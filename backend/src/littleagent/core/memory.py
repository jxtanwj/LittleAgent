"""长期记忆：把记忆存成一个 JSON 文件，并在上面做关键词检索。

存储和检索放在同一个模块，是因为两者缺了对方都没用。
存储这半边不知道"相关性"是什么，检索那半边不知道文件在哪。

这里是刻意不做成向量库的：没有 embedding 模型，也不依赖任何外部服务，
所以能离线跑、不花钱，代价是匹配"字面"而不是"意思"。

并发模型：进程内一把可重入锁把读-改-写串起来，落盘走"临时文件 + os.replace"，
所以读者要么看到旧文件、要么看到新文件，同一进程内的并发保存也不会互相覆盖。
跨进程（多个 uvicorn worker）不在保护范围内，详见 docs/memory.md。
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import tempfile
import threading
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 数据目录从环境变量读，默认放用户主目录，而不是从 __file__ 往上推：
# 那样推出的是安装位置，wheel 安装时会落在解释器内部（通常不可写，
# 而且升级解释器就把数据带走了）。开发时设 LITTLEAGENT_DATA_DIR=backend/data
# 可以让数据留在仓库里（该目录已被 .gitignore 排除）。
# 写成模块级变量而不是函数内部的常量，是为了让测试能把它指向一个临时文件。
DATA_DIR = Path(os.environ.get("LITTLEAGENT_DATA_DIR") or Path.home() / ".littleagent")
MEMORY_FILE = DATA_DIR / "memories.json"

Memory = dict[str, Any]

# 记忆的作用域。默认全局：单用户部署下所有会话共享一份记忆；
# 多用户时由 API 传入 user_id，写入和检索都按它隔离。
DEFAULT_SCOPE = "global"

# 一条记忆是一句话，这个上限防止整篇文档被塞进来。
# 它同时也是提示注入的缓解：能进上下文的东西越少越好。
MAX_CONTENT_LENGTH = 2000

# 这几个字段由存储层自己维护，update_memory 不接受外部覆盖。
# id 被改会让指向它的引用失效，scope 被改等于越权。
RESERVED_FIELDS = frozenset({"id", "created_at", "updated_at", "scope"})

# 保护下面所有读-改-写的锁。用可重入锁是因为公开函数会调用同样加锁的内部函数。
_LOCK = threading.RLock()

# 排序用的兜底时间：没有可用时间戳的条目不参与"新的优先"，排到最后。
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


# --- 字段与时间 --------------------------------------------------------------


def _now_iso() -> str:
    """带时区的 UTC 时间戳。旧数据是 naive 的本地时间，由 _parse_iso 兼容。"""
    return datetime.now(UTC).isoformat()


def _parse_iso(value: object) -> datetime | None:
    """解析时间戳，解不了返回 None。

    老条目是 naive 本地时间，直接和 aware 的比较会抛 TypeError，
    这里统一补成 UTC——排序只关心先后，差几个小时的偏移不影响结论。
    """
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed


def _timestamp_of(memory: Memory) -> float:
    """排序用的时间戳：优先最后修改时间，其次创建时间，都没有算 0。"""
    parsed = _parse_iso(memory.get("updated_at")) or _parse_iso(memory.get("created_at"))
    return parsed.timestamp() if parsed else 0.0


def _importance_of(memory: Memory) -> float:
    """取 importance，缺省或不是数字时按 0.5，并钳到 [0, 1]。"""
    value = memory.get("importance", 0.5)
    if not isinstance(value, (int, float)):
        return 0.5
    return max(0.0, min(float(value), 1.0))


def _id_of(memory: Memory) -> int:
    """取 id，缺 id 或不是整数时返回 -1（只用于排序）。"""
    value = memory.get("id")
    return value if isinstance(value, int) else -1


def _scope_of(memory: Memory) -> str:
    """取一条记忆的作用域；老条目没有这个字段，视为全局。"""
    scope = memory.get("scope")
    return scope if isinstance(scope, str) and scope else DEFAULT_SCOPE


def _in_scope(memory: Memory, scope: str | None) -> bool:
    """scope 为 None 表示不限作用域，留给管理用途。"""
    return scope is None or _scope_of(memory) == scope


# --- 读 ----------------------------------------------------------------------


def _read_locked() -> list[Memory]:
    """读出全部记忆；只要没有可读内容就返回空列表。

    文件不存在、为空、内容损坏，三种情况都算"没有记忆"，而不是报错——
    抛异常会让第一次使用就挂掉。但容忍不等于沉默：读不出来会被记录，
    而且 _write_locked 在覆盖读不出来的文件之前会先备份。

    非字典条目（手改的文件可能写成 [1, 2]）在这里被过滤掉：
    它们以前会让后面每个 .get 调用抛 AttributeError，把整个记忆功能带崩。
    """
    if not MEMORY_FILE.exists():
        return []

    raw = MEMORY_FILE.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("%s 不是合法 JSON，本次按没有记忆处理", MEMORY_FILE)
        return []

    # 手工编辑过的文件可能是个对象而不是数组。同样当成"没有记忆"处理，
    # 否则之后每次访问都会失败。
    if not isinstance(parsed, list):
        logger.warning("%s 的顶层不是数组，本次按没有记忆处理", MEMORY_FILE)
        return []

    memories = [entry for entry in parsed if isinstance(entry, dict)]
    if len(memories) != len(parsed):
        logger.warning(
            "%s 里有 %d 条不是对象的记录，已忽略", MEMORY_FILE, len(parsed) - len(memories)
        )
    return memories


def load_memories() -> list[Memory]:
    """读出全部记忆的副本。"""
    with _LOCK:
        return _read_locked()


def get_memory_by_id(memory_id: int, scope: str | None = None) -> Memory | None:
    """按 id 取一条记忆，没有则返回 None。拿到的是副本，不是能就地改的句柄。"""
    with _LOCK:
        for mem in _read_locked():
            if mem.get("id") == memory_id and _in_scope(mem, scope):
                return mem
    return None


# --- 写 ----------------------------------------------------------------------


def _needs_backup_locked() -> bool:
    """现有文件是否会在这次覆盖中丢掉内容。

    文件不存在或者为空，没有内容可丢；解析不出 JSON 数组说明里面有东西但读不出来，
    这正是必须先备份再覆盖的情形。
    """
    if not MEMORY_FILE.exists():
        return False

    raw = MEMORY_FILE.read_text(encoding="utf-8").strip()
    if not raw:
        return False

    try:
        return not isinstance(json.loads(raw), list)
    except json.JSONDecodeError:
        return True


def _backup_locked() -> None:
    """覆盖之前，把读不出来的文件复制成 .bak。

    复制而不是改名：改名会让 memories.json 出现"不存在"的窗口期，
    这期间任何读者看到的都是空记忆，而空记忆正是覆盖式丢数据的起点。

    固定一个 .bak 名字而不是按时间戳堆文件：需要的只是"这次覆盖前的内容"这一代。
    拷贝失败就让异常冒出去——宁可这次保存失败，也不能在没有备份的情况下覆盖。
    """
    if not _needs_backup_locked():
        return

    backup = MEMORY_FILE.with_name(MEMORY_FILE.name + ".bak")
    shutil.copy2(MEMORY_FILE, backup)
    logger.warning("%s 读不出来，已备份到 %s 再覆盖", MEMORY_FILE, backup)


def _write_locked(memories: list[Memory]) -> None:
    """把整个列表落盘：先写临时文件，再原子替换。

    替换而不是就地重写，是为了让读者永远看不到写了一半的文件。截断的 JSON
    会被读成"没有记忆"，然后在下一轮保存时把真实数据覆盖掉——这是丢数据的放大器。

    临时文件必须和目标在同一个目录：os.replace 的原子性依赖同一个文件系统，
    而且测试只把 MEMORY_FILE 指向临时目录，写到 DATA_DIR 会污染开发者的真实数据。
    """
    _backup_locked()

    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(memories, ensure_ascii=False, indent=2)
    # mkstemp 建出来的是 0600，替换前把原文件的权限位还回去，
    # 免得手改过的文件在保存之后变得只有自己能读。
    mode = MEMORY_FILE.stat().st_mode if MEMORY_FILE.exists() else None

    fd, temp_name = tempfile.mkstemp(
        dir=MEMORY_FILE.parent, prefix=f".{MEMORY_FILE.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            # fsync 之后才替换：否则断电时可能先落下"改名"、后落下数据块，
            # 留下一个空文件。
            os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(temp_name, mode)
        os.replace(temp_name, MEMORY_FILE)
    finally:
        # 替换成功后临时文件已经不存在了，missing_ok 让这条清理在成功和失败两条路径上都成立。
        Path(temp_name).unlink(missing_ok=True)


def _find_identical_locked(memories: list[Memory], content: object, scope: str) -> Memory | None:
    """找出同作用域内 content 完全相同的那条记忆。

    要求 content 是非空字符串：两条都没有 content 的坏条目不能因此被当成"相同"。
    """
    if not isinstance(content, str) or not content:
        return None

    for mem in memories:
        if (
            _scope_of(mem) == scope
            and mem.get("content") == content
            and isinstance(mem.get("id"), int)
        ):
            return mem
    return None


def save_memory(memory: Memory, scope: str = DEFAULT_SCOPE) -> int:
    """新增一条记忆，自动补上 id、created_at 和 scope，返回它的 id。

    同作用域内已经有内容完全相同的记忆时不再追加，而是刷新它——updated_at 更新、
    importance 取两者较大的——并返回既有 id。同一句话被反复保存很常见（模型每轮
    都可能"确认"一次），重复条目会在检索时挤占名额、把上下文塞满。
    memory_type 和 created_at 在合并时保持原样：类型不该被后来的说法改写，
    时间线也该指第一次记下来的时刻。
    """
    with _LOCK:
        memories = _read_locked()

        existing = _find_identical_locked(memories, memory.get("content"), scope)
        if existing is not None:
            existing["updated_at"] = _now_iso()
            existing["importance"] = max(_importance_of(existing), _importance_of(memory))
            _write_locked(memories)
            return int(existing["id"])

        # 取现有最大的 id，而不是"最后一个元素的 id"：手工编辑过的文件顺序可能是乱的，
        # 也可能有缺 id 的条目，那样会算出重复 id 甚至 KeyError。
        existing_ids = [m["id"] for m in memories if isinstance(m.get("id"), int)]
        new_id = max(existing_ids) + 1 if existing_ids else 1

        # 先复制再加工：调用方传进来的 dict 是调用方的，就地塞进 id 和 created_at
        # 是一种没人要的副作用。
        stored = dict(memory)
        stored["id"] = new_id
        stored["created_at"] = _now_iso()
        stored["scope"] = scope

        memories.append(stored)
        _write_locked(memories)
        return new_id


def update_memory(memory_id: int, new_data: Memory, scope: str | None = None) -> bool:
    """把 new_data 合并进指定 id 的记忆。没有这个 id 则返回 False。

    RESERVED_FIELDS 里的键被丢掉而不是报错：模型经常把整个对象原样回传，
    里面带着 id 和 created_at。真让它们覆盖会让引用失效、时间线错乱。
    """
    with _LOCK:
        memories = _read_locked()

        for mem in memories:
            if mem.get("id") != memory_id or not _in_scope(mem, scope):
                continue
            mem.update(
                {key: value for key, value in new_data.items() if key not in RESERVED_FIELDS}
            )
            mem["updated_at"] = _now_iso()
            _write_locked(memories)
            return True

    return False


def delete_memory(memory_id: int, scope: str | None = None) -> bool:
    """删掉指定 id 的记忆，返回是否真的删了。

    删除是记忆功能的一半：没有它，模型存错的东西只能靠手改 JSON 才能清掉。
    """
    with _LOCK:
        memories = _read_locked()
        remaining = [
            mem for mem in memories if not (mem.get("id") == memory_id and _in_scope(mem, scope))
        ]
        if len(remaining) == len(memories):
            return False

        _write_locked(remaining)
        return True


# --- 分词与打分 --------------------------------------------------------------

# 英文后缀，剥完至少要剩 3 个字符，免得 bus -> bu、is -> i 这类噪声。
_SUFFIXES = ("ies", "es", "s", "ed", "ing")
_MIN_STEM = 3


def _stem_forms(word: str) -> set[str]:
    """一个英文词的全部检索形式：原形，加上去掉各种后缀的样子。

    做成并集而不是"取一个词干"：只留词干的话，loved/love、classes/class
    这种一侧能剥、另一侧剥不动的词对仍然匹配不上——剥出来的不是同一个东西。
    多留几个形式只是让词表大一点，两边用同一个函数生成，交集照样成立。

    这仍然是很粗的启发式：hated 会顺带产出 hat 这种误匹配，
    news/new 也会撞上。真要做对得上词干提取库或者词向量，
    那超出了这个模块"零依赖、能离线跑"的边界。
    """
    forms = {word}
    for suffix in _SUFFIXES:
        if not word.endswith(suffix):
            continue

        stem = word[: -len(suffix)]
        # 剥完至少要剩 3 个字符，免得 bus -> bu、is -> i 这类噪声。
        if len(stem) < _MIN_STEM:
            continue
        forms.add(stem)

        if suffix == "ies":
            forms.add(stem + "y")
        elif suffix in ("ed", "ing"):
            # loved -> lov -> love：去掉 ed/ing 之后常常要把词尾的 e 加回来，
            # 否则 love/loved 这种最常见的时态变化反而匹配不上。
            forms.add(stem + "e")
            # running -> runn -> run：辅音双写同理。
            if stem[-1] == stem[-2] and len(stem) - 1 >= _MIN_STEM:
                forms.add(stem[:-1])
    return forms


def _extract_terms(text: str) -> set[str]:
    """把文本切成检索词：英文按词（含去后缀的形式），中文同时收单字和相邻两个字。"""
    text = text.lower()
    terms: set[str] = set()
    for word in re.findall(r"[a-z0-9_]+", text):
        terms |= _stem_forms(word)

    # 中文没有空格，相邻两个字承担英文里"单词"的角色。
    # 单字也一起收，而且两者都必须收：同一个函数既跑查询侧也跑每一条已存的记忆，
    # 两边词表必须用同样的方式构造，否则永远不可能相交。只收二字的话，
    # 用户输入单字「茶」时查询词是 {茶}，而记忆「用户喜欢喝茶」只会产出
    # {用户, 户喜, 喜欢, 欢喝, 喝茶}，两者没有交集，单字查询必然漏召回。
    for chunk in re.findall(r"[\u4e00-\u9fff]+", text):
        terms.update(chunk)
        for i in range(len(chunk) - 1):
            terms.add(chunk[i:i + 2])

    return terms


def _idf_map(documents: Sequence[set[str]], query_terms: set[str]) -> dict[str, float]:
    """给查询词算 IDF（BM25 的平滑版本），只算真正出现在候选文档里的词。

    两个细节都是必须的：

    - 语料里根本不存在的词不进表。把它们算进分母，会把唯一正确的部分匹配稀释
      成很低的分——两条语料时实测能低到 0.28，正确结果反而排不过噪声。
    - 平滑项让 df == 语料数时 IDF 仍然为正。语料只有一条记忆时，它的每个词都
      出现在全部文档里，没有平滑就会算成 0 分，那条记忆永远检索不到。
    """
    total = len(documents)
    idf: dict[str, float] = {}
    for term in query_terms:
        df = sum(1 for document in documents if term in document)
        if df:
            idf[term] = math.log(1 + (total - df + 0.5) / (df + 0.5))
    return idf


def _calc_score(
    memory: Memory,
    query_terms: set[str],
    idf: Mapping[str, float],
    query_idf_total: float,
) -> float:
    """给一条记忆对查询词打分。分数越高越相关，落在 [0, 1] 上。

    分子是命中词的 IDF 之和，分母是查询词里"有可能命中"的 IDF 之和。
    用 IDF 而不是简单的命中比例，是因为常见词（哪个记忆里都有）权重应该低：
    否则查询里的"用户""the"会和具体名词同权，长记忆也因为词多更容易蹭到重叠。
    """
    if not query_terms or query_idf_total <= 0:
        return 0.0

    memory_terms = _extract_terms(str(memory.get("content", "")))
    overlap = query_terms & memory_terms
    if not overlap:
        return 0.0

    lexical_score = sum(idf.get(term, 0.0) for term in overlap) / query_idf_total
    if lexical_score <= 0:
        return 0.0

    # importance 最多只能带来四分之一的加成，落在 [0.75, 1.0] 区间内。
    return lexical_score * (0.75 + 0.25 * _importance_of(memory))


def get_relevant_memories(
    query: str,
    limit: int = 5,
    memory_type: str | None = None,
    scope: str | None = DEFAULT_SCOPE,
    min_score: float = 0.0,
) -> list[Memory]:
    """返回最相关的至多 limit 条记忆，按相关度从高到低。

    和查询没有任何共同词的记忆会被丢掉，所以空结果的意思是"没有相关的"，
    而不是"什么都没存"。limit 小于等于 0 直接返回空列表：用切片实现的话，
    负数会从尾部切，返回的是"最后几条"这种没人要的结果。

    scope=None 表示不限作用域，默认只看全局作用域。同一内容只返回最新的一条：
    去重放在读取侧，因为旧文件里可能已经攒下了重复条目。
    """
    if limit <= 0:
        return []

    with _LOCK:
        memories = _read_locked()

    candidates = [
        mem
        for mem in memories
        if _in_scope(mem, scope) and (memory_type is None or mem.get("memory_type") == memory_type)
    ]

    query_terms = _extract_terms(query)
    if not query_terms:
        return []

    idf = _idf_map([_extract_terms(str(mem.get("content", ""))) for mem in candidates], query_terms)
    query_idf_total = sum(idf.values())

    scored: list[tuple[float, Memory]] = []
    for mem in candidates:
        score = _calc_score(mem, query_terms, idf, query_idf_total)
        if score > min_score:
            scored.append((score, mem))

    # 同分时新者优先，再同分按 id 倒序（时间戳可能撞到同一微秒）。
    # 文件顺序是老的在前面，直接用它会正好反过来：过时的记忆排在前面。
    scored.sort(key=lambda pair: (pair[0], _timestamp_of(pair[1]), _id_of(pair[1])), reverse=True)

    results: list[Memory] = []
    seen: set[str] = set()
    for _, mem in scored:
        content = str(mem.get("content", ""))
        if content in seen:
            continue
        seen.add(content)
        results.append(mem)
        if len(results) >= limit:
            break

    return results
