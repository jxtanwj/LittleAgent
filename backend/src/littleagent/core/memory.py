"""长期记忆：把记忆存成一个 JSON 文件，并在上面做关键词检索。

存储和检索放在同一个模块，是因为两者缺了对方都没用，合起来也不到两百行。
存储这半边不知道"相关性"是什么，检索那半边不知道文件在哪。

这里是刻意不做成向量库的：没有 embedding 模型，也不依赖任何外部服务，
所以能离线跑、不花钱，代价是匹配"字面"而不是"意思"。
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

# 从 core/memory.py 往上四级正好到 backend/，所以记忆文件放在包外面而不是包里面：
# 它是数据不是代码，绝不能被塞进 wheel。写成模块级变量而不是函数内部的常量，
# 是为了让测试能把它指向一个临时文件。
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
MEMORY_FILE = DATA_DIR / "memories.json"

Memory = dict[str, Any]


def load_memories() -> list[Memory]:
    """读出全部记忆；只要没有可读内容就返回空列表。

    文件不存在、为空、内容损坏，三种情况都算"没有记忆"，而不是报错。
    "为空"这条不是多余的防御：刚 clone 下来的仓库里 memories.json 就是 0 字节，
    对它调 json.load 会抛 JSONDecodeError，那会连带把搜索和保存一起搞挂在第一次使用上。
    """
    if not MEMORY_FILE.exists():
        return []

    raw = MEMORY_FILE.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    try:
        memories = json.loads(raw)
    except json.JSONDecodeError:
        return []

    # 手工编辑过的文件可能是个对象而不是数组。同样当成"没有记忆"处理，
    # 否则之后每次访问都会失败。
    if not isinstance(memories, list):
        return []

    return memories


def _write(memories: list[Memory]) -> None:
    """把整个列表落盘。

    不是原子写入——写到一半崩溃会留下截断的 JSON。这正是 load_memories
    要容忍损坏文件、而不是把异常抛出去的原因。
    """
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_FILE.write_text(
        json.dumps(memories, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_memory(memory: Memory) -> int:
    """新增一条记忆，自动补上 id 和 created_at，返回新 id。"""
    memories = load_memories()

    # 取现有最大的 id，而不是"最后一个元素的 id"：手工编辑过的文件顺序可能是乱的，
    # 也可能有缺 id 的条目，那样会算出重复 id 甚至 KeyError。
    existing_ids = [m["id"] for m in memories if isinstance(m.get("id"), int)]
    new_id = max(existing_ids) + 1 if existing_ids else 1

    # 先复制再加工：调用方传进来的 dict 是调用方的，就地塞进 id 和 created_at
    # 是一种没人要的副作用。
    stored = dict(memory)
    stored["id"] = new_id
    stored["created_at"] = datetime.now().isoformat()

    memories.append(stored)
    _write(memories)

    return new_id


def update_memory(memory_id: int, new_data: Memory) -> bool:
    """把 new_data 合并进指定 id 的记忆。没有这个 id 则返回 False。

    这里没有字段白名单，所以 new_data 里带上 id 或 created_at 会把它们覆盖掉，
    进而让指向这条记忆的引用失效。
    """
    memories = load_memories()

    for mem in memories:
        if mem.get("id") == memory_id:
            mem.update(new_data)
            mem["updated_at"] = datetime.now().isoformat()
            _write(memories)
            return True

    return False


def get_memory_by_id(memory_id: int) -> Memory | None:
    """按 id 取一条记忆，没有则返回 None。拿到的是副本，不是能就地改的句柄。"""
    for mem in load_memories():
        if mem.get("id") == memory_id:
            return mem
    return None


def _extract_terms(text: str) -> set[str]:
    """把文本切成检索词：英文按单词，中文同时收单字和相邻两个字。"""
    text = text.lower()
    terms = set(re.findall(r"[a-z0-9_]+", text))

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


def _calc_score(memory: Memory, query_terms: set[str]) -> float:
    """给一条记忆对查询词打分。分数越高越相关。"""
    memory_terms = _extract_terms(str(memory.get("content", "")))
    if not memory_terms or not query_terms:
        return 0.0

    overlap = query_terms & memory_terms
    if not overlap:
        return 0.0

    # 查询词里有多大比例被这条记忆命中。分母是查询词数而不是记忆词数，
    # 所以长记忆不会因为长度被惩罚——实际上它反而更容易蹭到重叠。
    lexical_score = len(overlap) / len(query_terms)

    importance = memory.get("importance", 0.5)
    if not isinstance(importance, (int, float)):
        importance = 0.5

    # importance 最多只能带来四分之一的加成，落在 [0.75, 1.0] 区间内。
    return lexical_score * (0.75 + 0.25 * max(0.0, min(float(importance), 1.0)))


def get_relevant_memories(query: str, limit: int = 5) -> list[Memory]:
    """返回最相关的至多 limit 条记忆，按相关度从高到低。

    和查询没有任何共同词的记忆会被丢掉，所以空结果的意思是"没有相关的"，
    而不是"什么都没存"。limit 不能传负数：它是用切片实现的，-1 会丢掉最后
    一条命中，而不是返回空。
    """
    memories = load_memories()
    query_terms = _extract_terms(query)

    if not query_terms:
        return []

    scored = []
    for mem in memories:
        score = _calc_score(mem, query_terms)
        if score > 0:
            scored.append((score, mem))

    # sort() 是稳定排序，分数相同的记忆会保持它们在文件里的顺序。
    scored.sort(key=lambda pair: pair[0], reverse=True)

    return [mem for _, mem in scored[:limit]]
