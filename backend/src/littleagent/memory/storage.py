# 负责读写 JSON

# src/littleagent/memory/storage.py

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# 记忆文件路径（放在 backend/data 下）
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
MEMORY_FILE = DATA_DIR / "memories.json"

Memory = dict[str, Any]


def load_memories() -> list[Memory]:
    """加载所有长期记忆。

    文件不存在、为空、或者内容损坏时都返回空列表。空文件这一条不是防御性
    编程：仓库里刚 clone 下来的 memories.json 就是 0 字节，直接 json.load
    会抛 JSONDecodeError，把整个搜索/保存链路带崩。
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

    # 手工编辑过的文件可能是 dict 而不是 list，当作没有记忆处理。
    if not isinstance(memories, list):
        return []

    return memories


def _write(memories: list[Memory]) -> None:
    """把记忆列表落盘。"""
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_FILE.write_text(
        json.dumps(memories, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_memory(memory: Memory) -> int:
    """
    新增一条记忆。
    会自动补上 id 和 created_at。
    返回新记忆的 id。
    """
    memories = load_memories()

    # 取现有最大 id + 1，而不是“最后一个元素 + 1”：手工编辑过的文件顺序可能
    # 是乱的，也可能有缺 id 的条目，那样会算错甚至 KeyError。
    existing_ids = [m["id"] for m in memories if isinstance(m.get("id"), int)]
    new_id = max(existing_ids) + 1 if existing_ids else 1

    # 复制一份，避免污染调用方传进来的 dict。
    stored = dict(memory)
    stored["id"] = new_id
    stored["created_at"] = datetime.now().isoformat()

    memories.append(stored)
    _write(memories)

    return new_id


def update_memory(memory_id: int, new_data: Memory) -> bool:
    """
    更新指定 id 的记忆。
    返回是否更新成功。
    """
    memories = load_memories()

    for mem in memories:
        if mem.get("id") == memory_id:
            mem.update(new_data)
            mem["updated_at"] = datetime.now().isoformat()
            _write(memories)
            return True

    return False


def get_memory_by_id(memory_id: int) -> Optional[Memory]:
    """根据 id 获取单条记忆。"""
    for mem in load_memories():
        if mem.get("id") == memory_id:
            return mem
    return None
