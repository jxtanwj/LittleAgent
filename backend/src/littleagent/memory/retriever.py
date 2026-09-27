# 负责搜索相关记忆

# src/littleagent/memory/retriever.py

import re
from .storage import load_memories


def _extract_terms(text: str) -> set[str]:
    """
    从文本里提取搜索关键词。
    英文按单词切，中文同时收单字和相邻两个字。
    """
    text = text.lower()
    terms = set(re.findall(r"[a-z0-9_]+", text))  # 英文和数字

    # 中文没有空格，单字和相邻两个字都要收。
    #
    # 只收 bigram 是不行的：那样记忆侧的「用户喜欢喝茶」只会产出
    # {用户, 户喜, 喜欢, 欢喝, 喝茶}，而用户只说一个「茶」时查询词是 {茶}，
    # 两个词表永不相交，单字查询必然漏召回。两边都收单字，词表才是对称的。
    for chunk in re.findall(r"[\u4e00-\u9fff]+", text):
        terms.update(chunk)
        for i in range(len(chunk) - 1):
            terms.add(chunk[i:i+2])

    return terms


def _calc_score(memory: dict, query_terms: set[str]) -> float:
    """
    计算一条记忆和查询词的相关分数。
    分数越高越相关。
    """
    memory_terms = _extract_terms(memory.get("content", ""))
    if not memory_terms or not query_terms:
        return 0.0

    # 计算重叠的关键词数量
    overlap = query_terms & memory_terms
    if not overlap:
        return 0.0

    # 基础相关度 = 重叠词数 / 查询词数
    lexical_score = len(overlap) / len(query_terms)

    # 再乘上重要性（如果有的话）
    importance = memory.get("importance", 0.5)
    if not isinstance(importance, (int, float)):
        importance = 0.5

    return lexical_score * (0.75 + 0.25 * max(0, min(importance, 1)))


def get_relevant_memories(query: str, limit: int = 5) -> list[dict]:
    """
    根据查询文本，返回最相关的几条记忆。
    
    参数：
        query: 用户当前说的话
        limit: 最多返回几条（默认 5）
    
    返回：
        按相关度排序后的记忆列表
    """
    memories = load_memories()
    query_terms = _extract_terms(query)

    if not query_terms:
        return []

    # 计算每条记忆的分数
    scored = []
    for mem in memories:
        score = _calc_score(mem, query_terms)
        if score > 0:
            scored.append((score, mem))

    # 按分数从高到低排序
    scored.sort(key=lambda x: x[0], reverse=True)

    # 只返回前 limit 条
    return [mem for score, mem in scored[:limit]]