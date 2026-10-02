"""Request routing between the Feature and Bugfix machines.

Pure scoring on normalized text — no hidden state, deterministic for tests.
"""
from __future__ import annotations

import re

# (flow_kind, weight, [keywords]) — longer/more-specific phrases weigh more.
_KEYWORDS: dict[str, list[tuple[int, str]]] = {
    "feature": [
        (3, "新增功能"), (3, "添加功能"), (3, "开发新模块"), (3, "需要实现"),
        (3, "加一个"), (3, "做一个"), (2, "新增"), (2, "添加"), (2, "新功能"),
        (2, "实现"), (2, "开发"), (1, "功能"), (1, "feature"), (1, "新模块"),
        (1, "创建"),
    ],
    "bugfix": [
        (3, "故障"), (3, "报错"), (3, "异常"), (3, "崩溃"), (3, "卡死"),
        (3, "闪退"), (3, "不工作"), (3, "没反应"), (3, "结果不对"), (3, "出bug"),
        (2, "bug"), (2, "错误"), (2, "失败"), (1, "问题"),
    ],
}

_RE_SPACE = re.compile(r"\s+")


def _score(text: str, kind: str) -> int:
    norm = _RE_SPACE.sub("", text).lower()
    total = 0
    for weight, kw in _KEYWORDS[kind]:
        if kw.lower() in norm:
            total += weight
    return total


def route(text: str) -> str | None:
    """Return the flow kind ('feature' | 'bugfix') for the request text, or None."""
    if not text or not text.strip():
        return None
    s_f = _score(text, "feature")
    s_b = _score(text, "bugfix")
    if s_f == 0 and s_b == 0:
        return None
    if s_f == s_b:
        return "bugfix"          # diagnosis-first on ties
    return "feature" if s_f > s_b else "bugfix"
