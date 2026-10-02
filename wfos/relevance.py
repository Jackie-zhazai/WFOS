"""One notion of "relevant to what this run is working on".

Two things consult this: which evidence reaches a prompt, and which memories of
previous runs do. They surface different material — one is this run's own
evidence, the other is what earlier runs left behind — but "relevant" has to mean
the same thing in both, or a reader comparing them is comparing two ideas.

Relevance here is deliberately crude: does the text name a file the run is
working on, and how much is the item trusted. There is no embedding model in this
project, and a keyword overlap the reader can predict is worth more than a
similarity score nobody can explain.

Ordering is `mentions desc, then trust desc`, with a recency-ish tiebreak left to
each caller's stable sort — so a prompt that lost an item to the cap today loses
the *same* item tomorrow rather than an arbitrary one. **The cap is always
reported**: "nothing was relevant" and "we cut it" have to stay distinguishable,
because only one of them means the operator should raise a limit.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Trust, as a number. An item with no stated confidence ranks with `low` rather
# than below it: we have no reason to rank it beneath an explicit `low`, only no
# reason to rank it above.
CONFIDENCE_RANK = {"high": 2, "medium": 1, "low": 0}
_UNSTATED_CONFIDENCE = 0


def working_names(working_files: Iterable[str]) -> frozenset[str]:
    """The strings worth looking for when asking whether text is about this run.

    Both the normalized repo-relative path and the bare filename, because the
    material quotes them differently: a tool result says `app.py`, a plan says
    `src/app.py`, and both mean the same file to a reader.
    """
    names: set[str] = set()
    for raw in working_files or ():
        norm = str(raw).replace("\\", "/").lstrip("./").lower()
        if not norm:
            continue
        names.add(norm)
        names.add(Path(norm).name)
    return frozenset(names)


def mentions(text: str, names: frozenset[str]) -> int:
    """How many distinct working-set names appear in `text`.

    Counted per name, not per occurrence: a log that says `app.py` twenty times
    is not twenty times more about `app.py` than one that says it once.
    """
    if not text or not names:
        return 0
    low = text.lower()
    return sum(1 for name in names if name in low)


# CJK runs are kept whole, as the repository's own query splitter does: a Chinese
# request shredded into single characters would overlap with everything.
_TOKENS = re.compile(r"[一-鿿]+|[A-Za-z0-9_]+")


def tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKENS.findall(text or "")}


def overlap_tokens(a: str, b: str) -> int:
    """How many distinct tokens two texts share.

    The coarse fallback for recognising "the same request as last time" when
    there are no files to match on — a run that failed before it planned or wrote
    anything. Coarser than an exact path intersection, which is why the caller
    needs a threshold rather than a truth test.
    """
    return len(tokens(a) & tokens(b))


def confidence_rank(confidence: Any) -> int:
    return CONFIDENCE_RANK.get(str(confidence or "").lower(), _UNSTATED_CONFIDENCE)


def relevance_key(text: str, confidence: Any, names: frozenset[str]) -> tuple[int, int]:
    """The sort key both callers order by. Negated, so `sorted` is best-first."""
    return (-mentions(text, names), -confidence_rank(confidence))


def cap(items: list, limit: int) -> dict:
    """`{"items", "total", "dropped"}` — the one shape a selection has.

    Callers sort first, so this is `items` best-first and `limit` keeps the head.
    """
    kept = items[:limit] if limit and limit > 0 else list(items)
    return {"items": kept, "total": len(items), "dropped": len(items) - len(kept)}


def select_evidence(rows: Iterable[dict], *, working_files: Iterable[str],
                    limit: int) -> dict:
    """A run's evidence, most relevant to its working set first, capped.

    Everything here belongs to *this* run, so the cap is not about relevance in
    the absolute — it is about which of a run's own evidence is worth the prompt
    budget when there is more than the budget affords. The ordering answers that;
    `dropped` says how much was left out.
    """
    names = working_names(working_files)
    ordered = sorted(
        rows,
        key=lambda row: (*relevance_key(f"{row.get('source', '')} {row.get('content', '')}",
                                        row.get("confidence"), names),
                         -(row.get("id") or 0)),
    )
    return cap(ordered, limit)
