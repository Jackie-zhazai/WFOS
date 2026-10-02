"""Relevance: the one notion evidence and memory both rank by.

Two things are pinned here. First the primitives, which are deliberately crude —
a keyword overlap a reader can predict, because this project has no embedding
model and a similarity score nobody can explain is worse than a coarse rule
everybody can. Second, and more importantly, **the cap is never silent**: an
empty or short selection has to say whether it is short because nothing was
relevant or because something was cut, since only one of those means the operator
should raise a limit.
"""
from __future__ import annotations

import asyncio

from wfos.relevance import (
    cap,
    confidence_rank,
    mentions,
    relevance_key,
    select_evidence,
    working_names,
)

FEATURE = "新增一个用户模块，在 app.py 中追加 feature_user() 函数"


# ------------------------------------------------------------------- primitives
def test_working_names_carries_both_the_path_and_the_basename():
    """A tool result says `app.py`, a plan says `src/app.py`, and both mean the
    same file to a reader."""
    names = working_names(["src/app.py", ".\\check.py", ""])
    assert "src/app.py" in names and "app.py" in names
    assert "check.py" in names
    assert "" not in names


def test_mentions_counts_names_not_occurrences():
    names = working_names(["app.py"])
    assert mentions("app.py app.py app.py", names) == 1
    assert mentions("no filenames here", names) == 0
    assert mentions("", names) == 0
    assert mentions("app.py", frozenset()) == 0


def test_mentions_ignores_case():
    assert mentions("APP.PY", working_names(["app.py"])) == 1


def test_confidence_rank_orders_trust_and_parks_unstated_at_low():
    """An unlabelled item ranks with `low`, not below it: there is no reason to
    rank it beneath an explicit `low`, only no reason to rank it above."""
    assert confidence_rank("high") > confidence_rank("medium") > confidence_rank("low")
    assert confidence_rank(None) == confidence_rank("low")
    assert confidence_rank("nonsense") == confidence_rank("low")


def test_relevance_key_puts_more_mentions_first_then_trust():
    names = working_names(["app.py", "check.py"])
    both = relevance_key("app.py check.py", "low", names)
    one = relevance_key("app.py", "high", names)
    assert both < one, "提到两个工作文件的应当排在只提一个的前面"
    assert relevance_key("app.py", "high", names) < relevance_key("app.py", "low", names)


def test_the_key_orders_best_first_under_a_plain_sort():
    """The key is already negated, so callers sort ascending and get best-first —
    a caller that negated it again would silently reverse every ranking."""
    names = working_names(["app.py"])
    rows = [{"t": "nothing here", "c": "high"}, {"t": "app.py", "c": "low"}]
    rows.sort(key=lambda r: relevance_key(r["t"], r["c"], names))
    assert rows[0]["t"] == "app.py"


def test_cap_reports_what_it_dropped():
    assert cap([1, 2, 3, 4, 5], 2) == {"items": [1, 2], "total": 5, "dropped": 3}


def test_cap_with_zero_keeps_everything():
    """`0` means no cap, not an empty selection — the distinction matters because
    the wrong reading turns a limit into a black hole."""
    assert cap([1, 2, 3], 0) == {"items": [1, 2, 3], "total": 3, "dropped": 0}


# --------------------------------------------------------------- what survives
def _evidence(n, *, source="other.py", confidence="medium"):
    return [{"id": i, "kind": "log", "source": f"{source}#{i}",
             "content": "x" * 10, "confidence": confidence} for i in range(n)]


def test_evidence_that_names_a_working_file_survives_the_cap():
    rows = _evidence(5) + [{"id": 99, "kind": "code", "source": "app.py",
                            "content": "def compute()", "confidence": "medium"}]
    picked = select_evidence(rows, working_files=["app.py"], limit=1)
    assert [row["id"] for row in picked["items"]] == [99]
    assert picked["total"] == 6 and picked["dropped"] == 5


def test_trust_breaks_a_tie_between_equally_relevant_items():
    rows = [{"id": 1, "kind": "log", "source": "app.py", "content": "", "confidence": "low"},
            {"id": 2, "kind": "log", "source": "app.py", "content": "", "confidence": "high"}]
    picked = select_evidence(rows, working_files=["app.py"], limit=1)
    assert [row["id"] for row in picked["items"]] == [2]


def test_a_tie_the_other_way_is_broken_by_id_so_the_cap_is_repeatable():
    """Otherwise the item that gets cut changes between runs on identical input,
    and a baseline diff would show churn that means nothing."""
    rows = [{"id": i, "kind": "log", "source": "app.py", "content": "",
             "confidence": "medium"} for i in (1, 2, 3)]
    first = select_evidence(rows, working_files=["app.py"], limit=2)
    shuffled = list(reversed(rows))
    assert ([r["id"] for r in first["items"]]
            == [r["id"] for r in select_evidence(shuffled, working_files=["app.py"],
                                                 limit=2)["items"]])


def test_a_run_with_no_working_files_keeps_the_most_trusted_evidence():
    """Before the plan exists there is nothing to be relevant *to*, so the cap
    falls back on trust rather than on nothing at all."""
    rows = [{"id": 1, "kind": "log", "source": "a", "content": "", "confidence": "low"},
            {"id": 2, "kind": "log", "source": "b", "content": "", "confidence": "high"}]
    picked = select_evidence(rows, working_files=[], limit=1)
    assert [row["id"] for row in picked["items"]] == [2]


# ------------------------------------------------------------------ end to end
def test_a_real_run_records_how_much_evidence_the_cap_left_out(app):
    h, repo, cfg = app["harness"], app["repo"], app["cfg"]
    cfg.harness.evidence_limit = 2
    run_id = h.create_run(FEATURE, kind="feature")["id"]
    for i in range(6):
        repo.add_evidence(run_id, "log", f"src{i}.py", "some finding", "medium")

    asyncio.run(h.advance(run_id))

    metas = [(s.get("input_json") or {}).get("prompt_metadata", {}).get("evidence") or {}
             for s in repo.steps_for_run(run_id)]
    assert any(m.get("dropped", 0) > 0 for m in metas), "被裁掉的证据没有留下任何痕迹"
    assert all(m.get("injected", 0) <= 2 for m in metas if m)
    assert any(m.get("total", 0) >= 6 for m in metas)
