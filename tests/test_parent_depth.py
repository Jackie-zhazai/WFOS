"""Parent/child recursion bound.

A child bugfix run is itself a bugfix run, so its own regression check can spawn
a further child. `max_parent_depth` is what stops that from recursing forever —
before it was enforced, the setting was read from config and never consulted.
"""
from __future__ import annotations

from wfos.harness import statemachine as sm


def _chain(repo, links: int) -> dict:
    """A root run with `links` generations of children below it."""
    cur = repo.create_run("bugfix", "root", "d")
    for i in range(links):
        cur = repo.create_run("bugfix", f"child{i}", "d", parent_run_id=cur["id"])
    return cur


def _regressed() -> dict:
    return {"regression": [{"module": "check.py", "ok": False, "detail": "failed"}]}


# ------------------------------------------------------------ depth arithmetic
def test_ancestor_depth_counts_parent_links(app):
    repo = app["repo"]
    root = repo.create_run("bugfix", "root", "d")
    assert repo.ancestor_depth(root["id"]) == 0
    child = repo.create_run("bugfix", "c", "d", parent_run_id=root["id"])
    assert repo.ancestor_depth(child["id"]) == 1
    assert child["root_run_id"] == root["id"]


def test_ancestor_depth_terminates_on_corrupt_cycle(app):
    """A run whose parent points back at itself must not hang the harness."""
    repo = app["repo"]
    root = repo.create_run("bugfix", "root", "d")
    repo.conn.execute("UPDATE runs SET parent_run_id=? WHERE id=?", (root["id"], root["id"]))
    repo.conn.commit()
    assert repo.ancestor_depth(root["id"]) == 1


# ------------------------------------------------------------- the enforcement
def test_spawn_is_refused_at_the_depth_limit_and_the_parent_fails(app):
    h, repo, cfg = app["harness"], app["repo"], app["cfg"]
    deepest = _chain(repo, cfg.harness.max_parent_depth)
    repo.update_run(deepest["id"], state="regression_verify")
    assert repo.ancestor_depth(deepest["id"]) == cfg.harness.max_parent_depth

    h._spawn_child(repo.get_run(deepest["id"]), _regressed())

    after = repo.get_run(deepest["id"])
    assert after["status"] == "failed"          # not waiting_child
    assert "深度" in (after["error"] or "")
    assert repo.child_runs(deepest["id"]) == []  # nothing was created
    assert after["payload"].get("child_run_id") is None


def test_spawn_still_creates_a_child_below_the_limit(app):
    h, repo, cfg = app["harness"], app["repo"], app["cfg"]
    parent = _chain(repo, cfg.harness.max_parent_depth - 1)
    repo.update_run(parent["id"], state="regression_verify")

    h._spawn_child(repo.get_run(parent["id"]), _regressed())

    after = repo.get_run(parent["id"])
    assert after["status"] == "waiting_child"
    children = repo.child_runs(parent["id"])
    assert len(children) == 1
    assert after["payload"]["child_run_id"] == children[0]["id"]
    assert children[0]["kind"] == "bugfix"


def test_spawn_records_the_refusal_as_a_transition(app):
    h, repo, cfg = app["harness"], app["repo"], app["cfg"]
    deepest = _chain(repo, cfg.harness.max_parent_depth)
    repo.update_run(deepest["id"], state="regression_verify")
    h._spawn_child(repo.get_run(deepest["id"]), _regressed())

    last = repo.transitions(deepest["id"])[-1]
    assert last["to_state"] == sm.SPAWN_BUGFIX
    assert "上限" in (last["reason"] or "")


def test_zero_disables_the_bound(app):
    """0 means 'no bound', matching repeated_call_threshold's convention."""
    h, repo, cfg = app["harness"], app["repo"], app["cfg"]
    cfg.harness.max_parent_depth = 0
    deepest = _chain(repo, 6)
    repo.update_run(deepest["id"], state="regression_verify")

    h._spawn_child(repo.get_run(deepest["id"]), _regressed())
    assert repo.get_run(deepest["id"])["status"] == "waiting_child"
