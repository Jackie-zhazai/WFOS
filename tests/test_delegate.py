"""Sub-agents as child runs.

wfos already had one kind of parent/child: the regression child the state machine
spawns. A delegated sub-task is a second, and it reuses the same machinery on
purpose — the child is an ordinary run with its own budget, audit trail, approvals
and lease, so nothing about the parent's constraints has to be re-implemented for
it.

Depth is gated the way the pattern that inspired this does it: past the limit the
tool is **removed from the surface the model is shown**, not refused when called.
A tool the model never sees is one it cannot spend a round on. The tool layer
refuses it as well, because a boundary the model can talk past is not one.
"""
from __future__ import annotations

import asyncio

from wfos.agents.investigator import InvestigatorAgent
from wfos.llm.scripted import ScriptedAdapter

FEATURE = "新增一个用户模块，在 app.py 中追加 feature_user() 函数"


def _nested(app, depth: int):
    """A chain of runs `depth` links deep; returns the deepest one's id."""
    repo = app["repo"]
    parent_id = None
    for level in range(depth + 1):
        run = repo.create_run("feature", f"L{level}", "任务", parent_run_id=parent_id)
        parent_id = run["id"]
    return parent_id


# ------------------------------------------------------------------- spawning
def test_delegating_spawns_a_child_run_and_drives_it(app):
    h, repo = app["harness"], app["repo"]
    parent = h.create_run(FEATURE, kind="feature")

    result = asyncio.run(h.delegate({"run_id": parent["id"]},
                                    {"task": "新增一个小工具模块并验证"}))

    assert result["ok"] is True
    child = repo.get_run(result["child_run_id"])
    assert child["parent_run_id"] == parent["id"]
    assert child["status"] in ("completed", "failed", "waiting_approval")
    # The result is structured, not a string — that is the point of returning
    # through the Harness rather than joining on prose.
    assert {"child_run_id", "kind", "status", "state", "steps",
            "changed_paths", "failure_classes"} <= set(result)


def test_the_child_is_a_real_run_with_its_own_record(app):
    """Not a side channel: it is auditable like any other."""
    h, repo = app["harness"], app["repo"]
    parent = h.create_run(FEATURE, kind="feature")
    result = asyncio.run(h.delegate({"run_id": parent["id"]}, {"task": "做一个独立的小模块"}))

    child_id = result["child_run_id"]
    assert repo.steps_for_run(child_id), "子流程没有留下步骤记录"
    assert repo.transitions(child_id), "子流程没有留下转换记录"


def test_a_childs_writes_are_not_the_parents(app):
    """Snapshot attribution is per-run, so a child's work cannot inflate the
    parent's change set — which is what the plan-deviation check reads."""
    h, repo = app["harness"], app["repo"]
    parent = h.create_run(FEATURE, kind="feature")
    asyncio.run(h.delegate({"run_id": parent["id"]}, {"task": "做一个独立的小模块"}))

    assert repo.affected_paths_for_run(parent["id"]) == []


def test_a_delegation_without_a_parent_run_is_refused(app):
    h = app["harness"]
    result = asyncio.run(h.delegate({}, {"task": "做一个独立的小模块"}))
    assert result["ok"] is False and result["error_code"] == "delegate_no_parent"


# --------------------------------------------------------------------- depth
def test_the_tool_is_not_declared_once_the_depth_limit_is_reached(app):
    """Removed from the surface, not refused on call."""
    h, repo = app["harness"], app["repo"]
    deep = _nested(app, h.cfg.harness.max_parent_depth)

    ctx = h._build_ctx(repo.get_run(deep))
    assert ctx["delegate_allowed"] is False
    names = [t["name"] for t in h.agents["investigator"]._tool_specs(ctx)]
    assert "delegate" not in names


def test_the_tool_is_declared_below_the_limit(app):
    h, repo = app["harness"], app["repo"]
    shallow = _nested(app, 0)
    ctx = h._build_ctx(repo.get_run(shallow))

    assert ctx["delegate_allowed"] is True
    assert "delegate" in [t["name"] for t in h.agents["investigator"]._tool_specs(ctx)]


def test_the_tool_layer_refuses_a_delegation_past_the_limit_too(app):
    """The surface gate is a courtesy to the model; this is the boundary."""
    h, repo = app["harness"], app["repo"]
    deep = _nested(app, h.cfg.harness.max_parent_depth)

    result = asyncio.run(h.delegate({"run_id": deep}, {"task": "做一个独立的小模块"}))

    assert result["ok"] is False
    assert result["error_code"] == "delegate_depth_exceeded"
    assert repo.child_runs(deep) == []


def test_zero_depth_means_unbounded(app):
    """The same `0 = disabled` convention the other limits use."""
    h, repo = app["harness"], app["repo"]
    h.cfg.harness.max_parent_depth = 0
    deep = _nested(app, 5)

    assert h._build_ctx(repo.get_run(deep))["delegate_allowed"] is True
    assert asyncio.run(h.delegate({"run_id": deep}, {"task": "做一个独立的小模块"}))["ok"] is True


# ------------------------------------------------------------- at the tool layer
def test_a_standalone_server_refuses_delegation_with_a_code(app):
    """With no Harness behind it the call must say so, not return an empty
    success that looks like a delegation which did nothing."""
    server = app["gateway"].server
    original, server._delegate = server._delegate, None
    try:
        out = asyncio.run(server._impl_delegate({"task": "随便做点什么"}))
    finally:
        server._delegate = original

    assert out["ok"] is False and out["error_code"] == "delegate_unavailable"


def test_the_harness_wires_the_handler_in(app):
    assert app["harness"].gateway.server._delegate is not None


def test_a_real_agent_can_call_it_through_the_gateway(app):
    """End to end through the tool surface the model actually uses."""
    h, repo = app["harness"], app["repo"]
    parent = h.create_run(FEATURE, kind="feature")
    llm = ScriptedAdapter()
    llm.set_script([
        {"tool_call": {"name": "delegate", "arguments": {"task": "做一个独立的小模块"}}},
        {"output": {"findings": [], "evidence": [], "project_context": {},
                    "next_step": {"suggested_state": "project_check", "reason": "done"}}}])
    agent = InvestigatorAgent(llm, app["gateway"], app["sandbox"])
    ctx = h._build_ctx(repo.get_run(parent["id"]))

    out = asyncio.run(agent.run(ctx))

    assert out["next_step"]["suggested_state"] == "project_check"
    assert len(repo.child_runs(parent["id"])) == 1
    row = repo.tool_calls(parent["id"])[-1]
    assert row["tool"] == "delegate" and row["ok"] == 1


def test_the_parent_keeps_its_own_principal_after_a_delegation(app):
    """A child run must not inherit the parent's calls — nor the reverse.

    `delegate` is driven *inside* the parent's tool call, so the child's every
    state sets a principal on the way through. Nothing restored the parent's, so
    the parent's remaining tool calls in that state were role-checked as the
    child's last role, scope-checked against the child's approved file set, and
    **audited to the child's run_id** — which silently removes them from the
    parent's `affected_paths_for_run`, the set the plan-deviation check reads.

    Observed here rather than reasoned about: the assertion is on the audit row.
    """
    h, repo = app["harness"], app["repo"]
    parent = h.create_run(FEATURE, kind="feature")

    llm = ScriptedAdapter()
    llm.set_script([
        {"tool_call": {"name": "delegate", "arguments": {"task": "做一个独立的小模块"}}},
        # The call whose attribution the child would steal.
        {"tool_call": {"name": "workspace.read", "arguments": {"path": "app.py"}}},
        {"output": {"findings": [], "evidence": [], "project_context": {},
                    "next_step": {"suggested_state": "project_check", "reason": "done"}}}])
    agent = InvestigatorAgent(llm, app["gateway"], app["sandbox"])
    ctx = h._build_ctx(repo.get_run(parent["id"]))

    asyncio.run(agent.run(ctx))

    child = repo.child_runs(parent["id"])
    assert len(child) == 1, "没有派生出子流程，这个测试就没在测它"
    child_id = child[0]["id"]

    rows = [t for t in repo.tool_calls(parent["id"]) if t["tool"] == "workspace.read"]
    assert rows, "父流程在 delegate 之后没有再调用工具"
    assert all(r["run_id"] == parent["id"] for r in rows), (
        f"delegate 之后父流程的工具调用被记到了别的 run 上："
        f"{[(r['run_id'], r['tool']) for r in rows]}（子流程是 {child_id}）")
