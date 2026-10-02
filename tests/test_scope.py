"""Plan-scope enforcement and the stable error vocabulary.

Two claims are pinned here: a write outside the approved plan never reaches
disk and becomes an approval instead; and every refusal is classified by
exception *type*, so failures are countable rather than only readable.
"""
from __future__ import annotations

import asyncio

from wfos.agents.implementer import ImplementerAgent
from wfos.agents.investigator import InvestigatorAgent
from wfos.llm.scripted import ScriptedAdapter


# ------------------------------------------------------------------ helpers
def _write_scenario(app, *paths: str) -> ScriptedAdapter:
    """An implementer that writes each path in turn, then reports success."""
    llm = ScriptedAdapter()
    llm.set_script(
        [{"tool_call": {"name": "workspace.write",
                        "arguments": {"path": p, "content": "x = 1\n"}}} for p in paths]
        + [{"output": {"changes": [{"file": p, "action": "create", "detail": "d"}
                                   for p in paths],
                       "failed": [],
                       "next_step": {"suggested_state": "build_test", "reason": "done"}}}])
    return llm


def _implement_run(app, plan_paths, *, state="implement"):
    """A run parked at implement whose approved plan allows `plan_paths`."""
    h, repo = app["harness"], app["repo"]
    run = h.create_run("新增一个模块", kind="feature")
    plan = {"summary": "s", "files": [{"path": p, "action": "modify"} for p in plan_paths]}
    repo.update_run(run["id"], state=state, payload={"plan": plan})
    return run, h, repo


def _call(gw, tool, args, *, allowed=None, role="implementer", run_id="scope-run"):
    root = str(gw._server._config.project_root)
    gw.set_principal(role=role, agent=role, run_id=run_id,
                     read_roots=[root], write_roots=[root],
                     allowed_write_paths=allowed)
    return asyncio.run(gw.call(tool, args))


def _row(repo, run_id="scope-run"):
    return repo.tool_calls(run_id)[-1]


# ------------------------------------------------------- scope at the tool layer
def test_out_of_plan_write_is_refused_and_never_reaches_disk(app):
    out = _call(app["gateway"], "workspace.write",
                {"path": "evil.py", "content": "x = 1\n"}, allowed=["app.py"])
    assert not out.ok
    assert out.structured["error_code"] == "plan_scope_denied"
    assert out.structured["security_event"] == "plan_scope_denied"
    assert not (app["sandbox"] / "evil.py").exists()


def test_in_plan_write_is_allowed(app):
    out = _call(app["gateway"], "workspace.write",
                {"path": "app.py", "content": "x = 1\n"}, allowed=["app.py"])
    assert out.ok, out.error


def test_patch_is_scoped_too(app):
    """Scope is not write-only: patch edits a file just as write does."""
    out = _call(app["gateway"], "workspace.patch",
                {"path": "other.py", "before": "a", "after": "b"}, allowed=["app.py"])
    assert not out.ok
    assert out.structured["error_code"] == "plan_scope_denied"


def test_absent_scope_means_unconstrained(app):
    """None = the plan declared no file set, so nothing can be enforced."""
    out = _call(app["gateway"], "workspace.write",
                {"path": "anywhere.py", "content": "x = 1\n"}, allowed=None)
    assert out.ok, out.error


def test_empty_scope_blocks_every_write(app):
    """An empty *list* is a real (empty) plan and does block."""
    out = _call(app["gateway"], "workspace.write",
                {"path": "app.py", "content": "x = 1\n"}, allowed=[])
    assert not out.ok
    assert out.structured["error_code"] == "plan_scope_denied"


def test_approved_escalation_widens_the_scope_for_one_path_only(app):
    gw, repo = app["gateway"], app["repo"]
    args = {"path": "evil.py", "content": "x = 1\n"}
    assert not _call(gw, "workspace.write", args, allowed=["app.py"]).ok

    a = repo.create_approval("scope-run", "workspace.write:evil.py", risk_level="high")
    repo.decide_approval(a["id"], "approved", by="tester")
    assert _call(gw, "workspace.write", args, allowed=["app.py"]).ok

    other = {"path": "worse.py", "content": "y = 2\n"}
    assert not _call(gw, "workspace.write", other, allowed=["app.py"]).ok


# ------------------------------------------------- the escalation chain, end to end
def test_out_of_plan_write_escalates_to_an_approval_and_blocks(app):
    run, h, repo = _implement_run(app, ["app.py"])
    h.agents["implementer"] = ImplementerAgent(
        _write_scenario(app, "evil.py"), app["gateway"], app["sandbox"],
        repeated_call_threshold=app["cfg"].harness.repeated_call_threshold)

    result = asyncio.run(h._execute_state(repo.get_run(run["id"])))
    assert result["blocked"] is True
    assert not (app["sandbox"] / "evil.py").exists()

    approvals = repo.list_approvals(run["id"])
    assert [a["action"] for a in approvals] == ["workspace.write:evil.py"]
    assert approvals[0]["status"] == "pending"
    # The refused step is dropped so the approved retry actually re-runs.
    assert repo.get_step(run["id"], "implement") is None

    asyncio.run(h._decide_transition(repo.get_run(run["id"]), result))
    assert repo.get_run(run["id"])["status"] == "waiting_approval"


def test_approving_the_escalation_lets_the_retry_write(app):
    run, h, repo = _implement_run(app, ["app.py"])
    llm = _write_scenario(app, "evil.py")
    h.agents["implementer"] = ImplementerAgent(
        llm, app["gateway"], app["sandbox"],
        repeated_call_threshold=app["cfg"].harness.repeated_call_threshold)

    asyncio.run(h._execute_state(repo.get_run(run["id"])))
    for a in repo.list_approvals(run["id"]):
        repo.decide_approval(a["id"], "approved", by="tester")

    # A fresh script: the same request, now permitted.
    llm.set_script([
        {"tool_call": {"name": "workspace.write",
                       "arguments": {"path": "evil.py", "content": "x = 1\n"}}},
        {"output": {"changes": [{"file": "evil.py", "action": "create", "detail": "d"}],
                    "failed": [],
                    "next_step": {"suggested_state": "build_test", "reason": "done"}}}])

    result = asyncio.run(h._execute_state(repo.get_run(run["id"])))
    assert not result.get("blocked")
    assert (app["sandbox"] / "evil.py").exists()
    assert result["output"]["actual_changes"] == ["evil.py"]


def test_escalated_path_does_not_re_block_the_retry(app):
    """Once a path is under an approval, the gate owns it — re-counting it as a
    violation would block the approved retry forever."""
    run, h, repo = _implement_run(app, ["app.py"])
    llm = _write_scenario(app, "evil.py")
    h.agents["implementer"] = ImplementerAgent(
        llm, app["gateway"], app["sandbox"],
        repeated_call_threshold=app["cfg"].harness.repeated_call_threshold)
    asyncio.run(h._execute_state(repo.get_run(run["id"])))

    assert repo.plan_violations_for_run(run["id"]) == []      # already escalated
    escaped = repo.plan_violations_for_run(run["id"])
    for a in repo.list_approvals(run["id"]):
        repo.decide_approval(a["id"], "approved", by="tester")
    assert repo.plan_violations_for_run(run["id"]) == escaped


def test_plan_scope_can_be_disabled(app):
    app["cfg"].harness.enforce_plan_scope = False
    run, h, repo = _implement_run(app, ["app.py"])
    h.agents["implementer"] = ImplementerAgent(
        _write_scenario(app, "evil.py"), app["gateway"], app["sandbox"],
        repeated_call_threshold=app["cfg"].harness.repeated_call_threshold)

    result = asyncio.run(h._execute_state(repo.get_run(run["id"])))
    assert not result.get("blocked")
    assert (app["sandbox"] / "evil.py").exists()


def test_a_plan_with_no_file_list_does_not_narrow_anything(app):
    """feature_design declares files: [] — narrowing to that would block all
    writes for a run that never planned one."""
    run, h, repo = _implement_run(app, [])
    h.agents["implementer"] = ImplementerAgent(
        _write_scenario(app, "free.py"), app["gateway"], app["sandbox"],
        repeated_call_threshold=app["cfg"].harness.repeated_call_threshold)

    result = asyncio.run(h._execute_state(repo.get_run(run["id"])))
    assert not result.get("blocked")
    assert (app["sandbox"] / "free.py").exists()


# ------------------------------------------------------- the error vocabulary
def test_path_escape_is_classified_from_the_exception_type(app):
    out = _call(app["gateway"], "workspace.read", {"path": "../outside.txt"},
                role="investigator")
    assert not out.ok
    row = _row(app["repo"])
    assert row["status"] == "rejected"
    assert row["error_code"] == "path_escape"
    assert row["security_event"] == "path_escape"


def test_role_denial_is_a_security_event(app):
    out = _call(app["gateway"], "workspace.write",
                {"path": "app.py", "content": "x = 1\n"}, allowed=None,
                role="investigator")
    assert not out.ok
    row = _row(app["repo"])
    assert (row["status"], row["error_code"], row["security_event"]) == \
        ("rejected", "role_denied", "role_denied")


def test_argument_error_is_too_short_for_a_strict_check(app):
    out = _call(app["gateway"], "workspace.read", {"path": ""}, role="investigator")
    assert not out.ok
    row = _row(app["repo"])
    assert (row["status"], row["error_code"]) == ("rejected", "invalid_arguments")
    assert row["security_event"] is None            # not a security boundary


def test_unclassified_failure_degrades_to_tool_failed(app):
    """A genuine execution error must not be mislabelled as a policy denial."""
    out = _call(app["gateway"], "workspace.read", {"path": "no_such_file.py"},
                role="investigator")
    assert not out.ok
    row = _row(app["repo"])
    assert (row["status"], row["error_code"], row["security_event"]) == \
        ("error", "tool_failed", None)


def test_successful_call_is_recorded_as_ok(app):
    assert _call(app["gateway"], "workspace.read", {"path": "app.py"},
                 role="investigator").ok
    row = _row(app["repo"])
    assert (row["status"], row["error_code"]) == ("ok", None)


def test_undeclared_tool_is_audited_and_answered(app):
    """The refusal is both an audit fact and a tool result.

    It used to abort the state on the first call; the model is now told which
    tools exist and gets to correct itself. What has to hold either way is that
    the refusal is recorded and the call never ran — `workspace.write` is a real
    tool, just not one the investigator was given, so this is a role boundary and
    not a typo.
    """
    gw, repo = app["gateway"], app["repo"]
    llm = ScriptedAdapter()
    llm.set_script([{"tool_call": {"name": "workspace.write",
                                   "arguments": {"path": "x.py", "content": "x"}}}])
    agent = InvestigatorAgent(llm, gw, app["sandbox"])

    asyncio.run(agent.run({"run": {"id": "undeclared-run"}, "state": "req_capture",
                           "plan": {}, "prior": {}, "evidence": []}))

    row = repo.tool_calls("undeclared-run")[-1]
    assert (row["status"], row["error_code"]) == ("rejected", "unknown_tool")
    assert not (app["sandbox"] / "x.py").exists()


def test_repeat_guard_records_the_stable_code(app):
    gw, repo = app["gateway"], app["repo"]
    llm = ScriptedAdapter()
    llm.set_script([
        {"tool_call": {"name": "workspace.read", "arguments": {"path": "app.py"}}},
        {"tool_call": {"name": "workspace.read", "arguments": {"path": "app.py"}}},
        {"output": {"findings": [], "evidence": [], "project_context": {},
                    "next_step": {"suggested_state": "project_check", "reason": "d"}}},
    ])
    agent = InvestigatorAgent(llm, gw, app["sandbox"], repeated_call_threshold=2)
    asyncio.run(agent.run({"run": {"id": "repeat-run"}, "state": "req_capture",
                           "plan": {}, "prior": {}, "evidence": []}))

    rows = repo.tool_calls("repeat-run")
    assert (rows[0]["status"], rows[0]["error_code"]) == ("ok", None)
    assert (rows[1]["status"], rows[1]["error_code"]) == \
        ("rejected", "repeated_identical_call")
