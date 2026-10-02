"""Plan deviation reporting and failure classification.

Two facts the Harness must be able to state about a run without asking a model:
how the implementation deviated from the approved file list, and *why* a state
did not succeed.
"""
from __future__ import annotations

import asyncio

from wfos.agents.implementer import ImplementerAgent
from wfos.harness.orchestrator import FAILURE_CLASSES, compute_deviation, failure_class
from wfos.llm.scripted import ScriptedAdapter


# ------------------------------------------------------------------ deviation
def test_deviation_is_empty_when_the_plan_is_satisfied():
    d = compute_deviation(["app.py"], ["app.py"])
    assert (d["missing"], d["extra"]) == ([], [])
    assert d["planned"] == ["app.py"] and d["unplanned"] is False


def test_deviation_reports_planned_but_untouched_files():
    d = compute_deviation(["app.py", "check.py"], ["app.py"])
    assert d["missing"] == ["check.py"] and d["extra"] == []


def test_deviation_reports_writes_outside_the_plan():
    d = compute_deviation(["app.py"], ["app.py", "extra.py"])
    assert d["extra"] == ["extra.py"] and d["missing"] == []


def test_deviation_normalises_windows_separators_and_dot_slash():
    d = compute_deviation(["./pkg\\mod.py"], ["pkg/mod.py"])
    assert (d["missing"], d["extra"]) == ([], [])


def test_no_planned_file_list_means_nothing_to_compare():
    """A plan that declared no files cannot be deviated from."""
    d = compute_deviation(None, ["anything.py"])
    assert d["unplanned"] is True
    assert d["planned"] is None
    assert (d["missing"], d["extra"]) == ([], [])


# ------------------------------------------------------- failure classification
def test_failure_class_covers_each_kind():
    cases = [
        ("build_error", "build_test",
         {"build": {"ok": False}, "tests": {"failed": 0}}),
        ("test_failure", "build_test",
         {"build": {"ok": True}, "tests": {"failed": 2}}),
        ("regression", "regression_verify",
         {"verdict": "fail", "regression": [{"module": "m", "ok": False}]}),
        ("verification_failed", "verify_regression",
         {"verdict": "fail", "regression": [{"module": "m", "ok": True}]}),
        ("no_change", "implement", {"changes": [], "failed": ["could not apply"]}),
    ]
    for expected, state, output in cases:
        got = failure_class(state, output)
        assert got == expected, f"{state} -> {got!r}, 期望 {expected!r}"
        assert got in FAILURE_CLASSES


def test_failure_class_is_none_when_the_state_succeeded():
    assert failure_class("build_test",
                        {"build": {"ok": True}, "tests": {"failed": 0}}) is None
    assert failure_class("regression_verify",
                        {"verdict": "pass", "regression": []}) is None
    assert failure_class("implement",
                        {"changes": [{"file": "a.py"}], "failed": []}) is None
    assert failure_class("req_capture", {}) is None       # not a verifiable state


def test_build_error_takes_precedence_over_test_failure():
    """Both broken: report the earlier, more actionable cause."""
    assert failure_class("build_test",
                        {"build": {"ok": False}, "tests": {"failed": 3}}) == "build_error"


# ------------------------------------------------------------------ integration
def _implementer(app, *writes):
    llm = ScriptedAdapter()
    llm.set_script(
        [{"tool_call": {"name": "workspace.write",
                        "arguments": {"path": p, "content": "x = 1\n"}}} for p in writes]
        + [{"output": {"changes": [{"file": p, "action": "create", "detail": "d"}
                                   for p in writes],
                       "failed": [],
                       "next_step": {"suggested_state": "build_test", "reason": "done"}}}])
    return ImplementerAgent(llm, app["gateway"], app["sandbox"],
                            repeated_call_threshold=app["cfg"].harness.repeated_call_threshold)


def _run_at_implement(app, planned):
    h, repo = app["harness"], app["repo"]
    run = h.create_run("新增一个模块", kind="feature")
    plan = {"summary": "s", "files": [{"path": p, "action": "modify"} for p in planned]}
    repo.update_run(run["id"], state="implement", payload={"plan": plan})
    return run, h, repo


def test_deviation_reaches_the_record_and_the_verifier_prompt(app):
    run, h, repo = _run_at_implement(app, ["app.py"])
    # Widen the scope for one file, the only way an out-of-plan write can land.
    a = repo.create_approval(run["id"], "workspace.write:extra.py", risk_level="high")
    repo.decide_approval(a["id"], "approved", by="tester")

    h.agents["implementer"] = _implementer(app, "extra.py")
    asyncio.run(h._execute_state(repo.get_run(run["id"])))

    dev = repo.get_step(run["id"], "implement")["output_json"]["plan_deviation"]
    assert dev["missing"] == ["app.py"]          # planned, never touched
    assert dev["extra"] == ["extra.py"]          # touched, never planned

    prompt = h.agents["verifier"].build_prompt(h._build_ctx(repo.get_run(run["id"])))
    assert "与已批准方案的偏离" in prompt
    assert "extra.py" in prompt and "app.py" in prompt


def test_no_deviation_line_when_the_plan_matches(app):
    run, h, repo = _run_at_implement(app, ["mod.py"])
    h.agents["implementer"] = _implementer(app, "mod.py")
    asyncio.run(h._execute_state(repo.get_run(run["id"])))

    dev = repo.get_step(run["id"], "implement")["output_json"]["plan_deviation"]
    assert (dev["missing"], dev["extra"]) == ([], [])

    prompt = h.agents["verifier"].build_prompt(h._build_ctx(repo.get_run(run["id"])))
    assert "与已批准方案的偏离" not in prompt


def test_deviation_is_marked_unplanned_when_no_files_were_planned(app):
    run, h, repo = _run_at_implement(app, [])
    h.agents["implementer"] = _implementer(app, "free.py")
    asyncio.run(h._execute_state(repo.get_run(run["id"])))

    dev = repo.get_step(run["id"], "implement")["output_json"]["plan_deviation"]
    assert dev["unplanned"] is True
