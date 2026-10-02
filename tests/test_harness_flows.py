"""Harness orchestration: happy path, failure->retry->cap, approval blocking,
confidence confirmation, parent-child regression, and crash recovery."""
from __future__ import annotations

import asyncio

from conftest import drive

FEATURE_TEXT = "新增一个用户模块，实现在 app.py 中追加 feature_user() 函数"
DELETE_TEXT = "删除 app.py 中的旧实现并新增用户模块功能"
MEDIUM_TEXT = "修复 compute 函数行为异常，重新实现 app.py 中 compute(x)"


def _feature_states(repo, run_id):
    return [t["to_state"] for t in repo.transitions(run_id)]


# --------------------------------------------------------------- happy path
def test_feature_happy_path_completes(app):
    h = app["harness"]
    run = drive(h, FEATURE_TEXT)
    assert run["status"] == "completed"
    assert run["state"] == "completed"
    states = _feature_states(app["repo"], run["id"])
    assert states[-1] == "completed"
    # every state persisted a step
    cur = app["repo"].conn.execute(
        "SELECT COUNT(*) FROM steps WHERE run_id=?", (run["id"],)).fetchone()
    assert cur[0] >= 8


# ------------------------------------------ build/test result drives transition
def test_repeated_same_failure_escalates_instead_of_burning_retries(app, bad_check):
    """check.py can never pass, so build_test fails the same way every time.
    Recurring the same failure class is not convergence: the run escalates to a
    human rather than spending the rest of its retry budget on it."""
    h, repo = app["harness"], app["repo"]
    run = drive(h, FEATURE_TEXT)
    assert run["status"] == "waiting_approval"

    # build.check is a syntax check and passes; check.py's *assertions* fail, so
    # the class is the more specific test_failure rather than build_error.
    escalations = [a for a in repo.list_approvals(run["id"])
                   if a["action"].startswith("repeated_failure:")]
    assert [a["action"] for a in escalations] == ["repeated_failure:build_test:test_failure"]
    assert escalations[0]["status"] == "pending"

    # It stopped early: well short of max_verify_attempts (3) implement re-runs.
    states = _feature_states(repo, run["id"])
    assert states.count("implement") == 2
    assert "同一失败类型" in repo.transitions(run["id"])[-1]["reason"]

    # The recorded step says why, so the failure is countable not just readable.
    assert repo.get_step(run["id"], "build_test")["failure_class"] == "test_failure"


def test_retry_cap_still_applies_when_escalation_is_disabled(app, bad_check):
    """Escalation off => the original contract: retry to the cap, then fail."""
    h, repo = app["harness"], app["repo"]
    app["cfg"].harness.repeated_failure_threshold = 0
    run = drive(h, FEATURE_TEXT)
    assert run["status"] == "failed"
    states = _feature_states(repo, run["id"])
    assert states.count("implement") >= 3          # retried more than once
    attempts = (run.get("payload") or {}).get("attempts", {})
    assert attempts.get("implement", 0) > 1        # retry budget exhausted


def test_rejecting_the_escalation_ends_the_run(app, bad_check):
    """A refused 'stop retrying this' cannot be honoured by looping again."""
    h, repo = app["harness"], app["repo"]
    run = drive(h, FEATURE_TEXT)
    assert run["status"] == "waiting_approval"
    escalation = [a for a in repo.list_approvals(run["id"])
                  if a["action"].startswith("repeated_failure:")][0]
    repo.decide_approval(escalation["id"], "rejected", by="tester")

    run = asyncio.run(h.advance(run["id"]))
    assert run["status"] == "failed"
    assert "升级审批被拒绝" in (run["error"] or "")


def test_approving_the_escalation_grants_a_fresh_retry_budget(app, bad_check):
    h, repo = app["harness"], app["repo"]
    run = drive(h, FEATURE_TEXT)
    escalation = [a for a in repo.list_approvals(run["id"])
                  if a["action"].startswith("repeated_failure:")][0]
    repo.decide_approval(escalation["id"], "approved", by="tester")
    assert (repo.get_run(run["id"])["payload"]
            .get("failure_attempts", {}).get("build_test:test_failure")) == 0

    run = asyncio.run(h.advance(run["id"]))
    # It retries, and the same failure recurring escalates again — a human keeps
    # a hand on a loop that is demonstrably not converging.
    assert run["status"] == "waiting_approval"
    escalations = [a for a in repo.list_approvals(run["id"])
                   if a["action"].startswith("repeated_failure:")]
    assert len(escalations) == 1          # same key, tracked not duplicated


# --------------------------------------------------------- approval: high risk
def test_delete_operation_requires_approval_and_blocks(app):
    h = app["harness"]
    run = drive(h, DELETE_TEXT)
    assert run["status"] == "waiting_approval"
    pending = app["repo"].pending_approvals_for_run(run["id"])
    actions = {a["action"] for a in pending}
    assert any(a.startswith("workspace.delete:") for a in actions)


def test_approve_resumes_and_completes(app):
    h = app["harness"]
    repo = app["repo"]
    r0 = h.create_run(DELETE_TEXT, kind="feature")

    async def scenario():
        run = await h.advance(r0["id"])
        assert run["status"] == "waiting_approval"
        for a in repo.pending_approvals_for_run(r0["id"]):
            h.approve(a["id"], by="tester")
        return await h.advance(r0["id"])

    run = asyncio.run(scenario())
    assert run["status"] == "completed"


def test_reject_fails_run(app):
    h = app["harness"]
    repo = app["repo"]
    r0 = h.create_run(DELETE_TEXT, kind="feature")

    async def scenario():
        await h.advance(r0["id"])
        for a in repo.pending_approvals_for_run(r0["id"]):
            h.reject(a["id"], by="tester", note="rejected")
        return await h.advance(r0["id"])

    run = asyncio.run(scenario())
    assert run["status"] == "failed"


# -------------------------------------------------- medium-confidence confirms
def test_medium_confidence_requires_confirmation(app):
    # A check.py without literal asserts -> root-cause confidence is medium.
    (app["sandbox"] / "check.py").write_text(
        'from app import compute\n'
        'print("PASS: compute(1)==", compute(1) == 2)\n'
        'print("PASS: compute(2)==", compute(2) == 4)\n', encoding="utf-8")
    h = app["harness"]
    repo = app["repo"]
    r0 = h.create_run(MEDIUM_TEXT, kind="bugfix")

    async def scenario():
        run = await h.advance(r0["id"])
        assert run["status"] == "waiting_approval"
        assert run["state"] == "confidence_assess"
        actions = {a["action"] for a in repo.pending_approvals_for_run(r0["id"])}
        assert "confirm_root_cause" in actions
        for a in repo.pending_approvals_for_run(r0["id"]):
            h.approve(a["id"], by="tester")
        return await h.advance(r0["id"])

    run = asyncio.run(scenario())
    assert run["status"] == "completed"
    states = _feature_states(repo, run["id"]) if False else \
        [t["to_state"] for t in repo.transitions(run["id"])]
    assert "fix_plan" in states


# ------------------------------------------------------ parent-child regression
def test_parent_child_regression_cycle(app):
    (app["sandbox"] / "regressions.txt").write_text("core\n", encoding="utf-8")
    h = app["harness"]
    repo = app["repo"]
    parent = h.create_run(FEATURE_TEXT, kind="feature")

    async def scenario():
        p = await h.advance(parent["id"])
        assert p["status"] == "waiting_child"
        child_id = p["payload"]["child_run_id"]
        assert repo.get_run(child_id)["kind"] == "bugfix"
        c = await h.advance(child_id)
        assert c["status"] == "completed"
        p2 = await h.resume_after_child(child_id)
        return p2, child_id

    parent, child_id = asyncio.run(scenario())
    assert parent["status"] == "completed"
    states = [t["to_state"] for t in repo.transitions(parent["id"])]
    assert "spawn_bugfix" in states
    # parent re-entered regression_verify after the child and finished it
    from_to = [(t["from_state"], t["to_state"]) for t in repo.transitions(parent["id"])]
    assert from_to.count(("regression_verify", "spawn_bugfix")) == 1
    assert ("regression_verify", "knowledge_distill") in from_to
    assert from_to[-1] == ("knowledge_distill", "completed")
    # regression markers cleared by the child's fix
    assert (app["sandbox"] / "regressions.txt").read_text(encoding="utf-8").strip() == ""


# ----------------------------------------------------------- crash recovery
def test_recovery_reuses_persisted_step(app):
    h = app["harness"]
    repo = app["repo"]
    run = h.create_run(FEATURE_TEXT, kind="feature")
    # Simulate: the previous process died right after persisting the req_capture
    # step but BEFORE the transition was persisted.
    repo.add_step(run["id"], "req_capture", "investigator",
                  {"findings": ["seeded"], "evidence": [],
                   "project_context": {}, "next_step": {"suggested_state": "project_check",
                                                        "reason": "seeded"}},
                  status="done")

    run = asyncio.run(h.advance(run["id"]))
    assert run["status"] == "completed"
    # The seeded step was reused, not re-created.
    cur = repo.conn.execute(
        "SELECT COUNT(*) FROM steps WHERE run_id=? AND state='req_capture'",
        (run["id"],)).fetchone()
    assert cur[0] == 1


# ------------------------------------------------------------ resume a parked run
def test_resume_unparks_a_paused_run(app):
    """`paused` used to be a trap: `advance` no-ops on it, and `resume`
    delegated straight back to `advance`, so a paused run could never move."""
    h, repo = app["harness"], app["repo"]
    run = h.create_run(FEATURE_TEXT, kind="feature")
    asyncio.run(h.advance(run["id"], max_loops=2))
    state_when_parked = repo.get_run(run["id"])["state"]
    progressed = [t["to_state"] for t in repo.transitions(run["id"])]

    repo.set_status(run["id"], "paused")
    # Advancing a parked run changes nothing; only resume un-parks it.
    parked = asyncio.run(h.advance(run["id"]))
    assert parked["status"] == "paused"
    assert parked["state"] == state_when_parked

    resumed = asyncio.run(h.resume(run["id"]))
    assert resumed["status"] == "completed"
    after = [t["to_state"] for t in repo.transitions(run["id"])]
    assert progressed == after[:len(progressed)]      # parked progress retained


# ------------------------------------------------------- a broken build reports
def test_a_syntax_error_is_reported_rather_than_raised(app):
    """The state that handles a broken build must not itself blow up.

    `build.check` used to emit dicts where BuildResult.errors expects strings,
    so any real syntax error raised AgentError straight out of the Verifier —
    the failure path was the crashing one."""
    (app["sandbox"] / "app.py").write_text("def compute(x)\n    return x\n",
                                          encoding="utf-8")
    run = drive(app["harness"], FEATURE_TEXT)         # must not raise
    assert run["status"] in ("waiting_approval", "failed")
    assert app["repo"].get_step(run["id"], "build_test")["failure_class"] == "build_error"
