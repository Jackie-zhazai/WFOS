"""The resume contract: persisted steps are reused only while still reproducible.

A step output is stuck in time — the model, the policy, and the files it was
produced against can all move on. Resuming onto it without checking is how a run
silently continues on reasoning that no longer holds.
"""
from __future__ import annotations

import asyncio

import pytest

from wfos.harness import identity
from wfos.harness.identity import (
    RESUME_IDENTITY_MISMATCH,
    RESUME_NO_FINGERPRINT,
    RESUME_VALID,
    RESUME_WORKSPACE_DRIFT,
    capture_key_files,
    validate_resume,
)
from wfos.llm.mock import MockAdapter


class _OtherBrain(MockAdapter):
    """A second mock that the fingerprint can tell apart from the first."""
    name = "mock-other"
    model = "a-different-brain"


# --------------------------------------------------------------- identity hash
def test_identity_changes_when_the_brain_changes(app):
    """The fingerprint follows the adapters, not the config that built them.

    Under a mock provider `llm.model` is read by nothing — the mock ignores it —
    so editing it changes no step's conditions and must not move the hash. A real
    model change rebuilds the adapters, and that is what this asserts.
    """
    agents = app["harness"].agents
    before = identity.identity_hash(app["cfg"], agents)

    for agent in agents.values():
        agent.llm = _OtherBrain()

    assert identity.identity_hash(app["cfg"], agents) != before


def test_the_configured_model_reaches_the_fingerprint_through_the_adapter(app):
    """The other half: for a real provider the config *is* the adapter's model.

    Adapters are built here but never called, so no network is involved — what is
    under test is the chain cfg.llm.model -> adapter.model -> fingerprint.
    """
    from wfos.harness.orchestrator import Harness

    cfg, repo, gateway, wiki = app["cfg"], app["repo"], app["gateway"], app["wiki"]
    cfg.llm.provider = "openai"
    cfg.llm.allow_missing_key = True          # never called, so no credential needed
    cfg.llm.model = "model-a"
    before = identity.identity_hash(cfg, Harness(cfg, repo, gateway, wiki).agents)

    cfg.llm.model = "model-b"
    after = identity.identity_hash(cfg, Harness(cfg, repo, gateway, wiki).agents)

    assert after != before


def test_routing_moves_the_fingerprint(app):
    """A per-role model is part of what a step was produced under.

    Without this, a run whose implementer answered on a cheap model could be
    resumed onto steps an expensive one produced, with nothing recording the swap.
    """
    from wfos.harness.orchestrator import Harness

    cfg, repo, gateway, wiki = app["cfg"], app["repo"], app["gateway"], app["wiki"]
    cfg.llm.provider = "openai"
    cfg.llm.allow_missing_key = True
    cfg.llm.model = "shared-model"
    before = identity.identity_hash(cfg, Harness(cfg, repo, gateway, wiki).agents)

    cfg.llm.routing = {"implementer": "a-better-model"}
    routed = Harness(cfg, repo, gateway, wiki)
    after = identity.identity_hash(cfg, routed.agents)

    assert after != before
    assert routed.agents["implementer"].llm.model == "a-better-model"
    assert routed.agents["verifier"].llm.model == "shared-model"


def test_identity_changes_when_the_rules_change(app):
    before = identity.identity_hash(app["cfg"], app["harness"].agents)
    app["cfg"].harness.enforce_plan_scope = not app["cfg"].harness.enforce_plan_scope
    assert identity.identity_hash(app["cfg"], app["harness"].agents) != before

    app["cfg"].harness.safe_commands = list(app["cfg"].harness.safe_commands)
    after = identity.identity_hash(app["cfg"], app["harness"].agents)
    assert after != before


def test_identity_changes_when_a_role_loses_a_tool(app):
    before = identity.identity_hash(app["cfg"], app["harness"].agents)
    app["harness"].agents["implementer"].allowed_tools = ["workspace.read"]
    assert identity.identity_hash(app["cfg"], app["harness"].agents) != before


def test_identity_is_stable_when_nothing_changed(app):
    assert identity.identity_hash(app["cfg"], app["harness"].agents) == \
        identity.identity_hash(app["cfg"], app["harness"].agents)


# ------------------------------------------------------------------ key files
def test_key_files_record_digests_and_missing_files_distinctly(app):
    root = app["sandbox"]
    (root / "gone.py").unlink(missing_ok=True)
    keys = capture_key_files(root, ["app.py", "gone.py"])
    assert set(keys) == {"app.py", "gone.py"}
    assert keys["app.py"]
    assert keys["gone.py"] is None          # absent, not omitted


# -------------------------------------------------------------- validate_resume
def test_no_recorded_identity_is_unknown_not_invalid(app):
    status, detail = validate_resume(None, {}, "abc", app["sandbox"], [])
    assert status == RESUME_NO_FINGERPRINT
    assert "未记录指纹" in detail["reason"]


def test_identity_mismatch_is_detected(app):
    status, detail = validate_resume("old-hash", {}, "new-hash", app["sandbox"], [])
    assert status == RESUME_IDENTITY_MISMATCH
    assert detail["recorded"] == "old-hash"     # truncated to 12 chars


def test_modified_key_file_is_drift(app):
    keys = capture_key_files(app["sandbox"], ["app.py"])
    (app["sandbox"] / "app.py").write_text("def compute(x):\n    return x * 9\n",
                                          encoding="utf-8")
    status, detail = validate_resume("h", keys, "h", app["sandbox"], [])
    assert status == RESUME_WORKSPACE_DRIFT
    assert detail["drifted"] == [{"path": "app.py", "state": "modified"}]


def test_deleted_key_file_is_drift(app):
    keys = capture_key_files(app["sandbox"], ["app.py"])
    (app["sandbox"] / "app.py").unlink()
    status, detail = validate_resume("h", keys, "h", app["sandbox"], [])
    assert status == RESUME_WORKSPACE_DRIFT
    assert detail["drifted"] == [{"path": "app.py", "state": "deleted"}]


def test_the_runs_own_writes_are_not_drift(app):
    """A run that edits a key file has not drifted from itself — without this
    the contract would fire on every run that writes anything."""
    keys = capture_key_files(app["sandbox"], ["app.py"])
    (app["sandbox"] / "app.py").write_text("changed by me\n", encoding="utf-8")
    status, _ = validate_resume("h", keys, "h", app["sandbox"], ["app.py"])
    assert status == RESUME_VALID


def test_untouched_key_files_stay_valid(app):
    keys = capture_key_files(app["sandbox"], ["app.py", "check.py"])
    assert validate_resume("h", keys, "h", app["sandbox"], [])[0] == RESUME_VALID


# ------------------------------------------------------------------ integration
def _run_with_plan(app, files):
    h, repo = app["harness"], app["repo"]
    run = h.create_run("新增一个模块", kind="feature")
    repo.update_run(run["id"], payload={"plan": {"summary": "s", "files": files}})
    h._record_fingerprint(run["id"])
    return run, h, repo


def test_a_fresh_run_records_a_fingerprint(app):
    run = app["harness"].create_run("新增一个模块", kind="feature")
    stored = app["repo"].get_run(run["id"])
    assert stored["identity_hash"]
    assert app["harness"].resume_status(stored)[0] == RESUME_VALID


def test_resume_is_refused_when_a_key_file_changed_behind_the_run(app):
    run, h, repo = _run_with_plan(app, [{"path": "app.py", "action": "modify"}])
    assert h.resume_status(repo.get_run(run["id"]))[0] == RESUME_VALID

    (app["sandbox"] / "app.py").write_text("def compute(x):\n    return 0\n",
                                          encoding="utf-8")
    repo.update_run(run["id"], status="running")
    result = asyncio.run(h.resume(run["id"]))
    assert result["status"] == "failed"
    assert "恢复被拒绝" in (result["error"] or "")
    assert RESUME_WORKSPACE_DRIFT in (result["error"] or "")


def test_force_stale_overrides_and_the_override_is_recorded(app):
    run, h, repo = _run_with_plan(app, [{"path": "app.py", "action": "modify"}])
    # Valid Python on purpose: this test is about the fingerprint override, and
    # a syntax error would detour through the build-check schema mismatch noted
    # in .claude/history.md.
    (app["sandbox"] / "app.py").write_text("def compute(x):\n    return x * 7\n",
                                          encoding="utf-8")

    before = len(repo.transitions(run["id"]))
    result = asyncio.run(h.resume(run["id"], force_stale=True))

    assert result["status"] != "failed"
    assert repo.get_run(run["id"])["payload"]["resume_status"] == RESUME_WORKSPACE_DRIFT
    assert repo.get_run(run["id"])["payload"]["force_stale"] is True
    assert any("显式接受过期状态" in (t["reason"] or "")
               for t in repo.transitions(run["id"])[before:])


def test_resume_is_refused_when_the_environment_changed(app):
    run, h, repo = _run_with_plan(app, [{"path": "app.py", "action": "modify"}])
    for agent in h.agents.values():
        agent.llm = _OtherBrain()
    repo.update_run(run["id"], status="running")

    result = asyncio.run(h.resume(run["id"]))
    assert result["status"] == "failed"
    assert RESUME_IDENTITY_MISMATCH in (result["error"] or "")


def test_key_file_baseline_does_not_advance_with_every_step(app):
    """If the baseline moved forward each step, drift mid-run would be absorbed
    into it and never detected."""
    run, h, repo = _run_with_plan(app, [{"path": "app.py", "action": "modify"}])
    baseline = dict(repo.get_run(run["id"])["key_files"])

    (app["sandbox"] / "app.py").write_text("drifted mid-run\n", encoding="utf-8")
    h._record_fingerprint(run["id"])              # as a later step would

    assert repo.get_run(run["id"])["key_files"] == baseline
    assert h.resume_status(repo.get_run(run["id"]))[0] == RESUME_WORKSPACE_DRIFT


# ---------------------------------------------------- finished runs are records
@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
def test_resume_leaves_a_finished_run_exactly_as_it_was(app, status):
    """Asking to resume a finished run must not rewrite it.

    The fingerprint verdict below ends in "mark this run failed", and it used to
    run before any check on the run's status — so `resume` on a `completed` run
    flipped it to `failed`. A drift is set up here on purpose: even a genuine
    mismatch must not touch a finished record, because there is nothing left to
    resume and no decision left to make.
    """
    run, h, repo = _run_with_plan(app, [{"path": "app.py", "action": "modify"}])
    repo.update_run(run["id"], status=status, error="原始错误" if status == "failed" else None)
    (app["sandbox"] / "app.py").write_text("drifted after the run ended\n",
                                          encoding="utf-8")
    before = repo.get_run(run["id"])
    transitions_before = len(repo.transitions(run["id"]))

    result = asyncio.run(h.resume(run["id"]))

    after = repo.get_run(run["id"])
    assert result["status"] == status
    assert after["status"] == status
    assert after["error"] == before["error"]
    assert "resume_status" not in (after["payload"] or {})
    assert len(repo.transitions(run["id"])) == transitions_before


def test_resume_still_refuses_a_drift_on_an_unfinished_run(app):
    """The guard above must not weaken the contract it sits in front of."""
    run, h, repo = _run_with_plan(app, [{"path": "app.py", "action": "modify"}])
    repo.update_run(run["id"], status="running")
    (app["sandbox"] / "app.py").write_text("drifted mid-run\n", encoding="utf-8")

    result = asyncio.run(h.resume(run["id"]))
    assert result["status"] == "failed"
    assert RESUME_WORKSPACE_DRIFT in (result["error"] or "")


# ------------------------------------------- the same gate on the advance path
def _with_a_step(app, files):
    """A run that has a persisted step — the thing the gate is about.

    `_run_with_plan` records a fingerprint but executes nothing, and a run with
    no steps has nothing to reuse, so the gate correctly stays out of its way.
    These tests need the step to exist.
    """
    run, h, repo = _run_with_plan(app, files)
    repo.add_step(run["id"], "req_capture", "investigator", {"ok": True}, status="done")
    assert repo.steps_for_run(run["id"]), "没有落库的步骤，测的就不是这道闸门"
    return run, h, repo


def test_advance_refuses_to_reuse_steps_from_a_changed_environment(app):
    """`resume` always asked this; `advance` never did.

    So any process with any configuration could take over an interrupted run and
    silently continue on steps produced under a different model, project root or
    policy set — the one thing the fingerprint exists to prevent. `wfos approve`
    reaches `advance` by exactly that road.
    """
    run, h, repo = _with_a_step(app, [{"path": "app.py", "action": "modify"}])
    for agent in h.agents.values():
        agent.llm = _OtherBrain()
    repo.update_run(run["id"], status="running")

    result = asyncio.run(h.advance(run["id"]))

    assert result["status"] == "failed"
    assert RESUME_IDENTITY_MISMATCH in (result["error"] or "")


def test_advance_does_not_refuse_a_run_with_nothing_persisted(app):
    """A fresh run has nothing to trust or distrust — the gate must stay out of
    its way, or every run would start by refusing itself."""
    h = app["harness"]
    run = h.create_run("新增一个模块", kind="feature")
    assert app["repo"].steps_for_run(run["id"]) == []
    for agent in h.agents.values():
        agent.llm = _OtherBrain()

    result = asyncio.run(h.advance(run["id"]))

    assert "拒绝复用" not in (result["error"] or "")


def test_an_accepted_override_covers_only_what_it_accepted(app):
    """`--force-stale` accepts a *condition*, not the whole run.

    A run let through while its workspace had drifted, which now also mismatches
    on identity, has met a new condition and needs its own decision. Treating one
    acceptance as blanket permission is how "accepted once" becomes "accepted
    forever" and the contract quietly stops existing.
    """
    run, h, repo = _with_a_step(app, [{"path": "app.py", "action": "modify"}])
    run_id = run["id"]

    # The operator accepted a workspace drift.
    repo.update_run(run_id, payload={"force_stale": True,
                                     "resume_status": RESUME_WORKSPACE_DRIFT})
    # Now the environment changed too — a condition nobody accepted.
    for agent in h.agents.values():
        agent.llm = _OtherBrain()

    refused = h._reuse_refused(repo.get_run(run_id))
    assert RESUME_IDENTITY_MISMATCH in refused, \
        f"换了一个条件后仍然放行：{refused!r}"

    # And the condition that *was* accepted is still covered.
    for agent in h.agents.values():
        agent.llm = MockAdapter()
    repo.update_run(run_id, payload={"force_stale": True,
                                     "resume_status": RESUME_WORKSPACE_DRIFT})
    assert h._reuse_refused(repo.get_run(run_id)) == "", \
        "已接受过的那个条件不该被重复拒绝"


def test_the_advance_gate_does_not_touch_a_terminal_run(app):
    """A finished record is a record. `resume` gained this short-circuit after it
    was caught rewriting history; the advance gate must not reintroduce it."""
    run, h, repo = _run_with_plan(app, [{"path": "app.py", "action": "modify"}])
    repo.update_run(run["id"], status="completed")
    for agent in h.agents.values():
        agent.llm = _OtherBrain()

    result = asyncio.run(h.advance(run["id"]))

    assert result["status"] == "completed"
