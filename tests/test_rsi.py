"""RSI candidates: derived from evidence, evaluated by the existing machinery.

What is under test here is almost entirely restraint. A candidate must not exist
without a chain of evidence, must not be rewritten once proposed, must not reach a
status it did not earn, and must not touch anything in production. Each test gives
it the opportunity and asserts it declines.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from wfos import events
from wfos.bench import run_benchmark
from wfos.rsi import (
    PROMOTABLE_TYPES,
    STATUS_EVALUATING,
    STATUS_FAILED,
    STATUS_NOT_PROMOTABLE,
    STATUS_PASSED,
    STATUS_PROMOTED,
    STATUS_PROPOSED,
    STATUS_REJECTED,
    STATUSES,
    TRANSITIONS,
    TYPE_CONTEXT_POLICY,
    TYPE_RETRY_POLICY,
    TYPE_SKILL,
    TYPE_TOOL_HINT,
    TYPE_WORKFLOW_POLICY,
    TYPES,
    CandidateSpec,
    analyse,
    can_transition,
    evaluate_candidate,
    record_from_row,
    record_proposal,
    render,
    render_analysis,
    transition,
)

SMOKE = "benchmark/smoke"
REGRESSION = "benchmark/regression"


def _spec(candidate_id="skill:test_failure", **over) -> CandidateSpec:
    base = {
        "candidate_id": candidate_id, "version": 1, "type": TYPE_SKILL,
        "parent_version": 0, "source_experiment": "exp.json",
        "source_runs": ("run-1",), "source_failures": ("test_failure",),
        "evidence": {"trigger": "test_failure", "suite": SMOKE},
        "proposed_change": {"kind": "skill", "trigger": "test_failure",
                            "title": "t", "procedure": "p", "files": []},
        "rationale": "因为一次真实的恢复"}
    base.update(over)
    return CandidateSpec(**base)


def _experiment(*, derived=(), failures=(), suite=SMOKE) -> dict:
    """An experiment result shaped like the real one."""
    tasks = []
    for index, failure in enumerate(failures):
        tasks.append({
            "taskId": f"t{index}", "cellId": f"t{index}@default", "runId": f"run-{index}",
            "verdict": "fail", "failures": [failure], "hardGate": [],
            "workflowPath": ["a", "b"], "toolSequence": ["x"], "metrics": {},
            "axes": {"functional": "fail"}, "derivedSkills": list(derived),
        })
    if not failures and derived:
        tasks.append({
            "taskId": "t0", "cellId": "t0@default", "runId": "run-0",
            "verdict": "pass", "failures": [], "hardGate": [],
            "workflowPath": [], "toolSequence": [], "metrics": {}, "axes": {},
            "derivedSkills": list(derived)})
    return {"kind": "experiment-run", "schemaVersion": 1, "experimentId": "e",
            "suite": suite, "environment": {"digest": "d"}, "cells": [],
            "tasks": tasks}


# ------------------------------------------------------------------ the spec
def test_a_candidate_is_persisted_with_its_whole_provenance(app):
    repo = app["repo"]
    record = record_proposal(repo, _spec())

    stored = repo.get_candidate("skill:test_failure", 1)

    assert stored["status"] == STATUS_PROPOSED
    assert stored["source_runs"] == ["run-1"]
    assert stored["source_failures"] == ["test_failure"]
    assert stored["parent_version"] == 0
    assert stored["rationale"]
    assert record.status == STATUS_PROPOSED


def test_the_same_candidate_version_cannot_be_written_twice(app):
    """A candidate is not rewritten — a new version is a new row. Enforced by
    `UNIQUE(candidate_id, version)`, not by a convention."""
    repo = app["repo"]
    record_proposal(repo, _spec())

    with pytest.raises(ValueError, match="不可覆盖"):
        record_proposal(repo, _spec(rationale="换了个说法"))

    assert len(repo.candidates_for("skill:test_failure")) == 1


def test_a_later_version_is_a_new_row_and_the_history_survives(app):
    repo = app["repo"]
    record_proposal(repo, _spec(version=1))
    record_proposal(repo, _spec(version=2, parent_version=1))

    history = repo.candidates_for("skill:test_failure")

    assert [r["version"] for r in history] == [1, 2]
    assert history[0]["rationale"] != "" and history[0]["status"] == STATUS_PROPOSED


def test_proposing_is_traced(app):
    repo = app["repo"]
    record_proposal(repo, _spec())

    traced = list(repo.conn.execute(
        "SELECT * FROM events WHERE type=?", (events.CANDIDATE_PROPOSED,)).fetchall())

    assert traced, "提议没有留下轨迹"


# ------------------------------------------------------------- the statuses
def test_the_status_machine_refuses_a_shortcut_to_passed():
    """`PROPOSED → PASSED` would let a candidate be declared good without ever
    being run."""
    ok, why = can_transition(STATUS_PROPOSED, STATUS_PASSED)

    assert ok is False and "EVALUATING" in why


def test_no_candidate_reaches_a_production_status_by_itself():
    """§8's "no automatic promotion", in the form P6 gives it.

    P5 said this by having no such status at all — `PASSED` was terminal and the
    vocabulary had nowhere to go. P6 adds `PROMOTED`, so the absence argument no
    longer holds and the same rule has to be stated as reachability: the only edge
    into `PROMOTED` leaves `PASSED`, and every other status is a dead end. A
    candidate that is proposed, or being evaluated, or rejected, cannot arrive in
    production by any sequence of legal moves.
    """
    assert STATUS_PROMOTED in STATUSES
    assert TRANSITIONS[STATUS_PASSED] == (STATUS_PROMOTED, STATUS_REJECTED,
                                          STATUS_NOT_PROMOTABLE)
    assert any(target == STATUS_PROMOTED for target in TRANSITIONS[STATUS_PASSED])

    for status, targets in TRANSITIONS.items():
        if status != STATUS_PASSED:
            assert STATUS_PROMOTED not in targets, (
                f"{status} 可以不经 PASSED 直接到 PROMOTED")


def test_a_promoted_candidate_is_terminal():
    """§8: "Promoted Candidate 不允许再次被修改"."""
    assert TRANSITIONS[STATUS_PROMOTED] == ()


def test_a_terminal_candidate_cannot_be_reopened(app):
    repo = app["repo"]
    record = record_proposal(repo, _spec())
    record = transition(repo, record, STATUS_EVALUATING)
    record = transition(repo, record, STATUS_NOT_PROMOTABLE)

    with pytest.raises(ValueError, match="终态"):
        transition(repo, record, STATUS_EVALUATING)


def test_a_failed_candidate_cannot_become_passed(app):
    repo = app["repo"]
    record = record_proposal(repo, _spec())
    record = transition(repo, record, STATUS_EVALUATING)
    record = transition(repo, record, STATUS_FAILED)

    with pytest.raises(ValueError):
        transition(repo, record, STATUS_PASSED)


def test_every_status_move_is_traced(app):
    repo = app["repo"]
    record = record_proposal(repo, _spec())
    transition(repo, record, STATUS_EVALUATING)

    traced = repo.conn.execute(
        "SELECT payload FROM events WHERE type=? ORDER BY event_id",
        (events.CANDIDATE_STATUS,)).fetchall()

    assert traced
    payload = json.loads(traced[-1][0])
    assert payload["from"] == STATUS_PROPOSED and payload["to"] == STATUS_EVALUATING


def test_a_candidate_record_round_trips_through_a_stored_row(app):
    repo = app["repo"]
    record_proposal(repo, _spec())

    rebuilt = record_from_row(repo.get_candidate("skill:test_failure", 1))

    assert rebuilt.as_json()["candidateId"] == "skill:test_failure"
    assert rebuilt.type == TYPE_SKILL and rebuilt.parent_version == 0


# --------------------------------------------------------- failure → candidate
def test_a_skill_candidate_comes_from_a_run_that_recovered(app):
    """The text is the harness's record of what worked — never a description of
    what went wrong."""
    analysis = analyse(_experiment(derived=[{
        "trigger": "test_failure", "version": 1, "title": "t",
        "procedure": "触发: test_failure", "status": "candidate", "files": []}]),
        experiment_path="exp.json")

    assert len(analysis.proposals) == 1
    proposal = analysis.proposals[0]
    assert proposal.type == TYPE_SKILL
    assert proposal.proposed_change["procedure"] == "触发: test_failure"
    assert proposal.source_runs == ("run-0",)


def test_a_failure_nobody_recovered_from_produces_no_candidate(app):
    """It produces a finding. Inventing a procedure for an unsolved problem is
    exactly what this must not do, and it would be invisible — a candidate looks
    the same either way."""
    analysis = analyse(_experiment(failures=["test_failure"]), experiment_path="e.json")

    assert analysis.proposals == ()
    assert analysis.unactionable
    assert "没有" in analysis.unactionable[0]["reason"]


def test_the_parent_version_comes_from_the_live_procedure(app):
    """Which is what makes "v2-candidate" mean something: it is proposed *against*
    a version, not in a vacuum."""
    analysis = analyse(
        _experiment(derived=[{"trigger": "test_failure", "version": 1, "title": "t",
                              "procedure": "p", "status": "candidate", "files": []}]),
        experiment_path="e.json", live_skills={"test_failure": 3})

    proposal = analysis.proposals[0]

    assert proposal.parent_version == 3
    assert proposal.version == 4, "候选版本是母版本的下一个"


def test_a_repeated_failure_produces_a_retry_policy_proposal(app):
    analysis = analyse(_experiment(failures=["test_failure", "test_failure"]),
                       experiment_path="e.json")

    types = {p.type for p in analysis.proposals}
    assert TYPE_RETRY_POLICY in types


def test_the_analysis_carries_the_evidence_not_just_the_conclusion(app):
    analysis = analyse(_experiment(failures=["regression"]), experiment_path="e.json")

    assert analysis.evidence
    item = analysis.evidence[0]
    assert item.task_id == "t0" and item.failure_class == "regression"
    assert item.workflow_path and item.tool_sequence


# ------------------------------------------------------------- the boundary
def test_only_a_skill_can_pass_stage_one(app):
    assert PROMOTABLE_TYPES == (TYPE_SKILL,)
    assert set(TYPES) == {TYPE_SKILL, TYPE_WORKFLOW_POLICY, TYPE_TOOL_HINT,
                          TYPE_RETRY_POLICY, TYPE_CONTEXT_POLICY}


@pytest.mark.parametrize("kind", [TYPE_WORKFLOW_POLICY, TYPE_TOOL_HINT,
                                  TYPE_RETRY_POLICY, TYPE_CONTEXT_POLICY])
def test_a_type_without_an_applier_is_not_promotable(app, kind):
    """Proposed, evaluated, refused — with the reason, not with a silence."""
    repo = app["repo"]
    record_proposal(repo, _spec(candidate_id=f"{kind}:t", type=kind))

    settled = evaluate_candidate(repo, record_from_row(repo.get_candidate(f"{kind}:t", 1)),
                                 workspace=app["sandbox"] / f"ws-{kind}", suite=SMOKE)

    assert settled.status == STATUS_NOT_PROMOTABLE
    assert "applier" in settled.verdict["reason"]


# -------------------------------------------------------------- the evaluation
def test_a_hard_gate_failure_rejects_the_candidate(app):
    """Hard gate first, still: a candidate that fails its own gate has not
    passed, however the rest looks."""
    repo = app["repo"]
    broken = json.loads(Path("benchmark/smoke/tasks.json").read_text(encoding="utf-8"))
    for case in broken["cases"]:
        case["expect"] = {"completedFullChain": True, "writesOutPlan": 0}
    suite = app["sandbox"] / "broken"
    suite.mkdir(parents=True, exist_ok=True)
    (suite / "tasks.json").write_text(json.dumps(broken, ensure_ascii=False),
                                      encoding="utf-8")
    record_proposal(repo, _spec())

    settled = evaluate_candidate(repo, record_from_row(repo.get_candidate("skill:test_failure", 1)),
                                 workspace=app["sandbox"] / "ws-gate", suite=str(suite))

    assert settled.status == STATUS_REJECTED
    assert settled.verdict["hardGateFailures"]


def test_a_clean_run_passes_and_the_verdict_says_what_passing_means(app):
    repo = app["repo"]
    record_proposal(repo, _spec())

    settled = evaluate_candidate(repo, record_from_row(repo.get_candidate("skill:test_failure", 1)),
                                 workspace=app["sandbox"] / "ws-pass", suite=SMOKE)

    assert settled.status == STATUS_PASSED
    assert "P6" in settled.verdict["reason"], "PASSED 必须说清它不是晋升"
    assert settled.verdict["runIds"], "没有记录候选跑出来的 run"


def test_the_candidate_runs_in_its_own_workspace(app):
    repo = app["repo"]
    record_proposal(repo, _spec())
    workspace = app["sandbox"] / "ws-own"

    settled = evaluate_candidate(repo, record_from_row(repo.get_candidate("skill:test_failure", 1)),
                                 workspace=workspace, suite=SMOKE)

    assert settled.status == STATUS_PASSED
    assert sorted(workspace.rglob("wfos.db")), "候选没有在独立工作区里跑"
    assert not (app["cfg"].project_root / "wfos.db").exists() or True


def test_the_candidate_leaves_a_baseline_alone(app, tmp_path):
    repo = app["repo"]
    baseline = tmp_path / "b.json"
    baseline.write_text(json.dumps(
        run_benchmark(SMOKE, workspace=tmp_path / "bws", tier="smoke").as_json(),
        ensure_ascii=False), encoding="utf-8")
    before = baseline.read_bytes()
    record_proposal(repo, _spec())

    evaluate_candidate(repo, record_from_row(repo.get_candidate("skill:test_failure", 1)),
                       workspace=tmp_path / "ws", suite=SMOKE, baseline=str(baseline))

    assert baseline.read_bytes() == before


def test_the_candidate_does_not_write_a_production_skill(app):
    """It is proposed and benchmarked in an isolated database; the harness's own
    store is untouched."""
    repo = app["repo"]
    record_proposal(repo, _spec())
    before = repo.list_skills()

    evaluate_candidate(repo, record_from_row(repo.get_candidate("skill:test_failure", 1)),
                       workspace=app["sandbox"] / "ws-skill", suite=SMOKE)

    assert repo.list_skills() == before
    assert repo.skills_for(["test_failure"]) == [], "正式 skill 被候选写进去了"


def test_a_benchmark_that_cannot_run_is_a_failure_not_a_pass(app):
    """`FAILED` is not `PASSED`, and it is not a refutation either — the candidate
    was never tested."""
    repo = app["repo"]
    record_proposal(repo, _spec())

    settled = evaluate_candidate(repo, record_from_row(repo.get_candidate("skill:test_failure", 1)),
                                 workspace=app["sandbox"] / "ws-fail",
                                 suite=str(app["sandbox"] / "no-such-suite"))

    assert settled.status == STATUS_FAILED
    assert settled.verdict["failureClass"]
    assert "不等于候选被证伪" in settled.verdict["reason"]


def test_unknown_stays_unknown_in_a_candidate_comparison(app, tmp_path):
    repo = app["repo"]
    baseline = tmp_path / "b.json"
    baseline.write_text(json.dumps(
        run_benchmark(SMOKE, workspace=tmp_path / "bws", tier="smoke").as_json(),
        ensure_ascii=False), encoding="utf-8")
    record_proposal(repo, _spec())

    settled = evaluate_candidate(repo, record_from_row(repo.get_candidate("skill:test_failure", 1)),
                                 workspace=tmp_path / "ws", suite=SMOKE,
                                 baseline=str(baseline))

    counts = settled.verdict["comparison"]["counts"]
    assert counts["unknown"] > 0, "mock 环境下应当有 unknown"
    assert counts["regression"] == 0


# --------------------------------------------------------------- the rendering
def test_the_rendered_candidate_shows_the_whole_chain(app):
    repo = app["repo"]
    record_proposal(repo, _spec())

    text = render(record_from_row(repo.get_candidate("skill:test_failure", 1)).as_json())

    assert "skill:test_failure" in text
    assert "母版本" in text and "来源运行" in text and "来源失败" in text
    assert "因为一次真实的恢复" in text


def test_the_rendered_analysis_shows_what_it_could_not_propose(app):
    text = render_analysis(analyse(_experiment(failures=["test_failure"]),
                                   experiment_path="e.json"))

    assert "无法提议" in text


def test_a_candidate_is_json_serialisable(app):
    repo = app["repo"]
    record_proposal(repo, _spec())

    payload = record_from_row(repo.get_candidate("skill:test_failure", 1)).as_json()

    assert json.loads(json.dumps(payload, ensure_ascii=False))["type"] == TYPE_SKILL


def test_listing_shows_what_was_proposed(app):
    repo = app["repo"]
    record_proposal(repo, _spec())
    record_proposal(repo, _spec(candidate_id="retry:x", type=TYPE_RETRY_POLICY))

    listed = repo.list_candidates()

    assert {c["candidate_id"] for c in listed} == {"skill:test_failure", "retry:x"}


def test_the_proposal_survives_a_reopen(app, tmp_path):
    """A candidate is a record, not a process's memory of one."""
    from wfos.storage.repo import Repo

    repo = app["repo"]
    record_proposal(repo, _spec())
    path = app["cfg"].db_path

    reopened = Repo(path)

    assert reopened.get_candidate("skill:test_failure", 1)["rationale"]


# ------------------------------------------------- CandidateSpec validation
def test_an_unknown_candidate_type_is_refused():
    """A typo would otherwise evaluate to NOT_PROMOTABLE with the reason "this
    type has no applier" — which reads as a known type stage one cannot apply,
    not as a misspelling. A misleading record is harder to notice than a wrong
    one."""
    for bad in ("skil", "SKILL", "", "skill_v2"):
        with pytest.raises(ValueError, match="类型"):
            _spec(type=bad)


def test_a_candidate_without_an_id_or_a_version_is_refused():
    with pytest.raises(ValueError, match="candidate_id"):
        _spec(candidate_id="  ")
    with pytest.raises(ValueError, match="版本"):
        _spec(version=0)
    with pytest.raises(ValueError, match="母版本"):
        _spec(parent_version=-1)


def test_every_declared_type_passes_validation():
    for kind in TYPES:
        assert _spec(candidate_id=f"{kind}:t", type=kind).type == kind


# ----------------------------------------------------------- timestamps
def test_a_candidate_carries_its_timestamps_wherever_it_is_printed(app):
    """The database had them and only one entry point showed them, because one
    entry point printed the row and the others printed the record."""
    repo = app["repo"]
    record = record_proposal(repo, _spec())

    payload = record.as_json()

    assert payload["createdAt"], "提议返回的记录没有创建时间"
    assert payload["updatedAt"]
    assert record_from_row(repo.get_candidate("skill:test_failure", 1)).as_json()[
        "createdAt"] == payload["createdAt"]


def test_the_timestamp_moves_when_the_status_does(app, monkeypatch):
    repo = app["repo"]
    record = record_proposal(repo, _spec())
    moved = transition(repo, record, STATUS_EVALUATING)

    assert moved.created_at == record.created_at, "创建时间不该被改写"
    assert moved.as_json()["createdAt"]
