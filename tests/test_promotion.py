"""P6: turning a candidate into a production skill version, and back again.

One test per claim in the phase's §13 list, because the list is the acceptance
criteria and a test named after a claim is the only kind that can fail for the
right reason.

The gate tests build their verdicts directly rather than running a suite. That is
not a shortcut around the gate — the gate *reads a recorded verdict and does not
re-run anything*, which is exactly what makes it re-checkable years later. What
has to be real is the recorded verdict, so the fixtures below are shaped like the
ones a real suite writes (including the `unknown` values a real suite produces for
axes a task makes no claim about). The end-to-end test at the bottom of the file
runs the whole chain for real.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from wfos import events
from wfos.bench import Arm, run_benchmark, write_baseline
from wfos.models import ORIGIN_INTERACTIVE
from wfos.rsi import (
    STATUS_EVALUATING,
    STATUS_FAILED,
    STATUS_NOT_PROMOTABLE,
    STATUS_PASSED,
    STATUS_PROMOTED,
    STATUS_PROPOSED,
    STATUS_REJECTED,
    CandidateSpec,
    evaluate_candidate,
    history,
    promote,
    promotion_gate,
    record_from_row,
    record_proposal,
    rollback,
    transition,
)
from wfos.skills import content_digest

SMOKE = "benchmark/smoke"
TRIGGER = "test_failure"
CANDIDATE = f"skill:{TRIGGER}"


# ------------------------------------------------------------------- fixtures
def _spec(**over) -> CandidateSpec:
    base = {
        "candidate_id": CANDIDATE, "version": 1, "type": "skill",
        "parent_version": 1, "source_experiment": "exp.json",
        "source_runs": ("run-1",), "source_failures": (TRIGGER,),
        "evidence": {"trigger": TRIGGER, "recoveredFrom": "run-1"},
        "proposed_change": {"kind": "skill", "trigger": TRIGGER,
                            "title": "v2 规程", "procedure": "改好的规程",
                            "files": []},
        "rationale": "因为一次真实的恢复",
    }
    base.update(over)
    return CandidateSpec(**base)


def _axes(**over) -> dict:
    """One task's five axes, defaulting to what a clean task reports."""
    base = {"functional": "pass", "safety": "pass", "efficiency": "unknown",
            "quality": "unknown", "operational": "pass"}
    base.update(over)
    return base


def _verdict(tasks=None, *, reason="好") -> dict:
    tasks = (("t0", _axes()),) if tasks is None else tasks
    return {
        "reason": reason,
        "runIds": [f"run-{i}" for i in range(len(tasks))],
        "taskVerdicts": {tid: "pass" for tid, _ in tasks},
        "axes": [{"taskId": tid, "cellId": f"{tid}@default", "axes": axes}
                 for tid, axes in tasks],
        "hardGateFailures": [],
        "comparison": {"ok": True, "counts": {}, "environment": {},
                       "regressions": []},
    }


def _seed_v1(repo, *, trigger=TRIGGER, procedure="v1 规程"):
    return repo.add_skill(trigger, "v1 标题", procedure, "seed-run")


def _settle(repo, record, status, verdict):
    """Walk a candidate to a status legally, so the state machine stays honest."""
    record = transition(repo, record, STATUS_EVALUATING)
    return transition(repo, record, status, verdict=verdict)


def _promotable(app, *, spec=None, verdict=None, seed=True, version=1):
    repo = app["repo"]
    # Only when production is empty. Seeding unconditionally would mint a new
    # skill version on every call, and the gate would then — correctly — refuse
    # the next candidate for naming a parent that is no longer current.
    if seed and repo.latest_skill(TRIGGER) is None:
        _seed_v1(repo)
    record = record_proposal(repo, spec or _spec(version=version))
    settled = _settle(repo, record, STATUS_PASSED, verdict or _verdict())
    return repo, record_from_row(repo.get_candidate(CANDIDATE, settled.version))


# -------------------------------------------------------------------- promotion
def test_a_passed_candidate_becomes_a_new_live_version(app):
    repo, record = _promotable(app)

    outcome = promote(repo, record, actor="rog", reason="因为证据")

    assert outcome["ok"] is True
    skill = outcome["skill"]
    assert skill["trigger"] == TRIGGER
    assert skill["version"] == 2
    assert skill["superseded"] == 0
    assert skill["status"] == "live"
    assert skill["procedure"] == "改好的规程"
    assert skill["parent_id"] == repo.skill_version(TRIGGER, 1)["id"]
    assert repo.get_candidate(CANDIDATE, 1)["status"] == STATUS_PROMOTED


def test_the_skill_version_is_the_parent_plus_one_not_the_candidate_version(app):
    """The candidate's *number* is not the skill's number.

    A candidate's version counts proposals — this is the third one — while a
    skill's version counts what has been in production. The two coincide in the
    happy path and must not be assumed to.
    """
    repo, record = _promotable(app, spec=_spec(version=3, parent_version=1))

    outcome = promote(repo, record, actor="rog")

    assert outcome["ok"] is True
    assert outcome["promotion"]["candidate_version"] == 3
    assert outcome["skill"]["version"] == 2, "正式版本只跟生产走"
    assert outcome["promotion"]["parent_skill_version"] == 1


def test_v1_is_not_overwritten_by_the_promotion(app):
    repo, record = _promotable(app)
    before = repo.skill_version(TRIGGER, 1)

    promote(repo, record, actor="rog")

    after = repo.skill_version(TRIGGER, 1)
    assert after["procedure"] == before["procedure"]
    assert after["title"] == before["title"]
    assert after["digest"] == before["digest"]
    assert after["superseded"] == 1, "v1 仍在，只是不再是 current"
    assert len(repo.skill_versions(TRIGGER)) == 2


def test_the_promotion_record_carries_the_whole_chain(app):
    repo, record = _promotable(app)

    outcome = promote(repo, record, baseline="", actor="rog", reason="因为证据")
    promotion = repo.get_promotion(outcome["promotion"]["promotion_id"])

    assert promotion["candidate_id"] == CANDIDATE
    assert promotion["candidate_version"] == 1
    assert promotion["parent_skill_version"] == 1
    assert promotion["promoted_skill_version"] == 2
    assert promotion["source_experiment"] == "exp.json"
    assert promotion["source_runs"] == ["run-1"]
    assert promotion["actor"] == "rog" and promotion["reason"] == "因为证据"
    assert promotion["created_at"], "时间戳由数据库在写入时给定"
    assert promotion["evaluation"]["axes"], "晋升记录保留它据以判断的判定"
    assert promotion["gate"]["verdict"] == STATUS_PASSED
    # §5's digest pair: what the parent was, and what the child is. Both are
    # written by something other than the row they describe — see the rollback
    # tamper test for why that matters.
    assert promotion["parent_digest"] == repo.skill_version(TRIGGER, 1)["digest"]
    assert promotion["promoted_digest"] == content_digest(repo.skill_version(TRIGGER, 2))


def test_the_recorded_digests_are_the_content_digests(app):
    repo, record = _promotable(app)
    outcome = promote(repo, record, actor="rog")

    assert outcome["promotion"]["promoted_digest"] == content_digest(outcome["skill"])


def test_a_candidate_that_was_never_passed_cannot_be_promoted(app):
    """§8: only `PASSED` promotes — and asking early changes nothing.

    Every one of these is refused *without* moving the candidate: a candidate
    that has not been evaluated has not been rejected either, and recording it as
    such would be writing down a judgement nobody made.
    """
    repo = app["repo"]
    _seed_v1(repo)

    # A fresh proposal per status: a candidate row is append-only, so these are
    # five candidates rather than one row edited five times.
    for version, status in enumerate(
            (STATUS_PROPOSED, STATUS_EVALUATING, STATUS_FAILED,
             STATUS_REJECTED, STATUS_NOT_PROMOTABLE), start=1):
        record = record_proposal(repo, _spec(version=version))
        if status != STATUS_PROPOSED:
            record = transition(repo, record, STATUS_EVALUATING, verdict=_verdict())
        if status not in (STATUS_PROPOSED, STATUS_EVALUATING):
            record = transition(repo, record, status, verdict=_verdict())

        outcome = promote(repo, record, actor="rog")

        assert outcome["ok"] is False, f"{status} 不该被晋升"
        assert outcome["status"] == status, f"{status} 被这次尝试改动了"
        assert repo.get_candidate(CANDIDATE, version)["status"] == status
        assert repo.latest_skill(TRIGGER)["version"] == 1, "生产版本不该动"


def test_a_functional_failure_rejects_the_promotion(app):
    repo, record = _promotable(app, verdict=_verdict(tasks=(("t0", _axes(functional="fail")),)))

    outcome = promote(repo, record, actor="rog")

    assert outcome["ok"] is False and outcome["gate"]["verdict"] == STATUS_REJECTED
    assert repo.get_candidate(CANDIDATE, 1)["status"] == STATUS_REJECTED
    assert repo.latest_skill(TRIGGER)["version"] == 1


def test_a_safety_failure_rejects_the_promotion(app):
    repo, record = _promotable(app, verdict=_verdict(tasks=(("t0", _axes(safety="fail")),)))

    outcome = promote(repo, record, actor="rog")

    assert outcome["gate"]["verdict"] == STATUS_REJECTED
    assert repo.latest_skill(TRIGGER)["version"] == 1


def test_one_task_failing_offsets_no_other_task_passing(app):
    """`fail` wins over `pass` on the same axis, whichever order they arrive in."""
    tasks = (("t0", _axes(safety="fail")), ("t1", _axes(safety="pass")))
    for version, ordered in enumerate((tasks, tuple(reversed(tasks))), start=1):
        repo, record = _promotable(app, verdict=_verdict(tasks=ordered),
                                   version=version)
        assert promote(repo, record, actor="rog")["gate"]["verdict"] == STATUS_REJECTED
        assert repo.latest_skill(TRIGGER)["version"] == 1


def test_a_hard_gate_failure_rejects_the_promotion(app):
    """§2's "critical failure": no axis average makes a tripped gate promotable."""
    verdict = _verdict()
    verdict["hardGateFailures"] = [["writesOutsidePlan", "3 个文件在计划之外"]]
    repo, record = _promotable(app, verdict=verdict)

    outcome = promote(repo, record, actor="rog")

    assert outcome["gate"]["verdict"] == STATUS_REJECTED
    assert "临界失败" in outcome["gate"]["reasons"][0]
    assert repo.latest_skill(TRIGGER)["version"] == 1


def test_an_unknown_axis_is_not_a_pass(app):
    """§2: "unknown 不得自动视为 pass. 没有足够证据不能伪造为通过"."""
    repo, record = _promotable(app, verdict=_verdict(tasks=(("t0", _axes(safety="unknown")),)))

    outcome = promote(repo, record, actor="rog")

    assert outcome["ok"] is False
    assert outcome["gate"]["verdict"] == STATUS_NOT_PROMOTABLE
    assert "unknown" in " ".join(outcome["gate"]["reasons"])
    assert repo.latest_skill(TRIGGER)["version"] == 1


def test_an_axis_no_task_spoke_to_is_unknown_not_pass(app):
    """An axis absent from every task is unmeasured, and unmeasured is not pass."""
    repo, record = _promotable(app, verdict=_verdict(tasks=(("t0", {"functional": "pass"}),)))

    outcome = promote(repo, record, actor="rog")

    assert outcome["gate"]["verdict"] == STATUS_NOT_PROMOTABLE
    assert "safety" in outcome["gate"]["reasons"][0]


def test_the_axis_verdict_does_not_depend_on_task_order(app):
    """The regression suite really does produce this shape.

    Four of its tasks check safety and report `pass`; the four resume tasks make
    no safety claim and report `unknown`. A last-one-wins fold made the promotion
    decision turn on which group happened to sort last, which is not a property of
    the candidate.
    """
    measured = ("t0", _axes(safety="pass"))
    silent = ("t1", _axes(safety="unknown"))

    for version, ordered in enumerate(((measured, silent), (silent, measured)), start=1):
        repo, record = _promotable(app, verdict=_verdict(tasks=ordered),
                                   version=version)
        assert promotion_gate(repo, record)["verdict"] == STATUS_PASSED, (
            f"{[t for t, _ in ordered]} 的顺序改变了判定")


def test_incomplete_provenance_is_not_promotable(app):
    """§2: evidence/provenance 不完整 → NOT_PROMOTABLE."""
    for version, missing in enumerate(
            ({"source_runs": ()}, {"source_experiment": ""},
             {"source_failures": ()}, {"evidence": {}},
             {"proposed_change": {}}), start=1):
        repo, record = _promotable(app, spec=_spec(version=version, **missing))
        outcome = promote(repo, record, actor="rog")

        assert outcome["ok"] is False
        assert outcome["gate"]["verdict"] == STATUS_NOT_PROMOTABLE, missing


def test_a_regression_against_the_baseline_rejects_the_promotion(app):
    """§2: critical regression > 0 → REJECT, using the compare verdict as it is.

    The gate does not recompute the comparison — it reads the one the evaluation
    recorded, which is P3's and not a second implementation of it.
    """
    verdict = _verdict()
    verdict["comparison"] = {
        "ok": False, "counts": {"regression": 1}, "environment": {},
        "regressions": [{"subject": "metric.latencyMs",
                         "verdict": "regression", "detail": "慢了 3 倍"}],
    }
    repo, record = _promotable(app, verdict=verdict)

    outcome = promote(repo, record, actor="rog")

    assert outcome["gate"]["verdict"] == STATUS_REJECTED
    assert "回归" in outcome["gate"]["reasons"][0]
    assert repo.latest_skill(TRIGGER)["version"] == 1


def test_a_candidate_with_no_comparison_cannot_be_signed_off_against_a_baseline(app):
    """§2 again: a missing comparison is missing evidence, not a clean sheet."""
    verdict = _verdict()
    verdict["comparison"] = {}
    repo, record = _promotable(app, verdict=verdict)

    outcome = promote(repo, record, baseline="base.json", actor="rog")

    assert outcome["gate"]["verdict"] == STATUS_NOT_PROMOTABLE
    assert repo.latest_skill(TRIGGER)["version"] == 1


def test_a_candidate_about_another_version_is_rejected(app):
    """§2: parent skill version is checked, and the wrong one is not a near miss."""
    repo, record = _promotable(app)
    _seed_v1(repo, procedure="v2 规程")          # production moved to v2

    outcome = promote(repo, record, actor="rog")   # candidate still says v1

    assert outcome["gate"]["verdict"] == STATUS_REJECTED
    assert "母版本不符" in outcome["gate"]["reasons"][0]
    assert repo.latest_skill(TRIGGER)["version"] == 2, "生产版本没有被改动"


def test_a_promotion_touches_no_baseline_no_task_and_no_evaluator_rule(app, tmp_path):
    """§9: the promotion may not move the goalposts it was measured against."""
    root = Path(__file__).resolve().parents[1]
    watched = [root / "wfos" / "eval.py", root / "wfos" / "bench_compare.py",
               root / "benchmark" / "smoke" / "tasks.json",
               root / "benchmark" / "regression" / "tasks.json",
               root / "benchmark" / "challenge" / "tasks.json"]
    before = {p: p.read_bytes() for p in watched}

    baseline = tmp_path / "base.json"
    baseline.write_bytes(b'{"kind": "benchmark-baseline", "sentinel": true}')
    before[baseline] = baseline.read_bytes()

    repo, record = _promotable(app)
    promote(repo, record, baseline=str(baseline), actor="rog")

    for path, content in before.items():
        assert path.read_bytes() == content, f"{path} 被晋升改动了"


def test_a_promoted_candidate_cannot_be_changed_afterwards(app):
    """§8: "Promoted Candidate 不允许再次被修改"."""
    repo, record = _promotable(app)
    promote(repo, record, actor="rog")
    promoted = record_from_row(repo.get_candidate(CANDIDATE, 1))

    for target in (STATUS_EVALUATING, STATUS_PASSED, STATUS_REJECTED,
                   STATUS_NOT_PROMOTABLE, STATUS_FAILED):
        with pytest.raises(ValueError, match="终态"):
            transition(repo, promoted, target)

    # And asking to promote it again does not produce a third version.
    again = promote(repo, promoted, actor="rog")
    assert again["ok"] is False
    assert repo.latest_skill(TRIGGER)["version"] == 2
    assert len(repo.skill_versions(TRIGGER)) == 2


def test_the_promotion_is_recorded_in_the_trace(app):
    repo, record = _promotable(app)
    outcome = promote(repo, record, actor="rog")

    trail = [e for e in repo.events_for_run("") if e["type"] == events.CANDIDATE_STATUS]
    promotion_id = outcome["promotion"]["promotion_id"]

    assert any(e["payload"].get("promotionId") == promotion_id for e in trail), (
        "晋升必须在 append-only trace 里留下痕迹")


# ----------------------------------------------------------------------- rollback
def _promoted(app, **over):
    """A skill with v1 and v2, current = v2, and the promotion record to prove it."""
    repo, record = _promotable(app, **over)
    outcome = promote(repo, record, actor="rog")
    return repo, outcome


def test_rollback_moves_the_pointer_and_keeps_both_versions(app):
    repo, outcome = _promoted(app)
    v2 = repo.skill_version(TRIGGER, 2)

    result = rollback(repo, TRIGGER, 1, actor="rog", reason="线上有问题")

    assert result["ok"] is True
    assert result["current"]["version"] == 1
    assert repo.skill_version(TRIGGER, 1)["superseded"] == 0
    after = repo.skill_version(TRIGGER, 2)
    assert after is not None, "回滚不是删除新版本"
    assert after["superseded"] == 1
    assert after["procedure"] == v2["procedure"] and after["digest"] == v2["digest"]


def test_rollback_writes_a_record_of_what_it_did(app):
    repo, outcome = _promoted(app)

    result = rollback(repo, TRIGGER, 1, actor="rog", reason="线上有问题")
    stored = repo.get_rollback(result["rollback"]["rollback_id"])

    assert stored["trigger"] == TRIGGER
    assert stored["from_version"] == 2 and stored["to_version"] == 1
    assert stored["actor"] == "rog" and stored["reason"] == "线上有问题"
    assert stored["created_at"]
    assert stored["from_digest"] == repo.skill_version(TRIGGER, 2)["digest"]
    assert stored["to_digest"] == repo.skill_version(TRIGGER, 1)["digest"]


def test_rollback_to_a_version_that_does_not_exist_is_refused(app):
    repo, _ = _promoted(app)

    result = rollback(repo, TRIGGER, 7, actor="rog")

    assert result["ok"] is False and "没有正在生效的 v7" in result["error"]
    assert repo.latest_skill(TRIGGER)["version"] == 2


def test_rollback_to_another_skills_version_is_refused(app):
    """A version number is not an identity: §7's "不匹配的 Skill identity"."""
    repo, _ = _promoted(app)
    repo.add_skill("build_error", "别的规程", "别的触发类", "seed-run")

    result = rollback(repo, TRIGGER, 1, actor="rog")
    other = rollback(repo, "build_error", 2, actor="rog")

    assert result["ok"] is True, "本 skill 的 v1 仍然可以回滚"
    assert other["ok"] is False, "另一个 trigger 只有 v1，v2 不存在"


def test_rollback_to_the_version_already_current_is_refused(app):
    repo, _ = _promoted(app)

    result = rollback(repo, TRIGGER, 2, actor="rog")

    assert result["ok"] is False and "已经是 v2" in result["error"]


def test_rollback_refuses_a_version_no_promotion_record_names(app):
    """Fail-closed: with no recorded digest there is nothing to check against."""
    repo = app["repo"]
    _seed_v1(repo)                                # seeded, never promoted
    repo.add_skill(TRIGGER, "v2", "手工写进去的", "manual")
    repo.set_current_skill(TRIGGER, 2)

    result = rollback(repo, TRIGGER, 1, actor="rog")

    assert result["ok"] is False
    assert "没有被任何晋升记录命名过" in result["error"]
    assert repo.latest_skill(TRIGGER)["version"] == 2


def test_rollback_refuses_a_tampered_version(app):
    """§7: 被篡改的历史版本 → 拒绝 rollback.

    The digest a rollback verifies against is the one the *promotion record* wrote
    down, not one read off the row being checked — otherwise editing the row would
    edit its own alibi.
    """
    repo, _ = _promoted(app)
    repo._conn.execute("UPDATE skills SET procedure=? WHERE trigger=? AND version=1",
                       ("被人改过的规程", TRIGGER))
    repo._conn.commit()

    result = rollback(repo, TRIGGER, 1, actor="rog")

    assert result["ok"] is False and "历史被改过" in result["error"]
    assert repo.latest_skill(TRIGGER)["version"] == 2, "仍然停在 v2"


def test_history_shows_both_versions_and_where_the_pointer_is(app):
    repo, outcome = _promoted(app)
    rollback(repo, TRIGGER, 1, actor="rog", reason="线上有问题")

    data = history(repo, TRIGGER)

    assert data["currentVersion"] == 1
    assert [v["version"] for v in data["versions"]] == [1, 2]
    assert len(data["promotions"]) == 1 and len(data["rollbacks"]) == 1


def test_the_gate_can_be_asked_again_without_changing_anything(app):
    """Re-checkability: the gate reads, and reading has no side effects."""
    repo, record = _promotable(app)
    before = repo.get_candidate(CANDIDATE, 1)

    first = promotion_gate(repo, record)
    second = promotion_gate(repo, record)

    assert first == second
    assert repo.get_candidate(CANDIDATE, 1) == before
    assert repo.latest_skill(TRIGGER)["version"] == 1


# ------------------------------------------------------------- the whole chain
def test_the_whole_chain_runs_for_real(app, tmp_path):
    """§14, with nothing stubbed: a real benchmark decides, and a person promotes.

    The chain end to end — v1 in production, a candidate derived from it, the
    candidate actually run through the suite and compared, an explicit promotion,
    the new version loading as production, then a rollback that puts the old one
    back while leaving the new one on the shelf.
    """
    from wfos.storage.repo import Repo

    repo = app["repo"]
    v1 = _seed_v1(repo, procedure="v1：先跑一次 check.py 再改")

    baseline = tmp_path / "base.json"
    run = run_benchmark(SMOKE, workspace=tmp_path / "bws", tier="smoke")
    write_baseline(baseline, run, source=SMOKE)
    baseline_bytes = baseline.read_bytes()

    proposed = record_proposal(repo, _spec(parent_version=1))

    settled = evaluate_candidate(
        repo, record_from_row(repo.get_candidate(CANDIDATE, proposed.version)),
        workspace=tmp_path / "cand", suite=SMOKE, baseline=str(baseline))
    assert settled.status == STATUS_PASSED, (
        f"候选没通过自己的评测：{settled.verdict.get('reason')}")

    outcome = promote(repo, record_from_row(repo.get_candidate(CANDIDATE, proposed.version)),
                      baseline=str(baseline), actor="rog", reason="P6 验收")
    assert outcome["ok"] is True, outcome.get("gate")
    promotion_record = repo.get_promotion(outcome["promotion"]["promotion_id"])

    # v2 is production: a prompt built now gets v2, and v1 is history, not gone.
    live = repo.skills_for([TRIGGER], origin=ORIGIN_INTERACTIVE)
    assert [s["version"] for s in live] == [2]
    assert live[0]["procedure"] == "改好的规程"
    assert repo.skill_version(TRIGGER, 1)["procedure"] == v1["procedure"]

    # 晋升记录引用的是**它据以判断的那些评测记录**，而且是 (id, 库) 成对的：一套
    # 隔离运行的每个任务写自己的库，只记 id 会指向八个不相干的第一行。
    cited = promotion_record["evaluation_ids"]
    assert cited, "晋升记录必须引用它据以判断的评测"
    assert all(isinstance(ref, dict) and ref.get("id") and ref.get("store")
               for ref in cited), cited
    for ref in cited:
        store = Repo(ref["store"])
        assert store.conn.execute("SELECT 1 FROM evaluations WHERE id=?",
                                  (ref["id"],)).fetchone() is not None,             f"引用的评测记录 {ref} 在它自己那个库里查不到"

    # And v2 really runs: an arm built from the promoted row is what the suite
    # executes, and it reaches the same verdicts the pre-promotion run did. (An
    # isolated run numbers skills in its own store, so the *text* in play is the
    # thing to look at, not the number.)
    after = run_benchmark(SMOKE, workspace=tmp_path / "after", tier="smoke",
                          arm=Arm(skills=(repo.skill_version(TRIGGER, 2),)))
    assert all(TRIGGER in r.skills for r in after.tasks)
    assert ([r.evaluation.verdict for r in after.tasks]
            == [r.evaluation.verdict for r in run.tasks])

    result = rollback(repo, TRIGGER, 1, actor="rog", reason="验收回滚")
    assert result["ok"] is True
    assert [s["version"] for s in
            repo.skills_for([TRIGGER], origin=ORIGIN_INTERACTIVE)] == [1]

    # Everything the chain produced is still there, and the baseline never moved.
    assert repo.skill_version(TRIGGER, 2) is not None
    assert repo.get_candidate(CANDIDATE, 1)["status"] == STATUS_PROMOTED
    assert len(repo.promotions_for_skill(TRIGGER)) == 1
    assert len(repo.rollbacks_for(TRIGGER)) == 1
    assert baseline.read_bytes() == baseline_bytes


def test_a_candidate_cannot_run_its_own_promotion(app):
    """§1: the promotion is a call a person makes, not a side effect of passing.

    Stated as a property of the code rather than of a run: nothing in the
    evaluation path names `promote`, so `PASSED` is as far as a candidate gets on
    its own.
    """
    import inspect

    from wfos import rsi

    for name in ("analyse", "evaluate_candidate", "record_proposal", "transition"):
        source = inspect.getsource(getattr(rsi, name))
        assert "promote(" not in source, f"{name} 自己调用了晋升"
