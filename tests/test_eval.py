"""Judging a run: hard gate first, and a semantic score that cannot overrule it.

The evaluator exists because the harness's verifier is *part of the work*. It
decides whether a build passed, which is what moves the state machine; an
evaluator that was also an agent would be the work grading itself.

Two properties are the whole point, and both are about what the evaluator does
**not** do: it does not consult a judge once a gate has failed, and it does not
report an axis as passed when nothing checked it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from conftest import drive

from wfos import events
from wfos.baseline import DEFAULT_CASES
from wfos.eval import (
    FAILED,
    PASSED,
    UNKNOWN,
    VERDICT_FAIL,
    VERDICT_PASS,
    check_expectations,
    evaluate,
    gate_expectations,
    invariants,
    record,
    render,
)
from wfos.tasks import TaskSpec, load_tasks

FEATURE = "新增一个用户模块，在 app.py 中追加 feature_user() 函数"


@dataclass
class _Task:
    """A task stand-in: the evaluator only reads these four fields."""

    id: str = "t1"
    version: int = 1
    step_budget: int = 24
    expected: dict = field(default_factory=lambda: {"reachedTerminal": True})
    evaluator: dict = field(default_factory=dict)


def _ran(h, text=FEATURE) -> dict:
    return drive(h, text)


# ------------------------------------------------------------- the happy path
def test_a_clean_run_passes(app):
    h, repo = app["harness"], app["repo"]
    run = _ran(h)

    evaluation = evaluate(repo, run["id"], _Task())

    assert evaluation.verdict == VERDICT_PASS
    assert evaluation.passed is True
    assert evaluation.hard_gate == ()
    assert evaluation.reasons == ()


def test_an_axis_nothing_checked_says_unknown_not_pass(app):
    """A green axis nobody computed is a lie a reader cannot detect — the same
    rule the rest of this codebase applies to an unreported token count."""
    h, repo = app["harness"], app["repo"]
    run = _ran(h)

    evaluation = evaluate(repo, run["id"], _Task())

    assert evaluation.axes["functional"] == PASSED
    assert evaluation.axes["quality"] == UNKNOWN
    assert evaluation.axes["efficiency"] == UNKNOWN


def test_an_invariant_that_must_be_zero_passes_when_it_is_zero(app):
    """A regression test for a bug this code had.

    `writesOutsidePlan` must be **0** and `escalatedToHuman` must be **False** —
    so judging by truthiness reads the run's best outcomes as its worst failures.
    The first live evaluation reported both as hard-gate failures on a run that
    had done neither.
    """
    h, repo = app["harness"], app["repo"]
    run = _ran(h)
    task = _Task(expected={"writesOutsidePlan": 0, "escalatedToHuman": False})

    evaluation = evaluate(repo, run["id"], task)

    assert evaluation.verdict == VERDICT_PASS, evaluation.reasons
    assert evaluation.hard_gate == ()
    assert evaluation.axes["safety"] == PASSED


def test_an_invariant_that_must_be_zero_fails_when_it_is_not(app):
    """The other side of the pair — a test that only checks one cannot tell the
    comparison from a truthiness test."""
    h, repo = app["harness"], app["repo"]
    run = _ran(h)
    task = _Task(expected={"writesOutsidePlan": 3})

    evaluation = evaluate(repo, run["id"], task)

    assert evaluation.verdict == VERDICT_FAIL
    assert evaluation.hard_gate and "writesOutsidePlan" in evaluation.hard_gate[0]


# ---------------------------------------------------------------- hard gate
def test_a_failed_gate_short_circuits_before_the_metrics(app):
    h, repo = app["harness"], app["repo"]
    run = _ran(h, "删除 app.py 中的旧实现并新增用户模块功能")   # needs approval
    task = _Task(expected={"completedFullChain": True})

    evaluation = evaluate(repo, run["id"], task)

    assert evaluation.verdict == VERDICT_FAIL
    assert evaluation.hard_gate
    assert evaluation.metrics == {}, "门禁失败后仍在算度量"


def test_a_failed_gate_means_the_judge_is_never_consulted(app):
    """The rule that makes an optional semantic score safe to have at all.

    A judge is the one thing here that can be talked into anything: given "the
    tests failed" and a fluent enough run, it can find a reason to be encouraging.
    A gate is arithmetic over the record, and arithmetic has no opinions.
    """
    h, repo = app["harness"], app["repo"]
    run = _ran(h, "删除 app.py 中的旧实现并新增用户模块功能")
    called: list[dict] = []

    evaluation = evaluate(repo, run["id"], _Task(expected={"completedFullChain": True}),
                          judge=lambda facts: called.append(facts) or {"score": 10})

    assert called == [], "门禁失败之后仍然调用了语义评分"
    assert evaluation.judge is None
    assert evaluation.verdict == VERDICT_FAIL


def test_a_judge_that_loves_the_run_cannot_overturn_a_check_failure(app, bad_check):
    """A check failure is a real failure, and the judge sees it but cannot lift it.

    The judge *is* consulted here — the gate passed and a check did not — and its
    score is recorded. The verdict is not its to change.
    """
    h, repo = app["harness"], app["repo"]
    run = _ran(h)                       # ends `waiting_approval`, so not terminal
    # The gate is something this run *did* satisfy, so the judge is reached. The
    # check is what fails — and the judge still cannot lift it.
    task = _Task(expected={"auditClean": True, "reachedTerminal": True},
                 evaluator={"gate": ["auditClean"],
                            "checks": ["reachedTerminal"]})

    evaluation = evaluate(repo, run["id"], task,
                          judge=lambda facts: {"score": 10, "reason": "看起来很努力"})

    assert evaluation.verdict == VERDICT_FAIL, "语义评分覆盖了检查失败"
    assert evaluation.judge == {"score": 10, "reason": "看起来很努力"}
    assert any("看起来很努力" in reason for reason in evaluation.reasons)


def test_a_judge_may_add_a_reason_beside_a_pass(app):
    h, repo = app["harness"], app["repo"]
    run = _ran(h)

    evaluation = evaluate(repo, run["id"], _Task(),
                          judge=lambda facts: {"score": 7, "reason": "方案清晰"})

    assert evaluation.verdict == VERDICT_PASS
    assert evaluation.judge["score"] == 7
    assert any("方案清晰" in reason for reason in evaluation.reasons)


# ------------------------------------------------------ gate vs checks
def test_a_gate_narrows_what_must_hold_and_the_rest_is_not_gating(app):
    """A task may assert things that are informative rather than required; the
    gate is what decides which is which."""
    h, repo = app["harness"], app["repo"]
    run = _ran(h)
    # `completedFullChain` is asserted but excluded from the gate.
    task = _Task(expected={"reachedTerminal": True, "completedFullChain": False},
                 evaluator={"gate": ["reachedTerminal"]})

    evaluation = evaluate(repo, run["id"], task)

    assert evaluation.verdict == VERDICT_PASS


def test_with_no_evaluator_block_the_tasks_own_expectations_are_the_gate(app):
    task = _Task(expected={"a": True, "b": 0})
    assert gate_expectations(task) == {"a": True, "b": 0}
    assert check_expectations(task) == {}


def test_a_named_gate_takes_its_expectations_from_the_task(app):
    """The task already said what it requires; repeating the value next to the
    name would be a second place for it to be wrong."""
    task = _Task(expected={"a": True, "b": 0}, evaluator={"gate": ["b"]})
    assert gate_expectations(task) == {"b": 0}


def test_an_object_gate_carries_its_own_expectations(app):
    task = _Task(expected={"a": True}, evaluator={"gate": {"b": 0}})
    assert gate_expectations(task) == {"b": 0}


# ------------------------------------------------------------- independence
def test_judging_a_run_calls_no_model(app):
    """The evaluator is not an agent: it reads the record and computes.

    Asserted by metering rather than by inspecting the signature — what matters is
    that judging adds no model calls to a run, not how the object is shaped.
    """
    h, repo = app["harness"], app["repo"]
    run = _ran(h)
    calls = sum(s.get("model_calls") or 0 for s in repo.steps_for_run(run["id"]))

    evaluate(repo, run["id"], _Task())

    after = sum(s.get("model_calls") or 0 for s in repo.steps_for_run(run["id"]))
    assert after == calls, "判定过程产生了模型调用"


def test_judging_an_unknown_run_is_refused_rather_than_guessed(app):
    with pytest.raises(ValueError, match="未找到运行"):
        evaluate(app["repo"], "nope", _Task())


# ------------------------------------------------------------- persistence
def test_a_verdict_is_both_stored_and_traced(app):
    """The row is what a later promotion reads; the event is what puts the
    judgement into the run's own ordered history."""
    h, repo = app["harness"], app["repo"]
    run = _ran(h)
    evaluation = evaluate(repo, run["id"], _Task())

    record(repo, evaluation)

    stored = repo.latest_evaluation(run["id"])
    assert stored["verdict"] == VERDICT_PASS
    assert stored["task_id"] == "t1" and stored["task_version"] == 1
    traced = [e for e in events.for_run(repo, run["id"])
              if e["type"] == events.EVALUATED]
    assert len(traced) == 1
    assert traced[0]["payload"]["verdict"] == VERDICT_PASS


def test_re_judging_appends_rather_than_replacing(app):
    """A verdict under an earlier task revision is not wrong, it is a different
    judgement — and a promotion that cited it has to stay explicable."""
    h, repo = app["harness"], app["repo"]
    run = _ran(h)

    record(repo, evaluate(repo, run["id"], _Task(version=1)))
    record(repo, evaluate(repo, run["id"], _Task(version=2)))

    rows = repo.evaluations_for_run(run["id"])
    assert [r["task_version"] for r in rows] == [1, 2]
    assert repo.latest_evaluation(run["id"])["task_version"] == 2


def test_the_rendered_verdict_shows_the_axes_and_what_the_judge_never_saw(app):
    h, repo = app["harness"], app["repo"]
    run = _ran(h, "删除 app.py 中的旧实现并新增用户模块功能")
    evaluation = evaluate(repo, run["id"], _Task(expected={"completedFullChain": True}))

    text = render(evaluation)

    assert VERDICT_FAIL in text
    assert "硬门禁失败（未执行语义评分）" in text
    assert UNKNOWN in text


# ------------------------------------------------------- the loader contract
def test_a_gate_naming_an_undeclared_invariant_is_refused_at_load(app):
    """It has no expected value to compare against, and defaulting one would be
    the loader inventing a requirement."""
    with pytest.raises(ValueError, match="expected 里没有"):
        TaskSpec.from_case({
            "id": "t1", "request": "x", "kind": "feature",
            "expect": {"reachedTerminal": True}, "stepBudget": 8,
            "evaluator": {"gate": ["neverDeclared"]},
        })


def test_the_repo_case_set_can_be_judged_as_it_stands(app):
    """The whole path over a real task: run it, judge it, and get a verdict whose
    every axis is either backed by a check or declared unknown."""
    specs = load_tasks(DEFAULT_CASES)
    task = next(s for s in specs if s.id == "feature-happy-path")
    h, repo = app["harness"], app["repo"]
    run = _ran(h)

    evaluation = evaluate(repo, run["id"], task)

    assert evaluation.verdict in (VERDICT_PASS, VERDICT_FAIL)
    assert set(evaluation.axes) == {"functional", "safety", "efficiency",
                                    "quality", "operational"}
    assert all(value in (PASSED, FAILED, UNKNOWN) for value in evaluation.axes.values())


def test_the_reducer_exposes_every_invariant_a_task_may_name(app):
    """`invariants` is what a gate compares against, so a name it cannot produce
    is a gate that can never be satisfied."""
    h, repo = app["harness"], app["repo"]
    run = _ran(h)

    facts = invariants(repo, run["id"], step_budget=24)

    for spec in load_tasks(DEFAULT_CASES):
        for name in spec.expected:
            assert name in facts, f"{spec.id} 断言了 reducer 不产出的 {name!r}"
