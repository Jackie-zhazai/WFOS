"""Judging a run: what it did, measured against what the task required.

The evaluator is **independent of the agent that did the work**, and that is not a
style preference. The harness already has a verifier agent, and the verifier is
part of the work — it decides whether a build passed, which is what moves the
state machine. An evaluator that was also an agent would be the work grading
itself, and the grade would be worth exactly as much as the work's own opinion.

**Hard gate first, and it short-circuits.** The order is gate → deterministic
checks → metrics → optional semantic judge, and a judge is never consulted once a
gate has failed. That rule exists because a semantic score is the one thing here
that can be talked into anything: given "the tests failed" and a fluent enough
run, a judge can find a reason to be encouraging. A gate is arithmetic over the
record, and arithmetic does not have opinions.

**The axes report `unknown`, not `pass`.** An axis nothing checked is not an axis
that passed — the same rule the rest of this codebase applies to an unreported
token count. A green axis nobody computed is a lie a reader cannot detect.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from . import events
from .failures import FAILURE_CLASSES
from .metrics import run_metrics
from .models import TERMINAL_STATUSES
from .storage.repo import Repo

# Error codes that mean the agent loop misbehaved rather than the task failing:
# a tool the agent was never given, or the same call re-issued verbatim.
_AUDIT_VIOLATIONS = ("unknown_tool", "repeated_identical_call")


def resume_status(run: dict) -> str:
    """The resume verdict, read from wherever the harness recorded it.

    A *refused* resume stops before the payload is stamped — the refusal is
    itself the terminal state — so on that path the verdict exists only inside
    the recorded error. Reporting it as absent would make the most important
    outcome (a resume that was correctly refused) look like no resume happened.
    """
    recorded = (run.get("payload") or {}).get("resume_status")
    if recorded:
        return str(recorded)
    error = run.get("error") or ""
    for status in ("identity-mismatch", "workspace-drift", "no-fingerprint"):
        if status in error:
            return status
    return ""


def invariants(repo: Repo, run_id: str, *, step_budget: int,
               child_run_id: str | None = None) -> dict[str, Any]:
    """Every scalar property a whole-flow case can be judged on.

    All values are scalars (bool / int / str) so the artifact diffs cleanly: a
    nested structure would make "this invariant moved" hard to see.
    """
    run = repo.get_run(run_id) or {}
    steps = repo.steps_for_run(run_id)
    states = [s["state"] for s in steps]
    tool_rows = repo.tool_calls(run_id)
    approvals = repo.list_approvals(run_id)
    observed = repo.affected_paths_for_run(run_id, agent="implementer")

    plan_files = [f["path"] for f in ((run.get("payload") or {}).get("plan") or {}).get("files") or []
                  if isinstance(f, dict) and f.get("path")]
    plan_norm = {p.replace("\\", "/").lstrip("./") for p in plan_files}
    outside = [p for p in observed
               if p.replace("\\", "/").lstrip("./") not in plan_norm]

    implement = repo.get_step(run_id, "implement") or {}
    deviation = (implement.get("output_json") or {}).get("plan_deviation") or {}
    prompt_metas = [(s.get("input_json") or {}).get("prompt_metadata") or {}
                    for s in steps]
    prompt_metas = [m for m in prompt_metas if m]

    children = repo.child_runs(run_id)
    if child_run_id and not any(c["id"] == child_run_id for c in children):
        children = children + [repo.get_run(child_run_id) or {}]

    return {
        "terminalStatus": run.get("status", ""),
        "finalState": run.get("state", ""),
        "stepsExecuted": len(steps),
        "stepsWithinBudget": len(steps) <= step_budget,
        "noStateRanTwice": len(states) == len(set(states)),
        "leaseReleased": run.get("owner") is None and run.get("lease_until") is None,
        "promptWithinBudget": bool(prompt_metas) and all(
            not m.get("over_budget", False) for m in prompt_metas),
        # Memory is what earlier runs left behind on these files. `injected` says
        # whether it reached a prompt at all; `withheld` says how much was held
        # back because its files had moved since — which has to stay visible, or
        # an empty memory section reads as "nothing was ever recorded".
        "memoryInjected": any(int((m.get("memory") or {}).get("injected") or 0) > 0
                              for m in prompt_metas),
        "memoryWithheld": max((int((m.get("memory") or {}).get("stale") or 0)
                               for m in prompt_metas), default=0),
        # Did the run reach a terminal status at all? The frozen mock baseline
        # asserts exact statuses; a live case cannot, so it asserts this instead —
        # which is the honest ceiling of what a non-deterministic brain allows.
        "reachedTerminal": run.get("status") in TERMINAL_STATUSES,
        # A failed run has to say *why*, in the shared vocabulary. This is the
        # live-case form of "failure has a readable classification": before the
        # wire codes existed, every real-provider failure recorded the same
        # catch-all, which is indistinguishable from having no diagnosis at all.
        "failuresAreClassified": (
            run.get("status") != "failed"
            or bool({s["failure_class"] for s in steps if s.get("failure_class")})
            and all(s.get("failure_class") in FAILURE_CLASSES
                    for s in steps if s.get("failure_class"))),
        # Evidence selection: what reached a prompt, and how much the count cap
        # held back. Recorded even though no case asserts it today — an evidence
        # ordering that starts dropping different items is exactly the kind of
        # change that should show up in a diff, not be discovered later.
        "evidenceInjected": any(int((m.get("evidence") or {}).get("injected") or 0) > 0
                                for m in prompt_metas),
        "evidenceDropped": max((int((m.get("evidence") or {}).get("dropped") or 0)
                                for m in prompt_metas), default=0),
        "auditClean": not any(t.get("error_code") in _AUDIT_VIOLATIONS for t in tool_rows),
        "completedFullChain": run.get("status") == "completed" and "knowledge_distill" in states,
        "wroteNothing": not observed,
        "writesOutsidePlan": len(outside),
        "planDeviationReported": bool(deviation.get("missing") or deviation.get("extra")),
        "approvalPending": sum(1 for a in approvals if a["status"] == "pending"),
        "approvalBlocked": (run.get("status") == "waiting_approval" and not observed),
        "escalatedToHuman": any(str(a["action"]).startswith("repeated_failure:")
                                for a in approvals),
        "failureClasses": ",".join(sorted({s["failure_class"] for s in steps
                                           if s.get("failure_class")})),
        "resumeStatus": resume_status(run),
        "resumeRefused": (run.get("status") == "failed"
                          and "恢复被拒绝" in (run.get("error") or "")),
        "childRuns": len(children),
    }


# --------------------------------------------------------------------- the axes
AXIS_FUNCTIONAL = "functional"
AXIS_SAFETY = "safety"
AXIS_EFFICIENCY = "efficiency"
AXIS_QUALITY = "quality"
AXIS_OPERATIONAL = "operational"
AXES = (AXIS_FUNCTIONAL, AXIS_SAFETY, AXIS_EFFICIENCY, AXIS_QUALITY,
        AXIS_OPERATIONAL)

PASSED = "pass"
FAILED = "fail"
# Nothing was checked on this axis. Deliberately not `passed`: a green axis that
# nobody computed is a claim the record does not support, and a reader cannot tell
# it from a real pass.
UNKNOWN = "unknown"

VERDICT_PASS = "pass"
VERDICT_FAIL = "fail"

# Which axis each invariant speaks to. Anything not named here is reported on
# `operational` — "the run itself" — rather than dropped, because a check that
# silently vanished would be a check nobody ever sees fail.
_AXIS_OF = {
    "terminalStatus": AXIS_FUNCTIONAL,
    "finalState": AXIS_FUNCTIONAL,
    "completedFullChain": AXIS_FUNCTIONAL,
    "reachedTerminal": AXIS_FUNCTIONAL,
    "wroteNothing": AXIS_FUNCTIONAL,
    "childRuns": AXIS_FUNCTIONAL,
    "planDeviationReported": AXIS_QUALITY,
    "evidenceInjected": AXIS_QUALITY,
    "memoryInjected": AXIS_QUALITY,
    "auditClean": AXIS_SAFETY,
    "writesOutsidePlan": AXIS_SAFETY,
    "approvalBlocked": AXIS_SAFETY,
    "escalatedToHuman": AXIS_SAFETY,
    "leaseReleased": AXIS_OPERATIONAL,
    "noStateRanTwice": AXIS_OPERATIONAL,
    "failuresAreClassified": AXIS_OPERATIONAL,
}


def axis_of(name: str) -> str:
    """Which axis an invariant speaks to; anything unmapped is `operational`."""
    return _AXIS_OF.get(name, AXIS_OPERATIONAL)


@dataclass(frozen=True)
class Evaluation:
    """A verdict, and everything it was reached from.

    `reasons` says why in sentences a person reads; `axes` says what was checked
    and what was not; `hard_gate` names the gate failures specifically, because
    "which of these did the judge never get to see" is the question a reader has
    when a run failed a gate and a score exists anyway.
    """

    run_id: str
    task_id: str
    task_version: int
    verdict: str
    axes: dict[str, str]
    reasons: tuple[str, ...]
    failures: tuple[str, ...]
    metrics: dict
    artifacts: dict
    hard_gate: tuple[str, ...] = ()
    judge: dict | None = None

    @property
    def passed(self) -> bool:
        return self.verdict == VERDICT_PASS

    def as_json(self) -> dict[str, Any]:
        """The verdict as data, for `--json` and for records that cite it.

        Tuples become lists and nothing is summarized: a caller that reads this
        is entitled to the same `unknown`s and the same `hard_gate` entries the
        run was judged on, not a friendlier version of them.
        """
        return {
            "runId": self.run_id, "taskId": self.task_id,
            "taskVersion": self.task_version, "verdict": self.verdict,
            "passed": self.passed, "axes": dict(self.axes),
            "reasons": list(self.reasons), "failures": list(self.failures),
            "hardGate": list(self.hard_gate), "metrics": dict(self.metrics),
            "artifacts": dict(self.artifacts),
            "judge": self.judge,
        }


def _expectations(task, declared) -> dict[str, Any]:
    """`{invariant: expected value}`, from either spelling the task may use.

    An **object** carries its own expectations. A **list** names invariants whose
    expectations come from the task's `expected` — the task has already declared
    what it requires, and repeating the value next to the name would be a second
    place for it to be wrong. A list naming something `expected` does not declare
    is refused by the loader, not silently ignored here.
    """
    if not declared:
        return {}
    if isinstance(declared, dict):
        return {str(name): value for name, value in declared.items()}
    return {str(name): task.expected[name] for name in declared}


def gate_expectations(task) -> dict[str, Any]:
    """The checks that must hold, and stop everything if they do not.

    `evaluator.gate` narrows it; with no `evaluator` block the task's own
    `expected` **is** the gate. That default is the honest one: a task declares
    what it requires, and a task whose requirements are not gating is a task that
    cannot fail.
    """
    return _expectations(task, (task.evaluator or {}).get("gate")) or dict(task.expected)


def check_expectations(task) -> dict[str, Any]:
    """Non-gating-but-reported checks declared alongside the gate.

    A check that fails still fails the run — it is not decoration. What it does
    *not* do is stop the judge from running, which is the gate's job.
    """
    return _expectations(task, (task.evaluator or {}).get("checks"))


def _failed(expectations: dict[str, Any], facts: dict) -> list[str]:
    """Which expectations did not hold, with what the run actually did.

    Compared **against the expected value**, never against truthiness. The
    invariants are not all "must be true": `writesOutsidePlan` must be **0** and
    `escalatedToHuman` must be **False**, so a truthiness test reads the run's
    best outcomes as its worst failures. The same distinction the rest of this
    codebase draws between an unreported count and a count of zero — here it is
    between a count of zero and a broken invariant.
    """
    return [f"{name}={facts.get(name)!r}（期望 {want!r}）"
            for name, want in expectations.items() if facts.get(name) != want]


def _axes_from(failures: list[str], checked: set[str], facts: dict) -> dict[str, str]:
    """Each axis: failed, passed, or unknown because nothing checked it."""
    failed_axes = {axis_of(entry.split("=", 1)[0]) for entry in failures}
    checked_axes = {axis_of(name) for name in checked}
    return {axis: (FAILED if axis in failed_axes
                   else PASSED if axis in checked_axes
                   else UNKNOWN)
            for axis in AXES}


def evaluate(repo, run_id: str, task, *,
             judge: Callable[[dict], dict] | None = None) -> Evaluation:
    """Judge one run against one task version.

    `judge` is an **optional** callable — the seam for a semantic score. It is
    called only if no gate failed, and its result is recorded under `judge`
    without ever touching `verdict`. No judge is implemented in this codebase;
    supplying one is the caller's decision, and the ordering guarantee is what is
    tested here.
    """
    run = repo.get_run(run_id)
    if run is None:
        raise ValueError(f"未找到运行 {run_id}")

    facts = invariants(repo, run_id, step_budget=task.step_budget)
    gate = gate_expectations(task)
    checks = check_expectations(task)

    gate_failures = _failed(gate, facts)
    if gate_failures:
        # Short-circuits: the metrics and the judge are both skipped. Not an
        # optimisation — a judge that ran here could only add a number the next
        # reader has to be told to ignore.
        return Evaluation(
            run_id=run_id, task_id=task.id, task_version=task.version,
            verdict=VERDICT_FAIL,
            axes=_axes_from(gate_failures, set(gate), facts),
            reasons=("硬门禁未通过：" + "、".join(gate_failures)
                     + "。语义评分不会被执行，也不会改变这个结论。",),
            failures=_failure_classes(repo, run_id), metrics={}, artifacts={},
            hard_gate=tuple(gate_failures), judge=None)

    check_failures = _failed(checks, facts)
    reasons = [f"检查未通过：{entry}" for entry in check_failures]

    report = run_metrics(repo, run_id)
    artifacts = {
        "changed_paths": list(repo.affected_paths_for_run(run_id, agent="implementer")),
        "steps": facts.get("stepsExecuted"),
        "final_state": facts.get("finalState"),
    }

    judged: dict | None = None
    if judge is not None:
        # Only reachable with a clean gate. A judge may *add* a reason; it may not
        # reach the verdict, which is the one rule that makes an optional
        # semantic score safe to have at all.
        judged = dict(judge({**facts, "metrics": report, "artifacts": artifacts}) or {})
        if judged.get("reason"):
            reasons.append(str(judged["reason"]))

    failed = gate_failures + check_failures
    return Evaluation(
        run_id=run_id, task_id=task.id, task_version=task.version,
        verdict=VERDICT_FAIL if failed else VERDICT_PASS,
        axes=_axes_from(failed, set(gate) | set(checks), facts),
        reasons=tuple(reasons), failures=_failure_classes(repo, run_id),
        metrics=report, artifacts=artifacts, hard_gate=(), judge=judged)


def record(repo, evaluation: Evaluation) -> int:
    """Persist a verdict and put it in the trace; return the evaluation row id.

    Two writes for one fact, on purpose. The row is what a later promotion reads
    — possibly from another process, possibly long after — and the event is what
    puts the judgement into the run's own ordered history. A verdict that lived
    only in the side table would be invisible to anyone reading `wfos trace show`,
    and one that lived only in the trace would have to be picked back out of
    JSON to be cited.
    """
    run = repo.get_run(evaluation.run_id) or {}
    event_id = events.record(
        repo, events.EVALUATED, run_id=evaluation.run_id,
        session_id=run.get("session_id") or "",
        payload={"task_id": evaluation.task_id,
                 "task_version": evaluation.task_version,
                 "verdict": evaluation.verdict,
                 "hard_gate": list(evaluation.hard_gate),
                 "judged": bool(evaluation.judge)})
    return repo.add_evaluation(evaluation, event_id=event_id)


def _failure_classes(repo, run_id: str) -> tuple[str, ...]:
    return tuple(sorted({s["failure_class"] for s in repo.steps_for_run(run_id)
                         if s.get("failure_class")}))


def render(evaluation: Evaluation) -> str:
    """The verdict as a person reads it: what was checked, and what was not."""
    lines = [f"判定 {evaluation.verdict}  任务 {evaluation.task_id} "
             f"v{evaluation.task_version}  运行 {evaluation.run_id}"]
    for axis in AXES:
        lines.append(f"  {axis:<12} {evaluation.axes[axis]}")
    if evaluation.hard_gate:
        lines.append("  硬门禁失败（未执行语义评分）: " + "、".join(evaluation.hard_gate))
    for reason in evaluation.reasons:
        lines.append(f"  - {reason}")
    if evaluation.failures:
        lines.append("  失败类型: " + "、".join(evaluation.failures))
    if evaluation.judge:
        lines.append(f"  语义评分: {evaluation.judge}（不改变判定）")
    return "\n".join(lines)
