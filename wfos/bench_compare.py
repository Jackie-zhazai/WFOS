"""Comparing a benchmark record against a baseline.

The rules here are all about **not** claiming more than the data supports, which
is harder than it sounds because the tempting thing to do is produce a number.

  * An axis nobody checked is `unknown` on both sides, and `unknown` against
    `unknown` is `unknown` — not `unchanged`, and certainly not `pass`. Turning
    it into 0 to make a total work is the one thing this module exists to refuse.
  * Metric movement is not a regression. It is `noise` inside a band the operator
    sets, and `unknown` when they have set none — because "this cost more" without
    a criterion for "too much more" is a fact, not a finding.
  * Latency is always `noise`. The same deterministic suite measured ×0.52 to
    ×1.23 between runs on this machine; a number that does that is measuring the
    machine.
  * A different route to the same verdict is `unknown`, not an improvement. A run
    that started passing by skipping verification has not got better, and nothing
    here can tell that apart from a genuinely better route — so it says so.
  * Two records produced under different environments are not comparable at all.
    The report says which part differs and every measurement between them is
    `unknown`; comparing tokens across two models is the same category error as
    dividing a token count by a call count.

`regression` and `improvement` are therefore narrow on purpose: they are claimed
only where a check moved in a direction that needs no interpretation — a check
that passed and now fails, a failure class that appeared, a task that ran and now
does not.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .eval import FAILED, PASSED, UNKNOWN, VERDICT_FAIL, VERDICT_PASS

REGRESSION = "regression"
IMPROVEMENT = "improvement"
UNCHANGED = "unchanged"
UNKNOWN_VERDICT = "unknown"
NOISE = "noise"

VERDICTS = (REGRESSION, IMPROVEMENT, UNCHANGED, UNKNOWN_VERDICT, NOISE)

# Figures whose movement is never a finding on its own. Latency is here because
# the evidence says so, not because it is hard: two runs of the same deterministic
# suite differed by ×0.52 to ×1.23, which is a measurement of the machine.
_NOISY = ("latencyMs",)

# The measurements that a threshold can turn into a finding.
_THRESHOLDED = ("inputTokens", "outputTokens", "costUsd", "modelCalls", "steps")


@dataclass(frozen=True)
class Finding:
    task_id: str
    subject: str
    verdict: str
    before: Any
    after: Any
    detail: str = ""

    def as_json(self) -> dict[str, Any]:
        return {"taskId": self.task_id, "subject": self.subject,
                "verdict": self.verdict, "before": self.before,
                "after": self.after, "detail": self.detail}


@dataclass(frozen=True)
class ComparisonReport:
    baseline: str
    subject: str
    findings: tuple[Finding, ...]
    # "" when both sides ran under the same conditions; otherwise why not.
    environment: str = ""

    @property
    def ok(self) -> bool:
        """No regression. `unknown` and `noise` do not fail — they are not claims."""
        return not any(f.verdict == REGRESSION for f in self.findings)

    def counts(self) -> dict[str, int]:
        out = dict.fromkeys(VERDICTS, 0)
        for finding in self.findings:
            out[finding.verdict] += 1
        return out

    def by_verdict(self, verdict: str) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.verdict == verdict)

    def as_json(self) -> dict[str, Any]:
        return {"kind": "comparison", "baseline": self.baseline,
                "subject": self.subject, "ok": self.ok,
                "environment": self.environment, "counts": self.counts(),
                "findings": [f.as_json() for f in self.findings]}


def _direction(before: Any, after: Any, *, worse: Any, better: Any) -> str:
    """Classify a before/after pair against explicit outcomes, never truthiness.

    A check's "good" value is whatever the task said it was, so `False` and `0` are
    passed to `better` as readily as `True` is. Judging by truthiness reads a run's
    best outcomes as its worst failures — this codebase has made that mistake once
    already, in the evaluator's gate.
    """
    if before == after:
        return UNCHANGED
    if after == worse and before != worse:
        return REGRESSION
    if after == better and before != better:
        return IMPROVEMENT
    return UNKNOWN_VERDICT


def _axes(before: dict, after: dict) -> list[Finding]:
    """One finding per axis, over the union so a missing axis is not a silence."""
    out: list[Finding] = []
    for axis in sorted(set(before) | set(after)):
        was, now = before.get(axis), after.get(axis)
        if was == UNKNOWN or now == UNKNOWN or was is None or now is None:
            # Unknown stays unknown. It is not a pass, it is not a regression, and
            # the two sides not agreeing it was unknown does not change that.
            out.append(Finding("", f"axis.{axis}", UNKNOWN_VERDICT, was, now,
                               "至少一侧没有检查过该轴"))
            continue
        out.append(Finding("", f"axis.{axis}", _direction(was, now, worse=FAILED,
                                                          better=PASSED), was, now))
    return out


def _metrics(before: dict, after: dict, limit: float | None) -> list[Finding]:
    """Measurements, classified as findings only where a criterion exists."""
    out: list[Finding] = []
    for figure in _NOISY + _THRESHOLDED:
        was, now = (before or {}).get(figure), (after or {}).get(figure)
        if was is None or now is None:
            # Unmeasured on one side. Not zero, and not a change.
            out.append(Finding("", f"metric.{figure}", UNKNOWN_VERDICT, was, now,
                               "一侧未上报该度量"))
            continue
        if was == now:
            out.append(Finding("", f"metric.{figure}", UNCHANGED, was, now))
            continue
        if figure in _NOISY:
            out.append(Finding("", f"metric.{figure}", NOISE, was, now,
                               "该项测的是机器负载，同一确定性套件两次可差 ×0.52~×1.23"))
            continue
        if limit is None:
            out.append(Finding("", f"metric.{figure}", UNKNOWN_VERDICT, was, now,
                               "未设阈值（harness.baseline_work_growth_limit），不判定"))
            continue
        ratio = (now / was) if was else None
        if ratio is None:
            out.append(Finding("", f"metric.{figure}", UNKNOWN_VERDICT, was, now,
                               "基数为 0，比值无意义"))
        elif ratio > 1 + limit:
            out.append(Finding("", f"metric.{figure}", REGRESSION, was, now,
                               f"×{ratio:.2f}，超过上限 {limit}"))
        elif ratio < 1 - limit:
            out.append(Finding("", f"metric.{figure}", IMPROVEMENT, was, now,
                               f"×{ratio:.2f}，低于下限 {1 - limit:.2f}"))
        else:
            out.append(Finding("", f"metric.{figure}", NOISE, was, now,
                               f"×{ratio:.2f}，在上限 {limit} 之内"))
    return out


def _sequence(subject: str, before: list, after: list) -> Finding:
    if before == after:
        return Finding("", subject, UNCHANGED, before, after)
    # A different route is a difference. Whether it is a better one cannot be read
    # off the route, and pretending otherwise would let "skipped verification" read
    # as an improvement.
    return Finding("", subject, UNKNOWN_VERDICT, before, after,
                   "路径不同，但无法据此判定好坏")


def _failures(before: list, after: list) -> Finding:
    was, now = set(before or ()), set(after or ())
    if was == now:
        return Finding("", "failureTaxonomy", UNCHANGED, sorted(was), sorted(now))
    if now - was:
        return Finding("", "failureTaxonomy", REGRESSION, sorted(was), sorted(now),
                       "出现了新的失败类型：" + "、".join(sorted(now - was)))
    return Finding("", "failureTaxonomy", IMPROVEMENT, sorted(was), sorted(now),
                   "消失的失败类型：" + "、".join(sorted(was - now)))


def _gate(before: list, after: list) -> Finding:
    was, now = tuple(before or ()), tuple(after or ())
    if was == now:
        return Finding("", "hardGate", UNCHANGED, list(was), list(now))
    if now and not was:
        return Finding("", "hardGate", REGRESSION, list(was), list(now),
                       "硬门禁开始失败 —— 判定语义必须保持失败")
    if was and not now:
        return Finding("", "hardGate", IMPROVEMENT, list(was), list(now))
    return Finding("", "hardGate", UNKNOWN_VERDICT, list(was), list(now),
                   "门禁失败项改变了，但不是从无到有或从有到无")


def _task_findings(before: dict, after: dict, limit: float | None) -> list[Finding]:
    task_id = str(after.get("cellId") or after.get("taskId")
                  or before.get("cellId") or before.get("taskId") or "?")
    out = [
        Finding(task_id, "verdict",
                _direction(before.get("verdict"), after.get("verdict"),
                           worse=VERDICT_FAIL, better=VERDICT_PASS),
                before.get("verdict"), after.get("verdict")),
        _gate(before.get("hardGate"), after.get("hardGate")),
        _failures(before.get("failures"), after.get("failures")),
        _sequence("workflowPath", before.get("workflowPath"), after.get("workflowPath")),
        _sequence("toolSequence", before.get("toolSequence"), after.get("toolSequence")),
        _sequence("models", before.get("models"), after.get("models")),
        _sequence("skillVersions", before.get("skills"), after.get("skills")),
    ]
    # The per-axis and per-metric findings each produce several, so they are
    # extended in rather than nested — a list inside `out` would put a list where
    # every later reader expects a Finding.
    out.extend(_axes(before.get("axes") or {}, after.get("axes") or {}))
    out.extend(_metrics(before.get("metrics") or {}, after.get("metrics") or {}, limit))
    return [Finding(task_id, f.subject, f.verdict, f.before, f.after, f.detail)
            for f in out]


def compare(baseline: dict, subject: dict, *,
            growth_limit: float | None = None) -> ComparisonReport:
    """Diff a benchmark record against a baseline record.

    Both sides are records of the same shape — a baseline is a benchmark run that
    was kept. Neither is modified: this is a reader, and a comparison that wrote to
    either side would make the next comparison a different question.
    """
    env_before = (baseline.get("environment") or {}).get("digest") or ""
    env_after = (subject.get("environment") or {}).get("digest") or ""
    environment = ""
    if env_before and env_after and env_before != env_after:
        environment = (f"两次运行的环境不同（{env_before[:12]} vs {env_after[:12]}）；"
                       f"度量之间不可比较")

    # A cell id when the record has one, the task id otherwise. An experiment runs
    # one task under several arms, so keying on the task alone would collapse three
    # cells into one row and report whichever happened to be last.
    def _key(record: dict) -> str:
        return str(record.get("cellId") or record.get("taskId"))

    before_tasks = {_key(t): t for t in baseline.get("tasks") or []}
    after_tasks = {_key(t): t for t in subject.get("tasks") or []}

    findings: list[Finding] = []
    # A task the baseline covers that this run did not is a regression, for the
    # same reason the frozen baseline treats a removed case as one: dropping
    # coverage is how a benchmark is made green.
    for task_id in sorted(set(before_tasks) - set(after_tasks)):
        findings.append(Finding(task_id, "presence", REGRESSION, "covered", "missing",
                                "基线覆盖的任务这次没有运行"))
    for task_id in sorted(set(after_tasks) - set(before_tasks)):
        findings.append(Finding(task_id, "presence", UNKNOWN_VERDICT, "missing", "covered",
                                "基线里没有这个任务，无从比较"))

    for task_id in sorted(set(before_tasks) & set(after_tasks)):
        findings.extend(_task_findings(before_tasks[task_id], after_tasks[task_id],
                                       growth_limit))

    if environment:
        # The whole point of saying so: a measurement between two environments is
        # not a measurement. Downgrade every metric finding rather than let one
        # read as a regression.
        findings = [
            Finding(f.task_id, f.subject, UNKNOWN_VERDICT, f.before, f.after, environment)
            if f.subject.startswith("metric.") and f.verdict != UNCHANGED else f
            for f in findings
        ]

    return ComparisonReport(
        baseline=str(baseline.get("suite") or baseline.get("kind") or "?"),
        subject=str(subject.get("suite") or subject.get("kind") or "?"),
        findings=tuple(findings), environment=environment)


def render(report: ComparisonReport) -> str:
    """The comparison as a person reads it: counts, then only what moved."""
    counts = report.counts()
    lines = [f"比较 {report.subject} ← 基线 {report.baseline}",
             "  " + "  ".join(f"{v}={counts[v]}" for v in VERDICTS)]
    if report.environment:
        lines.append(f"  ⚠ {report.environment}")
    moved = [f for f in report.findings
             if f.verdict not in (UNCHANGED,)]
    if not moved:
        lines.append("  没有变化。")
        return "\n".join(lines)
    for finding in moved:
        detail = f"  {finding.detail}" if finding.detail else ""
        lines.append(f"  [{finding.verdict}] {finding.task_id}.{finding.subject}: "
                     f"{finding.before!r} → {finding.after!r}{detail}")
    lines.append("  判定：" + ("有回归" if not report.ok else
                              "无回归（unknown / noise 不是结论）"))
    return "\n".join(lines)
