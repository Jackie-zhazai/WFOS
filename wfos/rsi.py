"""RSI: proposing changes from evidence, and finding out whether they help.

The thing this module must not do is invent an improvement. Every candidate has to
answer "why does this exist" with a chain that ends in something that actually
happened — an experiment, a run, an evaluation, a failure class — and a candidate
whose chain does not reach the record is not a candidate, it is an opinion with a
version number.

That shapes the whole design:

  * **Proposals are derived, never authored.** A `skill` candidate's text comes
    from a run that hit a failure class *and got past it* — the harness's own
    record of what worked. Nothing here writes a procedure from a description of
    a problem, which is what a model would do and what "verified improvement"
    cannot mean.
  * **A failure with no recovery produces no candidate.** It produces a finding.
    Inventing a procedure for a failure nobody has solved would be exactly the
    unfounded claim this is supposed to prevent, and it would be invisible,
    because a candidate looks the same either way.
  * **Stage one can only promote a skill**, so the other four types are proposed
    and then refused with a reason rather than quietly skipped. §2 asks for them
    to reach `NOT_PROMOTABLE`, and a refusal that says why is more useful than an
    absence.
  * **Nothing here promotes anything.** A candidate that did better is `PASSED`,
    which means "worth a human's attention", not "now in production". The words
    matter: `PASSED` is a verdict about a benchmark, and promotion is a decision
    a person makes.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import events
from .bench import Arm, load_record, run_benchmark, slug
from .bench_compare import REGRESSION
from .bench_compare import compare as compare_records
from .models import ORIGIN_INTERACTIVE, SKILL_LIVE

SCHEMA_VERSION = 1

# The five kinds a candidate may declare. Only the first has an applier, and only
# the first can pass stage one — see `PROMOTABLE_TYPES`.
TYPE_SKILL = "skill"
TYPE_WORKFLOW_POLICY = "workflow_policy"
TYPE_TOOL_HINT = "tool_hint"
TYPE_RETRY_POLICY = "retry_policy"
TYPE_CONTEXT_POLICY = "context_policy"
TYPES = (TYPE_SKILL, TYPE_WORKFLOW_POLICY, TYPE_TOOL_HINT, TYPE_RETRY_POLICY,
         TYPE_CONTEXT_POLICY)

# Stage one applies a change to a procedure and to nothing else. A candidate of
# another type is still proposed and still evaluated — it just cannot pass, and
# the report says which type lacked an applier rather than leaving a silence.
PROMOTABLE_TYPES = (TYPE_SKILL,)

STATUS_PROPOSED = "PROPOSED"
STATUS_EVALUATING = "EVALUATING"
STATUS_PASSED = "PASSED"
# Reachable only through `promote`, which re-runs the gate rather than trusting
# that whoever asked has already checked.
STATUS_PROMOTED = "PROMOTED"
STATUS_REJECTED = "REJECTED"
STATUS_NOT_PROMOTABLE = "NOT_PROMOTABLE"
STATUS_FAILED = "FAILED"
STATUSES = (STATUS_PROPOSED, STATUS_EVALUATING, STATUS_PASSED, STATUS_REJECTED,
            STATUS_NOT_PROMOTABLE, STATUS_FAILED, STATUS_PROMOTED)

# The only moves that exist. `PROPOSED -> PRODUCTION` is not among them and there
# is no status it could name, which is how §8's "no automatic promotion" is
# enforced: not by a check somewhere, but by the absence of a destination.
TRANSITIONS: dict[str, tuple[str, ...]] = {
    STATUS_PROPOSED: (STATUS_EVALUATING, STATUS_NOT_PROMOTABLE),
    STATUS_EVALUATING: (STATUS_PASSED, STATUS_REJECTED, STATUS_NOT_PROMOTABLE,
                        STATUS_FAILED),
    # PASSED is no longer terminal: a promotion can only be asked for explicitly,
    # and the gate can still refuse at that point (→ REJECTED / NOT_PROMOTABLE).
    # There is still no status meaning "in production by itself" — `PROMOTED` is
    # the *act*, and no status is reached without `promote` being called.
    STATUS_PASSED: (STATUS_PROMOTED, STATUS_REJECTED, STATUS_NOT_PROMOTABLE),
    STATUS_PROMOTED: (),
    STATUS_REJECTED: (),
    STATUS_NOT_PROMOTABLE: (),
    STATUS_FAILED: (),
}


def can_transition(current: str, target: str) -> tuple[bool, str]:
    """Whether a status move is legal, and why not when it is not."""
    if current not in TRANSITIONS:
        return False, f"未知状态 {current!r}"
    if target not in STATUSES:
        return False, f"未知状态 {target!r}"
    if target not in TRANSITIONS[current]:
        return False, (f"不允许 {current} → {target}；"
                       f"合规的下一步是 {TRANSITIONS[current] or '（终态）'}")
    return True, ""


@dataclass(frozen=True)
class CandidateSpec:
    """A proposed change, with everything needed to explain why it exists."""

    candidate_id: str
    version: int
    type: str
    parent_version: int
    source_experiment: str
    source_runs: tuple[str, ...]
    source_failures: tuple[str, ...]
    evidence: dict[str, Any]
    proposed_change: dict[str, Any]
    rationale: str

    def __post_init__(self) -> None:
        """Refuse a candidate that cannot be explained or evaluated.

        Validating the type matters more than it looks: an unknown type would
        evaluate to `NOT_PROMOTABLE` with the reason "this type has no applier",
        which reads as a known type that stage one cannot apply — not as a typo.
        The record would be misleading rather than wrong, which is harder to
        notice.
        """
        if not str(self.candidate_id or "").strip():
            raise ValueError("候选缺少 candidate_id")
        if self.type not in TYPES:
            raise ValueError(
                f"候选类型 {self.type!r} 未知；可选 {', '.join(TYPES)}")
        if int(self.version) < 1:
            raise ValueError(f"候选版本必须是正整数，实际 {self.version!r}")
        if int(self.parent_version) < 0:
            raise ValueError(f"母版本不能为负，实际 {self.parent_version!r}")

    def as_json(self) -> dict[str, Any]:
        return {"candidateId": self.candidate_id, "version": self.version,
                "type": self.type, "parentVersion": self.parent_version,
                "sourceExperiment": self.source_experiment,
                "sourceRuns": list(self.source_runs),
                "sourceFailures": list(self.source_failures),
                "evidence": self.evidence,
                "proposedChange": self.proposed_change,
                "rationale": self.rationale}


@dataclass(frozen=True)
class CandidateRecord:
    """A spec plus where it got to."""

    spec: CandidateSpec
    status: str
    verdict: dict[str, Any] = field(default_factory=dict)
    # When the row was written and when its status last moved. Carried on the
    # record rather than left in the database because every entry point that
    # prints a candidate prints the record — and the one that printed only the row
    # was the only one that had this field.
    created_at: str = ""
    updated_at: str = ""

    @property
    def candidate_id(self) -> str:
        return self.spec.candidate_id

    @property
    def version(self) -> int:
        return self.spec.version

    # The spec's fields, reachable through the record: the repository persists a
    # record, and a caller reading one back should not have to know which half of
    # it a given field lives in.
    @property
    def type(self) -> str:
        return self.spec.type

    @property
    def parent_version(self) -> int:
        return self.spec.parent_version

    @property
    def source_experiment(self) -> str:
        return self.spec.source_experiment

    @property
    def source_runs(self) -> tuple[str, ...]:
        return self.spec.source_runs

    @property
    def source_failures(self) -> tuple[str, ...]:
        return self.spec.source_failures

    @property
    def evidence(self) -> dict[str, Any]:
        return self.spec.evidence

    @property
    def proposed_change(self) -> dict[str, Any]:
        return self.spec.proposed_change

    @property
    def rationale(self) -> str:
        return self.spec.rationale

    def as_json(self) -> dict[str, Any]:
        return {**self.spec.as_json(), "status": self.status,
                "verdict": self.verdict,
                "createdAt": self.created_at, "updatedAt": self.updated_at}


@dataclass(frozen=True)
class FailureEvidence:
    """One thing that went wrong, as the record states it."""

    task_id: str
    cell_id: str
    run_id: str
    verdict: str
    failures: tuple[str, ...]
    hard_gate: tuple[str, ...]
    workflow_path: tuple[str, ...]
    tool_sequence: tuple[str, ...]
    status: str
    failure_class: str

    def as_json(self) -> dict[str, Any]:
        return {"taskId": self.task_id, "cellId": self.cell_id,
                "runId": self.run_id, "verdict": self.verdict,
                "failures": list(self.failures), "hardGate": list(self.hard_gate),
                "workflowPath": list(self.workflow_path),
                "toolSequence": list(self.tool_sequence), "status": self.status,
                "failureClass": self.failure_class}


@dataclass(frozen=True)
class Analysis:
    """What the evidence supports, and what it does not."""

    experiment: str
    experiment_id: str
    suite: str
    evidence: tuple[FailureEvidence, ...]
    proposals: tuple[CandidateSpec, ...]
    # Failures nobody has recovered from. Listed rather than dropped: "no
    # candidate" and "we did not look" are different facts, and only one of them
    # means the operator should go and fix something by hand.
    unactionable: tuple[dict[str, Any], ...]

    def as_json(self) -> dict[str, Any]:
        return {"experiment": self.experiment, "experimentId": self.experiment_id,
                "suite": self.suite, "evidence": [e.as_json() for e in self.evidence],
                "proposals": [p.as_json() for p in self.proposals],
                "unactionable": list(self.unactionable)}


# ------------------------------------------------------------------ analysis
def analyse(experiment: dict, *, experiment_path: str = "",
            live_skills: dict[str, int] | None = None) -> Analysis:
    """Read an experiment's record and say what it supports.

    `live_skills` maps a failure class to the version of the procedure currently
    in production for it. The candidate's `parent_version` comes from there, which
    is what makes "v2-candidate" mean something: it is proposed *against* a
    version, not in a vacuum.
    """
    live = dict(live_skills or {})
    evidence: list[FailureEvidence] = []
    proposals: list[CandidateSpec] = []
    unactionable: list[dict[str, Any]] = []

    records = experiment.get("tasks") or []
    cells = {str(c.get("cellId")): c for c in experiment.get("cells") or []}

    by_task: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_task.setdefault(str(record.get("taskId")), []).append(record)

    for record in records:
        cell = cells.get(str(record.get("cellId"))) or {}
        verdict = str(record.get("verdict") or "")
        failures = tuple(record.get("failures") or ())
        if verdict != "fail" and not failures:
            continue
        item = FailureEvidence(
            task_id=str(record.get("taskId") or ""),
            cell_id=str(record.get("cellId") or ""),
            run_id=str(record.get("runId") or ""),
            verdict=verdict, failures=failures,
            hard_gate=tuple(record.get("hardGate") or ()),
            workflow_path=tuple(record.get("workflowPath") or ()),
            tool_sequence=tuple(record.get("toolSequence") or ()),
            status="ran", failure_class=failures[0] if failures else "")
        evidence.append(item)

    # A cell that never started is evidence too, and it has no run to point at —
    # recorded so the analysis is not silently about a subset of what happened.
    for cell in experiment.get("cells") or []:
        if cell.get("status") == "ran":
            continue
        evidence.append(FailureEvidence(
            task_id=str(cell.get("taskId") or ""), cell_id=str(cell.get("cellId") or ""),
            run_id="", verdict="", failures=(),
            hard_gate=(), workflow_path=(), tool_sequence=(),
            status=str(cell.get("status") or ""),
            failure_class=str(cell.get("failureClass") or "")))

    # Every failure class the run *recovered from*, which is where a skill's text
    # comes from. A class nobody got past has no text to propose.
    recovered: dict[str, dict[str, Any]] = {}
    for record in records:
        for derived in record.get("derivedSkills") or []:
            trigger = str(derived.get("trigger") or "")
            if not trigger:
                continue
            recovered.setdefault(trigger, {**derived,
                                           "runId": str(record.get("runId") or ""),
                                           "taskId": str(record.get("taskId") or "")})

    for trigger, derived in sorted(recovered.items()):
        source_runs = tuple(sorted({str(derived.get("runId") or "")} - {""}))
        proposals.append(CandidateSpec(
            candidate_id=f"skill:{trigger}",
            # The proposed version is the parent's next one, so the id says what
            # it is against without having to look the parent up.
            version=int(live.get(trigger, 0)) + 1,
            type=TYPE_SKILL,
            parent_version=int(live.get(trigger, 0)),
            source_experiment=experiment_path,
            source_runs=source_runs,
            source_failures=(trigger,),
            evidence={"trigger": trigger,
                      # The suite is carried so the candidate can be benchmarked
                      # later without the caller having to remember which task set
                      # produced it.
                      "suite": str(experiment.get("suite") or ""),
                      "derivedFrom": source_runs[0] if source_runs else "",
                      "derivedTask": derived.get("taskId", ""),
                      "toolSequence": next((list(r.get("toolSequence") or ())
                                            for r in records
                                            if r.get("runId") in source_runs), []),
                      "workflowPath": next((list(r.get("workflowPath") or ())
                                            for r in records
                                            if r.get("runId") in source_runs), [])},
            proposed_change={"kind": TYPE_SKILL, "trigger": trigger,
                             "title": str(derived.get("title") or trigger),
                             "procedure": str(derived.get("procedure") or ""),
                             "files": list(derived.get("files") or [])},
            rationale=(f"运行 {source_runs[0] if source_runs else '?'} 撞上 {trigger} "
                       f"并从中恢复，Harness 由自己的记录推导出该做法；"
                       f"提议作为 v{int(live.get(trigger, 0)) + 1} 候选，"
                       f"母版本 v{int(live.get(trigger, 0))}")))

    # A class that recurs is evidence for a retry-policy proposal: the retry loop
    # is not converging, which is a policy question rather than a procedure one.
    counts: dict[str, int] = {}
    for item in evidence:
        if item.status == "ran" and item.failure_class:
            counts[item.failure_class] = counts.get(item.failure_class, 0) + 1
    for trigger, count in sorted(counts.items()):
        if count < 2 or trigger in recovered:
            continue
        runs = tuple(sorted({e.run_id for e in evidence
                             if e.failure_class == trigger and e.run_id}))
        proposals.append(CandidateSpec(
            candidate_id=f"retry:{trigger}", version=1, type=TYPE_RETRY_POLICY,
            parent_version=0, source_experiment=experiment_path,
            source_runs=runs, source_failures=(trigger,),
            evidence={"trigger": trigger, "occurrences": count, "runs": list(runs)},
            proposed_change={"kind": TYPE_RETRY_POLICY, "trigger": trigger,
                             "observedOccurrences": count},
            rationale=(f"{trigger} 在 {count} 处重复出现，说明重试循环没有收敛；"
                       f"提议作为重试策略候选（本阶段无 applier，不会被晋升）")))

    proposed_triggers = {p.source_failures[0] for p in proposals}
    for trigger, count in sorted(counts.items()):
        if trigger in proposed_triggers:
            continue
        unactionable.append({
            "failure": trigger, "occurrences": count,
            "reason": "没有任何运行从这个失败类型中恢复过，因此没有可提议的做法 —— "
                      "凭一个未解决的问题编一份 procedure 正是本模块要防的事"})

    return Analysis(experiment=experiment_path,
                    experiment_id=str(experiment.get("experimentId") or ""),
                    suite=str(experiment.get("suite") or ""),
                    evidence=tuple(evidence), proposals=tuple(proposals),
                    unactionable=tuple(unactionable))


# ----------------------------------------------------------------- proposal
def record_proposal(repo, spec: CandidateSpec, *, actor: str = events.ACTOR_HARNESS) -> CandidateRecord:
    """Persist a candidate at `PROPOSED`, and put the fact in the trace."""
    repo.add_candidate(CandidateRecord(spec=spec, status=STATUS_PROPOSED))
    # Read back rather than assumed: the timestamps are the database's, and a
    # caller that has to query for them anyway is a caller that will forget to.
    record = record_from_row(repo.get_candidate(spec.candidate_id, spec.version))
    events.record(repo, events.CANDIDATE_PROPOSED, session_id="",
                  actor=actor,
                  payload={"candidateId": spec.candidate_id, "version": spec.version,
                           "type": spec.type, "parentVersion": spec.parent_version,
                           "sourceRuns": list(spec.source_runs),
                           "sourceFailures": list(spec.source_failures)})
    return record


def transition(repo, record: CandidateRecord, status: str, *,
               verdict: dict[str, Any] | None = None) -> CandidateRecord:
    """Move a candidate's status, refusing an illegal move.

    The refusal matters more than the move: `PROPOSED → PASSED` would let a
    candidate be declared good without ever being run, and the only thing standing
    between that and a production change is this check.
    """
    ok, why = can_transition(record.status, status)
    if not ok:
        raise ValueError(f"候选 {record.candidate_id} v{record.version}: {why}")
    repo.set_candidate_status(record.candidate_id, record.version, status,
                              verdict=verdict)
    events.record(repo, events.CANDIDATE_STATUS, session_id="",
                  actor=events.ACTOR_HARNESS,
                  payload={"candidateId": record.candidate_id,
                           "version": record.version, "from": record.status,
                           "to": status})
    return CandidateRecord(spec=record.spec, status=status,
                           verdict=verdict if verdict is not None else record.verdict,
                           created_at=record.created_at, updated_at=record.updated_at)


# ---------------------------------------------------------------- evaluation
def evaluate_candidate(repo, record: CandidateRecord, *, workspace: str | Path,
                       suite: str, baseline: str = "") -> CandidateRecord:
    """Run the candidate's change through the benchmark, and settle its status.

    Uses P3's `run_benchmark` and P3's `compare` — the candidate is a run with one
    arm, and giving it its own execution or its own comparison would mean the
    thing being measured was not the thing that runs in production.

    The rule, stated once so it can be tested: **`PASSED` requires no hard-gate
    failure in the candidate's own run and no regression against the baseline.**
    Anything else is `REJECTED`, and a run that could not happen at all is
    `FAILED`. Improvement alone is not enough — a candidate that is better on
    quality while regressing on safety has not passed anything.
    """
    evaluating = transition(repo, record, STATUS_EVALUATING)
    spec = record.spec

    if spec.type not in PROMOTABLE_TYPES:
        return transition(
            repo, evaluating, STATUS_NOT_PROMOTABLE,
            verdict={"reason": (f"本阶段只对 {', '.join(PROMOTABLE_TYPES)} 有 applier；"
                                f"{spec.type} 没有 —— 因此它无法被验证，"
                                f"也无法被晋升"),
                     "type": spec.type,
                     "proposal": spec.proposed_change})

    try:
        arm = Arm(skills=(spec.proposed_change,))
        run = run_benchmark(suite, workspace=workspace, tier="",
                            arm=arm, name_prefix=f"rsi/{slug(spec.candidate_id)}")
    except Exception as e:  # noqa: BLE001
        from .llm.base import classify_wire_error
        return transition(repo, evaluating, STATUS_FAILED,
                          verdict={"error": str(e),
                                   "failureClass": classify_wire_error(e),
                                   "reason": "候选的 benchmark 没能跑起来；"
                                             "这不等于候选被证伪"})

    failed_gates = [t.evaluation.hard_gate for t in run.tasks
                    if t.evaluation.hard_gate]
    verdict: dict[str, Any] = {
        "runIds": [t.run_id for t in run.tasks],
        "taskVerdicts": {t.task_id: t.evaluation.verdict for t in run.tasks},
        "axes": [_axes_of(t) for t in run.tasks],
        "hardGateFailures": [list(g) for g in failed_gates],
        "comparison": {},
        # Cited, not restated: the promotion record names these rows so a reader
        # can open the verdict itself. Each is an (id, store) pair because an
        # isolated suite judges every task in its own database — a bare id would
        # read as eight references and resolve to eight unrelated first rows.
        "evaluationIds": [{"id": t.evaluation_id, "store": t.evaluation_store}
                          for t in run.tasks if t.evaluation_id],
    }

    if baseline and Path(baseline).exists():
        # Re-keyed to the baseline's own task ids. The candidate's records carry a
        # `cellId` (its arm is the point of them) and the baseline has never heard
        # of it, so comparing as-is makes every task look both removed and added —
        # which reads as a dozen regressions rather than as a key mismatch.
        subject = {**run.as_json(),
                   "tasks": [{**r.as_json(), "cellId": ""} for r in run.tasks]}
        report = compare_records(load_record(baseline), subject)
        verdict["comparison"] = {"ok": report.ok, "counts": report.counts(),
                                 "environment": report.environment,
                                 "regressions": [f.as_json() for f in
                                                 report.by_verdict(REGRESSION)]}

    regressed = not verdict["comparison"].get("ok", True)
    if failed_gates or regressed:
        verdict["reason"] = ("候选运行出现硬门禁失败" if failed_gates
                             else "候选相对基线有回归")
        return transition(repo, evaluating, STATUS_REJECTED, verdict=verdict)

    verdict["reason"] = ("候选的硬门禁全部通过，且相对基线没有回归。"
                         "PASSED 表示值得人工审阅 —— 晋升是 P6 里人的决定，"
                         "本阶段不会把任何候选变成正式版本。")
    return transition(repo, evaluating, STATUS_PASSED, verdict=verdict)


def _axes_of(task_record) -> dict[str, Any]:
    return {"taskId": task_record.task_id, "cellId": task_record.cell,
            "axes": dict(task_record.evaluation.axes)}



# ------------------------------------------------------------------ promotion
def _digest_of(row: dict) -> str:
    from .skills import content_digest

    return content_digest(row)


def provenance_gaps(spec: CandidateSpec) -> list[str]:
    """What is missing from "why does this exist". Empty means complete.

    Checked at promotion rather than trusted from proposal time: a candidate row
    can be read back by anything, and the gate is the last place before production
    where a chain with a hole in it can still be caught.
    """
    gaps: list[str] = []
    if not spec.source_experiment:
        gaps.append("没有来源实验")
    if not spec.source_runs:
        gaps.append("没有来源运行")
    if not spec.source_failures:
        gaps.append("没有来源失败类型")
    if not spec.evidence:
        gaps.append("没有证据")
    if not spec.proposed_change:
        gaps.append("没有提议内容")
    return gaps


def promotion_gate(repo, record: CandidateRecord, *,
                   baseline: str = "") -> dict[str, Any]:
    """Re-verify a candidate before it is allowed anywhere near production.

    Returns `{"verdict": PASS|REJECT|NOT_PROMOTABLE, "reasons": [...], ...}`.

    The distinction between the two refusals is not cosmetic:

      * **REJECT** — the evidence says this candidate is *worse*. It failed an axis
        that matters, or it regressed against the baseline.
      * **NOT_PROMOTABLE** — the evidence does not say anything. An `unknown` axis
        is not a pass, an incomplete provenance chain is not a reason to proceed,
        and neither is a reason to call the candidate bad either.

    `unknown` never becomes `pass` here. The axes are read as they were recorded,
    and an axis nobody measured refuses the promotion rather than being counted as
    satisfactory — the rule this codebase applies to an unreported token count,
    applied to the one decision where it matters most.
    """
    spec = record.spec

    if record.status != STATUS_PASSED:
        return {"verdict": STATUS_REJECTED, "reasons": [
            f"候选状态是 {record.status}，只有 {STATUS_PASSED} 才允许晋升"]}

    gaps = provenance_gaps(spec)
    if gaps:
        return {"verdict": STATUS_NOT_PROMOTABLE,
                "reasons": ["溯源不完整：" + "、".join(gaps)]}

    live = repo.latest_skill(spec.source_failures[0])
    live_version = int(live["version"]) if live else 0
    if live_version != spec.parent_version:
        return {"verdict": STATUS_REJECTED, "reasons": [
            f"母版本不符：候选声明 v{spec.parent_version}，"
            f"而 {spec.source_failures[0]} 当前是 v{live_version}；"
            f"候选是针对另一个版本做的，结论不适用"]}

    # A critical failure is not a matter of degree: if the candidate's own run
    # tripped a hard gate, no axis average and no comparison can make it fit to
    # promote, and the gate re-reads it rather than trusting that whoever ran the
    # evaluation drew the same conclusion.
    hard = [list(g) for g in record.verdict.get("hardGateFailures") or []]
    if hard:
        return {"verdict": STATUS_REJECTED, "reasons": [
            "候选自己的运行出现临界失败（硬门禁）："
            + "、".join("、".join(str(x) for x in g) for g in hard)]}

    # The axes this decision turns on. Read from the candidate's own recorded
    # evaluation — the gate does not re-run anything, it re-reads.
    #
    # Aggregated worst-first and **order-independently**. `fail` from any one task
    # stands — a task that passes safety does not offset a task that does not. A
    # task that *measured* the axis settles it. An axis **no task spoke to at all**
    # stays `unknown`, and unknown is never a pass.
    #
    # This used to be a last-one-wins fold, which made the verdict depend on the
    # order of the task list: the regression suite's four resume tasks make no
    # safety claim (so report `unknown`) while its other four check safety (so
    # report `pass`), and whichever group sorted last decided whether the same
    # candidate could be promoted.
    axes: dict[str, str] = {}
    for task_axes in record.verdict.get("axes") or []:
        for axis, value in (task_axes.get("axes") or {}).items():
            seen = axes.get(axis)
            if seen == "fail":
                continue
            if value == "fail" or seen is None or seen == "unknown":
                axes[axis] = value
    if not axes:
        return {"verdict": STATUS_NOT_PROMOTABLE,
                "reasons": ["候选记录里没有五轴判定，无法判断 functional/safety"]}

    for axis in ("functional", "safety", "operational"):
        value = axes.get(axis)
        if value == "fail":
            return {"verdict": STATUS_REJECTED,
                    "reasons": [f"{axis} 未通过（候选自己的判定为 fail）"]}
        if value != "pass":
            return {"verdict": STATUS_NOT_PROMOTABLE, "reasons": [
                f"{axis} 是 {value!r}，不是 pass —— "
                f"没有测量出来的东西不能当作通过（unknown 不是 pass）"]}

    comparison = record.verdict.get("comparison") or {}
    regressions = [f.get("subject") for f in comparison.get("regressions") or []]
    if regressions:
        critical = [s for s in regressions
                    if s in ("axis.functional", "axis.safety", "verdict", "hardGate")
                    or str(s).startswith("metric.")]
        return {"verdict": STATUS_REJECTED, "reasons": [
            "相对基线出现回归：" + "、".join(str(s) for s in regressions)
            + ("（其中含关键项）" if critical else "")]}
    if baseline and not comparison:
        return {"verdict": STATUS_NOT_PROMOTABLE,
                "reasons": ["给了基线但候选没有比较结果，无法判断是否退化"]}

    return {"verdict": STATUS_PASSED, "reasons": [], "axes": axes,
            "parentVersion": live_version, "comparison": comparison,
            "unknownAxes": sorted(a for a, v in axes.items() if v == "unknown")}


def promote(repo, record: CandidateRecord, *, baseline: str = "",
            actor: str = "cli", reason: str = "") -> dict[str, Any]:
    """Turn a `PASSED` candidate into a production skill version.

    Explicit by construction: this is the only function that writes a live skill
    from a candidate, nothing calls it automatically, and it re-runs the gate
    rather than trusting that whoever asks has already checked.

    Nothing here touches a baseline, a benchmark task or an evaluator rule. What a
    promotion produces is one new skill version and one record of having done it.
    """
    gate = promotion_gate(repo, record, baseline=baseline)
    if gate["verdict"] != STATUS_PASSED:
        # A refusal is only a *verdict* when there was something to judge. A
        # candidate that was never `PASSED` has not been evaluated, so a premature
        # `promote` leaves it exactly where it is — recording `PROPOSED` as
        # `REJECTED` would write down an evaluation that never happened. (It is
        # also not a legal move: nothing goes from `PROPOSED` to `REJECTED`.)
        if record.status != STATUS_PASSED:
            return {"ok": False, "status": record.status, "gate": gate,
                    "candidate": record.as_json()}
        target = (STATUS_REJECTED if gate["verdict"] == STATUS_REJECTED
                  else STATUS_NOT_PROMOTABLE)
        refused = transition(repo, record, target, verdict={
            **record.verdict, "gate": gate,
            "reason": "晋升被拒绝：" + "；".join(gate["reasons"])})
        return {"ok": False, "status": refused.status, "gate": gate,
                "candidate": refused.as_json()}

    spec = record.spec
    trigger = spec.source_failures[0]
    parent = repo.latest_skill(trigger)
    parent_digest = _digest_of(parent) if parent else ""
    parent_version = int(parent["version"]) if parent else 0

    change = spec.proposed_change
    created = repo.add_skill(
        trigger, str(change.get("title") or trigger),
        str(change.get("procedure") or ""),
        # The run that proved it, not the candidate id: a skill's evidence is
        # supposed to be a run, and the candidate is reachable through the
        # promotion record that names this version.
        (spec.source_runs[0] if spec.source_runs else ""),
        files=list(change.get("files") or []),
        origin=ORIGIN_INTERACTIVE, status=SKILL_LIVE,
        parent_id=int(parent["id"]) if parent else None)

    promotion = {
        "promotion_id": f"promo-{uuid.uuid4().hex[:12]}",
        "candidate_id": spec.candidate_id, "candidate_version": spec.version,
        "parent_skill_version": parent_version, "parent_digest": parent_digest,
        "promoted_skill_version": int(created["version"]),
        "promoted_digest": _digest_of(created),
        "source_experiment": spec.source_experiment,
        "source_runs": list(spec.source_runs),
        "evaluation": record.verdict,
        "evaluation_ids": list(record.verdict.get("evaluationIds") or []),
        "compare_result": record.verdict.get("comparison") or {},
        "gate": gate, "actor": actor,
        "reason": reason or "候选通过门禁，显式晋升",
    }
    repo.add_promotion(promotion)
    # Read back rather than returned as built: the row carries the timestamp and
    # the id the database assigned, and a renderer that reads a field the producer
    # never set is the same defect this codebase has now hit twice.
    promotion = repo.get_promotion(promotion["promotion_id"]) or promotion
    events.record(repo, events.CANDIDATE_STATUS, actor=actor,
                  payload={"candidateId": spec.candidate_id, "version": spec.version,
                           "from": STATUS_PASSED, "to": STATUS_PROMOTED,
                           "promotionId": promotion["promotion_id"]})
    promoted = transition(repo, record, STATUS_PROMOTED,
                          verdict={**record.verdict, "gate": gate,
                                   "promotionId": promotion["promotion_id"]})
    return {"ok": True, "promotion": promotion, "skill": created,
            "candidate": promoted.as_json(), "gate": gate}


# ------------------------------------------------------------------- rollback
def rollback(repo, trigger: str, version: int, *, actor: str = "cli",
             reason: str = "") -> dict[str, Any]:
    """Move the current pointer back to a version that already exists.

    Fail-closed on four things, in this order:

      * the version has to exist as a **live** production skill of this trigger —
        a candidate row, a superseded row of another trigger, or nothing at all is
        not a rollback target;
      * the version has to be **named by a promotion record**, because that record
        is where its digest was written down by something other than the row being
        checked;
      * the row's digest now has to **match** that record. A mismatch means the
        history was edited after the decision it justified, and returning to it
        would put a version into production that nobody ever approved;
      * and it must not already be current.

    The version being rolled back *from* is not deleted, not marked, and not
    touched. The only thing that changes is which row the pointer names.
    """
    target = repo.skill_version(trigger, version)
    if target is None:
        return {"ok": False, "error": f"{trigger} 没有正在生效的 v{version}"}

    recorded = _recorded_digest(repo, trigger, version)
    if recorded is None:
        return {"ok": False,
                "error": f"{trigger} v{version} 没有被任何晋升记录命名过；"
                         f"没有可核对的 digest，拒绝回滚（fail-closed）"}
    actual = _digest_of(target)
    if actual != recorded:
        return {"ok": False,
                "error": f"{trigger} v{version} 的内容与晋升记录不一致"
                         f"（记录 {recorded[:12]}，实际 {actual[:12]}）；"
                         f"历史被改过，拒绝回滚"}

    current = repo.latest_skill(trigger)
    if current is None:
        return {"ok": False, "error": f"{trigger} 当前没有生效版本"}
    if int(current["version"]) == version:
        return {"ok": False, "error": f"{trigger} 当前已经是 v{version}，无需回滚"}

    record = {
        "rollback_id": f"rb-{uuid.uuid4().hex[:12]}", "trigger": trigger,
        "from_version": int(current["version"]), "to_version": version,
        "from_digest": _digest_of(current), "to_digest": actual,
        "actor": actor, "reason": reason or f"回滚到 v{version}",
    }
    repo.add_rollback(record)
    repo.set_current_skill(trigger, version)
    events.record(repo, events.CANDIDATE_STATUS, actor=actor,
                  payload={"trigger": trigger, **record, "kind": "rollback"})
    return {"ok": True, "rollback": record,
            "current": repo.latest_skill(trigger)}


def _recorded_digest(repo, trigger: str, version: int) -> str | None:
    """The digest an append-only record wrote down for this version, if any."""
    for promotion in repo.promotions_for_skill(trigger):
        if int(promotion["promoted_skill_version"]) == version:
            return promotion["promoted_digest"]
        if int(promotion["parent_skill_version"]) == version:
            return promotion["parent_digest"]
    return None


def history(repo, trigger: str) -> dict[str, Any]:
    """Every version of a skill, where the pointer is, and how it got there."""
    versions = repo.skill_versions(trigger)
    current = repo.latest_skill(trigger)
    return {
        "trigger": trigger,
        "currentVersion": int(current["version"]) if current else 0,
        "versions": [
            {**{k: v[k] for k in ("version", "title", "digest", "created_at",
                                  "superseded", "parent_id")},
             "evidenceRun": v.get("evidence_run", "")}
            for v in versions],
        "promotions": repo.promotions_for_skill(trigger),
        "rollbacks": repo.rollbacks_for(trigger),
    }


def render_history(data: dict) -> str:
    lines = [f"Skill {data['trigger']}  当前 v{data['currentVersion']}",
             f"  版本 {len(data['versions'])} 个："]
    for v in data["versions"]:
        mark = "← 当前" if v["superseded"] == 0 else ""
        lines.append(f"    v{v['version']}  {v['title'][:30]}  "
                     f"digest={str(v['digest'])[:12]}  {v['created_at'][:19]} {mark}")
    for p in data["promotions"]:
        lines.append(f"  晋升 {p['promotion_id']}：候选 {p['candidate_id']} "
                     f"v{p['candidate_version']} → skill v{p['promoted_skill_version']}"
                     f"（母版本 v{p['parent_skill_version']}）")
    for r in data["rollbacks"]:
        lines.append(f"  回滚 {r['rollback_id']}：v{r['from_version']} → "
                     f"v{r['to_version']}  {r['reason']}")
    return "\n".join(lines)


def render_promotion(data: dict, *, promotion: dict | None = None) -> str:
    """A promotion as a person reads it: what it was, and what it became."""
    if promotion is None:
        promotion = data
    lines = [f"晋升 {promotion['promotion_id']}",
             f"  候选 {promotion['candidate_id']} v{promotion['candidate_version']}",
             f"  skill v{promotion['parent_skill_version']} → "
             f"v{promotion['promoted_skill_version']}",
             f"  来源实验 {promotion['source_experiment'] or '-'}",
             f"  来源运行 {'、'.join(promotion['source_runs']) or '-'}",
             f"  执行者 {promotion['actor']}   {promotion['created_at'][:19]}",
             f"  理由 {promotion['reason']}",
             f"  digest 母 {str(promotion['parent_digest'])[:12]} → "
             f"子 {str(promotion['promoted_digest'])[:12]}"]
    ids = promotion.get("evaluation_ids") or []
    if ids:
        stores = {str(i.get("store") or "") for i in ids if isinstance(i, dict)}
        lines.append(f"  评测记录 {len(ids)} 条，分布在 {len(stores)} 个隔离库中；"
                     f"每条都记着 (id, 库)（每个任务一个库，id 只在库内唯一）")
    return "\n".join(lines)

# ------------------------------------------------------------------ reading
def load_candidate(repo, candidate_id: str, version: int | None = None) -> dict | None:
    return repo.get_candidate(candidate_id, version)


def record_from_row(row: dict) -> CandidateRecord:
    """A stored row as a `CandidateRecord`.

    One shape for a candidate on the way out, so `render` and the JSON output
    cannot disagree: the row is the storage form, and everything a caller sees is
    the record's own `as_json`. Two shapes for one thing is how a renderer ends up
    reading a key that only one of its call sites produces.
    """
    return CandidateRecord(
        spec=CandidateSpec(
            candidate_id=row["candidate_id"], version=int(row["version"]),
            type=row["type"], parent_version=int(row["parent_version"]),
            source_experiment=row.get("source_experiment") or "",
            source_runs=tuple(row.get("source_runs") or ()),
            source_failures=tuple(row.get("source_failures") or ()),
            evidence=dict(row.get("evidence") or {}),
            proposed_change=dict(row.get("proposed_change") or {}),
            rationale=row.get("rationale") or ""),
        status=row["status"], verdict=dict(row.get("verdict") or {}),
        created_at=row.get("created_at") or "",
        updated_at=row.get("updated_at") or "")


def render(record: dict) -> str:
    """A candidate as a person reads it: the chain first, then the verdict.

    Takes the `as_json` shape — the same one the JSON output and the stored
    result use — so there is one spelling of a candidate's fields on the way out.
    """
    lines = [f"候选 {record['candidateId']} v{record['version']}"
             f"（{record['type']}）状态={record['status']}",
             f"  母版本: v{record['parentVersion']}",
             f"  来源实验: {record['sourceExperiment'] or '-'}",
             f"  来源运行: {'、'.join(record['sourceRuns']) or '-'}",
             f"  来源失败: {'、'.join(record['sourceFailures']) or '-'}",
             f"  理由: {record['rationale']}"]
    change = record.get("proposedChange") or {}
    if change.get("procedure"):
        first = str(change["procedure"]).splitlines()
        lines.append(f"  提议的 skill（{change.get('trigger')}）: {first[0] if first else ''}")
        for line in first[1:4]:
            lines.append(f"      {line}")
    verdict = record.get("verdict") or {}
    if verdict:
        lines.append(f"  判定: {verdict.get('reason', '')}")
        comparison = verdict.get("comparison") or {}
        if comparison:
            lines.append(f"  与基线比较: {comparison.get('counts')}"
                         f"（可判={comparison.get('ok')}）")
        if verdict.get("hardGateFailures"):
            lines.append(f"  硬门禁失败: {verdict['hardGateFailures']}")
    return "\n".join(lines)


def render_analysis(analysis: Analysis) -> str:
    """What the evidence supports — and what it does not."""
    lines = [f"失败分析（实验 {analysis.experiment_id}）",
             f"  证据 {len(analysis.evidence)} 条，候选提议 {len(analysis.proposals)} 个"]
    for proposal in analysis.proposals:
        lines.append(f"  [提议] {proposal.candidate_id} v{proposal.version}"
                     f"（母版本 v{proposal.parent_version}）{proposal.rationale}")
    for item in analysis.unactionable:
        lines.append(f"  [无法提议] {item['failure']}×{item['occurrences']}："
                     f"{item['reason']}")
    return "\n".join(lines)


def new_candidate_id(prefix: str = "cand") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def dump_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)
