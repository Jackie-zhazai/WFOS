"""Experiments: many arms of one question, run and recorded together.

An experiment is not a new kind of run. It is a *plan* — which tasks, under which
variations — and the same machinery P3 built runs every cell of it: `RunEnvironment`
for isolation, `TaskDriver` for the actions, `evaluator` for the judgement,
`compare` for the reading. This module is orchestration and nothing else, and the
import list is the evidence: it does not call a model, does not decide a verdict,
and does not compute a comparison.

**A cell is an ordinary run.** Each one gets its own workspace, database, trace and
evaluation, because a cell that shared any of those with another would make the
difference between two arms unreadable — which is the only thing the experiment
exists to show. That also means one cell failing cannot fail another: the failure
is recorded *in that cell*, with its taxonomy, and the rest of the matrix runs.

**The spec is the experiment.** Not the CLI arguments that produced it. A run of
`wfos experiment run` with nothing but flags would leave no record of what was
compared, and an experiment nobody can re-run is a set of numbers with a story
attached. `ExperimentSpec` is a file, versioned, and the result says which version
produced it.

**Nothing here writes a baseline.** Not on success, not on failure, not "because
the numbers improved". A baseline is created by a person deciding to freeze one
(`wfos baseline create`), and an experiment that could move it would make every
comparison after it a comparison against a moving target.
"""
from __future__ import annotations

import itertools
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .bench import Arm, load_record, run_benchmark, slug
from .bench_compare import compare as compare_records
from .storage.db import utcnow
from .tasks import load_tasks

SCHEMA_VERSION = 1

# The dimensions a matrix may vary. Named here so a typo is refused rather than
# silently producing a one-cell experiment that looks like a passing one.
DIMENSIONS = ("model", "skill", "workflow", "config")


@dataclass(frozen=True)
class ExperimentSpec:
    """A declared experiment. Persisted and versioned, not assembled from flags."""

    experiment_id: str
    version: int
    name: str
    description: str
    suite: str
    matrix: dict[str, list[Any]]
    repetitions: int = 1
    baseline: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def cells(self, task_ids: list[str]) -> list[dict[str, Any]]:
        """The matrix expanded: one entry per (task, arm) combination.

        The product is taken over the dimensions **in the spec**, so a matrix that
        varies two things produces the cross product rather than two separate
        sweeps. An empty matrix is one arm — the default one — rather than zero
        cells, because "run the suite unchanged" is a legitimate experiment (it is
        how a baseline is reproduced).
        """
        axes = [(name, values) for name, values in sorted(self.matrix.items())
                if values and any(v is not None for v in values)] or []
        arms: list[dict[str, Any]] = []
        for combination in itertools.product(*[values for _, values in axes]) or [()]:
            arms.append(dict(zip([name for name, _ in axes], combination,
                                 strict=True)))
        if not arms:
            arms = [{}]

        out: list[dict[str, Any]] = []
        for task_id in task_ids:
            for arm in arms:
                for repetition in range(self.repetitions):
                    out.append({"taskId": task_id, "arm": arm,
                                "repetition": repetition})
        return out


@dataclass(frozen=True)
class CellResult:
    """One cell of the matrix, as it actually went."""

    cell_id: str
    task_id: str
    repetition: int
    arm: Arm
    # "ran" or "failed_to_start" — a cell that could not even begin is a different
    # fact from a run that began and failed, and only one of them has a run_id.
    status: str
    run_id: str
    evaluation: dict[str, Any]
    failure_class: str
    error: str

    def as_json(self) -> dict[str, Any]:
        return {"cellId": self.cell_id, "taskId": self.task_id,
                "repetition": self.repetition, "arm": self.arm.as_json(),
                "status": self.status, "runId": self.run_id,
                "evaluation": self.evaluation,
                "failureClass": self.failure_class, "error": self.error}


@dataclass(frozen=True)
class ExperimentRun:
    """Every cell of one experiment, run and judged, plus what it compares to."""

    experiment_id: str
    experiment_version: int
    name: str
    suite: str
    benchmark_version: int
    session_id: str
    cells: tuple[CellResult, ...]
    # Every cell's task records, flattened, each carrying its `cellId` — the shape
    # P3's comparison already reads. A second comparison implementation for
    # experiments is exactly what §3 forbids.
    tasks: tuple[dict[str, Any], ...]
    environment: dict[str, Any]
    baseline: str = ""

    def rollup(self) -> dict[str, Any]:
        ran = [c for c in self.cells if c.status == "ran"]
        passed = sum(1 for c in ran if c.evaluation.get("verdict") == "pass")
        return {
            "cells": len(self.cells),
            "ran": len(ran),
            "failedToStart": len(self.cells) - len(ran),
            "passed": passed,
            "failed": len(ran) - passed,
            "failures": sorted({c.failure_class for c in self.cells if c.failure_class}),
        }

    def as_json(self) -> dict[str, Any]:
        return {
            "kind": "experiment-run",
            "schemaVersion": SCHEMA_VERSION,
            "experimentId": self.experiment_id,
            "experimentVersion": self.experiment_version,
            "name": self.name,
            "suite": self.suite,
            "benchmarkVersion": self.benchmark_version,
            "sessionId": self.session_id,
            "baseline": self.baseline,
            "environment": self.environment,
            "rollup": self.rollup(),
            "cells": [c.as_json() for c in self.cells],
            "tasks": list(self.tasks),
        }


# ------------------------------------------------------------------- the spec
def _arm_from(cell: dict[str, Any]) -> Arm:
    """Turn one expanded cell into the arm that configures a run."""
    skill = cell.get("skill")
    skills: tuple[dict[str, Any], ...] = ()
    if isinstance(skill, dict):
        skills = (skill,)
    elif isinstance(skill, list):
        skills = tuple(s for s in skill if isinstance(s, dict))
    config = cell.get("config")
    overrides: tuple[tuple[str, Any], ...] = ()
    if isinstance(config, dict):
        overrides = tuple(sorted((str(k), v) for k, v in config.items()))
    return Arm(model=cell.get("model") or None,
               workflow=cell.get("workflow") or None,
               skills=skills, overrides=overrides)


def load_experiment(path: str | Path) -> ExperimentSpec:
    """Read and validate one experiment spec. Fail-closed, like a task set."""
    src = Path(path)
    data = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{src}: 实验定义必须是对象")
    if data.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError(f"{src}: 不支持的 schemaVersion {data.get('schemaVersion')!r}")

    experiment_id = str(data.get("experimentId", "")).strip()
    if not experiment_id:
        raise ValueError(f"{src}: 缺少 experimentId")
    version = data.get("version", 1)
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ValueError(f"{src}: version 必须是正整数，实际 {version!r}")
    suite = str(data.get("suite", "")).strip()
    if not suite:
        raise ValueError(f"{src}: 缺少 suite —— 实验必须说清跑哪一批任务")

    matrix = data.get("matrix") or {}
    if not isinstance(matrix, dict):
        raise ValueError(f"{src}: matrix 必须是对象")
    for name, values in matrix.items():
        if name not in DIMENSIONS:
            raise ValueError(
                f"{src}: matrix 维度 {name!r} 未知；可选 {', '.join(DIMENSIONS)}。"
                f"拼错的维度会静默展开成单臂实验，看起来像一次通过的实验")
        if values is not None and not isinstance(values, list):
            raise ValueError(f"{src}: matrix.{name} 必须是列表")

    repetitions = data.get("repetitions", 1)
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions < 1:
        raise ValueError(f"{src}: repetitions 必须是正整数，实际 {repetitions!r}")

    return ExperimentSpec(
        experiment_id=experiment_id, version=version,
        name=str(data.get("name") or experiment_id),
        description=str(data.get("description") or ""),
        suite=suite, matrix={str(k): list(v or []) for k, v in matrix.items()},
        repetitions=repetitions, baseline=str(data.get("baseline") or ""),
        metadata=dict(data.get("metadata") or {}))


# ------------------------------------------------------------------ the run
def run_experiment(spec: ExperimentSpec, *, workspace: str | Path,
                   baseline: str = "", session_id: str | None = None) -> ExperimentRun:
    """Run every cell of `spec`, independently, and record what happened.

    Baseline counts as a parameter rather than a consequence: this function reads a
    baseline and never writes one. If a caller passed a path that does not exist,
    the run proceeds and the record says it compared against nothing — inventing a
    baseline here would make the next comparison meaningless.
    """
    tasks = load_tasks(spec.suite)
    if not tasks:
        raise ValueError(f"{spec.suite}: 没有可运行的任务")
    task_ids = [t.id for t in tasks]
    cells = spec.cells(task_ids)

    session = session_id or uuid.uuid4().hex[:16]
    results: list[CellResult] = []
    records: list[dict[str, Any]] = []
    environment: dict[str, Any] = {}
    benchmark_version = 0

    for cell in cells:
        task_id = str(cell["taskId"])
        arm = _arm_from(cell["arm"])
        cell_id = f"{task_id}@{arm.label()}" + (
            f"#r{cell['repetition']}" if spec.repetitions > 1 else "")
        # A cell's directory is named by its identity rather than by its position,
        # so a re-run lands the same cell in the same place — but **slugged**,
        # because the id is written for a person and contains characters a path
        # cannot hold. Found the hard way: `|` and `:` made half a matrix fail to
        # start on Windows, with an error that read like a network problem.
        name = f"{slug(spec.experiment_id)}/{slug(cell_id)}"
        try:
            run = run_benchmark(spec.suite, workspace=workspace, tier="",
                                session_id=session, arm=arm, name_prefix=name,
                                only_task=task_id)
        except Exception as e:  # noqa: BLE001
            # One cell failing must not fail the matrix. A provider that refused
            # one arm says nothing about the others, and marking them failed would
            # turn one real failure into a whole experiment's worth.
            results.append(CellResult(
                cell_id=cell_id, task_id=task_id, repetition=cell["repetition"],
                arm=arm, status="failed_to_start", run_id="", evaluation={},
                failure_class=_classify(e), error=str(e)))
            continue

        for record in run.tasks:
            records.append(record.as_json())
        if not environment:
            environment = run.environment
            benchmark_version = run.benchmark_version
        record = run.tasks[0]
        results.append(CellResult(
            cell_id=cell_id, task_id=task_id, repetition=cell["repetition"],
            arm=arm, status="ran", run_id=record.run_id,
            evaluation={"verdict": record.evaluation.verdict,
                        "axes": dict(record.evaluation.axes),
                        "hardGate": list(record.evaluation.hard_gate),
                        "reasons": list(record.evaluation.reasons)},
            failure_class=(record.evaluation.failures[0]
                           if record.evaluation.failures else ""),
            error=""))

    return ExperimentRun(
        experiment_id=spec.experiment_id, experiment_version=spec.version,
        name=spec.name, suite=spec.suite, benchmark_version=benchmark_version,
        session_id=session, cells=tuple(results), tasks=tuple(records),
        environment=environment or {"digest": ""}, baseline=baseline)


def _classify(exc: BaseException) -> str:
    """A cell that never started still gets a class from the shared vocabulary."""
    from .llm.base import classify_wire_error
    return classify_wire_error(exc)


def comparisons(run: ExperimentRun, baseline_path: str = "") -> list[dict[str, Any]]:
    """One P3 comparison per arm, against the baseline if there is one.

    Per *arm*, not per cell: an arm is the thing being compared, and comparing
    each cell separately would report the same difference once per task. The cells
    of one arm share an environment digest, so the comparison knows whether its
    measurements are comparable at all.

    With no baseline there is nothing to compare against, and this returns nothing
    rather than a report full of `unknown` — an absent comparison is a fact, and a
    fabricated one is not.
    """
    if not baseline_path or not Path(baseline_path).exists():
        return []
    baseline = load_record(baseline_path)

    by_arm: dict[str, list[dict[str, Any]]] = {}
    for record in run.tasks:
        by_arm.setdefault(str(record.get("cellId") or ""), []).append(record)

    out: list[dict[str, Any]] = []
    for cell_id, records in sorted(by_arm.items()):
        arm = next((c.arm for c in run.cells if c.cell_id == cell_id), None)
        for baseline_record in baseline.get("tasks") or []:
            task_id = baseline_record.get("taskId")
            mine = [r for r in records if r.get("taskId") == task_id]
            if not mine:
                continue
            subject = {"kind": "experiment-arm", "suite": run.suite,
                       "environment": run.environment,
                       # Re-keyed to the baseline's own task id: the cell id is how
                       # the experiment tells its arms apart, and the baseline has
                       # never heard of it.
                       "tasks": [{**r, "cellId": ""} for r in mine]}
            report = compare_records(baseline, subject)
            out.append({"cellId": cell_id, "taskId": task_id,
                        "arm": arm.as_json() if arm else {},
                        "ok": report.ok, "counts": report.counts(),
                        "environment": report.environment,
                        "findings": [f.as_json() for f in report.findings]})
    return out


# ------------------------------------------------------------------ storage
def write_experiment(path: str | Path, run: ExperimentRun,
                     baseline_path: str = "") -> Path:
    """Persist a result, refusing to write over one that exists.

    Same rule a baseline gets, for the same reason: a result is a measurement, and
    replacing it with a later measurement leaves no way to tell which one a
    conclusion was drawn from. Re-running the same spec is expected — it just
    writes somewhere else.
    """
    target = Path(path)
    if target.exists():
        raise FileExistsError(
            f"实验结果已存在：{target}\n"
            f"结果是一次测量，覆盖会丢掉它是哪一次跑的。换一个路径，"
            f"例如 {target.with_name(target.stem + '-2' + target.suffix)}")
    payload = {
        **run.as_json(),
        "createdAt": utcnow(),
        "comparisons": comparisons(run, baseline_path or run.baseline),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
                      + "\n", encoding="utf-8")
    return target


def load_experiment_result(path: str | Path) -> dict:
    """Read a stored result back, version-checked."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "cells" not in data:
        raise ValueError(f"{path}: 不是实验结果（缺少 cells）")
    if data.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError(f"{path}: 不支持的实验结果版本 {data.get('schemaVersion')!r}")
    return data


def render(run: ExperimentRun, comparisons_data: list[dict[str, Any]] | None = None) -> str:
    """The result as a person reads it: the matrix, then what each arm did."""
    rollup = run.rollup()
    lines = [f"实验 {run.experiment_id} v{run.experiment_version}  {run.name}",
             f"  套件 {run.suite}  session {run.session_id}",
             f"  单元 {rollup['cells']}（跑起来 {rollup['ran']}，未能启动 "
             f"{rollup['failedToStart']}），通过 {rollup['passed']}，失败 {rollup['failed']}"]
    if rollup["failures"]:
        lines.append("  失败分类: " + "、".join(rollup["failures"]))
    for cell in run.cells:
        if cell.status != "ran":
            lines.append(f"  [未能启动] {cell.cell_id}  {cell.failure_class}: {cell.error}")
            continue
        verdict = cell.evaluation.get("verdict")
        gate = cell.evaluation.get("hardGate") or []
        extra = f"  硬门禁失败: {'、'.join(gate)}" if gate else ""
        lines.append(f"  [{verdict:<4}] {cell.cell_id}  run={cell.run_id[:8]}{extra}")
    for item in comparisons_data or []:
        if item["ok"]:
            continue
        lines.append(f"  [回归] {item['cellId']}.{item['taskId']}: "
                     + "、".join(f"{f['subject']}" for f in item["findings"]
                                 if f["verdict"] == "regression"))
    return "\n".join(lines)
