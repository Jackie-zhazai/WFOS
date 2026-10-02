"""Running a benchmark: many tasks, one session, one comparable record.

A single task run answers "did this task work". A benchmark answers "did this
*set* of tasks work, under these conditions" — and the difference is entirely in
what has to be held constant and recorded. The tasks come from a versioned suite,
every run says which revision of which task produced it, and the whole set shares
one session so a later question ("what did that invocation do") has one answer.

**This module produces records; it does not judge them against anything.** A
benchmark run is a description of what happened, and comparing it to a baseline
is `compare`'s job. Merging the two would make the baseline a thing a run can
influence, which is the one property the baseline must not have.

**Task definitions are read, never written.** A benchmark that could edit its own
tasks could make itself pass. Nothing here opens a suite file for writing, and
`tests/test_bench.py` asserts the files are byte-identical after a run.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .eval import Evaluation, evaluate
from .eval import record as record_evaluation
from .harness import identity
from .models import ORIGIN_BENCHMARK
from .runner import RunEnvironment, TaskOutcome, TaskRunner
from .storage.db import utcnow
from .tasks import TaskSpec, load_tasks

# The version of *this record format* — what a baseline stores — as distinct from
# the version of a suite's tasks. A reader that meets a record it does not
# understand must be able to tell that from a record that is merely malformed.
BENCHMARK_VERSION = 1

# The three fixed sets. Named here so the CLI, the baseline writer and the tests
# all mean the same three things rather than three strings spelled in three
# places.
TIERS = ("smoke", "regression", "challenge")


@dataclass(frozen=True)
class Arm:
    """The knobs one experiment cell varies.

    A tuple of pairs rather than a dict so the arm is hashable and compares by
    value: two cells whose configuration is the same must be recognisable as the
    same arm, and a dict that compares equal but hashes differently would make
    that depend on insertion order.
    """

    model: str | None = None
    workflow: str | None = None
    skills: tuple[dict[str, Any], ...] = ()
    overrides: tuple[tuple[str, Any], ...] = ()

    def label(self) -> str:
        """A short, stable name for this arm, for a cell id and a report line."""
        parts = []
        if self.model:
            parts.append(f"model={self.model}")
        if self.workflow:
            parts.append(f"workflow={self.workflow}")
        if self.skills:
            parts.append("skill=" + "+".join(str(s.get("trigger")) for s in self.skills))
        if self.overrides:
            parts.append("config=" + ",".join(f"{k}:{v}" for k, v in self.overrides))
        return "|".join(parts) or "default"

    def as_json(self) -> dict[str, Any]:
        return {"model": self.model, "workflow": self.workflow,
                "skills": [dict(s) for s in self.skills],
                "config": dict(self.overrides), "label": self.label()}


@dataclass(frozen=True)
class TaskRecord:
    """One task's run, judged, with the facts a comparison needs.

    The workflow path and tool sequence are recorded because "the same verdict by
    a different route" is a real finding — a run that started passing by skipping
    the verification state has not improved, and only the path says so.
    """

    task_id: str
    task_version: int
    workflow: str
    run_id: str
    evaluation: Evaluation
    workflow_path: tuple[str, ...]
    tool_sequence: tuple[str, ...]
    models: tuple[str, ...]
    # `trigger -> version` for the procedures that were in play in this task's own
    # environment. Forward-looking and cheap: it is what a promoted skill (P6)
    # will have to be benchmarked against, and a record that does not say which
    # version ran cannot tell one promotion from another.
    skills: dict[str, int]
    # The procedures this run *derived*, with their text. Version numbers alone
    # say a skill exists; a candidate proposal needs to know what it says, and
    # reading it back out of an isolated database later would mean the analysis
    # depended on where that workspace happened to be.
    derived: tuple[dict[str, Any], ...]
    # Which arm of an experiment produced this. "" for a plain benchmark run —
    # `compare` keys on this when present and on `taskId` otherwise, so one
    # comparison serves both without a second implementation. Last because it has
    # a default and the fields before it do not.
    cell: str = ""
    # The `evaluations` row this verdict was persisted as, and the store it is
    # in. Carried as a pair because neither half resolves alone: every task of an
    # isolated suite is judged in its *own* database, so the ids run 1, 1, 1, …
    # across a suite and an id without its store points at nothing in particular.
    # A citation that cannot be resolved is not provenance.
    evaluation_id: int = 0
    evaluation_store: str = ""

    def as_json(self) -> dict[str, Any]:
        return {
            "taskId": self.task_id,
            "taskVersion": self.task_version,
            "workflow": self.workflow,
            "runId": self.run_id,
            "evaluationId": self.evaluation_id,
            "evaluationStore": self.evaluation_store,
            "verdict": self.evaluation.verdict,
            "axes": dict(self.evaluation.axes),
            "reasons": list(self.evaluation.reasons),
            "failures": list(self.evaluation.failures),
            "hardGate": list(self.evaluation.hard_gate),
            "metrics": self.evaluation.metrics,
            "workflowPath": list(self.workflow_path),
            "toolSequence": list(self.tool_sequence),
            "models": list(self.models),
            "cellId": self.cell,
            "derivedSkills": [dict(d) for d in self.derived],
            "skills": dict(self.skills),
        }


@dataclass(frozen=True)
class BenchmarkRun:
    """Every task of one suite, run and judged, under one session."""

    suite: str
    benchmark_version: int
    session_id: str
    environment: dict[str, Any]
    tasks: tuple[TaskRecord, ...]

    @property
    def rollup(self) -> dict[str, Any]:
        """Counts by verdict, plus how many tasks had a hard-gate failure.

        Counted rather than scored. A single number over mixed axes would have to
        decide how much a `safety` failure is worth against three `quality`
        passes, and whatever it decided would be the real semantics.
        """
        passed = sum(1 for t in self.tasks if t.evaluation.passed)
        gated = sum(1 for t in self.tasks if t.evaluation.hard_gate)
        unknown = sum(1 for t in self.tasks
                      for axis in t.evaluation.axes.values() if axis == "unknown")
        return {
            "tasks": len(self.tasks),
            "passed": passed,
            "failed": len(self.tasks) - passed,
            "hardGateFailures": gated,
            "unknownAxes": unknown,
        }

    def as_json(self) -> dict[str, Any]:
        return {
            "kind": "benchmark-run",
            "schemaVersion": BENCHMARK_VERSION,
            "suite": self.suite,
            "sessionId": self.session_id,
            "environment": self.environment,
            "rollup": self.rollup,
            "tasks": [t.as_json() for t in self.tasks],
        }


def slug(text: str) -> str:
    """A path-safe form of an identifier, keeping it recognisable.

    An id is a *label* — `task@model=x|config=y:0`, `rsi/skill:regression` — and a
    directory is a *path*. Conflating them is what made four of eight experiment
    cells fail to start on Windows, and then made a candidate's benchmark fail for
    the same reason at a second call site. One implementation, used by both.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")
    return safe or "cell"


def _environment_of(harness) -> dict[str, Any]:
    """What every run in this suite was produced under, for *comparing* two runs.

    Reuses the resume fingerprint's own view, minus `project_root` — and that
    subtraction is the whole point of this function existing rather than the
    digest being read straight off `identity_hash`.

    For resume, the project root is a condition: a step produced in one tree may
    not be reusable in another. For a benchmark it is scaffolding — every run gets
    a fresh workspace by design, so including it would make two runs of the same
    suite *always* look incomparable and quietly downgrade every measurement to
    `unknown`. Measured, not guessed: the first end-to-end comparison reported
    "环境不同" for two runs of the same mock suite differing only in where the
    throwaway tree happened to live.

    What remains is what makes runs comparable: the provider, the endpoint, the
    model per role, the sampling. The task's own fixture is covered by the task id,
    which both records carry.
    """
    facts = {k: v for k, v in identity.identity(harness.cfg, harness.agents).items()
             if k != "project_root"}
    blob = json.dumps(facts, sort_keys=True, ensure_ascii=False)
    return {
        "provider": facts["provider"],
        "baseUrl": facts["base_url"],
        "models": facts["models"],
        "temperature": facts["temperature"],
        "digest": hashlib.sha256(blob.encode("utf-8")).hexdigest(),
    }


def _record(runner: TaskRunner, task: TaskSpec, outcome: TaskOutcome,
            evaluation: Evaluation, *, cell: str = "",
            evaluation_id: int = 0) -> TaskRecord:
    repo = runner.repo
    steps = repo.steps_for_run(outcome.run_id)
    return TaskRecord(
        task_id=task.id, task_version=task.version, workflow=task.workflow,
        run_id=outcome.run_id, evaluation=evaluation,
        # The transitions, not the states: a state visited twice is a retry, and a
        # path that lost a retry is a different path.
        workflow_path=tuple(str(t["to_state"]) for t in repo.transitions(outcome.run_id)),
        tool_sequence=tuple(str(t["tool"]) for t in repo.tool_calls(outcome.run_id)),
        models=tuple(sorted({str(s["model"]) for s in steps if s.get("model")})),
        cell=cell, evaluation_id=evaluation_id,
        evaluation_store=str(getattr(runner.environment.config, "db_path", "")),
        # What was in play. Production only — this is the arm's own view of the
        # procedures it loaded, and a candidate that reached this table must not
        # be reported as one of them.
        skills={str(s["trigger"]): int(s["version"]) for s in repo.list_skills()},
        # What the run *derived* — a different question, and deliberately
        # unfiltered. A task or benchmark run's procedures are recorded as
        # `candidate` from a non-interactive origin (see `skills.record_from_run`),
        # so the production filter would empty this tuple for exactly the runs that
        # produce candidates. Asking "what does this store contain" is not loading
        # anything into production; `status` travels with each entry so the
        # analysis can judge it.
        derived=tuple({"trigger": str(s["trigger"]), "version": int(s["version"]),
                       "title": str(s["title"]), "procedure": str(s["procedure"]),
                       "status": str(s.get("status") or ""),
                       "files": list(s.get("files") or [])}
                      for s in repo.list_skills(status=None, origin=None)),
    )


def run_benchmark(path: str | Path, *, workspace: str | Path, tier: str = "",
                  origin: str = ORIGIN_BENCHMARK,
                  session_id: str | None = None,
                  approvals=None, arm: Arm | None = None,
                  name_prefix: str = "", only_task: str = "") -> BenchmarkRun:
    """Run every task in `path`, judge each, and return the record.

    One session for the whole set: a suite is one invocation, and "what did that
    run do" should have one answer that includes every task it started.

    Each task gets its own directory *and* its own database. Two tasks sharing a
    database would share memory, skills and audit rows, and a suite in which task
    B can see task A's post-mortem is not measuring the tasks.
    """
    tasks = load_tasks(path)
    if only_task:
        # A matrix cell runs exactly one task; the suite is read for its
        # definitions, not to run all of them again for every cell.
        tasks = [t for t in tasks if t.id == only_task]
        if not tasks:
            raise ValueError(f"{path}: 没有 id 为 {only_task!r} 的任务")
    if not tasks:
        raise ValueError(f"{path}: 没有可运行的任务")

    session = session_id or uuid.uuid4().hex[:16]
    records: list[TaskRecord] = []
    environment: dict[str, Any] = {}
    for task in tasks:
        parts = [p for p in (tier, name_prefix, task.id) if p]
        env = RunEnvironment.create(
            workspace, name="/".join(parts), fixture=task.fixture,
            live=task.is_live, pricing=task.pricing,
            model=arm.model if arm else None,
            overrides=dict(arm.overrides) if arm else None,
            skills=arm.skills if arm else ())
        runner = TaskRunner(env, session_id=session, approvals=approvals)
        # The arm's workflow overrides the task's: "what if this were treated as
        # a bugfix" is a question the task file cannot ask, and the run has to
        # record the machine it actually went through.
        effective = (replace(task, workflow=arm.workflow)
                     if arm and arm.workflow else task)
        outcome = runner.run(effective, origin=origin)
        evaluation = evaluate(runner.repo, outcome.run_id, effective)
        # Judged and recorded while this task's database is still open: the
        # evaluation lives beside the run it judges, in that run's own store.
        evaluation_id = record_evaluation(runner.repo, evaluation)
        if not environment:
            environment = _environment_of(runner.harness)
        records.append(_record(runner, effective, outcome, evaluation,
                               cell=arm.label() if arm else "",
                               evaluation_id=evaluation_id))

    return BenchmarkRun(suite=tier or str(path), benchmark_version=BENCHMARK_VERSION,
                        session_id=session, environment=environment,
                        tasks=tuple(records))


# ------------------------------------------------------------- baseline files
def write_baseline(path: str | Path, run: BenchmarkRun, *,
                   source: str = "") -> Path:
    """Write a baseline, and refuse to write over one that exists.

    This is the whole of "a baseline is immutable". Not a file permission and not
    a convention — the one function that writes a baseline will not replace one,
    so a candidate cannot produce a new baseline by re-running the suite; it has to
    name a new one. The refusal is loud on purpose: silently overwriting is exactly
    how a comparison stops meaning anything.
    """
    target = Path(path)
    if target.exists():
        raise FileExistsError(
            f"基线已存在：{target}\n"
            f"基线创建后不可被覆盖 —— 候选必须生成新的基线版本，而不是改写旧的。\n"
            f"换一个路径，例如 {target.with_name(target.stem + '-v2' + target.suffix)}")
    payload = {
        **run.as_json(),
        "kind": "benchmark-baseline",
        # Creation metadata, so a reader can tell when a stored baseline was taken
        # and from which suite directory — without it two baselines of the same
        # suite are indistinguishable after the fact.
        "createdAt": utcnow(),
        "source": source or str(path),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
                      + "\n", encoding="utf-8")
    return target


def load_record(path: str | Path) -> dict:
    """A baseline or a benchmark run, as a plain dict.

    Both are read the same way because they are the same shape — a baseline is a
    benchmark run that was kept. `compare` therefore takes either as either side.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "tasks" not in data:
        raise ValueError(f"{path}: 不是 benchmark 记录（缺少 tasks）")
    version = data.get("schemaVersion")
    if version != BENCHMARK_VERSION:
        raise ValueError(f"{path}: 不支持的记录版本 {version!r}")
    return data


def suite_paths(root: str | Path = "benchmark") -> dict[str, Path]:
    """The three tier directories that exist under `root`."""
    base = Path(root)
    return {tier: base / tier for tier in TIERS if (base / tier).is_dir()}
