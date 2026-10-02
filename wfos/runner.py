"""An isolated place for one run to work.

The harness has always had one working directory per *process*: `cfg.project_root`
is read once and handed to every agent, and the MCP tool layer resolves every path
against it. That is right for `wfos run`, where the point is to edit the project
the operator pointed at. It is wrong for anything that runs a *task* — a benchmark
case or an experiment — because two of those in one process would share a tree, a
database and a capability file, and nothing about the run would say which one a
tool call belonged to.

`RunEnvironment` is the isolation the baseline has been building for itself since
it existed (`_CaseRunner` gave every case its own workspace, config, database and
server). This gives that idea one name and one implementation, so the task runner
(P1) and the baseline cannot drift into two versions of it.

**Isolation here is about what a run can *reach*, not about what it may *do*.** A
fresh `project_root` and a fresh database file keep one run's files, memory,
skills and audit rows out of another's. What a run is permitted to do inside its
own tree is policy's job — roles, path roots, plan scope, approvals — and none of
that changes here.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import events
from .config import AppConfig, default_config, load_config
from .harness.orchestrator import Harness, RunLockedError
from .mcp.client import ToolGateway
from .mcp.server import WfosMcpServer
from .models import ORIGIN_INTERACTIVE, ORIGIN_TASK, TERMINAL_STATUSES
from .storage.repo import Repo
from .tasks import TaskSpec
from .wiki.wiki import WikiClient

ApprovalChecker = Callable[[str, str, dict], bool]

# Where a run's private database lives, relative to its workspace.
_DATA_DIR = ".data"


class RunEnvironment:
    """One run's own workspace, database and stack.

    Built, not configured: the point is that a caller cannot half-isolate a run by
    setting some of the four paths. `create` sets all of them together or the run
    does not start.
    """

    def __init__(self, root: Path, config: AppConfig,
                 skills: tuple[dict[str, Any], ...] = ()):
        self.root = root
        self.config = config
        # Procedures to put in this environment before it runs. An experiment
        # varies the skill that is in play, and a cell's skill has to exist in the
        # cell — seeding it is the only way to hold it constant for that cell
        # without reaching into another cell's database.
        self.skills = tuple(skills)

    @classmethod
    def create(cls, workspace: str | Path, *, name: str,
               fixture: dict[str, str] | None = None,
               live: bool = False,
               pricing: dict[str, Any] | None = None,
               model: str | None = None,
               overrides: dict[str, Any] | None = None,
               skills: tuple[dict[str, Any], ...] = ()) -> RunEnvironment:
        """Materialise `workspace/name` and the config that points only at it.

        `fixture` files are written before anything can read them, so a case never
        observes a half-built tree.

        `live` swaps in the operator's own configured provider. It is opt-in
        because the alternative — defaulting to whatever is configured — would make
        a *frozen* baseline impossible: the same case would answer with a
        different brain on every machine.

        `model` and `overrides` are what make an experiment arm an arm. `model`
        replaces the default model for every role; `overrides` are harness config
        fields, **validated against the dataclass** — an override naming a field
        that does not exist is refused rather than ignored, because an experiment
        whose variable silently did nothing would report the difference between
        two identical runs as a finding.
        """
        root = Path(workspace) / name
        root.mkdir(parents=True, exist_ok=True)
        for filename, content in (fixture or {}).items():
            target = root / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

        config = default_config()
        config.project_root = root
        config.data_dir = root / _DATA_DIR
        config.data_dir.mkdir(parents=True, exist_ok=True)
        config.db_path = config.data_dir / "wfos.db"
        config.wiki_path = config.data_dir / "wiki.db"
        if live:
            config.llm = load_config().llm
        else:
            config.llm.provider = "mock"
        config.pricing = pricing or {}
        if model:
            config.llm.model = model
        for field_name, value in (overrides or {}).items():
            if not hasattr(config.harness, field_name):
                raise ValueError(
                    f"config 覆盖项 {field_name!r} 不是 HarnessConfig 的字段；"
                    f"一个改不动任何东西的变量会把自己伪装成实验结论")
            setattr(config.harness, field_name, value)
        return cls(root, config, skills=skills)

    def config_digest(self) -> str:
        """A digest of the knobs a cell varies, for identifying the cell.

        Deliberately narrower than the environment digest: this is "which arm is
        this", not "were the conditions the same". Two cells differing only in a
        harness knob must have different digests or they would be one arm recorded
        twice.
        """
        harness = self.config.harness
        blob = json.dumps({
            "model": self.config.llm.model,
            "harness": {name: getattr(harness, name)
                        for name in sorted(vars(harness))
                        if name != "safe_commands"},
            "skills": [s.get("trigger") for s in self.skills],
        }, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def harness(self, *, approvals: ApprovalChecker | None = None) -> Harness:
        """A Harness wired to this environment and to nothing else.

        A fresh `Repo`, server, gateway and wiki each time, rather than one cached
        here: a caller that wants two runs must get two stacks, and a shared
        `Repo` would do the thing this class exists to prevent.

        `approvals` defaults to refusing everything — an isolated run has no human
        behind it, and a default that approved would be a default that widens what
        a benchmark can do.
        """
        repo = Repo(self.config.db_path)
        # Seeded as `interactive`, which is what makes them `live` — an experiment
        # that varies a procedure has to have that procedure actually in play, and
        # a candidate-tier row would be filtered out of the prompt instead.
        for seed in self.skills:
            repo.add_skill(str(seed["trigger"]), str(seed.get("title") or seed["trigger"]),
                           str(seed["procedure"]), str(seed.get("evidence_run") or "seed"),
                           files=list(seed.get("files") or []),
                           origin=ORIGIN_INTERACTIVE)
        server = WfosMcpServer(self.config, repo,
                               approval_checker=approvals or (lambda *_: False))
        return Harness(self.config, repo, ToolGateway(server, repo), WikiClient(repo))


class TaskDriver:
    """Drives a task's declared actions against one harness.

    A task is not always "create a run and advance it once". The recovery and
    cross-run cases interrupt, resume, mutate files behind the run's back and
    start a second run to be remembered by the next one. All of that is *how the
    task is set up*, and a runner that ignored `actions` would measure a different
    task than the file declares — silently, because a single `advance` still
    produces a run, a trace and a verdict.

    Shared by `TaskRunner` and the frozen baseline's case runner. Two drivers over
    the same task files would eventually disagree about what a task means, and the
    frozen baseline and the benchmark would then be measuring different things
    while both claiming to run the same cases.
    """

    def __init__(self, harness: Harness, root: str | Path):
        self.harness = harness
        # Where a `mutate` op writes. It stands in for a change made outside the
        # run, which is what the resume contract has to catch.
        self.root = Path(root)
        # The run everything is judged against. A `newRun` action moves it: a
        # single run cannot see its own memory, because memory is written when a
        # run ends.
        self.subject_run_id: str | None = None
        self.child_run_id: str | None = None

    def start(self, task: TaskSpec, *, origin: str,
              session_id: str | None = None) -> dict:
        """Create the subject run. A `newRun` action replaces it later.

        Always created, even when the first action is `newRun`: the run it
        replaces is then "the one to be remembered", which is exactly what the
        cross-run cases need and what the original case runner did.
        """
        run = self.harness.create_run(
            task.input, kind=task.workflow, title=task.id, origin=origin,
            task_id=task.id, task_version=task.version, session_id=session_id)
        self.subject_run_id = run["id"]
        return run

    def drive(self, task: TaskSpec, *, origin: str,
              session_id: str | None = None) -> None:
        """Apply every action in order."""
        for op in task.actions or ({"op": "advance"},):
            if op.get("op") == "mutate":
                # Not a harness call and not async: a change made *outside* the
                # run, which is exactly what the resume contract has to notice.
                for name, content in (op.get("files") or {}).items():
                    target = self.root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(content, encoding="utf-8")
                continue
            asyncio.run(self._apply(op, task, origin=origin, session_id=session_id))

    async def _apply(self, op: dict, task: TaskSpec, *, origin: str,
                     session_id: str | None) -> None:
        name = op.get("op")
        run_id = self.subject_run_id
        if name == "newRun":
            created = self.harness.create_run(
                str(op.get("request") or task.input),
                kind=str(op.get("kind") or task.workflow), title=task.id,
                origin=origin, task_id=task.id, task_version=task.version,
                session_id=session_id)
            self.subject_run_id = created["id"]
        elif name == "advance":
            await self.harness.advance(run_id)
        elif name == "interrupt":
            # Stands in for the process dying mid-run: the status is left at the
            # harness's own in-flight value, and the lease is free because the
            # claiming process is gone. This is the state a resume has to cope
            # with, and it cannot be reached by driving the API politely.
            self.harness.repo.update_run(run_id, status="running")
        elif name == "resume":
            await self.harness.resume(run_id, force_stale=bool(op.get("forceStale")))
        elif name == "advanceChild":
            child = self.harness.repo.get_run(run_id)["payload"].get("child_run_id")
            if child:
                self.child_run_id = child
                await self.harness.advance(child)
        elif name == "resumeChild":
            child = (self.child_run_id
                     or self.harness.repo.get_run(run_id)["payload"].get("child_run_id"))
            if child:
                self.child_run_id = child
                await self.harness.resume_after_child(child)
        else:
            raise ValueError(f"未知的 action op: {name!r}")


@dataclass(frozen=True)
class TaskOutcome:
    """What running one task produced, before anything judges it.

    Deliberately no verdict. Whether the run succeeded is not the runner's
    question — it is the evaluator's, and it answers it from the record rather
    than from the runner's opinion. A runner that scored its own work would make
    the evaluator's independence decorative.
    """

    task_id: str
    task_version: int
    run_id: str
    session_id: str
    # The run's own status: `completed`, `failed`, `waiting_approval`, …
    # Reported as it is, not folded into a boolean — `waiting_approval` is not a
    # failure, and a caller that needs to wait has to be able to tell.
    status: str
    state: str
    steps: int
    failure_classes: tuple[str, ...]
    changed_paths: tuple[str, ...]
    error: str

    @property
    def reached_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES


class TaskRunner:
    """Runs one task in one isolated environment.

    The pieces already existed and were not joined up: `TaskSpec` says what to
    run, `RunEnvironment` says where, and the Harness knows how. This is the
    join, and it is deliberately thin — anything it decided on its own would be a
    policy the task file cannot express and a reader cannot find.
    """

    def __init__(self, environment: RunEnvironment, *,
                 session_id: str | None = None,
                 approvals: ApprovalChecker | None = None):
        self.environment = environment
        # One per runner, because a session is one invocation. A suite that runs
        # ten tasks passes the same one to all ten, and "what did that run do" is
        # then answerable across every run it started.
        self.session_id = session_id or uuid.uuid4().hex[:16]
        self.harness = environment.harness(approvals=approvals)

    @property
    def repo(self) -> Repo:
        """The repository this runner wrote to, for reading the trace back."""
        return self.harness.repo

    def run(self, task: TaskSpec, *, origin: str = ORIGIN_TASK) -> TaskOutcome:
        """Load-check, run, and report what happened.

        `task.loaded` is emitted before the run exists, with no `run_id`. That is
        not a gap: loading and validating a task is a fact about the *invocation*,
        and it is the last thing that happens before a run can exist at all — a
        task that fails validation has no run to attach to. The run's own trace
        names its task through `run.created`'s `task_id`/`task_version`.
        """
        events.record(self.repo, events.TASK_LOADED, session_id=self.session_id,
                      actor=events.ACTOR_RUNNER,
                      payload={"task_id": task.id, "task_version": task.version,
                               "workflow": task.workflow, "brain": task.brain,
                               "origin": origin})
        driver = TaskDriver(self.harness, self.environment.root)
        driver.start(task, origin=origin, session_id=self.session_id)
        # Another process may hold the lease. The run is on the record either
        # way, and the outcome describes the state it actually reached.
        with contextlib.suppress(RunLockedError):
            driver.drive(task, origin=origin, session_id=self.session_id)
        return self._outcome(task, self.repo.get_run(driver.subject_run_id))

    def _outcome(self, task: TaskSpec, run: dict) -> TaskOutcome:
        steps = self.repo.steps_for_run(run["id"])
        return TaskOutcome(
            task_id=task.id, task_version=task.version,
            run_id=run["id"], session_id=run.get("session_id") or self.session_id,
            status=run["status"], state=run["state"], steps=len(steps),
            failure_classes=tuple(sorted({s["failure_class"] for s in steps
                                          if s.get("failure_class")})),
            changed_paths=tuple(self.repo.affected_paths_for_run(
                run["id"], agent="implementer")),
            error=run.get("error") or "")


def run_task(task: TaskSpec, *, workspace: str | Path, name: str | None = None,
             origin: str = ORIGIN_TASK,
             approvals: ApprovalChecker | None = None,
             session_id: str | None = None) -> TaskOutcome:
    """One task, one environment, one run — the whole of it.

    `name` decides the workspace directory and defaults to the task id. An
    experiment that runs the same task under several configurations must pass a
    different name for each, or they would share a tree and a database and stop
    being separate runs at all.
    """
    environment = RunEnvironment.create(
        workspace, name=name or task.id, fixture=task.fixture,
        live=task.is_live, pricing=task.pricing)
    return TaskRunner(environment, session_id=session_id,
                      approvals=approvals).run(task, origin=origin)
