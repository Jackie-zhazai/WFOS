"""Tasks: what to run, declared rather than implied.

Until now the harness's input was a sentence. `route()` scored it for keywords and
picked a state machine, which is the right shape for `wfos run` — a person typed
something and wants it done. It is the wrong shape for anything that must be
*reproducible*: a benchmark case, an experiment arm, an RSI candidate's proving
ground. Those need the request, the fixture, the workflow, the limits and the
success criteria to be declared up front, versioned, and the same next time.

**This does not invent a second format.** `benchmarks/cases.json` already is a
validated, versioned, fail-closed task set — id, request, kind, fixture, expect,
stepBudget, brain — and `baseline.load_cases` already refuses anything malformed. A
`TaskSpec` is a *view* over those dicts that names the fields the spec calls for
and validates the ones it adds. Two loaders over two formats would be two truths
about what a task is, and the second would drift.

Where the spec's name and the file's name differ, **both are accepted and they
must agree**. A file may migrate at its own pace; neither name silently wins.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .harness import statemachine as sm

# The workflows a *task* may declare. Not `sm.MACHINES`: that table also carries
# the interactive kind, which has no state machine to drive.
TASK_WORKFLOWS = sm.ADVANCEABLE_KINDS

SCHEMA_VERSION = 1
# The three-tier suite directory. One default for both the task layer and the
# reliability baseline, so "all the tasks" cannot mean two different sets.
DEFAULT_TASKS = Path("benchmark")

def _one_of(case: dict, canonical: str, alias: str) -> Any:
    """A field that may be spelled the spec's way or the file's way."""
    has_canonical, has_alias = canonical in case, alias in case
    if has_canonical and has_alias and case[canonical] != case[alias]:
        raise ValueError(
            f"task {case.get('id')!r} 的 {canonical} 与 {alias} 同时存在且不一致："
            f"{case[canonical]!r} vs {case[alias]!r}")
    if has_canonical:
        return case[canonical]
    return case.get(alias)


@dataclass(frozen=True)
class TaskSpec:
    """One task, fully declared.

    Frozen because a task is a thing that gets *compared against*: a run records
    which revision produced it, and a benchmark compares results across revisions.
    A task that could be edited in place after the fact would make every recorded
    result unanchorable.
    """

    id: str
    version: int
    description: str
    category: str
    # The state machine this task runs. The spec calls it `workflow`; a run
    # records the same value as its `kind`, and `MACHINES` is the one vocabulary
    # both are checked against.
    workflow: str
    # Which skill applies, "" for none. Carried now because an RSI candidate is
    # scoped to a task set, and a result has to say which skill was in play.
    skill: str
    input: str
    fixture: dict[str, str]
    # Limits. `stepBudget` is the one that exists today, normalised in here from
    # either spelling so there is a single place it is read and validated.
    constraints: dict[str, Any]
    # How success is judged. Empty until the evaluator (P2) fills it; declared
    # here so a task cannot be written without saying what would prove it.
    evaluator: dict[str, Any]
    expected: dict[str, Any]
    brain: str
    actions: tuple[dict, ...]
    pricing: dict[str, Any] = field(default_factory=dict)

    @property
    def step_budget(self) -> int:
        """A ceiling on executed states, not a target."""
        return int(self.constraints["stepBudget"])

    @classmethod
    def from_case(cls, case: dict) -> TaskSpec:
        """Build one from a case dict, or refuse it.

        Fail-closed throughout, on the same reasoning `load_cases` uses: a task
        missing an id, an input or an expectation is not "a task with defaults",
        it is one that would run without meaning anything. A task that asserts
        nothing passes forever, which is worse than having no task.
        """
        if not isinstance(case, dict):
            raise ValueError(f"task 必须是对象，实际 {type(case).__name__}")
        task_id = str(case.get("id", "")).strip()
        if not task_id:
            raise ValueError("task 缺少 id")

        version = case.get("version", 1)
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            raise ValueError(f"task {task_id} 的 version 必须是正整数，实际 {version!r}")

        workflow = str(_one_of(case, "workflow", "kind") or "")
        # A task drives the workflow state machine, so only the kinds that *have*
        # one are legal here. This used to read `sm.MACHINES`, which meant adding a
        # machine to the harness silently widened what a task file could declare —
        # `workflow: "chat"` would have been accepted and then driven into a
        # machine `advance` refuses. The task layer gets its own closed list.
        if workflow not in TASK_WORKFLOWS:
            raise ValueError(
                f"task {task_id} 的 workflow 未知：{workflow!r}；"
                f"可选 {', '.join(sorted(TASK_WORKFLOWS))}")

        text = str(_one_of(case, "input", "request") or "").strip()
        if not text:
            raise ValueError(f"task {task_id} 缺少 input/request")

        expected = _one_of(case, "expected", "expect")
        if not isinstance(expected, dict) or not expected:
            raise ValueError(
                f"task {task_id} 必须有非空的 expected —— 不断言任何东西的任务永远通过")

        fixture = case.get("fixture") or {}
        if not isinstance(fixture, dict):
            raise ValueError(f"task {task_id} 的 fixture 必须是对象")

        constraints = dict(case.get("constraints") or {})
        if "stepBudget" in case:
            if "stepBudget" in constraints and constraints["stepBudget"] != case["stepBudget"]:
                raise ValueError(
                    f"task {task_id} 的 stepBudget 同时存在且不一致："
                    f"{case['stepBudget']!r} vs {constraints['stepBudget']!r}")
            constraints.setdefault("stepBudget", case["stepBudget"])
        budget = constraints.get("stepBudget")
        if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
            raise ValueError(f"task {task_id} 缺少正整数 stepBudget，实际 {budget!r}")

        evaluator = case.get("evaluator") or {}
        if not isinstance(evaluator, dict):
            raise ValueError(f"task {task_id} 的 evaluator 必须是对象")
        # A gate naming an invariant the task never declared has no expected value
        # to compare against, and defaulting one would be this loader inventing a
        # requirement. Refused here rather than resolved at judgement time.
        for block in ("gate", "checks"):
            declared = evaluator.get(block)
            if isinstance(declared, list):
                unknown = [str(n) for n in declared if str(n) not in expected]
                if unknown:
                    raise ValueError(
                        f"task {task_id} 的 evaluator.{block} 引用了 expected 里没有的"
                        f"不变式：{'、'.join(unknown)}")

        brain = str(case.get("brain", "mock"))
        if brain not in ("mock", "live"):
            raise ValueError(f"task {task_id} 的 brain 必须是 mock 或 live，实际 {brain!r}")

        return cls(
            id=task_id, version=version,
            description=str(case.get("description", "")),
            category=str(case.get("category", "")),
            workflow=workflow,
            skill=str(case.get("skill", "")),
            input=text,
            fixture={str(k): str(v) for k, v in fixture.items()},
            constraints=constraints,
            evaluator=evaluator,
            expected=dict(expected),
            brain=brain,
            actions=tuple(case.get("actions") or ({"op": "advance"},)),
            pricing=dict(case.get("pricing") or {}),
        )

    @property
    def is_live(self) -> bool:
        return self.brain == "live"


def _cases_in(path: Path) -> list[dict]:
    """The case list in one file, schema-checked to the same version."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError(f"{path}: 不支持的 schemaVersion {data.get('schemaVersion')!r}")
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{path}: cases 必须是非空列表")
    return cases


def load_tasks(path: str | Path | None = None) -> list[TaskSpec]:
    """Every task in `path`, which may be one file or a directory of them.

    A directory is the shape the three tiers take (`benchmark/smoke`,
    `benchmark/regression`, `benchmark/challenge`), and reading it here means the
    tier sets do not need a loader of their own.

    Ids must be unique **across the whole input**, not per file: a task that
    appears twice is two answers to "what does this task require", and whichever
    loaded last would silently win.
    """
    src = Path(path or DEFAULT_TASKS)
    # Recursive, because the three tiers are directories of files
    # (`benchmark/smoke/*.json`), not a flat pile. A directory that only looked at
    # its own top level would find nothing there and report a suite of zero
    # tasks — which reads as "no failures", not as "nothing ran".
    files = sorted(src.rglob("*.json")) if src.is_dir() else [src]

    specs: list[TaskSpec] = []
    seen: dict[str, str] = {}
    for file in files:
        for case in _cases_in(file):
            spec = TaskSpec.from_case(case)
            if spec.id in seen:
                raise ValueError(
                    f"task id 重复: {spec.id}（{seen[spec.id]} 与 {file}）")
            seen[spec.id] = str(file)
            specs.append(spec)
    return specs
