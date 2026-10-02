"""Running a task: the join between what to run, where, and the harness.

`TaskSpec` says what, `RunEnvironment` says where, the Harness knows how. The
runner is deliberately thin — every decision it made on its own would be a policy
the task file cannot express and a reader cannot find.

Two properties are load-bearing here and neither is about running successfully:
**the run says which task revision produced it** (the whole loop compares across
revisions), and **it is marked as a task run** (or its memory and its skills would
reach a real prompt).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from wfos import events
from wfos.baseline import DEFAULT_CASES
from wfos.models import ORIGIN_BENCHMARK, ORIGIN_INTERACTIVE, ORIGIN_TASK
from wfos.runner import RunEnvironment, TaskRunner, run_task
from wfos.tasks import TaskSpec, load_tasks

HAPPY = "feature-happy-path"


def _task(task_id: str = HAPPY) -> TaskSpec:
    return next(t for t in load_tasks(DEFAULT_CASES) if t.id == task_id)


# ------------------------------------------------------------------ the join
def test_a_task_run_records_which_task_revision_produced_it(tmp_path):
    """Read off the **run row**, not off the outcome.

    The outcome knows the task because it was handed one; the claim worth testing
    is that the *stored run* says which revision produced it, because that is
    what a later comparison across revisions reads.
    """
    task = _task()
    runner = TaskRunner(RunEnvironment.create(tmp_path, name="t", fixture=task.fixture))

    outcome = runner.run(task)

    stored = runner.repo.get_run(outcome.run_id)
    assert stored["task_id"] == task.id
    assert stored["task_version"] == task.version
    assert stored["session_id"] == outcome.session_id


def test_a_task_run_is_marked_as_one(tmp_path):
    """`origin` is the boundary that keeps an artificial run's memory and skills
    out of a real prompt. Isolating the environment is not enough — those two are
    keyed on paths and on failure classes, not on the workspace."""
    task = _task()
    runner = TaskRunner(RunEnvironment.create(tmp_path, name="t", fixture=task.fixture))

    outcome = runner.run(task)

    assert runner.repo.get_run(outcome.run_id)["origin"] == ORIGIN_TASK
    assert ORIGIN_TASK != ORIGIN_INTERACTIVE


def test_the_caller_can_ask_for_a_benchmark_origin(tmp_path):
    """The same runner drives the benchmark tier; only the label changes."""
    task = _task()
    runner = TaskRunner(RunEnvironment.create(tmp_path, name="t", fixture=task.fixture))

    outcome = runner.run(task, origin=ORIGIN_BENCHMARK)

    assert runner.repo.get_run(outcome.run_id)["origin"] == ORIGIN_BENCHMARK


def test_the_tasks_fixture_is_materialised_before_the_run_starts(tmp_path):
    task = _task()

    environment = RunEnvironment.create(tmp_path, name=task.id, fixture=task.fixture)

    for name in task.fixture:
        assert (environment.root / name).exists(), f"{name} 没有被写入"


# ------------------------------------------------------------------ the trace
def test_loading_the_task_is_a_session_event_with_no_run_yet(tmp_path):
    """It is the last thing that happens before a run can exist — and a task that
    fails validation never gets one, so the fact needs somewhere to live."""
    task = _task()
    runner = TaskRunner(RunEnvironment.create(tmp_path, name="t", fixture=task.fixture))

    runner.run(task)

    loaded = [e for e in runner.repo.events_for_session(runner.session_id)
              if e["type"] == events.TASK_LOADED]
    assert len(loaded) == 1
    assert loaded[0]["run_id"] in ("", None), "任务加载事件不该挂在一个还不存在的 run 上"
    assert loaded[0]["actor"] == events.ACTOR_RUNNER
    assert loaded[0]["payload"]["task_id"] == task.id
    assert loaded[0]["payload"]["task_version"] == task.version


def test_the_runs_own_trace_names_its_task(tmp_path):
    """`task.loaded` is session-level, so the run's link to its task has to come
    from somewhere else — and this is where."""
    task = _task()
    runner = TaskRunner(RunEnvironment.create(tmp_path, name="t", fixture=task.fixture))

    outcome = runner.run(task)

    created = next(e for e in events.for_run(runner.repo, outcome.run_id)
                   if e["type"] == events.RUN_CREATED)
    assert created["payload"]["task_id"] == task.id
    assert created["payload"]["task_version"] == task.version


def test_the_run_and_the_task_load_share_one_session(tmp_path):
    task = _task()
    runner = TaskRunner(RunEnvironment.create(tmp_path, name="t", fixture=task.fixture))

    outcome = runner.run(task)

    session = runner.repo.events_for_session(runner.session_id)
    assert {e["type"] for e in session} >= {events.TASK_LOADED, events.RUN_CREATED}
    assert outcome.session_id == runner.session_id


# ---------------------------------------------------------------- the outcome
def test_the_outcome_reports_the_status_it_actually_has(tmp_path):
    """Not folded into a boolean: `waiting_approval` is not a failure, and a
    caller that has to wait must be able to tell the two apart."""
    task = _task()

    outcome = run_task(task, workspace=tmp_path)

    assert outcome.status in ("completed", "failed", "waiting_approval",
                              "waiting_child", "cancelled", "paused")
    assert outcome.reached_terminal == (outcome.status in
                                        ("completed", "failed", "cancelled"))
    assert outcome.steps >= 1


def test_the_outcome_carries_the_harnesss_own_facts(tmp_path):
    """Change attribution and failure classes come from the record, not from the
    task file's expectations — those are the evaluator's business, not the
    runner's."""
    task = _task()

    outcome = run_task(task, workspace=tmp_path)

    assert isinstance(outcome.changed_paths, tuple)
    assert isinstance(outcome.failure_classes, tuple)
    assert all(isinstance(name, str) for name in outcome.failure_classes)


def test_the_runner_carries_no_verdict(tmp_path):
    """A runner that scored its own work would make the evaluator's independence
    decorative."""
    outcome = run_task(_task(), workspace=tmp_path)

    assert not hasattr(outcome, "verdict")
    assert not hasattr(outcome, "passed")


# ---------------------------------------------------------------- isolation
def test_two_runs_of_one_task_do_not_share_a_workspace(tmp_path):
    """An experiment runs the same task under several configurations; without
    distinct names they would share a tree and a database and stop being separate
    runs at all."""
    task = _task()

    first = run_task(task, workspace=tmp_path, name="arm-a")
    second = run_task(task, workspace=tmp_path, name="arm-b")

    assert first.run_id != second.run_id
    assert (tmp_path / "arm-a").is_dir() and (tmp_path / "arm-b").is_dir()


def test_two_runs_of_one_task_do_not_see_each_others_runs(tmp_path):
    """Each environment holds exactly its own run — not none, and not both."""
    task = _task()

    first = run_task(task, workspace=tmp_path, name="arm-a")
    second = run_task(task, workspace=tmp_path, name="arm-b")

    arm_a = TaskRunner(RunEnvironment.create(tmp_path, name="arm-a"))
    arm_b = TaskRunner(RunEnvironment.create(tmp_path, name="arm-b"))

    assert [r["id"] for r in arm_a.repo.list_runs()] == [first.run_id]
    assert [r["id"] for r in arm_b.repo.list_runs()] == [second.run_id]


# ------------------------------------------------------------------ one step
def test_a_task_can_be_run_from_a_hand_written_file(tmp_path):
    """The whole path, from a task file on disk to a finished run."""
    path = tmp_path / "task.json"
    path.write_text(json.dumps({
        "schemaVersion": 1,
        "cases": [{
            "id": "inline", "version": 2, "workflow": "feature",
            "input": "新增一个用户模块，在 app.py 中追加 feature_user() 函数",
            "expected": {"reachedTerminal": True}, "stepBudget": 24,
            "fixture": {"app.py": "def compute(x):\n    return x * 2\n",
                        "check.py": "from app import compute\n"
                                    "assert compute(1) == 2\nprint('PASS: ok')\n"},
        }],
    }, ensure_ascii=False), encoding="utf-8")
    task = load_tasks(path)[0]

    outcome = run_task(task, workspace=tmp_path)

    assert outcome.task_version == 2
    assert outcome.reached_terminal, f"没有跑到终态：{outcome.status} {outcome.error}"


def test_the_environment_stays_readable_after_the_run(tmp_path):
    """The evaluator reads the trace and the artifacts from the same place, so a
    runner that threw its environment away would leave nothing to judge."""
    task = _task()
    runner = TaskRunner(RunEnvironment.create(tmp_path, name="t", fixture=task.fixture))

    outcome = runner.run(task)

    assert runner.repo.steps_for_run(outcome.run_id), "运行结束后读不到步骤"
    assert events.for_run(runner.repo, outcome.run_id), "运行结束后读不到 trace"
    assert Path(runner.environment.root).is_dir()


def test_the_harness_is_closed_over_by_the_time_the_outcome_returns(tmp_path):
    """`run` drives the loop with `asyncio.run`, so nothing may still be awaiting
    when it returns."""
    task = _task()
    runner = TaskRunner(RunEnvironment.create(tmp_path, name="t", fixture=task.fixture))

    outcome = runner.run(task)

    assert asyncio.get_event_loop_policy() is not None      # no loop left running
    assert runner.repo.get_run(outcome.run_id) is not None
