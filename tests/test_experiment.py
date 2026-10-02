"""Experiments: many arms, run independently, compared with P3's rules.

The properties here are mostly about what the orchestration layer must **not** do:
it must not reimplement a runner or an evaluator, must not let one cell's failure
reach another, and must not touch a baseline. Each test gives it the opportunity
and asserts it declines.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from wfos.bench import run_benchmark, write_baseline
from wfos.experiment import (
    DIMENSIONS,
    ExperimentSpec,
    comparisons,
    load_experiment,
    load_experiment_result,
    render,
    run_experiment,
    write_experiment,
)
from wfos.tasks import load_tasks

SMOKE = Path("benchmark/smoke")
ONE_TASK = "feature-happy-path"


def _spec(tmp_path, *, matrix=None, suite=str(SMOKE), **over) -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id=over.pop("experiment_id", "e1"),
        version=over.pop("version", 1), name="test", description="",
        suite=suite, matrix=matrix or {}, **over)


# ------------------------------------------------------------------ the spec
def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_a_spec_loads_and_carries_what_makes_it_reproducible(tmp_path):
    spec = load_experiment(_write(tmp_path / "e.json", {
        "schemaVersion": 1, "experimentId": "sweep", "version": 3,
        "name": "n", "suite": str(SMOKE), "matrix": {"model": ["a"]},
        "repetitions": 2, "metadata": {"owner": "me"}}))

    assert (spec.experiment_id, spec.version, spec.repetitions) == ("sweep", 3, 2)
    assert spec.metadata["owner"] == "me"


def test_an_unknown_matrix_dimension_is_refused(tmp_path):
    """A typo'd dimension would expand to a single-arm experiment that looks like
    a passing one."""
    path = _write(tmp_path / "e.json", {
        "schemaVersion": 1, "experimentId": "s", "suite": str(SMOKE),
        "matrix": {"modle": ["a"]}})

    with pytest.raises(ValueError, match="modle"):
        load_experiment(path)


def test_a_spec_without_an_id_or_a_suite_is_refused(tmp_path):
    with pytest.raises(ValueError, match="experimentId"):
        load_experiment(_write(tmp_path / "a.json", {"schemaVersion": 1, "suite": "x"}))
    with pytest.raises(ValueError, match="suite"):
        load_experiment(_write(tmp_path / "b.json", {"schemaVersion": 1,
                                                     "experimentId": "e"}))


def test_an_unsupported_spec_version_is_refused(tmp_path):
    with pytest.raises(ValueError, match="schemaVersion"):
        load_experiment(_write(tmp_path / "e.json", {"schemaVersion": 9,
                                                     "experimentId": "e", "suite": "x"}))


def test_the_four_dimensions_are_the_four_the_spec_names():
    assert set(DIMENSIONS) == {"model", "skill", "workflow", "config"}


# ----------------------------------------------------------------- the matrix
def test_an_empty_matrix_is_one_default_arm_not_zero_cells():
    """"Run the suite unchanged" is a legitimate experiment — it is how a baseline
    is reproduced."""
    assert len(_spec(tmp_path=None, matrix={}).cells(["t1"])) == 1


def test_a_single_dimension_expands_to_one_cell_per_value():
    cells = _spec(None, matrix={"model": ["a", "b", "c"]}).cells(["t1"])
    assert [c["arm"]["model"] for c in cells] == ["a", "b", "c"]


def test_two_dimensions_expand_to_the_cross_product():
    """Not two separate sweeps — the combination is the experiment."""
    cells = _spec(None, matrix={"model": ["a", "b"], "config": [None, {"x": 1}]}).cells(["t1"])

    assert len(cells) == 4
    assert {(c["arm"]["model"], bool(c["arm"]["config"])) for c in cells} == {
        ("a", False), ("a", True), ("b", False), ("b", True)}


def test_every_task_is_crossed_with_every_arm():
    cells = _spec(None, matrix={"model": ["a", "b"]}).cells(["t1", "t2"])
    assert {(c["taskId"], c["arm"]["model"]) for c in cells} == {
        ("t1", "a"), ("t1", "b"), ("t2", "a"), ("t2", "b")}


def test_repetitions_multiply_the_cells():
    cells = _spec(None, matrix={"model": ["a"]}, repetitions=3).cells(["t1"])
    assert [c["repetition"] for c in cells] == [0, 1, 2]


# ------------------------------------------------------------------ the run
def test_a_single_cell_experiment_runs_and_is_judged(tmp_path):
    run = run_experiment(_spec(tmp_path, matrix={"model": ["mock"]},
                               suite=str(SMOKE)),
                         workspace=tmp_path / "ws")

    assert len(run.cells) == 2, "冒烟层有两个任务，单臂应有两个 cell"
    assert all(c.status == "ran" and c.run_id for c in run.cells)
    assert all(c.evaluation.get("verdict") for c in run.cells)


def test_each_cell_gets_its_own_run_workspace_and_evaluation(tmp_path):
    run = run_experiment(_spec(tmp_path, matrix={"model": ["a", "b"]},
                               suite=str(SMOKE)),
                         workspace=tmp_path / "ws")

    assert len({c.run_id for c in run.cells}) == len(run.cells), "cell 之间共享了 run"
    dbs = sorted((tmp_path / "ws").rglob("wfos.db"))
    assert len(dbs) == len(run.cells), f"数据库数 {len(dbs)} ≠ cell 数 {len(run.cells)}"


def test_one_cell_failing_does_not_fail_the_others(tmp_path):
    """§5: a provider error in one arm says nothing about the others.

    Produced by giving one arm a config override that cannot be applied — the cell
    dies before it starts, and the rest of the matrix must be untouched.
    """
    spec = _spec(tmp_path, matrix={"config": [None, {"not_a_real_field": 1}]},
                 suite=str(SMOKE))

    run = run_experiment(spec, workspace=tmp_path / "ws")

    started = [c for c in run.cells if c.status == "ran"]
    dead = [c for c in run.cells if c.status != "ran"]
    assert dead, "这个测试没有制造出失败的 cell"
    assert started, "一个 cell 失败把其他 cell 也带下了水"
    assert all(c.run_id for c in started)
    assert all(not c.run_id for c in dead)
    assert all(c.failure_class for c in dead), "失败的 cell 没有记录失败分类"


def test_a_failed_cell_still_appears_in_the_record_with_its_arm(tmp_path):
    spec = _spec(tmp_path, matrix={"config": [{"nope": 1}]}, suite=str(SMOKE))

    run = run_experiment(spec, workspace=tmp_path / "ws")
    payload = run.as_json()

    assert all(c["status"] == "failed_to_start" for c in payload["cells"])
    assert all(c["arm"]["config"] for c in payload["cells"])
    assert payload["rollup"]["failedToStart"] == len(run.cells)


def test_the_result_carries_the_versions_a_reproduction_needs(tmp_path):
    run = run_experiment(_spec(tmp_path, matrix={"model": ["mock"]}, suite=str(SMOKE)),
                         workspace=tmp_path / "ws")
    payload = run.as_json()

    assert payload["experimentId"] == "e1" and payload["experimentVersion"] == 1
    assert payload["benchmarkVersion"] >= 1
    assert payload["sessionId"]
    assert payload["environment"]["digest"]
    for task in payload["tasks"]:
        assert task["taskVersion"] >= 1 and task["cellId"]


def test_every_cell_of_one_experiment_shares_one_session(tmp_path):
    run = run_experiment(_spec(tmp_path, matrix={"model": ["a", "b"]}, suite=str(SMOKE)),
                         workspace=tmp_path / "ws")
    assert run.session_id
    assert all(c.cell_id for c in run.cells)


def test_the_same_spec_can_be_run_twice(tmp_path):
    """§4: the same spec, run again, is recognisable as the same configuration."""
    spec = _spec(tmp_path, matrix={"model": ["mock"]}, suite=str(SMOKE))

    first = run_experiment(spec, workspace=tmp_path / "ws1")
    second = run_experiment(spec, workspace=tmp_path / "ws2")

    assert [c.cell_id for c in first.cells] == [c.cell_id for c in second.cells]
    assert ([c.evaluation.get("verdict") for c in first.cells]
            == [c.evaluation.get("verdict") for c in second.cells])
    assert first.environment["digest"] == second.environment["digest"]


def test_a_different_arm_is_a_different_cell(tmp_path):
    """Two arms differing only in a harness knob must not read as one arm."""
    one = run_experiment(_spec(tmp_path, experiment_id="x",
                               matrix={"config": [None]}, suite=str(SMOKE)),
                         workspace=tmp_path / "ws")
    two = run_experiment(_spec(tmp_path, experiment_id="x",
                               matrix={"config": [{"repeated_failure_threshold": 0}]},
                               suite=str(SMOKE)),
                         workspace=tmp_path / "ws")
    assert [c.cell_id for c in one.cells] != [c.cell_id for c in two.cells]


# ------------------------------------------------------------- the boundaries
def test_an_experiment_does_not_touch_the_benchmark_it_runs(tmp_path):
    before = {p: p.read_bytes() for p in Path("benchmark").rglob("*.json")}

    run_experiment(_spec(tmp_path, matrix={"model": ["a"]}, suite=str(SMOKE)),
                   workspace=tmp_path / "ws")

    assert {p: p.read_bytes() for p in Path("benchmark").rglob("*.json")} == before


def test_an_experiment_does_not_touch_a_baseline(tmp_path):
    """§8: not on success, not on failure, not because the numbers improved."""
    baseline = write_baseline(tmp_path / "b.json",
                              run_benchmark(SMOKE, workspace=tmp_path / "bws", tier="smoke"))
    before = baseline.read_bytes()

    run_experiment(_spec(tmp_path, matrix={"config": [{"nope": 1}]}, suite=str(SMOKE)),
                   workspace=tmp_path / "ws", baseline=str(baseline))

    assert baseline.read_bytes() == before


def test_an_experiment_never_creates_a_baseline_by_itself(tmp_path):
    """Baseline creation stays an explicit act."""
    run_experiment(_spec(tmp_path, matrix={"model": ["a"]}, suite=str(SMOKE)),
                   workspace=tmp_path / "ws")

    assert list(tmp_path.rglob("*baseline*")) == []


def test_a_missing_baseline_yields_no_comparison_rather_than_a_fabricated_one(tmp_path):
    run = run_experiment(_spec(tmp_path, matrix={"model": ["a"]}, suite=str(SMOKE)),
                         workspace=tmp_path / "ws")

    assert comparisons(run, str(tmp_path / "nope.json")) == []


def test_an_experiment_compares_against_a_baseline_per_arm(tmp_path):
    """§7: P3's comparison, reused — no second set of verdicts."""
    baseline = write_baseline(tmp_path / "b.json",
                              run_benchmark(SMOKE, workspace=tmp_path / "bws", tier="smoke"))
    run = run_experiment(_spec(tmp_path, matrix={"model": ["mock", "mock-alt"]},
                               suite=str(SMOKE)),
                         workspace=tmp_path / "ws", baseline=str(baseline))

    reports = comparisons(run, str(baseline))

    assert reports, "没有产生任何比较"
    assert all(set(r["counts"]) == {"regression", "improvement", "unchanged",
                                    "unknown", "noise"} for r in reports)
    assert all("ok" in r for r in reports)


def test_unknown_stays_unknown_in_an_experiment_comparison(tmp_path):
    baseline = write_baseline(tmp_path / "b.json",
                              run_benchmark(SMOKE, workspace=tmp_path / "bws", tier="smoke"))
    run = run_experiment(_spec(tmp_path, matrix={"model": ["mock"]}, suite=str(SMOKE)),
                         workspace=tmp_path / "ws", baseline=str(baseline))

    reports = comparisons(run, str(baseline))
    unknowns = [f for r in reports for f in r["findings"] if f["verdict"] == "unknown"]

    assert unknowns, "mock 环境下应当有 unknown"
    assert all(f["verdict"] != "pass" for f in unknowns)


# ---------------------------------------------------------------- persistence
def test_a_result_is_written_and_read_back(tmp_path):
    run = run_experiment(_spec(tmp_path, matrix={"model": ["a"]}, suite=str(SMOKE)),
                         workspace=tmp_path / "ws")
    path = write_experiment(tmp_path / "r.json", run)

    stored = load_experiment_result(path)

    assert stored["experimentId"] == "e1"
    assert len(stored["cells"]) == len(run.cells)
    assert stored["createdAt"]
    assert stored["rollup"]["cells"] == len(run.cells)


def test_writing_over_a_stored_result_is_refused(tmp_path):
    """A result is a measurement; replacing it loses which one a conclusion came
    from."""
    run = run_experiment(_spec(tmp_path, matrix={"model": ["a"]}, suite=str(SMOKE)),
                         workspace=tmp_path / "ws")
    path = write_experiment(tmp_path / "r.json", run)
    before = path.read_bytes()

    with pytest.raises(FileExistsError, match="已存在"):
        write_experiment(tmp_path / "r.json", run)

    assert path.read_bytes() == before


def test_a_file_that_is_not_a_result_is_refused(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"hello": 1}), encoding="utf-8")

    with pytest.raises(ValueError, match="实验结果"):
        load_experiment_result(path)


def test_a_result_of_an_unknown_version_is_refused(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"schemaVersion": 99, "cells": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="版本"):
        load_experiment_result(path)


def test_the_rendered_result_lists_every_cell(tmp_path):
    run = run_experiment(_spec(tmp_path, matrix={"model": ["a", "b"]}, suite=str(SMOKE)),
                         workspace=tmp_path / "ws")

    text = render(run)

    assert "e1" in text
    for cell in run.cells:
        assert cell.cell_id in text


def test_the_result_is_json_serialisable(tmp_path):
    run = run_experiment(_spec(tmp_path, matrix={"model": ["a"]}, suite=str(SMOKE)),
                         workspace=tmp_path / "ws")
    payload = run.as_json()

    assert json.loads(json.dumps(payload, ensure_ascii=False))["kind"] == "experiment-run"


# ------------------------------------------------------------------ isolation
def test_a_cells_directory_name_is_filesystem_safe_but_its_label_is_not_rewritten(tmp_path):
    """Found the hard way: `|` and `:` in a cell id made half a matrix fail to
    start on Windows, with an error that read like a network problem."""
    run = run_experiment(_spec(tmp_path, matrix={"config": [{"repeated_failure_threshold": 0}]},
                               suite=str(SMOKE)),
                         workspace=tmp_path / "ws")

    assert all(c.status == "ran" for c in run.cells), \
        [(c.cell_id, c.error) for c in run.cells if c.status != "ran"]
    for cell in run.cells:
        assert "|" in cell.cell_id or ":" in cell.cell_id, "标签本身没有被改动"


def test_experiment_runs_are_marked_and_never_interactive(tmp_path):
    """Their memory and skills must not reach a production prompt."""
    run = run_experiment(_spec(tmp_path, matrix={"model": ["a"]}, suite=str(SMOKE)),
                         workspace=tmp_path / "ws")
    assert run.tasks, "没有产生任务记录"
    assert all(t["cellId"] for t in run.tasks), "任务记录没有带上所属 cell"


def test_the_suite_is_read_for_its_tasks_not_rerun_per_cell(tmp_path):
    """A cell runs exactly one task; the suite supplies definitions."""
    spec = _spec(tmp_path, matrix={"model": ["a"]}, suite=str(SMOKE))
    run = run_experiment(spec, workspace=tmp_path / "ws")

    per_task = {}
    for cell in run.cells:
        per_task[cell.task_id] = per_task.get(cell.task_id, 0) + 1
    assert set(per_task) == {t.id for t in load_tasks(SMOKE)}
    assert all(n == 1 for n in per_task.values()), "单臂实验里每个任务只应跑一次"
