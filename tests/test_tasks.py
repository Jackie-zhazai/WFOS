"""The task layer: declared, versioned, and refused when it says nothing.

A task is what a run gets compared *against* — a benchmark baseline, an
experiment arm, an RSI candidate's proving ground. So the two properties that
matter are that it is fully declared (nothing about it is implied by a sentence)
and that it is frozen (a result recorded against revision 1 stays readable after
revision 2 lands).

The first test is the load-bearing one: the case set already in the repo parses
with no migration. If that ever stops being true, this module has become a second
format rather than a view over the first.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from wfos.baseline import DEFAULT_CASES, load_cases
from wfos.tasks import DEFAULT_TASKS, TaskSpec, load_tasks


def _case(**over) -> dict:
    base = {"id": "t1", "request": "新增一个模块", "kind": "feature",
            "expect": {"reachedTerminal": True}, "stepBudget": 8}
    base.update(over)
    return base


# --------------------------------------------------- the existing set parses
def test_the_case_set_already_in_the_repo_parses_unchanged():
    """No migration, no second format — the whole point of the design."""
    specs = load_tasks(DEFAULT_CASES)
    cases = load_cases(DEFAULT_CASES)["cases"]

    assert len(specs) == len(cases)
    assert {s.id for s in specs} == {c["id"] for c in cases}
    for spec in specs:
        assert spec.workflow in ("feature", "bugfix")
        assert spec.input
        assert spec.expected
        assert spec.step_budget > 0


def test_the_default_path_is_the_whole_suite(tmp_path):
    """One directory, one meaning of "all the tasks" — shared with the
    reliability baseline, so the two cannot drift into two different sets."""
    assert Path(DEFAULT_TASKS).name == "benchmark"
    assert Path(DEFAULT_TASKS).is_dir()
    assert load_tasks()[0].id == load_tasks(DEFAULT_TASKS)[0].id


# ---------------------------------------------------------------- the fields
def test_version_defaults_to_one_and_must_be_positive():
    assert TaskSpec.from_case(_case()).version == 1
    assert TaskSpec.from_case(_case(version=3)).version == 3
    for bad in (0, -1, "2", True):
        with pytest.raises(ValueError):
            TaskSpec.from_case(_case(version=bad))


def test_the_specs_name_and_the_files_name_are_both_accepted():
    """The spec says `workflow` / `input` / `expected`; the files say
    `kind` / `request` / `expect`. Either spelling builds the same task."""
    from_alias = TaskSpec.from_case(_case())
    from_canonical = TaskSpec.from_case(
        {"id": "t1", "workflow": "feature", "input": "新增一个模块",
         "expected": {"reachedTerminal": True}, "stepBudget": 8})

    assert from_alias == from_canonical


def test_two_spellings_that_disagree_are_refused_not_resolved():
    """Picking one silently makes the losing name a lie, and which one a reader
    believed would depend on where they happened to look."""
    with pytest.raises(ValueError, match="不一致"):
        TaskSpec.from_case(_case(workflow="bugfix"))


def test_an_unknown_workflow_is_refused_and_the_error_names_the_real_ones():
    with pytest.raises(ValueError) as excinfo:
        TaskSpec.from_case(_case(kind="skill"))

    assert "skill" in str(excinfo.value)
    assert "feature" in str(excinfo.value)


def test_a_task_that_asserts_nothing_is_refused():
    """It would pass forever, which is worse than having no task at all."""
    for empty in ({}, None, []):
        with pytest.raises(ValueError, match="expected"):
            TaskSpec.from_case(_case(expect=empty))


def test_a_task_without_an_input_or_an_id_is_refused():
    with pytest.raises(ValueError, match="input"):
        TaskSpec.from_case(_case(request=""), )
    with pytest.raises(ValueError, match="id"):
        TaskSpec.from_case(_case(id=""))


def test_the_step_budget_is_normalised_into_constraints():
    """One place it is read and validated, whichever spelling the file used."""
    # Built without the helper's own `stepBudget`, or carrying both would be the
    # conflict the next test is about.
    inside = {"id": "t1", "request": "x", "kind": "feature",
              "expect": {"a": True}, "constraints": {"stepBudget": 5}}
    both_forms = _case(constraints={"notes": "x"})

    assert TaskSpec.from_case(_case()).step_budget == 8
    assert TaskSpec.from_case(inside).step_budget == 5
    assert TaskSpec.from_case(both_forms).constraints["notes"] == "x"
    assert TaskSpec.from_case(both_forms).step_budget == 8


def test_a_missing_or_conflicting_step_budget_is_refused():
    with pytest.raises(ValueError, match="stepBudget"):
        TaskSpec.from_case({"id": "t1", "request": "x", "kind": "feature",
                            "expect": {"a": True}})
    with pytest.raises(ValueError, match="不一致"):
        TaskSpec.from_case(_case(stepBudget=8, constraints={"stepBudget": 9}))


def test_the_brain_must_be_one_of_the_two_the_harness_can_run():
    with pytest.raises(ValueError, match="brain"):
        TaskSpec.from_case(_case(brain="real"))


def test_the_evaluator_block_must_be_an_object():
    """Declared now, filled by P2 — but declared, so a task cannot be written
    without a place to say what would prove it."""
    assert TaskSpec.from_case(_case()).evaluator == {}
    # The gate names an invariant the task *does* declare: one it does not is
    # refused, because there would be no expected value to compare against.
    gate = {"gate": ["reachedTerminal"]}
    assert TaskSpec.from_case(_case(evaluator=gate)).evaluator == gate
    with pytest.raises(ValueError, match="evaluator"):
        TaskSpec.from_case(_case(evaluator=["x"]))


def test_a_task_is_frozen():
    """A task that could be edited in place would make every recorded result
    unanchorable — the run says which revision produced it, and that has to be a
    fact about a value that did not move."""
    import dataclasses

    spec = TaskSpec.from_case(_case())

    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.version = 2


# ------------------------------------------------------------- loading a set
def _write(path: Path, cases: list[dict]) -> Path:
    path.write_text(json.dumps({"schemaVersion": 1, "cases": cases}, ensure_ascii=False),
                    encoding="utf-8")
    return path


def test_a_directory_is_read_as_one_task_set(tmp_path):
    """The shape the three tiers take, so they do not need a loader of their own."""
    (tmp_path / "smoke").mkdir()
    (tmp_path / "regression").mkdir()
    _write(tmp_path / "smoke" / "a.json", [_case(id="s1")])
    _write(tmp_path / "regression" / "b.json", [_case(id="r1")])

    ids = {s.id for s in load_tasks(tmp_path)}

    assert ids == {"s1", "r1"}


def test_a_duplicate_id_across_files_is_refused(tmp_path):
    """Two answers to "what does this task require", and the last one would win."""
    _write(tmp_path / "a.json", [_case(id="same")])
    _write(tmp_path / "b.json", [_case(id="same")])

    with pytest.raises(ValueError, match="重复"):
        load_tasks(tmp_path)


def test_an_unsupported_schema_version_is_refused(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"schemaVersion": 99, "cases": [_case()]}), encoding="utf-8")

    with pytest.raises(ValueError, match="schemaVersion"):
        load_tasks(path)


def test_an_empty_case_list_is_refused(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"schemaVersion": 1, "cases": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="非空"):
        load_tasks(path)
