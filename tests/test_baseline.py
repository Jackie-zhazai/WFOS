"""The frozen whole-flow baseline: its case set, its diff semantics, its artifact.

Three things are pinned here:

  * the case set is well-formed and **every** case asserts something — a case with
    an empty `expect` would report as passing forever, which is worse than no case
    at all;
  * `compare` fails the check only in the direction that means "this got worse",
    so a baseline cannot be made green by deleting or weakening cases;
  * the committed `artifacts/harness-baseline.json` still matches a fresh run.

The last one is the expensive test in this file and deliberately so: it is the
only place a change to the workflow's whole-flow behaviour shows up.
"""
from __future__ import annotations

import json

import pytest

from wfos.baseline import (
    DEFAULT_ARTIFACT,
    DEFAULT_CASES,
    _ratio,
    compare,
    load_artifact,
    load_cases,
    render_diff,
    run_suite,
    work_units,
)


# ------------------------------------------------------------------ case set
def test_the_committed_case_set_is_valid_and_asserts_something():
    data = load_cases(DEFAULT_CASES)
    assert data["cases"]
    ids = [c["id"] for c in data["cases"]]
    assert len(ids) == len(set(ids))
    for case in data["cases"]:
        assert case["expect"], f"{case['id']} 没有断言任何东西"
        assert case["expect"].keys() <= set(_invariant_names()), (
            f"{case['id']} 断言的 {sorted(set(case['expect']) - set(_invariant_names()))} "
            "不是 runner 会产生的不变式 —— 那会永远判为失败")


def _invariant_names() -> list[str]:
    """Run nothing; read the vocabulary off a row of the committed artifact."""
    rows = load_artifact(DEFAULT_ARTIFACT)["rows"]
    return list(rows[0]["actual"])


def _write_cases(tmp_path, **overrides):
    """A minimal valid case, with fields removed or replaced.

    Built inline rather than by editing the suite on disk: `DEFAULT_CASES` is the
    three-tier directory now, and a test that read a real task to mutate it would
    fail for a reason that has nothing to do with what it is testing.
    """
    case = {"id": "t1", "request": "新增一个模块", "kind": "feature",
            "expect": {"reachedTerminal": True}, "stepBudget": 8}
    for key, value in overrides.items():
        if value is None:
            case.pop(key, None)
        else:
            case[key] = value
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"schemaVersion": 1, "cases": [case]},
                               ensure_ascii=False), encoding="utf-8")
    return path


def test_a_case_with_no_assertions_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="expect"):
        load_cases(_write_cases(tmp_path, expect={}))


def test_a_case_without_a_step_budget_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="stepBudget"):
        load_cases(_write_cases(tmp_path, stepBudget=None))


def test_a_case_without_a_request_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="request"):
        load_cases(_write_cases(tmp_path, request="  "))


def test_an_unsupported_schema_version_is_rejected(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"schemaVersion": 99, "cases": [
        {"id": "t1", "request": "x", "kind": "feature",
         "expect": {"reachedTerminal": True}, "stepBudget": 8}]},
        ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="schemaVersion"):
        load_cases(path)


# ---------------------------------------------------------------- diff rules
def _artifact(rows, pass_rate=1.0):
    return {"summary": {"passRate": pass_rate}, "rows": rows}


def _row(cid, passed=True, actual=None):
    return {"id": cid, "passed": passed, "actual": actual or {"terminalStatus": "completed"},
            "mismatches": [] if passed else [{"invariant": "terminalStatus",
                                              "expected": "completed", "actual": "failed"}]}


def test_a_case_that_passed_and_now_fails_is_a_regression():
    diff = compare(_artifact([_row("a", passed=False)]), _artifact([_row("a", passed=True)]))
    assert diff["ok"] is False
    assert [r["id"] for r in diff["regressions"]] == ["a"]


def test_a_case_that_failed_and_now_passes_is_fixed_not_a_regression():
    """Improvements must not fail the check, or the baseline can never be improved."""
    diff = compare(_artifact([_row("a", passed=True)]),
                   _artifact([_row("a", passed=False)], pass_rate=0.0))
    assert diff["ok"] is True
    assert diff["fixed"] == ["a"]


def test_deleting_a_case_from_the_run_fails_the_check():
    """Otherwise a red baseline could be made green by dropping the failing case."""
    diff = compare(_artifact([_row("a")]), _artifact([_row("a"), _row("b")]))
    assert diff["ok"] is False
    assert diff["removed"] == ["b"]


def test_a_case_the_baseline_does_not_cover_fails_the_check():
    """A baseline that quietly stops covering the cases that run is not a baseline."""
    diff = compare(_artifact([_row("a"), _row("b")]), _artifact([_row("a")]))
    assert diff["ok"] is False
    assert diff["added"] == ["b"]
    assert not diff["regressions"]      # it is stale, not regressed


def test_an_invariant_the_baseline_never_recorded_fails_the_check():
    """The same hole as an uncovered case, one level down.

    `compare` walks the keys both sides share, so an invariant the runner grew
    since the baseline was written would be skipped entirely — the baseline would
    quietly describe less than what runs, and still report itself consistent.
    """
    before = _artifact([_row("a", actual={"stepsExecuted": 8})])
    after = _artifact([_row("a", actual={"stepsExecuted": 8, "evidenceDropped": 0})])
    diff = compare(after, before)
    assert diff["ok"] is False
    assert diff["unrecorded"] == ["evidenceDropped"]
    assert not diff["regressions"]


def test_an_unasserted_invariant_moving_makes_the_baseline_stale():
    """Recording every invariant is what makes this catchable at all.

    Only asserted invariants are contracts, so movement outside them is not a
    regression — but it does mean the stored baseline no longer describes what
    runs, and a reader should hear about it rather than find out later.
    """
    before = _artifact([_row("a", actual={"stepsExecuted": 8, "childRuns": 0})])
    after = _artifact([_row("a", actual={"stepsExecuted": 99, "childRuns": 0})])
    diff = compare(after, before)
    assert diff["ok"] is False
    assert diff["changes"] == [{"id": "a", "invariant": "stepsExecuted",
                                "before": 8, "after": 99}]
    assert not diff["regressions"]


# ------------------------------------------------------- the real thing runs
def test_the_committed_baseline_still_matches_a_fresh_run(tmp_path):
    """The whole case set, re-run and diffed against the artifact in the repo.

    This is the only test that would notice a change to the workflow's
    end-to-end behaviour — its cost is the price of that coverage.
    """
    stored = load_artifact(DEFAULT_ARTIFACT)
    diff = compare(run_suite(DEFAULT_CASES, workspace=tmp_path), stored)
    assert diff["ok"], (
        "基线不匹配（回归、基线未覆盖的用例，或未被断言的不变式发生了移动——"
        "最后一种需要显式重刷基线，而不是让它静默通过）：\n"
        + json.dumps({k: diff[k] for k in
                      ("regressions", "removed", "added", "unrecorded", "changes")},
                     ensure_ascii=False, indent=2))


# ------------------------------------------------------- measured figures (趋势)
def _m(cid, **figures):
    """A row carrying measured figures, defaulting to what a mock run produces."""
    base = {"inputTokens": None, "outputTokens": None, "cachedTokens": None,
            "reasoningTokens": None, "modelCalls": 8, "latencyMs": 100,
            "costUsd": None, "models": "mock", "steps": 8, "stepsWithUsage": 0,
            "workUnit": "model_calls", "workUnits": 8}
    base.update(figures)
    unit, amount = work_units(base)
    base["workUnit"], base["workUnits"] = unit, amount
    return _row(cid) | {"measured": base}


def test_a_case_records_measured_figures_beside_its_invariants():
    """Beside, never among: `actual` is asserted, `measured` is only reported."""
    row = run_suite(DEFAULT_CASES, only=["feature-happy-path"])["rows"][0]

    assert "measured" in row and "measured" not in row["actual"]
    figures = row["measured"]
    # The mock brain has no provider, so tokens are unknown — not zero.
    assert figures["inputTokens"] is None and figures["outputTokens"] is None
    assert figures["stepsWithUsage"] == 0
    assert figures["costUsd"] is None
    assert figures["models"] == "mock"
    # Model calls are counted by the harness, so a mock case still has these.
    assert figures["modelCalls"] > 0 and figures["steps"] > 0


def test_the_work_unit_follows_what_could_actually_be_measured():
    """Tokens when a provider reported them, calls when it did not, "" for neither."""
    assert work_units({"inputTokens": 100, "outputTokens": 20,
                       "modelCalls": 3}) == ("tokens", 120)
    assert work_units({"inputTokens": None, "outputTokens": None,
                       "modelCalls": 3}) == ("model_calls", 3)
    assert work_units({}) == ("", None)
    # Cached and reasoning are breakdowns of input and output, so they are not
    # added again — the same rule the run budget uses.
    assert work_units({"inputTokens": 100, "outputTokens": 20,
                       "cachedTokens": 90, "reasoningTokens": 15})[1] == 120


def test_a_measured_figure_moving_does_not_fail_the_check():
    """The whole reason measured figures live outside `actual`.

    `compare` fails on any movement in an invariant nobody asserted, and that is
    right for behaviour. A token count moves every run by nature; if it lived in
    `actual`, every check after every run would be red and the baseline useless.
    """
    before = _artifact([_m("c1", inputTokens=1000, outputTokens=200, workUnits=1200,
                           workUnit="tokens")])
    after = _artifact([_m("c1", inputTokens=1300, outputTokens=260, workUnits=1560,
                          workUnit="tokens")])

    diff = compare(after, before)

    assert diff["ok"] is True
    changed = {c["figure"] for c in diff["measuredChanges"]}
    assert {"inputTokens", "outputTokens", "workUnits"} <= changed
    ratio = next(c["ratio"] for c in diff["measuredChanges"] if c["figure"] == "inputTokens")
    assert ratio == pytest.approx(1.3)


def test_a_baseline_without_measured_figures_is_behind_not_clean():
    """A baseline that cannot see the section must not report "consistent".

    Same rule as an invariant it has no column for: comparing nothing is not the
    same as comparing and finding no change.
    """
    stored = {"summary": {"passRate": 1.0}, "rows": [_row("c1")]}   # no `measured`
    diff = compare(_artifact([_m("c1")]), stored)

    assert diff["ok"] is False
    assert "measured" in diff["unrecorded"]


def test_latency_is_recorded_but_not_diffed_per_case():
    """It measures the machine. Eleven noise lines would drown the real figures."""
    before = _artifact([_m("c1", latencyMs=100), _m("c2", latencyMs=200)])
    after = _artifact([_m("c1", latencyMs=150), _m("c2", latencyMs=90)])

    diff = compare(after, before)

    assert not any(c["figure"] == "latencyMs" for c in diff["measuredChanges"])
    assert diff["noisy"] == [{"figure": "latencyMs", "before": 300, "after": 240}]
    assert diff["ok"] is True


# ------------------------------------------------------------------ work growth
def test_work_growth_past_the_limit_is_the_only_way_a_number_fails():
    before = _artifact([_m("c1", modelCalls=8, workUnits=8)])
    after = _artifact([_m("c1", modelCalls=12, workUnits=12)])

    assert compare(after, before, work_growth_limit=0.25)["ok"] is False
    assert compare(after, before, work_growth_limit=0.25)["work"]["exceeded"] is True
    # Within the limit, and unset, both pass — the figure is still reported.
    assert compare(after, before, work_growth_limit=1.0)["ok"] is True
    assert compare(after, before)["ok"] is True


def test_work_growth_is_measured_across_the_suite_not_per_case():
    """One case getting cheaper while another gets dearer is a wash."""
    # `_m` derives workUnits from the underlying figure, so these are set at the
    # source (model calls) rather than passed in — a helper that accepted both
    # could be handed a pair that disagrees with itself.
    before = _artifact([_m("c1", modelCalls=10), _m("c2", modelCalls=10)])
    after = _artifact([_m("c1", modelCalls=18), _m("c2", modelCalls=2)])

    assert before["rows"][0]["measured"]["workUnits"] == 10, "helper 没按预期推导"

    work = compare(after, before, work_growth_limit=0.25)["work"]

    assert work["before"] == 20 and work["after"] == 20
    assert work["exceeded"] is False


def test_work_is_not_compared_across_two_different_units():
    """Tokens over model calls is not a ratio, it is a category error."""
    mock = _artifact([_m("c1", workUnit="model_calls", workUnits=8)])
    live = _artifact([_m("c1", inputTokens=100000, outputTokens=5000,
                         workUnit="tokens", workUnits=105000)])

    diff = compare(live, mock, work_growth_limit=0.25)
    work = diff["work"]

    assert work["ratio"] is None and work["exceeded"] is False
    assert diff["ok"] is True
    assert "单位不同" in work["why"]


def test_a_mixed_unit_suite_has_no_total_at_all():
    """A frozen-mock case plus a live case totals to no unit whatsoever."""
    rows = [_m("c1", workUnit="model_calls", workUnits=8),
            _m("c2", inputTokens=50, workUnit="tokens", workUnits=50)]
    mixed = _artifact(rows)

    work = compare(mixed, mixed, work_growth_limit=0.25)["work"]

    assert work["after"] is None
    assert work["exceeded"] is False


def test_a_growth_limit_against_an_unmeasured_suite_does_nothing():
    """No measurement, no gate — it must not fire on an invented zero."""
    empty = _artifact([_row("c1") | {"measured": {}}])

    work = compare(empty, empty, work_growth_limit=0.0)["work"]

    assert work["exceeded"] is False and work["ratio"] is None
    assert work["why"]


# ------------------------------------------------------------------------ ratio
def test_a_ratio_is_none_wherever_the_division_would_mean_nothing():
    assert _ratio(100, 130) == pytest.approx(1.3)
    assert _ratio(0, 5) is None            # "grew from zero" has no ratio
    assert _ratio(None, 5) is None         # nothing was measured
    assert _ratio(5, None) is None
    assert _ratio(True, 2) is None         # bools are ints, but not quantities
    assert _ratio("mock", "mock") is None  # the model list is not a number


def test_the_work_line_is_rendered_even_with_no_limit():
    before = _artifact([_m("c1", modelCalls=8, workUnits=8)])
    after = _artifact([_m("c1", modelCalls=16, workUnits=16)])

    text = render_diff(compare(after, before))

    assert "工作量（model_calls）8 → 16" in text
    assert "×2.00" in text
    assert "未设上限" in text
