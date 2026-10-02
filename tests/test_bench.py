"""Benchmark, baseline and compare.

The three properties under test are all about what these things may **not** do.
A baseline must not be writable by anything a candidate can run; `unknown` must
not become `pass` and must not become `regression`; and a benchmark must not be
able to touch the tasks that judge it. Every one of those is a way to make a
result look better without improving anything, and each is checked by giving the
code the opportunity and asserting it declines.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from wfos.bench import (
    BENCHMARK_VERSION,
    TIERS,
    load_record,
    run_benchmark,
    suite_paths,
    write_baseline,
)
from wfos.bench_compare import (
    IMPROVEMENT,
    NOISE,
    REGRESSION,
    UNCHANGED,
    UNKNOWN_VERDICT,
    compare,
    render,
)
from wfos.tasks import load_tasks

SUITE = Path("benchmark")
SMOKE = SUITE / "smoke"


def _run(tmp_path, suite=SMOKE, name="ws", **kw):
    return run_benchmark(suite, workspace=tmp_path / name, tier=suite.name, **kw)


# ------------------------------------------------------------------ the suites
def test_the_three_tiers_exist_and_are_the_three_named_ones():
    assert set(suite_paths(SUITE)) == set(TIERS) == {"smoke", "regression", "challenge"}


def test_every_task_belongs_to_exactly_one_tier():
    """Two definitions of one task is two answers to "what does this require"."""
    seen: dict[str, str] = {}
    for tier, path in suite_paths(SUITE).items():
        for task in load_tasks(path):
            assert task.id not in seen, f"{task.id} 同时出现在 {seen[task.id]} 与 {tier}"
            seen[task.id] = tier
    # The count is derived, not frozen: a tier gaining a task is a normal change,
    # and hard-coding it here would make every such addition look like a failure of
    # this test rather than of the thing it is about.
    assert len(seen) == sum(len(load_tasks(p)) for p in suite_paths(SUITE).values())


def test_the_task_layer_is_the_only_source_of_expectations():
    """`TaskSpec` reads the same schema the suites use, so an evaluator never has
    to guess what a task wanted."""
    for task in load_tasks(SMOKE):
        assert task.expected, "任务没有声明期望"
        assert task.step_budget > 0


# ------------------------------------------------------------------ benchmark
def test_a_smoke_benchmark_runs_and_judges_every_task(tmp_path):
    run = _run(tmp_path)

    assert run.suite == "smoke"
    assert len(run.tasks) == 2
    assert run.session_id, "整个套件应当共用一个 session"
    for task in run.tasks:
        assert task.run_id and task.evaluation.run_id == task.run_id
        assert task.workflow_path, "没有记录 workflow 路径"
        assert task.task_version >= 1


def test_every_task_in_a_suite_shares_one_session(tmp_path):
    """One invocation, one answer to "what did that run do"."""
    run = _run(tmp_path)

    assert len({t.run_id for t in run.tasks}) == len(run.tasks), "运行 id 撞了"
    assert run.session_id


def test_two_tasks_in_one_suite_do_not_see_each_other(tmp_path):
    """Each task gets its own database. Two sharing one would share memory and
    audit rows, and a suite where B can read A's post-mortem is not measuring the
    tasks."""
    run = _run(tmp_path)
    dbs = sorted((tmp_path / "ws").rglob("wfos.db"))
    assert len(dbs) == len(run.tasks), f"数据库数 {len(dbs)} ≠ 任务数 {len(run.tasks)}"


def test_a_benchmark_never_writes_to_the_suite_it_runs(tmp_path):
    """A benchmark that could edit its own tasks could make itself pass."""
    before = {p: p.read_bytes() for p in SUITE.rglob("*.json")}

    _run(tmp_path)

    after = {p: p.read_bytes() for p in SUITE.rglob("*.json")}
    assert after == before, "任务定义被运行改动了"


def test_the_record_carries_the_versions_a_comparison_needs(tmp_path):
    run = _run(tmp_path)
    payload = run.as_json()

    assert payload["schemaVersion"] == BENCHMARK_VERSION
    assert payload["environment"]["digest"], "没有记录环境摘要"
    for task in payload["tasks"]:
        assert task["taskVersion"] >= 1
        assert task["workflow"]
        assert "skills" in task and "metrics" in task and "axes" in task


def test_the_comparability_digest_ignores_the_throwaway_workspace(tmp_path):
    """Every run gets a fresh tree by design, so the tree cannot be a condition.

    Measured, not assumed: the first end-to-end comparison reported "环境不同"
    for two runs of the same mock suite that differed only in where the
    throwaway tree lived, and quietly downgraded every measurement to unknown.
    """
    first = _run(tmp_path, name="ws-a")
    second = _run(tmp_path, name="ws-b")

    assert first.environment["digest"] == second.environment["digest"]
    assert str(first.environment["digest"]) != ""


def test_the_rollup_counts_and_does_not_score(tmp_path):
    run = _run(tmp_path)
    rollup = run.rollup

    assert rollup["tasks"] == len(run.tasks)
    assert rollup["passed"] + rollup["failed"] == rollup["tasks"]
    assert set(rollup) == {"tasks", "passed", "failed", "hardGateFailures", "unknownAxes"}


# ------------------------------------------------------------------- baseline
def test_a_baseline_is_written_with_its_creation_metadata(tmp_path):
    run = _run(tmp_path)
    path = write_baseline(tmp_path / "b.json", run, source=str(SMOKE))

    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["kind"] == "benchmark-baseline"
    assert stored["createdAt"] and stored["source"] == str(SMOKE)
    assert stored["suite"] == "smoke"


def test_creating_a_baseline_over_an_existing_one_is_refused(tmp_path):
    """This refusal is the whole of "a baseline is immutable" — the one function
    that writes a baseline will not replace one, so re-running cannot become the
    reference."""
    run = _run(tmp_path)
    path = write_baseline(tmp_path / "b.json", run)
    original = path.read_bytes()

    with pytest.raises(FileExistsError, match="不可被覆盖"):
        write_baseline(tmp_path / "b.json", run)

    assert path.read_bytes() == original, "被拒绝的写入仍然改动了基线"


def test_running_a_benchmark_cannot_modify_an_existing_baseline(tmp_path):
    """The property, from the other side: the thing a candidate *can* run must
    leave a baseline alone even when it is sitting right beside it."""
    run = _run(tmp_path)
    baseline = write_baseline(tmp_path / "b.json", run)
    before = baseline.read_bytes()

    _run(tmp_path, name="another-arm")

    assert baseline.read_bytes() == before


def test_comparing_does_not_write_to_either_side(tmp_path):
    baseline = _run(tmp_path, name="a").as_json()
    subject = _run(tmp_path, name="b").as_json()
    snapshot = (json.dumps(baseline, sort_keys=True), json.dumps(subject, sort_keys=True))

    compare(baseline, subject)

    assert (json.dumps(baseline, sort_keys=True), json.dumps(subject, sort_keys=True)) \
        == snapshot


def test_a_record_with_an_unknown_version_is_refused(tmp_path):
    path = tmp_path / "b.json"
    path.write_text(json.dumps({"kind": "benchmark-baseline", "schemaVersion": 99,
                                "tasks": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="版本"):
        load_record(path)


def test_a_file_that_is_not_a_record_is_refused(tmp_path):
    path = tmp_path / "b.json"
    path.write_text(json.dumps({"hello": "world"}), encoding="utf-8")

    with pytest.raises(ValueError, match="benchmark 记录"):
        load_record(path)


# -------------------------------------------------------------------- compare
def _record(tasks: list[dict], *, digest: str = "d1", suite: str = "s") -> dict:
    return {"kind": "benchmark-baseline", "schemaVersion": BENCHMARK_VERSION,
            "suite": suite, "environment": {"digest": digest}, "tasks": tasks}


def _task(task_id: str = "t1", **over) -> dict:
    base = {"taskId": task_id, "taskVersion": 1, "workflow": "feature",
            "runId": "r1", "verdict": "pass",
            "axes": {"functional": "pass", "safety": "pass", "efficiency": "unknown",
                     "quality": "unknown", "operational": "pass"},
            "reasons": [], "failures": [], "hardGate": [],
            "metrics": {"steps": 8, "latencyMs": None, "inputTokens": None},
            "workflowPath": ["a", "b"], "toolSequence": ["x"], "models": ["mock"],
            "skills": {}}
    base.update(over)
    return base


def test_comparing_a_record_with_itself_is_deterministic_and_clean():
    record = _record([_task()])

    first = compare(record, record)
    second = compare(record, record)

    assert first.as_json() == second.as_json()
    assert first.ok is True
    assert first.counts()[REGRESSION] == 0


def test_a_check_that_passed_and_now_fails_is_a_regression():
    before = _record([_task(axes={"functional": "pass", "safety": "pass"})])
    after = _record([_task(axes={"functional": "fail", "safety": "pass"})])

    report = compare(before, after)

    assert report.ok is False
    regression = report.by_verdict(REGRESSION)
    assert any(f.subject == "axis.functional" for f in regression)


def test_a_check_that_failed_and_now_passes_is_an_improvement():
    before = _record([_task(verdict="fail", axes={"functional": "fail"})])
    after = _record([_task(verdict="pass", axes={"functional": "pass"})])

    report = compare(before, after)

    assert report.ok is True, "改善不该判成回归"
    assert any(f.verdict == IMPROVEMENT for f in report.findings)


def test_unknown_is_not_a_pass():
    """The rule the whole axes design exists for."""
    before = _record([_task(axes={"functional": "pass", "quality": "unknown"})])
    after = _record([_task(axes={"functional": "pass", "quality": "unknown"})])

    findings = [f for f in compare(before, after).findings if f.subject == "axis.quality"]

    assert findings and findings[0].verdict == UNKNOWN_VERDICT


def test_unknown_is_not_a_regression():
    """The other half, and the one that is easy to get wrong: two things nobody
    measured are not two things that got worse."""
    before = _record([_task(axes={"efficiency": "unknown"})])
    after = _record([_task(axes={"efficiency": "unknown"})])

    report = compare(before, after)

    assert report.ok is True
    assert all(f.verdict != REGRESSION for f in report.findings)


def test_unknown_becoming_a_real_pass_is_not_a_regression_either():
    before = _record([_task(axes={"efficiency": "unknown"})])
    after = _record([_task(axes={"efficiency": "pass"})])

    findings = [f for f in compare(before, after).findings
                if f.subject == "axis.efficiency"]

    # We learned something; we did not go backwards. It is not an improvement
    # either — nothing was measured before, so there is no before to improve on.
    assert findings and findings[0].verdict == UNKNOWN_VERDICT


def test_a_hard_gate_failure_propagates_to_the_comparison():
    """The verdict must keep its failure semantics all the way through — a run
    that failed a gate cannot read as anything but a regression."""
    before = _record([_task(verdict="pass", hardGate=[])])
    after = _record([_task(verdict="fail", hardGate=["reachedTerminal=False（期望 True）"],
                           axes={"functional": "fail"})])

    report = compare(before, after)

    assert report.ok is False
    assert any(f.subject == "hardGate" and f.verdict == REGRESSION
               for f in report.findings)
    assert any(f.subject == "verdict" and f.verdict == REGRESSION
               for f in report.findings)


def test_a_task_the_baseline_covers_that_did_not_run_is_a_regression():
    """Dropping coverage is how a benchmark is made green."""
    before = _record([_task("a"), _task("b")])
    after = _record([_task("a")])

    report = compare(before, after)

    assert report.ok is False
    assert any(f.task_id == "b" and f.verdict == REGRESSION for f in report.findings)


def test_a_task_the_baseline_never_covered_is_unknown():
    report = compare(_record([_task("a")]), _record([_task("a"), _task("b")]))

    assert report.ok is True
    assert any(f.task_id == "b" and f.verdict == UNKNOWN_VERDICT for f in report.findings)


def test_a_new_failure_class_is_a_regression_and_a_gone_one_is_an_improvement():
    worse = compare(_record([_task(failures=[])]), _record([_task(failures=["test_failure"])]))
    better = compare(_record([_task(failures=["test_failure"])]), _record([_task(failures=[])]))

    assert worse.ok is False and any(f.verdict == REGRESSION for f in worse.findings)
    assert better.ok is True and any(f.verdict == IMPROVEMENT for f in better.findings)


def test_a_different_route_to_the_same_verdict_is_not_an_improvement():
    """A run that started passing by skipping verification has not got better,
    and nothing here can tell that apart from a genuinely better route."""
    before = _record([_task(workflowPath=["req", "implement", "verify", "done"])])
    after = _record([_task(workflowPath=["req", "done"])])

    report = compare(before, after)

    assert report.ok is True
    assert all(f.verdict != IMPROVEMENT for f in report.findings
               if f.subject == "workflowPath")
    assert any(f.subject == "workflowPath" and f.verdict == UNKNOWN_VERDICT
               for f in report.findings)


def test_metric_movement_without_a_threshold_is_unknown_not_regression():
    before = _record([_task(metrics={"steps": 8})])
    after = _record([_task(metrics={"steps": 40})])

    report = compare(before, after)

    assert report.ok is True, "没有阈值就不该判回归"
    assert any(f.subject == "metric.steps" and f.verdict == UNKNOWN_VERDICT
               for f in report.findings)


def test_metric_movement_beyond_the_threshold_is_a_regression():
    before = _record([_task(metrics={"steps": 8})])
    after = _record([_task(metrics={"steps": 40})])

    report = compare(before, after, growth_limit=0.25)

    assert report.ok is False
    assert any(f.subject == "metric.steps" and f.verdict == REGRESSION
               for f in report.findings)


def test_metric_movement_inside_the_threshold_is_noise():
    before = _record([_task(metrics={"steps": 8})])
    after = _record([_task(metrics={"steps": 9})])

    report = compare(before, after, growth_limit=0.25)

    assert report.ok is True
    assert any(f.subject == "metric.steps" and f.verdict == NOISE
               for f in report.findings)


def test_latency_movement_is_always_noise():
    """The evidence: the same deterministic suite measured ×0.52 to ×1.23."""
    before = _record([_task(metrics={"latencyMs": 1000})])
    after = _record([_task(metrics={"latencyMs": 3000})])

    report = compare(before, after, growth_limit=0.25)

    assert report.ok is True
    assert any(f.subject == "metric.latencyMs" and f.verdict == NOISE
               for f in report.findings)


def test_an_unmeasured_metric_is_unknown_not_zero():
    before = _record([_task(metrics={"inputTokens": None})])
    after = _record([_task(metrics={"inputTokens": 100})])

    report = compare(before, after)

    assert any(f.subject == "metric.inputTokens" and f.verdict == UNKNOWN_VERDICT
               for f in report.findings)


def test_two_environments_make_the_measurements_incomparable():
    """Tokens across two models is the same category error as dividing a token
    count by a call count."""
    before = _record([_task(metrics={"steps": 8})], digest="env-a")
    after = _record([_task(metrics={"steps": 40})], digest="env-b")

    report = compare(before, after, growth_limit=0.25)

    assert report.environment, "环境不同没有被指出"
    assert report.ok is True, "环境不同的度量不该判回归"
    assert any(f.subject == "metric.steps" and f.verdict == UNKNOWN_VERDICT
               for f in report.findings)


def test_the_renderer_says_unknown_is_not_a_conclusion(tmp_path):
    before = _record([_task()])
    after = _record([_task()])

    text = render(compare(before, after))

    assert "unknown" in text
    assert "不是结论" in text


def test_compare_is_json_serialisable(tmp_path):
    report = compare(_record([_task()]), _record([_task()]))
    payload = report.as_json()

    assert json.loads(json.dumps(payload, ensure_ascii=False))["kind"] == "comparison"
    assert set(payload["counts"]) == {REGRESSION, IMPROVEMENT, UNCHANGED,
                                      UNKNOWN_VERDICT, NOISE}
