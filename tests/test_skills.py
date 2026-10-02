"""Skills: procedures distilled from runs that got past a failure.

The distinction the tests defend: a skill is not memory. Memory records what
happened; a skill is a *procedure* keyed by the failure class it addresses, with
the run that proved it and a version — because procedural knowledge in free text
has no version, no evaluation and no way to be superseded.

And the content is the harness's own record, not a model's prose. That is what
lets it be written without a human in the loop and still count as evidence.
"""
from __future__ import annotations

import asyncio

from wfos.skills import class_of, record_from_run, recoveries, render, skills_for_run, stuck_on

FEATURE = "新增一个用户模块，在 app.py 中追加 feature_user() 函数"


def _recovered_run(app, *, failure_class="build_error", state="build_test"):
    """A completed run that hit `failure_class` at `state` and later got past it."""
    h, repo = app["harness"], app["repo"]
    run_id = h.create_run(FEATURE, kind="feature")["id"]
    repo.add_step(run_id, state, "verifier", {"build": {"ok": False}},
                  status="failed", failure_class=failure_class, error="构建坏了")
    repo.add_step(run_id, state, "verifier",
                  {"build": {"ok": True}, "tests": {"failed": 0}, "verdict": "pass"})
    repo.update_run(run_id, status="completed", state="completed")
    return repo.get_run(run_id), repo


# ------------------------------------------------------------------ what counts
def test_a_recovered_failure_is_a_procedure(app):
    run, repo = _recovered_run(app)
    assert recoveries(repo, run) == ["build_error"]


def test_a_failure_that_was_never_recovered_is_not_one(app):
    """The point is a procedure, not a post-mortem."""
    h, repo = app["harness"], app["repo"]
    run_id = h.create_run(FEATURE, kind="feature")["id"]
    repo.add_step(run_id, "build_test", "verifier", {}, status="failed",
                  failure_class="build_error")
    repo.update_run(run_id, status="failed", state="build_test")
    assert recoveries(repo, repo.get_run(run_id)) == []


def test_a_run_that_never_failed_teaches_nothing(app):
    """The common case, and it must stay free."""
    h, repo = app["harness"], app["repo"]
    run_id = h.create_run(FEATURE, kind="feature")["id"]
    repo.add_step(run_id, "req_capture", "investigator", {})
    repo.update_run(run_id, status="completed", state="completed")
    assert recoveries(repo, repo.get_run(run_id)) == []
    assert record_from_run(repo, repo.get_run(run_id)) == []


def test_an_unfinished_run_teaches_nothing_even_if_it_recovered(app):
    """Only a run that reached `completed` proves the procedure worked."""
    run, repo = _recovered_run(app)
    repo.update_run(run["id"], status="failed")
    assert recoveries(repo, repo.get_run(run["id"])) == []


# ------------------------------------------------------------------ the record
def test_the_recorded_skill_carries_its_evidence_and_a_version(app):
    run, repo = _recovered_run(app)
    recorded = record_from_run(repo, run)

    assert len(recorded) == 1
    skill = recorded[0]
    assert skill["trigger"] == "build_error"
    assert skill["evidence_run"] == run["id"]      # a run, not a claim
    assert skill["version"] == 1
    assert skill["superseded"] == 0
    # The procedure is the harness's own observations.
    assert "build_error" in skill["procedure"]
    assert run["id"] in skill["procedure"]
    assert "build.ok=True" in skill["procedure"]


def test_a_newer_procedure_supersedes_the_older_one(app):
    """Two live procedures for one class would have to be ranked by something, and
    the newest evidence is the only ranking that means anything."""
    first, repo = _recovered_run(app)
    record_from_run(repo, first)
    second, _ = _recovered_run(app)
    record_from_run(repo, second)

    live = repo.skills_for(["build_error"])
    assert len(live) == 1 and live[0]["version"] == 2
    assert live[0]["evidence_run"] == second["id"]
    # And the older one is kept, marked — the history of what we believed.
    history = repo.list_skills(live_only=False)
    assert len(history) == 2
    assert {h["version"]: h["superseded"] for h in history} == {1: 1, 2: 0}


def test_the_harness_records_skills_when_a_run_ends(app):
    """End to end through `advance`, not through a direct call."""
    h, repo = app["harness"], app["repo"]
    run_id = h.create_run(FEATURE, kind="feature")["id"]
    repo.add_step(run_id, "build_test", "verifier", {}, status="failed",
                  failure_class="build_error")
    repo.add_step(run_id, "build_test", "verifier", {"build": {"ok": True}})
    repo.update_run(run_id, status="completed", state="completed")

    h._record_skills(run_id)

    assert repo.latest_skill("build_error") is not None


# ---------------------------------------------------------------- retrieval
def test_stuck_on_reads_the_retry_counters(app):
    """Which classes are *live* for this run — a class that was hit and fully
    recovered from is no longer one of them."""
    assert stuck_on({"payload": {"failure_attempts": {"build_test:build_error": 2,
                                                      "implement:no_change": 0}}}) \
        == ["build_error"]
    assert class_of("build_test:build_error") == "build_error"
    assert class_of("bare") == "bare"


def test_a_run_stuck_on_a_class_gets_the_procedure(app):
    run, repo = _recovered_run(app)
    record_from_run(repo, run)

    stuck = {"id": "other", "payload": {"failure_attempts": {"build_test:build_error": 1}}}
    got = skills_for_run(repo, stuck)
    assert [s["trigger"] for s in got] == ["build_error"]

    # And a run stuck on something else gets nothing rather than everything.
    elsewhere = {"id": "other", "payload": {"failure_attempts": {"implement:no_change": 1}}}
    assert skills_for_run(repo, elsewhere) == []


def test_the_skill_reaches_the_prompt_naming_its_provenance(app):
    run, repo = _recovered_run(app)
    record_from_run(repo, run)

    h = app["harness"]
    repo.update_run(run["id"], status="running",
                    payload={"failure_attempts": {"build_test:build_error": 1}})
    ctx = h._build_ctx(repo.get_run(run["id"]))
    prompt = h.agents["verifier"].build_prompt(ctx)

    assert "已知修复手法" in prompt
    assert "build_error" in prompt
    assert run["id"] in prompt            # a reader can check where it came from
    assert "非模型结论" in prompt


def test_render_names_the_trigger_version_and_evidence(app):
    run, repo = _recovered_run(app)
    record_from_run(repo, run)
    text = render(repo.skills_for(["build_error"]))

    assert "build_error" in text and "版本 1" in text and run["id"] in text


def test_skills_do_not_appear_when_the_run_is_not_stuck(app):
    run, repo = _recovered_run(app)
    record_from_run(repo, run)

    h = app["harness"]
    repo.update_run(run["id"], status="running")
    ctx = h._build_ctx(repo.get_run(run["id"]))
    assert "已知修复手法" not in h.agents["verifier"].build_prompt(ctx)


# ------------------------------------------------------------------ the CLI
def test_the_cli_lists_skills(app):
    from wfos.cli import cmd_skills

    run, repo = _recovered_run(app)
    record_from_run(repo, run)
    assert cmd_skills(app["harness"], None) == 0


def test_a_full_mock_flow_records_no_skills(app):
    """A clean run has no failure to learn from — the feature must cost nothing
    when nothing went wrong."""
    h, repo = app["harness"], app["repo"]
    run_id = h.create_run(FEATURE, kind="feature")["id"]
    asyncio.run(h.advance(run_id))

    # `status=None, origin=None` on purpose: the default now means "production",
    # and a run that recorded a *candidate* would satisfy that default while
    # having recorded something. The claim here is "nothing at all".
    assert repo.list_skills(status=None, origin=None) == []


# ------------------------------------------------------------- production tier
def test_list_skills_returns_production_and_not_the_other_tiers(app):
    """§: the default answer to "what skills does this harness have".

    A candidate row and a benchmark-origin row are both *in this table* and
    neither is production. Before this, the default was "everything not
    superseded", which answered a different question — and the wrong answer to
    it is what a benchmark arm would have loaded.
    """
    from wfos.models import ORIGIN_BENCHMARK, ORIGIN_INTERACTIVE, ORIGIN_TASK
    repo = app["repo"]

    production = repo.add_skill("test_failure", "生产规程", "人工写下的",
                                "run-prod", origin=ORIGIN_INTERACTIVE)
    from_task = repo.add_skill("test_failure", "任务推导的规程", "任务里学到的",
                               "run-task", origin=ORIGIN_TASK)
    from_bench = repo.add_skill("build_error", "基准推导的规程", "基准里学到的",
                                "run-bench", origin=ORIGIN_BENCHMARK)

    listed = repo.list_skills()

    assert [s["id"] for s in listed] == [production["id"]]
    assert listed[0]["status"] == "live" and listed[0]["origin"] == "interactive"
    assert {s["id"] for s in listed}.isdisjoint({from_task["id"], from_bench["id"]})
    assert from_task["status"] == "candidate", "任务来源的规程本来就该是候选"

    # 过滤是一个**默认值**，不是删除：要全量的人说一声就能拿到
    everything = repo.list_skills(status=None, origin=None)
    assert {s["id"] for s in everything} == {
        production["id"], from_task["id"], from_bench["id"]}


def test_a_candidate_in_the_run_s_own_store_is_never_reported_as_in_play(app, tmp_path):
    """The leak this closes, at the point it would have happened.

    `TaskRecord.skills` is the arm's own view of what it loaded; `derived` is
    what the store contains. A task or benchmark run's learned procedure lands in
    its store as a `candidate` (see `skills.record_from_run`), so the two views
    must answer differently — otherwise an isolated run reports a procedure as
    production that no promotion ever approved.
    """
    from wfos import bench
    from wfos.eval import evaluate
    from wfos.models import ORIGIN_TASK
    from wfos.runner import RunEnvironment, TaskRunner
    from wfos.tasks import load_tasks

    task = load_tasks("benchmark/smoke")[0]
    env = RunEnvironment.create(tmp_path / "ws", name="cand-check")
    runner = TaskRunner(env)
    outcome = runner.run(task)
    # Exactly what a finishing task run writes for a class it recovered from.
    runner.repo.add_skill("test_failure", "候选规程", "在隔离环境里学到的",
                          outcome.run_id, origin=ORIGIN_TASK)

    record = bench._record(runner, task, outcome,
                           evaluate(runner.repo, outcome.run_id, task))

    assert "test_failure" not in record.skills, "候选被当成正在生效的规程了"
    derived = {d["trigger"]: d for d in record.derived}
    assert derived["test_failure"]["status"] == "candidate", (
        "候选必须仍然出现在 derived 里 —— 否则候选推导就没有输入了")
