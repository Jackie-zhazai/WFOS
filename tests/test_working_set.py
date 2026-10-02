"""The working-set section: which files are in play, and which attempt this is.

A retry loop only converges if the model can tell it is retrying. Without this
section every attempt reads as a first one, and the retry budget is spent
repeating the same answer.
"""
from __future__ import annotations

import asyncio

from wfos.agents.base import BaseAgent
from wfos.agents.investigator import InvestigatorAgent
from wfos.harness import context
from wfos.llm.scripted import ScriptedAdapter
from wfos.relevance import cap


def _agent(app):
    return InvestigatorAgent(ScriptedAdapter(), app["gateway"], app["sandbox"],
                             prompt_total_budget=context.DEFAULT_TOTAL_BUDGET)


def _ctx(app, **over):
    base = {"run": {"id": "ws-run", "kind": "feature", "title": "t", "description": "任务"},
            "state": "build_test", "plan": {}, "prior": {}, "evidence": cap([], 0)}
    base.update(over)
    return base


# ------------------------------------------------------------------ ordering
def test_working_set_sits_with_the_load_bearing_sections():
    assert "working_set" in context.SECTION_ORDER
    assert context.SECTION_ORDER[2] == "working_set"       # next to run_context
    assert "working_set" not in context.REDUCTION_ORDER    # never clipped


# --------------------------------------------------------------- what it says
def test_working_set_lists_the_runs_files(app):
    prompt = _agent(app).build_prompt(_ctx(app, working_files=["app.py", "check.py"]))
    assert "本运行涉及的文件" in prompt
    assert "app.py" in prompt and "check.py" in prompt


def test_working_set_reports_the_attempt_count_for_this_state(app):
    prompt = _agent(app).build_prompt(_ctx(app, state="implement", retry={
        "attempts": {"implement": 2, "build_test": 1},
        "failure_streaks": {}, "max_attempts": 3, "escalate_at": 2}))
    assert "本状态（implement）已尝试 2 次" in prompt
    assert "上限 3 次" in prompt


def test_working_set_reports_the_failure_streak_and_the_escalation_rule(app):
    prompt = _agent(app).build_prompt(_ctx(app, retry={
        "attempts": {}, "failure_streaks": {"build_test:test_failure": 1},
        "max_attempts": 3, "escalate_at": 2}))
    assert "build_test:test_failure 连续 1 次" in prompt
    assert "达到 2 次将升级人工审批" in prompt


def test_reset_streaks_are_not_reported_as_active(app):
    """An approved escalation resets a streak to 0; reporting "0 次" would
    suggest a failure is in progress when it is not."""
    prompt = _agent(app).build_prompt(_ctx(app, retry={
        "attempts": {}, "failure_streaks": {"build_test:test_failure": 0},
        "max_attempts": 3, "escalate_at": 2}))
    assert "同一失败类型已连续出现" not in prompt


def test_a_fresh_run_has_no_working_set_section(app):
    prompt = _agent(app).build_prompt(_ctx(app, working_files=[], retry={}))
    assert "## 工作集" not in prompt


def test_working_set_is_never_clipped_even_under_pressure(app):
    """It says which attempt this is — losing it makes the retry loop pointless."""
    agent = _agent(app)
    # Small enough that even with the reducible sections at their floors the
    # prompt cannot fit — the point is that working_set still survives.
    agent.prompt_total_budget = 2500
    ctx = _ctx(
        app,
        state="implement",
        working_files=[f"f{i}.py" for i in range(20)],
        retry={"attempts": {"implement": 2}, "failure_streaks": {},
               "max_attempts": 3, "escalate_at": 2},
        evidence=cap([{"kind": "log", "source": f"e{i}", "content": "x" * 3000}
                      for i in range(20)], 0))

    prompt = agent.build_prompt(ctx)
    meta = BaseAgent.record(ctx)["prompt_metadata"]
    assert meta["over_budget"] is True   # the prompt did not fit
    section = meta["sections"]["working_set"]
    assert section["raw_chars"] == section["rendered_chars"]
    assert "本状态（implement）已尝试 2 次" in prompt


# ------------------------------------------------------------- the real context
def test_the_harness_fills_the_working_set_from_the_run(app):
    h, repo = app["harness"], app["repo"]
    run = h.create_run("新增一个模块", kind="feature")
    repo.update_run(run["id"], state="implement", payload={
        "plan": {"summary": "s", "files": [{"path": "app.py", "action": "modify"}]},
        "attempts": {"implement": 2},
        "failure_attempts": {"build_test:test_failure": 1}})

    ctx = h._build_ctx(repo.get_run(run["id"]))
    assert ctx["working_files"] == ["app.py"]
    assert ctx["retry"]["attempts"] == {"implement": 2}
    assert ctx["retry"]["failure_streaks"] == {"build_test:test_failure": 1}
    assert ctx["retry"]["max_attempts"] == app["cfg"].harness.max_verify_attempts
    assert ctx["retry"]["escalate_at"] == app["cfg"].harness.repeated_failure_threshold

    prompt = h.agents["implementer"].build_prompt(ctx)
    assert "本运行涉及的文件" in prompt
    assert "本状态（implement）已尝试 2 次" in prompt


def test_the_working_set_grows_with_observed_changes(app):
    """Files the run actually changed belong in the set even if unplanned."""
    h, repo = app["harness"], app["repo"]
    run = h.create_run("新增一个模块", kind="feature")
    repo.log_tool_call(run["id"], "implementer", "workspace.write",
                       {"path": "extra.py"}, True, affected_paths=["extra.py"])
    ctx = h._build_ctx(repo.get_run(run["id"]))
    assert ctx["working_files"] == ["extra.py"]


def test_a_real_run_sends_the_working_set_to_the_model(app):
    """End to end: a state that is retrying says so in the prompt it builds."""
    h, repo = app["harness"], app["repo"]
    run = h.create_run("新增一个用户模块，改 app.py", kind="feature")
    # Far enough that the plan exists: a run with nothing planned and nothing
    # failed has no working set, which the test above already covers.
    asyncio.run(h.advance(run["id"], max_loops=4))

    ctx = h._build_ctx(repo.get_run(run["id"]))
    prompt = h.agents["investigator"].build_prompt(ctx)
    assert "## 工作集" in prompt
    assert "本运行涉及的文件" in prompt
    assert "app.py" in prompt
