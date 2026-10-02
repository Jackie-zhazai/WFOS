"""Runs must not contaminate each other, in a process or across one.

`Run 必须隔离` is a hard rule, and it is checked here at the level it is actually
violated: shared *objects*. Two runs advancing at once in one process share the
agent instances, the gateway and the server, so anything kept on those is one
slot for every run in flight.

The per-call principal (see `test_policy.py`) fixed that for tool calls. This
file covers the other half: what a step records about itself — the prompt the
model was shown and what the call cost. That used to live in `agent.last_prompt`
and `agent.last_usage`, which meant a step's prompt and its token counts could
belong to whichever run wrote them last.
"""
from __future__ import annotations

import asyncio

from wfos.llm.mock import MockAdapter


class _YieldingMock(MockAdapter):
    """The mock brain, but it actually yields control.

    Without a yield there is nothing to interleave: `MockAdapter.complete`
    returns without awaiting anything, so each state would run to completion and
    a sharing bug would never show. These yields are what make the test able to
    fail at all.
    """

    name = "yielding-mock"

    async def complete(self, **kwargs):
        for _ in range(8):
            await asyncio.sleep(0)
        return await super().complete(**kwargs)


def _prompts(repo, run_id: str) -> list[str]:
    """Every prompt this run's steps recorded, in order."""
    return [str((s.get("input_json") or {}).get("prompt") or "")
            for s in repo.steps_for_run(run_id)]


def _prompt_run_id(prompt: str) -> str | None:
    """The run the prompt's own context block names, if it has one.

    The assertion is on this line rather than on "the other run's text is
    absent". It has to be: two interactive runs in one project legitimately see
    each other's **memory**, which quotes the other run's title — so a marker
    search reports a working feature as contamination. This line can only come
    from the context the state was built with.
    """
    for line in prompt.splitlines():
        if line.startswith("- 运行 ID:"):
            return line.split(":", 1)[1].strip()
    return None


def test_two_runs_advancing_at_once_do_not_read_each_others_prompt(app):
    """The step record is per state of one run, not per agent object.

    The agent instances are shared by every run a Harness advances, so a field on
    them is one slot for the process — under that design a step's recorded prompt
    belongs to whichever run wrote it last.
    """
    h, repo = app["harness"], app["repo"]
    for agent in h.agents.values():
        agent.llm = _YieldingMock()

    alpha = h.create_run("新增一个模块 alpha_marker", kind="feature")
    beta = h.create_run("新增一个模块 beta_marker", kind="feature")

    async def both():
        await asyncio.gather(h.advance(alpha["id"]), h.advance(beta["id"]))

    asyncio.run(both())

    for run in (alpha, beta):
        prompts = _prompts(repo, run["id"])
        assert prompts, f"{run['id'][:8]} 没有产生任何步骤"
        named = {_prompt_run_id(p) for p in prompts}
        assert named == {run["id"]}, (
            f"运行 {run['id'][:8]} 的步骤里记录的 prompt 属于 {named}"
            " —— 记录不是按 run 隔离的")


def test_a_delegated_child_does_not_overwrite_the_parents_record(app):
    """The nested case, which is the one that happens without concurrency."""
    from wfos.agents.investigator import InvestigatorAgent
    from wfos.llm.scripted import ScriptedAdapter

    h, repo = app["harness"], app["repo"]
    parent = h.create_run("新增一个模块 parent_marker", kind="feature")

    llm = ScriptedAdapter()
    llm.set_script([
        {"tool_call": {"name": "delegate", "arguments": {"task": "做一个独立的小模块"}}},
        {"tool_call": {"name": "workspace.read", "arguments": {"path": "app.py"}}},
        {"output": {"findings": [], "evidence": [], "project_context": {},
                    "next_step": {"suggested_state": "project_check", "reason": "done"}}}])
    agent = InvestigatorAgent(llm, app["gateway"], app["sandbox"])
    ctx = h._build_ctx(repo.get_run(parent["id"]))

    asyncio.run(agent.run(ctx))

    record = agent.record_of(ctx)
    assert _prompt_run_id(record.get("prompt", "")) == parent["id"], \
        "父流程自己的 prompt 记录被子流程覆盖了"
