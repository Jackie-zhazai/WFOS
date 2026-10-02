"""Context budgeting: assemble a prompt under a budget without breaking it.

The load-bearing claim is that compression only ever touches reference material.
The rules, the task, the tool list and the output schema must survive intact —
a prompt that fits the budget by dropping the output schema is not a smaller
prompt, it is a guaranteed schema violation.
"""
from __future__ import annotations

import asyncio

from wfos.agents.base import BaseAgent
from wfos.agents.investigator import InvestigatorAgent
from wfos.harness.context import REDUCTION_ORDER, SECTION_ORDER, clip, fit
from wfos.llm.scripted import ScriptedAdapter
from wfos.relevance import cap

NEVER_CLIPPED = ("rules", "run_context", "tools", "output_schema")


class _Recorder(ScriptedAdapter):
    """Plays a script and keeps every messages list it was handed."""

    def __init__(self):
        super().__init__()
        self.seen: list[list[dict]] = []

    async def complete(self, *, messages=None, schema=None, tools=None,
                       temperature=None, max_tokens=None, ctx=None):
        self.seen.append(list(messages or []))
        return await super().complete(messages=messages, schema=schema, tools=tools,
                                      temperature=temperature, max_tokens=max_tokens,
                                      ctx=ctx)


def _agent_ctx(app, *, evidence=None, task="修复登录失败的问题"):
    # `ctx["evidence"]` is a selection ({items, total, dropped}), not a bare list —
    # the same shape memory uses. `cap(..., 0)` is the no-cap construction, so
    # these tests speak the production vocabulary instead of hand-rolling it.
    return {"run": {"id": "ctx-run", "kind": "bugfix", "title": task, "description": task},
            "state": "issue_capture", "plan": {}, "prior": {},
            "evidence": cap(list(evidence or []), 0), "actual_changes": []}


# ----------------------------------------------------------------------- clip
def test_clip_is_a_noop_under_the_limit():
    assert clip("short", 100) == "short"
    assert clip("short", None) == "short"


def test_clip_keeps_both_ends_and_states_what_it_dropped():
    text = "HEAD" + ("x" * 5000) + "TAIL"
    out = clip(text, 200)
    assert len(out) <= 200
    assert out.startswith("HEAD") and out.endswith("TAIL")   # neither end lost
    assert "省略" in out and "字符" in out                     # the loss is declared


# ------------------------------------------------------------------------ fit
def test_fit_leaves_a_small_prompt_untouched():
    sections = {"rules": "R" * 100, "run_context": "C" * 100, "evidence": "E" * 100}
    prompt, meta = fit(sections, total_budget=10_000)
    assert meta["over_budget"] is False
    assert meta["reductions"] == []
    assert all(meta["sections"][n]["raw_chars"] == meta["sections"][n]["rendered_chars"]
               for n in sections)


def test_fit_reduces_in_the_declared_order_skipping_absent_sections():
    """The property is the *order*, not which one happens to be first.

    `REDUCTION_ORDER` names sections cheapest-to-lose first, and a section a
    prompt does not have is simply skipped — so asserting `reductions[0]` equals
    `REDUCTION_ORDER[0]` only holds when that section is present, which is how
    this test used to pass and why adding `skills` ahead of `evidence` broke it.
    """
    sections = {"rules": "R" * 500, "run_context": "C" * 500,
                "evidence": "E" * 5000, "plan": "P" * 5000, "prior_plan": "Q" * 5000}
    _, meta = fit(sections, total_budget=6000, budgets={})   # no per-section caps
    assert meta["over_budget"] is False
    by_total = [r for r in meta["reductions"] if r["phase"] == "total_budget"]

    rank = {name: i for i, name in enumerate(REDUCTION_ORDER)}
    reduced = [r["section"] for r in by_total]
    assert reduced == sorted(reduced, key=rank.get), reduced
    assert reduced[0] == "evidence"          # the cheapest-to-lose one present here
    assert meta["sections"]["evidence"]["rendered_chars"] < 5000


def test_skills_are_the_first_thing_to_go_when_present():
    """A procedure is a suggestion about how to get past a failure; the plan is
    what the state is executing, so the suggestion goes first."""
    sections = {"rules": "R" * 500, "skills": "S" * 5000, "evidence": "E" * 5000,
                "plan": "P" * 5000}
    _, meta = fit(sections, total_budget=6000, budgets={})

    by_total = [r for r in meta["reductions"] if r["phase"] == "total_budget"]
    assert by_total[0]["section"] == "skills" == REDUCTION_ORDER[0]
    assert "skills" in SECTION_ORDER and "skills" not in (
        "rules", "run_context", "working_set")


def test_a_section_capped_by_its_own_budget_is_still_logged():
    """Clipping to a section ceiling is a reduction; recording it only in the
    total-budget loop would hide every routine cut."""
    sections = {"rules": "R" * 100, "evidence": "E" * 50_000}
    _, meta = fit(sections, total_budget=100_000)     # total never binds
    assert meta["over_budget"] is False
    assert meta["reductions"] == [{"section": "evidence", "phase": "section_budget",
                                   "from_chars": 50_000,
                                   "to_chars": meta["sections"]["evidence"]["rendered_chars"],
                                   "budget_chars": 8000}]


def test_fit_never_clips_the_load_bearing_sections():
    """Even absurdly large, the rules/task/tools/schema are left intact."""
    sections = {"rules": "R" * 4000, "run_context": "C" * 4000,
                "tools": "T" * 4000, "output_schema": "S" * 4000,
                "evidence": "E" * 40_000}
    prompt, meta = fit(sections, total_budget=12_000)
    for name in NEVER_CLIPPED:
        assert meta["sections"][name]["rendered_chars"] == \
            meta["sections"][name]["raw_chars"], name
        assert name in meta["sections"]
    assert "R" * 4000 in prompt and "S" * 4000 in prompt
    assert "E" * 40_000 not in prompt                      # only evidence shrank


def test_fit_reports_being_over_budget_rather_than_pretending():
    """When the unclippable part alone exceeds the budget, say so."""
    sections = {"rules": "R" * 9000, "run_context": "C" * 9000, "evidence": "E" * 9000}
    prompt, meta = fit(sections, total_budget=5000)
    assert meta["over_budget"] is True
    assert meta["total_chars"] == len(prompt)


def test_fit_respects_an_explicit_section_budget():
    sections = {"rules": "R" * 100, "evidence": "E" * 10_000}
    _, meta = fit(sections, total_budget=100_000, budgets={"evidence": 500})
    assert meta["sections"]["evidence"]["budget_chars"] == 500
    assert meta["sections"]["evidence"]["rendered_chars"] <= 500


def test_fit_respects_floors():
    """A section is not reduced below its floor even under heavy pressure."""
    sections = {"rules": "R" * 100, "evidence": "E" * 20_000}
    _, meta = fit(sections, total_budget=1500, floors={"evidence": 1200})
    assert meta["sections"]["evidence"]["rendered_chars"] >= 1200


def test_section_order_is_stable_and_complete():
    assert SECTION_ORDER[:2] == ("rules", "run_context")
    assert "output_schema" in SECTION_ORDER
    assert set(REDUCTION_ORDER).issubset(set(SECTION_ORDER))
    assert set(REDUCTION_ORDER).isdisjoint(set(NEVER_CLIPPED))


# ---------------------------------------------------------------- in the agent
def test_agent_prompt_never_loses_the_task(app):
    """The task is the one thing a compression bug must never eat."""
    evidence = [{"kind": "log", "source": f"f{i}.log", "content": "y" * 2000}
                for i in range(50)]
    agent = InvestigatorAgent(ScriptedAdapter(), app["gateway"], app["sandbox"])
    ctx = _agent_ctx(app, evidence=evidence, task="修复登录失败的问题")
    prompt = agent.build_prompt(ctx)

    assert "修复登录失败的问题" in prompt
    # The accounting rides in the context, not on the agent: the agent object is
    # shared by every run a Harness advances, so a field on it is one slot for
    # the whole process and two runs would read each other's. See
    # `BaseAgent.record_of`.
    meta = BaseAgent.record(ctx)["prompt_metadata"]
    assert meta["reductions"], "大量证据应当触发压缩"
    for name in NEVER_CLIPPED:
        if name not in meta["sections"]:
            continue                       # no tools passed to this prompt
        assert meta["sections"][name]["raw_chars"] == meta["sections"][name]["rendered_chars"]
    assert meta["sections"]["evidence"]["rendered_chars"] < \
        meta["sections"]["evidence"]["raw_chars"]


def test_agent_prompt_is_within_budget_for_a_normal_run(app):
    agent = InvestigatorAgent(ScriptedAdapter(), app["gateway"], app["sandbox"])
    ctx = _agent_ctx(app)
    agent.build_prompt(ctx, tools=agent._tool_specs())
    meta = BaseAgent.record(ctx)["prompt_metadata"]
    assert meta["over_budget"] is False
    assert meta["reductions"] == []                        # nothing needed cutting


def test_a_huge_tool_result_is_clipped_before_it_enters_the_conversation(app):
    (app["sandbox"] / "big.py").write_text("BIG = '" + ("z" * 40_000) + "'\n",
                                           encoding="utf-8")
    llm = _Recorder()
    llm.set_script([
        {"tool_call": {"name": "workspace.read", "arguments": {"path": "big.py"}}},
        {"output": {"findings": [], "evidence": [], "project_context": {},
                    "next_step": {"suggested_state": "project_check", "reason": "d"}}},
    ])
    agent = InvestigatorAgent(llm, app["gateway"], app["sandbox"],
                              tool_result_budget=2000)
    asyncio.run(agent.run(_agent_ctx(app)))

    tool_messages = [m for m in llm.seen[-1] if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    content = tool_messages[0]["content"]
    assert len(content) <= 2000
    assert "省略" in content                      # the cut is declared, not silent
    assert content.startswith("{") and content.rstrip().endswith("}")
    assert "z" * 100 in content                    # both ends of the payload survive
