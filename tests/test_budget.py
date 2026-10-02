"""Stopping a run that has spent its budget.

Before this existed the harness *measured* tokens and cost and nothing consumed
the numbers: no ceiling, no stop, no reader. That is the same defect in a third
place — plumbing installed and no water in it — so most of what is pinned here is
that the limit actually bites.

The harder half is the opposite: a limit must not appear to bite when it cannot.
Against a brain that reports no usage the totals are `None`, not 0, and the
honest behaviour is that the budget does nothing. A ceiling that silently does
nothing while looking armed is worse than no ceiling, so that case is tested
explicitly rather than left to fall out of the arithmetic.
"""
from __future__ import annotations

import asyncio

import pytest

from wfos.config import _harness_from_toml
from wfos.llm.base import LLMAdapter
from wfos.llm.mock import MockAdapter
from wfos.metrics import billable_tokens, run_metrics
from wfos.models import ModelResult, ModelUsage

FEATURE = "新增一个用户模块，在 app.py 中追加 feature_user() 函数"
PER_CALL_INPUT, PER_CALL_OUTPUT = 100, 20


class _MeteredAdapter(LLMAdapter):
    """The mock brain, reporting usage like a real provider would."""

    name = "metered"

    def __init__(self, inner=None):
        self._inner = inner or MockAdapter()

    async def complete(self, *, messages=None, schema=None, tools=None,
                       temperature=None, max_tokens=None, ctx=None) -> ModelResult:
        result = await self._inner.complete(messages=messages, schema=schema,
                                           tools=tools, temperature=temperature,
                                           max_tokens=max_tokens, ctx=ctx)
        result.usage = ModelUsage(input_tokens=PER_CALL_INPUT,
                                  output_tokens=PER_CALL_OUTPUT)
        return result


def _metered(app, *, price=None):
    h, cfg = app["harness"], app["cfg"]
    if price:
        cfg.pricing = {"mock": price}
    for agent in h.agents.values():
        agent.llm = _MeteredAdapter()


def _drive(h, text=FEATURE) -> str:
    run = h.create_run(text, kind="feature")
    asyncio.run(h.advance(run["id"]))
    return run["id"]


def _budget_transitions(repo, run_id):
    return [t for t in repo.transitions(run_id)
            if t["decision"] == "harness" and "预算耗尽" in (t["reason"] or "")]


# -------------------------------------------------------- the limit is inert
def test_with_no_budget_configured_nothing_changes(app):
    h, repo, cfg = app["harness"], app["repo"], app["cfg"]
    _metered(app)
    assert cfg.harness.run_token_budget is None
    assert cfg.harness.run_cost_budget_usd is None

    run_id = _drive(h)

    assert _budget_transitions(repo, run_id) == []
    assert "预算" not in (repo.get_run(run_id).get("error") or "")


def test_a_budget_above_what_the_run_spends_does_not_stop_it(app):
    """The check runs every loop; it must be a ceiling, not a tripwire."""
    h, repo, cfg = app["harness"], app["repo"], app["cfg"]
    _metered(app)
    cfg.harness.run_token_budget = 10_000_000

    run_id = _drive(h)

    assert _budget_transitions(repo, run_id) == []
    report = run_metrics(repo, run_id)
    assert report["steps_with_usage"] > 0, "没有上报用量，这个测试就没在测预算"


def test_an_unreported_usage_leaves_the_budget_unable_to_bite(app):
    """A limit against a number nobody gave cannot be enforced.

    The mock brain reports nothing, so the totals are None. Firing here would mean
    guessing a spend, and the run would be killed by a figure the harness made up.
    """
    h, repo, cfg = app["harness"], app["repo"], app["cfg"]
    cfg.harness.run_token_budget = 1          # would trip on any real usage at all

    run_id = _drive(h)

    report = run_metrics(repo, run_id)
    assert report["input_tokens"] is None and report["cost_usd"] is None
    assert _budget_transitions(repo, run_id) == [], \
        "用量未被上报时预算必须咬不到，否则就是按猜出来的数字杀人"
    assert repo.get_run(run_id)["error"] in (None, "")


def test_a_cost_budget_cannot_bite_without_prices(app):
    """Tokens reported, model unpriced -> cost is None, so the cost limit is moot.

    The distinction the project keeps everywhere: an unpriced model yields None,
    never 0, so a cost ceiling cannot be evaluated against a made-up zero.
    """
    h, repo, cfg = app["harness"], app["repo"], app["cfg"]
    _metered(app)                              # usage reported, cfg.pricing empty
    cfg.harness.run_cost_budget_usd = 0.000001

    run_id = _drive(h)

    assert run_metrics(repo, run_id)["cost_usd"] is None
    assert _budget_transitions(repo, run_id) == []


# ------------------------------------------------------------ the limit bites
def test_exceeding_the_token_budget_stops_the_run(app):
    h, repo, cfg = app["harness"], app["repo"], app["cfg"]
    _metered(app)
    cfg.harness.run_token_budget = 1

    run_id = _drive(h)

    run = repo.get_run(run_id)
    assert run["status"] == "failed"
    assert "token 预算耗尽" in run["error"] and "上限 1" in run["error"]
    fired = _budget_transitions(repo, run_id)
    assert len(fired) == 1


def test_exceeding_the_cost_budget_stops_the_run(app):
    h, repo, cfg = app["harness"], app["repo"], app["cfg"]
    _metered(app, price={"input": 1.0, "output": 1.0})
    cfg.harness.run_cost_budget_usd = 0.000001

    run_id = _drive(h)

    run = repo.get_run(run_id)
    assert run["status"] == "failed"
    assert "成本预算耗尽" in run["error"] and "$" in run["error"]


def test_the_stop_lands_between_states_not_inside_one(app):
    """The step that pushed the run over is recorded whole.

    Stopping mid-state would leave a half-written step and no way to tell whether
    its side effects had already landed, so the check sits at the top of the loop:
    it fires before the next state, never during the one in flight.
    """
    h, repo, cfg = app["harness"], app["repo"], app["cfg"]
    _metered(app)
    cfg.harness.run_token_budget = 1

    run_id = _drive(h)

    rows = repo.metric_rows(run_id)
    assert rows and all(r["input_tokens"] == PER_CALL_INPUT for r in rows), \
        "停止时那一步的用量没有完整落库"
    report = run_metrics(repo, run_id)
    # Everything the run measured is on a finished step: no spend was cut in half.
    assert report["steps_with_usage"] == report["steps"]
    fired = _budget_transitions(repo, run_id)[0]
    assert fired["from_state"] == fired["to_state"], \
        "预算停止不是状态迁移，记录里不该假装从一个状态走到了另一个"
    assert fired["from_state"] == repo.get_run(run_id)["state"], \
        "停止记录的状态与运行所在的状态不一致"


def test_the_budget_stops_the_run_earlier_than_it_would_have_gone(app):
    """The discriminating test: same scenario, one variable — the budget."""
    h, repo, cfg = app["harness"], app["repo"], app["cfg"]
    _metered(app)
    unbudgeted = _drive(h, FEATURE)

    cfg.harness.run_token_budget = 1
    budgeted = _drive(h, "新增一个别的模块，追加 feature_other() 函数")

    assert len(repo.steps_for_run(budgeted)) < len(repo.steps_for_run(unbudgeted)), \
        "预算生效时步骤数应当严格少于不设预算时"
    assert repo.get_run(unbudgeted)["status"] != "failed"
    assert repo.get_run(budgeted)["status"] == "failed"


def test_the_comparison_is_strictly_over_the_limit(app):
    """`>` not `>=`: spending exactly the ceiling is within budget.

    Pinned with a pair, because a test that only checks one side cannot tell the
    two operators apart.
    """
    h, repo, cfg = app["harness"], app["repo"], app["cfg"]
    _metered(app)
    cfg.harness.run_token_budget = 10_000_000
    run_id = _drive(h)
    spent = billable_tokens(run_metrics(repo, run_id))
    run = repo.get_run(run_id)
    assert spent and spent > 0

    cfg.harness.run_token_budget = spent
    assert h._budget_exhausted(run) == "", "正好花完上限不算超"

    cfg.harness.run_token_budget = spent - 1
    assert "预算耗尽" in h._budget_exhausted(run), "差 1 个 token 就该判超"


def test_the_budget_is_per_run_not_global(app):
    """One run's spending cannot kill another."""
    h, repo, cfg = app["harness"], app["repo"], app["cfg"]
    _metered(app)
    cfg.harness.run_token_budget = 10_000_000
    generous = _drive(h)

    cfg.harness.run_token_budget = 1
    small = _drive(h, "新增一个别的模块，追加 feature_other() 函数")

    assert repo.get_run(small)["status"] == "failed"
    assert _budget_transitions(repo, small)
    assert _budget_transitions(repo, generous) == [], \
        "另一次运行的预算被这一次的消耗触发了"


# ------------------------------------------------------------- the arithmetic
def test_billable_tokens_counts_input_plus_output_only():
    """Cached and reasoning are breakdowns, so adding them double-counts.

    A cached input token is still an input token; a reasoning token is counted in
    the output. Summing all four would inflate the figure the ceiling is compared
    against, and the run would be stopped for spending money it had not spent.
    """
    assert billable_tokens({"input_tokens": 100, "output_tokens": 20,
                            "cached_tokens": 90, "reasoning_tokens": 15}) == 120
    assert billable_tokens({"input_tokens": 100}) == 100
    assert billable_tokens({"output_tokens": 20}) == 20


def test_billable_tokens_is_none_when_nothing_was_reported():
    assert billable_tokens({"input_tokens": None, "output_tokens": None}) is None
    assert billable_tokens({}) is None


def test_a_reported_zero_is_a_number_not_a_missing_one():
    """`0` is a fact; the ceiling may be computed from it."""
    assert billable_tokens({"input_tokens": 0, "output_tokens": None}) == 0
    assert billable_tokens({"input_tokens": 0, "output_tokens": 0}) == 0


# ------------------------------------------------------------------- the knob
def test_the_budget_fields_parse_from_toml():
    assert _harness_from_toml({"run_token_budget": 5000}).run_token_budget == 5000
    assert _harness_from_toml({"run_cost_budget_usd": 2.5}).run_cost_budget_usd == 2.5
    assert _harness_from_toml({}).run_token_budget is None
    assert _harness_from_toml({}).run_cost_budget_usd is None


def test_a_budget_of_zero_is_not_the_same_as_no_budget():
    """TOML cannot express null, so absence is the only "unset" — 0 must survive.

    Collapsing 0 into None would turn "spend nothing" into "spend anything".
    """
    assert _harness_from_toml({"run_token_budget": 0}).run_token_budget == 0
    assert _harness_from_toml({"run_cost_budget_usd": 0}).run_cost_budget_usd == 0.0


def test_a_boolean_is_not_read_as_a_budget():
    """TOML `true` is an int in Python; a budget of 1 would be an accident."""
    assert _harness_from_toml({"run_token_budget": True}).run_token_budget is None
    assert _harness_from_toml({"run_cost_budget_usd": True}).run_cost_budget_usd is None


def test_a_budget_of_zero_aborts_once_anything_is_spent(app):
    """The end-to-end reading of the above: 0 means "no spend at all", not "unlimited"."""
    h, repo, cfg = app["harness"], app["repo"], app["cfg"]
    _metered(app)
    cfg.harness.run_token_budget = 0

    run_id = _drive(h)

    assert repo.get_run(run_id)["status"] == "failed"
    assert _budget_transitions(repo, run_id)


def test_an_unparsable_budget_does_not_crash_the_load():
    assert _harness_from_toml({"run_token_budget": "lots"}).run_token_budget is None
    assert _harness_from_toml({"run_cost_budget_usd": "cheap"}).run_cost_budget_usd is None


@pytest.mark.parametrize("field", ["run_token_budget", "run_cost_budget_usd"])
def test_the_knob_is_documented_in_the_shipped_defaults(field):
    """A knob an operator cannot discover is a knob that does not exist."""
    from wfos.config import _DEFAULTS_TOML
    assert field in _DEFAULTS_TOML
