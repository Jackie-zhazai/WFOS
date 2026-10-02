"""Cost accounting: what a run measured, and what it could not.

The load-bearing property here is that an *unreported* count stays `None` all
the way to the report — never 0. `0` is a real, reportable value (a provider can
say a call used no cached tokens), so defaulting an absent count to 0 would make
"we were not told" indistinguishable from "it was free" in every total computed
from it. Most of these tests exist to pin that distinction.

The second property is coverage: a sum over 10 steps where 2 reported is not the
run's token count, so the report carries how many steps each total was computed
over.
"""
from __future__ import annotations

import asyncio

import pytest

from wfos.llm.anthropic import parse_anthropic_response
from wfos.llm.base import LLMAdapter, int_or_none, usage_from_fields
from wfos.llm.gemini import parse_gemini_response
from wfos.llm.mock import MockAdapter
from wfos.llm.openai_compat import parse_response
from wfos.metrics import aggregate, cost_usd, render, run_metrics
from wfos.models import ModelResult, ModelUsage


# --------------------------------------------------------------- absent vs zero
def test_an_absent_count_is_not_a_zero_count():
    """`None` and `0` are different facts and must stay distinct."""
    assert int_or_none(None) is None
    assert int_or_none("n/a") is None
    assert int_or_none(True) is None          # bool is an int subclass; not a count
    assert int_or_none(0) == 0                # a reported zero is a measurement
    assert int_or_none(12.9) == 12


def test_a_usage_block_with_no_counts_collapses_to_none():
    assert usage_from_fields(input_tokens=None, output_tokens=None,
                             cached_tokens=None) is None
    assert usage_from_fields() is None


def test_reported_zero_survives_as_a_measurement():
    usage = usage_from_fields(input_tokens=0, output_tokens=None, cached_tokens=None)
    assert usage is not None and usage.input_tokens == 0


# ------------------------------------------------------------ provider shapes
def test_openai_usage_is_parsed_including_the_nested_cached_count():
    result = parse_response({
        "choices": [{"message": {"content": "{}"}}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 30,
                  "prompt_tokens_details": {"cached_tokens": 64}},
    })
    assert result.usage.input_tokens == 120
    assert result.usage.output_tokens == 30
    assert result.usage.cached_tokens == 64


def test_openai_response_without_usage_reports_nothing():
    result = parse_response({"choices": [{"message": {"content": "{}"}}]})
    assert result.usage is None


def test_anthropic_maps_cache_read_onto_cached_tokens():
    result = parse_anthropic_response({
        "content": [{"type": "text", "text": "{}"}],
        "usage": {"input_tokens": 200, "output_tokens": 40,
                  "cache_read_input_tokens": 150},
    })
    assert result.usage.input_tokens == 200
    assert result.usage.output_tokens == 40
    assert result.usage.cached_tokens == 150


def test_gemini_usage_comes_from_its_own_camelcase_block():
    result = parse_gemini_response({
        "candidates": [{"content": {"parts": [{"text": "{}"}]}}],
        "usageMetadata": {"promptTokenCount": 90, "candidatesTokenCount": 15,
                          "cachedContentTokenCount": 7},
    })
    assert result.usage.input_tokens == 90
    assert result.usage.output_tokens == 15
    assert result.usage.cached_tokens == 7


# -------------------------------------------------------------------- pricing
def test_a_model_with_no_configured_price_has_no_cost():
    """An unpriced model yields None, never 0 — 0 would claim it was free."""
    assert cost_usd("mystery-model", {}, input_tokens=100, output_tokens=10) is None
    assert cost_usd("mystery-model", {"other": {"input": 1.0}},
                    input_tokens=100, output_tokens=10) is None


def test_cost_prices_each_side_it_has_both_a_rate_and_a_count_for():
    both = {"m": {"input": 2.5, "output": 10.0}}
    assert cost_usd("m", both, input_tokens=1_000_000,
                    output_tokens=1_000_000) == pytest.approx(12.5)
    # Only the input side has a rate, so only that side is priced.
    assert cost_usd("m", {"m": {"input": 2.5}},
                    input_tokens=1_000_000) == pytest.approx(2.5)


def test_cost_needs_a_reported_count_to_work_from():
    pricing = {"m": {"input": 1.0, "output": 1.0}}
    assert cost_usd("m", pricing, input_tokens=None, output_tokens=None) is None
    assert cost_usd("m", pricing, input_tokens=1_000_000) == pytest.approx(1.0)
    assert cost_usd("m", pricing, input_tokens=0) == 0.0     # reported zero


# ------------------------------------------------------ end to end over a run
def _run_a_feature(h) -> str:
    run_id = h.create_run("新增一个用户模块，在 app.py 中追加 feature_user() 函数",
                          kind="feature")["id"]
    asyncio.run(h.advance(run_id))
    return run_id


def test_under_mock_nothing_is_reported_but_latency_is_still_measured(app):
    """The mock brain has no provider, so tokens are unknown — not zero.

    Latency and call count are measured by the harness itself, so those are
    always known and must be recorded even when the provider says nothing.
    """
    h, repo = app["harness"], app["repo"]
    run_id = _run_a_feature(h)

    rows = repo.metric_rows(run_id)
    assert rows, "运行没有产生任何 step"
    assert all(r["input_tokens"] is None for r in rows), "mock 不该产生 token 数"
    assert all(r["cost_usd"] is None for r in rows)
    assert all(r["model_calls"] >= 1 for r in rows), "调用次数是 harness 自己数的"
    assert all(r["latency_ms"] is not None for r in rows)

    report = run_metrics(repo, run_id)
    assert report["steps"] == len(rows)
    assert report["steps_with_usage"] == 0
    assert report["input_tokens"] is None, "没有上报应当是 None，而不是 0"
    assert report["cost_usd"] is None
    text = render(report, model="mock", pricing={})
    assert "未上报用量" in text
    # Unknown figures print as unknown rather than as a number.
    assert "input tokens   -" in text
    assert "成本         -" in text


class _MeteredAdapter(LLMAdapter):
    """Behaves like the mock brain but reports usage, standing in for a real
    provider so the whole accounting chain can be tested without a network."""

    name = "metered"

    def __init__(self, inner=None):
        self._inner = inner or MockAdapter()

    async def complete(self, *, messages=None, schema=None, tools=None,
                       temperature=None, max_tokens=None, ctx=None) -> ModelResult:
        result = await self._inner.complete(messages=messages, schema=schema,
                                           tools=tools, temperature=temperature,
                                           max_tokens=max_tokens, ctx=ctx)
        result.usage = ModelUsage(input_tokens=100, output_tokens=20, cached_tokens=5)
        return result


def test_usage_and_cost_reach_the_step_record_from_the_adapter(app):
    """Adapter -> agent accumulation -> step record -> aggregate, priced."""
    h, repo, cfg = app["harness"], app["repo"], app["cfg"]
    cfg.pricing = {"mock": {"input": 1.0, "output": 2.0}}
    metered = _MeteredAdapter()
    for agent in h.agents.values():
        agent.llm = metered

    run_id = _run_a_feature(h)

    rows = repo.metric_rows(run_id)
    used = [r for r in rows if r["input_tokens"] is not None]
    assert used, "provider 上报的用量没有落到 step 记录"
    per_step = 100 / 1e6 * 1.0 + 20 / 1e6 * 2.0
    for row in used:
        assert (row["input_tokens"], row["output_tokens"], row["cached_tokens"]) == (100, 20, 5)
        assert row["model_calls"] >= 1
        assert row["cost_usd"] == pytest.approx(per_step)

    report = run_metrics(repo, run_id)
    assert report["steps_with_usage"] == len(used) == report["steps_with_cost"]
    assert report["input_tokens"] == 100 * len(used)
    assert report["cost_usd"] == pytest.approx(per_step * len(used))
    assert "$" in render(report, model="mock", pricing=cfg.pricing)


def test_the_aggregate_carries_its_coverage(app):
    """A total computed over part of the data must say so."""
    h, repo, cfg = app["harness"], app["repo"], app["cfg"]
    cfg.pricing = {"mock": {"input": 1.0, "output": 1.0}}

    # One run whose provider reports usage, one whose provider does not.
    for agent in h.agents.values():
        agent.llm = _MeteredAdapter()
    metered_run = _run_a_feature(h)
    for agent in h.agents.values():
        agent.llm = MockAdapter()
    silent_run = h.create_run("新增一个模块", kind="feature")["id"]
    asyncio.run(h.advance(silent_run))

    report = aggregate(repo)
    assert report["runs"] == 2
    assert 0 < report["steps_with_usage"] < report["steps"]
    assert report["cost_usd"] is not None
    # The uncovered steps contributed nothing and are not counted as free.
    assert report["steps_with_cost"] == report["steps_with_usage"]
    assert set(report["by_kind"]) == {"feature"}
    assert repo.metric_rows(metered_run) and repo.metric_rows(silent_run)
