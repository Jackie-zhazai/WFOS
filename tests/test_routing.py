"""Per-role model selection.

The point of routing is that some roles are worth a stronger model and some are
not — the implementer writes the code, the curator distils a summary. Before this
the harness asked every role of the same model, all the way through.

Two things are pinned here that a "does the config parse" test would miss:

  * the routed adapter is the one that actually answers, and the step record says
    which model that was — otherwise the feature is invisible from the database;
  * cost is computed at the rate of the model that answered, not the default's.
    A wrong price written into a step is a historical fact that no later reader
    can correct.
"""
from __future__ import annotations

import asyncio

import pytest

from wfos.harness.orchestrator import AGENT_CLASSES, STATE_AGENT, Harness
from wfos.llm.factory import build_routed_adapters
from wfos.llm.mock import MockAdapter
from wfos.llm.openai_compat import OpenAICompatAdapter
from wfos.metrics import render, run_metrics
from wfos.models import ModelResult, ModelUsage

FEATURE = "新增一个用户模块，在 app.py 中追加 feature_user() 函数"

# Distinct rates so a mis-priced step cannot coincide with a correct one.
RATES = {"base": 1.0, "cheap": 0.1, "strong": 10.0}
REPORTED_INPUT_TOKENS = 1000


# ------------------------------------------------------------------ the factory
def _llm(**kw):
    from wfos.config import LLMConfig
    return LLMConfig(provider="mock", model="base", **kw)


def test_without_routing_every_role_shares_one_adapter(app):
    """An unrouted configuration must build exactly one adapter — routing is
    free when it is not used, and there is nothing to keep in sync."""
    default, by_role = build_routed_adapters(_llm(), roles=AGENT_CLASSES)

    assert by_role == {}
    assert isinstance(default, MockAdapter)


def test_a_route_gets_its_own_adapter(app):
    default, by_role = build_routed_adapters(
        _llm(routing={"implementer": "strong"}), roles=AGENT_CLASSES)

    assert set(by_role) == {"implementer"}
    assert by_role["implementer"] is not default


def test_a_route_naming_the_default_model_reuses_the_default_adapter(app):
    """No clone: the same model is the same adapter."""
    default, by_role = build_routed_adapters(
        _llm(routing={"verifier": "base"}), roles=AGENT_CLASSES)

    assert by_role["verifier"] is default


def test_two_roles_on_one_model_share_that_model_adapter(app):
    default, by_role = build_routed_adapters(
        _llm(routing={"verifier": "strong", "curator": "strong"}),
        roles=AGENT_CLASSES)

    assert by_role["verifier"] is by_role["curator"]
    assert by_role["verifier"] is not default


def test_an_empty_route_means_unrouted(app):
    """`implementer = ""` cannot mean "the model called empty string".

    Two layers, two jobs: the TOML reader drops the line entirely, and the factory
    falls back to the default adapter for any route that names no model — so a
    hand-built config reaches the same place the file does.
    """
    from wfos.config import _llm_from_toml

    assert _llm_from_toml({"routing": {"implementer": ""}}).routing == {}

    default, by_role = build_routed_adapters(
        _llm(routing={"implementer": ""}), roles=AGENT_CLASSES)
    assert by_role["implementer"] is default


def test_a_route_naming_a_role_that_does_not_exist_is_refused(app):
    """A typo must not become a setting that silently does nothing."""
    with pytest.raises(ValueError) as excinfo:
        build_routed_adapters(_llm(routing={"implementor": "strong"}),
                              roles=AGENT_CLASSES)

    message = str(excinfo.value)
    assert "implementor" in message
    assert "implementer" in message          # the message lists the real roles


def test_the_roles_that_can_be_routed_are_the_agents_that_exist():
    assert set(AGENT_CLASSES) == {"investigator", "architect", "implementer",
                                  "verifier", "curator"}


# ------------------------------------------------------------------- the wiring
def test_each_agent_gets_its_routed_adapter(app):
    cfg, repo, gateway, wiki = app["cfg"], app["repo"], app["gateway"], app["wiki"]
    cfg.llm.provider = "vllm"                # no credential needed
    cfg.llm.model = "base"
    cfg.llm.routing = {"implementer": "strong", "curator": "cheap"}

    h = Harness(cfg, repo, gateway, wiki)

    assert h.agents["implementer"].llm.model == "strong"
    assert h.agents["curator"].llm.model == "cheap"
    for role in ("investigator", "architect", "verifier"):
        assert h.agents[role].llm.model == "base"
    assert h.llm is h.agents["verifier"].llm, "默认适配器应当就是未路由角色的适配器"


def test_an_injected_brain_overrides_every_route(app):
    """The caller handed us one brain; quietly using a second one would make the
    run unreproducible from the outside."""
    cfg, repo, gateway, wiki = app["cfg"], app["repo"], app["gateway"], app["wiki"]
    cfg.llm.provider = "vllm"
    cfg.llm.routing = {"implementer": "strong"}
    injected = MockAdapter()

    h = Harness(cfg, repo, gateway, wiki, llm=injected)

    assert h.routed == {}
    assert all(a.llm is injected for a in h.agents.values())


def test_a_misconfigured_provider_is_not_checked_when_a_brain_is_injected(app):
    """`llm=` used to short-circuit the credential check, and it must keep doing
    so: building the routed adapters anyway would contact-check a provider this
    process never talks to."""
    cfg, repo, gateway, wiki = app["cfg"], app["repo"], app["gateway"], app["wiki"]
    cfg.llm.provider = "openai"              # would need a credential
    cfg.llm.api_key_env = "WFOS_DEFINITELY_UNSET"

    h = Harness(cfg, repo, gateway, wiki, llm=MockAdapter())   # must not raise

    assert all(a.llm.model == "mock" for a in h.agents.values())


# ---------------------------------------------------------------- end to end
def _record_calls(seen: list[str]):
    """A real adapter that answers like the mock, and reports usage."""
    inner = MockAdapter()

    async def complete(self, *, messages=None, schema=None, tools=None,
                       temperature=None, max_tokens=None, ctx=None) -> ModelResult:
        seen.append(self.model)
        result = await inner.complete(messages=messages, schema=schema, tools=tools,
                                      temperature=temperature, max_tokens=max_tokens,
                                      ctx=ctx)
        result.usage = ModelUsage(input_tokens=REPORTED_INPUT_TOKENS, output_tokens=0)
        return result
    return complete


def _routed_run(app, monkeypatch):
    cfg, repo, gateway, wiki = app["cfg"], app["repo"], app["gateway"], app["wiki"]
    cfg.llm.provider = "vllm"
    cfg.llm.model = "base"
    cfg.llm.routing = {"implementer": "strong", "curator": "cheap"}
    cfg.pricing = {name: {"input": rate, "output": 0.0} for name, rate in RATES.items()}

    seen: list[str] = []
    monkeypatch.setattr(OpenAICompatAdapter, "complete", _record_calls(seen))

    h = Harness(cfg, repo, gateway, wiki)
    run = h.create_run(FEATURE, kind="feature")
    asyncio.run(h.advance(run["id"]))
    return h, repo, run["id"], seen


def test_the_routed_model_is_the_one_that_answers(app, monkeypatch):
    """Not just wired up — used, and recorded per step."""
    h, repo, run_id, seen = _routed_run(app, monkeypatch)

    assert seen, "这次运行没有任何模型调用，测试就没在测路由"
    rows = [r for r in repo.metric_rows(run_id) if r["model"]]
    assert rows, "步骤没有记录是哪个模型回答的"
    for row in rows:
        role = STATE_AGENT.get(row["state"])
        assert row["model"] == h.agents[role].llm.model

    models = {row["model"] for row in rows}
    assert "strong" in models, "被路由到强模型的那个角色一次都没被问到"
    assert "base" in models


def test_a_step_is_priced_at_the_model_that_answered(app, monkeypatch):
    """The one thing `_step_metrics` exists to get right."""
    _h, repo, run_id, _seen = _routed_run(app, monkeypatch)

    priced = [r for r in repo.metric_rows(run_id) if r["cost_usd"] is not None]
    assert priced
    for row in priced:
        assert row["cost_usd"] == pytest.approx(
            row["input_tokens"] / 1e6 * RATES[row["model"]]), \
            f"{row['state']} 的单价与回答它的模型 {row['model']} 不符"

    assert {r["model"] for r in priced} > {"base"}, \
        "没有任何被路由的模型出现在计价里，这个测试就是空的"


def test_the_report_breaks_the_cost_down_by_model(app, monkeypatch):
    _h, repo, run_id, _seen = _routed_run(app, monkeypatch)

    report = run_metrics(repo, run_id)
    assert {"base", "strong"} <= set(report["by_model"])

    text = render(report, model="base", pricing={})
    assert "按模型：" in text and "strong" in text and "cheap" in text


def test_a_single_model_report_does_not_gain_a_breakdown_line(app):
    """With one model the breakdown is the whole report repeated."""
    h, repo = app["harness"], app["repo"]
    run = h.create_run(FEATURE, kind="feature")
    asyncio.run(h.advance(run["id"]))

    text = render(run_metrics(repo, run["id"]), model="mock", pricing={})

    assert "按模型：" not in text


def test_the_missing_price_message_names_the_models_that_ran(app, monkeypatch):
    """Not the configured default, which may not appear in the run at all."""
    cfg = app["cfg"]
    _h, repo, run_id, _seen = _routed_run(app, monkeypatch)
    cfg.pricing = {}

    text = render(run_metrics(repo, run_id), model="base", pricing={})

    assert "strong" in text.split("成本")[1]
