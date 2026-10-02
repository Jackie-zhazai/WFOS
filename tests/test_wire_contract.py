"""The model-interface contract: why a conversation produced no usable answer.

A third failure axis, alongside task failures (`build_error`, `test_failure`) and
tool refusals (`plan_scope_denied`). It exists because of what the first real
provider run showed: every failure recorded `agent_no_output`, so a **truncated**
answer and a **malformed** one read identically — and they need opposite
responses (give it more room vs. re-ask).

Two halves:

  * reading the provider's stop reason, which is the only signal that separates
    the two (`finish_reason: length` / `stop_reason: max_tokens` / `MAX_TOKENS`);
  * classifying an escaped exception by HTTP status, so a 401 stops looking like
    a flaky provider.
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from wfos.agents.base import AgentError
from wfos.agents.investigator import InvestigatorAgent
from wfos.llm.anthropic import parse_anthropic_response
from wfos.llm.base import (
    WIRE_AUTH_FAILED,
    WIRE_OUTPUT_MALFORMED,
    WIRE_OUTPUT_TRUNCATED,
    WIRE_PROVIDER_ERROR,
    WIRE_PROVIDER_REFUSED,
    WIRE_RATE_LIMITED,
    LLMAdapter,
    classify_wire_error,
)
from wfos.llm.gemini import parse_gemini_response
from wfos.llm.openai_compat import parse_response
from wfos.models import InvestigatorOutput, ModelResult

FEATURE = "新增一个用户模块，在 app.py 中追加 feature_user() 函数"


# ------------------------------------------------------- reading the stop reason
def test_openai_finish_reason_is_read():
    r = parse_response({"choices": [{"message": {"content": "{'a':1}"},
                                     "finish_reason": "length"}]})
    assert r.finish_reason == "length" and r.truncated


def test_anthropic_stop_reason_is_read():
    r = parse_anthropic_response({"content": [{"type": "text", "text": "{}"}],
                                  "stop_reason": "max_tokens"})
    assert r.finish_reason == "max_tokens" and r.truncated


def test_gemini_finish_reason_is_read():
    r = parse_gemini_response({"candidates": [{"content": {"parts": [{"text": "{}"}]},
                                               "finishReason": "MAX_TOKENS"}]})
    assert r.finish_reason == "MAX_TOKENS" and r.truncated


@pytest.mark.parametrize("reason", ["length", "max_tokens", "MAX_TOKENS",
                                    "max_output_tokens", "Length"])
def test_every_spelling_of_truncation_counts(reason):
    assert ModelResult(finish_reason=reason).truncated


@pytest.mark.parametrize("reason", ["stop", "end_turn", "tool_calls", "tool_use",
                                    None, ""])
def test_a_finished_answer_is_not_truncation(reason):
    assert not ModelResult(finish_reason=reason).truncated


# ------------------------------------------------- classifying an escaped error
@pytest.mark.parametrize("status,expected", [
    (401, WIRE_AUTH_FAILED), (403, WIRE_AUTH_FAILED),
    (429, WIRE_RATE_LIMITED),
    (400, WIRE_PROVIDER_REFUSED), (404, WIRE_PROVIDER_REFUSED),
    (422, WIRE_PROVIDER_REFUSED),
    (500, WIRE_PROVIDER_ERROR), (503, WIRE_PROVIDER_ERROR),
])
def test_an_http_status_maps_to_a_specific_wire_code(status, expected):
    req = httpx.Request("POST", "https://provider.invalid/v1")
    err = httpx.HTTPStatusError("x", request=req,
                                response=httpx.Response(status, request=req))
    assert classify_wire_error(err) == expected


def test_transport_trouble_keeps_the_name_it_already_had():
    """`network_error` is read by the repeated-failure escalation, so it keeps its
    name rather than becoming a second word for the same thing.

    The fixture is `ConnectionResetError`, not a bare `OSError("connection reset")`
    as it used to be. The old fixture only worked because every `OSError` answered
    `network_error` — which is the bug: the message said "connection", and the
    classifier was not reading messages, it was reading the type. With the type
    being all it reads, the test has to supply a type that says so.
    """
    import errno

    assert classify_wire_error(ConnectionResetError(errno.ECONNRESET, "reset")) ==         "network_error"
    assert classify_wire_error(OSError(errno.ENOSPC, "No space left")) == "disk_error"
    assert classify_wire_error(PermissionError(errno.EACCES, "denied")) ==         "permission_error"
    assert classify_wire_error(OSError("connection reset")) == "generic_os_error"
    assert classify_wire_error(RuntimeError("a real defect")) == "unexpected_error"


# ------------------------------------------------ the loop names the failure
class _ResultAdapter(LLMAdapter):
    """Replays exact `ModelResult`s, so a test can deliver a wire shape — a
    truncated answer, prose — without a provider."""

    name = "result_replay"

    def __init__(self, results):
        self._results = list(results)

    async def complete(self, *, messages=None, schema=None, tools=None,
                       temperature=None, max_tokens=None, ctx=None) -> ModelResult:
        return self._results.pop(0) if self._results else ModelResult()


def _ctx(run_id="wire-run"):
    return {"run": {"id": run_id, "kind": "feature", "title": "t",
                    "description": "d"},
            "state": "req_capture", "plan": {}, "prior": {}, "evidence": []}


def _agent(app, results, rounds=3):
    """`max_rounds` is a class attribute, not a constructor argument."""
    agent = InvestigatorAgent(_ResultAdapter(results), app["gateway"], app["sandbox"])
    agent.max_rounds = rounds
    return agent


def test_a_truncated_answer_is_reported_as_truncation(app):
    """The distinction that the first real-provider batch could not make."""
    agent = _agent(app, [ModelResult(text='{"findings": ["cut off',
                                     finish_reason="length")] * 3)

    with pytest.raises(AgentError) as exc:
        asyncio.run(agent.run(_ctx()))

    assert exc.value.wire_code == WIRE_OUTPUT_TRUNCATED
    assert "截断" in str(exc.value)


def test_a_prose_answer_is_reported_as_malformed(app):
    agent = _agent(app, [ModelResult(text="我觉得这个项目结构还行。")] * 3)

    with pytest.raises(AgentError) as exc:
        asyncio.run(agent.run(_ctx()))

    assert exc.value.wire_code == WIRE_OUTPUT_MALFORMED


def test_the_truncation_hint_asks_for_room_not_for_a_better_answer(app):
    """Telling a truncated model to "follow the schema" wastes the round: it is a
    budget problem, and the model cannot fix it by trying harder."""
    agent = InvestigatorAgent(_ResultAdapter([]), app["gateway"], app["sandbox"])
    truncated = agent._non_answer_hint(ModelResult(finish_reason="length"))
    prose = agent._non_answer_hint(ModelResult(text="嗯"))

    assert "截断" in truncated and "max_tokens" in truncated
    assert "截断" not in prose and "Schema" in prose


def test_the_run_records_the_specific_wire_class(app):
    """End to end: the class reaches the step record, not just the exception."""
    from wfos.harness.orchestrator import Harness
    from wfos.mcp.client import ToolGateway
    from wfos.mcp.server import WfosMcpServer
    from wfos.wiki.wiki import WikiClient

    repo, cfg = app["repo"], app["cfg"]
    truncated = _ResultAdapter([ModelResult(text='{"a":', finish_reason="length")] * 20)
    fresh = Harness(cfg, repo, ToolGateway(
        WfosMcpServer(cfg, repo, approval_checker=lambda *a: False), repo),
        WikiClient(repo), llm=truncated)
    run_id = fresh.create_run(FEATURE, kind="feature")["id"]

    asyncio.run(fresh.advance(run_id))

    step = repo.get_step(run_id, "req_capture")
    assert step["failure_class"] == WIRE_OUTPUT_TRUNCATED


# ------------------------------------------------- schema refusal is a round, not an end
def test_a_schema_refusal_is_fed_back_and_the_next_answer_is_taken(app):
    """The validator's own complaint is the most actionable thing we can hand back.

    It names the field and the reason. Replacing it with "follow the schema" told
    the model nothing it did not already know, and threw away the one detail that
    would let it fix the answer.
    """
    bad = {"next_step": {"suggested_state": "project_check"},
           "findings": {"oops": "not a list"}}
    good = {"findings": ["app.py: compute 是线性的"], "evidence": [],
            "project_context": {},
            "next_step": {"suggested_state": "project_check", "reason": "ok"}}
    agent = _agent(app, [ModelResult(output=bad), ModelResult(output=good)])

    out = asyncio.run(agent.run(_ctx()))

    assert out["next_step"]["reason"] == "ok"
    assert out["findings"] == ["app.py: compute 是线性的"]   # the second answer


def test_a_schema_refusal_that_keeps_failing_is_still_bounded(app):
    """Bounded by `max_rounds`, like every other unproductive round — a repair
    loop with its own private budget would be a second thing to reason about."""
    bad = {"next_step": {"suggested_state": "project_check"},
           "findings": {"oops": "not a list"}}
    agent = _agent(app, [ModelResult(output=bad)] * 10, rounds=3)

    with pytest.raises(AgentError) as exc:
        asyncio.run(agent.run(_ctx()))

    assert exc.value.wire_code == WIRE_OUTPUT_MALFORMED
    assert "Schema" in str(exc.value)


def test_the_schema_hint_names_the_field_and_shows_what_was_produced(app):
    """Both halves matter: what was wrong, and what the model actually emitted.

    The validator's own message already quotes the offending value, so that part
    of the payload does come back — which is useful, and small. What is *not*
    echoed is the whole payload: only its top-level key names, plus a clip on the
    message in case a nested error runs long.
    """
    bad = {"next_step": {"suggested_state": "project_check"},
           "findings": {"oops": "not a list"},
           "project_context": {"secret": "x" * 5000}}
    agent = _agent(app, [])
    with pytest.raises(AgentError) as exc:
        agent._validate(bad)
    hint = agent._schema_hint(exc.value, bad)

    assert "findings" in hint                 # the field the validator complained about
    assert "next_step" in hint                # the top-level keys it actually produced
    assert "secret" not in hint or "x" * 100 not in hint   # the payload is not echoed
    assert len(hint) < 3000                   # and it is bounded


# ------------------------------------------------------- provider capabilities
class _FakeResponse:
    def __init__(self, status, payload=None, url="http://provider.invalid/v1/chat/completions"):
        self.status_code = status
        self._payload = payload or {}
        self.request = httpx.Request("POST", url)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(f"{self.status_code}", request=self.request,
                                        response=self)


class _FakeClient:
    """Returns prepared responses in order and records the payloads it was given."""

    def __init__(self, results):
        self._results = list(results)
        self.sent: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, headers=None, json=None):
        self.sent.append(json)
        status, payload = self._results.pop(0)
        return _FakeResponse(status, payload)


_OK = (200, {"choices": [{"message": {"content": '{"a": 1}'},
                          "finish_reason": "stop"}]})


def _adapter(cfg, store=None):
    from wfos.llm.openai_compat import OpenAICompatAdapter
    return OpenAICompatAdapter(cfg, capabilities=store)


def _cfg(**over):
    from wfos.config import LLMConfig
    return LLMConfig(provider="openai", base_url="http://provider.invalid/v1",
                     model="m", allow_missing_key=True, **over)


def test_a_capability_store_round_trips_and_survives_corruption(tmp_path):
    from wfos.llm.capabilities import Capabilities, CapabilityStore

    path = tmp_path / "caps.json"
    store = CapabilityStore(path)
    caps = Capabilities(provider="openai", base_url="http://x/v1", model="m")
    caps.structured_output = "none"
    store.observe(caps)

    assert CapabilityStore(path).get(caps.key)["structured_output"] == "none"

    path.write_text("{ this is not json", encoding="utf-8")
    assert CapabilityStore(path).all() == {}     # corrupt reads as "nothing known"


def test_an_unreadable_cache_does_not_take_the_run_down(tmp_path):
    """A cache that cannot be written must not become a failure of the run."""
    from wfos.llm.capabilities import Capabilities, CapabilityStore

    store = CapabilityStore(tmp_path / "nope" / "caps.json")
    path_as_dir = tmp_path / "dir.json"
    path_as_dir.mkdir()
    CapabilityStore(path_as_dir).observe(
        Capabilities(provider="p", base_url="b", model="m"))
    store.observe(Capabilities(provider="p", base_url="b", model="m"))   # must not raise


def test_a_proven_downgrade_is_recorded(app, monkeypatch, tmp_path):
    """The probe asks by *doing*: the same request without `response_format`."""
    from wfos.llm.capabilities import CapabilityStore

    store = CapabilityStore(tmp_path / "caps.json")
    client = _FakeClient([(400, {}), _OK])       # json_schema rejected, plain works
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client)
    adapter = _adapter(_cfg(), store)

    asyncio.run(adapter.complete(messages=[{"role": "user", "content": "go"}],
                                 schema=InvestigatorOutput, tools=None))

    assert "response_format" in client.sent[0]           # the first attempt had it
    assert "response_format" not in client.sent[1]       # the probe did not
    key = "openai|http://provider.invalid/v1|m"
    assert store.get(key)["structured_output"] == "none"
    assert any("json_schema" in note for note in adapter.observations)


def test_an_unproven_downgrade_is_not_recorded(app, monkeypatch, tmp_path):
    """If the retry fails too, the 400 was about something else.

    Recording it as "json_schema is unsupported" would be a wrong conclusion the
    cache would then repeat on every later run — worse than not caching at all.
    """
    from wfos.llm.capabilities import CapabilityStore

    store = CapabilityStore(tmp_path / "caps.json")
    client = _FakeClient([(400, {}), (400, {})])         # both attempts rejected
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client)
    adapter = _adapter(_cfg(), store)

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(adapter.complete(messages=[{"role": "user", "content": "go"}],
                                     schema=InvestigatorOutput, tools=None))

    assert store.all() == {}, "未证实的降级被记进了缓存"
    assert adapter.observations == []


def test_a_recorded_capability_is_reused_instead_of_re_probed(app, monkeypatch, tmp_path):
    """The point of the cache: one discovery, not one per run."""
    from wfos.llm.capabilities import Capabilities, CapabilityStore

    store = CapabilityStore(tmp_path / "caps.json")
    known = Capabilities(provider="openai", base_url="http://provider.invalid/v1",
                         model="m", structured_output="none")
    store.observe(known)

    client = _FakeClient([_OK])                          # no 400 at all this time
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client)
    adapter = _adapter(_cfg(), store)

    asyncio.run(adapter.complete(messages=[{"role": "user", "content": "go"}],
                                 schema=InvestigatorOutput, tools=None))

    assert len(client.sent) == 1                         # one request, no probe
    assert "response_format" not in client.sent[0]
    assert any("沿用已观测到" in note for note in adapter.observations)


def test_the_cli_shows_what_has_been_learned(app):
    """A record an operator cannot read would just move the ignorance."""
    from wfos.cli import cmd_capabilities
    from wfos.llm.capabilities import Capabilities

    h = app["harness"]
    h.capabilities.observe(Capabilities(provider="openai", base_url="http://x/v1",
                                        model="deepseek-flash",
                                        structured_output="none"))
    assert cmd_capabilities(h, None) == 0


# ---------------------------------------------------------- reasoning is measured
def test_reasoning_tokens_are_read_from_the_provider_breakdown():
    """DeepSeek says how much it thought. Guessing from the size of the answer
    when the provider had already told us is the shape of mistake this whole
    module exists to stop."""
    r = parse_response({"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 100,
                                  "completion_tokens_details": {"reasoning_tokens": 80}}})
    assert r.usage.reasoning_tokens == 80
    assert r.usage.output_tokens == 100          # a breakdown, not an addition


def test_no_breakdown_reports_none_rather_than_zero():
    r = parse_response({"choices": [{"message": {"content": "{}"}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
    assert r.usage.reasoning_tokens is None


def _adapter_with(**facts):
    from wfos.llm.capabilities import Capabilities, CapabilityStore
    from wfos.llm.openai_compat import OpenAICompatAdapter
    store = CapabilityStore(None)
    # `structured_output` deliberately left unset: these tests are about the
    # reasoning facts, and setting it would add an unrelated "reusing" note.
    caps = Capabilities(provider="openai", base_url="http://provider.invalid/v1",
                        model="m", **facts)
    store.observe(caps)
    return OpenAICompatAdapter(_cfg(), capabilities=store), store


def test_measured_reasoning_raises_the_output_ceiling():
    adapter, _ = _adapter_with(reasoning=True, reasoning_tokens_observed=6000)
    assert adapter._max_tokens(8192) == 6000 * 3 + 2048
    assert any("提高到" in note for note in adapter.observations)


def test_the_ceiling_is_never_lowered_and_unmeasured_stays_untouched():
    """The operator's number is a floor. A big configured ceiling plus a small
    observed reasoning run must not shrink it, and an unmeasured model gets no
    adjustment at all."""
    generous, _ = _adapter_with(reasoning=True, reasoning_tokens_observed=100)
    assert generous._max_tokens(8192) == 8192
    assert generous.observations == []

    unmeasured, _ = _adapter_with()
    assert unmeasured._max_tokens(4096) == 4096


def test_the_explicit_field_is_used_instead_of_the_size_heuristic(app, monkeypatch, tmp_path):
    """When the provider reports reasoning, nothing is inferred from size."""
    from wfos.llm.capabilities import CapabilityStore

    store = CapabilityStore(tmp_path / "caps.json")
    client = _FakeClient([(200, {
        "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 900,
                  "completion_tokens_details": {"reasoning_tokens": 700}}})])
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client)
    adapter = _adapter(_cfg(), store)

    asyncio.run(adapter.complete(messages=[{"role": "user", "content": "go"}],
                                 schema=None, tools=None))

    key = "openai|http://provider.invalid/v1|m"
    assert store.get(key)["reasoning_tokens_observed"] == 700
    assert any("自报" in note for note in adapter.observations)
    assert not any("推测" in note for note in adapter.observations)


def test_without_a_breakdown_the_guess_says_it_is_a_guess(app, monkeypatch, tmp_path):
    from wfos.llm.capabilities import CapabilityStore

    store = CapabilityStore(tmp_path / "caps.json")
    client = _FakeClient([(200, {
        "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 900}})])
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client)
    adapter = _adapter(_cfg(), store)

    asyncio.run(adapter.complete(messages=[{"role": "user", "content": "go"}],
                                 schema=None, tools=None))

    assert any("推测" in note and "非测量" in note for note in adapter.observations)


def test_reasoning_tokens_reach_the_step_record_and_the_report(app):
    """Measured and consumed is not enough — it has to be visible, or the
    adjustment cannot be reviewed."""
    from wfos.metrics import render, run_metrics
    from wfos.models import ModelResult, ModelUsage

    h, repo = app["harness"], app["repo"]
    metered = _ResultAdapter([
        ModelResult(output={"findings": [], "evidence": [], "project_context": {},
                            "next_step": {"suggested_state": "project_check",
                                          "reason": "ok"}},
                    usage=ModelUsage(input_tokens=100, output_tokens=900,
                                     reasoning_tokens=700))])
    h.agents["investigator"].llm = metered
    run_id = h.create_run(FEATURE, kind="feature")["id"]
    asyncio.run(h.advance(run_id, max_loops=1))

    row = repo.metric_rows(run_id)[0]
    assert row["reasoning_tokens"] == 700
    assert "推理 tokens" in render(run_metrics(repo, run_id))


def test_wire_observations_are_recorded_with_the_prompt(app):
    """A capability discovered mid-run changed what the model was asked, so it
    belongs in the prompt's own record rather than only in a side file."""
    from wfos.models import ModelResult

    h, repo = app["harness"], app["repo"]
    metered = _ResultAdapter([
        ModelResult(output={"findings": [], "evidence": [], "project_context": {},
                            "next_step": {"suggested_state": "project_check",
                                          "reason": "ok"}})])
    metered.observations = ["测试用观测：预算已上调"]
    h.agents["investigator"].llm = metered
    run_id = h.create_run(FEATURE, kind="feature")["id"]
    asyncio.run(h.advance(run_id, max_loops=1))

    meta = (repo.get_step(run_id, "req_capture")["input_json"] or {})["prompt_metadata"]
    assert meta["wire_observations"] == ["测试用观测：预算已上调"]


# -------------------------------------------------------------- prompt on record
def test_a_failed_step_records_the_prompt_too(app):
    """A failure is exactly when a reader needs to know what the model was given.

    Recording it only on the success path would answer the question precisely
    when it does not matter — which is how the last few diagnoses went: the
    failing step had sizes but no content.
    """
    h, repo = app["harness"], app["repo"]
    truncated = _ResultAdapter([ModelResult(text='{"a":', finish_reason="length")] * 20)
    h.agents["investigator"].llm = truncated
    run_id = h.create_run(FEATURE, kind="feature")["id"]

    asyncio.run(h.advance(run_id, max_loops=1))

    step = repo.get_step(run_id, "req_capture")
    assert step["status"] == "failed"
    assert step["failure_class"] == WIRE_OUTPUT_TRUNCATED
    prompt = (step["input_json"] or {}).get("prompt")
    assert prompt and "## 运行上下文" in prompt


def test_the_assembled_prompt_is_persisted_with_the_step(app):
    """dsh's rule — if the model can see it, it is recorded.

    Without it, every diagnosis has to infer what the model was shown from the
    section sizes in `prompt_metadata`, which is how the last few rounds went.
    """
    h, repo = app["harness"], app["repo"]
    run = h.create_run("新增一个模块，改 app.py", kind="feature")
    asyncio.run(h.advance(run["id"], max_loops=1))

    prompt = (repo.get_step(run["id"], "req_capture")["input_json"] or {}).get("prompt")
    assert prompt, "prompt 没有落库"
    assert "## 运行上下文" in prompt
    assert "本状态允许建议的下一状态" in prompt


# ------------------------------------------------------------- early-failure memory
def test_a_run_that_failed_before_planning_is_still_remembered(app):
    """The case that recorded nothing, and the one a later run most needs.

    `_remember_ended_run` used to return early when the run had no planned and no
    observed files — which is exactly a run that died in the first states.
    """
    from wfos.memory import memory_for, render

    h, repo = app["harness"], app["repo"]
    run = h.create_run(FEATURE, kind="feature")
    # No plan, no writes: the shape of a run that fails at `req_capture`.
    repo.update_run(run["id"], status="failed", error="boom")

    h._remember_ended_run(run["id"])
    stored = repo.get_run(run["id"])["payload"]
    assert stored.get("memory_request") == FEATURE
    assert "memory_files" not in stored          # nothing was touched

    found = memory_for(repo, app["cfg"].project_root, [], request=FEATURE)
    assert [item["run_id"] for item in found["items"]] == [run["id"]]
    assert found["items"][0]["anchored"] == "request"
    # And it says its footing is weaker, rather than reading like verified memory.
    assert "未做新鲜度校验" in render(found)


def test_a_different_request_does_not_match_an_early_failure(app):
    from wfos.memory import memory_for

    h, repo = app["harness"], app["repo"]
    run = h.create_run(FEATURE, kind="feature")
    repo.update_run(run["id"], status="failed")
    h._remember_ended_run(run["id"])

    other = memory_for(repo, app["cfg"].project_root, [],
                       request="把数据库的索引重建一下")
    assert other["items"] == []


def test_a_file_anchored_run_is_not_also_matched_by_request(app):
    """One run, one entry: the stronger anchor wins, so a later run does not see
    the same memory twice with two different footings."""
    from wfos.memory import memory_for

    h, repo = app["harness"], app["repo"]
    run = h.create_run(FEATURE, kind="feature")
    asyncio.run(h.advance(run["id"]))            # plans, writes, completes

    found = memory_for(repo, app["cfg"].project_root, ["app.py"], request=FEATURE)
    assert [item["anchored"] for item in found["items"]] == ["files"]


# ------------------------------------------------------------------ the default
def test_the_output_budget_default_leaves_room_for_reasoning():
    """`max_tokens` is a cap, not a reservation, so a generous default costs
    nothing on small answers — and a reasoning model that runs out produces an
    unparseable answer rather than a short one."""
    from wfos.config import LLMConfig, _llm_from_toml, _parse_defaults

    assert LLMConfig().max_tokens >= 8192
    assert _llm_from_toml({}).max_tokens >= 8192
    assert _parse_defaults()["llm"]["max_tokens"] >= 8192


def test_the_prompt_size_accounting_still_reports_what_it_says(app):
    """The recorded prompt and the recorded metadata have to describe the same
    thing, or the sizes become a second, disagreeing account."""
    h, repo = app["harness"], app["repo"]
    run = h.create_run("新增一个模块，改 app.py", kind="feature")
    asyncio.run(h.advance(run["id"], max_loops=1))

    meta = (repo.get_step(run["id"], "req_capture")["input_json"] or {})
    assert len(meta["prompt"]) == meta["prompt_metadata"]["total_chars"]
    assert json.loads(json.dumps(meta["prompt_metadata"]))   # still serializable
