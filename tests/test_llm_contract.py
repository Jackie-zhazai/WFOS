"""Real-provider contract tests (no network).

Verifies the model-agnostic contract that the mock hides: payload
serializability, pydantic->JSON-schema conversion, tool-message translation per
provider, the agent's prompt context and tool loop, env-over-config precedence,
and the child-flow resume gate.
"""
from __future__ import annotations

import asyncio
import json
import re

import pytest

from wfos.agents.base import AgentError
from wfos.agents.investigator import InvestigatorAgent
from wfos.config import load_config
from wfos.llm.anthropic import _to_anthropic_messages, parse_anthropic_response
from wfos.llm.base import schema_to_json, tool_name_map
from wfos.llm.gemini import _to_gemini_contents
from wfos.llm.openai_compat import _to_openai_messages, build_payload, parse_response
from wfos.llm.scripted import ScriptedAdapter
from wfos.models import InvestigatorOutput


class _RecordingScripted(ScriptedAdapter):
    """Records every messages list handed to complete() so the test can assert
    that the call-side tool id equals the result-side tool id."""
    def __init__(self):
        super().__init__()
        self.seen: list[list[dict]] = []

    async def complete(self, *, messages=None, schema=None, tools=None,
                       temperature=None, max_tokens=None, ctx=None):
        self.seen.append(list(messages or []))
        return await super().complete(messages=messages, schema=schema, tools=tools,
                                      temperature=temperature, max_tokens=max_tokens,
                                      ctx=ctx)


# ------------------------------------------------------------- schema -> JSON
def test_schema_to_json_converts_pydantic_class():
    d = schema_to_json(InvestigatorOutput)
    assert isinstance(d, dict)
    assert d["title"] == "InvestigatorOutput"
    assert "next_step" in d["properties"]
    json.dumps(d)                       # fully serializable


def test_schema_to_json_passes_dicts_through():
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    assert schema_to_json(schema) is schema
    assert schema_to_json(None) == {}


# ------------------------------------------------------- OpenAI payload/translate
def test_openai_payload_serializes_with_pydantic_schema():
    payload = build_payload(messages=[{"role": "system", "content": "hi"}],
                            schema=InvestigatorOutput, tools=[],
                            temperature=0.1, max_tokens=512, model="m")
    json.dumps(payload)                 # must not raise
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["schema"]["title"] == "InvestigatorOutput"


def test_openai_payload_tools_serializable():
    tools = [{"name": "workspace.read", "description": "read",
              "inputSchema": {"type": "object",
                              "properties": {"path": {"type": "string"}}}}]
    payload = build_payload(messages=[{"role": "user", "content": "go"}], schema=None,
                            tools=tools, temperature=0.0, max_tokens=100, model="m")
    json.dumps(payload)
    # The tool declaration that reaches the wire must be one the provider accepts:
    # OpenAI's function names match `^[a-zA-Z0-9_-]+$`, and wfos's internal
    # `namespace.verb` names have a dot in them. Sending the internal name got
    # every request refused with a 400 — never caught, because the loopback fake
    # server accepts any payload it is handed.
    wire = payload["tools"][0]["function"]["name"]
    assert re.fullmatch(r"[a-zA-Z0-9_-]+", wire), wire
    assert wire == "workspace_read"
    assert tool_name_map(tools).to_internal(wire) == "workspace.read"


def test_anthropic_tool_declarations_use_provider_safe_names(app):
    """Same constraint on the Messages API: `^[a-zA-Z0-9_-]{1,64}$`."""
    from wfos.config import LLMConfig
    from wfos.llm.anthropic import AnthropicAdapter

    tools = [{"name": "workspace.list_files", "description": "list",
              "inputSchema": {"type": "object"}}]
    names = tool_name_map(tools)
    assert names.to_provider_tools(tools)[0]["name"] == "workspace_list_files"
    adapter = AnthropicAdapter(LLMConfig(provider="anthropic", allow_missing_key=True))
    assert adapter.name == "anthropic"          # constructed, not called


def test_the_name_map_is_exact_when_two_names_sanitize_alike():
    """`.` and `_` both become `_`, so `a.b` and `a_b` collide on the wire. The
    map must keep them apart in both directions — a name that maps back to the
    wrong tool is worse than an ugly one."""
    names = tool_name_map([{"name": "a.b"}, {"name": "a_b"}])
    assert names.to_provider("a.b") != names.to_provider("a_b")
    for internal in ("a.b", "a_b"):
        assert names.to_internal(names.to_provider(internal)) == internal


def test_a_name_nobody_declared_is_not_guessed_at():
    """It is returned unchanged and refused by the loop, which is the honest
    outcome for a tool that was never offered."""
    names = tool_name_map([{"name": "workspace.read"}])
    assert names.to_internal("workspace.invented") == "workspace.invented"


def test_an_unknown_structured_output_mode_is_refused_at_construction():
    """Falling through every branch would send no response_format at all — a
    silent wrong answer, which is what the check exists to prevent."""
    from wfos.config import LLMConfig
    from wfos.llm.openai_compat import OpenAICompatAdapter
    with pytest.raises(ValueError, match="structured_output"):
        OpenAICompatAdapter(LLMConfig(provider="openai", allow_missing_key=True,
                                      structured_output="json_schema_strict"))


def test_the_prompt_names_the_states_this_state_may_suggest(app):
    """A model asked to suggest a next state has to be told the vocabulary.

    A real provider invented `req_analysis`; the Harness correctly refused the
    illegal transition and the run failed on its first state. Refusing was right —
    asking the model to guess at the set of legal names was not.
    """
    h, repo = app["harness"], app["repo"]
    run = h.create_run("新增一个模块", kind="feature")
    ctx = h._build_ctx(repo.get_run(run["id"]))

    assert ctx["allowed_states"] == ["failed", "project_check"]
    prompt = h.agents["investigator"].build_prompt(ctx)
    assert "本状态允许建议的下一状态" in prompt
    assert "project_check" in prompt


def test_the_allowed_states_come_from_the_machine_and_follow_the_state(app):
    """The list is the machine's own, not a second copy that can drift from it."""
    from wfos.harness import statemachine as sm

    h, repo = app["harness"], app["repo"]
    run = h.create_run("新增一个模块", kind="feature")
    repo.update_run(run["id"], state="build_test")

    ctx = h._build_ctx(repo.get_run(run["id"]))
    assert ctx["allowed_states"] == ["failed", "implement", "regression_verify"]
    assert ctx["allowed_states"] == sorted(sm.allowed_transitions("feature", "build_test"))


def test_openai_message_translation_for_tool_loop():
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"name": "echo", "arguments": {"text": "hi"}}]},
        {"role": "tool", "tool_call_id": "call_0_0", "name": "echo",
         "content": '{"echo": "hi"}'},
    ]
    out = _to_openai_messages(messages)
    json.dumps(out)
    asst = out[2]
    assert asst["role"] == "assistant"
    assert asst["tool_calls"][0]["function"]["name"] == "echo"
    assert json.loads(asst["tool_calls"][0]["function"]["arguments"]) == {"text": "hi"}
    assert asst["tool_calls"][0]["id"].startswith("call_")
    assert out[3]["role"] == "tool"
    assert out[3]["tool_call_id"] == "call_0_0"


def test_openai_converter_preserves_existing_ids():
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_preserve_1", "name": "echo", "arguments": {"text": "hi"}}]},
        {"role": "tool", "tool_call_id": "call_preserve_1", "name": "echo",
         "content": "{}"},
    ]
    out = _to_openai_messages(messages)
    assert out[0]["tool_calls"][0]["id"] == "call_preserve_1"
    assert out[1]["tool_call_id"] == "call_preserve_1"


def test_openai_parse_preserves_provider_tool_call_id():
    data = {"choices": [{"message": {"role": "assistant", "tool_calls": [
        {"id": "call_xyz_123", "type": "function",
         "function": {"name": "echo", "arguments": '{"text": "hi"}'}}]}}]}
    res = parse_response(data)
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].id == "call_xyz_123"
    assert res.tool_calls[0].name == "echo"
    assert res.tool_calls[0].arguments == {"text": "hi"}


# -------------------------------------------------------- Anthropic translation
def test_anthropic_parse_preserves_provider_tool_use_id():
    data = {"content": [
        {"type": "tool_use", "id": "toolu_01ABC", "name": "echo",
         "input": {"text": "hi"}},
        {"type": "text", "text": '{"findings": [], "next_step": {"suggested_state": "", "reason": ""}}'},
    ]}
    res = parse_anthropic_response(data)
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].id == "toolu_01ABC"
    assert res.tool_calls[0].arguments == {"text": "hi"}
    assert res.output["next_step"]["suggested_state"] == ""


def test_anthropic_messages_translate_to_content_blocks():
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"name": "echo", "arguments": {"text": "hi"}}]},
        {"role": "tool", "tool_call_id": "toolu_0", "name": "echo",
         "content": '{"echo": "hi"}'},
    ]
    out = _to_anthropic_messages(messages)
    json.dumps(out)
    assert out[0]["content"] == "do it"           # system message dropped
    assert out[1]["content"][0]["type"] == "tool_use"
    assert out[1]["content"][0]["input"] == {"text": "hi"}
    assert out[2]["content"][0]["type"] == "tool_result"
    assert out[2]["content"][0]["tool_use_id"] == "toolu_0"


def test_anthropic_converter_preserves_existing_ids():
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "toolu_keep_1", "name": "echo", "arguments": {"text": "hi"}}]},
        {"role": "tool", "tool_call_id": "toolu_keep_1", "name": "echo", "content": "{}"},
    ]
    out = _to_anthropic_messages(messages)
    assert out[0]["content"][0]["type"] == "tool_use"
    assert out[0]["content"][0]["id"] == "toolu_keep_1"
    assert out[1]["content"][0]["type"] == "tool_result"
    assert out[1]["content"][0]["tool_use_id"] == "toolu_keep_1"


# --------------------------------------------------------- Gemini translation
def test_gemini_contents_translate_function_calls():
    messages = [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"name": "echo", "arguments": {"text": "hi"}}]},
        {"role": "tool", "tool_call_id": "x", "name": "echo", "content": '{"echo": "hi"}'},
    ]
    out = _to_gemini_contents(messages)
    json.dumps(out)
    assert out[1]["parts"][0]["functionCall"]["name"] == "echo"
    assert out[2]["parts"][0]["functionResponse"]["name"] == "echo"


# ----------------------------------------------------------- agent prompt/ctx
def test_agent_prompt_contains_task_and_tools(app):
    gw = app["gateway"]
    agent = InvestigatorAgent(ScriptedAdapter(), gw, app["sandbox"])
    ctx = {"run": {"id": "r9", "kind": "feature", "title": "T",
                   "description": "新增用户模块，实现 feature_user()"},
           "state": "req_capture", "plan": {}, "prior": {}, "evidence": [],
           "gateway": gw}
    tools = agent._tool_specs()
    names = {t["name"] for t in tools}
    # declared read-only tools are exposed...
    assert {"workspace.list_files", "workspace.search", "workspace.read",
            "wiki.search", "echo"} <= names
    # ...and write tools are NOT (investigator is read-only)
    assert "workspace.write" not in names
    assert "workspace.delete" not in names
    prompt = agent.build_prompt(ctx, tools=tools)
    assert "新增用户模块" in prompt              # the task is in the prompt
    assert "workspace.read" in prompt           # the tools are in the prompt
    assert "InvestigatorOutput" in prompt       # the output schema is in the prompt


def test_agent_tool_loop_executes_tools_then_returns_output(app):
    gw = app["gateway"]
    llm = ScriptedAdapter()
    llm.set_script([
        {"tool_call": {"name": "echo", "arguments": {"text": "hi"}}},
        {"output": {"findings": ["found"], "evidence": [],
                    "project_context": {},
                    "next_step": {"suggested_state": "project_check",
                                  "reason": "done"}}},
    ])
    agent = InvestigatorAgent(llm, gw, app["sandbox"])
    ctx = {"run": {"id": "r10", "kind": "feature", "title": "T",
                   "description": "调查模块"},
           "state": "req_capture", "plan": {}, "prior": {}, "evidence": [],
           "gateway": gw}
    out = asyncio.run(agent.run(ctx))
    assert out["findings"] == ["found"]
    assert out["next_step"]["suggested_state"] == "project_check"
    # the native tool call was actually executed through the gateway and audited
    calls = app["repo"].tool_calls("r10")
    assert any(c["tool"] == "echo" for c in calls)


def test_the_refusal_carries_the_tools_that_do_exist(app):
    """The refusal has to be *actionable*, and it has to be a tool result.

    A result must answer a `tool_call` in the preceding assistant message: a real
    provider rejects the whole conversation otherwise. That ordering was
    incidental while an undeclared tool aborted the state; now that the refusal
    is answered, it is the difference between a retry and a 400.
    """
    gw = app["gateway"]
    llm = _RecordingScripted()
    llm.set_script([
        {"tool_call": {"id": "call_x", "name": "workspace.foo",
                       "arguments": {"path": "a"}}},
        {"output": {"findings": [], "evidence": [], "project_context": {},
                    "next_step": {"suggested_state": "project_check", "reason": ""}}},
    ])
    agent = InvestigatorAgent(llm, gw, app["sandbox"])
    ctx = {"run": {"id": "r14"}, "state": "req_capture", "plan": {},
           "prior": {}, "evidence": [], "gateway": gw}

    asyncio.run(agent.run(ctx))

    second = llm.seen[1]
    roles = [m["role"] for m in second]
    assert roles.index("assistant") < roles.index("tool"), "tool result 出现在 assistant 轮之前"
    assert second[roles.index("assistant")]["tool_calls"][0]["id"] == "call_x"
    assert second[roles.index("tool")]["tool_call_id"] == "call_x"

    payload = json.loads(second[roles.index("tool")]["content"])
    assert payload["ok"] is False
    assert payload["error_code"] == "unknown_tool"
    assert "workspace.foo" in payload["message"]
    # the actionable half — what the model may use instead
    assert "workspace.read" in payload["available_tools"]
    assert "workspace.foo" not in payload["available_tools"]


def test_the_refusal_lists_exactly_the_tools_the_prompt_advertised(app):
    """The list handed back and the list in the prompt have to be the same set.

    Two lists that drift apart would be worse than one: the model would be
    pointed at a tool it cannot use, on the one turn where it is trying to
    correct itself.
    """
    gw = app["gateway"]
    llm = _RecordingScripted()
    llm.set_script([
        {"tool_call": {"id": "call_y", "name": "workspace.foo", "arguments": {}}},
        {"output": {"findings": [], "evidence": [], "project_context": {},
                    "next_step": {"suggested_state": "", "reason": ""}}},
    ])
    agent = InvestigatorAgent(llm, gw, app["sandbox"])
    ctx = {"run": {"id": "r15"}, "state": "req_capture", "plan": {},
           "prior": {}, "evidence": [], "gateway": gw}

    asyncio.run(agent.run(ctx))

    system = llm.seen[0][0]["content"]
    listed = json.loads([m for m in llm.seen[1] if m["role"] == "tool"][0]["content"])
    assert listed["available_tools"], "回灌里没有任何可用工具"
    for name in listed["available_tools"]:
        assert f"`{name}`" in system, f"{name} 在回灌里，但 prompt 没有宣告它"
    # And the other direction: nothing the prompt offers is missing from the list.
    for name in agent.allowed_tools:
        if gw.spec(name) is not None:
            assert name in listed["available_tools"]


def test_an_undeclared_tool_is_refused_and_the_model_gets_to_correct_it(app):
    """An unknown tool name is a slip, not a dead end.

    The refusal goes back as a structured tool result — carrying the tools that
    *do* exist, which is the part a model needs to correct itself — so the state
    survives a mistake that used to end it. What must not change either way is
    that the refusal is audited and the call never reaches the filesystem.
    """
    gw, repo = app["gateway"], app["repo"]
    llm = ScriptedAdapter()
    llm.set_script([
        {"tool_call": {"name": "workspace.write",
                       "arguments": {"path": "evil.py", "content": "x=1\n"}}},
        {"output": {"findings": [], "evidence": [], "project_context": {},
                    "next_step": {"suggested_state": "", "reason": ""}}},
    ])
    agent = InvestigatorAgent(llm, gw, app["sandbox"])
    ctx = {"run": {"id": "r11", "kind": "feature", "title": "T",
                   "description": "调查"}, "state": "req_capture",
           "plan": {}, "prior": {}, "evidence": [], "gateway": gw}

    out = asyncio.run(agent.run(ctx))

    assert out["next_step"]["suggested_state"] == ""     # the run continued
    assert not (app["sandbox"] / "evil.py").exists(), "越权写入落盘了"
    row = repo.tool_calls("r11")[-1]
    assert (row["status"], row["error_code"]) == ("rejected", "unknown_tool")


def test_the_undeclared_tool_retry_budget_is_finite(app):
    """Past the budget the state is abandoned, exactly as before it could retry."""
    gw, repo = app["gateway"], app["repo"]
    llm = ScriptedAdapter()
    llm.set_script([{"tool_call": {"name": "workspace.nope", "arguments": {}}}] * 5)
    agent = InvestigatorAgent(llm, gw, app["sandbox"], max_tool_retries=2)
    ctx = {"run": {"id": "r12"}, "state": "req_capture", "plan": {},
           "prior": {}, "evidence": [], "gateway": gw}

    with pytest.raises(AgentError) as exc:
        asyncio.run(agent.run(ctx))

    assert "未声明的工具" in str(exc.value)
    # Every refusal is audited, not just the one that ended the state: the call
    # *was* refused each time, and the retry budget deserves to be visible in the
    # record. Auditing only the last would make a three-attempt slip look like one.
    refused = [r for r in repo.tool_calls("r12") if r["error_code"] == "unknown_tool"]
    assert len(refused) == 3, f"2 次重试 + 1 次放弃 = 3 条审计，实际 {len(refused)}"


def test_a_zero_budget_aborts_on_the_first_undeclared_call(app):
    """`0` disables the retry, which is the behaviour this loop had before it
    could retry at all — kept reachable so the old contract stays testable."""
    gw, repo = app["gateway"], app["repo"]
    llm = ScriptedAdapter()
    llm.set_script([{"tool_call": {"name": "workspace.nope", "arguments": {}}}] * 3)
    agent = InvestigatorAgent(llm, gw, app["sandbox"], max_tool_retries=0)
    ctx = {"run": {"id": "r13"}, "state": "req_capture", "plan": {},
           "prior": {}, "evidence": [], "gateway": gw}

    with pytest.raises(AgentError):
        asyncio.run(agent.run(ctx))

    refused = [r for r in repo.tool_calls("r13") if r["error_code"] == "unknown_tool"]
    assert len(refused) == 1


def test_tool_call_id_round_trips_unchanged(app):
    """The id the provider issued must survive: assistant tool call -> gateway
    execution -> tool result. The converter must NOT mint a different id."""
    gw = app["gateway"]
    llm = _RecordingScripted()
    llm.set_script([
        {"tool_call": {"id": "call_abc_999", "name": "echo", "arguments": {"text": "hi"}}},
        {"output": {"findings": ["done"], "evidence": [], "project_context": {},
                    "next_step": {"suggested_state": "project_check", "reason": "ok"}}},
    ])
    agent = InvestigatorAgent(llm, gw, app["sandbox"])
    ctx = {"run": {"id": "r12", "kind": "feature", "title": "T",
                   "description": "调查"}, "state": "req_capture",
           "plan": {}, "prior": {}, "evidence": [], "gateway": gw}
    out = asyncio.run(agent.run(ctx))
    assert out["findings"] == ["done"]
    # round 1's messages (after the tool was executed) must carry the SAME id
    # on the assistant tool call and on the tool result.
    asst = next(m for m in llm.seen[1] if m.get("role") == "assistant")
    tool = next(m for m in llm.seen[1] if m.get("role") == "tool")
    assert asst["tool_calls"][0]["id"] == "call_abc_999"
    assert tool["tool_call_id"] == "call_abc_999"
    assert asst["tool_calls"][0]["id"] == tool["tool_call_id"]


# ------------------------------------------------------------- env precedence
def test_config_env_overrides_toml(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "wfos.toml").write_text(
        '[llm]\nprovider = "anthropic"\nmodel = "claude-3"\n', encoding="utf-8")
    monkeypatch.setenv("WFOS_DATA", str(data))
    monkeypatch.setenv("WFOS_PROVIDER", "ollama")
    monkeypatch.setenv("WFOS_MODEL", "qwen2.5")
    cfg = load_config()
    assert cfg.llm.provider == "ollama"          # env wins over the file
    assert cfg.llm.model == "qwen2.5"


def test_the_structured_output_mode_can_be_set_from_the_environment(tmp_path, monkeypatch):
    """It is provider-dependent, so it has to move with the provider.

    DeepSeek rejects `json_schema`, and the endpoint is chosen by an environment
    variable — leaving this one file-only made it the single LLM setting that
    could not be set the same way as the thing it depends on.
    """
    data = tmp_path / "data"
    data.mkdir()
    (data / "wfos.toml").write_text('[llm]\nstructured_output = "json_object"\n',
                                    encoding="utf-8")
    monkeypatch.setenv("WFOS_DATA", str(data))
    monkeypatch.setenv("WFOS_STRUCTURED", "none")

    cfg = load_config()
    assert cfg.llm.structured_output == "none"        # env wins over the file


def test_the_env_mode_survives_all_the_way_into_the_request(tmp_path, monkeypatch):
    """Reading it into the config is not the claim; keeping it out of the payload is."""
    from wfos.llm.factory import build_adapter

    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("WFOS_DATA", str(data))
    monkeypatch.setenv("WFOS_PROVIDER", "openai")
    monkeypatch.setenv("WFOS_BASE_URL", "http://127.0.0.1:1/v1")
    monkeypatch.setenv("WFOS_STRUCTURED", "none")
    monkeypatch.setenv("WFOS_API_KEY", "sk-not-used")

    adapter = build_adapter(load_config().llm)
    payload = build_payload(messages=[{"role": "user", "content": "go"}],
                            schema=InvestigatorOutput, tools=None,
                            temperature=0.0, max_tokens=16, model="m",
                            response_format=adapter._response_format)
    assert "response_format" not in payload


def test_config_env_used_without_file(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("WFOS_DATA", str(data))
    monkeypatch.setenv("WFOS_PROVIDER", "vllm")
    monkeypatch.setenv("WFOS_BASE_URL", "http://localhost:8000/v1")
    cfg = load_config()
    assert cfg.llm.provider == "vllm"
    assert cfg.llm.base_url == "http://localhost:8000/v1"


def test_config_file_used_when_no_env(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "wfos.toml").write_text(
        '[llm]\nprovider = "anthropic"\nmodel = "claude-sonnet-5"\n', encoding="utf-8")
    monkeypatch.setenv("WFOS_DATA", str(data))
    monkeypatch.delenv("WFOS_PROVIDER", raising=False)
    monkeypatch.delenv("WFOS_MODEL", raising=False)
    cfg = load_config()
    assert cfg.llm.provider == "anthropic"
    assert cfg.llm.model == "claude-sonnet-5"


# ------------------------------------------------------------- child resume gate
def test_resume_after_child_gates_on_child_status(app):
    h = app["harness"]
    repo = app["repo"]
    parent = h.create_run("新增用户模块", kind="feature")
    child = repo.create_run("bugfix", title="child", description="修复回归 [core]",
                            parent_run_id=parent["id"])
    repo.update_run(parent["id"], state="regression_verify", status="waiting_child",
                    payload={"child_run_id": child["id"],
                             "resume_state": "regression_verify"})

    # child still running -> parent stays waiting_child, child returned
    repo.update_run(child["id"], status="running")
    got = asyncio.run(h.resume_after_child(child["id"]))
    assert got["id"] == child["id"]
    assert repo.get_run(parent["id"])["status"] == "waiting_child"

    # child failed -> parent fails; it never silently resumes
    repo.update_run(child["id"], status="failed")
    got = asyncio.run(h.resume_after_child(child["id"]))
    assert got["id"] == parent["id"]
    assert got["status"] == "failed"            # returned object carries the NEW status
    assert repo.get_run(parent["id"])["status"] == "failed"

    # child completed -> parent resumes
    repo.update_run(parent["id"], state="regression_verify", status="waiting_child",
                    payload={"child_run_id": child["id"],
                             "resume_state": "regression_verify"})
    repo.update_run(child["id"], status="completed")
    got = asyncio.run(h.resume_after_child(child["id"]))
    assert got["id"] == parent["id"]
    assert repo.get_run(parent["id"])["status"] == "completed"


# ------------------------------------------------- loop hygiene: repeat guard
def _investigator_ctx(run_id: str) -> dict:
    return {"run": {"id": run_id, "kind": "feature", "title": "t", "description": "d"},
            "state": "req_capture", "plan": {}, "prior": {}, "evidence": []}


_INVESTIGATOR_OUTPUT = {"findings": [], "evidence": [], "project_context": {},
                        "next_step": {"suggested_state": "project_check", "reason": "done"}}


def _read_script(*paths: str) -> list[dict]:
    script = [{"tool_call": {"name": "workspace.read", "arguments": {"path": p}}}
              for p in paths]
    script.append({"output": _INVESTIGATOR_OUTPUT})
    return script


def test_repeated_identical_tool_call_is_not_re_executed(app):
    """A model re-issuing the exact same (name, arguments) pair is looping, not
    progressing: the repeat is answered with an error, never executed twice."""
    gw, repo = app["gateway"], app["repo"]
    llm = _RecordingScripted()
    llm.set_script(_read_script("app.py", "app.py"))
    agent = InvestigatorAgent(llm, gw, app["sandbox"], repeated_call_threshold=2)

    out = asyncio.run(agent.run(_investigator_ctx("rep-run")))
    assert out["next_step"]["suggested_state"] == "project_check"

    calls = repo.tool_calls("rep-run")
    assert len(calls) == 2
    assert calls[0]["ok"] == 1
    assert calls[1]["ok"] == 0                       # refused, not executed
    # The stable code lives in its own column, so recognising the refusal does not
    # mean matching on prose — which is exactly why the code stopped being a
    # prefix inside the message.
    assert calls[1]["error_code"] == "repeated_identical_call"

    refused = json.loads([m for m in llm.seen[-1] if m["role"] == "tool"][-1]["content"])
    assert refused["ok"] is False
    assert refused["error_code"] == "repeated_identical_call"
    assert refused["message"] and refused["hint"]


# What every refusal the loop produces must carry. Spelled out once so the two
# sites cannot drift apart again — they are the same event (the loop declining to
# execute a call) and must not describe themselves differently.
REFUSAL_VOCABULARY = {"ok", "error_code", "message", "hint"}


def test_the_two_refusals_speak_the_same_vocabulary(app):
    """An undeclared tool and a repeated identical call must not look different.

    A model that has to branch on two shapes is being asked to guess, and the
    stable code has to be a field rather than a prefix buried in prose.
    """
    gw = app["gateway"]

    undeclared_llm = _RecordingScripted()
    undeclared_llm.set_script([
        {"tool_call": {"name": "workspace.foo", "arguments": {}}},
        {"output": {"findings": [], "evidence": [], "project_context": {},
                    "next_step": {"suggested_state": "", "reason": ""}}}])
    asyncio.run(InvestigatorAgent(undeclared_llm, gw, app["sandbox"]).run(
        _investigator_ctx("vocab-a")))

    repeated_llm = _RecordingScripted()
    repeated_llm.set_script(_read_script("app.py", "app.py"))
    asyncio.run(InvestigatorAgent(repeated_llm, gw, app["sandbox"],
                                  repeated_call_threshold=2).run(
        _investigator_ctx("vocab-b")))

    first = json.loads([m for m in undeclared_llm.seen[-1]
                        if m["role"] == "tool"][0]["content"])
    second = json.loads([m for m in repeated_llm.seen[-1]
                         if m["role"] == "tool"][-1]["content"])

    assert set(first) >= REFUSAL_VOCABULARY, f"未声明工具的拒绝缺少字段: {set(first)}"
    assert set(second) >= REFUSAL_VOCABULARY, f"重复调用的拒绝缺少字段: {set(second)}"
    assert first["error_code"] == "unknown_tool"
    assert second["error_code"] == "repeated_identical_call"


def test_different_arguments_are_not_treated_as_a_repeat(app):
    gw, repo = app["gateway"], app["repo"]
    llm = ScriptedAdapter()
    llm.set_script(_read_script("app.py", "check.py"))
    agent = InvestigatorAgent(llm, gw, app["sandbox"], repeated_call_threshold=2)

    asyncio.run(agent.run(_investigator_ctx("diff-run")))
    calls = repo.tool_calls("diff-run")
    assert len(calls) == 2
    assert [c["ok"] for c in calls] == [1, 1]         # both actually ran


def test_repeat_guard_can_be_disabled(app):
    gw, repo = app["gateway"], app["repo"]
    llm = ScriptedAdapter()
    llm.set_script(_read_script("app.py", "app.py"))
    agent = InvestigatorAgent(llm, gw, app["sandbox"], repeated_call_threshold=0)

    asyncio.run(agent.run(_investigator_ctx("off-run")))
    calls = repo.tool_calls("off-run")
    assert [c["ok"] for c in calls] == [1, 1]         # guard off => both run


# --------------------------------------- change attribution reaches the prompt
def test_implement_step_records_observed_changes_not_self_reported_ones(app):
    """The implement step's record of what changed comes from the workspace
    snapshot diff, and the Verifier is told about it — so an implementation that
    never touched disk cannot pass as one that did."""
    h, repo, gw = app["harness"], app["repo"], app["gateway"]
    run = h.create_run("新增一个模块", kind="feature")
    repo.update_run(run["id"], state="implement",
                    payload={"plan": {"summary": "s", "files": [{"path": "new_mod.py"}]}})

    llm = ScriptedAdapter()
    llm.set_script([
        {"tool_call": {"name": "workspace.write",
                       "arguments": {"path": "new_mod.py", "content": "VALUE = 1\n"}}},
        # The model claims a *different* file than the one it actually wrote.
        {"output": {"changes": [{"file": "claimed_elsewhere.py", "action": "create",
                                 "detail": "self-reported"}],
                    "failed": [],
                    "next_step": {"suggested_state": "build_test", "reason": "done"}}},
    ])
    from wfos.agents.implementer import ImplementerAgent
    h.agents["implementer"] = ImplementerAgent(
        llm, gw, app["sandbox"], repeated_call_threshold=h.cfg.harness.repeated_call_threshold)

    run = repo.get_run(run["id"])
    asyncio.run(h._execute_state(run))

    step = repo.get_step(run["id"], "implement")
    output = step["output_json"]
    assert output["actual_changes"] == ["new_mod.py"]          # observed
    assert output["changes"][0]["file"] == "claimed_elsewhere.py"   # still recorded as claimed
    assert (app["sandbox"] / "new_mod.py").exists()

    # The Verifier's prompt carries the observed set, labelled as fact.
    run = repo.get_run(run["id"])
    verifier_prompt = h.agents["verifier"].build_prompt(h._build_ctx(run))
    assert "new_mod.py" in verifier_prompt
    assert "快照差分" in verifier_prompt


def test_implement_step_reports_no_changes_when_nothing_was_written(app):
    """A no-op implementation must not be attributed a change it never made."""
    h, repo, gw = app["harness"], app["repo"], app["gateway"]
    run = h.create_run("新增一个模块", kind="feature")
    repo.update_run(run["id"], state="implement", payload={"plan": {"summary": "s"}})

    llm = ScriptedAdapter()
    llm.set_script([
        {"output": {"changes": [{"file": "app.py", "action": "modify", "detail": "claimed"}],
                    "failed": [],
                    "next_step": {"suggested_state": "build_test", "reason": "done"}}},
    ])
    from wfos.agents.implementer import ImplementerAgent
    h.agents["implementer"] = ImplementerAgent(
        llm, gw, app["sandbox"],
        repeated_call_threshold=h.cfg.harness.repeated_call_threshold)

    asyncio.run(h._execute_state(repo.get_run(run["id"])))
    output = repo.get_step(run["id"], "implement")["output_json"]
    assert output["actual_changes"] == []                      # nothing happened

    prompt = h.agents["verifier"].build_prompt(h._build_ctx(repo.get_run(run["id"])))
    assert "实际变更" not in prompt                            # no false claim injected


# ------------------------------------------- schema violations and agent failure
def test_validate_rejects_a_malformed_payload_instead_of_repairing_it(app):
    """A wrong-typed field must be refused, not silently defaulted: the schemas
    carry decision-bearing fields, and filling those from defaults would turn a
    malformed reply into a confident wrong answer.

    Refusing is `_validate`'s job and is asserted directly on it. The *loop* now
    feeds the refusal back and asks again — a different thing from repairing it:
    the malformed payload never becomes an answer, and what comes back is the
    model's next attempt.
    """
    llm = _RecordingScripted()
    bad = {"next_step": {"suggested_state": "project_check"},
           "findings": {"oops": "not a list"}}
    llm.set_script([{"output": bad},
                    {"output": {"findings": [], "evidence": [], "project_context": {},
                                "next_step": {"suggested_state": "project_check",
                                              "reason": "fixed"}}}])
    agent = InvestigatorAgent(llm, app["gateway"], app["sandbox"])
    ctx = {"run": {"id": "bad-output-run"}, "state": "req_capture",
           "plan": {}, "prior": {}, "evidence": []}

    out = asyncio.run(agent.run(ctx))

    assert out["next_step"]["reason"] == "fixed"      # the second answer
    assert out["findings"] == []                      # not filled from the bad one

    # The hint handed back is the validator's own complaint, not "follow the schema".
    retry_prompt = llm.seen[-1][-1]["content"]
    assert "findings" in retry_prompt and "Schema 校验" in retry_prompt

    # And `_validate` still refuses outright — this is the behaviour that must not
    # soften into "fill the missing pieces in".
    with pytest.raises(AgentError) as exc:
        agent._validate(bad)
    assert "InvestigatorOutput" in str(exc.value) and "findings" in str(exc.value)


def _harness_with_prose_only_model(app, prose_turns: int = 12):
    """A Harness whose model only ever replies with prose — never structured
    output, never a tool call."""
    from wfos.harness.orchestrator import Harness
    llm = ScriptedAdapter()
    llm.set_script([{"text": "只回自然语言，从不产出结构化输出"}
                    for _ in range(prose_turns)])
    return Harness(app["cfg"], app["repo"], app["gateway"], app["wiki"], llm=llm)


def test_an_agent_that_never_answers_fails_the_run_instead_of_crashing(app):
    """AgentError used to escape `advance` as a traceback, stranding the run in
    `running` with no steps and no error to explain it."""
    h, repo = _harness_with_prose_only_model(app), app["repo"]
    run = h.create_run("新增一个用户模块，实现在 app.py 中追加 feature_user() 函数",
                       kind="feature")

    result = asyncio.run(h.advance(run["id"]))          # must not raise

    assert result["status"] == "failed"
    assert "未产出结构化输出" in (result["error"] or "")
    assert repo.get_run(run["id"])["status"] == "failed"   # not left `running`

    step = repo.get_step(run["id"], "req_capture")
    assert step is not None and step["status"] == "failed"
    # The *specific* model-interface failure, not the old catch-all: a prose
    # answer is `output_malformed` (it answered, badly) — a different problem from
    # `output_truncated` (it was cut off), and the two need opposite responses.
    assert step["failure_class"] == "output_malformed"
    assert "未产出结构化输出" in (step["error"] or "")

    last = repo.transitions(run["id"])[-1]
    assert last["to_state"] == "failed"
    assert "未产出可用输出" in (last["reason"] or "")


# --------------------------------------------------- exceptions escaping a state
class _RaisingAdapter(ScriptedAdapter):
    """A provider that fails instead of answering."""

    def __init__(self, exc: BaseException):
        super().__init__()
        self.exc = exc

    async def complete(self, *, messages=None, schema=None, tools=None,
                       temperature=None, max_tokens=None, ctx=None):
        raise self.exc


def _harness_with_failing_provider(app, exc: BaseException):
    from wfos.harness.orchestrator import Harness
    return Harness(app["cfg"], app["repo"], app["gateway"], app["wiki"],
                   llm=_RaisingAdapter(exc))


@pytest.mark.parametrize("exc, expected_class", [
    (ConnectionError("provider unreachable"), "network_error"),
    # A bare `OSError` used to land here as `network_error` too, on the strength
    # of being an `OSError` at all. It is not evidence of anything: nothing in it
    # says the failure came from the wire, and the same shape is what the disk and
    # the permission checks raise. It is classified as what it is — an OS error
    # nobody has narrowed — and only the structured cases below get a real name.
    (OSError("socket closed"), "generic_os_error"),
    (PermissionError("read-only tree"), "permission_error"),
    (OSError(28, "No space left on device"), "disk_error"),
    (TimeoutError("read timed out"), "network_error"),
    (RuntimeError("something we never anticipated"), "unexpected_error"),
    (KeyError("bug"), "unexpected_error"),
])
def test_a_provider_failure_is_recorded_then_re_raised(app, exc, expected_class):
    """The run must not be left looking alive — but the traceback is re-raised,
    so a genuine defect is not tidied away as an ordinary failed run."""
    h, repo = _harness_with_failing_provider(app, exc), app["repo"]
    run = h.create_run("新增一个模块，改 app.py", kind="feature")

    with pytest.raises(type(exc)):
        asyncio.run(h.advance(run["id"]))

    stored = repo.get_run(run["id"])
    assert stored["status"] == "failed"          # not stranded in `running`
    assert str(exc) in (stored["error"] or "")

    step = repo.get_step(run["id"], "req_capture")
    assert step["status"] == "failed"
    assert step["failure_class"] == expected_class
    assert repo.transitions(run["id"])[-1]["to_state"] == "failed"


def test_a_network_error_is_classified_as_such_not_as_a_bug(app):
    import errno

    import httpx

    from wfos.llm.base import is_network_error
    assert is_network_error(httpx.ConnectError("nope"))
    assert is_network_error(TimeoutError("slow"))
    assert not is_network_error(RuntimeError("bug"))
    assert not is_network_error(KeyError("bug"))

    # "Is it from the wire" and "is it the network" are different questions, and
    # an OSError answers neither by itself: the errno does.
    assert not is_network_error(OSError(errno.ENOSPC, "No space left on device"))
    assert not is_network_error(PermissionError(errno.EACCES, "Permission denied"))
    assert not is_network_error(OSError("socket closed"))          # no errno at all
    assert is_network_error(ConnectionResetError(errno.ECONNRESET, "reset"))


def test_the_local_failures_do_not_all_collapse_into_a_network_error():
    """The property the fix exists for, asserted directly.

    Before this, every `OSError` answered `network_error`, so a full disk and an
    unreachable host produced the same `failure_class` — and the two want
    opposite responses. A test that only checked "a network error is a network
    error" would have passed against the buggy version too.
    """
    import errno

    from wfos.llm.base import classify_wire_error

    cases = {
        OSError(errno.ENOSPC, "No space left on device"): "disk_error",
        OSError(errno.EROFS, "Read-only file system"): "disk_error",
        OSError(errno.EIO, "Input/output error"): "disk_error",
        PermissionError(errno.EACCES, "Permission denied"): "permission_error",
        PermissionError(errno.EPERM, "Operation not permitted"): "permission_error",
        ConnectionResetError(errno.ECONNRESET, "reset"): "network_error",
        ConnectionRefusedError(errno.ECONNREFUSED, "refused"): "network_error",
        OSError(errno.ENOENT, "No such file"): "generic_os_error",
        OSError("no errno here"): "generic_os_error",
    }
    got = {type(e).__name__ + str(getattr(e, "errno", "")): classify_wire_error(e)
           for e in cases}

    for exc, expected in cases.items():
        assert classify_wire_error(exc) == expected, f"{exc!r} → {classify_wire_error(exc)}"
    assert len(set(got.values())) >= 4, f"分类退化成了少数几类：{got}"


def test_interrupts_are_not_recorded_as_run_failures(app):
    """Ctrl-C and task cancellation are BaseException, not Exception: they must
    propagate without being written up as a failed run."""
    from wfos.harness.orchestrator import Harness
    h = Harness(app["cfg"], app["repo"], app["gateway"], app["wiki"],
                llm=_RaisingAdapter(KeyboardInterrupt()))
    run = h.create_run("新增一个模块，改 app.py", kind="feature")

    with pytest.raises(KeyboardInterrupt):
        asyncio.run(h.advance(run["id"]))

    assert app["repo"].get_step(run["id"], "req_capture") is None
    assert app["repo"].get_run(run["id"])["status"] == "running"   # untouched
