"""OpenAI-compatible chat-completions adapter (OpenAI, vLLM, Ollama /v1, ...).

Structured output uses the native `response_format: json_schema` when requested;
otherwise JSON is extracted from text and validated by the caller. Tool-calling
follows the OpenAI `tools` protocol.
"""
from __future__ import annotations

import json
from typing import Any

import httpx

from ..models import ModelResult, ModelUsage, ToolCall
from .base import (
    LLMAdapter,
    ToolNameMap,
    extract_json_object,
    schema_to_json,
    tool_name_map,
    usage_from_fields,
)
from .capabilities import capabilities_for

# How structured output is asked for. `json_schema` is the strictest and the
# default; `json_object` is the weaker OpenAI mode some providers require; `none`
# sends no response_format at all and relies on the prompt plus JSON extraction.
RESPONSE_FORMAT_JSON_SCHEMA = "json_schema"
RESPONSE_FORMAT_JSON_OBJECT = "json_object"
RESPONSE_FORMAT_NONE = "none"
RESPONSE_FORMAT_MODES = (RESPONSE_FORMAT_JSON_SCHEMA, RESPONSE_FORMAT_JSON_OBJECT,
                         RESPONSE_FORMAT_NONE)

# A reasoning model's thinking comes out of the same budget as its answer, so a
# ceiling that fits a plain model truncates a reasoning one. Sized from what has
# actually been observed rather than a fixed guess — and only ever raised, so the
# operator's number stays a floor.
_REASONING_HEADROOM_FACTOR = 3
_REASONING_ANSWER_ALLOWANCE = 2048


def _parse_usage(data: dict) -> ModelUsage | None:
    """Token counts from an OpenAI-compatible response.

    The cached portion is nested under `prompt_tokens_details` and the reasoning
    breakdown under `completion_tokens_details`; a response with no `usage` block
    yields None rather than a zeroed usage.
    """
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None
    cached = usage.get("prompt_tokens_details")
    cached = cached if isinstance(cached, dict) else {}
    completion = usage.get("completion_tokens_details")
    completion = completion if isinstance(completion, dict) else {}
    return usage_from_fields(input_tokens=usage.get("prompt_tokens"),
                             output_tokens=usage.get("completion_tokens"),
                             cached_tokens=cached.get("cached_tokens"),
                             reasoning_tokens=completion.get("reasoning_tokens"))


def _to_openai_messages(messages: list[dict], *,
                        tool_names: ToolNameMap | None = None) -> list[dict]:
    """Translate the agent's normalized messages to OpenAI's wire format.

    Normalized shape produced by `BaseAgent`: assistant turns carry `tool_calls`
    as [{name, arguments}] and tool results use `{"role": "tool"}`. OpenAI
    requires ids on both sides, so we synthesize `call_N` ids and stringify
    the arguments.
    """
    # A caller with no map gets pass-through names; the adapter always supplies
    # a real one, so this only affects direct callers (tests, tooling).
    tool_names = tool_names if tool_names is not None else ToolNameMap()
    out: list[dict] = []
    for m in messages or []:
        role = m.get("role")
        content = m.get("content", "")
        tool_calls = m.get("tool_calls")
        if role == "assistant" and tool_calls:
            out.append({
                "role": "assistant",
                "content": content or None,
                "tool_calls": [
                    {"id": tc.get("id") or f"call_{j}", "type": "function",
                     "function": {"name": tool_names.to_provider(tc["name"]),
                                  "arguments": json.dumps(tc.get("arguments") or {},
                                                          ensure_ascii=False)}}
                    for j, tc in enumerate(tool_calls)
                ],
            })
        elif role == "tool":
            out.append({"role": "tool",
                        "tool_call_id": m.get("tool_call_id") or "call_0",
                        "content": m.get("content", "")})
        else:
            out.append({"role": role or "user", "content": content or ""})
    return out


def build_payload(*, messages, schema, tools, temperature, max_tokens, model,
                  json_mode: bool = False, tool_names: ToolNameMap | None = None,
                  response_format: str = RESPONSE_FORMAT_JSON_SCHEMA) -> dict:
    """The chat-completions request body.

    `response_format` picks how structured output is asked for, and the default
    is not universally available: DeepSeek answers 400 to `json_schema` ("This
    response_format type is unavailable now") and returns empty content for
    `json_object`, so for it the honest setting is `none` — the prompt already
    demands the JSON and `parse_response` extracts it from the reply.
    """
    if tool_names is None:
        tool_names = tool_name_map(tools)
    payload: dict[str, Any] = {
        "model": model,
        "messages": _to_openai_messages(messages, tool_names=tool_names),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = [
            {"type": "function", "function": {
                "name": t["name"], "description": t.get("description", ""),
                "parameters": t.get("inputSchema", {"type": "object"})}}
            for t in tool_names.to_provider_tools(tools)
        ]
    if schema:
        schema_dict = schema_to_json(schema)
        if json_mode:
            payload["format"] = "json"          # ollama's own switch
        elif response_format == RESPONSE_FORMAT_JSON_SCHEMA:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "structured_output",
                                "schema": schema_dict, "strict": False},
            }
        elif response_format == RESPONSE_FORMAT_JSON_OBJECT:
            payload["response_format"] = {"type": "json_object"}
        # RESPONSE_FORMAT_NONE: send nothing and let the prompt plus
        # `extract_json_object` carry it.
    return payload


def parse_response(data: dict, *, tool_names: ToolNameMap | None = None) -> ModelResult:
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    result = ModelResult()
    result.usage = _parse_usage(data)
    result.finish_reason = choice.get("finish_reason")
    tool_calls = msg.get("tool_calls") or []
    if tool_calls:
        for tc in tool_calls:
            fn = tc.get("function") or {}
            try:
                args = fn.get("arguments") or "{}"
                args = json.loads(args) if isinstance(args, str) else args
            except Exception:
                args = {}
            # The model answers with the provider-safe name; map it back so the
            # rest of the harness (policy, audit, the loop's own declared-tool
            # check) only ever sees the internal one. A name nobody declared
            # passes through unchanged and is refused by the loop.
            name = fn.get("name", "")
            if tool_names is not None:
                name = tool_names.to_internal(name)
            # Preserve the provider-issued id so the agent can echo it back.
            result.tool_calls.append(ToolCall(id=tc.get("id"), name=name,
                                              arguments=args))
    content = msg.get("content")
    if isinstance(content, list):
        content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
    if content:
        result.text = content
        obj = extract_json_object(content)
        if obj is not None:
            result.output = obj
    return result


class OpenAICompatAdapter(LLMAdapter):
    name = "openai_compat"

    def __init__(self, cfg, *, json_mode: bool = False, capabilities=None) -> None:
        self._cfg = cfg
        self.model = getattr(cfg, "model", "") or ""
        self._json_mode = json_mode or cfg.provider == "ollama"
        mode = getattr(cfg, "structured_output", RESPONSE_FORMAT_JSON_SCHEMA)
        if mode not in RESPONSE_FORMAT_MODES:
            # An unrecognised mode would fall through every branch and quietly
            # send no response_format — the same silent-wrong-answer shape the
            # credential check exists to prevent.
            raise ValueError(
                f"未知的 llm.structured_output: {mode!r}；"
                f"可选 {', '.join(RESPONSE_FORMAT_MODES)}"
                f"（可由 wfos.toml 或环境变量 WFOS_STRUCTURED 设置）")
        # `capabilities` is a `CapabilityStore` when the Harness wires one up; None
        # elsewhere (tests, the smoke runner), in which case nothing is remembered
        # across processes but a discovery still holds for this adapter's lifetime.
        self._store = capabilities
        self._caps = capabilities_for(cfg)
        # Initialised before `_remembered_mode()`, which records into it.
        self.observations: list[str] = []
        self._headroom_noted = False
        self._response_format = self._load_remembered() or mode

    def _max_tokens(self, requested: int) -> int:
        """`requested`, raised when measured reasoning needs more room.

        Only ever raised, and only when the provider has actually reported
        reasoning usage: an unmeasured model gets the configured value untouched,
        and a measured one gets enough room for three times the thinking it has
        needed so far plus space for the answer.
        """
        observed = self._caps.reasoning_tokens_observed
        if not observed:
            return requested
        needed = observed * _REASONING_HEADROOM_FACTOR + _REASONING_ANSWER_ALLOWANCE
        if needed <= requested:
            return requested
        if not self._headroom_noted:
            self._headroom_noted = True
            self.observations.append(
                f"输出预算 {requested} 覆盖不住已观测到的推理用量 {observed}，"
                f"自本次起提高到 {needed}"
                f"（观测值×{_REASONING_HEADROOM_FACTOR}+{_REASONING_ANSWER_ALLOWANCE}）")
        return needed

    def _load_remembered(self) -> str | None:
        """Everything already observed about this endpoint, and the mode it implies.

        **All** the remembered facts come back, not just the structured-output
        mode: the reasoning figures are what size the output ceiling, and loading
        one field while leaving the rest in the file makes the profile decorative.

        An observation beats the configured preference — the config states what the
        operator wants, the store states what the endpoint actually accepted — and
        the difference is recorded rather than applied quietly.
        """
        if self._store is None:
            return None
        known = self._store.get(self._caps.key) or {}
        for name in ("tool_names_sanitized", "reasoning",
                     "reasoning_tokens_observed"):
            if known.get(name) is not None:
                setattr(self._caps, name, known[name])
        remembered = known.get("structured_output")
        if remembered not in RESPONSE_FORMAT_MODES:
            return None
        if remembered != self._caps.structured_output:
            self.observations.append(
                f"沿用已观测到的 structured_output={remembered}"
                f"（配置为 {self._caps.structured_output}）")
        return remembered

    def _record(self, **facts) -> None:
        for name, value in facts.items():
            setattr(self._caps, name, value)
        if self._store is not None:
            self._store.observe(self._caps)

    async def complete(self, *, messages=None, schema=None, tools=None,
                       temperature=None, max_tokens=None, ctx=None) -> ModelResult:
        cfg = self._cfg
        base_url = (cfg.base_url or "https://api.openai.com/v1").rstrip("/")
        url = f"{base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        # One map per call, built from the tools about to be declared: the same
        # list every round, so a name the model echoes back maps to the tool it
        # came from.
        names = tool_name_map(tools)
        base_payload = {
            "messages": messages, "schema": schema, "tools": tools,
            "temperature": temperature if temperature is not None else cfg.temperature,
            "max_tokens": self._max_tokens(
                max_tokens if max_tokens is not None else cfg.max_tokens),
            "model": cfg.model, "json_mode": self._json_mode, "tool_names": names}

        async with httpx.AsyncClient(timeout=cfg.timeout) as client:
            try:
                data = await self._post(client, url, headers, base_payload,
                                        self._response_format)
            except httpx.HTTPStatusError as e:
                data = await self._retry_without_schema(client, url, headers,
                                                        base_payload, e)
        result = parse_response(data, tool_names=names)
        self._note_reasoning(result)
        return result

    async def _post(self, client, url, headers, base_payload, mode) -> dict:
        payload = build_payload(**base_payload, response_format=mode)
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()

    async def _retry_without_schema(self, client, url, headers, base_payload, exc):
        """One behavioural probe: does the same request work without response_format?

        Nothing here reads the error message. A 400 has many causes, and guessing
        from prose which one it was is how a wrong conclusion gets recorded — so
        the question is asked by *doing*: if the same request without
        `response_format` succeeds, that field was the problem; if it fails too,
        the original error stands and **nothing is recorded**, because we still do
        not know whose fault it was.
        """
        if (self._response_format != RESPONSE_FORMAT_JSON_SCHEMA
                or exc.response.status_code != 400):
            raise exc
        try:
            data = await self._post(client, url, headers, base_payload,
                                    RESPONSE_FORMAT_NONE)
        except httpx.HTTPStatusError:
            raise exc from None          # unproven: report the original failure
        self._response_format = RESPONSE_FORMAT_NONE
        self._record(structured_output=RESPONSE_FORMAT_NONE)
        self.observations.append(
            "端点拒绝了 response_format=json_schema，同一请求改为不发送该字段后成功；"
            "已记为该 provider 的能力")
        return data

    def _note_reasoning(self, result) -> None:
        """Whether the model thinks before answering, and how much room that takes.

        Read from the provider's own breakdown when it reports one. Only when a
        provider reports nothing does this fall back to inferring from the size of
        the answer — and it says so, because a guess and a measurement should not
        be recorded as the same kind of fact.
        """
        usage = result.usage
        if usage is None:
            return

        if usage.reasoning_tokens is not None:
            first = self._caps.reasoning_tokens_observed is None
            seen = self._caps.reasoning_tokens_observed or 0
            if usage.reasoning_tokens > seen:
                self._record(reasoning=True,
                             reasoning_tokens_observed=usage.reasoning_tokens)
            if first:
                self.observations.append(
                    f"provider 自报 reasoning_tokens={usage.reasoning_tokens}："
                    f"已按观测值给输出预算留出推理余量")
            return

        if self._caps.reasoning is None and (usage.output_tokens or 0) > 512:
            self._record(reasoning=True)
            self.observations.append(
                "provider 未报告 reasoning_tokens；仅凭输出用量偏大推测是推理模型"
                "（推测，非测量）")
