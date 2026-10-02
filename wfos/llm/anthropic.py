"""Anthropic Messages API adapter."""
from __future__ import annotations

from typing import Any

import httpx

from ..models import ModelResult, ModelUsage, ToolCall
from .base import LLMAdapter, ToolNameMap, extract_json_object, tool_name_map, usage_from_fields


def _parse_usage(data: dict) -> ModelUsage | None:
    """Token counts from a Messages API response.

    Anthropic spells the cached portion `cache_read_input_tokens` and does not
    include it in `input_tokens`, so it maps straight onto our `cached_tokens`.
    """
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None
    return usage_from_fields(input_tokens=usage.get("input_tokens"),
                             output_tokens=usage.get("output_tokens"),
                             cached_tokens=usage.get("cache_read_input_tokens"))


def parse_anthropic_response(data: dict, *,
                             tool_names: ToolNameMap | None = None) -> ModelResult:
    """Parse a Messages API response, preserving each tool_use's original id."""
    result = ModelResult()
    result.usage = _parse_usage(data)
    # The Messages API calls it `stop_reason`; `max_tokens` is its spelling of
    # "cut off" (see `models.TRUNCATION_REASONS`).
    result.finish_reason = data.get("stop_reason")
    for block in data.get("content", []):
        if block.get("type") == "tool_use":
            # The model answers with the provider-safe name; map it back so the
            # rest of the harness (policy, audit, the loop's declared-tool check)
            # only ever sees the internal one. A name nobody declared passes
            # through unchanged and is refused by the loop.
            name = block["name"]
            if tool_names is not None:
                name = tool_names.to_internal(name)
            result.tool_calls.append(ToolCall(id=block.get("id"),
                                              name=name,
                                              arguments=block.get("input", {})))
        elif block.get("type") == "text":
            result.text = block["text"]
            obj = extract_json_object(block["text"])
            if obj is not None:
                result.output = obj
    return result


def _to_anthropic_messages(messages: list[dict], *,
                           tool_names: ToolNameMap | None = None) -> list[dict]:
    """Translate the agent's normalized messages to Anthropic content blocks.

    Assistant tool calls become `tool_use` blocks; tool results become
    `user` turns with a `tool_result` block carrying the same `tool_use_id`.
    """
    # A caller with no map gets pass-through names; the adapter always supplies
    # a real one, so this only affects direct callers (tests, tooling).
    tool_names = tool_names if tool_names is not None else ToolNameMap()
    out: list[dict] = []
    for m in messages or []:
        role = m.get("role")
        if role == "system":           # system prompts ride in `payload["system"]`
            continue
        content = m.get("content", "")
        tool_calls = m.get("tool_calls")
        if role == "assistant" and tool_calls:
            blocks: list[dict] = []
            if content:
                blocks.append({"type": "text", "text": content})
            for j, tc in enumerate(tool_calls):
                name = tc["name"]
                if tool_names is not None:
                    name = tool_names.to_provider(name)
                blocks.append({"type": "tool_use", "id": tc.get("id") or f"toolu_{j}",
                               "name": name, "input": tc.get("arguments") or {}})
            out.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            out.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": m.get("tool_call_id") or "toolu_0",
                "content": m.get("content", "")}]})
        else:
            out.append({"role": role if role in ("user", "assistant") else "user",
                        "content": content or ""})
    return out


class AnthropicAdapter(LLMAdapter):
    name = "anthropic"

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self.model = getattr(cfg, "model", "") or ""

    async def complete(self, *, messages=None, schema=None, tools=None,
                       temperature=None, max_tokens=None, ctx=None) -> ModelResult:
        cfg = self._cfg
        base_url = (cfg.base_url or "https://api.anthropic.com").rstrip("/")
        url = f"{base_url}/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": cfg.api_key or "",
            "anthropic-version": "2023-06-01",
        }
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        # Tool names must match `^[a-zA-Z0-9_-]{1,64}$`; wfos's `namespace.verb`
        # names do not. One map per call, built from the tools about to be
        # declared, so a name the model echoes back maps to what it came from.
        names = tool_name_map(tools)
        payload: dict[str, Any] = {
            "model": cfg.model,
            "max_tokens": max_tokens or cfg.max_tokens,
            "temperature": temperature if temperature is not None else cfg.temperature,
            "messages": _to_anthropic_messages(
                [m for m in messages if m.get("role") != "system"],
                tool_names=names),
        }
        if system_parts:
            payload["system"] = "\n".join(system_parts)
        if tools:
            payload["tools"] = [{
                "name": t["name"], "description": t.get("description", ""),
                "input_schema": t.get("inputSchema", {"type": "object"}),
            } for t in names.to_provider_tools(tools)]
        async with httpx.AsyncClient(timeout=cfg.timeout) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        return parse_anthropic_response(data, tool_names=names)
