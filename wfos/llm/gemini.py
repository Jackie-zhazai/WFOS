"""Google Gemini generateContent adapter."""
from __future__ import annotations

from typing import Any

import httpx

from ..models import ModelResult, ModelUsage, ToolCall
from .base import LLMAdapter, extract_json_object, usage_from_fields


def _parse_usage(data: dict) -> ModelUsage | None:
    """Token counts from a generateContent response.

    Gemini uses its own `usageMetadata` shape with camelCase keys, siblings of
    `candidates` rather than under a `usage` object.
    """
    usage = data.get("usageMetadata")
    if not isinstance(usage, dict):
        return None
    return usage_from_fields(input_tokens=usage.get("promptTokenCount"),
                             output_tokens=usage.get("candidatesTokenCount"),
                             cached_tokens=usage.get("cachedContentTokenCount"))


def parse_gemini_response(data: dict) -> ModelResult:
    """Parse a generateContent response.

    Gemini functionCall parts have no native id; tool results are matched back
    by function *name*, so `id` is left unset for the agent to synthesize.
    """
    result = ModelResult()
    result.usage = _parse_usage(data)
    # `finishReason` sits on the candidate, and spells truncation `MAX_TOKENS`.
    result.finish_reason = (data.get("candidates") or [{}])[0].get("finishReason")
    parts = (((data.get("candidates") or [{}])[0]).get("content") or {}).get("parts") or []
    for part in parts:
        if "functionCall" in part:
            fc = part["functionCall"]
            result.tool_calls.append(ToolCall(name=fc.get("name", ""), arguments=fc.get("args", {})))
        if "text" in part:
            result.text = part["text"]
            obj = extract_json_object(part["text"])
            if obj is not None:
                result.output = obj
    return result


def _to_gemini_contents(messages: list[dict]) -> list[dict]:
    """Translate the agent's normalized messages to Gemini contents.

    Assistant tool calls become `functionCall` parts (role "model"); tool
    results become `functionResponse` parts (role "user").
    """
    contents: list[dict] = []
    for m in messages or []:
        role = m.get("role")
        content = m.get("content", "")
        tool_calls = m.get("tool_calls")
        if role == "assistant" and tool_calls:
            parts: list[dict] = []
            if content:
                parts.append({"text": content})
            for tc in tool_calls:
                parts.append({"functionCall": {"name": tc["name"],
                                               "args": tc.get("arguments") or {}}})
            contents.append({"role": "model", "parts": parts})
        elif role == "tool":
            contents.append({"role": "user", "parts": [{
                "functionResponse": {"name": m.get("name", ""),
                                     "response": {"result": m.get("content", "")}}}]})
        else:
            contents.append({"role": "model" if role == "assistant" else "user",
                             "parts": [{"text": content or ""}]})
    return contents


class GeminiAdapter(LLMAdapter):
    name = "gemini"

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self.model = getattr(cfg, "model", "") or ""

    async def complete(self, *, messages=None, schema=None, tools=None,
                       temperature=None, max_tokens=None, ctx=None) -> ModelResult:
        cfg = self._cfg
        base_url = (cfg.base_url or "https://generativelanguage.googleapis.com").rstrip("/")
        url = f"{base_url}/v1beta/models/{cfg.model}:generateContent"
        params = {}
        if cfg.api_key:
            params["key"] = cfg.api_key
        contents, system = [], None
        for m in messages:
            if m.get("role") == "system":
                system = m["content"]
        contents = _to_gemini_contents([m for m in messages if m.get("role") != "system"])
        payload: dict[str, Any] = {"contents": contents}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            payload["tools"] = [{"functionDeclarations": [{
                "name": t["name"], "description": t.get("description", ""),
                "parameters": t.get("inputSchema", {"type": "object"}),
            } for t in tools]}]
        gen_cfg: dict[str, Any] = {"temperature": temperature if temperature is not None else cfg.temperature}
        if schema:
            gen_cfg["responseMimeType"] = "application/json"
        if max_tokens:
            gen_cfg["maxOutputTokens"] = max_tokens
        payload["generationConfig"] = gen_cfg
        async with httpx.AsyncClient(timeout=cfg.timeout) as client:
            resp = await client.post(url, params=params, json=payload)
            resp.raise_for_status()
            data = resp.json()
        return parse_gemini_response(data)
