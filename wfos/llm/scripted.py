"""Test-only adapter: plays back a scripted sequence of tool-calls / outputs.

Used by the security and control-flow tests to exercise the agent loop and the
Harness's enforcement (unauthorized tools, path escape, illegal transitions)
with a fully deterministic model.
"""
from __future__ import annotations

from ..models import ModelResult, ToolCall
from .base import LLMAdapter


class ScriptedAdapter(LLMAdapter):
    name = "scripted"
    model = "scripted"

    def __init__(self) -> None:
        self._script: list[dict] = []
        self._idx = 0

    def set_script(self, script: list[dict]) -> None:
        self._script = list(script)
        self._idx = 0

    async def complete(self, *, messages=None, schema=None, tools=None,
                       temperature=None, max_tokens=None, ctx=None) -> ModelResult:
        if self._idx >= len(self._script):
            # Exhausted: emit an empty output so the loop terminates deterministically.
            return ModelResult(output={"summary": "script exhausted",
                                       "next_step": {"suggested_state": "", "reason": "script end"}})
        item = self._script[self._idx]
        self._idx += 1
        if "tool_call" in item:
            tc = item["tool_call"]
            return ModelResult(tool_calls=[
                ToolCall(id=tc.get("id"), name=tc["name"],
                         arguments=tc.get("arguments", {}))])
        if "output" in item:
            return ModelResult(output=item["output"])
        return ModelResult(text=str(item))
