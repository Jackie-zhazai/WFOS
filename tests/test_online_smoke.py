"""Online smoke tests for the real provider wire path.

Two tiers:

1. `test_loopback_openai_compat_tool_round_trip` — hermetic, always runs.
   A fake OpenAI-compatible HTTP server stands in for the model and asserts on
   the *wire*: the adapter's payload carries the tool schema, the first
   completion returns a tool call with a provider id, the agent executes it via
   the gateway, and the fed-back tool result carries the SAME id on the second
   request. This exercises the real `httpx` request path, `parse_response`, the
   BaseAgent tool loop and the id round-trip end to end.

2. `test_live_provider_tool_round_trip` — runs only when a real endpoint is
   configured (WFOS_LIVE_PROVIDER / WFOS_LIVE_BASE_URL / WFOS_LIVE_MODEL /
   WFOS_LIVE_API_KEY, e.g. a local Ollama or a hosted OpenAI-compatible API).
   It drives the same loop against that endpoint and requires a validated
   structured output.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from wfos.agents.investigator import InvestigatorAgent
from wfos.config import LLMConfig
from wfos.llm.factory import build_adapter


class _FakeOpenAIHandler(BaseHTTPRequestHandler):
    """A deterministic OpenAI-compatible chat server.

    Call #1 -> a tool_call (echo) with provider id `call_loop_1`.
    Call #2 -> verifies the fed-back tool_result id matches the assistant
    tool_call id, then returns a structured output.
    """

    def do_POST(self):  # noqa: N802 (http.server API)
        if not self.path.endswith("/chat/completions"):
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        state = self.server.state
        state["calls"] += 1

        if state["calls"] == 1:
            # First call: the tools schema must be on the wire...
            tool_names = [t["function"]["name"] for t in body.get("tools", [])]
            assert "echo" in tool_names, f"echo tool missing from request tools: {tool_names}"
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_loop_1", "type": "function",
                    "function": {"name": "echo",
                                 "arguments": json.dumps({"text": "hi"})}}],
            }
        else:
            # Second call: the fed-back id MUST equal the assistant's id.
            msgs = body.get("messages", [])
            asst = [m for m in msgs if m.get("role") == "assistant" and m.get("tool_calls")]
            tool = [m for m in msgs if m.get("role") == "tool"]
            assert asst and tool, "assistant tool_calls / tool result missing"
            assert asst[0]["tool_calls"][0]["id"] == tool[0]["tool_call_id"], (
                f"id mismatch over the wire: {asst[0]['tool_calls'][0]['id']!r} != "
                f"{tool[0]['tool_call_id']!r}")
            message = {
                "role": "assistant",
                "content": json.dumps({
                    "findings": ["remote model completed the round trip"],
                    "evidence": [],
                    "project_context": {},
                    "next_step": {"suggested_state": "project_check",
                                  "reason": "remote ok"}},
                    ensure_ascii=False),
            }
        self._send(200, {"choices": [{"message": message}]})

    def _send(self, code: int, obj: dict) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):  # keep test output clean
        pass


@pytest.fixture()
def fake_openai_server():
    server = HTTPServer(("127.0.0.1", 0), _FakeOpenAIHandler)
    server.state = {"calls": 0}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()
    thread.join(timeout=5)


def _investigate(llm, app, run_id: str) -> dict:
    gw = app["gateway"]
    agent = InvestigatorAgent(llm, gw, app["sandbox"])
    ctx = {"run": {"id": run_id, "kind": "feature", "title": "T",
                   "description": "调查模块"},
           "state": "req_capture", "plan": {}, "prior": {}, "evidence": [],
           "gateway": gw}
    return asyncio.run(agent.run(ctx))


def test_loopback_openai_compat_tool_round_trip(app, fake_openai_server, monkeypatch):
    """Real OpenAI-compatible adapter over real HTTP: tool call -> feed back ->
    structured output, with the provider id round-tripping unchanged.

    `no_proxy` is set because httpx honours the system proxy by default, which is
    right for reaching a real provider and wrong here: on a machine with a local
    proxy configured, even `http://127.0.0.1:<port>` is routed through it and
    comes back 502. That is what this test saw before the exclusion was added.
    """
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    cfg = LLMConfig(provider="openai", base_url=fake_openai_server,
                    model="loopback", temperature=0.0, timeout=10,
                    # The loopback server authenticates nothing, and this says so
                    # explicitly rather than relying on a provider name.
                    allow_missing_key=True)
    llm = build_adapter(cfg)                     # the real OpenAICompatAdapter
    out = _investigate(llm, app, "r20")
    assert out["findings"] == ["remote model completed the round trip"]
    assert out["next_step"]["suggested_state"] == "project_check"
    # the tool call was really executed through the gateway and audited
    calls = app["repo"].tool_calls("r20")
    assert any(c["tool"] == "echo" and c["ok"] for c in calls)


def test_live_provider_tool_round_trip(app):
    """Real hosted provider smoke: requires an endpoint in the environment.

    Run with, e.g.:
      WFOS_LIVE_PROVIDER=ollama WFOS_LIVE_BASE_URL=http://localhost:11434/v1 \\
      WFOS_LIVE_MODEL=qwen2.5 pytest tests/test_online_smoke.py -q
    """
    base_url = os.environ.get("WFOS_LIVE_BASE_URL")
    if not base_url:
        pytest.skip("未配置 WFOS_LIVE_BASE_URL，跳过真实 provider 在线冒烟")
    provider = os.environ.get("WFOS_LIVE_PROVIDER", "openai")
    model = os.environ.get("WFOS_LIVE_MODEL", "qwen2.5")
    cfg = LLMConfig(provider=provider, base_url=base_url, model=model,
                    api_key_env=("WFOS_LIVE_API_KEY" if os.environ.get("WFOS_LIVE_API_KEY") else None),
                    # Not every provider accepts `json_schema`; DeepSeek answers
                    # 400 to it and empty content to `json_object`, so `none` is
                    # what it needs. Leaving the default here would make this
                    # smoke test fail on a provider the harness otherwise works
                    # with, which is its own kind of false report.
                    structured_output=os.environ.get("WFOS_LIVE_STRUCTURED", "json_schema"),
                    temperature=0.0, timeout=120)
    llm = build_adapter(cfg)
    out = _investigate(llm, app, "r_live")
    # whatever the model did (tool round trips, or a direct structured output),
    # it must have produced a schema-valid result.
    assert isinstance(out["findings"], list)
    assert out["next_step"]["suggested_state"]
