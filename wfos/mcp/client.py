"""In-process MCP gateway used by the agent runtime and the mock brain.

Owns the client session, sets the server-side principal (role / roots) before
an agent runs, and audits every tool call into the repo. Tool results are
always treated as *data*, never as instructions.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

from mcp.client import Client

from .. import events
from ..models import ToolOutcome
from ..storage.repo import Repo
from .policy import CODE_REPEATED_CALL, CODE_TOOL_FAILED, SOURCE_AGENT, ToolSpec, status_for
from .server import PRINCIPAL_META_KEY, WfosMcpServer


class ToolGateway:
    def __init__(self, server: WfosMcpServer, repo: Repo):
        self._server = server
        self._repo = repo
        self._client: Client | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def __aenter__(self) -> ToolGateway:
        await self._ensure()
        return self

    async def __aexit__(self, *exc) -> None:
        if self._client is not None:
            # Tearing down a client that is already broken must not mask the
            # reason the run is ending.
            with contextlib.suppress(Exception):
                await self._client.__aexit__(*exc)
            self._client = None

    async def _ensure(self) -> None:
        """Lazily open the in-process MCP client session (idempotent).

        The Harness may run inside `asyncio.run()` without an explicit
        `async with gateway`, so the client is created on first use. If a new
        event loop is in charge (e.g. separate `asyncio.run` calls in tests),
        the old transport is torn down and a fresh one opened.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if (self._client is not None and loop is not None and self._loop is loop):
            return
        if self._client is not None:
            with contextlib.suppress(Exception):
                await self._client.__aexit__(None, None, None)
            self._client = None
        # High-level Client(server) builds its own in-process InMemoryTransport
        # around the *same* server object, so a principal set on the server is
        # visible to every tool call.
        self._client = Client(self._server.server)
        await self._client.__aenter__()
        self._loop = loop

    # ---------------------------------------------------------------- principal
    def set_principal(self, *, role: str, agent: str, run_id: str,
                      read_roots: list[str], write_roots: list[str],
                      allowed_write_paths: list[str] | None = None,
                      session_id: str = "") -> None:
        """Fix the calling principal on the server.

        `allowed_write_paths` narrows file writes to the approved plan's file
        set; None means the plan declared none, so no narrowing applies.

        `session_id` is carried for the same reason `run_id` is — the gateway
        emits a trace event per tool call, and an event without a session cannot
        be grouped with the other runs of its invocation.
        """
        self._server.principal = {
            "role": role, "agent": agent, "run_id": run_id,
            "read_roots": read_roots, "write_roots": write_roots,
            "allowed_write_paths": allowed_write_paths,
            "session_id": session_id,
        }

    def _trace(self, principal: dict[str, Any], run_id: str, tool: str,
               ok: bool, error_code: str | None) -> int:
        """Trace one tool call and return its event id.

        Emitted here because this is where a call is audited, and audited is what
        a call is: one place records that it happened and one id joins the two.
        `actor=model` because a tool call is the model's action, not the
        harness's — a trace that attributed both to `harness` could not tell a
        decision the workflow made from one the model made.
        """
        return events.record(
            self._repo, events.TOOL_CALLED, run_id=run_id,
            session_id=principal.get("session_id") or "",
            actor=events.ACTOR_MODEL,
            payload={"tool": tool, "ok": ok, "error_code": error_code or ""})

    def bound(self, principal: dict[str, Any]) -> BoundGateway:
        """This gateway with `principal` attached to every call it makes."""
        return BoundGateway(self, principal)

    @property
    def server(self) -> WfosMcpServer:
        """The underlying MCP server (needed to serve an external stdio client)."""
        return self._server

    def spec(self, name: str) -> ToolSpec | None:
        return self._server.specs.get(name)

    async def list_tools(self) -> list[ToolSpec]:
        return list(self._server.specs.values())

    def record_rejected(self, tool: str, args: dict[str, Any], error: str, *,
                        error_code: str = CODE_REPEATED_CALL,
                        agent: str | None = None, run_id: str | None = None,
                        principal: dict[str, Any] | None = None) -> None:
        """Audit a call the agent loop refused to execute (repeat guard, or a
        tool the model was never given).

        Nothing reached the tool layer, but the refusal is a real event about
        the run and belongs in the same audit trail as executed calls.
        """
        principal = principal or self._server.principal
        run_id = run_id or principal.get("run_id") or ""
        self._repo.log_tool_call(
            run_id, agent or principal.get("agent") or "?", tool, args, False, error,
            status=status_for(error_code), error_code=error_code, source=SOURCE_AGENT,
            event_id=self._trace(principal, run_id, tool, False, error_code))

    # -------------------------------------------------------------------- call
    async def call(self, tool: str, args: dict[str, Any], *,
                   agent: str | None = None, run_id: str | None = None,
                   principal: dict[str, Any] | None = None) -> ToolOutcome:
        """Execute one tool call under `principal` (falling back to the server's).

        `principal` travels **with the request** rather than being read off the
        server, because the server's is one slot shared by every run in the
        process. Passing it means a call is authorised and audited as the run
        that made it, even when another run's state set the slot in between —
        which is what `delegate` does on purpose, by advancing a whole child run
        inside the parent's tool call.
        """
        principal = principal or self._server.principal
        agent = agent or principal.get("agent") or "?"
        run_id = run_id or principal.get("run_id") or ""
        spec = self._server.specs.get(tool)
        try:
            await self._ensure()
            result = await self._client.call_tool(
                tool, args, meta={PRINCIPAL_META_KEY: principal})
            text = _extract_text(result)
            structured: dict | None = None
            if text:
                try:
                    structured = json.loads(text)
                except json.JSONDecodeError:
                    structured = None
            ok = not bool(getattr(result, "is_error", getattr(result, "isError", False)))
            outcome = ToolOutcome(
                tool=tool, ok=ok,
                error=(structured or {}).get("error") if not ok else None,
                content=[{"type": "text", "text": text}] if text else [],
                structured=structured,
                side_effects=bool(spec and spec.side_effects))
            # The server diffs the workspace around every file-writing call and
            # classifies refusals from the exception type; persist both so the
            # change set is evidence rather than the model's own change list,
            # and failures are countable rather than only readable.
            self._repo.log_tool_call(
                run_id, agent, tool, args, ok, outcome.error,
                affected_paths=(structured or {}).get("affected_paths"),
                diff_summary=(structured or {}).get("diff_summary"),
                status="ok" if ok else status_for(
                    (structured or {}).get("error_code") or CODE_TOOL_FAILED),
                error_code=(structured or {}).get("error_code"),
                security_event=(structured or {}).get("security_event"),
                source=SOURCE_AGENT,
                event_id=self._trace(principal, run_id, tool, ok,
                                     (structured or {}).get("error_code")))
            return outcome
        except Exception as e:  # noqa: BLE001
            self._repo.log_tool_call(run_id, agent, tool, args, False, str(e),
                                     event_id=self._trace(principal, run_id, tool,
                                                          False, ""))
            return ToolOutcome(tool=tool, ok=False, error=f"调用失败: {e}",
                               side_effects=bool(spec and spec.side_effects))


class BoundGateway:
    """A `ToolGateway` that supplies one principal to every call.

    A proxy, not a copy: it shares the client, the server and the audit path, and
    only fills in what `call` would otherwise read off the server's shared slot.

    For a caller that drives tools itself rather than through an agent's tool
    loop — the mock brain does — naming the principal at each of a dozen call
    sites is a thing to forget, and forgetting falls back to that shared slot
    **silently**. Handing such a caller a bound gateway makes the right thing the
    automatic one.
    """

    def __init__(self, gateway: ToolGateway, principal: dict[str, Any]) -> None:
        self._gateway = gateway
        self._principal = principal

    def __getattr__(self, name: str) -> Any:
        return getattr(self._gateway, name)

    async def call(self, tool: str, args: dict[str, Any], *,
                   agent: str | None = None, run_id: str | None = None,
                   principal: dict[str, Any] | None = None) -> ToolOutcome:
        return await self._gateway.call(
            tool, args, agent=agent, run_id=run_id,
            principal=principal or self._principal)

    def record_rejected(self, tool: str, args: dict[str, Any], error: str, *,
                        error_code: str = CODE_REPEATED_CALL,
                        agent: str | None = None, run_id: str | None = None,
                        principal: dict[str, Any] | None = None) -> None:
        self._gateway.record_rejected(
            tool, args, error, error_code=error_code, agent=agent, run_id=run_id,
            principal=principal or self._principal)


def _extract_text(result: Any) -> str:
    parts = []
    for c in getattr(result, "content", []) or []:
        t = getattr(c, "text", None)
        if t:
            parts.append(t)
    return "".join(parts)
