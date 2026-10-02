"""End-to-end MCP stdio transport.

Spawns the real `wfos mcp` entry point as a subprocess and drives it with a
real MCP client over real protocol frames — the in-process gateway tests prove
the policy layer, but only this proves an external client can reach it at all.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from wfos.storage.repo import Repo

ROOT = Path(__file__).resolve().parents[1]


def _env(cfg) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["WFOS_PROJECT"] = str(cfg.project_root)
    env["WFOS_DATA"] = str(cfg.data_dir)
    env["WFOS_PROVIDER"] = "mock"
    return env


def _serve(cfg, role: str) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable, args=["-m", "wfos", "mcp", "--role", role],
        env=_env(cfg), cwd=str(ROOT))


async def _drive(params, errlog, scenario):
    """Run `scenario(session)` against a live stdio server, then shut it down."""
    async with (stdio_client(params, errlog=errlog) as (read, write),
                ClientSession(read, write) as session):
        await session.initialize()
        return await scenario(session)


@pytest.fixture()
def errlog(tmp_path: Path):
    """The child's stderr sink — stdout stays reserved for protocol frames."""
    with open(tmp_path / "mcp-errlog.txt", "w", encoding="utf-8") as fh:
        yield fh


def _text(result) -> str:
    return "".join(getattr(c, "text", "") for c in result.content)


def _failed(result) -> bool:
    return bool(getattr(result, "is_error", getattr(result, "isError", False)))


def test_stdio_handshake_lists_the_full_tool_suite(app, errlog):
    """The catalogue is complete for every caller; what a caller may actually
    do is enforced per call by the policy gate, not by hiding tools."""
    async def scenario(session):
        listed = await session.list_tools()
        return {t.name for t in listed.tools}

    names = asyncio.run(_drive(_serve(app["cfg"], "investigator"), errlog, scenario))
    assert {"workspace.read", "workspace.write", "workspace.delete",
            "wiki.search"} <= names


def test_stdio_serves_a_real_read_and_refuses_an_unauthorised_write(app, errlog):
    async def scenario(session):
        read = await session.call_tool("workspace.read", {"path": "app.py"})
        write = await session.call_tool("workspace.write",
                                        {"path": "evil.py", "content": "x = 1\n"})
        return read, write

    read, write = asyncio.run(_drive(_serve(app["cfg"], "investigator"), errlog, scenario))
    assert not _failed(read)
    assert "return x * 2" in _text(read)

    assert _failed(write)
    assert "无权调用" in _text(write)
    assert not (app["sandbox"] / "evil.py").exists()


def test_stdio_audits_calls_to_the_pseudo_run(app, errlog):
    """No in-process ToolGateway is in the path, so the server itself must
    write the audit trail for whatever an external client does."""
    async def scenario(session):
        return await session.call_tool("workspace.read", {"path": "app.py"})

    result = asyncio.run(_drive(_serve(app["cfg"], "investigator"), errlog, scenario))
    assert not _failed(result)

    repo = Repo(app["cfg"].db_path)
    try:
        calls = repo.tool_calls("external")
        assert [c["tool"] for c in calls] == ["workspace.read"]
        assert calls[0]["ok"] == 1
        assert calls[0]["agent"] == "investigator"      # a plain role name
        assert calls[0]["source"] == "mcp"              # origin, recorded separately
    finally:
        repo.close()


def test_stdio_implementer_can_write_and_change_is_attributed(app, errlog):
    """A writer role over stdio gets the same snapshot attribution that the
    in-process path records."""
    async def scenario(session):
        listed = await session.list_tools()
        write = await session.call_tool(
            "workspace.write", {"path": "fresh.py", "content": "a = 1\n"})
        return {t.name for t in listed.tools}, write

    names, write = asyncio.run(_drive(_serve(app["cfg"], "implementer"), errlog, scenario))
    assert "workspace.write" in names
    assert not _failed(write), _text(write)
    assert "fresh.py" in _text(write)                # affected_paths ride in the result
    assert (app["sandbox"] / "fresh.py").read_text(encoding="utf-8") == "a = 1\n"

    repo = Repo(app["cfg"].db_path)
    try:
        assert repo.affected_paths_for_run("external", agent="implementer") == ["fresh.py"]
        assert repo.tool_calls("external")[-1]["source"] == "mcp"
    finally:
        repo.close()


def test_stdio_diagnostics_never_touch_stdout(app, errlog):
    """A stray print() on stdout corrupts the protocol stream, so the startup
    banner must live on stderr."""
    async def scenario(session):
        return await session.call_tool("echo", {"text": "hi"})

    result = asyncio.run(_drive(_serve(app["cfg"], "investigator"), errlog, scenario))
    assert "hi" in _text(result)
    errlog.flush()
    assert "wfos MCP stdio server up" in Path(errlog.name).read_text(encoding="utf-8")


def test_stdio_rejects_an_unknown_role(app):
    """A bad --role must fail fast rather than serve some default capability."""
    proc = subprocess.run(
        [sys.executable, "-m", "wfos", "mcp", "--role", "root"],
        env=_env(app["cfg"]), cwd=str(ROOT), capture_output=True, text=True, timeout=60)
    assert proc.returncode != 0
    assert "invalid choice" in (proc.stderr or "").lower()
