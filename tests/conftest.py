"""Shared fixtures: a hermetic sandbox project + a wired Harness (mock brain)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wfos.config import default_config
from wfos.harness.orchestrator import Harness
from wfos.mcp.client import ToolGateway
from wfos.mcp.server import WfosMcpServer
from wfos.storage.repo import Repo
from wfos.wiki.wiki import WikiClient

PASSING_CHECK = (
    'from app import compute\n\n'
    'assert compute(1) == 2, "compute(1)"\n'
    'assert compute(2) == 4, "compute(2)"\n'
    'print("PASS: compute(1)==", compute(1) == 2)\n'
    'print("PASS: compute(2)==", compute(2) == 4)\n'
)


@pytest.fixture()
def sandbox(tmp_path: Path) -> Path:
    d = tmp_path / "proj"
    d.mkdir()
    (d / "app.py").write_text("def compute(x):\n    return x * 2\n", encoding="utf-8")
    (d / "check.py").write_text(PASSING_CHECK, encoding="utf-8")
    return d


@pytest.fixture()
def app(sandbox: Path, tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    cfg = default_config()
    cfg.project_root = sandbox
    cfg.data_dir = data
    cfg.db_path = data / "wfos.db"
    cfg.wiki_path = data / "wiki.db"
    cfg.llm.provider = "mock"
    repo = Repo(cfg.db_path)

    def approval_checker(run_id, tool, args):
        if not run_id:
            return False
        key = (args or {}).get("path") or ""
        return (repo.has_approved_approval(run_id, f"{tool}:{key}")
                or repo.has_approved_approval(run_id, tool))

    server = WfosMcpServer(cfg, repo, approval_checker=approval_checker)
    gateway = ToolGateway(server, repo)
    wiki = WikiClient(repo)
    harness = Harness(cfg, repo, gateway, wiki)
    return {"cfg": cfg, "repo": repo, "gateway": gateway, "wiki": wiki, "harness": harness,
            "sandbox": sandbox}


@pytest.fixture()
def bad_check(sandbox: Path) -> None:
    """Make check.py fail against the current app.py."""
    (sandbox / "check.py").write_text(
        'from app import compute\n\n'
        'assert compute(1) == 2, "compute(1)"\n'
        'assert compute(2) == 4, "compute(2)"\n'
        'assert compute(3) == 9, "compute(3)"\n'
        'print("PASS: compute(3)==", compute(3) == 9)\n', encoding="utf-8")


def drive(h: Harness, text: str, kind: str | None = None) -> dict:
    """Run one full scenario to a blocking/terminal state inside a single loop."""
    import asyncio

    run = h.create_run(text, kind=kind)

    async def scenario():
        return await h.advance(run["id"])

    return asyncio.run(scenario())
