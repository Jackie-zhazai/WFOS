"""End-to-end CLI integration: subprocess run, status, wiki search."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _run_cli(*args, sandbox: Path, data: Path) -> tuple[int, str]:
    env = dict(os.environ)
    env["WFOS_PROJECT"] = str(sandbox)
    env["WFOS_DATA"] = str(data)
    env["WFOS_DB"] = str(data / "wfos.db")
    env["WFOS_WIKI"] = str(data / "wiki.db")
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [PYTHON, "-m", "wfos", *args], cwd=ROOT, env=env,
        capture_output=True, text=True, timeout=120,
        encoding="utf-8", errors="replace")
    return proc.returncode, proc.stdout + proc.stderr


@pytest.fixture()
def cli_sandbox(tmp_path: Path):
    sandbox = tmp_path / "proj"
    sandbox.mkdir()
    (sandbox / "app.py").write_text("def compute(x):\n    return x * 2\n", encoding="utf-8")
    (sandbox / "check.py").write_text(
        'from app import compute\n'
        'assert compute(1) == 2, "compute(1)"\n'
        'assert compute(2) == 4, "compute(2)"\n'
        'print("PASS: compute(1)==", compute(1) == 2)\n'
        'print("PASS: compute(2)==", compute(2) == 4)\n', encoding="utf-8")
    return sandbox, tmp_path / "data"


def test_cli_feature_run_completes(cli_sandbox):
    sandbox, data = cli_sandbox
    code, out = _run_cli(
        "run", "新增一个用户模块，实现在 app.py 中追加 feature_user() 函数",
        sandbox=sandbox, data=data)
    assert code == 0, out
    assert "状态=completed" in out
    assert "功能开发" in out


def test_cli_history_and_status(cli_sandbox):
    sandbox, data = cli_sandbox
    code, out = _run_cli(
        "run", "新增一个用户模块", sandbox=sandbox, data=data)
    assert code == 0, out
    code, out = _run_cli("history", sandbox=sandbox, data=data)
    assert code == 0 and "feature" in out


def test_cli_wiki_search_after_run(cli_sandbox):
    sandbox, data = cli_sandbox
    _run_cli("run", "新增一个用户模块，实现在 app.py 中追加 feature_user() 函数",
             sandbox=sandbox, data=data)
    # curator distilled knowledge on completion (verified entries -> case tier)
    code, out = _run_cli("wiki", "list", sandbox=sandbox, data=data)
    assert code == 0
    assert ("candidate" in out) or ("case" in out) or ("authoritative" in out)
    code, out = _run_cli("wiki", "search", "新增一个用户模块", sandbox=sandbox, data=data)
    assert code == 0 and out.strip() != "无匹配结果。"


def test_cli_bugfix_history_retrieves_wiki(cli_sandbox):
    sandbox, data = cli_sandbox
    # seed a related knowledge case, then run a bugfix which searches history
    import sys
    sys.path.insert(0, str(ROOT))
    from wfos.storage.repo import Repo
    data.mkdir(parents=True, exist_ok=True)
    repo = Repo(str(data / "wfos.db"))     # creates the schema
    repo.add_wiki("case", "compute 修复经验", "compute 应采用线性实现",
                  verified=True, trust="high", status="published", tags=["bugfix"])
    repo.close()
    code, out = _run_cli("run", "--kind", "bugfix",
                         "修复 compute 结果不对的 bug，重新实现 compute 使其正确",
                         sandbox=sandbox, data=data)
    assert code == 0, out
    assert "问题修复" in out
    assert out.strip() and "completed" in out or "waiting" in out or "审批" in out
