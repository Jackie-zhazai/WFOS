"""What the command line promises a script that calls it.

Every check here is written from outside the process — the CLI is invoked as
`python -m wfos …` and judged on its exit code, its stdout and its stderr. That
is the only vantage point from which "stdout is pure JSON" and "the exit code is
right" are even observable; asserting against the functions in `cli.py` would
test the code's shape rather than the contract a caller depends on.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def env(tmp_path: Path) -> dict:
    data = tmp_path / "data"
    project = tmp_path / "proj"
    data.mkdir()
    project.mkdir()
    (project / "app.py").write_text("def compute(x):\n    return x * 2\n", encoding="utf-8")
    return {**os.environ,
            "WFOS_DATA": str(data), "WFOS_PROJECT": str(project),
            "WFOS_PROVIDER": "mock", "PYTHONIOENCODING": "utf-8"}


def wfos(*args: str, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "wfos", *args], cwd=str(ROOT),
                          capture_output=True, encoding="utf-8", errors="replace",
                          env=env)


def subcommands() -> list[str]:
    proc = subprocess.run([sys.executable, "-m", "wfos", "--help"], cwd=str(ROOT),
                          capture_output=True, encoding="utf-8", errors="replace")
    marker = "{" if "{" in proc.stdout else "["
    start = proc.stdout.index(marker) + 1
    end = proc.stdout.index("}" if marker == "{" else "]", start)
    return [name.strip() for name in proc.stdout[start:end].split(",") if name.strip()]


# --------------------------------------------------------------------- --help
def test_every_command_has_help_that_exits_zero():
    names = subcommands()
    assert len(names) >= 20, f"只发现 {len(names)} 个子命令：{names}"
    broken = []
    for name in names:
        proc = subprocess.run([sys.executable, "-m", "wfos", name, "--help"],
                              cwd=str(ROOT), capture_output=True, encoding="utf-8",
                              errors="replace")
        if proc.returncode != 0 or "usage" not in (proc.stdout + proc.stderr):
            broken.append((name, proc.returncode))
    assert not broken, f"这些子命令的 --help 有问题：{broken}"


# ------------------------------------------------------------------- exit codes
def test_an_unknown_command_is_a_usage_error(env):
    proc = wfos("definitely-not-a-command", env=env)

    assert proc.returncode == 2, "用法错误必须是 2（argparse 的约定），不是 1"
    assert "usage" in (proc.stdout + proc.stderr)


def test_a_missing_required_argument_is_a_usage_error(env):
    proc = wfos("compare", env=env)

    assert proc.returncode == 2


def test_an_unknown_run_is_not_found_not_a_crash(env):
    proc = wfos("status", "no-such-run", "--json", env=env)

    assert proc.returncode == 1
    assert json.loads(proc.stdout)["error"]
    assert "Traceback" not in proc.stderr


def test_an_unknown_task_id_is_refused_before_anything_is_written(env):
    """A bad argument must not leave a record behind."""
    created = wfos("run", "新增一个模块", "--json", env=env)
    assert created.returncode == 0
    run_id = json.loads(created.stdout)["run"]["id"]

    proc = wfos("evaluate", run_id, "benchmark/smoke", "not-a-task", "--json", env=env)

    assert proc.returncode == 2
    body = json.loads(proc.stdout)
    assert "not-a-task" in body["error"]
    assert body["available"], "拒绝时要说明有哪些可选，否则调用者只能猜"

    from wfos.storage.repo import Repo
    repo = Repo(Path(env["WFOS_DATA"]) / "wfos.db")
    assert repo.evaluations_for_run(run_id) == [], "被拒绝的调用不该写下半条判定"


# ---------------------------------------------------------------- JSON purity
@pytest.mark.parametrize("args", [
    ("history", "--json"),
    ("status", "--json"),
    ("rsi", "list", "--json"),
    ("skills", "--json"),
    ("capabilities", "--json"),
    ("metrics", "--json"),
])
def test_json_goes_to_stdout_and_parses(env, args):
    wfos("run", "新增一个模块", "--json", env=env)          # something to report on

    proc = wfos(*args, env=env)

    assert proc.returncode == 0, proc.stderr
    parsed = json.loads(proc.stdout)                        # the whole of stdout
    assert isinstance(parsed, (dict, list))


def test_json_stdout_carries_no_human_text_around_it(env):
    """The property `jq` depends on, checked by parsing stdout and nothing else.

    The probe is an **error** path, deliberately. Every command in `cli.py` that
    succeeds under `--json` happens to guard its progress lines with
    `if not args.json`, so a success path would pass even if the printer stopped
    routing human output away from stdout. The error branches do not guard — they
    just print a sentence and return a code — so they are where the routing is
    actually load-bearing.
    """
    proc = wfos("baseline", "create", "--suite", "no-such-suite", "--json", env=env)

    assert proc.returncode == 2
    text = proc.stdout.strip()
    assert text.startswith("{") and text.endswith("}"), f"stdout 不纯：{text[:160]!r}"
    json.loads(text)                       # 整段 stdout 就是那一份文档


def test_json_stdout_stays_pure_when_a_command_prints_progress(env, tmp_path):
    """And on a path that prints several human lines before the document."""
    target = tmp_path / "base.json"
    target.write_text('{"sentinel": true}', encoding="utf-8")

    proc = wfos("baseline", "create", "--suite", "smoke", "--out", str(target),
                "--workspace", str(tmp_path / "ws"), "--json", env=env)

    assert proc.returncode != 0
    body = json.loads(proc.stdout)          # 纯 JSON，人读的那几行去了 stderr
    assert body["error"]
    assert "已被占用" in proc.stderr or "基线" in proc.stderr


def test_a_failed_json_call_is_still_json(env):
    """An error is a result too. A script parsing stdout must not have to
    special-case the failure path — that is where it is most likely to break."""
    for args in (("status", "no-such-run", "--json"),
                 ("rsi", "show", "skill:nope", "--json"),
                 ("evaluate", "no-such-run", "benchmark/smoke", "t", "--json")):
        proc = wfos(*args, env=env)
        assert proc.returncode != 0, f"{args} 本该失败"
        json.loads(proc.stdout)


# --------------------------------------------------------------------- Windows
def test_a_windows_path_argument_is_accepted(env, tmp_path):
    """Backslashes in a path argument, without the shell eating them."""
    experiment = tmp_path / "exp.json"
    experiment.write_text(json.dumps({
        "kind": "experiment-spec", "schemaVersion": 1, "experimentId": "win-path",
        "name": "win", "description": "", "suite": "benchmark/smoke",
        "matrix": {}}), encoding="utf-8")

    proc = wfos("rsi", "propose", str(experiment), env=env)

    assert "Traceback" not in proc.stderr, proc.stderr
    assert "找不到" not in proc.stdout, "Windows 路径被当成了别的东西"


# --------------------------------------------------------------- repeatability
def test_a_read_command_repeats_itself_exactly(env):
    wfos("run", "新增一个模块", "--json", env=env)

    first = wfos("history", "--json", env=env).stdout
    second = wfos("history", "--json", env=env).stdout

    assert json.loads(first) == json.loads(second), "同样的输入两次结果不同"


# ------------------------------------------------------------- append-only-ness
def test_an_existing_baseline_is_not_overwritten(env, tmp_path):
    """`--out` points at a file that already exists: refused, and the file stays."""
    target = tmp_path / "base.json"
    target.write_text('{"sentinel": true}', encoding="utf-8")

    proc = wfos("baseline", "create", "--suite", "smoke", "--out", str(target),
                "--workspace", str(tmp_path / "ws"), env=env)

    assert proc.returncode != 0
    assert target.read_text(encoding="utf-8") == '{"sentinel": true}'
