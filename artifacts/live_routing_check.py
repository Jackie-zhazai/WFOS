"""Live check for ⑦: routing across two real models, and a budget that really bites.

Not a test — it needs a network, a credential and money, so it stays out of the
suite. It answers the two questions a mock cannot:

  * with `[llm.routing]` set, does the *recorded* model per step follow the route
    against a real endpoint, rather than every step naming the default?
  * against a provider that reports usage, does the token budget stop a run — with
    real counts in the reason, not counts the harness guessed?

Run A (no budget) and run B (a budget of one token) use the same task, so the
step counts can be compared: if B does not stop earlier than A, the budget did
nothing and the check says so.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

os.environ.setdefault("no_proxy", "api.deepseek.com")
os.environ.setdefault("NO_PROXY", "api.deepseek.com")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wfos.config import default_config  # noqa: E402
from wfos.harness.orchestrator import STATE_AGENT, Harness  # noqa: E402
from wfos.mcp.client import ToolGateway  # noqa: E402
from wfos.mcp.server import WfosMcpServer  # noqa: E402
from wfos.storage.repo import Repo  # noqa: E402
from wfos.wiki.wiki import WikiClient  # noqa: E402

KEY = Path("C:/Users/rog/AppData/Local/Temp/wfos_live_key").read_text().strip()
os.environ["WFOS_LIVE_KEY"] = KEY

DEFAULT_MODEL = "deepseek-flash"
STRONG_MODEL = "deepseek-v4-pro"
TASK = "新增一个 greet(name) 函数到 app.py，返回 'hello ' + name"

PROJECT = Path(os.environ.get("WFOS_LIVE_PROJECT", "C:/Users/rog/AppData/Local/Temp/wfos_e2e/proj"))


def make_harness(data: Path):
    cfg = default_config()
    cfg.project_root = PROJECT
    cfg.data_dir = data
    cfg.db_path = data / "wfos.db"
    cfg.wiki_path = data / "wiki.db"
    cfg.llm.provider = "openai"
    cfg.llm.base_url = "https://api.deepseek.com/v1"
    cfg.llm.model = DEFAULT_MODEL
    cfg.llm.api_key_env = "WFOS_LIVE_KEY"
    cfg.llm.structured_output = "none"
    cfg.llm.routing = {"investigator": STRONG_MODEL, "implementer": STRONG_MODEL}
    cfg.harness.auto_approve_low_risk_files = 8
    repo = Repo(cfg.db_path)

    def approval_checker(run_id, tool, args):
        if not run_id:
            return False
        key = (args or {}).get("path") or ""
        return (repo.has_approved_approval(run_id, f"{tool}:{key}")
                or repo.has_approved_approval(run_id, tool))

    server = WfosMcpServer(cfg, repo, approval_checker=approval_checker)
    return cfg, repo, Harness(cfg, repo, ToolGateway(server, repo), WikiClient(repo))


def drive(cfg, repo, h, label):
    run = h.create_run(TASK, kind="feature")
    run_id = run["id"]
    print(f"\n=== {label}：run {run_id} ===")
    if cfg.harness.run_token_budget is not None:
        print(f"    token 预算 = {cfg.harness.run_token_budget}")
    final = asyncio.run(h.advance(run_id))
    print(f"    status={final['status']} state={final['state']}")
    if final.get("error"):
        print(f"    error={final['error']}")
    rows = repo.metric_rows(run_id)
    for r in rows:
        print(f"    {r['state']:<18} model={r['model']:<16} "
              f"in={r['input_tokens']} out={r['output_tokens']} "
              f"reasoning={r['reasoning_tokens']} calls={r['model_calls']}")
    return run_id, rows


def main() -> int:
    data = Path("C:/Users/rog/AppData/Local/Temp/wfos_live_routing")
    cfg, repo, h = make_harness(data)

    run_a, rows_a = drive(cfg, repo, h, "A 无预算 + 路由")

    cfg.harness.run_token_budget = 1
    run_b, rows_b = drive(cfg, repo, h, "B token 预算 = 1")

    failures: list[str] = []

    # ---- routing: the recorded model follows the route, not the default
    routed_rows = [r for r in rows_a if r["model"]]
    if not routed_rows:
        failures.append("步骤没有记录模型")
    for r in routed_rows:
        role = STATE_AGENT.get(r["state"])
        want = STRONG_MODEL if role in ("investigator", "implementer") else DEFAULT_MODEL
        if role and r["model"] != want:
            failures.append(f"{r['state']}({role}) 记录了 {r['model']}，应为 {want}")
    if {r["model"] for r in routed_rows} != {DEFAULT_MODEL, STRONG_MODEL}:
        failures.append(
            f"两个模型没有都出现在记录里：{sorted({r['model'] for r in routed_rows})}")

    # ---- budget: real reported usage, and a real stop
    if not any(r["input_tokens"] is not None for r in rows_a):
        failures.append("provider 没有上报用量，预算这条就没被验证")
    final_b = repo.get_run(run_b)
    if "预算耗尽" not in (final_b.get("error") or ""):
        failures.append(f"预算没有咬到：status={final_b['status']} error={final_b.get('error')}")
    if len(rows_b) >= len(rows_a):
        failures.append(f"设了预算反而没有更早停下：B {len(rows_b)} 步 vs A {len(rows_a)} 步")
    if final_b["status"] != "failed":
        failures.append(f"预算中断的运行状态应为 failed，实际 {final_b['status']}")

    print("\n" + "=" * 60)
    if failures:
        print("未通过：")
        for f in failures:
            print("  -", f)
        return 1
    print(f"通过：{len(rows_a)} 步全部按路由记录模型；"
          f"预算 1 token 在 {len(rows_b)} 步处停下（无预算时 {len(rows_a)} 步）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
