"""P6 acceptance (§14): the whole chain, run for real, through the CLI.

    Skill v1 → Experiment → Candidate → Candidate Benchmark → Evaluation
    → Compare → PASSED → explicit promote → Skill v2
    → run v2 → rollback v2 → v1 → run again

Nothing here is stubbed and nothing is asserted from the code's shape: every
check reads a record the run produced. The CLI is the entry point for every step
after seeding, so the command surface §12 asks for is exercised rather than
described.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
W = pathlib.Path("C:/Users/rog/AppData/Local/Temp/wfos_p6accept")
SUITE = "benchmark/regression"
TRIGGER = "regression"

ok = True


def check(label, condition, detail=""):
    global ok
    ok = ok and bool(condition)
    print(f"  [{'OK ' if condition else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")


LAST = {"out": "", "err": ""}


def cli(*args: str):
    env = {**os.environ, "WFOS_DATA": str(W / "data"), "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run([sys.executable, "-m", "wfos", *args], cwd=str(ROOT),
                          capture_output=True, encoding="utf-8", errors="replace",
                          env=env)
    LAST["out"], LAST["err"] = proc.stdout, proc.stderr
    return proc


shutil.rmtree(W, ignore_errors=True)
(W / "data").mkdir(parents=True)
os.environ["WFOS_DATA"] = str(W / "data")

from wfos.bench import run_benchmark  # noqa: E402
from wfos.models import ORIGIN_INTERACTIVE  # noqa: E402
from wfos.skills import content_digest  # noqa: E402
from wfos.storage.repo import Repo  # noqa: E402

repo = Repo(W / "data" / "wfos.db")

print("\n=== 1. 生产里的 v1 ===")
v1 = repo.add_skill(TRIGGER, "v1 规程", "触发: regression\n步骤: 先跑 check.py，再改",
                    "seed-run", origin=ORIGIN_INTERACTIVE)
check("v1 就位且是 current", v1["version"] == 1 and v1["superseded"] == 0,
      f"digest={v1['digest'][:12]}")
v1_bytes = v1["procedure"]

print("\n=== 2. Experiment：跑出真实失败 ===")
proc = cli("experiment", "run", "experiments/recovery-regression.json",
           "--workspace", str(W / "exp"), "--out", str(W / "exp.json"))
check("experiment run 成功", proc.returncode == 0, proc.stderr.strip()[:80])
experiment = json.loads((W / "exp.json").read_text(encoding="utf-8"))
failures = sorted({f for t in experiment["tasks"] for f in t.get("failures") or []})
check("实验产生了真实失败与恢复", bool(failures), "、".join(failures))

print("\n=== 3. Candidate：从证据推导 ===")
proc = cli("rsi", "propose", str(W / "exp.json"))
check("rsi propose 成功", proc.returncode == 0, proc.stderr.strip()[:80])
candidates = [c for c in repo.list_candidates() if c["candidate_id"] == f"skill:{TRIGGER}"]
check("生成了 skill 候选", bool(candidates))
candidate = candidates[-1]
check("候选声明了母版本", candidate["parent_version"] == 1,
      f"{candidate['candidate_id']} v{candidate['version']} ← v{candidate['parent_version']}")

print("\n=== 4. Baseline + 候选 Benchmark + Evaluation + Compare ===")
# `baseline create` names a tier, not a path.
proc = cli("baseline", "create", "--suite", "regression", "--out", str(W / "base.json"),
           "--workspace", str(W / "bws"))
check("baseline create 成功", proc.returncode == 0,
      (proc.stderr or proc.stdout).strip()[-200:])
baseline_bytes = (W / "base.json").read_bytes()

proc = cli("rsi", "evaluate", candidate["candidate_id"], "--version",
           str(candidate["version"]), "--baseline", str(W / "base.json"),
           "--workspace", str(W / "cand"))
check("rsi evaluate 成功", proc.returncode == 0, proc.stderr.strip()[:120])
settled = repo.get_candidate(candidate["candidate_id"], candidate["version"])
check("候选状态是 PASSED", settled["status"] == "PASSED", settled["status"])
comp = (settled["verdict"] or {}).get("comparison") or {}
check("用了 P3 的 compare（不是第二套）", "counts" in comp,
      f"counts={comp.get('counts')}")
check("评测记录被引用", bool((settled["verdict"] or {}).get("evaluationIds")),
      f"{len((settled['verdict'] or {}).get('evaluationIds') or [])} 条")

print("\n=== 5. 显式晋升（没有人替你做这个决定）===")
proc = cli("rsi", "promote", candidate["candidate_id"], "--version",
           str(candidate["version"]), "--baseline", str(W / "base.json"),
           "--actor", "rog", "--reason", "P6 验收：候选通过门禁")
check("rsi promote 成功", proc.returncode == 0, proc.stderr.strip()[:120])
print("        " + "\n        ".join(proc.stdout.strip().splitlines()))

v2 = repo.skill_version(TRIGGER, 2)
check("v2 被创建", v2 is not None and v2["superseded"] == 0)
check("v1 仍然存在且内容未变",
      repo.skill_version(TRIGGER, 1)["procedure"] == v1_bytes
      and repo.skill_version(TRIGGER, 1)["superseded"] == 1)
check("v2 的 digest 是内容摘要",
      v2 is not None and v2["digest"] == content_digest(v2))
check("parent → child 链接正确",
      v2 is not None and v2["parent_id"] == repo.skill_version(TRIGGER, 1)["id"])

promotions = repo.promotions_for_skill(TRIGGER)
promotion = promotions[-1]
check("晋升记录完整（§5 的字段）",
      all(k in promotion and promotion[k] not in ("", None, [], {}) for k in
          ("promotion_id", "candidate_id", "candidate_version", "parent_skill_version",
           "promoted_skill_version", "source_experiment", "source_runs", "evaluation",
           "compare_result", "gate", "created_at", "actor", "reason")),
      f"{promotion['promotion_id']}，评测记录 "
      f"{len(promotion.get('evaluation_ids') or [])} 条")

print("\n=== 6. provenance 链可完整追溯 ===")
chain = {
    "production skill v2": f"{v2['trigger']} v{v2['version']} digest={v2['digest'][:12]}",
    "promotion": promotion["promotion_id"],
    "candidate": f"{promotion['candidate_id']} v{promotion['candidate_version']}",
    "experiment": promotion["source_experiment"],
    "runs": "、".join(r[:8] for r in promotion["source_runs"]),
    "evaluations": "、".join(str(i) for i in promotion.get("evaluation_ids") or []) or "-",
    "parent skill v1": f"v{promotion['parent_skill_version']} "
                       f"digest={promotion['parent_digest'][:12]}",
}
for name, value in chain.items():
    check(f"链上 {name}", bool(value) and value != "-", value)

print("\n=== 7. v2 作为正式 Skill 被加载并真的跑一遍 ===")
live = repo.skills_for([TRIGGER], origin=ORIGIN_INTERACTIVE)
check("交互式 prompt 现在加载的是 v2",
      [s["version"] for s in live] == [2], f"v{[s['version'] for s in live]}")
check("加载到的是 v2 的内容", live[0]["procedure"] == v2["procedure"])

before_run = run_benchmark(SUITE, workspace=W / "after-promote", tier="regression")
check("晋升后 benchmark 正常跑完",
      len(before_run.tasks) > 0,
      f"{len(before_run.tasks)} 个任务，rollup={before_run.rollup}")

print("\n=== 8. 回滚 v2 → v1 ===")
proc = cli("rsi", "rollback", TRIGGER, "1", "--actor", "rog", "--reason", "P6 验收：回滚")
check("rsi rollback <skill> <version> 成功", proc.returncode == 0,
      proc.stderr.strip()[:120])
print("        " + proc.stdout.strip().replace("\n", "\n        "))

check("current 指针回到 v1",
      repo.latest_skill(TRIGGER)["version"] == 1)
check("v2 仍然存在（回滚不是删除）",
      repo.skill_version(TRIGGER, 2) is not None
      and repo.skill_version(TRIGGER, 2)["superseded"] == 1)
check("v2 的内容没有被改动",
      repo.skill_version(TRIGGER, 2)["digest"] == content_digest(v2))

rollbacks = repo.rollbacks_for(TRIGGER)
check("回滚记录完整（§6 的字段）",
      bool(rollbacks) and all(k in rollbacks[-1] for k in
                              ("rollback_id", "trigger", "from_version", "to_version",
                               "from_digest", "to_digest", "actor", "reason",
                               "created_at")),
      f"{rollbacks[-1]['rollback_id']}：v{rollbacks[-1]['from_version']}"
      f" → v{rollbacks[-1]['to_version']}")

print("\n=== 9. 回滚后加载的是 v1，并且真的跑一遍 ===")
live = repo.skills_for([TRIGGER], origin=ORIGIN_INTERACTIVE)
check("交互式 prompt 现在加载的是 v1",
      [s["version"] for s in live] == [1], f"v{[s['version'] for s in live]}")
check("加载到的是 v1 的内容", live[0]["procedure"] == v1_bytes)
after_run = run_benchmark(SUITE, workspace=W / "after-rollback", tier="regression")
check("回滚后 benchmark 正常跑完", len(after_run.tasks) == len(before_run.tasks),
      f"{len(after_run.tasks)} 个任务")

print("\n=== 10. 什么都没被删、baseline 没被动 ===")
check("v1、v2、候选、晋升、回滚记录都还在",
      repo.skill_version(TRIGGER, 1) is not None
      and repo.skill_version(TRIGGER, 2) is not None
      and repo.get_candidate(candidate["candidate_id"], candidate["version"]) is not None
      and len(repo.promotions_for_skill(TRIGGER)) >= 1 and bool(rollbacks))
check("候选仍是 PROMOTED（晋升后不可篡改）",
      repo.get_candidate(candidate["candidate_id"], candidate["version"])["status"]
      == "PROMOTED")
check("baseline 未被修改", (W / "base.json").read_bytes() == baseline_bytes)

print("\n=== 11. CLI 的 --json 是纯 JSON ===")
for args in (["rsi", "history", TRIGGER, "--json"],
             ["rsi", "promotion", "show", promotion["promotion_id"], "--json"],
             ["rsi", "promotion", promotion["promotion_id"], "--json"]):
    proc = cli(*args)
    try:
        json.loads(proc.stdout)
        parsed = True
    except Exception:  # noqa: BLE001
        parsed = False
    check(f"{' '.join(args[:3])} --json 合法", parsed)

print()
print("全部通过" if ok else "有未通过项")
raise SystemExit(0 if ok else 1)
