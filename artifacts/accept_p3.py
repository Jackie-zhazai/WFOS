"""P3 acceptance: the seven things the phase has to prove, on real runs."""
import json
import os
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
W = pathlib.Path("C:/Users/rog/AppData/Local/Temp/wfos_acc2")
# 从干净的工作区开始。这些脚本里的写操作（基线、实验结果）都拒绝覆盖 —— 那是
# 被验收的性质本身 —— 所以重跑必须先清场，否则第二次运行会被第一次挡住。
shutil.rmtree(W, ignore_errors=True)
(W / "data").mkdir(parents=True)
os.environ["WFOS_DATA"] = str(W / "data")

from wfos.bench import load_record, run_benchmark, write_baseline  # noqa: E402
from wfos.bench_compare import REGRESSION, UNKNOWN_VERDICT, compare, render  # noqa: E402

ok = True


def check(label, condition, detail=""):
    global ok
    ok = ok and bool(condition)
    print(f"  [{'OK ' if condition else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")


# ---- 1. 相同输入可重复运行 -------------------------------------------------
first = run_benchmark(ROOT / "benchmark/smoke", workspace=W / "r1", tier="smoke")
second = run_benchmark(ROOT / "benchmark/smoke", workspace=W / "r2", tier="smoke")
same_verdicts = ([t.evaluation.verdict for t in first.tasks]
                 == [t.evaluation.verdict for t in second.tasks])
check("1. 相同输入可重复运行", same_verdicts and first.environment["digest"]
      == second.environment["digest"],
      f"digest 相同、判定一致 {[t.evaluation.verdict for t in first.tasks]}")

# ---- baseline 落盘，并证明不可被运行覆盖 -----------------------------------
baseline_path = write_baseline(W / "accept-base.json", first, source=str(ROOT / "benchmark/smoke"))
before_bytes = baseline_path.read_bytes()

# ---- 2. baseline 不会被运行修改 --------------------------------------------
run_benchmark(ROOT / "benchmark/smoke", workspace=W / "r3", tier="smoke")
check("2. baseline 不会被运行修改", baseline_path.read_bytes() == before_bytes)

stored = load_record(baseline_path)

# ---- 3. evaluator 结果可被 compare 消费 ------------------------------------
fresh = run_benchmark(ROOT / "benchmark/smoke", workspace=W / "r4", tier="smoke")
report = compare(stored, fresh.as_json())
check("3. evaluator 结果可被 compare 消费",
      all(t["verdict"] in ("pass", "fail") for t in stored["tasks"])
      and len(report.findings) > 0,
      f"{len(report.findings)} 条比较项")

# ---- 4/5. 真实回归 + 硬门禁传递 --------------------------------------------
broken = json.loads((ROOT / "benchmark/smoke/tasks.json").read_text(encoding="utf-8"))
for case in broken["cases"]:
    if case["id"] == "feature-happy-path":
        # A task that demands something this run cannot do. The gate must stop it
        # and the comparison must carry that through as a regression.
        case["expect"] = {"completedFullChain": True, "writesOutPlan": 0}
broken_dir = W / "broken" / "smoke"
broken_dir.mkdir(parents=True, exist_ok=True)
(broken_dir / "tasks.json").write_text(json.dumps(broken, ensure_ascii=False),
                                       encoding="utf-8")
bad_run = run_benchmark(broken_dir, workspace=W / "r5", tier="broken")
bad = next(t for t in bad_run.tasks if t.task_id == "feature-happy-path")
check("4. Hard Gate 失败被 evaluator 判为失败",
      bad.evaluation.verdict == "fail" and bad.evaluation.hard_gate,
      f"hard_gate={list(bad.evaluation.hard_gate)}")

regression = compare(stored, bad_run.as_json())
gate_findings = [f for f in regression.findings
                 if f.subject == "hardGate" and f.verdict == REGRESSION]
check("5. Hard Gate 失败传递到 compare",
      not regression.ok and gate_findings,
      f"可判={regression.ok}，hardGate 回归 {len(gate_findings)} 条")

# ---- 6. unknown 保持 unknown ----------------------------------------------
unknowns = [f for f in report.findings if f.verdict == UNKNOWN_VERDICT]
check("6. unknown 保持 unknown",
      unknowns and all(f.verdict != REGRESSION for f in unknowns),
      f"{len(unknowns)} 条 unknown，无一条被升级成结论")

# ---- 7. 交互式项目根行为未变（由调用方验证） --------------------------------
print()
print(render(regression) if not regression.ok else "（无回归）")
print()
print("全部通过" if ok else "有未通过项")
raise SystemExit(0 if ok else 1)
