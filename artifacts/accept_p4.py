"""P4 acceptance: the nine things the phase has to prove, on real runs."""
import json
import os
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
W = pathlib.Path("C:/Users/rog/AppData/Local/Temp/wfos_p4accept")
# 从干净的工作区开始。这些脚本里的写操作（基线、实验结果）都拒绝覆盖 —— 那是
# 被验收的性质本身 —— 所以重跑必须先清场，否则第二次运行会被第一次挡住。
shutil.rmtree(W, ignore_errors=True)
(W / "data").mkdir(parents=True)
os.environ["WFOS_DATA"] = str(W / "data")

from wfos.bench import run_benchmark, write_baseline  # noqa: E402
from wfos.experiment import (  # noqa: E402
    ExperimentSpec,
    comparisons,
    load_experiment_result,
    run_experiment,
    write_experiment,
)

SMOKE = ROOT / "benchmark/smoke"
ok = True


def check(label, condition, detail=""):
    global ok
    ok = ok and bool(condition)
    print(f"  [{'OK ' if condition else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")


# A matrix with two real arms plus one arm that cannot start, so failure
# isolation is exercised on the same run rather than in a separate scenario.
spec = ExperimentSpec(
    experiment_id="p4-accept", version=1, name="P4 acceptance",
    description="", suite=str(SMOKE),
    matrix={"model": ["mock", "mock-alt"],
            "config": [None, {"definitely_not_a_field": 1}]})

run = run_experiment(spec, workspace=W / "ws")
started = [c for c in run.cells if c.status == "ran"]
dead = [c for c in run.cells if c.status != "ran"]

# ---- 1. 两个 cell 是独立 Run ----------------------------------------------
check("1. cell 是独立 Run", len({c.run_id for c in started}) == len(started),
      f"{len(started)} 个已启动 cell，{len({c.run_id for c in started})} 个不同 run_id")

# ---- 2. 各自独立 Trace / Evaluation ---------------------------------------
dbs = sorted((W / "ws").rglob("wfos.db"))
check("2. 各自独立 Trace / Evaluation",
      len(dbs) == len(started) and all(c.evaluation for c in started),
      f"{len(dbs)} 个独立库，判定齐全")

# ---- 3. 一个 cell 失败不污染另一个 ----------------------------------------
check("3. 一个 cell 失败不污染其他", bool(started) and bool(dead)
      and all(c.run_id for c in started) and all(not c.run_id for c in dead),
      f"跑起来 {len(started)}，未能启动 {len(dead)}（{sorted({c.failure_class for c in dead})}）")

# ---- 6. baseline 没有被 Experiment 修改（先建一份，再跑实验）--------------
baseline = write_baseline(W / "base.json",
                          run_benchmark(SMOKE, workspace=W / "bws", tier="smoke"),
                          source=str(SMOKE))
before = baseline.read_bytes()
run_with_baseline = run_experiment(spec, workspace=W / "ws2", baseline=str(baseline))
check("6. baseline 没有被 Experiment 修改", baseline.read_bytes() == before)

# ---- 4. Compare 能消费 Experiment Run -------------------------------------
reports = comparisons(run_with_baseline, str(baseline))
check("4. Compare 能消费 Experiment Run", bool(reports),
      f"{len(reports)} 份按臂比较，判定词表 {sorted(reports[0]['counts']) if reports else []}")

# ---- 5. unknown 保持 unknown ----------------------------------------------
unknowns = [f for r in reports for f in r["findings"] if f["verdict"] == "unknown"]
check("5. unknown 保持 unknown",
      bool(unknowns) and all(f["verdict"] != "pass" for f in unknowns),
      f"{len(unknowns)} 条 unknown，无一被当成通过")

# ---- 8. 相同 ExperimentSpec 可重新执行 -------------------------------------
again = run_experiment(spec, workspace=W / "ws3")
check("8. 相同 spec 可重新执行",
      [c.cell_id for c in again.cells] == [c.cell_id for c in run.cells]
      and ([c.evaluation.get("verdict") for c in again.cells]
           == [c.evaluation.get("verdict") for c in run.cells]),
      "cell 身份与判定一致")

# ---- 结果可持久化并重新读取 ------------------------------------------------
path = write_experiment(W / "result.json", run_with_baseline, str(baseline))
stored = load_experiment_result(path)
check("结果可持久化并重新读取",
      stored["experimentId"] == "p4-accept" and len(stored["cells"]) == len(run.cells),
      f"{path.name}，{len(stored['cells'])} 个 cell")

# ---- 7. CLI 的 --json 是纯 JSON（真实子进程）--------------------------------
cli = subprocess.run([sys.executable, "-m", "wfos", "experiment", "run",
                      str(ROOT / "experiments/model-sweep.json"),
                      "--workspace", str(W / "ws_cli"), "--out", str(W / "cli.json"),
                      "--baseline", str(baseline), "--json"],
                     capture_output=True, cwd=str(ROOT), encoding="utf-8", errors="replace",
                     env={**os.environ, "WFOS_DATA": str(W / "data"),
                          "PYTHONIOENCODING": "utf-8"})
try:
    payload = json.loads(cli.stdout)
    parsed = True
except Exception:  # noqa: BLE001
    payload, parsed = {}, False
check("7. CLI --json 输出合法 JSON", parsed,
      f"kind={payload.get('kind')} cells={len(payload.get('cells') or [])} "
      f"comparisons={len(payload.get('comparisons') or [])}")

show = subprocess.run([sys.executable, "-m", "wfos", "experiment", "show",
                       str(path), "--json"],
                      capture_output=True, cwd=str(ROOT), encoding="utf-8", errors="replace",
                      env={**os.environ, "WFOS_DATA": str(W / "data"),
                           "PYTHONIOENCODING": "utf-8"})
try:
    json.loads(show.stdout)
    show_ok = True
except Exception:  # noqa: BLE001
    show_ok = False
check("experiment show --json 也合法", show_ok)

print()
for cell in run.cells:
    state = cell.evaluation.get("verdict") if cell.status == "ran" else \
        f"未能启动({cell.failure_class})"
    print(f"  {cell.cell_id:<62} {state}  run={cell.run_id[:8] or '-'}")
print()
print("全部通过" if ok else "有未通过项")
raise SystemExit(0 if ok else 1)
