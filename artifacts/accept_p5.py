"""P5 acceptance: the ten things the phase has to prove, on real runs."""
import json
import os
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
W = pathlib.Path("C:/Users/rog/AppData/Local/Temp/wfos_p5accept")
# 从干净的工作区开始。这些脚本里的写操作（基线、实验结果）都拒绝覆盖 —— 那是
# 被验收的性质本身 —— 所以重跑必须先清场，否则第二次运行会被第一次挡住。
shutil.rmtree(W, ignore_errors=True)
(W / "data").mkdir(parents=True)
os.environ["WFOS_DATA"] = str(W / "data")

from wfos.bench import run_benchmark, write_baseline  # noqa: E402
from wfos.experiment import (  # noqa: E402
    ExperimentSpec,
    load_experiment_result,
    run_experiment,
    write_experiment,
)
from wfos.rsi import (  # noqa: E402
    STATUS_PASSED,
    STATUSES,
    analyse,
    evaluate_candidate,
    record_from_row,
    record_proposal,
)
from wfos.storage.repo import Repo  # noqa: E402

SUITE = "benchmark/regression"
ok = True


def check(label, condition, detail=""):
    global ok
    ok = ok and bool(condition)
    print(f"  [{'OK ' if condition else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")


repo = Repo(W / "data" / "wfos.db")

# ---- the run that will be analysed -----------------------------------------
spec = ExperimentSpec(experiment_id="p5-accept", version=1, name="P5 acceptance",
                      description="", suite=SUITE, matrix={})
run = run_experiment(spec, workspace=W / "exp")
experiment_path = write_experiment(W / "exp.json", run)
experiment = load_experiment_result(experiment_path)

# ---- one real skill candidate ----------------------------------------------
live = {s["trigger"]: int(s["version"]) for s in repo.list_skills(live_only=True)}
analysis = analyse(experiment, experiment_path=str(experiment_path), live_skills=live)
skill_proposals = [p for p in analysis.proposals if p.type == "skill"]
check("未泄露：正式 skill 库仍为空",
      # 显式要全量：默认视图现在只含 production，而这里要断言的是"一行都没有"。
      repo.list_skills(status=None, origin=None) == [],
      "候选是在隔离环境里推导的，没有写进 harness 自己的库")
check("至少生成一个真实 Skill Candidate", bool(skill_proposals),
      f"{len(skill_proposals)} 个，另有 {len(analysis.unactionable)} 个失败被明确列为无法提议")

for proposal in skill_proposals:
    record_proposal(repo, proposal)
stored = repo.get_candidate(skill_proposals[0].candidate_id, skill_proposals[0].version)

# ---- 1. parent Skill/version -----------------------------------------------
check("1. 有明确 parent Skill/version",
      "parent_version" in stored and isinstance(stored["parent_version"], int),
      f"{stored['candidate_id']} v{stored['version']} ← 母版本 v{stored['parent_version']}")

# ---- 2. source run ---------------------------------------------------------
check("2. 有 source run", bool(stored["source_runs"]),
      "、".join(stored["source_runs"]))

# ---- 3. source failure -----------------------------------------------------
check("3. 有 source failure", bool(stored["source_failures"]),
      "、".join(stored["source_failures"]))

# ---- 4. 不覆盖正式 Skill ---------------------------------------------------
# 给正式库放一个同名 trigger 的规程，确认候选不碰它
repo.add_skill("regression", "正式规程", "这是生产在用的", "prod-run")
before_skills = json.dumps(repo.list_skills(status=None, origin=None),
                              sort_keys=True, ensure_ascii=False)

# ---- 5. 不修改 baseline ----------------------------------------------------
baseline = write_baseline(W / "base.json",
                          run_benchmark(SUITE, workspace=W / "bws", tier="regression"),
                          source=SUITE)
before_baseline = baseline.read_bytes()

settled = evaluate_candidate(repo, record_from_row(stored),
                             workspace=W / "cand-ws", suite=SUITE,
                             baseline=str(baseline))

check("4. 不覆盖正式 Skill",
      json.dumps(repo.list_skills(status=None, origin=None),
                 sort_keys=True, ensure_ascii=False) == before_skills,
      f"正式库仍有 {len(repo.list_skills(live_only=True))} 条，内容未变")
check("5. 不修改 baseline", baseline.read_bytes() == before_baseline)

# ---- 6. 独立 workspace -----------------------------------------------------
dbs = sorted((W / "cand-ws").rglob("wfos.db"))
check("6. 使用独立 workspace", bool(dbs),
      f"{len(dbs)} 个隔离库，均在候选自己的工作区下")

# ---- 7. 用 P2 Evaluator / 8. 用 P3 Compare ---------------------------------
verdict = settled.verdict
check("7. 使用 P2 Evaluator",
      bool(verdict.get("axes")) and bool(verdict.get("taskVerdicts")),
      "候选的每个任务都带五轴判定")
check("8. 使用 P3 Compare", "comparison" in verdict and bool(verdict["comparison"]),
      f"比较计数 {verdict['comparison'].get('counts')}")

# ---- 9. 明确状态 -----------------------------------------------------------
check("9. 得到明确状态", settled.status in STATUSES and settled.status != "PROPOSED",
      f"{settled.status}：{verdict.get('reason', '')[:40]}")

# ---- 10. 不会自动晋升 ------------------------------------------------------
check("10. 即使表现更好也不会自动晋升",
      settled.status == STATUS_PASSED and repo.list_skills(live_only=True)
      and [s["title"] for s in repo.list_skills(live_only=True)] == ["正式规程"],
      "PASSED 之后正式库仍是原来那一条 —— 晋升是 P6 里人的决定")

# ---- CLI 的 --json 是纯 JSON ------------------------------------------------
env = {**os.environ, "WFOS_DATA": str(W / "data"), "PYTHONIOENCODING": "utf-8"}
for args in (["rsi", "list", "--json"], ["rsi", "show", stored["candidate_id"], "--json"]):
    proc = subprocess.run([sys.executable, "-m", "wfos", *args],
                          capture_output=True, cwd=str(ROOT),
                          encoding="utf-8", errors="replace", env=env)
    try:
        json.loads(proc.stdout)
        parsed = True
    except Exception:  # noqa: BLE001
        parsed = False
    check(f"CLI {' '.join(args[:2])} --json 合法", parsed)

print()
print(f"  provenance 链: {stored['candidate_id']} v{stored['version']}"
      f" → 实验 {pathlib.Path(stored['source_experiment']).name}"
      f" → 运行 {stored['source_runs'][0][:8]}"
      f" → 失败 {stored['source_failures'][0]}"
      f" → 母版本 v{stored['parent_version']}")
print(f"  状态: PROPOSED → EVALUATING → {settled.status}")
print()
print("全部通过" if ok else "有未通过项")
raise SystemExit(0 if ok else 1)
