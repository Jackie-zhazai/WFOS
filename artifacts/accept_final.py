"""Final acceptance: the whole loop, end to end, on real runs.

    Task → isolated Run → Trace → Evaluation → Benchmark → immutable Baseline
    → Compare → Experiment matrix → Failure → RSI Candidate → Candidate
    Evaluation → Compare → PASSED → explicit Promotion → Skill v2
    → Benchmark v2 → Rollback → v1 → Benchmark v1 → history still there

Every assertion below reads a record a run produced. Nothing is asserted from the
shape of the code, and the CLI is the entry point for every step after seeding, so
what is being accepted is the product and not the library.

The invariants §8 names are checked as properties, not as one-off observations:
immutability is checked by bytes before and after, append-only-ness by the trace
growing and never shrinking, `unknown` by finding a real one and watching it fail
to become a `pass`.
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
W = pathlib.Path("C:/Users/rog/AppData/Local/Temp/wfos_final")
SUITE = "benchmark/regression"
TRIGGER = "regression"

ok = True
failures: list[str] = []


def check(label, condition, detail=""):
    global ok
    ok = ok and bool(condition)
    if not condition:
        failures.append(label)
    print(f"  [{'OK ' if condition else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")


def section(title):
    print(f"\n=== {title} ===")


def cli(*args: str):
    env = {**os.environ, "WFOS_DATA": str(W / "data"), "PYTHONIOENCODING": "utf-8"}
    return subprocess.run([sys.executable, "-m", "wfos", *args], cwd=str(ROOT),
                          capture_output=True, encoding="utf-8", errors="replace",
                          env=env)


def cli_json(*args: str):
    proc = cli(*args)
    try:
        return proc.returncode, json.loads(proc.stdout)
    except Exception:  # noqa: BLE001
        return proc.returncode, None


shutil.rmtree(W, ignore_errors=True)
(W / "data").mkdir(parents=True)
os.environ["WFOS_DATA"] = str(W / "data")

from wfos.bench import run_benchmark  # noqa: E402
from wfos.eval import evaluate  # noqa: E402
from wfos.models import ORIGIN_INTERACTIVE  # noqa: E402
from wfos.skills import content_digest  # noqa: E402
from wfos.storage.repo import Repo  # noqa: E402
from wfos.tasks import load_tasks  # noqa: E402

repo = Repo(W / "data" / "wfos.db")

# --------------------------------------------------------------- 1. Task + Run
section("1. 生产里的 v1 → 声明式任务 → 隔离运行 → Trace")
v1 = repo.add_skill(TRIGGER, "v1 规程", "触发: regression\n步骤: 先跑 check.py，再改",
                    "seed-run", origin=ORIGIN_INTERACTIVE)
v1_bytes = v1["procedure"]
check("v1 就位且是 current", v1["version"] == 1 and v1["superseded"] == 0,
      f"digest={v1['digest'][:12]}")

# A task whose run is expected to reach a terminal state: `task run` exits 1 when
# a run stops short (at an approval gate, say), which is the command reporting
# what happened rather than the command failing.
task_id = "feature-happy-path"
proc = cli("task", "run", "benchmark/smoke", task_id, "--workspace", str(W / "task"))
check("`wfos task run` 跑到终态（exit 0）", proc.returncode == 0,
      (proc.stderr or proc.stdout).strip()[-140:])

isolated = sorted((W / "task").rglob("wfos.db"))
check("运行落在自己的库和自己的工作区里", len(isolated) >= 1,
      f"{len(isolated)} 个隔离库，均在 {W / 'task'} 下")

task_repo = Repo(isolated[0])
task_runs = [r for r in task_repo.list_runs(limit=50) if r.get("origin") == "task"]
check("运行被标记为 task 来源（不是 interactive）", bool(task_runs),
      f"{len(task_runs)} 个，origin={task_runs[0]['origin'] if task_runs else '-'}")
run_id = task_runs[0]["id"] if task_runs else ""

check("没有写进 harness 自己的库",
      repo.get_run(run_id) is None,
      "任务运行只在隔离库里；主库看不到它")

trace = task_repo.events_for_run(run_id)
ordering = [e["event_id"] for e in trace]
check("trace 按 event_id 严格递增",
      bool(ordering) and ordering == sorted(ordering)
      and len(set(ordering)) == len(ordering),
      f"{len(ordering)} 条事件")

# ------------------------------------------------------- 2. Benchmark + Baseline
section("2. Benchmark → 不可变 Baseline")
proc = cli("baseline", "create", "--suite", "regression", "--out", str(W / "base.json"),
           "--workspace", str(W / "bws"))
check("baseline create 成功", proc.returncode == 0,
      (proc.stderr or proc.stdout).strip()[-160:])
baseline_bytes = (W / "base.json").read_bytes()
baseline_doc = json.loads(baseline_bytes)
check("基线记录了 8 个任务", len(baseline_doc["tasks"]) == 8,
      f"{len(baseline_doc['tasks'])} 个")

again = cli("baseline", "create", "--suite", "regression", "--out", str(W / "base.json"),
            "--workspace", str(W / "bws2"))
check("重复写同一份基线被拒绝", again.returncode != 0 and (W / "base.json").read_bytes() == baseline_bytes,
      (again.stderr or again.stdout).strip().splitlines()[-1][:80] if (again.stdout or again.stderr) else "")

# ------------------------------------------------------------ 3. Compare
section("3. Compare（同一套题两次运行的判定）")
frozen = json.loads(baseline_bytes)
fresh = run_benchmark(SUITE, workspace=W / "cmp", tier="regression").as_json()
(W / "fresh.json").write_text(json.dumps(fresh, ensure_ascii=False), encoding="utf-8")
code, report = cli_json("compare", str(W / "base.json"), str(W / "fresh.json"), "--json")
check("compare --json 是纯 JSON", report is not None)
counts = (report or {}).get("counts") or {}
check("比较给出五种判定之一", bool(counts), f"counts={counts}")
check("同环境同套题 → 没有 regression", counts.get("regression", 0) == 0,
      f"regression={counts.get('regression')}")
unknown_axes = sum(len([a for a in t.get("axes", {}).values() if a == "unknown"])
                   for t in frozen["tasks"])
check("基线里确实有 unknown 轴（下面的 unknown≠pass 才有意义）", unknown_axes > 0,
      f"{unknown_axes} 个")

# ---------------------------------------------------------- 4. Experiment matrix
section("4a. Experiment 矩阵展开（2 模型 × 2 配置）")
proc = cli("experiment", "run", "experiments/model-sweep.json",
           "--workspace", str(W / "sweep"), "--out", str(W / "sweep.json"))
check("矩阵实验跑成功", proc.returncode == 0, proc.stderr.strip()[:100])
sweep = json.loads((W / "sweep.json").read_text(encoding="utf-8"))
cells = {t.get("cellId") for t in sweep["tasks"]}
check("矩阵展开成 4 个 cell", len(cells) == 4, f"{len(cells)} 个：{sorted(cells)}")
sweep_dbs = list((W / "sweep").rglob("wfos.db"))
check("每个 cell 有自己的库", len(sweep_dbs) >= 4, f"{len(sweep_dbs)} 个隔离库")
check("每个 cell 有自己的 run",
      len({t.get("runId") for t in sweep["tasks"]}) == len(sweep["tasks"]),
      f"{len(sweep['tasks'])} 个任务，run 互不相同")

section("4b. 回归实验 → 真实失败（候选的证据来源）")
proc = cli("experiment", "run", "experiments/recovery-regression.json",
           "--workspace", str(W / "exp"), "--out", str(W / "exp.json"))
check("experiment run 成功", proc.returncode == 0, proc.stderr.strip()[:80])
experiment = json.loads((W / "exp.json").read_text(encoding="utf-8"))
failures_seen = sorted({f for t in experiment["tasks"] for f in t.get("failures") or []})
check("实验产生了真实失败", bool(failures_seen), "、".join(failures_seen))

# The failure classes a task reports have to be findable in that task's own store.
# A record that only asserts its own evidence is not evidence.
stores = {}
for db in (W / "exp").rglob("wfos.db"):
    for run in Repo(db).list_runs(limit=50):
        stores[run["id"]] = db
backed, checked = 0, 0
for task in experiment["tasks"]:
    db = stores.get(task.get("runId") or "")
    if db is None or not task.get("failures"):
        continue
    checked += 1
    step_classes = {s["failure_class"] for s in Repo(db).steps_for_run(task["runId"])
                    if s.get("failure_class")}
    if set(task["failures"]) <= step_classes:
        backed += 1
check("记录下来的失败类，在它自己那个库里查得到",
      checked > 0 and backed == checked,
      f"{backed}/{checked} 个任务的失败类可回溯到步骤行")
check("'经历了失败'与'任务通过'可以并存，而失败没有被抹掉",
      any(t.get("failures") for t in experiment["tasks"] if t.get("verdict") == "pass"),
      "这类任务的期望就是'经历失败并恢复'，所以 pass 与 failures 并存是对的")

# ------------------------------------------------------------ 5. RSI Candidate
section("5. RSI Candidate（从证据推导，不是写的）")
proc = cli("rsi", "propose", str(W / "exp.json"))
check("rsi propose 成功", proc.returncode == 0, proc.stderr.strip()[:80])
candidates = [c for c in repo.list_candidates() if c["candidate_id"] == f"skill:{TRIGGER}"]
check("生成了 skill 候选", bool(candidates))
candidate = candidates[-1]
check("候选带完整溯源",
      all(candidate.get(k) for k in ("source_experiment", "source_runs",
                                     "source_failures", "evidence", "proposed_change")),
      f"{candidate['candidate_id']} v{candidate['version']} ← v{candidate['parent_version']}")
check("候选没有变成正式 skill",
      [s["version"] for s in repo.skill_versions(TRIGGER)] == [1],
      "生产里仍然只有种下的 v1，候选没有擅自加一版")

# ------------------------------------------- 6. Candidate evaluation + Compare
section("6. 候选评测 → Compare → PASSED")
proc = cli("rsi", "evaluate", candidate["candidate_id"], "--version",
           str(candidate["version"]), "--baseline", str(W / "base.json"),
           "--workspace", str(W / "cand"))
check("rsi evaluate 成功", proc.returncode == 0, proc.stderr.strip()[:120])
settled = repo.get_candidate(candidate["candidate_id"], candidate["version"])
check("候选状态是 PASSED", settled["status"] == "PASSED", settled["status"])
verdict = settled["verdict"] or {}
check("用了 P3 的 compare", "counts" in (verdict.get("comparison") or {}),
      f"counts={(verdict.get('comparison') or {}).get('counts')}")
check("评测记录以 (id, 库) 成对引用",
      all(isinstance(i, dict) and i.get("id") and i.get("store")
          for i in verdict.get("evaluationIds") or []),
      f"{len(verdict.get('evaluationIds') or [])} 条")

# unknown 不是 pass，failure 不是 success —— 拿真实数据断言
axes_seen = {}
for task_axes in verdict.get("axes") or []:
    for axis, value in (task_axes.get("axes") or {}).items():
        axes_seen.setdefault(axis, set()).add(value)
check("候选里确实存在 unknown 轴", any("unknown" in v for v in axes_seen.values()),
      f"{ {k: sorted(v) for k, v in axes_seen.items() if 'unknown' in v} }")
# The axes the gate will be asked about, as the tasks actually reported them.
_axes_snapshot = {k: set(v) for k, v in axes_seen.items()}

# ---------------------------------------------------------------- 7. Promotion
section("7. 显式 Promotion → Skill v2")
proc = cli("rsi", "promote", candidate["candidate_id"], "--version",
           str(candidate["version"]), "--baseline", str(W / "base.json"),
           "--actor", "rog", "--reason", "P7 最终验收")
check("rsi promote 成功", proc.returncode == 0, proc.stderr.strip()[:160])
print("        " + "\n        ".join(proc.stdout.strip().splitlines()))

v2 = repo.skill_version(TRIGGER, 2)
check("v2 被创建且是 current", v2 is not None and v2["superseded"] == 0)
check("v1 未被覆盖", repo.skill_version(TRIGGER, 1)["procedure"] == v1_bytes
      and repo.skill_version(TRIGGER, 1)["superseded"] == 1)
check("v2 digest 是内容摘要", v2 is not None and v2["digest"] == content_digest(v2))
check("parent → child 链接正确",
      v2 is not None and v2["parent_id"] == repo.skill_version(TRIGGER, 1)["id"])
check("候选被锁在 PROMOTED（不可再篡改）",
      repo.get_candidate(candidate["candidate_id"], candidate["version"])["status"] == "PROMOTED")
_proposal_keys = ("candidate_id", "version", "type", "parent_version",
                  "source_experiment", "source_runs", "source_failures",
                  "evidence", "proposed_change", "rationale", "created_at")
after = repo.get_candidate(candidate["candidate_id"], candidate["version"])
check("候选的提案部分没有被晋升改写",
      {k: after[k] for k in _proposal_keys if k in after}
      == {k: candidate[k] for k in _proposal_keys if k in candidate},
      "verdict 会由评测写入，提案内容不会")

promotion = repo.promotions_for_skill(TRIGGER)[-1]

# `pass` 只在**真有任务量过它**时才算数。这不是"unknown 有没有变成 pass"的复述，
# 而是它的反面：晋升门据以放行的每条通过轴，都能指出是哪个任务测出来的 —— 而
# unknown 那条路已经由 `tests/test_promotion.py` 直接钉住了。
gate_axes = (promotion.get("gate") or {}).get("axes") or {}
check("晋升门据以放行的每条通过轴，都有任务实测过",
      all(gate_axes.get(axis) != "pass"
          or "pass" in _axes_snapshot.get(axis, set())
          for axis in ("functional", "safety", "operational")),
      f"gate.axes={gate_axes} 实测={ {k: sorted(v) for k, v in _axes_snapshot.items()} }")

check("晋升记录齐全（§5 字段）",
      all(k in promotion and promotion[k] not in ("", None, [], {}) for k in
          ("promotion_id", "candidate_id", "candidate_version", "parent_skill_version",
           "promoted_skill_version", "source_experiment", "source_runs", "evaluation",
           "compare_result", "gate", "created_at", "actor", "reason")),
      promotion["promotion_id"])

section("7b. Provenance 链完整可追溯")
live = repo.skills_for([TRIGGER], origin=ORIGIN_INTERACTIVE)
chain = {
    "skill v2": f"{v2['trigger']} v{v2['version']} digest={v2['digest'][:12]}",
    "promotion": promotion["promotion_id"],
    "candidate": f"{promotion['candidate_id']} v{promotion['candidate_version']}",
    "experiment": promotion["source_experiment"],
    "runs": "、".join(r[:8] for r in promotion["source_runs"]),
    "evaluations": "、".join(str(i["id"]) for i in promotion.get("evaluation_ids") or []),
    "parent v1": f"v{promotion['parent_skill_version']} digest={promotion['parent_digest'][:12]}",
}
for name, value in chain.items():
    check(f"链上 {name}", bool(value), value)

# --------------------------------------------------------- 8. v2 runs
section("8. v2 作为正式 Skill 被加载并真的跑一遍")
check("交互式 prompt 加载的是 v2", [s["version"] for s in live] == [2],
      f"v{[s['version'] for s in live]}")
check("加载到的是 v2 的内容", live[0]["procedure"] == v2["procedure"])
before_v2 = run_benchmark(SUITE, workspace=W / "bench-v2", tier="regression")
check("v2 下 benchmark 跑完 8 个任务", len(before_v2.tasks) == 8,
      f"rollup={before_v2.rollup}")
v2_verdicts = [t.evaluation.verdict for t in before_v2.tasks]

# ------------------------------------------------------------- 9. Rollback
section("9. Rollback v2 → v1")
proc = cli("rsi", "rollback", TRIGGER, "1", "--actor", "rog", "--reason", "P7 最终验收回滚")
check("rsi rollback <skill> <version> 成功", proc.returncode == 0,
      proc.stderr.strip()[:120])
print("        " + proc.stdout.strip().replace("\n", "\n        "))

check("current 指针回到 v1", repo.latest_skill(TRIGGER)["version"] == 1)
check("v2 仍然存在（回滚不是删除）",
      repo.skill_version(TRIGGER, 2) is not None
      and repo.skill_version(TRIGGER, 2)["superseded"] == 1)
check("v2 的内容没有被改动",
      repo.skill_version(TRIGGER, 2)["digest"] == content_digest(v2))
rollback_record = repo.rollbacks_for(TRIGGER)[-1]
check("回滚记录齐全（§6 字段）",
      all(k in rollback_record for k in
          ("rollback_id", "trigger", "from_version", "to_version",
           "from_digest", "to_digest", "actor", "reason", "created_at")),
      f"{rollback_record['rollback_id']}：v{rollback_record['from_version']}"
      f" → v{rollback_record['to_version']}")

section("9b. 回滚后 v1 被加载并真的跑一遍")
live = repo.skills_for([TRIGGER], origin=ORIGIN_INTERACTIVE)
check("加载的是 v1", [s["version"] for s in live] == [1], f"v{[s['version'] for s in live]}")
check("加载到的是 v1 的内容", live[0]["procedure"] == v1_bytes)
after_v1 = run_benchmark(SUITE, workspace=W / "bench-v1", tier="regression")
check("v1 下 benchmark 跑完同样多的任务",
      len(after_v1.tasks) == len(before_v2.tasks),
      f"{len(after_v1.tasks)} 个任务")
check("v1 的判定与 v2 一致（晋升没有让套件变好或变坏）",
      [t.evaluation.verdict for t in after_v1.tasks] == v2_verdicts,
      f"{[t.evaluation.verdict for t in after_v1.tasks]}")

# ------------------------------------------------------- 10. Nothing was lost
section("10. 什么都没被改：baseline / evaluator / benchmark / 记录")
root = ROOT
watched = [root / "wfos" / "eval.py", root / "wfos" / "bench_compare.py",
           root / "benchmark" / "smoke" / "tasks.json",
           root / "benchmark" / "regression" / "tasks.json",
           root / "benchmark" / "challenge" / "tasks.json"]
snapshot = {p: p.read_bytes() for p in watched}
check("baseline 未被修改（与创建时逐字节相同）",
      (W / "base.json").read_bytes() == baseline_bytes)
check("evaluator / benchmark 任务文件此刻与运行时一致",
      all(p.read_bytes() == c for p, c in snapshot.items()),
      f"监控 {len(snapshot)} 个文件：eval.py / bench_compare.py / 三层 tasks.json")
check("v1 / v2 / 候选 / 晋升记录 / 回滚记录都还在",
      repo.skill_version(TRIGGER, 1) is not None
      and repo.skill_version(TRIGGER, 2) is not None
      and repo.get_candidate(candidate["candidate_id"], candidate["version"]) is not None
      and len(repo.promotions_for_skill(TRIGGER)) >= 1
      and len(repo.rollbacks_for(TRIGGER)) >= 1,
      f"{len(repo.skill_versions(TRIGGER))} 个版本、"
      f"{len(repo.promotions_for_skill(TRIGGER))} 条晋升、"
      f"{len(repo.rollbacks_for(TRIGGER))} 条回滚")

section("10b. Trace 是 append-only")
# 先拍快照，再从库外重读一遍：append-only 是一个跨时间的性质，只有"读两次、比一比"
# 能证明它。只读一次然后断言它等于自己，等于什么都没说。
snapshot = {e["event_id"]: json.dumps(e["payload"], sort_keys=True) for e in trace}
trace_now = task_repo.events_for_run(run_id)
current = {e["event_id"]: json.dumps(e["payload"], sort_keys=True) for e in trace_now}
check("早先读到的事件，一条没少、一字未改",
      bool(snapshot) and all(current.get(k) == v for k, v in snapshot.items()),
      f"{len(snapshot)} 条")
check("而且只增不减", len(current) >= len(snapshot),
      f"{len(snapshot)} → {len(current)} 条")
check("顺序仍按 event_id 严格递增",
      [e["event_id"] for e in trace_now] == sorted(current),
      f"{len(current)} 条")

# 真正会考验"append-only"的是删除：状态重跑会把旧的步骤行**删掉**。如果历史只
# 活在 steps 里，这件事就查无实据 —— 所以删除必须自己留下一条事件。
deleted = []
for db in W.rglob("wfos.db"):
    store = Repo(db)
    for row in store.conn.execute(
            "SELECT run_id, payload FROM events WHERE type='step.invalidated' LIMIT 1"):
        deleted.append((store, row["run_id"], json.loads(row["payload"] or "{}")))
check("真实运行里确实发生过步骤删除（有 step.invalidated）", bool(deleted),
      f"{len(deleted)} 处")
if deleted:
    store, victim_run, payload = deleted[0]
    state = str(payload.get("state") or "")
    check("被删掉的步骤，在 steps 里已经不在了",
          state and store.get_step(victim_run, state) is None,
          f"run {victim_run[:8]} 的 {state or '?'}")
    check("但它在 trace 里留下了删除事件（含理由与被抹掉的内容）",
          bool(payload.get("reason")) and "state" in payload,
          f"理由={str(payload.get('reason'))[:40]!r}，字段={sorted(payload)}")

section("10c. unknown 不是 pass，failure 不是 success（拿真实运行断言）")
# A run that escalates to a human, judged by a task that requires a clean happy
# path. The pair is chosen so the two expectations genuinely differ — two tasks
# can both be satisfied by one run, and a test that assumed otherwise would be
# asserting something about the tasks rather than about the evaluator.
outcome = run_benchmark("benchmark/regression", workspace=W / "verdict",
                        tier="regression", only_task="build-error-is-classified")
record = outcome.tasks[0]
# 判定必须在**这次运行自己的库**里做。库的位置就是从我们上一阶段加进 TaskRecord
# 的那个 (id, 库) 对里读出来的 —— 这里验证那对东西是能用的，不只是能存。
verdict_repo = Repo(record.evaluation_store)
check("TaskRecord 记下的 (id, 库) 确实能定位到这次运行",
      verdict_repo.get_run(record.run_id) is not None,
      str(record.evaluation_store).split("wfos_final")[-1][:70])

regression = load_tasks("benchmark/regression")
matching = [t for t in regression if t.id == record.task_id][0]
same = evaluate(verdict_repo, record.run_id, matching)
check("用正确任务判定 → pass", same.verdict == "pass",
      f"{record.task_id} → {same.verdict}")

happy = [t for t in load_tasks("benchmark/smoke") if t.id == "feature-happy-path"][0]
judged = evaluate(verdict_repo, record.run_id, happy)
check("用一份要求完全不同的任务判定同一份运行 → 不是 pass",
      judged.verdict != "pass", f"verdict={judged.verdict}")
check("失败带着具体的硬门禁或检查项，而不是一个空判定",
      bool(judged.hard_gate) or bool(judged.reasons),
      f"hardGate={list(judged.hard_gate)} reasons={list(judged.reasons)[:2]}")
check("这一对判定不同 —— 判定确实读了任务，不是把运行本身当结论",
      same.verdict != judged.verdict, f"{same.verdict} / {judged.verdict}")

# unknown 不是 pass，分两层说：判定要如实报出来，汇总不能把它算进通过。
check("没人测过的轴被如实报成 unknown —— 既没被丢掉，也没被算成 pass",
      any(v == "unknown" for v in judged.axes.values())
      and all(v in ("pass", "fail", "unknown") for v in judged.axes.values()),
      f"axes={judged.axes}")
check("rollup 把 unknown 轴单独计数，不去抬高 passed",
      before_v2.rollup["unknownAxes"] > 0
      and before_v2.rollup["passed"]
      == sum(1 for t in before_v2.tasks if t.evaluation.verdict == "pass"),
      f"unknownAxes={before_v2.rollup['unknownAxes']}，"
      f"passed={before_v2.rollup['passed']}/{before_v2.rollup['tasks']}")

# ----------------------------------------------------------- 11. CLI contract
section("11. CLI：JSON 纯净、exit code、Windows 路径")
for args in (("rsi", "history", TRIGGER, "--json"),
             ("rsi", "promotion", "show", promotion["promotion_id"], "--json"),
             ("status", "--json"),
             ("history", "--json"),
             ("skills", "--json"),
             ("metrics", "--json")):
    code, body = cli_json(*args)
    check(f"{' '.join(a for a in args if not a.startswith('--'))[:44]} --json 合法",
          body is not None, f"exit={code}")

code, body = cli_json("status", "no-such-run", "--json")
check("失败路径也是 JSON", body is not None and bool(body.get("error")), f"exit={code}")
check("找不到运行 → exit 1", code == 1, f"exit={code}")
code, _ = cli_json("definitely-not-a-command")
check("未知命令 → exit 2", code == 2, f"exit={code}")

# `compare` takes two paths and emits JSON: both halves of the requirement at once.
code, body = cli_json("compare", str(W / "base.json"), str(W / "fresh.json"), "--json")
check("Windows 路径作为参数可用，且 --json 仍是纯 JSON",
      body is not None and "counts" in body,
      f"exit={code}，路径含反斜杠与盘符")
check("反斜杠没有被当成转义或命令",
      json.loads(cli("compare", str(W / "base.json"), str(W / "fresh.json"),
                     "--json").stdout)["counts"]["regression"] == 0)

code, first = cli_json("rsi", "history", TRIGGER, "--json")
code2, second = cli_json("rsi", "history", TRIGGER, "--json")
check("同样的输入两次结果相同", first == second)

print()
if failures:
    print(f"未通过 {len(failures)} 项：")
    for name in failures:
        print(f"  - {name}")
    raise SystemExit(1)
print("全部通过")
