"""Command-line interface.

Commands
  run <text> [--kind feature|bugfix] [--title T]   create + advance a run
  status [run_id]                                  run status / transition trace
  result <run_id>                                  steps, evidence, wiki of a run
  trace show <run_id>                              append-only trace (keeps deleted steps)
  evaluate <run_id> <任务集> <任务id>              按任务判定一次运行（hard gate 优先）
  experiment run|show <文件>                     声明式实验：矩阵展开为多个独立 run
  rsi promote|rollback|history|promotion show    显式晋升 / 回滚 / 版本历史（P6）
  task run <path> <task_id> [--workspace W]        run one declared task in isolation
  approve <approval_id> [--note N]                 approve a pending approval
  reject <approval_id> [--note N]                  reject a pending approval
  pending                                          list pending approvals / blocked runs
  resume <run_id>                                  resume a blocked/paused run
  resume-child <child_run_id>                      resume the parent after a child bugfix
  cancel <run_id>                                  cancel a run
  mcp [--role R] [--run-id ID]                     serve MCP over stdio to an external client
  history [--limit N]                              recent runs
  metrics [run_id] [--limit N]                     token/latency/cost usage
  baseline run|check [--cases P] [--file P] [--case ID]
                                                   frozen whole-flow baseline
  wiki search <query> [--kind k] [--tags t]        search the LLM wiki
  wiki list [--kind k] [--status s]                list wiki entries
  wiki promote <id> [--by NAME]                    promote a verified entry to authoritative
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .config import load_config
from .harness.orchestrator import Harness, RunLockedError
from .llm.base import MissingCredentialError
from .mcp.client import ToolGateway
from .mcp.server import EXTERNAL_ROLES, WfosMcpServer
from .metrics import aggregate, render, run_metrics
from .storage.repo import Repo
from .wiki.wiki import WikiClient


def build_harness(cfg=None) -> Harness:
    cfg = cfg or load_config()
    repo = Repo(cfg.db_path, redact=cfg.harness.redact_persistence)

    def approval_checker(run_id, tool, args):  # workspace.delete gate
        if not run_id:
            return False
        key = (args or {}).get("path") or ""
        return (repo.has_approved_approval(run_id, f"{tool}:{key}")
                or repo.has_approved_approval(run_id, tool))

    server = WfosMcpServer(cfg, repo, approval_checker=approval_checker)
    gateway = ToolGateway(server, repo)
    wiki = WikiClient(repo)
    return Harness(cfg, repo, gateway, wiki)


# ------------------------------------------------------------------- rendering
# Whether this invocation asked for JSON, and what it has said so far. Module
# state rather than a parameter because the contract it enforces is global to the
# process: stdout carries exactly one JSON document, whichever command ran.
_JSON_MODE = False
_JSON_WRITTEN = False
_LAST_LINE = ""


def _p(*args, **kw):
    """Human-facing output.

    In `--json` mode this goes to **stderr**. The document owns stdout, and a
    progress line sharing that stream is what breaks a caller's parser — but the
    information is not dropped, it moves. Nothing else in the file has to know:
    a command that prints prose and a command that prints records use the same
    call, and only the destination differs.
    """
    global _LAST_LINE
    text = " ".join(str(a) for a in args)
    _LAST_LINE = text
    print(*args, file=sys.stderr if _JSON_MODE else sys.stdout, **kw)


def _wants_json(args) -> bool:
    """Whether this invocation asked for JSON.

    `getattr` rather than `args.json` because the command functions are also
    called directly — by tests, and by anything that drives them — where the
    argument object may be absent entirely. An invocation with no arguments is
    not an invocation that asked for JSON.
    """
    return bool(getattr(args, "json", False))


def _json(payload) -> None:
    """The document. Always stdout, always the whole of it."""
    global _JSON_WRITTEN
    _JSON_WRITTEN = True
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _failure_reason(harness: Harness, run: dict) -> str:
    """Why a run stopped, in whichever voice actually has the answer.

    Two unrelated things end a run as `failed`, and they are treated completely
    differently by whoever reads the output:

      * **the workflow's own verdict** — the model looked, and concluded the
        problem cannot be located or that stopping beats guessing. There is
        nothing to fix; the run did its job.
      * **the harness giving up** — an exception escaped a state, or a state
        suggested a transition its machine does not allow. That one is a defect
        somewhere and wants attention.

    Printing both as `状态=failed` makes a considered conclusion read like a
    crash, so the two are named apart. The verdict is read from the step the
    model produced (`next_step.reason`), because the transition row only ever
    carries the machine's own "合法转换" — the judgement never reaches it.
    """
    if run.get("error"):
        return f"执行出错: {run['error']}"

    steps = harness.repo.steps_for_run(run["id"])
    judged = ((steps[-1].get("output_json") or {}).get("next_step") or {}) if steps else {}
    if judged.get("suggested_state") == "failed" and judged.get("reason"):
        return f"流程判定终止: {judged['reason']}"

    for transition in reversed(harness.repo.transitions(run["id"])):
        if transition["to_state"] == "failed" and transition.get("reason"):
            return f"迁移被拒: {transition['reason']}"
    return "（无终止原因 —— 运行以 failed 结束，但记录里没有说明）"


def _show_run(harness: Harness, run: dict) -> None:
    kind_label = {"feature": "功能开发", "bugfix": "问题修复"}.get(
        run["kind"], "交互会话")
    _p(f"运行 {run['id']}  [{kind_label}] 状态={run['status']}")
    if run["status"] == "failed":
        _p(f"  终止原因: {_failure_reason(harness, run)}")
        _p(f"  完整推理: wfos trace show {run['id']}")
    _p(f"  标题: {run['title']}")
    _p(f"  状态机当前状态: {run['state']}")
    if run.get("error"):
        _p(f"  错误: {run['error']}")
    if run["status"] == "waiting_approval":
        for a in harness.repo.pending_approvals_for_run(run["id"]):
            _p(f"  待审批: {a['id']}  {a['action']}  (风险 {a['risk_level']})")
    if run["status"] == "waiting_child":
        _p(f"  子流程: {run['payload'].get('child_run_id')}")
    resume_status = (run.get("payload") or {}).get("resume_status")
    if resume_status:
        detail = (run["payload"].get("resume_detail") or {}).get("reason", "")
        forced = "（已显式接受过期）" if run["payload"].get("force_stale") else ""
        _p(f"  恢复校验: {resume_status}{forced}  {detail}")
    violations = harness.repo.plan_violations_for_run(run["id"])
    if violations:
        _p("  越界写入待审批: " + "、".join(f"{v['tool']}:{v['path']}" for v in violations))
    events = sorted({t["security_event"] for t in harness.repo.tool_calls(run["id"])
                     if t.get("security_event")})
    if events:
        _p(f"  安全事件: {', '.join(events)}")
    for t in harness.repo.transitions(run["id"])[-8:]:
        _p(f"  转换: {t['from_state']} -> {t['to_state']}  ({t['reason']})")


# ----------------------------------------------------------------- async helpers
def _run(coro):
    """Run a Harness coroutine, reporting lease contention as a message.

    Contention is an expected outcome of two processes working the same database,
    not a defect — so it exits non-zero with the holder and the lease expiry, the
    two things needed to decide whether to wait. It needs its own exit code, so a
    caller can tell it apart from "not found" (1) and "bad arguments" (2).

    Every entry point that can reach `Harness.advance` goes through here, not
    just `advance` itself: `resume` and `resume_after_child` await it internally,
    so catching it only around `advance()` would leave those two leaking a
    traceback — the very thing this exists to prevent.
    """
    try:
        return asyncio.run(coro)
    except RunLockedError as e:
        _p(str(e))
        # `from None`: the message above already says everything worth saying, and
        # chaining would print a traceback for what is an expected outcome.
        raise SystemExit(3) from None


def _advance(harness: Harness, run_id: str) -> dict:
    return _run(harness.advance(run_id))


# -------------------------------------------------------------------- commands
def cmd_run(harness: Harness, args) -> int:
    text = " ".join(args.text)
    try:
        run = harness.create_run(text, kind=args.kind, title=args.title)
    except ValueError as e:
        hint = ("提示: 请求需含 '新增/实现/做一个' 等新功能词，"
                "或 '报错/故障/结果不对' 等诊断词。")
        if args.json:
            _json({"error": f"路由失败: {e}", "hint": hint})
        else:
            _p(f"路由失败: {e}")
            _p(hint)
        return 2
    if args.json:
        run = _advance(harness, run["id"])
        _json({"run": run, "pendingApprovals": harness.repo.pending_approvals_for_run(run["id"])})
        return 1 if run["status"] == "failed" else 0
    _p(f"已创建 {run['kind']} 运行 {run['id']}")
    run = _advance(harness, run["id"])
    _show_run(harness, run)
    if run["status"] == "waiting_approval":
        _p("> 运行已进入审批等待。用 `wfos approve <审批ID>` 批准后继续。")
    elif run["status"] == "waiting_child":
        child = harness.repo.get_run(run["payload"].get("child_run_id") or "")
        _p(f"> 已生成子回归流程 {child['id'] if child else '?'}。"
           f"子流程完成后用 `wfos resume-child <子ID>` 继续父流程。")
    # A run that ended `failed` exits non-zero. That was not the behaviour before
    # — it returned 0 whatever happened — but a script has no other way to tell
    # "the workflow concluded" from "it finished", and `waiting_approval` /
    # `waiting_child` deliberately keep 0: those are pauses a human resumes, not
    # outcomes.
    if run["status"] == "failed":
        return 1
    return 0


def cmd_chat(harness: Harness, args) -> int:
    """An interactive session: one run, one turn at a time.

    The only place in this file that reads stdin. Everything a turn actually does
    lives in `Harness.converse`, which takes its text as an argument — so a test
    drives the whole feature without a terminal, and this function is left with
    nothing but reading lines and printing answers.

    The session is a run, so it lands in `wfos history`, its turns are events, and
    `wfos trace show` reconstructs the conversation afterwards. It is **not** a
    state machine: `advance` refuses this kind outright.
    """
    once = getattr(args, "once", None)
    repo = harness.repo
    run = harness.create_run("（交互式会话）", kind="chat",
                             title=(once or "交互式会话")[:60])
    run_id = run["id"]

    try:
        if once:
            reporter = _tool_reporter()
            outcome = asyncio.run(harness.converse(run_id, once, on_tool=reporter))
            reporter.flush()
            print(outcome["reply"])
            _report_pending(outcome)
            return 0 if outcome["ok"] else 1

        print(f"交互会话 {run_id}   项目 {harness.cfg.project_root}",
              file=sys.stderr)
        print("直接说你想做什么；exit / quit / Ctrl+C 结束。\n", file=sys.stderr)
        reporter = _tool_reporter()

        async def session() -> None:
            # One event loop for the whole session, not one per turn: the gateway
            # caches its in-process MCP client per loop and rebuilds it whenever
            # the loop changes.
            while True:
                try:
                    line = input("wfos> ")
                except (EOFError, OSError, KeyboardInterrupt):
                    # OSError as well as EOFError: under a non-tty stdin (pytest's
                    # capture object, a closed pipe) reading raises OSError, and
                    # catching only EOFError turns a normal end-of-input into a
                    # crash. KeyboardInterrupt is Ctrl+C — a way out, not a fault.
                    _p("")
                    return
                text = (line or "").strip()
                if not text:
                    continue
                if text in ("exit", "quit", ":q"):
                    return
                try:
                    outcome = await harness.converse(run_id, text,
                                                     on_tool=reporter)
                except RunLockedError as e:
                    _p(f"（{e}）")
                    continue
                finally:
                    # Releases whatever run was still open when the turn ended.
                    reporter.flush()
                _p(outcome["reply"])
                _p("")
                _report_pending(outcome)

        asyncio.run(session())
        return 0
    finally:
        # The session ends however it ends — `exit`, EOF, Ctrl+C, an exception.
        # Closing the run here is what keeps it out of `waiting_approval`, which
        # would send `wfos approve` down `advance` into a machine that has no chat.
        if repo.get_run(run_id)["status"] not in ("completed", "failed", "cancelled"):
            repo.set_status(run_id, "completed")
        if not once:
            print(f"\n会话结束。`wfos trace show {run_id}` 可回看全过程。",
                  file=sys.stderr)


def _tool_text(name: str, args: dict) -> str:
    """How one tool call reads on a line: name plus the argument a person scans for.

    Picked by name rather than dumping the arguments — `path` and `pattern` are
    what a reader recognises, and a JSON blob per call turns the transcript into
    something you scroll past instead of read.
    """
    for key in ("path", "pattern", "command", "query", "text"):
        value = (args or {}).get(key)
        if value:
            text = str(value)
            # 48 keeps the whole line inside an 80-column terminal: the prefix and
            # the tool name take about thirty, and a line that wraps reads as two
            # actions rather than one.
            if len(text) > 48:
                text = text[:47] + "…"
            return f"{name} {text}"
    return name


def _tool_reporter(stream=None):
    """A callback that reports each *distinct* action once, as it happens.

    A model looking around a real project calls the same tool with the same
    arguments several times in a row — nine calls became five lines in one
    measured run, three of them the identical `workspace.search`. Printing each
    one is not information, it is noise with a timestamp.

    Runs are therefore held until they end, so the count is known before the line
    is written: `· workspace.read Equipment.cs ×3`. The cost is that a line
    appears when the *next* distinct action starts, which is also when there was
    anything new to say. `flush()` at the end of the turn releases the last one.
    """
    out = stream or sys.stderr
    state: dict = {"sig": None, "count": 0, "text": "", "ok": True}

    def flush() -> None:
        if not state["count"]:
            return
        suffix = f" ×{state['count']}" if state["count"] > 1 else ""
        mark = "·" if state["ok"] else "✗"
        print(f"   {mark} {state['text']}{suffix}", file=out, flush=True)
        state["sig"], state["count"] = None, 0

    def report(name: str, args: dict, payload: dict) -> None:
        ok = bool((payload or {}).get("ok", True))
        text = _tool_text(name, args)
        sig = (name, text, ok)
        if sig == state["sig"]:
            state["count"] += 1
            return
        flush()
        state.update(sig=sig, count=1, text=text, ok=ok)

    report.flush = flush          # type: ignore[attr-defined]
    return report


def _report_pending(outcome: dict) -> None:
    actions = outcome.get("pendingApprovals") or []
    if actions:
        _p("等待审批: " + "、".join(actions)
           + "   —— 用 `wfos pending` 查看，`wfos approve <ID>` 放行后再说一次。")


def cmd_status(harness: Harness, args) -> int:
    if args.run_id:
        run = harness.repo.get_run(args.run_id)
        if run is None:
            if args.json:
                _json({"error": f"未找到运行 {args.run_id}", "runId": args.run_id})
            else:
                _p(f"未找到运行 {args.run_id}")
            return 1
        if args.json:
            _json({"run": run,
                   "pendingApprovals": harness.repo.pending_approvals_for_run(run["id"])})
        else:
            _show_run(harness, run)
        return 0
    rows = harness.repo.list_runs(limit=20)
    if args.json:
        _json({"runs": rows})
        return 0
    if not rows:
        _p("尚无运行记录。")
        return 0
    _p(f"{'ID':<18}{'类型':<10}{'状态':<18}{'标题'}")
    for r in rows:
        _p(f"{r['id']:<18}{r['kind']:<10}{r['status']:<18}{r['title'][:40]}")
    return 0


def cmd_task(harness: Harness, args) -> int:
    """Run one declared task in its own environment.

    Not a replacement for `wfos run`: that one edits the project you pointed it
    at, which is what a person typing a request wants. This one refuses to touch
    anything outside a workspace of its own, so its result can be compared
    against a baseline later.
    """
    from .runner import RunEnvironment, TaskRunner
    from .tasks import load_tasks

    try:
        tasks = load_tasks(args.path)
    except (OSError, ValueError) as e:
        _p(f"任务集无法加载: {e}")
        return 2
    chosen = [t for t in tasks if t.id == args.task_id]
    if not chosen:
        _p(f"任务集里没有 {args.task_id!r}；可选: " + "、".join(t.id for t in tasks))
        return 2
    task = chosen[0]
    workspace = Path(args.workspace) if args.workspace else harness.cfg.data_dir / "task-runs"
    _p(f"任务 {task.id} v{task.version}（workflow={task.workflow} brain={task.brain}）")
    _p(f"隔离工作区: {workspace}")

    environment = RunEnvironment.create(
        workspace, name=args.name or task.id, fixture=task.fixture,
        live=task.is_live, pricing=task.pricing)
    runner = TaskRunner(environment)
    outcome = runner.run(task)

    _p(f"运行 {outcome.run_id}  状态={outcome.status} 终态={outcome.reached_terminal}")
    _p(f"  步骤 {outcome.steps}，session {outcome.session_id}")
    if outcome.failure_classes:
        _p("  失败类型: " + "、".join(outcome.failure_classes))
    if outcome.changed_paths:
        _p("  改动的文件: " + "、".join(outcome.changed_paths))
    if outcome.error:
        _p(f"  错误: {outcome.error}")
    _p("  提示: 判定这次运行是否达成任务目标，是 evaluator 的事，runner 不给自己打分。")
    # The run lives in its own database — that isolation is the point — so
    # `evaluate` cannot see it unless it is pointed there. Printed rather than
    # guessed: without this the loop `task run` → `evaluate` does not close.
    _p(f"  该运行的库目录: {environment.config.data_dir}")
    _p(f"  判定它: WFOS_DATA={environment.config.data_dir} wfos evaluate "
       f"{outcome.run_id} {args.path} {task.id}")
    return 0 if outcome.reached_terminal else 1


def cmd_evaluate(harness: Harness, args) -> int:
    """Judge one run against one task, and record the verdict.

    The task is required, not optional: a verdict is about *a run under a task
    version*, and judging without saying what was required would produce a number
    with nothing behind it.
    """
    from .eval import evaluate
    from .eval import record as record_evaluation
    from .eval import render as render_eval
    from .tasks import load_tasks

    try:
        tasks = load_tasks(args.path)
    except (OSError, ValueError) as e:
        if args.json:
            _json({"error": f"任务集无法加载: {e}", "path": args.path})
        else:
            _p(f"任务集无法加载: {e}")
        return 2
    chosen = [t for t in tasks if t.id == args.task_id]
    if not chosen:
        available = [t.id for t in tasks]
        if args.json:
            _json({"error": f"任务集里没有 {args.task_id!r}", "available": available})
        else:
            _p(f"任务集里没有 {args.task_id!r}；可选: " + "、".join(available))
        return 2
    if harness.repo.get_run(args.run_id) is None:
        if args.json:
            _json({"error": f"未找到运行 {args.run_id}", "runId": args.run_id})
        else:
            _p(f"未找到运行 {args.run_id}")
        return 1

    evaluation = evaluate(harness.repo, args.run_id, chosen[0])
    record_evaluation(harness.repo, evaluation)
    if args.json:
        _json(evaluation.as_json())
    else:
        _p(render_eval(evaluation))
    return 0 if evaluation.passed else 1


def cmd_benchmark(harness: Harness, args) -> int:
    """Run one tier of the benchmark, judge each task, and report.

    The suite is named, not guessed: `smoke` is the two minimal flows, and
    `challenge` needs a real provider. Running all three by default would make the
    common case pay for the expensive one.
    """
    from .bench import run_benchmark, suite_paths

    tiers = suite_paths()
    if args.suite not in tiers:
        _p(f"未知套件 {args.suite!r}；可选: " + "、".join(sorted(tiers)))
        return 2
    workspace = Path(args.workspace) if args.workspace else harness.cfg.data_dir / "bench"
    # `--json` means the output *is* JSON — a status line before it makes the
    # stream unparseable, which is the one thing a machine-readable mode is for.
    if not args.json:
        _p(f"套件 {args.suite}  工作区 {workspace}")
    run = run_benchmark(tiers[args.suite], workspace=workspace, tier=args.suite)
    if args.json:
        _json(run.as_json())
    else:
        _p(render_benchmark(run))
    return 0 if run.rollup["failed"] == 0 else 1


def cmd_compare(harness: Harness, args) -> int:
    """Compare a benchmark record against a baseline.

    Neither side is written to. A comparison that modified either would make the
    next comparison a different question.
    """
    from .bench import load_record
    from .bench_compare import compare
    from .bench_compare import render as render_comparison

    try:
        baseline = load_record(args.baseline)
        subject = load_record(args.record)
    except (OSError, ValueError) as e:
        _p(f"无法读取记录: {e}")
        return 2

    report = compare(baseline, subject,
                     growth_limit=harness.cfg.harness.baseline_work_growth_limit)

    if args.json:
        _json(report.as_json())
    else:
        _p(render_comparison(report))
    return 0 if report.ok else 1


def cmd_baseline_create(harness: Harness, args) -> int:
    """Run a tier and freeze the result as a baseline.

    Refuses to overwrite. That refusal is the whole of "a baseline is immutable":
    the one function that writes a baseline will not replace one, so a candidate
    cannot become the reference by re-running the suite.
    """
    from .bench import run_benchmark, suite_paths, write_baseline

    tiers = suite_paths()
    if args.suite not in tiers:
        _p(f"未知套件 {args.suite!r}；可选: " + "、".join(sorted(tiers)))
        return 2
    workspace = Path(args.workspace) if args.workspace else harness.cfg.data_dir / "bench"
    if not args.json:
        _p(f"套件 {args.suite}  工作区 {workspace}")
    run = run_benchmark(tiers[args.suite], workspace=workspace, tier=args.suite)
    try:
        target = write_baseline(args.out, run, source=str(tiers[args.suite]))
    except FileExistsError as e:
        _p(str(e))
        return 1
    if args.json:
        _json(run.as_json())
    else:
        _p(render_benchmark(run))
        _p(f"基线已写入 {target}（benchmark v{run.benchmark_version}，"
           f"{len(run.tasks)} 个任务）")
    return 0 if run.rollup["failed"] == 0 else 1


def cmd_baseline_check(harness: Harness, args) -> int:
    """Re-run a baseline's suite and compare the fresh record against it."""
    from .bench import load_record, run_benchmark, suite_paths
    from .bench_compare import compare
    from .bench_compare import render as render_comparison

    try:
        stored = load_record(args.baseline)
    except (OSError, ValueError) as e:
        _p(f"无法读取基线: {e}")
        return 2
    source = args.suite or stored.get("source") or ""
    if not source or not Path(source).exists():
        tiers = suite_paths()
        source = str(tiers.get(str(stored.get("suite")), ""))
    if not source or not Path(source).exists():
        _p(f"基线记录的套件路径不可用：{stored.get('source')!r}；用 --suite 指定")
        return 2

    workspace = Path(args.workspace) if args.workspace else harness.cfg.data_dir / "bench"
    _p(f"套件 {stored.get('suite')}  工作区 {workspace}")
    fresh = run_benchmark(source, workspace=workspace, tier=str(stored.get("suite") or ""))
    report = compare(stored, fresh.as_json(),
                     growth_limit=harness.cfg.harness.baseline_work_growth_limit)

    if args.json:
        _json(report.as_json())
    else:
        _p(render_benchmark(fresh))
        _p(render_comparison(report))
    return 0 if report.ok else 1


def render_benchmark(run) -> str:
    """A benchmark record as a person reads it."""
    lines = [f"套件 {run.suite}  benchmark v{run.benchmark_version}  session {run.session_id}",
             f"  环境: provider={run.environment.get('provider')} "
             f"digest={str(run.environment.get('digest'))[:12]}"]
    for task in run.tasks:
        unknown = sum(1 for v in task.evaluation.axes.values() if v == "unknown")
        gate = f"  硬门禁失败: {'、'.join(task.evaluation.hard_gate)}" if task.evaluation.hard_gate else ""
        lines.append(f"  [{task.evaluation.verdict:<4}] {task.task_id} v{task.task_version}"
                     f"  步骤={task.evaluation.artifacts.get('steps')}"
                     f"  工具={len(task.tool_sequence)}  未知轴={unknown}{gate}")
    rollup = run.rollup
    lines.append(f"  合计: {rollup['tasks']} 个任务，通过 {rollup['passed']}，"
                 f"失败 {rollup['failed']}，硬门禁失败 {rollup['hardGateFailures']}，"
                 f"未知轴 {rollup['unknownAxes']}")
    return "\n".join(lines)


def cmd_rsi(harness: Harness, args) -> int:
    """RSI candidates: propose from evidence, evaluate against a baseline, list.

    Nothing here promotes. `evaluate` settles a candidate at PASSED, REJECTED,
    NOT_PROMOTABLE or FAILED — and PASSED means "worth a human looking", which is
    as far as this stage goes.
    """
    from .experiment import load_experiment_result
    from .rsi import analyse, evaluate_candidate, record_from_row, record_proposal, render_analysis
    from .rsi import load_candidate as load_cand
    from .rsi import render as render_candidate

    repo = harness.repo

    if args.rsi_cmd in ("promote", "rollback", "promotion", "history"):
        from .rsi import history as skill_history
        from .rsi import promote, render_history, render_promotion, rollback

        if args.rsi_cmd == "promotion":
            promotion_id = (args.extra or args.target) if args.target == "show"                 else args.target
            record = repo.get_promotion(promotion_id)
            if record is None:
                _p(f"未找到晋升记录 {promotion_id}")
                return 1
            if args.json:
                _json(record)
            else:
                _p(render_promotion(record))
            return 0

        if args.rsi_cmd == "history":
            data = skill_history(repo, args.target)
            if not data["versions"]:
                _p(f"没有 {args.target} 的任何版本")
                return 1
            if args.json:
                _json(data)
            else:
                _p(render_history(data))
            return 0

        if args.rsi_cmd == "rollback":
            # §12 writes this as `wfos rsi rollback <skill> <version>`, so accept
            # the version positionally as well as through `--version`. The second
            # positional is only read here; everywhere else it is unused.
            version = args.version
            if version is None and str(args.extra).isdigit():
                version = int(args.extra)
            if version is None:
                _p("回滚需要指定版本：wfos rsi rollback <skill> <version>")
                return 2
            outcome = rollback(repo, args.target, version,
                               actor=args.actor or "cli", reason=args.reason or "")
            if args.json:
                _json(outcome)
            elif outcome["ok"]:
                rec = outcome["rollback"]
                _p(f"已回滚 {rec['trigger']}：v{rec['from_version']} → "
                   f"v{rec['to_version']}（v{rec['from_version']} 仍然存在）")
                _p(f"  回滚记录 {rec['rollback_id']}，digest "
                   f"v{rec['to_version']}={rec['to_digest'][:12]}")
            else:
                _p(f"回滚被拒绝：{outcome['error']}")
            return 0 if outcome["ok"] else 1

        # promote
        row = load_cand(repo, args.target, args.version)
        if row is None:
            _p(f"未找到候选 {args.target}")
            return 1
        outcome = promote(repo, record_from_row(row),
                          baseline=args.baseline or "",
                          actor=args.actor or "cli", reason=args.reason or "")
        if args.json:
            _json(outcome)
        elif outcome["ok"]:
            _p(render_promotion(outcome["promotion"]))
            _p(f"  正式 skill 现在是 {outcome['skill']['trigger']} "
               f"v{outcome['skill']['version']}（v{outcome['promotion']['parent_skill_version']}"
               f" 仍然存在，没有被覆盖）")
        else:
            _p(f"晋升被拒绝（候选 → {outcome['status']}）：")
            for reason_line in outcome["gate"]["reasons"]:
                _p(f"  - {reason_line}")
        return 0 if outcome["ok"] else 1

    if args.rsi_cmd == "list":
        rows = repo.list_candidates()
        if args.json:
            _json(rows)
            return 0
        if not rows:
            _p("尚无候选。用 `wfos rsi propose <实验结果>` 从证据里提出候选。")
            return 0
        _p(f"{'候选':<34}{'版本':<6}{'类型':<18}{'状态'}")
        for row in rows:
            _p(f"{row['candidate_id']:<34}v{row['version']:<5}{row['type']:<18}"
               f"{row['status']}")
        return 0

    if args.rsi_cmd == "show":
        record = load_cand(repo, args.target, args.version)
        if record is None:
            _p(f"未找到候选 {args.target}")
            return 1
        if args.json:
            _json(record)
        else:
            _p(render_candidate(record_from_row(record).as_json()))
        return 0

    if args.rsi_cmd == "propose":
        try:
            experiment = load_experiment_result(args.target)
        except (OSError, ValueError) as e:
            _p(f"无法读取实验结果: {e}")
            return 2
        live = {s["trigger"]: int(s["version"]) for s in repo.list_skills(live_only=True)}
        analysis = analyse(experiment, experiment_path=args.target, live_skills=live)
        for spec in analysis.proposals:
            try:
                record_proposal(repo, spec)
            except ValueError as e:
                # A candidate version already exists. Refusing is the rule — a
                # candidate is not rewritten — so the command reports it rather
                # than looking like it proposed something.
                _p(f"跳过：{e}")
        if args.json:
            _json(analysis.as_json())
        else:
            _p(render_analysis(analysis))
        return 0

    # evaluate
    record = load_cand(repo, args.target, args.version)
    if record is None:
        _p(f"未找到候选 {args.target}")
        return 1
    from .rsi import CandidateRecord, CandidateSpec
    spec = CandidateSpec(
        candidate_id=record["candidate_id"], version=record["version"],
        type=record["type"], parent_version=record["parent_version"],
        source_experiment=record["source_experiment"],
        source_runs=tuple(record["source_runs"]),
        source_failures=tuple(record["source_failures"]),
        evidence=record["evidence"], proposed_change=record["proposed_change"],
        rationale=record["rationale"])
    current = CandidateRecord(spec=spec, status=record["status"],
                              verdict=record["verdict"])
    workspace = Path(args.workspace) if args.workspace else         harness.cfg.data_dir / "rsi" / f"{spec.candidate_id}-v{spec.version}"
    suite = args.suite or str(spec.evidence.get("suite") or "")
    if not suite:
        _p("候选没有记录套件路径，无法评测；用 --suite 指定")
        return 2
    if not args.json:
        _p(f"候选 {spec.candidate_id} v{spec.version}  套件 {suite}")
        _p(f"  工作区 {workspace}")

    settled = evaluate_candidate(repo, current, workspace=workspace, suite=suite,
                                baseline=args.baseline or "")
    if args.json:
        _json(settled.as_json())
    else:
        _p(render_candidate(settled.as_json()))
    return 0 if settled.status in ("PASSED", "NOT_PROMOTABLE") else 1


def cmd_experiment(harness: Harness, args) -> int:
    """Run or read back a declared experiment.

    An experiment is a file, not a set of flags: a run driven by flags alone leaves
    no record of what was compared, and an experiment nobody can re-run is a set of
    numbers with a story attached.
    """
    from .experiment import (
        load_experiment,
        load_experiment_result,
        run_experiment,
        write_experiment,
    )
    from .experiment import render as render_experiment

    if args.experiment_cmd == "show":
        try:
            stored = load_experiment_result(args.path)
        except (OSError, ValueError) as e:
            _p(f"无法读取实验结果: {e}")
            return 2
        if args.json:
            _json(stored)
            return 0
        _p(render_stored_experiment(stored))
        return 0

    try:
        spec = load_experiment(args.path)
    except (OSError, ValueError) as e:
        _p(f"实验定义无法加载: {e}")
        return 2
    workspace = Path(args.workspace) if args.workspace else harness.cfg.data_dir / "experiments"
    if not args.json:
        _p(f"实验 {spec.experiment_id} v{spec.version}  套件 {spec.suite}")
        _p(f"  矩阵 {spec.matrix or '（空，单臂）'}  repetitions={spec.repetitions}")
        _p(f"  工作区 {workspace}")

    run = run_experiment(spec, workspace=workspace, baseline=args.baseline or spec.baseline)
    comparisons = comparisons_of(run, args.baseline or spec.baseline)

    if args.out:
        try:
            target = write_experiment(args.out, run, args.baseline or spec.baseline)
        except FileExistsError as e:
            _p(str(e))
            return 1
    else:
        target = None

    if args.json:
        payload = {**run.as_json(), "comparisons": comparisons}
        _json(payload)
    else:
        _p(render_experiment(run, comparisons))
        if target:
            _p(f"  结果已写入 {target}")
    regressed = any(not c["ok"] for c in comparisons)
    return 1 if (run.rollup()["failed"] or regressed) else 0


def comparisons_of(run, baseline_path: str):
    """P3's comparison, per arm. Empty when there is no baseline to compare to —
    an absent comparison is a fact, a fabricated one is not."""
    from .experiment import comparisons

    return comparisons(run, baseline_path)


def render_stored_experiment(stored: dict) -> str:
    """A stored result, rendered the way a fresh one is."""
    rollup = stored.get("rollup") or {}
    lines = [f"实验 {stored.get('experimentId')} v{stored.get('experimentVersion')}"
             f"  {stored.get('name')}",
             f"  套件 {stored.get('suite')}  跑于 {stored.get('createdAt', '?')[:19]}",
             f"  单元 {rollup.get('cells')}，通过 {rollup.get('passed')}，"
             f"失败 {rollup.get('failed')}，未能启动 {rollup.get('failedToStart')}"]
    for cell in stored.get("cells") or []:
        if cell.get("status") != "ran":
            lines.append(f"  [未能启动] {cell['cellId']}  {cell.get('failureClass')}: "
                         f"{cell.get('error')}")
            continue
        lines.append(f"  [{cell['evaluation'].get('verdict'):<4}] {cell['cellId']}"
                     f"  run={str(cell.get('runId'))[:8]}")
    return "\n".join(lines)


def cmd_trace(harness: Harness, args) -> int:
    """The append-only record of what a run did, in the order it did it.

    Distinct from `wfos result`, which shows the *current* working state. That
    one is missing whatever was deleted so a state could re-run; this one is not.
    """
    from .events import for_run
    from .events import render as render_trace

    run = harness.repo.get_run(args.run_id)
    if run is None:
        _p(f"未找到运行 {args.run_id}")
        return 1
    events = for_run(harness.repo, args.run_id)
    if not events:
        _p(f"运行 {args.run_id} 没有 trace 记录"
           + ("" if run["status"] == "created" else "（旧数据或未被推进过）"))
        return 0
    _p(f"=== 运行 {args.run_id} 的 trace（{len(events)} 条，按 event_id 排序）===")
    _p(f"session {run.get('session_id') or '-'}  origin {run.get('origin') or '-'}")
    _p(render_trace(events))
    return 0


def cmd_result(harness: Harness, args) -> int:
    run = harness.repo.get_run(args.run_id)
    if run is None:
        _p(f"未找到运行 {args.run_id}")
        return 1
    _p(f"=== 运行 {args.run_id} 步骤 ===")
    cur = harness.repo.conn.execute(
        "SELECT * FROM steps WHERE run_id=? ORDER BY id", (args.run_id,))
    for s in cur.fetchall():
        s = dict(s)
        fc = f"  失败类型={s['failure_class']}" if s.get("failure_class") else ""
        _p(f"[{s['state']}] agent={s['agent']} status={s['status']}{fc}")
        out = s.get("output_json") or "{}"
        try:
            import json
            o = json.loads(out)
            _p(f"    -> {json.dumps(o, ensure_ascii=False)[:220]}")
        except Exception:
            _p(f"    -> {str(out)[:220]}")
    _p(f"\n=== 运行 {args.run_id} 证据 ===")
    for e in harness.repo.list_evidence(args.run_id)[-10:]:
        _p(f"[{e['kind']}] {e['source']} ({e['confidence']}) {e['content'][:120]!r}")
    _p(f"\n=== 运行 {args.run_id} 工具调用 (最近 12) ===")
    for t in harness.repo.tool_calls(args.run_id)[-12:]:
        observed = f"  实际变更={t['diff_summary']}" if t.get("diff_summary") else ""
        code = f"  码={t['error_code']}" if t.get("error_code") else ""
        sec = f"  安全事件={t['security_event']}" if t.get("security_event") else ""
        _p(f"[{t.get('status') or ('ok' if t['ok'] else 'error'):<8}] "
           f"{t['agent']} {t['tool']} args={t['args_json'][:100]}{code}{sec}{observed}")
    violations = harness.repo.plan_violations_for_run(args.run_id)
    if violations:
        _p("\n=== 越界写入请求（未落盘，待审批） ===")
        for v in violations:
            _p(f"  {v['tool']}:{v['path']}")
    # Both writers, not just the state machine's: an interactive session's
    # edits are attributed to `assistant`, and filtering them out made
    # `wfos result` report "no changes" for a session that had made some.
    observed = sorted({p for role in ("implementer", "assistant")
                       for p in harness.repo.affected_paths_for_run(
                           args.run_id, agent=role)})
    deviation = (harness.repo.get_step(args.run_id, "implement") or {}) \
        .get("output_json", {}).get("plan_deviation") or {}
    if deviation and not deviation.get("unplanned") and (
            deviation.get("missing") or deviation.get("extra")):
        _p("\n=== 与已批准方案的偏离（报告项，非失败） ===")
        _p(f"  方案文件: {deviation.get('planned')}")
        _p(f"  漏改: {deviation.get('missing')}")
        _p(f"  多改: {deviation.get('extra')}")
    if observed:
        _p(f"\n=== 快照观测到的实际写入 ({len(observed)}) ===")
        for p in observed:
            _p(f"  {p}")
    return 0


def _resume_if_unblocked(harness: Harness, run_id: str) -> dict | None:
    run = harness.repo.get_run(run_id)
    if run is None or run["status"] != "waiting_approval":
        return run
    if harness.repo.pending_approvals_for_run(run_id):
        return run                    # still waiting on other approvals
    run = _advance(harness, run_id)   # all resolved -> continue past the gate
    _show_run(harness, run)
    return run


def cmd_approve(harness: Harness, args) -> int:
    a = harness.approve(args.approval_id, by="cli", note=args.note or "")
    if a is None:
        _p(f"审批不存在: {args.approval_id}")
        return 1
    _p(f"已批准 {a['id']} ({a['action']})")
    if a["status"] != "approved":
        _p(f"审批状态: {a['status']}")
    _resume_if_unblocked(harness, a["run_id"])
    return 0


def cmd_reject(harness: Harness, args) -> int:
    a = harness.reject(args.approval_id, by="cli", note=args.note or "")
    if a is None:
        _p(f"审批不存在: {args.approval_id}")
        return 1
    _p(f"已拒绝 {a['id']} ({a['action']})")
    _resume_if_unblocked(harness, a["run_id"])
    return 0


def cmd_pending(harness: Harness, args) -> int:
    approvals = harness.repo.list_pending_approvals()
    if approvals:
        _p("待审批:")
        for a in approvals:
            _p(f"  {a['id']}  run={a['run_id']}  {a['action']}  风险={a['risk_level']}")
    else:
        _p("暂无待审批。")
    blocked = harness.repo.list_runs(limit=100, status="waiting_approval")
    if blocked:
        _p("\n等待审批的运行:")
        for r in blocked:
            _p(f"  {r['id']}  {r['title'][:50]}")
    children = harness.repo.list_runs(limit=100, status="waiting_child")
    if children:
        _p("\n等待子流程的运行:")
        for r in children:
            _p(f"  {r['id']}  child={r['payload'].get('child_run_id')}  {r['title'][:50]}")
    return 0


def cmd_resume(harness: Harness, args) -> int:
    run = _run(harness.resume(args.run_id, force_stale=args.force_stale))
    if run is None:
        _p(f"未找到运行 {args.run_id}")
        return 1
    run = _advance(harness, run["id"])
    _show_run(harness, run)
    return 0


def cmd_resume_child(harness: Harness, args) -> int:
    run = _run(harness.resume_after_child(args.child_run_id))
    if run is None:
        _p(f"未找到子流程 {args.child_run_id}")
        return 1
    if run["id"] == args.child_run_id and run["status"] != "completed":
        _p(f"> 子流程 {args.child_run_id} 状态={run['status']}，尚未完成，父流程继续保持等待。")
        _p("> 子流程完成后再执行 `wfos resume-child <子ID>`。")
        return 0
    _show_run(harness, run)
    if run["status"] == "failed":
        _p(f"> 父流程已标记失败（子流程 {args.child_run_id} 回归未解决）。")
    return 0


def cmd_cancel(harness: Harness, args) -> int:
    run = harness.cancel(args.run_id, by="cli")
    if run is None:
        _p(f"未找到运行 {args.run_id}")
        return 1
    _p(f"已取消运行 {args.run_id}，状态={run['status']}")
    return 0


def cmd_mcp(harness: Harness, args) -> int:
    """Serve the tool suite over MCP stdio to an external client."""
    # stdout carries the MCP protocol stream, so every diagnostic goes to
    # stderr — and stays ASCII, because the launching client may well decode
    # that stream with a non-UTF-8 locale codec.
    if args.role not in EXTERNAL_ROLES:
        print(f"unsupported role {args.role!r}; choose one of: "
              f"{', '.join(EXTERNAL_ROLES)}", file=sys.stderr)
        return 2
    print(f"wfos MCP stdio server up: role={args.role} run_id={args.run_id} "
          f"project_root={harness.cfg.project_root}", file=sys.stderr)
    try:
        asyncio.run(harness.gateway.server.run_stdio(role=args.role, run_id=args.run_id))
    except KeyboardInterrupt:
        return 0
    return 0


def cmd_metrics(harness: Harness, args) -> int:
    """Token / latency / cost accounting, for one run or across runs.

    Every figure is printed with the coverage it was computed over: in this
    environment the brain is `mock`, which has no provider behind it and reports
    no usage, so the honest answer is "not measured" — not a zero that reads as
    a free run.
    """
    model, pricing = harness.cfg.llm.model, harness.cfg.pricing
    if args.run_id:
        if harness.repo.get_run(args.run_id) is None:
            _p(f"未找到运行 {args.run_id}")
            return 1
        report = run_metrics(harness.repo, args.run_id)
        if args.json:
            _json({"runId": args.run_id, "metrics": report,
                   "model": model, "pricing": pricing})
            return 0
        _p(f"=== 运行 {args.run_id} 用量 ===")
        _p(render(report, model=model, pricing=pricing))
        for state, s in (report.get("by_state") or {}).items():
            _p(f"    {state:<22} 步数={s['steps']} 调用={s['model_calls']} "
               f"延迟={s['latency_ms']}ms "
               f"in={s['input_tokens']} out={s['output_tokens']}")
        return 0

    report = aggregate(harness.repo, limit=args.limit)
    if args.json:
        _json({"metrics": report, "model": model, "pricing": pricing})
        return 0
    _p(f"=== 全部运行用量（{report['runs']} 个运行）===")
    _p(render(report, model=model, pricing=pricing))
    for kind, group in (report.get("by_kind") or {}).items():
        _p(f"    [{kind}] 步数={group['steps']} 调用={group['model_calls']} "
           f"in={group['input_tokens']} out={group['output_tokens']} "
           f"成本={group['cost_usd']}")
    origins = report.get("by_origin") or {}
    if len(origins) > 1:
        # Only when they differ: with one origin this repeats the whole report.
        _p("    按来源（基准/任务的运行会混进上面的合计，这里把它们分开）：")
        for origin, group in origins.items():
            _p(f"      {origin:<12} 步数={group['steps']} 调用={group['model_calls']} "
               f"in={group['input_tokens']} out={group['output_tokens']}")
    return 0


def cmd_baseline(harness: Harness, args) -> int:
    """Run the frozen whole-flow case set, and optionally diff it against the baseline.

    The cases build their own throwaway workspaces, so this ignores the harness
    it is handed — but it still shares the command dispatcher, which is what
    makes it reachable from the same CLI.

    Exit codes: 0 clean, 1 a case failed / a regression was found, 2 no baseline
    file to check against. A run without a stored baseline cannot be called
    clean, so it does not silently pass.
    """
    from .baseline import (
        DEFAULT_ARTIFACT,
        DEFAULT_CASES,
        compare,
        load_artifact,
        render_diff,
        render_measured,
        run_suite,
        write_artifact,
    )

    if args.baseline_cmd == "create":
        return cmd_baseline_create(harness, args)
    if args.baseline_cmd == "check" and args.baseline:
        # A positional names a *benchmark* baseline; `--file` names the frozen
        # reliability one. They are different artifacts with different comparison
        # rules, and the argument says which question is being asked.
        return cmd_baseline_check(harness, args)

    cases_path = args.cases or str(DEFAULT_CASES)
    store = Path(args.file or DEFAULT_ARTIFACT)
    if args.baseline_cmd == "live":
        return _run_live_cases(cases_path, args)

    suite = run_suite(cases_path, only=args.case)
    summary = suite["summary"]
    _p(f"用例 {summary['total']}，通过 {summary['passed']}，失败 {summary['failed']} "
       f"（通过率 {summary['passRate']:.2f}）")
    for row in suite["rows"]:
        _p(f"  [{'PASS' if row['passed'] else 'FAIL'}] {row['id']}")
        for mismatch in row["mismatches"]:
            _p(f"         {mismatch['invariant']}: 期望 {mismatch['expected']!r}，"
               f"实际 {mismatch['actual']!r}")
        if row.get("measured"):
            _p(f"         实测 {render_measured(row['measured'])}")
    if suite["invariantFailures"]:
        _p("失败集中在: " + "、".join(f"{k}×{v}"
                                     for k, v in suite["invariantFailures"].items()))

    if args.baseline_cmd == "run":
        write_artifact(store, suite)
        _p(f"基线已写入 {store}")
        return 0 if summary["failed"] == 0 else 1

    if not store.exists():
        _p(f"未找到基线文件 {store}；先执行 `wfos baseline run`")
        return 2
    diff = compare(suite, load_artifact(store),
                   work_growth_limit=harness.cfg.harness.baseline_work_growth_limit)
    _p(render_diff(diff))
    return 0 if diff["ok"] else 1


def _run_live_cases(cases_path: str, args) -> int:
    """Run the real-provider cases and report — deliberately no artifact.

    These cases exist because the mock baseline could not see what a real endpoint
    rejects: four of five real runs once failed on the wire, with every failure
    reading the same. They are an acceptance run, not a frozen baseline, and
    writing them to `artifacts/harness-baseline.json` would poison it — the next
    `check` would compare a *different* live run against a stored one and fail for
    no reason.
    """
    from .baseline import render_measured
    from .baseline import run_suite as _suite

    report = _suite(cases_path, only=args.case, include_live=True)
    if not report["rows"]:
        _p(f"{cases_path} 里没有 live 用例（需要一段 \"brain\": \"live\"）")
        return 2
    summary = report["summary"]
    _p(f"live 用例 {summary['total']}，通过 {summary['passed']}，失败 {summary['failed']}"
       f"（使用 load_config() 里配置的 provider，本机 {report['liveCases']} 条 live 用例）")
    for row in report["rows"]:
        _p(f"  [{'PASS' if row['passed'] else 'FAIL'}] {row['id']}")
        for mismatch in row["mismatches"]:
            _p(f"         {mismatch['invariant']}: 期望 {mismatch['expected']!r}，"
               f"实际 {mismatch['actual']!r}")
        if row.get("measured"):
            _p(f"         实测 {render_measured(row['measured'])}")
    if report["invariantFailures"]:
        _p("失败集中在: " + "、".join(f"{k}×{v}"
                                     for k, v in report["invariantFailures"].items()))
    _p("（live 结果不写入冻结基线，见 _run_live_cases 的说明）")
    return 0 if summary["failed"] == 0 else 1


def cmd_skills(harness: Harness, args) -> int:
    """The procedures the harness has distilled, and where each came from."""
    # Unfiltered on purpose: this prints, it does not load. An operator asking
    # "what has this harness distilled" wants the candidate rows too — they are
    # the ones with no evidence behind them yet, and hiding them would make the
    # list read as more settled than it is. Each row carries its own status.
    skills = harness.repo.list_skills(status=None, origin=None)
    if _wants_json(args):
        _json({"skills": skills, "count": len(skills)})
        return 0
    if not skills:
        _p("暂无手法记录。运行失败过一次、之后又通过时才会自动记下——"
           "只记过程，不记（也从不由模型撰写）经过。")
        return 0
    for skill in skills:
        _p(f"[{skill['trigger']}] 版本 {skill['version']}  "
           f"证据运行={skill['evidence_run']}  {skill['title']}")
        _p(f"    涉及文件: {', '.join(skill['files']) or '（无）'}")
        for line in (skill["procedure"] or "").splitlines():
            _p(f"    {line}")
        _p("")
    return 0


def cmd_capabilities(harness: Harness, args) -> int:
    """What this endpoint has been observed to accept.

    Readable on purpose: the discovery exists so an operator does not have to know
    that (say) DeepSeek rejects `json_schema`, and a record they cannot inspect
    would just move the ignorance rather than remove it.
    """
    entries = harness.capabilities.all()
    if _wants_json(args):
        _json({"capabilities": entries, "path": str(harness.capabilities.path)})
        return 0
    if not entries:
        _p(f"尚无观测记录（{harness.capabilities.path}）。"
           f"跑一次真实 provider 后会自动记下它接受什么。")
        return 0
    for key, facts in entries.items():
        _p(f"[{key}]")
        for name in ("structured_output", "tool_names_sanitized", "reasoning"):
            if name in facts:
                _p(f"  {name:24} = {facts[name]!r}")
    return 0


def cmd_history(harness: Harness, args) -> int:
    rows = harness.repo.list_runs(limit=args.limit or 30)
    if args.json:
        _json({"runs": rows, "count": len(rows)})
        return 0
    if not rows:
        _p("暂无运行记录。")
        return 0
    _p(f"{'ID':<18}{'类型':<10}{'状态':<18}{'创建时间':<22}{'标题'}")
    for r in rows:
        _p(f"{r['id']:<18}{r['kind']:<10}{r['status']:<18}{str(r['created_at'])[:20]:<22}{r['title'][:44]}")
    return 0


def cmd_wiki(harness: Harness, args) -> int:
    if args.wiki_cmd == "search":
        res = harness.wiki.search(args.query, kind=args.kind, tags=args.tags)
        if not res:
            _p("无匹配结果。")
            return 0
        for d in res:
            _p(f"[{d['kind']}] #{d['id']} 信任={d['trust']} 验证={bool(d['verified'])}  {d['title']}")
            _p(f"    tags={d['tags']}")
            _p(f"    {d['content'][:180]!r}")
        return 0
    if args.wiki_cmd == "list":
        for d in harness.wiki.list(kind=args.kind, status=args.status):
            _p(f"[{d['kind']}] #{d['id']} 状态={d['status']} 验证={bool(d['verified'])}  {d['title']}")
        return 0
    if args.wiki_cmd == "promote":
        if not args.by:
            _p("需指定 --by <审核人>")
            return 2
        try:
            d = harness.wiki.promote_to_authoritative(args.id, by=args.by)
        except PermissionError as e:
            _p(f"拒绝: {e}")
            return 1
        _p(f"已发布为权威: #{d['id']} {d['title']}")
        return 0
    _p("未知 wiki 子命令")
    return 2


# -------------------------------------------------------------------- argparse
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wfos", description="Model-agnostic engineering workflow")
    # Not `required`: with no subcommand this is an interactive session. That is
    # the whole point of the entry point — `wfos` on its own should start a
    # conversation, not print a usage error.
    sub = parser.add_subparsers(dest="cmd", required=False)

    p = sub.add_parser("run", help="创建并推进一个运行")
    p.add_argument("text", nargs="+")
    p.add_argument("--kind", choices=["feature", "bugfix"], default=None)
    p.add_argument("--title", default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("chat", help="交互式会话：多轮问答，可直接读写项目并跑测试")
    # Only here, never on the root parser: a subparser's default overwrites the
    # root's parsed value for the same dest, so defining it in both places makes
    # a bare `wfos` inherit `chat`'s default.
    p.add_argument("--once", default=None,
                   help="只跑一轮就退出（不读 stdin），便于脚本化")
    p.set_defaults(fn=cmd_chat)

    p = sub.add_parser("status", help="查看运行状态")
    p.add_argument("run_id", nargs="?", default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("task", help="在隔离环境里运行一个声明式任务")
    p.add_argument("task_cmd", choices=["run"])
    p.add_argument("path", help="任务集文件或目录")
    p.add_argument("task_id")
    p.add_argument("--workspace", default=None, help="隔离工作区根目录")
    p.add_argument("--name", default=None, help="工作区子目录名（同任务多配置时区分）")
    p.set_defaults(fn=cmd_task)

    p = sub.add_parser("evaluate", help="按任务判定一次运行，并记录判定")
    p.add_argument("run_id")
    p.add_argument("path", help="任务集文件或目录")
    p.add_argument("task_id")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("benchmark", help="跑一层固定任务集（smoke/regression/challenge）")
    p.add_argument("benchmark_cmd", choices=["run"])
    p.add_argument("suite", choices=["smoke", "regression", "challenge"])
    p.add_argument("--workspace", default=None, help="隔离工作区根目录")
    p.add_argument("--json", action="store_true", help="输出 JSON 而非人读格式")
    p.set_defaults(fn=cmd_benchmark)

    p = sub.add_parser("compare", help="比较一份 benchmark 记录与基线")
    p.add_argument("baseline", help="基线文件")
    p.add_argument("record", help="另一份 benchmark 记录（baseline 或 benchmark run）")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("rsi", help="RSI 候选：从证据提出、评测、查看（不晋升）")
    p.add_argument("rsi_cmd", choices=["propose", "show", "evaluate", "list",
                                       "promote", "rollback", "promotion", "history"])
    p.add_argument("target", nargs="?", default="", help="propose 用实验结果文件；show/evaluate 用候选 id")
    # `promotion show <id>` as the spec writes it, and `promotion <id>` for
    # anyone who leaves the word out. The second positional is what makes the
    # first form parseable at all; without it argparse reads "show" as the id.
    p.add_argument("extra", nargs="?", default="", help=argparse.SUPPRESS)
    p.add_argument("--version", type=int, default=None, help="候选版本（默认最新）")
    p.add_argument("--suite", default=None, help="评测用的任务集（默认取自候选证据）")
    p.add_argument("--baseline", default=None, help="评测时比较的基线")
    p.add_argument("--workspace", default=None)
    p.add_argument("--actor", default=None, help="晋升/回滚的执行者，记进记录")
    p.add_argument("--reason", default=None, help="晋升/回滚的理由，记进记录")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_rsi, candidate="", experiment="")

    p = sub.add_parser("experiment", help="跑一个声明式实验（矩阵展开为多个独立 run）")
    p.add_argument("experiment_cmd", choices=["run", "show"])
    p.add_argument("path", help="实验定义文件（run）或实验结果文件（show）")
    p.add_argument("--out", default=None, help="run 时把结果写入此路径（拒绝覆盖）")
    p.add_argument("--baseline", default=None, help="与该基线比较（省略则只保存原始结果）")
    p.add_argument("--workspace", default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_experiment)

    p = sub.add_parser("trace", help="append-only 运行轨迹（含已被删除的步骤）")
    p.add_argument("trace_cmd", choices=["show"])
    p.add_argument("run_id")
    p.set_defaults(fn=cmd_trace)

    p = sub.add_parser("result", help="查看运行步骤/证据/工具调用")
    p.add_argument("run_id")
    p.set_defaults(fn=cmd_result)

    p = sub.add_parser("approve", help="批准审批")
    p.add_argument("approval_id")
    p.add_argument("--note", default="")
    p.set_defaults(fn=cmd_approve)

    p = sub.add_parser("reject", help="拒绝审批")
    p.add_argument("approval_id")
    p.add_argument("--note", default="")
    p.set_defaults(fn=cmd_reject)

    p = sub.add_parser("pending", help="列出待审批")
    p.set_defaults(fn=cmd_pending)

    p = sub.add_parser("resume", help="恢复运行")
    p.add_argument("run_id")
    p.add_argument("--force-stale", action="store_true", dest="force_stale",
                   help="显式接受过期状态（环境或关键文件已变）继续，决定会被记录")
    p.set_defaults(fn=cmd_resume)

    p = sub.add_parser("resume-child", help="子流程完成后恢复父流程")
    p.add_argument("child_run_id")
    p.set_defaults(fn=cmd_resume_child)

    p = sub.add_parser("cancel", help="取消运行")
    p.add_argument("run_id")
    p.set_defaults(fn=cmd_cancel)

    p = sub.add_parser("history", help="最近运行")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_history)

    p = sub.add_parser("skills", help="已沉淀的修复手法（按失败类型，带证据运行）")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_skills)

    p = sub.add_parser("capabilities", help="已观测到的 provider 能力（结构化输出模式等）")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_capabilities)

    p = sub.add_parser("metrics", help="token/延迟/成本用量（含覆盖率）")
    p.add_argument("run_id", nargs="?", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_metrics)

    p = sub.add_parser("baseline", help="基线：run/check 冻结可靠性，create 冻结 benchmark，live 打真实 provider")
    p.add_argument("baseline_cmd", choices=["run", "check", "live", "create"])
    p.add_argument("baseline", nargs="?", default=None,
                   help="check 时指定 benchmark 基线文件（省略则检查冻结可靠性基线）")
    p.add_argument("--suite", default=None,
                   help="create 时指定层（smoke/regression/challenge），check 时可覆盖记录的套件")
    p.add_argument("--out", default=None, help="create 时基线写入路径")
    p.add_argument("--workspace", default=None, help="隔离工作区根目录")
    p.add_argument("--json", action="store_true")
    p.add_argument("--cases", default=None, help="用例集路径（默认 benchmarks/cases.json）")
    p.add_argument("--file", default=None, help="基线 artifact 路径")
    p.add_argument("--case", action="append", default=None, help="只跑指定用例（可重复）")
    p.set_defaults(fn=cmd_baseline)

    p = sub.add_parser("mcp", help="以 MCP stdio 服务端运行，供外部 MCP 客户端调用")
    p.add_argument("--role", default="investigator", choices=list(EXTERNAL_ROLES),
                   help="外部客户端可使用的工具集角色（默认只读 investigator）")
    p.add_argument("--run-id", default="external", dest="run_id",
                   help="工具调用归属的运行 ID；指向已有运行可复用其审批记录")
    p.set_defaults(fn=cmd_mcp)

    p = sub.add_parser("wiki", help="LLM 知识库")
    p.add_argument("wiki_cmd", choices=["search", "list", "promote"])
    p.add_argument("query", nargs="?")
    p.add_argument("--kind")
    p.add_argument("--tags")
    p.add_argument("--status")
    p.add_argument("--by")
    p.add_argument("id", type=int, nargs="?")
    p.set_defaults(fn=cmd_wiki)

    args = parser.parse_args(argv)
    global _JSON_MODE
    _JSON_MODE = bool(getattr(args, "json", False))
    try:
        try:
            harness = build_harness()
        except MissingCredentialError as e:
            # A configuration mistake, reported as one — with its own exit code, so
            # a wrapper can tell it apart from "not found" (1), "bad arguments" (2)
            # and "another process holds the run" (3).
            _p(str(e))
            return 4
        if args.cmd is None:            # bare `wfos` -> an interactive session
            return cmd_chat(harness, args)
        return args.fn(harness, args)
    finally:
        _seal_json()


def _seal_json() -> None:
    """Guarantee that `--json` produced exactly one document on stdout.

    Every failure branch in this file already prints a sentence and returns a
    code, and none of them was written with JSON in mind. Rather than teach each
    one a second dialect — and re-teach every one added later — the guarantee is
    applied once, here: if the invocation asked for JSON and nothing wrote any,
    the last thing it said becomes the document.

    That is why an error branch cannot break a caller's parser, and why the
    check is a property of the process rather than a rule about the code.
    """
    if _JSON_MODE and not _JSON_WRITTEN:
        _json({"error": _LAST_LINE or "命令没有产生输出"})


if __name__ == "__main__":
    sys.exit(main())
