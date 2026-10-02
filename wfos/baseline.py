"""Frozen reliability baseline: a declarative case set, run and diffed.

`tests/` covers units; this covers whole flows. The split is deliberate — a unit
test can assert that `check_write_scope` refuses a path, but only a whole-flow
case can assert that the refusal actually left the file untouched on disk, that
the run stopped in the right status, and that nothing else moved on the way.

Each case runs in its own throwaway workspace against the deterministic `mock`
brain, so the set is offline, repeatable, and needs no API key. A run's outcome
is reduced to a flat dictionary of **scalar invariants**. Every invariant is
recorded in the artifact; the case declares only the subset it requires.
Recording everything while asserting a subset is what makes a baseline worth
keeping: an invariant nobody asserted today is still visible in tomorrow's diff.

`compare` is the point of the exercise. A case that passed before and fails now
is a regression and fails the check; a case that failed before and passes now is
reported as fixed, not as a failure.
"""
from __future__ import annotations

import contextlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .eval import invariants
from .harness.orchestrator import Harness, RunLockedError
from .metrics import billable_tokens, totals_with_coverage
from .models import ORIGIN_BENCHMARK
from .runner import RunEnvironment, TaskDriver
from .storage.repo import Repo
from .tasks import TaskSpec

SCHEMA_VERSION = 1
DEFAULT_CASES = Path("benchmark")
DEFAULT_ARTIFACT = Path("artifacts/harness-baseline.json")


# ------------------------------------------------------------------- case input
def load_cases(path: Path | str = DEFAULT_CASES) -> dict:
    """Load and validate the case set.

    Validation is fail-closed: a case missing an id, a request, or an `expect`
    block is rejected rather than silently running with no assertions — a case
    that asserts nothing would report as passing forever.
    """
    src = Path(path)
    # A directory is the shape the three tiers take; reading it here means the
    # tiers do not need a loader of their own, and that the frozen reliability set
    # can be "everything" without anyone listing the files twice.
    files = sorted(src.rglob("*.json")) if src.is_dir() else [src]
    cases: list[dict] = []
    seen: set[str] = set()
    for file in files:
        data = json.loads(file.read_text(encoding="utf-8"))
        if data.get("schemaVersion") != SCHEMA_VERSION:
            raise ValueError(f"{file}: 不支持的 cases schemaVersion {data.get('schemaVersion')!r}")
        found = data.get("cases")
        if not isinstance(found, list) or not found:
            raise ValueError(f"{file}: cases 必须是非空列表")
        cases.extend(found)
    if not cases:
        raise ValueError(f"{src}: 没有找到任何用例")
    for case in cases:
        cid = str(case.get("id", "")).strip()
        if not cid:
            raise ValueError("case 缺少 id")
        if cid in seen:
            raise ValueError(f"case id 重复: {cid}")
        seen.add(cid)
        if not str(case.get("request", "")).strip():
            raise ValueError(f"case {cid} 缺少 request")
        if case.get("kind") not in ("feature", "bugfix"):
            raise ValueError(f"case {cid} 的 kind 必须是 feature 或 bugfix")
        if not isinstance(case.get("expect"), dict) or not case["expect"]:
            raise ValueError(f"case {cid} 必须有非空的 expect —— 不断言任何东西的用例永远通过")
        if not isinstance(case.get("stepBudget"), int) or case["stepBudget"] <= 0:
            raise ValueError(f"case {cid} 缺少正整数 stepBudget")
        if case.get("brain", "mock") not in ("mock", "live"):
            raise ValueError(f"case {cid} 的 brain 必须是 mock 或 live")
    # The merged set, not the last file read: returning `data` here meant a
    # directory of three tier files loaded as whichever one sorted last, and the
    # suite silently ran two tasks where twelve were expected.
    return {"schemaVersion": SCHEMA_VERSION, "cases": cases, "source": str(src)}


def is_live(case: dict) -> bool:
    return case.get("brain") == "live"


# ------------------------------------------------------------------- one case
class _CaseRunner:
    """Runs a single case in its own workspace and reduces it to invariants."""

    def __init__(self, case: dict, workspace: Path):
        self.case = case
        # Every case *is* a task: the case schema is what `TaskSpec` reads, and
        # going through it is what lets one driver serve both.
        self.task = TaskSpec.from_case(case)
        self.live = is_live(case)
        # Each case gets its own tree, database and stack. That isolation is the
        # whole reason a case's memory, skills and audit rows cannot reach a real
        # run — see `runner.RunEnvironment`, which now owns the one implementation.
        self.env = RunEnvironment.create(
            workspace, name=case["id"], fixture=case.get("fixture"),
            live=self.live, pricing=case.get("pricing"))
        self.root = self.env.root
        self.child_id: str | None = None
        # The run the invariants describe. `newRun` moves it, so a case can
        # observe a property that only exists between runs — a single run cannot
        # see its own memory, because memory is written when a run ends.
        self.subject_run_id: str | None = None

    def _harness(self) -> Harness:
        return self.env.harness()

    def run(self) -> tuple[dict, dict]:
        """`(invariants, measured)` — behaviour and cost, kept apart."""
        harness = self._harness()
        repo = harness.repo
        # A benchmark run says so. Every case builds its own harness, database
        # and workspace, but the *origin* is what keeps its memory and its skills
        # out of a production prompt — isolation of the environment alone would
        # not, because those two are keyed on paths and on failure classes.
        task = self.task
        # The same driver the task runner uses. A case *is* a task — the case
        # schema is what `TaskSpec` reads — so two drivers over these files would
        # end up disagreeing about what a case means, and the frozen baseline and
        # the benchmark would be measuring different things while both claimed to
        # run the same cases.
        driver = TaskDriver(harness, self.root)
        driver.start(task, origin=ORIGIN_BENCHMARK)
        figures: dict[str, Any] = {}
        try:
            # A lease held elsewhere: recorded through the run's own state, so
            # the invariants still describe what happened.
            with contextlib.suppress(RunLockedError):
                driver.drive(task, origin=ORIGIN_BENCHMARK)
            self.subject_run_id = driver.subject_run_id
            self.child_id = driver.child_run_id
            invariants = self._invariants(repo)
        finally:
            try:
                # Read before the connection goes: these come from the same tables.
                figures = measured(repo)
            finally:
                # Windows refuses to delete a workspace whose SQLite file is still
                # open, so the connection has to go before the caller cleans up —
                # including on the path where the reducer raised.
                repo.close()
        return invariants, figures

    def _invariants(self, repo: Repo) -> dict[str, Any]:
        return invariants(repo, self.subject_run_id,
                          step_budget=self.task.step_budget,
                          child_run_id=self.child_id)


# What a case's measured "work" is counted in. Tokens when the provider reported
# them, because that is what the money buys; model calls otherwise, because the
# harness counts those itself and so even a mock case — the frozen baseline —
# has a work figure. The unit travels with the number, and `compare` refuses to
# divide two different ones.
WORK_TOKENS = "tokens"
WORK_MODEL_CALLS = "model_calls"


def work_units(figures: dict) -> tuple[str, int | None]:
    """`(unit, amount)` for one case's measured work.

    `("", None)` when neither is available — which is not a zero. A growth limit
    cannot be applied to it, and must not be applied to a 0 invented here.
    """
    total = billable_tokens({"input_tokens": figures.get("inputTokens"),
                             "output_tokens": figures.get("outputTokens")})
    if total is not None:
        return WORK_TOKENS, total
    calls = figures.get("modelCalls")
    if calls is not None:
        return WORK_MODEL_CALLS, calls
    return "", None


def measured(repo: Repo) -> dict[str, Any]:
    """What a case cost, recorded next to the invariants rather than among them.

    One case owns a whole database, so this is every run in it: the case's cost is
    what the case spent, including a run a case creates only to be remembered by
    the next one.

    Kept out of `actual` on purpose. `compare` treats any movement in an invariant
    nobody asserted as a failure, and for *behaviour* that is right — a boolean
    that drifts unasserted is exactly what a reader would otherwise never hear
    about. A token count moves every single run; that is what measuring it is for.
    In one bag, either the baseline becomes useless or the measurement becomes a
    lie. So measured figures are diffed as information, and become a gate only
    through a threshold an operator sets.

    Every figure is either a measurement or None; None means the provider did not
    report it, never 0, so "unmeasured" cannot be read as "free".
    """
    rows = repo.metric_rows()
    got = totals_with_coverage(rows)
    figures = {
        "steps": got["steps"],
        "stepsWithUsage": got["steps_with_usage"],
        "inputTokens": got["input_tokens"],
        "outputTokens": got["output_tokens"],
        "cachedTokens": got["cached_tokens"],
        "reasoningTokens": got["reasoning_tokens"],
        # Counted by the harness rather than reported by a provider, so this is
        # always known: 0 here really is zero.
        "modelCalls": got["model_calls"],
        "latencyMs": got["latency_ms"],
        "costUsd": got["cost_usd"],
        # Which brains answered. A routed case and an unrouted one can cost the
        # same and still be different changes.
        "models": ",".join(sorted({str(r.get("model")) for r in rows if r.get("model")})),
    }
    unit, amount = work_units(figures)
    figures["workUnit"], figures["workUnits"] = unit, amount
    return figures


# Figures whose difference between two runs is not a property of the harness.
# Latency on a deterministic suite measures the machine, not the workflow: two
# runs of the same mock cases came out ×0.52 and ×1.23 apart on this checkout.
# Recording it is still right — the artifact is a record, and a tenfold jump is
# worth seeing — but diffing it per case would bury the figures that do mean
# something under a dozen noise lines, and a section a reader learns to skip is
# not a feature. They are reported as one suite-wide total instead.
_NOISY_FIGURES = ("latencyMs",)


def _ratio(before: Any, after: Any) -> float | None:
    """`after / before`, or None where the division would mean nothing.

    None for non-numbers, for booleans (ints in Python, but not quantities), and
    for a zero or unknown base: "grew from zero" has no ratio, and inventing one
    would put a precise-looking figure on a comparison that does not exist.
    """
    for value in (before, after):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
    if not before:
        return None
    return after / before


def render_measured(figures: dict) -> str:
    """One case's measured figures as a line, with unknowns shown as unknown.

    The live path is where these are actually measured — a mock case has no
    provider to report tokens — so it is the path that most needs them printed.
    A figure that was never reported prints as `-`, never as 0.
    """
    def n(value: Any) -> str:
        return "-" if value is None else str(value)

    unit = figures.get("workUnit") or ""
    return (f"steps={n(figures.get('steps'))} "
            f"有用量={n(figures.get('stepsWithUsage'))} "
            f"in={n(figures.get('inputTokens'))} out={n(figures.get('outputTokens'))} "
            f"cached={n(figures.get('cachedTokens'))} "
            f"reasoning={n(figures.get('reasoningTokens'))} "
            f"调用={n(figures.get('modelCalls'))} "
            f"成本={n(figures.get('costUsd'))} "
            f"模型={figures.get('models') or '-'} "
            f"工作量={n(figures.get('workUnits'))} {unit}")


def _sum_figures(rows: dict, figure: str) -> int | None:
    """One figure summed across a suite, or None when nothing carried it."""
    amounts = [(row.get("measured") or {}).get(figure) for row in rows.values()]
    present = [v for v in amounts if isinstance(v, (int, float))
               and not isinstance(v, bool)]
    return sum(present) if present else None


def _suite_work(rows: list[dict]) -> tuple[str, int | None]:
    """`(unit, total)` across the whole suite.

    All cases must agree on the unit, or the answer is `("", None)`: adding a mock
    case's model calls to a live case's tokens produces a total in no unit at all.
    """
    units, amounts = set(), []
    for row in rows:
        unit, amount = work_units(row.get("measured") or {})
        units.add(unit)
        if amount is not None:
            amounts.append(amount)
    if len(units) != 1 or not amounts:
        return "", None
    return units.pop(), sum(amounts)


def _work_growth(before: list[dict], now: list[dict], limit: float | None) -> dict:
    """Whether the suite's measured work grew past the operator's limit.

    Suite-wide rather than per case: one case getting cheaper while another gets
    dearer is a wash, and a per-case gate would fire on the wash.

    A unit change disables the gate and says so, rather than comparing numbers
    that are not the same kind of thing — a baseline frozen on mock and checked
    against a live provider would otherwise report a growth figure computed from
    tokens over model calls.
    """
    old_unit, old_total = _suite_work(before)
    new_unit, new_total = _suite_work(now)
    out = {"unit": new_unit, "before": old_total, "after": new_total,
           "ratio": None, "limit": limit, "exceeded": False, "why": ""}
    if limit is None:
        out["why"] = "未设上限（harness.baseline_work_growth_limit），只报告"
    elif not old_unit or old_unit != new_unit:
        out["why"] = f"度量单位不同（{old_unit or '无'} → {new_unit or '无'}），不做比较"
    elif not old_total or new_total is None:
        out["why"] = "上一次或本次没有可用的度量值"
    else:
        out["ratio"] = new_total / old_total
        out["exceeded"] = out["ratio"] > 1 + limit
    return out


def run_case(case: dict, workspace: Path) -> dict:
    actual, figures = _CaseRunner(case, workspace).run()
    mismatches = [{"invariant": name, "expected": want, "actual": actual.get(name)}
                  for name, want in case["expect"].items()
                  if actual.get(name) != want]
    return {
        "id": case["id"],
        "kind": case["kind"],
        "passed": not mismatches,
        "expect": dict(case["expect"]),
        "actual": actual,
        # A sibling of `actual`, never a member of it — see `measured`.
        "measured": figures,
        "mismatches": mismatches,
    }


# ------------------------------------------------------------------- full suite
def run_suite(cases_path: Path | str = DEFAULT_CASES,
              workspace: Path | str | None = None,
              only: list[str] | None = None, *,
              include_live: bool = False) -> dict:
    """Run one class of case and build an artifact.

    `include_live=False` (the default) runs the mock cases — the ones that can be
    **frozen**, because the same input gives the same outcome every time.
    `include_live=True` runs the real-provider cases instead, and the result is an
    acceptance report, not a baseline: it must not be written to the frozen
    artifact, because freezing a non-deterministic run would make the next `check`
    compare a *different* run against it and fail for no reason.

    The count of live cases is recorded either way, so a reader of the frozen
    artifact can see that more cases exist than it covers.
    """
    import tempfile

    data = load_cases(cases_path)
    live_total = sum(1 for c in data["cases"] if is_live(c))
    cases = [c for c in data["cases"]
             if is_live(c) == include_live and (not only or c["id"] in only)]
    with tempfile.TemporaryDirectory(prefix="wfos-baseline-") as tmp:
        rows = [run_case(case, Path(workspace or tmp)) for case in cases]

    passed = sum(1 for r in rows if r["passed"])
    failures: dict[str, int] = {}
    for row in rows:
        for mismatch in row["mismatches"]:
            failures[mismatch["invariant"]] = failures.get(mismatch["invariant"], 0) + 1
    return {
        "schemaVersion": SCHEMA_VERSION,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "cases": str(cases_path),
        "live": include_live,
        # How many real-provider cases exist. Shown even when they were not run, so
        # a green frozen baseline cannot be mistaken for "everything is covered".
        "liveCases": live_total,
        "summary": {
            "total": len(rows),
            "passed": passed,
            "failed": len(rows) - passed,
            "passRate": passed / len(rows) if rows else 0.0,
        },
        # Which invariant is the weak one, counted — a pass rate alone does not
        # say whether failures cluster on one property or spread across many.
        "invariantFailures": dict(sorted(failures.items())),
        "rows": rows,
    }


# ------------------------------------------------------------------------ diff
def compare(current: dict, stored: dict, *,
            work_growth_limit: float | None = None) -> dict:
    """Diff a fresh artifact against the stored baseline.

    Two different kinds of trouble, kept apart because they call for different
    responses:

      * a **regression** is behavioural — a case that passed and now fails, or a
        case the baseline covers that this run did not;
      * a **stale baseline** is bookkeeping — a case that is new since the
        baseline was written, or an invariant that moved without either run
        asserting it. The baseline no longer describes what is being run.

    Both fail the check. A frozen baseline that quietly stops covering the cases
    that run is not a baseline, and movement that no assertion catches is exactly
    what a reader would otherwise never hear about. A case that was failing and
    now passes is reported as *fixed* and fails nothing — otherwise a baseline
    could never be improved.
    """
    now = {r["id"]: r for r in current.get("rows", [])}
    before = {r["id"]: r for r in stored.get("rows", [])}

    regressions, fixed, changes = [], [], []
    for cid, row in sorted(before.items()):
        new = now.get(cid)
        if new is None:
            continue
        if row.get("passed") and not new.get("passed"):
            regressions.append({"id": cid, "mismatches": new.get("mismatches", [])})
        elif not row.get("passed") and new.get("passed"):
            fixed.append(cid)
        else:
            old_actual = row.get("actual") or {}
            for name, value in sorted((new.get("actual") or {}).items()):
                if name in old_actual and old_actual[name] != value:
                    changes.append({"id": cid, "invariant": name,
                                    "before": old_actual[name], "after": value})

    added = sorted(set(now) - set(before))
    removed = sorted(set(before) - set(now))
    # An invariant the runner produces today that the stored baseline has no
    # column for. Iterating only the keys both sides share would skip it
    # silently, so a baseline could quietly fall behind the runner and still
    # report "consistent" — which is the same hole as an uncovered *case*.
    unrecorded = sorted({name for cid, row in now.items() if cid in before
                         for name in (row.get("actual") or {})
                         if name not in (before[cid].get("actual") or {})})
    if any(r.get("measured") for r in now.values()) and not any(
            r.get("measured") for r in before.values()):
        # The stored baseline predates the measured section, so it cannot see any
        # of it. Same rule as an invariant it has no column for: a baseline that
        # quietly compares nothing is not a baseline, and the fix is to re-freeze.
        unrecorded = sorted({*unrecorded, "measured"})

    # Measured figures, diffed as information. They move by their nature, so a
    # change here is not a regression and never fails the check by itself.
    measured_changes = []
    for cid, row in sorted(before.items()):
        new = now.get(cid)
        if new is None:
            continue
        old_figures = row.get("measured") or {}
        new_figures = new.get("measured") or {}
        for name in sorted(set(old_figures) | set(new_figures)):
            was, is_now = old_figures.get(name), new_figures.get(name)
            if was == is_now or name in _NOISY_FIGURES:
                continue
            measured_changes.append({"id": cid, "figure": name, "before": was,
                                     "after": is_now, "ratio": _ratio(was, is_now)})

    noisy = [{"figure": name,
              "before": _sum_figures(before, name),
              "after": _sum_figures(now, name)}
             for name in _NOISY_FIGURES]

    work = _work_growth(list(before.values()), list(now.values()), work_growth_limit)
    return {
        "ok": not (regressions or removed or added or unrecorded or changes
                   or work["exceeded"]),
        "measuredChanges": measured_changes,
        "noisy": [n for n in noisy if n["before"] != n["after"]],
        "work": work,
        "regressions": regressions,
        "fixed": fixed,
        "added": added,
        "removed": removed,
        "unrecorded": unrecorded,
        "changes": changes,
        "summary": {
            "baselinePassRate": (stored.get("summary") or {}).get("passRate"),
            "currentPassRate": (current.get("summary") or {}).get("passRate"),
        },
    }


# ------------------------------------------------------------------------- io
def write_artifact(path: Path | str, artifact: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")


def load_artifact(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def render_diff(diff: dict) -> str:
    lines = [f"基线通过率 {diff['summary']['baselinePassRate']} → "
             f"当前 {diff['summary']['currentPassRate']}"]
    if diff["regressions"]:
        lines.append(f"回归 {len(diff['regressions'])} 例:")
        for item in diff["regressions"]:
            lines.append(f"  [回归] {item['id']}")
            for m in item["mismatches"]:
                lines.append(f"      {m['invariant']}: 期望 {m['expected']!r}，实际 {m['actual']!r}")
    if diff["fixed"]:
        lines.append(f"修复 {len(diff['fixed'])} 例: {', '.join(diff['fixed'])}")
    if diff["removed"]:
        lines.append(f"基线中存在但本次未跑（视为回归）: {', '.join(diff['removed'])}")
    if diff["added"]:
        lines.append(f"基线未覆盖的新增用例 {len(diff['added'])} 个（需重刷基线）: "
                     + ", ".join(diff["added"]))
    if diff.get("unrecorded"):
        lines.append(f"基线未记录的字段 {len(diff['unrecorded'])} 个（需重刷基线）: "
                     + ", ".join(diff["unrecorded"]))
    if diff["changes"]:
        lines.append(f"未被断言的不变式发生了变化 {len(diff['changes'])} 处"
                     "（基线已过时，需显式重刷）:")
        for c in diff["changes"]:
            lines.append(f"  {c['id']}.{c['invariant']}: {c['before']!r} → {c['after']!r}")
    work = diff.get("work") or {}
    if work:
        line = (f"工作量（{work.get('unit') or '未度量'}）"
                f"{work.get('before')} → {work.get('after')}")
        if work.get("ratio") is not None:
            line += f"，×{work['ratio']:.2f}"
        if work.get("exceeded"):
            line += f" —— 超出上限 {work['limit']}，判为回归"
        elif work.get("why"):
            line += f"（{work['why']}）"
        lines.append(line)
    if diff.get("measuredChanges"):
        lines.append(f"实测数字变化 {len(diff['measuredChanges'])} 处"
                     "（仅供参考，不判失败）:")
        for c in diff["measuredChanges"]:
            ratio = f"  ×{c['ratio']:.2f}" if c.get("ratio") is not None else ""
            lines.append(f"  {c['id']}.{c['figure']}: {c['before']!r} → {c['after']!r}{ratio}")
    for n in diff.get("noisy") or []:
        lines.append(f"（{n['figure']} 按套件合计 {n['before']} → {n['after']}，"
                     "逐用例不比较：它测的是机器负载而不是 harness）")
    if diff["ok"]:
        lines.append("与基线一致。")
    elif not diff["regressions"] and not diff["removed"]:
        lines.append("无回归，但基线已过时 —— 确认变化后重刷：`wfos baseline run`")
    return "\n".join(lines)
