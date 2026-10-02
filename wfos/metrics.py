"""Measured cost of running the workflow: tokens, latency, money.

Everything here reads counts a provider actually reported. A count that was
never reported stays `None` end to end — never 0 — so "we were not told" cannot
be averaged into "it was free". Money follows the same rule: a model with no
configured price yields `None`, because a price the harness invented is
indistinguishable from a real one once it is written into a report.

Two consequences worth stating plainly, because they are the difference between
a number that means something and one that does not:

  * Totals are reported **with their coverage**. A sum over 10 steps where only
    2 reported counts is not the run's token count, so `steps_with_usage` is
    returned next to the total rather than being left for the reader to guess.
  * Cost is computed when the step is written (see
    `Harness._step_metrics`), not when it is read. The model in use is only
    knowable while the step is being produced, and a price looked up later could
    belong to a different model — so the stored figure is the historical fact.

The unit is USD per 1,000,000 tokens, matching how providers publish prices.
"""
from __future__ import annotations

from typing import Any

# Token counts are the only figures here a provider may fail to report. Model
# call count and latency are measured by the harness itself, so they are always
# known and are stored as plain numbers.
TOKEN_FIELDS = ("input_tokens", "output_tokens", "cached_tokens",
                "reasoning_tokens")
SUM_FIELDS = TOKEN_FIELDS + ("model_calls", "latency_ms")

_PER_MILLION = 1_000_000


def cost_usd(model: str | None, pricing: dict | None, *,
             input_tokens: int | None = None,
             output_tokens: int | None = None) -> float | None:
    """Cost of one step in USD, or None when it cannot be computed honestly.

    Returns None — never 0 — when the model has no configured price, or when
    neither side has a reported count to price. A 0 would assert the call was
    free, which is a different claim from "we don't know what it cost".
    """
    rates = (pricing or {}).get(model or "")
    if not isinstance(rates, dict):
        return None
    total = 0.0
    priced = False
    for side, tokens in (("input", input_tokens), ("output", output_tokens)):
        rate = rates.get(side)
        if rate is None or tokens is None:
            continue
        total += tokens / _PER_MILLION * rate
        priced = True
    return total if priced else None


def _total(values: list[Any]) -> int | float | None:
    """Sum what is present, or None when nothing was reported.

    `sum([])` is 0, which would be a claim about data that does not exist; the
    distinction between "no rows reported" and "the total is genuinely 0" has to
    survive all the way to the report.
    """
    present = [v for v in values if v is not None]
    return sum(present) if present else None


def _coverage(rows: list[dict]) -> dict[str, int]:
    return {
        "steps": len(rows),
        "steps_with_usage": sum(
            1 for r in rows if any(r.get(f) is not None for f in TOKEN_FIELDS)),
        "steps_with_cost": sum(1 for r in rows if r.get("cost_usd") is not None),
        "steps_with_latency": sum(1 for r in rows if r.get("latency_ms") is not None),
    }


def _totals(rows: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {f: _total([r.get(f) for r in rows]) for f in SUM_FIELDS}
    out["cost_usd"] = _total([r.get("cost_usd") for r in rows])
    return out


def totals_with_coverage(rows: list[dict]) -> dict:
    """Every total for `rows`, each accompanied by how much of it was measured.

    Public because the reliability baseline records the same figures under its own
    (camelCase) names, and it must not grow a second implementation of "sum what
    was reported, and say how much that was".
    """
    return {**_coverage(rows), **_totals(rows)}


def billable_tokens(report: dict) -> int | None:
    """Input + output: the figure a token budget is about.

    Cached and reasoning are *breakdowns* of those two, not additions to them
    (a cached token is still an input token), so including them would
    double-count. None when neither side was reported — a budget cannot be
    enforced against a number nobody gave, and a limit that silently does nothing
    is worse than no limit.
    """
    inputs, outputs = report.get("input_tokens"), report.get("output_tokens")
    if inputs is None and outputs is None:
        return None
    return (inputs or 0) + (outputs or 0)


def run_metrics(repo, run_id: str) -> dict:
    """Measured cost of one run, with coverage."""
    run = repo.get_run(run_id)
    rows = repo.metric_rows(run_id)
    return {
        "run_id": run_id,
        "kind": (run or {}).get("kind"),
        "status": (run or {}).get("status"),
        **totals_with_coverage(rows),
        "by_state": _by_state(rows),
        "by_model": _by_model(rows),
    }


def _group_by(rows: list[dict], field: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(str(r.get(field) or "?"), []).append(r)
    return {name: totals_with_coverage(group) for name, group in groups.items()}


def _by_state(rows: list[dict]) -> dict[str, dict]:
    return _group_by(rows, "state")


def _by_model(rows: list[dict]) -> dict[str, dict]:
    """Which brain produced which steps.

    With `[llm.routing]` a run's steps can come from different models, and a total
    that mixes them cannot answer "what would this have cost on the cheap one" or
    "did the routed role actually get used". `?` is a step from before the model
    was recorded, or one with no model call behind it.
    """
    return _group_by(rows, "model")


def aggregate(repo, *, limit: int | None = None) -> dict:
    """Measured cost across runs, newest runs first when `limit` is set.

    Every total is accompanied by its coverage, so a figure computed over a
    fraction of the runs cannot be mistaken for a figure over all of them.
    """
    runs = repo.list_runs(limit=limit or 10_000)
    run_ids = [r["id"] for r in runs]
    rows: list[dict] = []
    for run_id in run_ids:
        rows.extend(repo.metric_rows(run_id))
    # Grouped, not filtered. `origin` separates real runs from benchmark runs, and
    # a total that silently excluded one of them would be a different kind of
    # dishonest — the reader would see a smaller number and no reason for it.
    # Showing the split keeps the total true and makes the mix visible.
    return {
        "runs": len(runs),
        **totals_with_coverage(rows),
        "by_kind": _by_run_field(rows, runs, "kind"),
        "by_origin": _by_run_field(rows, runs, "origin"),
        "by_model": _by_model(rows),
    }


def _by_run_field(rows: list[dict], runs: list[dict], field: str) -> dict[str, dict]:
    """Metric rows grouped by one column of the run each belongs to."""
    of = {r["id"]: r.get(field) or "?" for r in runs}
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(of.get(row["run_id"], "?")), []).append(row)
    return {name: totals_with_coverage(g) for name, g in sorted(groups.items())}


# ------------------------------------------------------------------- rendering
def _count(value, unknown: str = "-") -> str:
    if value is None:
        return unknown
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render(report: dict, *, model: str | None = None,
           pricing: dict | None = None) -> str:
    """Human-readable report, stating unknown figures as unknown."""
    steps, with_usage = report["steps"], report["steps_with_usage"]
    by_model = report.get("by_model") or {}
    if steps and with_usage == 0:
        why = "本环境无真实 provider，未上报用量"
    elif with_usage < steps:
        why = f"仅 {with_usage}/{steps} 步上报了用量，合计只是这部分的和"
    else:
        why = f"{with_usage}/{steps} 步均已上报"

    lines = [f"步骤数 {steps}，其中上报用量 {with_usage}、可计价 "
             f"{report['steps_with_cost']}、有延迟 {report['steps_with_latency']}",
             f"  （{why}）"]
    for label, field in (("input tokens ", "input_tokens"),
                         ("output tokens", "output_tokens"),
                         ("cached tokens", "cached_tokens"),
                         ("推理 tokens  ", "reasoning_tokens"),
                         ("模型调用     ", "model_calls")):
        lines.append(f"  {label}  {_count(report.get(field))}")
    latency = report.get("latency_ms")
    lines.append(f"  延迟         {_count(latency)} ms"
                 + (f"  ({latency / 1000:.1f}s)" if latency else ""))
    cost = report.get("cost_usd")
    if cost is None:
        detail = ("无步骤上报可计价的用量" if report["steps_with_usage"] == 0
                  else f"模型 {_named_models(report) or model or '?'} 未在 [pricing] 配置单价")
        lines.append(f"  成本         -  ({detail})")
    else:
        lines.append(f"  成本         ${cost:.4f}")
    if len(by_model) > 1:
        # Only when they differ: with one model this is the whole report repeated.
        lines.append("  按模型：")
        for name, group in sorted(by_model.items()):
            lines.append(f"    {name}  {group['steps']} 步，"
                         f"in {_count(group.get('input_tokens'))} / "
                         f"out {_count(group.get('output_tokens'))}，"
                         f"${_count(group.get('cost_usd'))}")
    return "\n".join(lines)


def _named_models(report: dict) -> str:
    """The models that produced steps, for a message about their missing price.

    Naming the configured default instead would point an operator at a model that
    may not appear in the run at all once routing is in use.
    """
    names = [n for n in sorted(report.get("by_model") or {}) if n != "?"]
    return ", ".join(names)
