"""Procedures learned from runs that got past a failure, kept as assets.

A skill is not memory. Memory records **what happened**; a skill is a
**procedure** — "when this class of failure appears, this is what got past it" —
and procedural knowledge needs what free-text memory does not have: a version,
evidence that it worked, and a way to be superseded. WorkBuddy's case for keeping
process knowledge out of long-term memory is the same one: "how to do things" in
prose has no version, no evaluation, no approval and no rollback.

The difference from the wiki is where the content comes from. A wiki entry is a
model's distillation. A skill here is **derived from the harness's own record** —
which run hit the class, what the run that got past it actually changed, and what
the verifier measured afterwards. That is what lets it be recorded without a human
in the loop and still carry evidence rather than a claim.

This is the honest form of "feed the failure classes back": not a sentence in a
prompt, but a versioned entry keyed by the class it addresses, retrievable exactly
when a run is stuck on that class.
"""
from __future__ import annotations

import hashlib
import json

from .models import ORIGIN_INTERACTIVE

# Where the trigger comes from. `payload.failure_attempts` is keyed
# `<state>:<class>`, the same counter the escalation rule reads.
_MAX_PROCEDURE_FILES = 12


def content_digest(row: dict) -> str:
    """sha256 of a skill version's *content*.

    Only the content: the trigger, the text, the files, the version number. Not
    `superseded`, because that is the current pointer rather than part of what a
    version says, and a pointer that moves is not a version that changed.
    """
    blob = json.dumps({k: row.get(k) for k in
                       ("trigger", "title", "procedure", "files", "version")},
                      sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def class_of(key: str) -> str:
    """The failure class out of a `failure_attempts` key (`state:class`)."""
    return key.split(":", 1)[1] if ":" in key else key


def stuck_on(run: dict) -> list[str]:
    """Failure classes this run has already stumbled on, most recent first.

    Read from the retry counters rather than from history: what matters is which
    classes are *live* for this run, and a class that was hit and fully recovered
    from is no longer one of them.
    """
    attempts = (run.get("payload") or {}).get("failure_attempts") or {}
    live = [class_of(k) for k, v in attempts.items() if v]
    return sorted(set(live))


def _failed_states(repo, run_id: str) -> dict[str, set[str]]:
    """`state -> the failure classes it has ever carried`, read from the trace.

    **Not from `steps`.** A failing step is *deleted* when its state re-runs, and
    the recovery that matters most — a child run repairing a regression — deletes
    exactly the step that held the evidence (`resume_after_child` invalidates the
    state it is about to re-open). Reading `steps` therefore cannot see the one
    recovery worth learning from, and reports "nothing to learn" instead.

    The trace is append-only and carries both halves: `step.failed` for a state
    that failed, and `step.invalidated` with the erased row's facts for a state
    that failed and was then re-run. This is the fact source the trace exists to
    be, and the derivation was simply still reading the working state instead.

    **The step rows are read as well**, and the two are unioned. A run whose record
    was seeded directly — a test fixture, or a database written before the trace
    existed — has no events at all, and reading only the trace would report that
    nothing was ever recovered from. The union is the set of (state, class) pairs
    either source remembers, which is what the question actually asks.
    """
    from . import events

    out: dict[str, set[str]] = {}
    for event in events.for_run(repo, run_id):
        payload = event.get("payload") or {}
        state = str(payload.get("state") or "")
        if not state:
            continue
        if event["type"] == events.STEP_FAILED:
            failure_class = str(payload.get("failure_class") or "")
        elif event["type"] == events.STEP_INVALIDATED:
            failure_class = str((payload.get("erased") or {}).get("failure_class") or "")
        else:
            continue
        if failure_class:
            out.setdefault(state, set()).add(failure_class)

    for step in repo.steps_for_run(run_id):
        failure_class = str(step.get("failure_class") or "")
        if failure_class:
            out.setdefault(str(step["state"]), set()).add(failure_class)
    return out


def recoveries(repo, run: dict) -> list[str]:
    """Classes a finished run hit *and* got past.

    Only a run that ended `completed` can teach this: a procedure learned from a
    run that also failed says nothing about getting past the failure. And the
    class has to have been genuinely recovered from — the state that carried it
    later has a `done` step — not merely recorded once on the way down.
    """
    if run.get("status") != "completed":
        return []
    failed = _failed_states(repo, run["id"])
    steps = repo.steps_for_run(run["id"])
    out: list[str] = []
    for failure_class in sorted({c for classes in failed.values() for c in classes}):
        states = {state for state, classes in failed.items() if failure_class in classes}
        if any(s["state"] in states and s["status"] == "done" for s in steps):
            out.append(failure_class)
    return out


def events_for_run(repo, run_id: str):
    """The run's trace. Imported here so `skills` stays usable without it."""
    from . import events

    return events.for_run(repo, run_id)


def _procedure(repo, run: dict, trigger: str) -> tuple[str, list[str], str]:
    """(title, files changed, procedure text) for one recovered class."""
    files = repo.affected_paths_for_run(run["id"])
    # Same source as the derivation, for the same reason: the failing step is
    # deleted when its state re-runs, so reading `steps` would describe a recovery
    # as having happened zero times and name no state at all.
    failed = _failed_states(repo, run["id"])
    failed_states = sorted(state for state, classes in failed.items()
                           if trigger in classes)
    attempts = sum(1 for event in events_for_run(repo, run["id"])
                   if (event.get("payload") or {}).get("state") in failed_states
                   and (event["type"] == "step.failed"
                        or event["type"] == "step.invalidated"))
    verifier = {}
    for state in ("build_test", "regression_verify", "verify_regression"):
        step = repo.get_step(run["id"], state)
        if step and step["status"] == "done":
            verifier = step["output_json"] or {}
            break

    lines = [
        f"触发: {trigger}",
        f"出处: 运行 {run['id']}（{run['kind']}，终态 {run['status']}）",
        f"失败发生在状态: {'、'.join(failed_states)}（共 {attempts} 次）",
    ]
    if files:
        shown = files[:_MAX_PROCEDURE_FILES]
        more = f" 等 {len(files)} 个" if len(files) > len(shown) else ""
        lines.append(f"最终改动的文件: {'、'.join(shown)}{more}")
    if verifier:
        build = verifier.get("build") or {}
        tests = verifier.get("tests") or {}
        lines.append(
            f"通过时的验证结果: build.ok={bool(build.get('ok'))}，"
            f"tests.failed={tests.get('failed', '?')}，"
            f"verdict={verifier.get('verdict', '?')}")
    lines.append(
        "这些是 Harness 自己观测到的，不是模型的结论；"
        "本次运行若再次落到该失败类型，可参照上面的改动范围与验证口径。")
    title = f"{trigger}：{run['title'][:40]}"
    return title, files, "\n".join(lines)


def record_from_run(repo, run: dict) -> list[dict]:
    """Distil a skill for every class this run hit and got past.

    Called for every finished run; a run that never failed produces nothing, which
    is the common case.

    The run's `origin` travels with the procedure and decides its status: an
    interactive run's goes live, a task/benchmark run's is a candidate. Without
    that, running a benchmark into a failure class was a way to write a procedure
    straight into a production prompt — no human, no benchmark, no gate.
    """
    recorded: list[dict] = []
    for trigger in recoveries(repo, run):
        title, files, procedure = _procedure(repo, run, trigger)
        recorded.append(repo.add_skill(trigger, title, procedure, run["id"],
                                       files=files, origin=origin_of(run)))
    return recorded


def origin_of(run: dict) -> str:
    """Where a run came from, defaulting to interactive.

    A row written before the column existed is an interactive run, and saying so
    is not a guess — it is what the code did.
    """
    return str(run.get("origin") or ORIGIN_INTERACTIVE)


def skills_for_run(repo, run: dict) -> list[dict]:
    """Procedures this run may use for the failure classes it is stuck on.

    Scoped to where the run came from: see `Repo.skills_for`.
    """
    return repo.skills_for(stuck_on(run), origin=origin_of(run))


def render(skills: list[dict]) -> str:
    """The skills block as it enters a prompt.

    Names the class, the evidence run and the version, because a procedure whose
    provenance a reader cannot check is indistinguishable from advice.
    """
    blocks: list[str] = []
    for skill in skills:
        blocks.append(
            f"### {skill['trigger']}（版本 {skill['version']}，"
            f"由运行 {skill['evidence_run']} 证明）\n{skill['procedure']}")
    return "\n\n".join(blocks)
