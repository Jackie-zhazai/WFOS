"""The append-only trace: what a run did, in the order it did it.

The harness already keeps four tables of what happened — steps, transitions,
approvals, tool calls — and they are the working state, not a record. Steps get
**deleted** so a state can be re-executed (`RE_RUNNABLE`), `runs.error` is
overwritten in place, and the four tables carry four independent AUTOINCREMENT
sequences with no key joining them and a `created_at` that is second-resolution,
local and naive. So "what did this run actually do, in order" had no answer that
could be rebuilt afterwards: the record showed fewer steps than were executed and
nothing said which.

This is that record. It is append-only: nothing is ever updated or deleted, and a
change that would have rewritten history is itself written as an event
(`STEP_INVALIDATED`). `steps` and `transitions` keep their semantics — the trace
explains them rather than replacing them.

**Ordering is `event_id` and nothing else.** SQLite assigns it atomically inside
the insert, so it is a total order with no clock involved. `seq` is a readable
position within one run and is correct while the execution lease is held, which
is where every harness-emitted event happens; `at` is for a human.

**If the model can see it, it is recorded** — the same rule the prompt record
already follows. An event that describes a decision carries what the decision was
made on, so a reader never has to infer it from a later state.
"""
from __future__ import annotations

from typing import Any

# Who caused an event. A decision the workflow made and a decision a person made
# are different facts about a run, and only one of them is reproducible.
ACTOR_HARNESS = "harness"
ACTOR_MODEL = "model"
ACTOR_HUMAN = "human"
ACTOR_RUNNER = "runner"

# The vocabulary. Named here so every kind of thing a trace can say is findable in
# one place — and so an emit site cannot quietly invent a spelling that no reader
# knows to look for.
RUN_CREATED = "run.created"
RUN_STATUS = "run.status"
RUN_RESUMED = "run.resumed"
TASK_LOADED = "task.loaded"
STEP_COMPLETED = "step.completed"
STEP_FAILED = "step.failed"
STEP_INVALIDATED = "step.invalidated"
TRANSITION = "transition"
APPROVAL_REQUESTED = "approval.requested"
APPROVAL_DECIDED = "approval.decided"
BUDGET_EXCEEDED = "budget.exceeded"
REUSE_REFUSED = "reuse.refused"
DELEGATED = "delegated"
TOOL_CALLED = "tool.called"
EVALUATED = "evaluated"
# RSI. A candidate's status move is a fact about a decision, and it belongs in the
# same ordered history as everything else the harness did.
CANDIDATE_PROPOSED = "rsi.candidate.proposed"
CANDIDATE_STATUS = "rsi.candidate.status"

TYPES = (RUN_CREATED, RUN_STATUS, RUN_RESUMED, TASK_LOADED, STEP_COMPLETED,
         STEP_FAILED, STEP_INVALIDATED, TRANSITION, APPROVAL_REQUESTED,
         APPROVAL_DECIDED, BUDGET_EXCEEDED, REUSE_REFUSED, DELEGATED,
         TOOL_CALLED, EVALUATED, CANDIDATE_PROPOSED, CANDIDATE_STATUS)

# Types whose payload the harness produces itself and whose absence of a payload
# would still be meaningful. Not enforced — listed so the set is visible.
_ = TYPES


def record(repo, type: str, *, run_id: str = "", session_id: str = "",
           actor: str = ACTOR_HARNESS, payload: dict[str, Any] | None = None) -> int:
    """Append one event; return its `event_id` so a caller can correlate a row.

    Returns the id rather than nothing because that is how the step, transition
    and tool-call rows name the event that describes them — without it the four
    tables stay unjoinable, which is the problem this table exists to solve.
    """
    return repo.add_event(type=type, run_id=run_id, session_id=session_id,
                          actor=actor, payload=payload)


def for_run(repo, run_id: str) -> list[dict]:
    """This run's events, oldest first."""
    return repo.events_for_run(run_id)


def render(events: list[dict]) -> str:
    """The trace as lines, one per event.

    `seq` and `event_id` are both shown. Only the second orders anything, and a
    reader who saw one number would not know which.
    """
    lines: list[str] = []
    for event in events:
        payload = event.get("payload") or {}
        detail = " ".join(f"{k}={_short(v)}" for k, v in payload.items())
        lines.append(f"#{event['seq']:<4} e{event['event_id']:<5} "
                     f"{event['actor']:<8} {event['type']:<20} {detail}".rstrip())
    return "\n".join(lines)


def _short(value: Any, limit: int = 80) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[:limit] + "…"
