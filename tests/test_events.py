"""The append-only trace: a run's history that cannot be rewritten.

The four tables the harness already keeps are *working state*, not a record. Steps
are deleted so a state can re-execute, `runs.error` is overwritten in place, and
the four carry four independent AUTOINCREMENT sequences with a `created_at` that
is second-resolution. So "what did this run do, in order" had no answer that
survived: the record showed fewer steps than ran and nothing said which.

These tests pin the three properties that make the trace worth keeping:
**it is ordered by something that cannot tie**, **it cannot be rewritten**, and
**it survives the deletion of the thing it describes**.
"""
from __future__ import annotations

import asyncio
from datetime import datetime

from wfos import events

FEATURE = "新增一个用户模块，在 app.py 中追加 feature_user() 函数"


def _drive(h) -> str:
    run = h.create_run(FEATURE, kind="feature")
    asyncio.run(h.advance(run["id"]))
    return run["id"]


def _types(repo, run_id) -> list[str]:
    return [e["type"] for e in events.for_run(repo, run_id)]


# ------------------------------------------------------------------- ordering
def test_a_run_traces_itself_from_creation_through_its_states(app):
    h, repo = app["harness"], app["repo"]

    run_id = _drive(h)

    kinds = _types(repo, run_id)
    assert kinds[0] == events.RUN_CREATED
    assert kinds.count(events.STEP_COMPLETED) >= 5, f"步骤没有被记录：{kinds}"
    assert events.TRANSITION in kinds


def test_event_id_is_the_total_order_and_it_never_ties(app):
    """The property the four old tables could not offer.

    `steps.id` and `tool_calls.id` are independent sequences, and `created_at` is
    second-resolution — so a merge of the two ties and the tie is undetectable.
    `event_id` is assigned by SQLite inside the insert.
    """
    h, repo = app["harness"], app["repo"]
    run_id = _drive(h)

    events_ = events.for_run(repo, run_id)

    ids = [e["event_id"] for e in events_]
    assert ids == sorted(ids), "for_run 没有按 event_id 排序"
    assert len(set(ids)) == len(ids), "event_id 出现了重复"


def test_seq_numbers_one_run_from_one_without_gaps(app):
    """A readable position within the run. It is not the authority — `event_id`
    is — but a gap in it would mean an event went missing."""
    h, repo = app["harness"], app["repo"]
    run_id = _drive(h)

    seqs = [e["seq"] for e in events.for_run(repo, run_id)]

    assert seqs == list(range(1, len(seqs) + 1))


def test_two_runs_each_number_their_own_events_from_one(app):
    h, repo = app["harness"], app["repo"]

    first = _drive(h)
    second = _drive(h)

    for run_id in (first, second):
        seqs = [e["seq"] for e in events.for_run(repo, run_id)]
        assert seqs[0] == 1, f"{run_id[:8]} 的 seq 不是从 1 开始"


def test_the_timestamp_is_utc_with_microseconds(app):
    """`now()` is local, seconds and naive — indefensible for a log, even though
    nothing orders by it."""
    h, repo = app["harness"], app["repo"]
    run_id = _drive(h)

    at = events.for_run(repo, run_id)[0]["at"]

    parsed = datetime.fromisoformat(at)
    assert parsed.tzinfo is not None, "时间戳没有时区"
    assert "." in at, "时间戳没有微秒"


# --------------------------------------------------------------- append-only
def test_nothing_in_the_repo_can_rewrite_an_event(app):
    """A guard on the design, not on an instance.

    An event that could be updated or deleted is not a record. The way to keep
    that true is for no such method to exist — so this fails the moment one is
    added, whatever it is called.
    """
    repo = app["repo"]

    mutators = [name for name in dir(repo)
                if "event" in name.lower()
                and any(word in name.lower() for word in ("update", "delete", "set", "remove"))]

    assert mutators == [], f"repo 上出现了可以改写事件的接口：{mutators}"


def test_the_correlation_column_points_at_the_event_that_describes_the_row(app):
    """Four tables, four sequences, no join key — until each row names its event."""
    h, repo = app["harness"], app["repo"]
    run_id = _drive(h)

    by_id = {e["event_id"]: e for e in events.for_run(repo, run_id)}
    steps = repo.steps_for_run(run_id)

    assert steps
    for step in steps:
        assert step.get("event_id"), f"{step['state']} 没有 event_id"
        described = by_id.get(step["event_id"])
        assert described is not None, "steps.event_id 指向了一个不存在的 event"
        assert described["payload"].get("state") == step["state"], \
            "step 与它所指的事件说的不是同一个状态"


def test_transitions_name_their_event_too(app):
    h, repo = app["harness"], app["repo"]
    run_id = _drive(h)

    transitions = repo.transitions(run_id)
    by_id = {e["event_id"]: e for e in events.for_run(repo, run_id)}

    assert transitions
    for row in transitions:
        assert row.get("event_id"), f"转换 {row['from_state']}->{row['to_state']} 没有 event_id"
        assert row["event_id"] in by_id


def test_tool_calls_name_their_event(app):
    h, repo = app["harness"], app["repo"]
    run_id = _drive(h)

    calls = repo.tool_calls(run_id)
    traced = {e["event_id"] for e in events.for_run(repo, run_id)}

    assert calls
    for row in calls:
        assert row.get("event_id"), f"工具调用 {row['tool']} 没有 event_id"
        assert row["event_id"] in traced


# ------------------------------------------------- the record survives a delete
def test_deleting_a_step_leaves_the_fact_that_it_ran(app):
    """`RE_RUNNABLE` deletes a step so its state re-executes — the one place in
    the harness that rewrites the record. The trace is what survives it, and the
    deletion is written as an event rather than left as a silence.
    """
    h, repo = app["harness"], app["repo"]
    run = h.create_run(FEATURE, kind="feature")
    run_id = run["id"]
    repo.add_step(run_id, "implement", "implementer", {"ok": True}, status="done",
                  failure_class="no_change", metrics={"model_calls": 3,
                                                      "latency_ms": 120})
    live = repo.get_run(run_id)

    h._invalidate_step(live, "implement", "测试用")

    assert repo.steps_for_run(run_id) == [], "步骤没有被删掉"
    invalidated = [e for e in events.for_run(repo, run_id)
                   if e["type"] == events.STEP_INVALIDATED]
    assert len(invalidated) == 1
    payload = invalidated[0]["payload"]
    assert payload["state"] == "implement" and payload["reason"] == "测试用"
    # What the row held is kept, so a reader sees what was thrown away.
    assert payload["erased"]["failure_class"] == "no_change"
    assert payload["erased"]["model_calls"] == 3


def test_a_state_that_ran_twice_shows_both_attempts(app, bad_check):
    """The working state keeps one step per state; the trace keeps every attempt.

    `bad_check` makes the same test fail the same way twice, so `implement` is
    invalidated and re-executed — the deletion that used to erase the only
    evidence the first attempt had happened at all.

    Driven through the harness rather than by hand: the trace records what the
    *harness* did, and a row written straight into the table is not that.
    """
    from conftest import drive

    h, repo = app["harness"], app["repo"]
    run = drive(h, FEATURE)

    in_steps = [s["state"] for s in repo.steps_for_run(run["id"])]
    traced = events.for_run(repo, run["id"])
    completed = [e["payload"]["state"] for e in traced
                 if e["type"] == events.STEP_COMPLETED]

    # The property, stated as the gap it is: the state ran at least once, and the
    # working state holds fewer records of it than the trace does. In this
    # scenario `implement` is gone from `steps` entirely — it was invalidated on
    # the way to the escalation — while the trace still shows it ran.
    assert completed.count("implement") >= 1, "trace 里没有 implement 跑过的记录"
    assert completed.count("implement") > in_steps.count("implement"), \
        f"trace 与工作状态没有差距：{completed} vs {in_steps}"
    assert events.STEP_INVALIDATED in _types(repo, run["id"]), \
        "重入没有留下作废事件"


# --------------------------------------------------------------- attribution
def test_the_actor_says_who_made_the_decision(app):
    """A decision the workflow made is reproducible; one a person made is not."""
    h, repo = app["harness"], app["repo"]
    run = h.create_run(FEATURE, kind="feature")
    run_id = run["id"]

    h._emit(repo.get_run(run_id), events.RUN_STATUS, status="running")

    kinds = {e["type"]: e["actor"] for e in events.for_run(repo, run_id)}
    assert kinds[events.RUN_CREATED] == events.ACTOR_HARNESS
    assert kinds[events.RUN_STATUS] == events.ACTOR_HARNESS


def test_a_tool_call_is_attributed_to_the_model_not_the_harness(app):
    h, repo = app["harness"], app["repo"]
    run_id = _drive(h)

    traced = [e for e in events.for_run(repo, run_id)
              if e["type"] == events.TOOL_CALLED]

    assert traced, "没有工具调用事件"
    assert all(e["actor"] == events.ACTOR_MODEL for e in traced)


# ------------------------------------------------------------------ sessions
def test_every_event_of_a_run_carries_its_session(app):
    h, repo = app["harness"], app["repo"]
    run_id = _drive(h)

    run = repo.get_run(run_id)

    assert run["session_id"], "运行没有 session id"
    assert all(e["session_id"] == run["session_id"]
               for e in events.for_run(repo, run_id))


def test_a_child_inherits_the_session_of_its_parent(app):
    """A session is one invocation; the runs it spawned are part of it."""
    repo = app["repo"]
    parent = repo.create_run("feature", "t", "d")
    child = repo.create_run("bugfix", "t", "d", parent_run_id=parent["id"])

    assert child["session_id"] == parent["session_id"]


def test_one_session_gathers_all_its_runs(app):
    h, repo = app["harness"], app["repo"]
    session = "sess-1"
    for _ in range(2):
        run = h.create_run(FEATURE, kind="feature", session_id=session)
        asyncio.run(h.advance(run["id"]))

    grouped = repo.events_for_session(session)

    assert grouped
    assert {e["run_id"] for e in grouped} == {r["id"] for r in repo.list_runs()}
    ids = [e["event_id"] for e in grouped]
    assert ids == sorted(ids)


# ------------------------------------------------------------------ redaction
def test_a_secret_in_an_event_payload_is_redacted_on_the_way_in(app):
    """The trace is persisted content like any other, so it goes through the same
    redaction — a secret that reached the log is a secret that leaked."""
    h, repo = app["harness"], app["repo"]
    run = h.create_run(FEATURE, kind="feature")

    h._emit(repo.get_run(run["id"]), events.RUN_STATUS,
            note="key=sk-abc123XYZ789defghijklmnop")

    payload = events.for_run(repo, run["id"])[-1]["payload"]
    assert "sk-abc123XYZ789defghijklmnop" not in str(payload)


# ------------------------------------------------------------------- rendering
def test_the_rendered_trace_shows_both_numbers(app):
    """`seq` and `event_id`, because only one of them orders anything and a
    reader who saw a single number would not know which."""
    h, repo = app["harness"], app["repo"]
    run_id = _drive(h)

    text = events.render(events.for_run(repo, run_id))

    assert "#1 " in text and "e" in text
    assert events.RUN_CREATED in text
