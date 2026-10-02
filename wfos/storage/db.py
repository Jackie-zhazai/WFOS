"""SQLite connection management and schema."""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id            TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,                -- feature | bugfix
    title         TEXT NOT NULL,
    description   TEXT NOT NULL,
    state         TEXT NOT NULL,
    status        TEXT NOT NULL,
    parent_run_id TEXT,
    root_run_id   TEXT,
    attempt       INTEGER NOT NULL DEFAULT 0,
    payload       TEXT NOT NULL DEFAULT '{}',   -- JSON: keywords, pending_state, child_run_id, rerun_state, plan...
    result        TEXT,                          -- JSON final structured result
    error         TEXT,
    -- Resume contract: the environment hash and key-file digests the persisted
    -- steps were produced under. See harness/identity.py.
    identity_hash TEXT,
    key_files     TEXT,                          -- JSON {path: sha256|null}
    -- Execution lease: who is currently advancing this run, and until when.
    -- A stale lease (expired) is claimable, so a killed process cannot wedge a
    -- run permanently.
    owner         TEXT,
    lease_until   TEXT,
    -- Groups the runs of one invocation: a benchmark suite is one session, an
    -- experiment cell is one, an interactive run is one of its own. Every trace
    -- event carries it, so "what did this invocation do" is answerable without
    -- walking run ids.
    session_id    TEXT,
    -- Where this run came from: `interactive` (a human asked for it), `task` or
    -- `benchmark` (a Runner did). It is the boundary that keeps an artificial run
    -- out of production: memory and skills are scoped by it, so a benchmark run
    -- cannot seed the prompt of a real one just by touching the same file.
    origin        TEXT NOT NULL DEFAULT 'interactive',
    -- Set when a task drove the run. `task_version` travels with the run because
    -- the loop's whole point is comparing results across task revisions, and a
    -- result that does not say which revision produced it cannot be compared.
    task_id       TEXT,
    task_version  INTEGER,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS steps (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    state       TEXT NOT NULL,
    agent       TEXT,
    status      TEXT NOT NULL,                   -- done | failed
    input_json  TEXT,
    output_json TEXT,
    error       TEXT,
    -- Why this state did not succeed: build_error | test_failure | regression |
    -- verification_failed | no_change. NULL means it succeeded. Retry budgets are
    -- counted per (state, class) so a repeated *kind* of failure can escalate.
    failure_class TEXT,
    -- What it cost to produce this step. NULL means the count was not reported
    -- (or, for cost_usd, that the model has no price configured) — never 0, so
    -- "not measured" cannot be averaged into "free" by any later aggregate.
    input_tokens  INTEGER,
    output_tokens INTEGER,
    cached_tokens INTEGER,
    -- The portion of output_tokens spent thinking, when the provider breaks it
    -- out. Same rule as the rest: NULL means unreported, never 0.
    reasoning_tokens INTEGER,
    model_calls   INTEGER,
    latency_ms    INTEGER,
    cost_usd      REAL,
    -- Which event in the trace describes this row. Four tables had four
    -- independent AUTOINCREMENT sequences and no key joining them, so "which
    -- event does this step belong to" had no answer.
    event_id      INTEGER,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_steps_run ON steps(run_id, state);

CREATE TABLE IF NOT EXISTS transitions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL,
    from_state   TEXT NOT NULL,
    to_state     TEXT NOT NULL,
    decision     TEXT NOT NULL,                  -- suggested | validated | blocked | rejected | auto | harness_override
    reason       TEXT,
    suggested_by TEXT,
    event_id     INTEGER,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS approvals (
    id          TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL,
    action      TEXT NOT NULL,
    scope       TEXT,
    risk_level  TEXT,
    reason      TEXT,
    status      TEXT NOT NULL,                   -- pending | approved | rejected
    decided_by  TEXT,
    decision_note TEXT,
    created_at  TEXT NOT NULL,
    decided_at  TEXT
);

CREATE TABLE IF NOT EXISTS evidence (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT NOT NULL,
    kind       TEXT NOT NULL,
    source     TEXT,
    content    TEXT,
    confidence TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT NOT NULL,
    agent      TEXT,
    tool       TEXT NOT NULL,
    args_json  TEXT,
    ok         INTEGER NOT NULL DEFAULT 0,
    error      TEXT,
    -- Stable outcome vocabulary, so failures can be counted instead of read.
    -- status: ok | error | rejected; error_code and security_event come from
    -- policy.classify_error (typed exceptions, never message text).
    status         TEXT,
    error_code     TEXT,
    security_event TEXT,
    -- Where the call came from: "agent" (in-process loop) or "mcp" (external
    -- client over stdio). The `agent` column stays a plain role name.
    source         TEXT,
    -- Change attribution: what the workspace snapshot diff actually observed
    -- for side-effecting tools (JSON array of paths, and a human summary).
    affected_paths TEXT,
    diff_summary   TEXT,
    event_id       INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS skills (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    -- What this procedure is for: the failure class it addresses. Not free text —
    -- it is the same vocabulary the retry budgets and the escalation rule use, so
    -- a skill is retrievable exactly when a run is stuck on that class.
    trigger     TEXT NOT NULL,
    title       TEXT NOT NULL,
    procedure   TEXT NOT NULL,
    -- The run that proved it: a run that hit `trigger` and still completed. The
    -- evidence is the harness's own record, not the model's claim about itself.
    evidence_run TEXT NOT NULL,
    -- Bumped when a newer run produces a procedure for the same trigger; the
    -- older row is kept so the history of "what we believed" survives.
    version     INTEGER NOT NULL DEFAULT 1,
    superseded  INTEGER NOT NULL DEFAULT 0,
    files       TEXT NOT NULL DEFAULT '[]',
    -- sha256 of this version's content, computed when it is written. Rollback
    -- compares it against the digest recorded in the promotion that named this
    -- version; a mismatch means the history was edited after the fact.
    digest      TEXT NOT NULL DEFAULT '',
    -- Lifecycle. `live` may be injected into a prompt; `candidate` may not.
    -- A skill derived from an *interactive* run is live on sight: a human drove
    -- that run, and the procedure is the harness's own record of it. One derived
    -- from a task/benchmark run is a candidate, because an artificial environment
    -- must not be able to write into production prompts by running into a failure
    -- class — that is the RSI loop's job, and it goes through the promotion gate.
    status      TEXT NOT NULL DEFAULT 'live',    -- candidate | verified | live
    -- Which environment the evidence run came from. Kept even after promotion, so
    -- a promoted skill can still be traced back to what proved it.
    origin      TEXT NOT NULL DEFAULT 'interactive',
    -- The skill this row was derived from, when it came from a proposal rather
    -- than from a run. Provenance a reader can follow; `superseded` only says
    -- which one is current.
    parent_id   INTEGER REFERENCES skills(id),
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_skills_trigger ON skills(trigger, superseded);

-- Promotion: the act that turns a candidate into a production skill version.
--
-- Append-only, and it carries the digests rather than only the version numbers.
-- A rollback has to be able to tell "this is the v1 that was promoted against"
-- from "this is a v1 somebody edited afterwards", and the only place a digest can
-- live and still be trustworthy is a record the tamperer does not control.
CREATE TABLE IF NOT EXISTS promotions (
    promotion_id     TEXT PRIMARY KEY,
    candidate_id     TEXT NOT NULL,
    candidate_version INTEGER NOT NULL,
    parent_skill_version   INTEGER NOT NULL,
    parent_digest          TEXT NOT NULL,
    promoted_skill_version INTEGER NOT NULL,
    promoted_digest        TEXT NOT NULL,
    source_experiment TEXT NOT NULL DEFAULT '',
    source_runs      TEXT NOT NULL DEFAULT '[]',
    evaluation       TEXT NOT NULL DEFAULT '{}',
    -- The `evaluations` rows this promotion was decided on, as ids. The
    -- verdict above says what was judged; these say which persisted
    -- rows say so, so a reader can walk back to (run, task) without
    -- trusting a copy.
    evaluation_ids   TEXT NOT NULL DEFAULT '[]',
    compare_result   TEXT NOT NULL DEFAULT '{}',
    gate             TEXT NOT NULL DEFAULT '{}',
    actor            TEXT NOT NULL DEFAULT 'cli',
    reason           TEXT NOT NULL DEFAULT '',
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_promotions_candidate ON promotions(candidate_id, candidate_version);

-- Rollback: moving the current pointer back to a version that already exists.
--
-- Not a delete. The version that was current stays exactly where it was — this
-- record is the whole of what changed, which is that something else is current now.
CREATE TABLE IF NOT EXISTS rollbacks (
    rollback_id  TEXT PRIMARY KEY,
    trigger      TEXT NOT NULL,
    from_version INTEGER NOT NULL,
    to_version   INTEGER NOT NULL,
    from_digest  TEXT NOT NULL,
    to_digest    TEXT NOT NULL,
    actor        TEXT NOT NULL DEFAULT 'cli',
    reason       TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rollbacks_skill ON rollbacks(trigger, created_at);

CREATE TABLE IF NOT EXISTS wiki (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    kind       TEXT NOT NULL,                    -- authoritative | case | candidate
    title      TEXT NOT NULL,
    content    TEXT NOT NULL,
    source     TEXT,
    evidence   TEXT NOT NULL DEFAULT '[]',
    scope      TEXT,
    verified   INTEGER NOT NULL DEFAULT 0,
    trust      TEXT NOT NULL DEFAULT 'medium',
    version    INTEGER NOT NULL DEFAULT 1,
    run_id     TEXT,
    status     TEXT NOT NULL DEFAULT 'pending',  -- pending | verified | published | rejected
    tags       TEXT NOT NULL DEFAULT '[]',
    checksum   TEXT,
    created_at TEXT NOT NULL
);

-- The append-only trace: what the run did, in the order it did it.
--
-- It does not replace `steps`/`transitions`; those stay the working state, with
-- their own semantics — including `RE_RUNNABLE` deleting a step so a state can
-- be re-executed. The trace is the *fact source*: the step was deleted, the fact
-- that it ran is not, and the deletion is itself an event.
--
-- `event_id` is the ordering authority and the only one. SQLite assigns it
-- atomically inside the insert, so it is a total order with no clock involved —
-- which matters because the four other tables have four independent
-- AUTOINCREMENT sequences and their `created_at` is second-resolution.
CREATE TABLE IF NOT EXISTS events (
    event_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    -- Position within one run, for a reader. Allocated inside the execution
    -- lease, so it is race-free exactly where the lease is held; `event_id`
    -- stays the authority everywhere.
    seq          INTEGER NOT NULL,
    -- UTC with microseconds, for a human. Never used for ordering.
    at           TEXT NOT NULL,
    run_id       TEXT,
    -- Groups the runs of one invocation: a benchmark suite is one session, an
    -- experiment cell is one session, an interactive run is one of its own.
    session_id   TEXT,
    -- Who caused it: harness | model | human | runner. A decision the workflow
    -- made and a decision a person made are different facts about a run.
    actor        TEXT NOT NULL,
    type         TEXT NOT NULL,
    payload      TEXT NOT NULL DEFAULT '{}',
    schema_version INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, seq);

-- Judgements. A verdict is a property of *a run under a task version*, and it has
-- to outlive the run: a promotion decision is justified by evaluations that were
-- recorded before it, possibly by a different process. `payload` was the obvious
-- home and the wrong one — it is already a catch-all, its merge is not atomic,
-- and nothing queries it.
CREATE TABLE IF NOT EXISTS evaluations (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL,
    task_id      TEXT NOT NULL,
    task_version INTEGER NOT NULL,
    verdict      TEXT NOT NULL,                  -- pass | fail
    -- JSON: axis -> pass|fail|unknown, and the lists the verdict was reached from.
    axes         TEXT NOT NULL DEFAULT '{}',
    reasons      TEXT NOT NULL DEFAULT '[]',
    failures     TEXT NOT NULL DEFAULT '[]',
    -- The gate checks that failed, named separately: "what did the judge never
    -- get to see" is the question a reader has when a score exists anyway.
    hard_gate    TEXT NOT NULL DEFAULT '[]',
    event_id     INTEGER,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_evaluations_run ON evaluations(run_id);

-- RSI candidates: a proposed change with its provenance and its status.
--
-- The content of a candidate is written once. Only `status` moves, and every move
-- is also an event — so "what was proposed" is a fact that cannot be edited, while
-- "where did it get to" is a state with a history. `UNIQUE(candidate_id, version)`
-- is what makes a new version a new row rather than a rewrite of the old one.
CREATE TABLE IF NOT EXISTS candidates (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL,
    version      INTEGER NOT NULL,
    type         TEXT NOT NULL,                  -- skill | workflow_policy | ...
    parent_version INTEGER NOT NULL DEFAULT 0,   -- the version this proposes against
    status       TEXT NOT NULL,
    -- Provenance: every one of these is required to answer "why does this exist".
    source_experiment TEXT NOT NULL DEFAULT '',
    source_runs  TEXT NOT NULL DEFAULT '[]',
    source_failures TEXT NOT NULL DEFAULT '[]',
    evidence     TEXT NOT NULL DEFAULT '{}',
    proposed_change TEXT NOT NULL DEFAULT '{}',
    rationale    TEXT NOT NULL DEFAULT '',
    verdict      TEXT NOT NULL DEFAULT '{}',
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    UNIQUE(candidate_id, version)
);
CREATE INDEX IF NOT EXISTS idx_candidates_status ON candidates(status);

"""


# Columns added after the initial schema. Applied idempotently to databases
# created by an older version: `SCHEMA` already carries them for fresh files,
# and `ALTER TABLE ADD COLUMN` is an O(1) metadata change that preserves rows.
_MIGRATIONS: dict[str, list[tuple[str, str]]] = {
    "tool_calls": [("affected_paths", "TEXT"), ("diff_summary", "TEXT"),
                   ("status", "TEXT"), ("error_code", "TEXT"),
                   ("security_event", "TEXT"), ("source", "TEXT"),
                   ("event_id", "INTEGER")],
    "steps": [("failure_class", "TEXT"),
              ("input_tokens", "INTEGER"), ("output_tokens", "INTEGER"),
              ("cached_tokens", "INTEGER"), ("reasoning_tokens", "INTEGER"),
              ("model_calls", "INTEGER"),
              ("latency_ms", "INTEGER"), ("cost_usd", "REAL"),
              ("model", "TEXT"),
              # Which event describes this row, so a trace reader can join the
              # four independent tables instead of guessing at `created_at`.
              ("event_id", "INTEGER")],
    "runs": [("identity_hash", "TEXT"), ("key_files", "TEXT"),
             ("owner", "TEXT"), ("lease_until", "TEXT"),
             # Defaults are the *old* behaviour on purpose: every row written
             # before these columns existed came from an interactive run and
             # every skill was immediately live. A migration that reclassified
             # existing data would be inventing history.
             ("origin", "TEXT NOT NULL DEFAULT 'interactive'"),
             ("task_id", "TEXT"), ("task_version", "INTEGER"),
             ("session_id", "TEXT")],
    "skills": [("digest", "TEXT NOT NULL DEFAULT ''"),
               ("status", "TEXT NOT NULL DEFAULT 'live'"),
               ("origin", "TEXT NOT NULL DEFAULT 'interactive'"),
               ("parent_id", "INTEGER")],
    # The correlation columns. Four tables had four independent AUTOINCREMENT
    # sequences and no key joining them, so "which event does this step belong
    # to" had no answer. Now each row names the event that describes it.
    "transitions": [("event_id", "INTEGER")],
    "promotions": [("evaluation_ids", "TEXT NOT NULL DEFAULT '[]'")],
}


def _migrate(conn: sqlite3.Connection) -> None:
    for table, columns in _MIGRATIONS.items():
        existing = {row["name"] for row in
                    conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if not existing:            # table missing: SCHEMA just created it fully
            continue
        for name, decl in columns:
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
    conn.commit()


def connect(path: str | Path) -> sqlite3.Connection:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def now() -> str:
    from datetime import datetime
    return datetime.now().isoformat(timespec="seconds")


# The trace's row format version, stamped on every event. A reader that meets an
# event it does not understand must be able to tell that from an event that is
# simply malformed — the same reason `cases.json` carries one.
EVENTS_SCHEMA_VERSION = 1


def utcnow() -> str:
    """A UTC timestamp with microseconds, for the event log.

    Deliberately not `now()`. That one is local, second-resolution and naive,
    which is fine for "when was this row written" and indefensible for a log
    meant to be read back in order: two events in the same second tie, and a tie
    has no defined order that any column can break.

    Even so, **nothing orders by this**. Ordering is `events.event_id`, which
    SQLite assigns atomically; a wall clock can move backwards and is not shared
    between processes. This field is for a human reading the log.
    """
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def after(seconds: int) -> str:
    """A local timestamp `seconds` in the future, in the same format as `now()`.

    Same format means the two compare correctly as plain strings in SQLite, so
    lease expiry needs no date parsing in the query.
    """
    from datetime import datetime, timedelta
    return (datetime.now() + timedelta(seconds=seconds)).isoformat(timespec="seconds")
