"""Repository: typed access to the SQLite database."""
from __future__ import annotations

import json
import sqlite3
import uuid
from typing import Any

from .. import redact
from ..harness import statemachine as sm
from ..models import (
    ORIGIN_INTERACTIVE,
    PRODUCTION_SKILL_ORIGIN,
    PRODUCTION_SKILL_STATUS,
    SKILL_CANDIDATE,
    SKILL_LIVE,
)
from ..skills import content_digest as skill_digest
from .db import EVENTS_SCHEMA_VERSION, after, connect, now, utcnow


def _j(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _promotion_row(row) -> dict:
    """One `promotions` row with its JSON columns parsed."""
    d = dict(row)
    d["source_runs"] = _unj(d.get("source_runs"), [])
    d["evaluation_ids"] = _unj(d.get("evaluation_ids"), [])
    for column in ("evaluation", "compare_result", "gate"):
        d[column] = _unj(d.get(column), {})
    return d


def trigger_of(candidate_id: str) -> str:
    """The failure class a `skill:<trigger>` candidate is about, or ""."""
    return candidate_id.split(":", 1)[1] if candidate_id.startswith("skill:") else ""


def _candidate_row(row) -> dict:
    """One `candidates` row with its JSON columns parsed."""
    d = dict(row)
    for column in ("source_runs", "source_failures"):
        d[column] = _unj(d.get(column), [])
    for column in ("evidence", "proposed_change", "verdict"):
        d[column] = _unj(d.get(column), {})
    return d


def _evaluation_row(row) -> dict:
    """One `evaluations` row with its JSON columns parsed."""
    d = dict(row)
    for column in ("axes", "reasons", "failures", "hard_gate"):
        d[column] = _unj(d.get(column), {} if column == "axes" else [])
    return d


def _event_row(row) -> dict:
    """One `events` row with its JSON payload parsed."""
    d = dict(row)
    d["payload"] = _unj(d.get("payload"), {})
    return d


def _unj(s: str | None, default: Any = None) -> Any:
    if not s:
        return default
    try:
        return json.loads(s)
    except (TypeError, ValueError):
        return default


class Repo:
    def __init__(self, db_path, *, redact: bool = True):
        self._conn = connect(db_path)
        self._redact = redact

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    # --------------------------------------------------------------- redaction
    def _clean_text(self, text: str | None) -> str | None:
        """Redact secret-shaped values from untrusted content on its way in.

        Applied to what a model or a subprocess produced — never to the
        harness's own fingerprints, which are sha256 digests and would be eaten
        by the long-hex pattern (see `redact.py`).
        """
        if not self._redact or not text:
            return text
        return redact.redact(text)

    def _clean_obj(self, obj: Any) -> Any:
        """Redact a structure's string values before it is serialized.

        Redacting the *serialized* text instead can produce invalid JSON: some
        patterns end in `\\S+`, which is greedy enough to eat a closing quote and
        the comma after it, and the row would then read back as its default.
        """
        if not self._redact or obj is None:
            return obj
        return redact.redact_structure(obj)

    # ------------------------------------------------------------------ runs
    def create_run(self, kind: str, title: str, description: str, *,
                   parent_run_id: str | None = None,
                   origin: str = ORIGIN_INTERACTIVE,
                   task_id: str | None = None,
                   task_version: int | None = None,
                   session_id: str | None = None) -> dict:
        run_id = uuid.uuid4().hex[:16]
        ts = now()
        root = run_id
        if parent_run_id:
            parent = self.get_run(parent_run_id)
            root = parent["root_run_id"]
            # A child inherits where its parent came from. A delegated sub-task of
            # a benchmark run is a benchmark run — inferring it fresh would let a
            # benchmark spawn children that seed production. It inherits the
            # *session* for the same reason: "what did this invocation do" has to
            # include the runs it spawned.
            origin = parent.get("origin") or origin
            session_id = session_id or parent.get("session_id")
        # Read off the machine declaration rather than repeated here. The literal
        # that used to sit on this line was a second copy of
        # `MACHINES[kind]["initial"]`, which is how a third machine would have
        # silently started every run in `req_capture`. `statemachine` is a leaf
        # (it imports nothing), so this direction is safe; if the machine table
        # ever moves, this follows it.
        initial_state = sm.initial_state(kind)
        self._conn.execute(
            "INSERT INTO runs (id,kind,title,description,state,status,parent_run_id,"
            "root_run_id,payload,origin,task_id,task_version,session_id,"
            "created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, kind, title, description, initial_state, "created",
             parent_run_id, root, "{}", origin, task_id, task_version,
             session_id or uuid.uuid4().hex[:16], ts, ts),
        )
        self._conn.commit()
        return self.get_run(run_id)

    @staticmethod
    def _run_row(row) -> dict:
        """One `runs` row with its JSON columns parsed.

        `get_run` used to parse these while `list_runs` and `child_runs` returned
        the raw text, so `run["payload"]` was a dict or a string depending on
        which method produced the run — and a caller written against one crashed
        against the other. Parsing lives in exactly one place now.
        """
        d = dict(row)
        for column in ("payload", "key_files"):
            if column in d:
                d[column] = _unj(d.get(column), {})
        return d

    def get_run(self, run_id: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return None if row is None else self._run_row(row)

    def set_fingerprint(self, run_id: str, identity_hash: str,
                        key_files: dict | None = None) -> None:
        """Record the conditions the run's persisted steps were produced under."""
        self._conn.execute(
            "UPDATE runs SET identity_hash=?, key_files=?, updated_at=? WHERE id=?",
            (identity_hash, _j(key_files or {}), now(), run_id))
        self._conn.commit()

    def list_runs(self, limit: int = 50, *, kind: str | None = None, status: str | None = None) -> list[dict]:
        sql = "SELECT * FROM runs"
        where, args = [], []
        if kind:
            where.append("kind=?")
            args.append(kind)
        if status:
            where.append("status=?")
            args.append(status)
        if where:
            sql += " WHERE " + " AND ".join(where)
        # `created_at` is second-resolution, so two runs created in the same
        # second tie — and with a tie the order is whatever SQLite happens to
        # return. `rowid` breaks it by insertion order, which is creation order,
        # so "newest first" is actually a promise rather than a tendency.
        sql += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        args.append(limit)
        return [self._run_row(r) for r in self._conn.execute(sql, args).fetchall()]

    def update_run(self, run_id: str, *, state: str | None = None, status: str | None = None,
                   payload: dict | None = None, attempt: int | None = None,
                   result: Any = None, error: str | None = None) -> None:
        sets, args = [], []
        if state is not None:
            sets.append("state=?")
            args.append(state)
        if status is not None:
            sets.append("status=?")
            args.append(status)
        if payload is not None:
            merged = self.get_run(run_id)["payload"]
            merged.update(payload)
            sets.append("payload=?")
            args.append(_j(merged))
        if attempt is not None:
            sets.append("attempt=?")
            args.append(attempt)
        if result is not None:
            sets.append("result=?")
            args.append(_j(self._clean_obj(result)))
        if error is not None:
            sets.append("error=?")
            args.append(self._clean_text(error))
        sets.append("updated_at=?")
        args.append(now())
        args.append(run_id)
        self._conn.execute(f"UPDATE runs SET {', '.join(sets)} WHERE id=?", args)
        self._conn.commit()

    def set_status(self, run_id: str, status: str) -> None:
        self.update_run(run_id, status=status)

    # ------------------------------------------------------------ trace (events)
    def add_event(self, *, type: str, run_id: str = "", session_id: str = "",
                  actor: str = "harness", payload: dict | None = None) -> int:
        """Append one event; return its `event_id`.

        Append-only: nothing here updates or deletes, and there is no method that
        does. A change that would have rewritten history is written as an event
        instead — `step.invalidated` exists for exactly that.

        `seq` is allocated in the same statement from the run's own events. That
        is race-free *while the execution lease is held*, which is where every
        harness-emitted event happens; outside it, two writers could read the same
        maximum. `event_id`, assigned by SQLite, is the ordering authority either
        way, so the worst case is a less tidy `seq` rather than a wrong order.
        """
        seq = 0
        if run_id:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM events WHERE run_id=?",
                (run_id,)).fetchone()
            seq = int(row[0])
        cur = self._conn.execute(
            "INSERT INTO events (seq,at,run_id,session_id,actor,type,payload,"
            "schema_version) VALUES (?,?,?,?,?,?,?,?)",
            (seq, utcnow(), run_id, session_id, actor, type,
             _j(self._clean_obj(payload or {})), EVENTS_SCHEMA_VERSION))
        self._conn.commit()
        return int(cur.lastrowid)

    def events_for_run(self, run_id: str) -> list[dict]:
        """This run's events, oldest first.

        Ordered by `event_id`, never by `seq` and never by `at`: `event_id` is
        assigned atomically so it cannot tie, `at` is a wall clock that can move
        backwards, and `seq` is only race-free inside the lease.
        """
        return [_event_row(r) for r in self._conn.execute(
            "SELECT * FROM events WHERE run_id=? ORDER BY event_id", (run_id,))]

    def skill_version(self, trigger: str, version: int, *,
                      origin: str = ORIGIN_INTERACTIVE,
                      status: str = SKILL_LIVE) -> dict | None:
        """One exact version of a skill, whatever the current pointer says."""
        row = self._conn.execute(
            "SELECT * FROM skills WHERE trigger=? AND version=? AND origin=? AND status=?",
            (trigger, version, origin, status)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["files"] = _unj(d.get("files"), [])
        return d

    def skill_versions(self, trigger: str, *,
                       origin: str = ORIGIN_INTERACTIVE,
                       status: str = SKILL_LIVE) -> list[dict]:
        """Every version of one skill, oldest first — the history, not just now."""
        out = []
        for row in self._conn.execute(
                "SELECT * FROM skills WHERE trigger=? AND origin=? AND status=? "
                "ORDER BY version", (trigger, origin, status)).fetchall():
            d = dict(row)
            d["files"] = _unj(d.get("files"), [])
            out.append(d)
        return out

    def set_current_skill(self, trigger: str, version: int, *,
                          origin: str = ORIGIN_INTERACTIVE,
                          status: str = SKILL_LIVE) -> None:
        """Move the current pointer to a version that already exists.

        `superseded` is a *pointer*, not content — which is why moving it is not
        editing history. The text of every version is untouched, nothing is
        deleted, and every move is written to `promotions`/`rollbacks` so the
        sequence of pointers stays reconstructible.
        """
        self._conn.execute(
            "UPDATE skills SET superseded=1 WHERE trigger=? AND origin=? AND status=?",
            (trigger, origin, status))
        self._conn.execute(
            "UPDATE skills SET superseded=0 WHERE trigger=? AND version=? AND origin=? "
            "AND status=?", (trigger, version, origin, status))
        self._conn.commit()

    # ------------------------------------------------------------- promotions
    def add_promotion(self, record: dict) -> None:
        """Append a promotion. There is no update path and no delete path."""
        self._conn.execute(
            "INSERT INTO promotions (promotion_id,candidate_id,candidate_version,"
            "parent_skill_version,parent_digest,promoted_skill_version,promoted_digest,"
            "source_experiment,source_runs,evaluation,evaluation_ids,compare_result,gate,"
            "actor,reason,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (record["promotion_id"], record["candidate_id"], record["candidate_version"],
             record["parent_skill_version"], record["parent_digest"],
             record["promoted_skill_version"], record["promoted_digest"],
             record.get("source_experiment", ""), _j(record.get("source_runs") or []),
             _j(record.get("evaluation") or {}),
             _j(record.get("evaluation_ids") or []),
             _j(record.get("compare_result") or {}),
             _j(record.get("gate") or {}), record.get("actor", "cli"),
             record.get("reason", ""), now()))
        self._conn.commit()

    def get_promotion(self, promotion_id: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM promotions WHERE promotion_id=?",
                                 (promotion_id,)).fetchone()
        return None if row is None else _promotion_row(row)

    def promotions_for(self, candidate_id: str) -> list[dict]:
        return [_promotion_row(r) for r in self._conn.execute(
            "SELECT * FROM promotions WHERE candidate_id=? ORDER BY created_at, rowid",
            (candidate_id,)).fetchall()]

    def promotions_for_skill(self, trigger: str) -> list[dict]:
        """Promotions that named this skill.

        The trigger comes out of the candidate id (`skill:<trigger>`), which is how
        that id is built. A second copy of the trigger on the promotion row would
        be a second place for the two to disagree.
        """
        return [r for r in self.all_promotions() if trigger_of(r["candidate_id"]) == trigger]

    def all_promotions(self) -> list[dict]:
        return [_promotion_row(r) for r in self._conn.execute(
            "SELECT * FROM promotions ORDER BY created_at, rowid").fetchall()]

    # -------------------------------------------------------------- rollbacks
    def add_rollback(self, record: dict) -> None:
        self._conn.execute(
            "INSERT INTO rollbacks (rollback_id,trigger,from_version,to_version,"
            "from_digest,to_digest,actor,reason,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (record["rollback_id"], record["trigger"], record["from_version"],
             record["to_version"], record["from_digest"], record["to_digest"],
             record.get("actor", "cli"), record.get("reason", ""), now()))
        self._conn.commit()

    def rollbacks_for(self, trigger: str) -> list[dict]:
        return [dict(r) for r in self._conn.execute(
            "SELECT * FROM rollbacks WHERE trigger=? ORDER BY created_at, rowid",
            (trigger,)).fetchall()]

    def get_rollback(self, rollback_id: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM rollbacks WHERE rollback_id=?",
                                 (rollback_id,)).fetchone()
        return None if row is None else dict(row)

    # -------------------------------------------------------------- candidates
    def add_candidate(self, candidate) -> int:
        """Persist a candidate. A new version is a new row; an old one is final.

        `UNIQUE(candidate_id, version)` does the enforcing: re-inserting a version
        that exists raises rather than replacing it. That is the whole of "a
        candidate cannot be rewritten" — not a convention, a constraint.
        """
        try:
            cur = self._conn.execute(
                "INSERT INTO candidates (candidate_id,version,type,parent_version,"
                "status,source_experiment,source_runs,source_failures,evidence,"
                "proposed_change,rationale,verdict,created_at,updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (candidate.candidate_id, candidate.version, candidate.type,
                 candidate.parent_version, candidate.status,
                 candidate.source_experiment, _j(list(candidate.source_runs)),
                 _j(list(candidate.source_failures)), _j(candidate.evidence),
                 _j(candidate.proposed_change), candidate.rationale,
                 _j(candidate.verdict), now(), now()))
        except sqlite3.IntegrityError as e:
            raise ValueError(
                f"候选 {candidate.candidate_id} v{candidate.version} 已存在；"
                f"候选创建后不可覆盖 —— 新版本应当是新的行") from e
        self._conn.commit()
        return int(cur.lastrowid)

    def set_candidate_status(self, candidate_id: str, version: int, status: str,
                             *, verdict: dict | None = None) -> None:
        """Move a candidate's status. The caller decides whether the move is legal."""
        self._conn.execute(
            "UPDATE candidates SET status=?, verdict=?, updated_at=? "
            "WHERE candidate_id=? AND version=?",
            (status, _j(verdict) if verdict is not None else _j({}), now(),
             candidate_id, version))
        self._conn.commit()

    def get_candidate(self, candidate_id: str, version: int | None = None) -> dict | None:
        """One candidate, the newest version by default."""
        if version is None:
            row = self._conn.execute(
                "SELECT * FROM candidates WHERE candidate_id=? ORDER BY version DESC "
                "LIMIT 1", (candidate_id,)).fetchone()
        else:
            row = self._conn.execute(
                "SELECT * FROM candidates WHERE candidate_id=? AND version=?",
                (candidate_id, version)).fetchone()
        return None if row is None else _candidate_row(row)

    def candidates_for(self, candidate_id: str) -> list[dict]:
        """Every version of one candidate, oldest first — the history, not just now."""
        return [_candidate_row(r) for r in self._conn.execute(
            "SELECT * FROM candidates WHERE candidate_id=? ORDER BY version",
            (candidate_id,))]

    def list_candidates(self, limit: int = 100) -> list[dict]:
        return [_candidate_row(r) for r in self._conn.execute(
            "SELECT * FROM candidates ORDER BY id DESC LIMIT ?", (limit,))]

    # ------------------------------------------------------------- evaluations
    def add_evaluation(self, evaluation, *, event_id: int | None = None) -> int:
        """Persist one verdict. Append-only, like the trace it is read beside.

        Re-judging a run inserts a second row rather than replacing the first: a
        verdict reached under an earlier task revision is not wrong, it is a
        different judgement, and a promotion decision that cited it has to stay
        explicable.
        """
        cur = self._conn.execute(
            "INSERT INTO evaluations (run_id,task_id,task_version,verdict,axes,"
            "reasons,failures,hard_gate,event_id,created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (evaluation.run_id, evaluation.task_id, evaluation.task_version,
             evaluation.verdict, _j(evaluation.axes),
             _j(list(evaluation.reasons)), _j(list(evaluation.failures)),
             _j(list(evaluation.hard_gate)), event_id, now()))
        self._conn.commit()
        return int(cur.lastrowid)

    def evaluations_for_run(self, run_id: str) -> list[dict]:
        """Every verdict recorded for this run, oldest first."""
        return [_evaluation_row(r) for r in self._conn.execute(
            "SELECT * FROM evaluations WHERE run_id=? ORDER BY id", (run_id,))]

    def latest_evaluation(self, run_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM evaluations WHERE run_id=? ORDER BY id DESC LIMIT 1",
            (run_id,)).fetchone()
        return None if row is None else _evaluation_row(row)

    def events_for_session(self, session_id: str) -> list[dict]:
        """Every event of one invocation, across all the runs it started."""
        return [_event_row(r) for r in self._conn.execute(
            "SELECT * FROM events WHERE session_id=? ORDER BY event_id", (session_id,))]

    # ------------------------------------------------------- execution lease
    def claim_run(self, run_id: str, owner: str, ttl_seconds: int) -> bool:
        """Atomically take the execution lease, if nobody else holds a live one.

        One conditional UPDATE, so two processes cannot both win: SQLite
        serializes the writes and only the one that matched the "free or expired"
        predicate gets `rowcount == 1`. An expired lease is claimable, so a
        killed process cannot wedge a run forever.
        """
        ts = now()
        cur = self._conn.execute(
            "UPDATE runs SET owner=?, lease_until=?, updated_at=? WHERE id=? "
            "AND (owner IS NULL OR lease_until IS NULL OR lease_until < ?)",
            (owner, after(ttl_seconds), ts, run_id, ts))
        self._conn.commit()
        return cur.rowcount == 1

    def renew_lease(self, run_id: str, owner: str, ttl_seconds: int) -> bool:
        """Extend a lease we already hold. False if it has moved on."""
        cur = self._conn.execute(
            "UPDATE runs SET lease_until=?, updated_at=? WHERE id=? AND owner=?",
            (after(ttl_seconds), now(), run_id, owner))
        self._conn.commit()
        return cur.rowcount == 1

    def release_run(self, run_id: str, owner: str) -> None:
        """Drop the lease. Guarded by owner, so a process that lost its lease
        (expired and re-claimed) cannot release the new holder's."""
        self._conn.execute(
            "UPDATE runs SET owner=NULL, lease_until=NULL, updated_at=? "
            "WHERE id=? AND owner=?", (now(), run_id, owner))
        self._conn.commit()

    def child_runs(self, run_id: str) -> list[dict]:
        return [self._run_row(r) for r in self._conn.execute(
            "SELECT * FROM runs WHERE parent_run_id=? ORDER BY created_at", (run_id,)).fetchall()]

    def root_run(self, run_id: str) -> dict | None:
        r = self.get_run(run_id)
        return self.get_run(r["root_run_id"]) if r else None

    # ----------------------------------------------------------------- steps
    def add_step(self, run_id: str, state: str, agent: str | None, output: dict | None,
                 *, status: str = "done", error: str | None = None,
                 input_json: dict | None = None,
                 failure_class: str | None = None,
                 event_id: int | None = None,
                 metrics: dict | None = None) -> int:
        """Persist a completed step.

        `failure_class` records *why* the state did not succeed (None = it did),
        so retries can be counted per kind of failure rather than per state.

        `metrics` carries what the step cost — `input_tokens` / `output_tokens` /
        `cached_tokens` / `model_calls` / `latency_ms` / `cost_usd`. A key that
        is absent is stored as NULL, never 0: an unreported token count and a
        count of zero are different facts, and only one of them is a measurement.
        """
        m = metrics or {}
        cur = self._conn.execute(
            "INSERT INTO steps (run_id,state,agent,status,input_json,output_json,error,"
            "failure_class,input_tokens,output_tokens,cached_tokens,reasoning_tokens,"
            "model_calls,latency_ms,cost_usd,model,event_id,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, state, agent, status,
             _j(self._clean_obj(input_json)) if input_json is not None else None,
             _j(self._clean_obj(output)) if output is not None else None,
             self._clean_text(error),
             failure_class or None,
             m.get("input_tokens"), m.get("output_tokens"), m.get("cached_tokens"),
             m.get("reasoning_tokens"),
             m.get("model_calls"), m.get("latency_ms"), m.get("cost_usd"),
             m.get("model") or None,
             event_id,
             now()),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def metric_rows(self, run_id: str | None = None) -> list[dict]:
        """Per-step cost columns, for aggregation.

        NULL counts are returned as None rather than coerced to 0, so the
        aggregator can report how many steps actually carry a measurement
        instead of presenting a partial sum as a complete one.
        """
        sql = ("SELECT run_id,state,agent,status,input_tokens,output_tokens,"
               "cached_tokens,reasoning_tokens,model_calls,latency_ms,cost_usd,model "
               "FROM steps")
        params: list[Any] = []
        if run_id is not None:
            sql += " WHERE run_id=?"
            params.append(run_id)
        sql += " ORDER BY id"
        return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def get_step(self, run_id: str, state: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM steps WHERE run_id=? AND state=? ORDER BY id DESC LIMIT 1",
            (run_id, state)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["output_json"] = _unj(d.get("output_json"), {})
        d["input_json"] = _unj(d.get("input_json"), {})
        return d

    def steps_for_run(self, run_id: str) -> list[dict]:
        """Every step of a run, oldest first, with the JSON columns parsed.

        `SELECT *` hands back text for the JSON columns, so a caller reading them
        straight from SQL gets strings where it expects structures — the kind of
        mismatch that surfaces as an AttributeError three frames away from the
        query. Parsing belongs here, next to the schema.
        """
        rows = self._conn.execute(
            "SELECT * FROM steps WHERE run_id=? ORDER BY id", (run_id,)).fetchall()
        out: list[dict] = []
        for row in rows:
            d = dict(row)
            d["output_json"] = _unj(d.get("output_json"), {})
            d["input_json"] = _unj(d.get("input_json"), {})
            out.append(d)
        return out

    def step_done(self, run_id: str, state: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM steps WHERE run_id=? AND state=? AND status='done' LIMIT 1",
            (run_id, state)).fetchone()
        return row is not None

    def delete_steps_for_state(self, run_id: str, state: str) -> None:
        self._conn.execute("DELETE FROM steps WHERE run_id=? AND state=?", (run_id, state))
        self._conn.commit()

    # ------------------------------------------------------------ transitions
    def add_transition(self, run_id: str, from_state: str, to_state: str,
                       decision: str, reason: str = "", suggested_by: str = "",
                       event_id: int | None = None) -> None:
        self._conn.execute(
            "INSERT INTO transitions (run_id,from_state,to_state,decision,reason,"
            "suggested_by,event_id,created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (run_id, from_state, to_state, decision, reason, suggested_by,
             event_id, now()))
        self._conn.commit()

    def transitions(self, run_id: str) -> list[dict]:
        return [dict(r) for r in self._conn.execute(
            "SELECT * FROM transitions WHERE run_id=? ORDER BY id", (run_id,)).fetchall()]

    # --------------------------------------------------------------- approvals
    def create_approval(self, run_id: str, action: str, *, scope: str = "", risk_level: str = "medium",
                        reason: str = "", required_by: list[str] | None = None) -> dict:
        a_id = uuid.uuid4().hex[:16]
        self._conn.execute(
            "INSERT INTO approvals (id,run_id,action,scope,risk_level,reason,status,created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (a_id, run_id, action, scope, risk_level,
             json.dumps({"reason": reason, "required_by": required_by or []}, ensure_ascii=False),
             "pending", now()))
        self._conn.commit()
        return self.get_approval(a_id)

    def get_approval(self, approval_id: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM approvals WHERE id=?", (approval_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["reason"] = _unj(d.get("reason"), {})
        return d

    def list_pending_approvals(self) -> list[dict]:
        return [dict(r) for r in self._conn.execute(
            "SELECT * FROM approvals WHERE status='pending' ORDER BY created_at").fetchall()]

    def pending_approvals_for_run(self, run_id: str) -> list[dict]:
        return [dict(r) for r in self._conn.execute(
            "SELECT * FROM approvals WHERE run_id=? AND status='pending' ORDER BY created_at",
            (run_id,)).fetchall()]

    def list_approvals(self, run_id: str) -> list[dict]:
        return [dict(r) for r in self._conn.execute(
            "SELECT * FROM approvals WHERE run_id=? ORDER BY created_at",
            (run_id,)).fetchall()]

    def list_approved(self, run_id: str) -> list[dict]:
        return [a for a in self.list_approvals(run_id) if a["status"] == "approved"]

    def has_approved_approval(self, run_id: str, action: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM approvals WHERE run_id=? AND status='approved' AND action=? LIMIT 1",
            (run_id, action)).fetchone()
        return row is not None

    def decide_approval(self, approval_id: str, decision: str, *, by: str, note: str = "") -> dict | None:
        a = self.get_approval(approval_id)
        if a is None or a["status"] != "pending":
            return a
        self._conn.execute(
            "UPDATE approvals SET status=?, decided_by=?, decision_note=?, decided_at=? WHERE id=?",
            (decision, by, note, now(), approval_id))
        self._conn.commit()
        return self.get_approval(approval_id)

    # --------------------------------------------------------------- evidence
    def add_evidence(self, run_id: str, kind: str, source: str, content: str, confidence: str = "medium") -> int:
        cur = self._conn.execute(
            "INSERT INTO evidence (run_id,kind,source,content,confidence,created_at) VALUES (?,?,?,?,?,?)",
            (run_id, kind, self._clean_text(source), self._clean_text(content),
             confidence, now()))
        self._conn.commit()
        return int(cur.lastrowid)

    def list_evidence(self, run_id: str) -> list[dict]:
        return [dict(r) for r in self._conn.execute(
            "SELECT * FROM evidence WHERE run_id=? ORDER BY id", (run_id,)).fetchall()]

    # -------------------------------------------------------------- tool_calls
    def log_tool_call(self, run_id: str, agent: str, tool: str, args: dict | None, ok: bool,
                      error: str | None = None, *,
                      affected_paths: list[str] | None = None,
                      diff_summary: str | None = None,
                      status: str | None = None,
                      error_code: str | None = None,
                      security_event: str | None = None,
                      source: str | None = None,
                      event_id: int | None = None) -> None:
        """Audit one tool call.

        `status`/`error_code`/`security_event` are the stable outcome
        vocabulary; `status` defaults to ok|error from `ok` so callers that do
        not classify still produce countable rows. `affected_paths` /
        `diff_summary` carry the snapshot-observed change set for tools that
        write files. `source` records the origin ("agent" / "mcp") while `agent`
        stays a plain role name.
        """
        self._conn.execute(
            "INSERT INTO tool_calls (run_id,agent,tool,args_json,ok,error,"
            "status,error_code,security_event,source,affected_paths,diff_summary,"
            "event_id,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, agent, tool,
             _j(self._clean_obj(args)) if args is not None else None, int(ok),
             self._clean_text(error), status or ("ok" if ok else "error"),
             # Empty string and NULL would be two spellings of "no value"; keep
             # NULL as the only one so the column means exactly one thing.
             error_code or None, security_event or None, source or None,
             # Paths are deliberately not redacted: they are the harness's own
             # attribution and redacting one would corrupt the change set.
             _j(affected_paths) if affected_paths is not None else None,
             diff_summary, event_id, now()))
        self._conn.commit()

    def tool_calls(self, run_id: str) -> list[dict]:
        return [dict(r) for r in self._conn.execute(
            "SELECT * FROM tool_calls WHERE run_id=? ORDER BY id", (run_id,)).fetchall()]

    def affected_paths_for_run(self, run_id: str, *, agent: str | None = None) -> list[str]:
        """Union of every path a workspace snapshot observed changing in this
        run — ground truth, independent of what any model claimed.

        This is the run-level footprint: it spans retries of the same state, so
        a path touched by an earlier attempt is still reported. Pass `agent` to
        narrow it to one role (e.g. only the implementer's writes, ignoring the
        artifacts a verifier's test run may leave behind).
        """
        sql = ("SELECT affected_paths FROM tool_calls "
               "WHERE run_id=? AND affected_paths IS NOT NULL")
        params: list[Any] = [run_id]
        if agent is not None:
            sql += " AND agent=?"
            params.append(agent)
        seen: set[str] = set()
        for row in self._conn.execute(sql, params).fetchall():
            paths = _unj(row["affected_paths"], []) or []
            seen.update(p for p in paths if isinstance(p, str))
        return sorted(seen)

    def plan_violations_for_run(self, run_id: str) -> list[dict]:
        """Writes the plan scope refused, as `{tool, path}`.

        These are requests that never reached disk. The Harness turns them into
        approvals, so an out-of-plan change needs a human instead of a retry.

        Paths that already have a pending or approved approval for this run are
        excluded: once escalated, the decision belongs to the approval gate, and
        counting them again would re-block the approved retry forever.
        """
        escalated = {a["action"] for a in self.list_approvals(run_id)
                     if a["status"] in ("pending", "approved")}
        out: list[dict] = []
        for row in self._conn.execute(
                "SELECT tool, args_json FROM tool_calls "
                "WHERE run_id=? AND error_code=? ORDER BY id",
                (run_id, "plan_scope_denied")).fetchall():
            args = _unj(row["args_json"], {}) or {}
            tool, path = row["tool"], args.get("path", "")
            if path and f"{tool}:{path}" in escalated:
                continue
            out.append({"tool": tool, "path": path})
        return out

    def ancestor_depth(self, run_id: str) -> int:
        """Number of parent links above this run (0 for a root run).

        Child bugfix runs are themselves bugfix runs, so they can spawn their
        own children; this is what bounds that recursion.
        """
        depth = 0
        seen: set[str] = set()
        cur = self.get_run(run_id)
        while cur is not None and cur.get("parent_run_id"):
            if cur["id"] in seen:            # defensive: never loop on corrupt data
                break
            seen.add(cur["id"])
            depth += 1
            cur = self.get_run(cur["parent_run_id"])
        return depth

    # ----------------------------------------------------------------- skills
    def add_skill(self, trigger: str, title: str, procedure: str, evidence_run: str,
                  *, files: list[str] | None = None,
                  origin: str = ORIGIN_INTERACTIVE,
                  status: str | None = None,
                  parent_id: int | None = None) -> dict:
        """Record a procedure, superseding the current one in the same scope.

        Superseding rather than appending: two live procedures for one failure
        class would have to be ranked by something, and the newest evidence is the
        only ranking that means anything. The old row stays, marked — the history
        of what the harness believed is part of the record.

        **Scoped by `origin`.** Global supersession had a hazard in the other
        direction from the one this column exists for: a benchmark run recording
        a procedure for `test_failure` would mark the *production* procedure
        superseded, and it would silently stop reaching a human's prompts. Running
        a task must not be able to switch off what production relies on.

        Status is derived rather than demanded, so a caller cannot get it wrong: a
        procedure learned from an interactive run goes live (a human drove that
        run and the text is the harness's own record of it), anything else is a
        candidate. Promotion passes an explicit status.
        """
        if status is None:
            status = SKILL_LIVE if origin == ORIGIN_INTERACTIVE else SKILL_CANDIDATE
        latest = self.latest_skill(trigger, origin=origin, status=status)
        version = (latest["version"] + 1) if latest else 1
        self._conn.execute(
            "UPDATE skills SET superseded=1 WHERE trigger=? AND origin=? AND status=? "
            "AND superseded=0", (trigger, origin, status))
        cleaned = self._clean_text(procedure)
        digest = skill_digest({
            "trigger": trigger, "title": title, "procedure": cleaned,
            "files": files or [], "version": version})
        cur = self._conn.execute(
            "INSERT INTO skills (trigger,title,procedure,evidence_run,version,"
            "superseded,files,digest,status,origin,parent_id,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (trigger, title, cleaned, evidence_run, version, 0,
             _j(files or []), digest, status, origin, parent_id, now()))
        self._conn.commit()
        return self.get_skill(int(cur.lastrowid))

    def get_skill(self, skill_id: int) -> dict | None:
        row = self._conn.execute("SELECT * FROM skills WHERE id=?", (skill_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["files"] = _unj(d.get("files"), [])
        return d

    def latest_skill(self, trigger: str, *, origin: str | None = None,
                     status: str | None = None) -> dict | None:
        """The current procedure for a failure class, or None.

        Unscoped by default (the newest current row for the trigger, whoever wrote
        it) — callers that mean a specific scope pass one.
        """
        sql = "SELECT * FROM skills WHERE trigger=? AND superseded=0"
        args: list[Any] = [trigger]
        if origin is not None:
            sql += " AND origin=?"
            args.append(origin)
        if status is not None:
            sql += " AND status=?"
            args.append(status)
        row = self._conn.execute(sql + " ORDER BY id DESC LIMIT 1", args).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["files"] = _unj(d.get("files"), [])
        return d

    def skills_for(self, triggers: list[str], *,
                   origin: str = ORIGIN_INTERACTIVE) -> list[dict]:
        """Procedures any of these failure classes may use, oldest first.

        An interactive run sees production procedures and nothing else: a
        `candidate` was produced by an environment nobody vouched for, and
        injecting it would be an artificial run writing into a real prompt —
        which is the whole reason `origin` exists.

        A task/benchmark run additionally sees candidates **from its own origin**,
        so a suite can measure whether a proposed procedure helps. It sees
        production procedures too, because exercising those is the point.
        """
        if not triggers:
            return []
        marks = ",".join("?" for _ in triggers)
        if origin == ORIGIN_INTERACTIVE:
            where = "superseded=0 AND status=? AND origin=? AND trigger IN (" + marks + ")"
            args: list[Any] = [SKILL_LIVE, ORIGIN_INTERACTIVE, *triggers]
        else:
            where = ("superseded=0 AND trigger IN (" + marks + ") AND "
                     "(status=? OR (status=? AND origin=?))")
            args = [*triggers, SKILL_LIVE, SKILL_CANDIDATE, origin]
        rows = self._conn.execute(
            f"SELECT * FROM skills WHERE {where} ORDER BY id", args).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            d["files"] = _unj(d.get("files"), [])
            out.append(d)
        return out

    def list_skills(self, *, live_only: bool = True, limit: int = 100,
                    status: str | None = PRODUCTION_SKILL_STATUS,
                    origin: str | None = PRODUCTION_SKILL_ORIGIN) -> list[dict]:
        """Procedures, **production by default**.

        The default answer to "what skills does this harness have" is the set a
        prompt may load and a default benchmark arm may run — live, from an
        interactive run. `list_skills()` was once "everything not superseded",
        which is a different question with a dangerous answer: the moment a
        `candidate`-status row existed in this table, an isolated run would have
        built its arm out of it and reported a procedure as production that no
        promotion had ever approved.

        A caller that genuinely wants a wider set says so, and the two that do are
        named where they are: the derived-skills record and the `wfos skills`
        viewer, neither of which loads anything.

        Pass `status=None, origin=None` for every row regardless of tier;
        `live_only=False` additionally includes superseded history.
        """
        clauses, args = [], []
        if live_only:
            clauses.append("superseded=0")
        if status is not None:
            clauses.append("status=?")
            args.append(status)
        if origin is not None:
            clauses.append("origin=?")
            args.append(origin)
        sql = "SELECT * FROM skills"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        out = []
        for row in self._conn.execute(sql, tuple(args)).fetchall():
            d = dict(row)
            d["files"] = _unj(d.get("files"), [])
            out.append(d)
        return out

    # ------------------------------------------------------------------- wiki
    def add_wiki(self, kind: str, title: str, content: str, *, source: str = "", evidence: list | None = None,
                 scope: str = "", verified: bool = False, trust: str = "medium", run_id: str = "",
                 status: str = "pending", tags: list | None = None, checksum: str = "") -> int:
        cur = self._conn.execute(
            "INSERT INTO wiki (kind,title,content,source,evidence,scope,verified,trust,run_id,status,tags,checksum,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (kind, title, self._clean_text(content), self._clean_text(source),
             _j(evidence or []), scope, int(verified), trust,
             run_id, status, _j(tags or []), checksum, now()))
        self._conn.commit()
        return int(cur.lastrowid)

    def update_wiki(self, wid: int, *, kind: str | None = None, status: str | None = None,
                    verified: bool | None = None, trust: str | None = None) -> None:
        sets, args = [], []
        if kind is not None:
            sets.append("kind=?")
            args.append(kind)
        if status is not None:
            sets.append("status=?")
            args.append(status)
        if verified is not None:
            sets.append("verified=?")
            args.append(int(verified))
        if trust is not None:
            sets.append("trust=?")
            args.append(trust)
        if not sets:
            return
        args.append(wid)
        self._conn.execute(f"UPDATE wiki SET {', '.join(sets)} WHERE id=?", args)
        self._conn.commit()

    def get_wiki(self, wid: int) -> dict | None:
        row = self._conn.execute("SELECT * FROM wiki WHERE id=?", (wid,)).fetchone()
        return dict(row) if row else None

    def list_wiki(self, kind: str | None = None, status: str | None = None, limit: int = 100) -> list[dict]:
        sql = "SELECT * FROM wiki"
        where, args = [], []
        if kind:
            where.append("kind=?")
            args.append(kind)
        if status:
            where.append("status=?")
            args.append(status)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        rows = [dict(r) for r in self._conn.execute(sql, args).fetchall()]
        for r in rows:
            r["evidence"] = _unj(r.get("evidence"), [])
            r["tags"] = _unj(r.get("tags"), [])
        return rows

    def search_wiki(self, query: str, *, kind: str | None = None, tags: str | None = None,
                    trust: str | None = None, limit: int = 10) -> list[dict]:
        """Metadata filter + full-text substring search.

        Full-text is LIKE-based so both CJK and Latin substrings match reliably.
        Metadata (kind / tags / trust) is applied as a hard filter; results are
        ranked by term-hit count, then recency.

        There is no vector retrieval, and this docstring used to say there was
        one "when wired in" — describing an enhancement that had no code behind
        it, in the one place a reader would take as authoritative. Ranking is
        keyword overlap, and a reader can predict it, which is worth more here
        than a similarity score nobody can explain.

        An FTS index used to sit next to this (`_sync_fts`, writing `wiki_fts`).
        It has been deleted, for two independent reasons. It was never called and
        its table was never created, so it did nothing but imply to a reader that
        one existed. And FTS5 would make *these* queries worse, not better: with
        `unicode61` a Chinese query matches nothing at all, and with `trigram` a
        two-character term like 登录 matches nothing while `.` in `app.py` is a
        syntax error to be escaped. LIKE handles both. Re-adding it would need
        that measurement revisited, not just a table definition.
        """
        terms = _split(query) or []
        sql = "SELECT * FROM wiki"
        where, args = [], []
        if terms:
            conds = []
            for t in terms:
                pat = f"%{_like_escape(t)}%"
                conds.append(
                    "(title LIKE ? ESCAPE '\\' OR content LIKE ? ESCAPE '\\' OR tags LIKE ? ESCAPE '\\')")
                args += [pat, pat, pat]
            where.append("(" + " OR ".join(conds) + ")")
        if kind:
            where.append("kind=?")
            args.append(kind)
        if tags:
            # Every named tag must be present (AND), matching how `kind` and
            # `trust` narrow — a filter that returned *more* rows for more
            # conditions would be surprising.
            #
            # This used to take `next(iter(set(_split(tags))))`: one arbitrary
            # tag out of however many were asked for, and which one changed
            # between processes, because set iteration order over strings follows
            # hash randomisation. The same query returned different rows on
            # different runs. Measured, not theorised: "alpha beta" selected
            # `alpha` in some processes and `beta` in others.
            #
            # The pattern is the JSON encoding of the tag (`"auth"`), not the bare
            # text: tags are stored as a JSON array, so a bare `%auth%` also
            # matches an entry tagged `oauth`.
            for tag in sorted(set(_split(tags))):
                where.append("tags LIKE ? ESCAPE '\\'")
                args.append(f"%{_like_escape(_j(tag))}%")
        if trust:
            where.append("trust=?")
            args.append(trust)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(min(limit * 5, 500))
        rows = [dict(r) for r in self._conn.execute(sql, args).fetchall()]

        def score(d: dict) -> tuple:
            hay = f"{d['title']} {d['content']} {' '.join(_unj(d.get('tags'), []))}"
            hits = sum(1 for t in terms if t.lower() in hay.lower())
            return (-hits, d["created_at"])

        rows.sort(key=score)
        rows = rows[:limit]
        for d in rows:
            d["evidence"] = _unj(d.get("evidence"), [])
            d["tags"] = _unj(d.get("tags"), [])
        return rows

    def close(self) -> None:
        self._conn.close()


def _like_escape(t: str) -> str:
    return t.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _split(s: str) -> list[str]:
    """Split a query into terms: whitespace-separated words, and each contiguous
    CJK run is treated as one term so multi-char Chinese queries match as a phrase."""
    import re
    parts = re.findall(r"[一-鿿]+|[^\s,]+", s or "")
    return [p for p in parts if p]
