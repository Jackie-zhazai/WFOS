"""Harness Orchestrator — the only place that decides state transitions.

The model (or mock) only *suggests* the next state. The Harness validates the
suggestion against the state machine, applies structured policy (verdicts,
confidence, risk, approval), persists the completed step BEFORE the transition,
spawns child Bugfix runs, and blocks on approvals. Every transition is audited.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import uuid
from typing import Any

from .. import events
from ..agents.architect import ArchitectAgent
from ..agents.base import RUN_RECORD, AgentError, BaseAgent
from ..agents.chat import ChatAgent
from ..agents.curator import CuratorAgent
from ..agents.implementer import ImplementerAgent
from ..agents.investigator import InvestigatorAgent
from ..agents.verifier import VerifierAgent
from ..config import AppConfig
from ..failures import FAILURE_CLASSES  # noqa: F401  (re-export)
from ..llm.base import LLMAdapter, classify_wire_error
from ..llm.capabilities import CapabilityStore
from ..llm.factory import build_routed_adapters
from ..mcp.client import ToolGateway
from ..memory import capture_end_state, memory_for
from ..metrics import billable_tokens, cost_usd, run_metrics
from ..models import ORIGIN_INTERACTIVE, TERMINAL_STATUSES
from ..relevance import select_evidence
from ..skills import origin_of, record_from_run, skills_for_run
from ..storage.repo import Repo
from ..wiki.wiki import WikiClient
from . import identity
from . import statemachine as sm
from .router import route

# state -> agent role
STATE_AGENT: dict[str, str] = {
    "req_capture": "investigator", "issue_capture": "investigator",
    "project_check": "investigator", "history_search": "investigator",
    "evidence_collect": "investigator",
    "feature_design": "architect", "tech_design": "architect",
    "root_cause": "architect", "fix_plan": "architect",
    "implement": "implementer",
    "build_test": "verifier", "regression_verify": "verifier",
    "verify_regression": "verifier",
    "knowledge_distill": "curator",
}

# role -> agent class. A table rather than five constructor calls, so that
# `[llm.routing]` can be validated against the roles that exist — a route naming
# a role nothing looks up would be a setting that silently does nothing.
AGENT_CLASSES: dict[str, type[BaseAgent]] = {
    "investigator": InvestigatorAgent,
    "architect": ArchitectAgent,
    "implementer": ImplementerAgent,
    "verifier": VerifierAgent,
    "curator": CuratorAgent,
}


# Harness-decision states (no model call).
HARNESS_STATES = {"risk_assess", "conditional_approval", "confidence_assess"}

# operation kinds that always require human approval
APPROVAL_KINDS = {"delete", "db_migrate", "prod_write", "external",
                  "permission", "secret", "irreversible", "core_modify"}

# states that get re-run on re-entry (delete leftover step)
RE_RUNNABLE = {"implement", "build_test", "verify_regression", "regression_verify",
               "root_cause", "evidence_collect", "fix_plan", "knowledge_distill"}


# How many per-run asyncio locks to keep before pruning idle ones.
_RUN_LOCK_CAP = 256

# How far one delegated child is driven before the tool returns. A child that
# needs more than this is a child whose task was too large for the parent to
# hand off, and its state is on the record either way.
_DELEGATE_MAX_LOOPS = 40


class RunLockedError(RuntimeError):
    """Another process holds the run's execution lease."""


class _LeaseHeartbeat:
    """Keeps an execution lease alive while a long state runs.

    Renewing once per loop iteration (once per state) is not enough: a single
    state may make up to `max_rounds` model calls — 8 for the implementer, 6 at
    240s for the verifier — so a state slower than the lease could have its
    claim expire mid-flight and be taken over by another process. Renewing at a
    third of the lease fixes that without lengthening the lease, which matters
    because a long lease also means a long wait before a killed process's run
    can be picked up by anyone else.

    Cooperative by construction: it renews only when the event loop is free. A
    state that blocks the loop for longer than the lease can still lose it — in
    that case `lost` is set and the advance stops at the next state boundary
    instead of continuing on a claim it no longer holds.
    """

    def __init__(self, repo: Repo, run_id: str, owner: str, seconds: int):
        self._repo = repo
        self._run_id = run_id
        self._owner = owner
        self._seconds = seconds
        self._interval = max(1, seconds // 3)
        # Set when a renewal is refused, i.e. somebody else holds the claim now.
        self.lost = asyncio.Event()
        self._task = asyncio.create_task(self._loop())

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            if not self._repo.renew_lease(self._run_id, self._owner, self._seconds):
                self.lost.set()
                return

    async def stop(self) -> None:
        """Cancel and await the heartbeat, so no task outlives the advance."""
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task


def _step_summary(step: dict | None) -> dict:
    """The facts a step row held, kept in the trace when the row is deleted."""
    if not step:
        return {}
    return {k: step.get(k) for k in ("status", "failure_class", "model_calls",
                                     "latency_ms", "cost_usd")
            if step.get(k) is not None}


def _approval_action(op: dict, tool: str) -> str:
    if tool:
        return f"{tool}:{op.get('path', '')}"
    return op.get("kind", "action")


# Why a state did not succeed. Retries are budgeted per (state, class) so that
# the *same* failure recurring is visible instead of being averaged away, and a
# repeated class can escalate to a human rather than burning the retry budget.
# `FAILURE_CLASSES` is defined in `wfos/failures.py` — a leaf, so the evaluator can
# reach the vocabulary without importing the runner's internals. Re-exported here
# because this module is where the harness reads it from.


def failure_class_for_exception(exc: BaseException) -> str:
    """Classify an exception that escaped a state.

    Delegates the model-interface cases to `classify_wire_error`, which reads an
    HTTP status rather than a message; the pre-existing `network_error` /
    `unexpected_error` names are preserved for transport and for genuine defects.
    """
    return classify_wire_error(exc)


def failure_class(state: str, output: dict) -> str | None:
    """Classify a failed state from its structured output. None means it passed."""
    if state == "build_test":
        if not (output.get("build") or {}).get("ok", False):
            return "build_error"
        if (output.get("tests") or {}).get("failed", 0):
            return "test_failure"
        return None
    if state in ("regression_verify", "verify_regression"):
        if output.get("verdict") == "pass":
            return None
        regressed = [r for r in (output.get("regression") or []) if not r.get("ok")]
        return "regression" if regressed else "verification_failed"
    if state == "implement":
        if not (output.get("changes") or []) and (output.get("failed") or []):
            return "no_change"
        return None
    return None


def compute_deviation(planned: list[str] | None, observed: list[str]) -> dict:
    """Compare the approved plan's file list against what the snapshot saw change.

    Report-only: a planned file may legitimately need no edit, so a deviation is
    evidence for review, not a failure. `planned is None` means the plan declared
    no file list, in which case there is nothing to compare against.
    """
    if planned is None:
        return {"planned": None, "missing": [], "extra": [], "unplanned": True}
    planned_norm = {p.replace("\\", "/").lstrip("./") for p in planned}
    observed_norm = {p.replace("\\", "/").lstrip("./") for p in observed}
    return {
        "planned": sorted(planned_norm),
        "missing": sorted(planned_norm - observed_norm),   # planned, never touched
        "extra": sorted(observed_norm - planned_norm),     # touched, never planned
        "unplanned": False,
    }


class Harness:
    def __init__(self, cfg: AppConfig, repo: Repo, gateway: ToolGateway,
                 wiki: WikiClient | None = None, llm: LLMAdapter | None = None):
        self.cfg = cfg
        self.repo = repo
        self.gateway = gateway
        self.wiki = wiki or WikiClient(repo)
        # The tool layer cannot spawn a run; the Harness can. Injected rather than
        # reached for from below, so the MCP server keeps working standalone
        # (where it refuses the call with a code).
        if hasattr(gateway, "server"):
            gateway.server.set_delegate(self.delegate)
        # What this provider has been observed to accept, so a fact learned once is
        # not rediscovered on every run. Kept next to the database rather than in
        # the package directory: it describes an endpoint, not an install — and it
        # is keyed by provider|endpoint|model, so one store serves every route.
        self.capabilities = CapabilityStore(cfg.data_dir / "capabilities.json")
        if llm is not None:
            # An injected brain is the only brain. Building the routed adapters
            # anyway would let a caller who handed us one adapter have a second one
            # answer part of the run — and would run the credential check for a
            # provider this process never contacts.
            self.llm, self.routed = llm, {}
        else:
            # `roles=` is the set a route may name, not the set of state-machine
            # agents — it has to include `assistant` or `[llm.routing] assistant =
            # "strong"` would be refused as a typo. `AGENT_CLASSES` stays at five.
            self.llm, self.routed = build_routed_adapters(
                cfg.llm, capabilities=self.capabilities,
                roles={**AGENT_CLASSES, "assistant": ChatAgent})
        root = cfg.project_root
        agent_kw = {
            "repeated_call_threshold": cfg.harness.repeated_call_threshold,
            "max_tool_retries": cfg.harness.max_tool_retries,
            "tool_result_budget": cfg.harness.tool_result_budget,
            "prompt_total_budget": cfg.harness.prompt_total_budget,
            "prompt_section_budgets": cfg.harness.prompt_section_budgets or None,
        }
        self.agents: dict[str, BaseAgent] = {
            role: cls(self.routed.get(role, self.llm), self.gateway, root, **agent_kw)
            for role, cls in AGENT_CLASSES.items()
        }
        # Deliberately **not** in `self.agents`. Three things read that mapping and
        # all three should keep seeing exactly the state-machine roles: `test_routing`
        # asserts its key set, `identity.identity_hash` folds its adapters into a
        # run's fingerprint (a sixth entry would move every existing run's hash and
        # churn the frozen benchmark digest), and `_record_failure` resolves a failing
        # state's agent through it. The chat turn reaches its agent through
        # `self.chat_agent` instead.
        self.chat_agent = ChatAgent(
            self.routed.get("assistant", self.llm), self.gateway, root, **agent_kw)
        # One asyncio lock per run, so concurrent callers in this process queue
        # up instead of interleaving two advances of the same run.
        self._run_locks: dict[str, asyncio.Lock] = {}

    # ---------------------------------------------------------------- lifecycle
    async def __aenter__(self) -> Harness:
        await self.gateway.__aenter__()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.gateway.__aexit__(*exc)

    # -------------------------------------------------------------- run control
    def create_run(self, text: str, *, kind: str | None = None, title: str | None = None,
                   parent_run_id: str | None = None,
                   origin: str = ORIGIN_INTERACTIVE,
                   task_id: str | None = None,
                   task_version: int | None = None,
                   session_id: str | None = None) -> dict:
        """Start a run.

        `origin` says where this run came from, and it is not decoration: it scopes
        the memory and the skills the run may see, so a task or benchmark run
        cannot seed the prompt of a real one. A caller that drives runs from a
        task must say so.

        `task_id`/`task_version` are recorded, not enforced — the loop's whole
        point is comparing results across task revisions, and a result that does
        not say which revision produced it cannot be compared.

        `session_id` groups the runs of one invocation. A caller that drives many
        runs — a benchmark suite, an experiment — passes the same one to each, and
        "what did this invocation do" is then answerable across all of them at
        once. Left unset, a run gets a session of its own.
        """
        kind = kind or route(text)
        # The vocabulary is the machine table, not a literal pair: `chat` is a real
        # kind (it has a machine), and a kind with no machine is refused here rather
        # than at `sm.initial_state` two frames down. A route that finds nothing
        # still returns None, so an unrouted request is still an error.
        if kind not in sm.MACHINES:
            raise ValueError(f"无法为请求路由到已知流程类型: {text[:40]!r}")
        first = re.split(r"[\n。；;]", text, maxsplit=1)[0][:60]
        run = self.repo.create_run(
            kind, title or first, text, parent_run_id=parent_run_id,
            origin=origin, task_id=task_id, task_version=task_version,
            session_id=session_id)
        self._record_fingerprint(run["id"])
        # The first line of the trace, and the one that says which task revision
        # and which environment everything after it belongs to.
        self._emit(run, events.RUN_CREATED, kind=run["kind"], state=run["state"],
                   origin=run["origin"], task_id=task_id or "",
                   task_version=task_version if task_version is not None else "")
        return self.repo.get_run(run["id"])

    # ------------------------------------------------------------------ chat
    # How many past turns a chat prompt carries. The full transcript stays in the
    # trace either way; this is only the window the model is shown.
    CHAT_HISTORY_TURNS = 20

    def chat_history(self, run_id: str, *,
                     turns: int | None = None) -> list[dict[str, str]]:
        """This session's turns, oldest first, read back out of the trace.

        Rebuilt rather than stored on the run. `events` is the append-only record
        *and* the redacted one (`repo.add_event` cleans its payload; `update_run`
        does not), so keeping the transcript in a `runs.payload` column instead
        would put a pasted secret in the database verbatim and rewrite the whole
        conversation on every turn. The trace already holds both halves — the
        person's words and the assistant's — because that is what a transcript is.
        """
        limit = self.CHAT_HISTORY_TURNS if turns is None else turns
        out: list[dict[str, str]] = []
        for event in self.repo.events_for_run(run_id):
            kind = event["type"]
            if kind == events.CHAT_USER:
                role = "user"
            elif kind == events.CHAT_ASSISTANT:
                role = "assistant"
            else:
                continue
            out.append({"role": role,
                        "text": str(event["payload"].get("text") or "")})
        return out[-limit:] if limit > 0 else out

    @staticmethod
    def _render_conversation(turns: list[dict[str, str]]) -> str:
        lines = [f"[{'用户' if t['role'] == 'user' else '助手'}] {t['text']}"
                 for t in turns]
        return "\n".join(lines)

    async def converse(self, run_id: str, text: str) -> dict:
        """One turn of an interactive session: record what was asked, let the
        assistant work, record what it answered.

        Not a state transition, and deliberately not routed through `advance`: a
        conversation has no machine to move through. Everything else is the same
        machinery a state gets — the execution lease (a turn makes up to
        `ChatAgent.max_rounds` model calls and must not lose its claim mid-flight),
        the same context assembly, the same gateway, the same trace. A turn is a
        step; it is just a step whose outcome is a sentence rather than a
        transition suggestion.
        """
        run = self.repo.get_run(run_id)
        if run is None:
            raise ValueError(f"未找到运行 {run_id}")
        if run["kind"] != "chat":
            raise ValueError(f"运行 {run_id} 不是交互会话（kind={run['kind']!r}）")
        if run["status"] in TERMINAL_STATUSES:
            raise ValueError(f"运行 {run_id} 已经结束（{run['status']}）")

        exhausted = self._budget_exhausted(run)
        if exhausted:
            self._stop_for_budget(run, exhausted)
            return {"runId": run_id, "reply": "", "ok": False,
                    "reason": exhausted, "pendingApprovals": []}

        lock = self._lock_for(run_id)
        async with lock:
            owner = f"{os.getpid()}:{uuid.uuid4().hex[:8]}"
            lease_seconds = self._lease_seconds()
            if not self.repo.claim_run(run_id, owner, lease_seconds):
                held = self.repo.get_run(run_id)
                raise RunLockedError(
                    f"运行 {run_id} 正被 {held.get('owner')} 推进"
                    f"（租约至 {held.get('lease_until')}），本次未执行任何一轮")
            heartbeat = _LeaseHeartbeat(self.repo, run_id, owner, lease_seconds)
            try:
                return await self._converse_claimed(run, text)
            finally:
                await heartbeat.stop()
                self.repo.release_run(run_id, owner)

    async def _converse_claimed(self, run: dict, text: str) -> dict:
        run_id = run["id"]
        self.repo.set_status(run_id, "running")
        # The person's words go in first: everything after this point is the
        # assistant reading a context that already contains the question.
        self._emit(run, events.CHAT_USER, actor=events.ACTOR_HUMAN, text=text)

        # Only rows written from here on belong to this turn. `tool_calls` is
        # per-run and the session is long-lived, so the id is the boundary.
        before = max((t["id"] for t in self.repo.tool_calls(run_id)), default=0)

        run = self.repo.get_run(run_id)
        ctx = self._build_ctx(run)
        history = self.chat_history(run_id)
        if history:
            ctx["conversation"] = self._render_conversation(history)

        failure = ""
        try:
            output = await self.chat_agent.run(ctx)
            reply = str(output.get("reply") or "")
        except AgentError as e:
            # A turn that could not finish is a sentence in the conversation, not
            # the end of the session: one request burning its rounds says nothing
            # about whether the next request will work.
            failure = str(e)
            reply = f"（这一轮没能完成：{e}）"

        event_id = self._emit(run, events.CHAT_ASSISTANT, actor=events.ACTOR_MODEL,
                              text=reply)
        record = BaseAgent.record(ctx)
        self.repo.add_step(
            run_id, "chat", self.chat_agent.role,
            {"reply": reply, **({"failed": True} if failure else {})},
            status="failed" if failure else "done", error=failure or None,
            failure_class="agent_no_output" if failure else None,
            input_json={k: v for k, v in ctx.items()
                        if k not in ("gateway", RUN_RECORD)},
            event_id=event_id, metrics=self._step_metrics(self.chat_agent, record))

        blocked = self._request_approvals_for(run, since=before)
        return {"runId": run_id, "reply": reply, "ok": not failure,
                "reason": failure, "pendingApprovals": blocked}

    def _request_approvals_for(self, run: dict, *, since: int) -> list[str]:
        """Turn this turn's policy refusals into approvals a person can act on.

        A refused tool call is already recorded — the gateway writes the row and
        the trace event — but nothing would ever let a human un-refuse it: the
        delete gate reads the `approvals` table, and in a state machine the only
        thing that writes there is the plan-violation escalation. A conversation
        has no plan to violate, so the refusal would otherwise be permanent.

        The action is `<tool>:<path>` and not the bare tool name: the checker tries
        both, and a bare name would unlock that tool for *every* path — one
        approval would authorise deleting the whole project.
        """
        created: list[str] = []
        for row in self.repo.tool_calls(run["id"]):
            if row["id"] <= since or row.get("error_code") != "approval_required":
                continue
            try:
                args = json.loads(row.get("args_json") or "{}")
            except ValueError:
                args = {}
            path = str(args.get("path") or "")
            action = f"{row['tool']}:{path}" if path else str(row["tool"])
            self._ensure_approval(run, action,
                                  {"path": path, "reason": "交互式会话中被策略拦下"})
            created.append(action)
        return created

    def _lease_seconds(self) -> int:
        """How long one claim stays good for, before anyone else may take over.

        Must comfortably exceed a single model call, so that a renewal missed
        because the event loop was busy cannot let the claim lapse mid-call. It
        does *not* have to cover a whole state: `_LeaseHeartbeat` renews at a
        third of this interval for as long as the state runs. The knob wins so
        an operator can raise it; the provider timeout is the floor.
        """
        return max(self.cfg.harness.run_lease_seconds,
                   int(self.cfg.llm.timeout) + 60)

    def _lock_for(self, run_id: str) -> asyncio.Lock:
        """The in-process half of the mutex, one lock per run.

        Pruned once it grows: an unbounded per-id lock table is a slow leak in a
        long-lived process.
        """
        lock = self._run_locks.get(run_id)
        if lock is None:
            if len(self._run_locks) >= _RUN_LOCK_CAP:
                for key in [k for k, v in self._run_locks.items() if not v.locked()]:
                    del self._run_locks[key]
            lock = asyncio.Lock()
            self._run_locks[run_id] = lock
        return lock

    async def advance(self, run_id: str, max_loops: int = 80) -> dict | None:
        """Advance a run, holding both halves of the mutex.

        Within one process an `asyncio.Lock` serialises callers; across
        processes a lease on the run row decides, claimed with a single
        conditional UPDATE so exactly one caller can win. Two advances of the
        same run used to interleave freely and execute a state twice.

        The claim is held by a heartbeat for as long as the run is being
        advanced, so a state slower than the lease does not lose it.
        """
        run = self.repo.get_run(run_id)
        if run is None:
            return None
        if run["status"] in ("completed", "failed", "cancelled",
                             "waiting_child", "paused"):
            return run
        async with self._lock_for(run_id):
            owner = f"{os.getpid()}:{uuid.uuid4().hex[:8]}"
            lease_seconds = self._lease_seconds()
            if not self.repo.claim_run(run_id, owner, lease_seconds):
                held = self.repo.get_run(run_id)
                raise RunLockedError(
                    f"运行 {run_id} 正被 {held.get('owner')} 推进"
                    f"（租约至 {held.get('lease_until')}），本次未执行任何状态")
            heartbeat = _LeaseHeartbeat(self.repo, run_id, owner, lease_seconds)
            try:
                return await self._advance_claimed(run_id, max_loops, owner,
                                                   heartbeat)
            finally:
                await heartbeat.stop()
                self.repo.release_run(run_id, owner)
                # Both hang off the one place every run passes through on its
                # way to a terminal status, rather than off each of the several
                # ways a run can end.
                self._remember_ended_run(run_id)
                self._record_skills(run_id)

    async def _advance_claimed(self, run_id: str, max_loops: int, owner: str,
                               heartbeat: _LeaseHeartbeat) -> dict:
        run = self.repo.get_run(run_id)
        if run["kind"] not in sm.ADVANCEABLE_KINDS:
            # Fail-closed and inert: a chat session is stepped by `converse`, one
            # turn at a time, and has no `STATE_AGENT` entry. Without this, any
            # caller that reaches `advance` with a chat run — `wfos resume`,
            # `resume_after_child`, `delegate`, a task file — would KeyError inside
            # `_run_state`, mark the run failed, and escape as a traceback.
            return run
        refused = self._reuse_refused(run)
        if refused:
            # Checked here, inside the lease, and *before* the status flips to
            # running: a refused run must not be left looking like it is going.
            self._emit(run, events.REUSE_REFUSED, reason=refused)
            self.repo.update_run(run_id, status="failed", error=refused)
            return self.repo.get_run(run_id)
        if run["status"] == "waiting_approval":
            # Re-enter the gate: it re-blocks while approvals remain pending,
            # otherwise it proceeds. (Calling advance on a waiting run is safe.)
            self.repo.set_status(run_id, "running")
            run = self.repo.get_run(run_id)
        self.repo.set_status(run_id, "running")
        for _ in range(max_loops):
            if heartbeat.lost.is_set() or not self.repo.renew_lease(
                    run_id, owner, self._lease_seconds()):
                # Our claim lapsed mid-run, so someone else may be advancing too.
                # Stop rather than risk executing a state twice. The heartbeat
                # reports a lapse that happened *during* a state; this renewal
                # reports one that lands between states.
                raise RunLockedError(
                    f"运行 {run_id} 的执行租约在执行途中失效（可能被其他进程接管）；"
                    f"已停止以避免重复执行。可调大 harness.run_lease_seconds")
            run = self.repo.get_run(run_id)
            if run["status"] not in ("created", "running"):
                break
            if sm.is_terminal(run["state"]):
                break
            spent = self._budget_exhausted(run)
            if spent:
                self._emit(run, events.BUDGET_EXCEEDED, reason=spent,
                           usage=run_metrics(self.repo, run["id"]))
                self._stop_for_budget(run, spent)
                break
            refused = self._rejected_escalation(run)
            if refused:
                # "Stop retrying this failure" cannot be honoured by looping
                # again, so a refused escalation ends the run.
                self.repo.update_run(
                    run_id, status="failed",
                    error=f"失败升级审批被拒绝（{refused}），停止重试")
                break
            try:
                result = await self._execute_state(run)
            except AgentError:
                # Already recorded, with the context of the state that failed.
                # A model-interface failure ends the state and the loop.
                break
            except Exception:  # noqa: BLE001
                # Also already recorded — but re-raised, because a provider that
                # dropped the connection and a genuine defect must not both be
                # tidied away as a "failed" run.
                raise
            run = self.repo.get_run(run_id)
            await self._decide_transition(run, result)
            run = self.repo.get_run(run_id)
            if run["status"] in ("waiting_approval", "waiting_child", "paused",
                                 "completed", "failed", "cancelled"):
                break
        return self.repo.get_run(run_id)

    def approve(self, approval_id: str, *, by: str = "cli", note: str = "") -> dict | None:
        a = self.repo.decide_approval(approval_id, "approved", by=by, note=note)
        if a is None:
            return None
        self._emit_decision(a, "approved", by, note)
        return a

    def reject(self, approval_id: str, *, by: str = "cli", note: str = "") -> dict | None:
        a = self.repo.decide_approval(approval_id, "rejected", by=by, note=note)
        if a is not None:
            self._emit_decision(a, "rejected", by, note)
        return a

    def _emit_decision(self, approval: dict, decision: str, by: str,
                       note: str) -> None:
        """Record a human's decision as theirs, not the harness's.

        The actor matters: a decision the workflow made is reproducible and a
        decision a person made is not, and a trace that spelled both `harness`
        would make the two indistinguishable after the fact.
        """
        run = self.repo.get_run(approval["run_id"])
        if run is None:
            return
        self._emit(run, events.APPROVAL_DECIDED, actor=events.ACTOR_HUMAN,
                   action=approval["action"], decision=decision, by=by, note=note)

    async def resume(self, run_id: str, *, force_stale: bool = False) -> dict | None:
        """Resume a run — but only onto steps that are still reproducible.

        Fail-closed: if the environment or a key file changed since the steps
        were persisted, the run fails with the reason rather than silently
        reusing outputs produced under conditions that no longer hold. A human
        can override with `force_stale`, and the override is recorded.
        """
        run = self.repo.get_run(run_id)
        if run is None:
            return None
        if run["status"] in TERMINAL_STATUSES:
            # Nothing to resume, and nothing to decide. The fingerprint check below
            # ends in "mark the run failed", which on a finished run means rewriting
            # history: asking to resume a `completed` run used to flip it to
            # `failed`. A finished record is a record, not a candidate for revival.
            return run
        status, detail = self.resume_status(run)
        if status in self._STALE_STATUSES:
            if not force_stale:
                self.repo.update_run(
                    run_id, status="failed",
                    error=self._stale_reason(status, detail, run_id,
                                             lead="恢复被拒绝"))
                return self.repo.get_run(run_id)
            self.repo.add_transition(run_id, run["state"], run["state"], "harness",
                                     f"显式接受过期状态继续（{status}）", "cli")
        self.repo.update_run(run_id, payload={"resume_status": status,
                                              "resume_detail": detail,
                                              "force_stale": bool(force_stale)})
        run = self.repo.get_run(run_id)
        if run["status"] == "paused":
            # `advance` deliberately no-ops on a paused run, so resuming has to
            # un-pause first — otherwise `resume` on a paused run did nothing.
            self.repo.set_status(run_id, "running")
            run = self.repo.get_run(run_id)
        if run["status"] in ("created", "running", "waiting_approval"):
            return await self.advance(run_id)
        return run

    def cancel(self, run_id: str, *, by: str = "cli") -> dict | None:
        run = self.repo.get_run(run_id)
        if run is None:
            return None
        self.repo.set_status(run_id, "cancelled")
        return self.repo.get_run(run_id)

    async def resume_after_child(self, child_run_id: str) -> dict | None:
        """Child Bugfix completed -> re-open the parent's regression stage.

        Gate: the parent is resumed ONLY after the child actually reached
        `completed`. A running/queued child never resumes the parent (the caller
        receives the child run so it can observe the true state); a failed or
        cancelled child leaves the parent's regression unresolved, so the parent
        is marked `failed` instead of advancing.
        """
        child = self.repo.get_run(child_run_id)
        if child is None:
            return None
        parent_id = child.get("parent_run_id")
        if not parent_id:
            return None
        parent = self.repo.get_run(parent_id)
        if parent is None:
            return None
        child_status = child.get("status")
        if child_status != "completed":
            if child_status in ("failed", "cancelled"):
                self.repo.update_run(
                    parent_id, status="failed",
                    error=f"子流程 {child_run_id} 状态={child_status}，回归未解决")
                return self.repo.get_run(parent_id)   # re-read: fresh status
            return child                # not finished yet -> do not touch the parent
        if parent["status"] != "waiting_child":
            return parent
        resume_state = parent["payload"].get("resume_state")
        if resume_state:
            # The parent re-runs the state it was blocked on, so the step it left
            # there is stale.
            self._invalidate_step(parent, resume_state, "子流程完成后重开回归阶段")
        self.repo.update_run(
            parent_id, status="running",
            payload={"resume_state": None, "child_run_id": None, "last_child": child_run_id})
        return await self.advance(parent_id)

    # ------------------------------------------------------------ state execution
    async def _execute_state(self, run: dict) -> dict:
        """Run one state, recording its failure here rather than at the caller.

        The failure is recorded on this side of the exception because this is
        where the state's `ctx` lives — and the record of what the model was
        shown and what it spent is in `ctx`, not on the shared agent object.
        Re-raised either way: the caller decides whether to stop the loop or let
        a genuine defect keep its traceback.

        `box` is a local, not a field. Handing the context back through `self`
        would have re-created the very sharing this is here to remove: two runs
        advancing at once would read whichever context was stored last.
        """
        box: dict[str, Any] = {}
        try:
            return await self._run_state(run, box)
        except AgentError as e:
            # The agent says *why* it could not answer (see `llm.base.WIRE_CODES`);
            # falling back to `agent_no_output` only when it has no opinion keeps
            # the old name for the old case without flattening the new ones.
            self._record_failure(run, getattr(e, "wire_code", None) or "agent_no_output",
                                 f"agent 未产出可用输出: {e}", e,
                                 BaseAgent.record(box.get("ctx")))
            raise
        except Exception as e:  # noqa: BLE001 - recorded, then re-raised
            self._record_failure(run, failure_class_for_exception(e),
                                 f"状态执行期间抛出异常: {e}", e,
                                 BaseAgent.record(box.get("ctx")))
            raise

    async def _run_state(self, run: dict, box: dict[str, Any]) -> dict:
        state = run["state"]
        if state in HARNESS_STATES:
            return await self._harness_state(run)
        agent = self.agents[STATE_AGENT[state]]
        run_id = run["id"]
        # None on the replay path: a step that is already `done` is loaded rather
        # than run, so no context is built and there is nothing new to measure.
        ctx: dict[str, Any] | None = None
        step = self.repo.get_step(run_id, state)
        if step and step["status"] == "done":
            output: dict[str, Any] = step["output_json"] or {}
            fc = step.get("failure_class")
        else:
            ctx = self._build_ctx(run)
            box["ctx"] = ctx
            # `run` seeds the record into this dict before copying it, so what it
            # was shown and what it spent are readable from here afterwards.
            output = await agent.run(ctx)
            if state == "implement":
                # Attach the workspace-snapshot ground truth of what changed, so
                # the record of this step does not rest on the implementer's own
                # change list. Covers the whole run, including earlier attempts.
                observed = self.repo.affected_paths_for_run(run_id, agent="implementer")
                output = {**output, "actual_changes": observed,
                          "plan_deviation": compute_deviation(
                              self._plan_write_paths(run), observed)}
                violations = self.repo.plan_violations_for_run(run_id)
                if violations:
                    output = {**output, "plan_violations": violations}
            fc = failure_class(state, output)
            # persist the completed step BEFORE any transition
            serializable_ctx = {k: v for k, v in ctx.items()
                                if k not in ("gateway", RUN_RECORD)}
            # The prompt itself, not just its sizes. "If the model can see it, it
            # is recorded" (dsh): without it, every diagnosis has to infer what
            # the model was shown from `prompt_metadata`'s section sizes.
            # Redaction happens on the way in, like every other untrusted field.
            record = BaseAgent.record(ctx)
            serializable_ctx = {**serializable_ctx,
                                **(self._agent_input(record) or {})}
            # Written before the step so the step can name it. `event_id` is
            # what joins this row to the trace; without it the four tables stay
            # four independent sequences.
            event_id = self._emit(
                run, events.STEP_COMPLETED if fc is None else events.STEP_FAILED,
                state=state, agent=agent.role, failure_class=fc or "")
            self.repo.add_step(run_id, state, agent.role, output, status="done",
                               input_json=serializable_ctx, failure_class=fc,
                               event_id=event_id,
                               metrics=self._step_metrics(agent, record))
            if state == "implement" and output.get("plan_violations"):
                # Writes outside the approved plan never reached disk; escalate
                # each one to a human and drop this step, so the approved retry
                # re-runs the implementation instead of replaying a refused one.
                self._escalate_plan_violations(run, output["plan_violations"])
                self._invalidate_step(run, "implement", "越界写入已升级为审批，本步作废")
                return {"state": state, "output": output, "blocked": True}
        self._persist_evidence(run_id, output.get("evidence") or [])
        if state in ("feature_design", "tech_design", "root_cause", "fix_plan"):
            self._store_plan(run_id, state, output)
        if state == "knowledge_distill":
            self._distill(run_id, output)
        # Stamp last, so the fingerprint includes the plan this state may have
        # just stored: recording it earlier would leave key_files empty until
        # some later state ran, and nothing would be checkable in between.
        self._record_fingerprint(run_id)
        return {"state": state, "output": output, "failure_class": fc}

    def _remember_ended_run(self, run_id: str) -> None:
        """Record what a finished run left behind, for later runs to read.

        Called from `advance`'s teardown — the one place every run passes through
        on its way to a terminal status, whichever of the several ways it got
        there. The captured digests are the run's *end* state, which is what makes
        them usable as a freshness anchor; the run's `key_files` fingerprint is
        taken when the plan is established and describes the state the run
        *started* from, so every file the run then wrote would read as drift
        against it.

        A no-op for a run that is not finished, which is the common case: with a
        small `max_loops` most `advance` calls leave the run mid-flight.
        """
        run = self.repo.get_run(run_id)
        if run is None or run["status"] not in TERMINAL_STATUSES:
            return
        payload = run.get("payload") or {}
        if payload.get("memory_files") or payload.get("memory_request"):
            return                       # already recorded
        remembered: dict[str, Any] = {}
        paths = sorted({*self._plan_file_paths(run),
                        *self.repo.affected_paths_for_run(run_id)})
        if paths:
            remembered["memory_files"] = capture_end_state(self.cfg.project_root, paths)
        # The request, always. A run that failed before it planned or wrote
        # anything has no files to be recognised by — and that is precisely the
        # failure a later run most needs to hear about, so it used to be the one
        # case that recorded nothing at all.
        request = run.get("description") or run.get("title") or ""
        if request:
            remembered["memory_request"] = request
        if remembered:
            self.repo.update_run(run_id, payload=remembered)

    @staticmethod
    def _agent_input(record: dict | None) -> dict | None:
        """What the agent was shown, for the step record.

        Read from the *state's* record rather than off the agent object. The agent
        is shared by every run a Harness advances, so `agent.last_prompt` is one
        slot for the whole process — two runs advancing at once would file each
        other's prompt against the wrong step, which is the same misattribution
        the per-call principal fixed for tool calls.

        Recorded for failed steps too. A failure is exactly when a reader needs to
        know what the model was given — recording it only on the success path
        would answer the question precisely when it does not matter.
        """
        record = record or {}
        recorded: dict[str, Any] = {}
        if record.get("prompt"):
            recorded["prompt"] = record["prompt"]
        if record.get("prompt_metadata"):
            recorded["prompt_metadata"] = record["prompt_metadata"]
        return recorded or None

    async def delegate(self, principal: dict, args: dict) -> dict:
        """Spawn a child run for a sub-task and drive it to a terminal status.

        Blocking from the calling tool's point of view: the agent is inside a tool
        call, so the child is advanced before the result comes back. That is what
        makes the join a **structured result** — status, run id, what it changed —
        rather than the raw string the delegation pattern this follows hands back.

        The child is an ordinary run: its own budget, its own audit trail, its own
        approvals, its own lease. Nothing about the parent's constraints is
        inherited, and nothing about the child's writes is attributed to the
        parent (the snapshot attribution is per-run).
        """
        parent_id = str((principal or {}).get("run_id") or "")
        parent = self.repo.get_run(parent_id) if parent_id else None
        if parent is None:
            return {"ok": False, "error_code": "delegate_no_parent",
                    "error": "派生需要一个已存在的父运行，当前 principal 没有 run_id"}

        depth = self.repo.ancestor_depth(parent["id"])
        max_depth = self.cfg.harness.max_parent_depth
        if max_depth and depth + 1 > max_depth:
            # The gate is also upstream: past the limit the tool is not declared to
            # the agent at all. Enforced here too, because a declared-tool list the
            # model can ignore is not a boundary.
            return {"ok": False, "error_code": "delegate_depth_exceeded",
                    "error": (f"子流程深度 {depth + 1} 超过上限 {max_depth}，拒绝派生")}

        task = str(args.get("task") or "").strip()
        kind = args.get("kind") or route(task) or "bugfix"
        child = self.repo.create_run(
            kind, title=f"[子任务] {task[:50]}", description=task,
            parent_run_id=parent["id"])
        self._emit(parent, events.DELEGATED, child_run_id=child["id"],
                   task=task[:200])
        try:
            await self.advance(child["id"], max_loops=_DELEGATE_MAX_LOOPS)
        except RunLockedError as e:              # pragma: no cover - contention
            return {"ok": False, "error_code": "delegate_locked",
                    "child_run_id": child["id"], "error": str(e)}

        child = self.repo.get_run(child["id"])
        steps = self.repo.steps_for_run(child["id"])
        return {
            "ok": True,
            "child_run_id": child["id"],
            "kind": child["kind"],
            "status": child["status"],
            "state": child["state"],
            "steps": len(steps),
            "changed_paths": self.repo.affected_paths_for_run(child["id"], agent="implementer"),
            "failure_classes": sorted({s["failure_class"] for s in steps
                                       if s.get("failure_class")}),
            "error": child.get("error") or None,
        }

    def _record_skills(self, run_id: str) -> None:
        """Distil a procedure for every failure class this run got past.

        A run that never failed records nothing, which is the common case. A
        failure that was never recovered from also records nothing: the point
        is a procedure, not a post-mortem.
        """
        run = self.repo.get_run(run_id)
        if run is None:
            return
        record_from_run(self.repo, run)

    def _step_metrics(self, agent: BaseAgent | None,
                      record: dict | None = None) -> dict | None:
        """What one agent run cost, priced at the model configured right now.

        Cost is computed here rather than at read time: the model in use is only
        knowable while the step is being produced, and a price looked up later
        could belong to a different model. Storing it makes the figure a
        historical fact rather than a re-derivation.

        The counts come from the state's record, not from the agent object, for
        the reason `_agent_input` gives. The agent is still asked for the model,
        because that is a property of the adapter rather than of a run.

        None means there is nothing to report — a harness-decision state makes
        no model call at all.
        """
        record = record or {}
        if agent is None and not record:
            return None
        usage = dict(record.get("usage") or {})
        # Priced at the model that actually answered, read off the adapter rather
        # than off `cfg.llm.model`. With `[llm.routing]` those differ, and pricing a
        # routed step at the default model's rate would store a wrong number as a
        # historical fact — which is the one thing this method exists to avoid.
        model = getattr(getattr(agent, "llm", None), "model", "") or self.cfg.llm.model
        return {
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "cached_tokens": usage.get("cached_tokens"),
            "reasoning_tokens": usage.get("reasoning_tokens"),
            # Counted by the harness, not reported by a provider, so these are
            # always known — 0 here genuinely means zero, not "unmeasured".
            "model_calls": record.get("model_calls", 0),
            "latency_ms": record.get("latency_ms", 0),
            "model": model,
            "cost_usd": cost_usd(model, self.cfg.pricing,
                                 input_tokens=usage.get("input_tokens"),
                                 output_tokens=usage.get("output_tokens")),
        }

    def _plan_file_paths(self, run: dict) -> list[str]:
        """Every path the stored plan names, regardless of enforcement settings."""
        files = ((run.get("payload") or {}).get("plan") or {}).get("files") or []
        return [f["path"] for f in files
                if isinstance(f, dict) and f.get("path")]

    def _plan_write_paths(self, run: dict) -> list[str] | None:
        """The file set the approved plan permits writing, or None if unbounded.

        None means the plan declared no file list at all — you cannot narrow a
        write to a set that was never planned, and an empty allow-list would
        block every write for such a run.
        """
        if not self.cfg.harness.enforce_plan_scope:
            return None
        return self._plan_file_paths(run) or None

    # ------------------------------------------------------------ resume contract
    def _record_fingerprint(self, run_id: str) -> None:
        """Stamp the run with what its persisted steps were produced under.

        The key-file baseline is captured once, when the plan is first
        established, and deliberately NOT refreshed afterwards: a baseline that
        moved forward with every step would absorb any drift that happened
        mid-run, which is exactly the drift this is meant to catch. A genuinely
        new plan re-baselines, because it names different files.
        """
        run = self.repo.get_run(run_id)
        if run is None:
            return
        plan_paths = self._plan_file_paths(run)
        recorded = run.get("key_files") or {}
        # Compare on one spelling of each path: the plan may say "./app.py" while
        # a captured key is "app.py", and a mismatch there would re-baseline the
        # fingerprint on every single step.
        if {identity.normalize_path(p) for p in plan_paths} != set(recorded):
            recorded = identity.capture_key_files(self.cfg.project_root, plan_paths)
        self.repo.set_fingerprint(
            run_id, identity.identity_hash(self.cfg, self.agents), recorded)

    def resume_status(self, run: dict) -> tuple[str, dict]:
        """Whether this run's persisted steps may be reused as-is."""
        return identity.validate_resume(
            run.get("identity_hash"),
            run.get("key_files") or {},
            identity.identity_hash(self.cfg, self.agents),
            self.cfg.project_root,
            self.repo.affected_paths_for_run(run["id"]),
        )

    def _build_ctx(self, run: dict) -> dict:
        payload = run.get("payload") or {}
        prior = {}
        for s in ("feature_design", "tech_design", "root_cause", "fix_plan"):
            key = f"plan_{s}"
            if key in payload:
                prior[s] = payload[key]
        observed = self.repo.affected_paths_for_run(run["id"], agent="implementer")
        working_files = sorted({*self._plan_file_paths(run), *observed})
        return {
            "run": run, "state": run["state"],
            # The legal next states for this state, straight from the machine the
            # Harness will validate against. The model is asked to *suggest* one,
            # and without the list it can only guess at the vocabulary: a real
            # model invented `req_analysis`, the Harness correctly refused the
            # illegal transition, and the run failed on its first state. Refusing
            # is right — withholding the list that makes a suggestion legal is not.
            "allowed_states": sorted(
                sm.allowed_transitions(run["kind"], run["state"])),
            # False once one more level would exceed the limit. The agent drops
            # `delegate` from its declared surface on seeing this, and the tool
            # layer refuses it too — a boundary the model can talk past is not
            # one.
            "delegate_allowed": (
                not self.cfg.harness.max_parent_depth
                or self.repo.ancestor_depth(run["id"]) + 1
                <= self.cfg.harness.max_parent_depth),
            "plan": payload.get("plan") or {},
            "prior": prior,
            # This run's evidence, most relevant to its working set first and
            # capped. The ordering matters because the prompt's evidence section
            # is clipped by characters afterwards: without a ranking, *which*
            # items got cut was arbitrary rather than "the least relevant ones".
            "evidence": select_evidence(self.repo.list_evidence(run["id"]),
                                        working_files=working_files,
                                        limit=self.cfg.harness.evidence_limit),
            # Observed on-disk changes (snapshot diff), so the Verifier can hold
            # the implementation against the approved plan instead of trusting it.
            "actual_changes": observed,
            # How the implementation deviated from the approved file list —
            # evidence for review, never a gate (a planned file may legitimately
            # need no edit).
            "plan_deviation": compute_deviation(self._plan_write_paths(run), observed),
            # What the approved plan allows writing; enforced again in the MCP
            # server, so the agent cannot widen its own scope.
            "plan_write_paths": self._plan_write_paths(run),
            # Retry progress, so the model knows it is on its Nth attempt and
            # what has already failed. Without it every attempt reads as a first
            # one, which is how a retry loop spends its budget repeating itself.
            "retry": {
                "attempts": dict(payload.get("attempts") or {}),
                "failure_streaks": dict(payload.get("failure_attempts") or {}),
                "max_attempts": self.cfg.harness.max_verify_attempts,
                "escalate_at": self.cfg.harness.repeated_failure_threshold,
            },
            # The files this run is about: planned ∪ observed.
            "working_files": working_files,
            # What earlier runs on these files left behind. Derived from the
            # record rather than from a model, so it exists for runs that failed
            # as much as for ones that completed — and freshness-checked, so a
            # memory describing a state that no longer exists is withheld.
            # Procedures for the failure classes this run has already stumbled
            # on. Retrieved by class rather than by similarity: the class is
            # what the retry budget, the escalation rule and the skill all key
            # on, so there is one taxonomy instead of two.
            "skills": skills_for_run(self.repo, run),
            "memory": memory_for(self.repo, self.cfg.project_root, working_files,
                                 request=run.get("description") or run.get("title") or "",
                                 exclude_run_id=run["id"],
                                 origin=origin_of(run)),
            "gateway": self.gateway,
        }

    def _persist_evidence(self, run_id: str, items: list[dict]) -> None:
        existing = {(e["kind"], e["source"], e["content"]) for e in self.repo.list_evidence(run_id)}
        for e in items or []:
            key = (e.get("kind", "other"), e.get("source", ""), e.get("content", ""))
            if key in existing:
                continue
            self.repo.add_evidence(run_id, e.get("kind", "other"), e.get("source", ""),
                                   e.get("content", ""), e.get("confidence", "medium"))
            existing.add(key)

    def _store_plan(self, run_id: str, state: str, output: dict) -> None:
        payload: dict[str, Any] = {f"plan_{state}": output}
        if state == "tech_design" or state == "fix_plan":
            payload["plan"] = output
        elif state == "root_cause":
            payload["confidence"] = output.get("confidence", "medium")
            payload["root_cause"] = output.get("root_cause", "")
        self.repo.update_run(run_id, payload=payload)

    def _distill(self, run_id: str, output: dict) -> None:
        run = self.repo.get_run(run_id)
        evidence = self.repo.list_evidence(run_id)
        candidates = (output or {}).get("knowledge") or []
        if not candidates:
            return
        self.wiki.distill(run_id, candidates, evidence=evidence, run=run)

    # -------------------------------------------------------- harness-decision states
    async def _harness_state(self, run: dict) -> dict:
        state = run["state"]
        if state == "risk_assess":
            return self._risk_assess(run)
        if state == "confidence_assess":
            return self._confidence_assess(run)
        if state == "conditional_approval":
            return self._conditional_approval(run)
        return {"state": state, "output": {}, "target": "failed"}

    def _plan(self, run: dict) -> dict:
        return run.get("payload", {}).get("plan") or {}

    def _risk_assess(self, run: dict) -> dict:
        plan = self._plan(run)
        ops = plan.get("operations") or []
        files = plan.get("files") or []
        risk = plan.get("risk_level") or "low"
        needs = [o for o in ops if o.get("kind") in APPROVAL_KINDS]
        for o in needs:
            tool = "workspace.delete" if o["kind"] == "delete" else ""
            self._ensure_approval(run, _approval_action(o, tool), o)
        if risk == "high":
            self._ensure_approval(run, f"high_risk:{run['title']}",
                                  {"kind": "core_modify", "path": "", "reason": "高风险核心修改"})
        auto = (risk == "low" and len(files) <= self.cfg.harness.auto_approve_low_risk_files
                and not needs)
        return {"state": "risk_assess", "target": "conditional_approval", "output": {
            "auto_approved": auto, "needs_approval": bool(needs) or risk == "high"}}

    def _ensure_approval(self, run: dict, action: str, op: dict) -> None:
        run_id = run["id"]
        existing = {a["action"] for a in self.repo.pending_approvals_for_run(run_id)
                    + self.repo.list_approved(run_id)}
        if action in existing:
            return
        self._emit(run, events.APPROVAL_REQUESTED, action=action,
                   scope=op.get("path", "") or action,
                   reason=op.get("reason", ""))
        self.repo.create_approval(
            run_id, action, scope=op.get("path", "") or action,
            risk_level=op.get("kind") if op.get("kind") in APPROVAL_KINDS else "high",
            reason=op.get("reason", ""), required_by=["policy"])

    def _record_failure(self, run: dict, failure_class: str, reason: str,
                        exc: BaseException, record: dict | None = None) -> None:
        """Record a state that threw, so the run neither looks alive nor hides why.

        An exception escaping the Harness used to reach the caller as a bare
        traceback, leaving the run stranded in `running` with no steps and no
        error — a run that looks like it is still going and cannot explain
        itself. The failure belongs in the record like any other.
        """
        run_id = run["id"]
        state = run["state"]
        agent = self.agents.get(STATE_AGENT.get(state, ""))
        # Traced like any other failure. A failed step that is *not* in the trace
        # is the worse half of the problem this table exists for: the state ran,
        # spent tokens, and the record would show a gap.
        event_id = self._emit(run, events.STEP_FAILED, state=state,
                              agent=getattr(agent, "role", None) or "",
                              failure_class=failure_class, raised=True)
        self.repo.add_step(run_id, state, getattr(agent, "role", None), {},
                           status="failed", error=str(exc),
                           failure_class=failure_class,
                           input_json=self._agent_input(record),
                           event_id=event_id,
                           # A state that failed still spent whatever it spent,
                           # so the record says so instead of reading as free.
                           # The record is written as the loop goes, so it holds
                           # the rounds that ran before the failure.
                           metrics=self._step_metrics(agent, record))
        self.repo.add_transition(run_id, state, "failed", "harness", reason, "")
        self.repo.update_run(run_id, status="failed", error=str(exc))

    def _budget_exhausted(self, run: dict) -> str:
        """Which budget this run has used up, or "" — one line saying so.

        Checked *between* states, never inside one. A state is the smallest unit
        that leaves a consistent record (a step written, a transition decided), so
        stopping mid-state would leave a half-written step and no way to tell
        whether its side effects had already landed.

        A limit only bites against a measured quantity. With a brain that reports
        no usage the totals are None — not 0 — and nothing here fires: an
        unenforceable limit that silently does nothing is worse than no limit, so
        the honest outcome is that it does nothing *visibly*, and `wfos metrics`
        shows the coverage that explains why.
        """
        token_limit = self.cfg.harness.run_token_budget
        cost_limit = self.cfg.harness.run_cost_budget_usd
        if token_limit is None and cost_limit is None:
            return ""
        report = run_metrics(self.repo, run["id"])
        if token_limit is not None:
            used = billable_tokens(report)
            if used is not None and used > token_limit:
                return f"token 预算耗尽：本运行已用 {used}，上限 {token_limit}"
        if cost_limit is not None:
            cost = report.get("cost_usd")
            if cost is not None and cost > cost_limit:
                return f"成本预算耗尽：本运行已用 ${cost:.4f}，上限 ${cost_limit:.4f}"
        return ""

    def _stop_for_budget(self, run: dict, reason: str) -> None:
        """End a run that has spent its budget.

        `failed`, not a fourth terminal status. The run did not finish, and every
        reader of a status already handles "did not finish"; a new word would have
        to be taught to each of them for one situation, and the existing three
        would still not be wrong. What must not be lost is *why* — so it is in the
        error, in the transition's reason, and in the decision being the
        harness's rather than the model's.
        """
        self.repo.add_transition(run["id"], run["state"], run["state"],
                                 "harness", reason, "")
        self.repo.update_run(run["id"], status="failed", error=reason)

    # The two resume verdicts that mean "these persisted steps were produced
    # under conditions that no longer hold". `no-fingerprint` is deliberately not
    # one of them: a run with nothing to compare has nothing to contradict it.
    _STALE_STATUSES = (identity.RESUME_IDENTITY_MISMATCH,
                       identity.RESUME_WORKSPACE_DRIFT)

    def _stale_reason(self, status: str, detail: dict, run_id: str, *,
                      lead: str = "拒绝复用已落库的步骤") -> str:
        """The one message both entry points refuse with.

        `lead` differs because the situations do: `resume` was asked and is
        answering no, while `advance` is refusing something nobody asked for.
        """
        return (f"{lead}（{status}）：{detail.get('reason', '')}；详情={detail}。"
                f"若确认要用旧结果继续，"
                f"请显式执行 `wfos resume {run_id} --force-stale`")

    def _reuse_refused(self, run: dict) -> str:
        """Why this run must not continue on its persisted steps, or "".

        `resume` has always asked this; `advance` never did. So any process with
        any configuration could take over an interrupted run and silently reuse
        steps produced under a different model, project root or policy set — the
        exact thing the fingerprint exists to prevent. `wfos approve` goes down
        that path (`_resume_if_unblocked` → `_advance`) with no check at all, so
        approving a blocked run from a differently-configured machine was enough.

        An accepted override covers only the condition it was accepted for. A run
        let through with `--force-stale` while its workspace had drifted, which
        now *also* mismatches on identity, has met a new condition and needs its
        own decision: "accepted once" is not "accepted forever".
        """
        if not self.repo.steps_for_run(run["id"]):
            # Nothing persisted, so nothing to trust or distrust. This is also
            # the fresh-run path, which must stay free.
            return ""
        status, detail = self.resume_status(run)
        if status not in self._STALE_STATUSES:
            return ""
        accepted = run.get("payload") or {}
        if accepted.get("force_stale") and accepted.get("resume_status") == status:
            return ""
        return self._stale_reason(status, detail, run["id"])

    def _emit(self, run: dict, type: str, *,
              actor: str = events.ACTOR_HARNESS, **payload: Any) -> int:
        """Append one trace event for this run, and return its `event_id`.

        The run dict rather than a run id: the session id lives on it, and an
        event that does not carry the session cannot be grouped with the other
        runs of the same invocation — which is the whole point of the column.

        Returns the id so the caller can hand it to `add_step` / `add_transition`
        / `log_tool_call`: that is how those four independently-numbered tables
        become joinable.
        """
        return events.record(self.repo, type, run_id=run.get("id") or "",
                             session_id=run.get("session_id") or "",
                             actor=actor, payload=payload)

    def _invalidate_step(self, run: dict, state: str, reason: str) -> None:
        """Delete a step so its state re-executes, and record that it happened.

        This is the one place in the harness that rewrites the record: the row
        goes entirely, taking what the state cost and what it produced with it.
        That is exactly why the trace is append-only *around* it — the fact that
        the state ran is not deleted, and the deletion is written as an event
        rather than left as a silence.

        `erased` carries the facts the row held, so a reader of the trace can see
        what was thrown away instead of inferring it from a gap.
        """
        erased = _step_summary(self.repo.get_step(run["id"], state))
        self.repo.delete_steps_for_state(run["id"], state)
        self._emit(run, events.STEP_INVALIDATED, state=state, reason=reason,
                   erased=erased)

    def _rejected_escalation(self, run: dict) -> str:
        """The action of a rejected repeated-failure escalation, or "".

        Scoped to the approvals this escalation creates, so general approval
        semantics (which only settle at `conditional_approval`) are untouched.
        """
        for a in self.repo.list_approvals(run["id"]):
            if a["status"] == "rejected" and str(a["action"]).startswith("repeated_failure:"):
                return a["action"]
        return ""

    def _escalate_plan_violations(self, run: dict, violations: list[dict]) -> None:
        """Turn refused out-of-plan writes into approvals.

        The action key must match exactly what the MCP approval checker looks up
        (`<tool>:<path>`), otherwise approving it would not re-open the scope.
        """
        for v in violations or []:
            path = v.get("path") or ""
            tool = v.get("tool") or "workspace.write"
            self._ensure_approval(run, f"{tool}:{path}", {
                "path": path,
                "reason": f"{path} 不在已批准方案的文件列表内，越界写入需单独审批",
            })

    def _confidence_assess(self, run: dict) -> dict:
        run_id = run["id"]
        confidence = run.get("payload", {}).get("confidence") or "medium"
        if confidence == "high":
            return {"state": "confidence_assess", "target": "fix_plan", "output": {"confidence": "high"}}
        if confidence == "low":
            return {"state": "confidence_assess", "target": "evidence_collect", "output": {"confidence": "low"}}
        # medium -> human confirmation required before committing a fix
        action = "confirm_root_cause"
        pending = {a["action"] for a in self.repo.pending_approvals_for_run(run_id)}
        approved = {a["action"] for a in self.repo.list_approved(run_id)}
        if action not in pending and action not in approved:
            self.repo.create_approval(
                run_id, action, scope=run.get("payload", {}).get("root_cause", ""),
                risk_level="medium", reason="根因置信度为 medium，需人工确认",
                required_by=["confidence"])
            pending.add(action)
        if action in pending:
            return {"state": "confidence_assess", "target": "confidence_assess",
                    "blocked": True, "output": {"confidence": "medium"}}
        # confirmation granted -> proceed with the fix
        return {"state": "confidence_assess", "target": "fix_plan",
                "output": {"confidence": "medium"}}

    def _conditional_approval(self, run: dict) -> dict:
        run_id = run["id"]
        pending = self.repo.pending_approvals_for_run(run_id)
        if pending:
            return {"state": "conditional_approval", "target": "conditional_approval",
                    "blocked": True, "output": {"pending": [p["action"] for p in pending]}}
        all_a = self.repo.list_approvals(run_id)
        rejected = [a for a in all_a if a["status"] == "rejected"]
        if rejected:
            return {"state": "conditional_approval", "target": "failed",
                    "output": {"rejected": [a["action"] for a in rejected]}}
        return {"state": "conditional_approval", "target": "implement", "output": {}}

    # ------------------------------------------------------------- transitions
    def _compute_target(self, run: dict, state: str, output: dict, suggested: str) -> str:
        if state == "build_test":
            return "regression_verify" if output.get("verdict") == "pass" else "implement"
        if state == "regression_verify":
            if output.get("verdict") == "pass":
                return "knowledge_distill"
            regressed = [r for r in (output.get("regression") or []) if not r.get("ok")]
            return sm.SPAWN_BUGFIX if regressed else "implement"
        if state == "verify_regression":
            return "knowledge_distill" if output.get("verdict") == "pass" else "root_cause"
        if state == "implement":
            changes, failed = output.get("changes") or [], output.get("failed") or []
            if not changes and failed:
                return "failed"
            return suggested
        if state == "knowledge_distill":
            return "completed"
        return suggested

    async def _decide_transition(self, run: dict, result: dict) -> None:
        run_id = run["id"]
        state = result["state"]
        output = result.get("output") or {}
        suggested = (output.get("next_step") or {}).get("suggested_state", "")

        if result.get("blocked"):
            self.repo.set_status(run_id, "waiting_approval")
            return

        target = result.get("target") or self._compute_target(run, state, output, suggested)
        kind = run["kind"]
        if target == sm.SPAWN_BUGFIX:
            self._spawn_child(run, output)
            return
        ok, msg = sm.validate_transition(kind, state, target)
        if not ok:
            event_id = self._emit(run, events.TRANSITION, **{
                "from": state, "to": target, "reason": msg, "rejected": True,
                "suggested_by": suggested})
            self.repo.add_transition(run_id, state, target, "harness", msg,
                                     suggested, event_id=event_id)
            self.repo.set_status(run_id, "failed")
            return
        if target in RE_RUNNABLE and self.repo.step_done(run_id, target):
            # Re-entry: replaying the old output would make the retry a no-op.
            self._invalidate_step(run, target, "重入可重跑状态")

        # A repeated *kind* of failure means the retry loop is not converging.
        # Escalate to a human instead of spending the remaining retries on it.
        fc = result.get("failure_class")
        threshold = self.cfg.harness.repeated_failure_threshold
        payload: dict[str, Any] = {}
        if fc and threshold and target in RE_RUNNABLE:
            seen = dict((run.get("payload") or {}).get("failure_attempts", {}))
            key = f"{state}:{fc}"
            count = seen.get(key, 0) + 1
            if count >= threshold:
                self._ensure_approval(run, f"repeated_failure:{key}", {
                    "path": "",
                    "reason": f"同一失败类型连续出现 {count} 次（{key}），已停止自动重试",
                })
                # An approval grants a fresh retry budget, so a human decision
                # buys another round rather than resuming mid-exhaustion.
                self.repo.update_run(run_id, payload={"failure_attempts": {**seen, key: 0}})
                self.repo.add_transition(
                    run_id, state, state, "harness",
                    f"同一失败类型（{fc}）连续 {count} 次，升级人工审批", suggested)
                self.repo.set_status(run_id, "waiting_approval")
                return
            payload = {"failure_attempts": {**seen, key: count}}

        if target in RE_RUNNABLE:
            attempts = dict((run.get("payload") or {}).get("attempts", {}))
            n = attempts.get(target, 0) + 1
            if n > self.cfg.harness.max_verify_attempts:
                target = "failed"
            else:
                attempts[target] = n
                payload = {**payload, "attempts": attempts}
        self.repo.update_run(run_id, state=target, payload=payload)
        event_id = self._emit(run, events.TRANSITION, **{
            "from": state, "to": target, "reason": msg, "suggested_by": suggested})
        self.repo.add_transition(run_id, state, target, "harness", msg, suggested,
                                 event_id=event_id)
        if target == "completed":
            self.repo.set_status(run_id, "completed")
        elif target == "failed":
            self.repo.set_status(run_id, "failed")

    def _spawn_child(self, run: dict, output: dict) -> None:
        run_id = run["id"]
        # Children are bugfix runs, and a bugfix run can spawn its own child, so
        # without this bound a recurring regression recurses without limit.
        depth = self.repo.ancestor_depth(run_id)
        max_depth = self.cfg.harness.max_parent_depth
        if max_depth and depth >= max_depth:
            self.repo.add_transition(run_id, run["state"], sm.SPAWN_BUGFIX, "harness",
                                     f"子流程深度 {depth} 已达上限 {max_depth}，停止派生", "")
            self.repo.update_run(
                run_id, status="failed",
                error=(f"回归未解决，且子流程深度 {depth}/{max_depth} 已达上限；"
                       f"停止继续派生，需人工介入（父流程 {run_id}）"))
            return
        regressed = [r.get("module", "?") for r in (output.get("regression") or [])
                     if not r.get("ok")]
        child = self.repo.create_run(
            "bugfix", title=f"[子回归修复] {run['title']}",
            description=f"修复回归模块 {regressed}: {run['description']}",
            parent_run_id=run_id)
        self.repo.update_run(run_id, status="waiting_child",
                             payload={"child_run_id": child["id"], "resume_state": run["state"]})
        self.repo.add_transition(run_id, run["state"], sm.SPAWN_BUGFIX, "harness",
                                 f"生成子流程 {child['id']}", "")
