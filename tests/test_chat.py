"""The interactive entry: a conversation is a run, a turn is a step.

What these tests defend, beyond "it works":

  * a turn goes through the same machinery a state does — the gateway, the policy
    for its own role, the trace, the execution lease — and is not a side door;
  * the transcript is the trace, not a second copy kept somewhere the harness does
    not redact;
  * the session survives a turn that fails, because one bad request says nothing
    about the next;
  * `advance` refuses this kind outright, so nothing walks a conversation into a
    state machine that has no state for it.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from wfos import events
from wfos.harness.orchestrator import Harness
from wfos.llm.scripted import ScriptedAdapter
from wfos.models import ChatOutput
from wfos.storage.repo import Repo

ROOT = Path(__file__).resolve().parents[1]


def _harness(app, script=None):
    """The app fixture's stack, with the interactive agent on a scripted brain.

    `llm=` replaces the default adapter for every role, which is what puts
    `chat_agent` under the script — it is built from `self.llm` in `__init__`.
    """
    llm = ScriptedAdapter()
    if script is not None:
        llm.set_script(script)
    h = Harness(app["cfg"], app["repo"], app["gateway"], app["wiki"], llm=llm)
    return h, llm


def _session(h: Harness, title="交互式会话"):
    return h.create_run("（交互式会话）", kind="chat", title=title)


def _turn(h: Harness, run_id: str, text: str) -> dict:
    return asyncio.run(h.converse(run_id, text))


REPLY = {"output": {"reply": "看过了。"}}


# ------------------------------------------------------------------ one turn
def test_a_turn_answers_and_records_both_halves(app):
    h, _ = _harness(app, [
        {"tool_call": {"name": "workspace.read", "arguments": {"path": "app.py"}}},
        REPLY])
    repo = app["repo"]
    run = _session(h)

    out = _turn(h, run["id"], "帮我看看 app.py")

    assert out["reply"] == "看过了。"
    assert out["ok"] is True
    # The tool call went through the gateway, so it is a policy-checked, attributed
    # row — not something the conversation did behind the harness's back.
    calls = repo.tool_calls(run["id"])
    assert [c["tool"] for c in calls] == ["workspace.read"]
    assert calls[0]["ok"] == 1 and calls[0]["agent"] == "assistant"

    kinds = [e["type"] for e in repo.events_for_run(run["id"])]
    assert kinds == [events.RUN_CREATED, events.CHAT_USER, events.TOOL_CALLED,
                     events.CHAT_ASSISTANT]
    # Actors are the point: a decision a person made and one the model made are
    # different facts, and a trace that spelled both `harness` would lose that.
    actors = {e["type"]: e["actor"] for e in repo.events_for_run(run["id"])}
    assert actors[events.CHAT_USER] == events.ACTOR_HUMAN
    assert actors[events.CHAT_ASSISTANT] == events.ACTOR_MODEL


def test_the_reply_lands_as_a_step_with_its_cost(app):
    h, _ = _harness(app, [REPLY])
    repo = app["repo"]
    run = _session(h)

    _turn(h, run["id"], "在吗")

    step = repo.get_step(run["id"], "chat")
    assert step["status"] == "done" and step["agent"] == "assistant"
    assert step["output_json"]["reply"] == "看过了。"


def test_the_transcript_lives_in_the_trace_not_in_the_run_payload(app):
    """`add_event` redacts its payload; `update_run` does not.

    A copy of the conversation in `runs.payload` would therefore put a pasted
    secret in the database verbatim — and would be rewritten in full on every
    turn. The transcript is read back out of the events instead, which is the
    append-only and cleaned record.
    """
    h, _ = _harness(app, [REPLY])
    repo = app["repo"]
    run = _session(h)

    _turn(h, run["id"], "我的 key 是 sk-abcdefghijklmnop1234")

    assert "conversation" not in (repo.get_run(run["id"])["payload"] or {})
    stored = json.dumps(repo.events_for_run(run["id"]), ensure_ascii=False)
    assert "sk-abcdefghijklmnop1234" not in stored
    assert "[已脱敏]" in stored or "sk-" not in stored


# --------------------------------------------------------------- two turns
def test_the_second_turn_carries_the_first(app):
    h, llm = _harness(app, [REPLY, REPLY])
    seen: list[str] = []

    class _Recorder(ScriptedAdapter):
        async def complete(self, **kw):
            seen.append(str((kw.get("messages") or [{}])[0].get("content") or ""))
            return await super().complete(**kw)

    h.chat_agent.llm = _Recorder()
    h.chat_agent.llm.set_script([REPLY, REPLY])
    run = _session(h)

    _turn(h, run["id"], "第一个问题：看看 app.py")
    _turn(h, run["id"], "第二个问题：顺便看看 check.py")

    assert "第一个问题" in seen[0]
    assert "第一个问题" in seen[1] and "第二个问题" in seen[1]
    assert h.chat_history(run["id"]) == [
        {"role": "user", "text": "第一个问题：看看 app.py"},
        {"role": "assistant", "text": "看过了。"},
        {"role": "user", "text": "第二个问题：顺便看看 check.py"},
        {"role": "assistant", "text": "看过了。"},
    ]
    _ = llm


# ------------------------------------------------------------------ the gate
def test_a_refused_delete_becomes_an_approval_a_person_can_grant(app):
    """The delete gate reads the approvals table, and nothing in a conversation
    writes there — so without this the refusal would simply be permanent."""
    repo = app["repo"]
    delete = {"tool_call": {"name": "workspace.delete", "arguments": {"path": "app.py"}}}
    h, llm = _harness(app, [delete, REPLY])
    run = _session(h)

    first = _turn(h, run["id"], "把 app.py 删掉")

    assert (app["sandbox"] / "app.py").exists(), "未经批准就不该落盘"
    assert first["pendingApprovals"] == ["workspace.delete:app.py"]
    pending = repo.pending_approvals_for_run(run["id"])
    assert [a["action"] for a in pending] == ["workspace.delete:app.py"]

    repo.decide_approval(pending[0]["id"], "approved", by="tester")
    llm.set_script([delete, REPLY])
    second = _turn(h, run["id"], "再删一次")

    assert second["reply"] == "看过了。"
    assert not (app["sandbox"] / "app.py").exists(), "批准之后应该真的删掉"


def test_an_approval_names_one_path_not_the_whole_tool(app):
    """A bare `<tool>` action would unlock that tool for every path."""
    h, _ = _harness(app, [
        {"tool_call": {"name": "workspace.delete", "arguments": {"path": "app.py"}}},
        REPLY])
    repo = app["repo"]
    run = _session(h)

    _turn(h, run["id"], "删掉 app.py")

    action = repo.pending_approvals_for_run(run["id"])[0]["action"]
    assert action == "workspace.delete:app.py"
    assert action != "workspace.delete"


# --------------------------------------------------------- role and refusal
def test_the_assistant_is_not_offered_tools_it_may_not_use(app):
    """It writes, and the tool it may not have is not even on the list.

    The refusal lands one layer earlier than the policy: `delegate` is absent from
    `ChatAgent.allowed_tools`, so it never reaches the model's tool list and a call
    to it is `unknown_tool` rather than `role_denied`. That is the better refusal —
    a model cannot plan around a tool it was never shown.
    """
    h, _ = _harness(app, [
        {"tool_call": {"name": "workspace.write",
                       "arguments": {"path": "made.py", "content": "x = 1\n"}}},
        {"tool_call": {"name": "delegate", "arguments": {"task": "做个别的"}}},
        REPLY])
    repo = app["repo"]
    run = _session(h)

    _turn(h, run["id"], "写一个文件")

    assert (app["sandbox"] / "made.py").exists()
    rows = {c["tool"]: c for c in repo.tool_calls(run["id"])}
    assert rows["workspace.write"]["ok"] == 1
    assert rows["delegate"]["ok"] == 0
    assert rows["delegate"]["error_code"] == "unknown_tool"


def test_the_policy_refuses_the_assistant_at_the_gateway_too(app):
    """And if it is asked for anyway, the gateway is what says no.

    Two independent layers on purpose: the tool list is what the model is told,
    and the policy is what is enforced. Only the second one is a boundary.
    """
    principal = {"role": "assistant", "agent": "assistant", "run_id": "chat-run",
                 "read_roots": ["."], "write_roots": ["."],
                 "allowed_write_paths": None, "session_id": "s"}

    refused = asyncio.run(app["gateway"].call(
        "wiki.add_candidate", {"title": "t", "content": "c"}, principal=principal))
    allowed = asyncio.run(app["gateway"].call(
        "workspace.write", {"path": "via-gateway.py", "content": "x = 1\n"},
        principal=principal))

    assert not refused.ok and refused.structured["error_code"] == "role_denied"
    assert allowed.ok, allowed.error


def test_a_turn_that_cannot_finish_is_a_sentence_not_the_end_of_the_session(app):
    """One request burning its rounds says nothing about the next one."""
    h, llm = _harness(app, [
        {"tool_call": {"name": "workspace.read", "arguments": {"path": "app.py"}}}])
    repo = app["repo"]
    run = _session(h)

    out = _turn(h, run["id"], "看看")

    assert out["ok"] is False
    assert "没能完成" in out["reply"]
    assert repo.get_step(run["id"], "chat")["status"] == "failed"
    assert repo.get_run(run["id"])["status"] != "failed", "会话没有因此结束"

    llm.set_script([REPLY])
    again = _turn(h, run["id"], "换个说法")
    assert again["ok"] is True and again["reply"] == "看过了。"


def test_a_finished_session_refuses_another_turn(app):
    h, _ = _harness(app, [REPLY])
    repo = app["repo"]
    run = _session(h)
    repo.set_status(run["id"], "completed")

    with pytest.raises(ValueError, match="已经结束"):
        _turn(h, run["id"], "还在吗")


def test_only_a_chat_run_can_be_conversed_with(app):
    h, _ = _harness(app, [REPLY])
    run = app["harness"].create_run("新增一个模块", kind="feature")

    with pytest.raises(ValueError, match="不是交互会话"):
        _turn(h, run["id"], "在吗")


# --------------------------------------------------------------- the guard
def test_advance_refuses_a_chat_run_instead_of_crashing(app):
    """A chat run has no `STATE_AGENT` entry.

    Without the guard, `wfos resume <chat_run>` — and `resume_after_child`,
    `delegate`, a task file — would KeyError inside `_run_state`, mark the run
    failed, and escape `advance` as a traceback.
    """
    h, _ = _harness(app, [REPLY])
    repo = app["repo"]
    run = _session(h)

    same = asyncio.run(h.advance(run["id"]))

    assert same["status"] == "created" and same["state"] == "chat"
    assert repo.transitions(run["id"]) == []
    assert [e["type"] for e in repo.events_for_run(run["id"])] == [events.RUN_CREATED]


def test_the_chat_kind_is_a_machine_but_not_an_advanceable_one(app):
    from wfos.harness import statemachine as sm

    assert "chat" in sm.MACHINES
    assert sm.initial_state("chat") == "chat"
    assert sm.allowed_transitions("chat", "chat") == set()
    assert "chat" not in sm.ADVANCEABLE_KINDS
    assert "chat" not in ("feature", "bugfix")


# ---------------------------------------------------------------- the CLI
def _env(app, tmp_path):
    return {**os.environ, "WFOS_DATA": str(tmp_path / "chatdata"),
            "WFOS_PROJECT": str(app["sandbox"]), "WFOS_PROVIDER": "mock",
            "PYTHONIOENCODING": "utf-8"}


def _run_cli(args, env, **kw):
    return subprocess.run([sys.executable, "-m", "wfos", *args], cwd=str(ROOT),
                          capture_output=True, encoding="utf-8", errors="replace",
                          env=env, **kw)


def test_the_bare_command_starts_a_session_and_ends_on_exit(app, tmp_path):
    (tmp_path / "chatdata").mkdir(exist_ok=True)
    env = _env(app, tmp_path)

    proc = _run_cli([], env, input="帮我看看这个项目\nexit\n", timeout=180)

    assert proc.returncode == 0, proc.stderr
    assert "交互会话" in proc.stdout
    assert "会话结束" in proc.stdout


def test_once_runs_a_single_turn_without_reading_stdin(app, tmp_path):
    (tmp_path / "chatdata").mkdir(exist_ok=True)
    env = _env(app, tmp_path)

    proc = _run_cli(["chat", "--once", "帮我看看这个项目"], env, timeout=180)

    assert proc.returncode == 0, proc.stderr
    assert "会话结束" in proc.stdout
    # And the run it left behind is an ordinary chat run in the ordinary store.
    repo = Repo(Path(env["WFOS_DATA"]) / "wfos.db")
    run = repo.list_runs(limit=1)[0]
    assert run["kind"] == "chat" and run["status"] == "completed"
    assert any(c["tool"] == "workspace.list_files" for c in repo.tool_calls(run["id"]))


def test_every_subcommand_still_has_help(app):
    """`chat` joined the surface; the sweep that checks the surface sees it."""
    proc = _run_cli(["--help"], {**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert "chat" in proc.stdout


def test_the_prompt_section_is_absent_for_every_other_agent(app):
    """The conversation is only ever produced by the interactive entry — and an
    empty section still emits its separator, so the key must be omitted, not
    blank. Anything else would move the prompts of every existing run."""
    from wfos.harness import context as ctx_mod

    prompts, _ = ctx_mod.fit({"rules": "R", "run_context": "C"})
    assert prompts == "R\n\nC"
    assert ctx_mod.SECTION_ORDER[:3] == ("rules", "run_context", "working_set")
    assert set(ctx_mod.REDUCTION_ORDER) <= set(ctx_mod.SECTION_ORDER)
    assert "conversation" in ctx_mod.SECTION_ORDER
    assert "conversation" in ctx_mod.REDUCTION_ORDER


def test_the_chat_output_schema_is_what_the_agent_is_told(app):
    assert list(ChatOutput.model_fields) == ["reply"]
    assert h_has_chat_agent(app)


def h_has_chat_agent(app) -> bool:
    h, _ = _harness(app, [REPLY])
    assert h.chat_agent.role == "assistant"
    assert h.chat_agent.output_schema is ChatOutput
    # And it is *not* one of the state machine's agents: three things read that
    # mapping, including the resume fingerprint.
    assert "assistant" not in h.agents
    assert set(h.agents) == {"investigator", "architect", "implementer",
                             "verifier", "curator"}
    return True
