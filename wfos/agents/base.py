"""Agent base: every agent declares permissions, then runs with them.

The Harness passes the run context (`ctx`). The agent fixes its principal on
the MCP gateway before it does anything, builds a system prompt that carries
the task, run state, plan, evidence, available tools and the output schema,
then drives a tool-calling loop against the LLM adapter. Any native tool call
the model makes is executed through the gateway (path/role constrained) and the
result is fed back into the conversation; the loop ends when the model returns
a structured output, which is validated against the agent's schema. Unstructured
text is never trusted as an execution result.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from ..harness import context
from ..llm.base import SYSTEM_RULES, WIRE_OUTPUT_MALFORMED, WIRE_OUTPUT_TRUNCATED, LLMAdapter
from ..mcp.client import ToolGateway
from ..mcp.policy import CODE_REPEATED_CALL, CODE_UNKNOWN_TOOL
from ..memory import render as render_memory
from ..models import Role, ToolCall
from ..skills import render as render_skills

# Where an agent keeps what it was shown and what it spent, for the state it is
# running. Keyed into `ctx` rather than stored on the agent — see `record_of`.
RUN_RECORD = "run_record"



class AgentError(Exception):
    """A state could not be completed by its agent.

    `wire_code` carries a model-interface failure class (`llm.base.WIRE_CODES`)
    when the cause is the conversation rather than the task. Without it every
    case was recorded as `agent_no_output`, so a truncated answer and a malformed
    one — which need opposite responses — read identically in the record.
    """

    def __init__(self, message: str, *, wire_code: str | None = None):
        super().__init__(message)
        self.wire_code = wire_code


# Consecutive identical tool calls tolerated before the loop stops executing
# them. Re-issuing the same (name, arguments) pair yields no new information —
# it is a loop, not progress. Overridden by HarnessConfig.repeated_call_threshold.
DEFAULT_REPEATED_CALL_THRESHOLD = 2

# How many undeclared-tool calls are answered with a structured error before the
# state is abandoned. Naming a tool that does not exist is a slip a model can
# usually correct once it is told what the tools are, and aborting the state on
# the first one turns a recoverable mistake into a failed run. Past the budget
# the old path resumes unchanged. 0 disables the retry (abort on the first call),
# matching the convention `repeated_call_threshold` and `max_parent_depth` use.
DEFAULT_TOOL_RETRY_LIMIT = 2


def _call_signature(name: str, arguments: dict | None) -> str:
    """Stable identity of a tool call: name plus canonicalized arguments."""
    return name + "\x00" + json.dumps(arguments or {}, sort_keys=True,
                                      ensure_ascii=False, default=str)


# The counts a provider may report. Named once so the accumulator and the record
# it feeds cannot drift apart.
USAGE_FIELDS = ("input_tokens", "output_tokens", "cached_tokens",
                "reasoning_tokens")


def _accumulate_usage(totals: dict[str, int], reported: set[str], usage) -> None:
    """Add one call's reported counts to the running totals.

    A count the provider did not report is skipped rather than added as 0, and
    the set of counts seen at least once is tracked separately — so the final
    record can contain only the fields that were actually observed. Averaging an
    unreported count as zero would report a cost saving that never happened.
    """
    if usage is None:
        return
    for field in USAGE_FIELDS:
        value = getattr(usage, field, None)
        if value is None:
            continue
        totals[field] = totals.get(field, 0) + value
        reported.add(field)


class AgentManifest(BaseModel):
    role: Role
    description: str = ""
    read_roots: list[str] = []
    write_roots: list[str] = []
    allowed_tools: list[str] = []
    timeout: int = 120
    max_rounds: int = 8
    failure_strategy: str = "report_failure"   # report_failure | fallback_suggest
    output_schema: str = ""


class BaseAgent:
    role: Role = "investigator"
    description = ""
    read_roots: list[str] = []
    write_roots: list[str] = []
    allowed_tools: list[str] = []
    timeout: int = 120
    max_rounds: int = 8
    failure_strategy = "report_failure"
    output_schema: type[BaseModel]

    def __init__(self, llm: LLMAdapter, gateway: ToolGateway, project_root: str | Path = ".",
                 *, repeated_call_threshold: int | None = None,
                 max_tool_retries: int | None = None,
                 tool_result_budget: int | None = None,
                 prompt_total_budget: int | None = None,
                 prompt_section_budgets: dict | None = None):
        self.llm = llm
        self.gateway = gateway
        self.project_root = Path(project_root).resolve()
        self.repeated_call_threshold = (
            DEFAULT_REPEATED_CALL_THRESHOLD if repeated_call_threshold is None
            else repeated_call_threshold)
        self.max_tool_retries = (
            DEFAULT_TOOL_RETRY_LIMIT if max_tool_retries is None else max_tool_retries)
        self.tool_result_budget = (
            context.DEFAULT_TOOL_RESULT_BUDGET if tool_result_budget is None
            else tool_result_budget)
        self.prompt_total_budget = (
            context.DEFAULT_TOTAL_BUDGET if prompt_total_budget is None
            else prompt_total_budget)
        self.prompt_section_budgets = prompt_section_budgets
        # Relative manifests are resolved against the project root so that tools
        # never resolve against the process CWD.
        self._read_roots = [str(self.project_root / r) for r in self.read_roots] \
            or [str(self.project_root)]
        self._write_roots = [str(self.project_root / r) for r in self.write_roots] or []

    # ------------------------------------------------------------------ helpers
    def manifest(self) -> AgentManifest:
        return AgentManifest(
            role=self.role, description=self.description,
            read_roots=list(self.read_roots), write_roots=list(self.write_roots),
            allowed_tools=list(self.allowed_tools), timeout=self.timeout,
            max_rounds=self.max_rounds, failure_strategy=self.failure_strategy,
            output_schema=self.output_schema.__name__)

    # ----------------------------------------------------------- tool surface
    def _tool_specs(self, ctx: dict | None = None) -> list[dict]:
        """Declared tools resolved to {name, description, inputSchema} dicts.

        Only tools the agent declared in `allowed_tools` are exposed to the
        model; the MCP server independently enforces the same set, so a model
        that fabricates a tool name can never execute it.

        `delegate` is further removed when the run is already as deep as it may
        go. Removed from the surface, not refused on call: a tool the model
        never sees is one it cannot spend a round on, and that is the shape
        that actually bounds recursion.
        """
        allowed = list(self.allowed_tools)
        if ctx is not None and ctx.get("delegate_allowed") is False:
            allowed = [n for n in allowed if n != "delegate"]
        specs: list[dict] = []
        for name in allowed:
            sp = self.gateway.spec(name)
            if sp is None:
                continue
            specs.append({"name": sp.name, "description": sp.description,
                          "inputSchema": sp.input_schema})
        return specs

    @staticmethod
    def record_of(ctx: dict) -> dict:
        """The per-state record this agent writes into `ctx`.

        Deliberately **not** instance state. The agent objects are built once per
        Harness and shared by every run it advances, so a `self.last_usage` is one
        slot for the whole process — two runs advancing concurrently read each
        other's prompt and each other's cost, and the step row is filed against
        the wrong run. That is the same misattribution the per-call principal
        fixed for tool calls, one layer up.

        `ctx` is built fresh for one state of one run, which is exactly the
        lifetime this record has. It also means a state that throws mid-loop
        still reports what it spent, because the record was being written as it
        went rather than only on the way out.
        """
        return ctx.setdefault(RUN_RECORD, {})

    @staticmethod
    def record(ctx: dict | None) -> dict:
        """The record from a `ctx`, or `{}` — for callers that may not have one."""
        return (ctx or {}).get(RUN_RECORD) or {}

    def build_prompt(self, ctx: dict, *, tools: list[dict] | None = None) -> str:
        """Full agent prompt, assembled under an explicit size budget.

        Sections are named so that compression is deliberate: the rules, the
        task, the tool list and the output schema are never clipped (shrinking
        them turns a long prompt into a wrong one), while evidence and plans are
        reduced in a fixed order once the total overruns. The budget accounting
        goes into `self.record_of(ctx)` so a shrinking prompt is visible.
        """
        run = ctx.get("run") or {}
        run_context = (
            "## 运行上下文\n"
            f"- 运行 ID: {run.get('id', '')}\n"
            f"- 流程类型: {run.get('kind', '')}\n"
            f"- 当前状态: {ctx.get('state', '')}\n"
            f"- 任务: {run.get('description') or run.get('title') or ''}")
        allowed = ctx.get("allowed_states") or []
        if allowed:
            # The vocabulary a `suggested_state` may be drawn from. Naming it is
            # the difference between asking the model to choose and asking it to
            # guess; the Harness refuses anything else, so a guess is a failed run.
            run_context += ("\n- 本状态允许建议的下一状态"
                            f"（suggested_state 必须取自这里）: {'、'.join(allowed)}")
        sections: dict[str, str] = {
            "rules": "\n\n".join([
                SYSTEM_RULES,
                f"你是 wfos 的 {self.role} 智能体。{self.description}",
            ]),
            "run_context": run_context,
        }
        plan = ctx.get("plan") or {}
        if plan:
            sections["plan"] = ("- 当前方案/计划: "
                                f"{json.dumps(plan, ensure_ascii=False, default=str)}")
        prior = ctx.get("prior") or {}
        if prior:
            sections["prior_plan"] = ("- 此前的方案: "
                                      f"{json.dumps(prior, ensure_ascii=False, default=str)}")
        evidence = (ctx.get("evidence") or {}).get("items") or []
        if evidence:
            sections["evidence"] = ("- 已有证据: "
                                    f"{json.dumps(evidence, ensure_ascii=False, default=str)}")
        working = self._render_working_set(ctx)
        if working:
            sections["working_set"] = "## 工作集\n" + working
        skills = ctx.get("skills") or []
        if skills:
            sections["skills"] = (
                "## 已知修复手法（由 Harness 从既往成功运行中推导，非模型结论）\n"
                + render_skills(skills))
        memory = ctx.get("memory") or {}
        if memory.get("items"):
            sections["memory"] = (
                "## 历史运行记忆（同一批文件上的既往运行，摘自 Harness 的运行记录）\n"
                + render_memory(memory))
        changes = self._render_changes(ctx)
        if changes:
            sections["changes"] = changes
        if tools:
            lines = ["## 可用工具（只能使用下列工具；未声明的工具调用会被拒绝）"]
            for t in tools:
                lines.append(
                    f"- `{t['name']}`: {t['description']}\n"
                    f"  参数 Schema: {json.dumps(t['inputSchema'], ensure_ascii=False)}")
            sections["tools"] = "\n".join(lines)
        out_schema = self.output_schema.model_json_schema()
        sections["output_schema"] = (
            "## 输出要求\n"
            "最终必须输出一个符合下列 JSON Schema 的 JSON 对象，不得附加任何解释。\n"
            f"```json\n{json.dumps(out_schema, ensure_ascii=False)}\n```")

        prompt, metadata = context.fit(
            sections, total_budget=self.prompt_total_budget,
            budgets=self.prompt_section_budgets)
        # Keep the assembled prompt itself, not just its section sizes. dsh's rule
        # — "if the model can see it, it is recorded" — is the one that makes a
        # failure answerable: every diagnosis so far could only infer what the
        # model had been shown from the sizes in `prompt_metadata`.
        self.record_of(ctx)["prompt"] = prompt
        # How much memory existed, was injected, and was withheld for staleness.
        # Deliberately not in the prompt: a model cannot act on the fact that
        # something went stale, but the run's audit should be able to show it —
        # an empty memory section that is empty for a reason is a different fact
        # from one that is empty because nothing was ever recorded.
        metadata["memory"] = {
            "injected": len(memory.get("items") or []),
            "stale": len(memory.get("stale") or []),
            "considered": int(memory.get("considered") or 0),
        }
        # Same reason: which evidence was cut for count, and how much. The section
        # budget reports what it clipped by characters; without this, evidence
        # that never reached the section at all would leave no trace.
        selection = ctx.get("evidence") or {}
        metadata["evidence"] = {
            "injected": len(selection.get("items") or []),
            "total": int(selection.get("total") or 0),
            "dropped": int(selection.get("dropped") or 0),
        }
        # What the adapter learned about this endpoint while talking to it. A
        # capability discovered mid-run changes what the model was asked — the
        # output ceiling moves, or the structured-output mode changes — so it
        # belongs in the same record as the prompt, not only in a side file.
        wire_notes = list(getattr(self.llm, "observations", []) or [])
        if wire_notes:
            metadata["wire_observations"] = wire_notes
        self.record_of(ctx)["prompt_metadata"] = metadata
        return prompt

    def _render_working_set(self, ctx: dict) -> str:
        """What this state is actually working on: the files in play, and how
        many attempts have already failed.

        A model that does not know it is on its Nth attempt with the same
        failure will repeat itself; saying so is cheaper than another retry.
        """
        lines: list[str] = []
        files = ctx.get("working_files") or []
        if files:
            lines.append(f"- 本运行涉及的文件: {json.dumps(files, ensure_ascii=False)}")

        retry = ctx.get("retry") or {}
        state = ctx.get("state")
        attempts = (retry.get("attempts") or {}).get(state, 0)
        if attempts:
            lines.append(f"- 本状态（{state}）已尝试 {attempts} 次，"
                         f"上限 {retry.get('max_attempts')} 次")
        streaks = {k: v for k, v in (retry.get("failure_streaks") or {}).items() if v}
        if streaks:
            described = "；".join(f"{k} 连续 {v} 次" for k, v in sorted(streaks.items()))
            lines.append(f"- 同一失败类型已连续出现: {described}"
                         f"（达到 {retry.get('escalate_at')} 次将升级人工审批，"
                         f"不再自动重试）")
        return "\n".join(lines)

    def _render_changes(self, ctx: dict) -> str:
        """Observed on-disk changes and how they compare to the approved plan."""
        blocks: list[str] = []
        changes = ctx.get("actual_changes") or []
        if changes:
            # Snapshot-derived, so it is fact rather than the implementer's claim.
            blocks.append(
                "- 工作区实际变更（由 Harness 快照差分得出，非模型自述）: "
                f"{json.dumps(changes, ensure_ascii=False)}")
        deviation = ctx.get("plan_deviation") or {}
        if (deviation and not deviation.get("unplanned")
                and (deviation.get("missing") or deviation.get("extra"))):
            blocks.append(
                "- 与已批准方案的偏离（比对对象: 方案文件列表 vs 快照实际变更）: "
                f"漏改={json.dumps(deviation.get('missing'), ensure_ascii=False)} "
                f"多改={json.dumps(deviation.get('extra'), ensure_ascii=False)}。"
                "漏改不一定是错误（方案文件可能本就无需改动），请在结论中说明判断。")
        return "\n".join(blocks)

    def _validate(self, raw: dict | None) -> dict[str, Any]:
        """Validate the model's structured output, strictly.

        A payload that fails the schema is rejected, never repaired: every agent
        schema carries decision-bearing fields (verdict, build, tests), and
        filling those from defaults would turn a malformed reply into a
        confident wrong answer.

        The earlier "coercion" branch was removed because it could not succeed —
        every schema requires `next_step`, so `model_validate({})` always raised
        and the fallback merely *looked* permissive while always failing.
        """
        try:
            return self.output_schema.model_validate(
                raw if raw is not None else {}).model_dump()
        except ValidationError as e:
            raise AgentError(f"{self.role} 输出无法通过 schema 校验: {e}",
                             wire_code=WIRE_OUTPUT_MALFORMED) from e

    def _undeclared_tool(self, call: ToolCall, available: list[str], seen: int) -> dict:
        """Answer a call to a tool the model was never given.

        Refusing it used to abort the state on the first slip, which turns a
        recoverable mistake into a failed run. The refusal is fed back as a
        structured result instead — carrying the tools that *do* exist, which is
        the part a model actually needs to correct itself — and the budget bounds
        that. Past the budget the old path resumes unchanged.

        Every refusal is audited, including the ones that earn a retry: the call
        *was* refused, and the audit trail is what makes refusals countable.
        Recording only the last one would hide a model that tried three times,
        and would make the retry budget invisible to whoever reads the run later.

        Raises AgentError once the budget is spent.
        """
        msg = f"{self.role} 调用了未声明的工具 {call.name}"
        self.gateway.record_rejected(call.name, call.arguments, msg,
                                     error_code=CODE_UNKNOWN_TOOL)
        if seen > self.max_tool_retries:
            raise AgentError(msg)
        return {
            "ok": False,
            "error_code": CODE_UNKNOWN_TOOL,
            "message": msg,
            "available_tools": available,
            "hint": ("只能使用 available_tools 中列出的工具；其他名字的工具不存在。"
                     "请改用其中之一，或直接依据已有结果输出符合 Schema 的 JSON。"),
        }

    # ----------------------------------------------------------------------- run
    async def run(self, ctx: dict) -> dict[str, Any]:
        """Set principal, drive the tool loop, return the validated output.

        Loop contract:
          - model returns tool calls -> each is executed through the gateway
            (declared tools only) and the results are fed back; continue.
          - model returns structured output -> validate and return.
          - model returns neither -> one corrective hint, then fail after
            `max_rounds`.
        """
        # Seeded in the caller's dict *before* it is copied, so whoever handed us
        # the context can read back what was shown and what it cost. The copy
        # below would otherwise swallow it: `record_of` would create the record
        # in the copy, and the caller would read an empty one.
        self.record_of(ctx)
        ctx = dict(ctx)
        run = ctx.get("run") or {}
        ctx["agent"] = self.role
        # Built once and carried on **every** call, not only installed on the
        # server. The server's slot is one shared place, and a nested run — a
        # delegated child driven inside this state's tool call — re-sets it for
        # each of its own states. Reading the slot back afterwards attributes
        # this run's remaining calls to the child.
        principal = {
            "role": self.role, "agent": self.role, "run_id": str(run.get("id", "")),
            # Carried for the same reason `run_id` is: the gateway emits the
            # trace event for a tool call, and an event without a session cannot
            # be grouped with the other runs of the same invocation.
            "session_id": str(run.get("session_id") or ""),
            "read_roots": self._read_roots, "write_roots": self._write_roots,
            # The Harness derives this from the *approved* plan, so the file set
            # a write is checked against never comes from the model.
            "allowed_write_paths": ctx.get("plan_write_paths"),
        }
        self.gateway.set_principal(**principal)
        # What a caller that drives tools itself picks up — the mock brain reads
        # `ctx["gateway"]`, and the bound view means it names the principal
        # automatically instead of relying on the shared slot.
        ctx["gateway"] = self.gateway.bound(principal)

        tools = self._tool_specs(ctx)
        system = self.build_prompt(ctx, tools=tools)
        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": "请依据上述上下文完成本阶段任务，并输出符合输出要求的 JSON 对象。"},
        ]
        # Consecutive identical tool calls: re-issuing the exact same (name,
        # arguments) pair yields no new information, so the repeat is answered
        # with an error result instead of being executed again. Tracked across
        # rounds, because that is where the loop actually shows up.
        threshold = self.repeated_call_threshold
        last_sig: str | None = None
        repeats = 0
        # Undeclared-tool calls seen so far in this state. Budgeted per state,
        # not per round: a per-round budget would let a model slip once in every
        # round for as long as the loop runs.
        undeclared = 0
        # The model-interface failure of the most recent unproductive round, kept
        # so the eventual AgentError can say which one it was.
        last_wire: str | None = None
        # Metering, updated after every call rather than once at the end: a
        # provider error escaping mid-loop leaves the counts describing what was
        # actually spent, not what a previous run spent.
        usage_totals: dict[str, int] = {}
        usage_reported: set[str] = set()
        model_calls = 0
        latency_ms = 0
        record = self.record_of(ctx)
        record["usage"], record["model_calls"], record["latency_ms"] = {}, 0, 0

        for rnd in range(self.max_rounds):
            started = time.monotonic()
            result = await self.llm.complete(
                messages=messages, schema=self.output_schema,
                tools=tools or None, ctx=ctx)
            latency_ms += int((time.monotonic() - started) * 1000)
            model_calls += 1
            _accumulate_usage(usage_totals, usage_reported, result.usage)
            record["usage"] = {f: usage_totals[f] for f in usage_reported}
            record["model_calls"] = model_calls
            record["latency_ms"] = latency_ms
            if result.tool_calls:
                # The assistant turn goes in BEFORE any result does. A tool result
                # has to answer a `tool_call` in the preceding assistant message,
                # and a real provider rejects the whole conversation otherwise.
                # That ordering was incidental while an undeclared tool aborted the
                # state outright; now that a refusal is *answered*, it is
                # load-bearing. The provider-issued id is preserved on both sides
                # for the same reason.
                tool_call_dicts = [
                    {"id": tc.id or f"call_{rnd}_{i}", "name": tc.name,
                     "arguments": tc.arguments}
                    for i, tc in enumerate(result.tool_calls)]
                messages.append({"role": "assistant", "content": result.text or "",
                                 "tool_calls": tool_call_dicts})
                available = [t["name"] for t in tools]
                for i, tc in enumerate(result.tool_calls):
                    if tc.name not in available:
                        undeclared += 1
                        payload = self._undeclared_tool(tc, available, undeclared)
                    else:
                        sig = _call_signature(tc.name, tc.arguments)
                        if sig == last_sig:
                            repeats += 1
                        else:
                            last_sig, repeats = sig, 1
                        if threshold and repeats >= threshold:
                            payload = {
                                "ok": False,
                                # Same vocabulary as a refused undeclared tool: a
                                # stable code in its own field, and a sentence for
                                # a human. The code used to be a prefix inside the
                                # message, which left matching on prose as the only
                                # way to recognise the refusal — the audit trail's
                                # own rule is that classification never reads text.
                                "error_code": CODE_REPEATED_CALL,
                                "message": (f"{self.role} 重复调用了 {tc.name}"
                                            f"（与上一次的调用完全相同），已跳过重复执行"),
                                "hint": ("同一工具、同一参数的连续调用不会带来新信息。"
                                         "请更换参数，或直接依据已有结果输出符合 Schema 的 JSON。"),
                            }
                            self.gateway.record_rejected(tc.name, tc.arguments,
                                                         payload["message"],
                                                         error_code=CODE_REPEATED_CALL,
                                                         principal=principal)
                        else:
                            outcome = await self.gateway.call(tc.name, tc.arguments,
                                                              principal=principal)
                            payload = outcome.structured
                            if payload is None:
                                payload = {"ok": outcome.ok, "error": outcome.error}
                    # Tool specs allow up to 200k chars of output; uncapped, a
                    # couple of test runs would crowd out everything else.
                    messages.append({
                        "role": "tool", "tool_call_id": tool_call_dicts[i]["id"],
                        "name": tc.name,
                        "content": context.clip(
                            json.dumps(payload, ensure_ascii=False, default=str),
                            self.tool_result_budget)})
                continue
            if result.output is not None:
                try:
                    return self._validate(result.output)
                except AgentError as e:
                    # A schema failure is an unproductive round, not the end of the
                    # state. The validator's own complaint names the field and the
                    # reason, which is the single most actionable thing we can hand
                    # back — and it used to be thrown away in favour of "follow the
                    # schema", a hint that tells the model nothing it did not already
                    # know. It costs one round, bounded by `max_rounds` like every
                    # other bad answer, so no new budget is needed.
                    last_wire = WIRE_OUTPUT_MALFORMED
                    messages.append({"role": "user",
                                     "content": self._schema_hint(e, result.output)})
                    continue
            # "The model answered badly" and "the model was cut off mid-answer"
            # need opposite responses — re-ask versus give it more room — and
            # used to read identically here, which is why every real-provider
            # failure in the record said only "no structured output".
            last_wire = (WIRE_OUTPUT_TRUNCATED if result.truncated
                         else WIRE_OUTPUT_MALFORMED)
            messages.append({"role": "user", "content": self._non_answer_hint(result)})
        raise AgentError(
            f"{self.role} 在 {self.max_rounds} 轮内未产出结构化输出"
            + ("（回复被输出配额截断）" if last_wire == WIRE_OUTPUT_TRUNCATED
               else "（回复不符合 Schema）"),
            wire_code=last_wire or WIRE_OUTPUT_MALFORMED)

    def _schema_hint(self, exc: AgentError, output: dict) -> str:
        """Hand the validator's own complaint back.

        The error text carries the field path and what was wrong with it; the
        model's last output is described by its top-level keys, so it can see what
        it actually produced without us echoing the whole payload back. Clipped,
        because a pydantic error over a nested object can be longer than the thing
        it is describing.
        """
        detail = context.clip(str(exc), 2000)
        keys = sorted(output) if isinstance(output, dict) else []
        return ("你的 JSON 能被解析，但没有通过 Schema 校验，因此没有被采纳。\n\n"
                f"校验器原话：\n{detail}\n\n"
                f"你上次输出的顶层字段是: {keys[:20]}\n"
                "请**只修正上面指出的问题**，然后重新输出完整的 JSON 对象；"
                "不要附加解释，也不要改动已经正确的字段。")

    def _non_answer_hint(self, result) -> str:
        """What to say when a completion produced nothing usable.

        A truncated answer gets a different instruction from a malformed one: the
        first is a budget problem the model cannot fix by trying harder, so telling
        it to "follow the schema" wastes the round.
        """
        if result.truncated:
            return (
                f"你的上一条回复因输出配额不足被截断（finish_reason={result.finish_reason}），"
                f"JSON 不完整，无法解析。请直接输出完整且精简的 JSON 对象本体，不要任何前言、"
                f"解释或 Markdown；若仍被截断，需提高 llm.max_tokens。")
        return ("你的回复中没有可解析的结构化输出。请严格按照给定 JSON Schema 输出一个 "
                "JSON 对象，不要附加任何解释或 Markdown。")
