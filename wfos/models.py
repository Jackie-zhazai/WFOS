"""Structured I/O schemas shared by agents, harness and storage.

Agents must emit pydantic-validated structured outputs. Execution results are
decided from these structured fields — never from natural-language keywords.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# `assistant` is the interactive entry's role. It exists so the policy layer can
# name a principal that may read, write *and* run the build/tests — the union no
# single state-machine role has, and one that would be wrong to give any of them:
# an implementer that could run its own tests is not the same separation as an
# independent verifier. It is deliberately **not** in `AGENT_CLASSES`.
Role = Literal["investigator", "architect", "implementer", "verifier", "curator",
               "assistant", "harness"]
RunKind = Literal["feature", "bugfix", "chat"]
RunStatus = Literal[
    "created", "running", "waiting_approval", "waiting_child", "paused",
    "completed", "failed", "cancelled",
]

# Statuses that mean the run is over. A *status* axis, distinct from the state
# machine's terminal *states* — the strings coincide today, but they answer
# different questions and reading one through the other is how they drift apart.
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})

# Every provider spells "I ran out of output budget" differently; all of them
# mean the answer was cut off rather than finished. Compared case-insensitively.
TRUNCATION_REASONS = frozenset({"length", "max_tokens", "max_output_tokens",
                                "maxtokens"})

# Where a run came from. This is the boundary that keeps an artificial run out of
# production: a `benchmark` run's memory and skills are scoped to `benchmark`, so
# running a task into a failure class cannot write into a real run's prompt.
ORIGIN_INTERACTIVE = "interactive"
ORIGIN_TASK = "task"
ORIGIN_BENCHMARK = "benchmark"
ORIGINS = (ORIGIN_INTERACTIVE, ORIGIN_TASK, ORIGIN_BENCHMARK)

# A skill's lifecycle. `live` may be injected into a prompt; `candidate` may not,
# whatever produced it, and `skills_for` is what enforces that.
#
# There is deliberately no `verified` rung. The spec's promotion path has one,
# but nothing produces it yet, and a name with no producer is a name a reader
# will find in the vocabulary and never in the data. It arrives with the
# promotion gate (P6) that writes it, not before.
SKILL_CANDIDATE = "candidate"
SKILL_LIVE = "live"
SKILL_STATUSES = (SKILL_CANDIDATE, SKILL_LIVE)

# What makes a skill **production** — the one definition, so a loader cannot
# answer the question differently from `skills_for`.
#
# Both halves are load-bearing and neither is redundant:
#
#   `status`  — a `candidate` was produced by an environment nobody vouched for.
#               Injecting it would be an artificial run writing into a real
#               prompt, which is the whole reason the two-tier skill store exists.
#   `origin`  — a `task` or `benchmark` run's output is evidence about a suite,
#               not a decision about production. Only an interactive run's
#               procedure has a human behind it.
#
# `superseded` is the third half, but it is a *pointer* rather than a property:
# `list_skills` and `skills_for` both apply it separately, because reading history
# is a legitimate thing to want and it is not the same question as "what is live".
PRODUCTION_SKILL_STATUS = SKILL_LIVE
PRODUCTION_SKILL_ORIGIN = ORIGIN_INTERACTIVE


class NextStep(BaseModel):
    """The model may only *suggest* the next state; the Harness validates and decides."""
    suggested_state: str = Field(
        description=("下一状态名。必须取自运行上下文中"
                     "『本状态允许建议的下一状态』列出的取值，"
                     "其他取值会被 Harness 拒绝并使运行失败"))
    reason: str = ""


class EvidenceItem(BaseModel):
    kind: Literal["log", "code", "config", "test_output", "user", "git", "build", "wiki", "other"] = "other"
    source: str = ""
    content: str = ""
    confidence: Literal["high", "medium", "low"] = "medium"


class ToolCall(BaseModel):
    id: str | None = None              # provider-issued tool-call id (round-trips)
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ModelUsage(BaseModel):
    """Token counts one model call reported.

    Every field is `int | None` deliberately: `None` means the provider did not
    report it, which is not the same as zero. Defaulting an unreported count to 0
    would make "we were not told" indistinguishable from "it was free" in every
    aggregate computed from it — and `0` is a real, reportable value.
    """
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    # Reasoning tokens are reported *inside* `output_tokens` by providers that
    # expose them, so this is a breakdown rather than an addition. Recorded
    # because it is the difference between needing output room and needing
    # output room *plus* room to think — and because a provider that says how much
    # it reasoned should not have to be guessed at from the size of the answer.
    reasoning_tokens: int | None = None


class ModelResult(BaseModel):
    """Unified return of an LLM completion (provider-independent)."""
    text: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    output: dict[str, Any] | None = None
    # None when the provider reported no usage at all (mock/scripted adapters
    # have no provider, so this is the normal case for them).
    usage: ModelUsage | None = None
    # The provider's own reason for stopping, verbatim (`length` / `max_tokens` /
    # `MAX_TOKENS` / `end_turn` / `tool_use` / …). Recorded because it is the one
    # signal that separates "the model answered badly" from "the model was cut
    # off mid-answer" — and the two need opposite responses. Its absence was why
    # every real-provider failure read as the same undiagnosable message.
    finish_reason: str | None = None

    @property
    def truncated(self) -> bool:
        """Whether the provider stopped because it ran out of output budget."""
        return (self.finish_reason or "").lower() in TRUNCATION_REASONS


# ---------------------------------------------------------------------------
# Agent outputs
# ---------------------------------------------------------------------------

class InvestigatorOutput(BaseModel):
    findings: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    project_context: dict[str, Any] = Field(default_factory=dict)
    next_step: NextStep


class Operation(BaseModel):
    kind: Literal[
        "write", "create", "delete", "db_migrate", "prod_write", "external",
        "permission", "secret", "irreversible", "core_modify",
    ] = "write"
    path: str = ""
    reason: str = ""


class PlanFile(BaseModel):
    path: str
    action: Literal["create", "modify", "delete"] = "modify"
    # patch payload
    before: str | None = None
    after: str | None = None
    content: str | None = None
    detail: str = ""


class ArchitectOutput(BaseModel):
    summary: str = ""
    files: list[PlanFile] = Field(default_factory=list)
    operations: list[Operation] = Field(default_factory=list)
    impact: list[str] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high"] = "low"
    verification_plan: list[str] = Field(default_factory=list)
    design_notes: str = ""
    confidence: Literal["high", "medium", "low"] = "high"   # for root-cause / fix-plan stages
    root_cause: str = ""
    causal_chain: list[str] = Field(default_factory=list)
    rollback: str = ""
    next_step: NextStep


class Change(BaseModel):
    file: str
    action: Literal["create", "modify", "delete"] = "modify"
    detail: str = ""


class ImplementerOutput(BaseModel):
    changes: list[Change] = Field(default_factory=list)
    failed: list[str] = Field(default_factory=list)
    next_step: NextStep


class ChatOutput(BaseModel):
    """One turn's answer, after whatever tool calls it needed.

    There is no `next_step`: a state-machine agent suggests where the workflow
    goes, and a conversation does not have one. The reply is the whole product —
    the tools it ran on the way are in the trace, not in this object.
    """
    reply: str


class BuildResult(BaseModel):
    ok: bool = False
    errors: list[str] = Field(default_factory=list)
    files_checked: list[str] = Field(default_factory=list)


class TestCase(BaseModel):
    name: str
    ok: bool
    detail: str = ""


class TestResult(BaseModel):
    ok: bool = False
    exit_code: int = -1
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    command: str = ""
    cases: list[TestCase] = Field(default_factory=list)
    output: str = ""


class RegressionCheck(BaseModel):
    module: str
    ok: bool
    detail: str = ""


class VerifierOutput(BaseModel):
    build: BuildResult = Field(default_factory=BuildResult)
    tests: TestResult = Field(default_factory=TestResult)
    regression: list[RegressionCheck] = Field(default_factory=list)
    verdict: Literal["pass", "fail"] = "fail"
    evidence: list[EvidenceItem] = Field(default_factory=list)
    next_step: NextStep

    @property
    def regression_ok(self) -> bool:
        return all(r.ok for r in self.regression)


class KnowledgeCandidate(BaseModel):
    title: str
    content: str
    tags: list[str] = Field(default_factory=list)
    source_run: str = ""
    evidence_refs: list[str] = Field(default_factory=list)


class CuratorOutput(BaseModel):
    knowledge: list[KnowledgeCandidate] = Field(default_factory=list)
    next_step: NextStep


# ---------------------------------------------------------------------------
# Approval
# ---------------------------------------------------------------------------

class ApprovalAction(BaseModel):
    action: str = ""                 # human-readable, e.g. "delete app.py"
    scope: str = ""                  # path / module affected
    risk_level: Literal["low", "medium", "high"] = "low"
    reason: str = ""
    required_by: list[Literal["policy", "risk", "confidence"]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Tool gate request/response (MCP layer)
# ---------------------------------------------------------------------------

class ToolOutcome(BaseModel):
    tool: str
    ok: bool
    error: str | None = None
    content: list[dict[str, Any]] = Field(default_factory=list)
    structured: dict[str, Any] | None = None
    side_effects: bool = False
