"""Two persistent state machines (Feature dev, Bugfix diagnosis).

The model may only *suggest* a next state; the Harness validates every
transition against the machine and its preconditions before persisting it.
"""
from __future__ import annotations

TERMINAL = ("completed", "failed", "cancelled")

FEATURE_STATES = [
    "req_capture", "project_check", "feature_design", "tech_design",
    "risk_assess", "conditional_approval", "implement",
    "build_test", "regression_verify", "knowledge_distill",
] + list(TERMINAL)

BUGFIX_STATES = [
    "issue_capture", "history_search", "evidence_collect", "root_cause",
    "confidence_assess", "fix_plan", "risk_assess", "conditional_approval",
    "implement", "verify_regression", "knowledge_distill",
] + list(TERMINAL)

FEATURE_TRANSITIONS = {
    "req_capture":          {"project_check", "failed"},
    "project_check":        {"feature_design", "failed"},
    "feature_design":       {"tech_design", "failed"},
    "tech_design":          {"risk_assess", "failed"},
    "risk_assess":          {"conditional_approval", "tech_design", "failed"},
    "conditional_approval": {"implement", "failed", "cancelled"},
    "implement":            {"build_test", "failed"},
    "build_test":           {"regression_verify", "implement", "failed"},
    "regression_verify":    {"knowledge_distill", "implement", "failed"},
    "knowledge_distill":    {"completed", "failed"},
}

BUGFIX_TRANSITIONS = {
    "issue_capture":        {"history_search", "failed"},
    "history_search":       {"evidence_collect", "failed"},
    "evidence_collect":     {"root_cause", "failed"},
    "root_cause":           {"confidence_assess", "failed"},
    "confidence_assess":    {"fix_plan", "evidence_collect", "failed"},
    "fix_plan":             {"risk_assess", "failed"},
    "risk_assess":          {"conditional_approval", "fix_plan", "failed"},
    "conditional_approval": {"implement", "failed", "cancelled"},
    "implement":            {"verify_regression", "failed"},
    "verify_regression":    {"knowledge_distill", "root_cause", "implement", "failed"},
    "knowledge_distill":    {"completed", "failed"},
}

# Sentinel pseudo-state used to signal the Harness to spawn a child Bugfix run.
SPAWN_BUGFIX = "spawn_bugfix"

MACHINES = {
    "feature": {"states": set(FEATURE_STATES), "transitions": FEATURE_TRANSITIONS,
                "initial": "req_capture", "completed": "completed"},
    "bugfix":  {"states": set(BUGFIX_STATES), "transitions": BUGFIX_TRANSITIONS,
                "initial": "issue_capture", "completed": "completed"},
}


def is_terminal(state: str) -> bool:
    return state in TERMINAL


def is_valid_state(kind: str, state: str) -> bool:
    return state in MACHINES[kind]["states"]


def initial_state(kind: str) -> str:
    return MACHINES[kind]["initial"]


def allowed_transitions(kind: str, state: str) -> set[str]:
    return MACHINES[kind]["transitions"].get(state, set())


def validate_transition(kind: str, from_state: str, to_state: str) -> tuple[bool, str]:
    """Validate a raw transition against the state machine topology.

    `to_state == from_state` is allowed only when the state is a re-entry
    (retry with new evidence); the Harness tracks that via rerun flags.
    """
    machine = MACHINES.get(kind)
    if machine is None:
        return False, f"未知流程类型: {kind}"
    if from_state not in machine["states"]:
        return False, f"未知状态: {from_state}"
    if to_state == from_state:
        return True, "同状态重入（携带新证据重试）"
    allowed = machine["transitions"].get(from_state, set())
    if to_state in allowed:
        return True, "合法转换"
    return False, f"状态 {from_state} → {to_state} 不是合法转换（允许: {sorted(allowed)}）"
