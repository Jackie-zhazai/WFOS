"""State machine topology and transition validation."""
from __future__ import annotations

from wfos.harness import statemachine as sm


def test_legal_transitions_feature():
    assert sm.validate_transition("feature", "req_capture", "project_check")[0]
    assert sm.validate_transition("feature", "tech_design", "risk_assess")[0]
    assert sm.validate_transition("feature", "build_test", "regression_verify")[0]
    assert sm.validate_transition("feature", "regression_verify", "knowledge_distill")[0]
    assert sm.validate_transition("feature", "knowledge_distill", "completed")[0]


def test_illegal_transitions_feature():
    ok, msg = sm.validate_transition("feature", "req_capture", "completed")
    assert not ok and "不是合法转换" in msg
    ok, _ = sm.validate_transition("feature", "implement", "completed")
    assert not ok


def test_legal_transitions_bugfix():
    assert sm.validate_transition("bugfix", "issue_capture", "history_search")[0]
    assert sm.validate_transition("bugfix", "confidence_assess", "fix_plan")[0]
    assert sm.validate_transition("bugfix", "verify_regression", "root_cause")[0]
    assert sm.validate_transition("bugfix", "verify_regression", "knowledge_distill")[0]


def test_same_state_reentry():
    ok, msg = sm.validate_transition("bugfix", "verify_regression", "verify_regression")
    assert ok and "重入" in msg


def test_unknown_kind_and_state():
    assert not sm.validate_transition("nope", "a", "b")[0]
    assert not sm.validate_transition("feature", "nope", "b")[0]


def test_terminal():
    assert sm.is_terminal("completed")
    assert sm.is_terminal("failed")
    assert sm.is_terminal("cancelled")
    assert not sm.is_terminal("implement")
