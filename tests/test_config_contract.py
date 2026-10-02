"""The config's contract with its readers, checked in both directions.

Both directions have already gone wrong here, in opposite ways:

  * `run_lease_seconds` was read by the orchestrator and never declared, so every
    `advance()` raised AttributeError — and 200-odd tests passed, because the
    tests that would have noticed all go through `advance()` themselves and so
    were broken by the same fault;
  * `max_agent_rounds` and `default_agent_timeout` were declared and never read,
    so setting either one silently did nothing.

One check catches one of those, so both are here.

Detection is anchored on *who holds the config*, not on the section name. That
distinction is not academic: `self.llm.complete(...)` is an adapter, not
`LLMConfig`, and a detector that said "`X.llm.FIELD` can only mean the config"
would report it as a bad reference. The write-up of that near-miss is in the
self-checks at the bottom, which exist so the next person can see what these
guards actually catch — a guard nobody has watched fail is not yet a guard.

What these checks cannot see: a reference written through `getattr`, or from
`tests/`. The first would slip past; the second is deliberate, because a field
only a test reads is still a field production never consults.
"""
from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

from wfos.config import HarnessConfig, LLMConfig

PACKAGE = Path(__file__).resolve().parents[1] / "wfos"
CONFIG_MODULE = "config.py"

# The sections reached through an attribute chain on a config object.
SECTIONS: dict[str, type] = {"harness": HarnessConfig, "llm": LLMConfig}

# Names that hold the config object. Anchoring on these is what keeps
# `self.llm.complete` — an adapter — out of the results.
CONFIG_HOLDERS = frozenset({"cfg", "config", "_config"})


# ------------------------------------------------------------------- detectors
def _holds_config(node: ast.AST) -> bool:
    """Whether `node` is one of the names that hold an `AppConfig`."""
    if isinstance(node, ast.Name):
        return node.id in CONFIG_HOLDERS
    if isinstance(node, ast.Attribute):
        return node.attr in CONFIG_HOLDERS
    return False


def _section_references_in(source: str) -> list[tuple[str, str]]:
    """`(section, member)` for every `HOLDER.<section>.<member>` in one source string.

    Members include methods, not just fields: `cfg.harness.shell_env(root)` is a
    legitimate read, and a check that only knew about fields would call it a
    typo.
    """
    found: list[tuple[str, str]] = []
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr in SECTIONS
                and _holds_config(node.value.value)):
            found.append((node.value.attr, node.attr))
    return found


def _harness_reads_in(source: str, filename: str) -> set[str]:
    """HarnessConfig member names read in one source string.

    Two shapes, because they are the only two that cannot mean anything else:
    `HOLDER.harness.FIELD`, and `self.FIELD` inside config.py — where `self` is a
    HarnessConfig method reading its own field, as `allows` and `shell_env` do.
    """
    found: set[str] = set()
    in_config = filename == CONFIG_MODULE
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Attribute):
            continue
        on_harness = (isinstance(node.value, ast.Attribute)
                      and node.value.attr == "harness"
                      and _holds_config(node.value.value))
        own_field = (in_config and isinstance(node.value, ast.Name)
                     and node.value.id == "self")
        if on_harness or own_field:
            found.add(node.attr)
    return found


def _members(section: str) -> set[str]:
    """Everything a config section offers: its fields and its methods."""
    cls = SECTIONS[section]
    return ({f.name for f in dataclasses.fields(cls)}
            | {name for name in dir(cls) if not name.startswith("__")})


def _package_sources():
    for path in sorted(PACKAGE.rglob("*.py")):
        yield path, path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------- checks
def test_every_referenced_config_member_exists():
    """The `run_lease_seconds` guard: a reader naming something nobody declared."""
    seen = [(section, member, path.name)
            for path, source in _package_sources()
            for section, member in _section_references_in(source)]
    unknown = [entry for entry in seen if entry[1] not in _members(entry[0])]
    assert not unknown, (
        "这些读取点引用了配置里不存在的成员（字段会 AttributeError，方法会 TypeError）：\n  "
        + "\n  ".join(f"{section}.{member}  读取自 {name}"
                      for section, member, name in unknown))


def test_every_harness_field_is_read_somewhere():
    """The other direction: a knob nothing consults."""
    read = {field for path, source in _package_sources()
            for field in _harness_reads_in(source, path.name)}
    unread = [f.name for f in dataclasses.fields(HarnessConfig) if f.name not in read]
    assert not unread, (
        "这些 HarnessConfig 字段没有任何读取点 —— 设置它们不会有任何效果，"
        "而一个静默无效的旋钮比没有旋钮更糟：\n  " + "\n  ".join(unread))


# ------------------------------------------------------------------ self-checks
def test_the_reference_detector_finds_a_member_that_does_not_exist():
    assert _section_references_in("x = cfg.harness.run_lease_seconds_typo\n") == [
        ("harness", "run_lease_seconds_typo")]
    assert _section_references_in("x = self.cfg.llm.model_typo\n") == [("llm", "model_typo")]


def test_the_reference_detector_ignores_chains_that_are_not_config():
    """Three shapes that look like config reads and are not.

    `self.llm.complete` is the one that actually caught this detector out: `llm`
    is the section name *and* the attribute an agent holds its adapter in, so
    anchoring on the section name alone reported a real method call as a bad
    reference.
    """
    assert _section_references_in("x = self.llm.complete(m)\n") == []
    assert _section_references_in("x = harness.repo.get_run(1)\n") == []
    assert _section_references_in("x = context.fit({})\n") == []


def test_a_config_method_is_not_mistaken_for_a_typo():
    """`shell_env` and `allows` are read off the config and are not fields."""
    assert _section_references_in("x = cfg.harness.shell_env(root)\n") == [
        ("harness", "shell_env")]
    assert "shell_env" in _members("harness") and "allows" in _members("harness")


def test_the_reader_detector_sees_both_shapes_and_nothing_else():
    assert _harness_reads_in("x = self.cfg.harness.run_lease_seconds\n", "run.py") == {
        "run_lease_seconds"}
    assert _harness_reads_in("x = self.safe_commands\n", CONFIG_MODULE) == {
        "safe_commands"}
    # `self.x` outside config.py is some other object's attribute; counting it
    # would let a dead field ride on an unrelated name.
    assert _harness_reads_in("x = self.safe_commands\n", "elsewhere.py") == set()
    assert _harness_reads_in("x = self.llm.complete(m)\n", "run.py") == set()
