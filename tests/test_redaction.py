"""Secret hygiene: what a subprocess may see, and what reaches the record.

Two boundaries are pinned here. A build/test subprocess must not inherit the
harness's credentials; and content produced by a model or a subprocess must not
land in the database carrying secrets. The third, quieter rule is that the
harness's own fingerprints are *not* redacted — they are sha256 digests, which
the long-hex pattern would otherwise eat, breaking the resume contract.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from wfos import redact
from wfos.storage.repo import Repo


def _call(gw, tool, args, *, role, run_id="redact-run", write_roots=()):
    root = str(gw._server._config.project_root)
    gw.set_principal(role=role, agent=role, run_id=run_id,
                     read_roots=[root], write_roots=list(write_roots) or [root])
    return asyncio.run(gw.call(tool, args))


# ----------------------------------------------------------------- the patterns
@pytest.mark.parametrize("text", [
    "Authorization: Bearer abc.def-ghi",
    "token sk-abcdefghijklmnop1234",
    "api_key = supersecretvalue",
    "password: hunter2",
    "deadbeefdeadbeefdeadbeefdeadbeef",
    "AKIAIOSFODNN7EXAMPLE",
])
def test_secret_shapes_are_redacted(text):
    assert redact.REDACTED in redact.redact(text)


@pytest.mark.parametrize("text", [
    "the value is deadbeef",                          # 8 hex, under the 32 floor
    "abcdef0123456789abcdef012345678",                # 31 hex, one short
    "token",                                          # a bare word, no value
    "sk-short",                                       # too short for the sk- form
    "PASS: compute(1)== True",
    "",
])
def test_ordinary_text_is_left_alone(text):
    assert redact.redact(text) == text


def test_redaction_is_idempotent():
    once = redact.redact("api_key=abc123")
    assert redact.redact(once) == once


def test_redact_structure_confines_each_match_to_its_own_string():
    """Redacting *serialized* text is unsafe: the value patterns end in `\\S+`,
    which is greedy enough to eat the closing quote and the comma after it —
    turning the document into invalid JSON that reads back as its default."""
    original = {"args": {"content": "password: hunter2"}, "path": "app.py"}
    # The whole `key: value` pair is replaced, not just the value.
    assert redact.redact_structure(original) == {
        "args": {"content": "[REDACTED]"}, "path": "app.py"}


def test_redacting_serialized_text_really_would_corrupt_the_document():
    """Pins why the repo redacts structures instead of serialized JSON."""
    blob = json.dumps({"content": "password: hunter2", "path": "app.py"})
    with pytest.raises(json.JSONDecodeError):
        json.loads(redact.redact(blob))


def test_redacted_json_columns_stay_parseable(app):
    """The regression guard: a secret-bearing argument must come back as valid
    JSON, not as the column's default."""
    repo = app["repo"]
    repo.log_tool_call("parse-run", "implementer", "workspace.write",
                       {"content": "password: hunter2", "path": "app.py"}, True)
    row = repo.tool_calls("parse-run")[-1]
    parsed = json.loads(row["args_json"])
    assert parsed == {"content": "[REDACTED]", "path": "app.py"}


# ------------------------------------------------------- persisted content
def test_evidence_is_redacted_on_the_way_in(app):
    repo = app["repo"]
    repo.add_evidence("redact-run", "log", "https://x.test/?token=abc123",
                      "found api_key=LEAKEDVALUE here")
    row = repo.list_evidence("redact-run")[-1]
    assert "LEAKEDVALUE" not in row["content"]
    assert "[REDACTED]" in row["content"]
    assert "abc123" not in row["source"]


def test_tool_call_args_and_error_are_redacted_but_paths_are_not(app):
    repo = app["repo"]
    repo.log_tool_call("redact-run", "implementer", "workspace.write",
                       {"content": "secret = leaky", "path": "app.py"}, False,
                       "failed with token=abc123def456",
                       affected_paths=["app.py"], diff_summary="修改 1")
    row = repo.tool_calls("redact-run")[-1]
    assert "leaky" not in row["args_json"]
    assert "abc123def456" not in (row["error"] or "")
    # Attribution must stay exact: redacting a path would corrupt the change set.
    assert json.loads(row["affected_paths"]) == ["app.py"]
    assert row["diff_summary"] == "修改 1"


def test_step_output_and_error_are_redacted(app):
    repo = app["repo"]
    repo.add_step("redact-run", "req_capture", "investigator",
                  {"findings": ["saw api_key=NESTEDSECRET"]},
                  input_json={"note": "Bearer tok.en"}, error="boom secret=xyz")
    row = repo.get_step("redact-run", "req_capture")
    assert "NESTEDSECRET" not in json.dumps(row["output_json"])
    assert "tok.en" not in json.dumps(row["input_json"])
    assert "xyz" not in (row["error"] or "")


def test_run_error_is_redacted(app):
    repo = app["repo"]
    run = repo.create_run("feature", "t", "d")
    repo.update_run(run["id"], error="provider said api_key=OOPS")
    assert "OOPS" not in (repo.get_run(run["id"])["error"] or "")


def test_redaction_can_be_switched_off(app, tmp_path):
    repo = Repo(tmp_path / "plain.db", redact=False)
    repo.add_evidence("r", "log", "s", "api_key=KEPT")
    assert "KEPT" in repo.list_evidence("r")[-1]["content"]


# ------------------------------- the harness's own fingerprints must survive
def test_fingerprints_are_not_redacted(app):
    """sha256 digests are 64 hex chars — exactly what the long-hex pattern
    matches. Redacting them would silently break the resume contract."""
    h, repo = app["harness"], app["repo"]
    run = h.create_run("新增一个模块", kind="feature")
    repo.update_run(run["id"], payload={
        "plan": {"summary": "s", "files": [{"path": "app.py", "action": "modify"}]}})
    h._record_fingerprint(run["id"])

    stored = repo.get_run(run["id"])
    assert stored["identity_hash"] and len(stored["identity_hash"]) == 64
    assert "[REDACTED]" not in stored["identity_hash"]
    digest = stored["key_files"]["app.py"]
    assert digest and len(digest) == 64 and "[REDACTED]" not in digest

    # And it still works: the contract validates rather than reporting drift.
    assert h.resume_status(stored)[0] == "full-valid"


# ------------------------------------------------------ the subprocess boundary
def test_build_and_test_subprocesses_do_not_inherit_credentials(app, monkeypatch):
    monkeypatch.setenv("WFOS_TEST_API_KEY", "LEAKEDVALUE")
    (app["sandbox"] / "check.py").write_text(
        'import os\n'
        'print("PASS: probe")\n'
        'print("api_key=%s" % os.environ.get("WFOS_TEST_API_KEY"))\n',
        encoding="utf-8")

    out = _call(app["gateway"], "test.run", {"command": "python check.py"},
                role="verifier")
    assert out.ok, out.error
    assert "LEAKEDVALUE" not in out.structured["output"]
    assert "api_key=None" in out.structured["output"]      # the var is simply absent


def test_subprocess_env_keeps_what_commands_need(app):
    env = app["cfg"].harness.shell_env(app["sandbox"])
    assert env["PATH"]                                     # nothing runs without it
    assert env["PWD"] == str(app["sandbox"])                # not the harness's cwd
    assert "PYTHONPATH" in env or "PYTHONPATH" not in __import__("os").environ
    assert "WFOS_PROVIDER" not in env


def test_subprocess_env_allowlist_is_configurable(app):
    app["cfg"].harness.shell_env_allowlist = ["PATH", "MY_BUILD_FLAG"]
    import os
    os.environ["MY_BUILD_FLAG"] = "on"
    try:
        env = app["cfg"].harness.shell_env(app["sandbox"])
        assert env["MY_BUILD_FLAG"] == "on"
        assert "SYSTEMROOT" not in env
    finally:
        del os.environ["MY_BUILD_FLAG"]


# ------------------------------------------------------------------ end to end
def test_secrets_in_tool_output_do_not_reach_the_record(app):
    """A command that prints a secret must not leave it in the audit trail."""
    (app["sandbox"] / "check.py").write_text(
        'print("PASS: compute(1)")\n'
        'print("leaked api_key=LEAKEDVALUE")\n', encoding="utf-8")
    h, repo = app["harness"], app["repo"]
    run = h.create_run("新增一个用户模块，改 app.py", kind="feature")
    asyncio.run(h.advance(run["id"]))

    rows = repo.list_evidence(run["id"])
    assert rows, "运行应当留下证据"
    assert all("LEAKEDVALUE" not in (r["content"] or "") for r in rows)
    assert any("[REDACTED]" in (r["content"] or "") for r in rows)
