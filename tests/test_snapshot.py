"""Workspace snapshot & diff — change attribution.

The Harness must be able to say what changed on disk without asking the model,
so these tests pin the fingerprinting rules, the ignore list, the large-tree
fallback, and the honest degradation when a diff spans a mode switch.
"""
from __future__ import annotations

from pathlib import Path

from wfos.mcp.snapshot import affected_paths, capture_tree, diff_snapshots


def _project(tmp_path: Path) -> Path:
    d = tmp_path / "proj"
    (d / "pkg").mkdir(parents=True)
    (d / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (d / "pkg" / "mod.py").write_text("X = 1\n", encoding="utf-8")
    return d


# ------------------------------------------------------------------ capture
def test_capture_hashes_and_uses_relative_posix_paths(tmp_path):
    d = _project(tmp_path)
    snap = capture_tree(d)
    assert snap["mode"] == "hash"
    assert snap["truncated"] is False
    assert set(snap["files"]) == {"app.py", "pkg/mod.py"}
    assert all(len(tok) == 64 for tok in snap["files"].values())   # sha256 hex


def test_capture_ignores_vcs_caches_and_bytecode(tmp_path):
    d = _project(tmp_path)
    for name in ("__pycache__", ".git", ".pytest_cache", ".venv", "node_modules"):
        (d / name).mkdir()
        (d / name / "junk.py").write_text("noise", encoding="utf-8")
    (d / "pkg" / "mod.pyc").write_text("bytecode", encoding="utf-8")
    snap = capture_tree(d)
    assert set(snap["files"]) == {"app.py", "pkg/mod.py"}


def test_capture_detects_content_change_not_just_mtime(tmp_path):
    d = _project(tmp_path)
    before = capture_tree(d)
    (d / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    after = capture_tree(d)
    assert before["files"]["app.py"] != after["files"]["app.py"]


def test_capture_falls_back_to_manifest_above_max_files(tmp_path):
    d = tmp_path / "proj"
    d.mkdir()
    for i in range(5):
        (d / f"f{i}.py").write_text(f"i = {i}\n", encoding="utf-8")
    snap = capture_tree(d, max_files=3)
    assert snap["mode"] == "manifest"
    assert snap["truncated"] is True
    assert len(snap["files"]) == 3
    assert all(":" in tok for tok in snap["files"].values())       # size:mtime


# --------------------------------------------------------------------- diff
def test_diff_buckets_created_modified_deleted(tmp_path):
    d = _project(tmp_path)
    before = capture_tree(d)
    (d / "app.py").write_text("VALUE = 99\n", encoding="utf-8")     # modified
    (d / "new.py").write_text("NEW = 1\n", encoding="utf-8")         # created
    (d / "pkg" / "mod.py").unlink()                                  # deleted
    diff = diff_snapshots(before, capture_tree(d))

    assert diff["created"] == ["new.py"]
    assert diff["modified"] == ["app.py"]
    assert diff["deleted"] == ["pkg/mod.py"]
    assert diff["total"] == 3 and diff["changed"] is True
    assert set(affected_paths(diff)) == {"new.py", "app.py", "pkg/mod.py"}


def test_diff_of_identical_snapshots_reports_no_change(tmp_path):
    d = _project(tmp_path)
    before = capture_tree(d)
    diff = diff_snapshots(before, capture_tree(d))
    assert diff["changed"] is False and diff["total"] == 0
    assert diff["summary"] == "无文件变更"
    assert affected_paths(diff) == []


def test_diff_across_a_mode_switch_does_not_invent_modifications(tmp_path):
    """Hashes and size+mtime tokens are incomparable — a tree that crossed the
    cap mid-diff must degrade to path-level accuracy, not claim everything was
    rewritten."""
    d = tmp_path / "proj"
    d.mkdir()
    for i in range(4):
        (d / f"f{i}.py").write_text(f"i = {i}\n", encoding="utf-8")
    before = capture_tree(d, max_files=10)          # hash mode: all 4 files
    (d / "extra.py").write_text("extra\n", encoding="utf-8")
    after = capture_tree(d, max_files=2)            # manifest mode: capped

    diff = diff_snapshots(before, after)
    assert diff["mode_changed"] is True
    assert diff["modified"] == []                   # NOT "every shared path"
    assert "extra.py" in diff["created"]            # path-level facts survive
    assert "快照模式切换" in diff["summary"]


def test_diff_truncates_large_path_lists_but_keeps_the_count(tmp_path):
    d = tmp_path / "proj"
    d.mkdir()
    for i in range(10):
        (d / f"f{i}.py").write_text(f"i = {i}\n", encoding="utf-8")
    diff = diff_snapshots({}, capture_tree(d), limit=4)
    assert len(diff["created"]) == 4
    assert diff["total"] == 10
    assert diff["truncated"] is True
    assert diff["summary"] == "新增 10"


def test_diff_handles_missing_snapshots():
    diff = diff_snapshots(None, None)
    assert diff["changed"] is False
    assert affected_paths(diff) == []
