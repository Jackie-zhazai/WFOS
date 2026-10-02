"""Workspace snapshot & diff — change attribution the model cannot fake.

A model's self-reported change list (`ImplementerOutput.changes`) is a claim,
not evidence. Before and after every side-effecting tool call the server
fingerprints the workspace and diffs the two snapshots, so the audit row
records what actually changed on disk.

Files are content-hashed (sha256). Above `max_files` the tree falls back to a
size+mtime manifest so the cost stays bounded on large repositories. Both modes
produce the same `{relpath: token}` shape, but the tokens are not comparable
across modes — a diff that spans a mode switch reports `mode_changed` and
degrades to path-level accuracy rather than inventing modifications.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

# Directory names pruned from every walk (matched per path component).
IGNORED_DIRS = frozenset({
    ".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".tox", ".venv", "venv", "env", "node_modules",
    ".idea", ".vscode", ".cache", "dist", "build", ".eggs",
})

# Suffixes never fingerprinted: bytecode, native objects, caches.
IGNORED_SUFFIXES = (".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib", ".o", ".obj")

# Per-bucket cap on the path lists carried in the audit row / tool result.
MAX_DIFF_PATHS = 50


def _walk(root: Path) -> list[Path]:
    """Every non-ignored file under `root`, in deterministic order."""
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
        for name in sorted(filenames):
            if name.endswith(IGNORED_SUFFIXES):
                continue
            out.append(Path(dirpath) / name)
    return out


def _token(path: Path, mode: str) -> str | None:
    """Fingerprint one file. `mode` is 'hash' or 'manifest'; None if unreadable."""
    try:
        if mode == "manifest":
            st = path.stat()
            return f"{st.st_size}:{st.st_mtime_ns}"
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def capture_tree(root: str | Path, *, max_files: int = 2000) -> dict[str, Any]:
    """Fingerprint a workspace tree.

    Returns `{"mode", "files": {relpath: token}, "truncated"}`. Paths are
    posix-style and relative to `root`, so snapshots are comparable across
    platforms. `truncated` is True when the tree exceeded `max_files` and only
    the first `max_files` entries were fingerprinted.
    """
    root = Path(root).resolve()
    paths = _walk(root)
    truncated = len(paths) > max_files
    if truncated:
        paths = paths[:max_files]
    # Above the cap, content hashing every run is the expensive path we are
    # avoiding, so the plan is deliberately cheaper than the counter.
    mode = "manifest" if truncated else "hash"
    files: dict[str, str] = {}
    for p in paths:
        token = _token(p, mode)
        if token is None:
            continue
        try:
            files[p.relative_to(root).as_posix()] = token
        except ValueError:
            continue
    return {"mode": mode, "files": files, "truncated": truncated}


def diff_snapshots(before: dict[str, Any] | None, after: dict[str, Any] | None,
                   *, limit: int = MAX_DIFF_PATHS) -> dict[str, Any]:
    """Compare two snapshots into created / modified / deleted buckets."""
    b_snap, a_snap = before or {}, after or {}
    b, a = b_snap.get("files") or {}, a_snap.get("files") or {}
    created = sorted(set(a) - set(b))
    deleted = sorted(set(b) - set(a))
    # A content hash and a size+mtime token are not comparable, so if the tree
    # crossed the `max_files` cap between the two captures the only honest diff
    # is by path: added/removed are still certain, modifications are unknown.
    mode_changed = (bool(b) and bool(a)
                    and b_snap.get("mode") != a_snap.get("mode"))
    modified = [] if mode_changed else sorted(k for k in set(a) & set(b) if a[k] != b[k])
    total = len(created) + len(modified) + len(deleted)

    parts = []
    if created:
        parts.append(f"新增 {len(created)}")
    if modified:
        parts.append(f"修改 {len(modified)}")
    if deleted:
        parts.append(f"删除 {len(deleted)}")
    if mode_changed:
        parts.append("快照模式切换，修改项无法比对")
    return {
        "created": created[:limit],
        "modified": modified[:limit],
        "deleted": deleted[:limit],
        "total": total,
        "changed": total > 0,
        "truncated": any(len(x) > limit for x in (created, modified, deleted)),
        "mode_changed": mode_changed,
        "summary": "，".join(parts) if parts else "无文件变更",
        "mode": a_snap.get("mode", "hash"),
    }


def affected_paths(diff: dict[str, Any] | None) -> list[str]:
    """The flat, sorted set of paths a diff touched (any bucket)."""
    d = diff or {}
    return sorted(set(d.get("created") or []) | set(d.get("modified") or [])
                  | set(d.get("deleted") or []))
