"""Resume-time fingerprints: what a persisted step's output was produced under.

A step output is only reusable while the conditions that produced it still hold.
Two things invalidate it: a different brain or rule set (`runtime_identity`), and
key files changing behind the run's back (`key_files`).

Key files are the *plan's* files, not the whole tree — that is the cheap,
targeted version of the question, and it is answerable for a large repository.
The run's own writes are excluded, because a run that edits a file has not
"drifted" from itself.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Resume validation outcomes. `no-fingerprint` is deliberately distinct from a
# mismatch: a run recorded before fingerprints existed (or a hand-seeded step)
# has nothing to compare, which is not the same as a failed comparison.
RESUME_VALID = "full-valid"
RESUME_IDENTITY_MISMATCH = "identity-mismatch"
RESUME_WORKSPACE_DRIFT = "workspace-drift"
RESUME_NO_FINGERPRINT = "no-fingerprint"


def identity(cfg, agents: dict) -> dict[str, Any]:
    """The conditions a run was executed under, as a canonical dict.

    Everything here changes what a model would be asked or allowed to do, so a
    difference in any of it means an earlier step's output may not be
    reproducible.
    """
    h = cfg.harness
    return {
        "provider": cfg.llm.provider,
        # Per role, not one value: with `[llm.routing]` the implementer and the
        # verifier may be answering on different models, and a step produced under
        # one route is not reproducible after that route changes. Read off the
        # adapter actually wired up rather than off `cfg.llm.model`, so an injected
        # brain is recorded as itself instead of as the configured default.
        "models": {role: getattr(getattr(a, "llm", None), "model", "")
                   for role, a in sorted(agents.items())},
        "base_url": cfg.llm.base_url or "",
        "temperature": cfg.llm.temperature,
        "max_tokens": cfg.llm.max_tokens,
        "project_root": str(Path(cfg.project_root).resolve()),
        "safe_commands": sorted(f"{sc.cmd}|{sc.timeout}|{sc.exact}" for sc in h.safe_commands),
        "roles": {role: sorted(getattr(a, "allowed_tools", []))
                  for role, a in sorted(agents.items())},
        "max_verify_attempts": h.max_verify_attempts,
        "repeated_call_threshold": h.repeated_call_threshold,
        "repeated_failure_threshold": h.repeated_failure_threshold,
        "enforce_plan_scope": h.enforce_plan_scope,
        "max_parent_depth": h.max_parent_depth,
    }


def identity_hash(cfg, agents: dict) -> str:
    blob = json.dumps(identity(cfg, agents), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def file_digest(path: str | Path) -> str | None:
    """sha256 of a file's bytes, or None if it cannot be read."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def capture_key_files(root: str | Path, paths: Iterable[str]) -> dict[str, str | None]:
    """Digest each key file. A missing file records None, not an omission —
    'this was deleted' and 'this was never key' must stay distinguishable."""
    base = Path(root).resolve()
    out: dict[str, str | None] = {}
    for rel in paths:
        norm = _norm(rel)
        if not norm:
            continue
        out[norm] = file_digest(base / norm)
    return out


def normalize_path(path: str) -> str:
    """Repo-relative, posix-separated, no leading `./` — one spelling per file."""
    return str(path).replace("\\", "/").lstrip("./")


_norm = normalize_path       # internal shorthand


def validate_resume(recorded_identity: str | None,
                    recorded_key_files: dict[str, str | None] | None,
                    current_identity: str,
                    root: str | Path,
                    own_changes: Iterable[str]) -> tuple[str, dict[str, Any]]:
    """Decide whether a run's persisted steps may be reused.

    Returns `(status, detail)`. `own_changes` are paths this run wrote; a change
    to one of them is the run's own doing, not drift.
    """
    if not recorded_identity:
        return RESUME_NO_FINGERPRINT, {"reason": "未记录指纹（旧数据或手工种入的步骤）"}

    if recorded_identity != current_identity:
        return RESUME_IDENTITY_MISMATCH, {
            "reason": "运行环境（模型/策略/工具集）与记录时不一致",
            "recorded": recorded_identity[:12],
            "current": current_identity[:12],
        }

    own = {_norm(p) for p in own_changes}
    drifted: list[dict[str, Any]] = []
    for rel, recorded in (recorded_key_files or {}).items():
        if rel in own:
            continue                      # the run wrote it; that is not drift
        now = file_digest(Path(root).resolve() / rel)
        if now != recorded:
            drifted.append({"path": rel,
                            "state": "deleted" if now is None else "modified"})
    if drifted:
        return RESUME_WORKSPACE_DRIFT, {
            "reason": "关键文件在运行之外被改动",
            "drifted": drifted,
        }
    return RESUME_VALID, {"reason": "环境与关键文件均未变化"}
