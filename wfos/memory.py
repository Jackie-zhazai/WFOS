"""Project-level durable memory: what previous runs left behind on these files.

Distinct from the wiki, which holds *knowledge* — a lesson a curator distilled,
in prose. This holds *record*: a run ended here, on these files, with this
outcome. Two consequences follow from deriving it from the harness's own record
rather than from a model:

  * it exists for runs that **failed**, which is when a later run most needs to
    know what was already tried — a curator only runs after a success, so
    knowledge-based memory is silent exactly where it would matter most;
  * it costs no model call, so remembering is not a budget decision.

**Freshness is the point.** A memory is anchored to the digests of the files as
the run left them. If any of those files has moved since, the memory describes a
state that no longer exists and is withheld rather than injected. Staleness is
reported, not hidden, so "no memory" stays distinguishable from "everything we
had went stale".

The anchor is written by `Harness._remember_ended_run` at the end of the run,
because the run's existing `key_files` fingerprint is taken when the plan is
established — it describes the state the run *started* from, and every file the
run then wrote would look like drift against it.
"""
from __future__ import annotations

from pathlib import Path

from .harness import identity
from .models import ORIGIN_INTERACTIVE, TERMINAL_STATUSES
from .relevance import cap, overlap_tokens, relevance_key, working_names

# How many prior runs to scan before giving up on finding relevant ones. Bounded
# so a long-lived project cannot make every prompt build walk its whole history.
_SCAN_LIMIT = 200
DEFAULT_LIMIT = 3
# How many request tokens two runs must share before "the same request as last
# time" is believed. Coarser than an exact path intersection, so it needs a
# threshold rather than a truth test.
REQUEST_OVERLAP_MIN = 2


def _norm(path: str) -> str:
    return identity.normalize_path(path)


def capture_end_state(project_root, paths) -> dict[str, str | None]:
    """Digest the files a finished run is remembered by.

    Reuses the resume fingerprint's own digest, so "the file moved" means the
    same thing to memory as it does to resume.
    """
    return identity.capture_key_files(project_root, paths)


def describe(repo, run: dict) -> dict:
    """What a finished run left behind, in the harness's own words.

    Every field comes from the record — states visited, failure classes, paths
    actually observed changing — so nothing here can be a model's claim about
    what it did.
    """
    steps = repo.steps_for_run(run["id"])
    return {
        "run_id": run["id"],
        "kind": run["kind"],
        "title": run["title"],
        "status": run["status"],
        "final_state": run["state"],
        "steps": len(steps),
        "failure_classes": sorted({s["failure_class"] for s in steps
                                   if s.get("failure_class")}),
        "changed_paths": repo.affected_paths_for_run(run["id"]),
    }


def memory_for(repo, project_root, working_files, *, request: str | None = None,
               exclude_run_id: str | None = None,
               origin: str = ORIGIN_INTERACTIVE,
               limit: int = DEFAULT_LIMIT) -> dict:
    """Prior runs relevant to what this run is about.

    Returns `{"items", "total", "dropped", "stale", "considered"}`, and the ways
    material can be missing are kept apart on purpose:

      * `dropped` — cut by `limit`;
      * `stale`   — withheld because a file this run is remembered by has changed
                    since. The memory describes a state that no longer exists.

    Both are returned rather than merely counted, because an empty memory section
    that is empty for a reason is a different fact from one that is empty because
    nothing was ever recorded — and only one of them means the operator should
    raise a limit.

    **Two kinds of anchor**, and each entry says which it has:

      * `files`   — exact normalized paths from the run's `memory_files`. The
                    strong signal, and the only one that can be freshness-checked.
      * `request` — the run stopped before it planned or wrote anything, so it has
                    no files to be recognised by. Matched on request-token
                    overlap. Coarser, and **uncheckable** — nothing can tell
                    whether the state it describes still holds, so `render` says
                    so rather than letting it read like verified memory.

    The `request` anchor exists because that early-failure case is exactly the
    one a later run most needs to know about, and it used to be the one case that
    recorded nothing at all.
    """
    wanted = {_norm(p) for p in (working_files or []) if p}
    request = request or ""
    if not wanted and not request:
        return {"items": [], "total": 0, "dropped": 0, "stale": [], "considered": 0}

    names = working_names(working_files)
    root = Path(project_root)
    fresh: list[dict] = []
    stale: list[dict] = []
    considered = 0

    for run in repo.list_runs(limit=_SCAN_LIMIT):
        if run["id"] == exclude_run_id or run["status"] not in TERMINAL_STATUSES:
            continue
        if str(run.get("origin") or ORIGIN_INTERACTIVE) != origin:
            # Memory is anchored to *repo-relative paths*, so two workspaces that
            # both contain `app.py` collide by construction, and a request-anchored
            # item is explicitly not freshness-checked. A benchmark run of "新增一个
            # 用户模块" would otherwise inject its post-mortem into a real run's
            # prompt with nothing to distinguish the two.
            continue
        payload = run.get("payload") or {}
        remembered = payload.get("memory_files")
        remembered = remembered if isinstance(remembered, dict) else {}
        touched = wanted & {_norm(p) for p in remembered} if remembered else set()

        if touched:
            anchor = "files"
        elif not remembered and overlap_tokens(
                request, str(payload.get("memory_request") or "")) >= REQUEST_OVERLAP_MIN:
            anchor = "request"
        else:
            continue

        considered += 1
        entry = {**describe(repo, run), "anchored": anchor, "files": sorted(touched)}
        if anchor == "files":
            changed = [p for p in sorted(touched)
                       if identity.file_digest(root / p) != remembered.get(p)]
            if changed:
                stale.append({**entry, "changed": changed})
                continue
        fresh.append(entry)

    ordered = sorted(fresh, key=lambda item: relevance_key(
        f"{item['title']} {' '.join(item.get('changed_paths') or [])}", "high", names))
    return {**cap(ordered, limit), "stale": stale, "considered": considered}


def render(memory: dict) -> str:
    """The memory block as it enters a prompt.

    Compact on purpose: this rides in every prompt that has anything to say, and
    a memory section that crowds out the plan is a net loss.
    """
    lines: list[str] = []
    for item in memory.get("items") or []:
        outcome = item["status"]
        if item["failure_classes"]:
            outcome += "（失败类型 " + "、".join(item["failure_classes"]) + "）"
        changed = "、".join(item["changed_paths"]) or "无"
        if item.get("anchored") == "request":
            # Weaker footing, and it says so. Nothing backs this one but a
            # request-similarity match, so no digest checked whether the state it
            # describes still holds — letting it read like the file-anchored kind
            # would be passing an unverified claim off as a verified one.
            anchor = "；**无文件锚点，未做新鲜度校验**"
        else:
            anchor = f"，涉及文件 {'、'.join(item['files'])}"
        # The run id is part of the line on purpose: memory a reader cannot trace
        # back to the record it came from is just an assertion.
        lines.append(
            f"- [{item['kind']}] {item['title']} → {outcome}；"
            f"终态 {item['final_state']}，{item['steps']} 步，"
            f"改动 {changed}{anchor}"
            f"（运行 {item['run_id']}）")
    return "\n".join(lines)
