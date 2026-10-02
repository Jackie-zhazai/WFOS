"""Project-level durable memory: what previous runs left behind, and when to trust it.

The two claims worth pinning:

  * memory is derived from the harness's own record, so a run that **failed** is
    remembered as readily as one that completed. That is the difference from the
    wiki, whose curator only runs after a success — knowledge-based memory is
    silent exactly where a later run would want it most;
  * memory is anchored to the files as the run left them, so a memory describing
    a state that no longer exists is **withheld** rather than injected, and the
    withholding is visible in the record instead of looking like "no memory".
"""
from __future__ import annotations

import asyncio

from wfos.memory import describe, memory_for, render

FEATURE = "新增一个用户模块，在 app.py 中追加 feature_user() 函数"
DELETE = "删除 app.py 中废弃的 compute 函数"


def _finish(h, request: str = FEATURE) -> str:
    run_id = h.create_run(request, kind="feature")["id"]
    asyncio.run(h.advance(run_id))
    return run_id


def _fail(h, repo) -> str:
    """A run that ends `failed`: run into the approval gate, then refuse it."""
    run_id = h.create_run(DELETE, kind="feature")["id"]
    asyncio.run(h.advance(run_id))
    for approval in repo.pending_approvals_for_run(run_id):
        h.reject(approval["id"], by="test", note="拒绝")
    asyncio.run(h.advance(run_id))
    return run_id


def _injected(repo, run_id: str) -> tuple[int, int]:
    """(memory items injected, worst stale count) across a run's step prompts."""
    injected = stale = 0
    for step in repo.steps_for_run(run_id):
        meta = ((step.get("input_json") or {}).get("prompt_metadata") or {}).get("memory") or {}
        injected += int(meta.get("injected") or 0)
        stale = max(stale, int(meta.get("stale") or 0))
    return injected, stale


# ------------------------------------------------------------- what gets recorded
def test_a_finished_run_records_the_files_it_left_behind(app):
    h, repo = app["harness"], app["repo"]
    run_id = _finish(h)

    remembered = repo.get_run(run_id)["payload"].get("memory_files")
    assert remembered and "app.py" in remembered
    # The digest is of the file as the run *left* it, which is what makes it
    # usable as a freshness anchor. The run's `key_files` fingerprint is taken at
    # plan time and would read as drift against every file the run wrote.
    from wfos.harness import identity
    assert remembered["app.py"] == identity.file_digest(app["sandbox"] / "app.py")


def test_a_run_still_in_flight_records_nothing(app):
    """Most `advance` calls leave a run mid-flight; those must not be remembered."""
    h, repo = app["harness"], app["repo"]
    run_id = h.create_run(FEATURE, kind="feature")["id"]
    asyncio.run(h.advance(run_id, max_loops=1))

    assert repo.get_run(run_id)["payload"].get("memory_files") is None
    assert memory_for(repo, app["cfg"].project_root, ["app.py"])["items"] == []


def test_a_failed_run_is_remembered_too(app):
    """The case a curator-based memory would miss entirely."""
    h, repo = app["harness"], app["repo"]
    run_id = _fail(h, repo)
    assert repo.get_run(run_id)["status"] == "failed"

    result = memory_for(repo, app["cfg"].project_root, ["app.py"])
    assert [item["run_id"] for item in result["items"]] == [run_id]
    assert result["items"][0]["status"] == "failed"


# ------------------------------------------------------------------- freshness
def test_memory_from_a_previous_run_reaches_the_next_prompt(app):
    h, repo = app["harness"], app["repo"]
    first = _finish(h)
    second = _finish(h)

    injected, stale = _injected(repo, second)
    assert injected > 0, "上一条运行没有进入下一条运行的 prompt"
    assert stale == 0

    metas = [(s.get("input_json") or {}).get("prompt_metadata", {}).get("memory") or {}
             for s in repo.steps_for_run(second)]
    # Exactly one prior run was a candidate, which is what makes `first` the
    # source of what got injected rather than merely a run that happened.
    assert any(m.get("considered") == 1 for m in metas)
    assert [item["run_id"] for item in memory_for(
        repo, app["cfg"].project_root, ["app.py"], exclude_run_id=second)["items"]] == [first]


def test_memory_is_withheld_when_its_files_changed(app):
    h, repo = app["harness"], app["repo"]
    _finish(h)
    (app["sandbox"] / "app.py").write_text("changed behind our back\n", encoding="utf-8")

    result = memory_for(repo, app["cfg"].project_root, ["app.py"])
    assert result["items"] == []
    assert result["stale"], "陈旧记忆应当被报出，而不是表现得像从未记录过"
    assert result["stale"][0]["changed"] == ["app.py"]


def test_withheld_memory_does_not_reach_the_prompt(app):
    h, repo = app["harness"], app["repo"]
    _finish(h)
    (app["sandbox"] / "app.py").write_text("changed behind our back\n", encoding="utf-8")
    second = _finish(h)

    injected, stale = _injected(repo, second)
    assert injected == 0
    assert stale >= 1, "扣留的事实要留在审计里，否则与『从未记录』无法区分"


def test_memory_is_scoped_to_the_files_a_run_touched(app):
    """A run about other files must not be injected into this one."""
    h, repo = app["harness"], app["repo"]
    _finish(h)
    assert memory_for(repo, app["cfg"].project_root, ["unrelated.py"])["items"] == []
    assert memory_for(repo, app["cfg"].project_root, [])["items"] == []


def test_a_run_is_not_its_own_memory(app):
    h, repo = app["harness"], app["repo"]
    run_id = _finish(h)
    assert memory_for(repo, app["cfg"].project_root, ["app.py"],
                      exclude_run_id=run_id)["items"] == []


def test_older_memories_come_after_newer_ones(app):
    h, repo = app["harness"], app["repo"]
    first, second = _finish(h), _finish(h)
    ids = [item["run_id"] for item in
           memory_for(repo, app["cfg"].project_root, ["app.py"], limit=5)["items"]]
    assert ids.index(second) < ids.index(first)


# ------------------------------------------------------------------- rendering
def test_describe_reports_what_the_record_shows(app):
    h, repo = app["harness"], app["repo"]
    run_id = _finish(h)
    described = describe(repo, repo.get_run(run_id))

    assert described["status"] == "completed"
    assert described["steps"] > 0
    assert described["changed_paths"] == ["app.py"]
    assert described["failure_classes"] == []


def test_the_rendered_block_names_the_outcome_and_the_files(app):
    h, repo = app["harness"], app["repo"]
    run_id = _finish(h)
    result = memory_for(repo, app["cfg"].project_root, ["app.py"])
    text = render(result)

    assert "completed" in text and "app.py" in text and run_id in text


def test_an_empty_memory_renders_to_an_empty_block(app):
    assert render({"items": [], "stale": [], "considered": 0}) == ""
