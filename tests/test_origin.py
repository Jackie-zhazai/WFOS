"""Where a run came from, and why it is a correctness boundary rather than a label.

A benchmark run and a production run share the same database shape, the same
failure vocabulary and the same repo-relative paths. Without this column the
only thing separating them is that nobody happened to look:

  * `skills_for` had no filter of any kind, and a recorded skill is live on
    insertion — so running a benchmark into a failure class wrote a procedure
    straight into a production prompt. No human, no benchmark, no gate.
  * `memory_for` scans every run in the database and matches on normalized paths,
    so two workspaces that both contain `app.py` collide by construction, and a
    request-anchored entry is explicitly not freshness-checked.

Both are now scoped by origin. `candidate` is the other half: an artificial run's
procedure is not merely *tagged*, it is ineligible.
"""
from __future__ import annotations

import sqlite3

from wfos.harness import identity
from wfos.memory import memory_for
from wfos.models import ORIGIN_BENCHMARK, ORIGIN_INTERACTIVE, SKILL_CANDIDATE, SKILL_LIVE
from wfos.storage.repo import Repo


# ------------------------------------------------------------------- skills
def test_a_benchmark_procedure_does_not_reach_a_production_prompt(app):
    """The injection channel this column exists to close."""
    repo = app["repo"]
    repo.add_skill("test_failure", "基准学到的", "过程", "bench-run",
                   origin=ORIGIN_BENCHMARK)

    assert repo.skills_for(["test_failure"], origin=ORIGIN_INTERACTIVE) == [], \
        "benchmark 跑出的规程进了生产 prompt"


def test_a_benchmark_run_can_see_its_own_candidates(app):
    """Otherwise the RSI loop could never measure whether a proposal helps."""
    repo = app["repo"]
    repo.add_skill("test_failure", "基准学到的", "过程", "bench-run",
                   origin=ORIGIN_BENCHMARK)

    got = repo.skills_for(["test_failure"], origin=ORIGIN_BENCHMARK)

    assert [s["status"] for s in got] == [SKILL_CANDIDATE]
    assert got[0]["title"] == "基准学到的"


def test_an_interactive_procedure_is_live_and_stays_visible(app):
    """A human drove that run, and the text is the harness's own record of it."""
    repo = app["repo"]
    repo.add_skill("test_failure", "生产的", "过程", "prod-run")

    got = repo.skills_for(["test_failure"], origin=ORIGIN_INTERACTIVE)

    assert [s["status"] for s in got] == [SKILL_LIVE]


def test_a_benchmark_run_sees_production_procedures_too(app):
    """Exercising the real procedure is the point of a benchmark."""
    repo = app["repo"]
    repo.add_skill("test_failure", "生产的", "过程", "prod-run")

    got = repo.skills_for(["test_failure"], origin=ORIGIN_BENCHMARK)

    assert [s["title"] for s in got] == ["生产的"]


def test_a_benchmark_candidate_cannot_retire_a_production_procedure(app):
    """Supersession is scoped, and this is the hazard that forced it.

    Superseding is per trigger globally, so a benchmark run recording anything
    for `test_failure` would mark the *production* procedure superseded — and it
    would silently stop reaching a human's prompts. Running a task must not be
    able to switch off what production relies on.
    """
    repo = app["repo"]
    repo.add_skill("test_failure", "prod v1", "过程", "prod-run")
    repo.add_skill("test_failure", "基准的", "过程", "bench-run",
                   origin=ORIGIN_BENCHMARK)

    live = repo.skills_for(["test_failure"], origin=ORIGIN_INTERACTIVE)

    assert [s["title"] for s in live] == ["prod v1"], "生产规程被基准运行顶掉了"


def test_the_version_series_is_per_scope(app):
    """Two scopes do not share a version counter, so "v2" always means the same
    thing within the scope a reader is looking at."""
    repo = app["repo"]
    first = repo.add_skill("test_failure", "prod v1", "p", "r1")
    repo.add_skill("test_failure", "bench v1", "p", "r2", origin=ORIGIN_BENCHMARK)
    third = repo.add_skill("test_failure", "prod v2", "p", "r3")

    assert (first["version"], third["version"]) == (1, 2)


def test_status_is_derived_so_a_caller_cannot_get_it_wrong(app):
    """Deriving beats demanding: `add_skill` is the one place the rule lives."""
    repo = app["repo"]
    assert repo.add_skill("a", "t", "p", "r")["status"] == SKILL_LIVE
    assert repo.add_skill("b", "t", "p", "r",
                          origin=ORIGIN_BENCHMARK)["status"] == SKILL_CANDIDATE


def test_promotion_can_write_an_explicit_status(app):
    """The gate needs to move a candidate up, and has to say which row it came
    from — `superseded` says which is current, `parent_id` says where it is from."""
    repo = app["repo"]
    seed = repo.add_skill("test_failure", "候选", "p", "bench-run",
                          origin=ORIGIN_BENCHMARK)
    promoted = repo.add_skill("test_failure", "晋升的", "p", "bench-run",
                              origin=ORIGIN_INTERACTIVE, status=SKILL_LIVE,
                              parent_id=seed["id"])

    assert promoted["parent_id"] == seed["id"]
    assert [s["title"] for s in repo.skills_for(["test_failure"])] == ["晋升的"]


# ------------------------------------------------------------------- memory
def _remembered_run(app, *, origin, request="新增一个用户模块"):
    """A finished run that left a memory anchor on `app.py`."""
    repo, root = app["repo"], app["sandbox"]
    run = repo.create_run("feature", "t", request, origin=origin)
    repo.update_run(run["id"], payload={
        "memory_files": {"app.py": identity.file_digest(root / "app.py")},
        "memory_request": request})
    repo.set_status(run["id"], "completed")
    return run


def test_benchmark_memory_does_not_reach_a_production_prompt(app):
    _remembered_run(app, origin=ORIGIN_BENCHMARK)

    got = memory_for(app["repo"], app["sandbox"], ["app.py"],
                     origin=ORIGIN_INTERACTIVE)

    assert got["items"] == [], f"基准运行的记忆进了生产 prompt：{got}"


def test_the_same_origin_still_sees_its_own_memory(app):
    """Scoping must not break within-suite memory — a case that runs twice in one
    suite is exactly what the `memory-from-a-previous-run` case depends on."""
    run = _remembered_run(app, origin=ORIGIN_BENCHMARK)

    got = memory_for(app["repo"], app["sandbox"], ["app.py"],
                     origin=ORIGIN_BENCHMARK, exclude_run_id="none")

    assert [item["run_id"] for item in got["items"]] == [run["id"]]


# --------------------------------------------------------------- inheritance
def test_a_child_inherits_where_its_parent_came_from(app):
    """A delegated sub-task of a benchmark run is a benchmark run. Inferring it
    fresh would let a benchmark spawn children that seed production."""
    repo = app["repo"]
    parent = repo.create_run("feature", "t", "d", origin=ORIGIN_BENCHMARK)

    child = repo.create_run("bugfix", "t", "d", parent_run_id=parent["id"])

    assert child["origin"] == ORIGIN_BENCHMARK


def test_an_interactive_run_is_the_default(app):
    repo = app["repo"]
    assert repo.create_run("feature", "t", "d")["origin"] == ORIGIN_INTERACTIVE


# ------------------------------------------------------------------ migration
def test_an_old_database_gets_the_behaviour_it_already_had(tmp_path):
    """Defaults must not reclassify history.

    Every row written before these columns existed came from an interactive run
    and every skill was immediately live. A migration that guessed otherwise
    would be inventing history — and would silently switch off procedures that
    production has been relying on.
    """
    path = tmp_path / "old.db"
    con = sqlite3.connect(str(path))
    con.executescript("""
      CREATE TABLE runs (id TEXT PRIMARY KEY, kind TEXT NOT NULL, title TEXT NOT NULL,
        description TEXT NOT NULL, state TEXT NOT NULL, status TEXT NOT NULL,
        parent_run_id TEXT, root_run_id TEXT, attempt INTEGER NOT NULL DEFAULT 0,
        payload TEXT NOT NULL DEFAULT '{}', result TEXT, error TEXT,
        identity_hash TEXT, key_files TEXT, owner TEXT, lease_until TEXT,
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
      CREATE TABLE skills (id INTEGER PRIMARY KEY AUTOINCREMENT, trigger TEXT NOT NULL,
        title TEXT NOT NULL, procedure TEXT NOT NULL, evidence_run TEXT NOT NULL,
        version INTEGER NOT NULL DEFAULT 1, superseded INTEGER NOT NULL DEFAULT 0,
        files TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL);
      INSERT INTO runs VALUES ('r1','feature','t','d','req_capture','completed',
        NULL,'r1',0,'{}',NULL,NULL,NULL,NULL,NULL,NULL,'2026-01-01','2026-01-01');
      INSERT INTO skills (trigger,title,procedure,evidence_run,created_at)
        VALUES ('test_failure','旧规程','p','r1','2026-01-01');
    """)
    con.commit()
    con.close()

    repo = Repo(path)

    assert repo.get_run("r1")["origin"] == ORIGIN_INTERACTIVE
    # And the old procedure is still usable, which is the whole point.
    assert [s["title"] for s in repo.skills_for(["test_failure"])] == ["旧规程"]
    assert repo.skills_for(["test_failure"])[0]["status"] == SKILL_LIVE


def test_the_migrated_columns_are_all_reachable(app):
    """A column nothing writes is a column that will read as its default forever."""
    cols = {r[1] for r in app["repo"].conn.execute("PRAGMA table_info(runs)")}
    assert {"origin", "task_id", "task_version"} <= cols
    cols = {r[1] for r in app["repo"].conn.execute("PRAGMA table_info(skills)")}
    assert {"status", "origin", "parent_id"} <= cols


# ------------------------------------------------------------------- metrics
def test_the_all_runs_report_splits_by_origin_without_hiding_any(app):
    """Grouped, not filtered.

    A total that silently excluded benchmark runs would be a different kind of
    dishonest: the reader sees a smaller number and no reason for it. The split
    keeps the total true and makes the mix visible — which matters because
    `metrics aggregate` otherwise mixes them into one figure with no sign.
    """
    from wfos.metrics import aggregate

    repo = app["repo"]
    for origin in (ORIGIN_INTERACTIVE, ORIGIN_BENCHMARK):
        run = repo.create_run("feature", "t", "d", origin=origin)
        repo.add_step(run["id"], "req_capture", "investigator", {"ok": True},
                      status="done")

    report = aggregate(repo)

    assert set(report["by_origin"]) == {ORIGIN_INTERACTIVE, ORIGIN_BENCHMARK}
    # The total still counts both, so nothing is hidden by the split.
    assert report["steps"] == 2
    assert sum(g["steps"] for g in report["by_origin"].values()) == 2


def test_the_skill_vocabulary_has_no_status_without_a_producer(app):
    """`verified` was removed for this reason.

    A rung nothing can write is a word a reader finds in the vocabulary and never
    in the data — the same defect as a config knob nothing reads or a failure
    name nothing produces. It comes back with the promotion gate that writes it.
    """
    from wfos.models import SKILL_STATUSES

    assert set(SKILL_STATUSES) == {SKILL_CANDIDATE, SKILL_LIVE}
    assert "verified" not in SKILL_STATUSES
