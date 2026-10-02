"""The execution lease: C2's cross-process mutex on advancing a run.

Two processes may work the same database. Within one process an `asyncio.Lock`
serialises callers; across processes a lease on the `runs` row decides, claimed
with a single conditional UPDATE. These tests pin both halves of the contract:
exactly one holder at a time, an expired lease never wedging a run, and — the
part that matters most — a lapsed lease stopping the advance rather than letting
it execute a state twice.

The lease is also deliberately *not* part of the resume fingerprint. `run_lease_seconds`
only affects scheduling; it does not change what a model is asked or allowed to
do, so including it would invalidate every persisted step whenever an operator
tuned it. That decision is pinned here too, so it cannot be undone by accident.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

from wfos.harness import identity
from wfos.harness.orchestrator import RunLockedError

ROOT = Path(__file__).resolve().parents[1]


def _run_cli(*args, sandbox: Path, data: Path) -> tuple[int, str]:
    env = dict(os.environ)
    env.update(WFOS_PROJECT=str(sandbox), WFOS_DATA=str(data),
               WFOS_DB=str(data / "wfos.db"), WFOS_WIKI=str(data / "wiki.db"),
               PYTHONIOENCODING="utf-8")
    proc = subprocess.run([sys.executable, "-m", "wfos", *args], cwd=ROOT,
                          env=env, capture_output=True, text=True,
                          timeout=120, encoding="utf-8", errors="replace")
    return proc.returncode, proc.stdout + proc.stderr


def _expire_lease(repo, run_id: str) -> None:
    """Backdate the lease, as a killed process would leave it behind."""
    repo.conn.execute("UPDATE runs SET lease_until=? WHERE id=?",
                      ("1970-01-01T00:00:00", run_id))
    repo.conn.commit()


def _new_run(app) -> str:
    return app["harness"].create_run("新增一个模块", kind="feature")["id"]


# --------------------------------------------------------------- config plumbing
def test_the_lease_length_survives_a_round_trip_through_config_file():
    """The knob has to exist in all three places, not just the dataclass.

    The original defect was exactly this: `orchestrator` read
    `cfg.harness.run_lease_seconds` while the dataclass, the TOML defaults and
    the TOML parser had never heard of it, so every `advance()` raised
    AttributeError. One assertion per place, so a field added to only one of
    them fails loudly instead of at the first run.
    """
    from wfos.config import _harness_from_toml, _parse_defaults

    defaults = _parse_defaults()
    assert "run_lease_seconds" in defaults["harness"], "TOML 默认值里缺 run_lease_seconds"
    assert _harness_from_toml({}).run_lease_seconds == 300
    assert _harness_from_toml({"run_lease_seconds": 42}).run_lease_seconds == 42


def test_a_config_file_can_raise_the_lease_length(tmp_path):
    from wfos.config import load_config

    cfg_file = tmp_path / "wfos.toml"
    cfg_file.write_text("[harness]\nrun_lease_seconds = 900\n", encoding="utf-8")
    assert load_config(cfg_file).harness.run_lease_seconds == 900


def test_the_lease_floor_is_the_provider_timeout(app):
    """The knob raises the lease; the provider timeout is the floor below it."""
    h, cfg = app["harness"], app["cfg"]
    cfg.harness.run_lease_seconds = 0
    assert h._lease_seconds() == int(cfg.llm.timeout) + 60
    cfg.harness.run_lease_seconds = 10_000
    assert h._lease_seconds() == 10_000


def test_the_lease_length_is_not_part_of_the_resume_fingerprint(app):
    """Tuning the lease must not invalidate steps persisted under the old value."""
    h, cfg = app["harness"], app["cfg"]
    assert "run_lease_seconds" not in identity.identity(cfg, h.agents)
    before = identity.identity_hash(cfg, h.agents)
    cfg.harness.run_lease_seconds = 1234
    assert identity.identity_hash(cfg, h.agents) == before


# ------------------------------------------------------------------ exclusivity
def test_a_lease_is_exclusive(app):
    repo = app["repo"]
    run_id = _new_run(app)
    assert repo.claim_run(run_id, "proc-A", 300) is True
    assert repo.claim_run(run_id, "proc-B", 300) is False
    assert repo.get_run(run_id)["owner"] == "proc-A"


def test_an_expired_lease_is_claimable(app):
    """A killed process must not wedge a run forever."""
    repo = app["repo"]
    run_id = _new_run(app)
    assert repo.claim_run(run_id, "proc-A", 300) is True
    _expire_lease(repo, run_id)
    assert repo.claim_run(run_id, "proc-B", 300) is True
    assert repo.get_run(run_id)["owner"] == "proc-B"


def test_renew_only_works_for_the_holder(app):
    repo = app["repo"]
    run_id = _new_run(app)
    assert repo.claim_run(run_id, "proc-A", 300) is True
    assert repo.renew_lease(run_id, "proc-A", 300) is True
    assert repo.renew_lease(run_id, "proc-B", 300) is False
    assert repo.get_run(run_id)["owner"] == "proc-A"


def test_release_is_owner_guarded(app):
    """A process that lost its lease must not release the new holder's."""
    repo = app["repo"]
    run_id = _new_run(app)
    assert repo.claim_run(run_id, "proc-A", 300) is True
    repo.release_run(run_id, "proc-B")                      # not ours: ignored
    assert repo.get_run(run_id)["owner"] == "proc-A"
    repo.release_run(run_id, "proc-A")
    assert repo.get_run(run_id)["owner"] is None


# ------------------------------------------------------------------- at the seam
def test_advance_refuses_while_another_owner_holds_the_lease(app):
    h, repo = app["harness"], app["repo"]
    run_id = _new_run(app)
    assert repo.claim_run(run_id, "other-process:1234", 300) is True

    with pytest.raises(RunLockedError) as exc:
        asyncio.run(h.advance(run_id))

    assert "other-process:1234" in str(exc.value)
    # Nothing ran: no state was entered, so no step was persisted.
    assert repo.get_step(run_id, "req_capture") is None


def test_advance_releases_the_lease_when_it_returns(app):
    h, repo = app["harness"], app["repo"]
    run_id = _new_run(app)
    run = asyncio.run(h.advance(run_id, max_loops=1))

    assert run["status"] == "running"          # one state ran, run not finished
    row = repo.get_run(run_id)
    assert row["owner"] is None
    assert row["lease_until"] is None


def test_a_stale_lease_does_not_wedge_a_run(app):
    h, repo = app["harness"], app["repo"]
    run_id = _new_run(app)
    assert repo.claim_run(run_id, "killed-process:0000", 300) is True
    _expire_lease(repo, run_id)

    run = asyncio.run(h.advance(run_id, max_loops=1))
    assert run["status"] == "running"
    assert repo.get_run(run_id)["owner"] is None   # taken over, then released


def test_a_lapsed_lease_stops_the_advance_instead_of_double_executing(app, monkeypatch):
    """The lease lapsing mid-run must stop, not continue on a claim we lost.

    Stopping leaves the state un-executed (recoverable by resuming); carrying on
    would risk a second process running the same state at the same time. The
    lease is still released on the way out, so the run is not left frozen.
    """
    h, repo = app["harness"], app["repo"]
    run_id = _new_run(app)
    monkeypatch.setattr(repo, "renew_lease", lambda *a, **k: False)

    with pytest.raises(RunLockedError) as exc:
        asyncio.run(h.advance(run_id, max_loops=2))

    assert "执行租约在执行途中失效" in str(exc.value)
    assert repo.get_step(run_id, "req_capture") is None
    assert repo.get_run(run_id)["owner"] is None


def test_the_heartbeat_holds_the_lease_through_a_state_slower_than_it(app, monkeypatch):
    """A state slower than the lease must not lose it.

    Renewal used to happen once per state, but one state can make up to
    `max_rounds` model calls (8 for the implementer, 6 at 240s for the
    verifier), so a slow state left a window in which another process could
    claim the run and execute the same state.

    Timestamps are second-resolution (`db.now` uses `timespec="seconds"`) and
    the expiry test is a strict `<`, so a 3s lease claimed at second S is still
    held through second S+3. The claim is therefore attempted at 4.5s — a whole
    second past the expiry — and is refused only because the heartbeat renewed
    in the meantime.
    """
    h, repo = app["harness"], app["repo"]
    monkeypatch.setattr(h, "_lease_seconds", lambda: 3)

    agent = h.agents["investigator"]
    real_run = agent.run

    async def slow_run(ctx):
        await asyncio.sleep(6.0)
        return await real_run(ctx)

    monkeypatch.setattr(agent, "run", slow_run)
    run_id = _new_run(app)

    async def scenario():
        advancing = asyncio.create_task(h.advance(run_id, max_loops=1))
        await asyncio.sleep(4.5)                    # a full second past expiry
        stolen = repo.claim_run(run_id, "intruder:0000", 300)
        await advancing
        return stolen

    assert asyncio.run(scenario()) is False, "租约在慢状态期间过期，被别的进程抢走了"


def test_a_heartbeat_that_loses_the_lease_stops_the_advance(app, monkeypatch):
    """Loss discovered by the heartbeat — not by the boundary renewal — stops too.

    The boundary renewal only runs between states, so without the heartbeat's
    flag a takeover during a state would go unnoticed until the next state
    boundary, long after the claim was gone.
    """
    h, repo = app["harness"], app["repo"]
    monkeypatch.setattr(h, "_lease_seconds", lambda: 3)

    real_renew = repo.renew_lease
    calls = {"n": 0}

    def renew_once_then_lose(run_id, owner, ttl):
        calls["n"] += 1
        return real_renew(run_id, owner, ttl) if calls["n"] == 1 else False

    monkeypatch.setattr(repo, "renew_lease", renew_once_then_lose)

    agent = h.agents["investigator"]
    real_run = agent.run

    async def slow_run(ctx):
        await asyncio.sleep(2.0)                    # long enough for the 1s heartbeat
        return await real_run(ctx)

    monkeypatch.setattr(agent, "run", slow_run)
    run_id = _new_run(app)

    with pytest.raises(RunLockedError):
        asyncio.run(h.advance(run_id, max_loops=3))

    counts = repo.conn.execute(
        "SELECT COUNT(*) AS n FROM steps WHERE run_id=?", (run_id,)).fetchone()
    assert counts["n"] == 1            # ran the state it held, then stopped
    assert repo.get_run(run_id)["owner"] is None


def test_advance_leaves_no_heartbeat_task_behind(app):
    """The heartbeat must not outlive the advance that started it."""
    h = app["harness"]
    run_id = _new_run(app)

    async def scenario():
        await h.advance(run_id, max_loops=1)
        return [t.get_name() for t in asyncio.all_tasks()
                if t is not asyncio.current_task()]

    assert asyncio.run(scenario()) == []


def test_two_advances_of_the_same_run_do_not_interleave(app, monkeypatch):
    """Concurrent callers in one process queue up instead of both executing.

    The yield inside the agent is what makes this discriminating: it is the
    point where two callers would otherwise interleave. Without the per-run
    lock the second caller would find the lease taken and be refused, so
    "both callers completed" is the property that proves they serialised.
    `max_loops=1` means each call runs exactly one state, so the two together
    must advance two *different* states and never one state twice.
    """
    h, repo = app["harness"], app["repo"]
    run_id = _new_run(app)

    agent = h.agents["investigator"]
    real_run = agent.run

    async def yielding_run(ctx):
        await asyncio.sleep(0)          # hand the loop to the second caller
        return await real_run(ctx)

    monkeypatch.setattr(agent, "run", yielding_run)

    async def both():
        return await asyncio.gather(
            h.advance(run_id, max_loops=1),
            h.advance(run_id, max_loops=1),
            return_exceptions=True)

    results = asyncio.run(both())
    assert not any(isinstance(r, BaseException) for r in results), results

    counts = repo.conn.execute(
        "SELECT state, COUNT(*) AS n FROM steps WHERE run_id=? GROUP BY state",
        (run_id,)).fetchall()
    assert sum(row["n"] for row in counts) == 2      # both callers did work
    assert max(row["n"] for row in counts) == 1      # but no state ran twice
    assert repo.get_run(run_id)["owner"] is None     # and both released


# ---------------------------------------------------------------- at the CLI
def _cli_dirs(tmp_path):
    sandbox = tmp_path / "proj"
    sandbox.mkdir()
    (sandbox / "app.py").write_text("def compute(x):\n    return x * 2\n",
                                    encoding="utf-8")
    data = tmp_path / "data"
    data.mkdir()
    return sandbox, data


def test_cli_resume_reports_lease_contention_instead_of_a_traceback(tmp_path):
    """Contention is an expected outcome: explain it and exit 3, don't traceback.

    This is the path a guard placed only around `advance()` would miss —
    `Harness.resume` awaits `advance` internally, so the raise happens inside
    the coroutine the CLI hands to `asyncio.run`.
    """
    from wfos.storage.repo import Repo

    sandbox, data = _cli_dirs(tmp_path)
    repo = Repo(str(data / "wfos.db"))               # creates the schema
    run_id = repo.create_run("feature", "标题", "新增一个模块")["id"]
    assert repo.claim_run(run_id, "other-process:4321", 300) is True
    repo.close()

    code, out = _run_cli("resume", run_id, sandbox=sandbox, data=data)
    assert code == 3, out
    assert "other-process:4321" in out
    assert "本次未执行任何状态" in out
    assert "Traceback" not in out


def test_cli_approve_reports_lease_contention_instead_of_a_traceback(tmp_path):
    """The other route into the lease: approving the last pending gate."""
    from wfos.storage.repo import Repo

    sandbox, data = _cli_dirs(tmp_path)
    repo = Repo(str(data / "wfos.db"))
    run_id = repo.create_run("feature", "标题", "新增一个模块")["id"]
    approval = repo.create_approval(run_id, "high_risk:标题")
    repo.set_status(run_id, "waiting_approval")
    assert repo.claim_run(run_id, "other-process:4321", 300) is True
    repo.close()

    code, out = _run_cli("approve", approval["id"], sandbox=sandbox, data=data)
    assert code == 3, out
    assert "已批准" in out
    assert "other-process:4321" in out
    assert "Traceback" not in out
