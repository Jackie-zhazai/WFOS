"""The run environment: one run's tree, database and stack, and nothing shared.

Isolation is what makes a task run's memory, skills and audit rows unable to
reach a real run's — the baseline has relied on it since it existed, and the task
runner (P1) needs the same guarantee. These tests pin the properties a second
implementation would have to keep, which is the reason there is only one.
"""
from __future__ import annotations

import asyncio

from wfos.runner import RunEnvironment

FIXTURE = {"app.py": "def compute(x):\n    return x * 2\n",
           "sub/check.py": "print('PASS: ok')\n"}


def test_each_environment_gets_its_own_tree_and_database(tmp_path):
    """Two environments must not share a single thing that carries state."""
    a = RunEnvironment.create(tmp_path, name="a")
    b = RunEnvironment.create(tmp_path, name="b")

    assert a.root != b.root
    assert a.config.db_path != b.config.db_path
    assert a.config.wiki_path != b.config.wiki_path
    assert a.config.data_dir != b.config.data_dir
    # And each points only at itself.
    assert a.config.project_root == a.root
    assert a.config.db_path.parent == a.config.data_dir


def test_the_fixture_is_written_before_anything_can_read_it(tmp_path):
    """A run must never observe a half-built tree."""
    env = RunEnvironment.create(tmp_path, name="case", fixture=FIXTURE)

    assert (env.root / "app.py").read_text(encoding="utf-8") == FIXTURE["app.py"]
    assert (env.root / "sub" / "check.py").exists(), "嵌套目录没有被创建"


def test_the_harness_is_wired_to_this_environment_only(tmp_path):
    env = RunEnvironment.create(tmp_path, name="case", fixture=FIXTURE)

    harness = env.harness()

    assert harness.cfg.project_root == env.root
    assert harness.repo is not None
    assert harness.agents, "没有装配 agent"


def test_two_harnesses_from_one_environment_do_not_share_a_repo(tmp_path):
    """A caller that wants two runs must get two stacks.

    Caching the `Repo` (and the server, gateway and wiki) here would do exactly
    the thing this class exists to prevent: two runs writing one database and one
    principal slot.
    """
    env = RunEnvironment.create(tmp_path, name="case", fixture=FIXTURE)

    first, second = env.harness(), env.harness()

    assert first.repo is not second.repo
    assert first.gateway is not second.gateway
    assert first.gateway.server is not second.gateway.server


def test_an_isolated_run_refuses_every_approval_by_default(tmp_path):
    """There is no human behind a benchmark case.

    A default that approved would be a default that widens what a benchmark is
    allowed to do — and the widening would be invisible, because the case would
    simply succeed.
    """
    env = RunEnvironment.create(tmp_path, name="case", fixture=FIXTURE)

    server = env.harness().gateway.server

    assert server._approval_checker("any-run", "workspace.delete", {}) is False


def test_a_non_live_environment_forces_the_mock_brain(tmp_path):
    """What makes a *frozen* baseline possible.

    Defaulting to whatever provider is configured would mean the same case
    answered with a different brain on every machine, and the artifact compared
    against it would mean nothing.
    """
    env = RunEnvironment.create(tmp_path, name="case", live=False)

    assert env.config.llm.provider == "mock"


def test_a_live_environment_takes_the_operators_own_provider(tmp_path, monkeypatch):
    """"Run the live baseline" should mean "run the brain I have configured"."""
    from wfos.config import default_config as real_default

    configured = real_default()
    configured.llm.provider = "openai"
    configured.llm.model = "an-operator-model"
    monkeypatch.setattr("wfos.runner.load_config",
                        lambda: type("C", (), {"llm": configured.llm})())

    env = RunEnvironment.create(tmp_path, name="case", live=True)

    assert env.config.llm.provider == "openai"
    assert env.config.llm.model == "an-operator-model"


def test_two_environments_do_not_see_each_others_runs(tmp_path):
    """The property the whole thing is for, stated end to end."""
    a = RunEnvironment.create(tmp_path, name="a", fixture=FIXTURE)
    b = RunEnvironment.create(tmp_path, name="b", fixture=FIXTURE)

    ha = a.harness()
    ha.create_run("新增一个模块", kind="feature")
    asyncio.run(ha.advance(ha.repo.list_runs()[0]["id"]))

    assert ha.repo.list_runs(), "环境 a 里没有运行"
    assert b.harness().repo.list_runs() == [], "环境 b 看到了 a 的运行"
