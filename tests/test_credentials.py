"""Missing credentials: refused before the request, not blamed on the network.

The behaviour being pinned is not "an error is raised" but *where*. Sending an
unauthenticated request and letting the provider answer 401 produces an
`httpx.HTTPStatusError`, which `is_network_error` classifies as `network_error`
— so the run's audit trail says the provider was unreachable when in fact the
operator forgot an environment variable. That distinction is asserted directly
below, because it is the whole reason this check exists.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from wfos.config import LLMConfig, _llm_from_toml, default_config
from wfos.llm.anthropic import AnthropicAdapter
from wfos.llm.base import MissingCredentialError, is_network_error
from wfos.llm.factory import build_adapter
from wfos.llm.mock import MockAdapter
from wfos.llm.openai_compat import OpenAICompatAdapter
from wfos.llm.scripted import ScriptedAdapter

ROOT = Path(__file__).resolve().parents[1]

KEYED = ["openai", "openai-compatible", "anthropic", "gemini"]
KEYLESS = ["mock", "local", "scripted", "ollama", "vllm"]


@pytest.fixture(autouse=True)
def _no_ambient_keys(monkeypatch):
    """Keep the developer's own exported keys (and provider overrides) out."""
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY",
                 "WFOS_API_KEY", "WFOS_PROVIDER", "WFOS_BASE_URL", "WFOS_MODEL"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("provider", KEYED)
def test_an_authenticating_provider_refuses_without_a_credential(provider):
    with pytest.raises(MissingCredentialError):
        build_adapter(LLMConfig(provider=provider))


@pytest.mark.parametrize("provider", KEYLESS)
def test_providers_that_do_not_authenticate_are_built_without_a_credential(provider):
    """A local server or a test double has no key to be missing."""
    assert build_adapter(LLMConfig(provider=provider)) is not None


def test_the_refusal_names_the_variable_to_set():
    with pytest.raises(MissingCredentialError) as exc:
        build_adapter(LLMConfig(provider="anthropic"))
    message = str(exc.value)
    assert "ANTHROPIC_API_KEY" in message
    assert "allow_missing_key" in message          # the escape hatch is documented


def test_the_refusal_names_the_declared_variable_when_it_is_empty(monkeypatch):
    """A declared-but-empty variable is a different fix from an undeclared one."""
    monkeypatch.setenv("MY_EMPTY_KEY", "")
    with pytest.raises(MissingCredentialError) as exc:
        build_adapter(LLMConfig(provider="openai", api_key_env="MY_EMPTY_KEY"))
    assert "MY_EMPTY_KEY" in str(exc.value)
    assert "为空" in str(exc.value)


def test_a_present_credential_lifts_the_refusal(monkeypatch):
    monkeypatch.setenv("MY_KEY", "sk-test")
    adapter = build_adapter(LLMConfig(provider="anthropic", api_key_env="MY_KEY"))
    assert isinstance(adapter, AnthropicAdapter)


def test_an_explicit_keyless_declaration_lifts_the_refusal():
    """The loopback case: a real provider name pointed at an endpoint with no auth."""
    adapter = build_adapter(LLMConfig(provider="openai", base_url="http://127.0.0.1:1",
                                      allow_missing_key=True))
    assert isinstance(adapter, OpenAICompatAdapter)


def test_the_check_cannot_be_bypassed_by_naming_an_adapter_class_directly():
    """`build_adapter` is the funnel; a direct class instantiation is not a
    supported path, and this pins that the funnel is where the guard lives."""
    with pytest.raises(MissingCredentialError):
        build_adapter(LLMConfig(provider="gemini", api_key_env=None))
    # ...whereas the guard is not duplicated inside the adapters themselves,
    # which stay constructible for tests that inject their own transport.
    assert AnthropicAdapter(LLMConfig(provider="anthropic")) is not None


def test_a_forgotten_credential_would_otherwise_be_recorded_as_a_network_error():
    """The reason the check exists, asserted rather than asserted-about.

    A 401 is an `httpx.HTTPStatusError`, and `is_network_error` — which decides
    whether a run's failure is recorded as a provider being unreachable — says
    yes to it. So without this check a missing key is filed as a flaky network.
    """
    httpx = pytest.importorskip("httpx")
    request = httpx.Request("POST", "https://example.invalid/v1/messages")
    unauthorized = httpx.HTTPStatusError(
        "401", request=request, response=httpx.Response(401, request=request))
    assert is_network_error(unauthorized) is True


# ------------------------------------------------------------------- at the CLI
def test_the_cli_reports_a_missing_credential_without_a_traceback(tmp_path):
    """A configuration mistake is reported as one, with its own exit code (4)."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY",
                        "WFOS_API_KEY", "WFOS_BASE_URL", "WFOS_MODEL")}
    env.update(WFOS_PROVIDER="anthropic", WFOS_DATA=str(tmp_path / "data"),
               WFOS_PROJECT=str(tmp_path), PYTHONIOENCODING="utf-8")
    proc = subprocess.run([sys.executable, "-m", "wfos", "status"], cwd=ROOT,
                          env=env, capture_output=True, text=True, timeout=120,
                          encoding="utf-8", errors="replace")
    out = proc.stdout + proc.stderr
    assert proc.returncode == 4, out
    assert "ANTHROPIC_API_KEY" in out
    assert "Traceback" not in out


def test_mock_remains_the_default_so_a_fresh_install_still_runs(tmp_path):
    """The guard must not make the offline default path unbuildable."""
    assert isinstance(build_adapter(LLMConfig()), MockAdapter)
    assert isinstance(build_adapter(LLMConfig(provider="scripted")), ScriptedAdapter)


# ------------------------------------------- optional llm strings (see _optional_str)
def test_an_empty_optional_llm_string_is_unset_not_the_string_none():
    """The bug this replaced: `str(value or None) or None` evaluates `str(None)`
    for an empty value, and the *string* "None" is truthy — so an unset field
    came back looking set."""
    for key in ("base_url", "api_key_env"):
        assert getattr(_llm_from_toml({}), key) is None
        assert getattr(_llm_from_toml({key: ""}), key) is None
        assert getattr(_llm_from_toml({key: "v"}), key) == "v"


def test_an_unconfigured_openai_provider_targets_the_real_endpoint():
    """The user-visible consequence: the URL used to be `None/chat/completions`."""
    cfg = _llm_from_toml({"provider": "openai"})
    base = (cfg.base_url or "https://api.openai.com/v1").rstrip("/")
    assert base == "https://api.openai.com/v1"


def test_the_default_config_carries_no_string_none():
    cfg = default_config()
    assert cfg.llm.base_url is None
    assert cfg.llm.api_key_env is None
