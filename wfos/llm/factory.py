"""Adapter factory — resolve a provider name to an adapter instance.

This is also where credentials are checked, because it is the single funnel
every real adapter is built through: a check here cannot be bypassed by reaching
for an adapter class directly.
"""
from __future__ import annotations

from dataclasses import replace

from ..config import LLMConfig
from .anthropic import AnthropicAdapter
from .base import LLMAdapter, MissingCredentialError
from .gemini import GeminiAdapter
from .mock import MockAdapter
from .openai_compat import OpenAICompatAdapter
from .scripted import ScriptedAdapter

# Providers that authenticate. A local server (ollama, vLLM) does not, and
# neither does a test double — so needing a credential is the exception here,
# not the default.
_KEYED_PROVIDERS = frozenset({"openai", "openai-compatible", "anthropic", "gemini"})

# What to tell an operator to set when nothing is configured yet.
_DEFAULT_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "openai-compatible": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def _require_credential(provider: str, cfg: LLMConfig) -> None:
    """Refuse to build an authenticating adapter with no credential.

    The alternative is to send the request anyway and let the provider answer
    401. That turns a configuration mistake into an `httpx.HTTPStatusError`,
    which `is_network_error` classifies as `network_error` — so the run's audit
    trail blames the provider for a problem the operator can fix in one line.
    """
    if provider not in _KEYED_PROVIDERS or cfg.allow_missing_key or cfg.api_key:
        return
    env = cfg.api_key_env or _DEFAULT_KEY_ENV.get(provider, "<API_KEY>")
    if cfg.api_key_env:
        why = f"环境变量 {env} 已声明（api_key_env={cfg.api_key_env}）但当前为空"
    else:
        why = f"未声明 api_key_env，且环境变量 {env} 未设置"
    raise MissingCredentialError(
        f"provider {provider!r} 需要凭据，但取不到：{why}。\n"
        f"  修法一：在 wfos.toml 写 [llm] api_key_env = \"{env}\"，并导出 {env}=...\n"
        f"  修法二：直接 export {env}=...\n"
        f"  若该端点确实不需要鉴权（例如本地服务或无密钥的测试替身），"
        f"请显式声明 [llm] allow_missing_key = true —— 不自动推断，"
        f"因为猜错的代价是静默的：401 会被记成 network_error，"
        f"配置错误看起来像 provider 不稳定。")


def build_adapter(cfg: LLMConfig, *, capabilities=None) -> LLMAdapter:
    """The adapter for `cfg`, optionally remembering what this provider accepts.

    `capabilities` is a `CapabilityStore`. The Harness passes one so that a fact
    learned once — "this endpoint rejects `json_schema`" — is not rediscovered on
    every run; callers that do not (tests, the smoke runner) get per-process
    behaviour only.
    """
    provider = (cfg.provider or "mock").lower()
    if provider in ("mock", "local"):
        return MockAdapter()
    if provider in ("scripted",):
        return ScriptedAdapter()
    if provider in ("openai", "openai-compatible", "vllm"):
        _require_credential(provider, cfg)
        return OpenAICompatAdapter(cfg, capabilities=capabilities)
    if provider in ("ollama",):
        # A local server with no auth; never requires a credential.
        return OpenAICompatAdapter(cfg, json_mode=True, capabilities=capabilities)
    if provider == "anthropic":
        _require_credential(provider, cfg)
        return AnthropicAdapter(cfg)
    if provider == "gemini":
        _require_credential(provider, cfg)
        return GeminiAdapter(cfg)
    raise ValueError(f"未知的 LLM provider: {provider!r}")


def build_routed_adapters(cfg: LLMConfig, *, capabilities=None,
                          roles=()) -> tuple[LLMAdapter, dict[str, LLMAdapter]]:
    """`(the default adapter, one adapter per routed role)`.

    A role with no route — and a route naming the default model — shares the
    default adapter rather than getting a clone of it, so an unrouted
    configuration builds exactly one adapter and routing costs nothing when it is
    not used. Adapters for the same model are also shared, so the per-endpoint
    capability discovery (which lives on the adapter) is not done twice for one
    model.

    Only the model is overridden. Provider, endpoint, credential and
    `structured_output` stay shared, because a route is meant to answer "is this
    role worth a stronger model", not to become a second and partial way of
    configuring the LLM.

    An unknown role is refused rather than ignored: a route that names a role
    which does not exist would be a setting that silently does nothing, which is
    the defect this codebase keeps removing.
    """
    default = build_adapter(cfg, capabilities=capabilities)
    known = set(roles) if roles else None
    by_role: dict[str, LLMAdapter] = {}
    cache: dict[str, LLMAdapter] = {cfg.model: default}
    for role, raw in (cfg.routing or {}).items():
        if known is not None and role not in known:
            raise ValueError(
                f"[llm.routing] 中的角色 {role!r} 不存在；可选："
                f"{', '.join(sorted(known))}。拼错角色名会让这条路由悄悄不生效，"
                f"所以在这里直接拒绝。")
        model = (raw or "").strip()
        if not model or model == cfg.model:
            by_role[role] = default
            continue
        if model not in cache:
            cache[model] = build_adapter(replace(cfg, model=model),
                                         capabilities=capabilities)
        by_role[role] = cache[model]
    return default, by_role
