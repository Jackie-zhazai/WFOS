"""Configuration loading and dataclasses.

Config file: TOML (stdlib `tomllib`). Defaults live in `config.example.toml`
next to the package; a per-user override may be placed at `<data_dir>/wfos.toml`.
Environment variables win over the file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomllib

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent


@dataclass
class LLMConfig:
    provider: str = "mock"          # mock | openai | anthropic | gemini | ollama | vllm | scripted
    base_url: str | None = None
    model: str = "mock"
    api_key_env: str | None = None
    # Declares that the selected endpoint genuinely needs no credential — a
    # local server, or a keyless test double. It is explicit rather than
    # inferred from the provider name because the failure mode of guessing
    # wrong is silent: an unauthenticated request answered with 401 gets filed
    # as a network error, so a misconfiguration looks like a flaky provider.
    allow_missing_key: bool = False
    # How structured output is requested from an OpenAI-compatible endpoint:
    #   json_schema   native `response_format: {type: json_schema}` (default)
    #   json_object   the weaker OpenAI mode; some providers accept only this
    #   none          send no response_format and rely on the prompt, which
    #                 already demands the JSON, plus JSON extraction from the reply
    # DeepSeek needs `none`: it answers 400 to json_schema ("This response_format
    # type is unavailable now") and returns empty content for json_object.
    structured_output: str = "json_schema"
    temperature: float = 0.1
    # A *cap*, not a reservation: an answer that needs fewer tokens costs fewer
    # tokens, so a generous value costs nothing on small replies. It is generous
    # because reasoning models spend this budget on thinking before writing the
    # answer, and a truncated answer is unparseable — the agent loop reports that
    # as `output_truncated` so it is not mistaken for a model that cannot follow
    # the schema. Raise it for a provider whose reasoning is verbose.
    max_tokens: int = 8192
    timeout: float = 120.0
    # role -> model, e.g. {"implementer": "deepseek-reasoner"}.
    # `model` above stays the default (and the fallback for unrouted roles); this
    # only overrides which model a given agent asks for. Everything else about the
    # endpoint — provider, base_url, credential, structured_output — is shared, on
    # purpose: letting a route change the provider too would make this a second,
    # partial way to configure the LLM, and two ways to set one thing is how they
    # drift apart.
    routing: dict[str, str] = field(default_factory=dict)

    @property
    def api_key(self) -> str | None:
        if not self.api_key_env:
            return None
        return os.environ.get(self.api_key_env)


# Environment variables passed through to build/test subprocesses. Everything
# else is withheld — notably any `*_API_KEY` the harness itself holds: a test run
# has no business reading the harness's credentials, and its output lands in the
# audit trail. The Windows entries are plumbing subprocesses need to start at
# all; none of them is sensitive. Extend per project in `wfos.toml`.
DEFAULT_SHELL_ENV_ALLOWLIST = (
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "LC_CTYPE",
    "TERM", "TMPDIR", "TMP", "TEMP", "PWD",
    "PYTHONPATH", "PYTHONIOENCODING",
    "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT",
)


@dataclass
class SafeCommand:
    """A build/test command that `build.run` / `test.run` may execute.

    The command is matched against the allowlist as an exact match or as a
    whitespace-prefix match on the normalized command string. The subprocess is
    launched WITHOUT a shell; shell metacharacters are rejected outright.
    """
    cmd: str
    timeout: int = 120
    exact: bool = True


@dataclass
class HarnessConfig:
    # `max_agent_rounds` and `default_agent_timeout` used to live here. Nothing
    # ever read them — each agent declares its own `max_rounds` / `timeout`, and
    # those per-role values are deliberate (the verifier gets 240s, the curator
    # 4 rounds) — so setting either one did nothing at all. A knob that silently
    # does nothing is worse than no knob, and `tests/test_config_contract.py`
    # now fails on any field nothing reads.
    max_verify_attempts: int = 3
    # Stop a run once it has spent this much. None disables the check. Enforced
    # only when the quantity is actually measured: a brain that reports no usage
    # leaves the budget unable to bite, and a limit that silently does nothing is
    # worse than no limit. Checked between states, never inside one.
    run_token_budget: int | None = None
    run_cost_budget_usd: float | None = None
    # How much a baseline case's measured work may grow before `baseline check`
    # fails, as a fraction (0.25 = fail past +25%). None means "report it, never
    # fail on it". Measured figures move on their own — that is what measuring
    # them is for — so they are information by default and a gate only when an
    # operator says what growth is unacceptable.
    baseline_work_growth_limit: float | None = None
    # How long one execution lease on a run stays good for, in seconds. Must
    # comfortably exceed a single model call, or a slow state could have its
    # lease expire mid-flight and be re-claimed by another process. The provider
    # timeout is enforced as a floor in `Harness._lease_seconds`, so this only
    # has to be raised for providers slower than that floor.
    run_lease_seconds: int = 300
    auto_approve_low_risk_files: int = 2      # <=N files + low risk => direct execution
    # How deep child bugfix runs may nest before the parent is failed instead of
    # spawning another child. 0 = unbounded.
    max_parent_depth: int = 3
    # Bound file writes to the approved plan's file set. Out-of-plan writes are
    # refused at the tool layer and escalated to an approval.
    enforce_plan_scope: bool = True
    # Reject the Nth consecutive identical tool call (same name AND arguments).
    # 0 disables the guard.
    repeated_call_threshold: int = 2
    # How many calls to an undeclared tool are answered with a structured error
    # (carrying the tools that do exist) before the state is abandoned. 0 aborts
    # on the first one, which is what the loop did before it could retry at all.
    max_tool_retries: int = 2
    # Escalate to an approval when the same (state, failure_class) recurs this
    # many times, instead of spending the remaining retries on the same failure.
    # 0 disables escalation (retries then run to max_verify_attempts as before).
    repeated_failure_threshold: int = 2
    # Above this many files a workspace snapshot uses a size+mtime manifest
    # instead of hashing every file's content.
    snapshot_max_files: int = 2000
    # Context budgeting, in CHARACTERS (there is no tokenizer; an approximation
    # would be a number that looks precise and cannot be checked).
    prompt_total_budget: int = 30000
    tool_result_budget: int = 8000
    # How many evidence items one prompt may carry. The section budget already
    # caps evidence by characters; this caps it by count, because a run with
    # hundreds of one-line items can crowd out the plan while every item still
    # looks small. Which ones survive is decided by relevance, and how many were
    # left out is recorded in the step's `prompt_metadata.evidence`.
    evidence_limit: int = 40
    # Per-section ceilings for the reducible prompt sections; None entries fall
    # back to harness.context.DEFAULT_SECTION_BUDGETS.
    prompt_section_budgets: dict[str, int] = field(default_factory=dict)
    # Environment handed to build/test subprocesses (see the constant above).
    shell_env_allowlist: list[str] = field(
        default_factory=lambda: list(DEFAULT_SHELL_ENV_ALLOWLIST))
    # Redact secret-shaped values from untrusted content before it is persisted.
    redact_persistence: bool = True
    safe_commands: list[SafeCommand] = field(default_factory=list)

    def shell_env(self, root: str | Path) -> dict[str, str]:
        """The environment a build/test subprocess is allowed to see.

        `PWD` is forced to the project root so a command cannot read the
        harness's own working directory, and `PATH` is preserved or nothing
        would execute.
        """
        env = {name: os.environ[name] for name in self.shell_env_allowlist
               if name in os.environ}
        env["PWD"] = str(root)
        env.setdefault("PATH", os.environ.get("PATH", ""))
        return env

    def allows(self, command: str) -> SafeCommand | None:
        norm = " ".join(command.split())
        for sc in self.safe_commands:
            cand = " ".join(sc.cmd.split())
            if sc.exact:
                if norm == cand:
                    return sc
            else:
                if norm.startswith(cand):
                    return sc
        return None


@dataclass
class AppConfig:
    data_dir: Path
    project_root: Path
    db_path: Path
    wiki_path: Path
    llm: LLMConfig = field(default_factory=LLMConfig)
    harness: HarnessConfig = field(default_factory=HarnessConfig)
    # model name -> {"input": USD per 1M tokens, "output": ...}. Only models the
    # operator priced appear here; anything else yields cost = None rather than
    # a guessed number (see `wfos.metrics.cost_usd`).
    pricing: dict[str, dict[str, float]] = field(default_factory=dict)


_DEFAULTS_TOML = r"""
# wfos default configuration (TOML)
[llm]
# provider: mock | openai | anthropic | gemini | ollama | vllm | scripted
provider = "mock"
base_url = ""
model = "mock"
api_key_env = ""
# 若所选端点确实不需要鉴权（本地服务或无密钥的测试替身），显式置 true。
# 不按 provider 名推断，是因为猜错的代价是静默的：未带凭据的请求被 401 拒绝后
# 会被归类成 network_error，配置错误看起来像 provider 不稳定。
allow_missing_key = false
# json_schema | json_object | none —— DeepSeek 需要 none，见 LLMConfig 的字段注释。
# 也可由环境变量 WFOS_STRUCTURED 覆盖（它与 provider 绑定，见 _apply_env_overrides）。
structured_output = "json_schema"
temperature = 0.1
max_tokens = 8192   # 帽值不是预留；推理模型会花掉它，见 LLMConfig 注释
timeout = 120.0

# 按角色选模型：只覆盖"用哪个模型"，端点/凭据/structured_output 仍然共用。
# 角色必须真实存在（investigator / architect / implementer / verifier / curator），
# 拼错会被当场拒绝而不是悄悄不生效。未列出的角色用上面的 model。
# [llm.routing]
# implementer = "deepseek-reasoner"
# verifier    = "deepseek-chat"
# curator     = "deepseek-flash"

[harness]
max_verify_attempts = 3
# 预算上限。省略即不设限（TOML 没有 null，所以这两项不写在默认值里，缺省就是 None）。
# 只有在用量确实被上报时才生效：mock/provider 不上报用量时预算咬不到，
# 而一个悄悄不生效的上限比没有上限更糟。
# run_token_budget = 2000000
# run_cost_budget_usd = 5.0
# 基线里"做了多少活"的增长上限，比例（0.25 = 涨过 25% 即判回归）。省略即只报告不判失败。
# 只在两次比较的度量单位相同时生效（真实 provider 比 token，mock 比模型调用数）。
# baseline_work_growth_limit = 0.25
run_lease_seconds = 300
auto_approve_low_risk_files = 2
max_parent_depth = 3
enforce_plan_scope = true
repeated_call_threshold = 2
max_tool_retries = 2
repeated_failure_threshold = 2
snapshot_max_files = 2000
prompt_total_budget = 30000
tool_result_budget = 8000
evidence_limit = 40

[[harness.safe_commands]]
cmd = "python check.py"
timeout = 60
exact = true

[[harness.safe_commands]]
cmd = "python -m pytest"
timeout = 120
exact = true

[[harness.safe_commands]]
cmd = "python -m unittest"
timeout = 120
exact = true

[[harness.safe_commands]]
cmd = "pytest"
timeout = 120
exact = true

[pricing]
# Cost per 1,000,000 tokens, keyed by model name. Example:
#   [pricing."gpt-4o"]
#   input = 2.50
#   output = 10.00
# A model that is not listed here gets cost = None, never 0 — the harness does
# not guess prices, and a guessed price in the database is indistinguishable
# from a real one.
"""


def default_config() -> AppConfig:
    data_dir = Path(os.environ.get("WFOS_DATA", str(PROJECT_DIR / ".data")))
    project_root = Path(os.environ.get("WFOS_PROJECT", str(PROJECT_DIR)))
    db_path = Path(os.environ.get("WFOS_DB", str(data_dir / "wfos.db")))
    wiki_path = Path(os.environ.get("WFOS_WIKI", str(data_dir / "wiki.db")))
    llm = _llm_from_env()
    defaults = _parse_defaults()
    harness = _harness_from_toml(defaults.get("harness", {}))
    return AppConfig(data_dir=data_dir, project_root=project_root, db_path=db_path,
                     wiki_path=wiki_path, llm=llm, harness=harness,
                     pricing=_pricing_from_toml(defaults.get("pricing", {})))


def load_config(path: str | Path | None = None) -> AppConfig:
    cfg = default_config()
    src = Path(path) if path else (cfg.data_dir / "wfos.toml")
    data: dict[str, Any] = {}
    if src.exists():
        with open(src, "rb") as fh:
            data = tomllib.load(fh)
    if src != Path(_parse_defaults_path()):
        data = _merge_dict(_parse_defaults(), data)
    # File wins over defaults, but environment variables win over BOTH.
    llm = _apply_env_overrides(_llm_from_toml(data.get("llm", {})))
    harness = _harness_from_toml(data.get("harness", {}))
    if path:
        src = Path(path)
        cfg.data_dir = src.parent
    cfg.llm = llm
    cfg.harness = harness
    cfg.pricing = _pricing_from_toml(data.get("pricing", {}))
    return cfg


def _parse_defaults() -> dict[str, Any]:
    return tomllib.loads(_DEFAULTS_TOML)


def _parse_defaults_path() -> Path:
    return Path("__defaults__")


def _merge_dict(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def _optional_str(value: Any) -> str | None:
    """A config value as an optional string: absent and empty both mean "not set".

    Written out because the obvious one-liner is wrong in a way that hides: with
    `value` empty, `str(value or None) or None` evaluates `str(None)`, which is
    the truthy *string* `"None"`. That is how `base_url` came to default to
    `"None"` — so an unconfigured `provider = "openai"` requested
    `None/chat/completions` rather than the provider's real endpoint, and
    `api_key_env` came to be `"None"`, looking up an environment variable that
    can never exist.
    """
    if value is None:
        return None
    text = str(value)
    return text or None


def _optional_number(value: Any, cast):
    """A config value as an optional number: absent means "no limit".

    `0` is a real value and `None` is not — the difference between "a budget of
    zero" and "no budget" is the difference between a run that stops immediately
    and one that never stops, so the two must not collapse into each other.
    Booleans are rejected because TOML `true` is an `int` in Python and would
    otherwise silently become a budget of 1.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        return cast(value)
    except (TypeError, ValueError):
        return None


def _llm_from_toml(d: dict) -> LLMConfig:
    return LLMConfig(
        provider=str(d.get("provider", "mock")),
        base_url=_optional_str(d.get("base_url")),
        model=str(d.get("model", "mock")),
        api_key_env=_optional_str(d.get("api_key_env")),
        allow_missing_key=bool(d.get("allow_missing_key", False)),
        structured_output=str(d.get("structured_output", "json_schema")),
        temperature=float(d.get("temperature", 0.1)),
        max_tokens=int(d.get("max_tokens", 8192)),
        timeout=float(d.get("timeout", 120.0)),
        routing=_routing_from_toml(d.get("routing", {})),
    )


def _routing_from_toml(d: dict) -> dict[str, str]:
    """role -> model, dropping entries that name no model.

    An empty value is the same as not writing the line: `implementer = ""` cannot
    mean "ask for the model called empty string", so it means the default.
    """
    out: dict[str, str] = {}
    for role, model in (d or {}).items():
        name = str(model or "").strip()
        if name:
            out[str(role)] = name
    return out


def _apply_env_overrides(llm: LLMConfig) -> LLMConfig:
    """Environment variables win over both defaults and the config file."""
    if os.environ.get("WFOS_PROVIDER"):
        llm.provider = os.environ["WFOS_PROVIDER"]
    if os.environ.get("WFOS_BASE_URL"):
        llm.base_url = os.environ["WFOS_BASE_URL"]
    if os.environ.get("WFOS_MODEL"):
        llm.model = os.environ["WFOS_MODEL"]
    if os.environ.get("WFOS_API_KEY"):
        llm.api_key_env = "WFOS_API_KEY"
    if os.environ.get("WFOS_STRUCTURED"):
        # This one is *provider-dependent* rather than a tuning knob, which is why
        # it belongs in the environment alongside provider/base_url/model: the
        # mode has to change together with the endpoint (DeepSeek rejects
        # `json_schema`), and the endpoint is chosen by an environment variable.
        # Leaving it file-only made it the single LLM setting that could not be
        # set the same way as the provider it depends on.
        llm.structured_output = os.environ["WFOS_STRUCTURED"]
    return llm


def _llm_from_env() -> LLMConfig:
    return _apply_env_overrides(_llm_from_toml({}))


def _pricing_from_toml(d: dict) -> dict[str, dict[str, float]]:
    """Model name -> {"input"/"output": USD per million tokens}.

    Only models the operator actually priced appear in the result. A model that
    is missing must stay missing (cost None) rather than defaulting to zero: a
    price the harness invented is indistinguishable from a real one once it is
    in the database, and it would silently understate every aggregate.
    """
    out: dict[str, dict[str, float]] = {}
    for model, rates in (d or {}).items():
        if not isinstance(rates, dict):
            continue
        price: dict[str, float] = {}
        for side in ("input", "output"):
            raw = rates.get(side)
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                price[side] = float(raw)
        if price:
            out[str(model)] = price
    return out


def _harness_from_toml(d: dict) -> HarnessConfig:
    safe = []
    for entry in d.get("safe_commands", []):
        safe.append(SafeCommand(
            cmd=str(entry.get("cmd", "")),
            timeout=int(entry.get("timeout", 120)),
            exact=bool(entry.get("exact", True)),
        ))
    return HarnessConfig(
        max_verify_attempts=int(d.get("max_verify_attempts", 3)),
        run_token_budget=_optional_number(d.get("run_token_budget"), int),
        run_cost_budget_usd=_optional_number(d.get("run_cost_budget_usd"), float),
        baseline_work_growth_limit=_optional_number(
            d.get("baseline_work_growth_limit"), float),
        run_lease_seconds=int(d.get("run_lease_seconds", 300)),
        auto_approve_low_risk_files=int(d.get("auto_approve_low_risk_files", 2)),
        max_parent_depth=int(d.get("max_parent_depth", 3)),
        enforce_plan_scope=bool(d.get("enforce_plan_scope", True)),
        repeated_call_threshold=int(d.get("repeated_call_threshold", 2)),
        max_tool_retries=int(d.get("max_tool_retries", 2)),
        repeated_failure_threshold=int(d.get("repeated_failure_threshold", 2)),
        snapshot_max_files=int(d.get("snapshot_max_files", 2000)),
        prompt_total_budget=int(d.get("prompt_total_budget", 30000)),
        tool_result_budget=int(d.get("tool_result_budget", 8000)),
        evidence_limit=int(d.get("evidence_limit", 40)),
        prompt_section_budgets=dict(d.get("prompt_section_budgets", {}) or {}),
        shell_env_allowlist=list(d.get("shell_env_allowlist",
                                       DEFAULT_SHELL_ENV_ALLOWLIST)),
        redact_persistence=bool(d.get("redact_persistence", True)),
        safe_commands=safe,
    )
