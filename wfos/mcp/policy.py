"""Per-tool policy: roles, path roots, params, timeout, output size, side effects.

Enforcement happens in the MCP server (`on_call_tool`) and again in the gateway,
so neither a model nor a misbehaving agent can bypass the Harness.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..config import HarnessConfig

SHELL_METACHARS = re.compile(r"[;&|<>`$()\n\r]|\brm\b|\bdel\b|--force|--delete|-rf\b|\brecursive\b")


class PathEscapeError(PermissionError):
    pass


class CommandDeniedError(PermissionError):
    pass


class RoleDeniedError(PermissionError):
    pass


class ApprovalRequiredError(PermissionError):
    pass


class ParamValidationError(ValueError):
    pass


class PlanScopeError(PermissionError):
    """A write to a path outside the file set of the approved plan."""


# Stable error vocabulary for the audit trail, keyed by exception *type*.
# Classification must never read exception text: editing a message must not be
# able to silently reclassify a security event. Value is
# (error_code, security_event) — an empty security_event means "not a security
# boundary crossing", whereas "the call failed for an ordinary reason".
ERROR_CODES: dict[type, tuple[str, str]] = {
    PlanScopeError:         ("plan_scope_denied", "plan_scope_denied"),
    PathEscapeError:        ("path_escape", "path_escape"),
    RoleDeniedError:        ("role_denied", "role_denied"),
    ApprovalRequiredError:  ("approval_required", "approval_denied"),
    CommandDeniedError:     ("command_denied", ""),
    ParamValidationError:   ("invalid_arguments", ""),
}

# Non-exception codes the agent loop assigns itself.
CODE_UNKNOWN_TOOL = "unknown_tool"
CODE_REPEATED_CALL = "repeated_identical_call"
CODE_TOOL_FAILED = "tool_failed"

STATUS_OK = "ok"
STATUS_ERROR = "error"
STATUS_REJECTED = "rejected"

# Where an audited call came from: the in-process agent loop, or an external MCP
# client driving the server itself over stdio.
SOURCE_AGENT = "agent"
SOURCE_MCP = "mcp"

# Codes meaning "refused before execution" rather than "ran and failed". Kept
# explicit so a new code has to be classified deliberately rather than
# inheriting a default that silently miscounts.
REJECTION_CODES = frozenset(code for code, _ in ERROR_CODES.values()) \
    | {CODE_REPEATED_CALL, CODE_UNKNOWN_TOOL}


def status_for(error_code: str) -> str:
    return STATUS_REJECTED if error_code in REJECTION_CODES else STATUS_ERROR


def classify_error(exc: BaseException) -> tuple[str, str]:
    """Map an exception to `(error_code, security_event)` via its type hierarchy.

    Walks the MRO so a subclass is classified as itself, and anything
    unrecognised degrades to `tool_failed` rather than being mislabelled as a
    policy denial.
    """
    for klass in type(exc).__mro__:
        hit = ERROR_CODES.get(klass)
        if hit:
            return hit
    return (CODE_TOOL_FAILED, "")


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict
    allowed_roles: list[str] = field(default_factory=list)
    path_roots: list[str] = field(default_factory=list)
    read_only: bool = True
    timeout: float = 30.0
    max_output: int = 200_000
    side_effects: bool = False
    approval_required: bool = False
    # Fingerprint the workspace before/after this call to attribute what the
    # tool actually changed. Reserved for tools that edit workspace *files*:
    # a build or test run has side effects too, but its artifacts are not part
    # of the change set we attribute, and snapshotting them is pure overhead.
    tracks_changes: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
            "readOnly": self.read_only,
            "sideEffects": self.side_effects,
        }


def resolve_within_roots(path: str, roots: list[str]) -> str:
    """Resolve `path` (possibly relative or absolute) and ensure it stays inside
    one of `roots`. Rejects any `..` escape or absolute path outside roots."""
    roots = [Path(r).resolve() for r in roots]
    if not roots:
        raise PathEscapeError("没有允许的路径根目录")
    p = Path(path)
    if not p.is_absolute():
        # Try each root; pick the first where the joined realpath stays inside.
        for root in roots:
            cand = root / p
            real = Path(os.path.realpath(str(cand)))
            try:
                common = os.path.commonpath([str(real), str(root)])
            except ValueError:
                continue
            if common == str(root):
                return str(real)
        raise PathEscapeError(f"路径越界: {path!r} 超出允许根目录")
    real = Path(os.path.realpath(str(p)))
    for root in roots:
        try:
            common = os.path.commonpath([str(real), str(root)])
        except ValueError:
            continue
        if common == str(root):
            return str(real)
    raise PathEscapeError(f"路径越界: {path!r} 不在允许根目录内")


def check_write_scope(path: str, allowed: list[str] | None) -> None:
    """Ensure `path` is one of the files the approved plan says may be written.

    `allowed is None` means the plan declared no file set at all, so nothing can
    be enforced and the check passes — narrowing to an empty set would block
    every write for a run that never planned one. An empty *list* is a real
    (empty) plan and does block.
    """
    if allowed is None:
        return
    norm = path.replace("\\", "/").lstrip("./")
    if norm not in {p.replace("\\", "/").lstrip("./") for p in allowed}:
        raise PlanScopeError(
            f"写入 {path!r} 不在已批准方案的文件列表内（方案允许: {sorted(allowed)}）；"
            "越过方案范围的写入需要单独审批")


def validate_command(command: str, harness: HarnessConfig) -> tuple[list[str], int]:
    """Check a build/test command against the allowlist.

    Returns (argv, timeout) or raises CommandDeniedError.
    """
    if not command or not command.strip():
        raise CommandDeniedError("空命令被拒绝")
    if SHELL_METACHARS.search(command):
        raise CommandDeniedError(f"命令含 shell 元字符或危险操作，被拒绝: {command!r}")
    sc = harness.allows(command)
    if sc is None:
        raise CommandDeniedError(f"命令不在安全白名单内，被拒绝: {command!r}")
    import shlex
    argv = shlex.split(command, posix=False)
    if not argv:
        raise CommandDeniedError("无法解析命令")
    return argv, sc.timeout


def validate_params(args: dict, schema: dict) -> dict:
    """Validate required params and types against a JSON-schema-like dict.

    Supports: type (string|integer|boolean|array|object), enum, minLength,
    maxLength, items. Extra keys are allowed (ignored) for forward compat.
    """
    if not isinstance(args, dict):
        raise ParamValidationError("工具参数必须是对象")
    props = schema.get("properties", {})
    required = schema.get("required", [])
    for name in required:
        if name not in args:
            raise ParamValidationError(f"缺少必填参数: {name}")
    for name, value in args.items():
        spec = props.get(name)
        if not spec:
            continue
        ptype = spec.get("type")
        if ptype == "string" and not isinstance(value, str):
            raise ParamValidationError(f"参数 {name} 需要字符串")
        if ptype == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
            raise ParamValidationError(f"参数 {name} 需要整数")
        if ptype == "boolean" and not isinstance(value, bool):
            raise ParamValidationError(f"参数 {name} 需要布尔")
        if ptype == "array":
            if not isinstance(value, list):
                raise ParamValidationError(f"参数 {name} 需要数组")
            item_type = (spec.get("items") or {}).get("type")
            if item_type == "string" and any(not isinstance(v, str) for v in value):
                raise ParamValidationError(f"参数 {name} 需要字符串数组")
        if isinstance(value, str):
            min_len = spec.get("minLength")
            max_len = spec.get("maxLength")
            if min_len is not None and len(value) < min_len:
                raise ParamValidationError(f"参数 {name} 过短")
            if max_len is not None and len(value) > max_len:
                raise ParamValidationError(f"参数 {name} 过长")
            enum = spec.get("enum")
            if enum and value not in enum:
                raise ParamValidationError(f"参数 {name} 不在允许取值内: {value!r}")
    return args
