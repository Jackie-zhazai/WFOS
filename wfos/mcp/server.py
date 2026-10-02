"""MCP server exposing the WFOS tool suite with policy enforcement.

All tools resolve paths inside the configured roots, honor per-tool roles,
timeouts and output caps, and declare side effects. No unrestricted shell is
ever exposed: build/test only run allowlisted commands.
"""
from __future__ import annotations

import asyncio
import glob as _glob
import json
import os
import re
from collections.abc import Callable
from typing import Any

from mcp.server.lowlevel import Server as McpServer
from mcp.types import CallToolRequestParams, CallToolResult, ListToolsResult, TextContent, Tool

from ..config import AppConfig
from ..storage.repo import Repo
from .policy import (
    CODE_TOOL_FAILED,
    SOURCE_MCP,
    STATUS_OK,
    ApprovalRequiredError,
    CommandDeniedError,
    ParamValidationError,
    PathEscapeError,
    PlanScopeError,
    RoleDeniedError,
    ToolSpec,
    check_write_scope,
    classify_error,
    resolve_within_roots,
    status_for,
    validate_command,
    validate_params,
)
from .snapshot import affected_paths, capture_tree, diff_snapshots

ApprovalChecker = Callable[[str, str, dict], bool]  # (run_id, tool, args) -> approved?

# Roles an external stdio client may assume.
EXTERNAL_ROLES = ("investigator", "architect", "implementer", "verifier", "curator")

# The `_meta` key a per-call principal travels under. Namespaced because the MCP
# `_meta` map also carries the SDK's own reserved `io.modelcontextprotocol/*`
# keys, and a bare "principal" would be a name this project does not own.
PRINCIPAL_META_KEY = "wfos/principal"


class WfosMcpServer:
    def __init__(self, config: AppConfig, repo: Repo,
                 approval_checker: ApprovalChecker | None = None,
                 delegate: Callable[[dict, dict], Any] | None = None):
        self._config = config
        self._repo = repo
        self._approval_checker = approval_checker or (lambda run_id, tool, args: False)
        # The Harness injects this: spawning and advancing a run is its job, not
        # the tool layer's. Left unset on a standalone stdio server, where the
        # call is refused rather than silently doing nothing.
        self._delegate = delegate
        self.principal: dict[str, Any] = {"role": None, "agent": None, "run_id": None,
                                          "read_roots": [], "write_roots": [],
                                          # None = the run's plan declared no file
                                          # set, so no scope narrowing is applied.
                                          "allowed_write_paths": None}
        # Flipped on by run_stdio(): an in-process agent is audited by the
        # ToolGateway, an external MCP session has no gateway and is audited here.
        self._audit_in_server = False
        self.specs = self._build_specs()
        self._server = McpServer(
            "wfos",
            on_list_tools=self._on_list_tools,
            on_call_tool=self._on_call_tool,
        )

    def set_delegate(self, handler: Callable[[dict, dict], Any] | None) -> None:
        """Wire the Harness's delegate handler in.

        A setter rather than a constructor argument because the Harness needs
        the gateway, and the gateway needs this server — constructing the pair
        either way round leaves one of them needing the other first.
        """
        self._delegate = handler

    # ------------------------------------------------------------- tool specs
    def _build_specs(self) -> dict[str, ToolSpec]:
        root = [str(self._config.project_root.resolve())]
        ALL = ["investigator", "architect", "implementer", "verifier", "curator", "harness"]
        READ = ALL
        WRITERS = ["implementer"]
        s = {}
        s["workspace.list_files"] = ToolSpec(
            "workspace.list_files", "按 glob 列出项目内文件（只读）",
            {"type": "object", "properties": {"pattern": {"type": "string", "minLength": 1},
                                              "path": {"type": "string"}}},
            allowed_roles=READ, path_roots=root, read_only=True, timeout=15, max_output=100_000)
        s["workspace.search"] = ToolSpec(
            "workspace.search", "在项目内做正则全文搜索（只读）",
            {"type": "object", "properties": {"pattern": {"type": "string", "minLength": 1},
                                              "path": {"type": "string"},
                                              "glob_filter": {"type": "string"}}},
            allowed_roles=READ, path_roots=root, read_only=True, timeout=30, max_output=150_000)
        s["workspace.read"] = ToolSpec(
            "workspace.read", "读取项目内文件内容（只读，受输出上限保护）",
            {"type": "object", "properties": {"path": {"type": "string", "minLength": 1},
                                              "offset": {"type": "integer"},
                                              "limit": {"type": "integer"}}},
            allowed_roles=READ, path_roots=root, read_only=True, timeout=15, max_output=100_000)
        s["workspace.patch"] = ToolSpec(
            "workspace.patch", "对项目内文件应用 before/after 文本替换（写）",
            {"type": "object", "properties": {"path": {"type": "string", "minLength": 1},
                                              "before": {"type": "string"},
                                              "after": {"type": "string"}},
             "required": ["path"]},
            allowed_roles=WRITERS, path_roots=root, read_only=False, timeout=20,
            max_output=20_000, side_effects=True, tracks_changes=True)
        s["workspace.write"] = ToolSpec(
            "workspace.write", "创建/覆盖项目内文件（写，仅 Implementer）",
            {"type": "object", "properties": {"path": {"type": "string", "minLength": 1},
                                              "content": {"type": "string"},
                                              "create_only": {"type": "boolean"}},
             "required": ["path"]},
            allowed_roles=WRITERS, path_roots=root, read_only=False, timeout=20,
            max_output=20_000, side_effects=True, tracks_changes=True)
        s["workspace.delete"] = ToolSpec(
            "workspace.delete", "删除项目内文件（写，必须已获审批）",
            {"type": "object", "properties": {"path": {"type": "string", "minLength": 1}},
             "required": ["path"]},
            allowed_roles=WRITERS, path_roots=root, read_only=False, timeout=20,
            max_output=20_000, side_effects=True, approval_required=True,
            tracks_changes=True)
        s["git.status"] = ToolSpec(
            "git.status", "查看 Git 工作区状态（只读）",
            {"type": "object", "properties": {}}, allowed_roles=READ, path_roots=root,
            read_only=True, timeout=20, max_output=100_000)
        s["git.diff"] = ToolSpec(
            "git.diff", "查看 Git 差异（只读）",
            {"type": "object", "properties": {"path": {"type": "string"}}},
            allowed_roles=READ, path_roots=root, read_only=True, timeout=20, max_output=150_000)
        s["git.log"] = ToolSpec(
            "git.log", "查看 Git 提交历史（只读）",
            {"type": "object", "properties": {"limit": {"type": "integer", "maximum": 200}}},
            allowed_roles=READ, path_roots=root, read_only=True, timeout=20, max_output=100_000)
        s["build.check"] = ToolSpec(
            "build.check", "进程内语法构建检查：对列出的 .py 文件执行 compile()",
            {"type": "object", "properties": {"files": {"type": "array", "items": {"type": "string"}}}},
            allowed_roles=["verifier"], path_roots=root, read_only=True, timeout=30,
            max_output=100_000, side_effects=False)
        s["build.run"] = ToolSpec(
            "build.run", "运行白名单内的构建命令（有副作用：产生构建产物）",
            {"type": "object", "properties": {"command": {"type": "string", "minLength": 1}},
             "required": ["command"]},
            allowed_roles=["verifier"], path_roots=root, read_only=False, timeout=120,
            max_output=200_000, side_effects=True)
        s["test.run"] = ToolSpec(
            "test.run", "运行白名单内的测试命令并解析 PASS/FAIL（有副作用）",
            {"type": "object", "properties": {"command": {"type": "string", "minLength": 1}},
             "required": ["command"]},
            allowed_roles=["verifier"], path_roots=root, read_only=False, timeout=120,
            max_output=200_000, side_effects=True)
        s["logs.read"] = ToolSpec(
            "logs.read", "读取日志文件，可按关键字过滤或取尾部（只读）",
            {"type": "object", "properties": {"path": {"type": "string", "minLength": 1},
                                              "keyword": {"type": "string"},
                                              "tail": {"type": "boolean"},
                                              "lines": {"type": "integer", "maximum": 2000}}},
            allowed_roles=READ, path_roots=root, read_only=True, timeout=20, max_output=150_000)
        s["wiki.search"] = ToolSpec(
            "wiki.search", "检索 LLM Wiki（权威/案例/候选），元数据过滤 + 全文检索",
            {"type": "object", "properties": {"query": {"type": "string", "minLength": 1},
                                              "kind": {"type": "string", "enum": ["authoritative", "case", "candidate"]},
                                              "tags": {"type": "string"},
                                              "limit": {"type": "integer", "maximum": 50}}},
            allowed_roles=READ, path_roots=[], read_only=True, timeout=15, max_output=100_000)
        s["wiki.add_candidate"] = ToolSpec(
            "wiki.add_candidate", "提交候选知识（不可直接写权威知识，仅 Curator）",
            {"type": "object", "properties": {"title": {"type": "string", "minLength": 1},
                                              "content": {"type": "string", "minLength": 1},
                                              "tags": {"type": "array", "items": {"type": "string"}},
                                              "run_id": {"type": "string"}},
             "required": ["title", "content"]},
            allowed_roles=["curator", "harness"], path_roots=[], read_only=False, timeout=15,
            max_output=20_000, side_effects=True)
        s["delegate"] = ToolSpec(
            "delegate",
            "把一个子任务交给独立的子流程运行。子流程有自己的预算、审计与审批；"
            "结果是结构化的，不是一段文字。深度用尽后本工具不再出现。",
            {"type": "object", "properties": {"task": {"type": "string", "minLength": 4},
                                              "kind": {"type": "string",
                                                       "enum": ["feature", "bugfix"]}},
             "required": ["task"]},
            allowed_roles=["investigator", "architect", "implementer", "verifier"],
            path_roots=[], read_only=False, timeout=900, max_output=20_000,
            side_effects=True)
        s["echo"] = ToolSpec(
            "echo", "回显工具（测试用）",
            {"type": "object", "properties": {"text": {"type": "string"}}},
            allowed_roles=READ, path_roots=[], read_only=True, timeout=10, max_output=10_000)
        return s

    # -------------------------------------------------------------- MCP hooks
    async def _on_list_tools(self, ctx, params=None) -> ListToolsResult:
        """Advertise the full tool suite; the policy gate decides per call.

        The list is not filtered by role: what a caller may actually do is
        enforced at call time (role, path roots, plan scope, approval), so the
        catalogue stays a stable, complete description of the system.
        """
        return ListToolsResult(tools=[Tool(name=sp.name, description=sp.description,
                                           inputSchema=sp.input_schema)
                                      for sp in self.specs.values()])

    def _principal_for(self, params: CallToolRequestParams) -> dict[str, Any]:
        """The principal *this* call runs under.

        In-process agents send their own on every call, so a tool call is never
        authorised by whichever run last set the server's slot. That slot was a
        real defect rather than a theoretical race: `delegate` drives a child run
        *inside* the parent's tool call, and every child state re-set the
        principal — so the parent's next call in that state was role-checked as
        the child's last role (usually `curator`), scope-checked against the
        child's approved file set, and audited to the **child's run_id**. The
        parent's own change set silently lost it.

        Only honoured for an in-process caller. `_meta` is caller-supplied, and an
        external stdio client that could name its own role and run id would have
        been handed exactly the authority `run_stdio` exists to withhold — it
        fixes one principal for the whole session, at connect time.
        """
        if self._audit_in_server:
            return self.principal
        meta = getattr(params, "meta", None) or {}
        supplied = meta.get(PRINCIPAL_META_KEY)
        if isinstance(supplied, dict) and supplied.get("role"):
            return supplied
        return self.principal

    async def _on_call_tool(self, ctx, params: CallToolRequestParams) -> CallToolResult:
        name = params.name
        args = params.arguments or {}
        spec = self.specs.get(name)
        if spec is None:
            return self._err(f"未知工具: {name}")
        principal = self._principal_for(params)
        role = principal.get("role")
        run_id = principal.get("run_id") or ""
        validated = args
        before: dict[str, Any] | None = None
        try:
            validated = validate_params(args, spec.input_schema)
            if role not in spec.allowed_roles:
                raise RoleDeniedError(
                    f"角色 {role!r} 无权调用 {name}（允许: {spec.allowed_roles}）")
            if spec.approval_required and not self._approval_checker(run_id, name, validated):
                raise ApprovalRequiredError(f"工具 {name} 需要已批准的审批记录")
            # File-writing tools are attributed by diffing the workspace around
            # the call; a model's self-reported change list is never treated as
            # evidence of what actually changed. The same call site is where the
            # approved plan's file set bounds what may be written at all.
            if spec.tracks_changes:
                try:
                    check_write_scope(validated.get("path", ""),
                                      principal.get("allowed_write_paths"))
                except PlanScopeError:
                    # An approval for exactly this tool+path widens the scope for
                    # that one file — the escape hatch a human opens, and the
                    # only way an out-of-plan write can ever reach disk.
                    if not self._approval_checker(run_id, name, validated):
                        raise
                before = await self._snapshot()
            handler = getattr(self, f"_impl_{name.replace('.', '_')}")
            result = await asyncio.wait_for(handler(validated), timeout=spec.timeout)
            attribution = await self._attribute(before)
            if attribution:
                result = {**result, **attribution}
            self._audit(run_id, role, name, validated, True, None, attribution,
                        status=STATUS_OK)
            return self._ok(result, spec.max_output)
        except PathEscapeError as e:
            return await self._fail(run_id, role, name, validated,
                                    f"拒绝路径越界: {e}", before, e)
        except PlanScopeError as e:
            return await self._fail(run_id, role, name, validated,
                                    f"方案范围拒绝: {e}", before, e)
        except (RoleDeniedError, CommandDeniedError, ApprovalRequiredError) as e:
            return await self._fail(run_id, role, name, validated, f"策略拒绝: {e}", before, e)
        except ParamValidationError as e:
            return await self._fail(run_id, role, name, validated, f"参数错误: {e}", before, e)
        except asyncio.TimeoutError:
            return await self._fail(
                run_id, role, name, validated, f"工具 {name} 超时（>{spec.timeout}s）",
                before)
        except Exception as e:  # noqa: BLE001
            return await self._fail(run_id, role, name, validated,
                                    f"工具执行失败: {type(e).__name__}: {e}", before, e)

    # -------------------------------------------------- change attribution
    async def _snapshot(self) -> dict[str, Any]:
        """Fingerprint the project tree without blocking the event loop."""
        return await asyncio.to_thread(
            capture_tree, self._root(),
            max_files=self._config.harness.snapshot_max_files)

    async def _attribute(self, before: dict[str, Any] | None) -> dict[str, Any]:
        """Diff the workspace against `before`; {} when nothing was snapshotted.

        A timed-out or failed handler is still attributed: it may have written
        files before it died, and reporting "no changes" would be a guess.
        """
        if before is None:
            return {}
        diff = diff_snapshots(before, await self._snapshot())
        return {"affected_paths": affected_paths(diff), "diff_summary": diff["summary"]}

    def _audit(self, run_id: str, role: str | None, name: str, args: dict,
               ok: bool, error: str | None, attribution: dict[str, Any], *,
               status: str | None = None, error_code: str | None = None,
               security_event: str | None = None) -> None:
        """Persist an audit row — but only when the server owns auditing.

        Serving an in-process agent, every call is already logged by the
        ToolGateway; only a self-driven stdio session has no gateway to do it.
        `agent` stays a plain role name and `source` records where the call came
        from, so an external client is distinguishable without re-encoding the
        role.
        """
        if not self._audit_in_server:
            return
        self._repo.log_tool_call(
            run_id, str(role or "external"), name, args, ok, error,
            affected_paths=attribution.get("affected_paths"),
            diff_summary=attribution.get("diff_summary"),
            status=status, error_code=error_code, security_event=security_event,
            source=SOURCE_MCP)

    async def _fail(self, run_id: str, role: str | None, name: str, args: dict,
                    message: str, before: dict[str, Any] | None = None,
                    exc: BaseException | None = None) -> CallToolResult:
        """Refuse a call, classifying it from the exception *type*."""
        attribution = await self._attribute(before)
        code, security = classify_error(exc) if exc is not None else (CODE_TOOL_FAILED, "")
        self._audit(run_id, role, name, args, False, message, attribution,
                    status=status_for(code), error_code=code, security_event=security)
        return self._err_payload({"ok": False, "error": message, "error_code": code,
                                  "security_event": security, **attribution})

    # ---------------------------------------------------------- MCP responses
    def _ok(self, data: dict, max_output: int) -> CallToolResult:
        text = json.dumps(data, ensure_ascii=False, default=str)
        if len(text) > max_output:
            text = text[:max_output] + "\n... (输出已截断)"
        return CallToolResult(content=[TextContent(type="text", text=text)])

    def _err_payload(self, payload: dict) -> CallToolResult:
        return CallToolResult(
            isError=True,
            content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))],
        )

    def _err(self, message: str) -> CallToolResult:
        return self._err_payload({"ok": False, "error": message})

    # ------------------------------------------------------------ transports
    async def run_stdio(self, *, role: str = "investigator",
                        run_id: str = "external") -> None:
        """Serve this tool suite to an external MCP client over stdio.

        The principal is fixed for the session: the client gets exactly the
        tool surface of `role` (read-only unless the role can write), and every
        call is audited by the server because no ToolGateway is in the path.
        Approvals still come from the database, so a gated write blocks until a
        human runs `wfos approve`.
        """
        from mcp.server.stdio import stdio_server

        self.principal = {
            "role": role, "agent": role, "run_id": run_id,
            "read_roots": [self._root()],
            "write_roots": [self._root()] if role == "implementer" else [],
            # An external client is not bound to a state machine's approved plan.
            "allowed_write_paths": None,
        }
        self._audit_in_server = True
        async with stdio_server() as (read_stream, write_stream):
            await self._server.run(read_stream, write_stream,
                                   self._server.create_initialization_options())

    # ------------------------------------------------------------ path helpers
    def _root(self) -> str:
        return str(self._config.project_root.resolve())

    def _resolve(self, path: str, *, write: bool = False) -> str:
        roots = self.principal.get("write_roots") or [] if write else self.principal.get("read_roots") or []
        if not roots:
            roots = [self._root()]
        return resolve_within_roots(path, roots)

    def _rel(self, abs_path: str) -> str:
        try:
            return os.path.relpath(abs_path, self._root())
        except ValueError:
            return abs_path

    def _shell_env(self) -> dict[str, str]:
        """The allowlisted environment for build/test subprocesses.

        Without this the child inherits the harness's whole environment,
        including its own provider credentials — which then flow back into the
        tool result and into the audit trail.
        """
        return self._config.harness.shell_env(self._root())

    # ------------------------------------------------------------ workspace
    async def _impl_workspace_list_files(self, args: dict) -> dict:
        pattern = args.get("pattern", "**/*")
        base = self._resolve(args.get("path", ".") or ".")
        base_real = os.path.realpath(base)
        full = os.path.join(base_real, pattern)
        files = [self._rel(p) for p in _glob.glob(full, recursive=True) if os.path.isfile(p)]
        files.sort()
        return {"files": files}

    async def _impl_workspace_search(self, args: dict) -> dict:
        pattern = args["pattern"]
        base = self._resolve(args.get("path", ".") or ".")
        glob_filter = args.get("glob_filter", "*")
        rx = re.compile(pattern)
        matches = []
        full = os.path.join(os.path.realpath(base), "**", glob_filter)
        for p in _glob.glob(full, recursive=True):
            if not os.path.isfile(p):
                continue
            if _is_binary(p):
                continue
            try:
                with open(p, encoding="utf-8", errors="replace") as fh:
                    for ln, line in enumerate(fh, 1):
                        if rx.search(line):
                            matches.append({"path": self._rel(p), "line": ln,
                                            "text": line.rstrip()[:300]})
                            if len(matches) >= 500:
                                return {"matches": matches, "truncated": True}
            except OSError:
                continue
        return {"matches": matches}

    async def _impl_workspace_read(self, args: dict) -> dict:
        path = self._resolve(args["path"])
        offset = args.get("offset") or 0
        limit = args.get("limit")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"文件不存在: {self._rel(path)}")
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        if offset or limit is not None:
            lines = lines[offset:(offset + limit) if limit else None]
        content = "".join(lines)
        return {"path": self._rel(path), "content": content[:95_000], "size": len(lines)}

    async def _impl_workspace_patch(self, args: dict) -> dict:
        path = self._resolve(args["path"], write=True)
        before = args.get("before") or ""
        after = args.get("after") or ""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"文件不存在: {self._rel(path)}")
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        if before:
            # Exact-match-once: an ambiguous anchor means we cannot know which
            # occurrence the caller meant, so refuse instead of editing the
            # first one and reporting success.
            count = text.count(before)
            if count == 0:
                raise ValueError(f"before 片段未命中: {before[:80]!r}")
            if count > 1:
                raise ValueError(
                    f"before 片段在文件中出现 {count} 次，无法确定唯一替换位置；"
                    f"请扩大上下文使其唯一: {before[:80]!r}")
            idx = text.find(before)
            new = text[:idx] + after + text[idx + len(before):]
            changed = new != text
        else:
            new = text + after
            changed = True
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(new)
        return {"path": self._rel(path), "changed": changed, "size": len(new)}

    async def _impl_workspace_write(self, args: dict) -> dict:
        path = self._resolve(args["path"], write=True)
        content = args.get("content") or ""
        create_only = bool(args.get("create_only"))
        existed = os.path.exists(path)
        if create_only and existed:
            raise ValueError(f"文件已存在且 create_only=true: {self._rel(path)}")
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(content)
        return {"path": self._rel(path), "written": True, "created": not existed,
                "size": len(content)}

    async def _impl_workspace_delete(self, args: dict) -> dict:
        path = self._resolve(args["path"], write=True)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"文件不存在: {self._rel(path)}")
        os.remove(path)
        return {"path": self._rel(path), "deleted": True}

    # ------------------------------------------------------------------- git
    async def _impl_git_status(self, args: dict) -> dict:
        return await self._git(["status", "--porcelain"])

    async def _impl_git_diff(self, args: dict) -> dict:
        cmd = ["diff"]
        if args.get("path"):
            p = self._resolve(args["path"])
            cmd += ["--", p]
        return await self._git(cmd)

    async def _impl_git_log(self, args: dict) -> dict:
        n = int(args.get("limit", 20))
        return await self._git(["log", "--oneline", "-n", str(n)])

    async def _git(self, cmd: list[str]) -> dict:
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "-C", self._root(), *cmd,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            out, err = await proc.communicate()
            return {"ok": proc.returncode == 0, "output": out.decode("utf-8", "replace"),
                    "error": err.decode("utf-8", "replace")}
        except FileNotFoundError:
            return {"ok": False, "output": "", "error": "git 不可用"}

    # ------------------------------------------------------------- build/test
    async def _impl_build_check(self, args: dict) -> dict:
        files = args.get("files")
        if not files:
            listing = await self._impl_workspace_list_files({"pattern": "**/*.py"})
            files = listing["files"]
        errors = []
        checked = []
        for rel in files:
            try:
                path = self._resolve(rel)
            except PathEscapeError:
                errors.append({"file": rel, "error": "路径越界"})
                continue
            if not os.path.isfile(path):
                errors.append({"file": rel, "error": "文件不存在"})
                continue
            with open(path, encoding="utf-8", errors="replace") as fh:
                src = fh.read()
            try:
                compile(src, path, "exec")
                checked.append(rel)
            except SyntaxError as e:
                # Strings, matching BuildResult.errors. Emitting dicts here made
                # every real syntax error fail schema validation in the Verifier
                # — the one path that is supposed to *report* a broken build.
                errors.append(f"{rel}:{e.lineno}: {e.msg}")
        return {"ok": not errors, "errors": errors, "files_checked": checked}

    async def _impl_build_run(self, args: dict) -> dict:
        argv, timeout = validate_command(args["command"], self._config.harness)
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=self._root(), env=self._shell_env(),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return {"ok": False, "exit_code": -1, "output": "命令超时", "command": args["command"]}
        text = out.decode("utf-8", "replace")
        return {"ok": proc.returncode == 0, "exit_code": proc.returncode,
                "output": text[-180_000:], "command": args["command"]}

    async def _impl_test_run(self, args: dict) -> dict:
        argv, timeout = validate_command(args["command"], self._config.harness)
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=self._root(), env=self._shell_env(),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return {"ok": False, "exit_code": -1, "passed": 0, "failed": 1, "skipped": 0,
                    "cases": [], "output": "测试超时", "command": args["command"]}
        text = out.decode("utf-8", "replace")
        cases = []
        passed = failed = skipped = 0
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("PASS:"):
                cases.append({"name": line[5:].strip(), "ok": True, "detail": ""})
                passed += 1
            elif line.startswith("FAIL:"):
                cases.append({"name": line[5:].strip(), "ok": False, "detail": ""})
                failed += 1
            elif line.startswith("SKIP:"):
                cases.append({"name": line[5:].strip(), "ok": True, "detail": "skipped"})
                skipped += 1
        if not cases:
            failed = 1 if proc.returncode != 0 else 0
            passed = 0 if proc.returncode != 0 else 1
        return {"ok": proc.returncode == 0 and failed == 0, "exit_code": proc.returncode,
                "passed": passed, "failed": failed, "skipped": skipped, "cases": cases,
                "output": text[-180_000:], "command": args["command"]}

    # ------------------------------------------------------------------- logs
    async def _impl_logs_read(self, args: dict) -> dict:
        path = self._resolve(args["path"])
        keyword = args.get("keyword")
        tail = bool(args.get("tail"))
        lines = int(args.get("lines", 200))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"日志文件不存在: {self._rel(path)}")
        with open(path, encoding="utf-8", errors="replace") as fh:
            all_lines = fh.readlines()
        if keyword:
            all_lines = [ln for ln in all_lines if keyword in ln]
        all_lines = all_lines[-lines:] if tail else all_lines[:lines]
        return {"path": self._rel(path), "lines": len(all_lines),
                "content": "".join(all_lines)[:140_000]}

    # ------------------------------------------------------------------- wiki
    async def _impl_wiki_search(self, args: dict) -> dict:
        results = self._repo.search_wiki(
            args["query"], kind=args.get("kind"), tags=args.get("tags"),
            limit=int(args.get("limit", 10)))
        return {"results": results}

    async def _impl_wiki_add_candidate(self, args: dict) -> dict:
        wid = self._repo.add_wiki(
            "candidate", args["title"], args["content"],
            tags=args.get("tags") or [], run_id=args.get("run_id") or "",
            status="pending", verified=False)
        return {"id": wid, "kind": "candidate", "status": "pending"}

    async def _impl_delegate(self, args: dict) -> dict:
        """Hand a sub-task to a child run, through the Harness.

        The tool layer cannot spawn or advance a run — that is the Harness's job —
        so it calls the handler the Harness injected. With none (a standalone
        stdio server) the call is refused with a code rather than returning an
        empty success, which would look like a delegation that did nothing.
        """
        if self._delegate is None:
            return {"ok": False, "error_code": "delegate_unavailable",
                    "error": "本服务端未接入 Harness，无法派生子流程"}
        return await self._delegate(self.principal, args)

    async def _impl_echo(self, args: dict) -> dict:
        return {"echo": args.get("text", "")}

    # -------------------------------------------------------------- assembly
    @property
    def server(self) -> McpServer:
        return self._server


def _is_binary(path: str) -> bool:
    try:
        with open(path, "rb") as fh:
            chunk = fh.read(8192)
        return b"\x00" in chunk
    except OSError:
        return True
