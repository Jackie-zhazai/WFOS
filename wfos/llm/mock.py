"""Deterministic, offline 'mock brain'.

The mock stands in for a hosted model so the whole system runs hermetically. Its
only 'intelligence' is scripted; every tool call it makes goes through the real
MCP gateway with full role/path/approval enforcement, and its Verifier reports
are computed from *real* build/test exit codes. Nothing here can be influenced
by file content or injected text.
"""
from __future__ import annotations

import re
from typing import Any

from ..models import ModelResult
from .base import LLMAdapter


def _slug(text: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z一-鿿]+", "_", text).strip("_")
    return s or "feature"


def _latin_slug(text: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_")
    return s or "feature"


def _find_py_target(description: str, default: str = "app.py") -> str:
    m = re.findall(r"[\w./\\-]+\.py\b", description)
    return m[0].replace("\\", "/") if m else default


def _find_danger_ops(description: str) -> list[str]:
    ops = []
    pairs = [
        (("删除", "delete", "移除", "remove"), "delete"),
        (("数据库", "db ", "migrate", "建表", "改表"), "db_migrate"),
        (("生产", "线上", "prod"), "prod_write"),
        (("外部", "外接", "external", "第三方接口"), "external"),
        (("密钥", "secret", "password", "token", "key"), "secret"),
        (("权限", "permission", "越权"), "permission"),
        (("不可逆", "irreversible", "回滚不了"), "irreversible"),
    ]
    low = description.lower()
    for keys, kind in pairs:
        if any(k in low or k in description for k in keys):
            ops.append(kind)
    return ops


def _asserts_from_test(test_src: str) -> list[tuple[str, float, float]]:
    """Parse `assert name(arg) == expected` lines. Returns (name, arg, expected)."""
    out = []
    for m in re.finditer(r"assert\s+(\w+)\s*\(\s*(-?\d+(?:\.\d+)?)\s*\)\s*==\s*(-?\d+(?:\.\d+)?)", test_src):
        name, arg, exp = m.group(1), float(m.group(2)), float(m.group(3))
        out.append((name, arg, exp))
    return out


def _fit_linear(pairs: list[tuple[float, float]]) -> tuple[float, float] | None:
    """Fit y = a*x + b to (x, y) pairs by least squares."""
    if not pairs:
        return None
    n = len(pairs)
    sx = sum(p[0] for p in pairs)
    sy = sum(p[1] for p in pairs)
    sxx = sum(p[0] ** 2 for p in pairs)
    sxy = sum(p[0] * p[1] for p in pairs)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-9:
        if n == 1:
            return 1.0, pairs[0][1] - pairs[0][0]
        return 0.0, sy / n
    a = (n * sxy - sx * sy) / denom
    b = (sy - a * sx) / n
    return round(a, 4), round(b, 4)


class MockAdapter(LLMAdapter):
    name = "mock"
    model = "mock"

    async def complete(self, *, messages=None, schema=None, tools=None,
                       temperature=None, max_tokens=None, ctx=None) -> ModelResult:
        ctx = ctx or {}
        agent = ctx.get("agent", "")
        state = ctx.get("state", "")
        handler = getattr(self, f"_h_{agent}__{state}", None) or getattr(self, f"_h_{agent}", None)
        if handler is None:
            return ModelResult(output={"summary": f"mock fallback for {agent}/{state}",
                                       "next_step": {"suggested_state": "", "reason": "unknown"}})
        output = await handler(ctx)
        return ModelResult(output=output)

    # ------------------------------------------------------------------ helpers
    async def _gateway_list_py(self, ctx) -> list[str]:
        gw = ctx.get("gateway")
        if gw is None:
            return []
        out = await gw.call(tool="workspace.list_files", args={"pattern": "**/*.py"})
        if not out.ok:
            return []
        return [f for f in (out.structured or {}).get("files", []) if f.endswith(".py")]

    async def _read(self, ctx, path: str) -> str | None:
        gw = ctx.get("gateway")
        if gw is None:
            return None
        out = await gw.call(tool="workspace.read", args={"path": path})
        if not out.ok:
            return None
        try:
            return out.structured["content"]
        except (KeyError, TypeError):
            return None

    # ----------------------------------------------------- investigator states
    async def _h_investigator__req_capture(self, ctx) -> dict:
        desc = ctx["run"]["description"]
        keywords = [k for k in re.split(r"[\s,，。；;]+", desc) if k][:5]
        return {
            "findings": [f"原始需求: {desc}"],
            "evidence": [{"kind": "user", "source": "request", "content": desc,
                          "confidence": "high"}],
            "project_context": {"keywords": keywords, "title": ctx["run"]["title"]},
            "next_step": {"suggested_state": "project_check", "reason": "需求已捕获，进入项目检查"},
        }

    async def _h_investigator__issue_capture(self, ctx) -> dict:
        desc = ctx["run"]["description"]
        keywords = [k for k in re.split(r"[\s,，。；;]+", desc) if k][:5]
        return {
            "findings": [f"问题描述: {desc}"],
            "evidence": [{"kind": "user", "source": "issue report", "content": desc,
                          "confidence": "high"}],
            "project_context": {"keywords": keywords, "title": ctx["run"]["title"]},
            "next_step": {"suggested_state": "history_search", "reason": "问题已捕获，检索历史"},
        }

    async def _h_investigator__history_search(self, ctx) -> dict:
        desc = ctx["run"]["description"]
        gw = ctx.get("gateway")
        results = []
        if gw is not None:
            out = await gw.call(tool="wiki.search", args={"query": desc[:60], "limit": 5})
            if out.ok and out.structured:
                results = out.structured.get("results") or []
        return {
            "findings": [f"历史记忆检索: 命中 {len(results)} 条相关知识"],
            "evidence": [{"kind": "wiki", "source": "llm-wiki",
                          "content": str(results)[:1500], "confidence": "medium"}],
            "project_context": {"wiki_hits": len(results)},
            "next_step": {"suggested_state": "evidence_collect",
                          "reason": "历史检索完成，收集现场证据"},
        }

    async def _h_investigator__project_check(self, ctx) -> dict:
        gw = ctx.get("gateway")
        files = await self._gateway_list_py(ctx)
        markers = []
        if gw:
            for pat in ("pyproject.toml", "package.json", "*.csproj", "*.sln", "CMakeLists.txt", "README*"):
                out = await gw.call(tool="workspace.list_files", args={"pattern": pat})
                if out.ok and out.structured:
                    markers.extend((out.structured or {}).get("files", []))
        project_type = "python"
        if any(m.endswith(".csproj") or m.endswith(".sln") for m in markers):
            project_type = "dotnet"
        elif any(m.endswith(".cpp") or m.endswith(".c") or m == "CMakeLists.txt" for m in markers):
            project_type = "cpp"
        return {
            "findings": [f"项目类型: {project_type}", f"Python 文件数: {len(files)}",
                         f"识别标志: {markers[:5]}"],
            "evidence": [{"kind": "other", "source": "project scan", "content": str(markers[:8]),
                          "confidence": "high"}],
            "project_context": {"files": files, "markers": markers, "project_type": project_type},
            "next_step": {"suggested_state": "feature_design", "reason": "项目检查完成"},
        }

    async def _h_investigator__evidence_collect(self, ctx) -> dict:
        desc = ctx["run"]["description"]
        target = _find_py_target(desc)
        src = await self._read(ctx, target) or ""
        evidence = [{"kind": "code", "source": target, "content": src[:2000], "confidence": "high"}]
        findings = [f"目标文件 {target} 已读取"]
        # A regressions file is project state, read as evidence when present.
        gw = ctx.get("gateway")
        if gw:
            out = await gw.call(tool="logs.read", args={"path": "regressions.txt", "tail": True})
            if out.ok and out.structured.get("content"):
                evidence.append({"kind": "log", "source": "regressions.txt",
                                 "content": out.structured["content"], "confidence": "medium"})
        return {
            "findings": findings,
            "evidence": evidence,
            "project_context": {"target": target},
            "next_step": {"suggested_state": "root_cause", "reason": "证据已收集"},
        }

    # ------------------------------------------------------- architect states
    async def _h_architect__feature_design(self, ctx) -> dict:
        desc = ctx["run"]["description"]
        return {
            "summary": f"功能: {ctx['run']['title']}",
            "design_notes": f"需求定义: {desc}。触发方式: 调用入口; 输入: 参数; 输出: 结构化结果。",
            "verification_plan": ["执行 build.check 确认语法", "运行 check.py 确认回归"],
            "files": [], "operations": [],
            "impact": [], "risk_level": "low", "confidence": "high", "root_cause": "",
            "causal_chain": [], "rollback": "git checkout 相关文件",
            "next_step": {"suggested_state": "tech_design", "reason": "功能设计完成"},
        }

    async def _h_architect__tech_design(self, ctx) -> dict:
        desc = ctx["run"]["description"]
        target = _find_py_target(desc)
        slug = _latin_slug(ctx["run"]["title"])
        files = [{"path": target, "action": "modify",
                  "before": "", "after": "",
                  "content": "", "detail": f"追加 feature_{slug}() 实现功能 {ctx['run']['title']}"}]
        ops = [{"kind": "write", "path": target, "reason": "追加功能代码"}]
        for kind in _find_danger_ops(desc):
            ops.append({"kind": kind, "path": target, "reason": f"需求涉及 {kind}"})
        risk = "high" if (len(files) > 2 or any(o["kind"] != "write" for o in ops)) else "low"
        return {
            "summary": f"方案: 在 {target} 追加功能模块",
            "files": files,
            "operations": ops,
            "impact": [target, "现有模块（回归检查）"],
            "risk_level": risk,
            "verification_plan": ["build.check", "python check.py", "回归检查"],
            "design_notes": "",
            "confidence": "high", "root_cause": "", "causal_chain": [], "rollback": "git checkout 相关文件",
            "next_step": {"suggested_state": "risk_assess", "reason": "技术方案完成，进行风险评估"},
        }

    async def _h_architect__root_cause(self, ctx) -> dict:
        desc = ctx["run"]["description"]
        target = _find_py_target(desc)
        src = await self._read(ctx, target) or ""
        test_src = await self._read(ctx, "check.py") or ""
        asserts = _asserts_from_test(test_src)
        evidence_based = bool(src) and bool(asserts)
        if evidence_based:
            names = sorted({n for n, _, _ in asserts})
            confidence = "high"
            root = f"{target} 中 {names[0]} 的实现与 check.py 断言不一致"
            chain = [f"{target} 定义了 {names[0]}", "check.py 断言期望特定返回值", "实现未满足断言"]
        elif "无法" in desc or "不知道" in desc or "推测" in desc:
            confidence = "low"
            root = "证据不足，根因未确认"
            chain = ["缺少可直接定位的日志/代码证据"]
        else:
            confidence = "medium"
            root = f"{target} 行为与预期不符（需确认）"
            chain = [f"{target} 被怀疑是问题源", "需补充证据确认"]
        return {
            "summary": f"根因: {root}",
            "root_cause": root,
            "causal_chain": chain,
            "confidence": confidence,
            "files": [], "operations": [], "impact": [target], "risk_level": "low",
            "verification_plan": ["修复后运行 check.py"], "design_notes": "",
            "rollback": "git checkout 相关文件",
            "next_step": {"suggested_state": "confidence_assess", "reason": "根因分析完成"},
        }

    async def _h_architect__fix_plan(self, ctx) -> dict:
        desc = ctx["run"]["description"]
        target = _find_py_target(desc)
        src = await self._read(ctx, target) or ""
        test_src = await self._read(ctx, "check.py") or ""
        asserts = _asserts_from_test(test_src)
        detail = ""
        patch: dict[str, Any] | None = None
        if asserts:
            by_fn: dict[str, list[tuple[float, float]]] = {}
            for name, arg, exp in asserts:
                by_fn.setdefault(name, []).append((arg, exp))
            for name, pairs in by_fn.items():
                fit = _fit_linear(pairs)
                if fit is not None and name in src:
                    a, b = fit
                    pm = "+" if b >= 0 else "-"
                    bval = f"{abs(b):g}"
                    aval = f"{a:g}"
                    if a == 1:
                        expr = f"x {pm} {bval}" if b else "x"
                    elif a == -1:
                        expr = f"-x {pm} {bval}" if b else "-x"
                    elif b == 0:
                        expr = f"{aval} * x"
                    else:
                        expr = f"{aval} * x {pm} {bval}"
                    body = f"    return {expr}\n"
                    new_src = re.sub(
                        rf"(def\s+{re.escape(name)}\s*\([^)]*\)\s*:\s*\n)(.*?)(?=\ndef |\Z)",
                        # `body` is bound as a default rather than closed over: it
                        # is read during this `re.sub` call today, but a closure
                        # over a loop variable breaks the moment the call is
                        # deferred, and silently.
                        lambda m, _body=body: m.group(1) + _body,
                        src, count=1, flags=re.S)
                    if new_src != src:
                        patch = {"path": target, "action": "modify",
                                 "before": src, "after": new_src, "content": "", "detail": "修复实现"}
                        detail = f"根据断言合成线性修复: {name} -> {expr}"
        if patch is None:
            # Non-breaking fallback: append a docstring.
            new_src = src.rstrip() + "\n\n# (fix-plan) 补充说明: 依据检查结果调整实现\n"
            patch = {"path": target, "action": "modify", "before": src, "after": new_src,
                     "content": "", "detail": "非破坏性补丁"}
            detail = "无可用断言，采用非破坏性补丁"
        files = [patch]

        # If this bugfix owns regression modules (from regressions.txt), clear
        # those markers so the parent regression re-verify can pass.
        gw = ctx.get("gateway")
        if gw is not None:
            mods = set()
            for grp in re.findall(r"\[([^\]]*)\]", desc):
                for part in grp.split(","):
                    part = part.strip().strip("'\"")
                    if part:
                        mods.add(part)
            if mods:
                ro = await gw.call(tool="logs.read", args={"path": "regressions.txt"})
                if ro.ok:
                    content = ro.structured.get("content", "")
                    lines = content.splitlines()
                    seen: set[str] = set()
                    kept = []
                    removed_any = False
                    for ln in lines:
                        key = ln.strip()
                        if key in mods and key not in seen:
                            seen.add(key)
                            removed_any = True
                            continue
                        kept.append(ln)
                    if removed_any:
                        new_content = "\n".join(kept)
                        if new_content:
                            new_content += "\n"
                        files.append({"path": "regressions.txt", "action": "modify",
                                      "before": content, "after": new_content,
                                      "content": "", "detail": "清除已修复回归标记"})
                        detail += "；清除回归标记"

        ops = [{"kind": "write", "path": target, "reason": "应用修复"}]
        for kind in _find_danger_ops(desc):
            ops.append({"kind": kind, "path": target, "reason": f"需求涉及 {kind}"})
        risk = "high" if (len(ops) > 1 and any(o["kind"] != "write" for o in ops)) else "low"
        return {
            "summary": detail,
            "root_cause": ctx.get("prior", {}).get("root_cause", {}).get("root_cause", ""),
            "files": files,
            "operations": ops,
            "impact": [target, "regressions.txt"],
            "risk_level": risk,
            "verification_plan": ["build.check", "python check.py"],
            "design_notes": detail, "confidence": "high", "causal_chain": [],
            "rollback": "git checkout 相关文件",
            "next_step": {"suggested_state": "risk_assess", "reason": "修复方案完成"},
        }

    # ------------------------------------------------------ implementer states
    async def _h_implementer__implement(self, ctx) -> dict:
        plan = ctx.get("plan") or {}
        files = plan.get("files") or []
        changes = []
        failed = []
        gw = ctx.get("gateway")
        for pf in files:
            path = pf.get("path", "")
            action = pf.get("action", "modify")
            if action == "create":
                out = await gw.call(tool="workspace.write", args={"path": path, "content": pf.get("content", "")})
            elif action == "delete":
                out = await gw.call(tool="workspace.delete", args={"path": path})
            else:
                out = await gw.call(tool="workspace.patch", args={
                    "path": path, "before": pf.get("before", ""), "after": pf.get("after", "")})
            if out.ok:
                changes.append({"file": path, "action": action, "detail": pf.get("detail", "")})
            else:
                failed.append(f"{path}: {out.error}")
        kind = ctx["run"]["kind"]
        suggested = "build_test" if kind == "feature" else "verify_regression"
        return {
            "changes": changes, "failed": failed,
            "next_step": {"suggested_state": suggested,
                          "reason": "实现完成" if not failed else "部分实现失败"},
        }

    # --------------------------------------------------------- verifier states
    async def _run_verifier(self, ctx, *, include_regression: bool = False) -> dict:
        gw = ctx.get("gateway")
        py_files = await self._gateway_list_py(ctx)
        build = {"ok": False, "errors": [], "files_checked": py_files}
        if py_files:
            out = await gw.call(tool="build.check", args={"files": py_files})
            if out.ok:
                build = out.structured or build
        test = {"ok": False, "exit_code": -1, "passed": 0, "failed": 0, "skipped": 0,
                "command": "python check.py", "cases": [], "output": ""}
        tout = await gw.call(tool="test.run", args={"command": "python check.py"})
        if tout.ok:
            test = tout.structured or test

        regression = [{"module": "core", "ok": True, "detail": "core 回归通过"}]
        if include_regression:
            ro = await gw.call(tool="logs.read", args={"path": "regressions.txt", "tail": True})
            if ro.ok and ro.structured.get("content"):
                for line in ro.structured["content"].splitlines():
                    line = line.strip()
                    if line:
                        regression.append({"module": line, "ok": False,
                                           "detail": f"regressions.txt 标记 {line} 回归"})

        all_ok = build.get("ok") and test.get("failed", 1) == 0 and all(r["ok"] for r in regression)
        verdict = "pass" if all_ok else "fail"
        return {
            "build": build, "tests": test, "regression": regression, "verdict": verdict,
            "evidence": [{"kind": "build", "source": "build.check", "content": str(build.get("errors")),
                          "confidence": "high"},
                         {"kind": "test_output", "source": test.get("command", ""),
                          "content": test.get("output", "")[:2000], "confidence": "high"}],
        }

    async def _h_verifier__build_test(self, ctx) -> dict:
        body = await self._run_verifier(ctx, include_regression=False)
        suggested = "regression_verify" if body["verdict"] == "pass" else "implement"
        body["next_step"] = {"suggested_state": suggested,
                             "reason": "构建测试通过" if body["verdict"] == "pass" else "构建/测试失败，携带证据返回实现"}
        return body

    async def _h_verifier__regression_verify(self, ctx) -> dict:
        body = await self._run_verifier(ctx, include_regression=True)
        if body["verdict"] == "fail":
            regressed = [r["module"] for r in body["regression"] if not r["ok"]]
            body["next_step"] = {
                "suggested_state": "spawn_bugfix" if regressed else "implement",
                "reason": f"回归模块: {regressed}" if regressed else "功能自身验证失败，返回实现"}
        else:
            body["next_step"] = {"suggested_state": "knowledge_distill", "reason": "回归验证通过"}
        return body

    async def _h_verifier__verify_regression(self, ctx) -> dict:
        body = await self._run_verifier(ctx, include_regression=True)
        if body["verdict"] == "pass":
            body["next_step"] = {"suggested_state": "knowledge_distill", "reason": "修复验证通过"}
        else:
            body["next_step"] = {"suggested_state": "root_cause",
                                 "reason": "验证失败，携带新证据返回诊断阶段"}
        return body

    # ------------------------------------------------------------ curator state
    async def _h_curator__knowledge_distill(self, ctx) -> dict:
        run = ctx["run"]
        kind_label = "功能开发" if run["kind"] == "feature" else "问题修复"
        evidence = (ctx.get("evidence") or {}).get("items") or []
        tags = [kind_label, _slug(run["title"])]
        content = (
            f"## {run['title']}\n\n"
            f"类型: {kind_label}\n描述: {run['description']}\n"
            f"结果: 流程完成\n关联运行: {run['id']}\n"
            f"证据数: {len(evidence)}\n"
        )
        return {
            "knowledge": [{
                "title": f"{run['title']}（{kind_label}记录）",
                "content": content,
                "tags": tags,
                "source_run": run["id"],
                "evidence_refs": [f"{e.get('kind')}:{e.get('source')}" for e in evidence[:5]],
            }],
            "next_step": {"suggested_state": "completed", "reason": "知识沉淀完成"},
        }
