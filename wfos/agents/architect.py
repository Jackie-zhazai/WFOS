"""Architect — planning only: design, tech plan, root cause, fix plan, risk.

The Architect never touches the filesystem write tools. It *declares* the
operations (files, actions, risk) that the Implementer will later carry out.
"""
from __future__ import annotations

from ..models import ArchitectOutput
from .base import BaseAgent


class ArchitectAgent(BaseAgent):
    role = "architect"
    description = ("方案设计：制定功能设计/技术方案/根因分析/修复方案，评估影响、"
                   "风险与回滚。只读，不执行任何写入。")
    read_roots = ["."]
    write_roots = []
    allowed_tools = [
        "workspace.list_files", "workspace.search", "workspace.read",
        "git.status", "git.diff", "git.log", "logs.read",
        "wiki.search", "echo",
    ]
    timeout = 120
    max_rounds = 8
    output_schema = ArchitectOutput
