"""Investigator — read-only evidence & context gathering."""
from __future__ import annotations

from ..models import InvestigatorOutput
from .base import BaseAgent


class InvestigatorAgent(BaseAgent):
    role = "investigator"
    description = ("只读调查：捕获需求/问题描述、检查项目结构、收集证据。"
                   "绝不修改任何文件。")
    read_roots = ["."]
    write_roots = []
    allowed_tools = [
        "workspace.list_files", "workspace.search", "workspace.read",
        "git.status", "git.diff", "git.log", "logs.read",
        "wiki.search", "delegate", "echo",
    ]
    timeout = 120
    max_rounds = 6
    output_schema = InvestigatorOutput
