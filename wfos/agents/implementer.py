"""Implementer — the only default write role.

Holds write roots over the project. Its execution follows the Architect's plan
file-by-file; every write goes through the gateway (path-constrained, logged).
"""
from __future__ import annotations

from ..models import ImplementerOutput
from .base import BaseAgent


class ImplementerAgent(BaseAgent):
    role = "implementer"
    description = ("代码执行：严格按照 Architect 方案逐文件落地（create/modify/delete）。"
                   "只做方案里列出的变更。")
    read_roots = ["."]
    write_roots = ["."]
    allowed_tools = [
        "workspace.list_files", "workspace.read", "workspace.patch",
        "workspace.write", "workspace.delete",
        "git.status", "git.diff", "logs.read", "echo",
    ]
    timeout = 180
    max_rounds = 8
    output_schema = ImplementerOutput
