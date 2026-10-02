"""Verifier — independent build/test/regression checking.

Runs with its own principal (no write roots) and reports *structured* results
built from real build/test exit codes. Never judges via prose.
"""
from __future__ import annotations

from ..models import VerifierOutput
from .base import BaseAgent


class VerifierAgent(BaseAgent):
    role = "verifier"
    description = ("独立验证：执行 build.check / test.run / 回归检查，输出结构化结果。"
                   "只读，绝不修改被测代码。")
    read_roots = ["."]
    write_roots = []
    allowed_tools = [
        "build.check", "build.run", "test.run", "logs.read",
        "workspace.list_files", "workspace.read", "git.diff", "echo",
    ]
    timeout = 240
    max_rounds = 6
    output_schema = VerifierOutput
