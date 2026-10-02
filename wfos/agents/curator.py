"""Knowledge Curator — distills run results into candidate knowledge.

Can only *submit candidates* (wiki.add_candidate). It has no authority to
write to the authoritative wiki tiers; promotion happens through the
distill pipeline and evidence verification.
"""
from __future__ import annotations

from ..models import CuratorOutput
from .base import BaseAgent


class CuratorAgent(BaseAgent):
    role = "curator"
    description = ("知识沉淀：把本次运行的成功经验蒸馏为知识候选并提交。"
                   "仅提交候选，无权直接发布权威知识。")
    read_roots = []
    write_roots = []
    allowed_tools = ["wiki.search", "wiki.add_candidate", "echo"]
    timeout = 60
    max_rounds = 4
    output_schema = CuratorOutput
