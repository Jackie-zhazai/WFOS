"""Chat — the interactive entry's agent.

The only role that reads, writes *and* runs the build/tests, because that is what
"帮我看看这个项目，然后改一下 app.py，再跑一下测试" is one sentence asking for. No
state-machine role has that union, and none should: an implementer that can run its
own tests has stopped being separate from the verifier.

Everything else is the ordinary loop — `BaseAgent.run` executes the tool calls
through the gateway, so each one is policy-checked against this role, attributed by
the workspace snapshot, and written to the trace. Nothing here is a side channel.
"""
from __future__ import annotations

from ..models import ChatOutput
from .base import BaseAgent


class ChatAgent(BaseAgent):
    role = "assistant"
    description = ("交互助手：在项目内读文件、改文件、跑构建与测试，并用自然语言回答。"
                   "需要先看清楚再动手；改动用 workspace.patch（要求锚点唯一）。")
    read_roots = ["."]
    write_roots = ["."]
    # The same set `mcp/server.py` grants the `assistant` role — a tool the model
    # is shown but the policy would refuse is worse than one it never sees.
    allowed_tools = [
        "workspace.list_files", "workspace.search", "workspace.read",
        "workspace.patch", "workspace.write", "workspace.delete",
        "git.status", "git.diff", "git.log",
        "build.check", "build.run", "test.run",
        "logs.read", "wiki.search", "echo",
    ]
    timeout = 180
    # Higher than any state-machine agent: a turn legitimately reads several files,
    # edits, then runs the tests, and cutting that off mid-investigation is worse
    # than a longer turn. Still bounded — the lease heartbeat covers it.
    max_rounds = 12
    output_schema = ChatOutput
    opening_prompt = ("请回答用户最后一条消息里的请求。需要了解现状或做改动时，"
                      "先调用工具（读、改、跑测试），完成后再输出回复。")
