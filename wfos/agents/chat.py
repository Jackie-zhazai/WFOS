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

from typing import Any

from ..models import ChatOutput
from .base import BaseAgent


class ChatAgent(BaseAgent):
    role = "assistant"
    description = "交互助手：按需读写项目内文件、跑构建与测试，然后简洁作答。"
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
    # than a longer turn. Still bounded — the lease heartbeat covers it, and the
    # last round is answered without tools either way.
    max_rounds = 20
    output_schema = ChatOutput
    # A conversation is the one place where "I looked at what I had time for and
    # here is what I think" beats an error. See `BaseAgent.answers_without_tools`.
    answers_without_tools = True

    @property
    def opening_prompt(self) -> str:
        # Short on purpose. The list of things *not* to do was longer than the
        # answer it was trying to shape, and a model spends its attention where
        # the instructions are — a five-line formatting spec produced five-line
        # formatting. Two facts have to survive: the budget (which is what stops
        # it exploring forever) and "plain text" (which is what makes it readable
        # in a terminal). Everything else is pico's one line: concise and concrete.
        return (
            f"请回答用户最后一条消息：需要时先调用工具，够了就回答。"
            f"最多 {self.max_rounds} 轮工具调用，最后一轮没有工具、必须直接作答，"
            f"所以不要反复搜索。回答简洁具体、用纯文本、不要 Markdown 标记。")

    def answer_from_text(self, text: str) -> dict[str, Any]:
        """Prose standing in for the structured answer, when the rounds ran out.

        Marked as such in the reply: a turn that was cut off while still looking is
        a different thing from one that finished, and a reader who cannot tell them
        apart will trust a partial answer more than it deserves.
        """
        said = (text or "").strip()
        if not said:
            return {"reply": f"（这一轮用完了 {self.max_rounds} 轮工具调用预算，"
                             f"仍没能得出结论；可以说得更具体一些，或让我只看某几个文件。）"}
        return {"reply": f"{said}\n\n（注：以上是在工具调用预算用尽前得到的结果，"
                         f"可能还不完整。）"}
