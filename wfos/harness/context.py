"""Context budgeting: assemble an agent prompt under an explicit size budget.

The prompt is built from named sections. Some are *load-bearing* and are never
clipped — the rules, the task, the tool list and the output schema are what make
the answer possible at all; shrinking them converts a long prompt into a wrong
one. The rest (evidence, plans, observed changes) are reference material and are
reduced, in a fixed order, only when the assembled prompt overruns.

Budgets are counted in **characters** and reported as such. There is no
tokenizer here, and inventing an approximation would produce a number that looks
precise and cannot be checked — worse than an honest character count.
"""
from __future__ import annotations

from typing import Any

SECTION_SEP = "\n\n"

# Fixed assembly order: stable, load-bearing sections first, and the sections a
# reader scans for last. Deterministic ordering is what makes a prompt diffable.
#
# `conversation` sits right after `working_set` because it plays the same role for
# an interactive turn that `working_set` plays for a state: it says what this
# exchange is about. Only the chat entry produces it, so for every other run the
# section is absent and the assembled prompt is byte-for-byte what it was —
# which is why adding it here does not move the frozen baseline.
SECTION_ORDER = ("rules", "run_context", "working_set", "conversation", "skills",
                 "memory", "plan", "prior_plan", "evidence", "changes", "tools",
                 "output_schema")

# Sections that may be shrunk, cheapest-to-lose first. Anything not listed here
# is never clipped, no matter how large it grows — `working_set` included: it
# says which attempt this is, and losing it makes a retry loop pointless.
# `memory` sits just behind `evidence`: prior runs are useful background, but the
# plan is what this state is executing, so memory goes first when something must.
# `skills` goes first: a procedure is a suggestion about how to get past a
# failure, and the plan is what this state is actually executing.
# `conversation` goes last, which makes it the *last* thing cut. It is reducible —
# a long session has to lose something — but what it holds is what the user
# actually asked for, so everything else goes before it.
REDUCTION_ORDER = ("skills", "evidence", "memory", "prior_plan", "plan", "changes",
                   "conversation")

# Default ceilings, in characters, for the reducible sections. Measured floors:
# the rules + tool schemas + output schema of a real agent run to ~3000-4600
# chars on their own, so the total budget has to clear that comfortably.
DEFAULT_TOTAL_BUDGET = 30_000
DEFAULT_SECTION_BUDGETS = {"evidence": 8_000, "memory": 3_000, "skills": 3_000,
                           "prior_plan": 4_000, "plan": 4_000, "changes": 3_000,
                           "conversation": 8_000}
# The conversation floor is one turn's worth: a prompt that has lost the question
# it was asked is not a cheaper prompt, it is a different task.
DEFAULT_SECTION_FLOORS = {"evidence": 1_000, "memory": 300, "skills": 300,
                          "prior_plan": 400, "plan": 400, "changes": 200,
                          "conversation": 600}

# Cap on a single tool result fed back into the conversation. The tool specs
# allow up to 200k chars for a test run; a handful of those would crowd out
# everything else in the window.
DEFAULT_TOOL_RESULT_BUDGET = 8_000


def clip(text: str, limit: int | None) -> str:
    """Shrink `text` to `limit` characters, keeping **both** ends.

    A head-only cut throws away the newest entries of a log or evidence list,
    which is usually the part that matters. The elision is marked, and the
    dropped count is stated rather than implied.
    """
    if limit is None or limit <= 0 or len(text) <= limit:
        return text
    marker = f"\n...[省略 {len(text) - limit} 字符]...\n"
    keep = max(0, limit - len(marker))
    head = keep // 2
    return text[:head] + marker + text[-keep + head:]


def fit(sections: dict[str, str], *, total_budget: int = DEFAULT_TOTAL_BUDGET,
        budgets: dict[str, int] | None = None,
        floors: dict[str, int] | None = None,
        order: tuple[str, ...] = REDUCTION_ORDER) -> tuple[str, dict[str, Any]]:
    """Assemble `sections` into a prompt no larger than `total_budget` if possible.

    Returns `(prompt, metadata)`. The metadata is the point of the exercise: it
    says which sections were cut and by how much, so a shrinking prompt is
    visible rather than silent. `over_budget` is reported honestly — if the
    non-reducible sections alone exceed the budget there is nothing left to cut.

    A caller that has nothing to say in a section must **omit the key**, not pass
    `""`: an empty section still contributes its separator, so `fit({"a": "x",
    "b": ""})` is `"x\\n\\n"` while `fit({"a": "x"})` is `"x"`. Measured, not
    assumed — it is the difference between a prompt that is unchanged for the
    runs that do not use the section and one that quietly gains a blank line.
    """
    raw = {name: (text or "") for name, text in sections.items()}
    limits: dict[str, int | None] = {name: (budgets or DEFAULT_SECTION_BUDGETS).get(name)
                                     for name in raw}
    floor_of = {**DEFAULT_SECTION_FLOORS, **(floors or {})}
    reductions: list[dict[str, Any]] = []

    def render() -> dict[str, str]:
        return {name: clip(text, limits.get(name)) if limits.get(name) else text
                for name, text in raw.items()}

    def joined(rendered: dict[str, str]) -> str:
        return SECTION_SEP.join(rendered[name] for name in SECTION_ORDER
                                if name in rendered)

    rendered = render()
    # A section clipped by its own ceiling is a reduction too, and recording it
    # only inside the total-budget loop below would make it invisible.
    for name, text in raw.items():
        limit = limits.get(name)
        if limit and len(text) > limit:
            reductions.append({"section": name, "phase": "section_budget",
                               "from_chars": len(text),
                               "to_chars": len(rendered[name]),
                               "budget_chars": limit})

    for _ in range(len(raw) * 4):          # bounded: each pass shrinks one section
        prompt = joined(rendered)
        if len(prompt) <= total_budget:
            break
        overflow = len(prompt) - total_budget
        for name in order:
            if name not in raw:
                continue
            current = len(rendered[name])
            floor = floor_of.get(name, 0)
            if current <= floor:
                continue
            new_limit = max(floor, current - overflow)
            limits[name] = new_limit
            reductions.append({"section": name, "phase": "total_budget",
                               "from_chars": current, "to_chars": new_limit,
                               "overflow_chars": overflow})
            rendered = render()
            break
        else:
            break                          # nothing reducible left

    prompt = joined(rendered)
    metadata = {
        "char_budget": total_budget,       # unit is characters, and says so
        "total_chars": len(prompt),
        "over_budget": len(prompt) > total_budget,
        "sections": {name: {"raw_chars": len(raw[name]),
                            "budget_chars": limits.get(name),
                            "rendered_chars": len(rendered[name])}
                     for name in raw},
        "reductions": reductions,
    }
    return prompt, metadata
