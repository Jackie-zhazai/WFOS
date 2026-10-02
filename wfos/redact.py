"""Secret redaction for content that enters the record from untrusted sources.

Two rules govern where this is applied:

1. **Only untrusted content.** Tool output, evidence, step output, tool
   arguments — anything a model, a subprocess, or the workspace produced.
2. **Never internal fingerprints.** One pattern matches bare long hex tokens,
   which is exactly the shape of a sha256 digest — and `runs.identity_hash` /
   `runs.key_files` are sha256 digests. Redacting those would silently break the
   resume contract, so persistence points that store them are explicitly
   excluded (see `storage/repo.py`).

Patterns are matched by *value shape*, so this is a net, not a guarantee.
"""
from __future__ import annotations

import re
from typing import Any

# Moved verbatim from wiki.py so redaction behaves identically everywhere.
PATTERNS = (
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]+"),
    re.compile(r"\b(sk-[A-Za-z0-9]{16,})\b"),
    re.compile(r"\b(api[_-]?key|apikey|secret|password|passwd|token)\b\s*[=:]\s*\S+", re.I),
    re.compile(r"\b[0-9a-f]{32,}\b"),                      # long hex tokens
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                   # AWS access key ids
)

REDACTED = "[REDACTED]"


def redact(text: str) -> str:
    """Replace secret-shaped values in `text`. Idempotent."""
    if not text:
        return text
    for rx in PATTERNS:
        text = rx.sub(REDACTED, text)
    return text


def redact_structure(value: Any) -> Any:
    """Redact every string inside a JSON-ish structure, preserving its shape.

    Use this for anything that will be serialized — **not** `redact()` on the
    serialized text. Several patterns end in `\\S+`, which is greedy and will
    happily consume a closing quote and following comma: redacting
    `{"content": "password: x"}` as text yields invalid JSON, and the row then
    reads back as its default (silent data loss). Redacting value by value
    confines each match to its own string.
    """
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: redact_structure(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_structure(v) for v in value]
    return value
