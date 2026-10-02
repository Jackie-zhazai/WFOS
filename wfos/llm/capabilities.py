"""What a provider turns out to accept, kept in one place.

Before this, the same assumptions were spread across the code and the operator's
head: `structured_output` had to be set by hand because DeepSeek rejects
`json_schema`, and knowing that came from a 400 somebody had to read. The rules
still have to be known; the point is that they are discovered **by behaviour**,
recorded once, and readable in one place.

Discovery never parses an error message. The question "does this endpoint accept
`response_format: json_schema`?" is answered by sending the same request without
it and seeing whether *that* succeeds — which is a fact, not a guess about what a
sentence meant. The tool-layer rule (classification never reads text) applies for
the same reason: a message is prose that can be reworded, a status is not.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


@dataclass
class Capabilities:
    """What is known about one (provider, endpoint, model) triple."""

    provider: str = ""
    base_url: str = ""
    model: str = ""
    # The mode observed to work. None means "not yet discovered" — distinct from
    # "none works", which is a finding.
    structured_output: str | None = None
    # Tool names are always sanitized on the wire today; recorded so that a future
    # provider which *does* accept dots can be recognised rather than assumed.
    tool_names_sanitized: bool = True
    # Whether the model spends output budget on reasoning before answering. Known
    # from usage once a call has been made — it is why a small max_tokens yields an
    # unparseable answer rather than a short one.
    reasoning: bool | None = None
    # The largest reasoning run seen, in tokens. Recorded so the output ceiling can
    # be sized from measurement: a reasoning model's thinking comes out of the same
    # budget as its answer, so a ceiling that fits a plain model truncates a
    # reasoning one — and truncated JSON is unparseable, not merely short.
    reasoning_tokens_observed: int | None = None

    @property
    def key(self) -> str:
        return f"{self.provider}|{self.base_url}|{self.model}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def capabilities_for(cfg) -> Capabilities:
    """The starting assumptions, from configuration alone."""
    return Capabilities(
        provider=getattr(cfg, "provider", "") or "",
        base_url=getattr(cfg, "base_url", "") or "",
        model=getattr(cfg, "model", "") or "",
        structured_output=getattr(cfg, "structured_output", None),
    )


class CapabilityStore:
    """A small JSON file of what has been learned, keyed by provider/endpoint/model.

    Written atomically: a half-written file would read back as "nothing is known",
    which is indistinguishable from a fresh install and would silently re-probe.
    """

    def __init__(self, path: Path | str | None):
        self.path = Path(path) if path else None
        self._entries: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return                       # unreadable or corrupt: start empty
        if isinstance(data, dict) and data.get("schemaVersion") == SCHEMA_VERSION:
            entries = data.get("entries")
            if isinstance(entries, dict):
                self._entries = entries

    def get(self, key: str) -> dict | None:
        entry = self._entries.get(key)
        return dict(entry) if isinstance(entry, dict) else None

    def observe(self, caps: Capabilities) -> None:
        """Record a fact about this provider. Absent fields are left as they were."""
        known = self._entries.get(caps.key) or {}
        merged = {**known, **{k: v for k, v in caps.to_dict().items()
                              if v is not None}}
        self._entries[caps.key] = merged
        self._save()

    def all(self) -> dict[str, dict]:
        return {k: dict(v) for k, v in sorted(self._entries.items())}

    def _save(self) -> None:
        if self.path is None:
            return
        payload = json.dumps({"schemaVersion": SCHEMA_VERSION,
                              "entries": self._entries},
                             ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(tmp, self.path)
        except OSError:
            # A cache that cannot be written must not take the run down with it.
            if os.path.exists(tmp):
                os.unlink(tmp)
