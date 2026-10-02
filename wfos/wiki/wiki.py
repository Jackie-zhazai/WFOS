"""LLM Wiki: three-tier knowledge store + distill pipeline.

Tiers:
  authoritative  verified, human-promoted, highest trust
  case          verified run experience (promoted automatically on evidence check)
  candidate     unverified submissions (never auto-published)

Distill pipeline (run completion -> candidates -> sanitize -> dedup ->
evidence-verify -> publish). Retrieved knowledge is *reference material*
only; it can never override system policy.
"""
from __future__ import annotations

import hashlib

from ..redact import redact
from ..storage.repo import Repo

_CASE = "case"
_AUTHORITATIVE = "authoritative"
_CANDIDATE = "candidate"


class WikiClient:
    def __init__(self, repo: Repo):
        self.repo = repo

    # ------------------------------------------------------------------ search
    def search(self, query: str, *, kind: str | None = None, tags: str | None = None,
               trust: str | None = None, limit: int = 10) -> list[dict]:
        return self.repo.search_wiki(query, kind=kind, tags=tags, trust=trust, limit=limit)

    def list(self, *, kind: str | None = None, status: str | None = None, limit: int = 100) -> list[dict]:
        return self.repo.list_wiki(kind=kind, status=status, limit=limit)

    def get(self, wid: int) -> dict | None:
        return self.repo.get_wiki(wid)

    # ------------------------------------------------------------------ intake
    def add_candidate(self, title: str, content: str, *, tags: list[str] | None = None,
                      run_id: str = "", source: str = "", evidence_refs: list[str] | None = None) -> int | None:
        """Add a single candidate. Sanitize + dedup; never auto-publish here."""
        content = self._sanitize(content)
        checksum = self._checksum(title, content)
        if self._exists_checksum(checksum):
            return None
        verified = self._verify_evidence(evidence_refs or [], [])
        return self.repo.add_wiki(
            _CANDIDATE, title, content, source=source, evidence=evidence_refs or [],
            verified=verified, trust="high" if verified else "low",
            run_id=run_id, status="verified" if verified else "pending",
            tags=tags or [], checksum=checksum)

    def distill(self, run_id: str, candidates: list[dict], *, evidence: list[dict],
                run: dict | None = None) -> dict:
        """Full pipeline for one run's candidate set. Returns per-candidate report."""
        report: list[dict] = []
        published = 0
        for c in candidates:
            title = c.get("title", "")
            content = self._sanitize(c.get("content", ""))
            checksum = self._checksum(title, content)
            refs = c.get("evidence_refs") or []
            tags = c.get("tags") or []
            source_run = c.get("source_run") or run_id
            if self._exists_checksum(checksum):
                report.append({"title": title, "status": "duplicate", "id": None})
                continue
            verified = self._verify_evidence(refs, evidence)
            wid = self.repo.add_wiki(
                _CANDIDATE, title, content, source=source_run, evidence=refs,
                verified=verified, trust="high" if verified else "low",
                run_id=run_id, status="verified" if verified else "pending",
                tags=tags, checksum=checksum)
            entry = {"title": title, "id": wid, "status": "verified" if verified else "pending"}
            if verified:
                # Verified run experience auto-promotes to the case tier.
                self.publish(wid, kind=_CASE, by="harness", note="证据校验通过")
                published += 1
                entry["published"] = _CASE
            report.append(entry)
        return {"total": len(candidates), "published": published, "report": report}

    # ------------------------------------------------------------- promote/verify
    def verify(self, wid: int) -> dict | None:
        row = self.repo.get_wiki(wid)
        if row is None:
            return None
        refs = row.get("evidence") or []
        ok = self._verify_evidence(refs, [])
        self.repo.update_wiki(wid, verified=ok, status="verified" if ok else "pending",
                              trust="high" if ok else "low")
        return self.repo.get_wiki(wid)

    def publish(self, wid: int, *, kind: str = _CASE, by: str = "harness", note: str = "") -> dict | None:
        """Publish to a visible tier. Authoritative requires human `by`."""
        row = self.repo.get_wiki(wid)
        if row is None:
            return None
        if kind == _AUTHORITATIVE and by in ("harness", "agent", "curator"):
            raise PermissionError("仅人工可将知识发布为权威（authoritative）")
        if not row.get("verified") and kind != _CANDIDATE:
            raise PermissionError("未经验证的知识不可发布")
        self.repo.update_wiki(wid, kind=kind, status="published", verified=True, trust="high")
        return self.repo.get_wiki(wid)

    def promote_to_authoritative(self, wid: int, *, by: str) -> dict | None:
        if not by or by in ("harness", "agent", "curator"):
            raise PermissionError("仅人工可将知识提升为权威")
        return self.publish(wid, kind=_AUTHORITATIVE, by=by)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _sanitize(text: str) -> str:
        """Redact secrets. Delegates to the shared redactor so the wiki and the
        run record cannot drift apart in what they consider a secret."""
        return redact(text)

    @staticmethod
    def _checksum(title: str, content: str) -> str:
        return hashlib.sha1((title + "\x00" + content).encode("utf-8", "replace")).hexdigest()

    def _exists_checksum(self, checksum: str) -> bool:
        cur = self.repo.conn.execute("SELECT 1 FROM wiki WHERE checksum=? LIMIT 1", (checksum,))
        return cur.fetchone() is not None

    @staticmethod
    def _verify_evidence(refs: list[str], evidence: list[dict]) -> bool:
        """A candidate is 'verified' when every evidence ref appears in the run's evidence."""
        if not refs:
            return False
        available = {f"{e.get('kind', 'other')}:{e.get('source', '')}" for e in evidence}
        available |= {f"log:{e.get('source','')}" for e in evidence if e.get('kind') == 'log'}
        for ref in refs:
            if ref not in available and ref not in {e.get('content', '')[:60] for e in evidence}:
                return False
        return True

    @staticmethod
    def _plan_evidence(refs: list[str]) -> bool:
        return bool(refs)
