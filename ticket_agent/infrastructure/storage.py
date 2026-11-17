"""Local files: the decision cache (for resume) and the per-ticket run log."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from ticket_agent.domain.masking import mask_recursively
from ticket_agent.domain.models import DecisionOutcome


class JsonDecisionStore:
    """Remembers each decision before it is posted.

    After a crash, a re-run reuses the stored decision instead of asking the model again,
    so the customer gets exactly what we had decided and we do not pay twice.
    Writes are atomic (temp file + rename), so a crash never leaves a half-written file.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._entries: dict[str, dict[str, Any]] = self._read()

    def load(self, ticket_id: str, fingerprint: str) -> DecisionOutcome | None:
        entry = self._entries.get(ticket_id)
        if entry is None or entry.get("fingerprint") != fingerprint:
            return None
        return DecisionOutcome.model_validate(entry["outcome"])

    def save(self, ticket_id: str, fingerprint: str, outcome: DecisionOutcome) -> None:
        self._entries[ticket_id] = {
            "fingerprint": fingerprint,
            "submitted": False,
            "outcome": outcome.model_dump(mode="json"),
        }
        self._flush()

    def mark_submitted(self, ticket_id: str) -> None:
        if ticket_id in self._entries:
            self._entries[ticket_id]["submitted"] = True
            self._flush()

    def clear(self) -> None:
        self._entries = {}
        self._flush()

    def _read(self) -> dict[str, dict[str, Any]]:
        if not self._path.exists():
            return {}
        data = json.loads(self._path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=self._path.parent, prefix=".decisions-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp:
                json.dump(self._entries, tmp, ensure_ascii=False, indent=1)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp_name, self._path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise


class JsonlRunLog:
    """One JSON line per ticket. Every string is masked again before it is written."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def write(self, entry: dict[str, Any]) -> None:
        line = json.dumps(mask_recursively(entry), ensure_ascii=False)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
