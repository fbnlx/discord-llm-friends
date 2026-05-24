"""Per-channel conversation history for cross-persona continuity.

Each /<persona> slash command interaction (when it produces a real LLM
response) is appended to state/history/<channel_id>.json. All personas
in the same channel share the same file, so one persona can see what
another just said and react.

File format: a JSON list of exchange dicts, oldest first. Bounded to
`CONFIG.history.retention_per_channel` entries — older ones get dropped
on each append.

Atomic write via tmp-then-rename so a crash mid-write can't corrupt the
file. Note: read-modify-write races between concurrent appends in the
same channel can still drop updates under high concurrency. Acceptable
for this project's scale.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from discord_llm_friends import config as cfg


logger = logging.getLogger("history")


@dataclass(frozen=True)
class Exchange:
    """One question→response turn recorded from a slash command."""

    timestamp: str           # ISO-8601 UTC
    user_id: int
    user_name: str
    persona: str             # persona id
    question: str
    response: str

    @classmethod
    def now(
        cls,
        *,
        user_id: int,
        user_name: str,
        persona: str,
        question: str,
        response: str,
    ) -> "Exchange":
        c = cfg.CONFIG.history
        return cls(
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            user_id=user_id,
            user_name=user_name,
            persona=persona,
            question=question[:c.max_question_chars],
            response=response[:c.max_response_chars],
        )


class ChannelHistory:
    """JSON-backed conversation history for a single Discord channel."""

    def __init__(self, channel_id: int) -> None:
        self.channel_id = channel_id
        self.retention = cfg.CONFIG.history.retention_per_channel
        self.path = cfg.HISTORY_DIR / f"{channel_id}.json"

    def load(self) -> list[Exchange]:
        """Return all stored exchanges, oldest first. Empty if file
        missing or corrupt."""
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception(
                "failed to parse %s — returning empty history", self.path,
            )
            return []
        return [Exchange(**entry) for entry in raw]

    def recent(self, n: int) -> list[Exchange]:
        """Last n exchanges, oldest first."""
        if n <= 0:
            return []
        return self.load()[-n:]

    def append(self, exchange: Exchange) -> None:
        """Append one exchange and persist atomically. Trims to retention."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entries = self.load()
        entries.append(exchange)
        if len(entries) > self.retention:
            entries = entries[-self.retention:]
        out = [asdict(e) for e in entries]
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)
