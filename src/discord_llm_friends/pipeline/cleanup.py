"""Filter raw corpus entries per persona into cleaned JSON.

Run as: `uv run python -m discord_llm_friends.pipeline.cleanup --persona <id>`.

Input:  personas/<id>/raw.json — a JSON array of strings.
Output: personas/<id>/cleaned.json — a JSON array of kept strings.

Filters (drop a comment if any apply, in order):
  1. empty after strip
  2. matches a persona-declared system-message string
  3. contains a persona-declared quote-block marker (multi-message
     reply chains pasted in)
  4. fewer than `min_words` words (per-persona override or project default)
  5. wrapped in ASCII or Hungarian quote marks start-to-end (pure quote);
     gated by CONFIG.cleanup.drop_pure_quotes
  6. exact duplicate of an earlier kept entry
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from discord_llm_friends import config as cfg
from discord_llm_friends import personas as personas_module


# Common quote-pair characters — applied to all personas. Includes ASCII
# and Hungarian-style pairs; extending to other languages is a one-line
# addition here.
_QUOTE_PAIRS: tuple[tuple[str, str], ...] = (
    ('"', '"'),
    ("„", "”"),    # Hungarian
    ("“", "”"),    # English curly
    ("«", "»"),    # French
)


def _is_pure_quote(text: str) -> bool:
    return any(text.startswith(o) and text.endswith(c) for o, c in _QUOTE_PAIRS)


def _word_count(text: str) -> int:
    return len(text.split())


def clean_persona(persona_id: str) -> dict:
    persona = personas_module.load(persona_id)
    raw_path = persona.raw_path
    if not raw_path.exists():
        raise FileNotFoundError(
            f"raw file not found: {raw_path}\n"
            f"Create personas/{persona_id}/raw.json (a JSON array of strings) "
            f"before running cleanup."
        )

    entries = json.loads(raw_path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError(
            f"{raw_path}: expected a JSON array of strings, got "
            f"{type(entries).__name__}"
        )

    min_words = personas_module.resolved_min_words(persona)
    drop_quote_block = persona.cleanup.drop_quote_block_markers
    drop_system = frozenset(persona.cleanup.drop_system_messages)
    drop_pure_quotes = cfg.CONFIG.cleanup.drop_pure_quotes

    stats: Counter[str] = Counter()
    stats["total"] = len(entries)
    seen: set[str] = set()
    kept: list[str] = []

    for entry in entries:
        if not isinstance(entry, str):
            stats["drop_non_string"] += 1
            continue
        text = entry.strip()
        if not text:
            stats["drop_empty"] += 1
            continue
        if text in drop_system:
            stats["drop_system"] += 1
            continue
        if any(marker in text for marker in drop_quote_block):
            stats["drop_quote_block"] += 1
            continue
        if _word_count(text) < min_words:
            stats["drop_short"] += 1
            continue
        if drop_pure_quotes and _is_pure_quote(text):
            stats["drop_pure_quote"] += 1
            continue
        if text in seen:
            stats["drop_duplicate"] += 1
            continue
        seen.add(text)
        kept.append(text)

    stats["kept"] = len(kept)

    out_path = persona.cleaned_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(kept, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return dict(stats)


def _format_stats(persona_id: str, stats: dict, out_path: Path) -> str:
    total = stats.get("total", 0)
    kept = stats.get("kept", 0)
    dropped = total - kept
    lines = [
        f"cleaned {persona_id}: {kept} kept / {dropped} dropped / {total} total",
        f"  written to {out_path}",
    ]
    drop_reasons = [
        "drop_non_string", "drop_empty", "drop_system", "drop_quote_block",
        "drop_short", "drop_pure_quote", "drop_duplicate",
    ]
    drop_lines = [
        f"    {r}: {stats[r]}" for r in drop_reasons if stats.get(r)
    ]
    if drop_lines:
        lines.append("  drop reasons:")
        lines.extend(drop_lines)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--persona", required=True,
        help="persona id (folder under personas/)",
    )
    args = parser.parse_args(argv)
    stats = clean_persona(args.persona)
    persona = personas_module.load(args.persona)
    print(_format_stats(args.persona, stats, persona.cleaned_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
