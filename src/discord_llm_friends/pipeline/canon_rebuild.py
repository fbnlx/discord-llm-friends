"""Rebuild a persona's canon collection from its ledger.

The ledger (personas/<id>/canon/facts.jsonl) is the source of truth; the
`<id>__canon` ChromaDB collection is a derived cache. This drops the
collection and re-embeds every ACTIVE ledger record into a fresh one — the
disaster-recovery path (collection lost/diverged) and the migration path
after an embedding-model change.

Run as:
    uv run python -m discord_llm_friends.pipeline.canon_rebuild --persona <id>
"""

from __future__ import annotations

import argparse
import sys

from discord_llm_friends import canon
from discord_llm_friends import personas as personas_module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--persona", required=True, help="persona id")
    args = parser.parse_args(argv)

    persona = personas_module.load(args.persona)
    if not persona.canon_ledger_path.exists():
        print(
            f"ERROR: no ledger at {persona.canon_ledger_path} — nothing to "
            f"rebuild from. Seed first (pipeline.canon_seed).",
            file=sys.stderr,
        )
        return 2

    stats = canon.rebuild_collection(persona)
    print(
        f"rebuilt {canon.collection_name(persona.id)}: "
        f"{stats['facts']} active fact(s) → {stats['rows']} row(s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
