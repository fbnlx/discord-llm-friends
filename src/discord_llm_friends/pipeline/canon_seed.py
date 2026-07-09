"""Seed a Synthetic persona's canon from its World bible.

Reads personas/<id>/canon/world_bible.json — the structured export of the
maintainer-reviewed World bible, where every fact already carries its
retrieval queries (authored at generation time). This step is therefore
purely mechanical: no LLM call, only embeddings.

Expected shape:

    {
      "facts": [
        {
          "fact_id": "WB-2031-03",          # stable id from the bible
          "year": 2031,                      # or null for undated facts
          "track": "hungary",                # world|europe|hungary|personal
          "text": "…one sentence…",
          "queries": ["…", "…", "…"],        # 3-7 mixed-register questions
          "tags": ["…"],                     # optional extra tags
          "date_scope": "2031-2035"          # optional; overrides year
        }, …
      ]
    }

Facts already in the ledger (same fact text) are skipped, so re-running
after a bible edit ingests only the new/changed ones. Removing or reworking
already-ledgered facts is consolidation's job, not the seeder's.

Run as:
    uv run python -m discord_llm_friends.pipeline.canon_seed --persona <id> [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys

from discord_llm_friends import canon
from discord_llm_friends import personas as personas_module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--persona", required=True, help="persona id")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print what would be ingested without writing anything",
    )
    args = parser.parse_args(argv)

    persona = personas_module.load(args.persona)
    if not persona.canon.enabled:
        print(
            f"WARNING: canon is not enabled for {persona.id} (persona.yaml "
            f"canon.enabled) — seeding anyway; the runtime will ignore it "
            f"until enabled.",
            file=sys.stderr,
        )

    path = persona.world_bible_path
    if not path.exists():
        print(f"ERROR: world bible not found: {path}", file=sys.stderr)
        return 2

    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("facts") or []

    facts: list[canon.CanonFact] = []
    skipped_invalid = 0
    for entry in entries:
        text = str(entry.get("text", "")).strip()
        queries = [
            str(q).strip() for q in (entry.get("queries") or [])
            if str(q).strip()
        ]
        if not text or not queries:
            skipped_invalid += 1
            continue
        year = entry.get("year")
        date_scope = str(
            entry.get("date_scope") or (str(year) if year else "undated")
        ).strip()
        if not canon._DATE_SCOPE_RE.match(date_scope):
            date_scope = "undated"
        track = entry.get("track")
        tags = ([str(track)] if track else []) + [
            str(t) for t in (entry.get("tags") or [])
        ]
        facts.append(canon.CanonFact.new(
            fact=text,
            queries=queries,
            date_scope=date_scope,
            tags=tags,
            source="seed",
            provenance={"fact_id": entry.get("fact_id")},
        ))

    existing = {f.id for f in canon.read_ledger(persona)}
    new = [f for f in facts if f.id not in existing]

    print(f"world bible facts : {len(entries)}")
    print(f"valid             : {len(facts)}  (skipped invalid: {skipped_invalid})")
    print(f"already in ledger : {len(facts) - len(new)}")
    print(f"to ingest         : {len(new)}")

    if args.dry_run:
        for fact in new[:20]:
            print(f"  [{fact.date_scope}] {fact.fact[:110]}")
        if len(new) > 20:
            print(f"  … and {len(new) - 20} more")
        print("(dry run — nothing written)")
        return 0

    if new:
        canon.add_facts(persona, new)
    rows = sum(len(f.queries) for f in new)
    print(
        f"ingested {len(new)} fact(s) / {rows} row(s) into "
        f"{canon.collection_name(persona.id)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
