"""Consolidate a Synthetic persona's canon (offline, maintainer-run).

The only thing allowed to rewrite canon (the runtime is append-only).
Three passes over the ledger:

  1. MERGE   — embed active fact texts, greedily cluster near-duplicates
               (--merge-distance, same squared-L2 scale as Chroma), reduce
               each cluster to one fact via call_llm; queries are unioned
               mechanically, the earliest created_at is kept, losers are
               marked status="superseded" pointing at the winner.
  2. PROMOTE — one LLM pass proposes load-bearing dated facts for the
               Timeline sheet; proposals are written to
               canon/timeline.proposed.md for maintainer review. This job
               NEVER edits timeline.md itself.
  3. COMPACT — rewrite facts.jsonl atomically (tmp-then-rename; the prior
               file is kept as facts.jsonl.bak) and rebuild the collection
               from surviving active records.

--dry-run prints the merge plan (LLM-free) and skips all writes.

Run as:
    uv run python -m discord_llm_friends.pipeline.canon_consolidate --persona <id> [--dry-run]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys

import numpy as np

from discord_llm_friends import canon, engine
from discord_llm_friends import personas as personas_module


def _cluster_near_duplicates(
    facts: list[canon.CanonFact], threshold: float,
) -> list[list[int]]:
    """Greedy union-find over pairs of fact texts closer than `threshold`
    (squared L2 on unit-norm embeddings, i.e. 2 - 2*cosine — the same
    scale Chroma reports). Returns clusters of indices, singletons omitted."""
    vecs = np.asarray(canon._embed([f.fact for f in facts]))
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    dists = 2.0 - 2.0 * (vecs @ vecs.T)

    parent = list(range(len(facts)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(facts)):
        for j in range(i + 1, len(facts)):
            if dists[i, j] < threshold:
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(len(facts)):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) > 1]


def _merge_cluster(
    persona: personas_module.Persona, members: list[canon.CanonFact],
) -> canon.CanonFact | None:
    """LLM-reduce one cluster to a single fact. Returns None on failure
    (the cluster is then left unmerged — safe default)."""
    numbered = "\n".join(
        f"{i + 1}. [{f.date_scope}] {f.fact}" for i, f in enumerate(members)
    )
    system = (
        f"You merge duplicate canon facts about {persona.display_name}'s "
        f"fictional world into ONE fact. Output STRICT JSON, nothing else:\n"
        f'{{"fact": "...", "date_scope": "...", "tags": ["..."]}}\n'
        f"Rules: one self-contained third-person {persona.language} "
        f"sentence carrying ALL non-contradictory specifics from the "
        f"inputs; if inputs conflict, the EARLIEST (first listed) wins; "
        f"date_scope like \"2032\" or \"2031-2035\" or \"undated\"; 1-3 "
        f"lowercase tags."
    )
    try:
        raw = engine.call_llm(system, f"FACTS (earliest first):\n{numbered}\n\nJSON:")
        text = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(text)
        fact_text = str(data["fact"]).strip()
        if not fact_text:
            return None
        date_scope = str(data.get("date_scope", "undated")).strip()
        if not canon._DATE_SCOPE_RE.match(date_scope):
            date_scope = "undated"
        queries: list[str] = []
        for m in members:
            for q in m.queries:
                if q not in queries:
                    queries.append(q)
        merged = canon.CanonFact.new(
            fact=fact_text,
            queries=queries[:7],
            date_scope=date_scope,
            tags=list(data.get("tags") or [])[:3],
            source="seed" if any(m.source == "seed" for m in members) else "emergent",
            provenance={"merged_from": [m.id for m in members]},
        )
        # Keep the earliest timestamp — the merge inherits, not restarts.
        return dataclasses.replace(
            merged, created_at=min(m.created_at for m in members if m.created_at),
        )
    except Exception:
        return None


def _propose_promotions(
    persona: personas_module.Persona,
    facts: list[canon.CanonFact],
    max_promotions: int,
) -> str | None:
    """One LLM pass proposing Timeline-sheet promotions. Returns the
    proposal markdown, or None on failure/empty."""
    dated = [f for f in facts if f.date_scope != "undated"]
    if not dated:
        return None
    listing = "\n".join(f"- [{f.date_scope}] {f.fact}" for f in dated)
    system = (
        f"You maintain the always-injected Timeline sheet of "
        f"{persona.display_name}'s fictional world. From the candidate "
        f"canon facts, pick AT MOST {max_promotions} that are load-bearing "
        f"for global consistency — dated facts other answers must not "
        f"contradict (who governed when, wars, disasters, major life "
        f"events) — and that are NOT already covered by the current "
        f"timeline. Output them as {persona.language} markdown bullets "
        f"('- YYYY: ...'), chronological, nothing else. Output NOTHING if "
        f"none qualify."
    )
    user = (
        f"CURRENT TIMELINE:\n{persona.timeline or '(empty)'}\n\n"
        f"CANDIDATE FACTS:\n{listing}\n\nPROPOSED ADDITIONS:"
    )
    try:
        proposal = engine.call_llm(system, user).strip()
        return proposal or None
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--persona", required=True, help="persona id")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the merge plan; write nothing, no LLM")
    parser.add_argument("--merge-distance", type=float, default=0.25,
                        help="fact-text distance below which facts are "
                             "considered duplicates (default: 0.25)")
    parser.add_argument("--max-promotions", type=int, default=10,
                        help="max timeline promotions to propose (default: 10)")
    args = parser.parse_args(argv)

    persona = personas_module.load(args.persona)
    ledger = canon.read_ledger(persona)
    if not ledger:
        print(f"ERROR: empty/missing ledger for {persona.id}", file=sys.stderr)
        return 2
    active = [f for f in ledger if f.status == "active"]
    inactive = [f for f in ledger if f.status != "active"]
    print(f"ledger: {len(ledger)} record(s), {len(active)} active")

    clusters = (
        _cluster_near_duplicates(active, args.merge_distance)
        if len(active) > 1 else []
    )
    print(f"near-duplicate clusters (< {args.merge_distance}): {len(clusters)}")
    for group in clusters:
        print("  cluster:")
        for idx in group:
            print(f"    [{active[idx].date_scope}] {active[idx].fact[:100]}")

    if args.dry_run:
        print("(dry run — no merges, no promotions, nothing written)")
        return 0

    # MERGE
    superseded: dict[str, str] = {}  # loser id -> winner id
    merged_facts: list[canon.CanonFact] = []
    for group in clusters:
        members = sorted((active[i] for i in group), key=lambda f: f.created_at)
        winner = _merge_cluster(persona, members)
        if winner is None:
            print(f"  merge failed for a {len(members)}-fact cluster — left unmerged")
            continue
        merged_facts.append(winner)
        for m in members:
            superseded[m.id] = winner.id

    survivors = [f for f in active if f.id not in superseded] + merged_facts
    losers = [
        dataclasses.replace(f, status="superseded", superseded_by=superseded[f.id])
        for f in active if f.id in superseded
    ]
    print(f"merged {len(superseded)} fact(s) into {len(merged_facts)}; "
          f"{len(survivors)} active after consolidation")

    # PROMOTE (proposal only — never edits timeline.md)
    proposal = _propose_promotions(persona, survivors, args.max_promotions)
    if proposal:
        proposed_path = persona.canon_dir / "timeline.proposed.md"
        proposed_path.write_text(
            "# Timeline promotions proposed by canon_consolidate\n"
            "# Review and merge into timeline.md by hand; then delete this file.\n\n"
            + proposal + "\n",
            encoding="utf-8",
        )
        print(f"timeline promotions proposed → {proposed_path}")
    else:
        print("no timeline promotions proposed")

    # COMPACT (atomic; prior ledger kept as .bak)
    path = persona.canon_ledger_path
    path.replace(path.with_suffix(".jsonl.bak"))
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for fact in inactive + losers + survivors:
            f.write(json.dumps(dataclasses.asdict(fact), ensure_ascii=False) + "\n")
    tmp.replace(path)
    print(f"ledger compacted (backup: {path.with_suffix('.jsonl.bak').name})")

    stats = canon.rebuild_collection(persona)
    print(f"collection rebuilt: {stats['facts']} fact(s) → {stats['rows']} row(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
