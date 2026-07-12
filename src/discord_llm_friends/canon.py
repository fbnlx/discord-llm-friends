"""Canon memory for Synthetic personas (ADR-0006).

A Synthetic persona (e.g. a future edition of a real persona) is allowed
to INVENT facts about its world. Inventions must stay consistent across
conversations, so they are recorded as Canon facts in three places:

  personas/<id>/canon/facts.jsonl   — append-only ledger, the source of truth
  <id>__canon ChromaDB collection   — derived retrieval cache (rebuildable
                                      from the ledger at any time)
  personas/<id>/canon/timeline.md   — Timeline sheet: always-injected dated
                                      skeleton (loaded by personas.py)

Write path (the Canonizer): after a response has been sent to the user,
`canonize_exchange` extracts NEW world/life facts the response asserted,
drops candidates that duplicate existing canon (embedding distance), then
commits survivors — ledger first, collection second. First written wins:
canon is never mutated at runtime; merging/superseding is the offline
consolidation job's business.

Read path: engine._retrieve_canon queries the collection per request and
injects the nearest facts into the prompt as non-negotiable context.

This module deliberately breaks the Pipeline/Runtime split (the runtime
embeds and calls an LLM after responding) — per-persona gated via the
`canon:` section in persona.yaml, quota-exempt, fire-and-forget.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from discord_llm_friends import config as cfg
from discord_llm_friends import personas as personas_module
from discord_llm_friends.history import Exchange, format_history


logger = logging.getLogger(__name__)

CANON_SUFFIX = "__canon"

_DATE_SCOPE_RE = re.compile(r"^(\d{4}(-\d{4})?|undated)$")


def collection_name(persona_id: str) -> str:
    return f"{persona_id}{CANON_SUFFIX}"


def _entry_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _fact_id(fact_text: str) -> str:
    return _entry_id(fact_text.strip())


# --- Fact records -----------------------------------------------------------

@dataclass(frozen=True)
class CanonFact:
    """One atomic canon fact: a self-contained third-person sentence plus
    the doc2query-style retrieval queries that point at it."""

    id: str
    fact: str
    queries: list[str]
    date_scope: str            # "2032" | "2031-2035" | "undated"
    tags: list[str]
    source: str                # "seed" | "emergent"
    created_at: str            # ISO-8601 UTC
    status: str = "active"     # "active" | "superseded" (consolidation only)
    superseded_by: str | None = None
    provenance: dict | None = None

    @classmethod
    def new(
        cls,
        *,
        fact: str,
        queries: list[str],
        date_scope: str,
        tags: list[str],
        source: str,
        provenance: dict | None = None,
    ) -> "CanonFact":
        return cls(
            id=_fact_id(fact),
            fact=fact.strip(),
            queries=[q.strip() for q in queries if q and q.strip()],
            date_scope=date_scope,
            tags=[str(t).strip() for t in tags if str(t).strip()],
            source=source,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            provenance=provenance,
        )


# --- Ledger (source of truth) ----------------------------------------------

def read_ledger(persona: personas_module.Persona) -> list[CanonFact]:
    """All ledger records, oldest first. Malformed lines (e.g. a truncated
    tail after a crash mid-append) are skipped with a warning."""
    path = persona.canon_ledger_path
    if not path.exists():
        return []
    facts: list[CanonFact] = []
    for lineno, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
            facts.append(CanonFact(
                id=raw["id"],
                fact=raw["fact"],
                queries=list(raw.get("queries") or []),
                date_scope=raw.get("date_scope", "undated"),
                tags=list(raw.get("tags") or []),
                source=raw.get("source", "emergent"),
                created_at=raw.get("created_at", ""),
                status=raw.get("status", "active"),
                superseded_by=raw.get("superseded_by"),
                provenance=raw.get("provenance"),
            ))
        except Exception:
            logger.warning("%s:%d: skipping malformed ledger line", path, lineno)
    return facts


def append_ledger(
    persona: personas_module.Persona, facts: list[CanonFact],
) -> None:
    """O_APPEND-style jsonl append — O(1) per fact, crash costs at most one
    truncated tail line (which read_ledger skips). Callers serialize via
    the per-persona lock."""
    path = persona.canon_ledger_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for fact in facts:
            f.write(json.dumps(asdict(fact), ensure_ascii=False) + "\n")


# Per-persona commit lock. The Canonizer runs in asyncio.to_thread worker
# threads, so concurrent same-persona commits are a *threading* concern:
# the lock serializes the dedup-check → insert sequence.
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock(persona_id: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(persona_id, threading.Lock())


# --- Collection (derived cache) ---------------------------------------------

def _embed(texts: list[str]) -> list[list[float]]:
    from discord_llm_friends import engine  # function-level: avoids cycle

    client = engine._openai_client()
    out: list[list[float]] = []
    batch = cfg.CONFIG.embedding.batch_size
    for i in range(0, len(texts), batch):
        resp = client.embeddings.create(
            model=cfg.CONFIG.embedding.model, input=texts[i:i + batch],
        )
        out.extend(d.embedding for d in resp.data)
    return out


def _collection(persona_id: str):
    from discord_llm_friends import engine  # function-level: avoids cycle

    return engine._chroma_client().get_or_create_collection(
        name=collection_name(persona_id),
        metadata={
            "persona_id": persona_id,
            "kind": "canon",
            "embedding_model": cfg.CONFIG.embedding.model,
            "embedding_input_kind": "canon_query",
        },
    )


def _flatten(facts: list[CanonFact]) -> tuple[list[str], list[str], list[str], list[dict]]:
    """One collection row per (query, fact) pair, doc2query style:
    the query is embedded, the fact text is the stored document."""
    ids: list[str] = []
    inputs: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []
    for fact in facts:
        for q in fact.queries:
            ids.append(_entry_id(f"{q}\x00{fact.id}"))
            inputs.append(q)
            docs.append(fact.fact)
            metas.append({
                "type": "canon",
                "fact_id": fact.id,
                "date_scope": fact.date_scope,
                "source": fact.source,
                "tags": ",".join(fact.tags),
                "query": q,
            })
    return ids, inputs, docs, metas


def _add_to_collection(
    persona_id: str,
    facts: list[CanonFact],
    vecs: list[list[float]] | None = None,
) -> int:
    """Upsert facts into the canon collection. `vecs`, when given, must
    align with the flattened (fact, query) order. Returns rows written."""
    ids, inputs, docs, metas = _flatten(facts)
    if not ids:
        return 0
    if vecs is None:
        vecs = _embed(inputs)
    _collection(persona_id).upsert(
        ids=ids, embeddings=vecs, documents=docs, metadatas=metas,
    )
    return len(ids)


def add_facts(persona: personas_module.Persona, facts: list[CanonFact]) -> None:
    """Commit new facts: ledger first (truth), collection second (cache).
    A collection failure is logged, not raised — canon_rebuild reconciles."""
    if not facts:
        return
    append_ledger(persona, facts)
    try:
        _add_to_collection(persona.id, facts)
    except Exception:
        logger.exception(
            "canon collection add failed for %s — facts are safe in the "
            "ledger; run pipeline.canon_rebuild to reconcile", persona.id,
        )


def rebuild_collection(persona: personas_module.Persona) -> dict:
    """Drop and rebuild the canon collection from active ledger records.
    Proof that the collection is a cache; also the model-migration path."""
    from discord_llm_friends import engine  # function-level: avoids cycle

    client = engine._chroma_client()
    try:
        client.delete_collection(collection_name(persona.id))
    except Exception:
        pass  # didn't exist — nothing to drop
    facts = [f for f in read_ledger(persona) if f.status == "active"]
    rows = _add_to_collection(persona.id, facts) if facts else 0
    if not facts:
        _collection(persona.id)  # recreate empty so retrieval finds it
    return {"facts": len(facts), "rows": rows}


# --- The Canonizer (post-response extraction) --------------------------------

def _extraction_prompt(
    persona: personas_module.Persona,
    question: str,
    response: str,
    injected_facts: tuple[str, ...],
    asker_name: str | None = None,
    asker_identity: str | None = None,
    history: list[Exchange] | None = None,
) -> tuple[str, str]:
    name = persona.display_name
    max_new = persona.canon.max_new_facts_per_exchange
    # Follow-up questions/replies are often elliptical ("hány szobás a
    # ház?") — the subject lives in the conversation, which the responder
    # saw. Without it the extractor guesses from the TIMELINE.
    history_rules = (
        (
            f"- The question and reply may be elliptical FOLLOW-UPS to the "
            f"RECENT CONVERSATION — resolve who and what they refer to from "
            f"it (e.g. \"the house\" means the house discussed there, owned "
            f"by whoever owns it there).\n"
            f"- Extract facts ONLY from {name}'s reply below. The RECENT "
            f"CONVERSATION is context for resolving references, never a "
            f"source of new facts — earlier replies were already "
            f"processed.\n"
        )
        if history else ""
    )
    system = (
        f"You maintain the canon fact database for {name}, a fictional "
        f"persona who is allowed to invent facts about their world. You "
        f"extract NEW canonical facts from {name}'s latest reply.\n"
        f"\n"
        f"A canonical fact is a concrete, reusable assertion that must stay "
        f"STABLE if the same question comes back later: (a) events, dates, "
        f"election results, people, places, technologies, outcomes and "
        f"biographical changes in {name}'s world or life — including what "
        f"he asserts about OTHER people's lives (the asker's, his "
        f"friends'); (b) standing "
        f"personal preferences, tastes, habits and self-descriptions {name} "
        f"asserts as durable truths about himself — favourites, loves/hates, "
        f"routines, names he gives to things or people around him. NOT "
        f"canonical: transient moods, jokes, insults, rhetorical "
        f"exaggeration, reactions specific to this one conversation, "
        f"hypotheticals, questions, hedged non-answers, or anything already "
        f"stated in the TIMELINE or ESTABLISHED FACTS (identical or "
        f"paraphrased).\n"
        f"\n"
        f'Output STRICT JSON, nothing else:\n'
        f'{{"facts": [{{"fact": "...", "queries": ["..."], '
        f'"date_scope": "...", "tags": ["..."]}}]}}\n'
        f"\n"
        f"Rules:\n"
        f"- \"fact\": ONE self-contained sentence in {persona.language}, "
        f"third person, about WHOEVER the reply actually asserts it about "
        f"— {name} himself, the asker, or another named person in his "
        f"world. NEVER re-attribute someone else's event to {name}: if the "
        f"reply says the asker will have surgery, the fact is about the "
        f"asker, not {name}. Second-person statements (\"you ...\") are "
        f"about the asker named above the question. Refer to people by the "
        f"short names the TIMELINE and ESTABLISHED FACTS use for them "
        f"(reserve \"{name}\" for {name} himself); write years explicitly. "
        f"Split independent assertions into separate facts.\n"
        f"{history_rules}"
        f"- Ground every fact in what the reply actually asserts. NEVER "
        f"import names, places or details from the TIMELINE that the reply "
        f"did not state. If you cannot determine WHO a fact is about from "
        f"the conversation, question and reply, DROP that fact — never "
        f"guess a subject from the TIMELINE.\n"
        f"- \"queries\": 3-5 short {persona.language} questions a user could "
        f"ask that this fact answers, mixed register (neutral and casual), "
        f"INCLUDING one phrasing at a higher abstraction level (e.g. for an "
        f"election result, also the {persona.language} equivalent of 'who "
        f"was in power then?').\n"
        f"- \"date_scope\": \"2032\" or \"2031-2035\" or \"undated\" (standing "
        f"preferences are usually \"undated\").\n"
        f"- For a standing preference, distill the DURABLE core and drop the "
        f"momentary rant around it (e.g. the {persona.language} equivalent "
        f"of '{name}'s favourite fruit is the cantaloupe' — not his "
        f"complaints about today's supermarkets).\n"
        f"- \"tags\": 1-3 lowercase {persona.language} topic tags.\n"
        f"- At most {max_new} facts. If the reply asserts nothing new and "
        f"concrete, output {{\"facts\": []}}.\n"
        f"- NEVER output a fact that contradicts the TIMELINE or ESTABLISHED "
        f"FACTS. If the reply itself contradicts them, output "
        f"{{\"facts\": []}}."
    )
    injected_block = (
        "\n".join(f"- {f}" for f in injected_facts) if injected_facts
        else "(none)"
    )
    asker_line = ""
    if asker_name:
        asker_line = (
            f" (asked by @{asker_name} — {asker_identity})" if asker_identity
            else f" (asked by @{asker_name})"
        )
    history_block = (
        f"RECENT CONVERSATION (context — the question and reply below "
        f"continue it):\n{format_history(history)}\n"
        f"\n"
        if history else ""
    )
    user = (
        f"TIMELINE (canon):\n{persona.timeline or '(none)'}\n"
        f"\n"
        f"ESTABLISHED FACTS shown to {name} for this reply (canon):\n"
        f"{injected_block}\n"
        f"\n"
        f"{history_block}"
        f"USER QUESTION{asker_line}:\n{question}\n"
        f"\n"
        f"{name.upper()}'S REPLY:\n{response}\n"
        f"\n"
        f"JSON:"
    )
    return system, user


def _parse_extraction(
    persona: personas_module.Persona, raw_text: str, question: str,
    asker_name: str | None = None,
) -> tuple[list[CanonFact], int]:
    """Parse the extractor's JSON into CanonFacts. Returns (facts, invalid
    count). Tolerant where safe (date_scope coerced to 'undated'), strict
    where it matters (fact text, queries)."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
    data = None
    try:
        data = json.loads(text)
    except Exception:
        # Models sometimes wrap the JSON in prose or a stray trailing fence;
        # fall back to the outermost {...} span before giving up.
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(text[start:end + 1])
            except Exception:
                data = None
    if data is None:
        logger.warning(
            "canonizer %s: unparseable extraction output: %.200r",
            persona.id, raw_text,
        )
        return [], 1
    entries = data.get("facts") or []

    facts: list[CanonFact] = []
    invalid = 0
    for entry in entries[: persona.canon.max_new_facts_per_exchange]:
        try:
            fact_text = str(entry["fact"]).strip()
            queries = [str(q).strip() for q in entry.get("queries") or []]
            queries = [q for q in queries if q][:7]
            if not fact_text or not queries:
                invalid += 1
                continue
            date_scope = str(entry.get("date_scope", "undated")).strip()
            if not _DATE_SCOPE_RE.match(date_scope):
                date_scope = "undated"
            facts.append(CanonFact.new(
                fact=fact_text,
                queries=queries,
                date_scope=date_scope,
                tags=list(entry.get("tags") or [])[:3],
                source="emergent",
                provenance=(
                    {"question": question[:300], "asker": asker_name}
                    if asker_name else {"question": question[:300]}
                ),
            ))
        except Exception:
            invalid += 1
    return facts, invalid


def canonize_exchange(
    persona: personas_module.Persona,
    question: str,
    response: str,
    injected_facts: tuple[str, ...] = (),
    asker_name: str | None = None,
    history: list[Exchange] | None = None,
) -> int:
    """Extract and commit new canon facts from one exchange. Returns the
    number of facts stored. Never raises for operational failures — the
    response has already been sent; a lost extraction costs one exchange's
    worth of memory, nothing else."""
    if not (persona.canon.enabled and persona.canon.extraction):
        return 0
    from discord_llm_friends import engine  # function-level: avoids cycle

    # PERSONA_DEBUG_PROMPT=1 makes the Canonizer transparent on stderr, the
    # way [retrieval]/[canon] expose the read path: extraction prompt, raw
    # model output, then a per-fact verdict for every candidate.
    debug = engine.DEBUG_PROMPT

    # The same cast resolution the responder saw: without it the extractor
    # cannot attribute second-person answers ("you will ...") to the asker.
    asker_identity = (
        engine._cast_identity(persona.cast_users, asker_name)
        if asker_name else None
    )
    system, user = _extraction_prompt(
        persona, question, response, injected_facts,
        asker_name=asker_name, asker_identity=asker_identity,
        history=history,
    )
    if debug:
        bar = "─" * 60
        print(
            f"\n{bar}\nCANONIZER PROMPT (persona={persona.id}):\nSYSTEM:\n"
            f"{system}\n{bar}\nUSER:\n{user}\n{bar}",
            file=sys.stderr,
        )
    try:
        raw = engine.call_llm(system, user)
    except Exception:
        logger.warning(
            "canonizer %s: extraction LLM call failed — skipping",
            persona.id, exc_info=True,
        )
        if debug:
            print("[canonizer] extraction LLM call FAILED — skipping",
                  file=sys.stderr)
        return 0
    if debug:
        print(f"[canonizer] raw model output:\n{raw}", file=sys.stderr)

    candidates, invalid = _parse_extraction(persona, raw, question, asker_name)
    if debug:
        print(
            f"[canonizer] parsed {len(candidates)} candidate fact(s), "
            f"{invalid} invalid",
            file=sys.stderr,
        )
    dropped_dup = 0
    kept: list[CanonFact] = []
    if candidates:
        with _lock(persona.id):
            # Exact re-assertions (identical fact text) short-circuit on
            # ledger ids; near-duplicates fall to the embedding check.
            existing_ids = {f.id for f in read_ledger(persona)}
            if debug:
                for c in candidates:
                    if c.id in existing_ids:
                        print(
                            f"[canonizer]   DROP exact-ledger-dup "
                            f"{c.id}: {c.fact[:90]!r}",
                            file=sys.stderr,
                        )
            candidates = [c for c in candidates if c.id not in existing_ids]

            flat_queries = [q for c in candidates for q in c.queries]
            vecs = _embed(flat_queries) if flat_queries else []
            collection = _collection(persona.id)
            populated = collection.count() > 0
            kept_vecs: list[list[float]] = []
            i = 0
            for cand in candidates:
                n = len(cand.queries)
                cand_vecs = vecs[i:i + n]
                i += n
                if populated:
                    res = collection.query(
                        query_embeddings=cand_vecs,
                        n_results=1,
                        include=["distances"],
                    )
                    best = min(
                        (d[0] for d in res["distances"] if d),
                        default=float("inf"),
                    )
                    if best < persona.canon.dedup_max_distance:
                        dropped_dup += 1
                        logger.info(
                            "canonizer %s: dropped duplicate (best=%.3f): %.80r",
                            persona.id, best, cand.fact,
                        )
                        if debug:
                            print(
                                f"[canonizer]   DROP near-dup (best={best:.3f} "
                                f"< {persona.canon.dedup_max_distance}): "
                                f"{cand.fact[:90]!r}",
                                file=sys.stderr,
                            )
                        continue
                    if debug:
                        print(
                            f"[canonizer]   KEEP (best={best:.3f} ≥ "
                            f"{persona.canon.dedup_max_distance}) "
                            f"[{cand.date_scope}] {cand.fact[:90]!r}",
                            file=sys.stderr,
                        )
                elif debug:
                    print(
                        f"[canonizer]   KEEP (collection empty) "
                        f"[{cand.date_scope}] {cand.fact[:90]!r}",
                        file=sys.stderr,
                    )
                kept.append(cand)
                kept_vecs.extend(cand_vecs)

            if kept:
                append_ledger(persona, kept)
                try:
                    _add_to_collection(persona.id, kept, vecs=kept_vecs)
                except Exception:
                    logger.exception(
                        "canon collection add failed for %s — facts are safe "
                        "in the ledger; run pipeline.canon_rebuild",
                        persona.id,
                    )

    logger.info(
        "canonizer persona=%s extracted=%d kept=%d dropped_dup=%d "
        "dropped_invalid=%d",
        persona.id, len(candidates) + dropped_dup, len(kept),
        dropped_dup, invalid,
    )
    if debug:
        print(
            f"[canonizer] kept {len(kept)} / dropped {dropped_dup} near-dup / "
            f"{invalid} invalid → ledger {persona.canon_ledger_path}",
            file=sys.stderr,
        )
    return len(kept)
