"""Embed cleaned corpus per persona into ChromaDB.

Run as: `uv run python -m discord_llm_friends.pipeline.embed --persona <id>`.

Pipeline:
  1. Load personas/<id>/cleaned.json — the DOCUMENTS we'll store + return
     at retrieval time.
  2. If personas/<id>/synthetic_queries.json exists and aligns, use it
     as the embedding INPUT (doc2query mode); otherwise embed the
     cleaned text itself (legacy mode).
  3. Hash (embedding-input + document) → stable id across runs.
     Including the document keeps two comments that happened to get the
     same synthetic query from colliding; including the input means
     re-running synth_queries (which regenerates inputs) correctly
     forces a re-embed of the changed entries.
  4. If the existing collection was built with a different
     embedding_input_kind, drop and recreate it (vector spaces don't
     mix).
  5. Embed only the not-yet-stored entries via CONFIG.embedding.model,
     in batches of CONFIG.embedding.batch_size.
  6. Store each new entry with vector + cleaned-text document + light
     metadata.

Idempotent within a mode: re-running on the same inputs does nothing.
Appending to cleaned.json only embeds the new ones. Switching modes
(cleaned ↔ synth_query) triggers a one-time full rebuild.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter

import chromadb
from chromadb.config import Settings
from openai import APIError, OpenAI, RateLimitError

from discord_llm_friends import config as cfg
from discord_llm_friends import personas as personas_module


def _entry_id(text: str) -> str:
    """Deterministic 16-hex-char id. 64 bits of entropy — collision-free
    for any corpus we'd realistically hold here."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _load_cleaned(persona_id: str) -> list[str]:
    persona = personas_module.load(persona_id)
    path = persona.cleaned_path
    if not path.exists():
        raise FileNotFoundError(
            f"cleaned file not found: {path}\n"
            f"run `uv run python -m discord_llm_friends.pipeline.cleanup "
            f"--persona {persona_id}` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _load_synth_queries(persona_id: str, expected_len: int) -> list[str] | None:
    """Try to load the doc2query sidecar. Returns the queries list on
    success, or None if absent / misaligned / has any unfilled slots
    (embed needs every slot populated; partial sidecars are unsafe).

    Side effect: prints why we're falling back when the file exists but
    isn't usable — so the user notices and can re-run synth_queries.
    """
    persona = personas_module.load(persona_id)
    path = persona.folder / "synthetic_queries.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(
            f"WARNING: {path.name} is malformed JSON ({e}) — falling back "
            f"to embedding cleaned text directly.",
            file=sys.stderr,
        )
        return None
    if not isinstance(data, list) or len(data) != expected_len:
        print(
            f"WARNING: {path.name} length doesn't match cleaned.json "
            f"({len(data) if isinstance(data, list) else type(data).__name__} "
            f"vs {expected_len}) — falling back to embedding cleaned text.",
            file=sys.stderr,
        )
        return None
    missing = [
        i for i, q in enumerate(data)
        if not (isinstance(q, str) and q.strip())
    ]
    if missing:
        print(
            f"WARNING: {path.name} has {len(missing)} unfilled slots "
            f"(first: {missing[:5]}). Re-run pipeline.synth_queries to "
            f"complete it, or pass --no-synth to embed cleaned text "
            f"directly. Falling back for now.",
            file=sys.stderr,
        )
        return None
    return [q.strip() for q in data]


def _get_collection(persona_id: str, embedding_input_kind: str):
    """Persistent ChromaDB collection per persona, no default embedding
    fn (we always provide pre-computed vectors).

    If the existing collection was built with a different
    embedding_input_kind (cleaned_text vs synth_query) or a different
    embedding model, drop and recreate it — the stored vectors are
    incompatible with new queries against a different space.
    """
    client = chromadb.PersistentClient(
        path=str(cfg.CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    existing_names = {c.name for c in client.list_collections()}
    if persona_id in existing_names:
        existing = client.get_collection(persona_id)
        meta = existing.metadata or {}
        stale_reasons: list[str] = []
        if meta.get("embedding_input_kind", "cleaned_text") != embedding_input_kind:
            stale_reasons.append(
                f"input kind {meta.get('embedding_input_kind', 'cleaned_text')!r} "
                f"→ {embedding_input_kind!r}"
            )
        if meta.get("embedding_model") != cfg.CONFIG.embedding.model:
            stale_reasons.append(
                f"model {meta.get('embedding_model')!r} "
                f"→ {cfg.CONFIG.embedding.model!r}"
            )
        if stale_reasons:
            print(
                f"  collection {persona_id!r} stale ({'; '.join(stale_reasons)}) "
                f"— dropping and rebuilding",
                file=sys.stderr,
            )
            client.delete_collection(persona_id)

    return client.get_or_create_collection(
        name=persona_id,
        embedding_function=None,
        metadata={
            "persona_id": persona_id,
            "embedding_model": cfg.CONFIG.embedding.model,
            "embedding_input_kind": embedding_input_kind,
        },
    )


def _find_existing_ids(collection, ids: list[str]) -> set[str]:
    existing: set[str] = set()
    for i in range(0, len(ids), 1000):
        chunk = ids[i:i + 1000]
        if not chunk:
            continue
        result = collection.get(ids=chunk, include=[])
        existing.update(result["ids"])
    return existing


def _embed_with_retry(openai_client: OpenAI, texts: list[str]) -> tuple[list[list[float]], int]:
    max_retries = cfg.CONFIG.embedding.max_retries
    delay = 1.0
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = openai_client.embeddings.create(
                model=cfg.CONFIG.embedding.model,
                input=texts,
            )
            vectors = [d.embedding for d in resp.data]
            tokens = resp.usage.prompt_tokens if resp.usage else 0
            return vectors, tokens
        except RateLimitError as e:
            last_err = e
            print(f"  rate limited (attempt {attempt + 1}/{max_retries}), "
                  f"sleeping {delay:.0f}s...", file=sys.stderr)
            time.sleep(delay)
            delay *= 2
        except APIError as e:
            last_err = e
            if attempt < max_retries - 1:
                print(f"  API error: {e} — retrying in {delay:.0f}s...",
                      file=sys.stderr)
                time.sleep(delay)
                delay *= 2
            else:
                raise
    raise RuntimeError(f"exhausted retries: {last_err}")


def _dedupe_preserving_order(
    items: list[tuple[str, str, str]],
) -> list[tuple[str, str, str]]:
    """Defensive id-level dedup — pipeline.cleanup dedupes by text already.

    Items are (id, embedding_input, document) tuples; dedup is by id.
    """
    seen: set[str] = set()
    out: list[tuple[str, str, str]] = []
    for entry_id, embed_input, document in items:
        if entry_id in seen:
            continue
        seen.add(entry_id)
        out.append((entry_id, embed_input, document))
    return out


def embed_persona(persona_id: str, use_synth: bool = True) -> dict:
    """Embed all not-yet-stored entries. Returns stats.

    If `use_synth` is True and a valid synthetic_queries.json sidecar is
    present, embed the synthetic queries; otherwise embed the cleaned
    text. The original cleaned text is stored as the document either way
    (it's what the runtime engine returns at retrieval).
    """
    documents = _load_cleaned(persona_id)
    queries = _load_synth_queries(persona_id, len(documents)) if use_synth else None

    if queries is not None:
        embedding_inputs = queries
        embedding_input_kind = "synth_query"
        print(
            f"loaded {len(documents)} cleaned entries for {persona_id} "
            f"(mode: doc2query — embedding synthetic queries)"
        )
    else:
        embedding_inputs = documents
        embedding_input_kind = "cleaned_text"
        print(
            f"loaded {len(documents)} cleaned entries for {persona_id} "
            f"(mode: legacy — embedding cleaned text directly)"
        )

    # Id is hashed from BOTH the embedding input AND the document. In
    # doc2query mode two distinct comments can receive the SAME synthetic
    # query (generic questions recur), and hashing the query alone would
    # collide → _dedupe_preserving_order would silently drop one of the
    # documents. Combining with the document keeps every distinct comment
    # indexable, while still changing the id whenever the query is
    # regenerated (so re-running synth_queries forces a correct re-embed).
    ids = [
        _entry_id(f"{e}\x00{d}")
        for e, d in zip(embedding_inputs, documents)
    ]
    collection = _get_collection(persona_id, embedding_input_kind)

    print("checking existing entries in collection...")
    already_have = _find_existing_ids(collection, ids)
    print(f"  {len(already_have)} already embedded")

    # Tuple shape: (id, embedding_input, document). Embedding goes
    # through the input; document gets stored verbatim for retrieval.
    new_items = _dedupe_preserving_order(
        [
            (i, e, d)
            for i, e, d in zip(ids, embedding_inputs, documents)
            if i not in already_have
        ]
    )

    # Counter holds integer counters; the string-valued `mode` is added
    # only at the dict-conversion step at the end of this function.
    stats: Counter[str] = Counter()
    stats["total"] = len(documents)
    stats["skipped_existing"] = len(already_have)
    stats["dropped_id_collision"] = (
        len(documents) - len(already_have) - len(new_items)
    )

    if not new_items:
        print("nothing new to embed.")
        out = dict(stats)
        out["mode"] = embedding_input_kind
        return out

    batch_size = cfg.CONFIG.embedding.batch_size
    print(f"embedding {len(new_items)} new entries in batches of {batch_size}...")

    openai_client = OpenAI()
    total_tokens = 0

    for batch_start in range(0, len(new_items), batch_size):
        batch = new_items[batch_start:batch_start + batch_size]
        batch_ids = [b[0] for b in batch]
        batch_inputs = [b[1] for b in batch]
        batch_docs = [b[2] for b in batch]
        batch_metas = [
            {
                "persona_id": persona_id,
                "doc_char_len": len(d),
                "embed_char_len": len(e),
                "embedding_input_kind": embedding_input_kind,
            }
            for d, e in zip(batch_docs, batch_inputs)
        ]

        vectors, tokens = _embed_with_retry(openai_client, batch_inputs)
        total_tokens += tokens

        collection.add(
            ids=batch_ids,
            embeddings=vectors,
            documents=batch_docs,
            metadatas=batch_metas,
        )
        stats["added"] += len(batch)
        done = min(batch_start + batch_size, len(new_items))
        print(f"  {done}/{len(new_items)} embedded "
              f"({total_tokens:,} tokens so far)")

    stats["tokens_used"] = total_tokens
    # text-embedding-3-large: $0.13
    rate_per_million = 0.13 if "large" in cfg.CONFIG.embedding.model else 0.02
    stats["estimated_cost_usd"] = round(
        total_tokens / 1_000_000 * rate_per_million, 6,
    )
    out = dict(stats)
    out["mode"] = embedding_input_kind
    return out


def _format_stats(persona_id: str, stats: dict) -> str:
    lines = [
        "",
        f"done — persona={persona_id}",
        f"  mode                  : {stats.get('mode', '?')}",
        f"  total cleaned entries : {stats['total']}",
        f"  already in collection : {stats.get('skipped_existing', 0)}",
        f"  newly added           : {stats.get('added', 0)}",
    ]
    if stats.get("dropped_id_collision"):
        lines.append(f"  id-collision drops    : {stats['dropped_id_collision']}")
    if stats.get("tokens_used"):
        lines.append(f"  input tokens used     : {stats['tokens_used']:,}")
        lines.append(f"  estimated cost (USD)  : ${stats['estimated_cost_usd']:.6f}")
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
    parser.add_argument(
        "--no-synth", action="store_true",
        help=(
            "force the legacy path: embed cleaned text directly, "
            "ignoring any synthetic_queries.json sidecar."
        ),
    )
    args = parser.parse_args(argv)

    if not os.getenv("OPENAI_API_KEY"):
        print(
            "ERROR: OPENAI_API_KEY is not set. Copy .env.example to .env "
            "and fill it in.",
            file=sys.stderr,
        )
        return 2

    stats = embed_persona(args.persona, use_synth=not args.no_synth)
    print(_format_stats(args.persona, stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
