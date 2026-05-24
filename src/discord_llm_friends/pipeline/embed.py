"""Embed cleaned corpus per persona into ChromaDB.

Run as: `uv run python -m discord_llm_friends.pipeline.embed --persona <id>`.

Pipeline:
  1. Load personas/<id>/cleaned.json.
  2. Hash each entry with SHA-256 (first 16 chars) → stable id across runs.
  3. Check which ids already exist in the persona's ChromaDB collection.
  4. Embed only the new entries via the configured embedding model,
     in batches of CONFIG.embedding.batch_size.
  5. Store each new entry with embedding + document + light metadata.

Idempotent: re-running on the same input does nothing. Re-running after
appending new entries to cleaned.json only embeds the new ones.
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


def _get_collection(persona_id: str):
    """Persistent ChromaDB collection per persona, no default embedding fn
    (we always provide pre-computed vectors)."""
    client = chromadb.PersistentClient(
        path=str(cfg.CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=persona_id,
        embedding_function=None,
        metadata={
            "persona_id": persona_id,
            "embedding_model": cfg.CONFIG.embedding.model,
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


def _dedupe_preserving_order(items: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Defensive id-level dedup — pipeline.cleanup dedupes by text already."""
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for entry_id, text in items:
        if entry_id in seen:
            continue
        seen.add(entry_id)
        out.append((entry_id, text))
    return out


def embed_persona(persona_id: str) -> dict:
    """Embed all not-yet-stored cleaned entries. Returns stats."""
    entries = _load_cleaned(persona_id)
    ids = [_entry_id(c) for c in entries]
    print(f"loaded {len(entries)} cleaned entries for {persona_id}")

    collection = _get_collection(persona_id)

    print("checking existing entries in collection...")
    already_have = _find_existing_ids(collection, ids)
    print(f"  {len(already_have)} already embedded")

    new_items = _dedupe_preserving_order(
        [(i, c) for i, c in zip(ids, entries) if i not in already_have]
    )

    stats: Counter[str] = Counter()
    stats["total"] = len(entries)
    stats["skipped_existing"] = len(already_have)
    stats["dropped_id_collision"] = (
        len(entries) - len(already_have) - len(new_items)
    )

    if not new_items:
        print("nothing new to embed.")
        return dict(stats)

    batch_size = cfg.CONFIG.embedding.batch_size
    print(f"embedding {len(new_items)} new entries in batches of {batch_size}...")

    openai_client = OpenAI()
    total_tokens = 0

    for batch_start in range(0, len(new_items), batch_size):
        batch = new_items[batch_start:batch_start + batch_size]
        batch_ids = [b[0] for b in batch]
        batch_texts = [b[1] for b in batch]
        batch_metas = [
            {"persona_id": persona_id, "char_len": len(t)}
            for t in batch_texts
        ]

        vectors, tokens = _embed_with_retry(openai_client, batch_texts)
        total_tokens += tokens

        collection.add(
            ids=batch_ids,
            embeddings=vectors,
            documents=batch_texts,
            metadatas=batch_metas,
        )
        stats["added"] += len(batch)
        done = min(batch_start + batch_size, len(new_items))
        print(f"  {done}/{len(new_items)} embedded "
              f"({total_tokens:,} tokens so far)")

    stats["tokens_used"] = total_tokens
    # text-embedding-3-small: $0.02 / 1M input tokens
    stats["estimated_cost_usd"] = round(total_tokens / 1_000_000 * 0.02, 6)
    return dict(stats)


def _format_stats(persona_id: str, stats: dict) -> str:
    lines = [
        "",
        f"done — persona={persona_id}",
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
    args = parser.parse_args(argv)

    if not os.getenv("OPENAI_API_KEY"):
        print(
            "ERROR: OPENAI_API_KEY is not set. Copy .env.example to .env "
            "and fill it in.",
            file=sys.stderr,
        )
        return 2

    stats = embed_persona(args.persona)
    print(_format_stats(args.persona, stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
