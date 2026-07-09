"""Preview what retrieval would pull for a given user query.

Run as:
    uv run python -m discord_llm_friends.dev.query_check \\
        --persona <id> --query "your question here" [--top-k 10]

Embeds the query, queries the persona's ChromaDB collection, prints
the nearest stored documents with cosine distances. The LLM is never
called — useful for eyeballing whether the corpus retrieves topically
relevant entries before running the full engine.
"""

from __future__ import annotations

import argparse
import os
import sys

import chromadb
from chromadb.config import Settings
from openai import OpenAI

from discord_llm_friends import canon
from discord_llm_friends import config as cfg
from discord_llm_friends import personas as personas_module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--persona", required=True, help="persona id")
    parser.add_argument("--query", required=True, help="query string")
    parser.add_argument(
        "--top-k", type=int, default=cfg.CONFIG.retrieval.top_k,
        help=f"how many results to show (default: {cfg.CONFIG.retrieval.top_k})",
    )
    parser.add_argument(
        "--canon", action="store_true",
        help="query the persona's canon collection (<id>__canon) instead of "
             "the corpus — for tuning canon.max_distance",
    )
    args = parser.parse_args(argv)

    # Validate persona exists (raises with a useful message if not).
    personas_module.load(args.persona)

    if not os.getenv("OPENAI_API_KEY"):
        print(
            "ERROR: OPENAI_API_KEY not found. Make sure .env exists at "
            "the repo root and contains a real key.",
            file=sys.stderr,
        )
        return 2

    print(f"query   : {args.query!r}")
    print(f"persona : {args.persona}{' (canon)' if args.canon else ''}")
    print(f"top-k   : {args.top_k}")
    print()

    openai_client = OpenAI()
    resp = openai_client.embeddings.create(
        model=cfg.CONFIG.embedding.model,
        input=[args.query],
    )
    query_vec = resp.data[0].embedding

    chroma = chromadb.PersistentClient(
        path=str(cfg.CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    collection_name = (
        canon.collection_name(args.persona) if args.canon else args.persona
    )
    collection = chroma.get_collection(name=collection_name)
    results = collection.query(query_embeddings=[query_vec], n_results=args.top_k)

    docs = results["documents"][0]
    dists = results["distances"][0]
    metas = results["metadatas"][0]
    print(f"top {args.top_k} nearest entries (distance: smaller = closer):")
    print()
    for i, (doc, dist, meta) in enumerate(zip(docs, dists, metas), start=1):
        snippet = doc.replace("\n", " ⏎ ")
        if len(snippet) > 160:
            snippet = snippet[:160] + "..."
        scope = (
            f"  [{(meta or {}).get('date_scope', '?')}]" if args.canon else ""
        )
        print(f"  {i:2}.  dist={dist:.3f}{scope}  {snippet}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
