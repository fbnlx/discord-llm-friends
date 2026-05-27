"""Generate per-document synthetic queries for doc2query indexing.

For each cleaned-corpus entry, asks Gemini what user question, topic, or
trigger sentence would most plausibly prompt that exact comment as a
reply. The output is written to a sidecar file
`personas/<id>/synthetic_queries.json` — a JSON array PARALLEL to
`cleaned.json` (same length, position i ↔ position i).

The next pipeline step (`pipeline.embed`) reads this sidecar and embeds
the synthetic queries into ChromaDB (with the ORIGINAL cleaned comment
still stored as the retrieval document). This shifts the embedding
vector space from answer-shaped (forum-comment phrasings) to question-
shaped (what a Discord user would ask), which improves recall on the
runtime question → corpus matching path. Helps most when:
  - the source language is one where small embedders struggle (Hungarian)
  - user questions and corpus comments live in very different registers
    (a polite question vs. a screaming forum reply)

Run as:
    uv run python -m discord_llm_friends.pipeline.synth_queries --persona <id>

Resumable: re-running picks up where the last run left off (unfilled
slots are `None` in the sidecar). `--rebuild` forces full regeneration.
`--dry-run` previews the first batch without writing.

Always uses the gemini-configured model (CONFIG.llm.models['gemini'])
regardless of CONFIG.llm.provider — the runtime provider and the doc2query
provider are independent choices.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from discord_llm_friends import config as cfg
from discord_llm_friends import personas as personas_module


# --- Prompt construction ---------------------------------------------------

def _response_schema(n: int) -> dict:
    """JSON schema dict for the structured Gemini output.

    Pinning minItems == maxItems == n forces a same-length array so we
    can position-match input comments to output queries without
    secondary alignment heuristics.
    """
    return {
        "type": "array",
        "minItems": n,
        "maxItems": n,
        "items": {"type": "string"},
    }


def _system_instruction(language: str) -> str:
    return (
        "You are reverse-engineering online forum comments to expose what "
        "each one was likely written in response to.\n"
        "\n"
        "For each numbered comment in the user message, write ONE short "
        f"{language} string that is the most plausible question, topic, "
        "or trigger sentence that would prompt that exact comment as a "
        "reply.\n"
        "\n"
        "Rules:\n"
        "- Output a JSON array of strings, one per input comment, in the "
        "same order.\n"
        f"- Each output is in {language}, ≤ 25 words.\n"
        "- Phrase it like something a real user would actually ask or "
        "say — colloquial register, not formal.\n"
        "- Do not echo the comment verbatim, do not add commentary, do "
        "not attribute or quote.\n"
        "- A short comment is usually still a direct answer to a simple "
        "question — write that question. Only if a comment is genuinely "
        "context-free (a bare interjection or keyboard-mash with no "
        f"inferable topic) output a 2-5 word {language} topic label "
        "instead.\n"
    )


def _format_batch(comments: list[str]) -> str:
    """Pack the batch into a numbered single-message input. Flatten
    newlines so each comment occupies one line — keeps the model's
    counting unambiguous."""
    lines = ["Comments:"]
    for i, c in enumerate(comments, start=1):
        flat = c.replace("\n", " ").replace("\r", " ")
        lines.append(f"{i}. {flat}")
    return "\n".join(lines)


# --- LLM dispatch ----------------------------------------------------------

async def _generate_one_batch(
    aio_client,
    model: str,
    system: str,
    comments: list[str],
    max_retries: int,
) -> list[str]:
    """One LLM call → an aligned list of N synthetic queries for N input
    comments. Raises after exhausting retries; caller handles fallback."""
    from google.genai import types

    schema = _response_schema(len(comments))
    contents = _format_batch(comments)

    last_err: Exception | None = None
    delay = 1.0
    for attempt in range(max_retries):
        try:
            resp = await aio_client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    response_json_schema=schema,
                    # Low-ish but not zero: we want some lexical variety
                    # in trigger phrasings, not deterministic templates.
                    temperature=0.4,
                    # Match engine.py — thinking tokens can leak into
                    # the visible JSON and break parsing.
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            text = (resp.text or "").strip()
            arr = json.loads(text)
            if not isinstance(arr, list):
                raise ValueError(f"non-list response: {text[:200]!r}")
            if len(arr) != len(comments):
                raise ValueError(
                    f"length mismatch: got {len(arr)}, expected {len(comments)}"
                )
            return [str(x).strip() for x in arr]
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                print(
                    f"  batch retry {attempt + 1}/{max_retries} after "
                    f"{type(e).__name__}: {e}",
                    file=sys.stderr,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)
            else:
                raise
    raise RuntimeError(f"exhausted retries: {last_err}")


async def _generate_batch_with_fallback(
    aio_client,
    model: str,
    system: str,
    comments: list[str],
    max_retries: int,
) -> list[str]:
    """Try the full batch; on persistent failure, fall back to one LLM
    call per comment (so a single bad apple can't poison the batch).
    Last-resort: if even a singleton fails, use the comment itself as
    the embedding text — keeps the entry retrievable, just bypasses
    doc2query for this row."""
    try:
        return await _generate_one_batch(
            aio_client, model, system, comments, max_retries,
        )
    except Exception:
        print(
            f"  batch failed — falling back to {len(comments)} singletons",
            file=sys.stderr,
        )
        out: list[str] = []
        for c in comments:
            try:
                singleton = await _generate_one_batch(
                    aio_client, model, system, [c], max_retries,
                )
                out.append(singleton[0])
            except Exception as e:
                print(
                    f"    singleton failed (err={e}) — using comment text as fallback",
                    file=sys.stderr,
                )
                out.append(c)
        return out


# --- Sidecar I/O -----------------------------------------------------------

def _load_existing_sidecar(path: Path, expected_len: int) -> list[str | None]:
    """Read the parallel-array sidecar if present and well-aligned.
    Returns a list of length `expected_len` with None for missing slots."""
    if not path.exists():
        return [None] * expected_len
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(
            f"WARNING: {path} is malformed JSON — starting fresh",
            file=sys.stderr,
        )
        return [None] * expected_len
    if not isinstance(data, list) or len(data) != expected_len:
        print(
            f"WARNING: {path} length doesn't match cleaned.json "
            f"({len(data) if isinstance(data, list) else type(data).__name__} "
            f"vs {expected_len}) — starting fresh",
            file=sys.stderr,
        )
        return [None] * expected_len
    return [
        x if (isinstance(x, str) and x.strip()) else None
        for x in data
    ]


def _atomic_write_json(path: Path, data) -> None:
    """Write JSON via tmp+rename so a crash mid-write can't corrupt the
    sidecar (the embed step depends on its integrity)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


# --- Orchestration ---------------------------------------------------------

async def _run_async(
    persona_id: str,
    rebuild: bool,
    dry_run: bool,
) -> dict:
    from google import genai

    persona = personas_module.load(persona_id)
    cleaned = personas_module.load_cleaned(persona)
    sidecar = persona.folder / "synthetic_queries.json"

    synth_cfg = cfg.CONFIG.synth_queries
    out_language = synth_cfg.output_language or persona.language

    print(
        f"persona={persona_id} cleaned={len(cleaned)} "
        f"sidecar={sidecar.relative_to(cfg.ROOT)} "
        f"language={out_language!r}",
        file=sys.stderr,
    )

    if rebuild or not sidecar.exists():
        results: list[str | None] = [None] * len(cleaned)
        if rebuild and sidecar.exists():
            print(f"  --rebuild: discarding existing {sidecar.name}", file=sys.stderr)
    else:
        results = _load_existing_sidecar(sidecar, len(cleaned))
        already = sum(1 for r in results if r is not None)
        print(f"  resuming: {already} already done", file=sys.stderr)

    pending_indices = [i for i, r in enumerate(results) if r is None]
    if not pending_indices:
        print("nothing to generate.", file=sys.stderr)
        if not dry_run:
            _atomic_write_json(sidecar, results)
        return {
            "total": len(cleaned),
            "generated": 0,
        }

    batches: list[list[int]] = []
    bs = synth_cfg.batch_size
    for start in range(0, len(pending_indices), bs):
        batches.append(pending_indices[start:start + bs])

    print(
        f"  generating {len(pending_indices)} synthetic queries — "
        f"{len(batches)} batches of up to {bs}, concurrency={synth_cfg.concurrency}",
        file=sys.stderr,
    )

    model = cfg.CONFIG.llm.models["gemini"]
    aio_client = genai.Client().aio
    system = _system_instruction(out_language)

    sem = asyncio.Semaphore(synth_cfg.concurrency)
    completed = 0
    start_time = time.monotonic()
    pending_total = len(pending_indices)

    async def run_batch(batch_indices: list[int]) -> tuple[list[int], list[str]]:
        nonlocal completed
        comments = [cleaned[i] for i in batch_indices]
        async with sem:
            queries = await _generate_batch_with_fallback(
                aio_client, model, system, comments,
                max_retries=synth_cfg.max_retries,
            )
        completed += len(batch_indices)
        elapsed = time.monotonic() - start_time
        rate = completed / elapsed if elapsed > 0 else 0.0
        eta = (pending_total - completed) / rate if rate > 0 else 0.0
        print(
            f"  {completed}/{pending_total} "
            f"({rate:.1f}/s, eta {eta:.0f}s)",
            file=sys.stderr,
        )
        return batch_indices, queries

    if dry_run:
        first = batches[0]
        idx, queries = await run_batch(first)
        print("\n--- dry run: first batch ---", file=sys.stderr)
        for i, q in zip(idx, queries):
            print(f"\nCOMMENT [{i}]: {cleaned[i]!r}", file=sys.stderr)
            print(f"     → QUERY: {q!r}", file=sys.stderr)
        return {
            "total": len(cleaned),
            "generated": len(first),
            "dry_run": True,
        }

    # Checkpoint partial results periodically so a crash mid-run only
    # loses the last few batches.
    CHECKPOINT_EVERY_BATCHES = 5
    batches_done = 0
    tasks = [asyncio.create_task(run_batch(b)) for b in batches]
    for fut in asyncio.as_completed(tasks):
        idx, queries = await fut
        for i, q in zip(idx, queries):
            results[i] = q
        batches_done += 1
        if batches_done % CHECKPOINT_EVERY_BATCHES == 0:
            _atomic_write_json(sidecar, results)

    _atomic_write_json(sidecar, results)
    return {
        "total": len(cleaned),
        "generated": pending_total,
    }


def generate_for_persona(
    persona_id: str,
    rebuild: bool = False,
    dry_run: bool = False,
) -> dict:
    """Sync wrapper for the async core. Returns stats dict."""
    return asyncio.run(_run_async(persona_id, rebuild, dry_run))


# --- CLI -------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--persona", required=True, help="persona id")
    parser.add_argument(
        "--rebuild", action="store_true",
        help="discard existing synthetic_queries.json and regenerate from scratch",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="generate the first batch only, print to stderr, write nothing",
    )
    args = parser.parse_args(argv)

    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        print(
            "ERROR: GOOGLE_API_KEY (or GEMINI_API_KEY) is not set. "
            "synth_queries always uses the gemini-configured model.",
            file=sys.stderr,
        )
        return 2

    if cfg.CONFIG.llm.provider != "gemini":
        print(
            f"NOTE: runtime LLM provider is {cfg.CONFIG.llm.provider!r}; "
            f"synth_queries always uses the gemini-configured model "
            f"({cfg.CONFIG.llm.models['gemini']!r}).",
            file=sys.stderr,
        )

    stats = generate_for_persona(args.persona, args.rebuild, args.dry_run)
    print("\ndone — " + ", ".join(f"{k}={v}" for k, v in stats.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
