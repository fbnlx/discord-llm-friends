# Three-tier persona memory for group-chat corpora

Personas built from multi-party group-chat logs get a layered memory:
**Windows** (speaker-labeled excerpts) and **Stance cards** (synthesized
per-topic opinions) as retrieval units in the persona's ChromaDB
collection, plus a **Dossier** (generated identity profile) appended to
every system prompt. The voice-sample pool is split into its own
artifact (`voice_pool.json`) so multi-speaker excerpts never enter the
voice path. The artifacts are produced offline by the sibling
`clean-job` pipeline and exported into `personas/<id>/` + `chroma/`.

## Context

The original corpus shape was isolated forum comments: one author, one
self-contained document, usable directly as both retrieval document and
voice sample. Group-chat logs broke both assumptions — a lone chat line
("no") is meaningless without neighbors, and no retrieved handful of
raw lines can answer aggregate questions ("what should I play next?")
that span the persona's whole history. LLM thread segmentation of the
logs was tried first and rejected: index-assignment over long sessions
hard-failed 59% above ~45 messages on local models, and the failure is
inherent to the task shape, not promptable away.

## Considered Options

- **Thread segmentation + per-thread retrieval.** Rejected (measured):
  unreliable segmentation, and still answers only topic-shaped
  questions — aggregate asks match nothing.
- **Three-tier memory** (chosen). Deterministic overlapping windows
  (complete by construction, no model in the loop) + claim extraction
  map-reduced into stance cards with category rollups + a dossier per
  persona. Every tier cites downward mechanically: cards cite claim
  ids, claims cite message ids whose speaker must match the claimant.
  Aggregate questions retrieve rollup cards; topical questions retrieve
  windows and topic cards; off-corpus questions fall back on the
  dossier already sitting in the system prompt.

## Consequences

- `cleaned.json`'s historical dual role (RAG source AND voice pool) is
  resolved: the engine samples `voice_pool.json` when present (persona's
  own lines only), falling back to `cleaned.json` for forum-only
  personas. Forum personas are unaffected end to end.
- One collection now mixes unit kinds. Several synthetic queries point
  at the same window/card, so the engine overfetches, dedups by
  `card_id`/`window_id` (document text for legacy rows), and injects up
  to `STANCE_SLOTS` (2) stance cards alongside `retrieval.top_k`
  excerpts as two separately-framed prompt sections.
- Window excerpts contain other speakers' lines. The prompt explicitly
  marks them as context, not voice or stance to absorb; only the
  persona's own lines carry their opinions.
- Prompts grow: a window is ~16 chat lines vs one forum comment, and the
  dossier adds ~600–1100 tokens to every call. `retrieval.top_k` may
  warrant lowering for window-heavy personas.
- Dossiers and cards regenerate cheaply from claims; claims are the
  expensive pass and accumulate append-only per month of corpus.
