# Runtime-writable canon memory for synthetic personas

A **Synthetic persona** (a persona with no real corpus for its era, e.g. a
future edition of a real person) is allowed to invent facts. Inventions
must stay consistent, so they become **Canon**: an append-only ledger
(`personas/<id>/canon/facts.jsonl`, the truth) mirrored into a separate
`<id>__canon` ChromaDB collection (a rebuildable cache), plus an
always-injected **Timeline sheet** (`canon/timeline.md`). New facts are
extracted post-response by the **Canonizer** — fire-and-forget, first
written wins.

## Context

The future persona speaks from 2050 with a manufactured identity and
a fictional 2026→2050 world. Two consistency requirements drove the
design: (1) a question with the same essence must always get the same
invented answer (the first answer is binding);
(2) related questions must respect established facts ("what did X do in 2032?" must honor that X was in Spain back then). Requirement (1)
is a retrieval problem; requirement (2) cannot be guaranteed by retrieval
alone — embedding neighborhoods miss cross-topic entailments.

## Considered Options

- **Retrieval-only canon.** Rejected: same-essence consistency only;
  cross-fact consistency becomes luck-of-retrieval.
- **Canon rows flagged by metadata in the main collection.** Rejected:
  `pipeline/embed.py` drops and recreates the main collection on
  model/mode staleness — runtime-written facts must not live in
  pipeline-disposable storage. Canon rows would also compete with
  cards/windows inside one overfetch budget and give one collection two
  writers with different lifecycles.
- **Ledger-as-truth + separate collection + Timeline sheet + async
  Canonizer** (chosen). The ledger is append-only at runtime (O_APPEND
  jsonl under a per-persona thread lock; a crash costs one truncated tail
  line, skipped on read). The collection is derived and rebuildable
  (`pipeline.canon_rebuild`). Load-bearing dated facts live in the
  Timeline sheet in every system prompt — that layer, not retrieval,
  guarantees the cross-fact case. The Canonizer runs after the reply is
  sent (`asyncio.create_task` → `asyncio.to_thread`), extracts 0–N new
  facts with their doc2query retrieval queries, drops candidates whose
  best distance to existing canon is under `canon.dedup_max_distance`
  (an accidental re-answer cannot overwrite the original — first written
  wins), and commits ledger-first, collection-second.

## Consequences

- **A sanctioned exception to the Pipeline/Runtime split**: a
  canon-enabled persona embeds queries against a second collection and,
  after replying, runs one extraction LLM call + one embedding call.
  Per-persona gated (`canon:` in persona.yaml, default off — all other
  personas are bit-identical), quota-exempt, never blocks or fails the
  reply.
- `engine.respond` returns a `RespondResult` (text + the canon facts
  actually injected) instead of a bare string — the Canonizer must judge
  novelty against exactly what the model saw; re-retrieval could not
  reproduce it. Callers: bot.py, dev/cli.py.
- First runtime-writable file under `personas/`: `canon/facts.jsonl` is
  the only irreplaceable runtime data in the project — include it in
  whatever backup covers `state/`.
- Runtime never mutates canon. Near-duplicate merging, superseding, and
  Timeline-sheet promotion happen only in the offline, maintainer-run
  consolidation pass (which may compact the ledger atomically).
- Retrieval-level consistency is best-effort; the Timeline sheet carries
  the guarantee. Facts that prove load-bearing must therefore migrate
  into the sheet — that is consolidation's promote step, and why the
  sheet is size-budgeted (warned, never truncated) rather than capped.
- The system-prompt IMPROVISE rule is conditionally swapped for canon
  personas: invention is allowed but must stay consistent with the
  Timeline sheet and injected facts, and the model is told its assertions
  become binding canon (commit to ONE concrete answer).
- Cost, canon persona only: ~1,100–1,400 extra prompt tokens per call
  (timeline + facts), plus one flash-tier extraction call and one
  embedding call per exchange (~+30–50% of an exchange), bounded by the
  existing daily quota. Collection footprint ≈ 12 MB per 1,000 rows —
  noise next to the existing store.
- Concurrency residual: two simultaneous different inventions on one
  topic, phrased far enough apart to clear dedup, can both land; accepted
  at this scale — consolidation merges later.
