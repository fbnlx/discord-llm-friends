# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A template for Discord bots that impersonate **personas** via retrieval-augmented few-shot prompting. Each persona is a self-contained folder under `personas/<id>/`; each maps to one Discord slash command and answers in that persona's voice and language.

[CONTEXT.md](CONTEXT.md) is the source of truth for domain vocabulary — **Persona, Corpus, Voice pool, Voice sample, Synthetic query, Window, Claim, Stance card, Dossier, Synthetic persona, Essence, World bible, Canon, Canon fact, Timeline sheet, Canonizer, Invention mode, Consolidation, Cast, Tic, Cleanup heuristic, Pipeline, Runtime**. It also lists _deprecated_ terms to avoid (e.g. "style anchor", "dataset", "friend", "thread", "lore"). Use this vocabulary in code, comments, and discussion; the maintainer cares about it. Architectural decisions live in [docs/adr/](docs/adr/).

## Commands

This is a `uv`-managed Python 3.12+ project. There is **no test suite and no linter configured** — verification is manual via the dev CLI and dry-run modes below.

```sh
uv sync                                    # install deps
cp .env.example .env                       # then fill in keys (see "API key dependencies")
```

**Pipeline** (offline, run once per persona to onboard it — order matters):

```sh
uv run python -m discord_llm_friends.pipeline.cleanup       --persona <id>   # raw.json -> cleaned.json
uv run python -m discord_llm_friends.pipeline.synth_queries --persona <id>   # cleaned.json -> synthetic_queries.json (resumable; --rebuild, --dry-run)
uv run python -m discord_llm_friends.pipeline.embed         --persona <id>   # -> ChromaDB (--no-synth to skip doc2query)
```

**Canon** (Synthetic personas only — ADR-0006):

```sh
uv run python -m discord_llm_friends.pipeline.canon_seed    --persona <id> [--dry-run]  # canon/world_bible.json -> facts.jsonl + <id>__canon (mechanical, no LLM)
uv run python -m discord_llm_friends.pipeline.canon_rebuild --persona <id>              # facts.jsonl -> <id>__canon (ledger is truth, collection is cache)
```

**STOP the bot before any pipeline step that writes ChromaDB** (`embed`, `canon_seed`, `canon_rebuild`, `canon_consolidate`): embedded Chroma is single-process, and rebuilding a collection under a live bot corrupts its WAL/HNSW segments ("Failed to apply logs to the hnsw segment writer" on later reads — the fix is another rebuild with the bot down). On the server: `systemctl stop` → pipeline step → `systemctl start`.

**Dev / manual testing** (no Discord needed):

```sh
uv run python -m discord_llm_friends.dev.cli --persona <id> "a question"    # + --asker "<display name>" (cast), --canonize (run Canonizer after; writes canon), --history-file exchanges.json (simulate channel history / follow-ups)
uv run python -m discord_llm_friends.dev.query_check --persona <id> --query "..." [--top-k N] [--canon]   # preview retrieval only, no LLM call; --canon targets <id>__canon

PERSONA_DRY_RUN=1     uv run python -m discord_llm_friends.dev.cli --persona <id> "q"   # assemble + log prompt to stderr, skip the LLM
PERSONA_DEBUG_PROMPT=1 uv run python -m discord_llm_friends.dev.cli --persona <id> "q"   # log prompt to stderr AND call the LLM
```

**Runtime:**

```sh
uv run python -m discord_llm_friends.bot --persona <id>   # one persona (debugging / single deploy)
uv run python -m discord_llm_friends.bot --all            # every persona except _example/, one process (production)
scripts/start.sh                                          # wrapper for --all
```

Set `DISCORD_GUILD_ID` in `.env` during development so slash commands sync to one test server instantly (global sync takes up to an hour).

`_example` (Marv, a fictional arcade-repair tech) ships with the repo as the format reference and for smoke-testing the whole pipeline end-to-end.

## Architecture

### Two phases: Pipeline vs Runtime

The **Pipeline** (`pipeline/`) is a one-shot, offline sequence that produces on-disk artifacts. The **Runtime** (`bot.py` + `engine.py`) is the long-lived process that consumes them. They share `personas.py` (loader) and `config.py` (config), nothing else. Keep this separation: the bot never generates synthetic queries or embeds at request time — **except the per-persona canon path (ADR-0006)**: a canon-enabled Synthetic persona queries its `<id>__canon` collection per request and, after the reply is sent, the Canonizer LLM-extracts + embeds new Canon facts (fire-and-forget `asyncio.create_task`, quota-exempt, never blocks the reply).

### A persona is a folder

`personas/<id>/` contains `persona.yaml` (structured config — Discord wiring, tics, in-voice fallback/quota messages, cleanup overrides), `description.md` (prose appended verbatim into the system prompt), `raw.json` (input corpus), `cleaned.json` (pipeline output), and `synthetic_queries.json` (doc2query sidecar). Chat-era personas (produced by the sibling `clean-job` project — see ADR-0005) additionally carry `voice_pool.json` (explicit voice-sample pool, the persona's own lines only) and `dossier.md` (generated identity profile, appended to the system prompt after the description). `personas/*` is **gitignored except `_example/`** — real personas never get committed. The persona `id` must equal the folder name and is reused as the ChromaDB collection name. `personas.discover()` excludes `_example` unless asked.

### How a response is assembled (`engine.respond`)

The persona's voice is rebuilt **per query** from three ingredients in the prompt:

1. **Tics + description + dossier** — static voice rules from `persona.yaml` / `description.md`, plus the generated `dossier.md` (identity, recurring opinions, inter-persona dynamics) when present.
2. **Voice sample** — a _fresh random_ `random.sample()` of the voice pool (`voice_pool.json` if present, else `cleaned.json`; size = `style.sample_size`, default 50), redrawn every call. Re-sampling per call is deliberate: it stops signature catchphrases from dominating every response. Do not "optimize" this into a cached fixed set.
3. **RAG hits** — the collection is overfetched (several synthetic queries point at the same unit), deduped by unit id (`card_id`/`window_id`; legacy comment rows dedup by document text), and injected as TWO sections: up to `STANCE_SLOTS` (2, engine constant) stance cards ("what this persona consistently thinks") and top-K (`retrieval.top_k`, default 4) excerpts ("what this persona has said" — forum comments and/or speaker-labeled group-chat windows, whose other-speaker lines the prompt frames as context, never the persona's own voice). Both tiers are gated by **per-type L2 distance floors** (`retrieval.max_card_distance` 0.80 / `max_window_distance` 0.70, as tuned in `config.yaml` — the code defaults are tighter): a hit beyond its floor is dropped, not injected — so an off-topic query injects little/nothing and falls back on the always-present dossier instead of irrelevant filler. When the first pass finds **nothing** within the floor, **conditional query expansion** fires: a cheap `call_llm` rewrites the question into a few register/topic variants (incl. a neutral, de-slang'd one), and `_retrieve` retries over all phrasings (best-distance merge) with a slightly looser floor (`retrieval.expanded_floor_bonus`). This rescues slangy/vulgar user phrasings that embed far from the stored card/window queries — without it, e.g. crude Hungarian slang misses cards that exist. Queries that already match never trigger it; it's skipped under `PERSONA_DRY_RUN` (keeps prompt inspection LLM-free). `PERSONA_DEBUG_PROMPT`/`PERSONA_DRY_RUN` print a `[retrieval]` line (phrasing count, nearest distances, kept/dropped) and an `[expansion]` line for calibration.

The historical dual use of `cleaned.json` (voice pool _and_ retrieval source) is resolved by the voice-pool split: forum-only personas still use `cleaned.json` for both; chat personas must not, because windows contain other speakers. Recent channel history is also injected (see below) so personas can react to each other.

A **length mode** (`curt`/`short`/`multi`/`rant`) is rolled from a weighted table on every call and injected as a target — pure prompt-described length options otherwise collapse onto one "safe" length. The picked mode is logged at INFO. The same collapse applies to signature Tics ("sometimes starts with X" becomes *always*): `rolled_tics:` in persona.yaml (`{p, text, off_text}` entries) injects a tic directive only when a per-call roll hits — `text` with probability `p`, else the counter-instruction `off_text` — so e.g. a signature opener lands ~10% of the time. Rolls are logged as `tic_rolls=fired/total` on the same INFO line. Keep the voice pool's own rate roughly in line with `p`: the few-shot sample outvotes instructions. For Synthetic personas a third roll, the **Invention mode** (`mundane`/`notable`/`dramatic`, engine constant `INVENTION_MODES`), sets the scale of NEW canon inventions per call — inert when the Timeline sheet / established facts already answer the question — and is logged as `invention_mode=` on the same line (`-` for non-canon personas).

### Canon memory & cast (Synthetic personas, ADR-0006)

A **Synthetic persona** (`canon.enabled` in persona.yaml; all knobs per-persona under `canon:`) may invent facts; inventions become **Canon** so answers stay consistent. Three pieces: `personas/<id>/canon/facts.jsonl` (append-only ledger, **the truth** — the only irreplaceable runtime data, back it up with `state/`), the `<id>__canon` ChromaDB collection (derived cache; rebuild via `pipeline.canon_rebuild`), and `canon/timeline.md` (**Timeline sheet**, always in the system prompt after the dossier — this layer, not retrieval, guarantees cross-fact consistency. Per query, up to `canon.max_facts` **Canon facts** are retrieved (own floor `canon.max_distance`; same embedding call as corpus retrieval) and injected as an "ESTABLISHED FACTS" section before the stances. After the reply is sent, the **Canonizer** (`canon.py`) extracts new asserted facts (strict-JSON `call_llm` pass) — attributing each to whoever the reply asserts it about (the extractor gets the asker's cast-resolved identity **and the same recent channel history the responder saw**, so second-person answers canonize as facts about the asker and elliptical follow-ups ("hány szobás a ház?") resolve to the right subject; a fact whose subject can't be resolved is dropped, not guessed from the Timeline sheet) — drops near-duplicates of existing canon (`canon.dedup_max_distance` — first written wins), and commits ledger-first. `engine.respond` returns a `RespondResult` (text + injected canon) so the Canonizer sees exactly what the model saw. For canon personas the "don't invent biographical facts" guidance bullet is swapped for a "invent consistently, commit to one answer" variant, targeted per call by the rolled Invention mode. Debug: `[canon]` stderr line mirrors `[retrieval]`; `dev.cli --canonize` tests the loop without Discord; under `PERSONA_DEBUG_PROMPT=1` the Canonizer additionally dumps its extraction prompt, the raw model output, and a per-fact verdict (`KEEP` / `DROP near-dup` with distances / `DROP exact-ledger-dup`) as `[canonizer]` stderr lines; the Canonizer is skipped under `PERSONA_DRY_RUN`.

A persona may also declare a **Cast** (`cast:` in persona.yaml): `personas:` maps other bot persona ids → who they are to this persona (e.g. 'futurepeter' knowing `peter` is his own younger self, and the 2026-era personas knowing `futurepeter` is Peter from 2050), `users:` maps Discord handles → the real human behind them (matched case-insensitively against the asker's display name; the question line gets annotated). Injected as a "WHO X MAY BE TALKING TO" system-prompt block.

### doc2query retrieval (`synth_queries.py` + `embed.py`)

Instead of embedding the cleaned comments directly, `synth_queries` asks Gemini for a one-line "what question/topic would prompt this comment as a reply?" per entry, written to `synthetic_queries.json` (a **parallel array** — same length and order as `cleaned.json`). `embed` then vectorizes _those synthetic queries_ but stores the _original comment_ as the retrieval document. This shifts the embedding space from answer-shaped to question-shaped — a meaningful recall win for non-English corpora where small embedders struggle with the declarative-vs-interrogative gap.

Embedding ids are hashed from `input \x00 document` (both), so: regenerating synth queries forces a correct re-embed, and two comments sharing a synthetic query don't collide. `embed` is idempotent within a mode; **switching embedding model or doc2query↔legacy mode drops and rebuilds the whole collection** (vector spaces don't mix). A missing/misaligned/partial sidecar silently falls back to legacy (embed cleaned text directly) with a warning.

### LLM dispatch and the three independent API dependencies

`engine.call_llm` routes to the provider in `CONFIG.llm.provider`, then **fails over** down `CONFIG.llm.fallback_order` on _any_ exception (overload, rate limit, empty/blocked response). Single-provider chains re-raise the original error; multi-provider chains raise a combined summary. Adding a provider = add to the `_PROVIDERS` dict in `engine.py`. Every provider client (and the embeddings client) is capped by `llm.request_timeout_seconds` (default 45) — without it an overloaded provider can hold the connection for minutes before erroring, and failover only starts after that wait. `LLM_PROVIDER` in `.env` is the quick lever to demote a provider having a bad day.

Three separate API keys serve three separate roles — don't conflate them:

- **Embeddings are always OpenAI** (`text-embedding-3-large`), regardless of LLM provider → `OPENAI_API_KEY` is required for _any_ embed/retrieval/runtime path.
- **`synth_queries` always uses the Gemini-configured model**, regardless of `CONFIG.llm.provider` → that pipeline step needs `GOOGLE_API_KEY`.
- The **runtime LLM** is whatever `provider`/`fallback_order` selects, needing that provider's key.

Gemini calls set `thinking_budget=0` — reasoning tokens otherwise leak into the visible response as English meta-commentary.

### Runtime topology (ADR-0003)

`--all` runs N `discord.Client` instances in **one process, one asyncio event loop** (`asyncio.gather`), because the deployment target is a 1 GB e2-micro and per-persona processes would triple the import/ChromaDB footprint. Consequences:

- **Failure isolation is in-process**: every command handler `defer()`s (Discord's 3s interaction limit) then wraps everything post-defer in a broad `try/except` that posts the persona's in-voice `fallback_messages`. A malformed persona or missing token is skipped + warned at boot, not fatal — but a true Python crash takes down all personas.
- Per-persona loggers are named `bot.<persona_id>` so one log stream stays filterable.
- The LLM call runs via `asyncio.to_thread` (the SDKs are sync) to avoid blocking the shared loop.

### State and config

`state/` (gitignored, created on the fly):

- `rate_limits/<persona-id>.json` — per-user **daily quota**, atomically persisted (tmp-then-rename). One file per persona, so no cross-process locking. Plus an in-process short-window `RateLimiter` (lost on restart).
- `history/<channel-id>.json` — per-channel exchange log, **shared by all personas in that channel**. This is the cross-persona continuity mechanism: the last `history_turns_in_prompt` exchanges are fed into each prompt. Atomic writes; concurrent appends in the same channel can race and drop an update (accepted at this scale).

Config layering (`config.py`), lowest to highest precedence: frozen dataclass defaults → `config.yaml` (selective overrides; unknown keys ignored for forward-compat) → env overrides for the documented subset only (`LLM_PROVIDER`, `DAILY_LIMIT`, `HISTORY_TURNS`). `CONFIG` is a singleton built at import; tests pass an explicit path to `load_config()`. Secrets live **only** in `.env`. Cleanup config is project-wide in `config.yaml` with optional per-persona overrides under `cleanup:` in `persona.yaml` (e.g. language-specific quote-block markers).
