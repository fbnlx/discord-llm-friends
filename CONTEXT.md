# discord-llm-friends

A template-shaped repo for running one or more Discord bots that impersonate
a personality through retrieval-augmented few-shot prompting. A user clones
the repo, defines one or more **Personas**, drops in raw source material,
and runs the pipeline.

## Language

**Persona**:
A character the bot impersonates. Self-contained on disk under
`personas/<id>/`. Carries a name, language, characteristic tics, and a
slash-command shape.
_Avoid_: friend, bot, character

**Corpus**:
The raw + cleaned source material for a **Persona** — forum-style comments
authored by the person the **Persona** is based on, and/or multi-party
group-chat logs the person took part in (prepared by the sibling
`clean-job` pipeline). Lives in the same per-persona folder.
_Avoid_: dataset, data, comments

**Voice pool**:
The set of texts the **Voice sample** is drawn from — exclusively the
persona's OWN words. Explicit artifact `voice_pool.json` when present
(required once **Windows** exist, since those contain other speakers);
otherwise the cleaned **Corpus** doubles as the pool, the historical
behavior.
_Avoid_: corpus (when meaning this), sample pool

**Voice sample**:
A randomly drawn subset of a **Persona**'s **Voice pool**, injected
verbatim into the LLM system prompt on each query for few-shot voice
mimicry. Re-drawn on every response so signature phrases don't dominate
over time. Subset size is set by `style.sample_size` in `config.yaml`
(default 50). Not persisted; lives only as the prompt body for one call.
_Avoid_: anchor, style example, sample (alone)

**Window**:
A retrieval unit cut deterministically from a group-chat session: a
speaker-labeled, overlapping span of ~16 messages (50% stride, hard cuts
at long silences). Carries the local context that makes one-word chat
lines meaningful. Embedded via several **Synthetic queries** (one per
distinct subject in the window); the rendered window text is the stored
document. Never segmented by a model — completeness holds by
construction.
_Avoid_: thread, segment, chunk

**Claim**:
One standalone sentence a persona ASSERTS (opinion, preference, habit,
plan, personal fact), extracted per **Window** with message-id citations
— at least one cited message must be authored by the claimant. The raw
material **Stance cards** and the **Dossier** are synthesized from;
never injected at runtime itself.
_Avoid_: fact, statement, quote

**Stance card**:
A synthesized "what this persona thinks about X" paragraph built from
**Claims** (per topic, plus per-category rollup cards that survey a
whole category, e.g. games). Embedded via representative questions —
rollups deliberately carry abstract phrasings ("mit játsszak?") so
aggregate questions that match no single **Window** retrieve a card
instead. Stored in the same collection as **Windows**, metadata
`type="stance"`.
_Avoid_: summary, profile, opinion entry

**Dossier**:
A generated per-persona identity document (personality, recurring
opinions, relationships with the other personas), reduced from **Stance
cards** and name-mentioning **Claims**. Lives as `dossier.md` in the
persona folder and is appended to EVERY system prompt after
`description.md` — the always-on identity core that covers off-corpus
questions. Distinct from `description.md`, which stays hand-written.
_Avoid_: bio, profile (alone), description

**Synthetic query**:
A one-line LLM-generated string that hypothesizes what user question,
topic, or trigger would plausibly prompt a given **Corpus** entry as a
reply. One per **Corpus** entry, position-aligned in
`synthetic_queries.json`. The embed pipeline vectorizes the **Synthetic
queries** (not the original **Corpus** entries) into ChromaDB, shifting
the retrieval vector space from answer-shaped to question-shaped — better
recall when the user asks a Discord question that doesn't lexically
overlap with the source forum-comment register. This is the doc2query
indexing technique.
_Avoid_: trigger, prompt, query (alone)

**Tic**:
A short prose description of a characteristic speech pattern of a **Persona**
(emoticon style, spelling quirks, register). Listed in `persona.yaml` and
injected into the system prompt alongside the **Voice sample**.
_Avoid_: trait, quirk, habit

**Cleanup heuristic**:
A filter rule applied during the raw → cleaned transform to drop noise
(short reactions, system messages, quote blocks). Some heuristics are
project-wide defaults; **Corpus**-specific heuristics (e.g. known
forum-system messages in a particular language) live in `persona.yaml`.
_Avoid_: filter, rule

**Pipeline**:
The one-shot sequence run when onboarding a **Persona**: define persona →
add raw → cleanup → generate **Synthetic queries** → embed. Distinct
from **Runtime**.
_Avoid_: setup, workflow

**Runtime**:
The long-lived Discord bot process(es) that handle slash commands. Reads
persona config + embedded **Corpus** at boot; draws a fresh **Voice
sample** and RAG hits from the **Corpus** per query.
_Avoid_: server, app

## Relationships

- A **Persona** has exactly one **Corpus**; a fresh **Voice sample** is
  drawn from its **Voice pool** into the system prompt on every query.
- A **Persona** declares a list of **Tics** which are injected into the
  system prompt alongside its **Voice sample**.
- The **Pipeline** produces the artifacts that the **Runtime** consumes:
  cleaned **Corpus** → one **Synthetic query** per entry → embedded
  (synth queries vectorized, original **Corpus** stored as the
  retrieval document) in ChromaDB for RAG retrieval.
- For group-chat corpora, the sibling `clean-job` pipeline produces the
  layered artifacts instead: **Windows** → **Claims** → **Stance cards**
  → **Dossier**, exporting **Windows** + **Stance cards** into the
  persona's ChromaDB collection, the persona's own lines into the
  **Voice pool**, and the **Dossier** into the persona folder.
- At **Runtime**, retrieval dedups to unique units and injects up to 2
  **Stance cards** and `retrieval.top_k` excerpts (**Windows** / forum
  comments) as two separately-framed prompt sections; the **Dossier**
  rides in the system prompt on every call.
- Each **Persona** maps to exactly one Discord slash command at **Runtime**.
