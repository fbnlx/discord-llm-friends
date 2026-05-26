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
The raw + cleaned source material for a **Persona** — by default, forum-style
comments authored by the person the **Persona** is based on. Lives in the
same per-persona folder.
_Avoid_: dataset, data, comments

**Voice sample**:
A randomly drawn subset of a **Persona**'s cleaned **Corpus**, injected
verbatim into the LLM system prompt on each query for few-shot voice
mimicry. Re-drawn on every response so signature phrases don't dominate
over time. Subset size is set by `style.sample_size` in `config.yaml`
(default 50). Not persisted; lives only as the prompt body for one call.
_Avoid_: anchor, style example, sample (alone)

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
add raw → cleanup → embed. Distinct from **Runtime**.
_Avoid_: setup, workflow

**Runtime**:
The long-lived Discord bot process(es) that handle slash commands. Reads
persona config + embedded **Corpus** at boot; draws a fresh **Voice
sample** and RAG hits from the **Corpus** per query.
_Avoid_: server, app

## Relationships

- A **Persona** has exactly one **Corpus**; a fresh **Voice sample** is
  drawn from that **Corpus** into the system prompt on every query.
- A **Persona** declares a list of **Tics** which are injected into the
  system prompt alongside its **Voice sample**.
- The **Pipeline** produces the artifacts that the **Runtime** consumes:
  cleaned **Corpus** → embedded in ChromaDB for RAG retrieval AND
  randomly sampled per query as the **Voice sample**.
- Each **Persona** maps to exactly one Discord slash command at **Runtime**.
