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

**Style anchor**:
A hand-curated example **Corpus** entry embedded verbatim in the LLM system
prompt for few-shot voice mimicry. Typically 40-60 per **Persona**.
_Avoid_: example, sample

**Tic**:
A short prose description of a characteristic speech pattern of a **Persona**
(emoticon style, spelling quirks, register). Listed in `persona.yaml` and
injected into the system prompt alongside **Style anchors**.
_Avoid_: trait, quirk, habit

**Cleanup heuristic**:
A filter rule applied during the raw → cleaned transform to drop noise
(short reactions, system messages, quote blocks). Some heuristics are
project-wide defaults; **Corpus**-specific heuristics (e.g. known
forum-system messages in a particular language) live in `persona.yaml`.
_Avoid_: filter, rule

**Pipeline**:
The one-shot sequence run when onboarding a **Persona**: define persona →
add raw → cleanup → curate style anchors → embed. Distinct from **Runtime**.
_Avoid_: setup, workflow

**Runtime**:
The long-lived Discord bot process(es) that handle slash commands. Reads
persona config + embedded **Corpus** + **Style anchors** at boot.
_Avoid_: server, app

## Relationships

- A **Persona** has exactly one **Corpus** and a set of **Style anchors**
  drawn from that **Corpus**.
- A **Persona** declares a list of **Tics** which are injected into the
  system prompt alongside its **Style anchors**.
- The **Pipeline** produces the artifacts that the **Runtime** consumes:
  cleaned **Corpus** → embedded in ChromaDB → retrieved at query time
  by the **Runtime**.
- Each **Persona** maps to exactly one Discord slash command at **Runtime**.
