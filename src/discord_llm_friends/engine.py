"""Persona engine — turns a user question into an in-character response.

Public entry point: `respond(persona_id, question, ...)`.

Pipeline:
  1. Load the persona's description + dossier + tics, and draw a fresh
     random sample of style examples from the voice pool (voice_pool.json
     if present, else the cleaned corpus).
  2. Embed the user's question with the embedding model in CONFIG.
  3. Query the persona's ChromaDB collection, overfetching because several
     synthetic queries can point at the same unit; dedup to unique units
     and split them into stance cards (distilled opinions) vs excerpts
     (comments / group-chat windows).
  4. Assemble a system prompt (who they are + dossier + style examples)
     and a user prompt (stances + excerpts + recent channel history + the
     question).
  5. Dispatch to the LLM provider selected by CONFIG.llm.provider,
     failing over to the next provider in CONFIG.llm.fallback_order on error.
  6. Return a RespondResult (reply text + the canon facts injected, so the
     Canonizer can see exactly what the model saw).

Synthetic personas (persona.canon.enabled) additionally get: canon-fact
retrieval from the `<id>__canon` collection, the always-injected Timeline
sheet, a relaxed improvisation rule (inventions allowed, must stay
consistent with canon), and a per-call rolled Invention mode setting the
scale of new inventions. See canon.py / ADR-0006.

Env-only knobs (no YAML equivalent):
  PERSONA_DEBUG_PROMPT=1 — print assembled prompts to stderr before the call.
  PERSONA_DRY_RUN=1      — skip the LLM call, return a placeholder.
"""

from __future__ import annotations

import logging
import os
import random
import sys
from dataclasses import dataclass
from functools import lru_cache

import chromadb
from chromadb.config import Settings
from openai import OpenAI

from discord_llm_friends import canon
from discord_llm_friends import config as cfg
from discord_llm_friends import personas as personas_module
from discord_llm_friends.history import Exchange, format_history


logger = logging.getLogger(__name__)


# --- Env-only debug knobs ---------------------------------------------------

DEBUG_PROMPT = os.getenv("PERSONA_DEBUG_PROMPT", "").strip() == "1"
DRY_RUN = os.getenv("PERSONA_DRY_RUN", "").strip() == "1"


# --- Response-length randomization -----------------------------------------
# Each call rolls a length mode from this weighted table; the chosen mode
# is injected into the system prompt as a target. This is the only
# reliable way to get length variety — pure prompt-described options
# collapse onto whichever length the model considers "safe".
#
# Weights are integers, summed and normalized by random.choices.
# Tune by reading INFO logs (each call logs the picked mode) and adjusting.

LENGTH_MODES: list[tuple[str, int, str]] = [
    # (name, weight, instruction-injected-into-prompt)
    ("curt",  30, "1-5 words. A grunt, dismissal, single quip, or short jab. Almost no elaboration."),
    ("short", 45, "ONE sentence. Direct answer in voice."),
    ("multi", 20, "2-4 sentences. A short opinion or explanation, optionally with one aside."),
    ("rant",   5, "5+ sentences. Worked-up, opinionated, possibly tangential."),
]


def _pick_length() -> tuple[str, str]:
    """Roll a length mode for this response. Returns (name, instruction)."""
    name, _weight, instruction = random.choices(
        LENGTH_MODES,
        weights=[w for _, w, _ in LENGTH_MODES],
        k=1,
    )[0]
    return name, instruction


# --- Invention-scale randomization (Synthetic personas only) ----------------
# A canon-enabled persona may invent new facts (ADR-0006). Left to itself
# the model collapses onto the safest possible invention every call — a
# generic, modal future ("normal boring life") — the same collapse family
# as length and rolled tics. So the SCALE of new inventions is rolled per
# call from this weighted table and injected as a target. The mode governs
# ONLY newly invented facts: when the Timeline sheet / established facts /
# retrieval already answer the question, the injected bullet tells the
# model to ignore it. Tune by reading INFO logs (invention_mode=) and the
# canon ledger.

INVENTION_MODES: list[tuple[str, int, str]] = [
    # (name, weight, instruction-injected-into-prompt)
    ("mundane",  30, "Everyday scale. New details stay small: routines, "
                     "chores, minor purchases, small gossip, work minutiae "
                     "— concrete but ordinary. No twists this reply."),
    ("notable",  45, "Memorable scale. Make the new detail worth retelling: "
                     "a named person, a dated event, a real change — a "
                     "move, a new venture, a breakup, a health scare, an "
                     "incident, interesting development. Specific, not filler."),
    ("dramatic", 25, "Bold scale. Assert ONE significant development as "
                     "hard fact — a birth, a death, a disaster brushing "
                     "somewhere familiar, a career rupture, a scandal, a "
                     "windfall, or something great and hopeful — named, dated, one vivid detail. In-genre "
                     "for this world; consequence, not apocalypse."),
]


def _pick_invention() -> tuple[str, str]:
    """Roll an invention mode for this response. Returns (name, instruction)."""
    name, _weight, instruction = random.choices(
        INVENTION_MODES,
        weights=[w for _, w, _ in INVENTION_MODES],
        k=1,
    )[0]
    return name, instruction


# --- Lazy shared clients (per-process singletons) --------------------------

@lru_cache(maxsize=1)
def _openai_client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY not set — embeddings require it. "
            "Make sure .env exists and is populated."
        )
    return OpenAI(timeout=cfg.CONFIG.llm.request_timeout_seconds)


@lru_cache(maxsize=1)
def _chroma_client():
    return chromadb.PersistentClient(
        path=str(cfg.CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


@lru_cache(maxsize=16)
def _persona(persona_id: str) -> personas_module.Persona:
    return personas_module.load(persona_id)


@lru_cache(maxsize=16)
def _voice_pool(persona_id: str) -> tuple[str, ...]:
    """Load the persona's voice pool once and cache it.

    Used as the pool for per-call style-example sampling — each
    `respond()` draws a fresh random subset from this tuple, so the
    system prompt shows different examples on every query. Returns a
    tuple so it's hashable for lru_cache and immutable to callers.
    """
    return tuple(personas_module.load_voice_pool(_persona(persona_id)))


# --- Retrieval -------------------------------------------------------------

# How many stance cards (distilled cross-session opinions, metadata
# type="stance") to inject on top of the excerpt budget. Cards are dense —
# one card answers "what does X think about this" wholesale — so a couple
# is plenty; excerpts carry the authentic texture.
STANCE_SLOTS = 2


def _embed_queries(queries: list[str]) -> list[list[float]]:
    """One embeddings API call for all phrasings — shared by corpus and
    canon retrieval, so a canon-enabled persona pays no extra embedding."""
    return [
        d.embedding
        for d in _openai_client().embeddings.create(
            model=cfg.CONFIG.embedding.model, input=queries,
        ).data
    ]


def _retrieve(
    persona_id: str, queries: list[str], n: int, floor_bonus: float = 0.0,
    query_vecs: list[list[float]] | None = None,
) -> tuple[list[str], list[str]]:
    """Embed the query phrasing(s) and pull the nearest stored units.

    `queries` is one or more phrasings of the user's question — more than one
    only when runtime query expansion kicks in. Each unit is scored by its
    BEST (nearest) distance across all phrasings, so an added neutral
    rephrasing can rescue a card a slangy phrasing missed. `floor_bonus` is
    added to both distance floors (the expansion retry passes a small bonus to
    accept a looser match rather than nothing).

    Returns (stances, excerpts): up to STANCE_SLOTS stance cards and up to
    `n` excerpts (forum comments / group-chat windows), nearest first — each
    within its per-type L2 distance floor (CONFIG.retrieval), so a query with
    no genuinely close hit injects little or nothing and the always-present
    dossier carries the answer instead of irrelevant filler.

    The collection holds several rows per unit (one per synthetic query
    pointing at it), so hits are deduped by unit id — card_id / window_id
    metadata when present, the document text itself for legacy comment rows.

    `query_vecs`, when given, must align with `queries` (caller already
    embedded them); when None they are embedded here.
    """
    if query_vecs is None:
        query_vecs = _embed_queries(queries)

    rc = cfg.CONFIG.retrieval
    collection = _chroma_client().get_collection(name=persona_id)
    overfetch = min((n + STANCE_SLOTS) * 4, collection.count())
    result = collection.query(
        query_embeddings=query_vecs,
        n_results=overfetch,
        include=["documents", "metadatas", "distances"],
    )

    # Merge across phrasings: keep each unit's nearest distance, deduped by id.
    best: dict[str, tuple[float, str, dict]] = {}
    for qi in range(len(query_vecs)):
        for doc, meta, dist in zip(
            result["documents"][qi], result["metadatas"][qi], result["distances"][qi]
        ):
            meta = meta or {}
            unit = meta.get("card_id") or meta.get("window_id") or doc
            if unit not in best or dist < best[unit][0]:
                best[unit] = (dist, doc, meta)

    card_floor = rc.max_card_distance + floor_bonus
    window_floor = rc.max_window_distance + floor_bonus
    stances: list[str] = []
    excerpts: list[str] = []
    nearest_card = nearest_window = None
    dropped_card = dropped_window = 0
    # Nearest first. A hit beyond its type's distance floor is dropped, not
    # injected — better to send nothing (and lean on the dossier) than to pad
    # the prompt with irrelevant excerpts. Cards get a looser floor than
    # windows (cleaner, fewer, denser).
    for dist, doc, meta in sorted(best.values(), key=lambda x: x[0]):
        is_stance = meta.get("type") == "stance"
        if is_stance and nearest_card is None:
            nearest_card = dist
        if not is_stance and nearest_window is None:
            nearest_window = dist
        if dist > (card_floor if is_stance else window_floor):
            if is_stance:
                dropped_card += 1
            else:
                dropped_window += 1
            continue
        if is_stance:
            if len(stances) < STANCE_SLOTS:
                stances.append(doc)
        elif len(excerpts) < n:
            excerpts.append(doc)

    logger.info(
        "retrieval persona=%s phrasings=%d kept_stances=%d kept_excerpts=%d "
        "nearest_card=%s nearest_window=%s dropped_beyond_floor=%d/%d(card/window)",
        persona_id, len(queries), len(stances), len(excerpts),
        f"{nearest_card:.3f}" if nearest_card is not None else "na",
        f"{nearest_window:.3f}" if nearest_window is not None else "na",
        dropped_card, dropped_window,
    )
    if DEBUG_PROMPT or DRY_RUN:
        nc = f"{nearest_card:.3f}" if nearest_card is not None else "—"
        nw = f"{nearest_window:.3f}" if nearest_window is not None else "—"
        print(
            f"[retrieval] {len(queries)} phrasing(s); nearest card={nc} "
            f"(floor {card_floor:.2f}), window={nw} "
            f"(floor {window_floor:.2f}) → kept {len(stances)} stance / "
            f"{len(excerpts)} excerpt; dropped {dropped_card} card / "
            f"{dropped_window} window beyond floor",
            file=sys.stderr,
        )
    return stances, excerpts


# Warn once per persona when canon is enabled but the collection is absent
# (enabled-but-not-yet-seeded is a legitimate transient state, not an error).
_CANON_MISSING_WARNED: set[str] = set()


def _retrieve_canon(
    persona: personas_module.Persona,
    queries: list[str],
    query_vecs: list[list[float]],
    floor_bonus: float = 0.0,
) -> list[str]:
    """Nearest active Canon facts for this question, deduped by fact_id.

    Returns up to persona.canon.max_facts fact documents within
    persona.canon.max_distance (+ floor_bonus), nearest first. The floor is
    deliberately permissive: a marginally related fact CONSTRAINS invention
    (the prompt says "never contradict", not "must use"), so recall beats
    precision here.
    """
    name = canon.collection_name(persona.id)
    try:
        collection = _chroma_client().get_collection(name=name)
    except Exception:
        if persona.id not in _CANON_MISSING_WARNED:
            _CANON_MISSING_WARNED.add(persona.id)
            logger.warning(
                "canon enabled for %s but collection %r is missing — run "
                "pipeline.canon_seed or pipeline.canon_rebuild",
                persona.id, name,
            )
        return []

    count = collection.count()
    if count == 0:
        return []
    cc = persona.canon
    result = collection.query(
        query_embeddings=query_vecs,
        n_results=min(cc.max_facts * 4, count),
        include=["documents", "metadatas", "distances"],
    )
    best: dict[str, tuple[float, str]] = {}
    for qi in range(len(query_vecs)):
        for doc, meta, dist in zip(
            result["documents"][qi], result["metadatas"][qi], result["distances"][qi]
        ):
            fact_id = (meta or {}).get("fact_id") or doc
            if fact_id not in best or dist < best[fact_id][0]:
                best[fact_id] = (dist, doc)

    floor = cc.max_distance + floor_bonus
    ranked = sorted(best.values(), key=lambda x: x[0])
    facts = [doc for dist, doc in ranked if dist <= floor][: cc.max_facts]
    nearest = ranked[0][0] if ranked else None
    dropped = len(ranked) - len(facts)

    logger.info(
        "canon persona=%s phrasings=%d kept=%d nearest=%s dropped=%d",
        persona.id, len(queries), len(facts),
        f"{nearest:.3f}" if nearest is not None else "na", dropped,
    )
    if DEBUG_PROMPT or DRY_RUN:
        nf = f"{nearest:.3f}" if nearest is not None else "—"
        print(
            f"[canon] {len(queries)} phrasing(s); nearest fact={nf} "
            f"(floor {floor:.2f}) → kept {len(facts)} / dropped {dropped}",
            file=sys.stderr,
        )
    return facts


# Runtime query expansion (conditional). Only fires when the first retrieval
# pass found nothing within the distance floor — typically a slangy/vulgar
# phrasing that embeds far from the (cleaner) stored card/window queries. A
# short rewrite into a few register/topic variants — crucially a NEUTRAL one —
# bridges the gap; normal queries that already match never pay for the call.
_EXPANSION_VARIANTS = 3


def _expand_query(question: str, persona: personas_module.Persona) -> list[str]:
    """Rewrite the user's question into a few alternative search phrasings.

    Returns [] on any failure — the caller then just keeps the empty
    first-pass result and falls back to the dossier.
    """
    system = (
        f"You rewrite a chat message into short {persona.language} search "
        f"phrasings used to look up {persona.display_name}'s opinions in a "
        f"database. Output {_EXPANSION_VARIANTS} alternative phrasings of the "
        f"SAME question, one per line, no numbering, each in "
        f"{persona.language}. Include at least one NEUTRAL, plain rephrasing "
        f"that strips slang and vulgarity and names the underlying topic "
        f"directly (e.g. a crude sexual phrasing becomes 'which women does he "
        f"find attractive' / 'sexual preferences', expressed in "
        f"{persona.language}). Keep the others near the original register. "
        f"Output ONLY the phrasings."
    )
    try:
        text = call_llm(system, question)
    except Exception:  # noqa: BLE001 — expansion is best-effort; never fatal
        return []
    variants = [
        line.strip(" -•\t").strip()
        for line in text.splitlines()
        if line.strip(" -•\t").strip()
    ]
    return variants[:_EXPANSION_VARIANTS]


# --- Prompt assembly -------------------------------------------------------

def _roll_tics(persona: personas_module.Persona) -> tuple[list[str], int]:
    """Per-call dice roll over the persona's RolledTics. Returns the
    directive lines to inject and how many rolled ON. Same reasoning as
    length modes: a prompt-described frequency ("use X sometimes") collapses
    into every-reply usage; a mechanical roll holds the target rate."""
    directives: list[str] = []
    fired = 0
    for tic in persona.rolled_tics:
        if random.random() < tic.p:
            directives.append(tic.text)
            fired += 1
        elif tic.off_text:
            directives.append(tic.off_text)
    return directives, fired


def _build_system_message(
    persona: personas_module.Persona,
    style_examples: tuple[str, ...],
    length_mode: str,
    length_instruction: str,
    tic_directives: list[str] | None = None,
    invention_mode: str | None = None,
    invention_instruction: str | None = None,
) -> str:
    tic_lines = (
        "\n".join(f"- {tic}" for tic in persona.tics)
        if persona.tics
        else "(none specified — infer from examples)"
    )
    if tic_directives:
        tic_lines += "\n" + "\n".join(
            f"- THIS REPLY ONLY: {d}" for d in tic_directives
        )
    example_lines = "\n".join(f"- {a}" for a in style_examples)
    name = persona.display_name

    dossier_block = (
        f"{name.upper()}'S PROFILE — distilled from {name}'s real chat "
        f"history: personality, recurring opinions, and how {name} treats "
        f"the other friends. Treat this as reliable background; when the "
        f"question touches one of these subjects or people, this is "
        f"{name}'s established take.\n"
        f"{persona.dossier}\n"
        f"\n"
        if persona.dossier
        else ""
    )

    timeline_block = (
        f"{name.upper()}'S WORLD TIMELINE — the established history of the "
        f"world {name} lives in. Everything here is canon: {name} lived "
        f"through these events and knows them as fact. NEVER contradict "
        f"this timeline. When asked about events or periods it does not "
        f"cover, invent SPECIFIC details that FIT it (who was in power, "
        f"what had already happened by then) — filling gaps with concrete, "
        f"consistent detail is the default. Staying vague is a LAST "
        f"resort, only for when no consistent detail is possible — never "
        f"an excuse to dodge a direct question. Contradicting the "
        f"timeline is never an option.\n"
        f"{persona.timeline}\n"
        f"\n"
        if persona.timeline
        else ""
    )

    cast_lines: list[str] = []
    if persona.cast_personas:
        cast_lines.append(
            "Bot personas that may appear in the conversation — treat "
            "their words accordingly:"
        )
        cast_lines += [
            f"- {pid}: {who}" for pid, who in persona.cast_personas.items()
        ]
    if persona.cast_users:
        cast_lines.append(
            f"Known humans by Discord handle — when one of these asks, "
            f"{name} knows exactly who is talking:"
        )
        cast_lines += [
            f"- {handle}: {who}" for handle, who in persona.cast_users.items()
        ]
    cast_block = (
        f"WHO {name.upper()} MAY BE TALKING TO:\n" + "\n".join(cast_lines) + "\n\n"
        if cast_lines
        else ""
    )

    invention_bullet = ""
    if persona.canon.enabled:
        improvise_bullet = (
            f"- IMPROVISE confidently. Extrapolate from {name}'s personality "
            f"to say new things this person could plausibly say — don't just "
            f"rearrange phrasings from the examples, invent in voice. You "
            f"MAY invent new facts about {name}'s life and world when the "
            f"timeline and established facts don't cover the question — but "
            f"inventions must be CONSISTENT with them. Invent SPECIFICS, "
            f"not generalities: names, years, places, concrete incidents — "
            f"a generic \"nothing much changed\" is a non-answer, not a "
            f"safe answer. What you assert "
            f"becomes canon you will be held to later, so commit: give ONE "
            f"concrete answer, not alternatives or hedges.\n"
        )
        if invention_mode and invention_instruction:
            invention_bullet = (
                f"- INVENTION SCALE FOR THIS RESPONSE: {invention_mode} — "
                f"{invention_instruction} Applies ONLY to NEW facts you "
                f"invent in this reply: when the timeline, established "
                f"facts, or retrieved material already answer the question, "
                f"answer from them and ignore this line — never escalate "
                f"established facts for effect. Consistency with the "
                f"timeline and established facts ALWAYS outranks this "
                f"scale. Override the scale only if the question itself "
                f"demands a different one.\n"
            )
    else:
        improvise_bullet = (
            f"- IMPROVISE confidently. Extrapolate from {name}'s personality "
            f"to say new things this person could plausibly say — don't just "
            f"rearrange phrasings from the examples, invent in voice. Just "
            f"don't invent biographical facts (no new family members, schools, "
            f"jobs, etc.).\n"
        )

    return (
        f"You are responding AS {name}. Stay completely in character. Respond "
        f"ONLY in {persona.language}, in {name}'s voice. Do not announce "
        f"yourself as an AI, do not add safety disclaimers, do not break the "
        f"fourth wall.\n"
        f"\n"
        f"WHO {name.upper()} IS:\n"
        f"{persona.description}\n"
        f"\n"
        f"{dossier_block}"
        f"{timeline_block}"
        f"{cast_block}"
        f"CHARACTERISTIC SPEECH TICS TO PRESERVE:\n"
        f"{tic_lines}\n"
        f"\n"
        f"EXAMPLES OF THINGS {name.upper()} HAS ACTUALLY WRITTEN — a fresh "
        f"random sample drawn from {name}'s corpus for this response, so the "
        f"set is different every call. Match the voice (register, length, "
        f"spelling tics, emoticon style). You may freely reuse short "
        f"signature phrases, expletives, dismissive interjections, and "
        f"characteristic catchphrases that appear in these examples — "
        f"they're part of the voice. Just don't reproduce whole example "
        f"messages verbatim; invent new sentences in the same idiom.\n"
        f"{example_lines}\n"
        f"\n"
        f"GUIDANCE:\n"
        f"- TARGET LENGTH FOR THIS RESPONSE: {length_mode} — "
        f"{length_instruction} Override only if the question genuinely "
        f"demands a different length (e.g., a list was explicitly requested "
        f"while \"curt\" was picked, or pure small-talk while \"rant\" was "
        f"picked).\n"
        f"- STAY ON SUBJECT. Don't pivot to unrelated topics — even within "
        f"a rant, any wandering should stay within the asked subject.\n"
        f"{improvise_bullet}"
        f"{invention_bullet}"
        f"- DO NOT bring up {name}'s hobbies, games, or specific interests "
        f"unless the user's question already mentions them. The persona "
        f"has many interests; resist the urge to advertise them in every "
        f"response.\n"
        f"- The user message may include distilled stance summaries and "
        f"real past material by {name} retrieved as relevant to this "
        f"question — they carry {name}'s actual opinions and knowledge on "
        f"the topic. Where one genuinely fits, lean on it for the SUBSTANCE "
        f"of your answer (the stance, the take, the facts), not merely the "
        f"wording. Judge each one and ignore the off-topic ones rather than "
        f"forcing them in. (The random style examples above however are "
        f"mostly voice reference, but feel free to use facts from them if "
        f"relevant)\n"
        f"- Match the crudeness and energy of the examples.\n"
        f"- For topics from after {name}'s era, riff in voice rather than "
        f"refusing or breaking character.\n"
        f"- If a recent conversation is shown, you can react to it — but "
        f"you're not obligated to reference earlier topics.\n"
        f"- Output ONLY {name}'s reply, in {persona.language}. No analysis, "
        f"no English meta-commentary, no markdown formatting, no thinking "
        f"aloud."
    )


def _cast_identity(
    cast_users: dict[str, str] | None, asker_name: str,
) -> str | None:
    """Case-insensitive lookup of a Discord handle in the persona's cast."""
    if not cast_users:
        return None
    needle = asker_name.strip().casefold()
    for handle, identity in cast_users.items():
        if handle.strip().casefold() == needle:
            return identity
    return None


def _build_user_message(
    stances: list[str],
    excerpts: list[str],
    question: str,
    name: str,
    history: list[Exchange] | None,
    asker_name: str | None,
    canon_facts: list[str] | None = None,
    cast_users: dict[str, str] | None = None,
) -> str:
    sections: list[str] = []

    if history:
        sections.append(
            "RECENT CONVERSATION IN THIS CHANNEL (oldest first):\n"
            + format_history(history)
        )

    # Canon precedes stances: world facts constrain the answer before
    # opinions color it.
    if canon_facts:
        fact_block = "\n".join(f"- {f}" for f in canon_facts)
        sections.append(
            f"ESTABLISHED FACTS OF {name.upper()}'S WORLD — canon facts "
            f"retrieved as relevant to this question. These are TRUE in "
            f"{name}'s world and non-negotiable: never contradict them, and "
            f"build on the ones that apply. If the question goes beyond "
            f"them, invent an answer CONSISTENT with them and with the "
            f"world timeline:\n"
            f"{fact_block}"
        )

    if stances:
        stance_block = "\n\n".join(stances)
        sections.append(
            f"WHAT {name.upper()} CONSISTENTLY THINKS — distilled stance "
            f"summaries synthesized from {name}'s whole chat history, "
            f"retrieved as relevant to this question. These are {name}'s "
            f"established opinions: rely on them for the SUBSTANCE of the "
            f"answer (the verdicts, the preferences, the reasons) whenever "
            f"the question touches them:\n"
            f"{stance_block}"
        )

    if excerpts:
        excerpt_block = "\n".join(f"- {r}" for r in excerpts)
        sections.append(
            f"WHAT {name.upper()} HAS SAID ABOUT THIS TOPIC — real past "
            f"material retrieved as relevant: standalone comments by {name}, "
            f"and/or excerpts of group conversations {name} took part in "
            f"(lines formatted '[n] author: text'). In conversation excerpts "
            f"ONLY the lines written by {name} are {name}'s own words and "
            f"opinions — the other speakers are context, never voice or "
            f"stance to absorb. Where relevant, use this material for the "
            f"SUBSTANCE of your answer — the stance, the take, the facts — "
            f"not just the voice. Skip any that turn out off-topic, and "
            f"rephrase in the moment rather than quoting verbatim:\n"
            f"{excerpt_block}"
        )

    asker_prefix = ""
    if asker_name:
        identity = _cast_identity(cast_users, asker_name)
        asker_prefix = (
            f" (from @{asker_name} — {identity})" if identity
            else f" (from @{asker_name})"
        )
    sections.append(f"User's new question{asker_prefix}: {question}")
    sections.append(f"Respond as {name}:")

    return "\n\n".join(sections)


def _maybe_log_prompt(system: str, user: str) -> None:
    if not (DEBUG_PROMPT or DRY_RUN):
        return
    bar = "─" * 60
    print(f"\n{bar}\nSYSTEM:\n{system}\n{bar}\nUSER:\n{user}\n{bar}\n",
          file=sys.stderr)


# --- LLM dispatch ----------------------------------------------------------

def _call_gemini(system: str, user: str) -> str:
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        raise RuntimeError("GOOGLE_API_KEY (or GEMINI_API_KEY) not set")
    from google import genai
    from google.genai import types

    client = genai.Client(http_options=types.HttpOptions(
        # google-genai expects milliseconds here. Without a cap, an
        # overloaded backend can sit on the connection for minutes before
        # returning its 503 — failover only starts after that wait.
        timeout=int(cfg.CONFIG.llm.request_timeout_seconds * 1000),
    ))
    resp = client.models.generate_content(
        model=cfg.CONFIG.llm.models["gemini"],
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=cfg.CONFIG.llm.temperature,
            max_output_tokens=cfg.CONFIG.llm.max_output_tokens,
            # Disable Gemini 2.5+ thinking mode — otherwise reasoning tokens
            # can leak into the visible response as English meta-commentary.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )

    if not resp.candidates:
        raise RuntimeError(
            f"Gemini returned no candidates — possible safety block. "
            f"prompt_feedback={getattr(resp, 'prompt_feedback', None)!r}"
        )
    parts = resp.candidates[0].content.parts or []
    text_parts = [
        p.text for p in parts
        if not getattr(p, "thought", False) and p.text
    ]
    text = "".join(text_parts).strip()
    if not text:
        raise RuntimeError(
            f"Gemini returned no non-thought text. finish_reason="
            f"{resp.candidates[0].finish_reason!r}, parts={parts!r}"
        )
    return text


def _call_claude(system: str, user: str) -> str:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    from anthropic import Anthropic

    client = Anthropic(timeout=cfg.CONFIG.llm.request_timeout_seconds)
    resp = client.messages.create(
        model=cfg.CONFIG.llm.models["claude"],
        max_tokens=cfg.CONFIG.llm.max_output_tokens,
        temperature=cfg.CONFIG.llm.temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    if not resp.content:
        raise RuntimeError(f"Claude returned empty content. stop_reason={resp.stop_reason!r}")
    return resp.content[0].text.strip()


def _call_openai(system: str, user: str) -> str:
    client = _openai_client()
    resp = client.chat.completions.create(
        model=cfg.CONFIG.llm.models["openai"],
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=cfg.CONFIG.llm.temperature,
        max_completion_tokens=cfg.CONFIG.llm.max_output_tokens,
    )
    text = resp.choices[0].message.content
    if not text:
        raise RuntimeError(
            f"OpenAI returned empty content. finish_reason={resp.choices[0].finish_reason!r}"
        )
    return text.strip()


# Single source of truth for which providers exist. call_llm() builds its
# failover chain from this map plus CONFIG.llm.fallback_order.
_PROVIDERS = {
    "gemini": _call_gemini,
    "claude": _call_claude,
    "openai": _call_openai,
}


def _provider_chain() -> list[str]:
    """Ordered providers to attempt: the configured `provider` first, then
    `CONFIG.llm.fallback_order`, skipping already-included and unknown names.
    """
    primary = cfg.CONFIG.llm.provider
    if primary not in _PROVIDERS:
        raise ValueError(
            f"unknown LLM provider {primary!r} — expected one of "
            f"{sorted(_PROVIDERS)}"
        )
    chain = [primary]
    for provider in cfg.CONFIG.llm.fallback_order:
        if provider in _PROVIDERS and provider not in chain:
            chain.append(provider)
    return chain


def call_llm(system: str, user: str) -> str:
    # Calls the configured LLM provider, failing over to the next on error.
    chain = _provider_chain()
    errors: list[str] = []
    last_exc: Exception | None = None
    for index, provider in enumerate(chain):
        try:
            text = _PROVIDERS[provider](system, user)
        except Exception as exc:
            # Catch-all is deliberate: any provider error triggers failover,
            # not just availability errors. The accumulated summary is
            # raised only if we exhaust the chain.
            last_exc = exc
            errors.append(f"{provider}: {type(exc).__name__}: {exc}")
            nxt = chain[index + 1] if index + 1 < len(chain) else None
            if nxt is not None:
                logger.warning(
                    "LLM provider %r failed (%s: %s) — failing over to %r",
                    provider, type(exc).__name__, exc, nxt,
                )
            else:
                logger.error(
                    "LLM provider %r failed (%s: %s) — no providers left",
                    provider, type(exc).__name__, exc,
                )
            continue
        if index > 0:
            logger.info("LLM provider %r succeeded after failover", provider)
        return text
    # Chain exhausted. With a single provider (failover disabled, or no other
    # known provider configured), re-raise its error unwrapped so the original
    # propagates cleanly. With several, raise a summary chained to the last.
    if len(chain) == 1 and last_exc is not None:
        raise last_exc
    raise RuntimeError(
        "all LLM providers failed — " + " | ".join(errors)
    ) from last_exc


# --- Public entry point ----------------------------------------------------

@dataclass(frozen=True)
class RespondResult:
    """respond()'s full result: the reply text plus the canon facts that
    were actually injected into the prompt. The Canonizer needs exactly
    what the model saw — re-retrieval could not reproduce it."""

    text: str
    canon_facts: tuple[str, ...] = ()


def respond(
    persona_id: str,
    user_question: str,
    history: list[Exchange] | None = None,
    asker_name: str | None = None,
) -> RespondResult:
    """Produce a single in-character response in the persona's voice.

    `history` is a list of recent Exchange objects (oldest first) to surface
    as conversational context. `asker_name` is the Discord display name of
    the person asking, surfaced in the prompt so the persona can address
    them naturally.
    """
    persona = _persona(persona_id)
    corpus = _voice_pool(persona_id)
    sample_size = cfg.CONFIG.style.sample_size
    style_examples = tuple(
        random.sample(corpus, min(sample_size, len(corpus)))
    )
    query_vecs = _embed_queries([user_question])
    stances, excerpts = _retrieve(
        persona_id, [user_question], cfg.CONFIG.retrieval.top_k,
        query_vecs=query_vecs,
    )
    canon_facts: list[str] = (
        _retrieve_canon(persona, [user_question], query_vecs)
        if persona.canon.enabled else []
    )
    # Conditional query expansion: only when the first pass found nothing
    # within the distance floor (e.g. a slangy phrasing that embeds far from
    # the cleaner stored queries). A neutral rephrasing usually rescues it;
    # queries that already matched never pay for the extra call. Skipped under
    # DRY_RUN to keep prompt inspection LLM-free.
    expanded = False
    if not stances and not excerpts and not DRY_RUN:
        variants = _expand_query(user_question, persona)
        if variants:
            expanded = True
            if DEBUG_PROMPT:
                print(f"[expansion] first pass empty — retrying with {variants}",
                      file=sys.stderr)
            all_queries = [user_question, *variants]
            expanded_vecs = _embed_queries(all_queries)
            stances, excerpts = _retrieve(
                persona_id, all_queries,
                cfg.CONFIG.retrieval.top_k,
                floor_bonus=cfg.CONFIG.retrieval.expanded_floor_bonus,
                query_vecs=expanded_vecs,
            )
            if persona.canon.enabled:
                canon_facts = _retrieve_canon(
                    persona, all_queries, expanded_vecs,
                    floor_bonus=cfg.CONFIG.retrieval.expanded_floor_bonus,
                )

    length_mode, length_instruction = _pick_length()
    tic_directives, tics_fired = _roll_tics(persona)
    # Keys on canon.enabled (not canon.extraction), matching the
    # improvise-bullet swap: a persona allowed to invent gets the scale
    # roll even if post-reply extraction is off.
    invention_mode, invention_instruction = (
        _pick_invention() if persona.canon.enabled else (None, None)
    )
    logger.info(
        "length_mode=%s invention_mode=%s persona=%s question=%r "
        "style_examples=%d stances=%d excerpts=%d canon=%d expanded=%s "
        "tic_rolls=%d/%d",
        length_mode, invention_mode or "-", persona_id, user_question[:80],
        len(style_examples), len(stances), len(excerpts), len(canon_facts),
        expanded, tics_fired, len(persona.rolled_tics),
    )

    system_msg = _build_system_message(
        persona, style_examples, length_mode, length_instruction,
        tic_directives=tic_directives,
        invention_mode=invention_mode,
        invention_instruction=invention_instruction,
    )
    user_msg = _build_user_message(
        stances=stances,
        excerpts=excerpts,
        question=user_question,
        name=persona.display_name,
        history=history,
        asker_name=asker_name,
        canon_facts=canon_facts,
        cast_users=persona.cast_users,
    )

    _maybe_log_prompt(system_msg, user_msg)

    if DRY_RUN:
        return RespondResult(
            text=(
                f"(PERSONA_DRY_RUN=1 — length_mode={length_mode}; "
                f"invention_mode={invention_mode or '-'}; "
                f"LLM call skipped; prompt logged to stderr)"
            ),
            canon_facts=tuple(canon_facts),
        )

    return RespondResult(
        text=call_llm(system_msg, user_msg),
        canon_facts=tuple(canon_facts),
    )
