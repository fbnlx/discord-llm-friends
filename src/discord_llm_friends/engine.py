"""Persona engine — turns a user question into an in-character response.

Public entry point: `respond(persona_id, question, ...)`.

Pipeline:
  1. Load the persona's description + tics, and draw a fresh random
     sample of style examples from the cleaned corpus.
  2. Embed the user's question with the embedding model in CONFIG.
  3. Query the persona's ChromaDB collection for top-K relevant comments.
  4. Assemble a system prompt (who they are + style examples) and a user
     prompt (retrieved comments + recent channel history + the question).
  5. Dispatch to the LLM provider selected by CONFIG.llm.provider.
  6. Return the generated response string.

Env-only knobs (no YAML equivalent):
  PERSONA_DEBUG_PROMPT=1 — print assembled prompts to stderr before the call.
  PERSONA_DRY_RUN=1      — skip the LLM call, return a placeholder.
"""

from __future__ import annotations

import logging
import os
import random
import sys
from functools import lru_cache

import chromadb
from chromadb.config import Settings
from openai import OpenAI

from discord_llm_friends import config as cfg
from discord_llm_friends import personas as personas_module
from discord_llm_friends.history import Exchange


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


# --- Lazy shared clients (per-process singletons) --------------------------

@lru_cache(maxsize=1)
def _openai_client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY not set — embeddings require it. "
            "Make sure .env exists and is populated."
        )
    return OpenAI()


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
def _cleaned_corpus(persona_id: str) -> tuple[str, ...]:
    """Load the persona's cleaned corpus once and cache it.

    Used as the pool for per-call style-example sampling — each
    `respond()` draws a fresh random subset from this tuple, so the
    system prompt shows different examples on every query. Returns a
    tuple so it's hashable for lru_cache and immutable to callers.
    """
    return tuple(personas_module.load_cleaned(_persona(persona_id)))


# --- Retrieval -------------------------------------------------------------

def _retrieve(persona_id: str, question: str, n: int) -> list[str]:
    """Embed the question and pull top-N nearest stored comments."""
    embed_resp = _openai_client().embeddings.create(
        model=cfg.CONFIG.embedding.model,
        input=[question],
    )
    query_vec = embed_resp.data[0].embedding

    collection = _chroma_client().get_collection(name=persona_id)
    result = collection.query(query_embeddings=[query_vec], n_results=n)
    return result["documents"][0]


# --- Prompt assembly -------------------------------------------------------

def _build_system_message(
    persona: personas_module.Persona,
    style_examples: tuple[str, ...],
    length_mode: str,
    length_instruction: str,
) -> str:
    tic_lines = (
        "\n".join(f"- {tic}" for tic in persona.tics)
        if persona.tics
        else "(none specified — infer from examples)"
    )
    example_lines = "\n".join(f"- {a}" for a in style_examples)
    name = persona.display_name

    return (
        f"You are responding AS {name}. Stay completely in character. Respond "
        f"ONLY in {persona.language}, in {name}'s voice. Do not announce "
        f"yourself as an AI, do not add safety disclaimers, do not break the "
        f"fourth wall.\n"
        f"\n"
        f"WHO {name.upper()} IS:\n"
        f"{persona.description}\n"
        f"\n"
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
        f"- IMPROVISE confidently. Extrapolate from {name}'s personality "
        f"to say new things this person could plausibly say — don't just "
        f"rearrange phrasings from the examples, invent in voice. Just "
        f"don't invent biographical facts (no new family members, schools, "
        f"jobs, etc.).\n"
        f"- DO NOT bring up {name}'s hobbies, games, or specific interests "
        f"unless the user's question already mentions them. The persona "
        f"has many interests; resist the urge to advertise them in every "
        f"response.\n"
        f"- The retrieved examples in the user message show how {name} "
        f"talks (vocabulary, register, emoticons). They are STYLE "
        f"reference, not a topic menu.\n"
        f"- Match the crudeness and energy of the examples.\n"
        f"- For topics from after {name}'s era, riff in voice rather than "
        f"refusing or breaking character.\n"
        f"- If a recent conversation is shown, you can react to it — but "
        f"you're not obligated to reference earlier topics.\n"
        f"- Output ONLY {name}'s reply, in {persona.language}. No analysis, "
        f"no English meta-commentary, no markdown formatting, no thinking "
        f"aloud."
    )


def _format_history(history: list[Exchange]) -> str:
    lines = []
    for ex in history:
        lines.append(f"[{ex.user_name} → {ex.persona}]: {ex.question}")
        lines.append(f"[{ex.persona}]: {ex.response}")
    return "\n".join(lines)


def _build_user_message(
    retrieved: list[str],
    question: str,
    name: str,
    history: list[Exchange] | None,
    asker_name: str | None,
) -> str:
    sections: list[str] = []

    if history:
        sections.append(
            "RECENT CONVERSATION IN THIS CHANNEL (oldest first):\n"
            + _format_history(history)
        )

    if retrieved:
        retrieved_block = "\n".join(f"- {r}" for r in retrieved)
        sections.append(
            f"HOW {name.upper()} TALKS (style/vocabulary reference — match "
            f"the voice, not the topics):\n{retrieved_block}"
        )

    asker_prefix = f" (from @{asker_name})" if asker_name else ""
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

def call_llm(system: str, user: str) -> str:
    """Dispatch to the provider selected by CONFIG.llm.provider."""
    provider = cfg.CONFIG.llm.provider
    if provider == "gemini":
        return _call_gemini(system, user)
    if provider == "claude":
        return _call_claude(system, user)
    if provider == "openai":
        return _call_openai(system, user)
    raise ValueError(
        f"unknown LLM provider {provider!r} — expected one of "
        f"{sorted(cfg.CONFIG.llm.models)}"
    )


def _call_gemini(system: str, user: str) -> str:
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        raise RuntimeError("GOOGLE_API_KEY (or GEMINI_API_KEY) not set")
    from google import genai
    from google.genai import types

    client = genai.Client()
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

    client = Anthropic()
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


# --- Public entry point ----------------------------------------------------

def respond(
    persona_id: str,
    user_question: str,
    history: list[Exchange] | None = None,
    asker_name: str | None = None,
) -> str:
    """Produce a single in-character response in the persona's voice.

    `history` is a list of recent Exchange objects (oldest first) to surface
    as conversational context. `asker_name` is the Discord display name of
    the person asking, surfaced in the prompt so the persona can address
    them naturally.
    """
    persona = _persona(persona_id)
    corpus = _cleaned_corpus(persona_id)
    sample_size = cfg.CONFIG.style.sample_size
    style_examples = tuple(
        random.sample(corpus, min(sample_size, len(corpus)))
    )
    retrieved = _retrieve(persona_id, user_question, cfg.CONFIG.retrieval.top_k)

    length_mode, length_instruction = _pick_length()
    logger.info(
        "length_mode=%s persona=%s question=%r style_examples=%d",
        length_mode, persona_id, user_question[:80], len(style_examples),
    )

    system_msg = _build_system_message(
        persona, style_examples, length_mode, length_instruction,
    )
    user_msg = _build_user_message(
        retrieved=retrieved,
        question=user_question,
        name=persona.display_name,
        history=history,
        asker_name=asker_name,
    )

    _maybe_log_prompt(system_msg, user_msg)

    if DRY_RUN:
        return (
            f"(PERSONA_DRY_RUN=1 — length_mode={length_mode}; "
            f"LLM call skipped; prompt logged to stderr)"
        )

    return call_llm(system_msg, user_msg)
