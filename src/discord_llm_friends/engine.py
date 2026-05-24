"""Persona engine — turns a user question into an in-character response.

Public entry point: `respond(persona_id, question, ...)`.

Pipeline:
  1. Load the persona's description + tics + style anchors.
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

import os
import sys
from functools import lru_cache

import chromadb
from chromadb.config import Settings
from openai import OpenAI

from discord_llm_friends import config as cfg
from discord_llm_friends import personas as personas_module
from discord_llm_friends.history import Exchange


# --- Env-only debug knobs ---------------------------------------------------

DEBUG_PROMPT = os.getenv("PERSONA_DEBUG_PROMPT", "").strip() == "1"
DRY_RUN = os.getenv("PERSONA_DRY_RUN", "").strip() == "1"


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
def _style_anchors(persona_id: str) -> tuple[str, ...]:
    # Return a tuple so it's hashable for lru_cache and immutable to callers.
    return tuple(personas_module.load_style_anchors(_persona(persona_id)))


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

def _build_system_message(persona: personas_module.Persona, anchors: tuple[str, ...]) -> str:
    tic_lines = (
        "\n".join(f"- {tic}" for tic in persona.tics)
        if persona.tics
        else "(none specified — infer from examples)"
    )
    anchor_lines = "\n".join(f"- {a}" for a in anchors)
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
        f"EXAMPLES OF THINGS {name.upper()} HAS ACTUALLY WRITTEN (match voice, "
        f"register, length, spelling tics, emoticon style — but do NOT quote "
        f"them verbatim):\n"
        f"{anchor_lines}\n"
        f"\n"
        f"GUIDANCE:\n"
        f"- Keep responses similar in length to the examples (usually 1–3 "
        f"sentences; longer rants only if the question really warrants).\n"
        f"- Match the energy and crudeness of the examples.\n"
        f"- For topics you wouldn't reasonably know about (e.g. events from "
        f"after the persona's era), riff in voice rather than refusing — "
        f"improvise plausibly, but don't invent biographical facts.\n"
        f"- Don't quote the examples verbatim. Use them as style reference.\n"
        f"- Other personas may have spoken earlier in this same channel — if "
        f"a recent conversation is shown below, you can reference what they "
        f"said, agree, disagree, mock them, whatever fits your voice.\n"
        f"- Output ONLY {name}'s reply, in {persona.language}. No analysis, "
        f"no explanations of your stylistic choices, no English meta-"
        f"commentary, no markdown formatting, no thinking aloud."
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
            f"Things {name} has previously written about topics similar to "
            f"this question:\n{retrieved_block}"
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
    anchors = _style_anchors(persona_id)
    retrieved = _retrieve(persona_id, user_question, cfg.CONFIG.retrieval.top_k)

    system_msg = _build_system_message(persona, anchors)
    user_msg = _build_user_message(
        retrieved=retrieved,
        question=user_question,
        name=persona.display_name,
        history=history,
        asker_name=asker_name,
    )

    _maybe_log_prompt(system_msg, user_msg)

    if DRY_RUN:
        return "(PERSONA_DRY_RUN=1 — LLM call skipped; prompt logged to stderr)"

    return call_llm(system_msg, user_msg)
