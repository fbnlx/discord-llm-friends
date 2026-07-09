"""Loads per-persona definitions from `personas/<id>/`.

Each persona is a folder containing:
  persona.yaml        — structured config (this module parses it)
  description.md      — long-form prose, appended to the system prompt
  raw.json            — input corpus (pipeline input)
  cleaned.json        — pipeline output (RAG index source; also the
                        voice-sample pool when no voice_pool.json exists)
  voice_pool.json     — optional: explicit voice-sample pool. Personas
                        whose retrieval documents are NOT all their own
                        words (group-chat windows) need this split —
                        sampling multi-speaker excerpts as the persona's
                        own voice would corrupt the mimicry.
  dossier.md          — optional: generated identity profile (stances,
                        relationships) appended to the system prompt
                        after description.md.

A persona id is the folder name. `_example` is shipped with the repo as
the format reference; `discover()` excludes it by default so `--all`
doesn't try to start a bot for it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from discord_llm_friends import config as cfg


# --- Schema -----------------------------------------------------------------

@dataclass(frozen=True)
class DiscordSpec:
    command_name: str
    command_description: str
    question_arg_description: str
    token_env: str


@dataclass(frozen=True)
class PersonaCleanup:
    """Per-persona overrides on top of CONFIG.cleanup."""
    min_words: int | None = None
    drop_quote_block_markers: list[str] = field(default_factory=list)
    drop_system_messages: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RolledTic:
    """A Tic injected only when a per-call dice roll hits, so a signature
    quirk appears at a controlled rate instead of in every reply — the same
    reasoning as the engine's length modes (prompt-described frequency
    collapses; mechanical rolls hold).

    `text` is injected with probability `p`; otherwise `off_text` (if set)
    is injected instead — an explicit counter-instruction for quirks the
    model would over-apply on its own."""
    p: float
    text: str
    off_text: str = ""


@dataclass(frozen=True)
class PersonaCanon:
    """Canon-memory knobs for a Synthetic persona (ADR-0006).

    Disabled by default. Enabling turns on canon retrieval (the
    `<id>__canon` collection), the always-injected Timeline sheet, and the
    post-response Canonizer. Distances are L2 in the shared embedding
    space, same scale as CONFIG.retrieval floors.
    """
    enabled: bool = False
    extraction: bool = True
    max_facts: int = 3
    max_distance: float = 0.80
    dedup_max_distance: float = 0.35
    max_new_facts_per_exchange: int = 3
    timeline_max_chars: int = 6000


@dataclass(frozen=True)
class Persona:
    id: str
    display_name: str
    language: str
    discord: DiscordSpec
    tics: list[str]
    rolled_tics: list[RolledTic]
    fallback_messages: list[str]
    quota_exhausted_messages: list[str]
    rate_limit_message: str
    cleanup: PersonaCleanup
    description: str
    """Full prose content of description.md, used as-is in the system prompt."""

    dossier: str
    """Full prose content of dossier.md ("" if absent) — generated identity
    profile appended to the system prompt after the description."""

    timeline: str
    """Timeline sheet (canon/timeline.md, "" if absent) — always-injected
    dated skeleton of a Synthetic persona's invented world; the canon
    analog of the dossier."""

    canon: PersonaCanon
    """Canon-memory configuration (`canon:` in persona.yaml)."""

    cast_personas: dict[str, str]
    """Cast: other bot persona ids → who they are to THIS persona (e.g.
    a future edition knowing `peter` is his own younger self)."""

    cast_users: dict[str, str]
    """Cast: Discord handles → the real identity behind them, so the
    persona knows who is prompting."""

    folder: Path
    """Where this persona lives on disk. Useful for derived paths."""

    # --- Derived paths --------------------------------------------------

    @property
    def raw_path(self) -> Path:
        return self.folder / "raw.json"

    @property
    def cleaned_path(self) -> Path:
        return self.folder / "cleaned.json"

    @property
    def voice_pool_path(self) -> Path:
        return self.folder / "voice_pool.json"

    @property
    def canon_dir(self) -> Path:
        return self.folder / "canon"

    @property
    def canon_ledger_path(self) -> Path:
        return self.canon_dir / "facts.jsonl"

    @property
    def timeline_path(self) -> Path:
        return self.canon_dir / "timeline.md"

    @property
    def world_bible_path(self) -> Path:
        return self.canon_dir / "world_bible.json"


# --- Loader -----------------------------------------------------------------

def load(persona_id: str, base_dir: Path | None = None) -> Persona:
    """Load a single persona by id. Raises FileNotFoundError if missing,
    ValueError if the YAML is malformed or missing required fields."""
    base = base_dir or cfg.PERSONAS_DIR
    folder = base / persona_id

    yaml_path = folder / "persona.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"persona config not found: {yaml_path}\n"
            f"Expected a folder personas/{persona_id}/ with persona.yaml + "
            f"description.md + raw.json. See personas/_example/ for the format."
        )

    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}

    try:
        discord = DiscordSpec(
            command_name=data["discord"]["command_name"],
            command_description=data["discord"]["command_description"],
            question_arg_description=data["discord"]["question_arg_description"],
            token_env=data["discord"]["token_env"],
        )
    except KeyError as e:
        raise ValueError(
            f"{yaml_path}: missing required discord field {e}"
        ) from e

    cleanup_data = data.get("cleanup") or {}
    cleanup = PersonaCleanup(
        min_words=cleanup_data.get("min_words"),
        drop_quote_block_markers=list(cleanup_data.get("drop_quote_block_markers") or []),
        drop_system_messages=list(cleanup_data.get("drop_system_messages") or []),
    )

    canon_data = data.get("canon") or {}
    canon = PersonaCanon(
        enabled=bool(canon_data.get("enabled", False)),
        extraction=bool(canon_data.get("extraction", True)),
        max_facts=int(canon_data.get("max_facts", 3)),
        max_distance=float(canon_data.get("max_distance", 0.80)),
        dedup_max_distance=float(canon_data.get("dedup_max_distance", 0.35)),
        max_new_facts_per_exchange=int(
            canon_data.get("max_new_facts_per_exchange", 3)
        ),
        timeline_max_chars=int(canon_data.get("timeline_max_chars", 6000)),
    )

    rolled_tics = []
    for entry in data.get("rolled_tics") or []:
        tic = RolledTic(
            p=float(entry.get("p", 0.0)),
            text=str(entry.get("text", "")).strip(),
            off_text=str(entry.get("off_text", "")).strip(),
        )
        if tic.text and 0.0 < tic.p <= 1.0:
            rolled_tics.append(tic)

    cast_data = data.get("cast") or {}
    cast_personas = {
        str(k): str(v) for k, v in (cast_data.get("personas") or {}).items()
    }
    cast_users = {
        str(k): str(v) for k, v in (cast_data.get("users") or {}).items()
    }

    description_path = folder / "description.md"
    description = (
        description_path.read_text(encoding="utf-8").strip()
        if description_path.exists()
        else ""
    )

    dossier_path = folder / "dossier.md"
    dossier = (
        dossier_path.read_text(encoding="utf-8").strip()
        if dossier_path.exists()
        else ""
    )

    timeline_file = folder / "canon" / "timeline.md"
    timeline = (
        timeline_file.read_text(encoding="utf-8").strip()
        if timeline_file.exists()
        else ""
    )
    if timeline and len(timeline) > canon.timeline_max_chars:
        # Never truncate — a reviewed load-bearing dated fact silently
        # dropped is worse than a fat prompt. Consolidation trims editorially.
        logging.getLogger(__name__).warning(
            "%s: canon/timeline.md is %d chars (budget %d) — injected whole; "
            "trim it via canon consolidation",
            persona_id, len(timeline), canon.timeline_max_chars,
        )

    try:
        return Persona(
            id=data["id"],
            display_name=data["display_name"],
            language=data["language"],
            discord=discord,
            tics=list(data.get("tics") or []),
            rolled_tics=rolled_tics,
            fallback_messages=list(data.get("fallback_messages") or []),
            quota_exhausted_messages=list(data.get("quota_exhausted_messages") or []),
            rate_limit_message=data.get("rate_limit_message", ""),
            cleanup=cleanup,
            description=description,
            dossier=dossier,
            timeline=timeline,
            canon=canon,
            cast_personas=cast_personas,
            cast_users=cast_users,
            folder=folder,
        )
    except KeyError as e:
        raise ValueError(
            f"{yaml_path}: missing required field {e}"
        ) from e


def discover(*, include_example: bool = False, base_dir: Path | None = None) -> list[str]:
    """Return persona ids found on disk, sorted. `_example` is excluded
    unless `include_example=True` — explicit loads of `_example` still work."""
    base = base_dir or cfg.PERSONAS_DIR
    if not base.exists():
        return []
    ids = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "persona.yaml").exists():
            continue
        if child.name == "_example" and not include_example:
            continue
        ids.append(child.name)
    return ids


def load_cleaned(persona: Persona) -> list[str]:
    """Read the persona's cleaned corpus. Raises if absent."""
    if not persona.cleaned_path.exists():
        raise FileNotFoundError(
            f"cleaned corpus missing: {persona.cleaned_path}\n"
            f"Run the cleanup pipeline to produce "
            f"{persona.cleaned_path.relative_to(cfg.ROOT)} from "
            f"{persona.raw_path.name} before starting the bot."
        )
    return json.loads(persona.cleaned_path.read_text(encoding="utf-8"))


def load_voice_pool(persona: Persona) -> list[str]:
    """The pool the engine samples voice examples from: voice_pool.json
    when present, else the cleaned corpus (the historical behavior, where
    cleaned.json served both retrieval and voice)."""
    if persona.voice_pool_path.exists():
        return json.loads(persona.voice_pool_path.read_text(encoding="utf-8"))
    return load_cleaned(persona)


def resolved_min_words(persona: Persona) -> int:
    """Per-persona override of CONFIG.cleanup.min_words, if set."""
    return (
        persona.cleanup.min_words
        if persona.cleanup.min_words is not None
        else cfg.CONFIG.cleanup.min_words
    )
