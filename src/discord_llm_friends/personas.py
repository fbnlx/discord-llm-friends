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
class Persona:
    id: str
    display_name: str
    language: str
    discord: DiscordSpec
    tics: list[str]
    fallback_messages: list[str]
    quota_exhausted_messages: list[str]
    rate_limit_message: str
    cleanup: PersonaCleanup
    description: str
    """Full prose content of description.md, used as-is in the system prompt."""

    dossier: str
    """Full prose content of dossier.md ("" if absent) — generated identity
    profile appended to the system prompt after the description."""

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

    try:
        return Persona(
            id=data["id"],
            display_name=data["display_name"],
            language=data["language"],
            discord=discord,
            tics=list(data.get("tics") or []),
            fallback_messages=list(data.get("fallback_messages") or []),
            quota_exhausted_messages=list(data.get("quota_exhausted_messages") or []),
            rate_limit_message=data.get("rate_limit_message", ""),
            cleanup=cleanup,
            description=description,
            dossier=dossier,
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
