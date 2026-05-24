"""Loads per-persona definitions from `personas/<id>/`.

Each persona is a folder containing:
  persona.yaml        — structured config (this module parses it)
  description.md      — long-form prose, appended to the system prompt
  raw.json            — input corpus (pipeline input)
  cleaned.json        — pipeline output (engine input)
  style_anchors.json  — hand-curated examples (engine input)

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
    def style_anchors_path(self) -> Path:
        return self.folder / "style_anchors.json"


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


def load_style_anchors(persona: Persona) -> list[str]:
    """Read the persona's hand-curated style anchors. Raises if absent."""
    if not persona.style_anchors_path.exists():
        raise FileNotFoundError(
            f"style anchors missing: {persona.style_anchors_path}\n"
            f"Curate {persona.style_anchors_path.relative_to(cfg.ROOT)} "
            f"(40-60 strings hand-picked from {persona.cleaned_path.name}) "
            f"before calling the engine."
        )
    return json.loads(persona.style_anchors_path.read_text(encoding="utf-8"))


def resolved_min_words(persona: Persona) -> int:
    """Per-persona override of CONFIG.cleanup.min_words, if set."""
    return (
        persona.cleanup.min_words
        if persona.cleanup.min_words is not None
        else cfg.CONFIG.cleanup.min_words
    )
