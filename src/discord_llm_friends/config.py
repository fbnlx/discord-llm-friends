"""Loads project-wide configuration from config.yaml + .env.

Layered defaults: every config field has a default baked into the
dataclasses below. config.yaml at the repo root may selectively override
any of them. A small subset of operational knobs (LLM_PROVIDER,
DAILY_LIMIT, HISTORY_TURNS) accept env-var overrides that take
precedence over both YAML and code defaults.

Secrets (API keys, Discord tokens) live in .env — load_dotenv() runs on
import so os.getenv() picks them up everywhere.

The exported `CONFIG` singleton is built once at import time. Tests that
want a different config can call `load_config(path=...)`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()


# --- Filesystem layout -------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent.parent
"""Repo root — three parents above this file (src/discord_llm_friends/config.py)."""

PERSONAS_DIR = ROOT / "personas"
CHROMA_DIR = ROOT / "chroma"
STATE_DIR = ROOT / "state"
RATE_LIMITS_DIR = STATE_DIR / "rate_limits"
HISTORY_DIR = STATE_DIR / "history"

CONFIG_PATH = ROOT / "config.yaml"


# --- Typed config dataclasses (defaults match docs in config.yaml) -----------

@dataclass(frozen=True)
class LLMConfig:
    provider: str = "gemini"
    models: dict[str, str] = field(default_factory=lambda: {
        "gemini": "gemini-flash-latest",
        "claude": "claude-haiku-4-5",
        "openai": "gpt-5.6-terra",
    })
    fallback_order: list[str] = field(default_factory=lambda: [
        "gemini", "openai", "claude",
    ])
    temperature: float = 0.6
    max_output_tokens: int = 2000
    # Per-request HTTP timeout applied to every provider call (generation,
    # expansion, canonizer, embeddings). Without it an overloaded provider
    # can hold a connection open for minutes before erroring, and failover
    # only starts after that wait.
    request_timeout_seconds: float = 45.0


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = "text-embedding-3-large"
    batch_size: int = 100
    max_retries: int = 5


@dataclass(frozen=True)
class RetrievalConfig:
    # Window/excerpt budget per query (stance cards get their own slots in the
    # engine, STANCE_SLOTS). Each candidate must ALSO clear the distance floor.
    top_k: int = 4
    # L2-distance floors: a hit farther than this is dropped rather than
    # injected as filler. Cards (synthesized, clean) get a looser floor than
    # windows (noisy multi-speaker excerpts). When nothing clears the floor the
    # prompt falls back to the always-present dossier. Tune by reading
    # dev.query_check distances on known-relevant vs irrelevant queries.
    max_card_distance: float = 0.80
    max_window_distance: float = 0.60
    # Added to both floors ONLY on the query-expansion retry. The first pass
    # found nothing clean, so on the retry we accept a looser match from the
    # neutral rephrasing (e.g. crude slang that only reaches ~0.83) rather than
    # inject nothing — while still excluding outright noise (~0.92+).
    expanded_floor_bonus: float = 0.08


@dataclass(frozen=True)
class SynthQueriesConfig:
    """Knobs for the doc2query synthetic-query generation pipeline step.

    For each cleaned-corpus entry, asks Gemini for a one-line synthetic
    user-question / topic / trigger that would plausibly prompt that
    comment as a reply. The embed step then vectorizes these synthetic
    queries instead of the comments themselves, shifting the embedding
    space from answer-shaped to question-shaped (better Hungarian recall
    on the runtime question → corpus matching path).
    """
    # Cleaned-corpus entries bundled per LLM call. Higher = fewer
    # requests, more risk of structured-output drift at the edges.
    batch_size: int = 20
    # Concurrent in-flight LLM requests. Bounded by Gemini RPM quota —
    # lower this if you see 429s in the logs.
    concurrency: int = 5
    # Per-request retry budget on transient failures.
    max_retries: int = 5
    # Override the output language for generated queries. If None, the
    # persona's `language` field is used (recommended).
    output_language: str | None = None


@dataclass(frozen=True)
class StyleConfig:
    """Knobs for per-call random style-example sampling.

    On each response, the engine draws a fresh random subset of cleaned-
    corpus entries and injects them into the system prompt as voice
    examples. This rotates which characteristic phrases the LLM sees
    instead of pinning the same set every call.
    """
    # How many cleaned-corpus entries to sample per response. Higher =
    # richer voice exposure; lower = leaner prompt. ~50 matches the
    # static-anchor count the engine used before sampling was introduced.
    sample_size: int = 50


@dataclass(frozen=True)
class BotConfig:
    rate_limit_seconds: int = 10
    daily_limit_per_user: int = 50
    max_discord_message_chars: int = 1900
    history_turns_in_prompt: int = 3
    echo_question_in_response: bool = True


@dataclass(frozen=True)
class HistoryConfig:
    retention_per_channel: int = 50
    max_question_chars: int = 1000
    max_response_chars: int = 2000


@dataclass(frozen=True)
class CleanupConfig:
    min_words: int = 4
    drop_pure_quotes: bool = True


@dataclass(frozen=True)
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    synth_queries: SynthQueriesConfig = field(default_factory=SynthQueriesConfig)
    style: StyleConfig = field(default_factory=StyleConfig)
    bot: BotConfig = field(default_factory=BotConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)


# --- Loader -----------------------------------------------------------------

def _override(dc: Any, section: dict[str, Any] | None) -> Any:
    """Return a new copy of `dc` with keys from `section` applied, if any.

    Unknown keys in `section` are silently ignored so a forward-compat
    config.yaml from a newer version doesn't blow up an older binary.
    """
    if not section:
        return dc
    fields = {f for f in dc.__dataclass_fields__}
    overrides = {k: v for k, v in section.items() if k in fields}
    return replace(dc, **overrides) if overrides else dc


def load_config(path: Path | None = None) -> Config:
    """Build a `Config` from defaults + config.yaml + env overrides."""
    cfg = Config()
    yaml_path = path or CONFIG_PATH

    if yaml_path.exists():
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        cfg = Config(
            llm=_override(cfg.llm, data.get("llm")),
            embedding=_override(cfg.embedding, data.get("embedding")),
            retrieval=_override(cfg.retrieval, data.get("retrieval")),
            synth_queries=_override(cfg.synth_queries, data.get("synth_queries")),
            style=_override(cfg.style, data.get("style")),
            bot=_override(cfg.bot, data.get("bot")),
            history=_override(cfg.history, data.get("history")),
            cleanup=_override(cfg.cleanup, data.get("cleanup")),
        )

    # Env overrides for the documented subset. Each accepts string input
    # and coerces to the appropriate type.
    if (env := os.getenv("LLM_PROVIDER")):
        cfg = replace(cfg, llm=replace(cfg.llm, provider=env.lower()))
    if (env := os.getenv("DAILY_LIMIT")):
        cfg = replace(cfg, bot=replace(cfg.bot, daily_limit_per_user=int(env)))
    if (env := os.getenv("HISTORY_TURNS")):
        cfg = replace(cfg, bot=replace(cfg.bot, history_turns_in_prompt=int(env)))

    return cfg


CONFIG: Config = load_config()
"""Process-wide singleton, built at import. Re-import or call load_config()
explicitly in tests that need a different config."""
