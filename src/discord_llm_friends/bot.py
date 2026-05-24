"""Discord bot runtime.

Two modes:

  python -m discord_llm_friends.bot --persona <id>
      Run a single persona. Used for debugging or single-persona deployments.

  python -m discord_llm_friends.bot --all
      Run every persona discovered under personas/ (excluding _example/),
      each as its own discord.Client, all in one process / one asyncio
      event loop. Production mode for the e2-micro deployment.

Personas with a missing token env var are skipped + warned. Personas with
malformed persona.yaml are skipped + warned (full traceback at log level).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
from datetime import date

import discord
from discord import app_commands

from discord_llm_friends import config as cfg
from discord_llm_friends import engine
from discord_llm_friends import personas as personas_module
from discord_llm_friends.history import ChannelHistory, Exchange
from discord_llm_friends.personas import Persona


# --- Daily quota (per-user, per-persona, JSON-persisted) ------------------

class DailyQuota:
    """Per-persona daily request quota with JSON persistence.

    File format: { "<user_id>": ["<date_iso>", count], ... }

    One file per persona (no cross-persona sharing), so no inter-process
    locking is needed even when multiple personas run in one process.
    """

    def __init__(self, persona_id: str, limit: int) -> None:
        self.persona_id = persona_id
        self.limit = limit
        self.path = cfg.RATE_LIMITS_DIR / f"{persona_id}.json"
        self.logger = logging.getLogger(f"bot.{persona_id}.quota")
        self._cache: dict[int, tuple[str, int]] = {}
        self._load()

    def _load(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._cache = {int(k): (v[0], int(v[1])) for k, v in raw.items()}
        except Exception:
            self.logger.exception(
                "failed to load quota file %s — starting fresh", self.path,
            )
            self._cache = {}

    def _save(self) -> None:
        out = {str(k): [v[0], v[1]] for k, v in self._cache.items()}
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(out), encoding="utf-8")
        tmp.replace(self.path)

    def check_and_increment(self, user_id: int) -> tuple[bool, int]:
        """Try to consume one request. Returns (allowed, remaining_after)."""
        today_iso = date.today().isoformat()
        last_date, count = self._cache.get(user_id, (today_iso, 0))
        if last_date != today_iso:
            count = 0
        if count >= self.limit:
            return False, 0
        new_count = count + 1
        self._cache[user_id] = (today_iso, new_count)
        self._save()
        return True, self.limit - new_count


# --- Per-user short-window rate limit (in-process) ------------------------

class RateLimiter:
    """Per-persona, in-process short-window rate limit. Survives only for
    the process lifetime."""

    def __init__(self, window_seconds: int) -> None:
        self.window = window_seconds
        self._last: dict[int, float] = {}

    def ok(self, user_id: int) -> bool:
        now = time.monotonic()
        last = self._last.get(user_id)
        if last is not None and (now - last) < self.window:
            return False
        self._last[user_id] = now
        return True


# --- Message splitting for Discord's 2000-char cap ------------------------

def _split_for_discord(text: str, limit: int) -> list[str]:
    """Split on paragraph/line/word boundaries to fit Discord's cap."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        cut = remaining.rfind("\n\n", 0, limit)
        if cut < limit // 2:
            cut = remaining.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = remaining.rfind(" ", 0, limit)
        if cut < limit // 4:
            cut = limit  # hard cut
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    return chunks


# --- Client builder -------------------------------------------------------

def _fallback_text(persona: Persona) -> str:
    options = persona.fallback_messages or ["something went wrong, try again."]
    return random.choice(options)


def _quota_exhausted_text(persona: Persona) -> str:
    options = persona.quota_exhausted_messages or [
        "Daily limit reached. Try again tomorrow."
    ]
    return random.choice(options)


def build_client(persona: Persona) -> discord.Client:
    """Create the Discord client + slash command for one persona."""
    logger = logging.getLogger(f"bot.{persona.id}")
    bot_cfg = cfg.CONFIG.bot

    guild_id_raw = os.getenv("DISCORD_GUILD_ID", "").strip()
    guild_obj = discord.Object(id=int(guild_id_raw)) if guild_id_raw else None

    quota = DailyQuota(persona.id, bot_cfg.daily_limit_per_user)
    rate_limiter = RateLimiter(bot_cfg.rate_limit_seconds)

    intents = discord.Intents.default()
    client = discord.Client(intents=intents)
    tree = app_commands.CommandTree(client)

    # --- Slash command --------------------------------------------------
    # The argument is locally renamed to a non-ASCII identifier when
    # configured (kept as `question` in this generic implementation —
    # personas with localized arg names use command_description /
    # question_arg_description for the user-facing text).
    @tree.command(
        name=persona.discord.command_name,
        description=persona.discord.command_description,
        guild=guild_obj,
    )
    @app_commands.describe(question=persona.discord.question_arg_description)
    async def _ask(interaction: discord.Interaction, question: str):
        user_id = interaction.user.id
        user_name = interaction.user.display_name
        channel_id = interaction.channel_id

        if not rate_limiter.ok(user_id):
            await interaction.response.send_message(
                persona.rate_limit_message or "Slow down — too many requests.",
                ephemeral=True,
            )
            return

        allowed, remaining = quota.check_and_increment(user_id)
        if not allowed:
            await interaction.response.send_message(
                _quota_exhausted_text(persona),
                ephemeral=True,
            )
            logger.info(
                "daily quota exhausted user=%s limit=%d", user_id, quota.limit,
            )
            return

        # Discord has a 3s timeout on interactions; defer before LLM call.
        await interaction.response.defer()

        # Once we've deferred, ANY exception escaping this handler will
        # leave the user hanging on "is thinking" until the followup token
        # expires (~15 minutes). Wrap everything post-defer in a broad
        # try/except so the user always gets a message — at minimum, the
        # persona's fallback line. Narrower try/excepts inside log the
        # specific failure points.
        try:
            history_store = ChannelHistory(channel_id) if channel_id else None
            try:
                recent_history = (
                    history_store.recent(bot_cfg.history_turns_in_prompt)
                    if history_store else []
                )
            except Exception:
                logger.exception(
                    "failed to load channel history — continuing without it",
                )
                recent_history = []

            logger.info(
                "request user=%s remaining=%d history=%d question=%r",
                user_id, remaining, len(recent_history), question[:200],
            )

            response_text = await asyncio.to_thread(
                engine.respond,
                persona.id,
                question,
                recent_history,
                user_name,
            )

            logger.info(
                "response user=%s length=%d", user_id, len(response_text),
            )

            if history_store is not None:
                try:
                    history_store.append(Exchange.now(
                        user_id=user_id,
                        user_name=user_name,
                        persona=persona.id,
                        question=question,
                        response=response_text,
                    ))
                except Exception:
                    logger.exception("failed to persist exchange to history")

            for chunk in _split_for_discord(response_text, bot_cfg.max_discord_message_chars):
                await interaction.followup.send(chunk)

        except Exception:
            logger.exception("command handler failed after defer")
            try:
                await interaction.followup.send(_fallback_text(persona))
            except Exception:
                logger.exception("fallback followup also failed")

    # --- Lifecycle -------------------------------------------------------

    @client.event
    async def on_ready():
        logger.info("logged in as %s", client.user)
        if guild_obj is not None:
            synced = await tree.sync(guild=guild_obj)
            logger.info(
                "synced %d command(s) to guild %s (instant)",
                len(synced), guild_id_raw,
            )
        else:
            synced = await tree.sync()
            logger.info(
                "synced %d command(s) globally — may take up to 1 hour to "
                "appear in clients", len(synced),
            )

    return client


# --- Runners ---------------------------------------------------------------

def _resolve_token(persona: Persona) -> str | None:
    """Look up the persona's bot token from env. Returns None if unset."""
    token = os.getenv(persona.discord.token_env)
    return token if token else None


def run_single(persona_id: str) -> None:
    """Run one persona in a fresh asyncio loop. Blocks until shutdown."""
    persona = personas_module.load(persona_id)
    token = _resolve_token(persona)
    if not token:
        raise RuntimeError(
            f"{persona.discord.token_env} not set in .env — see README for "
            f"Discord setup steps."
        )
    client = build_client(persona)
    # log_handler=None: we configure logging ourselves in main().
    client.run(token, log_handler=None)


async def run_many(persona_ids: list[str]) -> None:
    """Run multiple personas concurrently in one asyncio event loop."""
    logger = logging.getLogger("bot")
    starts: list = []

    for pid in persona_ids:
        try:
            persona = personas_module.load(pid)
        except Exception:
            logger.exception("skipping %s — failed to load persona", pid)
            continue

        token = _resolve_token(persona)
        if not token:
            logger.warning(
                "skipping %s — %s not set in .env",
                pid, persona.discord.token_env,
            )
            continue

        try:
            client = build_client(persona)
        except Exception:
            logger.exception("skipping %s — failed to build client", pid)
            continue

        starts.append(client.start(token))

    if not starts:
        raise RuntimeError(
            "no personas could be started — check tokens in .env and "
            "persona.yaml files under personas/."
        )

    logger.info("starting %d persona client(s)", len(starts))
    await asyncio.gather(*starts)


# --- Entry point ----------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--persona", help="persona id (folder under personas/) to run",
    )
    target.add_argument(
        "--all", action="store_true",
        help="run every persona under personas/ except _example/",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="logging verbosity (default: INFO)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.all:
        ids = personas_module.discover()
        if not ids:
            print(
                "ERROR: no personas found under personas/ (excluding _example/).",
                file=sys.stderr,
            )
            return 2
        asyncio.run(run_many(ids))
    else:
        run_single(args.persona)
    return 0


if __name__ == "__main__":
    sys.exit(main())
