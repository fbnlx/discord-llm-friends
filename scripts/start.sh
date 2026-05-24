#!/usr/bin/env bash
# Production-mode launcher: one process, all personas, one log stream.
#
# Personas are auto-discovered from personas/<id>/ (excluding _example/).
# Each persona's log lines are prefixed with `bot.<persona-id>` via the
# child-logger naming in src/discord_llm_friends/bot.py.
#
# Run from anywhere; the script `cd`s to the repo root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

exec uv run python -m discord_llm_friends.bot --all "$@"
