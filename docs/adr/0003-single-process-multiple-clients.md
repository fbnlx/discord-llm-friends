# Single process, multiple Discord clients

The production runtime is one Python process running N `discord.Client`
instances in one asyncio event loop via `asyncio.gather(*starts)`. Triggered
via `python -m discord_llm_friends.bot --all`. Single-persona invocation
`--persona <id>` remains available for debugging.

The driving constraint is the deployment target: a Google Cloud e2-micro
(1 GB RAM). One process per persona would load discord.py, chromadb,
openai, and google-genai imports three times over and run three
ChromaDB clients — too tight at 1 GB.

## Considered Options

- **One process per persona** (status quo until now, with
  `scripts/run-all-bots.sh` backgrounding N PIDs). Rejected: ~3× memory
  cost on the e2-micro, multiple systemd units, log multiplexing
  becomes the operator's problem.
- **Hybrid** (one process per persona behind a process supervisor like
  supervisord). Rejected: more moving parts than the e2-micro setup
  warrants.

## Consequences

- Failure isolation between personas is now process-internal. Every
  command handler is wrapped in a `try/except` that posts the persona's
  in-voice fallback message on LLM failure (`fallback_messages` in
  `persona.yaml`). A genuine Python-level crash still takes the whole
  process down.
- Personas are auto-discovered from `personas/<id>/` folders at boot
  (excluding `_example/` unless explicitly named). Malformed persona
  configs are logged and skipped, not fatal.
- Log lines are prefixed with the persona id so a single log stream is
  still parseable per-persona.
