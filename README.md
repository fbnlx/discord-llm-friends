# discord-llm-friends

A template for running Discord bots that impersonate personas through
retrieval-augmented few-shot prompting. Define one persona or many; each
gets its own slash command and answers in that persona's voice.

See [CONTEXT.md](./CONTEXT.md) for the domain vocabulary used throughout
this README and the code, and [docs/adr/](./docs/adr/) for the
architectural decisions worth their own writeup.

## Stack

- Python 3.12+ managed by [uv](https://github.com/astral-sh/uv)
- LLMs: Gemini (default), Claude, or OpenAI — selectable in `config.yaml`
- Embeddings: OpenAI `text-embedding-3-small`
- Vector store: ChromaDB, local, file-backed at `./chroma/`
- Discord: `discord.py`

## Quickstart (with the bundled example persona)

The repo ships with a fictional persona, `_example` (Marv, a 1990s
arcade-cabinet repair tech), so you can smoke-test the entire pipeline
before defining your own.

```sh
# 1. Install dependencies
uv sync

# 2. Configure secrets
cp .env.example .env
# Edit .env: fill in OPENAI_API_KEY (required for embeddings) and the
# provider key for whichever LLM you picked in config.yaml (default:
# GOOGLE_API_KEY for Gemini). Add DISCORD_TOKEN_EXAMPLE if you want to
# run Marv in a real Discord server.

# 3. Run the pipeline against Marv
uv run python -m discord_llm_friends.pipeline.cleanup --persona _example
uv run python -m discord_llm_friends.pipeline.embed   --persona _example

# 4. Test from the terminal (no Discord needed)
uv run python -m discord_llm_friends.dev.cli --persona _example "Why is the slam switch always the answer?"

# 5. Run as a Discord bot
uv run python -m discord_llm_friends.bot --persona _example
```

## Adding your own persona

```sh
# 1. Copy the example folder
cp -r personas/_example personas/your-id
# personas/<your-id>/ is gitignored automatically.

# 2. Edit personas/your-id/persona.yaml and description.md.
#    See personas/_example/ for every field with comments.

# 3. Drop your source comments into personas/your-id/raw.json — a JSON
#    array of strings, one entry per comment. Format example:
#    ["first comment", "second comment", "..."]

# 4. Clean the raw corpus
uv run python -m discord_llm_friends.pipeline.cleanup --persona your-id

# 5. Open personas/your-id/cleaned.json and hand-pick 40-60 entries that
#    cover the persona's voice. Save them as
#    personas/your-id/style_anchors.json (same JSON-array-of-strings format).

# 6. Embed the cleaned corpus into ChromaDB
uv run python -m discord_llm_friends.pipeline.embed --persona your-id

# 7. Create a Discord application at
#    https://discord.com/developers/applications, generate a bot token,
#    and add it to .env as DISCORD_TOKEN_<UPPER_ID> (e.g. DISCORD_TOKEN_YOUR_ID).

# 8. Test, then deploy
uv run python -m discord_llm_friends.dev.cli --persona your-id "Test question?"
uv run python -m discord_llm_friends.bot --persona your-id
```

## Running all personas at once (production mode)

`--all` runs every persona under `personas/` (excluding `_example/`) in
one process, sharing a single asyncio event loop, ChromaDB connection,
and Python interpreter. This is the mode designed for low-memory hosts
(e.g. a Google Cloud e2-micro with 1 GB RAM).

```sh
uv run python -m discord_llm_friends.bot --all
# or
scripts/start.sh
```

Personas with a missing `DISCORD_TOKEN_<UPPER_ID>` are skipped with a
warning rather than killing the whole process. Same for personas with
malformed `persona.yaml`.

## Configuration

| File | Purpose | Tracked in git? |
|---|---|---|
| `config.yaml` | Project-wide tunables (LLM provider, retrieval count, rate limits, history retention). Layered defaults — only override what you want to change. | yes |
| `.env`        | Secrets (API keys, Discord tokens). Also accepts env-var overrides for `LLM_PROVIDER`, `DAILY_LIMIT`, `HISTORY_TURNS`. | no |
| `personas/<id>/persona.yaml` | Per-persona config: name, language, tics, fallback/quota messages, Discord wiring, cleanup overrides. | only `_example/` |
| `personas/<id>/description.md` | Long-form prose description appended into the system prompt. | only `_example/` |

Open `personas/_example/persona.yaml` for an annotated reference. The
`config.yaml` shipped at the repo root contains commented defaults for
every tunable.

## Layout

```
config.yaml                        # project-wide tunables (committed)
.env / .env.example                # secrets only (.env gitignored)
personas/
  _example/                        # fictional Marv — format reference
  <your-id>/                       # your real personas (gitignored)
state/                             # runtime state (gitignored)
  rate_limits/<persona-id>.json    # per-user daily quota counts
  history/<channel-id>.json        # per-channel conversation log
chroma/                            # ChromaDB persistence (gitignored)
src/discord_llm_friends/
  bot.py                           # Discord runtime
  engine.py                        # retrieval + LLM dispatch
  history.py                       # channel-conversation persistence
  config.py                        # config loader (config.yaml + .env)
  personas.py                      # persona loader
  pipeline/
    cleanup.py                     # raw.json → cleaned.json
    embed.py                       # cleaned.json → ChromaDB
  dev/
    cli.py                         # ad-hoc Q&A
    query_check.py                 # preview retrieval results
scripts/
  start.sh                         # one-liner wrapper for `bot --all`
docs/adr/                          # architectural decision records
```

## Architecture in one paragraph

A persona's voice comes from two ingredients in the LLM system prompt:
the **tics + description** (general voice rules) and the **style anchors**
(40-60 verbatim examples). On each query, we additionally retrieve the
top-K most similar entries from the persona's full cleaned corpus via
ChromaDB and inject them as "things this persona has said about similar
topics". Recent in-channel exchanges from other personas are also
included so personas can react to each other. See `engine.py` for the
prompt assembly.

## Deploying to a Linux VM (e.g. Google Cloud e2-micro)

The repo ships with a sample systemd unit at
`deploy/discord-llm-friends.service`. A clean-VM deploy looks like:

1. **Provision the VM.** An e2-micro is enough — 1 GB RAM is the binding
   constraint, which is why the runtime is single-process (see ADR-0003).

2. **Install system deps + uv:**
   ```sh
   sudo apt update && sudo apt install -y git python3 python3-venv curl
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Create an unprivileged user and clone the repo:**
   ```sh
   sudo useradd --system --create-home --home-dir /opt/discord-llm-friends \
     --shell /usr/sbin/nologin discord-llm-friends
   sudo -u discord-llm-friends git clone <YOUR_REPO_URL> /opt/discord-llm-friends
   cd /opt/discord-llm-friends
   sudo -u discord-llm-friends /home/discord-llm-friends/.local/bin/uv sync
   ```

4. **Get your personas onto the host.** Either define them directly on
   the VM, or `rsync` your `personas/<id>/` folders from your laptop.
   Run cleanup + embed once per persona to populate ChromaDB:
   ```sh
   sudo -u discord-llm-friends uv run python -m discord_llm_friends.pipeline.cleanup --persona <id>
   sudo -u discord-llm-friends uv run python -m discord_llm_friends.pipeline.embed   --persona <id>
   ```

5. **Create the `.env` file with secrets, locked down:**
   ```sh
   sudo -u discord-llm-friends cp .env.example .env
   sudo -u discord-llm-friends nano .env
   sudo chmod 600 .env
   ```

6. **Install the systemd unit** after substituting the paths in it for
   your host (User, WorkingDirectory, ExecStart):
   ```sh
   sudo cp deploy/discord-llm-friends.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now discord-llm-friends.service
   sudo systemctl status discord-llm-friends
   ```

7. **Tail logs:**
   ```sh
   journalctl -u discord-llm-friends -f
   # filter to one persona:
   journalctl -u discord-llm-friends -f | grep 'bot\.<persona_id>'
   ```

Notes:
- Traffic is purely outbound HTTPS (Discord + LLM provider + OpenAI for
  embeddings). No inbound ports need opening — keep the VM's firewall
  locked down.
- ChromaDB persistence in `chroma/` survives restarts; you don't need to
  re-run `pipeline.embed` unless `cleaned.json` changes.

## Development

```sh
# Inspect what the LLM would see, without spending tokens
PERSONA_DRY_RUN=1 uv run python -m discord_llm_friends.dev.cli --persona _example "any question"

# Print the assembled prompt to stderr but still call the LLM
PERSONA_DEBUG_PROMPT=1 uv run python -m discord_llm_friends.dev.cli --persona _example "any question"

# Preview retrieval results without invoking the LLM
uv run python -m discord_llm_friends.dev.query_check --persona _example --query "your query"

# Force fast slash-command sync to one test server during dev
# (set DISCORD_GUILD_ID in .env; global sync can take up to an hour)
```
