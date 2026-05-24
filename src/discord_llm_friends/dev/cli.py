"""Ad-hoc CLI for testing personas without spinning up Discord.

Run as:
    uv run python -m discord_llm_friends.dev.cli --persona <id> "your question"

Honors PERSONA_DEBUG_PROMPT and PERSONA_DRY_RUN from the environment.
"""

from __future__ import annotations

import argparse
import sys

from discord_llm_friends import engine


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--persona", required=True, help="persona id")
    parser.add_argument("question", help="the question to ask the persona")
    args = parser.parse_args(argv)
    print(engine.respond(args.persona, args.question))
    return 0


if __name__ == "__main__":
    sys.exit(main())
