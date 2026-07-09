"""Ad-hoc CLI for testing personas without spinning up Discord.

Run as:
    uv run python -m discord_llm_friends.dev.cli --persona <id> "your question"

Honors PERSONA_DEBUG_PROMPT and PERSONA_DRY_RUN from the environment.

--asker simulates the Discord display name of the asker (exercises the
persona's cast map). --canonize additionally runs the Canonizer on the
response, synchronously, and prints its summary — the Discord-free way to
test canon growth. Off by default so casual CLI poking never writes canon;
refused under PERSONA_DRY_RUN (there is no real response to canonize).
"""

from __future__ import annotations

import argparse
import sys

from discord_llm_friends import canon, engine
from discord_llm_friends import personas as personas_module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--persona", required=True, help="persona id")
    parser.add_argument(
        "--asker", help="simulated Discord display name of the asker",
    )
    parser.add_argument(
        "--canonize", action="store_true",
        help="run the Canonizer on the response (writes canon; default off)",
    )
    parser.add_argument("question", help="the question to ask the persona")
    args = parser.parse_args(argv)

    if args.canonize and engine.DRY_RUN:
        print(
            "ERROR: --canonize is meaningless under PERSONA_DRY_RUN=1 — "
            "there is no real response to canonize.",
            file=sys.stderr,
        )
        return 2

    result = engine.respond(args.persona, args.question, asker_name=args.asker)
    print(result.text)

    if args.canonize:
        persona = personas_module.load(args.persona)
        if not (persona.canon.enabled and persona.canon.extraction):
            print(
                "[canonize] canon disabled for this persona — nothing to do",
                file=sys.stderr,
            )
            return 0
        stored = canon.canonize_exchange(
            persona, args.question, result.text, result.canon_facts,
            asker_name=args.asker,
        )
        print(f"[canonize] stored {stored} new fact(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
