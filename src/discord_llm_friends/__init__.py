"""discord-llm-friends — persona-driven Discord bots over RAG + few-shot.

Public entry points:
    discord_llm_friends.bot         — Discord runtime (--persona | --all)
    discord_llm_friends.pipeline.cleanup  — raw.json → cleaned.json
    discord_llm_friends.pipeline.embed    — cleaned.json → ChromaDB
    discord_llm_friends.dev.cli           — ad-hoc Q&A from the CLI
    discord_llm_friends.dev.query_check   — preview retrieval results
"""
