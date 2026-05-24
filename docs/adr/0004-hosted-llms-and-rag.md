# Hosted LLMs + RAG over local fine-tuning

The persona engine uses hosted frontier LLMs (Gemini / Claude / OpenAI)
with few-shot style anchors and retrieval-augmented context. Local model
fine-tuning was evaluated and rejected.

## Considered Options

- **Local open models fine-tuned per persona** (Llama 3.1 8B, Qwen 2.5
  7B, Aya Expanse 8B, Gemma 3 12B). Rejected: tested local open models
  have notably weaker non-English performance, and the project assumes
  personas may be defined in any language. Gemma 3 12B is borderline at
  best for non-English voices.
- **Hosted frontier models with few-shot + RAG** (chosen). Hosted
  Gemini / Claude / GPT have excellent multilingual performance. The
  typical corpus shape (a few thousand short forum comments per
  persona, no thread metadata) is better suited to style mimicry from
  examples than to fine-tuning.

## Consequences

- Per-query cost is non-zero (API tokens) but bounded by the daily
  per-user quota in `config.yaml`. Embedding cost is one-time per
  corpus.
- The system is portable across providers via the `llm.provider` switch
  in `config.yaml` — operators can swap without code changes.
- Local-only deployment is not supported. The runtime requires outbound
  HTTPS to the chosen LLM provider and to OpenAI for embeddings.
