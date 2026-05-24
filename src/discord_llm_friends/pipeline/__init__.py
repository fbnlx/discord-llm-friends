"""One-shot pipeline tools — run per persona during onboarding.

  pipeline.cleanup  — personas/<id>/raw.json → personas/<id>/cleaned.json
  pipeline.embed    — personas/<id>/cleaned.json → ChromaDB collection
"""
