"""Shared infrastructure used by every stage — no pipeline logic here.

  schema.py  the pydantic models of configs/*.yaml (`Config`) and `REPO_ROOT`
  config.py  load_config / save_config (guarded), secrets from .env, collection_name
  store.py   JSONL / Parquet persistence, resumable appends, run folders
  llm.py     one OpenAI-compatible chat client for every LLM role (Ollama or OpenAI)

A change here affects every stage: run the whole test suite after editing it.
"""
