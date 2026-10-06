"""chunkeval — chunking evaluation pipeline v2.

Package layout, in pipeline order:
  core/        shared infrastructure: config schema + loading, storage, LLM client
  corpus/      stages 1–3   parsing -> chunking -> embedding -> indexing
  qa/          stages 4–5   generation -> filtering
  evaluation/  stages 6–8   retrieval (+ retrieval_metrics) -> answering -> ragas_eval
  analysis.py  stage 9      report tables
  cli.py       entry point, one subcommand per stage (`uv run chunkeval <command>`)
See README.md for the data flow and the rationale behind each design decision.
"""
