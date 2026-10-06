"""Stages 6–8: scoring every chunking configuration.

  retrieval.py          6  top-k retrieval per question, collection and embedder
  retrieval_metrics.py  6  span-based retrieval metrics (deterministic, no API cost)
  answering.py          7  RAG answers from the retrieved chunks (fixed generator)
  ragas_eval.py         8  Ragas scores per question with the OpenAI judge
"""
