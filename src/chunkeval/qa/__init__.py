"""Stages 4–5: the ground-truth questions, independent of any chunker.

  generation.py  4   Q-A pairs from paragraph groups, evidence quotes located as char spans
  filtering.py   5   deterministic checks + LLM critic + dedup + final sample
"""
