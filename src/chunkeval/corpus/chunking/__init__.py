"""Chunkers under evaluation: character, sentence, recursive-chapter (each with/without overlap).

All chunkers implement `base.Chunker` and operate on `CanonicalDoc.text`, returning chunks with
character offsets into that text.

Reads:  data/parsed/<corpus>/<doc_id>.json
Writes: data/chunks/<corpus>/<method>_<overlap|no_overlap>.jsonl (+ size stats)
"""
from __future__ import annotations

from chunkeval.core.config import Secrets
from chunkeval.core.schema import Config

# TODO(you): once the classes exist, expose a small registry, e.g.
#   CHUNKERS = {"character": CharacterChunker, "sentence": SentenceChunker, "recursive": RecursiveChunker}
# so `run` can build any configuration from the YAML (methods x overlap).


def run(cfg: Config, secrets: Secrets, corpora: list[str], limit: int | None = None) -> None:
    """CLI entry (`chunkeval chunk`): chunk the parsed documents of each corpus in `corpora` with
    every (method, overlap) of the config. `limit`: only the first N documents per corpus.

    TODO(you):
      - Load the parsed docs (`parsing.load_doc` on data/parsed/<corpus>/*.json, sorted by name so
        `limit` always picks the same ones).
      - For each method in cfg.chunking.methods × overlap in cfg.chunking.overlap: build the
        chunker from the registry, chunk every doc, `base.validate_chunks`, then
        `store.write_jsonl` to data/chunks/<corpus>/<method>_<overlap|no_overlap>.jsonl. It is
        atomic, so a crash never leaves a half-written file that `embed` would read.
      - Write `base.size_stats` per config: the size distributions are part of the report.
    """
    raise NotImplementedError
