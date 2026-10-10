"""Chunkers under evaluation: character, sentence, recursive-chapter (each with/without overlap).

All chunkers implement `base.Chunker` and operate on `CanonicalDoc.text`, returning chunks with
character offsets into that text.

Reads:  data/parsed/<corpus>/<doc_id>.json
Writes: data/chunks/<corpus>/<method>_<overlap|no_overlap>.jsonl
        data/chunks/<corpus>/_size_stats.json   (size distribution per variant, for the report)
"""
from __future__ import annotations

import logging

from chunkeval.core.config import Secrets
from chunkeval.core.schema import Config
from chunkeval.core.store import write_json, write_jsonl
from chunkeval.corpus.chunking.base import Chunker, TokenCounter, load_chunks, load_counter, size_stats
from chunkeval.corpus.chunking.character import CharacterChunker
from chunkeval.corpus.chunking.recursive import RecursiveChunker
from chunkeval.corpus.chunking.sentence import SentenceChunker
from chunkeval.corpus.parsing import load_doc

log = logging.getLogger(__name__)

CHUNKERS: dict[str, type[Chunker]] = {
    "character": CharacterChunker,
    "sentence": SentenceChunker,
    "recursive": RecursiveChunker,
}


def build_chunker(method: str, overlap: bool, cfg: Config, counter: TokenCounter) -> Chunker:
    """One (method, overlap) configuration with the sizes of cfg.chunking. A CharacterChunker
    still needs `calibrate` on the corpus before use."""
    c = cfg.chunking
    extra = {"min_section_tokens": c.min_section_tokens} if method == "recursive" else {}
    return CHUNKERS[method](counter, c.target_tokens, overlap, c.overlap_ratio, c.max_tokens, **extra)


def run(cfg: Config, secrets: Secrets, corpora: list[str], limit: int | None = None) -> None:
    """CLI entry (`chunkeval chunk`): chunk the parsed documents of each corpus in `corpora` with
    every (method, overlap) of the config. `limit`: only the first N documents per corpus.

    Docs are sorted by name, so `limit` always picks the same ones. Each variant goes to its own
    JSONL (atomic write, so a crash never leaves a half-written file that `embed` would read);
    the character window is calibrated on the docs being chunked, so a `--limit` smoke test gets a
    slightly different window than the full corpus.
    """
    counter = load_counter(cfg.chunking.size_tokenizer)
    for corpus in corpora:
        parsed_dir = cfg.paths.data_dir / "parsed" / corpus
        paths = sorted(parsed_dir.glob("*.json"))[:limit]
        if not paths:
            raise FileNotFoundError(f"corpus '{corpus}': no parsed docs in {parsed_dir}, run `chunkeval parse` first")
        docs = [load_doc(p) for p in paths]
        out_dir = cfg.paths.data_dir / "chunks" / corpus

        stats: dict[str, dict[str, float]] = {}
        for method in cfg.chunking.methods:
            for overlap in cfg.chunking.overlap:
                chunker = build_chunker(method, overlap, cfg, counter)
                if isinstance(chunker, CharacterChunker):
                    log.info("%s: character window calibrated to %d chars", corpus, chunker.calibrate(docs))
                chunks = [c for doc in docs for c in chunker.chunk(doc)]
                write_jsonl(out_dir / f"{chunker.variant}.jsonl", chunks)
                stats[chunker.variant] = size_stats(chunks)
                if isinstance(chunker, CharacterChunker):
                    stats[chunker.variant]["window_chars"] = chunker.window_chars
                s = stats[chunker.variant]
                log.info("%s %-22s %5d chunks, tokens mean %5.1f p5 %3.0f p95 %3.0f max %3d, words %5.1f, overlap %4.1f",
                         corpus, chunker.variant, s["count"], s["mean"], s["p5"], s["p95"], s["max"],
                         s["words_mean"], s["overlap_tokens_mean"])
        write_json(out_dir / "_size_stats.json", stats)
        log.info("%s: %d docs chunked into %s", corpus, len(docs), out_dir)
