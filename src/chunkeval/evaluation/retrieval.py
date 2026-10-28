"""Stage 6a — retrieve the k_max best chunks for every question, collection and retriever.

Retrievers:
  * every embedder of `--embedder` (default retrieval.embedders): queries its own NAMED vector
    (`target_vector=`) in the 12 collections. Both embedders are 1024-dim, so a query encoded with
    the wrong model would pass any dimension check: each retriever is bound to one embedder name,
    which is both the query encoder and the target vector, and nothing else is ever passed around.
    `retrieval.mode: hybrid` adds BM25 (the follow-up experiment); the BM25 side is the same for
    both embedders, only the dense side changes.
  * two baselines, per (corpus, method, overlap), computed from the chunk files with no Weaviate
    and no embedder, scored like any other retriever by retrieval_metrics.py:
      - oracle: the chunks of the question's own document that cover the evidence, best first
        (`oracle_hits`). The ceiling THAT chunker allows: it separates "the chunker can't" from
        "the retriever didn't". It stops once the evidence is covered, so it has fewer than k_max
        hits, and its context_tokens is what the chunker needs to hand over all the evidence.
      - random: k_max chunks of the corpus drawn with a seed per question and variant (the floor).

Each question gets the k_max best chunks once; k=5 and the budget-matched view are cut from that
list by retrieval_metrics.py. Questions of corpus X are only asked against corpus-X collections
(each corpus is a separate index, as in the previous round).

Query vectors are computed ONCE per (embedder, corpus) with embedding.Embedder.encode_queries (the
model's query convention) and cached; this module only sends vectors to Weaviate. So retrieval
needs the GPU only when a question or a query setting changed (and stops while Ollama holds it).

Checks before anything is retrieved, so the span metrics never compare offsets into different texts:
  * the documents of the questions are the ones gen-qa saw (filtering.load_docs, text hashes);
  * every chunk of those documents is `doc.text[start:end]`, and every one of them has chunks;
  * the collection was built from exactly the current chunk file (indexing provenance, object count).

Re-runs: each retrieval file has a provenance file next to it: the chunk file's hash, a hash of the
questions (ids, text, evidence spans), k_max, and for an embedder its query settings (model,
revision, query prompt) and the mode, for the random baseline the seed. An up-to-date file is
skipped; a stale one is retrieved again (retrieval is free and deterministic). A different mode
is refused instead: it would replace one experiment's results with another's under the same run_id.

Reads:  data/qa/filtered.jsonl (+ raw.provenance.json), data/parsed/, data/chunks/, Weaviate v2 collections
Writes: runs/<run_id>/retrieval/<retriever>/<corpus>_<method>_<ovl>.jsonl   one line per question:
            {"question_id", "hits": [Hit without text, in rank order]}
        runs/<run_id>/retrieval/<retriever>/<corpus>_<method>_<ovl>.provenance.json
        data/embeddings/queries/<embedder>/<corpus>.npy (+ .json)   the cached query vectors
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import zlib
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from chunkeval.core.config import Secrets, collection_name
from chunkeval.core.schema import Config
from chunkeval.core.store import read_jsonl, write_json, write_jsonl
from chunkeval.corpus import indexing
from chunkeval.corpus.chunking.base import Chunk, load_chunks
from chunkeval.corpus.embedding import Embedder, EmbedderSpec, check_gpu_free, check_vectors, load_specs, save_npy
from chunkeval.evaluation.retrieval_metrics import evidence_intervals, merge_intervals, overlap_len
from chunkeval.qa.filtering import filter_paths, load_docs
from chunkeval.qa.generation import QAPair, load_pairs, qa_paths

if TYPE_CHECKING:
    import weaviate
    from chunkeval.corpus.parsing import CanonicalDoc

log = logging.getLogger(__name__)

# Bumped when the code changes what a retrieval file holds, so older files count as stale.
RETRIEVAL_VERSION = 1
# Embedder-independent retrievers, written once per (corpus, variant); also the `embedder` value of
# their rows in the metrics table.
BASELINES = ("oracle", "random")
# The chunk properties a hit needs; the text stays in the chunk file (see `Hit`).
HIT_PROPERTIES = ["chunk_id", "doc_id", "start", "end", "n_tokens"]


@dataclass
class Hit:
    """One retrieved chunk. `score`: cosine similarity (dense), Weaviate's fused score (hybrid),
    new evidence chars (oracle) or 0 (random).

    `text` is not stored in the retrieval files: at k_max=20 it would repeat each corpus's text
    dozens of times over. The chunk file holds it, by chunk_id, and the provenance guarantees the
    two match; whoever needs it (answering) fills it from there."""
    rank: int
    chunk_id: str
    doc_id: str
    start: int
    end: int
    n_tokens: int
    score: float
    text: str = ""


def hit_from_chunk(chunk: Chunk, rank: int, score: float) -> Hit:
    return Hit(rank, chunk.chunk_id, chunk.doc_id, chunk.start, chunk.end, chunk.n_tokens, float(score))


# --- inputs ---------------------------------------------------------------------------------------

def variants(cfg: Config) -> list[tuple[str, bool, str]]:
    """(method, overlap, variant name) of every configured chunk file, in config order."""
    return [(m, o, f"{m}_{'overlap' if o else 'no_overlap'}") for m in cfg.chunking.methods for o in cfg.chunking.overlap]


def load_questions(cfg: Config) -> dict[str, list[QAPair]]:
    """The final question set (filtered.jsonl) by corpus, in file order (document order)."""
    path = filter_paths(cfg)["filtered"]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `chunkeval filter-qa` first")
    questions: dict[str, list[QAPair]] = {}
    for qa in load_pairs(path):
        questions.setdefault(qa.corpus, []).append(qa)
    return questions


def questions_sha256(pairs: list[QAPair]) -> str:
    """What a retrieval file or a query-vector cache depends on: the questions in order, and the
    evidence spans the metrics compare the hits with."""
    key = [[qa.id, qa.doc_id, qa.question, [[e.start, e.end] for e in qa.evidence]] for qa in pairs]
    return hashlib.sha256(json.dumps(key, ensure_ascii=False).encode("utf-8")).hexdigest()


def variant_chunks(cfg: Config, corpus: str, variant: str) -> tuple[list[Chunk], dict[str, Any]]:
    """The chunks of one chunk file and the provenance `index` stores for a collection built from
    it (indexing.provenance, no --limit)."""
    path = cfg.paths.data_dir / "chunks" / corpus / f"{variant}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing, run `chunkeval chunk` first")
    chunks = load_chunks(path)
    return chunks, indexing.provenance(cfg, corpus, variant, chunks, None)


def check_chunk_offsets(chunks: list[Chunk], docs: dict[str, "CanonicalDoc"], where: str) -> None:
    """Every document in `docs` (those with questions, by doc_id) must have chunks, cut from exactly
    the text its evidence offsets point into. Otherwise the span metrics would compare offsets into
    two different texts, or score 0 for a document that is simply not indexed."""
    redo = "run `chunkeval chunk`, `chunkeval embed` and `chunkeval index --recreate`"
    missing = sorted(set(docs) - {c.doc_id for c in chunks})
    if missing:
        raise RuntimeError(f"{where}: no chunks of {', '.join(missing)}, which have questions; {redo}")
    bad = [c.chunk_id for c in chunks if c.doc_id in docs and docs[c.doc_id].text[c.start:c.end] != c.text]
    if bad:
        raise RuntimeError(f"{where}: {len(bad)} chunk(s) don't match the parsed text of their document, "
                           f"e.g. {bad[0]} (re-parsed after `chunk`); {redo}")


# --- query vectors --------------------------------------------------------------------------------

def query_settings(spec: EmbedderSpec) -> dict[str, Any]:
    """Everything that changes a QUERY vector: the document-side settings (same model and revision
    as the chunk vectors) plus the query prompt."""
    return {**spec.settings, "query_prompt": spec.query_prompt}


def query_cache_path(cfg: Config, embedder: str, corpus: str) -> Path:
    """data/embeddings/queries/<embedder>/<corpus>.npy; its metadata is the .json next to it."""
    return cfg.embedding_cache_dir / "queries" / embedder / f"{corpus}.npy"


def cached_query_vectors(cfg: Config, spec: EmbedderSpec, corpus: str, pairs: list[QAPair]) -> np.ndarray | None:
    """The cached query vectors of `pairs` (row i = pairs[i]), or None if the cache was made for
    other questions or with other settings (a new query prompt or model revision)."""
    path = query_cache_path(cfg, spec.name, corpus)
    meta_path = path.with_suffix(".json")
    if not (path.exists() and meta_path.exists()):
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("settings") != query_settings(spec) or meta.get("questions_sha256") != questions_sha256(pairs):
        return None
    vectors = np.load(path)
    check_vectors(vectors, len(pairs), spec.dim, f"{spec.name} {corpus} query cache")
    return vectors


def encode_questions(cfg: Config, secrets: Secrets, embedder: str,
                     questions: dict[str, list[QAPair]]) -> dict[str, np.ndarray]:
    """Query vectors per corpus of `questions` (row i = question i), encoded with `embedder`.

    Read from the cache where it is up to date. Otherwise the model is loaded once, after
    `check_gpu_free`, encodes every stale corpus with `encode_queries` (no instruction for bge-m3,
    the "Instruct: ...\\nQuery:" prompt for Qwen3; never `encode_documents`, which for Qwen3 would
    silently drop the instruction), caches the vectors and is unloaded.
    """
    spec = load_specs(cfg)[embedder]
    vectors: dict[str, np.ndarray] = {}
    for corpus, pairs in questions.items():
        cached = cached_query_vectors(cfg, spec, corpus, pairs)
        if cached is not None:
            vectors[corpus] = cached
    todo = [corpus for corpus in questions if corpus not in vectors]
    if not todo:
        log.info("%s: query vectors cached for %s, model not loaded", embedder, ", ".join(questions))
        return vectors
    check_gpu_free(secrets)
    with Embedder(spec) as model:
        for corpus in todo:
            pairs = questions[corpus]
            vectors[corpus] = model.encode_queries([qa.question for qa in pairs], cfg.embedding.batch_size)
            path = query_cache_path(cfg, embedder, corpus)
            meta_path = path.with_suffix(".json")
            meta_path.unlink(missing_ok=True)            # written last: only ever next to its complete matrix
            save_npy(path, vectors[corpus])
            write_json(meta_path, {"embedder": embedder, "corpus": corpus, "settings": query_settings(spec),
                                   "questions_sha256": questions_sha256(pairs), "n": len(pairs),
                                   "created": datetime.now().isoformat(timespec="seconds"),
                                   "question_ids": [qa.id for qa in pairs]})
            log.info("%s: %d %s questions encoded", embedder, len(pairs), corpus)
    return vectors


# --- retrievers -----------------------------------------------------------------------------------

class WeaviateRetriever:
    """One collection, searched in one named vector (dense, or hybrid with BM25).

    Generalised version of the old WeaviateRagasRetriever (ragas_evaluation.py): it returns the
    chunk's offsets and score, not just its text, and is bound to `target_vector`. The client is
    the caller's (`indexing.connect`, a context manager that closes it).
    """

    def __init__(self, client: "weaviate.WeaviateClient", collection_name: str, target_vector: str, cfg: Config,
                 index_provenance: dict[str, Any] | None = None) -> None:
        """Fails early: a missing collection, an unknown named vector, or (with `index_provenance`,
        what `index` would store for the current chunk file) a collection built from anything else."""
        if not client.collections.exists(collection_name):
            raise RuntimeError(f"{collection_name}: no such collection, run `chunkeval index`")
        self.collection = client.collections.get(collection_name)
        self.name, self.target_vector = collection_name, target_vector
        self.mode, self.alpha = cfg.retrieval.mode, cfg.retrieval.hybrid_alpha
        named = list(self.collection.config.get().vector_config or {})
        if target_vector not in named:
            raise ValueError(f"{collection_name}: no named vector {target_vector!r}, only {named}")
        if index_provenance is not None:
            self.check_index(index_provenance)

    def check_index(self, expected: dict[str, Any]) -> None:
        stored = indexing.stored_provenance(self.collection)
        if stored != expected:
            differs = [key for key in expected if (stored or {}).get(key) != expected[key]]
            raise RuntimeError(f"{self.name}: built from other inputs than the current chunk files "
                               f"({', '.join(differs)} differ); run `chunkeval index --recreate`")
        count = self.collection.aggregate.over_all(total_count=True).total_count
        if count != expected["n"]:
            raise RuntimeError(f"{self.name}: {count} objects for {expected['n']} chunks; "
                               f"run `chunkeval index --recreate`")

    def retrieve(self, question: str, query_vector: np.ndarray, k: int) -> list[Hit]:
        """The k best chunks, best first. Dense: score = 1 - cosine distance. Hybrid: Weaviate's
        fused score; alpha=1 is pure vector, 0 pure BM25 (on `text` only, see indexing.PROPERTIES)."""
        from weaviate.classes.query import MetadataQuery

        vector = query_vector.tolist()
        if self.mode == "dense":
            result = self.collection.query.near_vector(
                near_vector=vector, target_vector=self.target_vector, limit=k,
                return_metadata=MetadataQuery(distance=True), return_properties=HIT_PROPERTIES)
            scores = [1 - o.metadata.distance for o in result.objects]
        else:
            result = self.collection.query.hybrid(
                query=question, vector=vector, target_vector=self.target_vector, alpha=self.alpha, limit=k,
                return_metadata=MetadataQuery(score=True), return_properties=HIT_PROPERTIES)
            scores = [o.metadata.score for o in result.objects]
        return [Hit(rank=rank, score=float(score), **{p: o.properties[p] for p in HIT_PROPERTIES})
                for rank, (o, score) in enumerate(zip(result.objects, scores), start=1)]


def oracle_hits(qa: QAPair, doc_chunks: list[Chunk], k_max: int) -> list[Hit]:
    """The best ranking `doc_chunks` (the chunks of the question's document) allow: greedily the
    chunk adding the most evidence chars not yet covered (ties: fewer tokens, then position),
    until the evidence is covered. Score = the chars it added. Chunks cover the whole text, so the
    last hit always brings span_recall to 1 (within k_max)."""
    evidence = evidence_intervals(qa)
    pool = sorted((c for c in doc_chunks if overlap_len(evidence, [(c.start, c.end)])),
                  key=lambda c: (c.n_tokens, c.start))
    hits: list[Hit] = []
    union: list[tuple[int, int]] = []
    covered = 0
    while pool and len(hits) < k_max:
        gains = [overlap_len(evidence, merge_intervals([*union, (c.start, c.end)])) - covered for c in pool]
        best = max(range(len(pool)), key=lambda i: (gains[i], -i))   # first of the best in pool order
        if gains[best] <= 0:
            break
        chunk = pool.pop(best)
        hits.append(hit_from_chunk(chunk, len(hits) + 1, gains[best]))
        union = merge_intervals([*union, (chunk.start, chunk.end)])
        covered += gains[best]
    return hits


def random_hits(qa: QAPair, chunks: list[Chunk], variant: str, k_max: int, seed: int) -> list[Hit]:
    """k_max chunks of the corpus (all its documents) drawn without replacement, reproducibly per
    (seed, question, variant): the chance level of every metric."""
    rng = np.random.default_rng([seed, zlib.crc32(f"{qa.corpus}|{qa.id}|{variant}".encode())])
    rows = rng.choice(len(chunks), size=min(k_max, len(chunks)), replace=False)
    return [hit_from_chunk(chunks[i], rank, 0.0) for rank, i in enumerate(rows, start=1)]


# --- retrieval files ------------------------------------------------------------------------------

def retrieval_path(cfg: Config, retriever: str, corpus: str, variant: str) -> Path:
    """runs/<run_id>/retrieval/<retriever>/<corpus>_<variant>.jsonl"""
    return cfg.paths.runs_dir / cfg.run_id / "retrieval" / retriever / f"{corpus}_{variant}.jsonl"


def provenance_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.provenance.json")


def file_provenance(cfg: Config, retriever: str, corpus: str, variant: str, chunks_sha256: str,
                    pairs: list[QAPair]) -> dict[str, Any]:
    """What one retrieval file is made from; a file whose stored provenance differs is stale."""
    provenance: dict[str, Any] = {
        "version": RETRIEVAL_VERSION, "retriever": retriever, "corpus": corpus, "variant": variant,
        "chunks_sha256": chunks_sha256, "questions_sha256": questions_sha256(pairs), "k_max": cfg.retrieval.k_max,
    }
    if retriever == "random":
        provenance["seed"] = cfg.seed
    elif retriever not in BASELINES:
        r = cfg.retrieval
        provenance |= {"query": query_settings(load_specs(cfg)[retriever]), "mode": r.mode,
                       "hybrid_alpha": r.hybrid_alpha if r.mode == "hybrid" else None}
    return provenance


def stored_provenance(path: Path) -> dict[str, Any] | None:
    """The provenance stored next to a retrieval file; None if either file is missing."""
    meta = provenance_path(path)
    if not (path.exists() and meta.exists()):
        return None
    return json.loads(meta.read_text(encoding="utf-8"))


def write_hits(path: Path, hits: dict[str, list[Hit]], provenance: dict[str, Any]) -> None:
    """One line per question, in the order of `hits`. The provenance is removed first and written
    last, so it only ever sits next to the complete file it describes."""
    meta = provenance_path(path)
    meta.unlink(missing_ok=True)
    write_jsonl(path, ({"question_id": qid, "hits": [{k: v for k, v in asdict(h).items() if k != "text"}
                                                      for h in ranked]}
                       for qid, ranked in hits.items()))
    write_json(meta, provenance)


def load_hits(path: Path) -> dict[str, list[Hit]]:
    """question_id -> its hits in rank order (without text, see `Hit`)."""
    return {r["question_id"]: [Hit(**h) for h in r["hits"]] for r in read_jsonl(path)}


def check_same_mode(path: Path, stored: dict[str, Any] | None, expected: dict[str, Any]) -> None:
    """Refuse to overwrite an embedder's results retrieved in another mode: that is another
    experiment, which needs its own run_id."""
    if stored is None or stored.get("version") != expected["version"]:
        return
    then, now = (stored.get("mode"), stored.get("hybrid_alpha")), (expected["mode"], expected["hybrid_alpha"])
    if then != now:
        raise RuntimeError(f"{path} was retrieved with mode/hybrid_alpha {then}, the config says {now}: "
                           f"bump run_id to keep both experiments (or delete {path.parent.parent})")


# --- the stage ------------------------------------------------------------------------------------

def run(cfg: Config, secrets: Secrets, corpora: list[str], embedders: list[str]) -> None:
    """CLI entry (first half of `chunkeval retrieve`): for each corpus of `corpora` with questions
    and each (method, overlap), write the k_max hits of every question for the two baselines and
    for each of `embedders` (default retrieval.embedders), skipping files that are up to date.

    Order: the checks and the baselines first (no Weaviate, no GPU); then per embedder (outer, so
    each model is loaded at most once): the query vectors, then every stale collection.
    """
    questions = load_questions(cfg)
    for corpus in corpora:
        if not questions.get(corpus):
            log.warning("%s: no questions in %s, nothing to retrieve", corpus, filter_paths(cfg)["filtered"])
    corpora = [c for c in corpora if questions.get(c)]
    docs = load_docs(cfg, [qa for c in corpora for qa in questions[c]], qa_paths(cfg)["provenance"])
    r = cfg.retrieval
    written = up_to_date = 0
    jobs: list[tuple[str, str, str, bool, dict[str, Any], Path, dict[str, Any]]] = []   # embedder files to retrieve

    for corpus in corpora:
        pairs = questions[corpus]
        corpus_docs = {doc_id: doc for (c, doc_id), doc in docs.items() if c == corpus}
        for method, overlap, variant in variants(cfg):
            chunks, index = variant_chunks(cfg, corpus, variant)
            check_chunk_offsets(chunks, corpus_docs, f"{corpus}/{variant}")
            by_doc: dict[str, list[Chunk]] = {}
            for c in chunks:
                by_doc.setdefault(c.doc_id, []).append(c)
            for retriever in [*BASELINES, *embedders]:
                path = retrieval_path(cfg, retriever, corpus, variant)
                expected = file_provenance(cfg, retriever, corpus, variant, index["chunks_sha256"], pairs)
                stored = stored_provenance(path)
                if stored == expected:
                    up_to_date += 1
                    continue
                if retriever == "oracle":
                    write_hits(path, {qa.id: oracle_hits(qa, by_doc[qa.doc_id], r.k_max) for qa in pairs}, expected)
                elif retriever == "random":
                    write_hits(path, {qa.id: random_hits(qa, chunks, variant, r.k_max, cfg.seed) for qa in pairs},
                               expected)
                else:
                    check_same_mode(path, stored, expected)
                    jobs.append((retriever, corpus, method, overlap, index, path, expected))
                    continue
                written += 1
                log.debug("%s %s/%s: %d questions", retriever, corpus, variant, len(pairs))

    if jobs:
        with indexing.connect(cfg, secrets.weaviate_api_key.get_secret_value()) as client:
            for embedder in embedders:
                mine = [job for job in jobs if job[0] == embedder]
                if not mine:
                    continue
                needed = dict.fromkeys(job[1] for job in mine)          # corpora, in order
                vectors = encode_questions(cfg, secrets, embedder, {c: questions[c] for c in needed})
                start = time.monotonic()
                for _, corpus, method, overlap, index, path, expected in mine:
                    name = collection_name(cfg, corpus, method, overlap)
                    retriever = WeaviateRetriever(client, name, embedder, cfg, index)
                    pairs = questions[corpus]
                    hits = {qa.id: retriever.retrieve(qa.question, vectors[corpus][i], r.k_max)
                            for i, qa in enumerate(pairs)}
                    write_hits(path, hits, expected)
                    written += 1
                    log.debug("%s %s: %d questions", embedder, name, len(pairs))
                log.info("%s: %d collection(s) searched (%s, top %d) in %.1fs", embedder, len(mine), r.mode,
                         r.k_max, time.monotonic() - start)
    log.info("retrieval: %d file(s) written, %d up to date", written, up_to_date)
