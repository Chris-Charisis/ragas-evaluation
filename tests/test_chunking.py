"""Chunker invariants — the evaluation is only valid if these hold for all 3 methods x overlap.

Most tests chunk `tiny_doc` (conftest) with small sizes (target 40, max 80 tokens), so every rule
fires on text small enough to check by hand. `test_parsed_docs` runs the 6 configurations with the
config sizes on the first parsed docs in data/parsed/ (skipped until `chunkeval parse` has run).
`pack` and `overlap_start` are pure functions of unit sizes and are tested on their own.
"""
import dataclasses
import json
from collections import defaultdict

import pytest

from chunkeval.core.schema import REPO_ROOT
from chunkeval.core.store import write_json
from chunkeval.corpus.chunking import CHUNKERS, CharacterChunker, build_chunker, load_chunks, run
from chunkeval.corpus.chunking.base import overlap_start, pack, size_stats, validate_chunks
from chunkeval.corpus.chunking.character import SNAP_CHARS
from chunkeval.corpus.chunking.sentence import block_sentences, sentence_spans
from chunkeval.corpus.parsing import load_doc

METHODS = ["character", "sentence", "recursive"]
CONFIGS = [(m, o) for m in METHODS for o in (False, True)]
CONFIG_IDS = [f"{m}-{'overlap' if o else 'no_overlap'}" for m, o in CONFIGS]


def configs(method=None, overlap=None):
    """`chunked` restricted to some configurations: @pytest.mark.parametrize("chunked", ..., indirect=True)."""
    picked = [(m, o) for m, o in CONFIGS if method in (None, m) and overlap in (None, o)]
    return pytest.mark.parametrize("chunked", picked, indirect=True,
                                   ids=[f"{m}-{'overlap' if o else 'no_overlap'}" for m, o in picked])
SMALL = {"target_tokens": 40, "max_tokens": 80, "overlap_ratio": 0.25}   # overlap budget 10, cap 20


def make_chunker(method, overlap, counter, docs):
    extra = {"min_section_tokens": 15} if method == "recursive" else {}
    chunker = CHUNKERS[method](counter, overlap=overlap, **SMALL, **extra)
    if method == "character":
        chunker.calibrate(docs)
    return chunker


@pytest.fixture(params=CONFIGS, ids=CONFIG_IDS)
def chunked(request, counter, tiny_doc):
    """(chunker, chunks of tiny_doc) for each of the 6 configurations."""
    method, overlap = request.param
    chunker = make_chunker(method, overlap, counter, [tiny_doc])
    return chunker, chunker.chunk(tiny_doc)


def run_on_block(doc):
    """Span of tiny_doc's ~400-token run-on sentence, which every method splits into token windows."""
    return next((b.start, b.end) for b in doc.blocks if doc.text[b.start:b.end].startswith("Stations"))


# --- invariants, on all 6 configurations --------------------------------------------------------

def test_chunk_text_matches_offsets(chunked, tiny_doc):
    """For every chunk: chunk.text == doc.text[chunk.start:chunk.end], trimmed, and section_path
    is the path of the innermost section containing its start."""
    chunker, chunks = chunked
    for i, c in enumerate(chunks):
        assert c.text == tiny_doc.text[c.start:c.end] == c.text.strip()
        assert c.chunk_id == f"tiny:{chunker.variant}:{i}"
        innermost = max((s for s in tiny_doc.sections if s.start <= c.start < s.end), key=lambda s: s.start)
        assert c.section_path == innermost.path


def test_chunks_cover_document(chunked, tiny_doc):
    """The union of chunk spans covers all non-whitespace characters of doc.text."""
    _, chunks = chunked
    covered = [False] * len(tiny_doc.text)
    for c in chunks:
        covered[c.start:c.end] = [True] * (c.end - c.start)
    assert all(ok or ch.isspace() for ok, ch in zip(covered, tiny_doc.text))


def test_no_chunk_exceeds_max_tokens(chunked, counter):
    """n_tokens (bge-m3 tokenizer) <= max_tokens — the experimental cap that keeps size
    distributions comparable (bge-m3 itself would accept up to 8192). tiny_doc's run-on sentence
    (~400 tokens) is 5x the cap, so the token-window fallback must have split it."""
    _, chunks = chunked
    assert [c.n_tokens for c in chunks] == counter.count_many([c.text for c in chunks])
    assert max(c.n_tokens for c in chunks) <= SMALL["max_tokens"]


@configs(overlap=False)
def test_no_overlap_means_disjoint(chunked):
    """With overlap=False, consecutive spans do not overlap."""
    _, chunks = chunked
    assert all(a.end <= b.start for a, b in zip(chunks, chunks[1:]))
    assert all(c.meta["overlap_tokens"] == 0 for c in chunks)


@configs(overlap=True)
def test_overlap_within_budget(chunked, tiny_doc):
    """With overlap=True: the character chunker repeats ~overlap_ratio of a window at EVERY
    boundary (only snapping to whitespace moves it); the sentence methods repeat whole sentences,
    never more than half the target (`overlap_start`'s cap)."""
    chunker, chunks = chunked
    shared = [a.end - b.start for a, b in zip(chunks, chunks[1:])]
    tokens = [c.meta["overlap_tokens"] for c in chunks[1:]]
    if chunker.method == "character":
        expected = round(chunker.window_chars * chunker.overlap_ratio)
        snap = min(SNAP_CHARS, chunker.window_chars // 4)
        assert all(abs(n - expected) <= snap + 2 for n in shared)    # + a "\n\n" block separator
    else:
        assert any(n > 0 for n in tokens)
        assert all(n <= SMALL["target_tokens"] / 2 for n in tokens)
        # the overlap is whole sentences, so a chunk that has one starts on a sentence
        starts = {a for b in tiny_doc.blocks for a, _ in block_sentences(tiny_doc, b)}
        assert all(c.start in starts for c, n in zip(chunks[1:], tokens) if n)


def test_sentence_boundaries_respected(counter, tiny_doc):
    """Sentence/recursive chunks start and end on sentence boundaries — with the tricky
    abbreviations ("e.g.", "et al.", "Fig. 3", "3.5 t/ha") that broke the old regex splitter,
    and the ones pysbd itself gets wrong ("ca.", "Eq.", "approx.")."""
    text = ("Rice is grown in ca. 40 countries, e.g. in Côte d'Ivoire. Smith et al. (2020) reported "
            "3.5 t/ha for O. sativa. Fig. 3 shows the trend. See Eq. (1) and approx. 2 t/ha for "
            "irrigated vs. rainfed rice.")
    assert [text[a - 100:b - 100] for a, b in sentence_spans(text, base_offset=100)] == [
        "Rice is grown in ca. 40 countries, e.g. in Côte d'Ivoire.",
        "Smith et al. (2020) reported 3.5 t/ha for O. sativa.",
        "Fig. 3 shows the trend.",
        "See Eq. (1) and approx. 2 t/ha for irrigated vs. rainfed rice.",
    ]

    units = [s for b in tiny_doc.blocks for s in block_sentences(tiny_doc, b)]
    starts, ends = {a for a, _ in units}, {b for _, b in units}
    run_on = run_on_block(tiny_doc)
    for method in ("sentence", "recursive"):
        for overlap in (False, True):
            for c in make_chunker(method, overlap, counter, [tiny_doc]).chunk(tiny_doc):
                assert c.start in starts or run_on[0] < c.start < run_on[1], c.text
                assert c.end in ends or run_on[0] < c.end < run_on[1], c.text


@configs(method="character")
def test_character_cuts_between_words(chunked, tiny_doc):
    """The character baseline snaps every cut to whitespace: no word is split in half."""
    _, chunks = chunked
    text = tiny_doc.text
    assert all(c.start == 0 or text[c.start - 1].isspace() for c in chunks)
    assert all(c.end == len(text) or text[c.end].isspace() for c in chunks)


def test_token_counts_add_up(counter, tiny_doc):
    """`pack` sums unit counts instead of re-tokenising each chunk. That is exact for bge-m3
    (whitespace pre-tokenizer) when units are cut at whitespace, separators included."""
    units = [s for b in tiny_doc.blocks for s in block_sentences(tiny_doc, b)]
    assert sum(counter.count_many([tiny_doc.text[a:b] for a, b in units])) == counter.count(tiny_doc.text)


# --- recursive ------------------------------------------------------------------------------------

def test_recursive_never_crosses_sections(counter, tiny_doc):
    """Every recursive chunk lies inside one section group (merged tiny siblings allowed): the
    heading-only "2. Methods" joins its first subsection, the tiny "2.2 Data" and "2.3 Models"
    merge, and "1. Introduction" (big enough) stays alone. Overlap stays inside the group."""
    for overlap in (False, True):
        chunker = make_chunker("recursive", overlap, counter, [tiny_doc])
        groups = [[s.id for s in g] for g in chunker.section_groups(tiny_doc)]
        assert groups == [["s1"], ["s2", "s3"], ["s4", "s5"], ["s6"], ["s7"]]

        group_of = {sid: i for i, g in enumerate(groups) for sid in g}
        chunks, chunk_groups = chunker.chunk(tiny_doc), []
        for c in chunks:
            touched = {group_of[b.section_id] for b in tiny_doc.blocks if b.start < c.end and c.start < b.end}
            assert len(touched) == 1, c.text
            chunk_groups.append(touched.pop())
        # every group starts a new chunk, which opens with the group's heading...
        firsts = [c for i, c in enumerate(chunks) if i == 0 or chunk_groups[i] != chunk_groups[i - 1]]
        assert [c.text.split("\n")[0] for c in firsts] == ["1. Introduction", "2. Methods", "2.2 Data", "3. Results", "4. Conclusions"]
        # ...and no overlap reaches back into the previous group
        assert all(c.meta["overlap_tokens"] == 0 for c in firsts)


# --- stats ---------------------------------------------------------------------------------------

def test_size_stats_reported(counter, tiny_doc):
    """size_stats returns count/mean/p95/max and mean is close to target for the character method."""
    for overlap in (False, True):
        chunks = make_chunker("character", overlap, counter, [tiny_doc]).chunk(tiny_doc)
        stats = size_stats(chunks)
        assert stats["count"] == len(chunks)
        assert stats["p5"] <= stats["median"] <= stats["p95"] <= stats["max"] == max(c.n_tokens for c in chunks)
        assert abs(stats["mean"] - SMALL["target_tokens"]) <= 0.05 * SMALL["target_tokens"]
        assert stats["words_mean"] > 0
        assert (stats["overlap_tokens_mean"] > 0) is overlap
    assert size_stats([]) == {"count": 0}


# --- pack / overlap_start: the shared size rules ---------------------------------------------------

@pytest.mark.parametrize("sizes, target, max_tokens, expected", [
    ([30, 18], 40, 80, [(0, 2)]),                  # 48 is nearer to 40 than 30: soft target, not a max
    ([30, 22], 40, 80, [(0, 1), (1, 2)]),          # 52 is not
    ([30, 18], 40, 45, [(0, 1), (1, 2)]),          # max_tokens is hard (and blocks the tail merge)
    ([20, 60, 20], 40, 80, [(0, 1), (1, 2), (2, 3)]),   # a unit over the target is a chunk of its own
    ([40, 40, 10], 40, 80, [(0, 1), (1, 3)]),      # a tiny last chunk (< target/2) joins the previous
    ([10] * 12, 40, 80, [(0, 4), (4, 8), (8, 12)]),
    ([], 40, 80, []),
])
def test_pack(sizes, target, max_tokens, expected):
    assert pack(sizes, target, max_tokens) == expected


def test_pack_with_overlap():
    """Each chunk repeats the previous chunk's last unit (10 tokens = the budget); the overlap
    counts towards the chunk size, so every chunk still holds 40 tokens."""
    assert pack([10] * 12, 40, 80, overlap_budget=10) == [(0, 4), (3, 7), (6, 10), (9, 12)]


def test_pack_rejects_oversized_units():
    with pytest.raises(ValueError, match="split it first"):
        pack([10, 90], 40, 80)


@pytest.mark.parametrize("sizes, lo, budget, cap, expected", [
    ([30, 5, 5, 5], 1, 10, 20, 2),     # nearest whole units: 5+5 = 10, a third would give 15
    ([30, 12], 1, 10, 20, 1),          # at least one unit, even over the budget
    ([5, 5, 5, 30], 1, 10, 20, 4),     # ...unless it alone is over the cap: no overlap
    ([5, 5], 1, 100, 100, 1),          # never the whole previous chunk (lo)
    ([5, 5], 1, 0, 100, 2),            # budget 0 = overlap off
])
def test_overlap_start(sizes, lo, budget, cap, expected):
    assert overlap_start(sizes, lo, len(sizes), budget, cap) == expected


# --- validation and the stage ----------------------------------------------------------------------

@pytest.mark.parametrize("break_it", ["text", "too_long", "lost_content", "overlap", "order"])
def test_validate_chunks_fails_loudly(counter, tiny_doc, break_it):
    chunks = make_chunker("sentence", False, counter, [tiny_doc]).chunk(tiny_doc)
    validate_chunks(tiny_doc, chunks, SMALL["max_tokens"])
    a, b = chunks[0], chunks[1]
    if break_it == "text":
        chunks[0] = dataclasses.replace(a, text=a.text + " ")
    elif break_it == "too_long":
        chunks[0] = dataclasses.replace(a, n_tokens=SMALL["max_tokens"] + 1)
    elif break_it == "lost_content":
        del chunks[1]
    elif break_it == "overlap":
        chunks[1] = dataclasses.replace(b, start=a.end - 5, text=tiny_doc.text[a.end - 5:b.end])
    else:
        chunks[0], chunks[1] = b, a
    with pytest.raises(ValueError):
        validate_chunks(tiny_doc, chunks, SMALL["max_tokens"])


def test_run_writes_every_variant(cfg, counter, tiny_doc, tmp_path):
    """`chunkeval chunk`: one JSONL per (method, overlap) plus the size stats, from parsed JSONs."""
    cfg.paths.data_dir = tmp_path / "data"
    write_json(cfg.paths.data_dir / "parsed" / "papers" / "tiny.json", tiny_doc)
    run(cfg, None, corpora=["papers"])

    out = cfg.paths.data_dir / "chunks" / "papers"
    stats = json.loads((out / "_size_stats.json").read_text(encoding="utf-8"))
    assert set(stats) == {f"{m}_{'overlap' if o else 'no_overlap'}" for m, o in CONFIGS}
    for variant, s in stats.items():
        chunks = load_chunks(out / f"{variant}.jsonl")
        assert len(chunks) == s["count"]
        validate_chunks(tiny_doc, chunks, cfg.chunking.max_tokens)     # survived the JSONL round trip
    assert stats["character_no_overlap"]["window_chars"] == stats["character_overlap"]["window_chars"]

    method, overlap = cfg.chunking.methods[0], cfg.chunking.overlap[0]
    chunker = build_chunker(method, overlap, cfg, counter)
    if isinstance(chunker, CharacterChunker):
        chunker.calibrate([tiny_doc])
    assert load_chunks(out / f"{chunker.variant}.jsonl") == chunker.chunk(tiny_doc)


def test_run_without_parsed_docs(cfg, counter, tmp_path):
    cfg.paths.data_dir = tmp_path / "data"
    with pytest.raises(FileNotFoundError, match="chunkeval parse"):
        run(cfg, None, corpora=["papers"])


# --- the parsed docs, config sizes ---------------------------------------------------------------

def parsed_docs(per_corpus=2):
    """The first parsed docs of each corpus (at most `per_corpus`, so the test stays fast)."""
    docs = defaultdict(list)
    for path in sorted((REPO_ROOT / "data" / "parsed").glob("*/*.json")):
        if len(docs[path.parent.name]) < per_corpus:
            docs[path.parent.name].append(path)
    return dict(docs)


@pytest.mark.skipif(not parsed_docs(), reason="no parsed docs in data/parsed: run `chunkeval parse` first")
@pytest.mark.parametrize("method, overlap", CONFIGS, ids=CONFIG_IDS)
def test_parsed_docs(cfg, counter, method, overlap):
    """Every invariant (`validate_chunks` runs inside `chunk`) on real Docling output, with the
    config sizes; character and sentence chunks average the target (soft target)."""
    target = cfg.chunking.target_tokens
    for corpus, paths in parsed_docs().items():
        docs = [load_doc(p) for p in paths]
        chunker = build_chunker(method, overlap, cfg, counter)
        if method == "character":
            chunker.calibrate(docs)
        stats = size_stats([c for doc in docs for c in chunker.chunk(doc)])
        assert stats["max"] <= cfg.chunking.max_tokens
        if method != "recursive":            # recursive: sections smaller than the target stay whole
            assert abs(stats["mean"] - target) <= 0.05 * target, (corpus, stats)
