"""Stage 1b: audit the parsed corpus before it is chunked (docs/corpus.md, "Audit").

A diagnostic: it flags, it never fixes or drops anything. Each flag is a decision to make by hand:
replace or remove the PDF (then `chunkeval parse` again: only changed PDFs are parsed, and the
JSON of a removed one is deleted), add a row to parsing.title_overrides, or extend parsing.body's
lists. The corpus must be final before `chunk`, since adding a document later changes the top-k
of every question.

Reads:  data/parsed/<corpus>/*.json (+ _failures.jsonl)
Writes: runs/<run_id>/audit.txt    the summary, one part per check
        runs/<run_id>/audit.json   the per-document numbers behind it, for follow-up queries
Nothing in data/ changes. The checks that need a decision are also logged as warnings.

The checks:
  conversion   Docling failures and partial conversions
  ocr          the documents sent through full-page OCR, and those near an OCR threshold
  body         what each document lost to parsing.body: no start or end heading, share dropped
  drop list    section titles kept in 3+ documents of a corpus with a drop list
  titles       suspicious titles (banner, author list, too long or short, shared, the doc id);
               titles from parsing.title_overrides are not listed, they have been checked
  text         per text block: garbled text (bge-m3 tokens per word, share of English words) and
               other languages (French, Portuguese and Spanish function words). A broken text
               layer shows as garbled text in most of a document.
  duplicates   byte-identical PDFs, and near-duplicates by 8-word shingles
  size         text per document, and the largest documents' share of their corpus
"""
from __future__ import annotations

import collections
import hashlib
import logging
import re
import statistics
from pathlib import Path
from typing import Any

from chunkeval.core.config import Secrets
from chunkeval.core.schema import Config
from chunkeval.core.store import read_jsonl, run_dir, write_json
from chunkeval.corpus.chunking.base import TokenCounter, load_counter
from chunkeval.corpus.parsing import CanonicalDoc, load_doc, normalise_title

log = logging.getLogger(__name__)

# English word lists for the garbled-text check (Debian/Ubuntu: apt install wamerican wbritish).
WORD_LISTS = ("/usr/share/dict/american-english", "/usr/share/dict/british-english")

# A text block is checked when it has at least this many words (shorter ones: labels, numbers).
MIN_BLOCK_WORDS = 8
# Garbled: more bge-m3 tokens per word than this (normal text ~1.5; garbled OCR 3.4–17), or fewer
# English words than this share in a block that isn't in another language.
GARBLED_TOKENS_PER_WORD = 3.0
GARBLED_MIN_ENGLISH_SHARE = 0.6
GARBLED_LIST_SHARE = 0.02      # listed from this share of a document's checked text
# Warned: garbled text in most of a document is a broken text layer, not a few bad blocks. On the
# 2026-10-20 corpus the two broken layers came out at 62% and 83%; the worst kept paper, 20%.
BROKEN_LAYER_SHARE = 0.5

# Function words that aren't English words, per language. A block in which one language's words
# pass OTHER_LANGUAGE_BLOCK_SHARE of its alphabetic words is counted as that language.
# Left out as common in English text: "do", "son", "na", "nos" (Nos.), "com" (e-mail domains).
STOPWORDS = {
    "french": {"le", "la", "les", "des", "et", "du", "une", "pour", "dans", "est", "sont", "avec", "sur", "par",
               "aux", "qui", "que"},
    "portuguese": {"da", "das", "dos", "uma", "para", "não", "ao", "pela", "pelo", "são", "nas", "em", "se"},
    "spanish": {"el", "los", "las", "del", "una", "para", "con", "por", "es", "en", "que", "se"},
}
OTHER_LANGUAGE_BLOCK_SHARE = 0.08
OTHER_LANGUAGE_DOC_SHARE = 0.2     # warned from this share of a document's checked text

# Near-duplicates: containment = shared shingles / the smaller document's shingles.
SHINGLE_WORDS = 8
NEAR_DUPLICATE_CONTAINMENT = 0.2
# Prefilter: the bottom-SKETCH_SIZE shingle hashes are a random sample of a document's shingles, so
# the share of them found in the other document estimates containment; only pairs estimated at
# PREFILTER_CONTAINMENT or more are compared in full.
SKETCH_SIZE = 400
PREFILTER_CONTAINMENT = 0.1

BANNER = re.compile(r"intechopen|open access|publisher|downloaded|©|copyright|journal|vol\.|volume|doi|issn|isbn|"
                    r"http|www\.|elsevier|springer|wiley|mdpi|frontiers|creative commons|licen[cs]e|received|"
                    r"accepted", re.I)
# A number right after a word, not part of a version or a decimal: affiliation numbers in an author
# list ("John Omondi 1 , Mulundu Mwila 2") or manuscript line numbers ("CAN 1 CARBOHYDRATE…").
TRAILING_NUMBER = re.compile(r"[A-Za-z?]\s?\d\b(?![.\d])")
BACK_MATTER = re.compile(r"donor|trustee|board|staff|annex|appendi|partner|financ|glossary|index|acronym|abbreviat|"
                         r"contact|address|imprint|copyright|citation|disclaimer|publication|reference|bibliograph|"
                         r"acknowledg|foreword|preface|list of|table of|contents|about (us|the)|credits|photo", re.I)
ALPHA_WORD = re.compile(r"[A-Za-zÀ-ÿ]{2,}")

RULE = "=" * 100


def quantiles(values: list[float], ps: tuple[int, ...] = (50, 90, 95, 99)) -> dict[str, float]:
    v = sorted(values)
    if not v:
        return {}
    return {f"p{p}": round(v[min(len(v) - 1, int(p / 100 * len(v)))], 5) for p in ps} | {"max": round(v[-1], 5)}


def share_dropped(doc: CanonicalDoc) -> float:
    """Share of the converted text that parsing.body cut (front and back matter, dropped sections)."""
    dropped = doc.meta["body"]["chars_dropped"]
    return dropped / (dropped + len(doc.text)) if dropped + len(doc.text) else 0.0


def title_flags(title: str, doc_id: str, shared: bool) -> list[str]:
    """Why a document title looks wrong; empty when it looks like a title."""
    checks = (("long", len(title) > 200),
              ("short", len(title.split()) < 3),
              ("banner", bool(BANNER.search(title))),
              ("author list or line numbers", len(TRAILING_NUMBER.findall(title)) >= 2),
              ("same as another doc", shared),
              ("is the doc id", title == doc_id))
    return [name for name, bad in checks if bad]


def block_language(alpha_words: list[str]) -> str | None:
    """The other language a block is in, by its share of that language's function words; None for
    English (or too few function words to tell)."""
    shares = {lang: sum(w in words for w in alpha_words) / len(alpha_words) for lang, words in STOPWORDS.items()}
    lang, share = max(shares.items(), key=lambda x: x[1])
    return lang if share > OTHER_LANGUAGE_BLOCK_SHARE else None


def text_quality(doc: CanonicalDoc, counter: TokenCounter, english: set[str]
                 ) -> tuple[dict[str, Any], list[tuple[float, float]]]:
    """Garbled and other-language shares of a document's checked text (blocks of MIN_BLOCK_WORDS
    words or more, headings left out), with a few garbled examples; plus (tokens per word, English
    share) per checked block, for the corpus-wide distribution."""
    blocks = []
    for b in doc.blocks:
        if b.type == "heading":
            continue
        text = doc.text[b.start:b.end]
        alpha = [w.lower() for w in ALPHA_WORD.findall(text)]
        n_words = len(text.split())
        if n_words >= MIN_BLOCK_WORDS and alpha:
            blocks.append((text, n_words, alpha))
    tokens = counter.count_many([text for text, _, _ in blocks])

    total = garbled = 0
    other: collections.Counter[str] = collections.Counter()
    examples, per_block = [], []
    for (text, n_words, alpha), n_tokens in zip(blocks, tokens):
        tokens_per_word = n_tokens / n_words
        english_share = sum(w in english for w in alpha) / len(alpha)
        per_block.append((tokens_per_word, english_share))
        total += len(text)
        lang = block_language(alpha)
        if lang:
            other[lang] += len(text)
        # Few English words is garbling in English text, and expected in another language.
        if tokens_per_word > GARBLED_TOKENS_PER_WORD or (english_share < GARBLED_MIN_ENGLISH_SHARE and not lang):
            garbled += len(text)
            if len(examples) < 3:
                examples.append({"tokens_per_word": round(tokens_per_word, 2),
                                 "english_share": round(english_share, 2), "text": text[:110]})
    summary = {"checked_chars": total,
               "garbled_share": garbled / total if total else 0.0,
               "other_language_share": {lang: n / total for lang, n in other.items()},
               "garbled_examples": examples}
    return summary, per_block


def _shingles(text: str) -> set[int]:
    w = re.findall(r"\w+", text.lower())
    return {int(hashlib.blake2b(" ".join(w[k:k + SHINGLE_WORDS]).encode(), digest_size=8).hexdigest(), 16)
            for k in range(max(1, len(w) - SHINGLE_WORDS + 1))}


def near_duplicates(texts: dict[str, str]) -> list[dict[str, Any]]:
    """Pairs of documents whose containment reaches NEAR_DUPLICATE_CONTAINMENT, highest first."""
    sh = {k: _shingles(t) for k, t in texts.items()}
    sketch = {k: set(sorted(s)[:SKETCH_SIZE]) for k, s in sh.items()}
    keys = list(texts)
    found = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            estimate = max(len(sketch[a] & sh[b]) / len(sketch[a]), len(sketch[b] & sh[a]) / len(sketch[b]))
            if estimate < PREFILTER_CONTAINMENT:
                continue
            shared = len(sh[a] & sh[b])
            containment = shared / min(len(sh[a]), len(sh[b]))
            if containment >= NEAR_DUPLICATE_CONTAINMENT:
                found.append({"a": a, "b": b, "containment": round(containment, 3),
                              "jaccard": round(shared / len(sh[a] | sh[b]), 3)})
    return sorted(found, key=lambda p: -p["containment"])


def load_english_words() -> set[str]:
    missing = [f for f in WORD_LISTS if not Path(f).exists()]
    if missing:
        raise FileNotFoundError(f"English word lists missing: {missing} (Debian/Ubuntu: sudo apt install wamerican wbritish)")
    words: set[str] = set()
    for f in WORD_LISTS:
        words |= {w.strip().lower() for w in open(f, encoding="utf-8", errors="ignore")}
    return words


def run(cfg: Config, secrets: Secrets, corpora: list[str]) -> None:
    """CLI entry (`chunkeval audit`): audit the parsed documents of each corpus in `corpora` and
    write runs/<run_id>/audit.txt and audit.json (see the module docstring)."""
    docs: dict[str, CanonicalDoc] = {}            # "corpus/doc_id" -> doc
    failures: dict[str, list[dict[str, Any]]] = {}
    for corpus in corpora:
        parsed_dir = cfg.paths.data_dir / "parsed" / corpus
        paths = sorted(parsed_dir.glob("*.json"))
        if not paths:
            raise FileNotFoundError(f"corpus '{corpus}': no parsed docs in {parsed_dir}, run `chunkeval parse` first")
        for p in paths:
            doc = load_doc(p)
            docs[f"{corpus}/{doc.doc_id}"] = doc
        fail_path = parsed_dir / "_failures.jsonl"
        failures[corpus] = list(read_jsonl(fail_path)) if fail_path.exists() else []
    english = load_english_words()
    counter = load_counter(cfg.chunking.size_tokenizer, cfg.chunking.size_tokenizer_revision)

    def of(corpus: str) -> list[tuple[str, CanonicalDoc]]:
        return [(k, d) for k, d in docs.items() if d.corpus == corpus]

    out: list[str] = []
    warnings: list[str] = []
    per_doc: dict[str, dict[str, Any]] = {k: {"chars": len(d.text), "share_dropped": round(share_dropped(d), 4)}
                                          for k, d in docs.items()}

    out += [RULE, "CONVERSION"]
    for corpus in corpora:
        status = collections.Counter(d.meta["conversion_status"] for _, d in of(corpus))
        out.append(f"{corpus}: {len(of(corpus))} parsed, {len(failures[corpus])} failures, status {dict(status)}")
        out += [f"   failure: {f}"[:300] for f in failures[corpus]]
        partial = [d.doc_id for _, d in of(corpus) if d.meta["conversion_status"] != "success"]
        if failures[corpus] or partial:
            warnings.append(f"{corpus}: {len(failures[corpus])} conversion failures, not fully converted: {partial}")

    p = cfg.parsing
    out += [RULE, f"OCR (thresholds: replacement {p.ocr_max_replacement_char_ratio}, "
                  f"repeated {p.ocr_max_repeated_paragraph_ratio}; listed: full-page OCR, or a ratio above a third of its threshold)"]
    for corpus in corpora:
        ds = [d for _, d in of(corpus)]
        auto = [d.meta["text_layer"]["auto"] for d in ds]
        out.append(f"{corpus}: ocr {dict(collections.Counter(d.meta['ocr'] for d in ds))}; "
                   f"replacement_char_ratio {quantiles([a['replacement_char_ratio'] for a in auto])}; "
                   f"repeated_paragraph_ratio {quantiles([a['repeated_paragraph_ratio'] for a in auto])}")
        for d, a in zip(ds, auto):
            near = (a["replacement_char_ratio"] > p.ocr_max_replacement_char_ratio / 3
                    or a["repeated_paragraph_ratio"] > p.ocr_max_repeated_paragraph_ratio / 3)
            if d.meta["ocr"] != "auto" or near:
                out.append(f"   {d.doc_id:12} ocr={d.meta['ocr']:9} auto={a} after={d.meta['text_layer'].get('full_page')}")

    out += [RULE, "BODY SELECTION (what each document lost to parsing.body)"]
    for corpus in corpora:
        rule = p.body.get(corpus)
        ds = [d for _, d in of(corpus)]
        if rule is None:
            out.append(f"{corpus}: no body rule, every document keeps its whole text")
            continue
        if rule.start_titles:
            no_start = [d.doc_id for d in ds if d.meta["body"]["start"] is None]
            out.append(f"{corpus}: no start heading in {len(no_start)} of {len(ds)}: {no_start}")
        if rule.end_titles:
            no_end = [d.doc_id for d in ds if d.meta["body"]["end"] is None]
            out.append(f"{corpus}: no end found in {len(no_end)} of {len(ds)}: {no_end}")
        out.append(f"{corpus}: share of text dropped {quantiles([share_dropped(d) for d in ds])}")
        for d in sorted(ds, key=lambda d: -share_dropped(d))[:8]:
            b = d.meta["body"]
            out.append(f"   {d.doc_id:12} dropped {share_dropped(d):.0%}  start={b['start']!r:.60} end={b['end']!r:.60} "
                       f"dropped_sections={b['dropped_sections'][:6]}")
        for what, key in (("start headings", "start"), ("end headings", "end")):
            counts = collections.Counter(normalise_title(d.meta["body"][key]) for d in ds if d.meta["body"][key])
            if counts:
                out.append(f"{corpus}: {what} {dict(counts.most_common())}")
        dropped = collections.Counter(normalise_title(t) for d in ds for t in d.meta["body"]["dropped_sections"])
        if dropped:
            out.append(f"{corpus}: dropped sections {dict(dropped.most_common())}")

    out += [RULE, "SECTION TITLES KEPT, by number of documents (drop-list candidates; corpora with a drop list only)"]
    for corpus in corpora:
        rule = p.body.get(corpus)
        if rule is None or not (rule.drop_titles or rule.drop_title_prefixes):
            continue
        kept = collections.Counter()
        for _, d in of(corpus):
            kept.update({normalise_title(s.title) for s in d.sections if s.title})
        out.append(f"{corpus}: {({t: n for t, n in kept.most_common(80) if n >= 3})}")
        out.append(f"{corpus}: matching back-matter words: {({t: n for t, n in kept.most_common() if BACK_MATTER.search(t)})}")

    out += [RULE, "DOCUMENT TITLES (suspicious; fix in parsing.title_overrides)"]
    title_count = collections.Counter(d.title.strip().lower() for d in docs.values())
    overridden = 0
    for k, d in docs.items():
        flags = title_flags(d.title, d.doc_id, title_count[d.title.strip().lower()] > 1)
        per_doc[k]["title_flags"] = flags
        if d.meta.get("title_override"):
            overridden += 1
        elif flags:
            out.append(f"   {d.corpus:7} {d.doc_id:12} {flags}: {d.title[:160]!r}")
    out.append(f"   ({overridden} titles come from parsing.title_overrides and are not listed)")
    n_titles = sum(1 for k, d in docs.items() if per_doc[k]["title_flags"] and not d.meta.get("title_override"))
    if n_titles:
        warnings.append(f"{n_titles} suspicious titles without an override (see audit.txt)")

    out += [RULE, "GARBLED TEXT AND OTHER LANGUAGES (per text block: bge-m3 tokens per word, share of English "
                  "words, other languages' function words)"]
    blocks: list[tuple[float, float]] = []
    for k, d in docs.items():
        summary, per_block = text_quality(d, counter, english)
        per_doc[k].update(summary)
        blocks += per_block
    if blocks:
        english_sorted = sorted(b[1] for b in blocks)
        out.append(f"all blocks: tokens per word {quantiles([b[0] for b in blocks])}, English share p1/p5/p10 "
                   f"{[round(english_sorted[int(q / 100 * len(blocks))], 3) for q in (1, 5, 10)]}")
    out.append(f"garbled (>= {GARBLED_LIST_SHARE:.0%} of the checked text; >= {BROKEN_LAYER_SHARE:.0%} "
               f"points to a broken text layer):")
    for k, v in sorted(per_doc.items(), key=lambda x: -x[1]["garbled_share"]):
        if v["garbled_share"] >= GARBLED_LIST_SHARE:
            out.append(f"   {k:20} garbled {v['garbled_share']:.1%}  e.g. {v['garbled_examples'][:2]}")
    broken = [k for k, v in per_doc.items() if v["garbled_share"] >= BROKEN_LAYER_SHARE]
    if broken:
        warnings.append(f"garbled text in most of the document (broken text layer?): {broken}")
    out.append(f"other languages (>= {OTHER_LANGUAGE_DOC_SHARE:.0%} of the checked text):")
    foreign = []
    for k, v in per_doc.items():
        for lang, share in v["other_language_share"].items():
            if share >= OTHER_LANGUAGE_DOC_SHARE:
                out.append(f"   {k:20} {lang} {share:.0%}")
                foreign.append(f"{k} ({lang} {share:.0%})")
    if foreign:
        warnings.append(f"not in English: {foreign}")

    out += [RULE, f"DUPLICATES ({SHINGLE_WORDS}-word shingles; containment = shared / the smaller document's shingles)"]
    by_md5 = collections.defaultdict(list)
    for k, d in docs.items():
        by_md5[d.meta["pdf_md5"]].append(k)
    identical = [ks for ks in by_md5.values() if len(ks) > 1]
    out.append(f"   byte-identical PDFs: {identical or 'none'}")
    if identical:
        warnings.append(f"byte-identical PDFs: {identical}")
    pairs = near_duplicates({k: d.text for k, d in docs.items()})
    for pair in pairs:
        out.append(f"   containment {pair['containment']:.2f} jaccard {pair['jaccard']:.2f}: "
                   f"{pair['a']} ({len(docs[pair['a']].text):,} chars) ~ {pair['b']} ({len(docs[pair['b']].text):,} chars)")
    out.append(f"   {len(pairs)} pairs with containment >= {NEAR_DUPLICATE_CONTAINMENT}")
    if pairs:
        warnings.append(f"{len(pairs)} near-duplicate pairs (see audit.txt), highest: {pairs[0]}")

    out += [RULE, "SIZE (canonical text per document)"]
    for corpus in corpora:
        sizes = {d.doc_id: len(d.text) for _, d in of(corpus)}
        total = sum(sizes.values())
        largest = sorted(sizes.items(), key=lambda x: -x[1])
        out.append(f"{corpus}: {len(sizes)} docs, {total:,} chars, median {statistics.median(sizes.values()):,.0f}, "
                   f"largest: " + ", ".join(f"{i} {n:,} ({n / total:.1%})" for i, n in largest[:6])
                   + "; smallest: " + ", ".join(f"{i} {n:,}" for i, n in largest[::-1][:4]))

    txt = run_dir(cfg) / "audit.txt"
    txt.write_text("\n".join(out) + "\n", encoding="utf-8")
    json_path = write_json(run_dir(cfg) / "audit.json",
                           {"corpora": corpora, "per_doc": per_doc, "near_duplicates": pairs, "identical_pdfs": identical})
    for corpus in corpora:
        log.info("%s: %d documents audited", corpus, len(of(corpus)))
    log.info("%d suspicious titles, %d documents with >= %.0f%% garbled text, %d near-duplicate pairs",
             n_titles, sum(v["garbled_share"] >= GARBLED_LIST_SHARE for v in per_doc.values()),
             100 * GARBLED_LIST_SHARE, len(pairs))
    for w in warnings:
        log.warning(w)
    log.info("audit written to %s (summary) and %s (per document)", txt, json_path)
