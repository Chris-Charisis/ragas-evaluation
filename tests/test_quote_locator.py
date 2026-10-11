"""qa.generation.locate_quote — the ground truth spans depend on it."""
import random
import re

import pytest

from chunkeval.core.schema import REPO_ROOT
from chunkeval.corpus.parsing import SEPARATOR, load_doc
from chunkeval.qa.generation import _INVISIBLE, _QUOTE_EDGES, MIN_QUOTE_WORDS, locate_quote

THRESHOLD = 90   # cfg.qa.fuzzy_match_threshold

# Two paragraphs joined like blocks in CanonicalDoc.text. The second has an en dash, curly
# quotes and a soft hyphen ("irri\xadgation"), as Docling leaves them in the text.
TEXT = ("Trials ran at three sites in the Sahel and the humid forest zone. Each site had four "
        "replicates of every treatment. Soils were sampled before planting and after harvest."
        + SEPARATOR +
        "Yields rose at every site. The largest gain – 2.1 t/ha – was in the Sahel, where "
        "irri­gation doubled the “paddy” harvest.")
WHOLE = [(0, len(TEXT))]


def test_exact_match():
    """A verbatim quote returns its exact span with match == "exact"."""
    quote = "Each site had four replicates of every treatment."
    ev = locate_quote(quote, TEXT, WHOLE, THRESHOLD)
    assert (ev.match, ev.score, ev.quote) == ("exact", 100.0, quote)
    assert TEXT[ev.start:ev.end] == quote
    # Surrounding whitespace in the model's quote doesn't count as a difference.
    assert locate_quote(f"  {quote}\n", TEXT, WHOLE, THRESHOLD).match == "exact"


def test_whitespace_and_newline_drift():
    """Quote with single spaces matches text containing newlines / double spaces; returned offsets
    point into the ORIGINAL text (doc.text[start:end] reproduces the original substring)."""
    # Across a paragraph break, which the model writes as one space.
    ev = locate_quote("after harvest. Yields rose at every site", TEXT, WHOLE, THRESHOLD)
    assert ev.match == "normalized"
    assert TEXT[ev.start:ev.end] == "after harvest." + SEPARATOR + "Yields rose at every site"

    text = "Weather data were  recorded\ndaily at each station."
    ev = locate_quote("Weather data were recorded daily at each station", text, [(0, len(text))], THRESHOLD)
    assert ev.match == "normalized"
    assert text[ev.start:ev.end] == "Weather data were  recorded\ndaily at each station"


def test_typographic_and_case_drift():
    """ASCII dashes and quotes for typographic ones, a soft hyphen the model never copies, a quote
    capitalised because it starts mid-sentence, an added full stop: still a normalized match,
    and the span covers the original characters."""
    ev = locate_quote("The largest gain - 2.1 t/ha - was in the Sahel", TEXT, WHOLE, THRESHOLD)
    assert ev.match == "normalized"
    assert TEXT[ev.start:ev.end] == "The largest gain – 2.1 t/ha – was in the Sahel"

    ev = locate_quote('Irrigation doubled the "paddy" harvest.', TEXT, WHOLE, THRESHOLD)
    assert ev.match == "normalized"
    # The full stop was stripped from the quote, so the span ends before the text's own.
    assert TEXT[ev.start:ev.end] == "irri­gation doubled the “paddy” harvest"


def test_hyphenation_drift():
    """"nitrogen" in the quote matches "nitro-\\ngen" in the text."""
    text = "Plant uptake of nitro-\ngen was measured at three sites. Drought- and heat-tolerant lines were compared."
    spans = [(0, len(text))]
    ev = locate_quote("Plant uptake of nitrogen was measured", text, spans, THRESHOLD)
    assert ev.match == "normalized"
    assert text[ev.start:ev.end] == "Plant uptake of nitro-\ngen was measured"
    # A hyphen before a plain space is a real one and is never joined: at most a fuzzy match.
    ev = locate_quote("Droughtand heat-tolerant lines were compared", text, spans, THRESHOLD)
    assert ev is None or ev.match == "fuzzy"


def test_fuzzy_threshold():
    """A lightly paraphrased quote matches above the threshold; a heavily paraphrased one returns None."""
    text = "Trials ran at three sites in the Sahel. Each site had four replicats of every treatment. Soils were sampled."
    spans = [(0, len(text))]
    # The model fixed the typo "replicats": one character off, so only the fuzzy step finds it.
    light = "Each site had four replicates of every treatment"
    ev = locate_quote(light, text, spans, THRESHOLD)
    assert ev.match == "fuzzy" and THRESHOLD <= ev.score < 100
    assert text[ev.start:ev.end] == "Each site had four replicats of every treatment"   # whole words
    # The threshold is honoured: the same quote fails just above its score.
    assert locate_quote(light, text, spans, ev.score + 0.5) is None
    # A paraphrase is not evidence.
    assert locate_quote("Every treatment was replicated four times at each site", text, spans, THRESHOLD) is None
    # A quote longer than the span: rapidfuzz would align the whole span inside the quote with
    # score 100 if it were allowed to.
    short = [(0, text.index(" Each"))]
    assert locate_quote("Trials ran at three sites in the Sahel and the forest zone", text, short, THRESHOLD) is None


def test_search_restricted_to_unit():
    """An identical sentence elsewhere in the document is NOT returned — only the unit's spans are searched."""
    sentence = "Seed supply remained a bottleneck in remote villages."
    outside = "Prices for paddy were stable during the trial."
    text = f"{sentence} {outside}{SEPARATOR}Farmers adopted the new varieties quickly. {sentence}"
    unit = [(text.index("Farmers"), len(text))]
    ev = locate_quote(sentence, text, unit, THRESHOLD)
    assert ev.start == text.rindex(sentence)      # the copy inside the unit, not the first one
    # A quote from outside the unit is found by no step.
    assert locate_quote(outside, text, unit, THRESHOLD) is None
    # Every span of a two-span unit is searched.
    two_span = [(0, len(sentence)), unit[0]]
    assert locate_quote("Farmers adopted the new varieties quickly", text, two_span, THRESHOLD).match == "exact"


def test_too_short_quote_rejected():
    """Quotes below the minimum length (ambiguous) return None."""
    words = "Yields rose by 12% on average at every site".split()
    text = " ".join(words)
    spans = [(0, len(text))]
    assert locate_quote(" ".join(words[:MIN_QUOTE_WORDS - 1]), text, spans, THRESHOLD) is None
    assert locate_quote(" ".join(words[:MIN_QUOTE_WORDS]), text, spans, THRESHOLD).match == "exact"
    # Enough "words", but nothing to match once the punctuation is stripped.
    assert locate_quote(". . . . . .", text, spans, THRESHOLD) is None


# --- the real dev docs ----------------------------------------------------------------------------

PARSED = sorted((REPO_ROOT / "data" / "parsed").glob("*/*.json"))
# What a model types instead of the typographic characters in the text.
ASCII = str.maketrans({**dict.fromkeys("‐‑‒–—―−", "-"),
                       **dict.fromkeys("‘’‚‛′", "'"),
                       **dict.fromkeys("“”„‟", '"')})


@pytest.mark.skipif(not PARSED, reason="no parsed docs in data/parsed: run `chunkeval parse` first")
def test_parsed_docs():
    """One random word-aligned quote from every block of the parsed dev docs. Verbatim, it matches
    exactly. With the drift a model adds (ASCII dashes and quotes, soft hyphens dropped, first
    letter capitalised, a full stop added) it matches normalized, at the same span minus the
    stripped edge punctuation and invisible characters (paper_74 has a quote ending in a soft
    hyphen, "coun\\xad"). Real Docling text, so a character that breaks the offset map of
    `_normalize` shows up here."""
    edges = _QUOTE_EDGES + "".join(_INVISIBLE)
    rng = random.Random(0)
    for path in PARSED:
        doc = load_doc(path)
        for block in doc.blocks:
            words = [m.span() for m in re.finditer(r"\S+", doc.text[block.start:block.end])]
            if block.type == "heading" or len(words) < MIN_QUOTE_WORDS:
                continue
            i = rng.randrange(len(words) - MIN_QUOTE_WORDS + 1)
            j = rng.randrange(i + MIN_QUOTE_WORDS, min(len(words), i + 30) + 1)
            quote = doc.text[block.start + words[i][0]:block.start + words[j - 1][1]]
            # The first occurrence in the block: a short quote can occur twice (report_123).
            start = doc.text.find(quote, block.start, block.end)
            end = start + len(quote)
            unit = [(block.start, block.end)]

            ev = locate_quote(quote, doc.text, unit, THRESHOLD)
            assert (ev.match, ev.start, ev.end) == ("exact", start, end), (path.stem, quote)

            drifted = quote.translate(ASCII).replace("­", "")
            ev = locate_quote(drifted[0].upper() + drifted[1:] + ".", doc.text, unit, THRESHOLD)
            lead = len(quote) - len(quote.lstrip(edges))
            trail = len(quote) - len(quote.rstrip(edges))
            assert (ev.match, ev.start, ev.end) == ("normalized", start + lead, end - trail), (path.stem, quote)
