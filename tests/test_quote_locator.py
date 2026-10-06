"""qa.generation.locate_quote — the ground truth spans depend on it."""
import pytest


def test_exact_match():
    """A verbatim quote returns its exact span with match == "exact"."""
    pytest.skip("TODO(you)")


def test_whitespace_and_newline_drift():
    """Quote with single spaces matches text containing newlines / double spaces; returned offsets
    point into the ORIGINAL text (doc.text[start:end] reproduces the original substring)."""
    pytest.skip("TODO(you)")


def test_hyphenation_drift():
    """"nitrogen" in the quote matches "nitro-\\ngen" in the text."""
    pytest.skip("TODO(you)")


def test_fuzzy_threshold():
    """A lightly paraphrased quote matches above the threshold; a heavily paraphrased one returns None."""
    pytest.skip("TODO(you)")


def test_search_restricted_to_unit():
    """An identical sentence elsewhere in the document is NOT returned — only the unit's spans are searched."""
    pytest.skip("TODO(you)")


def test_too_short_quote_rejected():
    """Quotes below the minimum length (ambiguous) return None."""
    pytest.skip("TODO(you)")
