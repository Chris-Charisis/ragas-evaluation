"""Invariants of CanonicalDoc (see parsing.CanonicalDoc docstring)."""
import pytest


def test_block_and_section_offsets_within_text():
    """Every Block/Section has 0 <= start < end <= len(text)."""
    pytest.skip("TODO(you)")


def test_blocks_sorted_and_disjoint():
    """Blocks are in reading order and never overlap."""
    pytest.skip("TODO(you)")


def test_section_contains_children():
    """A section's range contains all of its child sections and blocks."""
    pytest.skip("TODO(you)")


def test_page_furniture_dropped():
    """Page headers/footers/page numbers from a real report PDF do not appear in text.

    Hint: pick one report you know has a running header and assert that string is absent.
    """
    pytest.skip("TODO(you)")
