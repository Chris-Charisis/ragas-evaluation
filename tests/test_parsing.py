"""Invariants of CanonicalDoc (see parsing.CanonicalDoc docstring)."""
import pytest

from chunkeval.core.schema import BodyRule
from chunkeval.corpus.parsing import Piece, assemble_doc, select_body, title_matches


def H(text, level=1):
    return Piece("heading", text, level=level)


def P(text):
    return Piece("paragraph", text)


@pytest.mark.parametrize("title, matches", [
    ("5. References", True), ("REFERENCES:", True), ("Appendix A", True), ("Appendix 1", True),
    ("Publications 2019", True), ("Contents of soil nitrogen", False), ("Funding mechanisms", False),
])
def test_title_matches(title, matches):
    assert title_matches(title, ["references", "appendix", "publications", "contents", "funding"]) is matches


PAPERS = BodyRule(start_titles=["introduction", "background"],
                  end_titles=["references", "appendix", "funding", "author contributions"])


def test_paper_body_is_introduction_to_conclusions():
    pieces = [H("A title"), P("Authors"), H("Abstract"), P("We study X."),
              H("1. Introduction"), P("Intro."), H("2. Conclusions"), P("Done."),
              P("Author Contributions: A.B. wrote it."), P("Funding: None."),
              H("Appendix A"), P("Survey."), H("References"), P("Smith 2020.")]
    kept, body = select_body(pieces, PAPERS)
    assert [p.text for p in kept] == ["1. Introduction", "Intro.", "2. Conclusions", "Done."]
    assert (body["start"], body["end"]) == ("1. Introduction", "Author Contributions")
    assert body["chars_dropped"] == sum(len(p.text) for p in pieces) - sum(len(p.text) for p in kept)


def test_paper_body_starts_at_background():
    kept, body = select_body([H("Abstract"), P("x"), H("Background"), P("y")], PAPERS)
    assert body["start"] == "Background" and [p.text for p in kept] == ["Background", "y"]


def test_paper_without_anchors_is_kept_whole():
    pieces = [P("Untitled text"), H("Results"), P("Numbers.")]
    kept, body = select_body(pieces, PAPERS)
    assert kept == pieces and body["start"] is None and body["end"] is None


def test_end_marker_before_the_start_is_ignored():
    """A "Funding:" line on the title page must not cut the paper."""
    pieces = [P("Funding: on page 1."), H("Introduction"), P("Intro."), H("References"), P("Smith.")]
    kept, _ = select_body(pieces, PAPERS)
    assert [p.text for p in kept] == ["Introduction", "Intro."]


def test_report_drops_sections_with_their_subsections():
    pieces = [H("Contents", 2), P("toc"), H("Message", 1), P("m"),
              H("Publications 2019", 1), H("Selected titles", 3), P("Smith 2019."),
              H("Finance", 1), P("$")]
    kept, body = select_body(pieces, BodyRule(drop_titles=["contents", "publications"]))
    assert [p.text for p in kept] == ["Message", "m", "Finance", "$"]
    assert body["dropped_sections"] == ["Contents", "Publications 2019"]


def test_selected_body_assembles_into_a_valid_doc():
    """The cut happens before offsets exist, so the invariants hold on the trimmed text."""
    kept, _ = select_body([P("Abstract"), H("Introduction"), P("Intro."), H("References"), P("x")], PAPERS)
    doc = assemble_doc("d", "papers", "t", kept)       # validate_doc runs inside
    assert doc.text == "Introduction\n\nIntro." and doc.sections[0].title == "Introduction"


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
