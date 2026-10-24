"""Invariants of CanonicalDoc (see parsing.CanonicalDoc docstring)."""
import pytest

from chunkeval.core.schema import BodyRule
from chunkeval.corpus.parsing import (Piece, assemble_doc, ocr_helps, select_body, title_has_prefix,
                                      title_matches)


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
    m = "m" * 500                               # the report's text, so the dropped sections stay small
    pieces = [H("Contents", 2), P("toc"), H("Message", 1), P(m),
              H("Publications 2019", 1), H("Selected titles", 3), P("Smith 2019."),
              H("Finance", 1), P("$")]
    kept, body = select_body(pieces, BodyRule(drop_titles=["contents", "publications"]))
    assert [p.text for p in kept] == ["Message", m, "Finance", "$"]
    assert body["dropped_sections"] == ["Contents", "Publications 2019"]
    assert body["dropped_own_text_only"] == []


def test_a_dropped_section_never_swallows_the_document():
    """report_112: "Table of contents" in a larger font than the chapters after it, so they all look
    like its subsections; over DROP_MAX_SHARE only its own text goes."""
    pieces = [H("Table of contents", 1), P("Introduction ... 3"), H("Introduction", 2), P("x" * 300),
              H("Results", 2), P("y" * 300), H("References", 2), P("Smith 2020.")]
    kept, body = select_body(pieces, BodyRule(drop_titles=["table of contents", "references"]))
    assert [p.text for p in kept] == ["Introduction", "x" * 300, "Results", "y" * 300]
    assert body["dropped_own_text_only"] == ["Table of contents"]


def test_drop_title_prefixes():
    pieces = [H("Results", 1), P("r" * 500), H("Annex 2: Workshop agenda", 1), P("8:30 Opening"),
              H("Appendix A Consumer questionnaire", 1), P("Q1"), H("Annexed land", 1), P("kept")]
    kept, body = select_body(pieces, BodyRule(drop_title_prefixes=["annex", "appendix"]))
    assert [p.text for p in kept] == ["Results", "r" * 500, "Annexed land", "kept"]
    assert body["dropped_sections"] == ["Annex 2: Workshop agenda", "Appendix A Consumer questionnaire"]


@pytest.mark.parametrize("title, matches", [
    ("Annex 2: Workshop agenda", True), ("ANNEX", True), ("Appendix A Consumer questionnaire", True),
    ("Annexed land", False), ("The annex", False), ("", False),
])
def test_title_has_prefix(title, matches):
    assert title_has_prefix(title, ["annex", "appendix"]) is matches


def test_a_parenthesis_is_not_a_lead_label():
    """paper_64: "(Additional file 1: Table S1) ..." opens a Results paragraph, it is not back matter."""
    pieces = [H("Background"), P("b" * 200), H("Results"), P("(Additional file 1: Table S1) shows it."),
              P("r" * 200), P("Additional file 1: Table S1."), H("References"), P("Smith.")]
    kept, body = select_body(pieces, BodyRule(start_titles=["background"], end_titles=["additional file", "references"]))
    assert body["end"] == "Additional file 1" and kept[-1].text == "r" * 200


def test_a_lead_label_in_the_first_quarter_is_front_matter():
    """paper_75: PLOS ONE's "Competing Interests: ..." footnote lands right after the Introduction heading."""
    pieces = [H("Introduction"), P("Competing Interests: The authors declare none."), P("i" * 400),
              H("Results"), P("r" * 400), P("Funding: Agency X."), H("References"), P("Smith.")]
    rule = BodyRule(start_titles=["introduction"], end_titles=["competing interests", "funding", "references"])
    kept, body = select_body(pieces, rule)
    assert body["end"] == "Funding" and [p.text for p in kept][-1] == "r" * 400


@pytest.mark.parametrize("auto, full_page, helps", [
    ({"replacement_char_ratio": 0.05, "repeated_paragraph_ratio": 0.01},
     {"replacement_char_ratio": 0.0, "repeated_paragraph_ratio": 0.015}, True),     # report_2: fixed
    ({"replacement_char_ratio": 0.0, "repeated_paragraph_ratio": 0.58},
     {"replacement_char_ratio": 0.0, "repeated_paragraph_ratio": 0.11}, True),      # report_77: more than halved
    ({"replacement_char_ratio": 0.0, "repeated_paragraph_ratio": 0.07},
     {"replacement_char_ratio": 0.0, "repeated_paragraph_ratio": 0.042}, True),     # paper_52: under the threshold
    ({"replacement_char_ratio": 0.0, "repeated_paragraph_ratio": 0.43},
     {"replacement_char_ratio": 0.0, "repeated_paragraph_ratio": 0.43}, False),     # report_133: repeats are real
    ({"replacement_char_ratio": 0.0, "repeated_paragraph_ratio": 0.067},
     {"replacement_char_ratio": 0.0, "repeated_paragraph_ratio": 0.054}, False),    # report_140
])
def test_ocr_helps(cfg, auto, full_page, helps):
    assert ocr_helps(auto, full_page, cfg) is helps


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
