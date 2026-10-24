"""The corpus audit (`chunkeval audit`): each check on hand-made text, then the stage on a small
parsed corpus in a temp folder. Offline: a fake token counter and a small English word list stand
in for bge-m3's tokenizer and /usr/share/dict."""
import json
import logging

import pytest

from chunkeval.core.store import write_json
from chunkeval.corpus import audit
from chunkeval.corpus.parsing import Piece, assemble_doc

ENGLISH = "Farmers in the region planted maize and beans after the first rains of the season."
EMAIL = "Contact the project team at the office in Nairobi or write to info@example.com for the full report."
FRENCH = "Les agriculteurs de la région ont planté du maïs et des haricots pour la saison des pluies."
PORTUGUESE = "Os agricultores da região plantaram milho e feijão para a estação das chuvas em maio."
GARBLED = ("The GanbanlPlatondastenegtls initiadedbovemgonicatienebaitks wasdeterminedby "
           "theseasonalrainfallpattern ofthenorthernregion inthelastdecade andbeyond")
WORDS = {w.lower() for w in (ENGLISH + " " + EMAIL).replace(".", " ").replace("@", " ").split()}


class FakeCounter:
    """About 4 characters per token, as bge-m3 on English: ~1.4 tokens per normal word, 5+ for the
    run-together words of garbled OCR."""

    def count_many(self, texts):
        return [len(t) // 4 for t in texts]


def make_doc(doc_id, corpus, paragraphs, title=None, **meta):
    title = title or f"Report on the {doc_id} farming systems"
    pieces = [Piece("heading", "Introduction", level=1)] + [Piece("paragraph", t) for t in paragraphs]
    base = {"conversion_status": "success", "ocr": "auto",
            "text_layer": {"auto": {"replacement_char_ratio": 0.0, "repeated_paragraph_ratio": 0.0}},
            "body": {"start": None, "end": None, "dropped_sections": [], "dropped_own_text_only": [],
                     "chars_dropped": 0},
            "pdf_md5": f"md5-{doc_id}", "title_override": None}
    return assemble_doc(doc_id, corpus, title, pieces, meta=base | meta)


# --- the checks ------------------------------------------------------------------------------------

@pytest.mark.parametrize("text, language", [
    (FRENCH, "french"), (PORTUGUESE, "portuguese"), (ENGLISH, None),
    (EMAIL, None),                       # "com" of an e-mail domain isn't Portuguese
])
def test_block_language(text, language):
    assert audit.block_language([w.lower() for w in audit.ALPHA_WORD.findall(text)]) == language


@pytest.mark.parametrize("title, flags", [
    ("John Okoth Omondi 1 , Mulundu Mwila 2 , Andson Banda 2", ["author list or line numbers"]),
    ("CONTRASTING SUGARS IN COCONUT AND OIL PALM: CAN 1 CARBOHYDRATE PATTERNS BE USED AS "
     "CHEMOTAXONOMIC 2 MARKERS? 3", ["author list or line numbers"]),
    ("Final report on sensory testing in Europe for Group 1 (AFTER project, Deliverable 4.4.1.1)", []),
    ("Duplicate and Conquer: Multiple Homologs of PHOSPHORUS-STARVATION TOLERANCE1 Enhance Phosphorus "
     "Acquisition", []),
    ("Climate change adaptation in the Sahel from 2010 to 2020", []),
    ("Foresight in International Development (IDS Bulletin Vol. 47 No. 4, 2016)", ["banner"]),
    ("We are IntechOpen, the world's leading publisher of Open Access books", ["banner"]),
    ("SPOTLIGHT", ["short"]),
    ("x" * 201, ["long", "short"]),
])
def test_title_flags(title, flags):
    assert audit.title_flags(title, "report_1", shared=False) == flags


def test_title_flags_shared_and_doc_id():
    assert audit.title_flags("report_7", "report_7", shared=True) == ["short", "same as another doc", "is the doc id"]


def test_text_quality_garbled_and_other_language():
    doc = make_doc("d", "reports", [ENGLISH, FRENCH, GARBLED, "Yields rose by half."])
    summary, per_block = audit.text_quality(doc, FakeCounter(), WORDS)
    checked = len(ENGLISH) + len(FRENCH) + len(GARBLED)       # the 4-word block isn't checked
    assert summary["checked_chars"] == checked
    assert len(per_block) == 3
    assert summary["garbled_share"] == pytest.approx(len(GARBLED) / checked)     # French isn't garbled
    assert summary["other_language_share"] == {"french": pytest.approx(len(FRENCH) / checked)}
    assert [e["text"] for e in summary["garbled_examples"]] == [GARBLED[:110]]


def test_near_duplicates():
    base = [f"w{i}" for i in range(200)]
    texts = {"a": " ".join(base),
             "b": " ".join(base[:150] + [f"x{i}" for i in range(50)]),
             "c": " ".join(f"y{i}" for i in range(200))}
    pairs = audit.near_duplicates(texts)
    assert [(p["a"], p["b"]) for p in pairs] == [("a", "b")]
    assert pairs[0]["containment"] == pytest.approx(143 / 193, abs=0.01)


# --- the stage -------------------------------------------------------------------------------------

@pytest.fixture
def corpus(cfg, tmp_path, monkeypatch):
    """papers: a and its byte-identical copy b; reports: one French, one garbled, one with a bad
    title from parsing.title_overrides, one clean."""
    cfg.paths.data_dir = tmp_path / "data"
    words = tmp_path / "words"
    words.write_text("\n".join(sorted(WORDS)), encoding="utf-8")
    monkeypatch.setattr(audit, "WORD_LISTS", (str(words),))
    monkeypatch.setattr(audit, "load_counter", lambda *args: FakeCounter())
    docs = [make_doc("a", "papers", [ENGLISH, EMAIL], pdf_md5="same"),
            make_doc("b", "papers", [ENGLISH, EMAIL], pdf_md5="same"),
            make_doc("r1", "reports", [FRENCH, FRENCH, ENGLISH]),
            make_doc("r2", "reports", [GARBLED, GARBLED, ENGLISH]),
            make_doc("r3", "reports", [EMAIL], title="SPOTLIGHT", title_override="SPOTLIGHT"),
            make_doc("r4", "reports", [EMAIL + " " + ENGLISH])]
    for d in docs:
        write_json(cfg.paths.data_dir / "parsed" / d.corpus / f"{d.doc_id}.json", d)
    return cfg


def test_run_writes_summary_and_per_document_numbers(corpus, caplog):
    before = sorted(p.name for p in corpus.paths.data_dir.rglob("*"))
    with caplog.at_level(logging.INFO):
        audit.run(corpus, None, corpora=["papers", "reports"])
    assert sorted(p.name for p in corpus.paths.data_dir.rglob("*")) == before     # data/ untouched

    out = corpus.paths.runs_dir / corpus.run_id
    text = (out / "audit.txt").read_text(encoding="utf-8")
    for header in ("CONVERSION", "OCR", "BODY SELECTION", "DOCUMENT TITLES", "GARBLED TEXT", "DUPLICATES", "SIZE"):
        assert header in text
    result = json.loads((out / "audit.json").read_text(encoding="utf-8"))
    per_doc = result["per_doc"]
    assert set(per_doc) == {"papers/a", "papers/b", "reports/r1", "reports/r2", "reports/r3", "reports/r4"}
    assert result["identical_pdfs"] == [["papers/a", "papers/b"]]
    assert ("papers/a", "papers/b", 1.0) in [(p["a"], p["b"], p["containment"]) for p in result["near_duplicates"]]
    assert per_doc["reports/r1"]["other_language_share"]["french"] >= audit.OTHER_LANGUAGE_DOC_SHARE
    assert per_doc["reports/r2"]["garbled_share"] >= audit.BROKEN_LAYER_SHARE
    assert per_doc["reports/r4"]["garbled_share"] == 0 and per_doc["reports/r4"]["other_language_share"] == {}

    # An overridden title has been checked by hand: its flags are kept, but it isn't listed.
    assert per_doc["reports/r3"]["title_flags"] == ["short"]
    titles = text.split("DOCUMENT TITLES")[1].split(audit.RULE)[0]
    assert "r3" not in titles and "1 titles come from parsing.title_overrides" in titles

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("not in English" in w and "reports/r1" in w for w in warnings)
    assert any("broken text layer" in w and "reports/r2" in w for w in warnings)
    assert any("byte-identical" in w for w in warnings)
    assert not any("suspicious titles" in w for w in warnings)


def test_run_without_parsed_docs(cfg, tmp_path):
    cfg.paths.data_dir = tmp_path / "data"
    with pytest.raises(FileNotFoundError, match="chunkeval parse"):
        audit.run(cfg, None, corpora=["papers"])


def test_run_without_word_lists(corpus, monkeypatch, tmp_path):
    monkeypatch.setattr(audit, "WORD_LISTS", (str(tmp_path / "missing"),))
    with pytest.raises(FileNotFoundError, match="apt install wamerican wbritish"):
        audit.run(corpus, None, corpora=["papers"])
