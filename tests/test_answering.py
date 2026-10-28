"""The `answer` stage: the prompt, abstention detection, and the stage against a fake generator.

The stage tests reuse the offline `retrieve` setup of test_retrieval.py (tiny_doc, its 3 questions,
one variant of overlapping windows, a fake retriever, no Weaviate) and answer with a fake generator,
so no Ollama.
"""
import json

import pytest

from chunkeval.core.config import Secrets
from chunkeval.core.llm import ContextOverflowError, LLMOutputError, LLMResult
from chunkeval.core.store import read_jsonl, run_dir, write_jsonl
from chunkeval.evaluation import answering
from chunkeval.evaluation.answering import (ABSTENTION_PHRASE, GOLD_PASSAGE, PROMPT_PATH, answer_cache_path,
                                            answer_path, build_messages, context, doc_titles, is_abstention,
                                            load_answers, load_prompt)
from chunkeval.evaluation.retrieval import (load_hits, load_questions, retrieval_path, stored_provenance,
                                            variant_chunks)
from chunkeval.evaluation.retrieval_metrics import question_metrics
from chunkeval.qa.filtering import filter_paths
from test_retrieval import VARIANT, make_pairs, offline, retrieve, stage  # noqa: F401  (fixtures used below)

SECRETS = Secrets(ollama_base_url="http://fake/v1")


# --- the prompt -----------------------------------------------------------------------------------

def test_prompt_file_holds_the_abstention_phrase(tmp_path):
    assert ABSTENTION_PHRASE in load_prompt()["system"]
    reworded = tmp_path / "rag_answer.md"
    reworded.write_text(PROMPT_PATH.read_text(encoding="utf-8").replace(ABSTENTION_PHRASE, "I don't know."),
                        encoding="utf-8")
    with pytest.raises(ValueError, match="abstention phrase"):
        load_prompt(reworded)


def test_messages_show_the_passages_in_order_then_the_question():
    """The system message is shared by every call; the user message is the passages, numbered in
    the order given, each with its document's title, then its text (a "$" in either survives),
    then the question."""
    system, user = build_messages("How much did it cost?", [("Seed prices in $ per kg", "Seed cost $5 per kg."),
                                                            ("A fertiliser trial", "Fertiliser was free.")])
    assert system == {"role": "system", "content": load_prompt()["system"]}
    assert build_messages("Other?", [("t", "x")])[0] == system
    content = user["content"]
    order = [content.index(s) for s in ("Passage 1\nDocument: Seed prices in $ per kg\n", "Seed cost $5 per kg.",
                                        "Passage 2\nDocument: A fertiliser trial\n", "Fertiliser was free.")]
    assert order == sorted(order) and user["role"] == "user"
    assert content.endswith("QUESTION: How much did it cost?")


def test_context_is_the_title_line_then_the_text():
    """What Ragas reads: the passage's title line as the generator saw it, then the text."""
    assert context("A fertiliser trial", "Fertiliser was free.") == "Document: A fertiliser trial\nFertiliser was free."
    assert "Document: $title\n" in load_prompt()["passage"]


# --- abstentions ----------------------------------------------------------------------------------

@pytest.mark.parametrize("response, expected", [
    (ABSTENTION_PHRASE, True),
    ("the context does not contain the answer", True),                      # case, no full stop
    ("**The context does not contain the answer.**", True),                 # markup
    ("The context does not contain the answer. It only covers yields in Mali.", True),   # the phrase + why
    ("The provided context doesn’t mention seed prices in Niger.", True),   # a paraphrase alone
    ("There is no information about seed prices in the context.", True),
    ("I don't know.", True),
    # partial or real answers
    ("The context does not specify the yield in Niger. In Mali, it was 3.4 t/ha.", False),
    ("MZ-DT 17 yielded 3.4 t/ha; the yield in Niger is not given.", False),
    ("The soil does not contain enough nitrogen for a second crop.", False),
    ("There is no information on prices available to farmers in remote villages.", False),
    ("Farmers said the context of the trial did not matter.", False),
])
def test_is_abstention(response, expected):
    assert is_abstention(response) is expected


# --- the stage: fake generator --------------------------------------------------------------------

class FakeGenerator:
    """Stands in for ChatLLM: answers every request with `respond(user_message)`, a string or an
    exception. `asked` records every user message, in order."""
    provider, base_url, model = "ollama", "http://fake/v1", "fake-generator"

    def __init__(self, respond):
        self.respond = respond
        self.asked: list[str] = []

    def chat_many(self, batch, json_schema=None, concurrency=None, on_result=None):
        results = []
        for i, messages in enumerate(batch):
            user = messages[-1]["content"]
            self.asked.append(user)
            out = self.respond(user)
            if not isinstance(out, Exception):
                out = LLMResult(f"  {out}\n", None, prompt_tokens=len(user) // 4, completion_tokens=len(out.split()),
                                model=self.model)
            results.append(out)
            on_result(i, out)
        return results


REPLICATES = "four replicates"


def replicates_or_abstain(user: str) -> str:
    """Answers when the replicates sentence is shown, abstains otherwise."""
    return "Each site had four replicates of every treatment." if REPLICATES in user else ABSTENTION_PHRASE


def paragraph_of(doc, quote: str) -> list[int]:
    """The span of the paragraph holding `quote`: the unit its question would come from."""
    start = doc.text.index(quote)
    block = next(b for b in doc.blocks if b.start <= start < b.end)
    return [block.start, block.end]


@pytest.fixture
def answers(offline, tiny_doc, monkeypatch):
    """`offline` with unit spans on its 3 questions (the paragraph of the first quote), retrieved
    (top_k 3), and a fake generator. Returns (cfg, generator, pairs)."""
    pairs = make_pairs(tiny_doc)
    for qa in pairs:
        qa.meta["unit_spans"] = [paragraph_of(tiny_doc, qa.evidence[0].quote)]
    write_jsonl(filter_paths(offline)["filtered"], pairs)
    retrieve(offline)
    generator = FakeGenerator(replicates_or_abstain)
    monkeypatch.setattr(answering, "ChatLLM", lambda role, cfg, secrets: generator)
    monkeypatch.setattr(answering, "ollama_model_digest", lambda base_url, model: "sha256:fake")
    return offline, generator, pairs


def answer(cfg):
    answering.run(cfg, SECRETS, corpora=["papers"])


def test_run_answers_every_setup_and_the_gold_passage(answers, tiny_doc):
    cfg, generator, pairs = answers
    answer(cfg)
    k = cfg.retrieval.top_k
    hits = load_hits(retrieval_path(cfg, "bge_m3", "papers", VARIANT))
    text = {c.chunk_id: c.text for c in variant_chunks(cfg, "papers", VARIANT)[0]}
    retrieved = load_answers(answer_path(cfg, "papers", VARIANT))
    gold = load_answers(answer_path(cfg, "papers", GOLD_PASSAGE))
    assert [a.question_id for a in retrieved] == [a.question_id for a in gold] == [qa.id for qa in pairs]

    for qa, a in zip(pairs, retrieved):                                  # the primary embedder's top k
        top = hits[qa.id][:k]
        assert a.chunk_ids == [h.chunk_id for h in top]
        assert a.contexts == [context(tiny_doc.title, text[h.chunk_id]) for h in top]
        assert a.evidence_shown == bool(question_metrics(qa, top, k)["span_hit"])
    for qa, a in zip(pairs, gold):                                       # the question's own paragraph
        start, end = qa.meta["unit_spans"][0]
        assert (a.contexts, a.chunk_ids, a.evidence_shown) == ([context(tiny_doc.title, tiny_doc.text[start:end])],
                                                               [], True)
    for a in retrieved + gold:
        assert a.abstained == (not any(REPLICATES in t for t in a.contexts))
        assert a.error is None and a.model == "fake-generator" and a.prompt_tokens > 0
        assert a.response == a.response.strip()
    assert not gold[0].abstained and gold[1].abstained

    # one request per question and setup; the generator sees exactly the stored contexts
    assert len(generator.asked) == 2 * len(pairs)
    shown = [(tiny_doc.title, text[h.chunk_id]) for h in hits[pairs[0].id][:k]]
    assert generator.asked[0] == build_messages(pairs[0].question, shown)[1]["content"]
    assert answer_path(cfg, "papers", VARIANT).with_name(f"papers_{VARIANT}.provenance.json").exists()

    summary = json.loads((run_dir(cfg) / "answer_summary.json").read_text(encoding="utf-8"))["papers"]
    counts = {"questions": 3, "answered": 1, "abstained": 2, "errors": 0, "abstention_rate": 0.6667,
              "evidence_shown": 3, "abstained_with_evidence": 2, "answered_without_evidence": 0}
    assert {key: summary[GOLD_PASSAGE][key] for key in counts} == counts
    assert summary[GOLD_PASSAGE]["response_words_mean"] == 8.0           # the one answer, not the abstentions
    s = summary[VARIANT]
    assert s["answered"] + s["abstained"] == 3 and s["evidence_shown"] == sum(a.evidence_shown for a in retrieved)


def test_rerun_asks_only_what_changed(answers):
    cfg, generator, pairs = answers
    answer(cfg)
    answer(cfg)
    assert len(generator.asked) == 6                                     # everything cached

    write_jsonl(filter_paths(cfg)["filtered"], pairs[:2])                 # filter-qa dropped a question
    retrieve(cfg)
    answer(cfg)
    assert len(generator.asked) == 6 and len(load_answers(answer_path(cfg, "papers", VARIANT))) == 2
    write_jsonl(filter_paths(cfg)["filtered"], pairs)                     # and brought it back: still cached
    retrieve(cfg)
    answer(cfg)
    assert len(generator.asked) == 6

    cfg.llm.rag_generator.max_tokens = 256                                # another generator setting
    answer(cfg)
    assert len(generator.asked) == 12


def test_a_retitled_document_is_asked_again(answers, tiny_doc):
    """A new title changes what the generator is shown, though the chunks and the retrieval stay the
    same: every passage of the document is asked again, and the provenance records the new titles."""
    cfg, generator, pairs = answers
    answer(cfg)
    before = stored_provenance(answer_path(cfg, "papers", VARIANT))["titles_sha256"]
    path = cfg.paths.data_dir / "parsed" / "papers" / f"{tiny_doc.doc_id}.json"
    path.write_text(json.dumps({**json.loads(path.read_text(encoding="utf-8")), "title": "A retitled paper"}),
                    encoding="utf-8")
    answer(cfg)
    assert len(generator.asked) == 12
    for setup in (VARIANT, GOLD_PASSAGE):
        assert all(c.startswith("Document: A retitled paper\n")
                   for a in load_answers(answer_path(cfg, "papers", setup)) for c in a.contexts)
        assert stored_provenance(answer_path(cfg, "papers", setup))["titles_sha256"] != before


def test_doc_titles_include_documents_without_questions(answers, tiny_doc):
    cfg, _, _ = answers
    parsed = cfg.paths.data_dir / "parsed" / "papers"
    other = {**json.loads((parsed / f"{tiny_doc.doc_id}.json").read_text(encoding="utf-8")),
             "doc_id": "other", "title": "Another paper"}
    (parsed / "other.json").write_text(json.dumps(other), encoding="utf-8")
    assert doc_titles(cfg, "papers") == {tiny_doc.doc_id: tiny_doc.title, "other": "Another paper"}


def test_stale_retrieval_is_refused(answers):
    cfg, generator, pairs = answers
    pairs[0].question = "How many replicates did each Sahel site have?"
    write_jsonl(filter_paths(cfg)["filtered"], pairs)                     # not retrieved again
    with pytest.raises(RuntimeError, match="run `chunkeval retrieve` first"):
        answer(cfg)
    assert generator.asked == []


@pytest.mark.parametrize("error", [
    RuntimeError("timeout"),
    # depends on the server's window, which no cache key holds: never cached, asked again once it is raised
    ContextOverflowError("[rag_generator] context window full"),
])
def test_api_error_writes_nothing_and_the_rerun_asks_only_it(answers, error):
    cfg, generator, pairs = answers
    generator.respond = lambda user: error if pairs[0].question in user else replicates_or_abstain(user)
    with pytest.raises(RuntimeError, match=r"2 question\(s\) got no answer"):
        answer(cfg)
    assert not answer_path(cfg, "papers", VARIANT).exists() and not answer_path(cfg, "papers", GOLD_PASSAGE).exists()
    assert len(list(read_jsonl(answer_cache_path(cfg)))) == 4               # the finished ones are kept

    generator.respond = replicates_or_abstain
    answer(cfg)
    assert len(generator.asked) == 6 + 2
    assert all(len(load_answers(answer_path(cfg, "papers", s))) == 3 for s in (VARIANT, GOLD_PASSAGE))


def test_unusable_answer_is_kept_with_its_error(answers):
    cfg, generator, pairs = answers
    generator.respond = (lambda user: LLMOutputError("cut off at max_tokens") if pairs[1].question in user
                         else replicates_or_abstain(user))
    answer(cfg)
    for setup in (VARIANT, GOLD_PASSAGE):
        a = load_answers(answer_path(cfg, "papers", setup))[1]
        assert (a.response, a.error, a.abstained) == ("", "cut off at max_tokens", False)
    summary = json.loads((run_dir(cfg) / "answer_summary.json").read_text(encoding="utf-8"))["papers"]
    assert summary[GOLD_PASSAGE]["errors"] == 1 and summary[GOLD_PASSAGE]["abstained"] == 1
    answer(cfg)
    assert len(generator.asked) == 6                                     # cached, not asked again
