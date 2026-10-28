"""qa.filtering: the deterministic checks, the critic's messages and schemas, the decision, the
de-duplication, and the `filter-qa` stage against a fake critic and a fake embedder (no LLM, no GPU)."""
import csv
import json
from dataclasses import replace

import jsonschema
import numpy as np
import pytest

from chunkeval.core.llm import LLMOutputError, LLMResult
from chunkeval.core.store import read_jsonl, run_dir, write_json, write_jsonl
from chunkeval.qa import filtering
from chunkeval.qa.filtering import (CRITERIA, CRITIC_PROMPT_PATH, build_critic_messages, critic_json_schema,
                                    decide, deduplicate, deterministic_checks, filter_paths, load_critic_prompt,
                                    numbers, run)
from chunkeval.qa.generation import EvidenceSpan, QAPair, qa_paths, section_path_at, text_sha256


def block_span(doc, startswith):
    return next((b.start, b.end) for b in doc.blocks if doc.text[b.start:b.end].startswith(startswith))


def make_pair(doc, question, answer, evidence, passages, kind="single", qtype="factoid"):
    """A QAPair on `doc`: the unit is the blocks starting with `passages`, each evidence quote is
    located exactly."""
    spans = [block_span(doc, p) for p in passages]
    located = [EvidenceSpan(doc.text.index(q), doc.text.index(q) + len(q), q, "exact") for q in evidence]
    return QAPair(id=f"{doc.doc_id}:{kind[0]}{spans[0][0]}", doc_id=doc.doc_id, corpus=doc.corpus,
                  unit_id=f"{kind[0]}{spans[0][0]}", question=question, answer=answer, question_type=qtype,
                  evidence=located, kind=kind, generator_model="fake-generator",
                  meta={"unit_spans": spans, "section_path": section_path_at(doc, spans[0][0])})


GOOD_Q = "How many replicates of every treatment did each Sahel trial site have?"
GOOD_A = "Each site had four replicates of every treatment."
GOOD_EV = ["Each site had four replicates of every treatment."]


def check(doc, question=GOOD_Q, answer=GOOD_A, evidence=GOOD_EV):
    return [r.split(":")[0] for r in deterministic_checks(make_pair(doc, question, answer, evidence, ["Trials ran"]), doc)]


# --- deterministic checks -------------------------------------------------------------------------

@pytest.mark.parametrize("question, code", [
    (GOOD_Q, None),
    ("How is collective action defined in the context of market participation in Mali?", None),
    ("How many replicates were used in this study of Sahel rice trials?", "source_reference"),
    ("IN THIS STUDY, how many replicates did each Sahel trial site have?", "source_reference"),
    ("How many replicates did each site have according to the passage?", "source_reference"),
    ("What did the authors report about replicates in the Sahel trials?", "source_reference"),
    ("How many sites were sampled for the study in the Sahel?", "source_reference"),
    ("How can Sahel farmers raise yields according to the study's conclusions?", "source_reference"),
    ("What are the uses of the Sahel trial dataset described in the abstract?", "source_reference"),
    ("They ran trials at how many sites in the Sahel and the forest zone?", "pronoun_opening"),
    ("What does Table 2 show about rice yields in Nigeria and Côte d'Ivoire?", "figure_reference"),
    ("How many replicates of every treatment did each Sahel trial site have", "not_a_question"),
    ("What is environment-sensitive genic male sterility, and what other name does it have?", "compound_question"),
    ("How many sites were there? And where were they in the Sahel?", "compound_question"),
    ("What is FIGS?", "question_too_short"),
    ("Which " + "very " * 40 + "long question about Sahel rice trials is this?", "question_too_long"),
], ids=lambda x: x[:40] if isinstance(x, str) else x)
def test_checks_question(tiny_doc, question, code):
    """Each rule flags its pattern, case-insensitively; "in the context of <topic>" passes."""
    codes = check(tiny_doc, question=question)
    assert codes == ([code] if code else [])


@pytest.mark.parametrize("answer, abstains", [
    ("The passage does not mention the number of replicates.", True),
    ("The number of replicates is not stated in the text.", True),
    ("This cannot be determined from the information given.", True),
    ("I don't know.", True),
    ("N/A", True),
    ("Farmers had no information on paddy prices.", False),
    (GOOD_A, False),
])
def test_checks_abstention(tiny_doc, answer, abstains):
    assert ("abstention" in check(tiny_doc, answer=answer)) == abstains


@pytest.mark.parametrize("answer, question, ok", [
    ("Each site had four replicates.", GOOD_Q, True),                  # no digit in the answer: nothing to check
    ("Each site had 4 replicates.", GOOD_Q, True),                     # "four" in the evidence
    ("Each site had 12 replicates of every treatment.", GOOD_Q, False),
    ("In 2019, each site had four replicates.", GOOD_Q, False),
    ("In 2019, each site had four replicates.", "How many replicates did each Sahel site have in 2019?", True),
])
def test_checks_numbers(tiny_doc, answer, question, ok):
    """Every number in the answer must be in the evidence or the question, in digits or words there:
    a number found only elsewhere in the passage means the evidence spans miss part of the answer."""
    codes = check(tiny_doc, question=question, answer=answer, evidence=GOOD_EV)
    assert ("number_not_in_evidence" not in codes) == ok


def test_numbers():
    """Thousands separators (comma, narrow no-break space), decimal commas and points, spelt-out numbers."""
    assert numbers("116,921 samples; 3,4 t/ha; 1·5; 12.5–13.0 h; 7 500 seeds; 2010") == \
           {116921, 3.4, 1.5, 12.5, 13.0, 7500, 2010}
    assert numbers("section 3.1.2") == {3, 1, 2}
    assert numbers("ten years, twenty-five farmers") == set()
    assert numbers("ten years, twenty-five farmers, a dozen", spelt=True) == {10, 25, 12}


# --- critic messages, prompt and schemas ---------------------------------------------------------

def test_critic_prompt_file_loads():
    prompt = load_critic_prompt()
    assert set(prompt) == {"system_passage", "system_question", "user_passage", "user_question", "passage", "two_span"}
    assert all(prompt.values()) and "Why the prompt is what it is" not in "".join(prompt.values())


def test_critic_prompt_file_checks(tmp_path):
    path = tmp_path / "critic.md"
    path.write_text(CRITIC_PROMPT_PATH.read_text(encoding="utf-8").replace("$quotes", "$quotez"), encoding="utf-8")
    with pytest.raises(ValueError, match="must use the placeholders"):
        load_critic_prompt(path)


def test_critic_messages_passage_call(tiny_doc):
    """The passage call: title, passage, question, answer and the evidence AS LOCATED in the text
    (here a fuzzy quote that differs from it)."""
    qa = make_pair(tiny_doc, GOOD_Q, GOOD_A, GOOD_EV, ["Trials ran"])
    qa.evidence[0].quote = "Each site had 4 replicates of every treatment"
    system, user = build_critic_messages(tiny_doc, qa, "passage")
    assert system["content"] == load_critic_prompt()["system_passage"]
    text = user["content"]
    assert "Document title: A tiny paper" in text and "PASSAGE\nSection: 2. Methods > 2.1 Sites\n" in text
    assert f"QUESTION: {GOOD_Q}\nANSWER: {GOOD_A}" in text
    assert "- Each site had four replicates of every treatment." in text and "had 4 replicates" not in text
    assert "PASSAGE A" not in text and "needs_both" not in text


def test_critic_messages_two_span(tiny_doc):
    qa = make_pair(tiny_doc, GOOD_Q, GOOD_A, GOOD_EV, ["Yields depend", "Trials ran"], kind="two_span")
    text = build_critic_messages(tiny_doc, qa, "passage")[1]["content"]
    assert text.index("PASSAGE A\n") < text.index("Yields depend") < text.index("PASSAGE B\n") < text.index("Trials ran")
    assert text.endswith(load_critic_prompt()["two_span"])


def test_critic_messages_question_call(tiny_doc):
    """The question call sees the question alone: no passage, no answer, no title."""
    qa = make_pair(tiny_doc, GOOD_Q, GOOD_A, GOOD_EV, ["Trials ran"])
    system, user = build_critic_messages(tiny_doc, qa, "question")
    assert system["content"] == load_critic_prompt()["system_question"]
    assert user["content"] == f"QUESTION: {GOOD_Q}"


def test_critic_system_messages_are_shared(tiny_doc):
    a = make_pair(tiny_doc, GOOD_Q, GOOD_A, GOOD_EV, ["Trials ran"])
    b = make_pair(tiny_doc, "Why did yields rise at every Sahel site?", "Irrigation.", ["Yields rose at every site."],
                  ["Yields rose"], kind="two_span")
    for call in ("passage", "question"):
        assert build_critic_messages(tiny_doc, a, call)[0] == build_critic_messages(tiny_doc, b, call)[0]


def fives(schema):
    """A valid critic answer to `schema` with every score 5."""
    return {name: ({"reason": "no problem", "score": 5} if prop["type"] == "object"
                   else [] if prop["type"] == "array" else "a fact")
            for name, prop in schema["properties"].items()}


@pytest.mark.parametrize("call, kind, expected", [
    ("passage", "single", ["question_asks_for", "groundedness", "answerability"]),
    ("passage", "two_span", ["question_asks_for", "groundedness", "answerability", "needs_both"]),
    ("question", "single", ["unresolved_references", "standalone", "specificity", "usefulness"]),
    ("question", "two_span", ["unresolved_references", "standalone", "specificity", "usefulness"]),
])
def test_critic_schema(call, kind, expected):
    """The note first, then the criteria; reason before score in each criterion."""
    schema = critic_json_schema(call, kind)
    assert list(schema["properties"]) == expected == schema["required"]
    assert list(schema["properties"][expected[1]]["properties"]) == ["reason", "score"]
    jsonschema.validate(fives(schema), schema)


@pytest.mark.parametrize("change", [
    lambda out: out.pop("question_asks_for"),
    lambda out: out["groundedness"].update(score=6),
    lambda out: out["groundedness"].update(score="5"),
    lambda out: out["groundedness"].pop("reason"),
    lambda out: out.update(verdict="keep"),
], ids=["no_note", "score_6", "score_string", "no_reason", "extra_field"])
def test_critic_schema_rejects(change):
    schema = critic_json_schema("passage", "single")
    out = fives(schema)
    change(out)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(out, schema)


def test_every_criterion_is_scored_once():
    """Every criterion belongs to exactly one call; needs_both only to a two-span passage call."""
    scored = [n for call in ("passage", "question") for n in filtering.criteria_for(call, "two_span")]
    assert sorted(scored) == sorted(CRITERIA) and len(set(scored)) == len(scored)


def test_decide(cfg):
    scores = {name: {"reason": "", "score": 5} for name in CRITERIA}
    scores["unresolved_references"] = ["the genebanks"]           # a note, not a criterion
    assert decide(scores, cfg) == []
    scores["specificity"]["score"], scores["groundedness"]["score"] = 3, 2
    assert decide(scores, cfg) == ["critic_groundedness: 2", "critic_specificity: 3"]
    cfg.qa.critic_min_score = 3
    assert decide(scores, cfg) == ["critic_groundedness: 2"]


# --- de-duplication -------------------------------------------------------------------------------

def bare(qid, corpus="papers"):
    return QAPair(id=qid, doc_id="d", corpus=corpus, unit_id="u", question=qid, answer="a", question_type="factoid")


def test_deduplicate():
    """Greedy, earlier wins, only within a corpus; the removed pair names its kept duplicate."""
    pairs = [bare("a"), bare("b"), bare("c"), bare("d", corpus="reports"), bare("e")]
    v = np.array([[1, 0], [0.99, 0.141], [0, 1], [1, 0], [0.9, 0.436]], dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    kept, removed = deduplicate(pairs, v, threshold=0.92)
    assert [p.id for p in kept] == ["a", "c", "d", "e"]          # d: same vector as a, other corpus
    assert [(p.id, dup) for p, dup, _ in removed] == [("b", "a")] and removed[0][2] == pytest.approx(0.99, abs=1e-3)


# --- the stage: fake critic, fake embedder -------------------------------------------------------

class FakeCritic:
    """Stands in for ChatLLM: answers every request with `respond(call, user_message, schema)`, a
    dict (validated against the schema, like `chat` does) or an exception."""
    provider, base_url, model = "ollama", "http://fake/v1", "fake-critic"

    def __init__(self, respond):
        self.respond = respond
        self.calls: list[tuple[str, str]] = []

    def chat_many(self, batch, json_schema=None, concurrency=None, on_result=None):
        call = "question" if "unresolved_references" in json_schema["properties"] else "passage"
        results = []
        for i, messages in enumerate(batch):
            self.calls.append((call, messages[-1]["content"]))
            out = self.respond(call, messages[-1]["content"], json_schema)
            if not isinstance(out, Exception):
                jsonschema.validate(out, json_schema)
                out = LLMResult(json.dumps(out), out, prompt_tokens=100, completion_tokens=20, model=self.model)
            results.append(out)
            on_result(i, out)
        return results


def strict_on_generic(call, user, schema):
    """All 5, except specificity 2 for a question containing "generic"."""
    out = fives(schema)
    if "generic" in user and "specificity" in out:
        out["specificity"]["score"] = 2
    return out


TOPICS = ["replicates", "African", "generic", "irrigated"]


def fake_embed(questions, cfg, secrets):
    """One-hot by the first topic word in each question: same topic = cosine 1."""
    vectors = np.zeros((len(questions), len(TOPICS)), dtype=np.float32)
    for i, q in enumerate(questions):
        vectors[i, next(k for k, t in enumerate(TOPICS) if t in q)] = 1
    return vectors


def stage_pairs(doc):
    """5 pairs on tiny_doc (or a copy in another corpus): one good, one failing the checks, one the
    critic rejects, a near-duplicate of the good one (later in the document), and a good two-span pair."""
    return {
        "good": make_pair(doc, GOOD_Q, GOOD_A, GOOD_EV, ["Trials ran"]),
        "checks": make_pair(doc, "In how many African countries is rice grown according to this study?",
                            "Rice is grown in ca. 40 countries in Africa.", ["Rice is grown in ca. 40 countries in Africa"],
                            ["Rice is grown"]),
        "critic": make_pair(doc, "What matters most for generic rice farming?", "Water management matters most.",
                            ["Water management matters most."], ["Water management"]),
        "duplicate": make_pair(doc, "How many replicates per treatment did the Sahel sites have?",
                               "Every site had four replicates.", ["Yields rose at every site."], ["Yields rose"]),
        "two_span": make_pair(doc, "How did irrigated and rainfed rice yields differ at the Sahel trial sites?",
                              "Irrigated vs. rainfed systems differ by approx. 2 t/ha.",
                              ["Irrigated vs. rainfed systems differ by approx. 2 t/ha",
                               "Trials ran at three sites in the Sahel"],
                              ["Yields depend", "Trials ran"], kind="two_span"),
    }


@pytest.fixture
def stage(cfg, tiny_doc, tmp_path, monkeypatch):
    """raw.jsonl with the 5 `stage_pairs` on tiny_doc in a temporary data dir. Returns (cfg, critic,
    pairs by name)."""
    cfg.paths.data_dir = tmp_path / "data"
    write_json(cfg.paths.data_dir / "parsed" / "papers" / "tiny.json", tiny_doc)
    pairs = stage_pairs(tiny_doc)
    paths = qa_paths(cfg)
    write_jsonl(paths["raw"], pairs.values())
    write_json(paths["provenance"], {"settings": {}, "docs": {"papers/tiny": text_sha256(tiny_doc)}})
    critic = FakeCritic(strict_on_generic)
    monkeypatch.setattr(filtering, "ChatLLM", lambda role, cfg, secrets: critic)
    monkeypatch.setattr(filtering, "ollama_model_digest", lambda base_url, model: "sha256:fake")
    monkeypatch.setattr(filtering, "unload_ollama_model", lambda base_url, model: None)
    monkeypatch.setattr(filtering, "embed_questions", fake_embed)
    return cfg, critic, pairs


def outputs(cfg):
    paths = filter_paths(cfg)
    kept = list(read_jsonl(paths["filtered"]))
    rejected = {r["id"]: r["meta"]["filter"] for r in read_jsonl(paths["rejected"])}
    return kept, rejected


def test_run(stage):
    """Each pair ends where its flaw puts it, with its stage and reasons; the critic is asked twice
    per pair that passed the checks; the outputs are QAPairs in document order."""
    cfg, critic, pairs = stage
    run(cfg, None, corpora=["papers"])
    kept, rejected = outputs(cfg)
    assert [r["id"] for r in kept] == [pairs["two_span"].id, pairs["good"].id]
    assert all(QAPair(**{**r, "evidence": []}).id == r["id"] for r in kept)
    assert kept[0]["meta"]["filter"]["critic"]["needs_both"]["score"] == 5
    assert kept[1]["meta"]["filter"]["critic"]["question_asks_for"] == "a fact"
    assert rejected[pairs["checks"].id]["stage"] == "checks"
    assert rejected[pairs["checks"].id]["reasons"] == ["source_reference: 'this study', 'according to this study'"]
    assert (rejected[pairs["critic"].id]["stage"], rejected[pairs["critic"].id]["reasons"]) == ("critic", ["critic_specificity: 2"])
    dup = rejected[pairs["duplicate"].id]
    assert (dup["stage"], dup["duplicate_of"], dup["cosine"]) == ("dedup", pairs["good"].id, 1.0)
    asked = {qa: f"QUESTION: {pairs[qa].question}" for qa in ("good", "critic", "duplicate", "two_span")}
    assert sorted(text for call, text in critic.calls if call == "question") == sorted(asked.values())
    assert sum(call == "passage" for call, _ in critic.calls) == 4

    summary = json.loads((run_dir(cfg) / "filter_qa_summary.json").read_text(encoding="utf-8"))["papers"]
    assert (summary["pairs"], summary["kept"], summary["keep_rate"]) == (5, 2, 0.4)
    assert summary["rejected"] == {"checks": 1, "critic": 1, "dedup": 1}
    assert summary["kept_kinds"] == {"single": 1, "two_span": 1}
    assert summary["critic_scores"]["specificity"] == {"scored": 4, "mean": 4.25, "below_min": 1}
    assert summary["critic_scores"]["needs_both"]["scored"] == 1


def test_run_labelling_sample(stage):
    """Kept and judged-rejected pairs (not duplicates), without the filter's decision; never overwritten."""
    cfg, _, pairs = stage
    run(cfg, None, corpora=["papers"])
    path = run_dir(cfg) / "filter_qa_labelling.csv"
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert sorted(r["id"] for r in rows) == sorted(pairs[k].id for k in ("good", "two_span", "checks", "critic"))
    assert list(rows[0]) == filtering.LABELLING_COLUMNS and all(r["label"] == "" for r in rows)
    two_span = next(r for r in rows if r["id"] == pairs["two_span"].id)
    assert "--- B ---" in two_span["passage"] and "Trials ran" in two_span["passage"]
    path.write_text("labelled by hand\n", encoding="utf-8")
    run(cfg, None, corpora=["papers"])
    assert path.read_text(encoding="utf-8") == "labelled by hand\n"


def test_run_resumes_from_the_cache(stage):
    """A re-run asks nothing; a new threshold asks nothing; a changed answer asks that pair's
    passage call only; another critic setting asks everything again."""
    cfg, critic, pairs = stage
    run(cfg, None, corpora=["papers"])
    first = outputs(cfg)
    critic.calls.clear()
    run(cfg, None, corpora=["papers"])
    assert critic.calls == [] and outputs(cfg) == first

    cfg.qa.critic_min_score = 2
    run(cfg, None, corpora=["papers"])
    assert critic.calls == [] and pairs["critic"].id in [r["id"] for r in outputs(cfg)[0]]

    raw = qa_paths(cfg)["raw"]
    records = list(read_jsonl(raw))
    records[0]["answer"] = "Four replicates per treatment at each site."
    write_jsonl(raw, records)
    run(cfg, None, corpora=["papers"])
    assert [call for call, _ in critic.calls] == ["passage"]

    critic.calls.clear()
    cfg.llm.qa_critic.temperature = 0.3
    run(cfg, None, corpora=["papers"])
    assert len(critic.calls) == 8


def test_run_api_error_writes_nothing(stage):
    """A pair without an answer stops the stage before anything is written; the answers that came
    are cached, and the re-run asks only the missing ones."""
    cfg, critic, pairs = stage
    critic.respond = lambda call, user, schema: (ConnectionError("down") if "generic" in user
                                                 else strict_on_generic(call, user, schema))
    with pytest.raises(RuntimeError, match="re-run `chunkeval filter-qa`"):
        run(cfg, None, corpora=["papers"])
    assert not filter_paths(cfg)["filtered"].exists() and not filter_paths(cfg)["rejected"].exists()
    assert len(list(read_jsonl(filter_paths(cfg)["critic"]))) == 6
    critic.calls.clear()
    critic.respond = strict_on_generic
    run(cfg, None, corpora=["papers"])
    assert len(critic.calls) == 2 and all("generic" in text for _, text in critic.calls)


def test_run_unusable_answer_is_rejected_and_cached(stage):
    cfg, critic, pairs = stage
    critic.respond = lambda call, user, schema: (LLMOutputError("[qa_critic] answer cut off") if "generic" in user
                                                 and call == "question" else strict_on_generic(call, user, schema))
    run(cfg, None, corpora=["papers"])
    reasons = outputs(cfg)[1][pairs["critic"].id]["reasons"]
    assert reasons == ["critic_bad_output: question call: [qa_critic] answer cut off"]
    critic.calls.clear()
    run(cfg, None, corpora=["papers"])
    assert critic.calls == []


def test_run_refuses_reparsed_doc(stage, tiny_doc):
    """Evidence offsets point into the text gen-qa saw: a document re-parsed since is refused."""
    cfg, critic, _ = stage
    tiny_doc.text = tiny_doc.text.replace("Rice is grown", "Rice is farmed")
    write_json(cfg.paths.data_dir / "parsed" / "papers" / "tiny.json", tiny_doc)
    with pytest.raises(RuntimeError, match="papers/tiny was re-parsed"):
        run(cfg, None, corpora=["papers"])
    assert critic.calls == []


@pytest.fixture
def two_corpora(stage, tiny_doc):
    """`stage` with tiny_doc again as reports/tinyr, with the same 5 pairs. Returns (cfg, critic,
    papers pairs, reports pairs)."""
    cfg, critic, pairs = stage
    doc = replace(tiny_doc, doc_id="tinyr", corpus="reports")
    write_json(cfg.paths.data_dir / "parsed" / "reports" / "tinyr.json", doc)
    reports = stage_pairs(doc)
    paths = qa_paths(cfg)
    write_jsonl(paths["raw"], [*pairs.values(), *reports.values()])
    write_json(paths["provenance"], {"settings": {}, "docs": {"papers/tiny": text_sha256(tiny_doc),
                                                               "reports/tinyr": text_sha256(doc)}})
    return cfg, critic, pairs, reports


def lines_of(path, corpus):
    return [line for line in path.read_text(encoding="utf-8").splitlines() if f'"corpus": "{corpus}"' in line]


def test_run_per_corpus_carries_the_others_over(two_corpora, caplog):
    """--corpus filters only those corpora: the others' lines stay exactly as the last run wrote
    them, each corpus's filter settings are recorded, and a carried corpus whose pairs in raw.jsonl
    changed since is flagged."""
    cfg, critic, pairs, reports = two_corpora
    paths = filter_paths(cfg)
    run(cfg, None, corpora=["papers", "reports"])
    before = {name: lines_of(paths[name], "reports") for name in ("filtered", "rejected")}
    cfg.qa.critic_min_score = 2
    critic.calls.clear()
    run(cfg, None, corpora=["papers"])
    kept = [r["id"] for r in read_jsonl(paths["filtered"])]
    assert pairs["critic"].id in kept and reports["critic"].id not in kept and critic.calls == []
    assert {name: lines_of(paths[name], "reports") for name in ("filtered", "rejected")} == before
    settings = json.loads(paths["provenance"].read_text(encoding="utf-8"))["settings_by_corpus"]
    assert (settings["papers"]["critic_min_score"], settings["reports"]["critic_min_score"]) == (2, 4)
    assert settings["papers"]["critic"]["digest"] == "sha256:fake"

    write_jsonl(qa_paths(cfg)["raw"], [*pairs.values(), *list(reports.values())[1:]])    # gen-qa ran again
    with caplog.at_level("WARNING"):
        run(cfg, None, corpora=["papers"])
    assert "lacks 1 filtered one(s): run `chunkeval filter-qa --corpus reports`" in caplog.text


def test_run_encodes_each_question_once(stage, monkeypatch):
    """The de-duplication's vectors are cached by question: a re-run encodes nothing, a changed
    question only itself."""
    cfg, _, pairs = stage
    encoded = []

    def recording(questions, cfg, secrets):
        encoded.extend(questions)
        return fake_embed(questions, cfg, secrets)

    monkeypatch.setattr(filtering, "embed_questions", recording)
    run(cfg, None, corpora=["papers"])
    assert sorted(encoded) == sorted(pairs[name].question for name in ("good", "duplicate", "two_span"))
    assert filtering.dedup_cache_path(cfg, "papers").is_relative_to(cfg.paths.data_dir)
    encoded.clear()
    run(cfg, None, corpora=["papers"])
    assert encoded == []

    raw = qa_paths(cfg)["raw"]
    records = list(read_jsonl(raw))
    records[0]["question"] = "How many replicates of every treatment did each trial site in the Sahel have?"
    write_jsonl(raw, records)
    run(cfg, None, corpora=["papers"])
    assert encoded == [records[0]["question"]]


def test_run_needs_raw_pairs(cfg, tmp_path):
    cfg.paths.data_dir = tmp_path / "data"
    with pytest.raises(FileNotFoundError, match="run `chunkeval gen-qa` first"):
        run(cfg, None, corpora=["papers"])
