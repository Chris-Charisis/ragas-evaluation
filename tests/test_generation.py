"""qa.generation: the prompt file, the chat messages and the output schema (no LLM calls).

The quote locator has its own file, test_quote_locator.py.
"""
import json
import re
from typing import get_args

import jsonschema
import numpy as np
import pytest

from chunkeval.core.llm import LLMOutputError, LLMResult
from chunkeval.core.schema import QA as QAConfig
from chunkeval.core.schema import REPO_ROOT
from chunkeval.core.store import read_jsonl, run_dir, write_json
from chunkeval.corpus.parsing import is_skippable_section, load_doc
from chunkeval.qa import generation
from chunkeval.qa.generation import (PROMPT_PATH, QA_JSON_SCHEMA, QUESTION_TYPES, DroppedUnit,
                                     GenerationUnit, QAPair, assign_types, build_messages,
                                     generate_for_doc, load_prompt, qa_paths, run, sample_units)


def block_span(doc, startswith):
    """(start, end) of the block whose text starts with `startswith`."""
    return next((b.start, b.end) for b in doc.blocks if doc.text[b.start:b.end].startswith(startswith))


# --- units ----------------------------------------------------------------------------------------
# tiny_doc's blocks in bge-m3 tokens: Introduction 57 + 44, 2.1 Sites 40, 2.2 Data 5, 2.3 Models 8,
# Results 90 (7 short sentences) + list items 9 and 6 + a run-on sentence of 391, Conclusions 15.

def units_of(doc, cfg, seed=0, **qa):
    for key, value in qa.items():
        setattr(cfg.qa, key, value)
    return sample_units(doc, cfg, np.random.default_rng(seed))


def check_units(doc, cfg, units):
    """The invariants of every unit list `sample_units` returns."""
    qa = cfg.qa
    sections = {s.id: s for s in doc.sections}
    body = [b for b in doc.blocks if b.type != "heading"]
    assert [u.spans[0][0] for u in units] == sorted(u.spans[0][0] for u in units)
    assert len(units) <= qa.max_questions_per_doc
    assert sum(u.kind == "two_span" for u in units) <= round(qa.max_questions_per_doc * qa.two_span_ratio)
    for i, u in enumerate(units):
        assert u.unit_id == f"{u.kind[0]}{u.spans[0][0]}"
        for s, e in u.spans:
            assert doc.text[s:e] == doc.text[s:e].strip() != ""
            blocks = [b for b in doc.blocks if b.start < e and s < b.end]
            assert all(b.type in ("paragraph", "list_item") for b in blocks)
            assert not any(is_skippable_section(sections[b.section_id], cfg) for b in blocks)
        if u.kind == "single":
            assert len(u.spans) == 1
            assert len({b.section_id for b in doc.blocks if b.start < u.spans[0][1] and u.spans[0][0] < b.end}) == 1
            assert qa.unit_min_tokens <= u.meta["n_tokens"] <= qa.unit_max_tokens
        else:
            # Two whole blocks, the second the next non-heading block after the first.
            (a_start, a_end), (b_start, b_end) = u.spans
            k = next(k for k, b in enumerate(body) if (b.start, b.end) == (a_start, a_end))
            assert (body[k + 1].start, body[k + 1].end) == (b_start, b_end)
            assert u.meta["crosses_section"] == (body[k].section_id != body[k + 1].section_id)
            assert u.meta["n_tokens"] <= qa.unit_max_tokens
        assert u.section_path == sections[next(b for b in doc.blocks if b.start <= u.spans[0][0] < b.end).section_id].path
        for other in units[i + 1:]:
            assert not any(s1 < e2 and s2 < e1 for s1, e1 in u.spans for s2, e2 in other.spans), (u, other)


def test_units_single(tiny_doc, cfg):
    """Paragraphs of one section, packed towards the middle of min..max; too-small sections give
    none; a paragraph over the max is cut into sentence groups; the run-on sentence (over the max on
    its own) is in no unit."""
    units = units_of(tiny_doc, cfg, unit_min_tokens=30, unit_max_tokens=80, max_questions_per_doc=20,
                     two_span_ratio=0.0)
    check_units(tiny_doc, cfg, units)
    by_start = {u.spans[0][0]: u for u in units}
    # Introduction: 57 and 44 tokens are two units (together 101 > 80); 2.1 Sites (40) is one.
    for text in ("Rice is grown", "Yields depend", "Trials ran"):
        span = block_span(tiny_doc, text)
        assert by_start[span[0]].spans == [span] and not by_start[span[0]].meta["split_paragraph"]
    # Results: the 90-token paragraph is cut at sentence boundaries; the list items join its last group.
    results = [u for u in units if u.section_path == ["3. Results"]]
    assert len(results) >= 2 and all(u.meta["split_paragraph"] for u in results)
    assert results[0].spans[0][0] == block_span(tiny_doc, "Yields rose")[0]
    assert results[-1].spans[0][1] == block_span(tiny_doc, "· Labour")[1]
    # Too small: 2.2 Data, 2.3 Models, Conclusions. Too long: the run-on sentence.
    run_on = block_span(tiny_doc, "Stations recorded")
    assert all(u.spans[0][1] <= run_on[0] for u in units)
    assert len(units) == 3 + len(results)


def test_units_two_span(tiny_doc, cfg):
    """Two adjacent whole paragraphs, here across a section boundary (Introduction -> 2.1 Sites:
    44 + 40 tokens); the single units never overlap it."""
    units = units_of(tiny_doc, cfg, unit_min_tokens=30, unit_max_tokens=100, max_questions_per_doc=20,
                     two_span_ratio=0.5)
    check_units(tiny_doc, cfg, units)
    two_span = [u for u in units if u.kind == "two_span"]
    assert [u.spans for u in two_span] == [[block_span(tiny_doc, "Yields depend"), block_span(tiny_doc, "Trials ran")]]
    assert two_span[0].unit_id == f"t{block_span(tiny_doc, 'Yields depend')[0]}"
    assert two_span[0].meta["crosses_section"] is True
    assert two_span[0].section_path == ["1. Introduction"]


def test_units_skippable_sections(tiny_doc, cfg):
    """No unit uses a block of a section that Q-A generation skips (here: Results)."""
    cfg.parsing.skip_section_titles = ["results"]
    units = units_of(tiny_doc, cfg, unit_min_tokens=30, unit_max_tokens=80, max_questions_per_doc=20)
    check_units(tiny_doc, cfg, units)
    results = next(s for s in tiny_doc.sections if s.title == "3. Results")
    assert units and all(e <= results.start for u in units for _, e in u.spans)


def test_units_cap_and_ratio(tiny_doc, cfg):
    """At most max_questions_per_doc units, of which round(max * two_span_ratio) two-span."""
    units = units_of(tiny_doc, cfg, unit_min_tokens=30, unit_max_tokens=100, max_questions_per_doc=2,
                     two_span_ratio=0.5)
    check_units(tiny_doc, cfg, units)
    assert sorted(u.kind for u in units) == ["single", "two_span"]


PARSED = sorted((REPO_ROOT / "data" / "parsed").glob("*/*.json"))


@pytest.mark.skipif(not PARSED, reason="no parsed docs in data/parsed: run `chunkeval parse` first")
def test_units_parsed_docs(cfg):
    """The config sizes on the dev docs: every invariant, the full quota (they have plenty of
    candidates), spread over the document rather than the first N, and reproducible by seed."""
    differs = False
    for path in PARSED:
        doc = load_doc(path)
        units = sample_units(doc, cfg, np.random.default_rng(42))
        check_units(doc, cfg, units)
        assert len(units) == cfg.qa.max_questions_per_doc
        assert sum(u.kind == "two_span" for u in units) == round(cfg.qa.max_questions_per_doc * cfg.qa.two_span_ratio)
        starts = [u.spans[0][0] / len(doc.text) for u in units]
        assert min(starts) < 0.2 and max(starts) > 0.8, (path.stem, starts)
        assert units == sample_units(doc, cfg, np.random.default_rng(42))
        differs |= units != sample_units(doc, cfg, np.random.default_rng(43))
    assert differs


# --- prompt file ----------------------------------------------------------------------------------

def test_prompt_file_loads():
    """The real prompt file has exactly the expected sections and placeholders (checked on load)."""
    prompt = load_prompt()
    assert set(prompt) == {"system", "user", "passage", "two_span"}
    assert all(prompt.values())
    # The notes above the first section are never sent.
    assert "Why the rules are what they are" not in "".join(prompt.values())


@pytest.mark.parametrize("change, error", [
    (lambda t: t.replace("## two_span", "## twospan"), "sections must be"),
    (lambda t: t.replace("$passages", "$pasages"), "must use the placeholders"),
    (lambda t: t.replace("$two_span", ""), "must use the placeholders"),
    (lambda t: t.replace("Question type: $question_type", "Question type: $question_type, costs $5"), "bad placeholder"),
])
def test_prompt_file_checks(tmp_path, change, error):
    """A missing section, a misspelt or missing placeholder, or a lone "$" fails on load."""
    path = tmp_path / "prompt.md"
    path.write_text(change(PROMPT_PATH.read_text(encoding="utf-8")), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        load_prompt(path)


def test_prompt_file_literal_dollar(tmp_path):
    """"$$" is a literal dollar sign in a template section, and plain "$" is fine in the system one."""
    text = PROMPT_PATH.read_text(encoding="utf-8")
    text = text.replace("Question type: $question_type", "Question type: $question_type (US$$ 5)")
    text = text.replace("You write evaluation questions", "You write US$ 5 evaluation questions")
    path = tmp_path / "prompt.md"
    path.write_text(text, encoding="utf-8")
    assert "US$ 5" in load_prompt(path)["system"]


# --- messages -------------------------------------------------------------------------------------

def test_build_messages_single(tiny_doc):
    """System + user; the user message holds the title, section path, passage and question type."""
    span = block_span(tiny_doc, "Trials ran")
    unit = GenerationUnit("u0", "tiny", "papers", [span], ["2. Methods", "2.1 Sites"])
    system, user = build_messages(tiny_doc, unit, "numeric")
    assert (system["role"], user["role"]) == ("system", "user")
    assert system["content"] == load_prompt()["system"]
    text = user["content"]
    assert "Document title: A tiny paper" in text
    assert "PASSAGE\nSection: 2. Methods > 2.1 Sites\n" in text
    assert tiny_doc.text[span[0]:span[1]] in text
    assert text.endswith("Question type: numeric")       # no two-span instruction, nothing after it
    assert "PASSAGE A" not in text


def test_build_messages_system_is_shared(tiny_doc):
    """Every call starts with the same system message, so Ollama can reuse its cached prefix."""
    a = GenerationUnit("u0", "tiny", "papers", [block_span(tiny_doc, "Trials ran")], [])
    b = GenerationUnit("u1", "tiny", "papers", [block_span(tiny_doc, "Yields rose")], [])
    assert build_messages(tiny_doc, a, "factoid")[0] == build_messages(tiny_doc, b, "reasoning")[0]


def test_build_messages_two_span(tiny_doc):
    """Passages A and B, each with its own section path (here across a section boundary), plus the
    two-span instruction."""
    spans = [block_span(tiny_doc, "We fitted"), block_span(tiny_doc, "Yields rose")]
    unit = GenerationUnit("u0", "tiny", "papers", spans, ["2. Methods", "2.3 Models"], kind="two_span")
    text = build_messages(tiny_doc, unit, "reasoning")[1]["content"]
    a = text.index("PASSAGE A\nSection: 2. Methods > 2.3 Models\n")
    b = text.index("PASSAGE B\nSection: 3. Results\n")
    assert a < text.index("We fitted a mixed model.") < b < text.index("Yields rose at every site.")
    assert text.endswith(load_prompt()["two_span"])


def test_build_messages_rejects_unknown_type(tiny_doc):
    unit = GenerationUnit("u0", "tiny", "papers", [block_span(tiny_doc, "Trials ran")], [])
    with pytest.raises(ValueError, match="unknown question type"):
        build_messages(tiny_doc, unit, "trivia")


# --- output schema --------------------------------------------------------------------------------

PAIR = {"question": "What yield did MZ-DT 17 reach in Chipata?", "answer": "It reached 3.4 t/ha.",
        "evidence_quotes": ["MZ-DT 17 yielded 3.4 t/ha on average"], "question_type": "numeric"}


@pytest.mark.parametrize("output", [
    PAIR,
    {**PAIR, "skip": False},                    # a model shown both shapes sometimes adds it
    {**PAIR, "evidence_quotes": ["one quote here", "two quote here", "three quote here"]},
    {"skip": True},
    {"skip": True, "reason": "contact details only"},
], ids=["pair", "pair_skip_false", "three_quotes", "skip", "skip_reason"])
def test_schema_accepts(output):
    jsonschema.validate(output, QA_JSON_SCHEMA)


@pytest.mark.parametrize("output", [
    {**PAIR, "evidence_quotes": []},
    {**PAIR, "evidence_quotes": ["a quote"] * 4},
    {**PAIR, "evidence_quotes": [""]},
    {**PAIR, "question_type": "trivia"},
    {**PAIR, "question": ""},
    {k: v for k, v in PAIR.items() if k != "answer"},
    {**PAIR, "notes": "extra field"},
    {**PAIR, "skip": True},                     # a pair can't be a skip
    {"skip": False},
    {"skip": True, "question": "half a pair"},
], ids=["no_quotes", "four_quotes", "empty_quote", "bad_type", "empty_question", "no_answer",
        "extra_field", "pair_skip_true", "skip_false_alone", "skip_with_pair_field"])
def test_schema_rejects(output):
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(output, QA_JSON_SCHEMA)


# --- generation loop (fake LLM) -------------------------------------------------------------------

def passages(user):
    """The passage texts of a rendered user message."""
    return re.findall(r'"""\n(.*?)\n"""', user, flags=re.DOTALL)


def asked_type(user):
    return re.search(r"^Question type: (\w+)$", user, flags=re.MULTILINE).group(1)


def quote_each_passage(user):
    """A good answer: one exact quote (the first 6 words) from every passage."""
    return {"question": "Which crop do the trials in the Sahel study?", "answer": "Rice.",
            "evidence_quotes": [" ".join(p.split()[:6]) for p in passages(user)],
            "question_type": asked_type(user)}


class FakeLLM:
    """Stands in for ChatLLM: answers every request with `respond(user_message)`, a dict (the parsed
    JSON, validated against the schema like `chat` does) or an exception."""
    provider, base_url, model = "ollama", "http://fake/v1", "fake-model"

    def __init__(self, respond=quote_each_passage):
        self.respond = respond
        self.calls: list[str] = []         # the user messages, in order

    def chat_many(self, batch, json_schema=None, concurrency=None, on_result=None):
        results = []
        for i, messages in enumerate(batch):
            self.calls.append(messages[-1]["content"])
            out = self.respond(messages[-1]["content"])
            if not isinstance(out, Exception):
                jsonschema.validate(out, json_schema)
                out = LLMResult(json.dumps(out), out, prompt_tokens=100, completion_tokens=20, model=self.model)
            results.append(out)
            if on_result is not None:
                on_result(i, out)
        return results


SMALL = {"unit_min_tokens": 30, "unit_max_tokens": 100, "max_questions_per_doc": 20, "two_span_ratio": 0.5}


@pytest.fixture
def small_cfg(cfg):
    """Sizes at which tiny_doc has 5 single units and one two-span unit (see test_units_two_span)."""
    for key, value in SMALL.items():
        setattr(cfg.qa, key, value)
    return cfg


def generate(doc, cfg, respond=quote_each_passage, done=frozenset()):
    llm = FakeLLM(respond)
    return generate_for_doc(doc, llm, cfg, np.random.default_rng(0), done), llm


def test_generate_pairs(tiny_doc, small_cfg):
    """Every unit gives a pair: stable id, exact evidence, both spans covered for the two-span unit,
    the type as labelled, and the unit's details in meta."""
    units = sample_units(tiny_doc, small_cfg, np.random.default_rng(0))
    outcomes, llm = generate(tiny_doc, small_cfg)
    assert len(llm.calls) == len(units) == len(outcomes) and all(isinstance(o, QAPair) for o in outcomes)
    by_id = {o.id: o for o in outcomes}
    assert set(by_id) == {f"tiny:{u.unit_id}" for u in units}
    for unit in units:
        pair = by_id[f"tiny:{unit.unit_id}"]
        assert (pair.kind, pair.unit_id, pair.corpus, pair.generator_model) == (unit.kind, unit.unit_id, "papers", "fake-model")
        assert [(ev.match, tiny_doc.text[ev.start:ev.end]) for ev in pair.evidence] == \
               [("exact", ev.quote) for ev in pair.evidence]
        assert [any(s <= ev.start < e for ev in pair.evidence) for s, e in unit.spans] == [True] * len(unit.spans)
        assert pair.meta["requested_types"] == [pair.question_type]
        assert pair.meta["unit_spans"] == unit.spans and pair.meta["n_tokens"] == unit.meta["n_tokens"]
        assert (pair.meta["prompt_tokens"], pair.meta["completion_tokens"]) == (100, 20)


def test_generate_skips_done_units(tiny_doc, small_cfg):
    units = sample_units(tiny_doc, small_cfg, np.random.default_rng(0))
    done = {f"tiny:{u.unit_id}" for u in units[1:]}
    outcomes, llm = generate(tiny_doc, small_cfg, done=done)
    assert [o.unit_id for o in outcomes] == [units[0].unit_id] and len(llm.calls) == 1


@pytest.mark.parametrize("respond, reason", [
    (lambda user: {"skip": True, "reason": "addresses only"}, "skip"),
    (lambda user: {**quote_each_passage(user), "evidence_quotes": ["Maize yields doubled in the Kenyan highlands"]},
     "quote_not_found"),
    (lambda user: LLMOutputError("[qa_generator] answer is not valid JSON"), "bad_output"),
    (lambda user: ConnectionError("server down"), "api_error"),
])
def test_generate_drop_reasons(tiny_doc, small_cfg, respond, reason):
    """Each failure becomes a DroppedUnit with its reason and the unit's details."""
    small_cfg.qa.question_types = ["factoid"]          # no retry: factoid is the fallback itself
    outcomes, llm = generate(tiny_doc, small_cfg, respond)
    assert outcomes and all(isinstance(o, DroppedUnit) and o.reason == reason for o in outcomes)
    assert len(llm.calls) == len(outcomes)
    assert all(o.meta["unit_spans"] and o.meta["requested_types"] == ["factoid"] for o in outcomes)


def test_generate_two_span_needs_both_sides(tiny_doc, small_cfg):
    """A two-span answer quoting only passage A is dropped; single units are unaffected."""
    def only_first(user):
        out = quote_each_passage(user)
        return {**out, "evidence_quotes": out["evidence_quotes"][:1]}
    outcomes, _ = generate(tiny_doc, small_cfg, only_first)
    assert {o.kind: type(o).__name__ for o in outcomes} == {"single": "QAPair", "two_span": "DroppedUnit"}
    dropped = next(o for o in outcomes if o.kind == "two_span")
    assert (dropped.reason, dropped.detail) == ("two_span_one_side", "quotes only in passage A")


def test_generate_retries_skip_as_factoid(tiny_doc, small_cfg):
    """A single unit's skip of another type is asked again once as factoid; a factoid skip and a
    two-span skip are final."""
    def factoid_only(user):
        return quote_each_passage(user) if asked_type(user) == "factoid" else {"skip": True, "reason": "no number"}
    small_cfg.qa.question_types = ["numeric", "definition", "factoid"]
    outcomes, llm = generate(tiny_doc, small_cfg, factoid_only)
    singles = [o for o in outcomes if o.kind == "single"]
    assert singles and all(isinstance(o, QAPair) and o.question_type == "factoid" for o in singles)
    retried = [o for o in outcomes if len(o.meta["requested_types"]) == 2]
    assert retried and all(o.kind == "single" and o.meta["requested_types"][1] == "factoid" != o.meta["requested_types"][0]
                           and o.meta["first_skip_reason"] == "no number" for o in retried)
    assert len(llm.calls) == len(outcomes) + len(retried)

    outcomes, _ = generate(tiny_doc, small_cfg, lambda user: {"skip": True})
    assert all(o.reason == "skip" for o in outcomes)
    for o in outcomes:
        first = o.meta["requested_types"][0]
        assert len(o.meta["requested_types"]) == (2 if o.kind == "single" and first != "factoid" else 1)


def test_assign_types(tiny_doc, small_cfg):
    """The types in turn; "numeric" passes to the next type for a unit without a digit."""
    units = sample_units(tiny_doc, small_cfg, np.random.default_rng(0))
    small_cfg.qa.question_types = ["numeric", "factoid"]
    asked = assign_types(tiny_doc, units, small_cfg, np.random.default_rng(0))
    for unit, qtype in zip(units, asked):
        if not any(re.search(r"\d", tiny_doc.text[s:e]) for s, e in unit.spans):
            assert qtype == "factoid", unit
    assert "numeric" in asked
    assert asked == assign_types(tiny_doc, units, small_cfg, np.random.default_rng(0))


# --- run: resume, provenance, drop log ------------------------------------------------------------

@pytest.fixture
def stage(small_cfg, tiny_doc, tmp_path, monkeypatch):
    """`run` on tiny_doc in a temporary data dir, with a FakeLLM: returns (cfg, llm, paths)."""
    small_cfg.paths.data_dir = tmp_path / "data"
    write_json(small_cfg.paths.data_dir / "parsed" / "papers" / "tiny.json", tiny_doc)
    llm = FakeLLM()
    monkeypatch.setattr(generation, "ChatLLM", lambda role, cfg, secrets: llm)
    monkeypatch.setattr(generation, "ollama_model_digest", lambda base_url, model: "sha256:fake")
    return small_cfg, llm, qa_paths(small_cfg)


def records(path):
    return list(read_jsonl(path)) if path.exists() else []


def test_run_writes_and_resumes(stage):
    cfg, llm, paths = stage
    run(cfg, None, corpora=["papers"])
    pairs, dropped = records(paths["raw"]), records(paths["dropped"])
    assert pairs and not dropped and len(llm.calls) == len(pairs)
    assert QAPair(**{**pairs[0], "evidence": []}).id == pairs[0]["id"]          # the records are QAPairs
    prov = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    assert prov["settings"]["generator"]["digest"] == "sha256:fake" and set(prov["docs"]) == {"papers/tiny"}
    summary = json.loads((run_dir(cfg) / "gen_qa_summary.json").read_text(encoding="utf-8"))
    assert summary["papers"]["pairs"] == len(pairs) and summary["papers"]["pair_rate"] == 1.0
    # tiny_doc's only two-span unit crosses from the Introduction into 2.1 Sites.
    assert summary["papers"]["two_span"] == {"same_section": {"units": 0, "pairs": 0},
                                             "cross_section": {"units": 1, "pairs": 1}}

    # Nothing left to ask: no call, files unchanged.
    llm.calls.clear()
    before = paths["raw"].read_text(encoding="utf-8")
    run(cfg, None, corpora=["papers"])
    assert llm.calls == [] and paths["raw"].read_text(encoding="utf-8") == before

    # A unit lost in a crash (its line never written) is asked again, and only that one.
    lines = before.splitlines(keepends=True)
    paths["raw"].write_text("".join(lines[:-1]), encoding="utf-8")
    run(cfg, None, corpora=["papers"])
    assert len(llm.calls) == 1 and paths["raw"].read_text(encoding="utf-8").splitlines() != []
    assert sorted(r["id"] for r in records(paths["raw"])) == sorted(r["id"] for r in pairs)


def test_run_saves_drops_but_not_api_errors(stage):
    """Drops are saved (not asked again); a unit without an answer is not, and the stage fails."""
    cfg, llm, paths = stage
    llm.respond = lambda user: (ConnectionError("down") if "Yields depend" in user
                                else {"skip": True} if "Rice is grown" in user else quote_each_passage(user))
    cfg.qa.question_types = ["factoid"]
    with pytest.raises(RuntimeError, match="re-run `chunkeval gen-qa`"):
        run(cfg, None, corpora=["papers"])
    saved = {r["id"] for r in records(paths["raw"]) + records(paths["dropped"])}
    assert records(paths["dropped"]) and all(r["reason"] == "skip" for r in records(paths["dropped"]))
    failed = [user for user in llm.calls if "Yields depend" in user]
    assert failed and len(saved) == len(llm.calls) - len(failed)
    # The re-run asks exactly the units without an answer, with the same messages.
    llm.calls.clear()
    llm.respond = quote_each_passage
    run(cfg, None, corpora=["papers"])
    assert llm.calls == failed
    assert len({r["id"] for r in records(paths["raw"]) + records(paths["dropped"])}) == len(saved) + len(failed)


@pytest.mark.parametrize("change, differs", [
    (lambda cfg: setattr(cfg.qa, "fuzzy_match_threshold", 80), "qa"),
    (lambda cfg: setattr(cfg.llm.qa_generator, "temperature", 0.2), "generator"),
    (lambda cfg: setattr(cfg, "seed", 7), "seed"),
])
def test_run_refuses_other_settings(stage, change, differs):
    """Records made with other settings are never mixed with new ones: refused until --recreate."""
    cfg, llm, paths = stage
    run(cfg, None, corpora=["papers"])
    change(cfg)
    llm.calls.clear()
    with pytest.raises(RuntimeError, match=rf"other settings \({differs} differ\).*--recreate"):
        run(cfg, None, corpora=["papers"])
    assert llm.calls == []
    run(cfg, None, corpora=["papers"], recreate=True)
    assert llm.calls and len(records(paths["raw"])) == len(llm.calls)


def test_run_refuses_other_prompt(stage, tmp_path, monkeypatch):
    cfg, _, paths = stage
    run(cfg, None, corpora=["papers"])
    prompt = tmp_path / "prompt.md"
    prompt.write_text(PROMPT_PATH.read_text(encoding="utf-8") + "\nOne more rule.\n", encoding="utf-8")
    monkeypatch.setattr(generation, "PROMPT_PATH", prompt)
    with pytest.raises(RuntimeError, match="prompt_sha256 differ"):
        run(cfg, None, corpora=["papers"])


def test_run_refuses_reparsed_doc(stage, tiny_doc):
    """Evidence offsets point into the text they were located in: a changed text is refused."""
    cfg, _, paths = stage
    run(cfg, None, corpora=["papers"])
    tiny_doc.text = tiny_doc.text.replace("Rice is grown", "Rice is farmed")
    write_json(cfg.paths.data_dir / "parsed" / "papers" / "tiny.json", tiny_doc)
    with pytest.raises(RuntimeError, match="re-parsed.*papers/tiny"):
        run(cfg, None, corpora=["papers"])


def test_run_refuses_records_without_provenance(stage):
    cfg, _, paths = stage
    paths["raw"].parent.mkdir(parents=True)
    paths["raw"].write_text('{"id": "tiny:s0"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="without raw.provenance.json"):
        run(cfg, None, corpora=["papers"])


def test_question_types_match_config():
    """The schema's enum and the config's allowed types are one list (schema.QuestionType)."""
    allowed = get_args(get_args(QAConfig.model_fields["question_types"].annotation)[0])
    assert QUESTION_TYPES == allowed
    assert QA_JSON_SCHEMA["anyOf"][0]["properties"]["question_type"]["enum"] == list(QUESTION_TYPES)
