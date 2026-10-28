"""corpus.removal (`remove-corpus`): everything of one corpus out of the data dir and Weaviate, the
other corpora untouched; refused while a config still uses it. A fake Weaviate client, no service."""
import contextlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from chunkeval.core.config import Secrets, format_collection_name
from chunkeval.core.store import write_json, write_jsonl
from chunkeval.corpus import indexing, removal
from chunkeval.corpus.parsing import doc_id_clashes
from chunkeval.qa.filtering import filter_paths
from chunkeval.qa.generation import qa_paths

from conftest import DEFAULT_CONFIG

SECRETS = Secrets(weaviate_api_key="key")


class FakeCollection:
    def __init__(self, chunks_path: str | None, n: int = 3):
        description = json.dumps({"chunks": chunks_path, "n": n}) if chunks_path else ""
        self.config = SimpleNamespace(get=lambda: SimpleNamespace(description=description))
        self.aggregate = SimpleNamespace(over_all=lambda total_count: SimpleNamespace(total_count=n))


class FakeClient:
    """collections.exists / get / delete over a dict of FakeCollections by name."""
    def __init__(self, collections: dict[str, FakeCollection]):
        self.existing = collections
        self.collections = SimpleNamespace(exists=lambda name: name in self.existing,
                                           get=lambda name: self.existing[name],
                                           delete=lambda name: self.existing.pop(name))


def pair(corpus, doc_id, unit="s0"):
    return {"id": f"{doc_id}:{unit}", "doc_id": doc_id, "corpus": corpus}


@pytest.fixture
def stage(cfg, tmp_path, monkeypatch):
    """Two corpora's artefacts in a temporary data dir, reports already out of the config, and a fake
    Weaviate holding each corpus's sentence collections plus one named like reports' but built from
    papers' chunks. Returns (cfg, client)."""
    cfg.paths.data_dir = tmp_path / "data"
    del cfg.paths.corpora["reports"]
    cfg.parsing.body.pop("reports")
    data, cache = cfg.paths.data_dir, cfg.embedding_cache_dir
    for corpus, doc_id in (("papers", "paper_1"), ("reports", "report_1")):
        write_json(data / "parsed" / corpus / f"{doc_id}.json", {})
        write_jsonl(data / "chunks" / corpus / "sentence_overlap.jsonl", [{"chunk_id": f"{doc_id}#0"}])
        for folder in (cache / "bge_m3" / corpus / "sentence_overlap", cache / "queries" / "bge_m3" / corpus,
                       cache / "dedup" / "bge_m3" / corpus):
            folder.parent.mkdir(parents=True, exist_ok=True)
            np.save(folder.with_suffix(".npy"), np.zeros((1, 2), dtype=np.float32))
            write_json(folder.with_suffix(".json"), {})
    gen, filt = qa_paths(cfg), filter_paths(cfg)
    write_jsonl(gen["raw"], [pair("papers", "paper_1"), pair("reports", "report_1"), pair("reports", "report_1", "s9")])
    write_jsonl(gen["dropped"], [pair("reports", "report_1", "s5")])
    write_json(gen["provenance"], {"settings_by_corpus": {"papers": {"seed": 42}, "reports": {"seed": 42}},
                                   "docs": {"papers/paper_1": "a", "reports/report_1": "b"}})
    write_jsonl(filt["filtered"], [pair("papers", "paper_1"), pair("reports", "report_1")])
    write_jsonl(filt["rejected"], [pair("reports", "report_1", "s9")])
    write_json(filt["provenance"], {"settings_by_corpus": {"papers": {"critic_min_score": 4},
                                                           "reports": {"critic_min_score": 4}}})
    names = {(c, o): format_collection_name(cfg, c, "sentence", o) for c in ("papers", "reports") for o in (False, True)}
    client = FakeClient({names[("papers", True)]: FakeCollection("data/chunks/papers/sentence_overlap.jsonl"),
                         names[("reports", True)]: FakeCollection("data/chunks/reports/sentence_overlap.jsonl"),
                         names[("reports", False)]: FakeCollection("data/chunks/papers/sentence_no_overlap.jsonl")})
    monkeypatch.setattr(indexing, "connect", lambda cfg, key: contextlib.nullcontext(client))
    monkeypatch.setattr(removal, "CONFIGS_DIR", tmp_path / "configs")      # no other config uses this data dir
    return cfg, client


def snapshot(cfg):
    """Every file under the data dir with its bytes."""
    return {p.relative_to(cfg.paths.data_dir): p.read_bytes() for p in cfg.paths.data_dir.rglob("*") if p.is_file()}


def test_without_yes_nothing_is_removed(stage, caplog):
    cfg, client = stage
    before, collections = snapshot(cfg), set(client.existing)
    with caplog.at_level("INFO"):
        removal.run(cfg, SECRETS, "reports")
    assert snapshot(cfg) == before and set(client.existing) == collections
    assert "nothing removed: run again with --yes to remove the 8 item(s) above" in caplog.text
    assert "1 in filtered.jsonl, 1 in rejected.jsonl" in caplog.text
    assert "2 in raw.jsonl, 1 in gen_dropped.jsonl, 1 document(s) in raw.provenance.json" in caplog.text


def test_removes_everything_of_the_corpus_and_nothing_else(stage, caplog):
    """Its folders, vectors, lines, provenance entries and collections go; papers' stay byte for byte,
    a collection under its name but built from other chunks stays, and its doc ids are free again."""
    cfg, client = stage
    papers = {p: b for p, b in snapshot(cfg).items() if "reports" not in str(p) and p.parent.name != "qa"}
    with caplog.at_level("WARNING"):
        removal.run(cfg, SECRETS, "reports", yes=True)
    after = snapshot(cfg)
    assert not [p for p in after if "reports" in str(p)]
    assert {p: b for p, b in after.items() if p.parent.name != "qa"} == papers
    gen, filt = qa_paths(cfg), filter_paths(cfg)
    for path in (gen["raw"], filt["filtered"]):
        assert [json.loads(line)["corpus"] for line in path.read_text(encoding="utf-8").splitlines()] == ["papers"]
    assert not gen["dropped"].exists() and not filt["rejected"].exists()           # left empty: deleted
    assert json.loads(gen["provenance"].read_text(encoding="utf-8")) == {
        "settings_by_corpus": {"papers": {"seed": 42}}, "docs": {"papers/paper_1": "a"}}
    assert list(json.loads(filt["provenance"].read_text(encoding="utf-8"))["settings_by_corpus"]) == ["papers"]
    assert sorted(client.existing) == sorted([format_collection_name(cfg, "papers", "sentence", True),
                                              format_collection_name(cfg, "reports", "sentence", False)])
    assert "not built by `index` from the chunks of reports" in caplog.text

    cfg.paths.corpora["reports"] = cfg.paths.data_dir.parent / "pdfs" / "reports"   # the same names again
    cfg.paths.corpora["reports"].mkdir(parents=True)
    (cfg.paths.corpora["reports"] / "report_1.pdf").write_bytes(b"")
    assert doc_id_clashes(cfg, ["reports"]) == []
    del cfg.paths.corpora["reports"]
    with pytest.raises(RuntimeError, match="nothing of a corpus 'reports'"):     # done: a second run finds nothing
        removal.run(cfg, SECRETS, "reports", yes=True)


def test_refused_while_a_config_uses_it(stage, tmp_path):
    """The config given, or any configs/*.yaml with the same data dir, still listing the corpus."""
    cfg, client = stage
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "other.yaml").write_text(yaml.safe_dump({"extends": str(DEFAULT_CONFIG), "paths": {"data_dir": str(
        cfg.paths.data_dir)}, "weaviate": {"namespace": ""}}), encoding="utf-8")
    (configs / "elsewhere.yaml").write_text(yaml.safe_dump({"extends": str(DEFAULT_CONFIG)}), encoding="utf-8")
    before = snapshot(cfg)
    with pytest.raises(RuntimeError, match=r"still in paths.corpora of \S*other\.yaml, with this data dir") as e:
        removal.run(cfg, SECRETS, "reports", yes=True)
    assert "elsewhere" not in str(e.value)                     # another data dir: its reports are another corpus
    cfg.paths.corpora["reports"] = cfg.paths.data_dir / "pdfs"
    with pytest.raises(RuntimeError, match="the config given"):
        removal.run(cfg, SECRETS, "reports", yes=True)
    assert snapshot(cfg) == before


def test_a_misspelt_name_finds_nothing(stage):
    cfg, _ = stage
    with pytest.raises(RuntimeError, match="nothing of a corpus 'reprots'.*check the name"):
        removal.run(cfg, SECRETS, "reprots", yes=True)
