"""core/config.py + core/schema.py — loading, validation, save round trip, secrets, naming.

All offline and fast. A config error caught here is one that would otherwise surface hours
into a run (or never, as a silently wrong setting).
"""
import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from chunkeval.core.config import (ConfigChanged, Secrets, collection_name, load_config, load_secrets,
                              require_secrets, save_config)
from chunkeval.core.schema import REPO_ROOT, Config, workspace_namespace

from conftest import DEFAULT_CONFIG


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------

def test_default_config_loads(cfg):
    assert isinstance(cfg, Config)


def test_relative_path_works_from_any_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_config("configs/default.yaml").run_id


def test_paths_are_absolute(cfg):
    p = cfg.prompts
    paths = [cfg.paths.data_dir, cfg.paths.llm_cache_dir, cfg.embedding_cache_dir, *cfg.paths.corpora.values(),
             p.qa_generation, p.qa_critic, p.rag_answer]
    assert all(p.is_absolute() for p in paths)


# --------------------------------------------------------------------------------------
# extends: a config holding only what differs
# --------------------------------------------------------------------------------------

def write_yaml(path, data):
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_extends_merges_on_top_of_its_base(tmp_path):
    """Mappings merge key by key, lists and values replace, null removes the key (its default then
    applies), and the base is found relative to the extending file; chains work."""
    write_yaml(tmp_path / "base.yaml", {"extends": str(DEFAULT_CONFIG), "run_id": "base-run"})
    child = write_yaml(tmp_path / "child.yaml", {
        "extends": "base.yaml",
        "paths": {"data_dir": "data/size512", "corpora": {"reports": None}},
        "parsing": {"body": {"reports": None}, "title_overrides": None},
        "chunking": {"target_tokens": 512, "overlap": [True]},
        "analysis": {"baseline": "character_overlap"}})
    default, cfg = load_config(DEFAULT_CONFIG), load_config(child)
    assert cfg.run_id == "base-run"
    assert cfg.paths.data_dir == REPO_ROOT / "data" / "size512" and cfg.paths.runs_dir == default.paths.runs_dir
    assert list(cfg.paths.corpora) == ["papers"] and list(cfg.parsing.body) == ["papers"]
    assert cfg.parsing.title_overrides is None and cfg.parsing.skip_section_titles == default.parsing.skip_section_titles
    assert (cfg.chunking.target_tokens, cfg.chunking.overlap, cfg.chunking.methods) == (512, [True], default.chunking.methods)


def test_extends_refuses_a_cycle(tmp_path):
    write_yaml(tmp_path / "a.yaml", {"extends": "b.yaml"})
    write_yaml(tmp_path / "b.yaml", {"extends": "a.yaml"})
    with pytest.raises(ValueError, match="extends itself"):
        load_config(tmp_path / "a.yaml")


def test_another_data_dir_gets_its_own_vectors_and_collections(tmp_path):
    """Another data dir caches its own vectors and names its own collections, so `embed` and
    `index --recreate` there never touch data/'s; weaviate.namespace: "" shares them on purpose."""
    default = load_config(DEFAULT_CONFIG)
    assert default.collection_namespace == "" and default.embedding_cache_dir == REPO_ROOT / "data" / "embeddings"
    assert default.paths.llm_cache_dir == REPO_ROOT / "data" / "llm_cache"

    cfg = load_config(write_yaml(tmp_path / "x.yaml", {"extends": str(DEFAULT_CONFIG), "paths": {"data_dir": "data/size512"}}))
    assert cfg.embedding_cache_dir == REPO_ROOT / "data" / "size512" / "embeddings"
    assert collection_name(cfg, "papers", "sentence", True) == "Size512_papers_sentence_overlap_v2"
    assert cfg.paths.llm_cache_dir == default.paths.llm_cache_dir            # the LLM caches stay shared
    shared = load_config(write_yaml(tmp_path / "y.yaml", {"extends": "x.yaml", "weaviate": {"namespace": ""}}))
    assert collection_name(shared, "papers", "sentence", True) == "Papers_sentence_overlap_v2"

    default.paths.data_dir = tmp_path / "data"                               # set after loading, as tests do
    assert default.embedding_cache_dir == tmp_path / "data" / "embeddings"


def test_smoke_config_is_a_workspace_of_its_own():
    """configs/smoke.yaml: the default settings in its own data dir, run and collections, so a smoke test
    never touches data/ or a run; only the LLM caches are shared, and the judge is capped low."""
    default, smoke = load_config(DEFAULT_CONFIG), load_config(REPO_ROOT / "configs" / "smoke.yaml")
    assert smoke.run_id != default.run_id and smoke.paths.runs_dir == default.paths.runs_dir
    assert smoke.paths.data_dir == REPO_ROOT / "data" / "smoke"
    assert smoke.embedding_cache_dir == REPO_ROOT / "data" / "smoke" / "embeddings"
    assert collection_name(smoke, "papers", "sentence", True) == "Smoke_papers_sentence_overlap_v2"
    assert smoke.paths.llm_cache_dir == default.paths.llm_cache_dir
    assert smoke.ragas.max_cost_usd <= 1
    # everything else as in the default config, so the smoke test runs the real settings
    same = {"paths": {"data_dir"}, "run_id": True, "ragas": {"max_cost_usd"}}
    for section, value in default.model_dump().items():
        if same.get(section) is True:
            continue
        theirs = smoke.model_dump()[section]
        if isinstance(value, dict):
            value = {k: v for k, v in value.items() if k not in same.get(section, set())}
            theirs = {k: v for k, v in theirs.items() if k not in same.get(section, set())}
        assert theirs == value, section


@pytest.mark.parametrize("data_dir, namespace", [
    ("data", ""), ("data/size512", "size512"), ("data/a/b-c", "a_b_c"), ("other", "other")])
def test_workspace_namespace(data_dir, namespace):
    assert workspace_namespace(REPO_ROOT / data_dir) == namespace


def test_reasoning_effort_parsed(cfg):
    assert cfg.llm.qa_generator.reasoning_effort == "none"
    assert cfg.llm.rag_generator.reasoning_effort == "none"
    assert cfg.llm.qa_critic.reasoning_effort == "low"
    assert cfg.llm.judge.reasoning_effort == "none"      # gpt-6-luna takes temperature 0 only with "none"


# --------------------------------------------------------------------------------------
# Validation: every bad config must fail at load time, naming the problem
# --------------------------------------------------------------------------------------

def _set(section, **values):
    return lambda raw: raw[section].update(values)


INVALID_CONFIGS = {
    # case id: (how to break the default config, text the error must contain)
    "unknown key": (lambda r: r.update(typo_key=1), "typo_key"),
    "missing section": (lambda r: r.pop("qa"), "qa"),
    "wrong type": (_set("retrieval", top_k="five"), "top_k"),
    "out of range": (_set("chunking", overlap_ratio=1.5), "overlap_ratio"),
    "bad provider": (lambda r: r["llm"]["judge"].update(provider="azure"), "provider"),
    "judge without a price": (lambda r: r["llm"]["judge"].update(model="gpt-4.1-2025-04-14"), "ragas.prices"),
    "removed bridge judge": (lambda r: r["llm"].update(bridge_judge={"provider": "openai", "model": "x"}),
                             "bridge_judge"),
    "removed ragas.embeddings_model": (_set("ragas", embeddings_model="BAAI/bge-m3"), "embeddings_model"),
    "bad reasoning_effort": (lambda r: r["llm"]["qa_critic"].update(reasoning_effort="max"), "reasoning_effort"),
    "removed ollama.seed": (_set("ollama", seed=1), "seed"),
    "TODO placeholder": (lambda r: r.update(run_id="TODO"), "TODO"),
    "target > max tokens": (_set("chunking", target_tokens=600), "target_tokens"),
    "qa unit min > max": (_set("qa", unit_min_tokens=500), "unit_min_tokens"),
    "unknown retrieval embedder": (_set("retrieval", embedders=["bge_m3", "nope"]), "nope"),
    "primary not retrieved": (_set("retrieval", embedders=["qwen3_06b"]), "primary"),
    "k above k_max": (_set("retrieval", k_values=[1, 5, 30]), "k_max"),
    "top_k above k_max": (_set("retrieval", top_k=25), "k_max"),
    "repeated k": (_set("retrieval", k_values=[1, 5, 5]), "repeated"),
    "bad baseline": (_set("analysis", baseline="character"), "baseline"),
    "body rule for unknown corpus": (lambda r: r["parsing"]["body"].update(thesis={}), "thesis"),
    "unknown body rule key": (lambda r: r["parsing"]["body"]["papers"].update(stop_titles=[]), "stop_titles"),
    "template missing placeholder": (_set("weaviate", collection_template="{corpus}_{method}_v2"), "placeholders"),
    "unknown template placeholder": (_set("weaviate", collection_template="{run}{corpus}_{method}_{overlap}"),
                                     "placeholders"),
    "no namespace for another data dir": (lambda r: (r["paths"].update(data_dir="data/size512"), r["weaviate"].update(
        collection_template="{corpus}_{method}_{overlap}_v2")), "no {namespace}"),
    "bad namespace": (_set("weaviate", namespace="2026_run"), "namespace"),
    "data dir without a valid namespace": (_set("paths", data_dir="data/2026"), "set weaviate.namespace"),
    "prompt file missing": (_set("prompts", qa_critic="prompts/nope.md"), "no such file"),
    "missing prompts section": (lambda r: r.pop("prompts"), "prompts"),
    "bad vector name": (lambda r: r["embedding"]["models"].update(
        {"bge-m3": r["embedding"]["models"]["bge_m3"]}), "bge-m3"),
    "primary also secondary": (_set("embedding", secondary=["qwen3_06b", "bge_m3"]), "listed twice"),
    "revision is a branch": (lambda r: r["embedding"]["models"]["bge_m3"].update(revision="main"), "revision"),
    "revision missing": (lambda r: r["embedding"]["models"]["qwen3_06b"].pop("revision"), "revision"),
    "tokenizer revision is a branch": (_set("chunking", size_tokenizer_revision="main"), "size_tokenizer_revision"),
    "tokenizer pinned apart from its embedder": (_set("chunking", size_tokenizer_revision="0" * 40), "differs from"),
}


@pytest.mark.parametrize("mutate, needle", INVALID_CONFIGS.values(), ids=INVALID_CONFIGS.keys())
def test_invalid_config_rejected(raw_config, mutate, needle):
    mutate(raw_config)
    with pytest.raises(ValidationError, match=needle):
        Config.model_validate(raw_config)


# --------------------------------------------------------------------------------------
# save_config
# --------------------------------------------------------------------------------------

def test_save_config_round_trip(cfg):
    out = save_config(cfg)
    assert out == cfg.paths.runs_dir / cfg.run_id / "config.yaml"
    saved = yaml.safe_load(out.read_text(encoding="utf-8"))
    # The saved (resolved) config must itself be a valid config describing the same run.
    assert Config.model_validate(saved).model_dump() == cfg.model_dump()
    assert list(saved)[:3] == ["run_id", "seed", "paths"]      # section order kept
    assert saved["paths"]["data_dir"] == "data"                # under the repo: relative, wherever it is
    assert saved["paths"]["runs_dir"] == str(cfg.paths.runs_dir)   # outside it: absolute


# --------------------------------------------------------------------------------------
# save_config's guard: a run's record changes only when accepted
# --------------------------------------------------------------------------------------

def record(cfg) -> Path:
    return cfg.paths.runs_dir / cfg.run_id / "config.yaml"


def history(cfg) -> list[Path]:
    return sorted((cfg.paths.runs_dir / cfg.run_id / "config_history").glob("*.yaml"))


def test_a_changed_setting_is_refused_until_accepted(cfg):
    save_config(cfg)
    save_config(cfg)                                           # the same config: saved again
    before = record(cfg).read_bytes()
    cfg.retrieval.k_max, cfg.qa.two_span_ratio = 10, 0.5
    with pytest.raises(ConfigChanged, match=r"(?s)qa.two_span_ratio: 0.2 -> 0.5\n.*retrieval.k_max: 20 -> 10\n"
                                            r".*Bump run_id .* --accept-config"):
        save_config(cfg)
    assert record(cfg).read_bytes() == before and history(cfg) == []     # nothing written

    save_config(cfg, accept=True)
    assert [p.read_bytes() for p in history(cfg)] == [before]            # the old record, kept
    assert yaml.safe_load(record(cfg).read_text(encoding="utf-8"))["retrieval"]["k_max"] == 10
    save_config(cfg)                                           # the new record is the reference now


def test_settings_that_cant_change_a_result_are_not_guarded(cfg, caplog):
    save_config(cfg)
    cfg.ragas.max_workers, cfg.ragas.max_cost_usd, cfg.ollama.concurrency = 8, 500, 4
    cfg.ollama.context_length = 16384               # only whether a prompt fits: an overflow is never stored
    with caplog.at_level("INFO"):
        save_config(cfg)
    assert ("can't change a result: ollama.concurrency: 2 -> 4; ollama.context_length: 8192 -> 16384; "
            "ragas.max_cost_usd") in caplog.text
    assert history(cfg) == [] and yaml.safe_load(record(cfg).read_text(encoding="utf-8"))["ragas"]["max_workers"] == 8


def edit_record(cfg, edit) -> None:
    saved = yaml.safe_load(record(cfg).read_text(encoding="utf-8"))
    edit(saved)
    record(cfg).write_text(yaml.safe_dump(saved, sort_keys=False), encoding="utf-8")


@pytest.mark.parametrize("edit, logged", [
    (lambda r: r.pop("prompts"), 'prompts.qa_critic: (not set) -> "prompts/qa_critic.md"'),   # required, added since
    (lambda r: r["weaviate"].pop("namespace"), "weaviate.namespace: (not set) -> null"),       # added at its default
    (lambda r: r["embedding"]["models"]["bge_m3"].pop("revision"),                             # in a map entry's section
     "embedding.models.bge_m3.revision: (not set) -> "),
    (lambda r: r["chunking"].update(old_knob=3), "chunking.old_knob: 3 -> (not set)"),           # removed since
])
def test_a_setting_the_code_added_or_removed_is_only_logged(cfg, caplog, edit, logged):
    """A key only one side has, in a section: a setting the code added or removed since the record
    was written. A new setting keeps the old behaviour, so it is no change."""
    save_config(cfg)
    edit_record(cfg, edit)
    with caplog.at_level("INFO"):
        save_config(cfg)
    assert f"settings the code added or removed since config.yaml was written, not a change: {logged}" in caplog.text
    assert history(cfg) == []


@pytest.mark.parametrize("edit, refused", [
    (lambda r: r["parsing"].pop("title_overrides"),                    # added since, but not at its default
     'parsing.title_overrides: (not set) -> "configs/titles.csv"'),
    (lambda r: r["paths"]["corpora"].pop("reports"),                   # a corpus added: an entry of a map
     'paths.corpora.reports: (not set) -> "data/corpora/reports"'),
])
def test_a_new_key_that_changes_what_is_computed_is_refused(cfg, edit, refused):
    save_config(cfg)
    edit_record(cfg, edit)
    with pytest.raises(ConfigChanged, match=re.escape(refused)):
        save_config(cfg)


def test_an_older_absolute_record_is_the_same_config(cfg, caplog):
    """Records written before paths were saved relative hold them absolute: the same paths."""
    save_config(cfg)
    edit_record(cfg, lambda r: r["paths"].update(data_dir=str(REPO_ROOT / "data")))
    with caplog.at_level("INFO"):
        save_config(cfg)
    assert "not a change" not in caplog.text and history(cfg) == []


# --------------------------------------------------------------------------------------
# Secrets
# --------------------------------------------------------------------------------------

def test_empty_secret_is_none():
    s = Secrets(openai_api_key="", ollama_base_url="")
    assert s.openai_api_key is None and s.ollama_base_url is None


def test_secret_is_masked():
    assert "sk-test" not in repr(Secrets(openai_api_key="sk-test"))


def test_missing_env_file_is_not_an_error(monkeypatch):
    for var in ["OPENAI_API_KEY", "HF_TOKEN", "WEAVIATE_API_KEY", "OLLAMA_BASE_URL"]:
        monkeypatch.delenv(var, raising=False)
    assert load_secrets("/nonexistent/.env") == Secrets()


ALL_SECRETS = Secrets(openai_api_key="k", ollama_base_url="u", weaviate_api_key="w")
STAGES = ["parse", "chunk", "embed", "index", "gen-qa", "filter-qa", "retrieve", "answer", "eval", "report"]


@pytest.mark.parametrize("stage", STAGES)
def test_require_secrets_passes_with_all_secrets(cfg, stage):
    require_secrets(stage, cfg, ALL_SECRETS)


@pytest.mark.parametrize("stage, var", [
    ("eval", "OPENAI_API_KEY"),
    ("gen-qa", "OLLAMA_BASE_URL"),
    ("filter-qa", "OLLAMA_BASE_URL"),
    ("answer", "OLLAMA_BASE_URL"),
    ("index", "WEAVIATE_API_KEY"),
    ("retrieve", "WEAVIATE_API_KEY"),
])
def test_require_secrets_names_missing_variable(cfg, stage, var):
    with pytest.raises(SystemExit, match=var):
        require_secrets(stage, cfg, Secrets())


@pytest.mark.parametrize("stage", ["parse", "chunk", "embed", "report"])
def test_offline_stages_need_no_secrets(cfg, stage):
    require_secrets(stage, cfg, Secrets())


def test_secret_follows_provider(cfg):
    """Switching a role to another provider changes which secret its stage needs."""
    cfg.llm.rag_generator.provider = "openai"
    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        require_secrets("answer", cfg, Secrets(ollama_base_url="u"))


# --------------------------------------------------------------------------------------
# collection_name
# --------------------------------------------------------------------------------------

def test_collection_name_format(cfg):
    assert collection_name(cfg, "papers", "sentence", True) == "Papers_sentence_overlap_v2"
    assert collection_name(cfg, "reports", "recursive", False) == "Reports_recursive_no_overlap_v2"


def test_collection_names_are_distinct(cfg):
    names = {collection_name(cfg, c, m, o)
             for c in cfg.paths.corpora for m in cfg.chunking.methods for o in cfg.chunking.overlap}
    expected = len(cfg.paths.corpora) * len(cfg.chunking.methods) * len(cfg.chunking.overlap)
    assert len(names) == expected == 12


@pytest.mark.parametrize("corpus, method", [("papers", "semantic"), ("books", "sentence")])
def test_collection_name_rejects_unknown(cfg, corpus, method):
    with pytest.raises(ValueError, match="unknown"):
        collection_name(cfg, corpus, method, True)
