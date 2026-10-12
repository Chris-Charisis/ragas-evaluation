"""core/config.py + core/schema.py — loading, validation, save round trip, secrets, naming.

All offline and fast. A config error caught here is one that would otherwise surface hours
into a run (or never, as a silently wrong setting).
"""
import pytest
import yaml
from pydantic import ValidationError

from chunkeval.core.config import (Secrets, collection_name, load_config, load_secrets,
                              require_secrets, save_config)
from chunkeval.core.schema import Config


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------

def test_default_config_loads(cfg):
    assert isinstance(cfg, Config)


def test_relative_path_works_from_any_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_config("configs/default.yaml").run_id


def test_paths_are_absolute(cfg):
    paths = [cfg.paths.data_dir, cfg.embedding.cache_dir, *cfg.paths.corpora.values()]
    assert all(p.is_absolute() for p in paths)


def test_reasoning_effort_parsed(cfg):
    assert cfg.llm.qa_generator.reasoning_effort == "none"
    assert cfg.llm.rag_generator.reasoning_effort == "none"
    assert cfg.llm.qa_critic.reasoning_effort == "low"
    assert cfg.llm.judge.reasoning_effort is None        # non-reasoning model: never sent


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
