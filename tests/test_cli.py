"""cli.py — argument parsing, the stage table, and what `main` does around every stage.

Offline: stage functions are replaced by fakes, Ollama calls by stubs.
"""
import inspect
from pathlib import Path

import pytest
import yaml

from chunkeval import cli
from chunkeval.core.config import Secrets
from chunkeval.core.schema import Config


@pytest.fixture
def config_file(tmp_path, raw_config) -> Path:
    """The default config with runs_dir in a temp dir, as a file for `--config`."""
    raw_config["paths"]["runs_dir"] = str(tmp_path / "runs")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")
    return path


@pytest.fixture
def calls(monkeypatch):
    """Replace every stage function with a fake that records its arguments."""
    recorded: list[dict] = []

    def fake_resolve(target):
        def fake_run(cfg, secrets, **options):
            recorded.append({"target": target, "cfg": cfg, "secrets": secrets, "options": options})
        return fake_run
    monkeypatch.setattr(cli, "resolve", fake_resolve)
    return recorded


@pytest.fixture
def ollama(monkeypatch):
    """Stub out the Ollama preflight and unload; record what was called."""
    from chunkeval.core import llm
    events: list[tuple] = []
    monkeypatch.setattr(llm, "check_ollama", lambda cfg, url, roles: events.append(("check", tuple(roles))))
    monkeypatch.setattr(llm, "unload_ollama_model", lambda url, model: events.append(("unload", model)))
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    return events


# --------------------------------------------------------------------------------------
# The stage table must match the stage code
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("command", cli.STAGES)
def test_stage_targets_accept_the_cli_options(command, cfg):
    """Every target imports, and its `run` accepts (cfg, secrets, **declared options).

    Fails when an option is added to the CLI but not to the stage's `run` (or the reverse
    for required parameters) — before anyone runs the stage for real.
    """
    stage = cli.STAGES[command]
    options = {"corpora": ["papers"], "limit": 1, "embedders": ["bge_m3"], "recreate": True, "pilot": 2,
               "dry_run": True, "corpus": "reports", "yes": True}
    for target in stage.targets:
        function = cli.resolve(target)
        inspect.signature(function).bind(cfg, Secrets(), **{o: options[o] for o in stage.options})


def test_every_stage_option_is_registered():
    assert {o for s in cli.STAGES.values() for o in s.options} <= set(cli.OPTIONS)


def test_llm_stages_match_config_roles():
    """Stages with LLM roles (config.STAGE_LLM_ROLES) must exist in the CLI."""
    from chunkeval.core.config import STAGE_LLM_ROLES
    assert set(STAGE_LLM_ROLES) <= set(cli.STAGES)


# --------------------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------------------

def test_options_only_where_declared():
    args = cli.build_parser().parse_args(["parse", "--corpus", "papers", "--limit", "2", "-v"])
    assert (args.corpora, args.limit, args.verbose) == (["papers"], 2, True)
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["report", "--corpus", "papers"])    # report has no --corpus
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["chunk", "--limit", "2"])           # it would replace the corpus's chunk files


@pytest.mark.parametrize("value", ["0", "-1", "two"])
def test_limit_must_be_positive_int(value):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["parse", "--limit", value])


def test_default_corpora_are_all_configured(cfg):
    args = cli.build_parser().parse_args(["parse"])
    assert cli.stage_options(cli.STAGES["parse"], args, cfg) == {"corpora": list(cfg.paths.corpora), "limit": None}


@pytest.mark.parametrize("stage", ["index", "gen-qa"])
def test_recreate_is_off_unless_given(cfg, stage):
    for argv, recreate in (([stage], False), ([stage, "--recreate"], True)):
        args = cli.build_parser().parse_args(argv)
        assert cli.stage_options(cli.STAGES[stage], args, cfg)["recreate"] is recreate


def test_repeated_corpus_deduplicated_in_order(cfg):
    args = cli.build_parser().parse_args(["chunk", "--corpus", "reports", "--corpus", "papers", "--corpus", "reports"])
    assert cli.stage_options(cli.STAGES["chunk"], args, cfg)["corpora"] == ["reports", "papers"]


def test_unknown_corpus_rejected(cfg):
    args = cli.build_parser().parse_args(["parse", "--corpus", "books"])
    with pytest.raises(SystemExit, match="unknown corpus"):
        cli.stage_options(cli.STAGES["parse"], args, cfg)


def test_default_embedders_primary_first(cfg):
    args = cli.build_parser().parse_args(["embed"])
    assert cli.stage_options(cli.STAGES["embed"], args, cfg)["embedders"] == [cfg.embedding.primary, *cfg.embedding.secondary]
    args = cli.build_parser().parse_args(["embed", "--embedder", "qwen3_06b", "--embedder", "qwen3_06b"])
    assert cli.stage_options(cli.STAGES["embed"], args, cfg)["embedders"] == ["qwen3_06b"]


def test_unknown_embedder_rejected(cfg):
    args = cli.build_parser().parse_args(["embed", "--embedder", "bge-m3"])
    with pytest.raises(SystemExit, match="unknown embedder"):
        cli.stage_options(cli.STAGES["embed"], args, cfg)


def test_ollama_models_per_stage(cfg):
    assert cli.ollama_models("parse", cfg) == []
    assert cli.ollama_models("gen-qa", cfg) == [cfg.llm.qa_generator.model]
    assert cli.ollama_models("eval", cfg) == []              # judges are OpenAI
    cfg.llm.judge.provider = "ollama"
    assert cli.ollama_models("eval", cfg) == [cfg.llm.judge.model]


# --------------------------------------------------------------------------------------
# main: what happens around every stage
# --------------------------------------------------------------------------------------

def test_main_runs_stage_saves_config_and_logs(config_file, calls, tmp_path):
    cli.main(["parse", "--config", str(config_file), "--limit", "2"])

    assert len(calls) == 1
    call = calls[0]
    assert call["target"] == "chunkeval.corpus.parsing:run"
    assert isinstance(call["cfg"], Config) and isinstance(call["secrets"], Secrets)
    assert call["options"] == {"corpora": ["papers", "reports"], "limit": 2}

    run = tmp_path / "runs" / call["cfg"].run_id
    assert (run / "config.yaml").exists()
    logs = list((run / "logs").glob("parse_*.log"))
    assert len(logs) == 1 and "stage parse done" in logs[0].read_text(encoding="utf-8")


def test_main_runs_all_targets_in_order(config_file, calls):
    cli.main(["retrieve", "--config", str(config_file)])
    assert [c["target"] for c in calls] == list(cli.STAGES["retrieve"].targets)
    cfg = calls[0]["cfg"]
    assert all(c["options"] == {"corpora": list(cfg.paths.corpora), "embedders": cfg.retrieval.embedders}
               for c in calls)                                   # the same options for both targets


def test_retrieve_embedders_default_to_the_retrieval_config(cfg):
    cfg.retrieval.embedders = ["bge_m3"]
    args = cli.build_parser().parse_args(["retrieve"])
    assert cli.stage_options(cli.STAGES["retrieve"], args, cfg)["embedders"] == ["bge_m3"]
    args = cli.build_parser().parse_args(["embed"])                # embed still embeds every model
    assert cli.stage_options(cli.STAGES["embed"], args, cfg)["embedders"] == ["bge_m3", "qwen3_06b"]


def test_main_stage_failure_exits_1_and_logs_traceback(config_file, monkeypatch, tmp_path):
    def broken(cfg, secrets, **options):
        raise ValueError("boom")
    monkeypatch.setattr(cli, "resolve", lambda target: broken)
    with pytest.raises(SystemExit) as e:
        cli.main(["parse", "--config", str(config_file)])
    assert e.value.code == 1
    log = next((tmp_path / "runs").rglob("parse_*.log")).read_text(encoding="utf-8")
    assert "stage parse failed" in log and "ValueError: boom" in log


def test_main_ctrl_c_exits_130(config_file, monkeypatch):
    def interrupted(cfg, secrets, **options):
        raise KeyboardInterrupt
    monkeypatch.setattr(cli, "resolve", lambda target: interrupted)
    with pytest.raises(SystemExit) as e:
        cli.main(["parse", "--config", str(config_file)])
    assert e.value.code == 130


def test_main_missing_secret_stops_before_any_work(config_file, calls, monkeypatch):
    monkeypatch.setattr(cli.config, "load_secrets", lambda: Secrets())
    with pytest.raises(SystemExit, match="OLLAMA_BASE_URL"):
        cli.main(["gen-qa", "--config", str(config_file)])
    assert calls == []


def test_main_llm_stage_preflight_and_unload(config_file, calls, ollama):
    cli.main(["gen-qa", "--config", str(config_file), "--corpus", "papers"])
    assert ollama[0] == ("check", ("qa_generator",))         # preflight before the stage, on its own model only
    assert calls and calls[0]["options"] == {"corpora": ["papers"], "limit": None, "recreate": False}
    assert ollama[-1][0] == "unload"                         # GPU freed after it


def test_main_unloads_even_when_stage_fails(config_file, monkeypatch, ollama):
    def broken(cfg, secrets, **options):
        raise RuntimeError("generation crashed")
    monkeypatch.setattr(cli, "resolve", lambda target: broken)
    with pytest.raises(SystemExit):
        cli.main(["gen-qa", "--config", str(config_file)])
    assert any(event == "unload" for event, _ in ollama)


def test_main_refuses_a_changed_config_before_any_work(config_file, calls, tmp_path):
    """Exit code 1, the stage not called, the refusal in the log; --accept-config goes ahead."""
    cli.main(["chunk", "--config", str(config_file)])
    raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    raw["retrieval"]["k_max"] = 10
    config_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        cli.main(["retrieve", "--config", str(config_file)])
    assert e.value.code == 1 and len(calls) == 1
    run = tmp_path / "runs" / raw["run_id"]
    log = next((run / "logs").glob("retrieve_*.log")).read_text(encoding="utf-8")
    assert "retrieval.k_max: 20 -> 10" in log and "stage retrieve" not in log

    cli.main(["retrieve", "--config", str(config_file), "--accept-config"])
    assert len(calls) == 3 and len(list((run / "config_history").glob("config_*.yaml"))) == 1
    with pytest.raises(SystemExit):                                      # no run record to guard
        cli.build_parser().parse_args(["remove-corpus", "reports", "--accept-config"])


def test_remove_corpus_needs_its_name_and_leaves_the_run_record_alone(config_file, calls, monkeypatch, tmp_path):
    """remove-corpus takes the corpus as an argument (no default: it never means "every corpus"),
    previews unless --yes, and doesn't overwrite a run's config.yaml."""
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["remove-corpus"])
    monkeypatch.setattr(cli.config, "load_secrets", lambda: Secrets(weaviate_api_key="w"))
    cli.main(["remove-corpus", "reports", "--config", str(config_file)])
    assert calls[0]["options"] == {"corpus": "reports", "yes": False}
    run = tmp_path / "runs" / calls[0]["cfg"].run_id
    assert not (run / "config.yaml").exists() and list((run / "logs").glob("remove-corpus_*.log"))


def test_main_non_llm_stage_never_touches_ollama(config_file, calls, ollama):
    cli.main(["chunk", "--config", str(config_file)])
    assert ollama == []
