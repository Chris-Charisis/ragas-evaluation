"""Command-line entry point: one subcommand per stage.

Usage:  uv run chunkeval <command> [options] [--config configs/default.yaml] [-v]
   or:  uv run python -m chunkeval.cli <command> …
(`chunkeval` is the console script declared in pyproject.toml's [project.scripts].)
Examples:
        uv run chunkeval parse --corpus papers --limit 2      # smoke test on 2 PDFs
        uv run chunkeval parse                                # every corpus in the config

What every stage gets from here, so stage code doesn't repeat it:
  1. config validated, secrets checked for this stage (fail in the first second);
  2. the resolved config saved to runs/<run_id>/config.yaml (what this run actually used);
  3. logging to the console and to runs/<run_id>/logs/<stage>_<timestamp>.log;
  4. for stages with local LLM roles: `check_ollama` before, and the models unloaded after
     (even when the stage fails), so the GPU is free for the next stage;
  5. the stage's `run(cfg, secrets, **options)`, timed.

Adding a stage option: register it in OPTIONS, list it in the stage's `options`, and add the
keyword argument (same name) to that stage's `run`. tests/test_cli.py checks that they match.

Uses stdlib argparse so it works before any new dependency is installed.
"""
from __future__ import annotations

import argparse
import importlib
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from chunkeval.core import config
from chunkeval.core.schema import Config
from chunkeval.core.store import run_dir

log = logging.getLogger("chunkeval")


@dataclass(frozen=True)
class Stage:
    help: str
    targets: tuple[str, ...]          # "module:function", called in this order
    options: tuple[str, ...] = ()     # keys of OPTIONS this stage accepts


STAGES: dict[str, Stage] = {
    "parse": Stage("Stage 1: PDFs -> CanonicalDoc JSON (Docling)",
                   ("chunkeval.corpus.parsing:run",), ("corpora", "limit")),
    "audit": Stage("Stage 1b: audit the parsed corpus before chunking (OCR, body, titles, garbled text, "
                   "languages, duplicates, size); flags only, changes nothing",
                   ("chunkeval.corpus.audit:run",), ("corpora",)),
    "chunk": Stage("Stage 2: CanonicalDoc -> chunks for every (method, overlap) + size stats",
                   ("chunkeval.corpus.chunking:run",), ("corpora", "limit")),
    "embed": Stage("Stage 2b: embed chunks with bge-m3 (primary) and Qwen3-Embedding (secondary), cached (GPU)",
                   ("chunkeval.corpus.embedding:run",), ("corpora", "embedders")),
    "index": Stage("Stage 3: load chunks + both named vectors into the v2 Weaviate collections",
                   ("chunkeval.corpus.indexing:run",), ("corpora", "limit", "recreate")),
    "gen-qa": Stage("Stage 4: generate Q-A pairs with located evidence spans",
                    ("chunkeval.qa.generation:run",), ("corpora", "limit", "recreate")),
    "filter-qa": Stage("Stage 5: deterministic checks + critic + dedup + final sample",
                       ("chunkeval.qa.filtering:run",)),
    "retrieve": Stage("Stage 6: top-k retrieval per question, collection and embedder + span metrics",
                      ("chunkeval.evaluation.retrieval:run", "chunkeval.evaluation.retrieval_metrics:run"),
                      ("corpora", "embedders")),
    "answer": Stage("Stage 7: RAG answers from the retrieved chunks and the gold passage (fixed generator)",
                    ("chunkeval.evaluation.answering:run",), ("corpora",)),
    "eval": Stage("Stage 8: Ragas metrics (OpenAI judge, cost-capped) + judge-free text metrics per question and setup",
                  ("chunkeval.evaluation.ragas_eval:run",), ("corpora", "pilot", "dry_run")),
    "report": Stage("Stage 9: tables with bootstrap CIs, paired diffs, breakdowns",
                    ("chunkeval.analysis:run",)),
}


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


# Option name = keyword argument of the stage's `run`.
OPTIONS: dict[str, Callable[[argparse.ArgumentParser], Any]] = {
    "corpora": lambda p: p.add_argument(
        "--corpus", dest="corpora", action="append", metavar="NAME",
        help="only this corpus (repeatable); default: every corpus in the config"),
    "limit": lambda p: p.add_argument(
        "--limit", type=_positive_int, metavar="N",
        help="smoke test: only the first N documents per corpus"),
    "embedders": lambda p: p.add_argument(
        "--embedder", dest="embedders", action="append", metavar="NAME",
        help="only this embedder (repeatable); default: embed: embedding.primary, then embedding.secondary; "
             "retrieve: retrieval.embedders"),
    "recreate": lambda p: p.add_argument(
        "--recreate", action="store_true",
        help="delete what this stage built before and build it again; default: reuse what is up to "
             "date, refuse what is stale (index: the selected collections; gen-qa: all Q-A records)"),
    "pilot": lambda p: p.add_argument(
        "--pilot", type=_positive_int, metavar="N",
        help="only the first N questions of each corpus's Ragas sample, to measure the cost per question; "
             "a later full run reuses what the pilot paid for"),
    "dry_run": lambda p: p.add_argument(
        "--dry-run", action="store_true",
        help="estimate the judge's cost of everything not cached yet, then stop: no API call, nothing written"),
}


def build_parser() -> argparse.ArgumentParser:
    # Options every stage has; `parents=` puts them after the subcommand: `chunkeval parse -v`.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="configs/default.yaml",
                        help="experiment config, relative to the repo root (default: %(default)s)")
    common.add_argument("-v", "--verbose", action="store_true", help="debug logging")

    parser = argparse.ArgumentParser(prog="chunkeval", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")
    for name, stage in STAGES.items():
        p = sub.add_parser(name, help=stage.help, description=stage.help, parents=[common])
        for option in stage.options:
            OPTIONS[option](p)
    return parser


def stage_options(stage: Stage, args: argparse.Namespace, cfg: Config) -> dict[str, Any]:
    """The keyword arguments for the stage's `run`, with defaults filled from the config."""
    options = {name: getattr(args, name) for name in stage.options}
    if "corpora" in options:
        # dict.fromkeys: drop repeats, keep the order given.
        corpora = list(dict.fromkeys(options["corpora"] or cfg.paths.corpora))
        unknown = [c for c in corpora if c not in cfg.paths.corpora]
        if unknown:
            raise SystemExit(f"unknown corpus {unknown}; configured: {list(cfg.paths.corpora)}")
        options["corpora"] = corpora
    if "embedders" in options:
        # embed: every model; retrieve: the named vectors the config queries
        default = (cfg.retrieval.embedders if args.command == "retrieve"
                   else [cfg.embedding.primary, *cfg.embedding.secondary])
        embedders = list(dict.fromkeys(options["embedders"] or default))
        unknown = [e for e in embedders if e not in cfg.embedding.models]
        if unknown:
            raise SystemExit(f"unknown embedder {unknown}; configured: {list(cfg.embedding.models)}")
        options["embedders"] = embedders
    return options


def resolve(target: str) -> Callable[..., None]:
    """Import "package.module:function" now — not at start-up, so e.g. `chunk` never loads
    docling, openai or ragas."""
    module, _, function = target.partition(":")
    return getattr(importlib.import_module(module), function)


def ollama_models(command: str, cfg: Config) -> list[str]:
    """Local models this stage's LLM roles use (deduplicated; empty for non-LLM stages)."""
    roles = config.STAGE_LLM_ROLES.get(command, [])
    return list(dict.fromkeys(getattr(cfg.llm, r).model for r in roles if getattr(cfg.llm, r).provider == "ollama"))


LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format=LOG_FORMAT, datefmt="%H:%M:%S")
    logging.getLogger().setLevel(level)        # also when basicConfig was a no-op (handlers existed)
    # The HTTP clients log every request at INFO: thousands of lines in an LLM stage.
    for noisy in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def add_log_file(cfg: Config, command: str) -> logging.FileHandler:
    path = run_dir(cfg, "logs") / f"{command}_{datetime.now():%Y%m%d-%H%M%S}.log"
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(handler)
    return handler


def _elapsed(start: float) -> str:
    seconds = time.monotonic() - start
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{seconds:02d}s" if hours else f"{minutes}m{seconds:02d}s"


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    stage = STAGES[args.command]
    setup_logging(args.verbose)

    # 1. Everything that can fail fast, before any work.
    cfg = config.load_config(args.config)
    secrets = config.load_secrets()
    config.require_secrets(args.command, cfg, secrets)
    options = stage_options(stage, args, cfg)
    functions = [resolve(target) for target in stage.targets]

    # 2.–3. Record what this run used, and keep a log file next to it.
    file_handler = add_log_file(cfg, args.command)
    saved = config.save_config(cfg)
    log.info("stage %s | run_id %s | options %s | config saved to %s",
             args.command, cfg.run_id, options, saved)

    # 4.–5. Preflight, run, always free the GPU.
    models = ollama_models(args.command, cfg)
    start = time.monotonic()
    try:
        if models:
            from chunkeval.core.llm import check_ollama
            check_ollama(cfg, secrets.ollama_base_url)
        for function in functions:
            function(cfg, secrets, **options)
    except KeyboardInterrupt:
        log.warning("stage %s interrupted after %s", args.command, _elapsed(start))
        raise SystemExit(130) from None        # the usual exit code for Ctrl+C
    except Exception:
        # The traceback goes to the console AND the log file, once; then a plain exit code 1.
        log.exception("stage %s failed after %s", args.command, _elapsed(start))
        raise SystemExit(1) from None
    else:
        log.info("stage %s done in %s", args.command, _elapsed(start))
    finally:
        if models:
            from chunkeval.core.llm import unload_ollama_model
            for model in models:
                try:
                    unload_ollama_model(secrets.ollama_base_url, model)
                except Exception as e:         # never hide the stage's own error behind this one
                    log.warning("could not unload %s: %s", model, e)
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()


if __name__ == "__main__":
    main()
