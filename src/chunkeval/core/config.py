"""Configuration loading: YAML (experiment settings) + .env (secrets).

Why: the old script mixed settings, paths and API keys inline. Here one YAML file
describes an experiment and secrets come only from the environment, so a config can be
committed/shared and a run is reproducible from (config, code version, seed).

Two kinds of input, kept strictly apart:
  - Settings (models, chunk sizes, paths, ...) live in configs/*.yaml. They are validated
    by the pydantic models in `schema.py`, so a typo or a wrong type fails at load
    time instead of silently using a default halfway through a run. A config for another
    experiment can `extends: default.yaml` and hold only what differs (`read_config`).
  - Secrets (API keys) live only in .env or the shell environment, never in YAML, so a
    config file can be committed and shared safely.

Typical use, at the start of every stage (see cli.py):
    cfg = load_config(args.config)              # validated `Config` object
    secrets = load_secrets()                    # keys from .env, masked when printed
    require_secrets(args.command, cfg, secrets) # stop now if this stage lacks a key
    save_config(cfg)                            # record what this run used; refused if it changed

Access settings with attributes, not dict keys: `cfg.embedding.primary`, `cfg.llm.judge.model`.

Reads:  configs/*.yaml, .env, runs/<run_id>/config.yaml
Used by: every stage (via cli.py).
"""
from __future__ import annotations

from datetime import datetime
import json
import logging
from pathlib import Path
import os
import shutil
from typing import Any, NamedTuple
from dotenv import load_dotenv
from pydantic import BaseModel, SecretStr, field_validator
from pydantic_core import to_jsonable_python
import yaml

from chunkeval.core.schema import REPO_ROOT, Config

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Secrets
# --------------------------------------------------------------------------------------

class Secrets(BaseModel):
    """Secrets loaded from the environment — never from YAML.

    Keys are `SecretStr`: printing, logging or a traceback shows '**********'. Read the real
    value only where a client is built: `secrets.openai_api_key.get_secret_value()`.

    Every field is optional here, because no single stage needs all of them (`chunk` needs
    none). Which ones are mandatory is decided per stage by `require_secrets`.
    Field names match the environment variable names in lower case; `require_secrets`
    relies on that to print the variable name in its error message.
    """

    openai_api_key: SecretStr | None = None
    hf_token: SecretStr | None = None       # optional: models are public; huggingface_hub reads HF_TOKEN from the env itself
    weaviate_api_key: SecretStr | None = None
    ollama_base_url: str | None = None      # not secret: OpenAI-compatible endpoint, e.g. http://localhost:11434/v1

    @field_validator("*", mode="before")
    @classmethod
    def empty_is_missing(cls, value):
        # Runs for every field ("*") on the raw value from os.getenv, before the conversion
        # to SecretStr ("before"). `KEY=` in .env gives "" — turn it into None so that
        # "unset" and "empty" mean the same thing and a missing key is always None.
        return value or None


def load_secrets(env_file: str | Path = ".env") -> Secrets:
    """Populate `Secrets` from .env / environment variables.

    A relative `env_file` is resolved against the repo root, so this works from any working
    directory. A missing .env file is not an error: the values then come from the shell
    environment only. Variables already set in the shell take precedence over .env
    (python-dotenv's default, override=False).

    This function never fails on a missing key — that check belongs to `require_secrets`,
    which knows which stage is running.
    """
    env_file = Path(env_file)
    if not env_file.is_absolute():
        env_file = REPO_ROOT / env_file

    # Copies the .env entries into os.environ. Side effect worth knowing: libraries that
    # read their own variables (huggingface_hub reads HF_TOKEN) pick them up from there too.
    load_dotenv(env_file)

    # os.getenv returns None for a variable that isn't set; empty strings are turned into
    # None by `Secrets.empty_is_missing`.
    return Secrets(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        hf_token=os.getenv("HF_TOKEN"),
        weaviate_api_key=os.getenv("WEAVIATE_API_KEY"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL"),
    )


# Which secrets each CLI stage needs. Stages not listed here need none.
#
# Fixed requirements, independent of the config:
# Weaviate runs with API-key auth (docker-compose.yml), so the stages that talk to it need the key.
STAGE_SECRETS = {
    "index": ["weaviate_api_key"],
    "retrieve": ["weaviate_api_key"],
    "remove-corpus": ["weaviate_api_key"],
}
# Requirements that depend on the config: the LLM roles (keys of the `llm:` section in the
# YAML) each stage calls. The secret then follows from that role's `provider`, so switching
# a role from "openai" to "ollama" in the YAML changes the requirement without code changes.
STAGE_LLM_ROLES = {
    "gen-qa": ["qa_generator"],
    "filter-qa": ["qa_critic"],
    "answer": ["rag_generator"],
    "eval": ["judge"],
}
# The `Secrets` field each provider needs.
PROVIDER_SECRET = {"ollama": "ollama_base_url", "openai": "openai_api_key"}


def require_secrets(stage: str, cfg: Config, secrets: Secrets) -> None:
    """Fail fast if a secret this stage needs is missing.

    Call it before the stage does any work, so a missing key stops the run in the first
    second instead of at the first API call an hour in.

    Raises `SystemExit` (message on stderr, exit code 1, no traceback) listing the missing
    variables, e.g. "Stage 'eval' needs OPENAI_API_KEY in .env".
    """
    # Fixed requirements of the stage (empty set for stages not in the table).
    needed = set(STAGE_SECRETS.get(stage, []))

    # Plus one secret per LLM role the stage calls: role -> its provider in the config ->
    # the secret that provider needs. A set, so two roles on the same provider count once.
    # getattr(cfg.llm, "judge") is the same as cfg.llm.judge, with the role name as a string.
    needed |= {PROVIDER_SECRET[getattr(cfg.llm, role).provider]
               for role in STAGE_LLM_ROLES.get(stage, [])}

    # Keep the ones that are None. sorted() keeps the error message order stable;
    # .upper() turns the field name back into the environment variable name.
    missing = [name.upper() for name in sorted(needed) if not getattr(secrets, name)]
    if missing:
        raise SystemExit(f"Stage '{stage}' needs {', '.join(missing)} in .env")


# --------------------------------------------------------------------------------------
# Experiment config (YAML)
# --------------------------------------------------------------------------------------

def merge_config(base: dict, override: dict) -> dict:
    """`override` on top of `base`, for `extends`: mappings merge key by key, at any depth; a list
    or a value replaces the base's; `null` removes the key, so the field's default applies
    (e.g. `title_overrides: null`, or `reports: null` under paths.corpora to leave a corpus out)."""
    merged = dict(base)
    for key, value in override.items():
        if value is None:
            merged.pop(key, None)
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def read_config(path: Path, seen: tuple[Path, ...] = ()) -> dict:
    """The YAML file as a plain dict, with its `extends` chain applied: `extends: default.yaml`
    (relative to the file's own folder) reads that file first and puts this one on top
    (`merge_config`), so a config for one experiment holds only what differs."""
    if path in seen:
        raise ValueError(f"config extends itself: {' -> '.join(map(str, (*seen, path)))}")
    # safe_load builds only plain Python types (dict, list, str, int, float, bool, None) and
    # never runs arbitrary constructors. Comments in the YAML are dropped here.
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}      # an empty file gives None
    base = raw.pop("extends", None)
    if base is None:
        return raw
    return merge_config(read_config((path.parent / base).resolve(), (*seen, path)), raw)


def load_config(path: str | Path) -> Config:
    """Load the YAML config (with its `extends` chain, see `read_config`) and validate it against
    `schema.Config`.

    A relative `path` is resolved against the repo root, like the paths inside the config,
    so the result doesn't depend on the current working directory. Invalid configs raise
    `pydantic.ValidationError` listing every bad field.

    What validation covers (see schema.py): unknown or missing keys, wrong types,
    value ranges, allowed values (e.g. provider, chunking method), cross-references between
    sections (e.g. retrieval.embedders must exist in embedding.models) and leftover "TODO"
    placeholder values. Paths in the returned object are absolute `Path`s.
    """
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    raw = read_config(path.resolve())

    # model_validate is inherited from pydantic.BaseModel: it checks the plain dict against
    # the Config fields, converts nested dicts into the section models, runs the validators,
    # and returns a typed Config object (or raises one ValidationError listing every problem).
    return Config.model_validate(raw)


# --------------------------------------------------------------------------------------
# The run's record of its config, and the guard on it
# --------------------------------------------------------------------------------------

# Settings that can't change a result, only how fast or how far a stage goes, what `eval`
# estimates it costs, or where the run folders are: changing one under the same run_id changes
# nothing the run measured.
UNGUARDED = ("paths.runs_dir", "ollama.concurrency", "ragas.max_workers", "ragas.timeout", "ragas.max_retries",
             "ragas.max_cost_usd", "ragas.prices")


class ConfigChanged(RuntimeError):
    """The config differs from the one this run_id recorded (`save_config` refuses)."""


class ConfigDiff(NamedTuple):
    """How a config differs from a run's record, one "key: old -> new" line per setting."""
    changed: list[str]       # can change a result: refused unless accepted
    unguarded: list[str]     # UNGUARDED settings
    code: list[str]          # settings the code added or removed since the record was written


def portable(value: Any) -> Any:
    """The dumped config with every path under the repo written relative to it, so the record
    doesn't depend on where the repo is (relative paths load the same: schema.py resolves them
    against the repo root)."""
    if isinstance(value, dict):
        return {k: portable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [portable(v) for v in value]
    if isinstance(value, str) and value.startswith(f"{REPO_ROOT}/"):
        return value[len(f"{REPO_ROOT}/"):]
    return value


def flatten(tree: dict, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    """Nested mappings as {(section, ..., key): value}; a list is one value, an empty mapping none.
    Tuples, not dotted strings: a key can hold dots ("gpt-4.1-mini" in ragas.prices)."""
    flat: dict[tuple[str, ...], Any] = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            flat |= flatten(value, (*prefix, key))
        else:
            flat[(*prefix, key)] = value
    return flat


NO_DEFAULT = object()


def section_field(cfg: Config, key: tuple[str, ...]) -> tuple[bool, Any]:
    """Whether `key` is (or was) a field of a config section, one the code defines, rather than an
    entry of a map the config fills in (a corpus in paths.corpora, a model in embedding.models);
    and its default as the record would hold it, or NO_DEFAULT (required, or removed since)."""
    node: Any = cfg
    field = None
    for part in key:
        if isinstance(node, BaseModel):
            field = type(node).model_fields.get(part)
            if field is None:
                return True, NO_DEFAULT        # a field this code doesn't have: removed since
            node = getattr(node, part)
        elif isinstance(node, dict) and part in node:
            node, field = node[part], None
        else:
            return False, NO_DEFAULT           # an entry missing from a map
    if field is None:
        return False, NO_DEFAULT
    if field.is_required():
        return True, NO_DEFAULT
    return True, portable(to_jsonable_python(field.get_default(call_default_factory=True)))


def run_config_changes(cfg: Config, recorded: dict) -> ConfigDiff:
    """How `cfg` differs from the config a run recorded (its runs/<run_id>/config.yaml).

    A key only one side has is a setting the code added or removed since the record was written,
    when it is a field of a section: not a change, because a new setting keeps the old behaviour,
    unless it has a default and is set to something else. An entry of a map the config fills in (a
    corpus added to paths.corpora) is a change like any other. Paths are compared relative to the
    repo root (`portable`)."""
    old, new = flatten(portable(recorded)), flatten(portable(cfg.model_dump(mode="json")))
    diff = ConfigDiff([], [], [])
    for key in sorted(old.keys() | new.keys()):
        if key in old and key in new and old[key] == new[key]:
            continue
        dotted = ".".join(key)
        line = f"{dotted}: {shown_value(old, key)} -> {shown_value(new, key)}"
        if any(dotted == u or dotted.startswith(f"{u}.") for u in UNGUARDED):
            diff.unguarded.append(line)
            continue
        is_field, default = section_field(cfg, key) if key not in old or key not in new else (False, NO_DEFAULT)
        if is_field and (key not in new or default is NO_DEFAULT or new[key] == default):
            diff.code.append(line)
        else:
            diff.changed.append(line)
    return diff


def shown_value(flat: dict, key: tuple[str, ...]) -> str:
    return json.dumps(flat[key], ensure_ascii=False) if key in flat else "(not set)"


def save_config(cfg: Config, accept: bool = False) -> Path:
    """Write the resolved config to runs/<run_id>/config.yaml — call at the start of every stage.

    "Resolved" means after validation: defaults filled in, every path absolute, and written relative
    to the repo root when under it (`portable`). Together with the code version and the seed, this
    file is what makes a run reproducible — it records what the run actually used, even if
    configs/default.yaml is edited later.

    Guarded: when the run recorded a config before and a setting that can change a result differs
    (`run_config_changes`), it raises ConfigChanged before anything is written, so the stages never
    replace a run's results with ones made under other settings unnoticed. Bump run_id for a new
    experiment, or `accept` (the CLI's --accept-config) to change this run: the old record is then
    kept as runs/<run_id>/config_history/config_<when it was replaced>.yaml. Settings the code added
    or removed, and UNGUARDED ones, are only logged.
    """
    out = cfg.paths.runs_dir / cfg.run_id / "config.yaml"
    if out.exists():
        diff = run_config_changes(cfg, yaml.safe_load(out.read_text(encoding="utf-8")) or {})
        if diff.code:
            log.info("settings the code added or removed since %s was written, not a change: %s",
                     out.name, "; ".join(diff.code))
        if diff.unguarded:
            log.info("changed, but can't change a result: %s", "; ".join(diff.unguarded))
        if diff.changed and not accept:
            raise ConfigChanged(
                f"the config differs from the one run {cfg.run_id} recorded ({shown_path(out)}):\n"
                + "".join(f"    {line}\n" for line in diff.changed)
                + "The stages would replace this run's results with ones made under these settings. Bump run_id "
                  "for a new experiment, or re-run with --accept-config to change this run (its old config.yaml "
                  "is then kept in config_history/)")
        if diff.changed:
            history = out.parent / "config_history" / f"config_{datetime.now():%Y%m%d-%H%M%S}.yaml"
            history.parent.mkdir(exist_ok=True)
            shutil.copy2(out, history)
            log.warning("run %s now records a changed config (--accept-config): %s; the old one is %s",
                        cfg.run_id, "; ".join(diff.changed), shown_path(history))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        # mode="json" turns Path objects back into strings so yaml.safe_dump can write them;
        # sort_keys=False keeps the section order of the original YAML.
        yaml.safe_dump(portable(cfg.model_dump(mode="json")), f, sort_keys=False, allow_unicode=True)
    return out


def shown_path(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def collection_name(cfg: Config, corpus: str, method: str, overlap: bool) -> str:
    """Build the Weaviate collection name from `weaviate.collection_template`.

    Returns the name the way Weaviate stores it — first letter capitalised — so comparisons
    with names Weaviate returns match: ("papers", "sentence", True) -> "Papers_sentence_overlap_v2".

    Every stage that touches a collection (index creates them, retrieve queries them) must get
    the name from here, so the naming rule exists in one place. The template's "_v2" suffix
    keeps these collections apart from the previous round's, which must not be overwritten.
    `{namespace}` is "" for the repo's data/ and "<namespace>_" for any other data dir
    (`Config.collection_namespace`): data/size512 gives "Size512_papers_sentence_overlap_v2".

    Raises `ValueError` for a corpus or method that isn't in the config, so a typo fails with
    a clear message instead of querying a collection that doesn't exist.
    """
    if corpus not in cfg.paths.corpora:
        raise ValueError(f"unknown corpus '{corpus}'; configured: {list(cfg.paths.corpora)}")
    if method not in cfg.chunking.methods:
        raise ValueError(f"unknown method '{method}'; configured: {cfg.chunking.methods}")
    return format_collection_name(cfg, corpus, method, overlap)


def format_collection_name(cfg: Config, corpus: str, method: str, overlap: bool) -> str:
    """`collection_name` without its checks that the corpus and method are configured: for the
    collections of a corpus already taken out of the config (`remove-corpus`)."""
    # The template is a plain string from the YAML, e.g. "{corpus}_{method}_{overlap}_v2";
    # str.format fills each {placeholder} with the keyword argument of the same name.
    # The placeholders themselves are checked when the config loads (Weaviate model).
    # Overlap words: same as the analysis.baseline check in schema.py.
    namespace = cfg.collection_namespace
    name = cfg.weaviate.collection_template.format(
        namespace=f"{namespace}_" if namespace else "",
        corpus=corpus, method=method, overlap="overlap" if overlap else "no_overlap"
    )
    return name[0].upper() + name[1:]   # not str.capitalize(): it would lowercase the rest
