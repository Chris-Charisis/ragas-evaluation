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
    save_config(cfg)                            # record what this run used

Access settings with attributes, not dict keys: `cfg.embedding.primary`, `cfg.llm.judge.model`.

Reads:  configs/*.yaml, .env
Used by: every stage (via cli.py).
"""
from __future__ import annotations

from pathlib import Path
import os
from dotenv import load_dotenv
from pydantic import BaseModel, SecretStr, field_validator
import yaml

from chunkeval.core.schema import REPO_ROOT, Config


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


def save_config(cfg: Config) -> Path:
    """Write the resolved config to runs/<run_id>/config.yaml — call at the start of every stage.

    "Resolved" means after validation: absolute paths, defaults filled in. Together with the
    code version and the seed, this file is what makes a run reproducible — it records what
    the run actually used, even if configs/default.yaml is edited later.
    Overwritten by each stage of the same run_id; bump run_id for a new experiment.
    """
    out = cfg.paths.runs_dir / cfg.run_id / "config.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        # mode="json" turns Path objects back into strings so yaml.safe_dump can write them;
        # sort_keys=False keeps the section order of the original YAML.
        yaml.safe_dump(cfg.model_dump(mode="json"), f, sort_keys=False, allow_unicode=True)
    return out


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
