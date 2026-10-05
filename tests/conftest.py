"""Shared test setup.

`chunkeval` is imported as the installed (editable) package: run `uv sync` once, then
`uv run pytest`. pytest itself is in the dev dependency group.

TODO(you): add fixtures here as you need them, e.g.
  - `tiny_doc`: a hand-built CanonicalDoc (3 sections, a table, a few tricky sentences such as
    "e.g. maize", "Smith et al. 2020", "Fig. 3", "3.5 t/ha") — small enough to reason about by hand;
  - `counter`: a TokenCounter on the bge-m3 tokenizer (session-scoped: loading it is slow);
  - `embedders`: bge-m3 + Qwen3-Embedding-0.6B loaded once per session — mark the tests that use
    them (e.g. `@pytest.mark.gpu`) so the fast tests can run without loading 2 models.
"""
import pytest
import yaml

from chunkeval.core.config import load_config
from chunkeval.core.schema import REPO_ROOT

DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


@pytest.fixture
def raw_config() -> dict:
    """configs/default.yaml as a plain dict — mutate it to build invalid configs."""
    return yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))


@pytest.fixture
def cfg(tmp_path):
    """The validated default config, with runs_dir redirected to a temp dir.

    Fresh per test (tests may mutate it), and nothing a test writes via `run_dir` /
    `save_config` / `check_ollama` ends up in the real runs/ folder.
    """
    config = load_config(DEFAULT_CONFIG)
    config.paths.runs_dir = tmp_path / "runs"
    return config
