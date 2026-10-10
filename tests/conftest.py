"""Shared test setup.

`chunkeval` is imported as the installed (editable) package: run `uv sync` once, then
`uv run pytest`. pytest itself is in the dev dependency group.

TODO(you): add fixtures here as you need them, e.g.
  - `embedders`: bge-m3 + Qwen3-Embedding-0.6B loaded once per session — mark the tests that use
    them (e.g. `@pytest.mark.gpu`) so the fast tests can run without loading 2 models.
"""
import pytest
import yaml

from chunkeval.core.config import load_config
from chunkeval.core.schema import REPO_ROOT
from chunkeval.corpus.parsing import Piece, assemble_doc

DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


@pytest.fixture(scope="session")
def counter():
    """TokenCounter on the size tokenizer (bge-m3): tokenizer files only, ~20 MB, downloaded on
    first use. Session-scoped, since loading takes a few seconds; skips the test when the
    tokenizer can't be loaded (offline before the first download)."""
    from chunkeval.corpus.chunking.base import load_counter

    try:
        return load_counter(load_config(DEFAULT_CONFIG).chunking.size_tokenizer)
    except OSError as e:
        pytest.skip(f"size tokenizer unavailable: {e}")


@pytest.fixture
def tiny_doc():
    """A hand-built CanonicalDoc, small enough to reason about by hand: nested sections, a
    heading-only parent ("2. Methods"), tiny sibling sections ("2.2", "2.3"), list items, the
    abbreviations that broke the old regex sentence splitter, and a run-on sentence of ~400
    tokens with no full stop (longer than any max_tokens the tests use)."""
    def H(text, level=1):
        return Piece("heading", text, level=level)

    def P(text, type="paragraph"):
        return Piece(type, text)

    pieces = [
        H("1. Introduction"),
        P("Rice is grown in ca. 40 countries in Africa, e.g. in Côte d'Ivoire and Nigeria. "
          "Smith et al. (2020) reported yields of 3.5 t/ha for O. sativa. Fig. 3 shows the trend since 2010."),
        P("Yields depend on water, see Eq. (1) and Table 2. Irrigated vs. rainfed systems differ by "
          "approx. 2 t/ha. The gap has widened over the last decade."),
        H("2. Methods"),
        H("2.1 Sites", level=2),
        P("Trials ran at three sites in the Sahel and the humid forest zone. Each site had four "
          "replicates of every treatment. Soils were sampled before planting and after harvest."),
        H("2.2 Data", level=2),
        P("Weather data were daily."),
        H("2.3 Models", level=2),
        P("We fitted a mixed model."),
        H("3. Results"),
        P("Yields rose at every site. The largest gain was in the Sahel, where irrigation doubled "
          "the harvest. In the forest zone the gain was smaller but steady. Weeds were the main "
          "constraint in rainfed plots. Farmers adopted the new varieties quickly. Seed supply "
          "remained a bottleneck in remote villages. Prices for paddy were stable during the trial."),
        P("· Yield rose by 12% on average.", type="list_item"),
        P("· Labour costs fell.", type="list_item"),
        P("Stations recorded rainfall as follows: " + ", ".join(f"site {i} had {i * 7} mm" for i in range(60)) + "."),
        H("4. Conclusions"),
        P("Water management matters most. Further trials are planned for 2025."),
    ]
    return assemble_doc("tiny", "papers", "A tiny paper", pieces)


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
