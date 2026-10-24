import string
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from pathlib import Path


def _find_repo_root() -> Path:
    """The nearest folder above this file that holds pyproject.toml.

    Searched instead of a fixed `parents[N]`, which silently points to the wrong folder as soon
    as this module moves to another depth in the package. Works with the editable install
    (`uv sync`), because the code still runs from src/.
    """
    here = Path(__file__).resolve()
    for folder in here.parents:
        if (folder / "pyproject.toml").is_file():
            return folder
    raise RuntimeError(f"no pyproject.toml above {here}: run chunkeval from the repo (uv sync)")


# Every relative path (configs, data/, runs/, .env) resolves against this, never the CWD.
REPO_ROOT = _find_repo_root()

# Embedding model names double as Weaviate named-vector names: letters, digits, underscores only.
VectorName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]

# A full Hugging Face Hub commit hash. Not a branch or tag ("main" moves when the authors push).
CommitHash = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")



class Paths(StrictConfig):
    data_dir: Path
    runs_dir: Path
    corpora: dict[str, Path]


    @model_validator(mode="after")
    def resolve_against_repo_root(self):
        self.data_dir = REPO_ROOT / self.data_dir
        self.runs_dir = REPO_ROOT / self.runs_dir
        self.corpora = {name: REPO_ROOT / p for name, p in self.corpora.items()}
        return self


# Docling labels parsing.py can turn into text (the keys of parsing.LABEL_BLOCK_TYPES).
DoclingLabel = Literal["title", "section_header", "text", "paragraph", "list_item", "caption", "footnote"]


class BodyRule(StrictConfig):
    """Which part of a document becomes CanonicalDoc.text (see parsing.select_body). An empty
    list switches its step off."""
    start_titles: list[str] = []
    end_titles: list[str] = []
    drop_titles: list[str] = []
    drop_title_prefixes: list[str] = []     # first word of the title: "Annex 2: Workshop agenda"


class Parsing(StrictConfig):
    include_labels: list[DoclingLabel] = Field(min_length=1)
    heading_levels: Literal["font_size", "numbering", "flat"]
    ocr_max_replacement_char_ratio: float = Field(ge=0, le=1)
    ocr_max_repeated_paragraph_ratio: float = Field(ge=0, le=1)
    body: dict[str, BodyRule] = {}       # per corpus; a corpus without a rule keeps its whole text
    skip_section_titles: list[str]
    # CSV (corpus, doc_id, title) of titles that replace the one Docling found; relative to the repo root.
    title_overrides: Path | None = None

    @model_validator(mode="after")
    def resolve_against_repo_root(self):
        if self.title_overrides is not None:
            self.title_overrides = REPO_ROOT / self.title_overrides
        return self


class EmbModel(StrictConfig):
    model_id: str
    revision: CommitHash
    query_prompt: str
    document_prompt: str
    max_seq_length: int = Field(gt=0)
    dim: int = Field(gt=0)

class Embedding(StrictConfig):
    primary: str
    secondary: list[str]
    batch_size: int = Field(gt=0)
    cache_dir: Path
    models: dict[VectorName, EmbModel]

    @model_validator(mode="after")
    def check_and_resolve(self):
        names = [self.primary, *self.secondary]
        unknown = [n for n in names if n not in self.models]
        if unknown:
            raise ValueError(f"unknown embedder(s) {unknown}; defined in models: {list(self.models)}")
        # `embed` would embed (and the analysis count) such an embedder twice
        if len(set(names)) != len(names):
            raise ValueError(f"an embedder is listed twice in primary + secondary: {names}")
        self.cache_dir = REPO_ROOT / self.cache_dir
        return self

class Chunking(StrictConfig):
    size_tokenizer: str
    size_tokenizer_revision: CommitHash
    target_tokens: int = Field(gt=0)
    max_tokens: int = Field(gt=0)
    overlap_ratio: float = Field(ge=0, lt=1)
    min_section_tokens: int = Field(ge=0)
    methods: list[Literal["character", "sentence", "recursive"]] = Field(min_length=1)
    overlap: list[bool] = Field(min_length=1)
    size_sweep: list[Annotated[int, Field(gt=0)]]

    @model_validator(mode="after")
    def check_sizes(self):
        if self.target_tokens > self.max_tokens:
            raise ValueError(f"target_tokens ({self.target_tokens}) > max_tokens ({self.max_tokens})")
        return self

class Weaviate(StrictConfig):
    host: str
    port: int = Field(ge=1, le=65535)
    grpc_port: int = Field(ge=1, le=65535)
    collection_template: str

    @model_validator(mode="after")
    def check_template_placeholders(self):
        # all three are needed, or two configs would share (and overwrite) one collection
        fields = {f for _, f, _, _ in string.Formatter().parse(self.collection_template) if f is not None}
        expected = {"corpus", "method", "overlap"}
        if fields != expected:
            raise ValueError(f"collection_template placeholders must be {sorted(expected)}, got {sorted(fields)}")
        return self

class Ollama(StrictConfig):
    context_length: int = Field(gt=0)
    concurrency: int = Field(ge=1)

class LLMModel(StrictConfig):
    provider: Literal["ollama", "openai"]
    model: str
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)   # None = no explicit limit
    # Sent as `reasoning_effort` only when set: non-reasoning OpenAI models reject it.
    # On Ollama /v1, "none" is the only switch that turns Qwen3.5's thinking off (Gemma 4's too); gpt-oss
    # can't turn it off, but "low" cuts it to a few dozen tokens.
    reasoning_effort: Literal["none", "low", "medium", "high"] | None = None


class LLM(StrictConfig):
    qa_generator: LLMModel
    qa_critic: LLMModel
    rag_generator: LLMModel
    judge: LLMModel

# Question types the Q-A generator can be asked for (prompts/qa_generation.md defines each one);
# also the enum of qa.generation.QA_JSON_SCHEMA.
QuestionType = Literal["factoid", "numeric", "definition", "reasoning"]


class QA(StrictConfig):
    docs_per_corpus: int = Field(gt=0)
    max_questions_per_doc: int = Field(gt=0)
    unit_min_tokens: int = Field(gt=0)
    unit_max_tokens: int = Field(gt=0)
    two_span_ratio: float = Field(ge=0, le=1)
    question_types: list[QuestionType] = Field(min_length=1)
    fuzzy_match_threshold: float = Field(ge=0, le=100)   # rapidfuzz score
    critic_min_score: int = Field(ge=1, le=5)
    dedup_cosine_threshold: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def check_unit_sizes(self):
        if self.unit_min_tokens > self.unit_max_tokens:
            raise ValueError(f"unit_min_tokens ({self.unit_min_tokens}) > unit_max_tokens ({self.unit_max_tokens})")
        return self

class Retrieval(StrictConfig):
    embedders: list[str] = Field(min_length=1)
    mode: Literal["dense", "hybrid"]
    hybrid_alpha: float = Field(ge=0, le=1)
    top_k: int = Field(gt=0)
    k_max: int = Field(gt=0)
    k_values: list[Annotated[int, Field(gt=0)]] = Field(min_length=1)
    budget_tokens: int = Field(gt=0)

    @model_validator(mode="after")
    def check_k(self):
        # every k is cut from the k_max hits retrieved once
        too_large = [k for k in (self.top_k, *self.k_values) if k > self.k_max]
        if too_large:
            raise ValueError(f"top_k and k_values must be <= k_max ({self.k_max}), got {too_large}")
        # a repeated k would give two metric rows with the same key
        if len(set(self.k_values)) != len(self.k_values):
            raise ValueError(f"retrieval.k_values has a repeated value: {self.k_values}")
        return self

class Price(StrictConfig):
    """USD per 1M tokens."""
    input: float = Field(ge=0)
    output: float = Field(ge=0)

class Ragas(StrictConfig):
    sample_per_corpus: int = Field(gt=0)
    max_workers: int = Field(ge=1)
    timeout: int = Field(gt=0)
    max_retries: int = Field(ge=0)
    max_cost_usd: float = Field(gt=0)
    prices: dict[str, Price] = {}        # by judge model; an OpenAI judge must have one

class Analysis(StrictConfig):
    bootstrap_iterations: int = Field(gt=0)
    ci: float = Field(gt=0, lt=1)
    baseline: str


def find_placeholders(value: Any, path: str = "") -> list[str]:
    """Dotted paths of every string value that still contains "TODO"."""
    if isinstance(value, str):
        return [path] if "TODO" in value else []
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in find_placeholders(v, f"{path}.{k}" if path else k)]
    if isinstance(value, list):
        return [p for i, v in enumerate(value) for p in find_placeholders(v, f"{path}[{i}]")]
    return []


class Config(StrictConfig):

    run_id: str
    seed: int
    paths: Paths
    parsing: Parsing
    embedding: Embedding
    chunking: Chunking
    weaviate: Weaviate
    ollama: Ollama
    llm: LLM
    qa: QA
    retrieval: Retrieval
    ragas: Ragas
    analysis: Analysis

    @model_validator(mode="after")
    def check_cross_references(self):
        models = self.embedding.models
        unknown = [n for n in self.retrieval.embedders if n not in models]
        if unknown:
            raise ValueError(f"retrieval.embedders {unknown} not in embedding.models: {list(models)}")
        # answering and Ragas run on the primary embedder's retrieval results
        if self.embedding.primary not in self.retrieval.embedders:
            raise ValueError(f"embedding.primary '{self.embedding.primary}' missing from retrieval.embedders")
        # The size tokenizer is an embedder's own (bge-m3): pinned to the same commit, so updating
        # one pin without the other fails here instead of measuring chunks with another version.
        c = self.chunking
        for name, m in models.items():
            if m.model_id == c.size_tokenizer and m.revision != c.size_tokenizer_revision:
                raise ValueError(f"chunking.size_tokenizer_revision ({c.size_tokenizer_revision}) differs from "
                                 f"embedding.models.{name}.revision ({m.revision}), the same model {m.model_id}")

        variants = [f"{m}_{'overlap' if o else 'no_overlap'}"
                    for m in self.chunking.methods for o in self.chunking.overlap]
        if self.analysis.baseline not in variants:
            raise ValueError(f"analysis.baseline '{self.analysis.baseline}' must be one of {variants}")

        unknown = [c for c in self.parsing.body if c not in self.paths.corpora]
        if unknown:
            raise ValueError(f"parsing.body {unknown} not in paths.corpora: {list(self.paths.corpora)}")

        # `eval` estimates and caps the judge's cost, so a paid judge needs its price.
        judge = self.llm.judge
        if judge.provider == "openai" and judge.model not in self.ragas.prices:
            raise ValueError(f"ragas.prices has no price for the judge {judge.model!r}: add it (USD per 1M "
                             f"input and output tokens), so `eval` can estimate and cap its cost")
        return self

    @model_validator(mode="after")
    def check_no_placeholders(self):
        placeholders = find_placeholders(self.model_dump(mode="json"))
        if placeholders:
            raise ValueError(f"replace the TODO placeholders before running: {placeholders}")
        return self
