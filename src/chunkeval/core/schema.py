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


class Parsing(StrictConfig):
    drop_labels: list[str]
    skip_section_titles: list[str]


class EmbModel(StrictConfig):
    model_id: str
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
        unknown = [n for n in [self.primary, *self.secondary] if n not in self.models]
        if unknown:
            raise ValueError(f"unknown embedder(s) {unknown}; defined in models: {list(self.models)}")
        self.cache_dir = REPO_ROOT / self.cache_dir
        return self

class Chunking(StrictConfig):
    size_tokenizer: str
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
    # On Ollama /v1, "none" is the only switch that turns Qwen3.5's thinking off; gpt-oss
    # can't turn it off, but "low" cuts it to a few dozen tokens.
    reasoning_effort: Literal["none", "low", "medium", "high"] | None = None


class LLM(StrictConfig):
    qa_generator: LLMModel
    qa_critic: LLMModel
    rag_generator: LLMModel
    judge: LLMModel
    bridge_judge: LLMModel

class QA(StrictConfig):
    docs_per_corpus: int = Field(gt=0)
    max_questions_per_doc: int = Field(gt=0)
    unit_min_tokens: int = Field(gt=0)
    unit_max_tokens: int = Field(gt=0)
    two_span_ratio: float = Field(ge=0, le=1)
    question_types: list[Literal["factoid", "numeric", "definition", "reasoning"]] = Field(min_length=1)
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
    budget_tokens: int = Field(gt=0)

class Ragas(StrictConfig):
    embeddings_model: str
    sample_per_corpus: int = Field(gt=0)
    max_workers: int = Field(ge=1)
    timeout: int = Field(gt=0)
    max_retries: int = Field(ge=0)
    max_wait: int = Field(ge=0)

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

        variants = [f"{m}_{'overlap' if o else 'no_overlap'}"
                    for m in self.chunking.methods for o in self.chunking.overlap]
        if self.analysis.baseline not in variants:
            raise ValueError(f"analysis.baseline '{self.analysis.baseline}' must be one of {variants}")
        return self

    @model_validator(mode="after")
    def check_no_placeholders(self):
        placeholders = find_placeholders(self.model_dump(mode="json"))
        if placeholders:
            raise ValueError(f"replace the TODO placeholders before running: {placeholders}")
        return self
