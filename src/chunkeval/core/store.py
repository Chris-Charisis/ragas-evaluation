"""Artefact persistence: JSONL / Parquet helpers and stage caching.

Why: the old `save_data_with_pickle` / `load_data_with_pickle` wrote *globals* into a pickle.
That is fragile (silent overwrites of globals, Python-version-dependent, not inspectable).
JSONL is human-readable and diff-able; Parquet is compact for the per-question metric tables.

Layout produced (see README):
  data/parsed/<corpus>/<doc_id>.json
  data/chunks/<corpus>/<method>_<ovl>.jsonl
  data/qa/{raw,filtered,rejected}.jsonl
  runs/<run_id>/{retrieval,answers,ragas,report}/...
"""
from __future__ import annotations

from pathlib import Path
import dataclasses
from typing import Any, Iterable, Iterator, Sequence
import json
import tempfile
import os

from chunkeval.core.schema import REPO_ROOT, Config


def _to_line(record: Any) -> str:
    """Serialize one record (dict or dataclass instance) as a JSONL line, "\\n" included."""
    # Dataclass instances -> plain dicts (json can't serialize them); dicts pass through.
    # Note: `is_dataclass` is also True for the class itself (e.g. `Chunk`, not
    # `Chunk(...)`). The strict, docs-recommended instance-only check is:
    #     if dataclasses.is_dataclass(record) and not isinstance(record, type):
    # Not needed here: a class passed by mistake fails either way, and `asdict`
    # gives the clearer error ("asdict() should be called on dataclass instances").
    if dataclasses.is_dataclass(record):
        record = dataclasses.asdict(record)
    # ensure_ascii=False keeps symbols like α, ≤, é readable instead of \uXXXX escapes.
    # No `indent`: JSONL requires exactly one record per line.
    return json.dumps(record, ensure_ascii=False) + "\n"


def _iter_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield the records of a JSONL file, tolerating what a crashed append can leave behind.

    - Blank lines are skipped (e.g. an empty line an editor added at the end).
    - An unterminated last line that isn't valid JSON is an append cut off by a crash: it is
      skipped, so that item counts as not done and is simply redone.
    - Invalid JSON anywhere else is real corruption and raises, naming the line.
    """
    # Binary mode: a crash can also cut a multi-byte UTF-8 character in half, which would make
    # a text-mode file object raise before we see the line. json.loads decodes the UTF-8 bytes
    # itself and reports a bad cut as a ValueError, which the except below handles.
    with open(path, "rb") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as e:      # JSONDecodeError and UnicodeDecodeError both subclass it
                if not line.endswith(b"\n"):
                    return               # broken last line from a crash: ignore it
                raise ValueError(f"{path}:{lineno}: invalid JSON line ({e})") from e
            yield record


def _fix_unterminated_last_line(path: Path) -> None:
    """Make sure the next append starts on a fresh line.

    Our writers end every record with "\\n", so a last line without one means an append was cut
    off by a crash. Appending straight after it would glue the new record onto the broken one
    and corrupt both. If the fragment is a complete record (the crash hit right before the
    "\\n"), terminate it; otherwise cut it off, matching `_iter_records`, which skips it.
    """
    if not path.exists() or path.stat().st_size == 0:
        return
    with open(path, "rb+") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return                       # the normal case costs a single 1-byte read
        # Rare (only right after a crash), so reading the whole file is fine.
        f.seek(0)
        data = f.read()
        start = data.rfind(b"\n") + 1    # where the last line starts (0 if it's the only line)
        try:
            json.loads(data[start:])
            f.write(b"\n")               # complete record: just terminate it (we're at EOF)
        except ValueError:
            f.truncate(start)            # partial record: drop it, the item will be redone


def write_jsonl(path: str | Path, records: Iterable[Any]) -> int:
    """Atomically write records (dicts or dataclass instances) to JSONL, return the count.

    The file is written to a temp file and swapped in at the end, so a crash never leaves a
    half-written file that a later stage would happily read.
    """
    # Relative paths resolve against the repo root, so callers work from any CWD.
    # (An absolute `path` wins: `REPO_ROOT / "/abs/x"` == `/abs/x`.)
    path = REPO_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    # Temp file lives in the *target* directory: `os.replace` is only atomic within one
    # filesystem, and /tmp is often a separate tmpfs ("Invalid cross-device link").
    # delete=False because we rename it ourselves instead of letting it be deleted on close.
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False, encoding="utf-8") as tmp:
        try:
            for record in records:
                tmp.write(_to_line(record))
                # Count while writing: `records` may be a generator, which can't be iterated twice.
                total += 1
        except BaseException:
            # Any failure (bad value, Ctrl+C): drop the partial temp file and leave any
            # existing target untouched. BaseException so KeyboardInterrupt also cleans up.
            tmp.close()
            os.unlink(tmp.name)
            raise

    # Atomic swap: readers see either the old complete file or the new complete file, never a mix.
    os.replace(tmp.name, path)

    return total


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream records from a JSONL file (generator, so large files don't load at once).

    Skips blank lines and a crash-truncated last line; see `_iter_records`.
    """
    yield from _iter_records(REPO_ROOT / path)


def append_jsonl(path: str | Path, record: Any) -> None:
    """Append one record — used for resumable LLM stages (one line per finished item).

    Deliberately does no duplicate check: by the time a record is appended, its LLM call has
    already been paid for. Skipping finished items is the caller's job, *before* the call:

        done = done_ids(out_path, key="question_id")     # read the file once, at the start
        for item in items:
            if item.id in done:
                continue                                  # finished in an earlier run
            append_jsonl(out_path, call_llm(item))        # saved as soon as it's finished
    """
    path = REPO_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    _fix_unterminated_last_line(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(_to_line(record))


def done_ids(path: str | Path, key: str = "id") -> set[str]:
    """Return the set of ids already present in an (append-only) JSONL file.

    A missing file means nothing is done yet (first run). Records without `key` are ignored.
    """
    path = REPO_ROOT / path
    if not path.exists():
        return set()
    return {record[key] for record in _iter_records(path) if key in record}


def write_table(
    path: str | Path,
    rows: list[dict[str, Any]],
    unique_by: Sequence[str] | None = None,
) -> Path:
    """Persist a per-question metrics table (long format) and return the path actually written.

    Parquet needs pyarrow; without it the table is written as CSV next to the requested path
    (`x.parquet` -> `x.csv`), which is why the real path is returned.

    `unique_by`: the columns that must identify a row, e.g. ("question_id", "corpus", "method",
    "overlap"). Long format makes the paired comparisons in analysis.py a simple pivot, and a
    duplicated row would silently count a question twice, so duplicates raise instead.
    """
    import pandas as pd   # imported here so the JSONL helpers don't pay pandas' import time

    path = REPO_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)

    if unique_by:
        dup = df.duplicated(subset=list(unique_by), keep=False)
        if dup.any():
            raise ValueError(
                f"{path.name}: {int(dup.sum())} rows share a {list(unique_by)} key, e.g.:\n"
                f"{df.loc[dup, list(unique_by)].head()}"
            )

    # Same write-then-swap as write_jsonl, so a crash never leaves a truncated table.
    try:
        tmp = path.with_name(path.name + ".tmp")
        df.to_parquet(tmp, engine="pyarrow", index=False)
    except ImportError:              # pandas raises ImportError when pyarrow is missing
        path = path.with_suffix(".csv")
        tmp = path.with_name(path.name + ".tmp")
        df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return path


def run_dir(cfg: Config, *parts: str) -> Path:
    """Return <runs_dir>/<run_id>/<parts...>, creating it if needed."""
    # paths.runs_dir comes from the config (already absolute), not a hardcoded "runs".
    path = cfg.paths.runs_dir.joinpath(cfg.run_id, *parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
