"""store.py — atomic writes, crash-tolerant appends for resumable stages, tables, run dirs.

All offline and fast; everything is written under pytest's `tmp_path`.
"""
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from chunkeval.core import store


@dataclass
class Rec:
    id: str
    text: str


def ids(path: Path, key: str = "id") -> list:
    return [r[key] for r in store.read_jsonl(path)]


# --------------------------------------------------------------------------------------
# write_jsonl / read_jsonl
# --------------------------------------------------------------------------------------

def test_write_jsonl_counts_a_generator(tmp_path):
    """Regression: the count used to be taken by re-iterating `records` (0 for a generator)."""
    p = tmp_path / "a" / "b" / "w.jsonl"           # parent dirs don't exist yet
    assert store.write_jsonl(p, (Rec(str(i), "x") for i in range(5))) == 5
    assert ids(p) == ["0", "1", "2", "3", "4"]


def test_one_line_per_record_with_raw_unicode(tmp_path):
    p = tmp_path / "u.jsonl"
    store.write_jsonl(p, [{"id": 1, "text": "α ≤ é ✓"}, {"id": 2, "text": "line\nbreak"}])
    content = p.read_text(encoding="utf-8")
    assert content.count("\n") == 2                 # the "\n" inside a value is escaped
    assert "α ≤ é ✓" in content                     # ensure_ascii=False: no \uXXXX escapes


def test_dicts_and_dataclasses_mixed(tmp_path):
    p = tmp_path / "mix.jsonl"
    assert store.write_jsonl(p, [{"id": "x"}, Rec("y", "t")]) == 2
    assert ids(p) == ["x", "y"]


def test_empty_iterable(tmp_path):
    p = tmp_path / "empty.jsonl"
    assert store.write_jsonl(p, []) == 0
    assert p.read_text() == ""


def test_failed_write_keeps_old_file_and_no_temp(tmp_path):
    p = tmp_path / "w.jsonl"
    store.write_jsonl(p, [{"id": "old"}])
    with pytest.raises(TypeError):
        store.write_jsonl(p, iter([{"id": "new"}, {"id": object()}]))   # 2nd record not serialisable
    assert ids(p) == ["old"]
    assert not list(tmp_path.glob("*.tmp"))


def test_relative_path_resolves_against_repo_root(tmp_path):
    rel = Path(os.path.relpath(tmp_path / "rel.jsonl", store.REPO_ROOT))
    store.write_jsonl(rel, [{"id": 1}])
    assert (tmp_path / "rel.jsonl").exists()


def test_read_jsonl_is_lazy(tmp_path):
    p = tmp_path / "w.jsonl"
    store.write_jsonl(p, [{"id": 1}])
    assert hasattr(store.read_jsonl(p), "__next__")


def test_blank_lines_skipped(tmp_path):
    p = tmp_path / "blank.jsonl"
    p.write_text('{"id": 1}\n\n{"id": 2}\n\n')
    assert ids(p) == [1, 2]


def test_mid_file_corruption_raises_with_line_number(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"id": 1}\n{oops\n{"id": 2}\n')
    with pytest.raises(ValueError, match=":2:"):
        list(store.read_jsonl(p))


# --------------------------------------------------------------------------------------
# append_jsonl / done_ids — the resume mechanism of the LLM stages
# --------------------------------------------------------------------------------------

@pytest.fixture
def appended(tmp_path) -> Path:
    p = tmp_path / "res" / "out.jsonl"
    for i in range(3):
        store.append_jsonl(p, Rec(f"q{i}", "x"))
    return p


def test_filter_jsonl_keeps_the_rest_in_order(tmp_path):
    """One corpus out of a shared file: the others stay in order; a file left empty is deleted."""
    path = tmp_path / "x.jsonl"
    store.write_jsonl(path, [{"id": i, "corpus": c} for i, c in enumerate(["a", "b", "a", "b"])])
    before = path.read_bytes()
    assert store.filter_jsonl(path, lambda r: True) == 0 and path.read_bytes() == before     # untouched
    assert store.filter_jsonl(path, lambda r: r["corpus"] != "a") == 2 and ids(path) == [1, 3]
    assert store.filter_jsonl(path, lambda r: False) == 2 and not path.exists()
    assert store.filter_jsonl(path, lambda r: False) == 0                                    # missing file


def test_done_ids_missing_file_is_empty(tmp_path):
    assert store.done_ids(tmp_path / "nope.jsonl") == set()


def test_append_and_done_ids(appended):
    assert store.done_ids(appended) == {"q0", "q1", "q2"}
    assert store.done_ids(appended, key="text") == {"x"}


def test_records_without_key_are_ignored(appended):
    store.append_jsonl(appended, {"other": 1})
    assert store.done_ids(appended) == {"q0", "q1", "q2"}


def test_append_does_not_deduplicate(appended):
    """Skipping finished items is the caller's job (before paying for the LLM call)."""
    store.append_jsonl(appended, Rec("q0", "again"))
    assert ids(appended).count("q0") == 2


def test_crash_partial_last_line(appended):
    """A crash mid-append can cut the last line anywhere — even inside a UTF-8 character."""
    with open(appended, "ab") as f:
        f.write('{"id": "q3", "text": "é'.encode()[:-1])
    assert store.done_ids(appended) == {"q0", "q1", "q2"}      # q3 counts as not done
    assert len(list(store.read_jsonl(appended))) == 3
    store.append_jsonl(appended, Rec("q3", "redo"))            # resume repairs the tail first
    assert store.done_ids(appended) == {"q0", "q1", "q2", "q3"}
    assert appended.read_text(encoding="utf-8").count("\n") == 4   # nothing glued together


def test_crash_before_newline_keeps_complete_record(appended):
    with open(appended, "a") as f:
        f.write('{"id": "q3"}')                                # complete, only "\n" missing
    assert "q3" in store.done_ids(appended)
    store.append_jsonl(appended, {"id": "q4"})
    assert store.done_ids(appended) == {"q0", "q1", "q2", "q3", "q4"}


# --------------------------------------------------------------------------------------
# write_table
# --------------------------------------------------------------------------------------

ROWS = [{"question_id": f"q{i}", "method": m, "score": i / 10} for i in range(3) for m in ("a", "b")]


def test_write_table_parquet(tmp_path):
    out = store.write_table(tmp_path / "t" / "m.parquet", ROWS, unique_by=["question_id", "method"])
    assert out.suffix == ".parquet"
    assert pd.read_parquet(out).shape == (6, 3)
    assert not list(out.parent.glob("*.tmp"))


def test_write_table_rejects_duplicate_keys(tmp_path):
    with pytest.raises(ValueError, match="share"):
        store.write_table(tmp_path / "dup.parquet", ROWS + ROWS[:1], unique_by=["question_id", "method"])
    assert not (tmp_path / "dup.parquet").exists()


def test_write_table_csv_fallback_without_pyarrow(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pyarrow", None)          # import pyarrow -> ImportError
    out = store.write_table(tmp_path / "f.parquet", ROWS)
    assert out == tmp_path / "f.csv"                           # the caller learns the real path
    assert len(pd.read_csv(out)) == 6


# --------------------------------------------------------------------------------------
# run_dir
# --------------------------------------------------------------------------------------

def test_run_dir(cfg):
    d = store.run_dir(cfg, "ragas", "sub")
    assert d == cfg.paths.runs_dir / cfg.run_id / "ragas" / "sub"
    assert d.is_dir()
    assert store.run_dir(cfg) == cfg.paths.runs_dir / cfg.run_id
