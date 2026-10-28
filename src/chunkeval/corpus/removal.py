"""Maintenance — take a corpus out of the pipeline for good: everything the stages made from it.

`chunkeval remove-corpus NAME` lists what it would remove and removes nothing; with `--yes` it
removes it. What goes, from this config's data dir (paths.data_dir, the embedding cache) and Weaviate:
  * data/parsed/<corpus>/ and data/chunks/<corpus>/;
  * its vectors: data/embeddings/<embedder>/<corpus>/, its query vectors (queries/<embedder>/<corpus>.*)
    and its de-duplication vectors (dedup/<embedder>/<corpus>.*);
  * its lines in data/qa/ (raw.jsonl, gen_dropped.jsonl, filtered.jsonl, rejected.jsonl) and its
    entries in raw.provenance.json and filtered.provenance.json; the other corpora's lines stay as they are;
  * its Weaviate collections, for every chunking method and overlap under this data dir's namespace,
    but only those whose provenance says `index` built them from this corpus's chunks.
Its doc ids are then free again (parsing.doc_id_clashes).

What stays, on purpose:
  * the inputs: its PDFs, the configs and configs/titles.csv. The stage edits no input, so it removes
    nothing that can't be rebuilt from them;
  * runs/: the results of finished runs, the record of what was measured;
  * the caches keyed by content (data/llm_cache/, data/qa/critic_scores.jsonl): an entry only ever
    serves the identical call, so a leftover can't touch another corpus, and if the corpus comes
    back, nothing is asked or paid for again.

Refused while a config still uses the corpus with this data dir (the one given, or any of
configs/*.yaml): the next `parse` would bring it back, or another experiment still needs it.
Removing is idempotent: after a crash, run it again.

Reads:  configs/*.yaml, data/, Weaviate
Writes: removes the files and collections above; runs/<run_id>/logs/remove-corpus_<timestamp>.log
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from chunkeval.core.config import Secrets, format_collection_name, load_config
from chunkeval.core.schema import REPO_ROOT, Config
from chunkeval.core.store import read_jsonl
from chunkeval.corpus import indexing
from chunkeval.corpus.chunking import CHUNKERS
from chunkeval.corpus.parsing import title_overrides
from chunkeval.qa import filtering, generation

if TYPE_CHECKING:
    import weaviate

log = logging.getLogger(__name__)

# Every config here is checked: one that still uses the corpus with the same data dir refuses the removal.
CONFIGS_DIR = REPO_ROOT / "configs"


@dataclass
class Item:
    """One thing to remove: what it is, where, how much, and how."""
    what: str
    where: str
    size: str
    remove: Callable[[], None]


def shown(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def configs_using(cfg: Config, corpus: str) -> list[str]:
    """The configs that still use `corpus` with this data dir: the one given, and every
    configs/*.yaml (one that doesn't load is skipped, with a warning)."""
    users = ["the config given"] if corpus in cfg.paths.corpora else []
    for path in sorted(CONFIGS_DIR.glob("*.yaml")):
        try:
            other = load_config(path)
        except Exception as e:                   # a broken config can't use anything
            log.warning("%s doesn't load, so it isn't checked for %s (%s)", shown(path), corpus, type(e).__name__)
            continue
        if other.paths.data_dir == cfg.paths.data_dir and corpus in other.paths.corpora:
            users.append(shown(path))
    return users


def folder_item(what: str, path: Path) -> Item | None:
    if not path.is_dir():
        return None
    n = sum(p.is_file() for p in path.rglob("*"))
    return Item(what, shown(path), f"{n} file(s)", lambda: shutil.rmtree(path))


def lines_of(path: Path, corpus: str) -> int:
    return sum(r["corpus"] == corpus for r in read_jsonl(path)) if path.exists() else 0


def removal_plan(cfg: Config, corpus: str, client: "weaviate.WeaviateClient") -> list[Item]:
    """Everything of `corpus` there is to remove, in the order it is removed (see the module docstring)."""
    items: list[Item] = []
    for method in CHUNKERS:
        for overlap in (False, True):
            name = format_collection_name(cfg, corpus, method, overlap)
            if not client.collections.exists(name):
                continue
            collection = client.collections.get(name)
            stored = indexing.stored_provenance(collection)
            source = stored["chunks"].split("/")[2] if stored and stored.get("chunks", "").count("/") == 3 else None
            if source != corpus:
                log.warning("%s: not built by `index` from the chunks of %s (%s), left alone", name, corpus,
                            stored["chunks"] if stored else "no provenance")
                continue
            count = collection.aggregate.over_all(total_count=True).total_count
            items.append(Item("Weaviate collection", name, f"{count} objects",
                              lambda name=name: client.collections.delete(name)))

    cache = cfg.embedding_cache_dir
    per_corpus_files = {"queries": "query vectors", "dedup": "de-duplication vectors"}
    for folder in sorted(p for p in cache.iterdir() if p.is_dir()) if cache.is_dir() else []:
        if folder.name in per_corpus_files:
            for embedder in sorted(p for p in folder.iterdir() if p.is_dir()):
                files = sorted(embedder.glob(f"{corpus}.*"))
                if files:
                    items.append(Item(f"{per_corpus_files[folder.name]}, {embedder.name}", shown(embedder / f"{corpus}.*"),
                                      f"{len(files)} file(s)", lambda files=files: [f.unlink() for f in files]))
        elif item := folder_item(f"chunk vectors, {folder.name}", folder / corpus):
            items.append(item)
    for what, folder in (("chunk files", "chunks"), ("parsed documents", "parsed")):
        if item := folder_item(what, cfg.paths.data_dir / folder / corpus):
            items.append(item)

    filt = filtering.filter_paths(cfg)
    kept, rejected = lines_of(filt["filtered"], corpus), lines_of(filt["rejected"], corpus)
    if kept or rejected or corpus in filtering.read_filter_provenance(cfg):
        items.append(Item("question set lines", shown(filt["filtered"].parent),
                          f"{kept} in filtered.jsonl, {rejected} in rejected.jsonl",
                          lambda: filtering.delete_records(filt, [corpus])))
    gen = generation.qa_paths(cfg)
    settings, hashes = generation.read_provenance(gen["provenance"])
    raw, dropped = lines_of(gen["raw"], corpus), lines_of(gen["dropped"], corpus)
    docs = sum(key.split("/")[0] == corpus for key in hashes)
    if raw or dropped or docs or corpus in settings:
        items.append(Item("Q-A records", shown(gen["raw"].parent),
                          f"{raw} in raw.jsonl, {dropped} in gen_dropped.jsonl, {docs} document(s) in raw.provenance.json",
                          lambda: generation.delete_records(gen, [corpus])))
    return items


def run(cfg: Config, secrets: Secrets, corpus: str, yes: bool = False) -> None:
    """CLI entry (`chunkeval remove-corpus NAME`): list everything of `corpus` there is to remove,
    then remove it if `yes`. Refused while a config still uses the corpus (`configs_using`) and when
    nothing of it is found (a misspelt name). Weaviate must be reachable, so the plan is complete
    before anything is removed."""
    users = configs_using(cfg, corpus)
    if users:
        raise RuntimeError(f"{corpus} is still in paths.corpora of {', '.join(users)}, with this data dir "
                           f"({shown(cfg.paths.data_dir)}): take it out of paths.corpora and parsing.body of each "
                           f"first, so no stage brings it back")
    with indexing.connect(cfg, secrets.weaviate_api_key.get_secret_value()) as client:
        plan = removal_plan(cfg, corpus, client)
        if not plan:
            raise RuntimeError(f"nothing of a corpus {corpus!r} in {shown(cfg.paths.data_dir)} or in Weaviate: "
                               f"check the name")
        for item in plan:
            log.info("%-26s %-58s %s", item.what, item.where, item.size)
        if not yes:
            log.warning("nothing removed: run again with --yes to remove the %d item(s) above", len(plan))
            return
        for item in plan:
            item.remove()
            log.info("removed %s: %s", item.what, item.where)
    overrides = sum(c == corpus for c, _ in title_overrides(cfg))
    log.info("%s is out of %s and Weaviate. Left as they were: its PDFs, the configs%s, runs/ and the caches "
             "keyed by content", corpus, shown(cfg.paths.data_dir),
             f", and {overrides} title override(s) in {shown(cfg.parsing.title_overrides)}" if overrides else "")
