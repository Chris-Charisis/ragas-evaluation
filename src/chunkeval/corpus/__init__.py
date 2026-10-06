"""Stages 1–3: from PDFs to indexed chunks.

  parsing.py    1   PDFs -> CanonicalDoc (Docling), one canonical text per document
  chunking/     2   the chunkers under evaluation (character, sentence, recursive)
  embedding.py  2b  chunk vectors for both embedders, cached on disk
  indexing.py   3   chunks + named vectors -> the Weaviate v2 collections
"""
