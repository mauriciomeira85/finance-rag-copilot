"""Leitura do corpus, chunking estrutural e indexacao."""

from finrag.ingestion.chunking import Block, chunk_markdown, estimate_tokens, parse_blocks
from finrag.ingestion.loaders import (
    DocumentLoadError,
    iter_corpus,
    load_document,
    parse_front_matter,
)
from finrag.ingestion.pipeline import IngestionReport, build_chunks, ingest

__all__ = [
    "Block",
    "DocumentLoadError",
    "IngestionReport",
    "build_chunks",
    "chunk_markdown",
    "estimate_tokens",
    "ingest",
    "iter_corpus",
    "load_document",
    "parse_blocks",
    "parse_front_matter",
]
