"""Pipeline de ingestao: corpus -> chunks -> indice hibrido."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from finrag.ingestion.chunking import chunk_markdown
from finrag.ingestion.loaders import iter_corpus
from finrag.logging_setup import get_logger
from finrag.models import Chunk
from finrag.observability import span
from finrag.retrieval.vectorstore import HybridVectorStore
from finrag.settings import get_settings

logger = get_logger(__name__)


@dataclass(slots=True)
class IngestionReport:
    documents: int = 0
    chunks: int = 0
    indexed: int = 0
    per_document: dict[str, int] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "documents": self.documents,
            "chunks": self.chunks,
            "indexed": self.indexed,
            "per_document": self.per_document,
            "skipped": self.skipped,
        }


def build_chunks(corpus_dir: Path | None = None) -> tuple[list[Chunk], IngestionReport]:
    """Le o corpus e produz os chunks, sem tocar no banco vetorial.

    Separar essa etapa da indexacao permite testar o chunking sem subir o
    Qdrant nem baixar modelo de embedding.
    """
    settings = get_settings()
    corpus_dir = corpus_dir or settings.corpus_dir
    report = IngestionReport()
    chunks: list[Chunk] = []

    for source, body in iter_corpus(corpus_dir):
        report.documents += 1
        produced = chunk_markdown(body, source, settings.chunking)
        if not produced:
            report.skipped.append(source.path)
            logger.warning("documento_sem_chunks", doc_id=source.doc_id)
            continue
        report.per_document[source.doc_id] = len(produced)
        chunks.extend(produced)

    report.chunks = len(chunks)
    return chunks, report


def ingest(
    corpus_dir: Path | None = None,
    store: HybridVectorStore | None = None,
    recreate: bool = True,
) -> IngestionReport:
    """Executa a ingestao completa e devolve o relatorio.

    ``recreate=True`` e o padrao porque o corpus e pequeno e versionado: e mais
    seguro reconstruir do zero do que conviver com chunk orfao de um documento
    que mudou de estrutura.
    """
    store = store or HybridVectorStore()
    with span("ingestion.run") as current:
        chunks, report = build_chunks(corpus_dir)
        store.ensure_collection(recreate=recreate)
        report.indexed = store.upsert(chunks)
        current.attributes.update(
            documents=report.documents, chunks=report.chunks, indexed=report.indexed
        )

    logger.info(
        "ingestao_concluida",
        documents=report.documents,
        chunks=report.chunks,
        indexed=report.indexed,
    )
    return report
