"""Leitura do corpus.

Markdown com front matter e o formato canonico do corpus versionado. PDF e
suportado porque documento financeiro real chega em PDF; o extrator e
proposital simples, e a limpeza pesada de PDF fica em um projeto proprio de
Document AI.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from finrag.logging_setup import get_logger
from finrag.models import DocumentSource

logger = get_logger(__name__)

_FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}


class DocumentLoadError(RuntimeError):
    """Documento presente no corpus mas ilegivel."""


def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    """Le um front matter ``chave: valor`` simples, sem dependencia de YAML."""
    match = _FRONT_MATTER.match(raw)
    if not match:
        return {}, raw
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip().lower()] = value.strip().strip("\"'")
    return meta, raw[match.end() :]


def _source_from_meta(path: Path, meta: dict[str, str]) -> DocumentSource:
    return DocumentSource(
        doc_id=meta.get("doc_id") or path.stem.lower().replace(" ", "-"),
        title=meta.get("title") or path.stem.replace("-", " ").replace("_", " ").title(),
        path=path.as_posix(),
        doc_type=meta.get("doc_type", "documento"),
        period=meta.get("period") or None,
        version=meta.get("version") or None,
    )


def load_text_document(path: Path) -> tuple[DocumentSource, str]:
    raw = path.read_text(encoding="utf-8")
    meta, body = parse_front_matter(raw)
    return _source_from_meta(path, meta), body


def load_pdf_document(path: Path) -> tuple[DocumentSource, str]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise DocumentLoadError("pypdf nao instalado; nao e possivel ler PDF") from exc

    reader = PdfReader(str(path))
    pages: list[str] = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            # O marcador de pagina vira titulo, o que preserva rastreabilidade
            # na citacao ("pagina 4") depois do chunking.
            pages.append(f"## Pagina {number}\n\n{text}")
    if not pages:
        raise DocumentLoadError(f"Nenhum texto extraido de {path}. PDF digitalizado exige OCR.")
    meta = {"doc_type": "pdf"}
    return _source_from_meta(path, meta), "\n\n".join(pages)


def load_document(path: Path) -> tuple[DocumentSource, str]:
    if path.suffix.lower() == ".pdf":
        return load_pdf_document(path)
    return load_text_document(path)


def iter_corpus(corpus_dir: Path) -> Iterator[tuple[DocumentSource, str]]:
    """Percorre o corpus em ordem estavel, ignorando arquivos nao suportados."""
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Diretorio de corpus inexistente: {corpus_dir}")

    for path in sorted(corpus_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            logger.debug("arquivo_ignorado", path=path.as_posix())
            continue
        try:
            yield load_document(path)
        except (DocumentLoadError, UnicodeDecodeError) as exc:
            logger.error("falha_ao_carregar", path=path.as_posix(), error=str(exc))
