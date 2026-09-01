"""Leitura do corpus e pipeline de ingestao."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from finrag.ingestion import build_chunks, ingest, iter_corpus, load_document, parse_front_matter
from finrag.ingestion.loaders import DocumentLoadError, load_pdf_document

DOCUMENT = """---
doc_id: pol-mdr
title: Tabela de MDR
doc_type: tabela
period: 2025-09
version: "3"
---

# Tabela de MDR

## Adquirente Beta

O credito parcelado em 2 a 6 vezes tem taxa de 3,15% para Visa e Mastercard,
e a liquidacao ocorre em trinta dias corridos apos a captura da transacao.
"""


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    directory = tmp_path / "corpus"
    directory.mkdir()
    (directory / "01-mdr.md").write_text(DOCUMENT, encoding="utf-8")
    (directory / "leia-me.docx").write_bytes(b"binario qualquer")
    return directory


def test_front_matter_alimenta_a_proveniencia() -> None:
    meta, body = parse_front_matter(DOCUMENT)

    assert meta["doc_id"] == "pol-mdr"
    assert meta["version"] == "3"
    assert body.lstrip().startswith("# Tabela de MDR")


def test_documento_sem_front_matter_deriva_metadados(tmp_path: Path) -> None:
    path = tmp_path / "politica-de-caixa.md"
    path.write_text("# Titulo\n\nTexto.\n", encoding="utf-8")

    source, body = load_document(path)

    assert source.doc_id == "politica-de-caixa"
    assert source.title == "Politica De Caixa"
    assert source.doc_type == "documento"
    assert source.period is None
    assert body.startswith("# Titulo")


def test_formato_nao_suportado_e_ignorado(corpus: Path) -> None:
    documents = list(iter_corpus(corpus))

    assert len(documents) == 1
    assert documents[0][0].doc_id == "pol-mdr"


def test_corpus_inexistente_falha_alto(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="corpus inexistente"):
        list(iter_corpus(tmp_path / "nao-existe"))


def test_arquivo_ilegivel_nao_interrompe_a_ingestao(corpus: Path) -> None:
    """Um documento problematico nao pode impedir a indexacao dos outros."""
    (corpus / "02-quebrado.md").write_bytes(b"\xff\xfe\x00 invalido em utf-8")

    documents = list(iter_corpus(corpus))

    assert [source.doc_id for source, _ in documents] == ["pol-mdr"]


def test_pdf_sem_texto_pede_ocr(tmp_path: Path) -> None:
    from pypdf import PdfWriter

    path = tmp_path / "digitalizado.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with path.open("wb") as handle:
        writer.write(handle)

    with pytest.raises(DocumentLoadError, match="OCR"):
        load_pdf_document(path)


def test_relatorio_de_chunks_por_documento(corpus: Path) -> None:
    chunks, report = build_chunks(corpus)

    assert report.documents == 1
    assert report.chunks == len(chunks)
    assert report.per_document["pol-mdr"] == len(chunks)
    assert report.skipped == []


def test_ingestao_indexa_e_relata(corpus: Path, store: Any) -> None:
    report = ingest(corpus, store, recreate=True)

    assert report.indexed == report.chunks
    assert store.count() == report.chunks
    assert report.as_dict()["documents"] == 1


def test_corpus_versionado_do_projeto_continua_legivel() -> None:
    """Pega edicao no corpus que quebre front matter ou zere um documento."""
    _, report = build_chunks(Path(__file__).resolve().parent.parent / "data" / "corpus")

    assert report.documents == 10
    assert report.chunks > 50
    assert report.skipped == []
