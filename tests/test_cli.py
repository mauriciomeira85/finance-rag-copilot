"""Comandos da linha de comando.

A CLI e a porta de entrada usada na demo e no CI, entao vale garantir que
cada comando roda, imprime o que promete e devolve o codigo de saida certo.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from finrag.cli import app
from finrag.graph import CorrectiveRAGPipeline
from finrag.llm import LLMProviderError
from tests.helpers import FakeLLMClient, FakeRetriever, make_scored

runner = CliRunner()


@pytest.fixture
def fake_pipeline(monkeypatch: pytest.MonkeyPatch) -> CorrectiveRAGPipeline:
    """Substitui a construcao do pipeline em todos os pontos de uso da CLI."""
    pipeline = CorrectiveRAGPipeline(
        retriever=FakeRetriever(make_scored(2)),  # type: ignore[arg-type]
        client=FakeLLMClient(generate="A taxa e de 3,15% [1]."),  # type: ignore[arg-type]
    )
    monkeypatch.setattr("finrag.graph.CorrectiveRAGPipeline", lambda: pipeline)
    return pipeline


def test_version_mostra_a_configuracao_efetiva() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert "chave configurada" in result.stdout


def test_corpus_resume_chunks_por_documento() -> None:
    result = runner.invoke(app, ["corpus"])

    assert result.exit_code == 0
    assert "documentos" in result.stdout
    assert "tabulares" in result.stdout


def test_ask_imprime_resposta_fontes_e_custo(fake_pipeline: CorrectiveRAGPipeline) -> None:
    result = runner.invoke(app, ["ask", "Qual a taxa de MDR?"])

    assert result.exit_code == 0, result.stdout
    assert "3,15%" in result.stdout
    assert "Fontes" in result.stdout
    assert "custo" in result.stdout


def test_ask_com_trace_lista_os_spans(fake_pipeline: CorrectiveRAGPipeline) -> None:
    result = runner.invoke(app, ["ask", "Qual a taxa de MDR?", "--show-trace"])

    assert result.exit_code == 0, result.stdout
    assert "Trace" in result.stdout


def test_ask_traduz_recusa_do_provedor(
    fake_pipeline: CorrectiveRAGPipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Saldo esgotado no meio de uma demo nao deve virar traceback."""

    async def refuse(*args: object, **kwargs: object) -> object:
        raise LLMProviderError(402, "saldo insuficiente na conta do provedor (HTTP 402)")

    monkeypatch.setattr(fake_pipeline, "answer", refuse)

    result = runner.invoke(app, ["ask", "Qual a taxa de MDR?"])

    assert result.exit_code == 2
    assert "saldo insuficiente" in result.stdout
    assert "Traceback" not in result.stdout


def test_eval_falha_quando_abaixo_do_minimo(
    fake_pipeline: CorrectiveRAGPipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """E assim que o CI barra merge que degrada a qualidade."""
    dataset = tmp_path / "golden.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "caso-1",
                "question": "Qual a taxa?",
                "reference_answer": "3,15%",
                "expected_doc_ids": ["outro-documento"],
                "must_include": ["99,99%"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FINRAG_GOLDEN_DATASET", str(dataset))
    from finrag.settings import reset_settings_cache

    reset_settings_cache()

    result = runner.invoke(app, ["eval", "--min-pass-rate", "0.9", "--no-save"])

    assert result.exit_code == 1
    assert "abaixo do minimo" in result.stdout


def test_eval_aprovado_salva_relatorio(
    fake_pipeline: CorrectiveRAGPipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "golden.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "caso-1",
                "question": "Qual a taxa?",
                "reference_answer": "3,15%",
                "expected_doc_ids": ["mdr"],
                "must_include": ["3,15%"],
                "category": "taxas",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FINRAG_GOLDEN_DATASET", str(dataset))
    reports = tmp_path / "reports"
    monkeypatch.setenv("FINRAG_REPORTS_DIR", str(reports))
    from finrag.settings import reset_settings_cache

    reset_settings_cache()

    result = runner.invoke(app, ["eval", "--min-pass-rate", "1.0"])

    assert result.exit_code == 0, result.stdout
    assert (reports / "eval-latest.json").exists()
    assert "Por categoria" in result.stdout


def test_eval_com_categoria_inexistente_falha_claro(
    fake_pipeline: CorrectiveRAGPipeline,
) -> None:
    result = runner.invoke(app, ["eval", "--category", "categoria-que-nao-existe"])

    assert result.exit_code == 1
    assert "Nenhum caso selecionado" in result.stdout


def test_ingest_relata_o_que_indexou(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from tests.helpers import FakeEmbedder

    monkeypatch.setenv("FINRAG_VECTORSTORE__URL", ":memory:")
    from finrag.retrieval.vectorstore import HybridVectorStore
    from finrag.settings import reset_settings_cache

    reset_settings_cache()

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc.md").write_text(
        "---\ndoc_id: doc\ndoc_type: politica\n---\n\n# Doc\n\n## Secao\n\n"
        + "Texto de politica com tamanho suficiente para virar um chunk valido. " * 3,
        encoding="utf-8",
    )

    def build_store() -> HybridVectorStore:
        return HybridVectorStore(embedder=FakeEmbedder())  # type: ignore[arg-type]

    monkeypatch.setattr("finrag.retrieval.HybridVectorStore", build_store)

    result = runner.invoke(app, ["ingest", "--corpus-dir", str(corpus)])

    assert result.exit_code == 0, result.stdout
    assert "Indexacao concluida" in result.stdout
    assert "1 documentos" in result.stdout
