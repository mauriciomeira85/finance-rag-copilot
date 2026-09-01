"""Indice hibrido e recuperador.

Roda contra o Qdrant embutido em memoria com embeddings falsos: verifica o
encanamento (dois vetores por ponto, filtros de payload, fusao, re-rank e
fallback), nao a qualidade semantica do modelo real.
"""

from __future__ import annotations

from typing import Any

import pytest

from finrag.models import ScoredChunk
from finrag.retrieval.reranker import (
    CrossEncoderReranker,
    LLMReranker,
    NoOpReranker,
    build_reranker,
)
from finrag.retrieval.retriever import HybridRetriever
from finrag.settings import RerankerKind, RetrievalSettings
from tests.helpers import FakeLLMClient, make_chunk, make_scored

CORPUS = [
    make_chunk(
        chunk_id="mdr-0000-aaa",
        doc_id="mdr",
        text="Adquirente Beta ADQ-8802 credito parcelado 2 a 6x taxa de 3,15%",
        doc_type="tabela",
        period="2025-09",
    ),
    make_chunk(
        chunk_id="mdr-0001-bbb",
        doc_id="mdr",
        text="Adquirente Alfa ADQ-4471 credito a vista taxa de 2,45%",
        doc_type="tabela",
        period="2025-09",
    ),
    make_chunk(
        chunk_id="dre-0000-ccc",
        doc_id="dre",
        text="Margem EBITDA consolidada do terceiro trimestre de 2025 foi de 18,4%",
        doc_type="dre",
        period="2025-Q3",
    ),
]


@pytest.fixture
def indexed(store: Any) -> Any:
    store.ensure_collection(recreate=True)
    store.upsert(CORPUS)
    return store


def test_ponto_guarda_os_dois_vetores(indexed: Any) -> None:
    """Sem os dois vetores no mesmo ponto nao ha busca hibrida de verdade."""
    points = indexed.client.scroll(indexed.collection, limit=1, with_vectors=True)[0]

    assert points
    assert set(points[0].vector) == {"dense", "sparse"}


def test_contagem_reflete_o_corpus(indexed: Any) -> None:
    assert indexed.count() == len(CORPUS)


def test_reindexar_nao_duplica(indexed: Any) -> None:
    """O id do ponto vem do chunk_id, entao upsert repetido sobrescreve."""
    indexed.upsert(CORPUS)

    assert indexed.count() == len(CORPUS)


def test_busca_lexical_acha_codigo_exato(indexed: Any) -> None:
    """O motivo de existir o indice esparso: codigo de adquirente."""
    results = indexed.search_sparse("ADQ-8802", limit=3)

    assert results
    assert results[0].chunk.chunk_id == "mdr-0000-aaa"
    assert results[0].sparse_score is not None


def test_filtro_por_tipo_de_documento(indexed: Any) -> None:
    results = indexed.search_dense("taxa", limit=5, doc_types=["dre"])

    assert {item.chunk.source.doc_type for item in results} == {"dre"}


def test_filtro_por_competencia(indexed: Any) -> None:
    results = indexed.search_sparse("2025", limit=5, periods=["2025-Q3"])

    assert all(item.chunk.source.period == "2025-Q3" for item in results)


def test_remocao_por_documento(indexed: Any) -> None:
    indexed.delete_document("mdr")

    assert indexed.count() == 1


def test_recriar_colecao_limpa_o_indice(indexed: Any) -> None:
    indexed.ensure_collection(recreate=True)

    assert indexed.count() == 0


async def test_recuperador_funde_e_reordena(indexed: Any) -> None:
    retriever = HybridRetriever(
        store=indexed,
        reranker=NoOpReranker(),
        settings=RetrievalSettings(top_k_dense=3, top_k_sparse=3, top_k_final=2),
    )

    result = await retriever.retrieve("taxa de MDR do credito parcelado ADQ-8802")

    assert len(result.chunks) == 2
    assert result.dense_hits > 0
    assert result.sparse_hits > 0
    assert all(item.fused_score is not None for item in result.chunks)


async def test_ramo_desligado_nao_consulta_o_indice(indexed: Any) -> None:
    """``top_k=0`` isola um dos indices; e a base do benchmark de ablacao."""
    retriever = HybridRetriever(
        store=indexed,
        reranker=NoOpReranker(),
        settings=RetrievalSettings(top_k_dense=0, top_k_sparse=3, top_k_final=3),
    )

    result = await retriever.retrieve("ADQ-8802")

    assert result.dense_hits == 0
    assert result.sparse_hits > 0
    assert all(item.dense_score is None for item in result.chunks)


async def test_rerank_por_llm_reordena_pela_nota() -> None:
    client = FakeLLMClient(rerank={"ranking": [{"index": 3, "score": 0.99}]})
    candidates = make_scored(3)

    ranked = await LLMReranker(client).rerank("pergunta", candidates, top_k=3)  # type: ignore[arg-type]

    assert ranked[0].chunk.chunk_id == candidates[2].chunk.chunk_id
    assert ranked[0].rerank_score == 0.99


async def test_rerank_ignora_indice_fora_da_faixa() -> None:
    """Modelo alucinando posicao nao pode gerar IndexError."""
    client = FakeLLMClient(rerank={"ranking": [{"index": 99, "score": 0.9}]})
    candidates = make_scored(2)

    ranked = await LLMReranker(client).rerank("pergunta", candidates, top_k=2)  # type: ignore[arg-type]

    assert [item.chunk.chunk_id for item in ranked] == [
        candidates[0].chunk.chunk_id,
        candidates[1].chunk.chunk_id,
    ]


async def test_falha_no_rerank_preserva_a_ordem_da_fusao() -> None:
    """Componente de qualidade nao pode derrubar a resposta."""
    client = FakeLLMClient(rerank=RuntimeError("provedor fora do ar"))
    candidates = make_scored(3)

    ranked = await LLMReranker(client).rerank("pergunta", candidates, top_k=2)  # type: ignore[arg-type]

    assert [item.chunk.chunk_id for item in ranked] == [
        candidates[0].chunk.chunk_id,
        candidates[1].chunk.chunk_id,
    ]


async def test_cross_encoder_ordena_pela_sigmoide_do_logit() -> None:
    """O logit do cross-encoder vira [0, 1] para ficar comparavel ao re-rank por LLM."""
    reranker = CrossEncoderReranker("modelo-falso")
    reranker._model = type("Model", (), {"rerank": lambda self, q, docs: [-4.0, 6.0, 0.0]})()
    candidates = make_scored(3)

    ranked = await reranker.rerank("pergunta", candidates, top_k=3)

    assert [item.chunk.chunk_id for item in ranked] == [
        candidates[1].chunk.chunk_id,
        candidates[2].chunk.chunk_id,
        candidates[0].chunk.chunk_id,
    ]
    assert ranked[0].rerank_score == pytest.approx(0.9975, abs=1e-4)
    assert ranked[1].rerank_score == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("configurado", "esperado"),
    [("none", NoOpReranker), ("llm", LLMReranker), ("cross_encoder", CrossEncoderReranker)],
)
def test_reranker_e_escolhido_pela_configuracao(configurado: RerankerKind, esperado: type) -> None:
    reranker = build_reranker(
        RetrievalSettings(reranker=configurado),
        client=FakeLLMClient(),  # type: ignore[arg-type]
    )

    assert isinstance(reranker, esperado)


async def test_cross_encoder_indisponivel_cai_no_fallback() -> None:
    reranker = CrossEncoderReranker("modelo-que-nao-existe")
    candidates = make_scored(3)

    ranked = await reranker.rerank("pergunta", candidates, top_k=2)

    assert len(ranked) == 2


async def test_candidato_unico_dispensa_rerank() -> None:
    client = FakeLLMClient(rerank=RuntimeError("nao deveria ser chamado"))
    single: list[ScoredChunk] = make_scored(1)

    ranked = await LLMReranker(client).rerank("pergunta", single, top_k=5)  # type: ignore[arg-type]

    assert ranked == single
    assert client.calls == []
