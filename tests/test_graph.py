"""Ciclo de Corrective RAG.

Cada teste fixa um ramo do grafo: resposta direta, reescrita da consulta,
abstencao e regeneracao apos auditoria de ancoragem. Sao esses quatro caminhos
que diferenciam o projeto de um RAG ingenuo, e por isso precisam de teste.
"""

from __future__ import annotations

from typing import Any

from finrag.graph import ABSTENTION_MESSAGE, CorrectiveRAGPipeline
from finrag.models import RetrievalRoute
from finrag.observability import METRICS
from tests.helpers import FakeLLMClient, FakeRetriever, make_scored


def build(retriever: Any, client: Any) -> CorrectiveRAGPipeline:
    return CorrectiveRAGPipeline(retriever=retriever, client=client)


async def test_contexto_bom_responde_direto() -> None:
    pipeline = build(FakeRetriever(make_scored(3)), FakeLLMClient())

    answer, trace_id = await pipeline.answer("Qual a taxa de MDR da Adquirente Beta?")

    assert answer.route is RetrievalRoute.DIRECT
    assert answer.grounded is True
    assert answer.rewrites == []
    assert len(answer.citations) == 3
    assert trace_id


async def test_documento_irrelevante_sai_do_prompt() -> None:
    """O grading existe para o contexto nao ser contaminado."""
    verdicts = iter(
        [
            {"relevant": True, "score": 0.9, "reason": "responde"},
            {"relevant": False, "score": 0.1, "reason": "fora do tema"},
            {"relevant": True, "score": 0.8, "reason": "complementa"},
        ]
    )
    client = FakeLLMClient(grade=lambda _: next(verdicts))
    pipeline = build(FakeRetriever(make_scored(3)), client)

    answer, _ = await pipeline.answer("Qual a taxa de MDR?")

    assert len(answer.citations) == 2
    assert answer.route is RetrievalRoute.DIRECT


async def test_contexto_pobre_dispara_reescrita() -> None:
    """Poucos documentos relevantes: reformula e busca de novo antes de desistir."""
    retriever = FakeRetriever(make_scored(1), make_scored(3))
    grades = iter([{"relevant": False, "score": 0.1, "reason": "nao trata"}])
    client = FakeLLMClient(
        grade=lambda _: next(grades, {"relevant": True, "score": 0.9, "reason": "trata"}),
        rewrite="taxa de desconto do lojista por bandeira",
    )
    pipeline = build(retriever, client)

    answer, _ = await pipeline.answer("quanto some do valor da venda?")

    assert answer.rewrites == ["taxa de desconto do lojista por bandeira"]
    assert answer.route is RetrievalRoute.REWRITTEN
    assert retriever.queries == [
        "quanto some do valor da venda?",
        "taxa de desconto do lojista por bandeira",
    ]
    assert answer.citations


async def test_sem_contexto_o_sistema_se_abstem() -> None:
    """Resposta inventada em conciliacao vira decisao errada de caixa."""
    client = FakeLLMClient(grade={"relevant": False, "score": 0.0, "reason": "nada a ver"})
    pipeline = build(FakeRetriever(make_scored(2)), client)

    answer, _ = await pipeline.answer("Qual e a politica de home office?")

    assert answer.route is RetrievalRoute.INSUFFICIENT_CONTEXT
    assert answer.answer == ABSTENTION_MESSAGE
    assert answer.citations == []
    assert "generate" not in client.calls


async def test_reescrita_respeita_o_limite_configurado(monkeypatch: Any) -> None:
    """Sem limite, contexto ruim faria o grafo girar indefinidamente."""
    monkeypatch.setenv("FINRAG_GRAPH__MAX_REWRITES", "1")
    from finrag.settings import reset_settings_cache

    reset_settings_cache()
    client = FakeLLMClient(grade={"relevant": False, "score": 0.0, "reason": "nada"})
    retriever = FakeRetriever(make_scored(2))
    pipeline = build(retriever, client)

    answer, _ = await pipeline.answer("pergunta sem resposta no corpus")

    assert len(answer.rewrites) == 1
    assert len(retriever.queries) == 2
    assert answer.route is RetrievalRoute.INSUFFICIENT_CONTEXT


async def test_afirmacao_sem_respaldo_gera_nova_redacao() -> None:
    client = FakeLLMClient(
        grounding={
            "grounded": False,
            "unsupported_claims": ["a taxa cai para 2% em dezembro"],
            "reason": "nao consta",
        },
        generate="A taxa e 3,15% e cai para 2% em dezembro.",
        regenerate="A taxa e 3,15% [1].",
    )
    pipeline = build(FakeRetriever(make_scored(2)), client)

    answer, _ = await pipeline.answer("Qual a taxa de MDR?")

    assert answer.route is RetrievalRoute.REGENERATED
    assert answer.answer == "A taxa e 3,15% [1]."
    assert "regenerate" in client.calls


async def test_regeneracao_acontece_uma_vez_so() -> None:
    """Loop de regeneracao queimaria tokens sem convergir."""
    client = FakeLLMClient(
        grounding={"grounded": False, "unsupported_claims": ["x"], "reason": "nao consta"}
    )
    pipeline = build(FakeRetriever(make_scored(2)), client)

    answer, _ = await pipeline.answer("Qual a taxa de MDR?")

    assert client.calls.count("regenerate") == 1
    assert answer.grounded is False


async def test_falha_no_avaliador_nao_descarta_documento() -> None:
    """Em duvida o documento passa: a geracao decide o que usar."""
    client = FakeLLMClient(grade=RuntimeError("provedor instavel"))
    pipeline = build(FakeRetriever(make_scored(3)), client)

    answer, _ = await pipeline.answer("Qual a taxa de MDR?")

    assert answer.citations
    assert answer.route is RetrievalRoute.DIRECT


async def test_auditoria_pode_ser_desligada(monkeypatch: Any) -> None:
    monkeypatch.setenv("FINRAG_GRAPH__ENABLE_GROUNDING_CHECK", "false")
    from finrag.settings import reset_settings_cache

    reset_settings_cache()
    client = FakeLLMClient()
    pipeline = build(FakeRetriever(make_scored(2)), client)

    answer, _ = await pipeline.answer("Qual a taxa de MDR?")

    assert answer.grounded is None
    assert "grounding" not in client.calls


async def test_filtros_chegam_ao_recuperador() -> None:
    retriever = FakeRetriever(make_scored(2))
    pipeline = build(retriever, FakeLLMClient())

    await pipeline.answer("Qual a taxa?", doc_types=["tabela"], periods=["2025-09"])

    assert retriever.filters[0] == {"doc_types": ["tabela"], "periods": ["2025-09"]}


async def test_custo_e_consumo_sao_contabilizados() -> None:
    """Custo descoberto no fim do mes e custo que ninguem controla."""
    pipeline = build(FakeRetriever(make_scored(2)), FakeLLMClient())

    answer, _ = await pipeline.answer("Qual a taxa de MDR?")

    assert answer.usage.calls >= 3
    assert answer.usage.total_tokens > 0
    assert answer.cost_usd > 0
    assert answer.latency_ms > 0


async def test_metricas_registram_a_rota() -> None:
    client = FakeLLMClient(grade={"relevant": False, "score": 0.0, "reason": "nada"})
    pipeline = build(FakeRetriever(make_scored(2)), client)

    await pipeline.answer("pergunta impossivel")

    snapshot = METRICS.snapshot()
    assert snapshot["counters"]["queries_total"] == 1.0
    assert snapshot["counters"]["abstentions_total"] == 1.0
    assert snapshot["routes"]["insufficient_context"] == 1


async def test_pergunta_original_e_usada_na_resposta() -> None:
    """A reescrita serve a busca, nao ao usuario: a citacao volta ao original."""
    captured: list[str] = []

    def capture(messages: Any) -> str:
        captured.append(str(messages[-1].content))
        return "resposta"

    retriever = FakeRetriever(make_scored(1), make_scored(3))
    grades = iter([{"relevant": False, "score": 0.1, "reason": "nao"}])
    client = FakeLLMClient(
        grade=lambda _: next(grades, {"relevant": True, "score": 0.9, "reason": "sim"}),
        rewrite="consulta reformulada",
        generate=capture,
    )
    pipeline = build(retriever, client)

    answer, _ = await pipeline.answer("pergunta do usuario")

    assert answer.question == "pergunta do usuario"
    assert "pergunta do usuario" in captured[0]
