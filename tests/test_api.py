"""Contrato HTTP.

O pipeline e injetado direto no estado da aplicacao. O ``TestClient`` e usado
sem gerenciador de contexto de proposito: assim o lifespan nao roda e as rotas
sao exercitadas sem carregar modelo de embedding nem abrir indice.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from finrag.api import main as api_main
from finrag.graph import CorrectiveRAGPipeline
from finrag.llm import LLMProviderError
from tests.helpers import FakeLLMClient, FakeRetriever, make_scored


def build_pipeline(**overrides: Any) -> CorrectiveRAGPipeline:
    return CorrectiveRAGPipeline(
        retriever=FakeRetriever(make_scored(2)),  # type: ignore[arg-type]
        client=FakeLLMClient(**overrides),  # type: ignore[arg-type]
    )


@pytest.fixture
def client() -> Iterator[TestClient]:
    api_main._state["pipeline"] = build_pipeline()
    yield TestClient(api_main.app)
    api_main._state.clear()


@pytest.fixture
def broken_client() -> Iterator[TestClient]:
    api_main._state["pipeline"] = None
    api_main._state["error"] = "FINRAG_LLM__API_KEY nao definida"
    yield TestClient(api_main.app)
    api_main._state.clear()


def test_health_reporta_a_configuracao_efetiva(client: TestClient) -> None:
    payload = client.get("/health").json()

    assert payload["status"] == "ok"
    assert payload["indexed_chunks"] == 12
    assert payload["llm_configured"] is True
    assert payload["reranker"] in {"llm", "cross_encoder", "none"}


def test_health_responde_mesmo_sem_llm(broken_client: TestClient) -> None:
    """O healthcheck e como o operador descobre falta de chave, sem crash em loop."""
    payload = broken_client.get("/health").json()

    assert payload["status"] == "degraded"
    assert payload["llm_configured"] is False


def test_recusa_do_provedor_vira_502(monkeypatch: pytest.MonkeyPatch) -> None:
    """A causa esta fora do processo, entao 500 mandaria depurar o lugar errado."""
    pipeline = build_pipeline()

    async def refuse(*args: Any, **kwargs: Any) -> Any:
        raise LLMProviderError(402, "saldo insuficiente na conta do provedor (HTTP 402)")

    monkeypatch.setattr(pipeline, "answer", refuse)
    api_main._state["pipeline"] = pipeline
    http = TestClient(api_main.app, raise_server_exceptions=False)

    response = http.post("/query", json={"question": "Qual a taxa de MDR?"})
    api_main._state.clear()

    assert response.status_code == 502
    assert response.json()["error_type"] == "llm_provider_error"
    assert "saldo insuficiente" in response.json()["detail"]


def test_consulta_devolve_resposta_com_citacoes(client: TestClient) -> None:
    response = client.post("/query", json={"question": "Qual a taxa de MDR?"})
    payload = response.json()

    assert response.status_code == 200
    assert payload["answer"]
    assert payload["route"] == "direct"
    assert payload["grounded"] is True
    assert len(payload["citations"]) == 2
    assert payload["citations"][0]["chunk_id"]
    assert payload["trace_id"]
    assert payload["cost_usd"] > 0


def test_contexto_integral_nao_vaza_na_resposta(client: TestClient) -> None:
    """O contexto existe para a auditoria de fidelidade, nao para o cliente."""
    payload = client.post("/query", json={"question": "Qual a taxa de MDR?"}).json()

    assert "context" not in payload
    assert all(len(citation["excerpt"]) <= 340 for citation in payload["citations"])


def test_consulta_aceita_filtros(client: TestClient) -> None:
    response = client.post(
        "/query",
        json={"question": "Qual a taxa de MDR?", "doc_types": ["tabela"], "periods": ["2025-09"]},
    )

    assert response.status_code == 200


def test_pergunta_curta_e_rejeitada_na_borda(client: TestClient) -> None:
    response = client.post("/query", json={"question": "oi"})

    assert response.status_code == 422


def test_consulta_sem_pipeline_devolve_503(broken_client: TestClient) -> None:
    response = broken_client.post("/query", json={"question": "Qual a taxa de MDR?"})

    assert response.status_code == 503
    assert "API_KEY" in response.json()["detail"]


def test_streaming_emite_estagios_e_resposta(client: TestClient) -> None:
    with client.stream("POST", "/query/stream", json={"question": "Qual a taxa?"}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events: list[tuple[str, dict[str, Any]]] = []
        name = ""
        for line in response.iter_lines():
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                events.append((name, json.loads(line.split(":", 1)[1].strip())))

    kinds = [name for name, _ in events]
    assert "answer" in kinds
    final = next(payload for name, payload in events if name == "answer")
    assert final["citations"]


def test_streaming_reporta_erro_como_evento(client: TestClient) -> None:
    """Erro no meio do stream nao pode virar conexao pendurada."""
    api_main._state["pipeline"] = build_pipeline(generate=RuntimeError("provedor fora do ar"))

    with client.stream("POST", "/query/stream", json={"question": "Qual a taxa?"}) as response:
        body = "".join(response.iter_text())

    assert "event: error" in body
    assert "provedor fora do ar" in body


def test_metricas_em_formato_prometheus(client: TestClient) -> None:
    client.post("/query", json={"question": "Qual a taxa de MDR?"})

    body = client.get("/metrics").text

    assert "finrag_queries_total 1.0" in body
    assert 'finrag_route_total{route="direct"} 1' in body


def test_estatisticas_agregam_custo(client: TestClient) -> None:
    client.post("/query", json={"question": "Qual a taxa de MDR?"})

    payload = client.get("/stats").json()

    assert payload["counters"]["queries_total"] == 1.0
    assert payload["cost_usd_total"] > 0
    assert payload["routes"]["direct"] == 1


def test_trace_da_consulta_pode_ser_recuperado(client: TestClient) -> None:
    trace_id = client.post("/query", json={"question": "Qual a taxa de MDR?"}).json()["trace_id"]

    payload = client.get(f"/traces/{trace_id}").json()

    assert payload["trace_id"] == trace_id
    assert any(span["name"] == "query" for span in payload["spans"])


def test_trace_inexistente_devolve_404(client: TestClient) -> None:
    assert client.get("/traces/nao-existe").status_code == 404


def test_documentacao_openapi_descreve_as_rotas(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    assert "/query" in schema["paths"]
    assert "/query/stream" in schema["paths"]
    assert schema["info"]["title"] == "finance-rag-copilot"
