"""Tracing e metricas."""

from __future__ import annotations

import json

import pytest

from finrag.observability import METRICS, current_trace_id, get_recorder, span, trace


def test_span_filho_herda_o_trace_do_pai() -> None:
    """Sem hierarquia o trace nao mostra qual etapa chamou qual."""
    with trace("query") as root:
        trace_id = current_trace_id()
        with span("retrieval") as child:
            child.attributes["top_k"] = 5

    assert trace_id == root.trace_id
    records = get_recorder().read_trace(root.trace_id)
    names = {record["name"]: record for record in records}
    assert names["retrieval"]["parent_id"] == root.span_id
    assert names["retrieval"]["attributes"]["top_k"] == 5
    assert names["query"]["parent_id"] is None


def test_trace_sai_do_escopo_ao_terminar() -> None:
    with trace("query"):
        pass

    assert current_trace_id() is None


def test_excecao_marca_o_span_e_propaga() -> None:
    with (
        pytest.raises(RuntimeError, match="falha simulada"),
        trace("query") as root,
        span("generate"),
    ):
        raise RuntimeError("falha simulada")

    record = next(
        item for item in get_recorder().read_trace(root.trace_id) if item["name"] == "generate"
    )
    assert record["status"] == "error"
    assert "RuntimeError: falha simulada" in record["error"]


def test_atributo_gigante_e_truncado() -> None:
    """Trace nao pode virar deposito de prompt inteiro."""
    with trace("query", prompt="x" * 5000) as root:
        pass

    record = get_recorder().read_trace(root.trace_id)[0]
    assert record["attributes"]["prompt"].endswith("...[truncado]")
    assert len(record["attributes"]["prompt"]) < 5000


def test_valor_nao_serializavel_vira_texto() -> None:
    with trace("query", objeto=object()) as root:
        pass

    record = get_recorder().read_trace(root.trace_id)[0]
    assert isinstance(record["attributes"]["objeto"], str)


def test_arquivo_de_trace_e_jsonl_valido() -> None:
    with trace("query"), span("retrieval"):
        pass

    content = get_recorder().path.read_text(encoding="utf-8")
    lines = [line for line in content.splitlines() if line.strip()]
    assert len(lines) == 2
    assert all(json.loads(line)["trace_id"] for line in lines)


def test_trace_inexistente_devolve_lista_vazia() -> None:
    assert get_recorder().read_trace("nao-existe") == []


def test_percentis_da_latencia() -> None:
    for value in range(1, 101):
        METRICS.observe("query_latency_ms", float(value))

    latency = METRICS.snapshot()["latency_ms"]

    assert latency["count"] == 100
    assert latency["p50"] == pytest.approx(51.0, abs=1.0)
    assert latency["p95"] == pytest.approx(96.0, abs=1.0)


def test_custo_por_consulta_divide_pelo_total() -> None:
    METRICS.increment("queries_total", 4)
    METRICS.increment("llm_cost_usd", 0.02)

    snapshot = METRICS.snapshot()

    assert snapshot["cost_usd_total"] == pytest.approx(0.02)
    assert snapshot["cost_usd_per_query"] == pytest.approx(0.005)


def test_sem_consulta_nao_divide_por_zero() -> None:
    assert METRICS.snapshot()["cost_usd_per_query"] == 0.0


def test_exposicao_prometheus_tem_tipo_e_rotulo() -> None:
    METRICS.increment("queries_total", 3)
    METRICS.observe("query_latency_ms", 120.0)
    METRICS.count_route("direct")

    output = METRICS.render_prometheus()

    assert "# TYPE finrag_queries_total counter" in output
    assert "finrag_queries_total 3.0" in output
    assert 'finrag_query_latency_ms{quantile="0.95"} 120.0' in output
    assert 'finrag_route_total{route="direct"} 1' in output
