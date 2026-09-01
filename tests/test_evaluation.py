"""Metricas de qualidade e execucao da avaliacao."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from finrag.evaluation import (
    Evaluator,
    GoldenCase,
    abstention_correct,
    contains_fact,
    fact_coverage,
    load_golden_dataset,
    normalize,
    retrieval_metrics,
    save_report,
)
from finrag.graph import CorrectiveRAGPipeline
from finrag.models import Answer, Citation, RetrievalRoute
from tests.helpers import FakeLLMClient, FakeRetriever, make_scored


def answer_with(*chunk_ids: str, route: RetrievalRoute = RetrievalRoute.DIRECT) -> Answer:
    return Answer(
        question="Qual a taxa?",
        answer="A taxa e de 3,15%.",
        citations=[
            Citation(chunk_id=chunk_id, document="Doc", excerpt="trecho", score=0.9)
            for chunk_id in chunk_ids
        ],
        route=route,
    )


@pytest.mark.parametrize(
    ("texto", "fato"),
    [
        ("A taxa é de 3,15% ao mês.", "3,15%"),
        ("A taxa e de 3.15% ao mes.", "3,15%"),
        ("Receita de R$ 1.256.750.000 no periodo.", "1256750000"),
        ("A MARGEM EBITDA foi de 18,4%.", "margem ebitda"),
        ("Liquidação em 30 dias corridos.", "30 dias corridos"),
    ],
)
def test_fato_e_reconhecido_apesar_da_formatacao(texto: str, fato: str) -> None:
    """Numero em documento brasileiro alterna virgula e ponto; a metrica nao pode falhar por isso."""
    assert contains_fact(texto, fato)


def test_fato_ausente_nao_e_inventado() -> None:
    assert not contains_fact("A taxa e de 3,15%.", "4,05%")


def test_normalizacao_remove_acento_e_caixa() -> None:
    assert normalize("Liquidação  EM 30 Dias") == "liquidacao em 30 dias"


def test_cobertura_de_fatos_lista_o_que_faltou() -> None:
    coverage, missing = fact_coverage("A taxa e 3,15% para Visa.", ["3,15%", "Visa", "Elo"])

    assert coverage == pytest.approx(2 / 3)
    assert missing == ["Elo"]


def test_documento_esperado_e_extraido_do_chunk_id() -> None:
    metrics = retrieval_metrics(answer_with("mdr-0000-abc", "dre-0003-def"), ["mdr"])

    assert metrics.hit is True
    assert metrics.context_precision == pytest.approx(0.5)
    assert metrics.context_recall == 1.0
    assert metrics.retrieved_docs == ["mdr", "dre"]


def test_nenhum_documento_esperado_premia_nao_recuperar() -> None:
    """Caso de abstencao: citar qualquer coisa e erro, nao acerto parcial."""
    vazio = retrieval_metrics(answer_with(), [])
    com_citacao = retrieval_metrics(answer_with("mdr-0000-abc"), [])

    assert vazio.hit is True
    assert com_citacao.hit is False


def test_abstencao_e_acerto_apenas_quando_nao_ha_resposta() -> None:
    abstida = answer_with(route=RetrievalRoute.INSUFFICIENT_CONTEXT)
    respondida = answer_with("mdr-0000-abc")

    assert abstention_correct(abstida, answerable=False)
    assert not abstention_correct(abstida, answerable=True)
    assert abstention_correct(respondida, answerable=True)
    assert not abstention_correct(respondida, answerable=False)


def test_golden_dataset_do_projeto_e_valido() -> None:
    """O dataset e o artefato que sustenta qualquer numero do README."""
    path = Path(__file__).resolve().parent.parent / "evals" / "golden_dataset.jsonl"

    cases = load_golden_dataset(path)

    assert len(cases) >= 30
    assert len({case.id for case in cases}) == len(cases)
    assert any(not case.answerable for case in cases), "faltam casos de abstencao"
    for case in cases:
        if case.answerable:
            assert case.expected_doc_ids, f"{case.id} sem documento esperado"
            assert case.must_include, f"{case.id} sem fato obrigatorio"


def test_linha_invalida_no_dataset_aponta_o_numero(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text('{"id": "ok"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Linha 1"):
        load_golden_dataset(path)


async def test_avaliacao_produz_relatorio_com_configuracao() -> None:
    """Metrica sem prompt, modelo e configuracao de busca nao e comparavel."""
    pipeline = CorrectiveRAGPipeline(
        retriever=FakeRetriever(make_scored(2)),  # type: ignore[arg-type]
        client=FakeLLMClient(generate="A taxa e de 3,15% [1]."),  # type: ignore[arg-type]
    )
    cases = [
        GoldenCase(
            id="mdr-01",
            question="Qual a taxa?",
            reference_answer="3,15%",
            expected_doc_ids=["mdr"],
            must_include=["3,15%"],
            category="taxas",
        )
    ]

    report = await Evaluator(pipeline).run(cases)

    assert report.summary["cases"] == 1
    assert report.summary["pass_rate"] == 1.0
    assert report.summary["retrieval_hit_rate"] == 1.0
    assert report.model == "fake-model"
    assert report.prompt_version
    assert report.by_category["taxas"]["pass_rate"] == 1.0
    assert report.failures() == []


async def test_caso_reprovado_registra_o_fato_ausente() -> None:
    pipeline = CorrectiveRAGPipeline(
        retriever=FakeRetriever(make_scored(2)),  # type: ignore[arg-type]
        client=FakeLLMClient(generate="Nao sei dizer a taxa."),  # type: ignore[arg-type]
    )
    cases = [
        GoldenCase(
            id="mdr-02",
            question="Qual a taxa?",
            reference_answer="3,15%",
            expected_doc_ids=["mdr"],
            must_include=["3,15%"],
        )
    ]

    report = await Evaluator(pipeline).run(cases)

    assert report.summary["pass_rate"] == 0.0
    assert report.failures()[0].missing_facts == ["3,15%"]


async def test_juiz_llm_e_opcional() -> None:
    pipeline = CorrectiveRAGPipeline(
        retriever=FakeRetriever(make_scored(2)),  # type: ignore[arg-type]
        client=FakeLLMClient(generate="A taxa e de 3,15% [1]."),  # type: ignore[arg-type]
    )
    cases = [
        GoldenCase(
            id="mdr-03",
            question="Qual a taxa?",
            reference_answer="3,15%",
            expected_doc_ids=["mdr"],
            must_include=["3,15%"],
        )
    ]

    sem_juiz = await Evaluator(pipeline).run(cases)
    com_juiz = await Evaluator(pipeline, use_judge=True).run(cases)

    assert sem_juiz.summary["faithfulness"] is None
    assert com_juiz.summary["faithfulness"] == 1.0
    assert com_juiz.summary["correctness"] == 1.0


async def test_relatorio_salvo_tem_versao_mais_recente(tmp_path: Path) -> None:
    pipeline = CorrectiveRAGPipeline(
        retriever=FakeRetriever(make_scored(2)),  # type: ignore[arg-type]
        client=FakeLLMClient(),  # type: ignore[arg-type]
    )
    cases = [
        GoldenCase(
            id="mdr-04",
            question="Qual a taxa?",
            reference_answer="3,15%",
            expected_doc_ids=["mdr"],
            must_include=[],
        )
    ]
    report = await Evaluator(pipeline).run(cases)

    path = save_report(report, tmp_path)

    assert path.exists()
    latest = json.loads((tmp_path / "eval-latest.json").read_text(encoding="utf-8"))
    assert latest["config"]["model"] == "fake-model"
    assert latest["summary"]["cases"] == 1
