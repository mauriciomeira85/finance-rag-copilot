"""Execucao da avaliacao sobre o golden dataset.

O relatorio final e o artefato que sustenta qualquer afirmacao de qualidade do
projeto. Ele registra tambem a versao do prompt, o modelo e a configuracao de
recuperacao, porque metrica sem essas tres informacoes nao e comparavel entre
execucoes.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from finrag.evaluation.metrics import (
    JUDGE_CORRECTNESS_SYSTEM,
    JUDGE_FAITHFULNESS_SYSTEM,
    CorrectnessVerdict,
    FaithfulnessVerdict,
    abstention_correct,
    fact_coverage,
    retrieval_metrics,
)
from finrag.graph.corrective_rag import CorrectiveRAGPipeline
from finrag.llm.prompts import PROMPT_VERSION, format_context
from finrag.logging_setup import get_logger
from finrag.models import Answer
from finrag.settings import get_settings

logger = get_logger(__name__)


class GoldenCase(BaseModel):
    id: str
    question: str
    reference_answer: str
    expected_doc_ids: list[str] = Field(default_factory=list)
    must_include: list[str] = Field(default_factory=list)
    answerable: bool = True
    category: str = "geral"
    difficulty: str = "media"


class CaseResult(BaseModel):
    case_id: str
    question: str
    category: str
    difficulty: str
    answer: str
    route: str
    latency_ms: float
    cost_usd: float
    total_tokens: int

    hit: bool
    context_precision: float
    context_recall: float
    fact_coverage: float
    missing_facts: list[str] = Field(default_factory=list)
    abstention_ok: bool

    faithfulness: float | None = None
    correctness: float | None = None
    judge_notes: str | None = None

    @property
    def deterministic_pass(self) -> bool:
        """CritÃ©rio de aprovacao usado no gate do CI."""
        return self.hit and self.fact_coverage >= 1.0 and self.abstention_ok


class EvaluationReport(BaseModel):
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    prompt_version: str = PROMPT_VERSION
    model: str
    reranker: str
    embedding_model: str
    top_k_final: int
    judge_enabled: bool

    cases: list[CaseResult] = Field(default_factory=list)

    # ------------------------------------------------------------- agregados
    def _mean(self, attribute: str) -> float:
        values = [value for case in self.cases if (value := getattr(case, attribute)) is not None]
        return round(sum(values) / len(values), 4) if values else 0.0

    @property
    def summary(self) -> dict[str, Any]:
        answerable = [case for case in self.cases if case.abstention_ok is not None]
        return {
            "cases": len(self.cases),
            "pass_rate": round(
                sum(case.deterministic_pass for case in self.cases) / max(len(self.cases), 1), 4
            ),
            "retrieval_hit_rate": self._mean("hit"),
            "context_precision": self._mean("context_precision"),
            "context_recall": self._mean("context_recall"),
            "fact_coverage": self._mean("fact_coverage"),
            "abstention_accuracy": round(
                sum(case.abstention_ok for case in answerable) / max(len(answerable), 1), 4
            ),
            "faithfulness": self._mean("faithfulness") if self.judge_enabled else None,
            "correctness": self._mean("correctness") if self.judge_enabled else None,
            "avg_latency_ms": round(self._mean("latency_ms"), 1),
            "p95_latency_ms": self._percentile("latency_ms", 0.95),
            "total_cost_usd": round(sum(case.cost_usd for case in self.cases), 6),
            "avg_cost_usd": round(self._mean("cost_usd"), 6),
            "avg_tokens": round(self._mean("total_tokens")),
        }

    def _percentile(self, attribute: str, q: float) -> float:
        values = sorted(getattr(case, attribute) for case in self.cases)
        if not values:
            return 0.0
        index = min(int(q * (len(values) - 1) + 0.5), len(values) - 1)
        return round(values[index], 1)

    @property
    def by_category(self) -> dict[str, dict[str, float]]:
        grouped: dict[str, list[CaseResult]] = {}
        for case in self.cases:
            grouped.setdefault(case.category, []).append(case)
        return {
            category: {
                "cases": len(items),
                "pass_rate": round(sum(item.deterministic_pass for item in items) / len(items), 4),
                "context_recall": round(sum(item.context_recall for item in items) / len(items), 4),
            }
            for category, items in sorted(grouped.items())
        }

    def failures(self) -> list[CaseResult]:
        return [case for case in self.cases if not case.deterministic_pass]

    def to_json(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at.isoformat(),
            "config": {
                "prompt_version": self.prompt_version,
                "model": self.model,
                "reranker": self.reranker,
                "embedding_model": self.embedding_model,
                "top_k_final": self.top_k_final,
                "judge_enabled": self.judge_enabled,
            },
            "summary": self.summary,
            "by_category": self.by_category,
            "cases": [case.model_dump() for case in self.cases],
        }


def load_golden_dataset(path: Path | None = None) -> list[GoldenCase]:
    path = path or get_settings().golden_dataset
    if not path.exists():
        raise FileNotFoundError(f"Golden dataset nao encontrado: {path}")
    cases: list[GoldenCase] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                cases.append(GoldenCase.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"Linha {number} invalida no golden dataset: {exc}") from exc
    return cases


class Evaluator:
    def __init__(
        self,
        pipeline: CorrectiveRAGPipeline | None = None,
        use_judge: bool = False,
        concurrency: int = 3,
    ) -> None:
        self._pipeline = pipeline or CorrectiveRAGPipeline()
        self._use_judge = use_judge
        self._semaphore = asyncio.Semaphore(concurrency)

    async def _judge_faithfulness(self, answer: Answer) -> FaithfulnessVerdict | None:
        if not answer.citations:
            return None
        # Julga contra o contexto integral, o mesmo que o gerador recebeu. Usar
        # o trecho curto da citacao faria afirmacao legitima parecer sem
        # respaldo e subestimaria a fidelidade.
        context = answer.context
        try:
            verdict, _ = await self._pipeline.client.structured(
                FaithfulnessVerdict,
                JUDGE_FAITHFULNESS_SYSTEM,
                f"CONTEXTO:\n{context}\n\nRESPOSTA:\n{answer.answer}",
                step="judge_faithfulness",
            )
            return verdict
        except Exception as exc:
            logger.warning("juiz_fidelidade_falhou", error=str(exc)[:200])
            return None

    async def _judge_correctness(self, answer: Answer, reference: str) -> CorrectnessVerdict | None:
        try:
            verdict, _ = await self._pipeline.client.structured(
                CorrectnessVerdict,
                JUDGE_CORRECTNESS_SYSTEM,
                (
                    f"PERGUNTA:\n{answer.question}\n\n"
                    f"RESPOSTA DE REFERENCIA:\n{reference}\n\n"
                    f"RESPOSTA GERADA:\n{answer.answer}"
                ),
                step="judge_correctness",
            )
            return verdict
        except Exception as exc:
            logger.warning("juiz_correcao_falhou", error=str(exc)[:200])
            return None

    async def _run_case(self, case: GoldenCase) -> CaseResult:
        async with self._semaphore:
            answer, _ = await self._pipeline.answer(case.question)

        retrieval = retrieval_metrics(answer, case.expected_doc_ids)
        coverage, missing = fact_coverage(answer.answer, case.must_include)

        result = CaseResult(
            case_id=case.id,
            question=case.question,
            category=case.category,
            difficulty=case.difficulty,
            answer=answer.answer,
            route=answer.route.value,
            latency_ms=answer.latency_ms,
            cost_usd=answer.cost_usd,
            total_tokens=answer.usage.total_tokens,
            hit=retrieval.hit,
            context_precision=round(retrieval.context_precision, 4),
            context_recall=round(retrieval.context_recall, 4),
            fact_coverage=round(coverage, 4),
            missing_facts=missing,
            abstention_ok=abstention_correct(answer, case.answerable),
        )

        if self._use_judge:
            faithfulness = await self._judge_faithfulness(answer)
            if faithfulness is not None:
                result.faithfulness = round(faithfulness.score, 4)
            correctness = await self._judge_correctness(answer, case.reference_answer)
            if correctness is not None:
                result.correctness = round(correctness.score, 4)
                result.judge_notes = correctness.reason

        logger.info(
            "caso_avaliado",
            case_id=case.id,
            passou=result.deterministic_pass,
            recall=result.context_recall,
            fatos=result.fact_coverage,
        )
        return result

    async def run(self, cases: list[GoldenCase] | None = None) -> EvaluationReport:
        settings = get_settings()
        cases = cases or load_golden_dataset()
        results = await asyncio.gather(*(self._run_case(case) for case in cases))
        report = EvaluationReport(
            # Vem do cliente, e nao da configuracao, para o relatorio registrar
            # o modelo que realmente respondeu.
            model=self._pipeline.client.model,
            reranker=settings.retrieval.reranker,
            embedding_model=settings.embedding.model,
            top_k_final=settings.retrieval.top_k_final,
            judge_enabled=self._use_judge,
            cases=list(results),
        )
        return report


def save_report(report: EvaluationReport, directory: Path | None = None) -> Path:
    directory = directory or get_settings().reports_dir
    directory.mkdir(parents=True, exist_ok=True)
    stamp = report.created_at.strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"eval-{stamp}.json"
    path.write_text(
        json.dumps(report.to_json(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    latest = directory / "eval-latest.json"
    latest.write_text(
        json.dumps(report.to_json(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


__all__ = [
    "CaseResult",
    "EvaluationReport",
    "Evaluator",
    "GoldenCase",
    "format_context",
    "load_golden_dataset",
    "save_report",
]
