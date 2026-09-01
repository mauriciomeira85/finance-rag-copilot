"""Golden dataset, metricas e relatorio de avaliacao."""

from finrag.evaluation.metrics import (
    RetrievalMetrics,
    abstention_correct,
    contains_fact,
    fact_coverage,
    normalize,
    retrieval_metrics,
)
from finrag.evaluation.runner import (
    CaseResult,
    EvaluationReport,
    Evaluator,
    GoldenCase,
    load_golden_dataset,
    save_report,
)

__all__ = [
    "CaseResult",
    "EvaluationReport",
    "Evaluator",
    "GoldenCase",
    "RetrievalMetrics",
    "abstention_correct",
    "contains_fact",
    "fact_coverage",
    "load_golden_dataset",
    "normalize",
    "retrieval_metrics",
    "save_report",
]
