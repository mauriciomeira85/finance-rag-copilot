"""Contratos de entrada e saida da API.

Os schemas da API sao separados dos modelos de dominio de proposito: mudar a
representacao interna nao deve quebrar o contrato publico.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from finrag.models import Answer


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    doc_types: list[str] | None = Field(
        default=None,
        description="Filtra por tipo de documento, ex.: ['politica', 'contrato']",
    )
    periods: list[str] | None = Field(
        default=None, description="Filtra por competencia, ex.: ['2025-09']"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "Qual a taxa de MDR do crédito parcelado na Adquirente Beta?",
                    "doc_types": None,
                    "periods": None,
                }
            ]
        }
    }


class CitationOut(BaseModel):
    chunk_id: str
    document: str
    section: str | None
    excerpt: str
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationOut]
    route: str
    rewrites: list[str]
    grounded: bool | None
    trace_id: str | None
    latency_ms: float
    cost_usd: float
    total_tokens: int

    @classmethod
    def from_answer(cls, answer: Answer, trace_id: str | None) -> QueryResponse:
        return cls(
            question=answer.question,
            answer=answer.answer,
            citations=[CitationOut(**citation.model_dump()) for citation in answer.citations],
            route=answer.route.value,
            rewrites=answer.rewrites,
            grounded=answer.grounded,
            trace_id=trace_id,
            latency_ms=answer.latency_ms,
            cost_usd=answer.cost_usd,
            total_tokens=answer.usage.total_tokens,
        )


class IngestRequest(BaseModel):
    recreate: bool = Field(default=True, description="Recria a colecao antes de indexar o corpus")


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    indexed: int
    per_document: dict[str, int]
    skipped: list[str]


class HealthResponse(BaseModel):
    status: str
    version: str
    model: str
    embedding_model: str
    reranker: str
    indexed_chunks: int
    llm_configured: bool


class StatsResponse(BaseModel):
    counters: dict[str, float]
    routes: dict[str, int]
    latency_ms: dict[str, float]
    cost_usd_total: float
    cost_usd_per_query: float


class TraceResponse(BaseModel):
    trace_id: str
    spans: list[dict[str, Any]]


class ErrorResponse(BaseModel):
    detail: str
    error_type: str
