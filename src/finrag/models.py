"""Modelos de dominio.

Sao os contratos que atravessam ingestao, recuperacao, grafo e API. Manter
tudo em Pydantic permite validar nas bordas e serializar sem trabalho extra.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class BlockKind(StrEnum):
    """Natureza do bloco de texto, usada para escolher a estrategia de chunking."""

    PROSE = "prose"
    TABLE = "table"
    HEADING = "heading"
    LIST = "list"


class DocumentSource(BaseModel):
    """Metadados de proveniencia de um documento do corpus."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    title: str
    path: str
    doc_type: str = Field(description="Ex.: politica, manual, dre, contrato, glossario")
    period: str | None = Field(default=None, description="Competencia, ex.: 2025-Q3")
    version: str | None = None


class Chunk(BaseModel):
    """Unidade indexada no banco vetorial."""

    chunk_id: str
    doc_id: str
    text: str
    kind: BlockKind = BlockKind.PROSE
    ordinal: int = Field(ge=0, description="Posicao do chunk dentro do documento")
    heading_path: list[str] = Field(
        default_factory=list,
        description="Trilha de titulos ate o chunk, ex.: ['Conciliacao', 'Cartoes']",
    )
    source: DocumentSource

    @computed_field  # type: ignore[prop-decorator]
    @property
    def citation(self) -> str:
        """Referencia curta e legivel para exibir junto da resposta."""
        trail = " > ".join(self.heading_path) if self.heading_path else ""
        return f"{self.source.title}" + (f" — {trail}" if trail else "")

    def to_payload(self) -> dict[str, Any]:
        """Serializa para o payload do Qdrant."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "kind": self.kind.value,
            "ordinal": self.ordinal,
            "heading_path": self.heading_path,
            "source": self.source.model_dump(),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Self:
        return cls.model_validate(payload)


class ScoredChunk(BaseModel):
    """Chunk recuperado, com a trilha de scores de cada estagio da busca."""

    chunk: Chunk
    dense_score: float | None = None
    sparse_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None
    relevance_score: float | None = Field(
        default=None, description="Nota do avaliador de relevancia do Corrective RAG"
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def final_score(self) -> float:
        """Score efetivo usado na ordenacao final, do estagio mais tardio disponivel."""
        for candidate in (self.rerank_score, self.fused_score, self.dense_score, self.sparse_score):
            if candidate is not None:
                return candidate
        return 0.0


class RetrievalRoute(StrEnum):
    """Caminho que o grafo percorreu. Vai para a resposta como dado de auditoria."""

    DIRECT = "direct"
    REWRITTEN = "rewritten"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    REGENERATED = "regenerated"


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def merge(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            calls=self.calls + other.calls,
        )

    def cost_usd(self, per_mtok_input: float, per_mtok_output: float) -> float:
        return (
            self.prompt_tokens / 1_000_000 * per_mtok_input
            + self.completion_tokens / 1_000_000 * per_mtok_output
        )


class Citation(BaseModel):
    """Citacao exibida ao usuario, ligada ao chunk que a originou."""

    chunk_id: str
    document: str
    section: str | None = None
    excerpt: str
    score: float


class Answer(BaseModel):
    """Resposta final, com tudo que a trilha de auditoria precisa."""

    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    route: RetrievalRoute = RetrievalRoute.DIRECT
    rewrites: list[str] = Field(default_factory=list)
    grounded: bool | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    context: str = Field(
        default="",
        exclude=True,
        repr=False,
        description=(
            "Contexto integral que gerou a resposta. Fica fora da serializacao "
            "porque nao interessa ao cliente da API, mas a auditoria de "
            "fidelidade precisa julgar contra o mesmo texto que o modelo viu; "
            "julgar contra o trecho curto da citacao subestima a fidelidade."
        ),
    )

    @model_validator(mode="after")
    def _abstain_without_citations(self) -> Self:
        """Sem contexto suficiente nao existe resposta com citacao."""
        if self.route is RetrievalRoute.INSUFFICIENT_CONTEXT:
            self.citations = []
        return self


class RelevanceVerdict(BaseModel):
    """Saida estruturada do no avaliador de documentos."""

    relevant: bool
    score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(max_length=400)


class GroundingVerdict(BaseModel):
    """Saida estruturada do no que verifica se a resposta se sustenta no contexto."""

    grounded: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    reason: str = Field(max_length=400)
