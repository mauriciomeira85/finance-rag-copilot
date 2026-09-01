"""Dubles e fabricas usadas pelos testes.

Nenhum teste toca a rede ou baixa modelo: o cliente de LLM e o modelo de
embedding sao substituidos por dubles deterministicos. E o que permite rodar a
suite inteira no CI em segundos.
"""

from __future__ import annotations

import json
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from finrag.llm.client import LLMResponse
from finrag.models import Chunk, DocumentSource, ScoredChunk, TokenUsage
from finrag.observability import METRICS
from finrag.settings import get_settings

# Payload devolvido quando o teste nao especifica um para o passo.
DEFAULT_STRUCTURED: dict[str, Any] = {
    "RelevanceVerdict": {"relevant": True, "score": 0.9, "reason": "trata do tema"},
    "GroundingVerdict": {"grounded": True, "unsupported_claims": [], "reason": "tudo citado"},
    "_RankingResponse": {"ranking": [{"index": 1, "score": 0.95}, {"index": 2, "score": 0.8}]},
    "FaithfulnessVerdict": {"total_claims": 3, "supported_claims": 3, "unsupported": []},
    "CorrectnessVerdict": {"score": 1.0, "reason": "equivalente"},
}


class FakeLLMClient:
    """Cliente de LLM com respostas roteadas pelo nome do passo.

    Poder programar cada passo (``grade``, ``generate``, ``grounding``,
    ``rerank``) e o que permite exercitar cada ramo do grafo sem depender de
    como o modelo se comporta no dia. Passar uma excecao como valor simula
    falha do provedor naquele passo.
    """

    def __init__(self, **overrides: Any) -> None:
        self.model = "fake-model"
        self.calls: list[str] = []
        self._overrides = overrides
        self.text = overrides.get("text", "Resposta ancorada no contexto [1].")

    def cost_of(self, usage: TokenUsage) -> float:
        return usage.total_tokens / 1_000_000 * 0.3

    def _account(self, usage: TokenUsage) -> None:
        """Espelha a contabilidade do cliente real.

        Sem isso o endpoint de metricas apareceria zerado nos testes e o custo
        acumulado ficaria sem cobertura.
        """
        METRICS.increment("llm_calls_total", usage.calls)
        METRICS.increment("llm_prompt_tokens_total", usage.prompt_tokens)
        METRICS.increment("llm_completion_tokens_total", usage.completion_tokens)
        METRICS.increment("llm_cost_usd", self.cost_of(usage))

    async def complete(
        self,
        messages: Sequence[BaseMessage],
        *,
        step: str = "complete",
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls.append(step)
        value = self._overrides.get(step, self.text)
        if isinstance(value, Exception):
            raise value
        if callable(value):
            value = value(messages)
        usage = TokenUsage(prompt_tokens=100, completion_tokens=20, calls=1)
        self._account(usage)
        return LLMResponse(text=str(value), usage=usage)

    async def structured(
        self,
        schema: type[BaseModel],
        system: str,
        user: str,
        *,
        step: str = "structured",
    ) -> tuple[Any, TokenUsage]:
        self.calls.append(step)
        payload = self._overrides.get(step)
        if payload is None:
            payload = DEFAULT_STRUCTURED[schema.__name__]
        if isinstance(payload, Exception):
            raise payload
        if callable(payload):
            payload = payload(user)
        usage = TokenUsage(prompt_tokens=80, completion_tokens=15, calls=1)
        self._account(usage)
        return schema.model_validate(json.loads(json.dumps(payload))), usage


@dataclass(slots=True)
class SparseEmbedding:
    """Mesma forma do que o fastembed devolve: arrays numpy com ``tolist()``."""

    indices: Any
    values: Any


class FakeEmbedder:
    """Embeddings por bag-of-words, sem ONNX.

    Nao pretendem ser bons: pretendem ser estaveis. O hash e crc32 e nao
    ``hash()`` porque o embutido e aleatorizado por processo, o que deixaria o
    resultado dependente do PYTHONHASHSEED.
    """

    dimension = 32

    def __init__(self) -> None:
        self.settings = get_settings().embedding

    @staticmethod
    def _bucket(token: str, size: int) -> int:
        return zlib.crc32(token.lower().encode()) % size

    def _dense(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in text.split():
            vector[self._bucket(token, self.dimension)] += 1.0
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]

    def _sparse(self, text: str) -> SparseEmbedding:
        counts: dict[int, float] = {}
        for token in text.split():
            index = self._bucket(token, 10_000)
            counts[index] = counts.get(index, 0.0) + 1.0
        return SparseEmbedding(
            indices=numpy.array(list(counts), dtype=numpy.int64),
            values=numpy.array(list(counts.values()), dtype=numpy.float32),
        )

    def embed_documents(self, texts: Sequence[str]) -> tuple[list[list[float]], list[Any]]:
        return [self._dense(text) for text in texts], [self._sparse(text) for text in texts]

    def embed_query(self, text: str) -> tuple[list[float], Any]:
        return self._dense(text), self._sparse(text)


class FakeStore:
    """Substitui o indice quando o teste so precisa da contagem."""

    def __init__(self, total: int = 12) -> None:
        self.total = total
        self.closed = False

    def count(self) -> int:
        return self.total

    def close(self) -> None:
        self.closed = True


class FakeRetriever:
    """Recuperador programavel.

    ``batches`` define o que cada chamada devolve, em ordem: e assim que o
    teste simula "a primeira busca veio pobre, a busca apos a reescrita veio
    boa" e verifica o ciclo corretivo.
    """

    def __init__(self, *batches: Sequence[ScoredChunk], store: Any = None) -> None:
        self._batches = [list(batch) for batch in batches] or [[]]
        self.queries: list[str] = []
        self.filters: list[dict[str, Any]] = []
        self.store = store or FakeStore()

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        doc_types: Sequence[str] | None = None,
        periods: Sequence[str] | None = None,
    ) -> Any:
        from finrag.retrieval.retriever import RetrievalResult

        self.queries.append(query)
        self.filters.append({"doc_types": doc_types, "periods": periods})
        index = min(len(self.queries) - 1, len(self._batches) - 1)
        chunks = self._batches[index]
        return RetrievalResult(
            chunks=chunks,
            dense_hits=len(chunks),
            sparse_hits=len(chunks),
            fused_hits=len(chunks),
        )


def make_chunk(
    chunk_id: str = "mdr-0000-aaa",
    text: str = "A taxa de MDR do credito parcelado na Adquirente Beta e de 3,15%.",
    doc_id: str = "mdr",
    doc_type: str = "tabela",
    period: str | None = "2025-09",
    heading_path: Sequence[str] = ("Adquirente Beta",),
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        ordinal=0,
        heading_path=list(heading_path),
        source=DocumentSource(
            doc_id=doc_id,
            title="Tabela de MDR",
            path=f"data/corpus/{doc_id}.md",
            doc_type=doc_type,
            period=period,
        ),
    )


def make_scored(count: int = 3, **kwargs: Any) -> list[ScoredChunk]:
    return [
        ScoredChunk(
            chunk=make_chunk(chunk_id=f"mdr-{index:04d}-aaa", **kwargs),
            fused_score=1.0 - index * 0.1,
        )
        for index in range(count)
    ]
