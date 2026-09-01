"""Re-ranking.

A fusao hibrida entrega recall; o re-rank entrega precisao. Sem ele o prompt
recebe cinco chunks em que o primeiro nem sempre e o melhor, e o modelo tende
a se ancorar no inicio do contexto.

Duas implementacoes, com trade-off explicito:

* ``LLMReranker`` — listwise, sem download. Custa uma chamada de LLM e ~800
  tokens de entrada por consulta. Entende portugues bem e nao adiciona peso
  a imagem do container.
* ``CrossEncoderReranker`` — modelo local dedicado. Custo de API zero e
  latencia menor depois do warm-up, ao preco de 1,1 GB de download.

Em ambos, se o re-rank falhar a ordem da fusao e preservada. Componente de
qualidade nunca deve derrubar a resposta.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from finrag.llm.client import LLMClient
from finrag.llm.prompts import RERANK_SYSTEM, RERANK_USER
from finrag.logging_setup import get_logger
from finrag.models import ScoredChunk
from finrag.observability import span
from finrag.settings import RetrievalSettings, get_settings

logger = get_logger(__name__)

_EXCERPT_CHARS = 700


class _RankedItem(BaseModel):
    index: int = Field(ge=1)
    score: float = Field(ge=0.0, le=1.0)


class _RankingResponse(BaseModel):
    ranking: list[_RankedItem]


class Reranker(ABC):
    @abstractmethod
    async def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]: ...


class NoOpReranker(Reranker):
    """Mantem a ordem da fusao. Usado no benchmark de ablacao."""

    async def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        return candidates[:top_k]


class LLMReranker(Reranker):
    def __init__(self, client: LLMClient) -> None:
        self._client = client

    async def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        if len(candidates) <= 1:
            return candidates[:top_k]

        listing = "\n\n".join(
            f"[{position}] {scored.chunk.citation}\n{scored.chunk.text[:_EXCERPT_CHARS]}"
            for position, scored in enumerate(candidates, start=1)
        )
        with span("rerank.llm", candidates=len(candidates)) as current:
            try:
                verdict, _ = await self._client.structured(
                    _RankingResponse,
                    RERANK_SYSTEM,
                    RERANK_USER.format(question=query, documents=listing),
                    step="rerank",
                )
            except Exception as exc:
                logger.warning("rerank_llm_falhou", error=str(exc)[:200])
                current.attributes["fallback"] = True
                return candidates[:top_k]

        ordered: list[ScoredChunk] = []
        seen: set[int] = set()
        for item in verdict.ranking:
            position = item.index - 1
            if position in seen or not 0 <= position < len(candidates):
                continue
            seen.add(position)
            scored = candidates[position].model_copy(deep=True)
            scored.rerank_score = item.score
            ordered.append(scored)

        # Candidatos que o modelo esqueceu entram no fim, na ordem da fusao.
        ordered.extend(
            candidates[position] for position in range(len(candidates)) if position not in seen
        )
        return ordered[:top_k]


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        settings = get_settings().embedding
        with span("rerank.load_model", model=self._model_name):
            self._model = TextCrossEncoder(self._model_name, cache_dir=str(settings.cache_dir))

    async def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        if len(candidates) <= 1:
            return candidates[:top_k]
        with span("rerank.cross_encoder", candidates=len(candidates)) as current:
            try:
                self._load()
                scores = list(
                    self._model.rerank(
                        query, [scored.chunk.text[:_EXCERPT_CHARS] for scored in candidates]
                    )
                )
            except Exception as exc:
                logger.warning("rerank_cross_encoder_falhou", error=str(exc)[:200])
                current.attributes["fallback"] = True
                return candidates[:top_k]

        ranked: list[ScoredChunk] = []
        for scored, raw in zip(candidates, scores, strict=True):
            item = scored.model_copy(deep=True)
            # Cross-encoder devolve logit; a sigmoide traz para [0, 1] e deixa
            # o score comparavel com o do re-rank por LLM.
            item.rerank_score = 1.0 / (1.0 + pow(2.718281828459045, -float(raw)))
            ranked.append(item)
        ranked.sort(key=lambda item: item.rerank_score or 0.0, reverse=True)
        return ranked[:top_k]


def build_reranker(
    settings: RetrievalSettings | None = None,
    client: LLMClient | None = None,
) -> Reranker:
    settings = settings or get_settings().retrieval
    if settings.reranker == "none":
        return NoOpReranker()
    if settings.reranker == "cross_encoder":
        return CrossEncoderReranker(settings.cross_encoder_model)
    if client is None:
        from finrag.llm.client import get_llm_client

        client = get_llm_client()
    return LLMReranker(client)
