"""Recuperador hibrido.

Orquestra o pipeline de recuperacao em tres estagios:

    consulta -> busca densa (k=12) --\\
                                      >-- RRF --> re-rank --> top 5
                consulta -> BM25 (k=12) --/

Recuperar 24 candidatos para entregar 5 e deliberado: a fusao precisa de
material para reordenar, e o re-rank e o unico estagio caro o suficiente para
justificar um funil estreito.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from finrag.logging_setup import get_logger
from finrag.models import ScoredChunk
from finrag.observability import span
from finrag.retrieval.fusion import reciprocal_rank_fusion
from finrag.retrieval.reranker import Reranker, build_reranker
from finrag.retrieval.vectorstore import HybridVectorStore
from finrag.settings import RetrievalSettings, get_settings

logger = get_logger(__name__)


@dataclass(slots=True)
class RetrievalResult:
    chunks: list[ScoredChunk]
    dense_hits: int
    sparse_hits: int
    fused_hits: int


class HybridRetriever:
    def __init__(
        self,
        store: HybridVectorStore | None = None,
        reranker: Reranker | None = None,
        settings: RetrievalSettings | None = None,
    ) -> None:
        self._store = store or HybridVectorStore()
        self._settings = settings or get_settings().retrieval
        self._reranker = reranker if reranker is not None else build_reranker(self._settings)

    @property
    def store(self) -> HybridVectorStore:
        return self._store

    @staticmethod
    async def _search(
        search: Callable[..., list[ScoredChunk]],
        query: str,
        limit: int,
        doc_types: Sequence[str] | None,
        periods: Sequence[str] | None,
    ) -> list[ScoredChunk]:
        """Executa uma das buscas fora do event loop, ou nenhuma se desligada."""
        if limit == 0:
            return []
        return await asyncio.to_thread(search, query, limit, doc_types, periods)

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        doc_types: Sequence[str] | None = None,
        periods: Sequence[str] | None = None,
    ) -> RetrievalResult:
        top_k = top_k or self._settings.top_k_final

        with span("retrieval.hybrid", query=query, top_k=top_k) as current:
            # Os dois indices vivem no mesmo Qdrant e o cliente e sincrono;
            # to_thread evita bloquear o event loop da API.
            dense, sparse = await asyncio.gather(
                self._search(
                    self._store.search_dense, query, self._settings.top_k_dense, doc_types, periods
                ),
                self._search(
                    self._store.search_sparse,
                    query,
                    self._settings.top_k_sparse,
                    doc_types,
                    periods,
                ),
            )

            fused = reciprocal_rank_fusion([dense, sparse], k=self._settings.rrf_k)
            # O re-rank recebe o dobro do alvo: candidatos suficientes para
            # reordenar sem pagar tokens por uma lista longa demais.
            candidates = fused[: max(top_k * 2, top_k + 3)]
            reranked = await self._reranker.rerank(query, candidates, top_k)

            current.attributes.update(
                dense_hits=len(dense),
                sparse_hits=len(sparse),
                fused_hits=len(fused),
                returned=len(reranked),
                reranker=self._settings.reranker,
            )

        return RetrievalResult(
            chunks=reranked,
            dense_hits=len(dense),
            sparse_hits=len(sparse),
            fused_hits=len(fused),
        )
