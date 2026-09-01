"""Fusao de rankings.

Busca densa e busca lexical produzem scores em escalas incomparaveis: cosseno
entre 0 e 1 de um lado, soma de pesos BM25 sem limite superior do outro.
Normalizar essas escalas e fragil porque depende da distribuicao de cada
consulta. O Reciprocal Rank Fusion resolve o problema usando apenas a posicao
no ranking, nao o valor do score.

    RRF(d) = soma sobre os rankings r de  1 / (k + posicao_r(d))

k amortece o peso das primeiras posicoes; 60 e o valor do artigo original
(Cormack et al., 2009) e continua sendo um default solido.
"""

from __future__ import annotations

from collections.abc import Sequence

from finrag.models import ScoredChunk


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ScoredChunk]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[ScoredChunk]:
    """Funde varios rankings em um, preenchendo ``fused_score``.

    Args:
        rankings: listas ja ordenadas da melhor para a pior.
        k: constante de amortecimento do RRF.
        weights: peso por ranking. Igual para todos quando omitido.

    Returns:
        Lista unica ordenada por score fundido, sem duplicatas, preservando
        os scores de origem de cada estagio.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights precisa ter o mesmo tamanho de rankings")

    accumulated: dict[str, float] = {}
    merged: dict[str, ScoredChunk] = {}

    for ranking, weight in zip(rankings, weights, strict=True):
        for position, scored in enumerate(ranking, start=1):
            chunk_id = scored.chunk.chunk_id
            accumulated[chunk_id] = accumulated.get(chunk_id, 0.0) + weight / (k + position)

            if chunk_id not in merged:
                merged[chunk_id] = scored.model_copy(deep=True)
            else:
                # Mesmo chunk vindo dos dois rankings: guarda os dois scores.
                existing = merged[chunk_id]
                if scored.dense_score is not None:
                    existing.dense_score = scored.dense_score
                if scored.sparse_score is not None:
                    existing.sparse_score = scored.sparse_score

    for chunk_id, score in accumulated.items():
        merged[chunk_id].fused_score = score

    return sorted(merged.values(), key=lambda item: item.fused_score or 0.0, reverse=True)
