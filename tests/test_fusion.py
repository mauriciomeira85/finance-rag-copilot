"""Reciprocal Rank Fusion."""

from __future__ import annotations

import pytest

from finrag.models import ScoredChunk
from finrag.retrieval.fusion import reciprocal_rank_fusion
from tests.helpers import make_chunk


def dense(chunk_id: str, score: float) -> ScoredChunk:
    return ScoredChunk(chunk=make_chunk(chunk_id=chunk_id), dense_score=score)


def sparse(chunk_id: str, score: float) -> ScoredChunk:
    return ScoredChunk(chunk=make_chunk(chunk_id=chunk_id), sparse_score=score)


def test_documento_nos_dois_rankings_sobe() -> None:
    """O ganho do hibrido vem daqui: concordancia entre os dois indices."""
    denso = [dense("a", 0.9), dense("b", 0.8), dense("c", 0.7)]
    lexical = [sparse("c", 12.0), sparse("d", 9.0), sparse("a", 4.0)]

    fused = reciprocal_rank_fusion([denso, lexical], k=60)

    assert [item.chunk.chunk_id for item in fused][:2] == ["a", "c"]


def test_scores_de_origem_sao_preservados() -> None:
    """A trilha de scores e o que permite depurar por que um trecho subiu."""
    fused = reciprocal_rank_fusion([[dense("a", 0.9)], [sparse("a", 12.0)]])

    assert len(fused) == 1
    assert fused[0].dense_score == 0.9
    assert fused[0].sparse_score == 12.0
    assert fused[0].fused_score == pytest.approx(2 / 61)


def test_escala_de_score_nao_influencia_o_resultado() -> None:
    """RRF usa posicao, nao valor: BM25 sem limite superior nao domina."""
    denso = [dense("a", 0.51), dense("b", 0.50)]
    lexical = [sparse("b", 900.0), sparse("a", 1.0)]

    fused = reciprocal_rank_fusion([denso, lexical])

    assert fused[0].fused_score == pytest.approx(fused[1].fused_score)


def test_peso_por_ranking_desempata() -> None:
    denso = [dense("a", 0.9)]
    lexical = [sparse("b", 9.0)]

    fused = reciprocal_rank_fusion([denso, lexical], weights=[2.0, 1.0])

    assert fused[0].chunk.chunk_id == "a"


def test_peso_incompativel_falha_alto() -> None:
    with pytest.raises(ValueError, match="mesmo tamanho"):
        reciprocal_rank_fusion([[dense("a", 0.9)]], weights=[1.0, 1.0])


def test_ranking_vazio_nao_quebra() -> None:
    assert reciprocal_rank_fusion([[], []]) == []


def test_final_score_usa_o_estagio_mais_tardio() -> None:
    scored = ScoredChunk(chunk=make_chunk(), dense_score=0.4, fused_score=0.7, rerank_score=0.95)

    assert scored.final_score == 0.95
