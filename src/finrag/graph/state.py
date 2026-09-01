"""Estado do grafo de Corrective RAG."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from finrag.models import RetrievalRoute, ScoredChunk, TokenUsage


def _merge_usage(left: TokenUsage, right: TokenUsage) -> TokenUsage:
    """Redutor do consumo de tokens.

    Cada no que chama o LLM devolve apenas o proprio consumo; o redutor
    acumula. Sem isso, nos que rodam em paralelo sobrescreveriam a contagem.
    """
    return left.merge(right)


def _append(left: list[str], right: list[str]) -> list[str]:
    return [*left, *right]


class GraphState(TypedDict, total=False):
    """Estado compartilhado entre os nos.

    ``question`` e a consulta corrente, que a reescrita substitui.
    ``original_question`` nunca muda: a resposta final e a avaliacao precisam
    da pergunta que o usuario realmente fez.
    """

    question: str
    original_question: str
    documents: list[ScoredChunk]
    rewrites: Annotated[list[str], _append]
    attempts: int
    route: RetrievalRoute
    answer: str
    grounded: bool | None
    unsupported_claims: list[str]
    regenerated: bool
    usage: Annotated[TokenUsage, _merge_usage]
    filters: dict[str, Any]
