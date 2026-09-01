"""Grafo de Corrective RAG.

    retrieve -> grade_documents -+-> generate -> check_grounding -+-> FIM
                                 |                               |
                                 +-> rewrite_query -> retrieve    +-> regenerate -> FIM
                                 |
                                 +-> abstain -> FIM

RAG ingenuo entrega ao modelo o que a busca trouxe e confia. Aqui existem tres
pontos de controle:

1. **Grading** — cada chunk recuperado e avaliado; irrelevantes saem do prompt
   antes de contaminar a resposta.
2. **Reescrita** — se sobrou pouco contexto, a consulta e reformulada para o
   vocabulario dos documentos e a busca roda de novo (ate ``max_rewrites``).
3. **Ancoragem** — a resposta e auditada contra o contexto; havendo afirmacao
   sem respaldo, ha uma regeneracao dirigida.

Se depois de todas as tentativas o contexto continuar insuficiente, o sistema
se abstem em vez de inventar.
"""

from __future__ import annotations

import time
from typing import Any

from langgraph.graph import END, StateGraph

from finrag.graph.nodes import CorrectiveRAGNodes
from finrag.graph.state import GraphState
from finrag.llm.client import LLMClient, get_llm_client
from finrag.llm.prompts import format_context
from finrag.logging_setup import get_logger
from finrag.models import Answer, Citation, RetrievalRoute, ScoredChunk, TokenUsage
from finrag.observability import METRICS, current_trace_id, trace
from finrag.retrieval.retriever import HybridRetriever
from finrag.settings import get_settings

logger = get_logger(__name__)

_EXCERPT_CHARS = 320


def build_graph(nodes: CorrectiveRAGNodes) -> Any:
    """Monta e compila o grafo."""
    graph: StateGraph = StateGraph(GraphState)

    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("grade_documents", nodes.grade_documents)
    graph.add_node("rewrite_query", nodes.rewrite_query)
    graph.add_node("generate", nodes.generate)
    graph.add_node("check_grounding", nodes.check_grounding)
    graph.add_node("regenerate", nodes.regenerate)
    graph.add_node("abstain", nodes.abstain)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        nodes.route_after_grading,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "abstain": "abstain",
        },
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", "check_grounding")
    graph.add_conditional_edges(
        "check_grounding",
        nodes.route_after_grounding,
        {"regenerate": "regenerate", "finish": END},
    )
    graph.add_edge("regenerate", END)
    graph.add_edge("abstain", END)

    return graph.compile()


def _citations(documents: list[ScoredChunk]) -> list[Citation]:
    citations: list[Citation] = []
    for scored in documents:
        chunk = scored.chunk
        excerpt = chunk.text.strip().replace("\n", " ")
        citations.append(
            Citation(
                chunk_id=chunk.chunk_id,
                document=chunk.source.title,
                section=" > ".join(chunk.heading_path) or None,
                excerpt=excerpt[:_EXCERPT_CHARS] + ("..." if len(excerpt) > _EXCERPT_CHARS else ""),
                score=round(scored.relevance_score or scored.final_score, 4),
            )
        )
    return citations


class CorrectiveRAGPipeline:
    """Ponto de entrada unico da aplicacao."""

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        client: LLMClient | None = None,
    ) -> None:
        self._client = client or get_llm_client()
        self._retriever = retriever or HybridRetriever()
        self._nodes = CorrectiveRAGNodes(self._retriever, self._client)
        self._graph = build_graph(self._nodes)

    @property
    def retriever(self) -> HybridRetriever:
        return self._retriever

    @property
    def client(self) -> LLMClient:
        """Exposto para a avaliacao reaproveitar o mesmo cliente e contabilidade."""
        return self._client

    @property
    def graph(self) -> Any:
        return self._graph

    async def answer(
        self,
        question: str,
        *,
        doc_types: list[str] | None = None,
        periods: list[str] | None = None,
    ) -> tuple[Answer, str | None]:
        """Responde uma pergunta. Devolve a resposta e o id do trace."""
        started = time.perf_counter()
        settings = get_settings()
        # O recursion_limit protege contra ciclo infinito de reescrita: cada
        # rodada consome tres passos do grafo, mais margem para a auditoria.
        recursion_limit = 6 + settings.graph.max_rewrites * 3

        with trace("query", question=question) as root:
            trace_id = current_trace_id()
            final: dict[str, Any] = await self._graph.ainvoke(
                {
                    "question": question,
                    "original_question": question,
                    "attempts": 0,
                    "rewrites": [],
                    "usage": TokenUsage(),
                    "regenerated": False,
                    "route": RetrievalRoute.DIRECT,
                    "filters": {"doc_types": doc_types, "periods": periods},
                },
                config={"recursion_limit": recursion_limit},
            )

            usage: TokenUsage = final.get("usage") or TokenUsage()
            cost = self._client.cost_of(usage)
            latency_ms = (time.perf_counter() - started) * 1000
            route: RetrievalRoute = final.get("route", RetrievalRoute.DIRECT)

            documents: list[ScoredChunk] = final.get("documents", [])
            answer = Answer(
                question=question,
                answer=final.get("answer", ""),
                citations=_citations(documents),
                context=format_context(documents),
                route=route,
                rewrites=final.get("rewrites", []),
                grounded=final.get("grounded"),
                usage=usage,
                cost_usd=round(cost, 6),
                latency_ms=round(latency_ms, 1),
            )

            root.attributes.update(
                route=route.value,
                grounded=answer.grounded,
                citations=len(answer.citations),
                total_tokens=usage.total_tokens,
                cost_usd=answer.cost_usd,
                latency_ms=answer.latency_ms,
            )

        METRICS.increment("queries_total")
        METRICS.observe("query_latency_ms", latency_ms)
        METRICS.count_route(route.value)
        if answer.grounded is False:
            METRICS.increment("ungrounded_answers_total")
        if route is RetrievalRoute.INSUFFICIENT_CONTEXT:
            METRICS.increment("abstentions_total")

        logger.info(
            "resposta_gerada",
            route=route.value,
            grounded=answer.grounded,
            citations=len(answer.citations),
            tokens=usage.total_tokens,
            cost_usd=answer.cost_usd,
            latency_ms=answer.latency_ms,
        )
        return answer, trace_id
