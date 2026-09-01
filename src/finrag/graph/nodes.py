"""Nos do grafo de Corrective RAG.

Cada no e uma funcao assincrona que recebe o estado e devolve apenas as
chaves que alterou. Manter os nos pequenos e sem estado proprio e o que
permite testar cada etapa isoladamente com um LLM falso.
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from finrag.graph.state import GraphState
from finrag.llm.client import LLMClient
from finrag.llm.prompts import (
    ANSWER_SYSTEM,
    ANSWER_USER,
    GRADE_SYSTEM,
    GRADE_USER,
    GROUNDING_SYSTEM,
    GROUNDING_USER,
    REWRITE_SYSTEM,
    REWRITE_USER,
    format_context,
)
from finrag.logging_setup import get_logger
from finrag.models import (
    GroundingVerdict,
    RelevanceVerdict,
    RetrievalRoute,
    ScoredChunk,
    TokenUsage,
)
from finrag.observability import span
from finrag.retrieval.retriever import HybridRetriever
from finrag.settings import GraphSettings, get_settings

logger = get_logger(__name__)

ABSTENTION_MESSAGE = (
    "Nao encontrei essa informacao nos documentos disponiveis. "
    "Reformule a pergunta ou verifique se o documento correspondente foi indexado."
)

_MAX_PARALLEL_GRADES = 6


class CorrectiveRAGNodes:
    """Implementacao dos nos, com as dependencias injetadas.

    Injetar retriever e cliente de LLM no construtor e o que torna o grafo
    testavel sem rede: os testes passam dublês.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        client: LLMClient,
        settings: GraphSettings | None = None,
    ) -> None:
        self._retriever = retriever
        self._client = client
        self._settings = settings or get_settings().graph

    # ------------------------------------------------------------------ nos
    async def retrieve(self, state: GraphState) -> dict[str, Any]:
        filters = state.get("filters") or {}
        result = await self._retriever.retrieve(
            state["question"],
            doc_types=filters.get("doc_types"),
            periods=filters.get("periods"),
        )
        return {"documents": result.chunks}

    async def grade_documents(self, state: GraphState) -> dict[str, Any]:
        """Avalia relevancia de cada chunk recuperado.

        O gargalo aqui e latencia, nao CPU: as avaliacoes rodam concorrentes,
        com semaforo para nao estourar o rate limit do provedor.
        """
        documents: list[ScoredChunk] = state.get("documents", [])
        if not documents:
            return {"documents": [], "usage": TokenUsage()}

        semaphore = asyncio.Semaphore(_MAX_PARALLEL_GRADES)
        question = state["question"]

        async def grade(scored: ScoredChunk) -> tuple[ScoredChunk, TokenUsage]:
            async with semaphore:
                try:
                    verdict, usage = await self._client.structured(
                        RelevanceVerdict,
                        GRADE_SYSTEM,
                        GRADE_USER.format(question=question, document=scored.chunk.text[:1500]),
                        step="grade",
                    )
                except Exception as exc:
                    # Falha no avaliador nao pode descartar o documento: em caso
                    # de duvida ele passa, e a geracao decide o que usar.
                    logger.warning("grading_falhou", error=str(exc)[:200])
                    updated = scored.model_copy(deep=True)
                    updated.relevance_score = self._settings.relevance_threshold
                    return updated, TokenUsage()
            updated = scored.model_copy(deep=True)
            updated.relevance_score = verdict.score
            return updated, usage

        with span("graph.grade_documents", documents=len(documents)) as current:
            results = await asyncio.gather(*(grade(scored) for scored in documents))
            total = TokenUsage()
            graded: list[ScoredChunk] = []
            for scored, usage in results:
                total = total.merge(usage)
                if (scored.relevance_score or 0.0) >= self._settings.relevance_threshold:
                    graded.append(scored)
            graded.sort(key=lambda item: item.relevance_score or 0.0, reverse=True)
            current.attributes.update(kept=len(graded), dropped=len(documents) - len(graded))

        return {"documents": graded, "usage": total}

    async def rewrite_query(self, state: GraphState) -> dict[str, Any]:
        previous = state.get("rewrites", [])
        listing = "\n".join(f"- {item}" for item in previous) or "- (nenhuma)"
        response = await self._client.complete(
            [
                SystemMessage(content=REWRITE_SYSTEM),
                HumanMessage(
                    content=REWRITE_USER.format(
                        question=state.get("original_question", state["question"]),
                        previous=listing,
                    )
                ),
            ],
            step="rewrite",
        )
        rewritten = response.text.strip().strip('"')
        logger.info("consulta_reescrita", original=state["question"], rewritten=rewritten)
        return {
            "question": rewritten,
            "rewrites": [rewritten],
            "attempts": state.get("attempts", 0) + 1,
            "route": RetrievalRoute.REWRITTEN,
            "usage": response.usage,
        }

    async def generate(self, state: GraphState) -> dict[str, Any]:
        documents: list[ScoredChunk] = state.get("documents", [])
        context = format_context(documents)
        response = await self._client.complete(
            [
                SystemMessage(content=ANSWER_SYSTEM),
                HumanMessage(
                    content=ANSWER_USER.format(
                        context=context,
                        question=state.get("original_question", state["question"]),
                    )
                ),
            ],
            step="generate",
        )
        return {"answer": response.text, "usage": response.usage}

    async def regenerate(self, state: GraphState) -> dict[str, Any]:
        """Segunda tentativa quando a auditoria acha afirmacao sem respaldo.

        Em vez de simplesmente repetir a chamada, o prompt recebe as
        afirmacoes problematicas, o que direciona a correcao.
        """
        documents: list[ScoredChunk] = state.get("documents", [])
        claims = state.get("unsupported_claims", [])
        listing = "\n".join(f"- {claim}" for claim in claims) or "- (nao especificado)"
        response = await self._client.complete(
            [
                SystemMessage(content=ANSWER_SYSTEM),
                HumanMessage(
                    content=(
                        ANSWER_USER.format(
                            context=format_context(documents),
                            question=state.get("original_question", state["question"]),
                        )
                        + "\n\nA tentativa anterior continha afirmacoes sem respaldo no "
                        f"contexto:\n{listing}\n\nReescreva a resposta removendo o que nao "
                        "estiver no contexto. Se o que sobrar for insuficiente, use a "
                        "formula de abstencao da regra 3."
                    )
                ),
            ],
            step="regenerate",
        )
        return {
            "answer": response.text,
            "regenerated": True,
            "route": RetrievalRoute.REGENERATED,
            "usage": response.usage,
        }

    async def check_grounding(self, state: GraphState) -> dict[str, Any]:
        if not self._settings.enable_grounding_check:
            return {"grounded": None, "usage": TokenUsage()}

        documents: list[ScoredChunk] = state.get("documents", [])
        if not documents:
            return {"grounded": False, "usage": TokenUsage()}

        try:
            verdict, usage = await self._client.structured(
                GroundingVerdict,
                GROUNDING_SYSTEM,
                GROUNDING_USER.format(
                    context=format_context(documents), answer=state.get("answer", "")
                ),
                step="grounding",
            )
        except Exception as exc:
            logger.warning("grounding_falhou", error=str(exc)[:200])
            return {"grounded": None, "usage": TokenUsage()}

        if not verdict.grounded:
            logger.warning("resposta_nao_ancorada", claims=verdict.unsupported_claims[:3])
        return {
            "grounded": verdict.grounded,
            "unsupported_claims": verdict.unsupported_claims,
            "usage": usage,
        }

    async def abstain(self, state: GraphState) -> dict[str, Any]:
        """Abstencao explicita.

        Responder errado custa mais do que nao responder: em conciliacao
        financeira, um numero inventado vira decisao errada de caixa.
        """
        logger.info("abstencao", question=state.get("original_question"))
        return {
            "answer": ABSTENTION_MESSAGE,
            "documents": [],
            "route": RetrievalRoute.INSUFFICIENT_CONTEXT,
            "grounded": None,
        }

    # ------------------------------------------------------ arestas condicionais
    def route_after_grading(self, state: GraphState) -> str:
        documents = state.get("documents", [])
        if len(documents) >= self._settings.min_relevant_docs:
            return "generate"
        if state.get("attempts", 0) < self._settings.max_rewrites:
            return "rewrite_query"
        # Um unico documento relevante ainda vale uma resposta com citacao;
        # zero nao vale.
        return "generate" if documents else "abstain"

    def route_after_grounding(self, state: GraphState) -> str:
        if state.get("grounded") is False and not state.get("regenerated", False):
            return "regenerate"
        return "finish"
