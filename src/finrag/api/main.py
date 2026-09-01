"""Aplicacao FastAPI.

O pipeline e construido uma vez no lifespan e reaproveitado: instanciar o
modelo de embedding por requisicao custaria segundos e memoria a cada chamada.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from finrag import __version__
from finrag.api.schemas import (
    ErrorResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    StatsResponse,
    TraceResponse,
)
from finrag.graph.corrective_rag import CorrectiveRAGPipeline
from finrag.ingestion.pipeline import ingest
from finrag.llm.client import LLMNotConfiguredError, LLMProviderError
from finrag.logging_setup import get_logger, setup_logging
from finrag.observability import METRICS, get_recorder
from finrag.settings import get_settings

logger = get_logger(__name__)

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_logging()
    settings = get_settings()
    try:
        pipeline = CorrectiveRAGPipeline()
        _state["pipeline"] = pipeline
        logger.info("pipeline_pronto", model=settings.llm.model)
        if pipeline.retriever.store.count() == 0:
            # A demo publica sobe o indice sozinha. Sem isso o visitante
            # encontraria a interface no ar e o copiloto sem documento nenhum.
            report = ingest(store=pipeline.retriever.store, recreate=True)
            logger.info("ingestao_inicial_ok", chunks=report.indexed)
    except LLMNotConfiguredError as exc:
        # A API sobe mesmo sem chave para que /health responda e o operador
        # descubra o problema pelo healthcheck, nao por crash em loop.
        _state["pipeline"] = None
        _state["error"] = str(exc)
        logger.error("pipeline_indisponivel", error=str(exc))
    yield
    pipeline = _state.get("pipeline")
    if pipeline is not None:
        pipeline.retriever.store.close()
    _state.clear()


def get_pipeline() -> CorrectiveRAGPipeline:
    pipeline = _state.get("pipeline")
    if not isinstance(pipeline, CorrectiveRAGPipeline):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_state.get("error", "Pipeline indisponivel"),
        )
    return pipeline


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="finance-rag-copilot",
        version=__version__,
        summary="RAG corretivo sobre documentos financeiros de uma rede de franquias",
        description=(
            "Busca hibrida (densa + BM25) com fusao RRF, re-ranking e ciclo de "
            "Corrective RAG: avaliacao de relevancia, reescrita de consulta, "
            "auditoria de ancoragem e abstencao explicita."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.exception_handler(LLMProviderError)
    async def provider_failed(request: Request, exc: Exception) -> JSONResponse:
        """Traduz recusa do provedor em 502.

        A causa esta fora do processo, entao devolver 500 mandaria o operador
        procurar o defeito no lugar errado.
        """
        assert isinstance(exc, LLMProviderError)
        logger.error("provedor_recusou", status_code=exc.status_code, detail=str(exc))
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content=ErrorResponse(detail=str(exc), error_type="llm_provider_error").model_dump(),
        )

    @app.middleware("http")
    async def log_requests(request: Request, call_next: Any) -> Any:
        response = await call_next(request)
        if request.url.path not in ("/health", "/metrics"):
            logger.info(
                "http",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
            )
        return response

    # ------------------------------------------------------------- rotas
    @app.get("/health", response_model=HealthResponse, tags=["operacao"])
    async def health() -> HealthResponse:
        pipeline = _state.get("pipeline")
        indexed = 0
        if pipeline is not None:
            try:
                indexed = pipeline.retriever.store.count()
            except Exception as exc:  # pragma: no cover
                logger.warning("contagem_indisponivel", error=str(exc))
        return HealthResponse(
            status="ok" if pipeline is not None and indexed > 0 else "degraded",
            version=__version__,
            model=settings.llm.model,
            embedding_model=settings.embedding.model,
            reranker=settings.retrieval.reranker,
            indexed_chunks=indexed,
            llm_configured=pipeline is not None,
        )

    @app.post("/query", response_model=QueryResponse, tags=["consulta"])
    async def query(
        payload: QueryRequest,
        pipeline: CorrectiveRAGPipeline = Depends(get_pipeline),
    ) -> QueryResponse:
        answer, trace_id = await pipeline.answer(
            payload.question, doc_types=payload.doc_types, periods=payload.periods
        )
        return QueryResponse.from_answer(answer, trace_id)

    @app.post("/query/stream", tags=["consulta"])
    async def query_stream(
        payload: QueryRequest,
        pipeline: CorrectiveRAGPipeline = Depends(get_pipeline),
    ) -> StreamingResponse:
        """Server-Sent Events com o progresso do grafo e a resposta final.

        O ciclo corretivo pode levar alguns segundos; emitir o estagio atual
        evita a tela parada, que e a principal reclamacao de usuario em
        aplicacao de RAG.
        """

        async def event_stream() -> AsyncIterator[str]:
            def sse(event: str, data: dict[str, Any]) -> str:
                return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

            task = asyncio.create_task(
                pipeline.answer(
                    payload.question, doc_types=payload.doc_types, periods=payload.periods
                )
            )
            stages = [
                ("retrieve", "Buscando nos documentos"),
                ("grade", "Avaliando relevancia dos trechos"),
                ("generate", "Redigindo a resposta"),
                ("grounding", "Auditando a ancoragem"),
            ]
            index = 0
            while not task.done():
                if index < len(stages):
                    stage, label = stages[index]
                    yield sse("stage", {"stage": stage, "label": label})
                    index += 1
                await asyncio.sleep(0.6)

            try:
                answer, trace_id = task.result()
            except Exception as exc:
                logger.error("stream_falhou", error=str(exc))
                yield sse("error", {"detail": str(exc)})
                return
            yield sse("answer", QueryResponse.from_answer(answer, trace_id).model_dump())

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/ingest", response_model=IngestResponse, tags=["operacao"])
    async def run_ingestion(
        payload: IngestRequest,
        pipeline: CorrectiveRAGPipeline = Depends(get_pipeline),
    ) -> IngestResponse:
        report = await asyncio.to_thread(ingest, None, pipeline.retriever.store, payload.recreate)
        return IngestResponse.model_validate(report.as_dict())

    @app.get("/stats", response_model=StatsResponse, tags=["operacao"])
    async def stats() -> StatsResponse:
        return StatsResponse.model_validate(METRICS.snapshot())

    @app.get("/metrics", response_class=PlainTextResponse, tags=["operacao"])
    async def prometheus_metrics() -> str:
        return METRICS.render_prometheus()

    @app.get("/traces/{trace_id}", response_model=TraceResponse, tags=["operacao"])
    async def get_trace(trace_id: str) -> TraceResponse:
        spans = get_recorder().read_trace(trace_id)
        if not spans:
            raise HTTPException(status_code=404, detail="Trace nao encontrado")
        return TraceResponse(trace_id=trace_id, spans=spans)

    return app


app = create_app()
