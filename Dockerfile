# Imagem da API. Dois estagios: o primeiro resolve dependencias, o segundo so
# recebe o venv pronto e o codigo, o que mantem a imagem final sem cache de
# build nem compilador.
FROM ghcr.io/astral-sh/uv:0.5-python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# As dependencias sao instaladas antes do codigo: enquanto o lock nao muda,
# esta camada e reaproveitada e o build fica em segundos.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --extra ui

COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra ui


FROM python:3.12-slim-bookworm AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FINRAG_OBSERVABILITY__LOG_JSON=true \
    FINRAG_EMBEDDING__CACHE_DIR=/app/.fastembed_cache

RUN useradd --create-home --uid 10001 finrag

WORKDIR /app

COPY --from=builder --chown=finrag:finrag /app/.venv /app/.venv
COPY --chown=finrag:finrag src/ ./src/
COPY --chown=finrag:finrag app/ ./app/
COPY --chown=finrag:finrag data/ ./data/
COPY --chown=finrag:finrag evals/ ./evals/
COPY --chown=finrag:finrag .streamlit/ ./.streamlit/

# O modelo ONNX de embedding e baixado no build, nao no primeiro request: a
# alternativa e um cold start de dezenas de segundos e um container que precisa
# de rede para responder.
RUN mkdir -p /app/.fastembed_cache /app/traces /app/reports \
    && chown -R finrag:finrag /app/.fastembed_cache /app/traces /app/reports

USER finrag

RUN python -c "\
from fastembed import SparseTextEmbedding, TextEmbedding; \
TextEmbedding('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', cache_dir='/app/.fastembed_cache'); \
SparseTextEmbedding('Qdrant/bm25', cache_dir='/app/.fastembed_cache')"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "finrag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
