"""Logging estruturado.

Em desenvolvimento sai colorido e legivel; em container sai JSON, que e o
formato que qualquer coletor (CloudWatch, Loki, Datadog) consegue indexar.
"""

from __future__ import annotations

import logging
import sys

import structlog

from finrag.settings import get_settings

_configured = False


def setup_logging(force: bool = False) -> None:
    global _configured
    if _configured and not force:
        return

    settings = get_settings().observability
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level, force=True)
    # Reduz ruido das bibliotecas de rede sem esconder erros. Uma linha por
    # chamada de LLM poluiria o log da avaliacao, que faz centenas delas.
    # httpx2 entra na lista porque o cliente da OpenAI ja migrou para ele.
    for noisy in (
        "httpx",
        "httpx2",
        "httpcore",
        "urllib3",
        "qdrant_client",
        "openai",
        "langsmith",
    ):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))

    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer()
        if settings.log_json
        else structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    setup_logging()
    return structlog.get_logger(name)
