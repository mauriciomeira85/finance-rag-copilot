"""Tracing das etapas do pipeline.

Uma aplicacao de LLM sem tracing e impossivel de depurar: nao se sabe qual
chunk entrou no prompt, quantos tokens foram gastos nem qual no do grafo
falhou. Aqui cada etapa vira um span com duracao, atributos e erro.

O destino padrao e um arquivo JSONL local, para que o projeto funcione sem
depender de servico externo. Se as chaves do Langfuse estiverem preenchidas,
os spans tambem sao enviados para la.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from finrag.logging_setup import get_logger
from finrag.settings import get_settings

logger = get_logger(__name__)

_current_trace: ContextVar[str | None] = ContextVar("finrag_trace_id", default=None)
_current_span: ContextVar[str | None] = ContextVar("finrag_span_id", default=None)


@dataclass(slots=True)
class Span:
    name: str
    span_id: str
    trace_id: str
    parent_id: str | None
    started_at: float
    attributes: dict[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None
    status: str = "ok"
    error: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "ts": datetime.now(UTC).isoformat(),
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "duration_ms": round(self.duration_ms or 0.0, 2),
            "status": self.status,
            "error": self.error,
            "attributes": _sanitize(self.attributes),
        }


def _sanitize(attributes: dict[str, Any]) -> dict[str, Any]:
    """Trunca textos longos e remove valores nao serializaveis."""
    clean: dict[str, Any] = {}
    for key, value in attributes.items():
        if isinstance(value, str) and len(value) > 2000:
            clean[key] = value[:2000] + "...[truncado]"
        elif isinstance(value, int | float | bool | str | type(None)):
            clean[key] = value
        elif isinstance(value, list | tuple):
            clean[key] = [str(item)[:300] for item in value[:20]]
        elif isinstance(value, dict):
            clean[key] = _sanitize(value)
        else:
            clean[key] = str(value)[:300]
    return clean


class TraceRecorder:
    """Escreve spans em JSONL e, opcionalmente, espelha no Langfuse."""

    def __init__(self, path: Path | None = None) -> None:
        settings = get_settings().observability
        self._path = path or settings.trace_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._langfuse = self._init_langfuse() if settings.langfuse_enabled else None

    @property
    def path(self) -> Path:
        """Destino dos spans. Util para inspecao em teste e em suporte."""
        return self._path

    @staticmethod
    def _init_langfuse() -> Any | None:
        settings = get_settings().observability
        try:
            from langfuse import Langfuse
        except ImportError:
            logger.warning(
                "langfuse_indisponivel",
                hint="instale com: uv sync --extra langfuse",
            )
            return None
        assert settings.langfuse_public_key and settings.langfuse_secret_key
        return Langfuse(
            public_key=settings.langfuse_public_key.get_secret_value(),
            secret_key=settings.langfuse_secret_key.get_secret_value(),
            host=settings.langfuse_host,
        )

    def record(self, span: Span) -> None:
        record = span.to_record()
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self._langfuse is not None:
            self._emit_langfuse(record)

    def _emit_langfuse(self, record: dict[str, Any]) -> None:
        try:
            self._langfuse.create_event(  # type: ignore[union-attr]
                name=record["name"],
                metadata=record,
                trace_id=record["trace_id"],
            )
        except Exception as exc:  # pragma: no cover - falha de telemetria nao quebra o fluxo
            logger.warning("langfuse_emit_falhou", error=str(exc))

    def read_trace(self, trace_id: str) -> list[dict[str, Any]]:
        """Le de volta os spans de um trace. Usado pela interface e pelos testes."""
        if not self._path.exists():
            return []
        spans: list[dict[str, Any]] = []
        with self._path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("trace_id") == trace_id:
                    spans.append(record)
        return spans


_recorder: TraceRecorder | None = None


def get_recorder() -> TraceRecorder:
    global _recorder
    if _recorder is None:
        _recorder = TraceRecorder()
    return _recorder


def reset_recorder() -> None:
    """Forca nova instancia. Usado pelos testes com diretorio temporario."""
    global _recorder
    _recorder = None


def current_trace_id() -> str | None:
    return _current_trace.get()


@contextmanager
def trace(name: str, **attributes: Any) -> Iterator[Span]:
    """Abre um trace novo (raiz). Use uma vez por requisicao."""
    trace_id = attributes.pop("trace_id", None) or uuid.uuid4().hex
    token = _current_trace.set(trace_id)
    try:
        with span(name, **attributes) as root:
            yield root
    finally:
        _current_trace.reset(token)


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Span]:
    """Abre um span filho do span corrente.

    Atributos podem ser acrescentados durante a execucao mutando
    ``span.attributes``, o que e util para registrar contagens que so se
    conhecem no fim da etapa.
    """
    trace_id = _current_trace.get() or uuid.uuid4().hex
    trace_token = _current_trace.set(trace_id)
    current = Span(
        name=name,
        span_id=uuid.uuid4().hex[:16],
        trace_id=trace_id,
        parent_id=_current_span.get(),
        started_at=time.perf_counter(),
        attributes=dict(attributes),
    )
    span_token = _current_span.set(current.span_id)
    try:
        yield current
    except Exception as exc:
        current.status = "error"
        current.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        current.duration_ms = (time.perf_counter() - current.started_at) * 1000
        _current_span.reset(span_token)
        _current_trace.reset(trace_token)
        get_recorder().record(current)
        logger.debug(
            "span",
            name=current.name,
            duration_ms=round(current.duration_ms, 1),
            status=current.status,
        )
