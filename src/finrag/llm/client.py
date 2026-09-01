"""Cliente de LLM.

Fala com qualquer endpoint compativel com a API da OpenAI; o padrao aponta
para a DeepSeek. Duas decisoes valem explicacao:

1. Saida estruturada e feita com ``response_format=json_object`` mais
   validacao Pydantic e uma retentativa que devolve o erro de validacao ao
   modelo. E mais portavel do que ``with_structured_output`` baseado em tool
   calling, que varia de provedor para provedor.
2. Todo consumo de tokens e contabilizado e convertido em USD na hora. Custo
   descoberto no fim do mes e custo que ninguem controla.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Sequence
from typing import Any, TypeVar

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from finrag.logging_setup import get_logger
from finrag.models import TokenUsage
from finrag.observability import METRICS, span
from finrag.settings import LLMSettings, get_settings

logger = get_logger(__name__)

TModel = TypeVar("TModel", bound=BaseModel)

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class LLMNotConfiguredError(RuntimeError):
    """Levantado quando falta a chave da API."""


class LLMProviderError(RuntimeError):
    """Falha vinda do provedor, com a causa traduzida.

    Existe para que saldo esgotado, chave revogada ou limite de taxa cheguem ao
    operador como uma frase acionavel, e nao como um traceback de trinta
    quadros no meio de uma avaliacao.
    """

    def __init__(self, status_code: int | None, message: str) -> None:
        self.status_code = status_code
        super().__init__(message)


class StructuredOutputError(RuntimeError):
    """O modelo nao devolveu JSON valido para o schema pedido."""


class LLMResponse(BaseModel):
    text: str
    usage: TokenUsage


def _extract_json(raw: str) -> str:
    """Isola o objeto JSON de uma resposta que pode vir com cerca ou prosa."""
    fenced = _JSON_BLOCK.search(raw)
    if fenced:
        return fenced.group(1).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        return raw[start : end + 1]
    return raw.strip()


_PROVIDER_HINTS = {
    401: "chave rejeitada pelo provedor; confira FINRAG_LLM__API_KEY",
    402: "saldo insuficiente na conta do provedor",
    403: "chave sem permissao para este modelo",
    404: "modelo inexistente no endpoint configurado",
    429: "limite de taxa atingido; reduza a concorrencia da avaliacao",
}


def _as_provider_error(exc: Exception) -> LLMProviderError | None:
    """Traduz erro HTTP do provedor. Devolve ``None`` se nao for esse caso."""
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        return None
    hint = _PROVIDER_HINTS.get(status, "falha no provedor de LLM")
    return LLMProviderError(status, f"{hint} (HTTP {status})")


def _usage_from_message(message: AIMessage) -> TokenUsage:
    usage = getattr(message, "usage_metadata", None) or {}
    if usage:
        return TokenUsage(
            prompt_tokens=int(usage.get("input_tokens", 0)),
            completion_tokens=int(usage.get("output_tokens", 0)),
            calls=1,
        )
    legacy = (message.response_metadata or {}).get("token_usage", {})
    return TokenUsage(
        prompt_tokens=int(legacy.get("prompt_tokens", 0)),
        completion_tokens=int(legacy.get("completion_tokens", 0)),
        calls=1,
    )


class LLMClient:
    """Fachada sobre o ChatOpenAI com contabilidade de custo e tracing."""

    def __init__(self, settings: LLMSettings | None = None) -> None:
        self._settings = settings or get_settings().llm
        if not self._settings.is_configured:
            raise LLMNotConfiguredError(
                "FINRAG_LLM__API_KEY nao definida. Copie .env.example para .env "
                "e preencha a chave da DeepSeek."
            )
        self._chat = ChatOpenAI(
            model=self._settings.model,
            base_url=self._settings.base_url,
            api_key=self._settings.api_key,
            temperature=self._settings.temperature,
            max_tokens=self._settings.max_tokens,
            timeout=self._settings.timeout_seconds,
            max_retries=0,  # a retentativa e nossa, com tracing
        )

    @property
    def model(self) -> str:
        return self._settings.model

    def cost_of(self, usage: TokenUsage) -> float:
        return usage.cost_usd(
            self._settings.cost_per_mtok_input,
            self._settings.cost_per_mtok_output,
        )

    def _account(self, usage: TokenUsage) -> None:
        cost = self.cost_of(usage)
        METRICS.increment("llm_calls_total", usage.calls)
        METRICS.increment("llm_prompt_tokens_total", usage.prompt_tokens)
        METRICS.increment("llm_completion_tokens_total", usage.completion_tokens)
        METRICS.increment("llm_cost_usd", cost)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        reraise=True,
    )
    async def complete(
        self,
        messages: Sequence[BaseMessage],
        *,
        step: str = "complete",
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        with span(f"llm.{step}", model=self.model) as current:
            kwargs: dict[str, Any] = {}
            if response_format is not None:
                kwargs["response_format"] = response_format
            try:
                message = await self._chat.ainvoke(list(messages), **kwargs)
            except Exception as exc:
                provider_error = _as_provider_error(exc)
                if provider_error is None:
                    raise
                raise provider_error from exc
            assert isinstance(message, AIMessage)
            usage = _usage_from_message(message)
            self._account(usage)
            current.attributes.update(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cost_usd=round(self.cost_of(usage), 6),
            )
            text = message.content if isinstance(message.content, str) else str(message.content)
            return LLMResponse(text=text.strip(), usage=usage)

    async def structured(
        self,
        schema: type[TModel],
        system: str,
        user: str,
        *,
        step: str = "structured",
    ) -> tuple[TModel, TokenUsage]:
        """Pede JSON aderente ao schema e valida. Uma retentativa com o erro."""
        contract = json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)
        base_system = (
            f"{system}\n\n"
            "Responda EXCLUSIVAMENTE com um objeto JSON valido que satisfaca o "
            f"schema abaixo. Sem texto antes ou depois, sem cercas de codigo.\n\n"
            f"Schema:\n{contract}"
        )
        messages: list[BaseMessage] = [
            SystemMessage(content=base_system),
            HumanMessage(content=user),
        ]
        total = TokenUsage()
        last_error: Exception | None = None

        for attempt in range(2):
            response = await self.complete(
                messages,
                step=step if attempt == 0 else f"{step}.retry",
                response_format={"type": "json_object"},
            )
            total = total.merge(response.usage)
            try:
                payload = json.loads(_extract_json(response.text))
                return schema.model_validate(payload), total
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                logger.warning(
                    "saida_estruturada_invalida",
                    step=step,
                    attempt=attempt + 1,
                    error=str(exc)[:300],
                )
                messages.extend(
                    [
                        AIMessage(content=response.text),
                        HumanMessage(
                            content=(
                                "A resposta anterior nao passou na validacao. "
                                f"Erro: {exc}. Corrija e devolva apenas o JSON valido."
                            )
                        ),
                    ]
                )

        raise StructuredOutputError(
            f"Modelo nao produziu JSON valido para {schema.__name__}: {last_error}"
        )

    async def stream(
        self,
        messages: Sequence[BaseMessage],
        *,
        step: str = "stream",
    ) -> AsyncIterator[str]:
        """Streaming de tokens. O usuario ve a resposta nascer."""
        with span(f"llm.{step}", model=self.model) as current:
            chunks = 0
            async for piece in self._chat.astream(list(messages)):
                content = piece.content
                if isinstance(content, str) and content:
                    chunks += 1
                    yield content
            current.attributes["chunks"] = chunks


_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def reset_llm_client() -> None:
    global _client
    _client = None
