"""Cliente de LLM: saida estruturada, contabilidade e falhas."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, SystemMessage
from pydantic import BaseModel, Field, SecretStr

from finrag.llm.client import (
    LLMClient,
    LLMNotConfiguredError,
    LLMProviderError,
    StructuredOutputError,
    _extract_json,
    get_llm_client,
    reset_llm_client,
)
from finrag.models import TokenUsage
from finrag.observability import METRICS
from finrag.settings import LLMSettings, reset_settings_cache


class Verdict(BaseModel):
    ok: bool
    score: float = Field(ge=0.0, le=1.0)


class ProviderRefusal(Exception):
    """Imita a forma do erro da SDK da OpenAI: excecao com ``status_code``."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}")


class FakeChat:
    """Substitui o ChatOpenAI devolvendo respostas na ordem programada."""

    def __init__(self, *responses: str, usage: dict[str, int] | None = None) -> None:
        self._responses = list(responses)
        self._usage = usage or {"input_tokens": 120, "output_tokens": 30}
        self.received: list[list[Any]] = []

    async def ainvoke(self, messages: list[Any], **kwargs: Any) -> AIMessage:
        self.received.append(messages)
        content = self._responses[min(len(self.received) - 1, len(self._responses) - 1)]
        return AIMessage(content=content, usage_metadata={**self._usage, "total_tokens": 150})


def build_client(*responses: str, **usage: int) -> tuple[LLMClient, FakeChat]:
    client = LLMClient(LLMSettings(api_key=SecretStr("chave"), model="modelo-de-teste"))
    chat = FakeChat(*responses, usage=usage or None)
    client._chat = chat
    return client, chat


def test_falta_de_chave_falha_na_construcao(monkeypatch: pytest.MonkeyPatch) -> None:
    """Melhor falhar ao subir do que na primeira pergunta do usuario."""
    monkeypatch.delenv("FINRAG_LLM__API_KEY", raising=False)
    monkeypatch.setenv("FINRAG_LLM__API_KEY", "")
    reset_settings_cache()
    reset_llm_client()

    with pytest.raises(LLMNotConfiguredError, match="FINRAG_LLM__API_KEY"):
        LLMClient()


def test_cliente_e_reaproveitado() -> None:
    reset_llm_client()

    assert get_llm_client() is get_llm_client()

    reset_llm_client()


async def test_json_com_cerca_de_codigo_e_aceito() -> None:
    """Modelo devolve bloco cercado com frequencia, apesar da instrucao."""
    client, _ = build_client('```json\n{"ok": true, "score": 0.8}\n```')

    verdict, usage = await client.structured(Verdict, "sistema", "usuario")

    assert verdict.ok is True
    assert usage.calls == 1


async def test_json_com_prosa_em_volta_e_recuperado() -> None:
    client, _ = build_client('Claro! Segue: {"ok": false, "score": 0.1} Espero ter ajudado.')

    verdict, _ = await client.structured(Verdict, "sistema", "usuario")

    assert verdict.ok is False


async def test_json_invalido_gera_retentativa_com_o_erro() -> None:
    """Devolver o erro de validacao ao modelo e mais eficaz que repetir o pedido.

    O primeiro score viola o limite do schema (5 acima de 1), o que exercita
    validacao de dominio e nao apenas sintaxe de JSON.
    """
    client, chat = build_client('{"ok": true, "score": 5}', '{"ok": true, "score": 0.9}')

    verdict, usage = await client.structured(Verdict, "sistema", "usuario")

    assert verdict.score == 0.9
    assert len(chat.received) == 2
    assert "nao passou na validacao" in str(chat.received[1][-1].content)
    assert usage.calls == 2


async def test_duas_falhas_seguidas_levantam_erro() -> None:
    client, chat = build_client("nao sou json", "continuo nao sendo json")

    with pytest.raises(StructuredOutputError, match="Verdict"):
        await client.structured(Verdict, "sistema", "usuario")

    assert len(chat.received) == 2


async def test_consumo_e_custo_vao_para_as_metricas() -> None:
    client, _ = build_client("texto", input_tokens=1_000_000, output_tokens=1_000_000)

    response = await client.complete([SystemMessage(content="oi")])

    counters = METRICS.snapshot()["counters"]
    assert response.usage.prompt_tokens == 1_000_000
    assert counters["llm_calls_total"] == 1.0
    # 1 Mtok de entrada + 1 Mtok de saida aos precos default da DeepSeek.
    assert counters["llm_cost_usd"] == pytest.approx(0.28 + 0.42)


def test_custo_usa_os_precos_configurados() -> None:
    client, _ = build_client("texto")
    usage = TokenUsage(prompt_tokens=500_000, completion_tokens=200_000, calls=1)

    assert client.cost_of(usage) == pytest.approx(0.5 * 0.28 + 0.2 * 0.42)


@pytest.mark.parametrize(
    ("status_code", "fragment"),
    [
        (401, "chave rejeitada"),
        (402, "saldo insuficiente"),
        (429, "limite de taxa"),
        (500, "falha no provedor"),
    ],
)
async def test_erro_http_do_provedor_vira_mensagem_acionavel(
    status_code: int, fragment: str
) -> None:
    """Sem a traducao, saldo esgotado chega como traceback de trinta quadros."""
    client, _ = build_client("texto")

    class RefusingChat(FakeChat):
        async def ainvoke(self, messages: list[Any], **kwargs: Any) -> AIMessage:
            raise ProviderRefusal(status_code)

    client._chat = RefusingChat()

    with pytest.raises(LLMProviderError, match=fragment) as excinfo:
        await client.complete([SystemMessage(content="oi")])

    assert excinfo.value.status_code == status_code
    assert isinstance(excinfo.value.__cause__, ProviderRefusal)


async def test_erro_sem_status_http_sobe_intacto() -> None:
    """Falha de rede nao e recusa do provedor e nao deve ser mascarada."""
    client, _ = build_client("texto")

    class BrokenChat(FakeChat):
        async def ainvoke(self, messages: list[Any], **kwargs: Any) -> AIMessage:
            raise TimeoutError("conexao caiu")

    client._chat = BrokenChat()

    with pytest.raises(TimeoutError):
        await client.complete([SystemMessage(content="oi")])


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"a": 1}', '{"a": 1}'),
        ('```json\n{"a": 1}\n```', '{"a": 1}'),
        ('prefixo {"a": 1} sufixo', '{"a": 1}'),
        ("sem json", "sem json"),
    ],
)
def test_extracao_de_json_cobre_os_formatos_comuns(raw: str, expected: str) -> None:
    assert _extract_json(raw) == expected


async def test_consumo_legado_em_response_metadata_e_lido() -> None:
    """Nem todo provedor compativel preenche ``usage_metadata``."""
    client, _ = build_client("texto")

    class LegacyChat(FakeChat):
        async def ainvoke(self, messages: list[Any], **kwargs: Any) -> AIMessage:
            self.received.append(messages)
            return AIMessage(
                content="texto",
                response_metadata={"token_usage": {"prompt_tokens": 42, "completion_tokens": 7}},
            )

    client._chat = LegacyChat()

    response = await client.complete([SystemMessage(content="oi")])

    assert response.usage.prompt_tokens == 42
    assert response.usage.completion_tokens == 7
