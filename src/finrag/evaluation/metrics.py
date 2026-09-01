"""Metricas de qualidade.

As metricas estao divididas em dois grupos, e a divisao e proposital:

* **Deterministicas** — acerto de recuperacao, precisao e cobertura de
  contexto, presencia de fatos obrigatorios e acerto de abstencao. Nao chamam
  LLM, custam zero, dao o mesmo resultado sempre e por isso rodam no CI como
  porta de merge.
* **Com juiz LLM** — fidelidade ao contexto e correcao da resposta. Capturam o
  que a comparacao textual nao alcanca, mas custam tokens e tem variancia, e
  por isso rodam sob demanda.

Confiar apenas no juiz LLM e um erro comum: a nota flutua entre execucoes e o
gate do CI passa a ser ruido.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from pydantic import BaseModel, Field

from finrag.models import Answer, RetrievalRoute

_PUNCTUATION = re.compile(r"[^\w\s%,./-]+", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Normaliza para comparacao: minusculas, sem acento, espacos colapsados.

    Numero em documento financeiro brasileiro aparece como "3,15%" e as vezes
    como "3.15%"; a normalizacao unifica os dois para que a verificacao de fato
    obrigatorio nao falhe por formatacao.
    """
    lowered = text.lower()
    decomposed = unicodedata.normalize("NFKD", lowered)
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
    cleaned = _PUNCTUATION.sub(" ", without_accents)
    return _WHITESPACE.sub(" ", cleaned).strip()


def contains_fact(answer: str, fact: str) -> bool:
    """Verifica presenca de um fato, tolerando separador decimal e milhar."""
    haystack = normalize(answer)
    needle = normalize(fact)
    if needle in haystack:
        return True
    # "1.256.750.000" e "1256750000" devem casar; idem "3,15" e "3.15".
    digits_only = re.sub(r"[^\d]", "", needle)
    if len(digits_only) >= 3 and digits_only in re.sub(r"[^\d]", "", haystack):
        return True
    swapped = needle.replace(",", ".")
    return swapped in haystack.replace(",", ".")


class RetrievalMetrics(BaseModel):
    hit: bool = Field(description="Ao menos um documento esperado foi recuperado")
    context_precision: float = Field(ge=0.0, le=1.0)
    context_recall: float = Field(ge=0.0, le=1.0)
    retrieved_docs: list[str] = Field(default_factory=list)


def retrieval_metrics(answer: Answer, expected_doc_ids: Sequence[str]) -> RetrievalMetrics:
    """Precisao e cobertura de contexto em nivel de documento.

    A granularidade e o documento, nao o chunk: a resposta correta pode estar
    em qualquer chunk do documento certo, e exigir o chunk exato tornaria a
    metrica sensivel a mudanca de chunking em vez de a qualidade da busca.
    """
    retrieved = [_doc_id_of(citation.chunk_id) for citation in answer.citations]
    expected = set(expected_doc_ids)

    if not expected:
        # Pergunta sem documento esperado (caso de abstencao): recuperar nada e
        # o comportamento correto.
        return RetrievalMetrics(
            hit=not retrieved,
            context_precision=1.0 if not retrieved else 0.0,
            context_recall=1.0,
            retrieved_docs=retrieved,
        )

    if not retrieved:
        return RetrievalMetrics(hit=False, context_precision=0.0, context_recall=0.0)

    relevant = [doc for doc in retrieved if doc in expected]
    return RetrievalMetrics(
        hit=bool(relevant),
        context_precision=len(relevant) / len(retrieved),
        context_recall=len(expected & set(retrieved)) / len(expected),
        retrieved_docs=retrieved,
    )


def _doc_id_of(chunk_id: str) -> str:
    """chunk_id tem o formato ``<doc_id>-<ordinal>-<hash>``."""
    parts = chunk_id.rsplit("-", 2)
    return parts[0] if len(parts) == 3 else chunk_id


def fact_coverage(answer_text: str, must_include: Sequence[str]) -> tuple[float, list[str]]:
    """Fracao dos fatos obrigatorios presentes, e quais faltaram."""
    if not must_include:
        return 1.0, []
    missing = [fact for fact in must_include if not contains_fact(answer_text, fact)]
    return (len(must_include) - len(missing)) / len(must_include), missing


def abstention_correct(answer: Answer, answerable: bool) -> bool:
    """Abstencao e acerto quando a pergunta nao e respondivel, e erro quando e.

    Em conciliacao financeira, abstencao indevida custa produtividade, mas
    resposta inventada custa decisao errada de caixa. As duas contam como erro,
    e o relatorio separa os dois tipos.
    """
    abstained = answer.route is RetrievalRoute.INSUFFICIENT_CONTEXT or not answer.citations
    return abstained if not answerable else not abstained


# ---------------------------------------------------------------------------
# Metricas com juiz LLM
# ---------------------------------------------------------------------------
JUDGE_FAITHFULNESS_SYSTEM = """\
Voce audita fidelidade de resposta ao contexto que a gerou.

Quebre a resposta em afirmacoes factuais atomicas. Para cada uma, decida se e
sustentada pelo contexto. Devolva o total de afirmacoes e quantas sao
sustentadas.

Nao julgue se a resposta esta certa no mundo real: julgue apenas se ela se
sustenta no contexto fornecido.
"""

JUDGE_CORRECTNESS_SYSTEM = """\
Voce compara uma resposta gerada com uma resposta de referencia escrita por
especialista.

Atribua score de 0 a 1:
- 1.0: equivalente em conteudo, com todos os numeros e condicoes corretos
- 0.7 a 0.9: correta no essencial, com omissao de detalhe secundario
- 0.4 a 0.6: parcialmente correta, ou com numero certo e interpretacao errada
- 0.1 a 0.3: majoritariamente errada
- 0.0: contradiz a referencia ou nao responde

Diferenca de estilo ou de ordem nao penaliza. Diferenca de numero penaliza
sempre.
"""


class FaithfulnessVerdict(BaseModel):
    total_claims: int = Field(ge=0)
    supported_claims: int = Field(ge=0)
    unsupported: list[str] = Field(default_factory=list)

    @property
    def score(self) -> float:
        if self.total_claims == 0:
            return 1.0
        return min(self.supported_claims / self.total_claims, 1.0)


class CorrectnessVerdict(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(max_length=500)
