"""Interface do copiloto.

Fala com a API por HTTP em vez de importar o pipeline. Isso mantem uma unica
copia do modelo de embedding e do indice no processo da API, e deixa a
interface substituivel sem tocar no nucleo.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("FINRAG_API_URL", "http://localhost:8000")
TIMEOUT = 180

EXAMPLES = [
    "Qual a taxa de MDR do crédito parcelado em 2 a 6 vezes na Adquirente Beta?",
    "Qual o limite de tolerância para divergência de valor na conciliação de cartões?",
    "Como é calculada a base de cálculo dos royalties e qual a alíquota vigente?",
    "Qual foi a margem EBITDA do terceiro trimestre de 2025 e o que explicou a variação?",
    "Em quantos dias a Adquirente Alfa deve disponibilizar o arquivo de conciliação?",
    "Qual é a política de home office da rede?",
]

ROUTE_LABELS = {
    "direct": ("Resposta direta", "Os trechos recuperados já eram suficientes."),
    "rewritten": ("Consulta reescrita", "A pergunta foi reformulada para achar o documento certo."),
    "regenerated": (
        "Resposta refeita",
        "A auditoria achou afirmação sem respaldo e pediu nova geração.",
    ),
    "insufficient_context": (
        "Abstenção",
        "Nada no corpus sustenta uma resposta; o copiloto preferiu não inventar.",
    ),
}

st.set_page_config(
    page_title="Copiloto financeiro (RAG)",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background-color: #0e1117; }
      .citation {
        border-left: 3px solid #2e6fdb;
        padding: 0.35rem 0 0.35rem 0.85rem;
        margin-bottom: 0.9rem;
        font-size: 0.88rem;
        color: #c9d1d9;
      }
      .citation-title { color: #58a6ff; font-weight: 600; }
      .badge {
        display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px;
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.03em;
      }
      .badge-ok { background: #16351f; color: #56d364; }
      .badge-warn { background: #3a2a12; color: #e3b341; }
      .badge-info { background: #12283a; color: #58a6ff; }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_get(path: str) -> dict[str, Any] | None:
    try:
        response = requests.get(f"{API_URL}{path}", timeout=15)
        response.raise_for_status()
        return dict(response.json())
    except requests.RequestException:
        return None


def ask(question: str, doc_types: list[str] | None, periods: list[str] | None) -> dict[str, Any]:
    """Consome o endpoint de streaming e reporta o estagio corrente."""
    payload = {"question": question, "doc_types": doc_types, "periods": periods}
    status = st.status("Consultando os documentos...", expanded=True)
    answer: dict[str, Any] | None = None
    error: str | None = None

    with requests.post(
        f"{API_URL}/query/stream", json=payload, stream=True, timeout=TIMEOUT
    ) as response:
        response.raise_for_status()
        event = ""
        for raw in response.iter_lines(decode_unicode=True):
            if raw is None or raw == "":
                continue
            if raw.startswith("event:"):
                event = raw.split(":", 1)[1].strip()
            elif raw.startswith("data:"):
                data = json.loads(raw.split(":", 1)[1].strip())
                if event == "stage":
                    status.write(data["label"])
                elif event == "answer":
                    answer = data
                elif event == "error":
                    error = data["detail"]

    if error:
        status.update(label="Falhou", state="error")
        raise RuntimeError(error)
    status.update(label="Concluído", state="complete", expanded=False)
    if answer is None:
        raise RuntimeError("A API não devolveu resposta.")
    return answer


def render_answer(result: dict[str, Any]) -> None:
    route = result["route"]
    label, explanation = ROUTE_LABELS.get(route, (route, ""))
    grounded = result["grounded"]

    badges = [f'<span class="badge badge-info">{label}</span>']
    if grounded is True:
        badges.append('<span class="badge badge-ok">ancorada no documento</span>')
    elif grounded is False:
        badges.append('<span class="badge badge-warn">ancoragem parcial</span>')
    st.markdown(" ".join(badges), unsafe_allow_html=True)
    if explanation:
        st.caption(explanation)

    st.markdown(result["answer"])

    if result["rewrites"]:
        with st.expander(f"A pergunta foi reescrita {len(result['rewrites'])}×"):
            for index, rewrite in enumerate(result["rewrites"], start=1):
                st.write(f"{index}. {rewrite}")

    if result["citations"]:
        st.markdown("#### De onde veio a resposta")
        for index, citation in enumerate(result["citations"], start=1):
            section = f" › {citation['section']}" if citation["section"] else ""
            st.markdown(
                f'<div class="citation">'
                f'<span class="citation-title">[{index}] {citation["document"]}{section}</span>'
                f"<br/>{citation['excerpt']}</div>",
                unsafe_allow_html=True,
            )

    columns = st.columns(4)
    columns[0].metric("Tempo", f"{result['latency_ms'] / 1000:.1f} s")
    columns[1].metric("Tokens", f"{result['total_tokens']:,}".replace(",", "."))
    columns[2].metric("Custo", f"US$ {result['cost_usd']:.5f}")
    columns[3].metric("Trechos citados", len(result["citations"]))

    if result.get("trace_id"):
        with st.expander("Trace de execução"):
            trace = api_get(f"/traces/{result['trace_id']}")
            if trace and trace["spans"]:
                frame = pd.DataFrame(trace["spans"])[["name", "duration_ms", "status"]]
                frame.columns = ["etapa", "duração (ms)", "status"]
                st.dataframe(frame, hide_index=True, width="stretch")


with st.sidebar:
    st.title("Copiloto financeiro")
    st.caption("Perguntas e respostas sobre os documentos da Rede Aurora Cosméticos.")

    health = api_get("/health")
    if health is None:
        st.error("O copiloto está temporariamente indisponível. Tente de novo em alguns segundos.")
        st.stop()
    if health["indexed_chunks"] == 0:
        st.warning("Os documentos ainda estão sendo indexados. Atualize a página em instantes.")
        st.stop()

    st.markdown("**Documentos na base**")
    st.markdown(
        """
- Política de conciliação de cartões
- Tabela de MDR (Alfa e Beta)
- Manual de fluxo de caixa
- DRE gerencial 3T2025
- Política de royalties e marketing
- Fechamento mensal
- Conciliação bancária set/2025
- Glossário financeiro
- Política de inadimplência
- Contrato ADQ-4471 (Adquirente Alfa)
        """
    )

    st.markdown("**Stack**")
    st.caption(
        "LangGraph · Qdrant (denso + BM25) · FastAPI · Streamlit · "
        f"{health['model']} · {health['embedding_model'].split('/')[-1]}"
    )

    st.markdown("**Filtros (opcional)**")
    doc_types = st.multiselect(
        "Tipo de documento",
        [
            "politica",
            "manual",
            "tabela",
            "dre",
            "procedimento",
            "relatorio",
            "glossario",
            "contrato",
        ],
        help="Vazio busca em todo o corpus.",
    )
    periods = st.multiselect("Competência", ["2025-09", "2025-Q3", "2025-06"])

    stats = api_get("/stats")
    if stats and stats["counters"].get("queries_total"):
        st.markdown("**Uso desta instância**")
        st.code(
            f"consultas    {int(stats['counters']['queries_total'])}\n"
            f"p95          {stats['latency_ms']['p95'] / 1000:.1f} s\n"
            f"custo total  US$ {stats['cost_usd_total']:.4f}\n"
            f"por consulta US$ {stats['cost_usd_per_query']:.5f}",
            language=None,
        )

st.markdown("## Copiloto financeiro — Rede Aurora")
st.write(
    "Pergunte sobre conciliação, MDR, royalties, DRE, fluxo de caixa, "
    "inadimplência ou o contrato com a Adquirente Alfa. A resposta cita o "
    "documento. O último exemplo da lista não está na base: o sistema deve "
    "dizer que não encontrou."
)

with st.expander("Como a busca funciona"):
    st.markdown(
        """
A pergunta vai para o Qdrant em duas frentes (embedding e BM25). O LangGraph
decide se os trechos servem, reescreve a consulta se for preciso e só então
pede a resposta ao DeepSeek, com citação. Se nada no texto aguentar a
afirmação, a resposta é recusada.
        """
    )

if "pending" not in st.session_state:
    st.session_state.pending = ""

chosen = st.selectbox(
    "Comece por um exemplo",
    ["—", *EXAMPLES],
    help="O último exemplo não tem resposta no corpus: serve para ver a abstenção.",
)
if chosen != "—":
    st.session_state.pending = chosen

question = st.text_area(
    "Pergunta",
    value=st.session_state.pending,
    height=90,
    placeholder="Ex.: qual o prazo de liquidação do crédito à vista na Adquirente Alfa?",
)

if st.button("Perguntar", type="primary", disabled=not question.strip()):
    try:
        render_answer(ask(question.strip(), doc_types or None, periods or None))
    except Exception as exc:
        st.error(f"Não foi possível responder: {exc}")
