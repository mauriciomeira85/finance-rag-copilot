"""Interface do copiloto financeiro da Rede Aurora."""

from __future__ import annotations

import html
import json
import os
from typing import Any

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("FINRAG_API_URL", "http://localhost:8000")
TIMEOUT = 180

EXAMPLES = [
    (
        "MDR parcelado — Beta",
        "Qual a taxa de MDR do crédito parcelado em 2 a 6 vezes na Adquirente Beta?",
    ),
    (
        "Tolerância da conciliação",
        "Qual o limite de tolerância para divergência de valor na conciliação de cartões?",
    ),
    (
        "Base de royalties",
        "Como é calculada a base de cálculo dos royalties e qual a alíquota vigente?",
    ),
    (
        "EBITDA do 3T2025",
        "Qual foi a margem EBITDA do terceiro trimestre de 2025 e o que explicou a variação?",
    ),
    (
        "Arquivo de conciliação",
        "Em quantos dias a Adquirente Alfa deve disponibilizar o arquivo de conciliação?",
    ),
    (
        "Home office (fora da base)",
        "Qual é a política de home office da rede?",
    ),
]

DOCUMENTS = [
    "Política de conciliação de cartões",
    "Tabela de MDR (Alfa e Beta)",
    "Manual de fluxo de caixa",
    "DRE gerencial 3T2025",
    "Royalties e fundo de marketing",
    "Fechamento mensal",
    "Conciliação bancária set/2025",
    "Glossário financeiro",
    "Inadimplência de franqueados",
    "Contrato ADQ-4471",
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
        "Nada na base sustenta uma resposta; o sistema não inventou o número.",
    ),
}

st.set_page_config(
    page_title="Copiloto financeiro · Rede Aurora",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=Source+Sans+3:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
      :root {
        --ink: #1c1917;
        --muted: #57534e;
        --paper: #f3efe6;
        --card: #fffcf7;
        --navy: #1f3d4d;
        --navy-2: #2a5366;
        --gold: #a0784a;
        --line: #e4ddd0;
        --ok: #276749;
        --warn: #9a3412;
      }
      html, body, [data-testid="stAppViewContainer"], .stApp {
        background: var(--paper) !important;
        color: var(--ink);
        font-family: "Source Sans 3", "Segoe UI", sans-serif;
      }
      [data-testid="stHeader"], [data-testid="stToolbar"],
      #MainMenu, footer, .stDeployButton { display: none !important; }
      [data-testid="stSidebarNav"],
      [data-testid="stSidebarNavSearch"] { display: none !important; }
      [data-testid="stSidebar"] {
        background: #ebe6da !important;
        border-right: 1px solid var(--line);
      }
      [data-testid="stSidebar"] * { font-family: "Source Sans 3", sans-serif; }
      .block-container {
        padding-top: 1.6rem !important;
        padding-bottom: 3rem !important;
        max-width: 920px;
      }
      h1, h2, h3, .hero-title {
        font-family: Fraunces, Georgia, serif !important;
        font-weight: 550 !important;
        letter-spacing: -0.02em;
        color: var(--ink) !important;
      }
      .hero {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1.35rem 1.5rem 1.2rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 10px 30px rgba(28, 25, 23, 0.04);
      }
      .kicker {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--gold);
        margin-bottom: 0.35rem;
      }
      .hero-title { font-size: 2rem; line-height: 1.15; margin: 0 0 0.4rem 0; }
      .hero p { color: var(--muted); margin: 0; font-size: 1.02rem; line-height: 1.5; }
      .doc-chip {
        display: inline-block;
        background: #fff;
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 0.18rem 0.6rem;
        margin: 0.15rem 0.15rem 0 0;
        font-size: 0.75rem;
        color: var(--navy);
      }
      .badge {
        display: inline-block; padding: 0.18rem 0.7rem; border-radius: 999px;
        font-size: 0.75rem; font-weight: 650;
      }
      .badge-ok { background: #dcfce7; color: var(--ok); }
      .badge-warn { background: #ffedd5; color: var(--warn); }
      .badge-info { background: #e7eef2; color: var(--navy); }
      .answer-card, .cite-card {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 1.05rem 1.2rem;
        margin: 0.55rem 0;
      }
      .cite-card { border-left: 3px solid var(--gold); }
      .cite-title { color: var(--navy); font-weight: 650; font-size: 0.92rem; }
      .cite-body {
        color: var(--muted); font-size: 0.88rem; margin-top: 0.35rem; line-height: 1.45;
      }
      .status-dot {
        display: inline-block; width: 8px; height: 8px; border-radius: 50%;
        background: #16a34a; margin-right: 0.4rem;
      }
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #44403c; }
      .stButton > button {
        border-radius: 10px !important;
        font-weight: 650 !important;
        font-family: "Source Sans 3", sans-serif !important;
      }
      .stButton > button[kind="primary"] {
        background: var(--navy) !important;
        color: #fff !important;
        border: 0 !important;
      }
      .stButton > button[kind="secondary"],
      .stButton > button[kind="tertiary"] {
        background: var(--card) !important;
        color: var(--navy) !important;
        border: 1px solid var(--line) !important;
      }
      textarea {
        background: var(--card) !important;
        border-radius: 12px !important;
      }
      [data-testid="stMetricValue"] { font-family: Fraunces, Georgia, serif; }
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
    payload = {"question": question, "doc_types": doc_types, "periods": periods}
    status = st.status("Consultando a base documental...", expanded=True)
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
        status.update(label="Não foi possível concluir", state="error")
        raise RuntimeError(error)
    status.update(label="Consulta concluída", state="complete", expanded=False)
    if answer is None:
        raise RuntimeError("A API não devolveu resposta.")
    return answer


def render_answer(result: dict[str, Any]) -> None:
    route = result["route"]
    label, explanation = ROUTE_LABELS.get(route, (route, ""))
    grounded = result["grounded"]

    badges = [f'<span class="badge badge-info">{html.escape(label)}</span>']
    if grounded is True:
        badges.append('<span class="badge badge-ok">ancorada no documento</span>')
    elif grounded is False:
        badges.append('<span class="badge badge-warn">ancoragem parcial</span>')

    st.markdown(
        f'<div class="answer-card">{" ".join(badges)}'
        f'<p style="color:#57534e;margin:0.45rem 0 0.8rem;font-size:0.9rem">'
        f"{html.escape(explanation)}</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown(result["answer"])

    if result["rewrites"]:
        with st.expander(f"A pergunta foi reescrita {len(result['rewrites'])}×"):
            for index, rewrite in enumerate(result["rewrites"], start=1):
                st.write(f"{index}. {rewrite}")

    if result["citations"]:
        st.markdown("#### Fontes")
        for index, citation in enumerate(result["citations"], start=1):
            section = f" · {citation['section']}" if citation["section"] else ""
            title = html.escape(f"[{index}] {citation['document']}{section}")
            excerpt = html.escape(citation["excerpt"])
            st.markdown(
                f'<div class="cite-card"><div class="cite-title">{title}</div>'
                f'<div class="cite-body">{excerpt}</div></div>',
                unsafe_allow_html=True,
            )

    columns = st.columns(4)
    columns[0].metric("Tempo", f"{result['latency_ms'] / 1000:.1f} s")
    columns[1].metric("Tokens", f"{result['total_tokens']:,}".replace(",", "."))
    columns[2].metric("Custo", f"US$ {result['cost_usd']:.5f}")
    columns[3].metric("Fontes", len(result["citations"]))

    if result.get("trace_id"):
        with st.expander("Trace de execução"):
            trace = api_get(f"/traces/{result['trace_id']}")
            if trace and trace["spans"]:
                frame = pd.DataFrame(trace["spans"])[["name", "duration_ms", "status"]]
                frame.columns = ["etapa", "duração (ms)", "status"]
                st.dataframe(frame, hide_index=True, width="stretch")


health = api_get("/health")
if health is None:
    st.error("O copiloto está temporariamente indisponível. Tente de novo em alguns segundos.")
    st.stop()
if health["indexed_chunks"] == 0:
    st.warning("Os documentos ainda estão sendo indexados. Atualize a página em instantes.")
    st.stop()

with st.sidebar:
    st.markdown(
        '<p class="kicker">Rede Aurora</p><h2 style="margin:0 0 0.4rem;font-size:1.35rem">'
        "Copiloto financeiro</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p><span class="status-dot"></span>{health["indexed_chunks"]} trechos indexados · '
        f"{html.escape(health['model'])}</p>",
        unsafe_allow_html=True,
    )
    st.markdown("**Documentos na base**")
    chips = "".join(f'<span class="doc-chip">{html.escape(name)}</span>' for name in DOCUMENTS)
    st.markdown(chips, unsafe_allow_html=True)

    with st.expander("Filtrar a busca"):
        doc_types = st.multiselect(
            "Tipo",
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
        )
        periods = st.multiselect("Competência", ["2025-09", "2025-Q3", "2025-06"])

    stats = api_get("/stats")
    if stats and stats["counters"].get("queries_total"):
        st.caption(
            f"{int(stats['counters']['queries_total'])} consultas nesta instância · "
            f"p95 {stats['latency_ms']['p95'] / 1000:.1f}s"
        )

st.markdown(
    """
    <div class="hero">
      <div class="kicker">Controladoria · Rede Aurora Cosméticos</div>
      <h1 class="hero-title">Pergunte à documentação financeira</h1>
      <p>A resposta vem com o trecho do manual, da política ou do contrato.
      Se a informação não estiver na base, o sistema avisa — não inventa taxa nem prazo.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "question" not in st.session_state:
    st.session_state.question = ""

st.caption("Exemplos")
chip_cols = st.columns(3)
for index, (label, prompt) in enumerate(EXAMPLES):
    with chip_cols[index % 3]:
        if st.button(label, key=f"ex-{index}", use_container_width=True):
            st.session_state.question = prompt

question = st.text_area(
    "Pergunta",
    key="question",
    height=100,
    placeholder="Ex.: qual o prazo de liquidação do crédito à vista na Adquirente Alfa?",
    label_visibility="collapsed",
)

ask_col, hint_col = st.columns([1, 3])
with ask_col:
    submitted = st.button("Perguntar", type="primary", use_container_width=True)
with hint_col:
    st.caption("O exemplo “home office” não está na base — serve para ver a recusa.")

if submitted and question.strip():
    try:
        render_answer(ask(question.strip(), doc_types or None, periods or None))
    except Exception as exc:
        st.error(f"Não foi possível responder: {exc}")
elif submitted:
    st.warning("Escreva uma pergunta ou escolha um exemplo.")
