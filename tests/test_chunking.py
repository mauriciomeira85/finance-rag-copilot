"""Chunking sensivel a estrutura."""

from __future__ import annotations

from finrag.ingestion.chunking import chunk_markdown, estimate_tokens, parse_blocks
from finrag.models import BlockKind, DocumentSource
from finrag.settings import ChunkingSettings

SOURCE = DocumentSource(
    doc_id="pol-teste",
    title="Politica de Teste",
    path="data/corpus/teste.md",
    doc_type="politica",
    period="2025-09",
)

TABLE_DOC = """# Politica de Teste

## Taxas por bandeira

| Bandeira | Debito | Credito |
| --- | --- | --- |
| Visa | 1,20% | 2,45% |
| Elo | 1,45% | 2,80% |
| Amex | 1,90% | 3,30% |

## Regras gerais

O prazo de liquidacao do debito e de um dia util.
"""


def test_tabela_nao_e_quebrada_no_meio() -> None:
    """Tabela cortada perde o cabecalho e vira numero sem rotulo."""
    blocks = parse_blocks(TABLE_DOC)
    tables = [block for block in blocks if block.kind is BlockKind.TABLE]

    assert len(tables) == 1
    assert "Visa" in tables[0].text
    assert "Amex" in tables[0].text
    assert tables[0].text.count("|") >= 12


def test_titulo_do_documento_fica_fora_da_trilha() -> None:
    """Repetir o titulo em cada chunk infla o BM25 e poluiria a citacao."""
    chunks = chunk_markdown(TABLE_DOC, SOURCE, ChunkingSettings())

    assert chunks
    for chunk in chunks:
        assert "Politica de Teste" not in chunk.heading_path
    assert any(chunk.heading_path == ["Taxas por bandeira"] for chunk in chunks)


def test_h1_repetido_permanece_como_secao() -> None:
    """Com mais de um H1 eles sao secoes de verdade, nao titulo do documento."""
    body = "Paragrafo com tamanho suficiente para passar do minimo de caracteres exigido. " * 3
    document = f"# Parte A\n\n{body}\n\n# Parte B\n\n{body}"

    chunks = chunk_markdown(document, SOURCE, ChunkingSettings())

    trails = {tuple(chunk.heading_path) for chunk in chunks}
    assert ("Parte A",) in trails
    assert ("Parte B",) in trails


def test_chunk_carrega_proveniencia_e_ordinal() -> None:
    chunks = chunk_markdown(TABLE_DOC, SOURCE, ChunkingSettings())

    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.doc_id == "pol-teste" for chunk in chunks)
    assert all(chunk.chunk_id.startswith("pol-teste-") for chunk in chunks)
    assert all(chunk.source.period == "2025-09" for chunk in chunks)


def test_id_do_chunk_e_estavel_entre_execucoes() -> None:
    """Reindexar o mesmo documento nao pode duplicar registro no Qdrant."""
    primeira = chunk_markdown(TABLE_DOC, SOURCE, ChunkingSettings())
    segunda = chunk_markdown(TABLE_DOC, SOURCE, ChunkingSettings())

    assert [chunk.chunk_id for chunk in primeira] == [chunk.chunk_id for chunk in segunda]


def test_texto_indexado_inclui_a_trilha_de_titulos() -> None:
    """Sem a trilha, um chunk de tabela nao tem a palavra que o usuario digita."""
    chunks = chunk_markdown(TABLE_DOC, SOURCE, ChunkingSettings())
    table_chunk = next(chunk for chunk in chunks if chunk.kind is BlockKind.TABLE)

    assert "Taxas por bandeira" in table_chunk.text


def test_paragrafo_longo_em_linha_unica_e_dividido() -> None:
    """Markdown escreve paragrafo em uma linha; cortar por linha nao basta."""
    paragraph = "Esta e uma frase de politica financeira com termos suficientes. "
    document = "# Doc\n\n## Secao\n\n" + paragraph * 120
    settings = ChunkingSettings(target_tokens_prose=120, overlap_tokens=20)

    chunks = chunk_markdown(document, SOURCE, settings)

    assert len(chunks) > 1
    assert all(estimate_tokens(chunk.text) <= settings.target_tokens_prose * 2 for chunk in chunks)


def test_tabela_grande_repete_o_cabecalho_em_cada_fatia() -> None:
    rows = "\n".join(f"| Bandeira {index} | 1,20% | 2,45% |" for index in range(60))
    document = f"# Doc\n\n## Taxas\n\n| Bandeira | Debito | Credito |\n| --- | --- | --- |\n{rows}"
    settings = ChunkingSettings(target_tokens_table=120)

    chunks = chunk_markdown(document, SOURCE, settings)

    assert len(chunks) > 1
    assert all("| Bandeira | Debito | Credito |" in chunk.text for chunk in chunks)


def test_documento_minusculo_ainda_gera_um_chunk() -> None:
    """O minimo de caracteres filtra sobra de secao, nao pode zerar o documento."""
    chunks = chunk_markdown("# Doc\n\n## S\n\nok\n", SOURCE, ChunkingSettings(min_chunk_chars=120))

    assert len(chunks) == 1


def test_sobra_de_secao_curta_e_descartada() -> None:
    document = TABLE_DOC + "\n\n## Nota\n\nok\n"

    chunks = chunk_markdown(document, SOURCE, ChunkingSettings())

    assert all("Nota" not in chunk.heading_path for chunk in chunks)
