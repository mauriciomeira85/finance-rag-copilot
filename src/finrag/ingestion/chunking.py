"""Chunking sensivel a estrutura.

Chunking de tamanho fixo e a causa mais comum de RAG ruim em documento
financeiro: corta a tabela no meio da linha e o modelo perde a associacao
entre o rotulo e o valor. Aqui o documento e primeiro quebrado em blocos
estruturais (titulo, paragrafo, tabela, lista) e so depois agrupado em chunks,
com dois orcamentos de tamanho: prosa curta e tabela longa.

Cada chunk carrega a trilha de titulos, o que serve para dois fins: citacao
legivel para o usuario e sinal extra de recuperacao, ja que o titulo e
concatenado ao texto indexado.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from finrag.models import BlockKind, Chunk, DocumentSource
from finrag.settings import ChunkingSettings

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_SENTENCE_END = re.compile(r"(?<=[.;:!?])\s+")

# Razao caracteres/token medida em portugues com tokenizadores BPE modernos.
# Fica entre 3.4 e 3.9; 3.6 e um meio conservador e evita depender do tiktoken,
# que nao cobre tokenizadores de todos os provedores.
CHARS_PER_TOKEN = 3.6


def estimate_tokens(text: str) -> int:
    """Estimativa de tokens sem tokenizador especifico de provedor."""
    return max(1, round(len(text) / CHARS_PER_TOKEN))


@dataclass(slots=True)
class Block:
    kind: BlockKind
    text: str
    heading_path: list[str] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)


def parse_blocks(markdown: str, drop_document_title: bool = True) -> list[Block]:
    """Quebra markdown em blocos estruturais preservando a trilha de titulos.

    Args:
        markdown: conteudo do documento, sem front matter.
        drop_document_title: quando o documento tem exatamente um titulo de
            nivel 1, ele e o titulo do documento e nao uma secao. Nesse caso
            fica fora da trilha, porque a citacao ja exibe o titulo e repetir
            o termo no texto indexado enviesa o BM25.
    """
    blocks: list[Block] = []
    heading_stack: list[tuple[int, str]] = []
    buffer: list[str] = []
    buffer_kind = BlockKind.PROSE
    top_level_count = sum(
        1
        for line in markdown.splitlines()
        if _HEADING.match(line.rstrip()) and line.startswith("# ")
    )
    skip_level = 1 if (drop_document_title and top_level_count == 1) else 0

    def flush() -> None:
        nonlocal buffer, buffer_kind
        if not buffer:
            return
        text = "\n".join(buffer).strip()
        if text:
            blocks.append(
                Block(
                    kind=buffer_kind,
                    text=text,
                    heading_path=[title for level, title in heading_stack if level > skip_level],
                )
            )
        buffer = []
        buffer_kind = BlockKind.PROSE

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()

        heading = _HEADING.match(line)
        if heading:
            flush()
            level, title = len(heading.group(1)), heading.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            continue

        if not line.strip():
            flush()
            continue

        line_kind = (
            BlockKind.TABLE
            if _TABLE_ROW.match(line)
            else BlockKind.LIST
            if _LIST_ITEM.match(line)
            else BlockKind.PROSE
        )
        if buffer and line_kind is not buffer_kind:
            flush()
        buffer_kind = line_kind
        buffer.append(line)

    flush()
    return blocks


def _budget(kind: BlockKind, settings: ChunkingSettings) -> int:
    return settings.target_tokens_table if kind is BlockKind.TABLE else settings.target_tokens_prose


def _tail_overlap(text: str, overlap_tokens: int) -> str:
    """Ultimas frases do chunk anterior, para nao cortar o encadeamento."""
    if overlap_tokens <= 0:
        return ""
    limit = int(overlap_tokens * CHARS_PER_TOKEN)
    if len(text) <= limit:
        return text
    tail = text[-limit:]
    boundary = tail.find(". ")
    return tail[boundary + 2 :] if boundary != -1 else tail


def _soft_wrap(line: str, limit: int) -> list[str]:
    """Fatia uma linha longa em limites de frase.

    Paragrafo em markdown e uma linha unica. Sem esta etapa, um paragrafo
    grande sairia inteiro como um chunk acima do orcamento, porque o corte por
    linha nao teria onde quebrar.
    """
    if len(line) <= limit:
        return [line]

    grouped: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(line):
        candidate = f"{current} {sentence}" if current else sentence
        if current and len(candidate) > limit:
            grouped.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        grouped.append(current)

    # Frase unica acima do orcamento (tabela em texto corrido, enumeracao sem
    # pontuacao): corta por caracteres como ultimo recurso.
    wrapped: list[str] = []
    for part in grouped:
        while len(part) > limit:
            wrapped.append(part[:limit])
            part = part[limit:]
        if part:
            wrapped.append(part)
    return wrapped


def _split_oversized(block: Block, budget: int, overlap_tokens: int) -> list[Block]:
    """Fatia um bloco maior que o orcamento, respeitando limites de linha."""
    if block.tokens <= budget:
        return [block]

    limit = int(budget * CHARS_PER_TOKEN)
    lines = block.text.split("\n")
    # Cabecalho de tabela markdown (rotulos + separador) e repetido em cada
    # fatia; sem isso a fatia 2 em diante fica sem os nomes das colunas.
    header = (
        lines[:2]
        if block.kind is BlockKind.TABLE and len(lines) > 2 and set(lines[1]) <= set("|-: ")
        else []
    )
    body = lines[len(header) :]
    if block.kind is not BlockKind.TABLE:
        body = [piece for line in body for piece in _soft_wrap(line, limit)]

    pieces: list[Block] = []
    current: list[str] = list(header)
    current_len = sum(len(line) for line in current)

    for line in body:
        if current_len + len(line) > limit and len(current) > len(header):
            pieces.append(Block(block.kind, "\n".join(current), list(block.heading_path)))
            carry = _tail_overlap("\n".join(current[len(header) :]), overlap_tokens)
            current = [*header, carry] if carry else list(header)
            current_len = sum(len(item) for item in current)
        current.append(line)
        current_len += len(line)

    if len(current) > len(header):
        pieces.append(Block(block.kind, "\n".join(current), list(block.heading_path)))
    return pieces or [block]


def chunk_markdown(
    markdown: str,
    source: DocumentSource,
    settings: ChunkingSettings,
) -> list[Chunk]:
    """Converte um documento markdown em chunks prontos para indexacao."""
    blocks = parse_blocks(markdown)
    chunks: list[Chunk] = []
    ordinal = 0

    pending: list[Block] = []
    pending_tokens = 0

    def emit() -> None:
        nonlocal pending, pending_tokens, ordinal
        if not pending:
            return
        text = "\n\n".join(block.text for block in pending).strip()
        if len(text) >= settings.min_chunk_chars or not chunks:
            kind = (
                BlockKind.TABLE
                if any(block.kind is BlockKind.TABLE for block in pending)
                else pending[0].kind
            )
            heading_path = pending[0].heading_path
            # O titulo entra no texto indexado: em documento financeiro o
            # contexto hierarquico costuma conter o termo que o usuario busca.
            indexed = (" > ".join(heading_path) + "\n" + text) if heading_path else text
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(source.doc_id, ordinal, text),
                    doc_id=source.doc_id,
                    text=indexed,
                    kind=kind,
                    ordinal=ordinal,
                    heading_path=list(heading_path),
                    source=source,
                )
            )
            ordinal += 1
        pending = []
        pending_tokens = 0

    for block in blocks:
        budget = _budget(block.kind, settings)
        for piece in _split_oversized(block, budget, settings.overlap_tokens):
            same_section = not pending or pending[0].heading_path == piece.heading_path
            fits = pending_tokens + piece.tokens <= budget
            # Tabela nunca se mistura com prosa no mesmo chunk.
            mixes_table = bool(pending) and (
                (piece.kind is BlockKind.TABLE) != (pending[0].kind is BlockKind.TABLE)
            )
            if not (same_section and fits) or mixes_table:
                emit()
            pending.append(piece)
            pending_tokens += piece.tokens

    emit()
    return chunks


def _chunk_id(doc_id: str, ordinal: int, text: str) -> str:
    """Id deterministico: reindexar o mesmo conteudo nao duplica registro."""
    digest = hashlib.sha256(f"{doc_id}:{ordinal}:{text}".encode()).hexdigest()[:12]
    return f"{doc_id}-{ordinal:04d}-{digest}"
