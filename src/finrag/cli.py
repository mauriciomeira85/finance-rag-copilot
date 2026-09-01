"""Interface de linha de comando.

Concentra as operacoes do ciclo de vida do projeto: indexar o corpus,
perguntar, rodar a avaliacao, comparar configuracoes e subir a API.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from finrag import __version__
from finrag.llm import LLMNotConfiguredError, LLMProviderError
from finrag.logging_setup import setup_logging
from finrag.settings import get_settings, reset_settings_cache

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="finance-rag-copilot: RAG corretivo sobre documentos financeiros.",
)
console = Console()

def run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    """Executa a corrotina traduzindo falha de configuracao e de provedor.

    Sem isso, chave ausente ou saldo esgotado no meio de uma avaliacao sobem
    como traceback longo, e a causa real fica enterrada no fim.
    """
    try:
        return asyncio.run(coroutine)
    except LLMNotConfiguredError as exc:
        console.print(f"[red]Configuracao incompleta:[/red] {exc}")
        raise typer.Exit(code=2) from None
    except LLMProviderError as exc:
        console.print(f"[red]O provedor de LLM recusou a chamada:[/red] {exc}")
        raise typer.Exit(code=2) from None


@app.callback()
def main() -> None:
    setup_logging()


@app.command()
def version() -> None:
    """Mostra a versao e a configuracao efetiva."""
    settings = get_settings()
    table = Table(show_header=False, box=None)
    table.add_row("versao", __version__)
    table.add_row("modelo de LLM", settings.llm.model)
    table.add_row("endpoint", settings.llm.base_url)
    table.add_row("embeddings", settings.embedding.model)
    table.add_row("banco vetorial", settings.vectorstore.url)
    table.add_row("re-ranking", settings.retrieval.reranker)
    table.add_row("chave configurada", "sim" if settings.llm.is_configured else "NAO")
    console.print(Panel(table, title="finance-rag-copilot", border_style="cyan"))


@app.command("corpus")
def corpus_stats() -> None:
    """Estatisticas do corpus e do chunking, sem tocar no banco vetorial."""
    from finrag.ingestion import build_chunks
    from finrag.ingestion.chunking import estimate_tokens

    chunks, report = build_chunks()
    tokens = [estimate_tokens(chunk.text) for chunk in chunks]

    table = Table(title="Corpus", header_style="bold cyan")
    table.add_column("documento")
    table.add_column("chunks", justify="right")
    table.add_column("tokens", justify="right")
    for doc_id, count in report.per_document.items():
        doc_tokens = sum(estimate_tokens(chunk.text) for chunk in chunks if chunk.doc_id == doc_id)
        table.add_row(doc_id, str(count), f"{doc_tokens:,}".replace(",", "."))
    console.print(table)

    tables = sum(1 for chunk in chunks if chunk.kind.value == "table")
    console.print(
        f"\n[bold]{report.documents}[/bold] documentos, "
        f"[bold]{report.chunks}[/bold] chunks "
        f"([bold]{tables}[/bold] tabulares), "
        f"tokens por chunk: min {min(tokens)} / mediana "
        f"{sorted(tokens)[len(tokens) // 2]} / max {max(tokens)}"
    )


@app.command()
def ingest(
    recreate: Annotated[bool, typer.Option(help="Recria a colecao antes de indexar")] = True,
    corpus_dir: Annotated[Path | None, typer.Option(help="Diretorio do corpus")] = None,
) -> None:
    """Indexa o corpus no banco vetorial hibrido."""
    from finrag.ingestion import ingest as run_ingest
    from finrag.retrieval import HybridVectorStore

    with HybridVectorStore() as store:
        report = run_ingest(corpus_dir, store, recreate)
        total = store.count()

    console.print(
        f"[green]Indexacao concluida.[/green] {report.documents} documentos, "
        f"{report.chunks} chunks, {report.indexed} pontos gravados. "
        f"Total na colecao: {total}."
    )
    if report.skipped:
        console.print(f"[yellow]Ignorados:[/yellow] {', '.join(report.skipped)}")


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Pergunta em linguagem natural")],
    doc_type: Annotated[list[str] | None, typer.Option(help="Filtra por tipo")] = None,
    period: Annotated[list[str] | None, typer.Option(help="Filtra por competencia")] = None,
    show_trace: Annotated[bool, typer.Option(help="Exibe os spans do trace")] = False,
) -> None:
    """Pergunta ao copiloto."""
    from finrag.graph import CorrectiveRAGPipeline
    from finrag.observability import get_recorder

    async def consult() -> None:
        pipeline = CorrectiveRAGPipeline()
        try:
            answer, trace_id = await pipeline.answer(question, doc_types=doc_type, periods=period)
        finally:
            pipeline.retriever.store.close()

        console.print(Panel(answer.answer, title="Resposta", border_style="green"))

        if answer.citations:
            table = Table(title="Fontes", header_style="bold cyan", show_lines=False)
            table.add_column("#", justify="right", width=3)
            table.add_column("documento")
            table.add_column("secao")
            table.add_column("score", justify="right", width=6)
            for index, citation in enumerate(answer.citations, start=1):
                table.add_row(
                    str(index), citation.document, citation.section or "—", f"{citation.score:.2f}"
                )
            console.print(table)

        meta = (
            f"rota: [bold]{answer.route.value}[/bold]  "
            f"ancorada: [bold]{answer.grounded}[/bold]  "
            f"tokens: [bold]{answer.usage.total_tokens}[/bold]  "
            f"custo: [bold]US$ {answer.cost_usd:.6f}[/bold]  "
            f"latencia: [bold]{answer.latency_ms:.0f} ms[/bold]"
        )
        console.print(meta)
        if answer.rewrites:
            console.print(f"[yellow]reescritas:[/yellow] {' | '.join(answer.rewrites)}")

        if show_trace and trace_id:
            spans = get_recorder().read_trace(trace_id)
            trace_table = Table(title=f"Trace {trace_id[:12]}", header_style="bold magenta")
            trace_table.add_column("span")
            trace_table.add_column("ms", justify="right")
            trace_table.add_column("status")
            for record in spans:
                trace_table.add_row(
                    record["name"], f"{record['duration_ms']:.0f}", record["status"]
                )
            console.print(trace_table)

    run(consult())


@app.command("eval")
def run_eval(
    judge: Annotated[bool, typer.Option(help="Ativa as metricas com juiz LLM")] = False,
    limit: Annotated[int | None, typer.Option(help="Avalia apenas os N primeiros casos")] = None,
    category: Annotated[str | None, typer.Option(help="Filtra por categoria")] = None,
    min_pass_rate: Annotated[
        float, typer.Option(help="Taxa minima de aprovacao; abaixo dela o comando falha")
    ] = 0.0,
    save: Annotated[bool, typer.Option(help="Grava o relatorio em reports/")] = True,
) -> None:
    """Roda a avaliacao sobre o golden dataset."""
    from finrag.evaluation import Evaluator, load_golden_dataset, save_report
    from finrag.graph import CorrectiveRAGPipeline

    async def evaluate() -> float:
        cases = load_golden_dataset()
        if category:
            cases = [case for case in cases if case.category == category]
        if limit:
            cases = cases[:limit]
        if not cases:
            console.print("[red]Nenhum caso selecionado.[/red]")
            raise typer.Exit(code=1)

        pipeline = CorrectiveRAGPipeline()
        try:
            report = await Evaluator(pipeline, use_judge=judge).run(cases)
        finally:
            pipeline.retriever.store.close()

        summary = report.summary
        table = Table(title="Resumo da avaliacao", header_style="bold cyan", show_header=False)
        for key, value in summary.items():
            if value is not None:
                table.add_row(key, str(value))
        console.print(table)

        by_category = Table(title="Por categoria", header_style="bold cyan")
        by_category.add_column("categoria")
        by_category.add_column("casos", justify="right")
        by_category.add_column("aprovacao", justify="right")
        by_category.add_column("cobertura de contexto", justify="right")
        for name, values in report.by_category.items():
            by_category.add_row(
                name,
                str(int(values["cases"])),
                f"{values['pass_rate']:.0%}",
                f"{values['context_recall']:.0%}",
            )
        console.print(by_category)

        failures = report.failures()
        if failures:
            console.print(f"\n[yellow]{len(failures)} caso(s) reprovado(s):[/yellow]")
            for case in failures:
                reason = []
                if not case.hit:
                    reason.append("nenhum documento esperado recuperado")
                if case.missing_facts:
                    reason.append(f"fatos ausentes: {', '.join(case.missing_facts)}")
                if not case.abstention_ok:
                    reason.append("abstencao incorreta")
                console.print(f"  [red]{case.case_id}[/red] {case.question}")
                console.print(f"    {'; '.join(reason)}")

        if save:
            path = save_report(report)
            console.print(f"\nRelatorio salvo em [cyan]{path}[/cyan]")

        return float(summary["pass_rate"])

    pass_rate = run(evaluate())
    if pass_rate < min_pass_rate:
        console.print(
            f"[red]Taxa de aprovacao {pass_rate:.0%} abaixo do minimo {min_pass_rate:.0%}.[/red]"
        )
        raise typer.Exit(code=1)


@app.command()
def benchmark(
    output: Annotated[Path, typer.Option(help="Arquivo JSON de saida")] = Path(
        "reports/benchmark.json"
    ),
) -> None:
    """Compara configuracoes de recuperacao no mesmo golden dataset.

    Mede o efeito de cada estagio do pipeline: so denso, so BM25, hibrido sem
    re-rank e hibrido com re-rank. E o experimento que sustenta as escolhas de
    arquitetura documentadas no README.
    """
    import os

    from finrag.evaluation import Evaluator, load_golden_dataset
    from finrag.graph import CorrectiveRAGPipeline

    configurations = [
        (
            "denso_apenas",
            {"FINRAG_RETRIEVAL__TOP_K_SPARSE": "0", "FINRAG_RETRIEVAL__RERANKER": "none"},
        ),
        (
            "bm25_apenas",
            {"FINRAG_RETRIEVAL__TOP_K_DENSE": "0", "FINRAG_RETRIEVAL__RERANKER": "none"},
        ),
        ("hibrido_sem_rerank", {"FINRAG_RETRIEVAL__RERANKER": "none"}),
        ("hibrido_rerank_cross_encoder", {"FINRAG_RETRIEVAL__RERANKER": "cross_encoder"}),
        ("hibrido_rerank_llm", {"FINRAG_RETRIEVAL__RERANKER": "llm"}),
    ]

    original = {
        key: os.environ.get(key)
        for key in (
            "FINRAG_RETRIEVAL__TOP_K_DENSE",
            "FINRAG_RETRIEVAL__TOP_K_SPARSE",
            "FINRAG_RETRIEVAL__RERANKER",
        )
    }
    results: dict[str, dict[str, object]] = {}

    async def compare() -> None:
        cases = load_golden_dataset()
        for name, overrides in configurations:
            for key, value in overrides.items():
                os.environ[key] = value
            reset_settings_cache()
            console.print(f"[cyan]Executando[/cyan] {name}...")
            pipeline = CorrectiveRAGPipeline()
            try:
                report = await Evaluator(pipeline).run(cases)
            except LLMProviderError as exc:
                # Uma comparacao completa custa varias centenas de chamadas. Se o
                # provedor cortar no meio, as configuracoes ja medidas continuam
                # validas e vao para a tabela; perde-las obrigaria a repetir tudo.
                console.print(f"[red]Interrompido em {name}:[/red] {exc}")
                break
            finally:
                pipeline.retriever.store.close()
                for key in overrides:
                    previous = original.get(key)
                    if previous is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = previous
                reset_settings_cache()
            results[name] = report.summary

    run(compare())
    if not results:
        raise typer.Exit(code=2)

    table = Table(title="Benchmark de recuperacao", header_style="bold cyan")
    table.add_column("configuracao")
    table.add_column("aprovacao", justify="right")
    table.add_column("hit rate", justify="right")
    table.add_column("precisao", justify="right")
    table.add_column("cobertura", justify="right")
    table.add_column("custo medio", justify="right")
    table.add_column("p95 ms", justify="right")
    for name, summary in results.items():
        table.add_row(
            name,
            f"{summary['pass_rate']:.0%}",
            f"{summary['retrieval_hit_rate']:.0%}",
            f"{summary['context_precision']:.0%}",
            f"{summary['context_recall']:.0%}",
            f"US$ {summary['avg_cost_usd']:.5f}",
            f"{summary['p95_latency_ms']:.0f}",
        )
    console.print(table)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"Resultado salvo em [cyan]{output}[/cyan]")


@app.command()
def serve(
    host: Annotated[str | None, typer.Option()] = None,
    port: Annotated[int | None, typer.Option()] = None,
    reload: Annotated[bool, typer.Option(help="Recarrega ao salvar (desenvolvimento)")] = False,
) -> None:
    """Sobe a API."""
    import uvicorn

    settings = get_settings().api
    uvicorn.run(
        "finrag.api.main:app",
        host=host or settings.host,
        port=port or settings.port,
        reload=reload,
        log_config=None,
    )


if __name__ == "__main__":
    app()
