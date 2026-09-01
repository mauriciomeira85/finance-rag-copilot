"""finance-rag-copilot: RAG corretivo sobre documentos financeiros.

Camadas, de baixo para cima:

    settings / models          contratos e configuracao
    ingestion                  corpus -> chunks estruturais
    retrieval                  busca hibrida -> RRF -> re-rank
    graph                      Corrective RAG em LangGraph
    api / cli / app            interfaces de entrada
    observability              tracing, metricas e custo
    evaluation                 golden dataset e metricas de qualidade
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
