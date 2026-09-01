"""Configuracao central da aplicacao.

Toda configuracao entra por variaveis de ambiente com o prefixo ``FINRAG_`` e
delimitador ``__`` para os grupos aninhados (por exemplo
``FINRAG_LLM__MODEL``). Nenhum valor sensivel tem default no codigo.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

RerankerKind = Literal["llm", "cross_encoder", "none"]


class LLMSettings(BaseModel):
    """Provedor de LLM. Qualquer endpoint compativel com a API da OpenAI serve."""

    api_key: SecretStr = SecretStr("")
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0)

    # Precos em USD por 1 milhao de tokens, usados no calculo de custo por resposta.
    cost_per_mtok_input: float = Field(default=0.28, ge=0)
    cost_per_mtok_output: float = Field(default=0.42, ge=0)

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key.get_secret_value())


class EmbeddingSettings(BaseModel):
    """Embeddings densos, executados localmente em ONNX, sem chamada de API.

    O default e o modelo leve (384 dimensoes, 220 MB) porque cabe no runner do
    CI e no laptop de quem clona o repositorio. Para maxima qualidade em
    portugues use ``intfloat/multilingual-e5-large`` (1024 dimensoes, 2,2 GB);
    o ganho medido esta em ``docs/benchmarks.md``.
    """

    model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    sparse_model: str = "Qdrant/bm25"
    batch_size: int = Field(default=32, gt=0)
    cache_dir: Path = Path(".fastembed_cache")

    @property
    def needs_e5_prefix(self) -> bool:
        """Modelos da familia E5 exigem prefixo ``query:``/``passage:``.

        Sem o prefixo a qualidade cai de forma silenciosa, o que e um dos
        erros mais comuns em pipelines de RAG.
        """
        return "e5" in self.model.lower()


class VectorStoreSettings(BaseModel):
    """Qdrant em tres modos.

    O default e o modo embutido com persistencia em disco, porque ``ingest`` e
    ``ask`` sao processos diferentes e um indice em memoria se perderia entre
    eles. ``:memory:`` serve aos testes, e uma URL http aponta para o servidor
    do docker-compose, que e o modo com indice de payload e acesso concorrente.
    """

    url: str = ".qdrant"
    collection: str = "finrag_docs"
    persist_path: Path = Path(".qdrant")
    api_key: SecretStr | None = None

    @property
    def is_remote(self) -> bool:
        return self.url.startswith(("http://", "https://"))


class RetrievalSettings(BaseModel):
    """Busca hibrida e re-ranking."""

    # Zero desliga o ramo. E o que permite medir a contribuicao de cada indice
    # separadamente no comando ``finrag benchmark``.
    top_k_dense: int = Field(default=12, ge=0)
    top_k_sparse: int = Field(default=12, ge=0)
    top_k_final: int = Field(default=5, gt=0)
    rrf_k: int = Field(default=60, gt=0, description="Constante do Reciprocal Rank Fusion")
    # llm: sem download, custa tokens e latencia de rede.
    # cross_encoder: 1,1 GB de download, custo de API zero e latencia menor.
    reranker: RerankerKind = "llm"
    cross_encoder_model: str = "jinaai/jina-reranker-v2-base-multilingual"


class GraphSettings(BaseModel):
    """Parametros do ciclo de Corrective RAG."""

    max_rewrites: int = Field(default=2, ge=0, le=5)
    relevance_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    min_relevant_docs: int = Field(default=2, ge=1)
    enable_grounding_check: bool = True


class ObservabilitySettings(BaseModel):
    log_level: str = "INFO"
    log_json: bool = False
    trace_path: Path = Path("traces/spans.jsonl")
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    @property
    def langfuse_enabled(self) -> bool:
        return bool(
            self.langfuse_public_key
            and self.langfuse_public_key.get_secret_value()
            and self.langfuse_secret_key
            and self.langfuse_secret_key.get_secret_value()
        )


class APISettings(BaseModel):
    # Escuta em todas as interfaces porque o servico roda em container.
    host: str = "0.0.0.0"
    port: int = Field(default=8000, gt=0, le=65535)
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:8501"])


class ChunkingSettings(BaseModel):
    """Chunking sensivel a estrutura do documento.

    Tabelas financeiras perdem sentido se cortadas no meio, por isso o
    tamanho alvo e maior para blocos tabulares do que para prosa.
    """

    target_tokens_prose: int = Field(default=320, gt=0)
    target_tokens_table: int = Field(default=768, gt=0)
    overlap_tokens: int = Field(default=64, ge=0)
    min_chunk_chars: int = Field(default=120, gt=0)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FINRAG_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    corpus_dir: Path = Path("data/corpus")
    golden_dataset: Path = Path("evals/golden_dataset.jsonl")
    reports_dir: Path = Path("reports")

    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    vectorstore: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    graph: GraphSettings = Field(default_factory=GraphSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    api: APISettings = Field(default_factory=APISettings)

    @field_validator("corpus_dir", "golden_dataset", "reports_dir", mode="before")
    @classmethod
    def _expand(cls, value: object) -> object:
        if isinstance(value, str):
            return Path(value).expanduser()
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Instancia unica de configuracao, cacheada pelo processo."""
    return Settings()


def reset_settings_cache() -> None:
    """Limpa o cache. Usado pelos testes que alteram variaveis de ambiente."""
    get_settings.cache_clear()
