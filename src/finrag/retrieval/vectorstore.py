"""Indice hibrido no Qdrant.

Uma unica colecao guarda dois vetores por chunk:

* ``dense``  — embedding semantico (ONNX local, sem custo de API);
* ``sparse`` — BM25, que continua imbativel para termo exato.

O segundo existe porque busca densa erra justamente onde documento financeiro
mais dói: codigo de adquirente, numero de contrato, sigla de bandeira, nome de
conta contabil. "MDR" e "taxa de desconto" ficam proximos no espaco vetorial,
mas "ADQ-4471" so aparece se houver casamento lexical.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from finrag.logging_setup import get_logger
from finrag.models import Chunk, ScoredChunk
from finrag.observability import span
from finrag.settings import EmbeddingSettings, VectorStoreSettings, get_settings

if TYPE_CHECKING:  # pragma: no cover
    from qdrant_client import QdrantClient

logger = get_logger(__name__)

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"
_NAMESPACE = uuid.UUID("6f0a2a1e-2d33-4b0a-9c5f-9a1f8d8f0001")


def _point_id(chunk_id: str) -> str:
    """Qdrant aceita UUID ou inteiro; derivamos UUID estavel do chunk_id."""
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


class EmbeddingModel:
    """Embeddings densos e esparsos, carregados sob demanda.

    Carregamento tardio importa: instanciar o modelo ONNX custa segundos e
    memoria, e o processo da API nao deve pagar isso na importacao. O lock
    existe porque a busca densa e a esparsa rodam em threads diferentes: sem
    ele as duas carregariam o modelo em paralelo na primeira consulta.
    """

    def __init__(self, settings: EmbeddingSettings | None = None) -> None:
        self._settings = settings or get_settings().embedding
        self._dense: Any = None
        self._sparse: Any = None
        self._lock = threading.Lock()

    @property
    def settings(self) -> EmbeddingSettings:
        return self._settings

    def _load(self) -> None:
        if self._dense is not None:
            return
        import warnings

        from fastembed import SparseTextEmbedding, TextEmbedding

        with self._lock:
            if self._dense is not None:
                return
            cache = str(self._settings.cache_dir)
            Path(cache).mkdir(parents=True, exist_ok=True)
            with span("embedding.load", model=self._settings.model), warnings.catch_warnings():
                # O fastembed avisa que trocou CLS por mean pooling. Indexacao
                # e consulta usam a mesma versao, entao o aviso e informativo;
                # trocar de versao exige reindexar o corpus.
                warnings.filterwarnings("ignore", message=".*mean pooling.*")
                sparse = SparseTextEmbedding(self._settings.sparse_model, cache_dir=cache)
                dense = TextEmbedding(self._settings.model, cache_dir=cache)
            self._sparse = sparse
            # Publicado por ultimo porque e a sentinela lida sem o lock.
            self._dense = dense

    @property
    def dimension(self) -> int:
        self._load()
        for description in self._dense.list_supported_models():
            if description["model"] == self._settings.model:
                return int(description["dim"])
        raise ValueError(f"Modelo de embedding desconhecido: {self._settings.model}")

    def _prefix(self, texts: Sequence[str], role: str) -> list[str]:
        if not self._settings.needs_e5_prefix:
            return list(texts)
        return [f"{role}: {text}" for text in texts]

    def embed_documents(self, texts: Sequence[str]) -> tuple[list[list[float]], list[Any]]:
        self._load()
        prepared = self._prefix(texts, "passage")
        dense = [
            vector.tolist()
            for vector in self._dense.embed(prepared, batch_size=self._settings.batch_size)
        ]
        sparse = list(self._sparse.embed(list(texts), batch_size=self._settings.batch_size))
        return dense, sparse

    def embed_query(self, text: str) -> tuple[list[float], Any]:
        self._load()
        dense = next(iter(self._dense.query_embed(self._prefix([text], "query")[0]))).tolist()
        sparse = next(iter(self._sparse.query_embed(text)))
        return dense, sparse


class HybridVectorStore:
    """Fachada sobre o Qdrant com indexacao e busca dos dois vetores."""

    def __init__(
        self,
        settings: VectorStoreSettings | None = None,
        embedder: EmbeddingModel | None = None,
    ) -> None:
        self._settings = settings or get_settings().vectorstore
        self._embedder = embedder or EmbeddingModel()
        self._client: QdrantClient | None = None
        self._lock = threading.Lock()

    @property
    def collection(self) -> str:
        return self._settings.collection

    @property
    def client(self) -> QdrantClient:
        """Conexao unica, criada sob demanda e protegida por lock.

        O lock nao e zelo excessivo: no modo embutido o Qdrant trava a pasta de
        armazenamento com lock exclusivo, e as buscas densa e esparsa rodam em
        threads separadas. Sem ele, a primeira consulta abre dois clientes e o
        segundo morre com "storage folder is already accessed".
        """
        if self._client is not None:
            return self._client

        from qdrant_client import QdrantClient

        with self._lock:
            if self._client is not None:
                return self._client
            if self._settings.is_remote:
                key = self._settings.api_key.get_secret_value() if self._settings.api_key else None
                client = QdrantClient(url=self._settings.url, api_key=key, timeout=30)
            elif self._settings.url == ":memory:":
                client = QdrantClient(location=":memory:")
            else:
                self._settings.persist_path.mkdir(parents=True, exist_ok=True)
                client = QdrantClient(path=str(self._settings.persist_path))
            self._client = client
            logger.info("qdrant_conectado", url=self._settings.url)
            return client

    def ensure_collection(self, recreate: bool = False) -> None:
        from qdrant_client import models as qm

        exists = self.client.collection_exists(self.collection)
        if exists and not recreate:
            return
        if exists:
            self.client.delete_collection(self.collection)

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE_VECTOR: qm.VectorParams(
                    size=self._embedder.dimension,
                    distance=qm.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR: qm.SparseVectorParams(
                    modifier=qm.Modifier.IDF,  # BM25 exige IDF calculado pelo servidor
                )
            },
        )
        # Filtro por tipo de documento e competencia e caso de uso frequente
        # ("politica vigente em 2025-09"), entao vale indice dedicado. O modo
        # embutido do Qdrant ignora indices de payload, logo so criamos no
        # servidor real para nao poluir o log com aviso a cada ingestao.
        if self._settings.is_remote:
            for field in ("source.doc_type", "source.period", "doc_id"):
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=qm.PayloadSchemaType.KEYWORD,
                )
        logger.info("colecao_criada", collection=self.collection, dim=self._embedder.dimension)

    def upsert(self, chunks: Sequence[Chunk], batch_size: int = 64) -> int:
        from qdrant_client import models as qm

        if not chunks:
            return 0
        self.ensure_collection()
        total = 0
        with span("vectorstore.upsert", chunks=len(chunks)):
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start : start + batch_size]
                dense, sparse = self._embedder.embed_documents([chunk.text for chunk in batch])
                points = [
                    qm.PointStruct(
                        id=_point_id(chunk.chunk_id),
                        vector={
                            DENSE_VECTOR: dense_vector,
                            SPARSE_VECTOR: qm.SparseVector(
                                indices=sparse_vector.indices.tolist(),
                                values=sparse_vector.values.tolist(),
                            ),
                        },
                        payload=chunk.to_payload(),
                    )
                    for chunk, dense_vector, sparse_vector in zip(batch, dense, sparse, strict=True)
                ]
                self.client.upsert(collection_name=self.collection, points=points, wait=True)
                total += len(points)
        return total

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return int(self.client.count(self.collection, exact=True).count)

    def _filter(self, doc_types: Sequence[str] | None, periods: Sequence[str] | None) -> Any:
        from qdrant_client import models as qm

        conditions = []
        if doc_types:
            conditions.append(
                qm.FieldCondition(key="source.doc_type", match=qm.MatchAny(any=list(doc_types)))
            )
        if periods:
            conditions.append(
                qm.FieldCondition(key="source.period", match=qm.MatchAny(any=list(periods)))
            )
        return qm.Filter(must=conditions) if conditions else None

    def search_dense(
        self,
        query: str,
        limit: int,
        doc_types: Sequence[str] | None = None,
        periods: Sequence[str] | None = None,
    ) -> list[ScoredChunk]:
        dense, _ = self._embedder.embed_query(query)
        with span("vectorstore.search_dense", limit=limit):
            result = self.client.query_points(
                collection_name=self.collection,
                query=dense,
                using=DENSE_VECTOR,
                limit=limit,
                with_payload=True,
                query_filter=self._filter(doc_types, periods),
            )
        return [
            ScoredChunk(chunk=Chunk.from_payload(point.payload or {}), dense_score=point.score)
            for point in result.points
        ]

    def search_sparse(
        self,
        query: str,
        limit: int,
        doc_types: Sequence[str] | None = None,
        periods: Sequence[str] | None = None,
    ) -> list[ScoredChunk]:
        from qdrant_client import models as qm

        _, sparse = self._embedder.embed_query(query)
        with span("vectorstore.search_sparse", limit=limit):
            result = self.client.query_points(
                collection_name=self.collection,
                query=qm.SparseVector(
                    indices=sparse.indices.tolist(), values=sparse.values.tolist()
                ),
                using=SPARSE_VECTOR,
                limit=limit,
                with_payload=True,
                query_filter=self._filter(doc_types, periods),
            )
        return [
            ScoredChunk(chunk=Chunk.from_payload(point.payload or {}), sparse_score=point.score)
            for point in result.points
        ]

    def delete_document(self, doc_id: str) -> None:
        from qdrant_client import models as qm

        self.client.delete(
            collection_name=self.collection,
            points_selector=qm.Filter(
                must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))]
            ),
        )

    def close(self) -> None:
        """Fecha o cliente.

        Necessario no modo embutido: o Qdrant local mantem lock de arquivo e,
        se o processo termina sem fechar, o portalocker tenta liberar o lock
        durante o shutdown do interpretador, quando ``msvcrt`` ja foi
        descarregado, e o traceback aparece no stderr.
        """
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> HybridVectorStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def iter_chunks(self, batch_size: int = 256) -> Iterable[Chunk]:
        """Varre a colecao. Usado pela avaliacao para calcular context recall."""
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
            )
            for point in points:
                yield Chunk.from_payload(point.payload or {})
            if offset is None:
                break
