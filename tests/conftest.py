"""Fixtures compartilhadas."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from finrag.observability import METRICS, reset_recorder
from finrag.settings import reset_settings_cache
from tests.helpers import FakeEmbedder, FakeLLMClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Indice em memoria, traces em tmp e metricas zeradas em cada teste.

    Corpus e golden dataset apontam para caminho absoluto: assim a suite passa
    independentemente do diretorio de onde o pytest foi chamado.
    """
    monkeypatch.setenv("FINRAG_LLM__API_KEY", "chave-de-teste")
    monkeypatch.setenv("FINRAG_VECTORSTORE__URL", ":memory:")
    monkeypatch.setenv("FINRAG_OBSERVABILITY__TRACE_PATH", str(tmp_path / "spans.jsonl"))
    monkeypatch.setenv("FINRAG_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("FINRAG_CORPUS_DIR", str(PROJECT_ROOT / "data" / "corpus"))
    monkeypatch.setenv(
        "FINRAG_GOLDEN_DATASET", str(PROJECT_ROOT / "evals" / "golden_dataset.jsonl")
    )
    reset_settings_cache()
    reset_recorder()
    METRICS.reset()
    yield
    reset_settings_cache()
    reset_recorder()
    METRICS.reset()


@pytest.fixture
def fake_llm() -> FakeLLMClient:
    return FakeLLMClient()


@pytest.fixture
def store() -> Iterator[Any]:
    from finrag.retrieval.vectorstore import HybridVectorStore

    instance = HybridVectorStore(embedder=FakeEmbedder())  # type: ignore[arg-type]
    try:
        yield instance
    finally:
        instance.close()
