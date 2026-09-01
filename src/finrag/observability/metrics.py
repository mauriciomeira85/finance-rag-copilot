"""Metricas agregadas em memoria e exposicao no formato Prometheus.

Nao usa a biblioteca oficial de proposito: o objetivo aqui e um coletor
minusculo, sem estado global de registro, que o endpoint ``/metrics`` serve
em texto. Isso mantem o container leve e o comportamento testavel.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TypedDict


class LatencySnapshot(TypedDict):
    count: int
    p50: float
    p95: float
    p99: float


class MetricsSnapshot(TypedDict):
    counters: dict[str, float]
    routes: dict[str, int]
    latency_ms: LatencySnapshot
    cost_usd_total: float
    cost_usd_per_query: float


@dataclass
class Histogram:
    """Histograma simples com os quantis que interessam em producao."""

    values: list[float] = field(default_factory=list)

    def observe(self, value: float) -> None:
        self.values.append(value)

    def quantile(self, q: float) -> float:
        if not self.values:
            return 0.0
        ordered = sorted(self.values)
        index = min(int(q * (len(ordered) - 1) + 0.5), len(ordered) - 1)
        return ordered[index]

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def total(self) -> float:
        return sum(self.values)


class MetricsRegistry:
    """Contadores e histogramas do servico, seguros para uso concorrente."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, Histogram] = defaultdict(Histogram)
        self._route_counts: dict[str, int] = defaultdict(int)

    def increment(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._counters[name] += value

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._histograms[name].observe(value)

    def count_route(self, route: str) -> None:
        with self._lock:
            self._route_counts[route] += 1

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            latency = self._histograms.get("query_latency_ms", Histogram())
            cost = self._counters.get("llm_cost_usd", 0.0)
            queries = max(self._counters.get("queries_total", 0.0), 1.0)
            return MetricsSnapshot(
                counters=dict(self._counters),
                routes=dict(self._route_counts),
                latency_ms=LatencySnapshot(
                    count=latency.count,
                    p50=round(latency.quantile(0.50), 1),
                    p95=round(latency.quantile(0.95), 1),
                    p99=round(latency.quantile(0.99), 1),
                ),
                cost_usd_total=round(cost, 6),
                cost_usd_per_query=round(cost / queries, 6),
            )

    def render_prometheus(self) -> str:
        """Formato de exposicao do Prometheus, para o Grafana ler direto."""
        snap = self.snapshot()
        lines: list[str] = []
        for name, value in sorted(snap["counters"].items()):
            lines.append(f"# TYPE finrag_{name} counter")
            lines.append(f"finrag_{name} {value}")

        latency = snap["latency_ms"]
        lines.append("# TYPE finrag_query_latency_ms summary")
        quantiles = (("0.5", latency["p50"]), ("0.95", latency["p95"]), ("0.99", latency["p99"]))
        for quantile, value in quantiles:
            lines.append(f'finrag_query_latency_ms{{quantile="{quantile}"}} {value}')
        lines.append(f"finrag_query_latency_ms_count {latency['count']}")

        lines.append("# TYPE finrag_route_total counter")
        for route, count in sorted(snap["routes"].items()):
            lines.append(f'finrag_route_total{{route="{route}"}} {count}')
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._histograms.clear()
            self._route_counts.clear()


METRICS = MetricsRegistry()
