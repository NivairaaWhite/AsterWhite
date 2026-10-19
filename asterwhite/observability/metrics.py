"""
AsterWhite :: observability.metrics

Prometheus metrics for asterwhite.serving. Import `registry` and mount it
at /metrics (serving.app does this via prometheus_client's ASGI app) --
kept in its own module so metric objects are created exactly once even if
multiple parts of the app import this file (prometheus_client raises on
duplicate registration otherwise).
"""
from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "asterwhite_requests_total", "Total inference requests", ["route", "outcome"], registry=registry,
)
REQUEST_LATENCY = Histogram(
    "asterwhite_request_latency_seconds", "Request latency in seconds", ["route"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120), registry=registry,
)
TOKENS_GENERATED = Counter(
    "asterwhite_tokens_generated_total", "Total tokens generated", registry=registry,
)
ACTIVE_REQUESTS = Gauge(
    "asterwhite_active_requests", "Requests currently being served", registry=registry,
)
QUEUE_REJECTIONS = Counter(
    "asterwhite_queue_rejections_total", "Requests rejected for exceeding max_concurrent_requests", registry=registry,
)
