"""Multi-Agent AIOps 的 Prometheus 指标。

只使用低基数 label（route/mode/tool/model/status/priority），避免把 task_id、query、
trace_id 等高基数字段塞进时序库。模块在 prometheus-client 缺失时自动退化为空操作，
使队列迁移脚本和轻量单测仍可运行；生产镜像通过 requirements.txt 安装真实依赖。
"""

from __future__ import annotations

import os
from typing import Any

try:  # 可选导入，避免只安装最小依赖时阻断业务模块加载。
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        REGISTRY,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
        multiprocess,
        start_http_server,
    )

    PROMETHEUS_AVAILABLE = True
except Exception:  # pragma: no cover - dependency guard
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    REGISTRY = None
    PROMETHEUS_AVAILABLE = False


class _NoopMetric:
    def labels(self, *_args: Any, **_kwargs: Any) -> "_NoopMetric":
        return self

    def inc(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def dec(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def set(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def observe(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _counter(name: str, description: str, labels: tuple[str, ...]) -> Any:
    if not PROMETHEUS_AVAILABLE:
        return _NoopMetric()
    return Counter(name, description, labels)


def _histogram(
    name: str,
    description: str,
    labels: tuple[str, ...],
    buckets: tuple[float, ...],
) -> Any:
    if not PROMETHEUS_AVAILABLE:
        return _NoopMetric()
    return Histogram(name, description, labels, buckets=buckets)


def _gauge(name: str, description: str, labels: tuple[str, ...], **kwargs: Any) -> Any:
    if not PROMETHEUS_AVAILABLE:
        return _NoopMetric()
    return Gauge(name, description, labels, **kwargs)


HTTP_REQUESTS = _counter(
    "aiops_http_requests_total",
    "HTTP requests handled by the AIOps API.",
    ("method", "route", "status"),
)
HTTP_DURATION = _histogram(
    "aiops_http_request_duration_seconds",
    "AIOps API request latency.",
    ("method", "route"),
    (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)

KAFKA_MESSAGES = _counter(
    "aiops_kafka_messages_total",
    "Kafka incident queue operations.",
    ("action", "result", "priority"),
)
KAFKA_CONSUMER_LAG = _gauge(
    "aiops_kafka_consumer_lag",
    "Committed consumer-group lag by incident priority.",
    ("priority",),
    multiprocess_mode="livemax",
)
KAFKA_DLQ_DEPTH = _gauge(
    "aiops_kafka_dlq_depth",
    "Retained records in the incident DLQ.",
    (),
    multiprocess_mode="livemax",
)

DIAGNOSIS_RUNS = _counter(
    "aiops_diagnosis_runs_total",
    "Diagnosis graph runs by mode and terminal status.",
    ("mode", "status"),
)
DIAGNOSIS_DURATION = _histogram(
    "aiops_diagnosis_duration_seconds",
    "End-to-end diagnosis graph duration.",
    ("mode", "status"),
    (0.5, 1, 2.5, 5, 10, 20, 30, 60, 120, 300, 600, 1200),
)
DIAGNOSIS_IN_PROGRESS = _gauge(
    "aiops_diagnosis_in_progress",
    "Diagnosis graph runs currently executing.",
    ("mode",),
    multiprocess_mode="livesum",
)

TOOL_CALLS = _counter(
    "aiops_agent_tool_calls_total",
    "Agent tool calls by tool and result.",
    ("tool", "status"),
)
TOOL_DURATION = _histogram(
    "aiops_agent_tool_duration_seconds",
    "Agent tool call latency.",
    ("tool", "status"),
    (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)

LLM_CALLS = _counter(
    "aiops_llm_calls_total",
    "Logical LLM rounds by model and result.",
    ("model", "status"),
)
LLM_DURATION = _histogram(
    "aiops_llm_call_duration_seconds",
    "Logical LLM round latency including stream fallback.",
    ("model", "status"),
    (0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60, 120),
)
LLM_TOKENS = _counter(
    "aiops_llm_tokens_total",
    "LLM tokens reported by the provider.",
    ("model", "direction"),
)


def record_http(method: str, route: str, status: int, elapsed_seconds: float) -> None:
    HTTP_REQUESTS.labels(method=method, route=route, status=str(status)).inc()
    HTTP_DURATION.labels(method=method, route=route).observe(max(0.0, elapsed_seconds))


def record_kafka(action: str, result: str = "ok", priority: str = "base") -> None:
    KAFKA_MESSAGES.labels(action=action, result=result, priority=priority).inc()


def set_kafka_lag(lag_by_level: dict[str, Any], dlq_depth: Any = None) -> None:
    for level, value in (lag_by_level or {}).items():
        try:
            KAFKA_CONSUMER_LAG.labels(priority=str(level)).set(max(0, int(value)))
        except (TypeError, ValueError):
            continue
    if dlq_depth is not None:
        try:
            KAFKA_DLQ_DEPTH.set(max(0, int(dlq_depth)))
        except (TypeError, ValueError):
            pass


def diagnosis_started(mode: str) -> None:
    DIAGNOSIS_IN_PROGRESS.labels(mode=mode or "unknown").inc()


def diagnosis_finished(mode: str, status: str, elapsed_seconds: float) -> None:
    normalized_mode = mode or "unknown"
    normalized_status = status or "unknown"
    DIAGNOSIS_IN_PROGRESS.labels(mode=normalized_mode).dec()
    DIAGNOSIS_RUNS.labels(mode=normalized_mode, status=normalized_status).inc()
    DIAGNOSIS_DURATION.labels(mode=normalized_mode, status=normalized_status).observe(
        max(0.0, elapsed_seconds)
    )


def record_tool_call(tool: str, status: str, elapsed_seconds: float) -> None:
    normalized_tool = tool or "unknown"
    normalized_status = status or "unknown"
    TOOL_CALLS.labels(tool=normalized_tool, status=normalized_status).inc()
    TOOL_DURATION.labels(tool=normalized_tool, status=normalized_status).observe(
        max(0.0, elapsed_seconds)
    )


def record_llm_call(
    model: str,
    status: str,
    elapsed_seconds: float,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    normalized_model = (model or "unknown")[:120]
    normalized_status = status or "unknown"
    LLM_CALLS.labels(model=normalized_model, status=normalized_status).inc()
    LLM_DURATION.labels(model=normalized_model, status=normalized_status).observe(
        max(0.0, elapsed_seconds)
    )
    if input_tokens > 0:
        LLM_TOKENS.labels(model=normalized_model, direction="input").inc(input_tokens)
    if output_tokens > 0:
        LLM_TOKENS.labels(model=normalized_model, direction="output").inc(output_tokens)


def render_metrics() -> tuple[bytes, str]:
    """Render the current process or Prometheus multiprocess registry."""
    if not PROMETHEUS_AVAILABLE:
        return b"# prometheus_client is not installed\n", CONTENT_TYPE_LATEST
    multiproc_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR", "").strip()
    if multiproc_dir:
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return generate_latest(registry), CONTENT_TYPE_LATEST
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


def start_worker_metrics_server(port: int, addr: str = "0.0.0.0") -> bool:
    """Expose metrics from a standalone Worker process."""
    if not PROMETHEUS_AVAILABLE or int(port) <= 0:
        return False
    start_http_server(int(port), addr=addr)
    return True
