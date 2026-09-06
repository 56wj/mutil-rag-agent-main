"""OpenTelemetry 初始化与跨 Kafka W3C Trace Context 传播。

所有导入均为 best-effort。OTLP Collector 暂时离线时，BatchSpanProcessor 会在后台
重试/丢弃 telemetry，不改变告警入队、消费和诊断结果。
"""

from __future__ import annotations

import contextlib
import os
from typing import Any, Iterator, Mapping

from app.config import settings

_initialized = False
_httpx_instrumented = False


def _headers(value: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in (value or "").split(","):
        key, sep, val = item.strip().partition("=")
        if sep and key.strip():
            result[key.strip()] = val.strip()
    return result


def _trace_endpoint(base: str) -> str:
    value = (base or "").rstrip("/")
    if not value:
        return ""
    return value if value.endswith("/v1/traces") else f"{value}/v1/traces"


def setup_tracing(component: str, fastapi_app: Any | None = None) -> bool:
    """Initialize an OTLP trace provider once per process and instrument HTTP clients."""
    global _initialized, _httpx_instrumented
    if not settings.observability_enabled or not settings.otel_enabled:
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
    except Exception:
        return False

    if not _initialized:
        resource = Resource.create(
            {
                "service.name": f"{settings.otel_service_name}-{component}",
                "service.namespace": settings.app_name,
                "service.version": settings.app_version,
                "deployment.environment.name": settings.otel_environment,
                "service.instance.id": os.environ.get("HOSTNAME", component),
            }
        )
        provider = TracerProvider(
            resource=resource,
            sampler=ParentBased(
                TraceIdRatioBased(min(1.0, max(0.0, settings.otel_trace_sample_ratio)))
            ),
        )
        endpoint = _trace_endpoint(settings.otel_exporter_otlp_endpoint)
        if endpoint:
            exporter = OTLPSpanExporter(
                endpoint=endpoint,
                headers=_headers(settings.otel_exporter_otlp_headers),
                timeout=settings.otel_export_timeout_sec,
            )
            provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _initialized = True

    if not _httpx_instrumented:
        try:
            from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

            HTTPXClientInstrumentor().instrument()
            _httpx_instrumented = True
        except Exception:
            pass

    if fastapi_app is not None:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            FastAPIInstrumentor.instrument_app(
                fastapi_app,
                excluded_urls=settings.otel_excluded_urls,
            )
        except Exception:
            pass
    return True


def inject_trace_context() -> dict[str, str]:
    """Serialize current W3C trace context into a JSON-safe Kafka carrier."""
    if not settings.observability_enabled or not settings.otel_enabled:
        return {}
    try:
        from opentelemetry import propagate

        carrier: dict[str, str] = {}
        propagate.inject(carrier)
        return {
            str(key): str(value)
            for key, value in carrier.items()
            if key.lower() in {"traceparent", "tracestate", "baggage"}
        }
    except Exception:
        return {}


def extract_trace_context(carrier: Mapping[str, Any] | None) -> Any | None:
    """Extract a parent context from an untrusted Kafka message carrier."""
    if not carrier or not settings.observability_enabled or not settings.otel_enabled:
        return None
    try:
        from opentelemetry import propagate

        normalized = {
            str(key): str(value)
            for key, value in carrier.items()
            if str(key).lower() in {"traceparent", "tracestate", "baggage"}
        }
        return propagate.extract(normalized)
    except Exception:
        return None


@contextlib.contextmanager
def start_span(
    name: str,
    *,
    context: Any | None = None,
    attributes: Mapping[str, Any] | None = None,
    kind: str = "internal",
) -> Iterator[Any | None]:
    """Start a span when OTel is present; otherwise behave like nullcontext."""
    try:
        from opentelemetry import trace
        from opentelemetry.trace import SpanKind
    except ImportError:
        yield None
        return

    span_kind = {
        "producer": SpanKind.PRODUCER,
        "consumer": SpanKind.CONSUMER,
        "client": SpanKind.CLIENT,
        "server": SpanKind.SERVER,
    }.get(kind.lower(), SpanKind.INTERNAL)
    tracer = trace.get_tracer("multi-agent-aiops")
    try:
        manager = tracer.start_as_current_span(
            name,
            context=context,
            kind=span_kind,
            attributes=dict(attributes or {}),
            record_exception=True,
            set_status_on_exception=True,
        )
    except Exception:
        yield None
        return
    # 不把业务代码抛出的 ImportError/Exception 吞进 fallback；它们必须按原样传播。
    with manager as span:
        yield span


def mark_span_error(span: Any | None, exc: BaseException) -> None:
    if span is None:
        return
    try:
        from opentelemetry.trace import Status, StatusCode

        span.record_exception(exc)
        span.set_status(Status(StatusCode.ERROR, f"{type(exc).__name__}: {exc}"[:500]))
    except Exception:
        pass


def current_trace_id() -> str:
    try:
        from opentelemetry import trace

        context = trace.get_current_span().get_span_context()
        if not context.is_valid:
            return ""
        return format(context.trace_id, "032x")
    except Exception:
        return ""
