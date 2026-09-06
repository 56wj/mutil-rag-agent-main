"""AgentOps 可观测性公共入口。

指标与 tracing 都是业务链路的旁路能力：依赖缺失或后端不可达时不阻塞诊断。
"""

from app.observability.tracing import (
    current_trace_id,
    extract_trace_context,
    inject_trace_context,
    setup_tracing,
    start_span,
)
from app.observability.langfuse_client import (
    flush_langfuse,
    get_langfuse_client,
    langfuse_configured,
    make_langfuse_callback,
    start_diagnosis_observation,
)

__all__ = [
    "current_trace_id",
    "extract_trace_context",
    "inject_trace_context",
    "setup_tracing",
    "start_span",
    "flush_langfuse",
    "get_langfuse_client",
    "langfuse_configured",
    "make_langfuse_callback",
    "start_diagnosis_observation",
]
