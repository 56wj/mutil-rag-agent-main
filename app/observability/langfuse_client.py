"""Langfuse 的可选 Agent tracing 适配层。

设计边界：
  - Langfuse 记录诊断策略、LangGraph 节点、LLM generation、Tool 与评测分数；
  - Prometheus 记录 API/Worker/Kafka 的低基数运行指标；
  - Langfuse SDK、密钥或服务缺失时，诊断链路保持原行为。

Langfuse Python SDK v4 基于 OpenTelemetry。这里复用进程级 global
TracerProvider，使 Langfuse CallbackHandler 创建的 observation 与现有 W3C trace
上下文保持在同一棵树中。
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator, Mapping
from typing import Any

from loguru import logger

from app.config import settings

_client: Any | None = None
_client_signature: tuple[Any, ...] | None = None
_client_lock = threading.Lock()
_warning_signatures: set[str] = set()


def _warn_once(signature: str, message: str) -> None:
    if signature in _warning_signatures:
        return
    _warning_signatures.add(signature)
    logger.warning(message)


def _configuration_signature() -> tuple[Any, ...]:
    return (
        settings.langfuse_enabled,
        settings.langfuse_public_key,
        settings.langfuse_secret_key,
        settings.langfuse_base_url,
        settings.langfuse_environment,
        settings.langfuse_release,
        settings.langfuse_sample_rate,
    )


def langfuse_configured() -> bool:
    """只有显式启用且公私钥齐全时才初始化 SDK。"""
    return bool(
        settings.observability_enabled
        and settings.langfuse_enabled
        and settings.langfuse_public_key.strip()
        and settings.langfuse_secret_key.strip()
    )


def get_langfuse_client() -> Any | None:
    """返回进程级 Langfuse client；任何接入异常都降级为空操作。"""
    global _client, _client_signature

    signature = _configuration_signature()
    if _client is not None and _client_signature == signature:
        return _client
    if not langfuse_configured():
        if settings.langfuse_enabled:
            _warn_once(
                "credentials",
                "[Langfuse] 已启用但缺少 LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY，"
                "本进程仅保留 Prometheus/OTel 观测",
            )
        return None

    with _client_lock:
        if _client is not None and _client_signature == signature:
            return _client
        try:
            from langfuse import Langfuse

            _client = Langfuse(
                public_key=settings.langfuse_public_key.strip(),
                secret_key=settings.langfuse_secret_key.strip(),
                base_url=settings.langfuse_base_url.rstrip("/"),
                environment=settings.langfuse_environment.strip() or "local",
                release=settings.langfuse_release.strip() or settings.app_version,
                sample_rate=float(settings.langfuse_sample_rate),
            )
            _client_signature = signature
            logger.info(
                "[Langfuse] Agent tracing enabled | "
                f"host={settings.langfuse_base_url} environment={settings.langfuse_environment}"
            )
            return _client
        except ImportError:
            _warn_once(
                "dependency", "[Langfuse] 缺少 langfuse SDK，跳过 Agent trace 导出"
            )
        except Exception as exc:
            _warn_once(
                f"init:{type(exc).__name__}",
                f"[Langfuse] 初始化失败，诊断继续运行: {type(exc).__name__}: {exc}",
            )
    return None


def make_langfuse_callback() -> Any | None:
    """为单次 LangGraph 调用创建 CallbackHandler，捕获 chain/LLM/tool 子树。"""
    if get_langfuse_client() is None:
        return None
    try:
        from langfuse.langchain import CallbackHandler

        return CallbackHandler()
    except Exception as exc:
        _warn_once(
            f"callback:{type(exc).__name__}",
            f"[Langfuse] CallbackHandler 创建失败: {type(exc).__name__}: {exc}",
        )
        return None


@contextlib.contextmanager
def start_diagnosis_observation(
    *,
    query: str,
    session_id: str,
    requested_mode: str,
    effective_mode: str,
    metadata: Mapping[str, Any] | None = None,
) -> Iterator[Any | None]:
    """创建一条语义类型为 agent 的根 observation，并传播比较维度。"""
    client = get_langfuse_client()
    if client is None:
        yield None
        return

    merged_metadata = {
        "requested_mode": requested_mode,
        "effective_mode": effective_mode,
        "app_name": settings.app_name,
        **dict(metadata or {}),
    }
    stack = contextlib.ExitStack()
    try:
        from langfuse import propagate_attributes

        observation = stack.enter_context(
            client.start_as_current_observation(
                as_type="agent",
                name=f"aiops-diagnosis-{effective_mode}",
                input={"query": query},
                metadata=merged_metadata,
                version=settings.langfuse_release.strip() or settings.app_version,
            )
        )
        stack.enter_context(
            propagate_attributes(
                session_id=session_id,
                tags=["aiops", "diagnosis", effective_mode],
                metadata=merged_metadata,
                version=settings.langfuse_release.strip() or settings.app_version,
            )
        )
    except Exception as exc:
        stack.close()
        _warn_once(
            f"observation:{type(exc).__name__}",
            f"[Langfuse] observation 异常，诊断结果不受影响: {type(exc).__name__}: {exc}",
        )
        yield None
        return

    # setup/flush 异常都只影响 telemetry；业务异常仍按原样传播。
    try:
        yield observation
    finally:
        try:
            stack.close()
        except Exception as exc:
            _warn_once(
                f"close:{type(exc).__name__}",
                f"[Langfuse] observation 收尾失败: {type(exc).__name__}: {exc}",
            )


def update_observation(
    observation: Any | None,
    *,
    output: Any | None = None,
    status: str,
    elapsed_ms: int,
    error: BaseException | None = None,
) -> None:
    """收尾根 observation；只写评测有用的输出和运行状态。"""
    if observation is None:
        return
    payload: dict[str, Any] = {
        "output": output,
        "metadata": {"status": status, "elapsed_ms": elapsed_ms},
    }
    if error is not None:
        payload.update(
            level="ERROR",
            status_message=f"{type(error).__name__}: {error}"[:500],
        )
    elif status == "cancelled":
        payload.update(level="WARNING", status_message="diagnosis cancelled")
    try:
        observation.update(**payload)
    except Exception:
        pass


def flush_langfuse() -> bool:
    """短生命周期 benchmark 进程退出前强制发送缓存 trace。"""
    client = _client
    if client is None:
        return False
    try:
        client.flush()
        return True
    except Exception as exc:
        logger.warning(f"[Langfuse] flush failed: {type(exc).__name__}: {exc}")
        return False


def _reset_for_tests() -> None:
    global _client, _client_signature
    _client = None
    _client_signature = None
    _warning_signatures.clear()
