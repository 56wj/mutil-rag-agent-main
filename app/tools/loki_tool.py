"""Loki HTTP API 只读工具：让 LogAgent 查询真实原始日志，而不是只搜模板知识库。"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from langchain_core.tools import tool
from loguru import logger

from app.config import settings

_LABEL_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
_SECRET_PATTERNS = (
    (re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[A-Za-z0-9._~+/=-]+"), r"\1[REDACTED]"),
    (
        re.compile(
            r"(?i)\b(password|passwd|pwd|token|api[_-]?key|secret)\b"
            r"(\s*[:=]\s*)(\"[^\"]*\"|'[^']*'|[^\s,;]+)"
        ),
        r"\1\2[REDACTED]",
    ),
    (re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"), "[REDACTED_JWT]"),
)


def _not_configured_md(action: str) -> str:
    return (
        "## Loki 未配置\n"
        f"动作 `{action}` 未执行：LOKI_URL 为空。\n"
        "LogAgent 将继续使用知识库中的日志模板；接真实日志时配置 "
        "`LOKI_URL=http://loki:3100`。"
    )


def _error_md(action: str, exc: BaseException) -> str:
    return (
        "## Loki 调用失败\n"
        f"动作: `{action}`\n"
        f"错误类型: `{type(exc).__name__}`\n"
        f"错误信息: {str(exc)[:1000]}\n"
        "后续动作: 检查 LogQL、租户/Token、查询窗口和 Loki 连通性。"
    )


def _request_headers() -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if settings.loki_bearer_token:
        headers["Authorization"] = f"Bearer {settings.loki_bearer_token}"
    if settings.loki_tenant_id:
        headers["X-Scope-OrgID"] = settings.loki_tenant_id
    return headers


async def _http_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    import httpx

    url = f"{settings.loki_url.rstrip('/')}{path}"
    async with httpx.AsyncClient(
        timeout=float(settings.loki_timeout_sec),
        verify=bool(settings.loki_tls_verify),
        headers=_request_headers(),
    ) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        payload = response.json()
    if payload.get("status") != "success":
        raise RuntimeError(f"loki error: {payload.get('errorType')}: {payload.get('error')}")
    data = payload.get("data")
    return data if isinstance(data, dict) else {"values": data or []}


def _format_timestamp(nanoseconds: Any) -> str:
    try:
        seconds = int(str(nanoseconds)) / 1_000_000_000
        return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat(timespec="milliseconds")
    except Exception:
        return str(nanoseconds)


def _format_labels(labels: dict[str, Any]) -> str:
    if not labels:
        return "{}"
    body = ",".join(f'{key}="{value}"' for key, value in sorted(labels.items()))
    return "{" + body + "}"


def _redact_line(line: str) -> str:
    if not settings.loki_redact_secrets:
        return line
    redacted = line
    for pattern, replacement in _SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def _format_streams(
    result: List[Dict[str, Any]],
    limit: int,
    *,
    newest_first: bool = True,
) -> str:
    rows: list[tuple[int, str, str]] = []
    for stream in result:
        labels = _format_labels(stream.get("stream") or {})
        for value in stream.get("values") or []:
            if not isinstance(value, (list, tuple)) or len(value) < 2:
                continue
            try:
                sort_key = int(str(value[0]))
            except Exception:
                sort_key = 0
            line = _redact_line(str(value[1])).replace("\r", " ").replace("\n", "\\n")
            rows.append((sort_key, labels, line[:2000]))
    rows.sort(key=lambda row: row[0], reverse=newest_first)
    rows = rows[:limit]
    if not rows:
        return "(查询成功，但窗口内无日志)"
    lines = ["| timestamp (UTC) | stream | line |", "|---|---|---|"]
    for timestamp, labels, line in rows:
        safe_line = line.replace("|", "\\|").replace("`", "\\`")
        safe_labels = labels.replace("|", "\\|").replace("`", "\\`")
        lines.append(f"| {_format_timestamp(timestamp)} | `{safe_labels}` | `{safe_line}` |")
    return "\n".join(lines)


@tool
async def loki_query_range(
    logql: str,
    lookback_seconds: int = 600,
    limit: int = 100,
    direction: str = "backward",
) -> str:
    """执行 Loki LogQL range query，返回最近一段时间的原始日志证据。

    参数:
      logql: LogQL，例如 `{service="checkout"} |= "ERROR"`。
      lookback_seconds: 回看秒数，限制在 60 到 86400 秒。
      limit: 返回日志条数，受 LOKI_MAX_ENTRIES 硬上限保护。
      direction: backward（新到旧）或 forward（旧到新）。
    """
    if not settings.loki_url:
        return _not_configured_md(f"loki_query_range({logql})")
    query = (logql or "").strip()
    if not query or len(query) > 4000:
        return "## Loki 参数错误\nLogQL 不能为空且长度不得超过 4000 字符。"
    normalized_direction = direction.lower().strip()
    if normalized_direction not in {"backward", "forward"}:
        normalized_direction = "backward"
    window = min(86400, max(60, int(lookback_seconds)))
    bounded_limit = min(max(1, int(limit)), max(1, int(settings.loki_max_entries)))
    end_ns = time.time_ns()
    start_ns = end_ns - window * 1_000_000_000
    try:
        data = await _http_get(
            "/loki/api/v1/query_range",
            {
                "query": query,
                "start": str(start_ns),
                "end": str(end_ns),
                "limit": bounded_limit,
                "direction": normalized_direction,
            },
        )
        result = data.get("result") or []
        return (
            f"## Loki LogQL: `{query}`\n"
            f"窗口: 最近 {window}s · streams {len(result)} · limit {bounded_limit}\n\n"
            + _format_streams(
                result,
                bounded_limit,
                newest_first=normalized_direction == "backward",
            )
        )
    except Exception as exc:
        logger.warning(f"[loki] query_range failed logql={query!r}: {exc}")
        return _error_md(f"loki_query_range({query})", exc)


@tool
async def loki_label_values(label: str, lookback_seconds: int = 3600) -> str:
    """列出 Loki 标签取值，用来发现 service/namespace/app/cluster 等查询维度。"""
    if not settings.loki_url:
        return _not_configured_md(f"loki_label_values({label})")
    normalized = (label or "").strip()
    if not _LABEL_RE.fullmatch(normalized):
        return "## Loki 参数错误\nlabel 必须是字母/数字/下划线组成的合法标签名。"
    window = min(86400, max(60, int(lookback_seconds)))
    end_ns = time.time_ns()
    try:
        data = await _http_get(
            f"/loki/api/v1/label/{normalized}/values",
            {"start": str(end_ns - window * 1_000_000_000), "end": str(end_ns)},
        )
        values = data.get("values") or data.get("result") or []
        if not values:
            return f"## Loki label `{normalized}`\n(无取值)"
        rows = "\n".join(f"- `{value}`" for value in values[:500])
        return f"## Loki label `{normalized}` ({len(values)} 个)\n\n{rows}"
    except Exception as exc:
        logger.warning(f"[loki] label_values failed label={normalized!r}: {exc}")
        return _error_md(f"loki_label_values({normalized})", exc)


def get_loki_tools() -> List[Any]:
    return [loki_query_range, loki_label_values] if settings.loki_url else []


def is_configured() -> bool:
    return bool(settings.loki_url)
