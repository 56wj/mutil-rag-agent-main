"""Shared Redis client for rate limits and distributed execution slots.

The incident queue itself is Kafka-backed. Redis remains an optional runtime
store for the features above and RAG chat keeps its existing isolated client.
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.config import settings

_client: Any | None = None
_lock: asyncio.Lock | None = None


def _client_lock() -> asyncio.Lock:
    global _lock
    if _lock is None:
        _lock = asyncio.Lock()
    return _lock


async def get_redis_client() -> Any:
    """Return a connected process-local Redis client."""
    global _client
    if _client is not None:
        return _client

    async with _client_lock():
        if _client is not None:
            return _client
        try:
            from redis.asyncio import Redis
        except Exception as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("redis 包不可用") from exc

        client = Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=10,
            health_check_interval=30,
        )
        await client.ping()
        _client = client
        return client


async def close_redis_client() -> None:
    """Close the shared Redis client created in this process."""
    global _client
    if _client is None:
        return
    await _client.aclose()
    _client = None
