#!/usr/bin/env python3
"""Copy the unacknowledged Redis Streams backlog into Kafka.

The source records are left untouched so the cutover remains reversible.  Stop
the old Redis consumers before running this script; otherwise the backlog can
change while it is being copied.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any, AsyncIterator

from app.config import settings
from app.queue.kafka import PRIORITY_LEVELS, incident_queue


def source_streams(base: str, priority_enabled: bool) -> list[str]:
    if not priority_enabled:
        return [base]
    return [*(f"{base}:{level}" for level in PRIORITY_LEVELS), base]


async def pending_ids(client: Any, stream: str, group: str, batch_size: int) -> list[str]:
    ids: list[str] = []
    cursor = "-"
    while True:
        rows = await client.xpending_range(
            stream, group, min=cursor, max="+", count=batch_size
        )
        if not rows:
            break
        batch = [str(row.get("message_id") or "") for row in rows]
        batch = [message_id for message_id in batch if message_id]
        ids.extend(batch)
        if len(rows) < batch_size or not batch:
            break
        cursor = f"({batch[-1]}"
    return ids


async def group_last_delivered_id(client: Any, stream: str, group: str) -> str | None:
    for row in await client.xinfo_groups(stream):
        if str(row.get("name") or "") == group:
            return str(row.get("last-delivered-id") or "0-0")
    return None


async def iter_stream_backlog(
    client: Any,
    *,
    stream: str,
    group: str,
    batch_size: int,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Yield PEL records plus records after the group's last delivered ID."""
    try:
        last_delivered = await group_last_delivered_id(client, stream, group)
    except Exception as exc:
        if "no such key" in str(exc).lower():
            return
        raise

    seen: set[str] = set()
    if last_delivered is not None:
        for message_id in await pending_ids(client, stream, group, batch_size):
            rows = await client.xrange(stream, min=message_id, max=message_id, count=1)
            if rows:
                seen.add(message_id)
                yield str(rows[0][0]), dict(rows[0][1] or {})

    cursor = "-" if last_delivered is None else f"({last_delivered}"
    while True:
        rows = await client.xrange(stream, min=cursor, max="+", count=batch_size)
        if not rows:
            break
        for message_id, fields in rows:
            message_id = str(message_id)
            if message_id not in seen:
                seen.add(message_id)
                yield message_id, dict(fields or {})
        if len(rows) < batch_size:
            break
        cursor = f"({rows[-1][0]}"


def decode_payload(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(str(raw or "{}"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


async def migrate(args: argparse.Namespace) -> dict[str, Any]:
    from redis.asyncio import Redis

    client = Redis.from_url(args.redis_url, decode_responses=True)
    copied = 0
    scanned = 0
    by_stream: dict[str, int] = {}
    try:
        await client.ping()
        if not args.dry_run:
            await incident_queue.connect()
        for stream in source_streams(args.source_stream, args.priority_enabled):
            stream_count = 0
            async for source_id, fields in iter_stream_backlog(
                client,
                stream=stream,
                group=args.consumer_group,
                batch_size=args.batch_size,
            ):
                scanned += 1
                stream_count += 1
                if args.dry_run:
                    print(f"[dry-run] {stream} {source_id} task={fields.get('task_id', '')}")
                    continue
                try:
                    priority = int(fields.get("priority") or 100)
                except Exception:
                    priority = 100
                await incident_queue.enqueue_task(
                    task_id=str(fields.get("task_id") or ""),
                    incident_group_id=str(fields.get("incident_group_id") or ""),
                    incident_id=str(fields.get("incident_id") or ""),
                    diagnosis_mode=str(fields.get("diagnosis_mode") or "fast"),
                    priority=priority,
                    level=str(fields.get("level") or "") or None,
                    payload=decode_payload(fields.get("payload")),
                )
                copied += 1
            by_stream[stream] = stream_count
    finally:
        await client.aclose()
        await incident_queue.close()
    return {
        "dry_run": args.dry_run,
        "scanned": scanned,
        "copied": copied,
        "by_stream": by_stream,
        "source_unchanged": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy pending/unread Redis Stream diagnosis records into Kafka"
    )
    parser.add_argument("--redis-url", default=settings.redis_url)
    parser.add_argument("--source-stream", default="aiops:incident_tasks")
    parser.add_argument("--consumer-group", default="diagnosis-workers")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--priority-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="read :critical/:high/:normal/:low source streams plus the legacy base stream",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(asyncio.run(migrate(parse_args())), ensure_ascii=False, indent=2))
