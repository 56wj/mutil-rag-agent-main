"""Kafka-backed diagnosis task queue.

Delivery is at-least-once: a Worker commits a record offset only after the
Postgres task state has been persisted.  Task IDs are used as Kafka keys so
retries for the same task retain partition ordering.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from app.config import settings
from app.observability.metrics import record_kafka, set_kafka_lag
from app.observability.tracing import inject_trace_context, mark_span_error, start_span

PRIORITY_LEVELS = ["critical", "high", "normal", "low"]
SCHEMA_VERSION = 1


def level_for_severity(severity: str) -> str:
    """Map alert severity to a queue priority topic."""
    value = str(severity or "").lower().strip()
    if value in {"critical", "page", "p0"}:
        return "critical"
    if value in {"high", "p1"}:
        return "high"
    if value in {"info", "low", "p3"}:
        return "low"
    return "normal"


def level_for_priority(priority: int) -> str:
    """Map the legacy integer priority (smaller is more urgent) to a level."""
    try:
        value = int(priority)
    except Exception:
        return "normal"
    if value <= 10:
        return "critical"
    if value <= 50:
        return "high"
    if value <= 100:
        return "normal"
    return "low"


class KafkaIncidentQueue:
    """Small Kafka adapter used by the API and Diagnosis Workers."""

    def __init__(self) -> None:
        self._producer: Any | None = None
        self._consumers: dict[str, Any] = {}
        self._consumer_name: str | None = None
        self._connect_lock: asyncio.Lock | None = None
        self._consumer_lock: asyncio.Lock | None = None

    @staticmethod
    def _lock(current: asyncio.Lock | None) -> asyncio.Lock:
        return current or asyncio.Lock()

    def _producer_lock(self) -> asyncio.Lock:
        self._connect_lock = self._lock(self._connect_lock)
        return self._connect_lock

    def _workers_lock(self) -> asyncio.Lock:
        self._consumer_lock = self._lock(self._consumer_lock)
        return self._consumer_lock

    @staticmethod
    def _connection_kwargs() -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "bootstrap_servers": settings.kafka_bootstrap_servers,
            "security_protocol": settings.kafka_security_protocol,
        }
        if settings.kafka_security_protocol.upper().startswith("SASL"):
            kwargs.update(
                sasl_mechanism=settings.kafka_sasl_mechanism,
                sasl_plain_username=settings.kafka_sasl_username,
                sasl_plain_password=settings.kafka_sasl_password,
            )
        return kwargs

    def _base_topic(self) -> str:
        return settings.kafka_incident_topic

    def _topic_for_level(self, level: str) -> str:
        if not settings.incident_queue_priority_enabled:
            return self._base_topic()
        normalized = level if level in PRIORITY_LEVELS else "normal"
        return f"{self._base_topic()}.{normalized}"

    def _consume_topics(self) -> list[str]:
        if not settings.incident_queue_priority_enabled:
            return [self._base_topic()]
        return [f"{self._base_topic()}.{level}" for level in PRIORITY_LEVELS]

    @staticmethod
    def _level_from_topic(topic: str) -> str:
        suffix = topic.rsplit(".", 1)[-1]
        return suffix if suffix in PRIORITY_LEVELS else "base"

    async def connect(self) -> None:
        """Connect the producer and create missing topics when configured."""
        if self._producer is not None:
            return
        async with self._producer_lock():
            if self._producer is not None:
                return
            try:
                from aiokafka import AIOKafkaProducer
            except Exception as exc:  # pragma: no cover - dependency guard
                raise RuntimeError("aiokafka 包不可用，Kafka 队列未启动") from exc

            if settings.kafka_auto_create_topics:
                await self._ensure_topics()

            producer = AIOKafkaProducer(
                **self._connection_kwargs(),
                client_id=f"{settings.kafka_client_id}-producer",
                request_timeout_ms=settings.kafka_request_timeout_ms,
                enable_idempotence=True,
            )
            await producer.start()
            self._producer = producer
            logger.info(
                f"[incident-queue] Kafka connected brokers={settings.kafka_bootstrap_servers} "
                f"topics={','.join(self._consume_topics())}"
            )

    async def _ensure_topics(self) -> None:
        from aiokafka.admin import AIOKafkaAdminClient, NewTopic
        from aiokafka.errors import TopicAlreadyExistsError, for_code

        admin = AIOKafkaAdminClient(
            **self._connection_kwargs(),
            client_id=f"{settings.kafka_client_id}-admin",
            request_timeout_ms=settings.kafka_request_timeout_ms,
        )
        await admin.start()
        try:
            topics = [*self._consume_topics(), settings.kafka_incident_dlq_topic]
            for topic in dict.fromkeys(topics):
                new_topic = NewTopic(
                    name=topic,
                    num_partitions=max(1, settings.kafka_topic_partitions),
                    replication_factor=max(1, settings.kafka_topic_replication_factor),
                    topic_configs={"retention.ms": str(settings.kafka_topic_retention_ms)},
                )
                response = await admin.create_topics([new_topic])
                for topic_error in getattr(response, "topic_errors", []) or []:
                    _, error_code, *details = topic_error
                    if not error_code:
                        continue
                    error_type = for_code(error_code)
                    if error_type is TopicAlreadyExistsError:
                        break
                    error_message = details[0] if details else ""
                    raise error_type(
                        f"创建 Kafka topic={topic} 失败: {error_message or error_type.__name__}"
                    )
                else:
                    logger.info(f"[incident-queue] created Kafka topic={topic}")
        finally:
            await admin.close()

    async def close(self) -> None:
        consumers = list(self._consumers.values())
        self._consumers = {}
        self._consumer_name = None
        for consumer in consumers:
            try:
                await consumer.stop()
            except Exception as exc:
                logger.warning(
                    f"[incident-queue] Kafka consumer close failed: {type(exc).__name__}: {exc}"
                )
        if self._producer is not None:
            await self._producer.stop()
            self._producer = None
        logger.info("[incident-queue] Kafka clients closed")

    async def is_healthy(self) -> bool:
        """Perform a broker metadata request instead of trusting a cached socket."""
        admin = None
        try:
            await self.connect()
            from aiokafka.admin import AIOKafkaAdminClient

            admin = AIOKafkaAdminClient(
                **self._connection_kwargs(),
                client_id=f"{settings.kafka_client_id}-health",
                request_timeout_ms=settings.kafka_request_timeout_ms,
            )
            await admin.start()
            cluster = await admin.describe_cluster()
            return bool(cluster.get("brokers"))
        except Exception:
            return False
        finally:
            if admin is not None:
                await admin.close()

    async def _ensure_consumers(self, consumer_name: str) -> None:
        if self._consumers:
            if self._consumer_name != consumer_name:
                raise RuntimeError(
                    f"Kafka queue 已绑定 consumer={self._consumer_name}，不能切换到 {consumer_name}"
                )
            return

        async with self._workers_lock():
            if self._consumers:
                return
            await self.connect()
            from aiokafka import AIOKafkaConsumer

            max_poll_interval_ms = max(
                settings.kafka_max_poll_interval_ms,
                settings.diagnosis_task_timeout_sec * 1000 + 60000,
            )
            started: dict[str, Any] = {}
            try:
                for topic in self._consume_topics():
                    consumer = AIOKafkaConsumer(
                        topic,
                        **self._connection_kwargs(),
                        client_id=f"{settings.kafka_client_id}-{consumer_name}-{self._level_from_topic(topic)}",
                        group_id=settings.kafka_consumer_group,
                        enable_auto_commit=False,
                        auto_offset_reset=settings.kafka_auto_offset_reset,
                        request_timeout_ms=settings.kafka_request_timeout_ms,
                        max_poll_interval_ms=max_poll_interval_ms,
                        max_poll_records=1,
                    )
                    await consumer.start()
                    started[topic] = consumer
            except Exception:
                for consumer in started.values():
                    await consumer.stop()
                raise
            self._consumers = started
            self._consumer_name = consumer_name
            logger.info(
                f"[incident-queue] Kafka consumers started name={consumer_name} "
                f"group={settings.kafka_consumer_group}"
            )

    async def enqueue_task(
        self,
        *,
        task_id: str,
        incident_group_id: str,
        incident_id: str,
        diagnosis_mode: str,
        priority: int,
        payload: dict[str, Any],
        level: str | None = None,
    ) -> str:
        await self.connect()
        if level is None:
            severity = str((payload or {}).get("severity") or "")
            level = level_for_severity(severity) if severity else level_for_priority(priority)
        topic = self._topic_for_level(level)
        with start_span(
            "kafka.incident.publish",
            kind="producer",
            attributes={
                "messaging.system": "kafka",
                "messaging.destination.name": topic,
                "messaging.operation.name": "publish",
                "aiops.diagnosis.mode": diagnosis_mode,
                "aiops.incident.priority": level,
            },
        ) as span:
            trace_context = inject_trace_context()
            value = {
                "schema_version": SCHEMA_VERSION,
                "task_id": task_id,
                "incident_group_id": incident_group_id,
                "incident_id": incident_id,
                "diagnosis_mode": diagnosis_mode,
                "priority": int(priority),
                "level": level,
                "payload": payload or {},
                # JSON carrier 兼容迁移/重放脚本；Kafka headers 是在线链路的标准传播面。
                "trace_context": trace_context,
                "enqueued_at": datetime.now(timezone.utc).isoformat(),
            }
            try:
                metadata = await self._producer.send_and_wait(
                    topic,
                    key=task_id.encode("utf-8"),
                    value=json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"),
                    headers=[
                        (key, val.encode("utf-8"))
                        for key, val in trace_context.items()
                    ],
                )
            except Exception as exc:
                record_kafka("enqueue", "failed", level)
                mark_span_error(span, exc)
                raise
        record_kafka("enqueue", "ok", level)
        message_id = self._format_message_id(metadata.topic, metadata.partition, metadata.offset)
        logger.info(
            f"[incident-queue] enqueued task={task_id} group={incident_group_id} "
            f"level={level} topic={topic} msg={message_id}"
        )
        return message_id

    async def read_tasks(
        self,
        *,
        consumer_name: str,
        count: int = 1,
        block_ms: int | None = None,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Poll one record at a time, checking priority topics high-to-low."""
        await self._ensure_consumers(consumer_name)
        timeout_ms = block_ms if block_ms is not None else settings.diagnosis_worker_block_ms
        deadline = time.monotonic() + max(0, timeout_ms) / 1000.0

        while True:
            for topic in self._consume_topics():
                consumer = self._consumers[topic]
                records = await consumer.getmany(timeout_ms=0, max_records=max(1, count))
                tasks = self._parse_records(records)
                if tasks:
                    for _, item in tasks:
                        result = "decode_failed" if item.get("__decode_error__") else "ok"
                        record_kafka(
                            "consume",
                            result,
                            str(item.get("level") or self._level_from_topic(topic)),
                        )
                    return tasks
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return []
            await asyncio.sleep(min(0.05, remaining))

    def _parse_records(self, records: Any) -> list[tuple[str, dict[str, Any]]]:
        tasks: list[tuple[str, dict[str, Any]]] = []
        for topic_partition, messages in (records or {}).items():
            for message in messages:
                raw = message.value
                try:
                    text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                    item = json.loads(text)
                    if not isinstance(item, dict):
                        raise ValueError("message JSON must be an object")
                except Exception as exc:
                    item = {
                        "payload": {},
                        "__decode_error__": f"{type(exc).__name__}: {exc}",
                        "__raw_value__": repr(raw)[:4000],
                    }
                item["__topic__"] = message.topic
                item["__partition__"] = message.partition
                item["__offset__"] = message.offset
                header_carrier: dict[str, str] = {}
                for key, raw_header in getattr(message, "headers", None) or []:
                    if str(key).lower() not in {"traceparent", "tracestate", "baggage"}:
                        continue
                    try:
                        header_carrier[str(key)] = (
                            raw_header.decode("utf-8")
                            if isinstance(raw_header, bytes)
                            else str(raw_header)
                        )
                    except Exception:
                        continue
                if header_carrier:
                    item["trace_context"] = header_carrier
                message_id = self._format_message_id(
                    topic_partition.topic, topic_partition.partition, message.offset
                )
                tasks.append((message_id, item))
        return tasks

    @staticmethod
    def _format_message_id(topic: str, partition: int, offset: int) -> str:
        return f"{topic}:{int(partition)}:{int(offset)}"

    @staticmethod
    def _parse_message_id(message_id: str) -> tuple[str, int, int]:
        try:
            topic, partition, offset = str(message_id).rsplit(":", 2)
            return topic, int(partition), int(offset)
        except Exception as exc:
            raise ValueError(f"无效 Kafka message id: {message_id!r}") from exc

    async def ack(self, message_id: str) -> None:
        topic, partition, offset = self._parse_message_id(message_id)
        consumer = self._consumers.get(topic)
        if consumer is None:
            raise RuntimeError(f"Kafka topic={topic} 没有活动 consumer，不能提交 offset")
        from aiokafka import TopicPartition

        level = self._level_from_topic(topic)
        try:
            await consumer.commit({TopicPartition(topic, partition): offset + 1})
        except Exception:
            record_kafka("ack", "failed", level)
            raise
        record_kafka("ack", "ok", level)

    async def dead_letter(
        self,
        *,
        message_id: str,
        item: dict[str, Any],
        reason: str,
    ) -> str:
        """Publish to the DLQ first, then commit the source record."""
        await self.connect()
        value = {
            "schema_version": SCHEMA_VERSION,
            "original_message_id": message_id,
            "reason": reason[:2000],
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "message": item,
        }
        task_id = str(item.get("task_id") or "")
        level = str(item.get("level") or self._level_from_topic(str(item.get("__topic__") or "")))
        try:
            metadata = await self._producer.send_and_wait(
                settings.kafka_incident_dlq_topic,
                key=(task_id or message_id).encode("utf-8"),
                value=json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"),
            )
        except Exception:
            record_kafka("dlq", "failed", level)
            raise
        dlq_id = self._format_message_id(metadata.topic, metadata.partition, metadata.offset)
        await self.ack(message_id)
        record_kafka("dlq", "ok", level)
        logger.warning(
            f"[incident-queue] dead-lettered msg={message_id} dlq={dlq_id} reason={reason[:120]}"
        )
        return dlq_id

    async def status(self) -> dict[str, Any]:
        """Return retained record counts and committed consumer-group lag."""
        out: dict[str, Any] = {
            "configured": True,
            "backend": "kafka",
            "topic": settings.kafka_incident_topic,
            "consumer_group": settings.kafka_consumer_group,
            "dlq_topic": settings.kafka_incident_dlq_topic,
            "priority_enabled": settings.incident_queue_priority_enabled,
            "depth": None,
            "pending": None,
            "lag": None,
            "topic_length": None,
            "dlq_depth": None,
            "workers": [],
            "alive_workers": None,
            "warnings": [],
        }
        try:
            await self.connect()
            from aiokafka import AIOKafkaConsumer, TopicPartition
        except Exception as exc:
            out["configured"] = False
            out["warnings"].append(f"Kafka 不可达: {type(exc).__name__}: {exc}")
            return out

        probe = AIOKafkaConsumer(
            **self._connection_kwargs(),
            client_id=f"{settings.kafka_client_id}-status",
            group_id=settings.kafka_consumer_group,
            enable_auto_commit=False,
            request_timeout_ms=settings.kafka_request_timeout_ms,
        )
        try:
            await probe.start()
            topic_names = [*self._consume_topics(), settings.kafka_incident_dlq_topic]
            topic_partitions = await self._describe_topic_partitions(topic_names)
            depth_by_level: dict[str, int] = {}
            length_by_level: dict[str, int] = {}
            total_depth = 0
            total_length = 0
            for topic in self._consume_topics():
                partitions = topic_partitions.get(topic, [])
                tps = [TopicPartition(topic, part) for part in partitions]
                if not tps:
                    out["warnings"].append(f"topic {topic} 没有分区或不可见")
                    continue
                beginnings = await probe.beginning_offsets(tps)
                ends = await probe.end_offsets(tps)
                topic_depth = 0
                topic_length = 0
                for tp in tps:
                    beginning = int(beginnings[tp])
                    end = int(ends[tp])
                    committed = await probe.committed(tp)
                    cursor = beginning if committed is None else max(beginning, int(committed))
                    topic_depth += max(0, end - cursor)
                    topic_length += max(0, end - beginning)
                level = self._level_from_topic(topic)
                depth_by_level[level] = topic_depth
                length_by_level[level] = topic_length
                total_depth += topic_depth
                total_length += topic_length

            out["depth"] = total_depth
            out["lag"] = total_depth
            out["depth_by_level"] = depth_by_level
            out["lag_by_level"] = dict(depth_by_level)
            out["topic_length"] = total_length
            out["topic_length_by_level"] = length_by_level
            out["dlq_depth"] = await self._retained_topic_size(
                probe,
                settings.kafka_incident_dlq_topic,
                topic_partitions.get(settings.kafka_incident_dlq_topic, []),
                TopicPartition,
            )
            set_kafka_lag(depth_by_level, out["dlq_depth"])
            workers, group_state = await self._describe_workers()
            out["workers"] = workers
            out["alive_workers"] = len(workers)
            out["consumer_group_state"] = group_state
        except Exception as exc:
            out["warnings"].append(f"Kafka 状态查询失败: {type(exc).__name__}: {exc}")
        finally:
            await probe.stop()
        return out

    async def _describe_workers(self) -> tuple[list[dict[str, Any]], str | None]:
        """Collapse the per-priority Kafka consumers into logical Worker rows."""
        from aiokafka.admin import AIOKafkaAdminClient

        admin = AIOKafkaAdminClient(
            **self._connection_kwargs(),
            client_id=f"{settings.kafka_client_id}-group-status",
            request_timeout_ms=settings.kafka_request_timeout_ms,
        )
        await admin.start()
        try:
            responses = await admin.describe_consumer_groups([settings.kafka_consumer_group])
        finally:
            await admin.close()

        workers: dict[str, dict[str, Any]] = {}
        group_state: str | None = None
        prefix = f"{settings.kafka_client_id}-"
        topic_suffixes = [*PRIORITY_LEVELS, "base"]
        for response in responses:
            for group in getattr(response, "groups", []) or []:
                # Raw DescribeGroups response fields:
                # error_code, group, state, protocol_type, protocol, members, ...
                if len(group) < 6 or str(group[1]) != settings.kafka_consumer_group:
                    continue
                group_state = str(group[2])
                for member in group[5] or []:
                    if len(member) < 3:
                        continue
                    client_id = str(member[1])
                    if not client_id.startswith(prefix):
                        continue
                    worker_name = client_id[len(prefix):]
                    for suffix in topic_suffixes:
                        marker = f"-{suffix}"
                        if worker_name.endswith(marker):
                            worker_name = worker_name[: -len(marker)]
                            break
                    row = workers.setdefault(
                        worker_name,
                        {
                            "name": worker_name,
                            "host": str(member[2]),
                            "alive": True,
                            "consumer_clients": 0,
                        },
                    )
                    row["consumer_clients"] += 1
        return sorted(workers.values(), key=lambda row: row["name"]), group_state

    async def _describe_topic_partitions(self, topics: list[str]) -> dict[str, list[int]]:
        """Fetch partition IDs explicitly; status consumers have no subscription."""
        from aiokafka.admin import AIOKafkaAdminClient

        admin = AIOKafkaAdminClient(
            **self._connection_kwargs(),
            client_id=f"{settings.kafka_client_id}-topic-status",
            request_timeout_ms=settings.kafka_request_timeout_ms,
        )
        await admin.start()
        try:
            descriptions = await admin.describe_topics(list(dict.fromkeys(topics)))
        finally:
            await admin.close()

        result: dict[str, list[int]] = {}
        for description in descriptions:
            if int(description.get("error_code") or 0) != 0:
                continue
            result[str(description["topic"])] = sorted(
                int(partition["partition"])
                for partition in description.get("partitions", [])
                if int(partition.get("error_code") or 0) == 0
            )
        return result

    @staticmethod
    async def _retained_topic_size(
        probe: Any,
        topic: str,
        partitions: list[int],
        topic_partition_cls: Any,
    ) -> int | None:
        tps = [topic_partition_cls(topic, part) for part in partitions]
        if not tps:
            return None
        beginnings = await probe.beginning_offsets(tps)
        ends = await probe.end_offsets(tps)
        return sum(max(0, int(ends[tp]) - int(beginnings[tp])) for tp in tps)


incident_queue = KafkaIncidentQueue()
