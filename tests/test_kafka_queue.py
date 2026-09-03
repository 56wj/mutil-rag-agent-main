from __future__ import annotations

import json
import unittest
from collections import namedtuple
from types import SimpleNamespace
from unittest.mock import patch

from app.config import settings
from app.queue.kafka import (
    KafkaIncidentQueue,
    level_for_priority,
    level_for_severity,
)
from scripts.migrate_redis_streams_to_kafka import decode_payload, source_streams


FakeTopicPartition = namedtuple("FakeTopicPartition", "topic partition")


class FakeConsumer:
    def __init__(self, batches=None):
        self.batches = list(batches or [])
        self.commits = []

    async def getmany(self, **_kwargs):
        return self.batches.pop(0) if self.batches else {}

    async def commit(self, offsets):
        self.commits.append(offsets)


class FakeProducer:
    def __init__(self):
        self.calls = []

    async def send_and_wait(self, topic, *, key, value):
        self.calls.append({"topic": topic, "key": key, "value": value})
        return SimpleNamespace(topic=topic, partition=2, offset=41)


class KafkaQueueTests(unittest.IsolatedAsyncioTestCase):
    def test_priority_mapping(self):
        self.assertEqual(level_for_severity("P0"), "critical")
        self.assertEqual(level_for_severity("warning"), "normal")
        self.assertEqual(level_for_severity("info"), "low")
        self.assertEqual(level_for_priority(10), "critical")
        self.assertEqual(level_for_priority(51), "normal")
        self.assertEqual(level_for_priority("bad"), "normal")

    def test_topic_mapping(self):
        queue = KafkaIncidentQueue()
        with (
            patch.object(settings, "kafka_incident_topic", "incident.tasks"),
            patch.object(settings, "incident_queue_priority_enabled", True),
        ):
            self.assertEqual(queue._topic_for_level("critical"), "incident.tasks.critical")
            self.assertEqual(
                queue._consume_topics(),
                [
                    "incident.tasks.critical",
                    "incident.tasks.high",
                    "incident.tasks.normal",
                    "incident.tasks.low",
                ],
            )

    def test_message_id_round_trip(self):
        message_id = KafkaIncidentQueue._format_message_id("incident.tasks", 3, 99)
        self.assertEqual(message_id, "incident.tasks:3:99")
        self.assertEqual(
            KafkaIncidentQueue._parse_message_id(message_id),
            ("incident.tasks", 3, 99),
        )
        with self.assertRaises(ValueError):
            KafkaIncidentQueue._parse_message_id("broken")

    def test_redis_backlog_migration_helpers(self):
        self.assertEqual(
            source_streams("old:tasks", True),
            [
                "old:tasks:critical",
                "old:tasks:high",
                "old:tasks:normal",
                "old:tasks:low",
                "old:tasks",
            ],
        )
        self.assertEqual(decode_payload('{"severity":"critical"}'), {"severity": "critical"})
        self.assertEqual(decode_payload("[]"), {})

    async def test_enqueue_serializes_versioned_message(self):
        queue = KafkaIncidentQueue()
        producer = FakeProducer()
        queue._producer = producer
        with (
            patch.object(settings, "kafka_incident_topic", "incident.tasks"),
            patch.object(settings, "incident_queue_priority_enabled", True),
        ):
            message_id = await queue.enqueue_task(
                task_id="task-1",
                incident_group_id="group-1",
                incident_id="incident-1",
                diagnosis_mode="deep",
                priority=10,
                payload={"severity": "critical", "query": "cpu high"},
            )
        self.assertEqual(message_id, "incident.tasks.critical:2:41")
        sent = producer.calls[0]
        self.assertEqual(sent["topic"], "incident.tasks.critical")
        self.assertEqual(sent["key"], b"task-1")
        value = json.loads(sent["value"])
        self.assertEqual(value["schema_version"], 1)
        self.assertEqual(value["task_id"], "task-1")
        self.assertEqual(value["payload"]["query"], "cpu high")

    async def test_read_checks_high_priority_before_normal(self):
        queue = KafkaIncidentQueue()
        topics = [
            "incident.tasks.critical",
            "incident.tasks.high",
            "incident.tasks.normal",
            "incident.tasks.low",
        ]
        normal_tp = FakeTopicPartition(topics[2], 0)
        normal_record = SimpleNamespace(
            topic=topics[2],
            partition=0,
            offset=7,
            value=json.dumps({"task_id": "task-normal", "payload": {}}).encode(),
        )
        queue._consumers = {
            topics[0]: FakeConsumer([{}]),
            topics[1]: FakeConsumer([{}]),
            topics[2]: FakeConsumer([{normal_tp: [normal_record]}]),
            topics[3]: FakeConsumer([]),
        }
        queue._consumer_name = "worker-1"
        with (
            patch.object(settings, "kafka_incident_topic", "incident.tasks"),
            patch.object(settings, "incident_queue_priority_enabled", True),
        ):
            tasks = await queue.read_tasks(
                consumer_name="worker-1", count=1, block_ms=0
            )
        self.assertEqual(tasks[0][0], "incident.tasks.normal:0:7")
        self.assertEqual(tasks[0][1]["task_id"], "task-normal")

    async def test_ack_commits_next_offset(self):
        queue = KafkaIncidentQueue()
        consumer = FakeConsumer()
        queue._consumers = {"incident.tasks": consumer}
        await queue.ack("incident.tasks:1:8")
        offsets = consumer.commits[0]
        [(topic_partition, offset)] = offsets.items()
        self.assertEqual((topic_partition.topic, topic_partition.partition), ("incident.tasks", 1))
        self.assertEqual(offset, 9)

    async def test_worker_status_collapses_priority_consumers(self):
        group = (
            0,
            settings.kafka_consumer_group,
            "Stable",
            "consumer",
            "roundrobin",
            [
                ("member-1", f"{settings.kafka_client_id}-worker-1-critical", "/host-a", b"", b""),
                ("member-2", f"{settings.kafka_client_id}-worker-1-normal", "/host-a", b"", b""),
                ("member-3", f"{settings.kafka_client_id}-worker-2-low", "/host-b", b"", b""),
            ],
        )

        class FakeAdmin:
            async def start(self):
                return None

            async def close(self):
                return None

            async def describe_consumer_groups(self, _groups):
                return [SimpleNamespace(groups=[group])]

        queue = KafkaIncidentQueue()
        with patch("aiokafka.admin.AIOKafkaAdminClient", return_value=FakeAdmin()):
            workers, state = await queue._describe_workers()
        self.assertEqual(state, "Stable")
        self.assertEqual([worker["name"] for worker in workers], ["worker-1", "worker-2"])
        self.assertEqual(workers[0]["consumer_clients"], 2)

    async def test_status_partition_metadata_comes_from_admin(self):
        class FakeAdmin:
            async def start(self):
                return None

            async def close(self):
                return None

            async def describe_topics(self, topics):
                return [
                    {
                        "error_code": 0,
                        "topic": topics[0],
                        "partitions": [
                            {"error_code": 0, "partition": 2},
                            {"error_code": 0, "partition": 0},
                        ],
                    },
                    {
                        "error_code": 3,
                        "topic": topics[1],
                        "partitions": [],
                    },
                ]

        queue = KafkaIncidentQueue()
        with patch("aiokafka.admin.AIOKafkaAdminClient", return_value=FakeAdmin()):
            partitions = await queue._describe_topic_partitions(["topic-a", "topic-b"])
        self.assertEqual(partitions, {"topic-a": [0, 2]})

    async def test_topic_bootstrap_treats_existing_topics_as_success(self):
        class FakeAdmin:
            def __init__(self):
                self.created = []

            async def start(self):
                return None

            async def close(self):
                return None

            async def create_topics(self, topics):
                self.created.extend(topic.name for topic in topics)
                return SimpleNamespace(topic_errors=[(topics[0].name, 36, "exists")])

        admin = FakeAdmin()
        queue = KafkaIncidentQueue()
        with patch("aiokafka.admin.AIOKafkaAdminClient", return_value=admin):
            await queue._ensure_topics()
        self.assertEqual(len(admin.created), 5)
        self.assertIn(settings.kafka_incident_dlq_topic, admin.created)


if __name__ == "__main__":
    unittest.main()
