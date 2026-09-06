from __future__ import annotations

import json
import asyncio
import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path

from benchmark.agent_strategy_eval import (
    build_report,
    keyword_group_recall,
    load_cases,
    score_output,
    run_case,
    EvaluationCase,
)


class AgentStrategyEvalTests(unittest.TestCase):
    def test_group_recall_is_and_between_groups_or_within_group(self):
        score = keyword_group_recall(
            "Root cause: connection pool exhausted at max_connections",
            [["连接池", "connection pool"], ["max_connections"], ["cpu"]],
        )
        self.assertEqual(score, 0.6667)

    def test_score_output_uses_report_and_structured_evidence(self):
        output = {
            "status": "succeeded",
            "report": "根因是 Redis 达到 maxmemory，建议扩容并检查 TTL。",
            "evidence": [{"summary": "evicted_keys increased"}],
            "rca": {},
        }
        expected = {
            "root_cause_groups": [["maxmemory"], ["淘汰", "evict"]],
            "evidence_groups": [["evicted_keys"]],
            "remediation_groups": [["扩容"], ["ttl"]],
        }
        self.assertEqual(
            score_output(output, expected),
            {
                "diagnosis_success": 1.0,
                "root_cause_recall": 1.0,
                "evidence_recall": 1.0,
                "remediation_recall": 1.0,
            },
        )

    def test_failed_run_scores_are_all_zero(self):
        output = {
            "status": "failed",
            "report": "MySQL max_connections 已耗尽，建议扩容连接池。",
            "evidence": [{"summary": "Threads_connected=500"}],
            "rca": {"root_cause": "连接池耗尽"},
        }
        expected = {
            "root_cause_groups": [["max_connections"]],
            "evidence_groups": [["Threads_connected"]],
            "remediation_groups": [["扩容"]],
        }
        self.assertEqual(
            score_output(output, expected),
            {
                "diagnosis_success": 0.0,
                "root_cause_recall": 0.0,
                "evidence_recall": 0.0,
                "remediation_recall": 0.0,
            },
        )

    def test_score_does_not_credit_verbatim_incident_echo(self):
        query = "MySQL max_connections=500，Threads_connected=500"
        output = {
            "status": "succeeded",
            "report": f"## 问题\n{query}\n## 结论\n需进一步人工确认。",
            "evidence": [],
            "rca": {},
        }
        expected = {
            "root_cause_groups": [["max_connections"]],
            "evidence_groups": [["Threads_connected"]],
        }
        scores = score_output(output, expected, input_text=query)
        self.assertEqual(scores["root_cause_recall"], 0.0)
        self.assertEqual(scores["evidence_recall"], 0.0)

    def test_loader_rejects_duplicate_case_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.jsonl"
            row = {"id": "same", "input": {"query": "alert"}}
            path.write_text(json.dumps(row) + "\n" + json.dumps(row), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "重复"):
                load_cases(path)

    def test_report_contains_only_fast_and_deep(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.jsonl"
            path.write_text('{"id":"c1","input":{"query":"q"}}\n', encoding="utf-8")
            rows = [
                {
                    "case_id": "c1",
                    "strategy": "fast",
                    "scores": {"root_cause_recall": 0.5},
                    "elapsed_ms": 100,
                    "usage": {"total_tokens": 10, "tool_calls": 1},
                },
                {
                    "case_id": "c1",
                    "strategy": "deep",
                    "scores": {"root_cause_recall": 1.0},
                    "elapsed_ms": 200,
                    "usage": {"total_tokens": 20, "tool_calls": 2},
                },
            ]
            report = build_report(path, rows, experiment_name="x", run_id="r")
        self.assertEqual(report["strategies"], ["fast", "deep"])
        self.assertEqual(report["deep_minus_fast"]["scores"]["root_cause_recall"], 0.5)
        self.assertEqual(report["deep_minus_fast"]["mean_latency_ms"], 100.0)

    def test_run_case_disables_mutable_experience_for_paired_eval(self):
        captured = {}

        async def fake_run(query, **kwargs):
            captured.update({"query": query, **kwargs})
            yield {
                "type": "mode_selected",
                "stage": "diagnosis_mode",
                "data": {"effective_mode": "fast"},
            }
            yield {
                "type": "report",
                "stage": "report_generated",
                "data": {"report": "disk full"},
            }
            yield {
                "type": "progress",
                "stage": "stats",
                "data": {"total_tokens": 12, "tool_calls": 2},
            }
            yield {
                "type": "complete",
                "stage": "diagnosis_complete",
                "data": {"langfuse_trace_id": "f" * 32},
            }

        module = types.SimpleNamespace(run_diagnosis_graph=fake_run)
        case = EvaluationCase(
            id="disk",
            scenario="host",
            input={"query": "disk alert"},
            expected_output={"root_cause_groups": [["disk"]]},
            metadata={},
        )
        with mock.patch.dict(
            sys.modules, {"app.orchestration.diagnosis_runner": module}
        ):
            result = asyncio.run(
                run_case(case, "fast", experiment_name="exp", run_id="run")
            )
        self.assertFalse(captured["experience_recall_enabled"])
        self.assertFalse(captured["persist_experience"])
        self.assertEqual(captured["trace_metadata"]["strategy"], "fast")
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["effective_strategy"], "fast")
        self.assertEqual(result["langfuse_trace_id"], "f" * 32)

    def test_run_case_rejects_completed_authentication_fallback(self):
        async def fake_run(_query, **_kwargs):
            yield {
                "type": "mode_selected",
                "stage": "diagnosis_mode",
                "data": {"effective_mode": "fast"},
            }
            yield {
                "type": "report",
                "stage": "report_generated",
                "data": {
                    "report": "[执行失败: OpenAIAuthenticationError: invalid api key]"
                },
            }
            yield {
                "type": "progress",
                "stage": "stats",
                "data": {"total_tokens": 0, "tool_calls": 0},
            }
            yield {
                "type": "complete",
                "stage": "diagnosis_complete",
                "data": {},
            }

        module = types.SimpleNamespace(run_diagnosis_graph=fake_run)
        case = EvaluationCase(
            id="auth",
            scenario="provider",
            input={"query": "alert"},
            expected_output={"root_cause_groups": [["invalid api key"]]},
            metadata={},
        )
        with mock.patch.dict(
            sys.modules, {"app.orchestration.diagnosis_runner": module}
        ):
            result = asyncio.run(
                run_case(case, "fast", experiment_name="exp", run_id="run")
            )
        self.assertEqual(result["status"], "failed")
        self.assertIn("model_provider_authentication_failed", result["errors"])
        self.assertEqual(result["scores"]["root_cause_recall"], 0.0)


if __name__ == "__main__":
    unittest.main()
