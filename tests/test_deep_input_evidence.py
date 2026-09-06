from __future__ import annotations

import sys
import types
import unittest

# 精简宿主/CI 只跑节点纯逻辑时，不要求安装完整 LangGraph 与日志栈。
try:
    import loguru  # noqa: F401
except ImportError:
    sys.modules["loguru"] = types.SimpleNamespace(
        logger=types.SimpleNamespace(
            info=lambda *_a, **_k: None,
            warning=lambda *_a, **_k: None,
            exception=lambda *_a, **_k: None,
            debug=lambda *_a, **_k: None,
        )
    )

try:
    import langgraph.graph  # noqa: F401
except ImportError:
    graph_module = types.ModuleType("langgraph.graph")
    graph_module.END = "__end__"
    graph_module.START = "__start__"
    graph_module.StateGraph = object
    langgraph_module = types.ModuleType("langgraph")
    langgraph_module.graph = graph_module
    sys.modules["langgraph"] = langgraph_module
    sys.modules["langgraph.graph"] = graph_module

from app.diagnosis_graphs.deep_diagnosis_graph import (
    evidence_reducer_node,
    incident_manager_node,
)
from app.orchestration.diagnosis_runner import _convert_node_event


class DeepInputEvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_manual_input_becomes_ranked_alert_evidence(self):
        state = {
            "input": (
                "checkout-api 持续 5xx；MySQL Threads_connected=500，"
                "max_connections=500；日志出现 Too many connections"
            )
        }

        patch = await incident_manager_node(state)

        self.assertEqual(len(patch["evidences"]), 1)
        evidence = patch["evidences"][0]
        self.assertEqual(evidence["source"], "alert")
        self.assertEqual(evidence["type"], "alert_payload")
        self.assertEqual(evidence["metadata"]["agent"], "incident_manager")
        self.assertEqual(evidence["metadata"]["provenance"], "manual_input")
        self.assertEqual(evidence["metadata"]["trust_level"], "reported")
        self.assertIn("Threads_connected=500", evidence["summary"])

        reduced = await evidence_reducer_node({"evidences": patch["evidences"]})
        self.assertEqual(len(reduced["candidates"]), 1)
        self.assertEqual(reduced["candidates"][0]["type"], "alert_payload")
        self.assertEqual(reduced["candidates"][0]["support_score"], 0.7)
        self.assertEqual(reduced["candidates"][0]["evidence_ids"], ["ev_0"])

    async def test_empty_input_does_not_create_placeholder_evidence(self):
        patch = await incident_manager_node({"input": "   "})
        self.assertEqual(patch["evidences"], [])

    async def test_runner_emits_full_provenance_for_incident_evidence(self):
        node_output = await incident_manager_node({"input": "Redis maxmemory reached"})
        events = [
            event
            async for event in _convert_node_event("incident_manager", node_output)
        ]

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["type"], "evidence")
        self.assertEqual(event["stage"], "incident_input_evidence")
        self.assertEqual(event["data"]["source"], "alert")
        self.assertEqual(event["data"]["evidence_type"], "alert_payload")
        self.assertEqual(
            event["data"]["evidence"]["metadata"]["provenance"],
            "manual_input",
        )


if __name__ == "__main__":
    unittest.main()
