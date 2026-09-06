from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from app.config import settings
from app.observability.tracing import (
    _headers,
    _trace_endpoint,
    current_trace_id,
    extract_trace_context,
    inject_trace_context,
    setup_tracing,
    start_span,
)
from app.tools import loki_tool, prom_tool


class TracingHelpersTests(unittest.TestCase):
    def test_trace_endpoint_appends_signal_path_once(self):
        self.assertEqual(
            _trace_endpoint("http://collector:4318"),
            "http://collector:4318/v1/traces",
        )
        self.assertEqual(
            _trace_endpoint("http://collector:4318/v1/traces"),
            "http://collector:4318/v1/traces",
        )

    def test_otlp_headers_are_parsed(self):
        self.assertEqual(_headers("api-key=TOKEN, tenant = blue"), {"api-key": "TOKEN", "tenant": "blue"})

    def test_w3c_context_round_trip_keeps_trace_id(self):
        with patch.object(settings, "otel_exporter_otlp_endpoint", ""):
            self.assertTrue(setup_tracing("test"))
        with start_span("producer"):
            parent_trace_id = current_trace_id()
            carrier = inject_trace_context()
        self.assertRegex(carrier.get("traceparent", ""), r"^00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}$")
        with start_span("consumer", context=extract_trace_context(carrier)):
            self.assertEqual(current_trace_id(), parent_trace_id)


class LokiToolTests(unittest.IsolatedAsyncioTestCase):
    def test_stream_formatter_bounds_and_escapes_output(self):
        result = [
            {
                "stream": {"service": "checkout"},
                "values": [
                    ["1710000000000000000", "older|line"],
                    ["1710000001000000000", "newer\nline"],
                ],
            }
        ]
        rendered = loki_tool._format_streams(result, limit=1)
        self.assertIn("2024-03-09", rendered)
        self.assertIn("newer\\nline", rendered)
        self.assertNotIn("older", rendered)

    def test_auth_and_tenant_headers(self):
        with (
            patch.object(settings, "loki_bearer_token", "TOKEN"),
            patch.object(settings, "loki_tenant_id", "tenant-a"),
        ):
            self.assertEqual(
                loki_tool._request_headers(),
                {
                    "Accept": "application/json",
                    "Authorization": "Bearer TOKEN",
                    "X-Scope-OrgID": "tenant-a",
                },
            )

    def test_secret_redaction_before_logs_reach_agent(self):
        line = "password=hunter2 Authorization: Bearer abc.def api_key=sk-live eyJabc.def.ghi"
        with patch.object(settings, "loki_redact_secrets", True):
            redacted = loki_tool._redact_line(line)
        self.assertNotIn("hunter2", redacted)
        self.assertNotIn("abc.def", redacted)
        self.assertNotIn("sk-live", redacted)
        self.assertIn("[REDACTED]", redacted)

    async def test_query_range_enforces_configured_limit(self):
        response = {
            "resultType": "streams",
            "result": [
                {"stream": {"service": "api"}, "values": [["1710000000000000000", "ERROR"]]}
            ],
        }
        with (
            patch.object(settings, "loki_url", "http://loki:3100"),
            patch.object(settings, "loki_max_entries", 20),
            patch("app.tools.loki_tool._http_get", new=AsyncMock(return_value=response)) as get,
        ):
            output = await loki_tool.loki_query_range.ainvoke(
                {"logql": '{service="api"} |= "ERROR"', "lookback_seconds": 30, "limit": 999}
            )
        params = get.await_args.args[1]
        self.assertEqual(params["limit"], 20)
        self.assertIn("limit 20", output)
        self.assertIn("ERROR", output)


class PrometheusToolTests(unittest.TestCase):
    def test_auth_and_tenant_headers(self):
        with (
            patch.object(settings, "prometheus_bearer_token", "TOKEN"),
            patch.object(settings, "prometheus_tenant_id", "tenant-b"),
        ):
            self.assertEqual(
                prom_tool._request_headers(),
                {
                    "Accept": "application/json",
                    "Authorization": "Bearer TOKEN",
                    "X-Scope-OrgID": "tenant-b",
                },
            )


if __name__ == "__main__":
    unittest.main()
