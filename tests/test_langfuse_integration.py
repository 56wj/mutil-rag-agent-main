from __future__ import annotations

import contextlib
import sys
import types
import unittest
from unittest.mock import patch

# 这个测试只验证可选 Langfuse 适配层；精简 CI/Python 环境未安装项目完整依赖时，
# 给 loguru 提供最小替身，避免把依赖安装状态误判成适配逻辑失败。
try:
    import loguru  # noqa: F401
except ImportError:

    class _Logger:
        def warning(self, *_args, **_kwargs):
            pass

        def info(self, *_args, **_kwargs):
            pass

    sys.modules["loguru"] = types.SimpleNamespace(logger=_Logger())

from app.config import settings
from app.observability import langfuse_client


class _FakeObservation:
    trace_id = "a" * 32

    def __init__(self):
        self.updates = []

    def update(self, **kwargs):
        self.updates.append(kwargs)


class _FakeClient:
    def __init__(self):
        self.observation = _FakeObservation()

    @contextlib.contextmanager
    def start_as_current_observation(self, **_kwargs):
        yield self.observation


class LangfuseAdapterTests(unittest.TestCase):
    def tearDown(self):
        langfuse_client._reset_for_tests()

    def test_disabled_client_is_noop(self):
        with patch.object(settings, "langfuse_enabled", False):
            self.assertFalse(langfuse_client.langfuse_configured())
            self.assertIsNone(langfuse_client.get_langfuse_client())

    def test_enabled_without_credentials_is_noop(self):
        with (
            patch.object(settings, "langfuse_enabled", True),
            patch.object(settings, "langfuse_public_key", ""),
            patch.object(settings, "langfuse_secret_key", ""),
        ):
            self.assertFalse(langfuse_client.langfuse_configured())
            self.assertIsNone(langfuse_client.get_langfuse_client())

    def test_diagnosis_observation_propagates_and_updates(self):
        fake_client = _FakeClient()

        @contextlib.contextmanager
        def propagate_attributes(**_kwargs):
            yield

        fake_module = types.SimpleNamespace(propagate_attributes=propagate_attributes)
        with (
            patch.object(
                langfuse_client, "get_langfuse_client", return_value=fake_client
            ),
            patch.dict(sys.modules, {"langfuse": fake_module}),
        ):
            with langfuse_client.start_diagnosis_observation(
                query="alert",
                session_id="s1",
                requested_mode="fast",
                effective_mode="fast",
            ) as observation:
                self.assertEqual(observation.trace_id, "a" * 32)
                langfuse_client.update_observation(
                    observation,
                    output={"report": "rca"},
                    status="succeeded",
                    elapsed_ms=12,
                )
        self.assertEqual(
            fake_client.observation.updates[0]["output"], {"report": "rca"}
        )
        self.assertEqual(
            fake_client.observation.updates[0]["metadata"]["elapsed_ms"], 12
        )


if __name__ == "__main__":
    unittest.main()
