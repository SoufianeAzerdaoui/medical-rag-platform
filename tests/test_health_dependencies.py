from __future__ import annotations

import unittest
from unittest.mock import patch

import backend_api


class TestHealthDependencies(unittest.TestCase):
    def test_health_includes_dependencies_payload(self) -> None:
        fake_deps = {
            "app_db": {"ok": True, "error": None},
            "clamav": {"ok": True, "required": False, "available": False},
            "ingestion_jobs": {"ok": True},
        }
        with (
            patch.object(backend_api, "_health_dependencies", return_value=fake_deps),
            patch.object(backend_api, "_chunks_count", return_value=10),
        ):
            payload = backend_api.health()
        self.assertIn("dependencies", payload)
        self.assertEqual(payload.get("status"), "ok")
        self.assertEqual(payload.get("dependencies"), fake_deps)


if __name__ == "__main__":
    unittest.main()
