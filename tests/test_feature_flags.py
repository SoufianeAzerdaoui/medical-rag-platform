from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover
    TestClient = None


class TestFeatureFlags(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.db_path = Path(self._tmpdir.name) / "app_state_test.sqlite3"

        import backend.config as backend_config
        import backend.database as backend_database
        from backend.database import init_schema
        from backend.services import feature_flag_service

        backend_config.APP_DB_PATH = self.db_path
        backend_database.APP_DB_PATH = self.db_path
        init_schema()
        feature_flag_service.ensure_feature_flags_seeded()

    def test_default_flag_exists(self) -> None:
        from backend.services.feature_flag_service import list_feature_flags

        flags = {f["name"]: f for f in list_feature_flags()}
        self.assertIn("REFERENCE_RANGE_STRICT_MODE", flags)
        self.assertTrue(flags["REFERENCE_RANGE_STRICT_MODE"]["enabled"])

    def test_toggle_flag_without_restart(self) -> None:
        from backend.services.feature_flag_service import get_feature_flag, set_feature_flag

        set_feature_flag("REFERENCE_RANGE_STRICT_MODE", False, updated_by="tester@example.com")
        self.assertFalse(get_feature_flag("REFERENCE_RANGE_STRICT_MODE"))
        set_feature_flag("REFERENCE_RANGE_STRICT_MODE", True, updated_by="tester@example.com")
        self.assertTrue(get_feature_flag("REFERENCE_RANGE_STRICT_MODE"))

    def test_endpoint_protected(self) -> None:
        if TestClient is None:
            self.skipTest("fastapi testclient indisponible")
        import backend_api

        client = TestClient(backend_api.app)
        with patch.object(backend_api.config, "ENABLE_FEATURE_FLAG_ADMIN_API", False):
            backend_api.app.dependency_overrides[backend_api.get_current_user] = lambda: {"id": "u1", "email": "user@example.com"}
            try:
                resp = client.patch("/feature-flags/REFERENCE_RANGE_STRICT_MODE", json={"enabled": False})
            finally:
                backend_api.app.dependency_overrides.clear()
            self.assertEqual(resp.status_code, 403)

    def test_endpoint_update(self) -> None:
        if TestClient is None:
            self.skipTest("fastapi testclient indisponible")
        import backend_api

        client = TestClient(backend_api.app)
        with patch.object(backend_api.config, "ENABLE_FEATURE_FLAG_ADMIN_API", True), patch.object(
            backend_api.config, "ADMIN_EMAILS", ("admin@example.com",)
        ):
            backend_api.app.dependency_overrides[backend_api.get_current_user] = lambda: {"id": "u1", "email": "admin@example.com"}
            try:
                resp_patch = client.patch("/feature-flags/REFERENCE_RANGE_STRICT_MODE", json={"enabled": False})
                resp_get = client.get("/feature-flags")
            finally:
                backend_api.app.dependency_overrides.clear()
            self.assertEqual(resp_patch.status_code, 200)
            self.assertEqual(resp_get.status_code, 200)
            items = resp_get.json()
            flags = {f["name"]: f for f in items}
            self.assertIn("REFERENCE_RANGE_STRICT_MODE", flags)
            self.assertFalse(bool(flags["REFERENCE_RANGE_STRICT_MODE"]["enabled"]))


if __name__ == "__main__":
    unittest.main()
