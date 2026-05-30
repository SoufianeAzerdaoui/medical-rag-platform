from __future__ import annotations

import unittest
from unittest.mock import patch

from backend.services import p0_readiness_service


class TestP0ReadinessService(unittest.TestCase):
    def test_secret_strength_rules(self) -> None:
        self.assertFalse(p0_readiness_service._secret_strength("short"))  # type: ignore[attr-defined]
        self.assertFalse(p0_readiness_service._secret_strength("pfe-medical-rag-dev-secret-change-me-1234567890"))  # type: ignore[attr-defined]
        self.assertTrue(p0_readiness_service._secret_strength("A" * 48))  # type: ignore[attr-defined]

    def test_run_report_aggregates_blocking_failures(self) -> None:
        with (
            patch.object(p0_readiness_service, "_check_clamav", return_value={"id": "a", "required": True, "passed": True}),
            patch.object(p0_readiness_service, "_check_encryption", return_value={"id": "b", "required": True, "passed": False}),
            patch.object(p0_readiness_service, "_check_jwt", return_value={"id": "c", "required": True, "passed": True}),
            patch.object(p0_readiness_service, "_check_rbac_ops", return_value={"id": "d", "required": True, "passed": True}),
            patch.object(p0_readiness_service, "_check_audit_immutable", return_value={"id": "e", "required": True, "passed": True}),
            patch.object(p0_readiness_service, "_check_jobs_and_registry", return_value={"id": "f", "required": True, "passed": True}),
            patch.object(p0_readiness_service, "_check_backup_artifacts", return_value={"id": "g", "required": True, "passed": True}),
            patch.object(p0_readiness_service, "_check_e2e_artifacts", return_value={"id": "h", "required": True, "passed": True}),
        ):
            report = p0_readiness_service.run_p0_readiness_check()
        self.assertEqual(report.get("overall_status"), "fail")
        self.assertEqual(int(report.get("blocking_failures") or 0), 1)
        self.assertEqual(len(report.get("checks") or []), 8)


if __name__ == "__main__":
    unittest.main()
