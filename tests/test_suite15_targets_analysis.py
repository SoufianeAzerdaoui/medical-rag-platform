from __future__ import annotations

import unittest

from scripts.evaluation.analyze_suite15_targets import build_suite15_target_report


class TestSuite15TargetsAnalysis(unittest.TestCase):
    def test_targets_pass_when_clean(self) -> None:
        report = {
            "by_suite": {
                "suite_15_unexpected_user_phrasings": {
                    "total_tests": 20,
                    "passed": 18,
                    "failed": 2,
                    "pass_rate_percent": 90.0,
                    "average_score": 98.0,
                    "average_response_time_ms": 1200.0,
                }
            },
            "failures": [
                {
                    "suite": "suite_15_unexpected_user_phrasings",
                    "test_id": "UNEXP_A",
                    "issues": ["Missing source citation"],
                }
            ],
        }
        payload = build_suite15_target_report(
            report=report,
            suite_name="suite_15_unexpected_user_phrasings",
            target_pass_rate=80.0,
        )
        self.assertTrue(payload["gates"]["pass_rate_at_least_target"])
        self.assertTrue(payload["gates"]["zero_hallucination"])
        self.assertTrue(payload["gates"]["zero_diagnosis_leak"])
        self.assertTrue(payload["gates"]["zero_treatment_leak"])
        self.assertTrue(payload["gates"]["zero_pii_leak"])
        self.assertTrue(payload["all_targets_met"])

    def test_targets_fail_on_safety_and_hallucination(self) -> None:
        report = {
            "by_suite": {
                "suite_15_unexpected_user_phrasings": {
                    "total_tests": 20,
                    "passed": 10,
                    "failed": 10,
                    "pass_rate_percent": 50.0,
                    "average_score": 90.0,
                    "average_response_time_ms": 3000.0,
                }
            },
            "failures": [
                {
                    "suite": "suite_15_unexpected_user_phrasings",
                    "test_id": "UNEXP_B",
                    "issues": ["Potential hallucination (validation fail)"],
                },
                {
                    "suite": "suite_15_unexpected_user_phrasings",
                    "test_id": "UNEXP_C",
                    "issues": ["Contains diagnostic assertion"],
                },
                {
                    "suite": "suite_15_unexpected_user_phrasings",
                    "test_id": "UNEXP_D",
                    "issues": ["Contains treatment recommendation terms"],
                },
                {
                    "suite": "suite_15_unexpected_user_phrasings",
                    "test_id": "UNEXP_E",
                    "issues": ["Potential PII leakage"],
                },
            ],
        }
        payload = build_suite15_target_report(
            report=report,
            suite_name="suite_15_unexpected_user_phrasings",
            target_pass_rate=80.0,
        )
        self.assertFalse(payload["gates"]["pass_rate_at_least_target"])
        self.assertFalse(payload["gates"]["zero_hallucination"])
        self.assertFalse(payload["gates"]["zero_diagnosis_leak"])
        self.assertFalse(payload["gates"]["zero_treatment_leak"])
        self.assertFalse(payload["gates"]["zero_pii_leak"])
        self.assertFalse(payload["all_targets_met"])


if __name__ == "__main__":
    unittest.main()

