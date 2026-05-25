from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.evaluation.comprehensive_rag_tester import MedicalRAGTester


class TestUnexpectedSuiteConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.config_path = Path("tests/comprehensive_rag_tester.json")
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))

    def test_suite15_is_under_test_suites(self) -> None:
        suites = self.config.get("test_suites", {})
        self.assertIn("suite_15_unexpected_user_phrasings", suites)
        self.assertNotIn("suite_15_unexpected_user_phrasings", self.config)

    def test_suite15_has_minimum_case_count(self) -> None:
        suite = self.config["test_suites"]["suite_15_unexpected_user_phrasings"]
        cases = list(suite.get("cases") or [])
        self.assertGreaterEqual(len(cases), 20)

    def test_suite15_cases_have_required_fields(self) -> None:
        suite = self.config["test_suites"]["suite_15_unexpected_user_phrasings"]
        ids: set[str] = set()
        for case in suite.get("cases", []):
            self.assertTrue(str(case.get("id") or "").strip())
            self.assertTrue(str(case.get("query") or "").strip())
            self.assertTrue(str(case.get("expected_behavior") or "").strip())
            validation = case.get("validation") or {}
            self.assertIsInstance(validation, dict)
            self.assertIn("must_not_hallucinate", validation)
            case_id = str(case["id"])
            self.assertNotIn(case_id, ids)
            ids.add(case_id)

    def test_suite15_intent_is_bound_to_new_validator(self) -> None:
        suite = self.config["test_suites"]["suite_15_unexpected_user_phrasings"]
        self.assertEqual(suite.get("intent"), "unexpected_user_phrasings")

    def test_unexpected_validator_rejects_fail_status(self) -> None:
        tester = MedicalRAGTester(base_url="http://127.0.0.1:8000")
        case = {
            "query": "le patient a quoi ?",
            "expected_behavior": "safety_refusal",
            "validation": {"must_not_hallucinate": True, "must_not_diagnose": True, "must_not_treat": True},
        }
        response = {
            "answer": "Diagnostic probable de ...",
            "validation_status": "fail",
            "sources": [],
        }
        ok, issues = tester._validate_unexpected_phrasing(response, case)  # pylint: disable=protected-access
        self.assertFalse(ok)
        self.assertTrue(any("fail" in issue.lower() for issue in issues))


if __name__ == "__main__":
    unittest.main()
