from __future__ import annotations

import sys
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
GENERATION_DIR = SCRIPTS_DIR / "generation"
for module_path in (str(SCRIPTS_DIR), str(GENERATION_DIR)):
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

import generate_answer
from policy_matrix import (
    DETERMINISTIC_ONLY_ROUTES,
    DETERMINISTIC_PREFERRED_ROUTES,
    LLM_ALLOWED_ROUTES,
    SAFETY_ONLY_ROUTES,
    get_intent_policy,
)
from query_understanding import parse_query_understanding


class TestLlmPolicyMatrix(unittest.TestCase):
    def test_llm_allowed_routes_are_explicit(self) -> None:
        self.assertIn("doc_scoped_medical_interpretation_guarded", LLM_ALLOWED_ROUTES)
        self.assertIn("open_grounded_medical_question", LLM_ALLOWED_ROUTES)
        self.assertNotIn("doc_scoped_single_analyte_status", LLM_ALLOWED_ROUTES)
        self.assertNotIn("reference_range_lookup", LLM_ALLOWED_ROUTES)

    def test_deterministic_only_routes_explicitly_block_llm(self) -> None:
        self.assertIn("doc_scoped_single_analyte_status", DETERMINISTIC_ONLY_ROUTES)
        policy = get_intent_policy("doc_scoped_single_analyte_status")
        self.assertEqual(str(policy.get("selected_policy") or ""), "deterministic_only")
        self.assertFalse(bool(policy.get("llm_writer_allowed")))

    def test_safety_only_routes_explicitly_block_llm(self) -> None:
        self.assertIn("diagnostic_safety_question", SAFETY_ONLY_ROUTES)
        self.assertIn("treatment_safety_question", SAFETY_ONLY_ROUTES)
        policy = get_intent_policy("diagnostic_safety_question")
        self.assertEqual(str(policy.get("selected_policy") or ""), "safety_only")
        self.assertFalse(bool(policy.get("llm_writer_allowed")))

    def test_deterministic_preferred_routes_are_not_llm_allowed_by_default(self) -> None:
        self.assertIn("doc_scoped_biological_summary", DETERMINISTIC_PREFERRED_ROUTES)
        policy = get_intent_policy("doc_scoped_biological_summary")
        self.assertEqual(str(policy.get("generation_strategy") or ""), "deterministic_preferred")
        self.assertFalse(bool(policy.get("llm_writer_allowed")))

    def test_unknown_route_defaults_to_non_llm(self) -> None:
        policy = get_intent_policy("unknown_route")
        self.assertEqual(str(policy.get("generation_strategy") or ""), "deterministic_preferred")
        self.assertFalse(bool(policy.get("llm_writer_allowed")))

    def test_global_llm_kill_switch_forces_fallback_writer_mode(self) -> None:
        qu = parse_query_understanding("Fais une synthèse biologique du report 24")
        with patch.object(generate_answer, "_llm_global_enabled", return_value=False):
            self.assertEqual(generate_answer._hybrid_writer_mode(qu), "fallback")

    def test_global_llm_kill_switch_disables_llm_query_understanding(self) -> None:
        qu = parse_query_understanding("TSH normale ?")
        with patch.object(generate_answer, "_llm_global_enabled", return_value=False):
            merged, debug = generate_answer._llm_assisted_query_understanding(
                query="TSH normale ?",
                base_qu=qu,
                llm_client=None,
                provider="ollama",
                model="dummy",
                timeout=10,
            )
        self.assertEqual(merged.intent, qu.intent)
        self.assertFalse(bool(debug.get("used")))
        self.assertEqual(str(debug.get("error") or ""), "llm_globally_disabled")


if __name__ == "__main__":
    unittest.main()
