from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
GENERATION_ROOT = SCRIPTS_ROOT / "generation"
for root in (SCRIPTS_ROOT, GENERATION_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from query_planner import build_execution_plan
from generate_answer import run_generation
from query_understanding import parse_query_understanding


class TestQueryPlannerPhase4(unittest.TestCase):
    def test_build_execution_plan_single_analyte_doc_scope_high_confidence(self) -> None:
        plan = build_execution_plan(
            {
                "intent": "doc_scoped_single_analyte_status",
                "intent_candidates": [{"intent": "doc_scoped_single_analyte_status", "confidence": 0.91}],
                "intent_confidence": 0.91,
                "scope_confidence": 0.95,
                "ambiguity_flags": [],
                "medical_topics": [{"topic": "renal", "confidence": 0.9}],
                "requested_doc_ids": ["report_29"],
                "requested_analytes": ["creatinine"],
                "technical_condition": "below_reference",
                "safety_intent": None,
            },
            "la créat du report 29 est basse ?",
        )
        self.assertEqual(plan["selected_plan"], "doc_scoped_single_analyte_status")
        self.assertTrue(plan["route_candidates"])
        self.assertGreaterEqual(float(plan["route_candidates"][0]["confidence"]), 0.80)
        self.assertEqual(plan["route_candidates"][0]["policy"], "deterministic_only")

    def test_build_execution_plan_summary_fallback_candidate_present(self) -> None:
        plan = build_execution_plan(
            {
                "intent": "doc_scoped_summary",
                "intent_candidates": [{"intent": "doc_scoped_summary", "confidence": 0.82}],
                "intent_confidence": 0.82,
                "scope_confidence": 0.88,
                "ambiguity_flags": [],
                "medical_topics": [{"topic": "general_biology", "confidence": 0.7}],
                "requested_doc_ids": ["report_24"],
                "requested_analytes": [],
                "technical_condition": "out_of_reference",
                "safety_intent": None,
            },
            "y a quoi d’anormal dans report 24 ?",
        )
        routes = [str(c["route"]) for c in plan["route_candidates"]]
        self.assertIn("doc_scoped_biological_summary", routes)

    def test_run_generation_numeric_result_lookup_routes_to_doc_scoped_results(self) -> None:
        result = run_generation(
            query="Dans le PDF report (12).pdf, quels sont les résultats chiffrés importants ? Donne la valeur, l’unité, la référence si elle existe, et la source.",
            mode="keyword",
            top_k=20,
            index_dir="data/indexes",
        )
        debug = dict(result.get("debug") or {})
        self.assertEqual(str(debug.get("selected_route") or ""), "doc_scoped_results")
        self.assertNotEqual(str(result.get("generation_mode") or ""), "deterministic_no_evidence_response")

    def test_build_execution_plan_numeric_result_lookup_prefers_results_route(self) -> None:
        qu = parse_query_understanding(
            "Dans le PDF report (12).pdf, quels sont les résultats chiffrés importants ? Donne la valeur, l’unité, la référence si elle existe, et la source."
        )
        self.assertTrue(bool((qu.intents or {}).get("doc_scoped_numeric_result_lookup")))
        self.assertEqual(str(qu.intent), "doc_scoped_results")
        plan = build_execution_plan(
            {
                "intent": qu.intent,
                "intent_candidates": [{"intent": qu.intent, "confidence": 0.91}],
                "intent_confidence": 0.91,
                "scope_confidence": 0.95,
                "ambiguity_flags": [],
                "medical_topics": [{"topic": "general_biology", "confidence": 0.7}],
                "requested_doc_ids": list(qu.requested_doc_ids or []),
                "requested_analytes": list(qu.requested_analytes or []),
                "technical_condition": None,
                "safety_intent": None,
            },
            "Dans le PDF report (12).pdf, quels sont les résultats chiffrés importants ?",
        )
        self.assertEqual(plan["selected_plan"], "doc_scoped_results")

    def test_parse_query_understanding_multi_analyte_report_100_routes_to_results(self) -> None:
        qu = parse_query_understanding(
            "Dans le PDF report (100), retrouve les anomalies de LDH, CKMB, bilirubine directe et ammonium. Si elles ne sont pas présentes, dis-le explicitement sans inventer de valeur."
        )
        requested = [str(a).strip().lower() for a in list(qu.requested_analytes or []) if str(a).strip()]
        self.assertIn("ldh", requested)
        self.assertIn("ckmb", requested)
        self.assertIn("bilirubine_directe", requested)
        self.assertIn("ammonium", requested)
        self.assertEqual(str(qu.intent), "doc_scoped_results")
        self.assertTrue(bool((qu.intents or {}).get("multi_analyte_results")))
        self.assertTrue(bool((qu.intents or {}).get("doc_scoped_results")))
        plan = build_execution_plan(
            {
                "intent": qu.intent,
                "intent_candidates": [{"intent": qu.intent, "confidence": 0.93}],
                "intent_confidence": 0.93,
                "scope_confidence": 0.96,
                "ambiguity_flags": [],
                "medical_topics": [{"topic": "hepatic", "confidence": 0.8}],
                "requested_doc_ids": list(qu.requested_doc_ids or []),
                "requested_analytes": list(qu.requested_analytes or []),
                "technical_condition": "out_of_reference",
                "safety_intent": None,
            },
            "Dans le PDF report (100), retrouve les anomalies de LDH, CKMB, bilirubine directe et ammonium.",
        )
        self.assertEqual(plan["selected_plan"], "doc_scoped_results")

    def test_build_execution_plan_medical_summary_prefers_doc_scoped_summary(self) -> None:
        qu = parse_query_understanding(
            "Dans le PDF report (45).pdf, fais un résumé médical clair et fidèle en 4 à 6 lignes, sans interprétation excessive."
        )
        self.assertEqual(str(qu.answer_style), "doctor_note")
        self.assertTrue(bool((qu.intents or {}).get("doc_scoped_medical_summary")))
        self.assertEqual(str(qu.intent), "doc_scoped_summary")
        plan = build_execution_plan(
            {
                "intent": qu.intent,
                "intent_candidates": [{"intent": qu.intent, "confidence": 0.88}],
                "intent_confidence": 0.88,
                "scope_confidence": 0.94,
                "ambiguity_flags": [],
                "medical_topics": [{"topic": "general_biology", "confidence": 0.7}],
                "requested_doc_ids": list(qu.requested_doc_ids or []),
                "requested_analytes": list(qu.requested_analytes or []),
                "technical_condition": None,
                "safety_intent": None,
            },
            "Dans le PDF report (45).pdf, fais un résumé médical clair et fidèle en 4 à 6 lignes, sans interprétation excessive.",
        )
        self.assertIn(plan["selected_plan"], {"doc_scoped_biological_summary", "doc_scoped_summary"})
        self.assertNotEqual(plan["selected_plan"], "unstructured")

    def test_parse_query_understanding_medical_summary_with_explicit_doc_is_not_clarification(self) -> None:
        qu = parse_query_understanding(
            "Dans le PDF report (7), fais un résumé médical clair et fidèle en 4 à 6 lignes, sans interprétation excessive."
        )
        self.assertEqual(str(qu.intent), "doc_scoped_summary")
        self.assertTrue(list(qu.requested_doc_ids or []))
        self.assertNotEqual(str(qu.response_strategy), "ask_clarification")
        self.assertIn(str(qu.response_strategy), {"render_table", "answer_directly"})

    def test_build_execution_plan_safety_intent_prioritizes_refusal_route(self) -> None:
        plan = build_execution_plan(
            {
                "intent": "doc_scoped_results",
                "intent_candidates": [{"intent": "doc_scoped_results", "confidence": 0.80}],
                "intent_confidence": 0.80,
                "scope_confidence": 0.10,
                "ambiguity_flags": ["insufficient_clinical_scope"],
                "medical_topics": [],
                "requested_doc_ids": [],
                "requested_analytes": [],
                "technical_condition": None,
                "safety_intent": "diagnostic_safety_question",
            },
            "le patient a quoi ?",
        )
        self.assertEqual(plan["selected_plan"], "diagnostic_safety_question")
        self.assertIn("diagnosis_refusal", list(plan["fallback_candidates"]))

    def test_build_execution_plan_ambiguous_scope_adds_fallback(self) -> None:
        plan = build_execution_plan(
            {
                "intent": "doc_scoped_results",
                "intent_candidates": [{"intent": "doc_scoped_results", "confidence": 0.58}],
                "intent_confidence": 0.58,
                "scope_confidence": 0.30,
                "ambiguity_flags": ["missing_doc_scope", "multiple_candidates_clustered"],
                "medical_topics": [{"topic": "renal", "confidence": 0.6}],
                "requested_doc_ids": [],
                "requested_analytes": ["creatinine"],
                "technical_condition": None,
                "safety_intent": None,
            },
            "la créat est haute ?",
        )
        fallbacks = list(plan["fallback_candidates"])
        self.assertIn("ambiguous_scope", fallbacks)

    def test_shadow_mode_default_no_takeover(self) -> None:
        os.environ.pop("MEDICAL_RAG_PLANNER_SHADOW_MODE", None)
        os.environ.pop("MEDICAL_RAG_PLANNER_ENABLE_TAKEOVER", None)
        plan = build_execution_plan(
            {
                "intent": "doc_scoped_results",
                "intent_candidates": [{"intent": "doc_scoped_results", "confidence": 0.99}],
                "intent_confidence": 0.99,
                "scope_confidence": 0.99,
                "ambiguity_flags": [],
                "medical_topics": [],
                "requested_doc_ids": ["report_12"],
                "requested_analytes": ["creatinine"],
                "technical_condition": None,
                "safety_intent": None,
            },
            "test",
        )
        self.assertTrue(bool(plan["shadow_mode"]))
        self.assertFalse(bool(plan["takeover_allowed"]))
        self.assertEqual(str(plan["takeover_reason"]), "shadow_mode_default")

    def test_takeover_disabled_even_when_confident_by_default(self) -> None:
        os.environ["MEDICAL_RAG_PLANNER_SHADOW_MODE"] = "1"
        os.environ["MEDICAL_RAG_PLANNER_ENABLE_TAKEOVER"] = "1"
        plan = build_execution_plan(
            {
                "intent": "doc_scoped_single_analyte_status",
                "intent_candidates": [{"intent": "doc_scoped_single_analyte_status", "confidence": 0.99}],
                "intent_confidence": 0.99,
                "scope_confidence": 0.99,
                "ambiguity_flags": [],
                "medical_topics": [{"topic": "renal", "confidence": 1.0}],
                "requested_doc_ids": ["report_29"],
                "requested_analytes": ["creatinine"],
                "technical_condition": "below_reference",
                "safety_intent": None,
            },
            "test",
        )
        self.assertFalse(bool(plan["takeover_allowed"]))
        self.assertEqual(str(plan["takeover_reason"]), "shadow_mode_default")

    def test_confidence_clamped_between_0_and_1(self) -> None:
        plan = build_execution_plan(
            {
                "intent": "doc_scoped_results",
                "intent_candidates": [
                    {"intent": "doc_scoped_results", "confidence": 9.0},
                    {"intent": "reference_range_lookup", "confidence": -3.0},
                ],
                "intent_confidence": 4.0,
                "scope_confidence": -2.0,
                "ambiguity_flags": [],
                "medical_topics": [],
                "requested_doc_ids": ["report_12"],
                "requested_analytes": ["creatinine"],
                "technical_condition": None,
                "safety_intent": None,
            },
            "test",
        )
        for cand in list(plan["route_candidates"]):
            conf = float(cand["confidence"])
            self.assertGreaterEqual(conf, 0.0)
            self.assertLessEqual(conf, 1.0)

    def test_no_hardcoded_report_dependency(self) -> None:
        qu = {
            "intent": "doc_scoped_single_analyte_status",
            "intent_candidates": [{"intent": "doc_scoped_single_analyte_status", "confidence": 0.87}],
            "intent_confidence": 0.87,
            "scope_confidence": 0.91,
            "ambiguity_flags": [],
            "medical_topics": [{"topic": "renal", "confidence": 0.9}],
            "requested_analytes": ["creatinine"],
            "technical_condition": "below_reference",
            "safety_intent": None,
        }
        p1 = build_execution_plan({**qu, "requested_doc_ids": ["report_111"]}, "q1")
        p2 = build_execution_plan({**qu, "requested_doc_ids": ["report_999"]}, "q2")
        self.assertEqual(p1["selected_plan"], p2["selected_plan"])

    def test_debug_payload_contains_planner_fields(self) -> None:
        result = run_generation(
            query="la créat du report 29 est basse ?",
            mode="keyword",
            top_k=20,
            index_dir="data/indexes",
        )
        debug = dict(result.get("debug") or {})
        self.assertIn("route_candidates", debug)
        self.assertIn("rejected_routes", debug)
        self.assertIn("selected_plan", debug)
        self.assertIn("fallback_candidates", debug)
        self.assertIn("fallback_decision_path", debug)
        self.assertIn("canonical_requested_analytes", debug)
        self.assertIn("intent_candidates", debug)
        self.assertIn("intent_confidence", debug)
        self.assertIn("scope_confidence", debug)
        self.assertIn("ambiguity_flags", debug)
        self.assertIn("medical_topics", debug)
        self.assertIn("planner_shadow_mode", debug)
        self.assertIn("planner_takeover_allowed", debug)
        self.assertIn("planner_takeover_reason", debug)
        self.assertIn("planner_version", debug)

    def test_build_execution_plan_exposes_rejected_routes(self) -> None:
        plan = build_execution_plan(
            {
                "intent": "doc_scoped_single_analyte_status",
                "intent_candidates": [
                    {"intent": "doc_scoped_single_analyte_status", "confidence": 0.91},
                    {"intent": "doc_scoped_summary", "confidence": 0.55},
                ],
                "intent_confidence": 0.91,
                "scope_confidence": 0.95,
                "ambiguity_flags": [],
                "medical_topics": [{"topic": "renal", "confidence": 0.9}],
                "requested_doc_ids": ["report_29"],
                "requested_analytes": ["creatinine"],
                "technical_condition": "below_reference",
                "safety_intent": None,
            },
            "la créat du report 29 est basse ?",
        )
        self.assertIn("rejected_routes", plan)
        self.assertIsInstance(plan["rejected_routes"], list)

    def test_backward_compat_routing_unchanged_when_shadow_mode(self) -> None:
        os.environ["MEDICAL_RAG_PLANNER_SHADOW_MODE"] = "1"
        os.environ["MEDICAL_RAG_PLANNER_ENABLE_TAKEOVER"] = "0"
        result = run_generation(
            query="la créat du report 29 est basse ?",
            mode="keyword",
            top_k=20,
            index_dir="data/indexes",
        )
        debug = dict(result.get("debug") or {})
        self.assertEqual(str(debug.get("selected_route") or ""), "doc_scoped_single_analyte_status")
        self.assertFalse(bool(debug.get("planner_takeover_allowed")))


if __name__ == "__main__":
    unittest.main()
