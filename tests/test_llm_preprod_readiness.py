from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from scripts.evaluation.analyze_llm_preprod_readiness import (
    ReadinessThresholds,
    build_preprod_readiness_report,
)


class TestLlmPreprodReadiness(unittest.TestCase):
    def _base_go_nogo(self) -> dict:
        return {
            "thresholds": {},
            "suite15_targets": {
                "gates": {
                    "zero_hallucination": True,
                    "zero_diagnosis_leak": True,
                    "zero_treatment_leak": True,
                    "zero_pii_leak": True,
                }
            },
            "gates": {
                "zero_hallucination_final_accepted": True,
                "zero_diagnosis_leak": True,
                "zero_treatment_leak": True,
                "zero_pii_leak": True,
                "fallback_after_llm_rate_acceptable": True,
                "system_better_with_llm_on_allowed_routes": True,
            },
            "llm_on_metrics": {
                "llm_expected_count": 4,
                "llm_attempt_rate": 1.0,
                "llm_accept_rate": 0.75,
                "llm_timeout_rate": 0.05,
                "fallback_after_llm_rate": 0.10,
                "avg_llm_writer_ms": 1200.0,
                "p95_llm_writer_ms": 1900.0,
                "p95_response_time_ms": 2200.0,
            },
            "llm_off_metrics": {
                "llm_expected_count": 4,
                "llm_timeout_rate": 0.02,
                "avg_score_llm_expected": 8.0,
            },
            "llm_vs_baseline": {"enabled": True, "score_delta": 0.5},
        }

    def _row(self, *, llm: bool, fallback: str = "", score: float = 9.0) -> dict:
        return {
            "question_id": "Q4",
            "answer": "Réponse.\n\nSources :\n- [report (16).pdf — page 1](/viewer/pdf?doc_id=report_16&page=1)",
            "generation_mode": "hybrid_structured_llm_writer" if llm else "deterministic_guarded_medical_interpretation",
            "generation_writer": "llm_writer" if llm and not fallback else "professional_fallback" if fallback else "professional_fallback",
            "validation_status": "pass",
            "quality_final_status": "pass",
            "selected_route": "doc_scoped_medical_interpretation_guarded",
            "llm_expected": bool(llm),
            "llm_writer_attempted": bool(llm),
            "llm_writer_accepted": bool(llm and not fallback),
            "fallback_reason": fallback or None,
            "response_time": 0.8 if not fallback else 1.3,
            "sources_count": 1,
            "displayed_count": 1,
            "score": score,
            "validation_errors": [],
            "validation_warnings": [],
            "llm_candidate_validation_errors": [],
            "llm_candidate_validation_warnings": [],
        }

    def test_preprod_readiness_pass(self) -> None:
        go_nogo = self._base_go_nogo()
        on_rows = [self._row(llm=True), self._row(llm=True, fallback="llm_timeout")]
        off_rows = [self._row(llm=False, score=8.0), self._row(llm=False, score=8.5)]
        with patch.dict(
            os.environ,
            {
                "MEDICAL_RAG_LLM_MODEL": "llama3.2:latest",
                "MEDICAL_RAG_LLM_TIMEOUT": "30",
                "MEDICAL_RAG_LLM_MAX_TOKENS": "160",
                "MEDICAL_RAG_LLM_TEMPERATURE": "0.0",
                "MEDICAL_RAG_LLM_MAX_RETRY_ATTEMPTS": "1",
                "MEDICAL_RAG_LLM_TIMEOUT_CIRCUIT_ENABLED": "1",
            },
            clear=False,
        ):
            with patch(
                "scripts.evaluation.analyze_llm_preprod_readiness._load_llm_route_policy",
                return_value={
                    "llm_allowed_routes": ["doc_scoped_medical_interpretation_guarded", "response_transform"],
                    "deterministic_only_routes": ["doc_scoped_single_analyte_status", "reference_range_lookup"],
                },
            ), patch(
                "scripts.evaluation.analyze_llm_preprod_readiness._load_feature_flags_seed",
                return_value={"LLM_GLOBAL_ENABLED"},
            ):
                report = build_preprod_readiness_report(
                    go_nogo_report=go_nogo,
                    llm_on_rows=on_rows,
                    llm_off_rows=off_rows,
                    thresholds=ReadinessThresholds(),
                )
        self.assertTrue(report["technical_readiness"]["all_pass"])
        self.assertTrue(report["product_readiness"]["all_pass"])
        self.assertTrue(report["final_decision"]["all_pass"])
        self.assertTrue(report["final_decision"]["ready_for_preprod"])

    def test_preprod_readiness_fails_when_env_and_coverage_missing(self) -> None:
        go_nogo = self._base_go_nogo()
        go_nogo["gates"]["system_better_with_llm_on_allowed_routes"] = False
        go_nogo["suite15_targets"]["gates"]["zero_hallucination"] = False
        on_rows = [self._row(llm=True, fallback="llm_timeout")]
        off_rows = [self._row(llm=False, score=5.0)]
        with patch.dict(
            os.environ,
            {
                "MEDICAL_RAG_LLM_MODEL": "",
                "MEDICAL_RAG_LLM_TIMEOUT": "",
                "MEDICAL_RAG_LLM_MAX_TOKENS": "",
                "MEDICAL_RAG_LLM_TEMPERATURE": "0.5",
                "MEDICAL_RAG_LLM_MAX_RETRY_ATTEMPTS": "",
                "MEDICAL_RAG_LLM_TIMEOUT_CIRCUIT_ENABLED": "0",
            },
            clear=False,
        ):
            with patch(
                "scripts.evaluation.analyze_llm_preprod_readiness._load_llm_route_policy",
                return_value={
                    "llm_allowed_routes": [],
                    "deterministic_only_routes": [],
                },
            ), patch(
                "scripts.evaluation.analyze_llm_preprod_readiness._load_feature_flags_seed",
                return_value=set(),
            ):
                report = build_preprod_readiness_report(
                    go_nogo_report=go_nogo,
                    llm_on_rows=on_rows,
                    llm_off_rows=off_rows,
                    thresholds=ReadinessThresholds(),
                )
        self.assertFalse(report["technical_readiness"]["all_pass"])
        self.assertFalse(report["product_readiness"]["all_pass"])
        self.assertFalse(report["final_decision"]["all_pass"])
        self.assertFalse(report["final_decision"]["ready_for_preprod"])
        self.assertGreater(len(report["final_decision"]["blocking_items"]), 0)


if __name__ == "__main__":
    unittest.main()
