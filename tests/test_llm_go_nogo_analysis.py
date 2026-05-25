from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.evaluation.analyze_llm_go_nogo import (
    GoNoGoThresholds,
    _benchmark_metrics,
    _parse_chat_summary_events,
    _runtime_metrics_from_logs,
    build_llm_go_nogo_report,
)


class TestLlmGoNoGoAnalysis(unittest.TestCase):
    def test_parse_chat_summary_events_and_runtime_metrics(self) -> None:
        content = "\n".join(
            [
                'INFO chat_request_summary {"event":"chat_request_summary","llm_route_class":"llm_allowed","llm_attempt_rate":1.0,"llm_accept_rate":1.0,"llm_reject_rate":0.0,"llm_timeout_rate":0.0,"repair_attempt_rate":0.0,"repair_success_rate":0.0,"fallback_after_llm_rate":0.0,"hallucination_rejection_rate":0.0,"response_time_ms":220.0,"llm_writer_ms":180.0,"failure_signals":[]}',
                '{"event":"chat_request_summary","llm_route_class":"llm_allowed","llm_attempt_rate":1.0,"llm_accept_rate":0.0,"llm_reject_rate":1.0,"llm_timeout_rate":1.0,"repair_attempt_rate":1.0,"repair_success_rate":0.0,"fallback_after_llm_rate":1.0,"hallucination_rejection_rate":1.0,"response_time_ms":880.0,"llm_writer_ms":530.0,"failure_signals":["hallucination"]}',
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "app.log"
            path.write_text(content, encoding="utf-8")
            events = _parse_chat_summary_events(path)
            self.assertEqual(len(events), 2)
            m = _runtime_metrics_from_logs(events)
            self.assertAlmostEqual(m["llm_accept_rate_llm_allowed"], 0.5)
            self.assertAlmostEqual(m["llm_timeout_rate"], 0.5)
            self.assertAlmostEqual(m["fallback_after_llm_rate"], 0.5)
            self.assertEqual(m["failure_signals"].get("hallucination"), 1)

    def test_build_go_nogo_report_passes_when_clean(self) -> None:
        suite15_targets = {
            "violations": {
                "hallucination": {"count": 0, "cases": []},
                "diagnosis": {"count": 0, "cases": []},
                "treatment": {"count": 0, "cases": []},
                "pii": {"count": 0, "cases": []},
            }
        }
        llm_on = {
            "llm_timeout_rate": 0.02,
            "fallback_after_llm_rate": 0.10,
            "llm_accept_rate": 0.75,
            "p95_response_time_ms": 1200.0,
            "p95_llm_writer_ms": 800.0,
            "professional_llm_accept_rate": 1.0,
            "avg_score_llm_expected": 9.2,
        }
        llm_off = {
            "avg_score_llm_expected": 8.5,
        }
        report = build_llm_go_nogo_report(
            suite15_targets=suite15_targets,
            runtime_metrics={},
            llm_on_metrics=llm_on,
            llm_off_metrics=llm_off,
            thresholds=GoNoGoThresholds(),
        )
        self.assertTrue(report["all_targets_met"])

    def test_build_go_nogo_report_fails_on_leaks_and_perf(self) -> None:
        suite15_targets = {
            "violations": {
                "hallucination": {"count": 1, "cases": ["A"]},
                "diagnosis": {"count": 0, "cases": []},
                "treatment": {"count": 0, "cases": []},
                "pii": {"count": 0, "cases": []},
            }
        }
        llm_on = {
            "llm_timeout_rate": 0.40,
            "fallback_after_llm_rate": 0.40,
            "llm_accept_rate": 0.10,
            "p95_response_time_ms": 4500.0,
            "p95_llm_writer_ms": 3300.0,
            "professional_llm_accept_rate": 0.20,
            "avg_score_llm_expected": 7.0,
        }
        llm_off = {
            "avg_score_llm_expected": 7.5,
        }
        report = build_llm_go_nogo_report(
            suite15_targets=suite15_targets,
            runtime_metrics={},
            llm_on_metrics=llm_on,
            llm_off_metrics=llm_off,
            thresholds=GoNoGoThresholds(),
        )
        self.assertFalse(report["all_targets_met"])
        self.assertFalse(report["gates"]["zero_hallucination_final_accepted"])
        self.assertFalse(report["gates"]["llm_timeout_rate_low"])
        self.assertFalse(report["gates"]["system_better_with_llm_on_allowed_routes"])

    def test_benchmark_metrics_extracts_professionality(self) -> None:
        rows = [
            {
                "llm_expected": True,
                "llm_writer_attempted": True,
                "llm_writer_accepted": True,
                "generation_writer": "llm_writer",
                "validation_status": "pass",
                "hard_gate_rejected": False,
                "validation_errors": [],
                "validation_warnings": [],
                "llm_candidate_validation_errors": [],
                "llm_candidate_validation_warnings": [],
                "answer": "Synthèse technique factuelle.",
                "llm_writer_ms": 210.0,
                "response_time": 0.5,
                "score": 9.5,
                "fallback_reason": "",
                "repair_attempted": False,
                "repair_success": False,
            },
            {
                "llm_expected": True,
                "llm_writer_attempted": True,
                "llm_writer_accepted": False,
                "generation_writer": "professional_fallback",
                "validation_status": "warning",
                "hard_gate_rejected": False,
                "validation_errors": ["llm_hallucination"],
                "llm_candidate_validation_errors": ["llm_hallucination"],
                "answer": "",
                "llm_writer_ms": 320.0,
                "response_time": 0.8,
                "score": 8.0,
                "fallback_reason": "llm_validation_failed",
                "repair_attempted": True,
                "repair_success": False,
            },
        ]
        m = _benchmark_metrics(rows)
        self.assertAlmostEqual(m["llm_accept_rate"], 0.5)
        self.assertAlmostEqual(m["fallback_after_llm_rate"], 0.5)
        self.assertGreater(m["professional_llm_accept_rate"], 0.9)


if __name__ == "__main__":
    unittest.main()

