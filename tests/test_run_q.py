from __future__ import annotations

import contextlib
import io
import unittest
from unittest.mock import patch

from scripts.ops import run_q as run_q_module


class TestRunQ(unittest.TestCase):
    def test_run_q_reports_effective_runtime_fields(self) -> None:
        fake_result = {
            "answer": "ok",
            "generation_time_seconds": 1.234,
            "generation_mode": "hybrid_structured_llm_writer",
            "provider": "ollama",
            "model": "llama3.2:latest",
            "selected_route": "doc_scoped_medical_interpretation_guarded",
            "validation": {
                "validation_status": "pass",
                "warnings": [],
                "errors": [],
            },
            "quality_report": {"final_status": "pass"},
            "query_understanding": {"intent": "doc_scoped_summary", "output_format": "bullet_list"},
            "sources": [{"id": "s1"}],
            "displayed_evidences": [{"id": "e1"}],
            "writer_profile": {"provider": "gemini", "model": "gemini-2.5-flash"},
            "llm_provider_effective_runtime": "gemini",
            "llm_model_effective_runtime": "gemini-2.5-flash",
            "debug": {
                "generation_writer": "llm_writer",
                "selected_route": "doc_scoped_medical_interpretation_guarded",
                "llm_provider": "gemini",
                "llm_model": "gemini-2.5-flash",
                "ollama_endpoint": "http://ollama:11434",
                "ollama_api_kind": "generate",
                "ollama_model": "llama3.2:latest",
                "messages_count": 3,
                "system_prompt_chars": 512,
                "user_prompt_chars": 2048,
                "prompt_chars": 2560,
                "llm_elapsed_ms": 123.4,
                "tokens_per_second_estimate": 18.2,
                "validation_status": "pass",
                "model_verified": True,
            },
        }

        with patch.object(run_q_module, "run_generation", return_value=fake_result):
            report = run_q_module.run_q(query="Quelle est la conclusion ?", provider="ollama", model="llama3.2:latest")

        summary = report["summary"]
        self.assertEqual(summary["routing"]["selected_route"], "doc_scoped_medical_interpretation_guarded")
        self.assertEqual(summary["model"]["provider_requested"], "ollama")
        self.assertEqual(summary["model"]["provider_effective_runtime"], "gemini")
        self.assertEqual(summary["model"]["model_effective_runtime"], "gemini-2.5-flash")
        self.assertTrue(summary["routing"]["model_verified"])
        self.assertEqual(summary["validation"]["status"], "pass")
        self.assertEqual(summary["response"]["sources_count"], 1)
        self.assertEqual(summary["response"]["displayed_evidences_count"], 1)

    def test_print_report_includes_raw_debug(self) -> None:
        fake_report = {
            "request": {"query": "q", "provider": "ollama", "model": "llama3.2:latest"},
            "summary": {
                "routing": {"selected_route": "doc_scoped_summary"},
                "model": {"provider_requested": "ollama", "model_requested": "llama3.2:latest"},
                "validation": {"status": "pass", "errors": [], "warnings": []},
                "response": {"answer": "ok", "generation_time_seconds": 0.5, "response_time_ms": 500.0},
                "raw_debug": {"ollama_model": "llama3.2:latest"},
            },
            "result": {"sources": [], "displayed_evidences": []},
        }

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            run_q_module._print_report(fake_report, show_context=True, show_raw_debug=True)

        output = buffer.getvalue()
        self.assertIn("run_q report", output)
        self.assertIn("Model Runtime", output)
        self.assertIn("Raw Debug", output)
        self.assertIn("llama3.2:latest", output)

    def test_run_q_comparison_reports_multiple_variants(self) -> None:
        fake_results = [
            {
                "answer": "LLM A",
                "generation_time_seconds": 1.0,
                "generation_mode": "hybrid_structured_llm_writer",
                "provider": "ollama",
                "model": "llama3.2:latest",
                "selected_route": "doc_scoped_biological_summary",
                "validation": {"validation_status": "pass", "warnings": [], "errors": []},
                "quality_report": {"final_status": "pass"},
                "query_understanding": {"intent": "doc_scoped_biological_summary", "output_format": "bullet_list"},
                "sources": [{"id": "s1"}],
                "displayed_evidences": [{"id": "e1"}],
                "debug": {
                    "generation_writer": "llm_writer",
                    "selected_route": "doc_scoped_biological_summary",
                    "llm_provider": "ollama",
                    "llm_model": "llama3.2:latest",
                    "llm_expected": True,
                    "final_answer_source": "llm_writer",
                },
            },
            {
                "answer": "LLM B",
                "generation_time_seconds": 2.0,
                "generation_mode": "deterministic_doc_scoped_biological_summary",
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "selected_route": "doc_scoped_biological_summary",
                "validation": {"validation_status": "warning", "warnings": ["narrative_too_short"], "errors": []},
                "quality_report": {"final_status": "warning"},
                "query_understanding": {"intent": "doc_scoped_biological_summary", "output_format": "bullet_list"},
                "sources": [{"id": "s2"}],
                "displayed_evidences": [{"id": "e2"}],
                "debug": {
                    "generation_writer": "professional_fallback",
                    "selected_route": "doc_scoped_biological_summary",
                    "llm_provider": "gemini",
                    "llm_model": "gemini-2.5-flash",
                    "llm_expected": True,
                    "final_answer_source": "deterministic_renderer",
                    "fallback_reason": "quality_gate_failed",
                },
            },
        ]

        with patch.object(run_q_module, "run_generation", side_effect=fake_results):
            report = run_q_module.run_q(
                query="Compare",
                provider="ollama",
                model="llama3.2:latest",
                variants=[("ollama", "llama3.2:latest"), ("gemini", "gemini-2.5-flash")],
            )

        self.assertIn("comparison", report)
        self.assertEqual(len(report["comparison"]), 2)
        self.assertEqual(report["comparison"][0]["provider"], "ollama")
        self.assertEqual(report["comparison"][1]["model"], "gemini-2.5-flash")
        self.assertEqual(report["comparison"][0]["llm_attempted"], True)
        self.assertEqual(report["comparison"][1]["llm_accepted"], False)

    def test_print_report_includes_comparison_table(self) -> None:
        fake_report = {
            "request": {"query": "q", "provider": "ollama", "model": "llama3.2:latest"},
            "variants": [
                {"provider": "ollama", "model": "llama3.2:latest"},
                {"provider": "gemini", "model": "gemini-2.5-flash"},
            ],
            "comparison": [
                {
                    "variant": "run_1",
                    "provider": "ollama",
                    "model": "llama3.2:latest",
                    "llm_attempted": True,
                    "llm_accepted": True,
                    "fallback_reason": None,
                    "latency_ms": 123.4,
                    "final_mode": "hybrid_structured_llm_writer",
                    "final_answer_preview": "Answer A",
                    "final_answer": "Answer A",
                },
                {
                    "variant": "run_2",
                    "provider": "gemini",
                    "model": "gemini-2.5-flash",
                    "llm_attempted": True,
                    "llm_accepted": False,
                    "fallback_reason": "quality_gate_failed",
                    "latency_ms": 45.6,
                    "final_mode": "deterministic_doc_scoped_biological_summary",
                    "final_answer_preview": "Answer B",
                    "final_answer": "Answer B",
                },
            ],
            "reports": [
                {
                    "summary": {
                        "routing": {
                            "generation_mode": "hybrid_structured_llm_writer",
                            "final_answer_source": "llm_writer",
                            "fallback_reason": None,
                            "llm_expected": True,
                        },
                        "model": {"provider_effective_runtime": "ollama", "model_effective_runtime": "llama3.2:latest"},
                        "response": {"answer": "Answer A", "response_time_ms": 123.4},
                        "validation": {"status": "pass", "errors": [], "warnings": []},
                        "raw_debug": {"ollama_model": "llama3.2:latest"},
                    },
                    "result": {"sources": [], "displayed_evidences": []},
                },
                {
                    "summary": {
                        "routing": {
                            "generation_mode": "deterministic_doc_scoped_biological_summary",
                            "final_answer_source": "deterministic_renderer",
                            "fallback_reason": "quality_gate_failed",
                            "llm_expected": True,
                        },
                        "model": {"provider_effective_runtime": "gemini", "model_effective_runtime": "gemini-2.5-flash"},
                        "response": {"answer": "Answer B", "response_time_ms": 45.6},
                        "validation": {"status": "warning", "errors": [], "warnings": ["narrative_too_short"]},
                        "raw_debug": {"gemini_model": "gemini-2.5-flash"},
                    },
                    "result": {"sources": [], "displayed_evidences": []},
                },
            ],
        }

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            run_q_module._print_report(fake_report, show_context=False, show_raw_debug=False)

        output = buffer.getvalue()
        self.assertIn("run_q comparison", output)
        self.assertIn("Comparison Table", output)
        self.assertIn("llama3.2:latest", output)
        self.assertIn("gemini-2.5-flash", output)
        self.assertIn("Answer A", output)
        self.assertIn("Answer B", output)


if __name__ == "__main__":
    unittest.main()
