from __future__ import annotations

import json
import logging
import os
import unittest
from typing import Iterable
from unittest.mock import patch

try:
    import pytest
except ModuleNotFoundError:  # pragma: no cover - unittest-only minimal env
    class _PytestCompat:
        class mark:  # type: ignore[valid-type]
            @staticmethod
            def backend_contract(obj):
                return obj

    pytest = _PytestCompat()  # type: ignore[assignment]

try:
    from backend.models import ChatRequest
    from backend.services import chat_service
    _BACKEND_IMPORT_ERROR: ImportError | ModuleNotFoundError | None = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - optional backend deps in lightweight env
    ChatRequest = None  # type: ignore[assignment]
    chat_service = None  # type: ignore[assignment]
    _BACKEND_IMPORT_ERROR = exc


pytestmark = pytest.mark.backend_contract


def _missing_backend_deps_from_exc(exc: ImportError | ModuleNotFoundError) -> list[str]:
    missing: list[str] = []
    if isinstance(exc, ModuleNotFoundError):
        name = str(getattr(exc, "name", "") or "").strip()
        if name:
            missing.append(name)
    # ImportError often carries module path in msg.
    msg = str(exc)
    for token in ("pydantic", "fastapi", "starlette", "backend"):
        if token in msg and token not in missing:
            missing.append(token)
    return missing or ["unknown_dependency"]


def _format_skip_message(missing: Iterable[str]) -> str:
    ordered = []
    seen = set()
    for dep in missing:
        key = str(dep).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return (
        "Skipping chat API contract test: backend dependencies are missing: "
        + ", ".join(ordered)
    )


class _FakeStateService:
    def __init__(self) -> None:
        self._states: dict[str, dict] = {}

    def cleanup_expired(self) -> int:
        return 0

    def hydrate_from_db_if_present(self, conversation_id: str) -> None:
        self._states.setdefault(conversation_id, {"state_version": 1})

    def load(self, conversation_id: str) -> dict:
        return self._states.setdefault(conversation_id, {"state_version": 1})

    def update_from_generation(self, *, conversation_id: str, state: dict, generation: dict, user_message: str) -> None:
        state["state_version"] = int(state.get("state_version") or 1) + 1
        state["last_intent"] = str(((generation.get("query_understanding") or {}).get("intent") or ""))
        self._states[conversation_id] = state

    def save_to_db(self, conversation_id: str, state: dict) -> None:
        self._states[conversation_id] = state


class TestChatApiContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if _BACKEND_IMPORT_ERROR is not None:
            missing = _missing_backend_deps_from_exc(_BACKEND_IMPORT_ERROR)
            raise unittest.SkipTest(_format_skip_message(missing))

    def setUp(self) -> None:
        self._orig_require = chat_service.conversation_service.require_owned_conversation
        self._orig_save = chat_service.message_service.save_message
        self._orig_touch = chat_service.conversation_service.touch_conversation
        self._orig_app_env = os.environ.get("APP_ENV")
        self._orig_chat_debug_errors = os.environ.get("CHAT_DEBUG_ERRORS")
        os.environ["APP_ENV"] = "test"
        os.environ.pop("CHAT_DEBUG_ERRORS", None)
        chat_service.conversation_service.require_owned_conversation = lambda *_args, **_kwargs: None
        chat_service.message_service.save_message = lambda *_args, **_kwargs: None
        chat_service.conversation_service.touch_conversation = lambda *_args, **_kwargs: None

    def tearDown(self) -> None:
        chat_service.conversation_service.require_owned_conversation = self._orig_require
        chat_service.message_service.save_message = self._orig_save
        chat_service.conversation_service.touch_conversation = self._orig_touch
        if self._orig_app_env is None:
            os.environ.pop("APP_ENV", None)
        else:
            os.environ["APP_ENV"] = self._orig_app_env
        if self._orig_chat_debug_errors is None:
            os.environ.pop("CHAT_DEBUG_ERRORS", None)
        else:
            os.environ["CHAT_DEBUG_ERRORS"] = self._orig_chat_debug_errors

    def test_report19_contract_includes_displayed_and_stage_timings(self) -> None:
        def _fake_run_generation(**_kwargs):
            return {
                "answer": "ok",
                "generation_time_seconds": 1.8,
                "generation_mode": "deterministic_doc_scoped_abnormal_results",
                "validation": {
                    "validation_status": "pass",
                    "warnings": ["missing_conclusion", "missing_conclusion"],
                    "errors": [],
                },
                "quality_report": {"final_status": "warning"},
                "query_understanding": {"intent": "doc_scoped_summary", "requested_doc_ids": ["report_19"]},
                "sources": [
                    {
                        "doc_id": "report_19",
                        "filename": "report (19).pdf",
                        "page": 1,
                        "row": 1,
                        "label": "report (19).pdf — page 1, ligne 1",
                        "viewer_url": "/viewer/pdf?doc_id=report_19&page=1",
                    }
                ],
                "displayed_evidences": [],
                "debug": {
                    "selected_route": "doc_scoped_abnormal_results",
                    "included_rows": [
                        {
                            "doc_id": "report_19",
                            "document_name": "report (19).pdf",
                            "page": 1,
                            "row": 1,
                            "analyte": "INSULINE",
                            "current_value": "23,00",
                            "unit": "uU/mL",
                            "reference": "4 à 20 µIU/mL",
                            "technical_status_code": "above_reference",
                            "source_label": "report (19).pdf — page 1, ligne 1",
                            "source_excerpt": "INSULINE 23,00 uU/mL 4 à 20 µIU/mL",
                        },
                        {
                            "doc_id": "report_19",
                            "document_name": "report (19).pdf",
                            "page": 1,
                            "row": 2,
                            "analyte": "T4 LIBRE",
                            "current_value": "22,00",
                            "unit": "pmol/l",
                            "reference": "9,01 à 19,05 pmol/l",
                            "technical_status_code": "above_reference",
                            "source_label": "report (19).pdf — page 1, ligne 2",
                            "source_excerpt": "T4 LIBRE 22,00 pmol/l 9,01 à 19,05 pmol/l",
                        },
                    ],
                },
            }

        response = chat_service.process_chat(
            payload=ChatRequest(conversation_id="conv_1", message="q"),
            current_user={"id": "u1"},
            state_service=_FakeStateService(),
            run_generation=_fake_run_generation,
            logger=logging.getLogger("test.chat.contract"),
        )
        self.assertEqual(response.validation_status, "pass")
        self.assertEqual(len(response.displayed_evidences), 2)
        self.assertEqual(response.debug["validation"]["warnings"], ["missing_conclusion"])
        self.assertEqual((response.debug or {}).get("debug_contract_version"), "v2")
        timings = response.debug.get("stage_timings_ms") or {}
        for key in [
            "query_understanding_ms",
            "routing_ms",
            "retrieval_ms",
            "evidence_build_ms",
            "llm_writer_ms",
            "validation_ms",
            "repair_ms",
            "fallback_ms",
            "total_ms",
        ]:
            self.assertIn(key, timings)

    def test_report16_contract_uses_preview_when_needed(self) -> None:
        def _fake_run_generation(**_kwargs):
            return {
                "answer": "ok",
                "generation_time_seconds": 0.5,
                "generation_mode": "deterministic_doc_scoped_abnormal_results",
                "validation": {"validation_status": "pass", "warnings": [], "errors": []},
                "quality_report": {"final_status": "pass"},
                "query_understanding": {"intent": "doc_scoped_abnormal_results", "requested_doc_ids": ["report_16"]},
                "sources": [],
                "displayed_evidences": [],
                "debug": {
                    "evidence_rows_preview": [
                        {"doc_id": "report_16", "document_name": "report (16).pdf", "page": 1, "row": i, "analyte": f"A{i}", "current_value": str(i), "unit": "u", "reference": "r", "technical_status_code": "above_reference", "source_label": f"s{i}", "source_excerpt": f"x{i}"}
                        for i in range(1, 6)
                    ]
                },
            }

        response = chat_service.process_chat(
            payload=ChatRequest(conversation_id="conv_2", message="q"),
            current_user={"id": "u1"},
            state_service=_FakeStateService(),
            run_generation=_fake_run_generation,
            logger=logging.getLogger("test.chat.contract"),
        )
        self.assertEqual(response.validation_status, "pass")
        self.assertEqual(len(response.displayed_evidences), 5)
        self.assertIsInstance((response.debug or {}).get("stage_timings_ms"), dict)

    def test_chat_contract_exposes_repaired_writer_without_fallback_flags(self) -> None:
        def _fake_run_generation(**_kwargs):
            return {
                "answer": "Synthèse réparée valide.\n\nSource : report_12, pages 1-2.",
                "generation_time_seconds": 0.9,
                "generation_mode": "hybrid_structured_llm_writer",
                "validation": {"validation_status": "pass", "warnings": [], "errors": []},
                "quality_report": {"final_status": "pass"},
                "query_understanding": {"intent": "doc_scoped_biological_summary", "requested_doc_ids": ["report_12"]},
                "sources": [],
                "displayed_evidences": [],
                "final_answer_source": "llm_writer_repaired",
                "fallback_reason": None,
                "llm_repair_attempted": True,
                "llm_repair_status": "passed",
                "llm_quality_escalation_used": True,
                "llm_quality_escalation_reason": "doc_scoped_biological_summary_strict_repair",
                "quality_final_status": "pass",
                "synthesis_quality_reason": None,
                "llm_quality_gate": {"pass": False, "reasons": ["summary_too_poor_for_available_facts"], "score": 0.75, "threshold": 0.85},
                "final_answer_quality_gate": {"pass": True, "reasons": [], "score": 1.0, "threshold": 0.85},
                "debug": {
                    "final_answer_source": "llm_writer_repaired",
                    "fallback_reason": None,
                    "llm_repair_status": "passed",
                    "llm_quality_escalation_used": True,
                    "llm_quality_escalation_reason": "doc_scoped_biological_summary_strict_repair",
                    "quality_final_status": "pass",
                    "synthesis_quality_reason": None,
                },
            }

        response = chat_service.process_chat(
            payload=ChatRequest(conversation_id="conv_3", message="q"),
            current_user={"id": "u1"},
            state_service=_FakeStateService(),
            run_generation=_fake_run_generation,
            logger=logging.getLogger("test.chat.contract"),
        )
        self.assertEqual(response.final_answer_source, "llm_writer_repaired")
        self.assertIsNone(response.fallback_reason)
        self.assertEqual(response.llm_repair_status, "passed")
        self.assertEqual(response.quality_final_status, "pass")
        self.assertEqual((response.debug or {}).get("final_answer_source"), "llm_writer_repaired")
        self.assertIsNone((response.debug or {}).get("fallback_reason"))

    def test_llm_model_override_is_applied_only_in_devtest(self) -> None:
        captured: list[dict] = []

        def _fake_run_generation(**kwargs):
            captured.append(dict(kwargs))
            return {
                "answer": "ok",
                "generation_time_seconds": 0.1,
                "generation_mode": "hybrid_structured_llm_writer",
                "provider": "ollama",
                "model": kwargs.get("model"),
                "validation": {"validation_status": "pass", "warnings": [], "errors": []},
                "quality_report": {"final_status": "pass"},
                "query_understanding": {"intent": "doc_scoped_summary", "requested_doc_ids": ["report_24"]},
                "sources": [],
                "displayed_evidences": [],
                "debug": {
                    "generation_writer": "llm_writer",
                    "ollama_model": kwargs.get("model"),
                },
            }

        old_app_env = os.environ.get("APP_ENV")
        old_debug = os.environ.get("CHAT_DEBUG_ERRORS")
        try:
            os.environ["APP_ENV"] = "test"
            os.environ.pop("CHAT_DEBUG_ERRORS", None)
            response = chat_service.process_chat(
                payload=ChatRequest(conversation_id="conv_dev", message="q", llm_model_override="qwen2.5:7b-instruct"),
                current_user={"id": "u1"},
                state_service=_FakeStateService(),
                run_generation=_fake_run_generation,
                logger=logging.getLogger("test.chat.contract"),
            )
            self.assertEqual(captured[-1]["model"], "qwen2.5:7b-instruct")
            self.assertEqual((response.debug or {}).get("llm_model_requested"), "qwen2.5:7b-instruct")
            self.assertEqual((response.debug or {}).get("llm_model_effective"), "qwen2.5:7b-instruct")
            self.assertTrue(bool((response.debug or {}).get("llm_model_override_applied")))

            os.environ["APP_ENV"] = "prod"
            os.environ.pop("CHAT_DEBUG_ERRORS", None)
            response_prod = chat_service.process_chat(
                payload=ChatRequest(conversation_id="conv_prod", message="q", llm_model_override="qwen2.5:7b-instruct"),
                current_user={"id": "u1"},
                state_service=_FakeStateService(),
                run_generation=_fake_run_generation,
                logger=logging.getLogger("test.chat.contract"),
            )
            self.assertNotEqual(captured[-1]["model"], "qwen2.5:7b-instruct")
            self.assertIsNone((response_prod.debug or {}).get("llm_model_override_applied"))
            self.assertIsNone((response_prod.debug or {}).get("llm_model_override_rejected"))
        finally:
            if old_app_env is None:
                os.environ.pop("APP_ENV", None)
            else:
                os.environ["APP_ENV"] = old_app_env
            if old_debug is None:
                os.environ.pop("CHAT_DEBUG_ERRORS", None)
            else:
                os.environ["CHAT_DEBUG_ERRORS"] = old_debug

    def test_safe_ui_model_override_is_applied_in_prod(self) -> None:
        captured: list[dict] = []

        def _fake_run_generation(**kwargs):
            captured.append(dict(kwargs))
            return {
                "answer": "ok",
                "generation_time_seconds": 0.1,
                "generation_mode": "hybrid_structured_llm_writer",
                "provider": kwargs.get("provider"),
                "model": kwargs.get("model"),
                "validation": {"validation_status": "pass", "warnings": [], "errors": []},
                "quality_report": {"final_status": "pass"},
                "query_understanding": {"intent": "doc_scoped_summary", "requested_doc_ids": ["report_24"]},
                "sources": [],
                "displayed_evidences": [],
                "debug": {
                    "generation_writer": "llm_writer",
                    "ollama_model": kwargs.get("model"),
                    "llm_provider": kwargs.get("provider"),
                },
            }

        old_app_env = os.environ.get("APP_ENV")
        old_debug = os.environ.get("CHAT_DEBUG_ERRORS")
        try:
            os.environ["APP_ENV"] = "prod"
            os.environ.pop("CHAT_DEBUG_ERRORS", None)
            response = chat_service.process_chat(
                payload=ChatRequest(
                    conversation_id="conv_prod_ui",
                    message="q",
                    llm_provider_override="ollama",
                    llm_model_override="llama3.2:latest",
                ),
                current_user={"id": "u1"},
                state_service=_FakeStateService(),
                run_generation=_fake_run_generation,
                logger=logging.getLogger("test.chat.contract"),
            )
            self.assertEqual(captured[-1]["provider"], "ollama")
            self.assertEqual(captured[-1]["model"], "llama3.2:latest")
            self.assertEqual(response.provider, "ollama")
            self.assertEqual(response.model, "llama3.2:latest")
        finally:
            if old_app_env is None:
                os.environ.pop("APP_ENV", None)
            else:
                os.environ["APP_ENV"] = old_app_env
            if old_debug is None:
                os.environ.pop("CHAT_DEBUG_ERRORS", None)
            else:
                os.environ["CHAT_DEBUG_ERRORS"] = old_debug

    def test_debug_requested_analytes_are_canonical_with_labels_preserved(self) -> None:
        def _fake_run_generation(**_kwargs):
            return {
                "answer": "ok",
                "generation_time_seconds": 0.1,
                "generation_mode": "deterministic_single_analyte_lookup",
                "validation": {"validation_status": "pass", "warnings": [], "errors": []},
                "quality_report": {"final_status": "pass"},
                "query_understanding": {
                    "intent": "doc_scoped_results",
                    "requested_doc_ids": ["report_29"],
                    "requested_analytes": ["créat", "TSH"],
                },
                "sources": [],
                "displayed_evidences": [],
                "debug": {},
            }

        response = chat_service.process_chat(
            payload=ChatRequest(conversation_id="conv_canon", message="q"),
            current_user={"id": "u1"},
            state_service=_FakeStateService(),
            run_generation=_fake_run_generation,
            logger=logging.getLogger("test.chat.contract"),
        )
        dbg = response.debug or {}
        self.assertEqual(dbg.get("debug_contract_version"), "v2")
        self.assertEqual(dbg.get("requested_analytes"), ["creatinine", "tsh"])
        self.assertEqual(dbg.get("requested_analyte_labels"), ["créat", "TSH"])

    def test_debug_requested_analytes_fallback_from_generation_debug(self) -> None:
        def _fake_run_generation(**_kwargs):
            return {
                "answer": "ok",
                "generation_time_seconds": 0.1,
                "generation_mode": "deterministic_no_evidence_response",
                "validation": {"validation_status": "warning", "warnings": [], "errors": []},
                "quality_report": {"final_status": "warning"},
                "query_understanding": {
                    "intent": "cohort_search",
                    "requested_doc_ids": [],
                    "requested_analytes": [],
                },
                "detected_analytes": ["uricémie"],
                "sources": [],
                "displayed_evidences": [],
                "debug": {
                    "detected_analytes": ["uricémie"],
                    "answerability_not_found_analytes": ["acide_urique"],
                },
            }

        response = chat_service.process_chat(
            payload=ChatRequest(conversation_id="conv_fallback", message="q"),
            current_user={"id": "u1"},
            state_service=_FakeStateService(),
            run_generation=_fake_run_generation,
            logger=logging.getLogger("test.chat.contract"),
        )
        dbg = response.debug or {}
        self.assertEqual(dbg.get("debug_contract_version"), "v2")
        self.assertEqual(dbg.get("requested_analytes"), ["acide_urique"])
        self.assertIn("uricémie", list(dbg.get("requested_analyte_labels") or []))
        raw = dbg.get("raw_debug") if isinstance(dbg.get("raw_debug"), dict) else {}
        self.assertEqual(raw.get("requested_analytes"), ["acide_urique"])
        self.assertIn("uricémie", list(raw.get("requested_analyte_labels") or []))

    def test_debug_contract_version_present_in_controlled_error_fallback(self) -> None:
        def _boom(**_kwargs):
            raise RuntimeError("synthetic failure")

        response = chat_service.process_chat(
            payload=ChatRequest(conversation_id="conv_err", message="q"),
            current_user={"id": "u1"},
            state_service=_FakeStateService(),
            run_generation=_boom,
            logger=logging.getLogger("test.chat.contract"),
        )
        dbg = response.debug or {}
        self.assertEqual(dbg.get("debug_contract_version"), "v2")
        self.assertEqual(response.generation_mode, "controlled_error_fallback")

    def test_debug_advanced_fields_are_hidden_in_prod(self) -> None:
        def _fake_run_generation(**_kwargs):
            return {
                "answer": "ok",
                "generation_time_seconds": 0.1,
                "generation_mode": "deterministic_single_analyte_lookup",
                "validation": {"validation_status": "pass", "warnings": [], "errors": []},
                "quality_report": {"final_status": "pass"},
                "query_understanding": {
                    "intent": "doc_scoped_results",
                    "requested_doc_ids": ["report_29"],
                    "requested_analytes": ["créat"],
                },
                "sources": [],
                "displayed_evidences": [],
                "debug": {
                    "selected_route": "doc_scoped_single_analyte_status",
                    "route_reason": "heuristic",
                },
            }

        old_app_env = os.environ.get("APP_ENV")
        try:
            os.environ["APP_ENV"] = "production"
            response = chat_service.process_chat(
                payload=ChatRequest(conversation_id="conv_prod_debug", message="q"),
                current_user={"id": "u1"},
                state_service=_FakeStateService(),
                run_generation=_fake_run_generation,
                logger=logging.getLogger("test.chat.contract"),
            )
        finally:
            if old_app_env is None:
                os.environ.pop("APP_ENV", None)
            else:
                os.environ["APP_ENV"] = old_app_env

        dbg = response.debug or {}
        self.assertEqual(dbg.get("debug_contract_version"), "v2")
        self.assertIsNone(dbg.get("raw_debug"))
        self.assertIsNone(dbg.get("query_understanding"))
        self.assertIsNone(dbg.get("requested_analytes"))
        self.assertIsNone(dbg.get("selected_route"))

    def test_structured_request_summary_log_is_emitted(self) -> None:
        def _fake_run_generation(**_kwargs):
            return {
                "answer": "ok",
                "generation_time_seconds": 0.123,
                "generation_mode": "deterministic_single_analyte_lookup",
                "validation": {"validation_status": "warning", "warnings": ["llm_hallucination"], "errors": []},
                "quality_report": {"final_status": "warning"},
                "query_understanding": {
                    "intent": "doc_scoped_results",
                    "requested_doc_ids": ["report_29"],
                    "requested_analytes": ["créat"],
                },
                "sources": [],
                "displayed_evidences": [],
                "debug": {
                    "selected_route": "doc_scoped_single_analyte_status",
                    "answerability_status": "answerable_alias",
                    "specialized_fallback_kind": "partial_answer",
                    "generation_writer": "professional_fallback",
                    "llm_route_class": "deterministic_only",
                    "llm_writer_attempted": True,
                    "llm_writer_accepted": False,
                    "fallback_reason": "llm_validation_failed",
                },
            }

        logger = logging.getLogger("test.chat.contract.structured")
        with patch.object(logger, "info") as info_mock:
            chat_service.process_chat(
                payload=ChatRequest(conversation_id="conv_structured_log", message="q"),
                current_user={"id": "u1"},
                state_service=_FakeStateService(),
                run_generation=_fake_run_generation,
                logger=logger,
            )

        joined_calls = [" ".join(str(arg) for arg in call.args) for call in info_mock.call_args_list]
        summary_call = next((entry for entry in joined_calls if "chat_request_summary" in entry), "")
        self.assertIn('"intent": "doc_scoped_results"', summary_call)
        self.assertIn('"selected_route": "doc_scoped_single_analyte_status"', summary_call)
        self.assertIn('"generation_writer": "professional_fallback"', summary_call)
        self.assertIn('"answerability_status": "answerable_alias"', summary_call)
        self.assertIn('"fallback_kind": "partial_answer"', summary_call)
        self.assertIn('"llm_route_class": "deterministic_only"', summary_call)
        self.assertIn('"llm_attempt_rate": 1.0', summary_call)
        self.assertIn('"llm_accept_rate": 0.0', summary_call)
        self.assertIn('"llm_reject_rate": 1.0', summary_call)
        self.assertIn('"llm_timeout_rate": 0.0', summary_call)
        self.assertIn('"repair_attempt_rate": 0.0', summary_call)
        self.assertIn('"repair_success_rate": 0.0', summary_call)
        self.assertIn('"fallback_after_llm_rate": 1.0', summary_call)
        self.assertIn('"hallucination_rejection_rate": 1.0', summary_call)
        self.assertIn('"llm_writer_ms": 0.0', summary_call)
        self.assertIn('"contract_violation_count": 0', summary_call)
        self.assertIn('"validation_hard_gate_reason": null', summary_call)
        self.assertIn('"validation_hard_gate_reason_count": 0', summary_call)
        self.assertIn('"response_time_ms": 123.0', summary_call)
        self.assertIn('"failure_signals": ["hallucination"]', summary_call)

    def test_advanced_debug_exposes_llm_observability_fields(self) -> None:
        def _fake_run_generation(**_kwargs):
            return {
                "answer": "ok",
                "generation_time_seconds": 0.1,
                "generation_mode": "hybrid_structured_llm_writer",
                "validation": {"validation_status": "pass", "warnings": [], "errors": []},
                "quality_report": {"final_status": "pass"},
                "query_understanding": {
                    "intent": "doc_scoped_results",
                    "requested_doc_ids": ["report_16"],
                    "requested_analytes": ["TSH"],
                },
                "sources": [],
                "displayed_evidences": [],
                "debug": {
                    "selected_route": "doc_scoped_medical_interpretation_guarded",
                    "generation_writer": "llm_writer",
                    "selected_policy": "hybrid_controlled",
                    "policy_level": "hybrid_controlled",
                    "llm_writer_allowed": True,
                    "llm_writer_attempted": True,
                    "llm_writer_accepted": True,
                    "llm_prompt_policy_version": "v2",
                },
            }

        old_app_env = os.environ.get("APP_ENV")
        try:
            os.environ["APP_ENV"] = "test"
            response = chat_service.process_chat(
                payload=ChatRequest(conversation_id="conv_llm_debug", message="q"),
                current_user={"id": "u1"},
                state_service=_FakeStateService(),
                run_generation=_fake_run_generation,
                logger=logging.getLogger("test.chat.contract"),
            )
        finally:
            if old_app_env is None:
                os.environ.pop("APP_ENV", None)
            else:
                os.environ["APP_ENV"] = old_app_env

        dbg = response.debug or {}
        self.assertEqual(dbg.get("llm_route_class"), "llm_allowed")
        self.assertEqual(dbg.get("llm_prompt_policy_version"), "v2")
        self.assertEqual(dbg.get("llm_attempt_rate"), 1.0)
        self.assertEqual(dbg.get("llm_accept_rate"), 1.0)
        self.assertEqual(dbg.get("llm_reject_rate"), 0.0)
        self.assertEqual(dbg.get("llm_timeout_rate"), 0.0)
        self.assertEqual(dbg.get("repair_attempt_rate"), 0.0)
        self.assertEqual(dbg.get("repair_success_rate"), 0.0)
        self.assertEqual(dbg.get("fallback_after_llm_rate"), 0.0)
        self.assertEqual(dbg.get("hallucination_rejection_rate"), 0.0)
        self.assertEqual(dbg.get("contract_violation_count"), 0)
        self.assertEqual(dbg.get("validation_hard_gate_reason_count"), 0)
        raw = dbg.get("raw_debug") if isinstance(dbg.get("raw_debug"), dict) else {}
        self.assertEqual(raw.get("llm_route_class"), "llm_allowed")
        self.assertEqual(raw.get("llm_prompt_policy_version"), "v2")
        self.assertEqual(raw.get("llm_reject_rate"), 0.0)
        self.assertEqual(raw.get("llm_timeout_rate"), 0.0)
        self.assertEqual(raw.get("repair_attempt_rate"), 0.0)
        self.assertEqual(raw.get("repair_success_rate"), 0.0)
        self.assertEqual(raw.get("hallucination_rejection_rate"), 0.0)
        self.assertEqual(raw.get("contract_violation_count"), 0)
        self.assertEqual(raw.get("validation_hard_gate_reason_count"), 0)

    def test_contract_violation_resets_llm_attempt_metrics_in_debug_and_logs(self) -> None:
        def _fake_run_generation(**_kwargs):
            return {
                "answer": "Réponse de fallback déterministe.",
                "generation_time_seconds": 0.05,
                "generation_mode": "writer_contract_violation_fallback",
                "validation": {"validation_status": "pass", "warnings": [], "errors": []},
                "quality_report": {"final_status": "pass"},
                "query_understanding": {
                    "intent": "doc_scoped_biological_summary",
                    "requested_doc_ids": ["report_12"],
                    "requested_analytes": ["CRP"],
                },
                "sources": [],
                "displayed_evidences": [],
                "debug": {
                    "selected_route": "doc_scoped_biological_summary",
                    "generation_writer": "professional_fallback",
                    "selected_policy": "hybrid_controlled",
                    "policy_level": "hybrid_controlled",
                    "llm_writer_allowed": True,
                    "llm_writer_attempted": True,
                    "llm_writer_accepted": False,
                    "llm_prompt_policy_version": "v2",
                    "fallback_reason": "writer_evidence_contract_violation",
                    "contract_violation": ["scope_incoherent", "results_locked_empty"],
                },
            }

        old_app_env = os.environ.get("APP_ENV")
        logger = logging.getLogger("test.chat.contract.contract_violation")
        try:
            os.environ["APP_ENV"] = "test"
            with patch.object(logger, "info") as info_mock:
                response = chat_service.process_chat(
                    payload=ChatRequest(conversation_id="conv_contract_violation", message="q"),
                    current_user={"id": "u1"},
                    state_service=_FakeStateService(),
                    run_generation=_fake_run_generation,
                    logger=logger,
                )
        finally:
            if old_app_env is None:
                os.environ.pop("APP_ENV", None)
            else:
                os.environ["APP_ENV"] = old_app_env

        dbg = response.debug or {}
        self.assertEqual(dbg.get("contract_violation_count"), 2)
        self.assertEqual(dbg.get("llm_attempt_rate"), 0.0)
        self.assertEqual(dbg.get("llm_accept_rate"), 0.0)
        self.assertEqual(dbg.get("llm_reject_rate"), 0.0)
        self.assertEqual(dbg.get("llm_timeout_rate"), 0.0)
        self.assertEqual(dbg.get("repair_attempt_rate"), 0.0)
        self.assertEqual(dbg.get("repair_success_rate"), 0.0)
        self.assertEqual(dbg.get("fallback_after_llm_rate"), 0.0)
        raw = dbg.get("raw_debug") if isinstance(dbg.get("raw_debug"), dict) else {}
        self.assertEqual(raw.get("contract_violation_count"), 2)
        self.assertEqual(raw.get("llm_attempt_rate"), 0.0)
        self.assertEqual(raw.get("llm_reject_rate"), 0.0)
        self.assertEqual(raw.get("validation_hard_gate_reason_count"), 0)
        joined_calls = [" ".join(str(arg) for arg in call.args) for call in info_mock.call_args_list]
        summary_call = next((entry for entry in joined_calls if "chat_request_summary" in entry), "")
        self.assertIn('"contract_violation_count": 2', summary_call)
        self.assertIn('"llm_attempt_rate": 0.0', summary_call)

    def test_hard_gate_reason_is_exposed_in_debug_and_summary(self) -> None:
        def _fake_run_generation(**_kwargs):
            return {
                "answer": "Fallback déterministe.",
                "generation_time_seconds": 0.04,
                "generation_mode": "deterministic_safety_fallback_after_llm_validation_failure",
                "validation": {"validation_status": "warning", "warnings": [], "errors": []},
                "quality_report": {"final_status": "warning"},
                "query_understanding": {
                    "intent": "doc_scoped_results",
                    "requested_doc_ids": ["report_12"],
                    "requested_analytes": ["crp"],
                },
                "sources": [],
                "displayed_evidences": [],
                "debug": {
                    "selected_route": "doc_scoped_results",
                    "hard_gate_triggered": True,
                    "hard_gate_errors": ["source_mismatch", "raw_internal_source"],
                },
            }

        old_app_env = os.environ.get("APP_ENV")
        logger = logging.getLogger("test.chat.contract.hard_gate_reason")
        try:
            os.environ["APP_ENV"] = "test"
            with patch.object(logger, "info") as info_mock:
                response = chat_service.process_chat(
                    payload=ChatRequest(conversation_id="conv_hg", message="q"),
                    current_user={"id": "u1"},
                    state_service=_FakeStateService(),
                    run_generation=_fake_run_generation,
                    logger=logger,
                )
        finally:
            if old_app_env is None:
                os.environ.pop("APP_ENV", None)
            else:
                os.environ["APP_ENV"] = old_app_env

        dbg = response.debug or {}
        self.assertEqual(dbg.get("validation_hard_gate_reason"), "source_mismatch")
        self.assertEqual(dbg.get("validation_hard_gate_reason_count"), 2)
        self.assertEqual(dbg.get("validation_hard_gate_reasons"), ["source_mismatch", "raw_internal_source"])
        raw = dbg.get("raw_debug") if isinstance(dbg.get("raw_debug"), dict) else {}
        self.assertEqual(raw.get("validation_hard_gate_reason"), "source_mismatch")
        self.assertEqual(raw.get("validation_hard_gate_reason_count"), 2)
        joined_calls = [" ".join(str(arg) for arg in call.args) for call in info_mock.call_args_list]
        summary_call = next((entry for entry in joined_calls if "chat_request_summary" in entry), "")
        self.assertIn('"validation_hard_gate_reason": "source_mismatch"', summary_call)
        self.assertIn('"validation_hard_gate_reason_count": 2', summary_call)

    def test_llm_timeout_and_repair_rates_are_exposed(self) -> None:
        def _fake_run_generation(**_kwargs):
            return {
                "answer": "Fallback déterministe.",
                "generation_time_seconds": 0.06,
                "generation_mode": "deterministic_safety_fallback_after_llm_validation_failure",
                "validation": {"validation_status": "warning", "warnings": [], "errors": []},
                "quality_report": {"final_status": "warning"},
                "query_understanding": {
                    "intent": "doc_scoped_biological_summary",
                    "requested_doc_ids": ["report_16"],
                    "requested_analytes": ["tsh"],
                },
                "sources": [],
                "displayed_evidences": [],
                "debug": {
                    "selected_route": "doc_scoped_biological_summary",
                    "generation_writer": "professional_fallback",
                    "llm_writer_attempted": True,
                    "llm_writer_accepted": False,
                    "fallback_reason": "llm_timeout",
                    "llm_repair_attempted": True,
                },
            }

        response = chat_service.process_chat(
            payload=ChatRequest(conversation_id="conv_timeout_rates", message="q"),
            current_user={"id": "u1"},
            state_service=_FakeStateService(),
            run_generation=_fake_run_generation,
            logger=logging.getLogger("test.chat.contract.timeout_rates"),
        )

        dbg = response.debug or {}
        self.assertEqual(dbg.get("llm_attempt_rate"), 1.0)
        self.assertEqual(dbg.get("llm_reject_rate"), 1.0)
        self.assertEqual(dbg.get("llm_timeout_rate"), 1.0)
        self.assertEqual(dbg.get("repair_attempt_rate"), 1.0)
        self.assertEqual(dbg.get("repair_success_rate"), 0.0)

    def test_debug_exposes_fallback_decision_path(self) -> None:
        def _fake_run_generation(**_kwargs):
            return {
                "answer": "Clarification requise.",
                "generation_time_seconds": 0.02,
                "generation_mode": "deterministic_no_evidence_response",
                "validation": {"validation_status": "warning", "warnings": [], "errors": []},
                "quality_report": {"final_status": "warning"},
                "query_understanding": {
                    "intent": "doc_scoped_results",
                    "requested_doc_ids": [],
                    "requested_analytes": [],
                },
                "sources": [],
                "displayed_evidences": [],
                "debug": {
                    "selected_route": "doc_scoped_results",
                    "generation_writer": "deterministic_clarification",
                    "generation_mode_before_fallback": "hybrid_structured_llm_writer",
                    "fallback_decision_path": [
                        "planner:selected_plan:doc_scoped_results",
                        "answerability:ambiguous",
                        "fallback_stage:answerability_gate",
                        "fallback_reason:ambiguous_scope",
                        "specialized_fallback:ambiguous_document_scope",
                    ],
                },
            }

        old_app_env = os.environ.get("APP_ENV")
        try:
            os.environ["APP_ENV"] = "test"
            response = chat_service.process_chat(
                payload=ChatRequest(conversation_id="conv_fallback_path", message="q"),
                current_user={"id": "u1"},
                state_service=_FakeStateService(),
                run_generation=_fake_run_generation,
                logger=logging.getLogger("test.chat.contract.fallback_path"),
            )
        finally:
            if old_app_env is None:
                os.environ.pop("APP_ENV", None)
            else:
                os.environ["APP_ENV"] = old_app_env

        dbg = response.debug or {}
        self.assertEqual(
            dbg.get("fallback_decision_path"),
            [
                "planner:selected_plan:doc_scoped_results",
                "answerability:ambiguous",
                "fallback_stage:answerability_gate",
                "fallback_reason:ambiguous_scope",
                "specialized_fallback:ambiguous_document_scope",
            ],
        )
        self.assertEqual(str(dbg.get("generation_mode_before_fallback") or ""), "hybrid_structured_llm_writer")

    def test_integration_run_generation_process_chat_observability_contract(self) -> None:
        from scripts.generation.generate_answer import run_generation as real_run_generation

        logger = logging.getLogger("test.chat.contract.integration_observability")
        with patch.object(logger, "info") as info_mock:
            response = chat_service.process_chat(
                payload=ChatRequest(
                    conversation_id="conv_obs_contract_integration",
                    message="Dans le report 29, la créatinine est-elle normale, basse ou élevée ?",
                ),
                current_user={"id": "u1"},
                state_service=_FakeStateService(),
                run_generation=real_run_generation,
                logger=logger,
            )

        dbg = response.debug or {}
        required_debug_keys = [
            "llm_expected",
            "llm_writer_attempted",
            "llm_writer_accepted",
            "llm_skipped_reason",
            "generation_mode_before_fallback",
            "fallback_decision_path",
        ]
        for key in required_debug_keys:
            self.assertIn(key, dbg)

        payload: dict[str, object] | None = None
        for call in info_mock.call_args_list:
            args = list(call.args)
            if not args:
                continue
            if str(args[0]).strip() != "chat_request_summary %s":
                continue
            if len(args) < 2:
                continue
            try:
                payload = json.loads(str(args[1]))
            except Exception:
                payload = None
            break
        self.assertIsNotNone(payload)
        payload = dict(payload or {})

        required_summary_keys = [
            "intent",
            "selected_route",
            "generation_mode",
            "generation_writer",
            "validation_status",
            "quality_final_status",
            "answerability_status",
            "fallback_kind",
            "response_time_ms",
            "llm_attempt_rate",
            "llm_accept_rate",
            "llm_reject_rate",
            "llm_timeout_rate",
            "repair_attempt_rate",
            "repair_success_rate",
            "fallback_after_llm_rate",
            "hallucination_rejection_rate",
            "avg_llm_writer_ms",
            "p95_llm_writer_ms",
        ]
        for key in required_summary_keys:
            self.assertIn(key, payload)


if __name__ == "__main__":
    unittest.main()
