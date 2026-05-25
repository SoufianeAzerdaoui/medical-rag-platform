from __future__ import annotations

import logging
import os
import unittest
from typing import Iterable

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


if __name__ == "__main__":
    unittest.main()
