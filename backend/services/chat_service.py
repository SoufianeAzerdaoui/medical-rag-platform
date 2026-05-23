from __future__ import annotations

import logging
import os
import traceback
from typing import Any, Callable
from uuid import uuid4

from fastapi import HTTPException

from backend.models import ChatRequest, ChatResponse, SourceItem
from backend.services import conversation_service, message_service
from backend.services.conversation_state_store import ConversationStateService, transformable_context
from scripts.generation.source_normalization import dedup_normalized_sources
from scripts.generation.model_settings import (
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_NUM_CTX,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_TIMEOUT,
)


def _debug_or_devtest_enabled() -> bool:
    if str(os.getenv("CHAT_DEBUG_ERRORS", "")).strip().lower() in {"1", "true", "yes", "on"}:
        return True
    return str(os.getenv("APP_ENV", "")).strip().lower() in {"dev", "development", "test", "local"}


def to_source_items(result: dict[str, Any]) -> list[SourceItem]:
    items: list[SourceItem] = []

    structured = dedup_normalized_sources(list(result.get("sources") or []))
    if structured:
        for idx, src in enumerate(structured, start=1):
            doc_id = str(src.get("doc_id") or "").strip() or None
            filename = str(src.get("filename") or src.get("source_pdf") or "").strip() or None
            page = src.get("page")
            row = src.get("line", src.get("row"))
            label = str(src.get("label") or "").strip() or None
            url = str(src.get("url") or "").strip() or None
            viewer_url = str(src.get("viewer_url") or "").strip() or None
            warning = None if url else "source_pdf_unavailable"
            items.append(
                SourceItem(
                    id=f"source-{idx}",
                    documentName=filename or "Document médical",
                    documentId=doc_id,
                    page=page if isinstance(page, int) else None,
                    section=None,
                    excerpt=label,
                    score=None,
                    type="pdf_source",
                    warning=warning,
                    doc_id=doc_id,
                    filename=filename,
                    row=row if isinstance(row, int) else None,
                    label=label,
                    url=url,
                    viewer_url=viewer_url,
                )
            )
        return items

    evidences = list(result.get("displayed_evidences") or result.get("evidence_pack") or [])
    for idx, ev in enumerate(evidences, start=1):
        page_num: int | None = None
        try:
            page_value = ev.get("page_number")
            page_num = int(page_value) if page_value is not None else None
        except Exception:
            page_num = None
        items.append(
            SourceItem(
                id=f"legacy-{idx}",
                documentName=str(ev.get("source_pdf") or ev.get("doc_id") or "Document médical"),
                documentId=str(ev.get("doc_id") or "") or None,
                page=page_num,
                section=str(ev.get("section") or "") or None,
                excerpt=str(ev.get("text_excerpt") or ev.get("source") or "")[:600] or None,
                score=float(ev.get("final_score")) if ev.get("final_score") not in (None, "") else None,
                type=str(ev.get("chunk_type") or "") or None,
                warning=None,
            )
        )
    return items


def confidence_from_result(result: dict[str, Any]) -> float | None:
    validation = result.get("validation") or {}
    status = str(validation.get("validation_status") or "").lower()
    if status == "pass":
        return 0.9
    if status == "warning":
        return 0.7
    if status == "fail":
        return 0.4
    return None


def _dedup_keep_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = str(item or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _stage_timings_from_generation(generation: dict[str, Any]) -> dict[str, float | None]:
    base = {
        "query_understanding_ms": 0.0,
        "routing_ms": 0.0,
        "retrieval_ms": 0.0,
        "evidence_build_ms": 0.0,
        "llm_writer_ms": 0.0,
        "validation_ms": 0.0,
        "repair_ms": 0.0,
        "fallback_ms": 0.0,
        "total_ms": 0.0,
    }
    debug = generation.get("debug") if isinstance(generation.get("debug"), dict) else {}
    raw = debug.get("stage_timings_ms") if isinstance(debug.get("stage_timings_ms"), dict) else {}
    for key in base:
        value = raw.get(key)
        try:
            base[key] = float(value) if value is not None else 0.0
        except Exception:
            base[key] = 0.0
    if (not base.get("total_ms")) and generation.get("generation_time_seconds") is not None:
        try:
            base["total_ms"] = round(float(generation.get("generation_time_seconds") or 0.0) * 1000.0, 3)
        except Exception:
            base["total_ms"] = 0.0
    return base


def _normalize_displayed_evidence_row(row: dict[str, Any]) -> dict[str, Any]:
    doc_id = str(row.get("doc_id") or row.get("documentId") or "").strip()
    document_name = str(row.get("document_name") or row.get("documentName") or row.get("filename") or row.get("source_pdf") or "").strip()
    page = row.get("page_number", row.get("page"))
    row_no = row.get("row_index", row.get("row", row.get("line")))
    analyte = str(row.get("analyte") or "").strip()
    value = str(row.get("current_value") or row.get("value_raw") or row.get("value") or "").strip()
    unit = str(row.get("unit") or "").strip()
    reference = str(row.get("reference") or row.get("reference_range") or "").strip()
    status = str(row.get("technical_status_code") or row.get("interpretation_status") or row.get("status") or "").strip()
    source_label = str(
        row.get("source_label")
        or row.get("label")
        or row.get("source")
        or row.get("text_excerpt")
        or ""
    ).strip()
    source_excerpt = str(row.get("source_excerpt") or row.get("text_excerpt") or source_label).strip()
    out: dict[str, Any] = {
        "doc_id": doc_id or None,
        "document_name": document_name or None,
        "page": int(page) if isinstance(page, int) else page,
        "row": int(row_no) if isinstance(row_no, int) else row_no,
        "analyte": analyte or None,
        "value": value or None,
        "unit": unit or None,
        "reference": reference or None,
        "status": status or None,
        "source_label": source_label or None,
        "source_excerpt": source_excerpt or None,
    }
    return out


def _resolve_displayed_evidences(generation: dict[str, Any]) -> list[dict[str, Any]]:
    direct = generation.get("displayed_evidences")
    if isinstance(direct, list) and direct:
        return [_normalize_displayed_evidence_row(r if isinstance(r, dict) else {}) for r in direct]
    debug = generation.get("debug") if isinstance(generation.get("debug"), dict) else {}
    included = debug.get("included_rows")
    if isinstance(included, list) and included:
        return [_normalize_displayed_evidence_row(r if isinstance(r, dict) else {}) for r in included]
    preview = debug.get("evidence_rows_preview")
    if isinstance(preview, list) and preview:
        return [_normalize_displayed_evidence_row(r if isinstance(r, dict) else {}) for r in preview]
    return []


def process_chat(
    *,
    payload: ChatRequest,
    current_user: dict[str, Any],
    state_service: ConversationStateService,
    run_generation: Callable[..., dict[str, Any]],
    logger: logging.Logger,
) -> ChatResponse:
    request_id = str(uuid4())
    conversation_id = payload.resolved_conversation_id()
    user_id = str(current_user["id"])

    try:
        if not conversation_id:
            raise HTTPException(status_code=400, detail="conversation_id requis")

        if state_service.cleanup_expired() > 0:
            logger.info("state_store_action=cleanup_expired request_id=%s", request_id)

        conversation_service.require_owned_conversation(conversation_id, user_id)

        state_service.hydrate_from_db_if_present(conversation_id)
        state = state_service.load(conversation_id)
        state_version_before = int(state.get("state_version") or 1)

        previous_intent = str(state.get("last_intent") or "").strip().lower() or "none"
        transformable = transformable_context(state)
        qualitative_context = (
            state.get("last_qualitative_evidence_pack")
            if isinstance(state.get("last_qualitative_evidence_pack"), dict)
            else None
        )
        has_last_evidence_pack = isinstance(state.get("last_evidence_pack"), dict) and bool(state.get("last_evidence_pack"))
        has_last_transformable = isinstance(transformable, dict) and bool(transformable)
        has_last_qualitative = isinstance(qualitative_context, dict) and bool(qualitative_context)
        last_data_context_type_before = str(state.get("last_data_context_type") or "none")
        query_text = payload.message.lower()
        visualization_request_detected = any(
            token in query_text
            for token in ["radar", "chart", "graphique", "graphe", "visualisation", "courbe", "line graph", "bar chart", "diagramme"]
        )

        logger.info(
            "qa_state_pre request_id=%s conversation_id=%s previous_intent=%s has_last_evidence_pack=%s has_last_transformable_evidence_pack=%s transformable_context_used=%s visualization_request_detected=%s",
            request_id,
            conversation_id,
            previous_intent,
            bool(has_last_evidence_pack),
            bool(has_last_transformable),
            bool(has_last_transformable),
            bool(visualization_request_detected),
        )

        query = f"{payload.message} doc_id {payload.document_id}" if payload.document_id else payload.message
        requested_model_override = str(payload.llm_model_override or "").strip() or None
        allow_model_override = _debug_or_devtest_enabled()
        model_for_request = requested_model_override if (allow_model_override and requested_model_override) else DEFAULT_LLM_MODEL
        generation = run_generation(
            query=query,
            top_k=5,
            mode="hybrid",
            provider=DEFAULT_LLM_PROVIDER,
            model=model_for_request,
            temperature=DEFAULT_LLM_TEMPERATURE,
            num_ctx=DEFAULT_LLM_NUM_CTX,
            max_tokens=max(600, DEFAULT_LLM_MAX_TOKENS),
            timeout=DEFAULT_LLM_TIMEOUT,
            index_dir="data/indexes",
            collection="medical_chunks",
            previous_structured_evidence_pack=transformable,
            previous_displayed_evidence_pack=(
                state.get("last_displayed_evidence_pack")
                if isinstance(state.get("last_displayed_evidence_pack"), dict)
                else None
            ),
            previous_displayed_context=(
                state.get("last_displayed_context")
                if isinstance(state.get("last_displayed_context"), dict)
                else None
            ),
            previous_context_intent=str(state.get("last_intent") or ""),
            previous_data_context_intent=str(state.get("last_data_context_intent") or ""),
            previous_data_context_type=str(state.get("last_data_context_type") or ""),
            previous_doc_scope=(
                list((state.get("last_doc_scope") or {}).get("doc_ids") or [])
                if isinstance(state.get("last_doc_scope"), dict)
                else []
            ),
            previous_qualitative_evidence_pack=(
                state.get("last_qualitative_evidence_pack")
                if isinstance(state.get("last_qualitative_evidence_pack"), dict)
                else None
            ),
            previous_has_patient_inventory=bool(state.get("last_patient_inventory")),
            previous_patient_inventory=(
                state.get("last_patient_inventory")
                if isinstance(state.get("last_patient_inventory"), list)
                else None
            ),
            recent_style_history=state.get("recent_style_history") or [],
        )

        state_service.update_from_generation(
            conversation_id=conversation_id,
            state=state,
            generation=generation,
            user_message=payload.message,
        )
        new_state = state_service.load(conversation_id)
        state_service.save_to_db(conversation_id, new_state)

        message_service.save_message(conversation_id, "user", payload.message)
        state_version_after = int(new_state.get("state_version") or state_version_before)
        current_intent = str(((generation.get("query_understanding") or {}).get("intent") or "")).strip().lower() or "unknown"
        visualization = generation.get("visualization") if isinstance(generation.get("visualization"), dict) else None
        rendered_type = str((visualization or {}).get("rendered_type") or "").strip().lower() or None
        requested_type = str((visualization or {}).get("requested_type") or "").strip().lower() or None

        logger.info(
            "qa_state_post request_id=%s conversation_id=%s current_intent=%s previous_intent=%s has_last_evidence_pack=%s has_last_transformable_evidence_pack=%s transformable_context_used=%s visualization_request_detected=%s retrieval_skipped_due_to_no_transformable_context=%s response_has_visualization=%s response_rendered_type=%s response_requested_type=%s",
            request_id,
            conversation_id,
            current_intent,
            previous_intent,
            bool(isinstance(new_state.get("last_evidence_pack"), dict) and new_state.get("last_evidence_pack")),
            bool(isinstance(new_state.get("last_transformable_evidence_pack"), dict) and new_state.get("last_transformable_evidence_pack")),
            bool(has_last_transformable),
            bool(visualization_request_detected),
            bool(((generation.get("debug") or {}).get("retrieval_skipped_due_to_no_transformable_context"))),
            bool(visualization),
            rendered_type,
            requested_type,
        )
        logger.info(
            "qa_state_versions request_id=%s conversation_id=%s state_version_before=%s state_version_after=%s last_data_context_type_before=%s last_data_context_type_after=%s requested_analytes=%s requested_date_iso=%s latest_report=%s requested_clickable_sources=%s previous_doc_scope_used=%s resolved_doc_scope=%s has_transformable_pack=%s has_qualitative_pack=%s sources_count=%s chart_generated=%s smalltalk_blocked=%s validator_status=%s",
            request_id,
            conversation_id,
            state_version_before,
            state_version_after,
            last_data_context_type_before,
            str(new_state.get("last_data_context_type") or "none"),
            list((generation.get("query_understanding") or {}).get("requested_analytes") or []),
            (generation.get("query_understanding") or {}).get("requested_date_iso"),
            bool((generation.get("query_understanding") or {}).get("latest_report")),
            bool((generation.get("query_understanding") or {}).get("source_clickable_requested")),
            bool(((generation.get("debug") or {}).get("context_resolution") or {}).get("reuse_doc_scope")),
            ((generation.get("debug") or {}).get("context_resolution") or {}).get("effective_doc_scope"),
            bool(isinstance(new_state.get("last_transformable_evidence_pack"), dict) and new_state.get("last_transformable_evidence_pack")),
            bool(isinstance(new_state.get("last_qualitative_evidence_pack"), dict) and new_state.get("last_qualitative_evidence_pack")),
            len(list(generation.get("sources") or [])),
            bool(visualization),
            (
                "bonjour ! je suis prêt" not in str(generation.get("answer") or "").lower()
                and "comment puis-je vous aider" not in str(generation.get("answer") or "").lower()
            ),
            str((generation.get("validation") or {}).get("validation_status") or ""),
        )

        answer = str(generation.get("answer") or "").strip()
        if not answer:
            answer = "Aucune réponse générée. Cette réponse ne remplace pas l'avis médical."

        message_service.save_message(conversation_id, "assistant", answer)
        conversation_service.touch_conversation(conversation_id)

        sources = to_source_items(generation)
        document_ids = sorted({item.documentId for item in sources if item.documentId})
        validation = generation.get("validation") if isinstance(generation.get("validation"), dict) else {}
        validation_warnings = _dedup_keep_order([str(w) for w in (validation.get("warnings") or []) if str(w).strip()])
        validation_errors = _dedup_keep_order([str(e) for e in (validation.get("errors") or []) if str(e).strip()])
        validation["warnings"] = validation_warnings
        validation["errors"] = validation_errors
        qu = generation.get("query_understanding") if isinstance(generation.get("query_understanding"), dict) else {}
        displayed_evidences = _resolve_displayed_evidences(generation)
        stage_timings_ms = _stage_timings_from_generation(generation)
        generation_debug = generation.get("debug") if isinstance(generation.get("debug"), dict) else {}
        llm_provider = str(generation.get("provider") or "") or None
        llm_model_requested = str(generation.get("model") or "") or None
        llm_model_effective = str(generation_debug.get("ollama_model") or "") or None
        model_verified = None
        if llm_model_requested or llm_model_effective:
            model_verified = llm_model_requested == llm_model_effective
        response_debug = {
            "intent": str(qu.get("intent") or "") or None,
            "safety_intent": str(qu.get("safety_intent") or "") or None,
            "intent_arbitration": (qu.get("intent_arbitration") if isinstance(qu.get("intent_arbitration"), dict) else None),
            "technical_condition": str(qu.get("technical_condition") or "") or None,
            "requested_doc_ids": list(qu.get("requested_doc_ids") or []),
            "requested_analytes": list(qu.get("requested_analytes") or []),
            "query_understanding": qu if qu else None,
            "selected_route": str((generation_debug.get("selected_route") or "")) or None,
            "route_reason": str((generation_debug.get("route_reason") or "")) or None,
            "evidence_rows_preview": list((generation_debug.get("evidence_rows_preview") or [])),
            "included_rows": list((generation_debug.get("included_rows") or [])),
            "excluded_rows": list((generation_debug.get("excluded_rows") or [])),
            "fallback_reason": str((generation_debug.get("fallback_reason") or "")) or None,
            "generation_mode_before_fallback": str((generation_debug.get("generation_mode_before_fallback") or "")) or None,
            "llm_provider": llm_provider,
            "llm_model_override_requested": requested_model_override,
            "llm_model_override_allowed": allow_model_override,
            "llm_model_override_applied": bool(requested_model_override and allow_model_override),
            "llm_model_override_rejected": bool(requested_model_override and not allow_model_override),
            "llm_model_requested": llm_model_requested,
            "llm_model_effective": llm_model_effective,
            "ollama_model": str((generation_debug.get("ollama_model") or "")) or None,
            "model_verified": model_verified,
            "llm_writer_attempted": bool(generation_debug.get("llm_writer_allowed") or generation_debug.get("llm_writer_used")),
            "llm_writer_accepted": str((generation_debug.get("generation_writer") or "")).strip().lower() == "llm_writer",
            "hard_gate_rejected": bool(generation_debug.get("hard_gate_triggered")),
            "repair_attempted": bool(generation_debug.get("llm_repair_attempted") or generation_debug.get("llm_candidate_repair_used")),
            "repair_success": bool(generation_debug.get("llm_repair_attempted") or generation_debug.get("llm_candidate_repair_used"))
            and not bool(generation_debug.get("hard_gate_triggered"))
            and str((generation_debug.get("generation_writer") or "")).strip().lower() == "llm_writer",
            "validation": {
                "status": str(validation.get("validation_status") or "") or None,
                "errors": validation_errors,
                "warnings": validation_warnings,
                "unsupported_claims": list(validation.get("unsupported_claims") or []),
            },
            "stage_timings_ms": stage_timings_ms,
            "raw_debug": generation_debug,
        }

        return ChatResponse(
            conversation_id=conversation_id,
            answer=answer,
            sources=sources,
            confidence=confidence_from_result(generation),
            document_ids=document_ids,
            response_time=float(generation.get("generation_time_seconds") or 0.0),
            quality_report=(generation.get("quality_report") if isinstance(generation.get("quality_report"), dict) else None),
            validation_status=str((generation.get("validation") or {}).get("validation_status") or "") or None,
            generation_mode=str(generation.get("generation_mode") or "") or None,
            generation_writer=str(((generation.get("debug") or {}).get("generation_writer") or "")) or None,
            visualization=(generation.get("visualization") if isinstance(generation.get("visualization"), dict) else None),
            chart_data=(generation.get("chart_data") if isinstance(generation.get("chart_data"), dict) else None),
            patients=generation.get("patients"),
            inventory_view=(generation.get("inventory_view") if isinstance(generation.get("inventory_view"), dict) else None),
            displayed_evidences=displayed_evidences,
            debug=response_debug,
        )

    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        logger.exception(
            "chat_generation_failed request_id=%s provider=%s intent=unknown query=%r error=%s\n%s",
            request_id,
            "ollama",
            payload.message,
            str(exc),
            traceback.format_exc(),
        )
        safe_answer = (
            "Une erreur interne a empêché la génération complète de la réponse. "
            "Les données indexées restent disponibles ; veuillez relancer la demande ou simplifier la formulation."
        )
        expose_error_detail = str(os.getenv("CHAT_DEBUG_ERRORS", "")).strip().lower() in {"1", "true", "yes", "on"}
        if not expose_error_detail:
            app_env = str(os.getenv("APP_ENV", "")).strip().lower()
            expose_error_detail = app_env in {"dev", "development", "test", "local"}
        debug_payload: dict[str, Any] = {
            "intent": None,
            "validation": {
                "status": "warning",
                "errors": ["controlled_error_fallback"],
                "warnings": [],
                "unsupported_claims": [],
            },
        }
        if expose_error_detail:
            debug_payload["controlled_error_detail"] = str(exc)
            debug_payload["controlled_error_traceback"] = traceback.format_exc()
        return ChatResponse(
            conversation_id=conversation_id,
            answer=safe_answer,
            sources=[],
            confidence=0.0,
            document_ids=[],
            response_time=0.0,
            quality_report=None,
            validation_status="warning",
            generation_mode="controlled_error_fallback",
            generation_writer="professional_fallback",
            visualization=None,
            chart_data=None,
            patients=None,
            inventory_view=None,
            debug=debug_payload,
        )
