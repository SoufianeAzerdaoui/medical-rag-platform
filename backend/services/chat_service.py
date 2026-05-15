from __future__ import annotations

import logging
import traceback
from typing import Any, Callable
from uuid import uuid4

from fastapi import HTTPException

from backend.models import ChatRequest, ChatResponse, SourceItem
from backend.services import conversation_service, message_service
from backend.services.conversation_state_store import ConversationStateService, transformable_context
from scripts.generation.source_normalization import dedup_normalized_sources


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
        generation = run_generation(
            query=query,
            top_k=5,
            mode="hybrid",
            provider="ollama",
            model="qwen3:4b",
            temperature=0.0,
            num_ctx=4096,
            max_tokens=600,
            timeout=120,
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
        )
