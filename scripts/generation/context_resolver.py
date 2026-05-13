from __future__ import annotations

from typing import Any

from query_understanding import QueryUnderstanding, norm_text


def resolve_context_for_turn(message: str, query_understanding: QueryUnderstanding, state: dict[str, Any]) -> dict[str, Any]:
    qn = norm_text(message or "")
    intent = str(query_understanding.intent or "").strip().lower()
    requested_doc_ids = list(query_understanding.requested_doc_ids or [])
    requested_analytes = list(query_understanding.requested_analytes or [])
    previous_doc_scope = state.get("last_doc_scope") if isinstance(state.get("last_doc_scope"), dict) else None
    data_context_type = str(state.get("last_data_context_type") or "none")
    has_inventory = bool(state.get("last_patient_inventory"))
    has_transformable = isinstance(state.get("last_transformable_evidence_pack"), dict) and bool(state.get("last_transformable_evidence_pack"))
    has_qualitative = isinstance(state.get("last_qualitative_evidence_pack"), dict) and bool(state.get("last_qualitative_evidence_pack"))

    deictic_followup = bool(
        requested_analytes
        and not requested_doc_ids
        and any(token in qn for token in ["et ", "ensuite", "puis", "meme rapport", "même rapport", "ce rapport"])
    )

    reuse_doc_scope = bool(deictic_followup and previous_doc_scope and (previous_doc_scope.get("doc_ids") or []))
    effective_doc_scope = previous_doc_scope if reuse_doc_scope else None
    reuse_patient_inventory = intent == "inventory_visualization_render" and (data_context_type == "patient_inventory" or has_inventory)
    reuse_transformable = intent == "response_transform" and has_transformable
    reuse_qualitative = intent in {"qualitative_comment_render", "visualization_recommendation"} and (
        data_context_type == "medical_qualitative_comment" or has_qualitative
    )
    is_deictic = any(token in qn for token in [" ca", "ça", "ces donnees", "ces données", "affiche", "mets"])
    asks_table = any(token in qn for token in [" table", "table ", "tableau"])
    deictic_table_inventory = bool(is_deictic and asks_table and data_context_type == "patient_inventory")

    should_skip_retrieval = False
    reason = "default_retrieval"
    if intent == "response_transform" and not has_transformable:
        should_skip_retrieval = True
        reason = "response_transform_no_transformable_context"
    elif intent == "inventory_visualization_render" and reuse_patient_inventory:
        should_skip_retrieval = True
        reason = "inventory_render_reuses_previous_inventory"
    elif intent == "qualitative_comment_render":
        should_skip_retrieval = True
        reason = "qualitative_render_reuses_qualitative_pack"
    elif intent == "visualization_recommendation":
        should_skip_retrieval = True
        reason = "recommendation_no_retrieval"
    elif deictic_table_inventory:
        should_skip_retrieval = True
        reason = "inventory_deictic_table_reuse"
    elif reuse_doc_scope:
        reason = "followup_reuse_doc_scope"

    return {
        "requires_retrieval": not should_skip_retrieval,
        "reuse_doc_scope": reuse_doc_scope,
        "reuse_patient_inventory": reuse_patient_inventory,
        "reuse_transformable_pack": reuse_transformable,
        "reuse_qualitative_pack": reuse_qualitative,
        "effective_doc_scope": effective_doc_scope,
        "effective_data_context_type": data_context_type,
        "deictic_table_request": bool(is_deictic and asks_table),
        "should_skip_retrieval": should_skip_retrieval,
        "reason": reason,
    }
