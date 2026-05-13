from __future__ import annotations

import re
from typing import Any


TRANSFORMABLE_INTENTS = {
    "doc_scoped_results",
    "cohort_search",
    "global_patient_lookup",
    "multi_doc_comparison",
    "multi_doc_presence_diff",
    "immunoanalysis_summary",
    "toxicology_summary",
    "doc_scoped_summary",
    "response_transform",
}

NON_TRANSFORMABLE_INTENTS = {
    "patient_inventory",
    "patient_inventory_count",
    "small_talk",
    "identity_question",
    "help_question",
    "capability_question",
}


def extract_intent(result: dict[str, Any]) -> str:
    qu = result.get("query_understanding") or {}
    generation_mode = str(result.get("generation_mode") or "").strip().lower()
    structured_pack = result.get("structured_evidence_pack")
    if generation_mode == "deterministic_response_transform" and (not isinstance(structured_pack, dict) or not structured_pack):
        return "response_transform_no_context"
    if isinstance(qu, dict):
        return str(qu.get("intent") or "").strip().lower()
    return ""


def evidence_pack_is_transformable(pack: Any) -> bool:
    if not isinstance(pack, dict):
        return False
    evidences = pack.get("evidences")
    if not isinstance(evidences, list):
        evidences = pack.get("results")
    if not isinstance(evidences, list) or not evidences:
        return False
    for ev in evidences:
        if not isinstance(ev, dict):
            continue
        analyte = str(ev.get("analyte") or ev.get("parameter") or "").strip()
        if not analyte:
            continue
        analyte_norm = analyte.strip().lower()
        if analyte_norm in {"commentaire", "comment", "observation", "note"}:
            continue
        result_kind = str(ev.get("result_kind") or "").strip().lower()
        if result_kind in {"qualitative", "comment"}:
            continue
        value_numeric = ev.get("value_numeric", ev.get("value"))
        current_value = str(ev.get("current_value") or ev.get("value_raw") or ev.get("value") or "").strip()
        if isinstance(value_numeric, str):
            try:
                value_numeric = float(value_numeric.replace(",", ".").strip())
            except Exception:
                value_numeric = None
        if value_numeric in (None, ""):
            raw_norm = current_value.replace(",", ".").strip()
            if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", raw_norm):
                continue
        has_source = bool(
            str(ev.get("source_url") or ev.get("viewer_url") or "").strip()
            or str(ev.get("source") or "").strip()
            or str(ev.get("doc_id") or "").strip()
            or str(ev.get("source_id") or "").strip()
        )
        if has_source:
            return True
    return False


def get_transformable_context(state: dict[str, Any]) -> dict[str, Any] | None:
    pack = state.get("last_transformable_evidence_pack")
    return pack if isinstance(pack, dict) and pack else None


def update_conversation_state(
    *,
    state_store: dict[str, dict[str, Any]],
    chat_id: str,
    state: dict[str, Any],
    generation: dict[str, Any],
    user_message: str,
) -> None:
    new_state = state_store.setdefault(chat_id, {})
    intent = extract_intent(generation)
    evidence_pack = generation.get("structured_evidence_pack")
    patients = generation.get("patients")
    sources = generation.get("sources")

    new_state["last_intent"] = intent
    new_state["last_user_question"] = user_message
    new_state["last_answer"] = generation.get("answer")
    new_state["last_query_understanding"] = generation.get("query_understanding")
    new_state["last_rendered_rows"] = generation.get("displayed_evidences")
    new_state["last_sources"] = sources if sources is not None else state.get("last_sources")
    new_state["last_visualization"] = generation.get("visualization")
    qu = generation.get("query_understanding") if isinstance(generation.get("query_understanding"), dict) else {}
    qu_doc_ids = list(qu.get("requested_doc_ids") or [])
    if qu_doc_ids:
        new_state["last_doc_scope"] = qu_doc_ids

    if patients is not None:
        new_state["last_patient_inventory"] = patients

    if evidence_pack is not None:
        new_state["last_evidence_pack"] = evidence_pack

    if intent in TRANSFORMABLE_INTENTS and evidence_pack_is_transformable(evidence_pack):
        new_state["last_transformable_evidence_pack"] = evidence_pack
    elif intent in NON_TRANSFORMABLE_INTENTS or intent.startswith("deterministic_patient_"):
        new_state["last_transformable_evidence_pack"] = None
    elif evidence_pack is None and intent not in TRANSFORMABLE_INTENTS:
        new_state["last_transformable_evidence_pack"] = None

    # Preserve a separate "last data context" that should survive technical no-context turns.
    if intent == "patient_inventory" and patients:
        new_state["last_data_context_intent"] = "patient_inventory"
        new_state["last_data_context_type"] = "patient_inventory"
        new_state["last_data_context_payload"] = {"patients_count": len(patients)}
    elif intent == "patient_inventory_count":
        new_state["last_data_context_intent"] = "patient_inventory_count"
        new_state["last_data_context_type"] = "patient_count"
        new_state["last_data_context_payload"] = {"count_text": str(generation.get("answer") or "")[:200]}
    elif intent in {"comment_without_measured_value"} or str(qu.get("requested_context_type") or "").strip() == "medical_qualitative_comment":
        new_state["last_data_context_intent"] = intent
        new_state["last_data_context_type"] = "medical_qualitative_comment"
        new_state["last_data_context_payload"] = {"answer_preview": str(generation.get("answer") or "")[:240]}
        new_state["last_transformable_evidence_pack"] = None
        if isinstance(evidence_pack, dict) and (evidence_pack.get("comment_text") or evidence_pack.get("evidences")):
            new_state["last_qualitative_evidence_pack"] = evidence_pack
    elif intent in TRANSFORMABLE_INTENTS and evidence_pack_is_transformable(evidence_pack):
        new_state["last_data_context_intent"] = intent
        new_state["last_data_context_type"] = "biological_results"
        new_state["last_data_context_payload"] = {"evidence_count": len((evidence_pack or {}).get("evidences") or [])}

    style_hist = list(state.get("recent_style_history") or [])
    style_entry = generation.get("style_memory_entry")
    if isinstance(style_entry, dict):
        style_hist.append(style_entry)
    new_state["recent_style_history"] = style_hist[-20:]
