from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any, Literal, TypedDict


DataContextType = Literal[
    "none",
    "patient_inventory",
    "patient_count",
    "biological_numeric_results",
    "medical_qualitative_comment",
]


class DocScope(TypedDict, total=False):
    doc_ids: list[str]
    source_pdf: str | None
    report_date: str | None
    sample_id: str | None
    patient_id: str | None
    patient_name: str | None


class ConversationState(TypedDict, total=False):
    conversation_id: str
    state_version: int
    last_intent: str | None
    last_data_context_type: DataContextType
    last_patient_inventory: list[dict[str, Any]] | None
    last_evidence_pack: dict[str, Any] | None
    last_displayed_evidence_pack: dict[str, Any] | None
    last_transformable_evidence_pack: dict[str, Any] | None
    last_qualitative_evidence_pack: dict[str, Any] | None
    last_doc_scope: DocScope | None
    last_visualization: dict[str, Any] | None
    last_inventory_view: dict[str, Any] | None
    last_qualitative_view: dict[str, Any] | None
    recent_turns: list[dict[str, Any]]
    created_at: str
    updated_at: str
    # Compatibility fields kept during migration window.
    last_data_context_intent: str | None
    last_data_context_payload: dict[str, Any] | None
    last_answer: str | None
    last_user_question: str | None
    last_sources: list[dict[str, Any]] | None
    last_query_understanding: dict[str, Any] | None
    last_rendered_rows: list[dict[str, Any]] | None
    recent_style_history: list[dict[str, Any]]


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

TECHNICAL_INTENTS = {
    "response_transform_no_context",
    "visualization_recommendation",
    "inventory_visualization_render",
    "qualitative_comment_render",
    "error",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_qualitative_evidence_pack(pack: dict[str, Any] | None) -> bool:
    if not isinstance(pack, dict) or not pack:
        return False
    if pack.get("comment_text") or pack.get("raw_comment_text") or pack.get("display_comment_text"):
        return True
    rows = pack.get("evidences")
    if not isinstance(rows, list):
        rows = pack.get("results")
    if not isinstance(rows, list) or not rows:
        return False
    for row in rows:
        if not isinstance(row, dict):
            continue
        kind = str(row.get("result_kind") or row.get("kind") or "").strip().lower()
        analyte = str(row.get("analyte") or row.get("parameter") or row.get("subject") or "").strip().lower()
        value = str(row.get("value") or row.get("current_value") or row.get("comment_text") or "").strip().lower()
        if kind in {"comment", "qualitative"}:
            return True
        if analyte in {"commentaire", "comment", "observation", "note"}:
            return True
        if "commentaire" in value or "valeur seuil" in value or "attention" in value:
            return True
    return False


def _empty_state(conversation_id: str) -> ConversationState:
    now = _now_iso()
    return {
        "conversation_id": conversation_id,
        "state_version": 1,
        "last_intent": None,
        "last_data_context_type": "none",
        "last_patient_inventory": None,
        "last_evidence_pack": None,
        "last_displayed_evidence_pack": None,
        "last_transformable_evidence_pack": None,
        "last_qualitative_evidence_pack": None,
        "last_doc_scope": None,
        "last_visualization": None,
        "last_inventory_view": None,
        "last_qualitative_view": None,
        "recent_turns": [],
        "created_at": now,
        "updated_at": now,
        "last_data_context_intent": None,
        "last_data_context_payload": None,
        "last_answer": None,
        "last_user_question": None,
        "last_sources": None,
        "last_query_understanding": None,
        "last_rendered_rows": None,
        "recent_style_history": [],
    }


def migrate_conversation_state(state: dict[str, Any] | None, *, conversation_id: str) -> ConversationState:
    if not isinstance(state, dict) or not state:
        return _empty_state(conversation_id)

    out = _empty_state(conversation_id)
    out.update(state)
    out["conversation_id"] = str(state.get("conversation_id") or conversation_id)
    out["state_version"] = int(state.get("state_version") or 1)
    if out.get("created_at") is None:
        out["created_at"] = _now_iso()
    out["updated_at"] = _now_iso()

    # Backward compatibility for old list-based doc_scope.
    lds = out.get("last_doc_scope")
    if isinstance(lds, list):
        out["last_doc_scope"] = {"doc_ids": [str(d).strip() for d in lds if str(d).strip()]}
    elif isinstance(lds, dict):
        out["last_doc_scope"] = {
            "doc_ids": [str(d).strip() for d in (lds.get("doc_ids") or []) if str(d).strip()],
            "source_pdf": lds.get("source_pdf"),
            "report_date": lds.get("report_date"),
            "sample_id": lds.get("sample_id"),
            "patient_id": lds.get("patient_id"),
            "patient_name": lds.get("patient_name"),
        }
    else:
        out["last_doc_scope"] = None

    if out.get("last_data_context_type") not in {
        "none",
        "patient_inventory",
        "patient_count",
        "biological_numeric_results",
        "medical_qualitative_comment",
    }:
        # Migrate legacy values.
        legacy = str(out.get("last_data_context_type") or "").strip().lower()
        if legacy in {"biological_results"}:
            out["last_data_context_type"] = "biological_numeric_results"
        else:
            out["last_data_context_type"] = "none"
    return out


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


def get_qualitative_context(state: dict[str, Any]) -> dict[str, Any] | None:
    pack = state.get("last_qualitative_evidence_pack")
    return pack if isinstance(pack, dict) and pack else None


def _build_doc_scope(*, qu: dict[str, Any], evidence_pack: dict[str, Any] | None) -> DocScope | None:
    doc_ids = [str(d).strip() for d in (qu.get("requested_doc_ids") or []) if str(d).strip()]
    if not doc_ids and isinstance(evidence_pack, dict):
        doc_ids = sorted(
            {
                str(ev.get("doc_id") or "").strip()
                for ev in (evidence_pack.get("evidences") or evidence_pack.get("results") or [])
                if str(ev.get("doc_id") or "").strip()
            }
        )
    if not doc_ids:
        return None
    sample_id = None
    source_pdf = None
    patient_id = None
    patient_name = None
    report_date = qu.get("requested_date_iso")
    if isinstance(evidence_pack, dict):
        for ev in (evidence_pack.get("evidences") or evidence_pack.get("results") or []):
            if not isinstance(ev, dict):
                continue
            sample_id = sample_id or ev.get("sample_id") or ev.get("sample_token")
            source_pdf = source_pdf or ev.get("source_pdf") or ev.get("filename")
            patient_id = patient_id or ev.get("patient_id") or ev.get("patient_token")
            patient_name = patient_name or ev.get("patient_name")
            report_date = report_date or ev.get("report_date")
            if sample_id and source_pdf and patient_id and patient_name and report_date:
                break
    return {
        "doc_ids": doc_ids,
        "source_pdf": source_pdf,
        "report_date": report_date,
        "sample_id": sample_id,
        "patient_id": patient_id,
        "patient_name": patient_name,
    }


def _infer_data_context_type(*, intent: str, evidence_pack: dict[str, Any] | None, qualitative_pack: dict[str, Any] | None, patient_inventory: list[dict[str, Any]] | None, qu: dict[str, Any]) -> DataContextType | None:
    if intent == "patient_inventory" and patient_inventory:
        return "patient_inventory"
    if intent in {"patient_inventory_count", "patient_count"}:
        return "patient_count"
    if intent in {"comment_without_measured_value"} or str(qu.get("requested_context_type") or "").strip() == "medical_qualitative_comment":
        if _is_qualitative_evidence_pack(qualitative_pack):
            return "medical_qualitative_comment"
        if _is_qualitative_evidence_pack(evidence_pack):
            return "medical_qualitative_comment"
        # Keep qualitative context classification when the user explicitly asked
        # for comment/note context even if extraction produced no structured row.
        return "medical_qualitative_comment"
    if evidence_pack_is_transformable(evidence_pack):
        return "biological_numeric_results"
    return None


def update_conversation_state_reducer(
    state: ConversationState,
    *,
    intent: str,
    answer: str | None = None,
    evidence_pack: dict[str, Any] | None = None,
    qualitative_evidence_pack: dict[str, Any] | None = None,
    patient_inventory: list[dict[str, Any]] | None = None,
    doc_scope: DocScope | None = None,
    visualization: dict[str, Any] | None = None,
    inventory_view: dict[str, Any] | None = None,
    qualitative_view: dict[str, Any] | None = None,
    sources: list[dict[str, Any]] | None = None,
    user_message: str | None = None,
    query_understanding: dict[str, Any] | None = None,
    style_entry: dict[str, Any] | None = None,
) -> ConversationState:
    out = dict(state)
    qu = query_understanding or {}
    out["state_version"] = int(out.get("state_version") or 1) + 1
    out["updated_at"] = _now_iso()
    out["last_intent"] = intent or out.get("last_intent")
    out["last_answer"] = answer if answer is not None else out.get("last_answer")
    out["last_user_question"] = user_message if user_message is not None else out.get("last_user_question")
    out["last_query_understanding"] = qu or out.get("last_query_understanding")
    out["last_sources"] = sources if sources is not None else out.get("last_sources")
    out["last_visualization"] = visualization if visualization is not None else out.get("last_visualization")
    if inventory_view is not None:
        out["last_inventory_view"] = inventory_view
    if qualitative_view is not None:
        out["last_qualitative_view"] = qualitative_view

    if patient_inventory is not None:
        out["last_patient_inventory"] = patient_inventory
    if evidence_pack is not None:
        out["last_evidence_pack"] = evidence_pack
        out["last_displayed_evidence_pack"] = evidence_pack

    if doc_scope is None:
        doc_scope = _build_doc_scope(qu=qu, evidence_pack=evidence_pack)
    if doc_scope:
        out["last_doc_scope"] = doc_scope

    if evidence_pack_is_transformable(evidence_pack) and intent not in NON_TRANSFORMABLE_INTENTS:
        out["last_transformable_evidence_pack"] = evidence_pack
    elif intent in NON_TRANSFORMABLE_INTENTS or intent.startswith("deterministic_patient_"):
        out["last_transformable_evidence_pack"] = None
    elif intent in {"comment_without_measured_value", "qualitative_comment_render"}:
        out["last_transformable_evidence_pack"] = None
    elif evidence_pack is None and intent not in TRANSFORMABLE_INTENTS:
        # Technical intents should not resurrect old transformable contexts.
        if intent in TECHNICAL_INTENTS:
            pass
        else:
            out["last_transformable_evidence_pack"] = None

    if qualitative_evidence_pack is None and _is_qualitative_evidence_pack(evidence_pack):
        qualitative_evidence_pack = evidence_pack
    if _is_qualitative_evidence_pack(qualitative_evidence_pack):
        out["last_qualitative_evidence_pack"] = qualitative_evidence_pack
        out["last_transformable_evidence_pack"] = None

    context_type = _infer_data_context_type(
        intent=intent,
        evidence_pack=evidence_pack,
        qualitative_pack=qualitative_evidence_pack,
        patient_inventory=patient_inventory,
        qu=qu,
    )
    if context_type:
        out["last_data_context_type"] = context_type
        out["last_data_context_intent"] = intent
        if context_type == "patient_inventory":
            out["last_data_context_payload"] = {"patients_count": len(patient_inventory or [])}
            out["last_qualitative_evidence_pack"] = None
        elif context_type == "patient_count":
            out["last_data_context_payload"] = {"count_text": str(answer or "")[:200]}
            out["last_qualitative_evidence_pack"] = None
        elif context_type == "biological_numeric_results":
            out["last_data_context_payload"] = {
                "evidence_count": len((evidence_pack or {}).get("evidences") or []),
            }
            out["last_qualitative_evidence_pack"] = None
        elif context_type == "medical_qualitative_comment":
            out["last_data_context_payload"] = {"answer_preview": str(answer or "")[:240]}
    elif out.get("last_data_context_type") not in {
        "none",
        "patient_inventory",
        "patient_count",
        "biological_numeric_results",
        "medical_qualitative_comment",
    }:
        out["last_data_context_type"] = "none"

    recent_turns = list(out.get("recent_turns") or [])
    if user_message:
        recent_turns.append({"role": "user", "content": str(user_message)[:600], "intent": intent})
    if answer:
        recent_turns.append({"role": "assistant", "content": str(answer)[:900], "intent": intent})
    out["recent_turns"] = recent_turns[-20:]

    style_hist = list(out.get("recent_style_history") or [])
    if isinstance(style_entry, dict):
        style_hist.append(style_entry)
    out["recent_style_history"] = style_hist[-20:]
    return out  # type: ignore[return-value]


def update_conversation_state(
    *,
    state_store: dict[str, dict[str, Any]],
    chat_id: str,
    state: dict[str, Any],
    generation: dict[str, Any],
    user_message: str,
) -> None:
    migrated = migrate_conversation_state(state, conversation_id=chat_id)
    intent = extract_intent(generation)
    evidence_pack = generation.get("structured_evidence_pack") if isinstance(generation.get("structured_evidence_pack"), dict) else None
    displayed = generation.get("displayed_evidences") if isinstance(generation.get("displayed_evidences"), list) else []
    # State should reflect what was actually shown to the user (especially follow-up analyte turns).
    if displayed:
        displayed_pack: dict[str, Any] = {
            "question": (evidence_pack or {}).get("question") if isinstance(evidence_pack, dict) else None,
            "intent": (evidence_pack or {}).get("intent") if isinstance(evidence_pack, dict) else intent,
            "output_format": (evidence_pack or {}).get("output_format") if isinstance(evidence_pack, dict) else None,
            "answer_style": (evidence_pack or {}).get("answer_style") if isinstance(evidence_pack, dict) else None,
            "requested_doc_ids": list(((evidence_pack or {}).get("requested_doc_ids") or []) if isinstance(evidence_pack, dict) else []),
            "requested_analytes": list(((evidence_pack or {}).get("requested_analytes") or []) if isinstance(evidence_pack, dict) else []),
            "evidences": list(displayed),
            "results": list(displayed),
        }
        evidence_pack = displayed_pack
    patients = generation.get("patients") if isinstance(generation.get("patients"), list) else None
    sources = generation.get("sources") if isinstance(generation.get("sources"), list) else None
    qu = generation.get("query_understanding") if isinstance(generation.get("query_understanding"), dict) else {}
    updated = update_conversation_state_reducer(
        migrated,
        intent=intent,
        answer=str(generation.get("answer") or ""),
        evidence_pack=evidence_pack,
        qualitative_evidence_pack=(evidence_pack if _is_qualitative_evidence_pack(evidence_pack) else None),
        patient_inventory=patients,
        doc_scope=_build_doc_scope(qu=qu, evidence_pack=evidence_pack),
        visualization=generation.get("visualization") if isinstance(generation.get("visualization"), dict) else None,
        inventory_view=generation.get("inventory_view") if isinstance(generation.get("inventory_view"), dict) else None,
        qualitative_view=generation.get("qualitative_view") if isinstance(generation.get("qualitative_view"), dict) else None,
        sources=sources,
        user_message=user_message,
        query_understanding=qu,
        style_entry=generation.get("style_memory_entry") if isinstance(generation.get("style_memory_entry"), dict) else None,
    )
    state_store[chat_id] = updated
