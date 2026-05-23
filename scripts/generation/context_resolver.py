from __future__ import annotations

import re
from typing import Any

from query_understanding import QueryUnderstanding, norm_text


def resolve_deictic_request(message: str, query_understanding: QueryUnderstanding, state: dict[str, Any]) -> dict[str, Any]:
    qn = norm_text(message or "")
    def _has_phrase(phrase: str) -> bool:
        p = norm_text(phrase or "")
        if not p:
            return False
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(p)}(?![a-z0-9])", qn))
    requested_doc_ids = list(query_understanding.requested_doc_ids or [])
    requested_analytes = list(query_understanding.requested_analytes or [])
    explicit_scope = bool(requested_doc_ids or getattr(query_understanding, "requested_date_iso", None))
    last_displayed_context = state.get("last_displayed_context") if isinstance(state.get("last_displayed_context"), dict) else None
    has_inventory = bool(state.get("last_patient_inventory"))
    has_transformable = bool(state.get("last_transformable_evidence_pack")) if isinstance(state.get("last_transformable_evidence_pack"), dict) else False
    has_qualitative = bool(state.get("last_qualitative_evidence_pack")) if isinstance(state.get("last_qualitative_evidence_pack"), dict) else False
    previous_doc_scope = state.get("last_doc_scope") if isinstance(state.get("last_doc_scope"), dict) else None
    last_intent = str(state.get("last_intent") or "").strip().lower()
    prev_doc_ids = [str(d).strip() for d in (previous_doc_scope.get("doc_ids") or [])] if isinstance(previous_doc_scope, dict) else []
    prev_ctx_type = str((last_displayed_context or {}).get("context_type") or state.get("last_data_context_type") or "none").strip().lower()

    is_deictic = any(
        k in qn
        for k in [
            " ca",
            "ça",
            "ce commentaire",
            "ce resultat",
            "ce résultat",
            "cette valeur",
            "ce resultat",
            "ce résultat",
            "ceci",
            "affiche ca",
            "affiche ça",
            "mets ca",
            "mets ça",
            "fais la meme chose",
            "fais la même chose",
            "pareil pour",
            "la meme chose pour",
            "la même chose pour",
        ]
    )
    asks_table = any(
        k in qn
        for k in [
            " tableau",
            "tableau",
            " en table",
            " table ",
            "une table",
            "dans une table",
            " tabl",
            "table",
        ]
    )
    asks_chart = any(k in qn for k in ["graphique", "chart", "courbe", "radar", "bar chart", "line graph", "visualise"])
    asks_summary = any(k in qn for k in ["resume", "résume", "synthese", "synthèse"])
    asks_note = any(_has_phrase(k) for k in ["note interpretative", "note interprétative", "encadre", "encadré"])
    asks_card = any(
        _has_phrase(k)
        for k in [
            "fiche",
            "carte",
            "carte d information",
            "carte d’information",
            "fiche medicale",
            "fiche médicale",
            "format fiche",
            "format carte",
        ]
    )
    asks_source = any(k in qn for k in ["d ou vient", "d'où vient", "source", "quelle page", "quel rapport", "ouvre la source"])
    asks_status = ("hors reference" in qn or "hors de la reference" in qn or "hors référence" in qn) and ("ce resultat" in qn or "ce résultat" in qn or "cette valeur" in qn)
    has_global_scope_phrase = any(
        k in qn
        for k in [
            "sur l ensemble des rapports",
            "sur l’ensemble des rapports",
            "parmi les rapports disponibles",
            "dans tous les rapports",
            "rapports disponibles",
            "rapports indexes",
            "rapports indexés",
            "donnees disponibles",
            "données disponibles",
        ]
    )
    has_correction_marker = any(
        k in qn
        for k in [
            "non",
            "plutot",
            "plutôt",
            "pas ca",
            "pas ça",
            "je voulais",
            "je veux",
        ]
    )
    correction_format_followup = bool(
        asks_table
        and has_correction_marker
        and not explicit_scope
        and not requested_doc_ids
        and not requested_analytes
    )

    same_action_markers = [
        "fais la meme chose pour",
        "fais la même chose pour",
        "la meme chose pour",
        "la même chose pour",
        "pareil pour",
        "meme chose avec",
        "même chose avec",
        "donne pareil pour",
        "montre pareil pour",
        "et pour",
    ]
    same_action_for_subject = bool(requested_analytes and any(m in qn for m in same_action_markers))

    out = {
        "resolved": False,
        "intent": str(query_understanding.intent or "").strip().lower(),
        "context_type": prev_ctx_type or "none",
        "reuse_last_displayed_context": False,
        "reuse_doc_scope": False,
        "effective_doc_scope": {"doc_ids": prev_doc_ids} if prev_doc_ids else None,
        "target_subject": (requested_analytes[0] if requested_analytes else None),
        "render_type": None,
        "skip_retrieval": False,
        "reason": "no_deictic_resolution",
    }

    has_any_context = bool(
        last_displayed_context
        or prev_ctx_type in {"patient_inventory", "biological_numeric_results", "medical_qualitative_comment"}
        or has_inventory
        or has_transformable
        or has_qualitative
        or prev_doc_ids
    )

    # 1) Explicit source follow-up from last displayed context (or equivalent context state).
    if asks_source and not explicit_scope and has_any_context:
        out.update(
            {
                "resolved": True,
                "intent": "source_followup",
                "reuse_last_displayed_context": True,
                "skip_retrieval": True,
                "reason": "source_followup_from_last_displayed_context",
            }
        )
        return out

    # 2) same action for analyte -> keep previous doc scope
    if same_action_for_subject and not explicit_scope and last_intent == "reference_range_lookup":
        out.update(
            {
                "resolved": True,
                "intent": "reference_range_lookup",
                "reuse_last_displayed_context": True,
                "reuse_doc_scope": bool(prev_doc_ids),
                "skip_retrieval": False,
                "reason": "same_action_for_subject_reuse_reference_lookup",
            }
        )
        return out

    if same_action_for_subject and prev_doc_ids and not explicit_scope:
        out.update(
            {
                "resolved": True,
                "intent": "doc_scoped_results",
                "reuse_last_displayed_context": True,
                "reuse_doc_scope": True,
                "skip_retrieval": False,
                "reason": "same_action_for_subject_reuse_doc_scope",
            }
        )
        return out

    # 3) render/summary/status from last displayed context
    if (is_deictic or correction_format_followup or (asks_summary and not has_global_scope_phrase)) and not explicit_scope and has_any_context:
        intent = out["intent"]
        render_type = None
        if asks_summary:
            intent = "context_summary_render"
            out["skip_retrieval"] = True
        elif asks_status and prev_ctx_type == "biological_numeric_results":
            intent = "response_transform"
            render_type = "status_check"
            out["skip_retrieval"] = True
        elif prev_ctx_type == "medical_qualitative_comment":
            if asks_chart:
                intent = "qualitative_comment_render"
                render_type = "text_table"
                out["skip_retrieval"] = True
            elif asks_note:
                intent = "qualitative_comment_render"
                render_type = "interpretive_note"
                out["skip_retrieval"] = True
            elif asks_card:
                intent = "qualitative_comment_render"
                render_type = "medical_info_card"
                out["skip_retrieval"] = True
            elif asks_table:
                intent = "qualitative_comment_render"
                render_type = "text_table"
                out["skip_retrieval"] = True
        elif prev_ctx_type == "patient_inventory":
            if asks_table:
                intent = "inventory_visualization_render"
                render_type = "filterable_table"
                out["skip_retrieval"] = True
            elif asks_chart:
                intent = "inventory_visualization_render"
                render_type = "patient_cards"
                out["skip_retrieval"] = True
        elif prev_ctx_type == "biological_numeric_results":
            if asks_chart or asks_table:
                intent = "response_transform"
                out["skip_retrieval"] = True
        out.update(
            {
                "resolved": True,
                "intent": intent,
                "reuse_last_displayed_context": True,
                "render_type": render_type,
                "reason": "deictic_render_from_last_displayed_context",
            }
        )
        return out

    # 4) no-context guard
    # Only trigger no-context for genuinely deictic phrasing (or explicit summarize request),
    # and only when absolutely no reusable context exists.
    if (is_deictic or (asks_summary and not has_global_scope_phrase)) and not explicit_scope and not has_any_context:
        out.update(
            {
                "resolved": True,
                "intent": "deictic_no_context",
                "skip_retrieval": True,
                "reason": "deictic_no_context_guard",
            }
        )
        return out

    return out


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
    reuse_qualitative = intent in {"qualitative_comment_render", "visualization_recommendation", "context_summary_render"} and (
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
    elif intent == "context_summary_render":
        should_skip_retrieval = True
        reason = "context_summary_no_retrieval"
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
