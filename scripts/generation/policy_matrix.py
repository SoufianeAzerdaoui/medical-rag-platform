from __future__ import annotations

from typing import Any

STRICT_DETERMINISTIC_ROUTES = {
    "single_analyte_lookup",
    "doc_scoped_single_analyte_status",
    "doc_scoped_abnormal_results",
    "global_analyte_abnormal_search",
    "global_toxicology_search",
    "doc_scoped_toxicology_threshold_search",
    "doc_scoped_toxicology_summary",
    "doc_pair_comparison",
    "multi_doc_comparison",
    "reference_range_lookup",
}

LEVEL2_HYBRID_INTENT_POLICY: dict[str, dict[str, Any]] = {
    "doc_scoped_biological_summary": {
        "selected_policy": "hybrid_controlled",
        "policy_level": "hybrid_controlled",
        "facts_source": "evidence_rows_only",
        "llm_allowed": True,
        "timeout_s": 60,
        "max_tokens": 220,
        "validator_policy": "facts_safety_style",
        "fallback_mode": "deterministic_doc_scoped_biological_summary",
    },
    "doc_scoped_priority_anomalies": {
        "selected_policy": "hybrid_optional",
        "policy_level": "hybrid_optional",
        "facts_source": "evidence_rows_only",
        "llm_allowed": True,
        "timeout_s": 60,
        "max_tokens": 200,
        "validator_policy": "facts_priority_safety",
        "fallback_mode": "deterministic_doc_scoped_priority_anomalies",
    },
    "doc_scoped_medical_interpretation_guarded": {
        "selected_policy": "hybrid_controlled",
        "policy_level": "hybrid_controlled",
        "facts_source": "evidence_rows_only",
        "llm_allowed": True,
        "timeout_s": 90,
        "max_tokens": 280,
        "validator_policy": "facts_safety_unit",
        "fallback_mode": "deterministic_guarded_medical_interpretation",
    },
    "open_grounded_medical_question": {
        "selected_policy": "hybrid_controlled",
        "policy_level": "hybrid_controlled",
        "facts_source": "evidence_rows_only",
        "llm_allowed": True,
        "timeout_s": 90,
        "max_tokens": 280,
        "validator_policy": "facts_safety_insufficient_context",
        "fallback_mode": "deterministic_no_evidence_response",
    },
}

HARD_GATE_ERRORS = {
    "value_changed",
    "unit_mismatch",
    "reference_changed",
    "status_changed",
    "doc_id_changed",
    "source_changed",
    "unsupported_value",
    "unsupported_source",
    "unsupported_reference",
    "source_alignment_mismatch",
    "source_alignment_mismatch_doc_level",
    "llm_hallucination",
    "diagnostic_affirmation",
    "treatment_recommendation",
    "raw_internal_field_visible",
    "chunk_id_visible",
    "evidence_id_visible",
    "forbidden_internal_field",
    "source_format_bad",
    "general_conversation_no_retrieval_violation",
    "small_talk_triggered_retrieval",
}


def get_intent_policy(intent_or_route: str) -> dict[str, Any]:
    route_norm = str(intent_or_route or "").strip().lower()
    if route_norm in STRICT_DETERMINISTIC_ROUTES:
        return {
            "selected_policy": "deterministic_strict",
            "policy_level": "deterministic_strict",
            "facts_source": "evidence_rows_only",
            "llm_writer_allowed": False,
            "validator_policy": "strict_fact",
        }
    if route_norm in LEVEL2_HYBRID_INTENT_POLICY:
        p = dict(LEVEL2_HYBRID_INTENT_POLICY[route_norm])
        p["llm_writer_allowed"] = bool(p.get("llm_allowed", False))
        return p
    return {
        "selected_policy": "standard",
        "policy_level": "standard",
        "facts_source": "mixed",
        "llm_writer_allowed": True,
        "validator_policy": "default",
    }


__all__ = [
    "STRICT_DETERMINISTIC_ROUTES",
    "LEVEL2_HYBRID_INTENT_POLICY",
    "HARD_GATE_ERRORS",
    "get_intent_policy",
]
