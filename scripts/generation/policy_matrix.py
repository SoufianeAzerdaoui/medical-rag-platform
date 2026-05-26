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

DETERMINISTIC_ONLY_ROUTES = {
    "global_toxicology_search",
    "doc_scoped_toxicology_threshold_search",
    "doc_scoped_toxicology_summary",
    "reference_range_lookup",
    "single_analyte_lookup",
    "doc_scoped_single_analyte_status",
}

DETERMINISTIC_PREFERRED_ROUTES = {
    "doc_scoped_biological_summary",
    "doc_scoped_priority_anomalies",
}

SAFETY_ONLY_ROUTES = {
    "diagnostic_safety_question",
    "treatment_safety_question",
}

LLM_ALLOWED_ROUTES = {
    "doc_scoped_medical_interpretation_guarded",
    "open_grounded_medical_question",
    "response_transform",
}

LLM_WRITER_EXPECTED_ROUTES = set(LLM_ALLOWED_ROUTES)

LEVEL2_HYBRID_INTENT_POLICY: dict[str, dict[str, Any]] = {
    "doc_scoped_biological_summary": {
        "selected_policy": "deterministic_preferred",
        "policy_level": "deterministic_preferred",
        "facts_source": "evidence_rows_only",
        "llm_allowed": False,
        "timeout_s": 60,
        "max_tokens": 220,
        "validator_policy": "facts_safety_style",
        "fallback_mode": "deterministic_doc_scoped_biological_summary",
    },
    "doc_scoped_priority_anomalies": {
        "selected_policy": "deterministic_preferred",
        "policy_level": "deterministic_preferred",
        "facts_source": "evidence_rows_only",
        "llm_allowed": False,
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
        "timeout_s": 30,
        "max_tokens": 160,
        "preferred_model": "llama3.2:latest",
        "enforce_model_lock": True,
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
    "source_mismatch",
    "unsupported_value",
    "unsupported_source",
    "unsupported_reference",
    "source_alignment_mismatch",
    "source_alignment_mismatch_doc_level",
    "llm_hallucination",
    "diagnostic_affirmation",
    "treatment_recommendation",
    "pii_exposure",
    "raw_internal_source",
    "raw_internal_field_visible",
    "chunk_id_visible",
    "evidence_id_visible",
    "forbidden_internal_field",
    "source_format_bad",
    "false_no_abnormal_summary",
    "summary_missing_abnormal_coverage",
    "general_conversation_no_retrieval_violation",
    "small_talk_triggered_retrieval",
}


def get_intent_policy(intent_or_route: str) -> dict[str, Any]:
    route_norm = str(intent_or_route or "").strip().lower()
    if route_norm in SAFETY_ONLY_ROUTES:
        return {
            "selected_policy": "safety_only",
            "policy_level": "safety_only",
            "generation_strategy": "deterministic_only",
            "llm_expected": False,
            "deterministic_preferred_reason": "route_marked_safety_only",
            "facts_source": "validated_evidence_or_refusal_only",
            "llm_writer_allowed": False,
            "validator_policy": "strict_safety",
        }
    if route_norm in DETERMINISTIC_ONLY_ROUTES:
        return {
            "selected_policy": "deterministic_only",
            "policy_level": "deterministic_only",
            "generation_strategy": "deterministic_only",
            "llm_expected": False,
            "deterministic_preferred_reason": "route_marked_deterministic_only",
            "facts_source": "evidence_rows_only",
            "llm_writer_allowed": False,
            "validator_policy": "strict_fact",
        }
    if route_norm in DETERMINISTIC_PREFERRED_ROUTES:
        return {
            "selected_policy": "deterministic_preferred",
            "policy_level": "deterministic_preferred",
            "generation_strategy": "deterministic_preferred",
            "llm_expected": False,
            "deterministic_preferred_reason": "factual_route_backend_structure_first",
            "facts_source": "evidence_rows_only",
            "llm_writer_allowed": False,
            "validator_policy": "strict_fact",
        }
    if route_norm in LLM_WRITER_EXPECTED_ROUTES:
        base = dict(LEVEL2_HYBRID_INTENT_POLICY.get(route_norm) or {})
        if not base:
            base = {
                "selected_policy": "hybrid_controlled",
                "policy_level": "hybrid_controlled",
                "facts_source": "evidence_rows_only",
                "llm_allowed": True,
                "validator_policy": "facts_safety_style",
            }
        base["generation_strategy"] = "llm_writer_expected"
        base["llm_expected"] = True
        base["deterministic_preferred_reason"] = None
        base["llm_writer_allowed"] = bool(base.get("llm_allowed", True))
        return base
    if route_norm in STRICT_DETERMINISTIC_ROUTES:
        return {
            "selected_policy": "deterministic_strict",
            "policy_level": "deterministic_strict",
            "generation_strategy": "deterministic_only",
            "llm_expected": False,
            "deterministic_preferred_reason": "strict_deterministic_route",
            "facts_source": "evidence_rows_only",
            "llm_writer_allowed": False,
            "validator_policy": "strict_fact",
        }
    if route_norm in LEVEL2_HYBRID_INTENT_POLICY:
        p = dict(LEVEL2_HYBRID_INTENT_POLICY[route_norm])
        p["llm_writer_allowed"] = bool(p.get("llm_allowed", False))
        p.setdefault("generation_strategy", "deterministic_preferred" if not p["llm_writer_allowed"] else "llm_writer_expected")
        p.setdefault("llm_expected", bool(p.get("llm_allowed", False)))
        if p.get("generation_strategy") == "deterministic_preferred":
            p.setdefault("deterministic_preferred_reason", "level2_route_deterministic_preferred")
        return p
    return {
        "selected_policy": "standard",
        "policy_level": "standard",
        "generation_strategy": "deterministic_preferred",
        "llm_expected": False,
        "deterministic_preferred_reason": "default_route_not_explicitly_llm_allowed",
        "facts_source": "mixed_validated",
        "llm_writer_allowed": False,
        "validator_policy": "default",
    }


def get_llm_route_class(intent_or_route: str, policy: dict[str, Any] | None = None) -> str:
    route_norm = str(intent_or_route or "").strip().lower()
    p = dict(policy or get_intent_policy(route_norm))
    selected_policy = str(p.get("selected_policy") or "").strip().lower()
    if route_norm in SAFETY_ONLY_ROUTES or selected_policy == "safety_only":
        return "safety_only"
    if bool(p.get("llm_writer_allowed", False)):
        return "llm_allowed"
    if (
        route_norm in DETERMINISTIC_ONLY_ROUTES
        or route_norm in STRICT_DETERMINISTIC_ROUTES
        or selected_policy in {"deterministic_only", "deterministic_strict"}
    ):
        return "deterministic_only"
    return "deterministic_preferred"


__all__ = [
    "STRICT_DETERMINISTIC_ROUTES",
    "DETERMINISTIC_ONLY_ROUTES",
    "DETERMINISTIC_PREFERRED_ROUTES",
    "SAFETY_ONLY_ROUTES",
    "LLM_ALLOWED_ROUTES",
    "LLM_WRITER_EXPECTED_ROUTES",
    "LEVEL2_HYBRID_INTENT_POLICY",
    "HARD_GATE_ERRORS",
    "get_intent_policy",
    "get_llm_route_class",
]
