from __future__ import annotations

import os
from typing import Any, TypedDict


class RouteCandidate(TypedDict):
    route: str
    confidence: float
    policy: str
    reason: str


class ExecutionPlan(TypedDict):
    route_candidates: list[RouteCandidate]
    rejected_routes: list[RouteCandidate]
    selected_plan: str
    fallback_candidates: list[str]
    shadow_mode: bool
    takeover_allowed: bool
    takeover_reason: str
    planner_version: str


PLANNER_VERSION = "v1"


INTENT_TO_ROUTE: dict[str, str] = {
    "doc_scoped_single_analyte_status": "doc_scoped_single_analyte_status",
    "single_analyte_lookup": "doc_scoped_single_analyte_status",
    "doc_scoped_analyte_query": "doc_scoped_single_analyte_status",
    "doc_scoped_abnormal_results": "doc_scoped_abnormal_results",
    "doc_scoped_summary": "doc_scoped_biological_summary",
    "doc_scoped_biological_summary": "doc_scoped_biological_summary",
    "doc_scoped_priority_anomalies": "doc_scoped_priority_anomalies",
    "reference_range_lookup": "reference_range_lookup",
    "cohort_search": "cohort_search",
    "global_analyte_abnormal_search": "cohort_search",
    "global_patient_lookup": "cohort_search",
    "global_toxicology_search": "global_toxicology_search",
    "doc_scoped_toxicology_summary": "doc_scoped_toxicology_summary",
    "doc_scoped_toxicology_threshold_search": "doc_scoped_toxicology_threshold_search",
    "doc_pair_comparison": "doc_pair_comparison",
    "multi_doc_comparison": "multi_doc_comparison",
    "multi_doc_presence_diff": "multi_doc_presence_diff",
    "doc_scoped_medical_interpretation_guarded": "doc_scoped_medical_interpretation_guarded",
    "diagnostic_safety_question": "diagnostic_safety_question",
    "general_conversation": "general_conversation",
    "small_talk": "general_conversation",
    "help_question": "general_conversation",
    "identity_question": "general_conversation",
    "capability_question": "general_conversation",
}


ROUTE_POLICY: dict[str, str] = {
    "doc_scoped_single_analyte_status": "deterministic_only",
    "doc_scoped_abnormal_results": "deterministic_preferred",
    "doc_scoped_biological_summary": "deterministic_preferred",
    "doc_scoped_priority_anomalies": "deterministic_preferred",
    "reference_range_lookup": "deterministic_only",
    "cohort_search": "deterministic_preferred",
    "global_toxicology_search": "deterministic_preferred",
    "doc_scoped_toxicology_summary": "deterministic_preferred",
    "doc_scoped_toxicology_threshold_search": "deterministic_only",
    "doc_pair_comparison": "deterministic_preferred",
    "multi_doc_comparison": "deterministic_preferred",
    "multi_doc_presence_diff": "deterministic_preferred",
    "doc_scoped_medical_interpretation_guarded": "safety_only",
    "diagnostic_safety_question": "safety_only",
    "general_conversation": "deterministic_preferred",
}


DOC_SCOPED_ROUTES = {
    "doc_scoped_single_analyte_status",
    "doc_scoped_abnormal_results",
    "doc_scoped_biological_summary",
    "doc_scoped_priority_anomalies",
    "doc_scoped_toxicology_summary",
    "doc_scoped_toxicology_threshold_search",
    "doc_pair_comparison",
}


ANALYTE_SCOPED_ROUTES = {
    "doc_scoped_single_analyte_status",
    "reference_range_lookup",
    "cohort_search",
}


ABNORMAL_CONDITION_ROUTES = {
    "doc_scoped_abnormal_results",
    "cohort_search",
    "doc_scoped_priority_anomalies",
}


CRITICAL_AMBIGUITY_FLAGS = {
    "missing_doc_scope",
    "multiple_doc_scope_ambiguous",
    "confidence_below_threshold",
    "multiple_candidates_clustered",
    "insufficient_clinical_scope",
}


def _norm_bool_env(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "1" if default else "0") or "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _as_list_str(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        txt = str(item or "").strip()
        if txt:
            out.append(txt)
    return out


def _to_route(intent: str) -> str:
    key = str(intent or "").strip().lower()
    if key in INTENT_TO_ROUTE:
        return INTENT_TO_ROUTE[key]
    return key if key else "general_conversation"


def _route_policy(route: str) -> str:
    return ROUTE_POLICY.get(str(route or "").strip().lower(), "deterministic_preferred")


def _base_confidence_from_qu(query_understanding: dict[str, Any], fallback: float = 0.55) -> float:
    try:
        val = float(query_understanding.get("intent_confidence", fallback))
    except Exception:
        val = fallback
    return _clamp01(val)


def _score_route_candidate(
    *,
    route: str,
    base_conf: float,
    reason: str,
    requested_doc_ids: list[str],
    requested_analytes: list[str],
    technical_condition: str,
    safety_intent: str,
    ambiguity_flags: list[str],
) -> RouteCandidate:
    conf = float(base_conf)
    route_norm = str(route or "").strip().lower()
    condition_norm = str(technical_condition or "").strip().lower()
    safety_norm = str(safety_intent or "").strip().lower()
    flags = {str(f).strip().lower() for f in ambiguity_flags}

    if route_norm in DOC_SCOPED_ROUTES:
        conf += 0.12 if requested_doc_ids else -0.15
    if route_norm in ANALYTE_SCOPED_ROUTES:
        conf += 0.10 if requested_analytes else -0.15
    if condition_norm and route_norm in ABNORMAL_CONDITION_ROUTES:
        conf += 0.06

    if "missing_doc_scope" in flags:
        conf -= 0.08
    if "multiple_doc_scope_ambiguous" in flags:
        conf -= 0.12
    if "confidence_below_threshold" in flags:
        conf -= 0.08
    if "multiple_candidates_clustered" in flags:
        conf -= 0.06
    if "insufficient_clinical_scope" in flags:
        conf -= 0.15

    if safety_norm:
        if route_norm in {"diagnostic_safety_question", "doc_scoped_medical_interpretation_guarded"}:
            conf = max(conf, 0.95)
        else:
            conf -= 0.25

    conf = round(_clamp01(conf), 2)
    return {
        "route": route_norm,
        "confidence": conf,
        "policy": _route_policy(route_norm),
        "reason": reason,
    }


def _merge_candidates(candidates: list[RouteCandidate]) -> list[RouteCandidate]:
    by_route: dict[str, RouteCandidate] = {}
    for cand in candidates:
        route = str(cand.get("route") or "").strip().lower()
        if not route:
            continue
        current = by_route.get(route)
        if current is None:
            by_route[route] = dict(cand)
            continue
        cur_conf = float(current.get("confidence", 0.0))
        new_conf = float(cand.get("confidence", 0.0))
        if new_conf > cur_conf:
            merged_reason = str(current.get("reason") or "")
            if str(cand.get("reason") or "") and str(cand.get("reason") or "") not in merged_reason:
                merged_reason = (merged_reason + " | " + str(cand.get("reason") or "")).strip(" |")
            by_route[route] = {
                "route": route,
                "confidence": round(new_conf, 2),
                "policy": str(cand.get("policy") or current.get("policy") or "deterministic_preferred"),
                "reason": merged_reason or str(cand.get("reason") or ""),
            }
        else:
            prev_reason = str(current.get("reason") or "")
            new_reason = str(cand.get("reason") or "")
            if new_reason and new_reason not in prev_reason:
                current["reason"] = (prev_reason + " | " + new_reason).strip(" |")
                by_route[route] = current
    out = list(by_route.values())
    out.sort(key=lambda x: (-float(x.get("confidence", 0.0)), str(x.get("route") or "")))
    return out


def _fallback_candidates(
    *,
    requested_doc_ids: list[str],
    requested_analytes: list[str],
    safety_intent: str,
    ambiguity_flags: list[str],
    medical_topics: list[dict[str, Any]],
) -> list[str]:
    fallback: list[str] = []
    flags = {str(f).strip().lower() for f in ambiguity_flags}
    safety_norm = str(safety_intent or "").strip().lower()

    if requested_analytes:
        fallback.append("analyte_not_found")
    if "missing_doc_scope" in flags or (not requested_doc_ids and requested_analytes):
        fallback.append("ambiguous_scope")
    if not requested_doc_ids and not requested_analytes and medical_topics:
        fallback.append("insufficient_evidence")
    if "multiple_doc_scope_ambiguous" in flags or "multiple_candidates_clustered" in flags:
        fallback.append("ambiguous_scope")

    if "diagnostic" in safety_norm:
        fallback.append("diagnosis_refusal")
    if "treatment" in safety_norm:
        fallback.append("treatment_refusal")
    if "pii" in safety_norm:
        fallback.append("pii_refusal")

    if not requested_doc_ids and "missing_doc_scope" in flags:
        fallback.append("document_not_found")

    dedup: list[str] = []
    for item in fallback:
        token = str(item or "").strip().lower()
        if token and token not in dedup:
            dedup.append(token)
    return dedup


def _is_takeover_safe(plan: ExecutionPlan, query_understanding: dict[str, Any], legacy_route: str = "") -> tuple[bool, str]:
    if bool(plan.get("shadow_mode")):
        return False, "shadow_mode_default"
    if not _norm_bool_env("MEDICAL_RAG_PLANNER_ENABLE_TAKEOVER", False):
        return False, "takeover_disabled"

    candidates = list(plan.get("route_candidates") or [])
    if not candidates:
        return False, "no_candidates"
    best = candidates[0]
    best_conf = float(best.get("confidence", 0.0))
    if best_conf < 0.90:
        return False, "low_confidence"

    flags = {
        str(f).strip().lower()
        for f in _as_list_str(query_understanding.get("ambiguity_flags", []))
    }
    if flags.intersection(CRITICAL_AMBIGUITY_FLAGS):
        return False, "critical_ambiguity_flags"

    safety_intent = str(query_understanding.get("safety_intent") or "").strip().lower()
    selected_route = str(best.get("route") or "").strip().lower()
    if safety_intent and selected_route not in {"diagnostic_safety_question", "doc_scoped_medical_interpretation_guarded"}:
        return False, "safety_conflict"

    legacy_norm = str(legacy_route or "").strip().lower()
    if legacy_norm and selected_route != legacy_norm and selected_route not in {
        "diagnostic_safety_question",
        "doc_scoped_medical_interpretation_guarded",
    }:
        return False, "legacy_route_mismatch"

    return True, "takeover_safe_high_confidence"


def build_execution_plan(query_understanding: dict[str, Any], query: str) -> ExecutionPlan:
    qu = dict(query_understanding or {})
    _ = str(query or "")

    requested_doc_ids = _as_list_str(qu.get("requested_doc_ids", []))
    requested_analytes = _as_list_str(qu.get("requested_analytes", []))
    safety_intent = str(qu.get("safety_intent") or "").strip().lower()
    technical_condition = str(qu.get("technical_condition") or "").strip().lower()
    ambiguity_flags = _as_list_str(qu.get("ambiguity_flags", []))
    medical_topics = list(qu.get("medical_topics") or []) if isinstance(qu.get("medical_topics"), list) else []
    primary_intent = str(qu.get("intent") or "").strip().lower()
    scope_conf = _clamp01(float(qu.get("scope_confidence", 0.5) or 0.5))
    base_conf = _base_confidence_from_qu(qu, fallback=max(0.45, scope_conf))

    candidates: list[RouteCandidate] = []
    primary_route = _to_route(primary_intent)
    candidates.append(
        _score_route_candidate(
            route=primary_route,
            base_conf=max(base_conf, 0.55),
            reason="primary_intent_mapping",
            requested_doc_ids=requested_doc_ids,
            requested_analytes=requested_analytes,
            technical_condition=technical_condition,
            safety_intent=safety_intent,
            ambiguity_flags=ambiguity_flags,
        )
    )

    intent_candidates = qu.get("intent_candidates")
    if isinstance(intent_candidates, list):
        for item in intent_candidates:
            if not isinstance(item, dict):
                continue
            cand_intent = str(item.get("intent") or "").strip().lower()
            if not cand_intent:
                continue
            try:
                cand_conf = float(item.get("confidence", base_conf))
            except Exception:
                cand_conf = base_conf
            mapped_route = _to_route(cand_intent)
            candidates.append(
                _score_route_candidate(
                    route=mapped_route,
                    base_conf=cand_conf,
                    reason="intent_candidates_mapping",
                    requested_doc_ids=requested_doc_ids,
                    requested_analytes=requested_analytes,
                    technical_condition=technical_condition,
                    safety_intent=safety_intent,
                    ambiguity_flags=ambiguity_flags,
                )
            )

    if requested_doc_ids and requested_analytes:
        candidates.append(
            _score_route_candidate(
                route="doc_scoped_single_analyte_status",
                base_conf=max(base_conf, 0.62),
                reason="doc_id + analyte",
                requested_doc_ids=requested_doc_ids,
                requested_analytes=requested_analytes,
                technical_condition=technical_condition,
                safety_intent=safety_intent,
                ambiguity_flags=ambiguity_flags,
            )
        )
    if requested_doc_ids and not requested_analytes:
        candidates.append(
            _score_route_candidate(
                route="doc_scoped_biological_summary",
                base_conf=max(base_conf - 0.05, 0.50),
                reason="doc_scope_summary_fallback",
                requested_doc_ids=requested_doc_ids,
                requested_analytes=requested_analytes,
                technical_condition=technical_condition,
                safety_intent=safety_intent,
                ambiguity_flags=ambiguity_flags,
            )
        )
    if safety_intent:
        candidates.append(
            _score_route_candidate(
                route="diagnostic_safety_question",
                base_conf=0.98,
                reason="safety_intent_priority",
                requested_doc_ids=requested_doc_ids,
                requested_analytes=requested_analytes,
                technical_condition=technical_condition,
                safety_intent=safety_intent,
                ambiguity_flags=ambiguity_flags,
            )
        )

    merged = _merge_candidates(candidates)
    if not merged:
        merged = [
            {
                "route": "general_conversation",
                "confidence": 0.35,
                "policy": _route_policy("general_conversation"),
                "reason": "fallback_no_candidate",
            }
        ]

    selected_plan = str(merged[0].get("route") or "general_conversation").strip().lower()
    fallback_candidates = _fallback_candidates(
        requested_doc_ids=requested_doc_ids,
        requested_analytes=requested_analytes,
        safety_intent=safety_intent,
        ambiguity_flags=ambiguity_flags,
        medical_topics=medical_topics,
    )

    shadow_mode = _norm_bool_env("MEDICAL_RAG_PLANNER_SHADOW_MODE", True)
    plan: ExecutionPlan = {
        "route_candidates": merged[:5],
        "rejected_routes": merged[1:5],
        "selected_plan": selected_plan,
        "fallback_candidates": fallback_candidates,
        "shadow_mode": shadow_mode,
        "takeover_allowed": False,
        "takeover_reason": "shadow_mode_default" if shadow_mode else "takeover_not_evaluated",
        "planner_version": PLANNER_VERSION,
    }
    takeover_allowed, takeover_reason = _is_takeover_safe(plan, qu, legacy_route=str(qu.get("intent") or ""))
    plan["takeover_allowed"] = takeover_allowed
    plan["takeover_reason"] = takeover_reason
    return plan


__all__ = [
    "build_execution_plan",
]
