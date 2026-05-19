from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from query_understanding import detect_requested_doc_ids, parse_query_understanding


MEDICAL_TOPIC_MAP: dict[str, list[str]] = {
    "thyroid": ["TSH", "TSHus", "T4 LIBRE", "T3 LIBRE", "ANTI-TG", "ANTI-TPO", "TRAK"],
    "inflammation": ["CRP", "PROCALCITONINE", "VS"],
    "diabetes_metabolism": ["GLUCOSE", "INSULINE", "PEPTIDE C", "HBA1C"],
    "renal": ["CREATININE", "UREE", "CLAIRANCE DE LA CREATININE", "MICROALBUMINURIE"],
    "liver": ["ASAT", "ALAT", "GGT", "BILIRUBINE TOTALE", "BILIRUBINE DIRECTE"],
    "vitamins": ["VITAMINE D", "VITAMINE B12", "FOLATES"],
    "toxicology": ["LITHIUM", "CARBAMAZEPINE", "ACIDE VALPROIQUE", "AMPHETAMINE", "BENZODIAZEPINE", "COCAINE"],
}


_ALLOWED_INTENTS = {
    "single_analyte_lookup",
    "doc_scoped_abnormal_results",
    "doc_scoped_priority_anomalies",
    "doc_scoped_summary",
    "doc_pair_comparison",
    "global_result_search",
    "global_abnormal_search",
    "guarded_medical_interpretation",
    "open_grounded_medical_question",
    "reference_range_lookup",
    "unsupported_or_insufficient_context",
}
_ALLOWED_SCOPES = {"single_document", "multi_document", "all_documents", "retrieval_required"}
_ALLOWED_TECH_CONDITIONS = {
    "above_reference_only",
    "below_reference_only",
    "out_of_reference",
    "within_reference",
    "any_result",
    "not_applicable",
}
_ALLOWED_SAFETY_MODES = {
    "technical_only",
    "no_diagnosis",
    "no_treatment",
    "grounded_no_diagnosis_no_treatment",
}


@dataclass(frozen=True)
class QueryPlan:
    intent: str
    scope: str
    requested_doc_ids: list[str]
    requested_analytes: list[str]
    medical_topics: list[str]
    technical_condition: str | None
    comparison_targets: list[str]
    requires_llm_writer: bool
    safety_mode: str
    planner_source: str = "deterministic_fallback"


def _norm(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _detect_topics(analytes: list[str]) -> list[str]:
    up = {str(a or "").strip().upper().replace("_", " ") for a in analytes if str(a or "").strip()}
    topics: list[str] = []
    for topic, topic_analytes in MEDICAL_TOPIC_MAP.items():
        topic_set = {a.upper() for a in topic_analytes}
        if up & topic_set:
            topics.append(topic)
    return topics


def _map_qu_intent(qu_intent: str) -> str:
    mapping = {
        "single_analyte_lookup": "single_analyte_lookup",
        "doc_scoped_abnormal_results": "doc_scoped_abnormal_results",
        "doc_scoped_priority_anomalies": "doc_scoped_priority_anomalies",
        "doc_scoped_summary": "doc_scoped_summary",
        "doc_pair_comparison": "doc_pair_comparison",
        "multi_doc_comparison": "doc_pair_comparison",
        "cohort_search": "global_result_search",
        "global_analyte_abnormal_search": "global_abnormal_search",
        "doc_scoped_medical_interpretation_guarded": "guarded_medical_interpretation",
        "diagnostic_safety_question": "guarded_medical_interpretation",
        "reference_range_lookup": "reference_range_lookup",
        "unstructured": "open_grounded_medical_question",
    }
    return mapping.get(str(qu_intent or "").strip().lower(), "unsupported_or_insufficient_context")


def _resolve_scope(doc_ids: list[str], intent: str) -> str:
    if intent in {"global_result_search", "global_abnormal_search"}:
        return "all_documents"
    if len(doc_ids) >= 2:
        return "multi_document"
    if len(doc_ids) == 1:
        return "single_document"
    return "retrieval_required"


def _resolve_tech_condition(raw: str | None, intent: str) -> str:
    cond = str(raw or "").strip().lower()
    if intent in {"reference_range_lookup", "guarded_medical_interpretation"}:
        return "not_applicable"
    mapping = {
        "above_reference": "above_reference_only",
        "below_reference": "below_reference_only",
        "out_of_reference": "out_of_reference",
        "within_reference": "within_reference",
    }
    return mapping.get(cond, "any_result")


def _resolve_safety_mode(qu: Any, intent: str) -> str:
    q_safety = str(getattr(qu, "safety_intent", "") or "").strip().lower()
    if intent == "guarded_medical_interpretation":
        return "grounded_no_diagnosis_no_treatment"
    if q_safety == "diagnostic_safety_question":
        return "no_diagnosis"
    return "technical_only"


def _plan_from_qu(message: str) -> QueryPlan:
    qu = parse_query_understanding(message)
    intent = _map_qu_intent(str(getattr(qu, "intent", "") or ""))
    doc_ids = [str(d).strip() for d in list(getattr(qu, "requested_doc_ids", []) or []) if str(d).strip()]
    analytes = [str(a).strip() for a in list(getattr(qu, "requested_analytes", []) or []) if str(a).strip()]
    if not doc_ids:
        doc_ids = detect_requested_doc_ids(message)
    if intent == "unsupported_or_insufficient_context" and (doc_ids or analytes):
        intent = "open_grounded_medical_question"
    scope = _resolve_scope(doc_ids, intent)
    tech = _resolve_tech_condition(getattr(qu, "technical_condition", None), intent)
    comp_targets = list(doc_ids[:2]) if intent == "doc_pair_comparison" else []
    safety_mode = _resolve_safety_mode(qu, intent)
    topics = _detect_topics(analytes)
    return QueryPlan(
        intent=intent,
        scope=scope,
        requested_doc_ids=doc_ids,
        requested_analytes=analytes,
        medical_topics=topics,
        technical_condition=tech,
        comparison_targets=comp_targets,
        requires_llm_writer=(intent != "reference_range_lookup"),
        safety_mode=safety_mode,
        planner_source="deterministic_fallback",
    )


def _validate_plan_payload(payload: dict[str, Any]) -> QueryPlan | None:
    try:
        intent = str(payload.get("intent") or "").strip()
        scope = str(payload.get("scope") or "").strip()
        safety_mode = str(payload.get("safety_mode") or "").strip()
        if intent not in _ALLOWED_INTENTS or scope not in _ALLOWED_SCOPES or safety_mode not in _ALLOWED_SAFETY_MODES:
            return None
        raw_cond = payload.get("technical_condition")
        technical_condition = None if raw_cond is None else str(raw_cond).strip()
        if technical_condition not in _ALLOWED_TECH_CONDITIONS and technical_condition is not None:
            return None
        req_docs = [str(d).strip() for d in list(payload.get("requested_doc_ids") or []) if str(d).strip()]
        req_analytes = [str(a).strip() for a in list(payload.get("requested_analytes") or []) if str(a).strip()]
        topics = [str(t).strip() for t in list(payload.get("medical_topics") or []) if str(t).strip()]
        comp = [str(c).strip() for c in list(payload.get("comparison_targets") or []) if str(c).strip()]
        requires_llm_writer = bool(payload.get("requires_llm_writer", True))
        return QueryPlan(
            intent=intent,
            scope=scope,
            requested_doc_ids=req_docs,
            requested_analytes=req_analytes,
            medical_topics=topics,
            technical_condition=technical_condition,
            comparison_targets=comp,
            requires_llm_writer=requires_llm_writer,
            safety_mode=safety_mode,
            planner_source="llm_json",
        )
    except Exception:
        return None


def _llm_plan_message(message: str) -> str:
    return (
        "Tu es un planificateur de requête médicale.\n"
        "Réponds en JSON strict uniquement.\n"
        "Ne donne aucune explication.\n"
        "Champs obligatoires: intent, scope, requested_doc_ids, requested_analytes, medical_topics, "
        "technical_condition, comparison_targets, requires_llm_writer, safety_mode.\n"
        "Intent autorisés: single_analyte_lookup, doc_scoped_abnormal_results, doc_scoped_summary, "
        "doc_scoped_priority_anomalies, "
        "doc_pair_comparison, global_result_search, global_abnormal_search, guarded_medical_interpretation, "
        "open_grounded_medical_question, reference_range_lookup, unsupported_or_insufficient_context.\n"
        "Scope autorisés: single_document, multi_document, all_documents, retrieval_required.\n"
        "Safety autorisés: technical_only, no_diagnosis, no_treatment, grounded_no_diagnosis_no_treatment.\n"
        f"Question: {message.strip()}"
    )


def understand_medical_query(
    message: str,
    *,
    llm_client: Any | None = None,
    model: str | None = None,
    timeout: int = 20,
) -> QueryPlan:
    if llm_client is not None:
        try:
            raw = str(
                llm_client.generate(
                    prompt=_llm_plan_message(message),
                    model=model,
                    temperature=0.0,
                    max_tokens=320,
                    timeout=max(8, int(timeout)),
                )
                or ""
            ).strip()
            if raw:
                start = raw.find("{")
                end = raw.rfind("}")
                if start >= 0 and end > start:
                    parsed = json.loads(raw[start : end + 1])
                    if isinstance(parsed, dict):
                        plan = _validate_plan_payload(parsed)
                        if plan is not None:
                            return plan
        except Exception:
            pass
    return _plan_from_qu(message)


def plan_to_payload(plan: QueryPlan) -> dict[str, Any]:
    return asdict(plan)
