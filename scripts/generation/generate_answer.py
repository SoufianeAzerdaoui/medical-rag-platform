from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import sys
import time
import unicodedata
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

# Ensure scripts/ is importable so we can use retrieval package as-is.
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
GENERATION_DIR = Path(__file__).resolve().parent
if str(GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATION_DIR))

_RETRIEVAL_IMPORT_ERROR: Exception | None = None
try:
    from retrieval.models import RetrievalFilters
    from retrieval.search import SearchEngine
except Exception as _retrieval_import_error:  # pragma: no cover - import guard for lightweight test environments
    _RETRIEVAL_IMPORT_ERROR = _retrieval_import_error
    try:
        from scripts.retrieval.models import RetrievalFilters  # type: ignore[no-redef]
        from scripts.retrieval.search import SearchEngine  # type: ignore[no-redef]
    except Exception:  # pragma: no cover - final fallback when retrieval deps are unavailable
        class RetrievalFilters(SimpleNamespace):  # type: ignore[no-redef]
            def __init__(self, **kwargs: Any):
                super().__init__(**kwargs)

        class SearchEngine:  # type: ignore[no-redef]
            def __init__(self, *_args: Any, **_kwargs: Any):
                raise RuntimeError(
                    "Retrieval backend dependencies are unavailable. "
                    "Install retrieval dependencies (e.g. numpy) to enable search."
                ) from _RETRIEVAL_IMPORT_ERROR

from answer_validator import validate_answer
from citation_builder import append_source_citations, build_citations, build_source_citations
try:
    from source_resolver import DocPdfResolver
except Exception:
    from scripts.generation.source_resolver import DocPdfResolver
try:
    from source_normalization import normalize_source_for_response
except Exception:
    from scripts.generation.source_normalization import normalize_source_for_response
from evidence_builder import build_evidence_pack as build_retrieval_evidence_pack
from llm_client import LLMClient, LLMClientError
from model_settings import (
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_NUM_CTX,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_TIMEOUT,
)
from professional_answer_composer import (
    compose_professional_answer,
    render_professional_fallback,
    compose_patient_inventory_answer,
    compose_patient_inventory_count_answer,
)
from prompt_builder import INSUFFICIENT_CONTEXT_SENTENCE, build_prompt
from query_understanding import (
    QueryUnderstanding,
    analyte_display_name,
    build_intent_arbitration_debug,
    contains_exact_term,
    decide_response_strategy,
    detect_exact_analyte,
    detect_exact_analytes,
    detect_technical_condition,
    detect_query_intents,
    detect_requested_doc_ids,
    get_analyte_aliases,
    match_analyte,
    parse_query_understanding,
    norm_text,
)
from followup_scope_utils import resolve_followup_doc_scope
from context_resolver import resolve_context_for_turn, resolve_deictic_request
from conversation_state_utils import evidence_pack_is_transformable
from medical_query_planner import understand_medical_query, plan_to_payload
from qualitative_comment_utils import (
    build_qualitative_comment_answer,
    build_sourced_comment_block,
    clean_qualitative_comment_text,
    dedup_sources_for_qualitative,
    escape_markdown_table_cell,
    format_clickable_source_markdown,
    extract_comment_text_for_subject,
)
from analyte_aliases import ANALYTE_ALIAS_GROUPS
from analyte_resolver import resolve_requested_analytes, load_available_analytes, normalize_analyte_text
from config_loader import get_analyte_families_config, get_priority_scoring_config
from general_conversation import (
    detect_general_conversation,
    get_general_conversation_response,
    is_pure_general_conversation,
    render_general_conversation_response,
)
from medical_topics import detect_medical_topic, get_topic_analytes, get_topic_exclusions
from policy_matrix import (
    HARD_GATE_ERRORS,
    LEVEL2_HYBRID_INTENT_POLICY,
    STRICT_DETERMINISTIC_ROUTES,
    get_intent_policy,
)
from priority_scoring import compute_priority_score as _compute_priority_score_external
from reference_range_parser import parse_reference_ranges
from reference_range_selector import select_reference_range
from reference_range_lookup_flow import run_reference_range_lookup_from_rows
try:
    from backend.services.feature_flag_service import get_feature_flag as _runtime_get_feature_flag
except Exception:  # pragma: no cover - fallback for CLI-only runs without backend package
    _runtime_get_feature_flag = None

LOGGER = logging.getLogger("medical_rag.generation")


def _is_feature_enabled(name: str, default: bool = True) -> bool:
    if _runtime_get_feature_flag is None:
        return default
    try:
        return bool(_runtime_get_feature_flag(str(name)))
    except Exception:
        return default


def _is_critical_medical_intent(query_understanding: QueryUnderstanding) -> bool:
    intent = str(getattr(query_understanding, "intent", "") or "").strip().lower()
    if intent in {"reference_range_lookup", "diagnostic_safety_question"}:
        return True
    if intent in {"doc_scoped_results", "previous_result_comparison", "cohort_search", "global_patient_lookup"}:
        # Numeric medical ask with threshold/range semantics should stay deterministic-first.
        if str(getattr(query_understanding, "requested_value", "") or "").strip():
            return True
        if str(getattr(query_understanding, "comparison_operator", "") or "").strip():
            return True
        if str(getattr(query_understanding, "technical_condition", "") or "").strip():
            return True
    return False


def _strict_policy_for_route(route: str) -> dict[str, Any]:
    return get_intent_policy(route)


def _is_hybrid_structured_writer_intent(query_understanding: QueryUnderstanding) -> bool:
    intent = str(getattr(query_understanding, "intent", "") or "").strip().lower()
    technical_condition = str(getattr(query_understanding, "technical_condition", "") or "").strip().lower()
    if intent in {
        "doc_scoped_abnormal_results",
        "doc_scoped_priority_anomalies",
        "global_analyte_abnormal_search",
        "doc_pair_comparison",
        "multi_doc_comparison",
        "doc_scoped_medical_interpretation_guarded",
        "diagnostic_safety_question",
    }:
        return True
    if intent in {"doc_scoped_summary", "cohort_search"} and technical_condition == "out_of_reference":
        return True
    return False


def _hybrid_writer_mode(query_understanding: QueryUnderstanding) -> str:
    force_llm_writer = str(os.getenv("MEDICAL_RAG_FORCE_LLM_WRITER", "1")).strip().lower() in {"1", "true", "yes", "on"}
    intent_norm = str(getattr(query_understanding, "intent", "") or "").strip().lower()
    if intent_norm == "multi_doc_comparison":
        return "fallback"
    deterministic_routes = {
        "doc_scoped_abnormal_results",
        "doc_scoped_biological_summary",
        "doc_scoped_priority_anomalies",
        "global_analyte_abnormal_search",
        "doc_pair_comparison",
        "multi_doc_comparison",
        "single_analyte_lookup",
        "doc_scoped_medical_interpretation_guarded",
        "reference_range_lookup",
    }
    if not force_llm_writer and str(getattr(query_understanding, "intent", "") or "").strip().lower() in deterministic_routes:
        return "fallback"
    llm_rewrite_enabled = _is_feature_enabled("LLM_REWRITE_ENABLED", default=True)
    llm_fallback_non_critical_only = _is_feature_enabled("LLM_FALLBACK_NON_CRITICAL_ONLY", default=True)
    if not llm_rewrite_enabled:
        return "fallback"
    if _is_hybrid_structured_writer_intent(query_understanding):
        return "hybrid_structured_llm_writer"
    if llm_fallback_non_critical_only and _is_critical_medical_intent(query_understanding):
        # deterministic backend selection + optional rewrite only
        return "fallback"
    return "auto"


def _level2_llm_runtime_config(
    *,
    selected_route: str,
    selected_policy: dict[str, Any],
    force_llm_writer: bool,
    safety_intent_norm: str,
    displayed_evidences: list[dict[str, Any]],
    default_timeout: int,
    default_max_tokens: int,
) -> dict[str, Any]:
    route_norm = str(selected_route or "").strip().lower()
    llm_allowed = bool(selected_policy.get("llm_writer_allowed", False))
    is_level2 = route_norm in LEVEL2_HYBRID_INTENT_POLICY
    has_evidence = bool(displayed_evidences)
    hard_block_safety = safety_intent_norm in {"diagnostic_safety_question"} and route_norm not in {
        "doc_scoped_medical_interpretation_guarded",
    }
    use_llm = bool(
        is_level2
        and llm_allowed
        and has_evidence
        and not hard_block_safety
        and force_llm_writer
    )
    timeout_s = int(selected_policy.get("timeout_s") or default_timeout)
    max_tok = int(selected_policy.get("max_tokens") or default_max_tokens)
    if not use_llm:
        return {
            "use_llm": False,
            "compose_mode": "fallback",
            "timeout_s": timeout_s,
            "max_tokens": max_tok,
        }
    return {
        "use_llm": True,
        "compose_mode": "hybrid_structured_llm_writer",
        "timeout_s": max(15, timeout_s),
        "max_tokens": max(120, max_tok),
    }


def _force_deterministic_mode_for_summary_anomalies(query_understanding: QueryUnderstanding, query_norm: str) -> bool:
    intent = str(getattr(query_understanding, "intent", "") or "").strip().lower()
    if intent not in {"doc_scoped_summary", "immunoanalysis_summary"}:
        return False
    return _query_requests_out_of_reference_only(query_norm)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1) if fenced else raw
    # Try direct parse first
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    # Best-effort: first balanced object
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(candidate[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
    return None


def _normalize_llm_qu_intent(value: Any) -> str | None:
    intent = str(value or "").strip().lower()
    if not intent:
        return None
    allowed = {
        "reference_range_lookup",
        "doc_scoped_results",
        "doc_scoped_summary",
        "response_transform",
        "cohort_search",
        "global_patient_lookup",
        "comment_without_measured_value",
        "small_talk",
    }
    return intent if intent in allowed else None


def _llm_assisted_query_understanding(
    *,
    query: str,
    base_qu: QueryUnderstanding,
    llm_client: LLMClient | None,
    provider: str,
    model: str,
    timeout: int,
) -> tuple[QueryUnderstanding, dict[str, Any]]:
    debug: dict[str, Any] = {"enabled": False, "used": False, "error": None}
    if not _is_feature_enabled("LLM_QUERY_UNDERSTANDING_ENABLED", default=False):
        return base_qu, debug
    debug["enabled"] = True
    client = llm_client or LLMClient(provider=provider)
    prompt = (
        "Tu es un routeur de requêtes médicales. "
        "Retourne UNIQUEMENT un JSON valide sans texte autour.\n"
        "Ne déduis pas de valeurs biologiques. Tu fais seulement du query-understanding.\n"
        "Schema:\n"
        "{"
        "\"intent\": string,"
        "\"requested_analytes\": string[],"
        "\"requested_report_type\": string|null,"
        "\"requested_date_iso\": string|null,"
        "\"requested_reference_profile\": object|null,"
        "\"use_patient_profile\": boolean,"
        "\"request_all_reference_ranges\": boolean,"
        "\"output_format\": string|null,"
        "\"answer_style\": string|null"
        "}\n"
        f"Question: {query}\n"
    )
    try:
        raw = client.generate(
            prompt=prompt,
            model=model,
            temperature=0.0,
            num_ctx=max(2048, int(DEFAULT_LLM_NUM_CTX)),
            max_tokens=220,
            timeout=max(8, int(timeout)),
            keep_alive="5m",
        )
        parsed = _extract_json_object(raw)
        if not isinstance(parsed, dict):
            debug["error"] = "invalid_json"
            return base_qu, debug
        debug["used"] = True
        intent = _normalize_llm_qu_intent(parsed.get("intent")) or base_qu.intent
        llm_analytes = [str(a).strip().lower() for a in (parsed.get("requested_analytes") or []) if str(a).strip()]
        base_analytes = list(base_qu.requested_analytes or [])
        # Keep deterministic extraction as anchor; allow LLM additions only when deterministic found nothing.
        requested_analytes = base_analytes if base_analytes else llm_analytes
        report_type = str(parsed.get("requested_report_type") or "").strip() or base_qu.requested_report_type
        date_iso = str(parsed.get("requested_date_iso") or "").strip() or base_qu.requested_date_iso
        profile = parsed.get("requested_reference_profile")
        if not isinstance(profile, dict):
            profile = base_qu.requested_reference_profile
        use_patient_profile = bool(parsed.get("use_patient_profile")) if "use_patient_profile" in parsed else base_qu.use_patient_profile
        request_all = bool(parsed.get("request_all_reference_ranges")) if "request_all_reference_ranges" in parsed else base_qu.request_all_reference_ranges
        output_format = str(parsed.get("output_format") or "").strip().lower() or base_qu.output_format
        answer_style = str(parsed.get("answer_style") or "").strip().lower() or base_qu.answer_style
        merged = replace(
            base_qu,
            intent=intent,
            requested_analytes=requested_analytes,
            requested_report_type=report_type,
            requested_date_iso=date_iso,
            requested_reference_profile=profile,
            use_patient_profile=use_patient_profile,
            request_all_reference_ranges=request_all,
            output_format=output_format,
            answer_style=answer_style,
            is_response_transform=(intent == "response_transform"),
        )
        debug["llm_intent"] = intent
        debug["llm_requested_analytes"] = requested_analytes
        return merged, debug
    except Exception as exc:
        debug["error"] = f"llm_error:{exc}"
        return base_qu, debug

_GENERIC_COMMENT_SUBJECTS = {
    "commentaire",
    "commentaire medical",
    "commentaire médical",
    "comment",
    "note",
    "observation",
    "interpretation",
    "interprétation",
    "qualitative",
    "resultat",
    "résultat",
    "valeur",
}


def normalize_query(query: str) -> str:
    q = re.sub(r"\s+", " ", (query or "").strip())
    return q


def _has_any_reference_profile_slot(profile: dict[str, Any] | None) -> bool:
    if not isinstance(profile, dict):
        return False
    return any(profile.get(k) not in (None, "") for k in ["sex", "population", "condition", "age", "age_min", "age_max", "age_operator", "age_value"])


def _looks_like_reference_range_followup(query: str) -> bool:
    qn = norm_text(query or "")
    if not qn:
        return False
    markers = [
        "et pour",
        "et chez",
        "meme chose pour",
        "même chose pour",
        "pour aussi",
        "aussi pour",
    ]
    if any(m in qn for m in markers):
        return True
    return qn.startswith("et ")


def _arbitrate_resolution(
    *,
    query_understanding: QueryUnderstanding,
    context_resolution: dict[str, Any],
    deictic_resolution: dict[str, Any],
) -> dict[str, Any]:
    base_intent = str(getattr(query_understanding, "intent", "") or "").strip().lower()
    ctx_reason = str((context_resolution or {}).get("reason") or "").strip()
    deictic_resolved = bool((deictic_resolution or {}).get("resolved"))
    deictic_intent = str((deictic_resolution or {}).get("intent") or "").strip().lower()
    deictic_reason = str((deictic_resolution or {}).get("reason") or "").strip()

    chosen = "context"
    if deictic_resolved and deictic_intent not in {"", "deictic_no_context"}:
        chosen = "deictic"
    elif deictic_intent == "deictic_no_context":
        chosen = "deictic_guard"

    conflict = bool(
        deictic_resolved
        and deictic_intent not in {"", "deictic_no_context"}
        and base_intent
        and deictic_intent != base_intent
    )
    return {
        "base_intent": base_intent,
        "context_reason": ctx_reason,
        "deictic_resolved": deictic_resolved,
        "deictic_intent": deictic_intent,
        "deictic_reason": deictic_reason,
        "chosen": chosen,
        "conflict": conflict,
        "priority_rule": "deictic_if_resolved_else_context",
    }


def _should_force_reference_range_lookup(query_norm: str, query_understanding: QueryUnderstanding) -> bool:
    reference_terms = (
        "norme",
        "plage",
        "référence",
        "reference",
        "valeur normale",
        "valeurs physiologiques",
        "intervalle de référence",
        "plage de référence",
    )
    has_reference_wording = any(term in query_norm for term in reference_terms)
    if not has_reference_wording:
        return False
    tc = detect_technical_condition(query_norm)
    if tc in {"above_reference", "below_reference", "out_of_reference"}:
        return False
    if not list(query_understanding.requested_analytes or []):
        return False
    if str(query_understanding.intent or "").strip().lower() == "reference_range_lookup":
        return False

    measurement_terms = (
        "resultat",
        "résultat",
        "valeur mesuree",
        "valeur mesurée",
        "valeur actuelle",
        "du rapport",
        "dans le rapport",
        "montre moi",
        "affiche moi",
        "donne moi",
    )
    has_measurement_wording = any(t in query_norm for t in measurement_terms)
    has_explicit_scope = bool(
        list(getattr(query_understanding, "requested_doc_ids", None) or [])
        or str(getattr(query_understanding, "requested_date_iso", "") or "").strip()
        or str(getattr(query_understanding, "requested_report_type", "") or "").strip()
    )
    measurement_intent_like = str(getattr(query_understanding, "intent", "") or "").strip().lower() in {
        "doc_scoped_results",
        "biological_numeric_results",
        "cohort_search",
        "global_patient_lookup",
        "global_analyte_abnormal_search",
        "doc_scoped_abnormal_results",
        "doc_pair_comparison",
        "single_analyte_lookup",
    }
    has_global_scope_markers = any(
        t in query_norm
        for t in [
            "tous les rapports",
            "tous les documents",
            "quels documents",
            "y a t il des rapports",
            "y a-t-il des rapports",
            "rapports disponibles",
            "rapports indexes",
            "rapports indexés",
            "sur l ensemble des rapports",
            "sur l’ensemble des rapports",
            "retrouve tous les cas",
            "dans les documents",
            "patients qui ont",
        ]
    )
    has_abnormal_wording = any(
        t in query_norm
        for t in ["hors reference", "anormal", "above_reference", "below_reference", "above reference", "below reference"]
    )
    return not (has_measurement_wording or has_explicit_scope or measurement_intent_like or (has_global_scope_markers and has_abnormal_wording))


def _is_generic_subject(label: str | None) -> bool:
    return norm_text(str(label or "")) in _GENERIC_COMMENT_SUBJECTS


def resolve_medical_subject(
    *,
    query_understanding: QueryUnderstanding | None,
    evidence: dict[str, Any] | None,
    state_context: dict[str, Any] | None,
    query: str,
) -> str:
    qu = query_understanding
    ev = evidence if isinstance(evidence, dict) else {}
    st = state_context if isinstance(state_context, dict) else {}

    requested = []
    if qu is not None:
        requested = list(getattr(qu, "requested_analytes", []) or [])
    if requested:
        return analyte_display_name(requested[0], requested[0]).strip() or str(requested[0]).strip().title()

    detected = detect_exact_analytes(query or "")
    if detected:
        candidate = str(detected[0]).strip().lower()
        return analyte_display_name(candidate, candidate).strip() or candidate.title()

    ev_subject = str(ev.get("subject") or "").strip()
    if ev_subject and not _is_generic_subject(ev_subject):
        return ev_subject
    ev_analyte = str(ev.get("analyte") or ev.get("parameter") or "").strip()
    if ev_analyte and not _is_generic_subject(ev_analyte):
        return ev_analyte

    ctx_subject = str(st.get("subject") or "").strip()
    if ctx_subject and not _is_generic_subject(ctx_subject):
        return ctx_subject

    return "Commentaire médical"


def _resolve_comment_subject_from_query(
    *,
    query_understanding: QueryUnderstanding,
    query: str,
) -> tuple[str, str]:
    requested = [str(a).strip().lower() for a in (getattr(query_understanding, "requested_analytes", []) or []) if str(a).strip()]
    if requested:
        norm = requested[0]
        label = analyte_display_name(norm, norm).strip() or str(norm).strip().upper()
        return norm, label
    detected = detect_exact_analytes(query or "")
    if detected:
        norm = str(detected[0]).strip().lower()
        label = analyte_display_name(norm, norm).strip() or str(norm).strip().upper()
        return norm, label
    return "", "Commentaire médical"


def _wants_all_comments_listing(query: str) -> bool:
    qn = norm_text(query or "")
    if "commentaire" not in qn and "commentaires" not in qn and "note" not in qn and "notes" not in qn:
        return False
    return any(
        k in qn
        for k in [
            "liste",
            "lister",
            "montre",
            "affiche",
            "tous les commentaires",
            "toutes les notes",
            "tous les comment",
        ]
    )


def _wants_single_comment(query: str) -> bool:
    qn = norm_text(query or "")
    if any(k in qn for k in ["tous les commentaires", "toutes les notes", "tous les comment"]):
        return False
    # Singular requests should yield one comment even without the word "seule".
    if re.search(r"\bliste\s+(un|une)\s+commentaire\b", qn):
        return True
    return any(
        k in qn
        for k in [
            "une seule commentaire",
            "un seul commentaire",
            "seulement un commentaire",
            "juste un commentaire",
            "1 commentaire",
            "single comment",
        ]
    )


def _row_looks_like_qualitative_comment(row: dict[str, Any]) -> bool:
    analyte = norm_text(str(row.get("analyte") or row.get("analyte_norm") or ""))
    section = norm_text(str(row.get("section") or row.get("section_norm") or ""))
    merged = norm_text(
        " ".join(
            [
                str(row.get("value_raw") or ""),
                str(row.get("text_for_keyword") or ""),
                str(row.get("text_for_embedding") or ""),
            ]
        )
    )
    if analyte in {"commentaire", "commentaire medical", "commentaire médical", "note", "interpretation", "interprétation"}:
        return True
    if "commentaire" in section or "note" in section or "interpretation" in section or "interprétation" in section:
        return True
    return any(marker in merged for marker in ["commentaire", "valeur seuil", "attention"])


def _is_low_signal_comment_text(text: str) -> bool:
    tn = norm_text(text or "")
    if not tn:
        return True
    low_signal_markers = [
        "resume du rapport medical",
        "type de document",
        "laboratoire",
        "section medicale",
        "laboratory results",
    ]
    if any(m in tn for m in low_signal_markers):
        return True
    # Keep very short but clinically meaningful comments such as "<4,11 IU/ml".
    return False


def _norm_comment_fingerprint(text: str) -> str:
    fp = norm_text(text or "")
    fp = re.sub(r"[^a-z0-9\s]", " ", fp)
    fp = re.sub(r"\s+", " ", fp).strip()
    return fp


def _doc_recency_key(doc_id: str) -> tuple[int, str]:
    d = str(doc_id or "").strip().lower()
    m = re.search(r"(\d+)$", d)
    if m:
        return (int(m.group(1)), d)
    return (0, d)


def _format_comment_for_readability(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    out = raw
    # Restore paragraph separation when packed on one line.
    out = re.sub(r"\s+(Attention\s*:)", r"\n\1", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+(Valeur\s+seuil\s*:)", r"\n\1", out, flags=re.IGNORECASE)
    # Restore bullet list markers.
    out = re.sub(r"\s+-\s+", r"\n- ", out)
    out = re.sub(r"\s*;\s*", " ; ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _merge_comment_variants(texts: list[str]) -> str:
    max_variants = 4
    max_chars = 1200

    cleaned = [str(t or "").strip() for t in texts if str(t or "").strip()]
    if not cleaned:
        return ""
    # Keep the richest version first, then append only complementary variants.
    cleaned.sort(key=len, reverse=True)
    merged_parts: list[str] = [cleaned[0][:max_chars].strip()]
    merged = merged_parts[0]
    merged_fp = _norm_comment_fingerprint(merged)
    for candidate in cleaned[1:]:
        if len(merged_parts) >= max_variants:
            break
        cand_fp = _norm_comment_fingerprint(candidate)
        if not cand_fp or cand_fp in merged_fp or merged_fp in cand_fp:
            continue
        remaining = max_chars - len(merged)
        if remaining <= 0:
            break
        to_add = candidate
        prefix_len = 1  # newline separator
        if len(to_add) + prefix_len > remaining:
            trim_to = max(0, remaining - prefix_len - 1)
            if trim_to <= 0:
                break
            to_add = to_add[:trim_to].rstrip() + "…"
            merged_parts.append(to_add)
            merged = "\n".join(merged_parts)
            merged_fp = _norm_comment_fingerprint(merged)
            break
        merged_parts.append(to_add)
        merged = "\n".join(merged_parts)
        merged_fp = _norm_comment_fingerprint(merged)
    return _format_comment_for_readability(merged[:max_chars].rstrip())


def _enrich_comment_with_unit_if_missing(text: str, unit: str) -> str:
    comment = str(text or "").strip()
    u = str(unit or "").strip()
    if not comment or not u:
        return comment
    # If the comment is a compact threshold/value without explicit unit, append unit.
    # Example: "<4,11" -> "<4,11 IU/ml"
    has_letters = bool(re.search(r"[a-zA-Zµ]", comment))
    looks_numeric_threshold = bool(re.fullmatch(r"[<>]=?\s*\d+(?:[.,]\d+)?", comment))
    if looks_numeric_threshold and not has_letters:
        return f"{comment} {u}"
    # Also handle prefixed compact comments such as "Commentaire : <4,11".
    prefixed_threshold = re.fullmatch(r"(?:(?P<prefix>[^:]{1,40}:\s*)?)(?P<th>[<>]=?\s*\d+(?:[.,]\d+)?)", comment)
    if prefixed_threshold and not re.search(rf"\b{re.escape(u)}\b", comment, flags=re.IGNORECASE):
        prefix = str(prefixed_threshold.group("prefix") or "")
        threshold = str(prefixed_threshold.group("th") or "").strip()
        return f"{prefix}{threshold} {u}".strip()
    return comment


def _build_multi_comment_answer(evidences: list[dict[str, Any]]) -> str:
    if not evidences:
        return "Aucun commentaire exploitable n’a été retrouvé dans les données indexées."
    lines = ["### Commentaires retrouvés", ""]
    use_bullets = len(evidences) > 1
    for idx, ev in enumerate(evidences[:12], start=1):
        subject = str(ev.get("subject") or ev.get("analyte") or "Commentaire médical").strip()
        comment_text = str(ev.get("display_comment_text") or ev.get("comment_text") or ev.get("current_value") or "").strip()
        if not comment_text:
            continue
        pretty = _format_comment_for_readability(comment_text)
        if use_bullets:
            one_line = " ".join(part.strip() for part in pretty.splitlines() if part.strip())
            lines.append(f"- **{subject}** : {one_line}")
        else:
            lines.append(f"{idx}. **{subject}**")
            lines.append(pretty)
        lines.append("")
    if len(lines) <= 2:
        return "Aucun commentaire exploitable n’a été retrouvé dans les données indexées."
    if len(evidences) > 12:
        lines.append(f"- … {len(evidences) - 12} autre(s) commentaire(s) disponible(s).")
    return "\n".join(lines)


_INTERNAL_REASONING_PATTERNS = [
    "okay, the user",
    "the user said",
    "the user wants",
    "i need to",
    "i should",
    "first, i'll",
    "first i ll",
    "first, i will",
    "first i will",
    "i will",
    "let me",
    "je dois",
    "je vais",
    "raisonnement",
    "stratégie",
    "strategie",
    "plan interne",
    "<think>",
    "</think>",
    "je dois répondre",
    "je vais répondre",
]

SMALL_TALK_FALLBACK_ANSWER = "Bonjour ! Je suis prêt à vous aider à analyser vos rapports médicaux."
VISUALIZATION_REGISTRY: dict[str, dict[str, Any]] = {
    "bar": {
        "label_fr": "graphique en barres",
        "supported": True,
        "best_for": ["comparaison", "écart à la référence", "plusieurs analytes"],
        "requires_same_unit": False,
    },
    "line": {
        "label_fr": "courbe",
        "supported": True,
        "best_for": ["évolution temporelle", "série chronologique"],
        "requires_same_unit": True,
    },
    "radar": {
        "label_fr": "graphique radar",
        "supported": False,
        "best_for": ["profil multivarié normalisé"],
        "requires_normalized_values": True,
    },
    "scatter": {
        "label_fr": "nuage de points",
        "supported": False,
        "best_for": ["relation entre deux variables"],
        "requires_two_numeric_axes": True,
    },
    "heatmap": {
        "label_fr": "heatmap",
        "supported": False,
        "best_for": ["matrice de valeurs"],
        "requires_matrix_data": True,
    },
    "unknown": {
        "label_fr": "format visuel demandé",
        "supported": False,
        "best_for": [],
    },
}
GENERAL_CONVERSATION_INTENTS = {"general_conversation", "small_talk", "identity_question", "capability_question", "help_question", "thanks"}


def _is_treatment_request(query_norm: str) -> bool:
    return any(
        k in query_norm
        for k in [
            "traitement",
            "recommandes tu",
            "recommande",
            "prescrire",
            "prescription",
            "supplementation",
            "supplémentation",
        ]
    )


def _select_hard_gate_fallback_mode(
    *,
    hard_gate_errors: set[str],
    selected_route: str,
    safety_intent_norm: str,
    has_evidence: bool,
    query_norm: str,
) -> str:
    if hard_gate_errors.intersection({"general_conversation_no_retrieval_violation", "small_talk_triggered_retrieval"}):
        return "deterministic_general_conversation"
    if _is_treatment_request(query_norm):
        return "deterministic_treatment_refusal_with_technical_summary"
    if safety_intent_norm == "diagnostic_safety_question":
        return "deterministic_diagnostic_refusal_with_technical_summary"
    if not has_evidence:
        return "deterministic_no_evidence_response"
    per_route = {
        "doc_scoped_biological_summary": "deterministic_doc_scoped_biological_summary",
        "doc_scoped_priority_anomalies": "deterministic_doc_scoped_priority_anomalies",
        "doc_scoped_medical_interpretation_guarded": "deterministic_guarded_medical_interpretation",
        "doc_scoped_abnormal_results": "deterministic_doc_scoped_abnormal_results",
        "global_analyte_abnormal_search": "deterministic_global_analyte_abnormal_search",
        "doc_pair_comparison": "deterministic_doc_pair_comparison",
        "multi_doc_comparison": "deterministic_multi_doc_comparison",
        "single_analyte_lookup": "deterministic_single_analyte_lookup",
        "reference_range_lookup": "deterministic_reference_range_lookup",
    }
    return per_route.get(str(selected_route or "").strip().lower(), "deterministic_safety_fallback_after_llm_validation_failure")


def _is_general_conversation_fastpath_candidate(
    *,
    query_norm: str,
    requested_doc_ids: list[str],
    requested_analytes: list[str],
    requested_value: str | None,
    comparison_operator: str | None,
) -> bool:
    if requested_doc_ids or requested_analytes or requested_value or comparison_operator:
        return False
    return is_pure_general_conversation(query_norm)


def _contains_internal_reasoning_leak(answer: str) -> bool:
    body = norm_text(answer or "")
    if not body:
        return False
    if "<think>" in (answer or "").lower() or "</think>" in (answer or "").lower():
        return True
    return any(norm_text(p) in body for p in _INTERNAL_REASONING_PATTERNS)


def sanitize_final_answer(answer: str) -> str:
    raw = (answer or "").strip()
    if not raw:
        return raw
    raw = re.sub(r"(?is)<think>.*?</think>", "", raw).strip()
    raw = re.sub(r"(?im)^thinking\s*:\s*", "", raw).strip()
    raw = re.sub(r"(?im)^reasoning\s*:\s*", "", raw).strip()
    raw = re.sub(r"(?im)^plan\s*:\s*", "", raw).strip()
    raw = re.sub(r"(?im)^réponse\s*:\s*", "", raw).strip()
    raw = re.sub(r"(?im)^reponse\s*:\s*", "", raw).strip()
    return raw.strip()


def _rewrite_final_without_reasoning(
    *,
    leaked_answer: str,
    user_message: str,
    llm_client: LLMClient | None,
    provider: str,
    model: str,
    timeout: int,
) -> tuple[str, str | None]:
    client = llm_client or LLMClient(provider=provider)
    prompt = (
        "Tu es l'assistant spécialisé d'une plateforme d'analyse médicale.\n"
        "Réécris la réponse finale pour l'utilisateur de manière fluide et professionnelle.\n"
        "Supprime tout raisonnement interne, plan d'action ou mention technique de ton fonctionnement.\n"
        "Conserve uniquement les faits médicaux et la structure de la réponse.\n"
        "Sortie : texte final épuré uniquement.\n"
        "/no_think\n\n"
        f"Message utilisateur : {user_message.strip()}\n\n"
        f"Texte à purifier :\n{leaked_answer.strip()}\n"
    )
    try:
        rewritten = client.generate(
            prompt=prompt,
            model=model,
            temperature=0.0,
            num_ctx=2048,
            max_tokens=180,
            timeout=max(6, int(timeout)),
            keep_alive="5m",
        )
    except LLMClientError as exc:
        return "", str(exc)
    return sanitize_final_answer(rewritten), None


def sanitize_final_answer_with_retry(
    *,
    answer: str,
    user_message: str,
    llm_client: LLMClient | None,
    provider: str,
    model: str,
    timeout: int,
    fallback_answer: str,
) -> tuple[str, bool, str | None]:
    sanitized = sanitize_final_answer(answer)
    if sanitized and not _contains_internal_reasoning_leak(sanitized):
        return sanitized, False, None

    rewritten, rewrite_err = _rewrite_final_without_reasoning(
        leaked_answer=sanitized or answer,
        user_message=user_message,
        llm_client=llm_client,
        provider=provider,
        model=model,
        timeout=timeout,
    )
    rewritten = sanitize_final_answer(rewritten)
    if rewritten and not _contains_internal_reasoning_leak(rewritten):
        return rewritten, True, None
    return sanitize_final_answer(fallback_answer), True, rewrite_err


def _query_is_sensitive_or_treatment(query: str) -> bool:
    q = normalize_query(query).lower()
    sensitive_markers = [
        "nom du patient",
        "date de naissance",
        "prescripteur",
        "telephone",
        "numéro",
        "numero",
        "patient id",
        "patient_id",
    ]
    treatment_markers = [
        "traitement",
        "prescrire",
        "posologie",
        "dose",
        "medicament",
        "médicament",
    ]
    return any(k in q for k in (sensitive_markers + treatment_markers))


def _build_structured_fallback_answer(query: str, evidence_pack: list[dict[str, Any]], exact_analyte: str | None = None) -> str:
    if not evidence_pack:
        return INSUFFICIENT_CONTEXT_SENTENCE

    qu = parse_query_understanding(query)
    candidates = list(evidence_pack)
    if exact_analyte:
        exact_candidates = [
            e
            for e in candidates
            if contains_exact_term(str(e.get("analyte_norm") or ""), exact_analyte)
            or contains_exact_term(str(e.get("analyte") or ""), exact_analyte)
        ]
        if exact_candidates:
            candidates = exact_candidates

    structured_evidences: list[dict[str, Any]] = []
    for ev in candidates:
        status_code = str(ev.get("technical_status_code") or ev.get("interpretation_status") or "").strip().lower()
        status_label = _interpretation_fr(status_code)
        structured_evidences.append(
            {
                "doc_id": ev.get("doc_id"),
                "patient_token": ev.get("patient_token"),
                "page": ev.get("page_number"),
                "row": ev.get("row_index"),
                "analyte": ev.get("analyte_display") or ev.get("analyte") or ev.get("parameter") or "non précisé",
                "analyte_norm": ev.get("analyte_norm"),
                "current_value": ev.get("value_raw"),
                "unit": ev.get("unit"),
                "reference": ev.get("reference_range"),
                "previous_result": ev.get("previous_result"),
                "technical_status_code": status_code or "not_interpretable",
                "technical_status": status_label,
                "variation": _variation_label(ev.get("value_raw"), ev.get("previous_result")),
            }
        )

    fallback_pack = {
        "question": query,
        "intent": qu.intent,
        "requested_doc_ids": list(qu.requested_doc_ids or []),
        "requested_analytes": list(qu.requested_analytes or []),
        "requested_table_columns": list(qu.requested_table_columns or []),
        "output_format": qu.output_format,
        "answer_style": qu.answer_style,
        "technical_condition": qu.technical_condition,
        "evidences": structured_evidences,
        "missing_items": [],
    }
    composed = render_professional_fallback(
        evidence_pack=fallback_pack,
        query_understanding=qu,
        user_question=query,
        source_citations=[],
    )
    return str(composed.get("answer") or INSUFFICIENT_CONTEXT_SENTENCE).strip()


def _answer_needs_fallback(text: str) -> bool:
    if not text.strip():
        return True
    low = text.lower()
    noisy_markers = [
        "okay,",
        "let's",
        "i need to",
        "the user is asking",
        "let me",
        "first,",
    ]
    if any(m in low for m in noisy_markers):
        return True
    if len(text) > 2200:
        return True
    return False


def _value_to_float(value: Any) -> float | None:
    if value is None:
        return None
    s = str(value).strip().replace(",", ".")
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _has_mixed_units(evidences: list[dict[str, Any]]) -> bool:
    units = {str(ev.get("unit") or "").strip().lower() for ev in evidences if str(ev.get("unit") or "").strip()}
    return len(units) > 1


def _visualization_label(kind: str | None) -> str:
    key = str(kind or "unknown").strip().lower()
    entry = VISUALIZATION_REGISTRY.get(key) or VISUALIZATION_REGISTRY["unknown"]
    return str(entry.get("label_fr") or "visualisation")


def _normalize_requested_visualization_type(chart_type: str | None, raw_format_phrase: str | None) -> str:
    direct = str(chart_type or "").strip().lower()
    if direct in VISUALIZATION_REGISTRY:
        return direct
    raw_norm = norm_text(str(raw_format_phrase or ""))
    if any(k in raw_norm for k in ["bar chart", "bar graph", "barres", "barre", "histogramme"]):
        return "bar"
    if any(k in raw_norm for k in ["line graph", "line chart", "line-graph", "arithmetic line graph", "courbe"]):
        return "line"
    if any(k in raw_norm for k in ["radar", "spider"]):
        return "radar"
    if any(k in raw_norm for k in ["scatter", "nuage de points"]):
        return "scatter"
    if any(k in raw_norm for k in ["heatmap", "carte thermique", "matrix", "matrice"]):
        return "heatmap"
    return "unknown"


def _has_temporal_axis(evidences: list[dict[str, Any]]) -> bool:
    temporal_keys = ("observation_date", "date", "datetime", "timestamp", "collected_at", "report_date")
    for ev in evidences:
        for key in temporal_keys:
            if str(ev.get(key) or "").strip():
                return True
    return False


def _normalize_status_code(status: str | None, status_code: str | None = None) -> str:
    code = str(status_code or "").strip().lower()
    if code in {"above_reference", "below_reference", "within_reference", "not_interpretable"}:
        return code
    if code in {"in_reference", "normal"}:
        return "within_reference"
    if code in {"qualitative", "missing_value"}:
        return "not_interpretable"
    text = norm_text(str(status or ""))
    if any(k in text for k in ["au dessus", "au-dessus", "above reference", "superieur", "supérieur"]):
        return "above_reference"
    if any(k in text for k in ["en dessous", "below reference", "inferieur", "inférieur"]):
        return "below_reference"
    if any(k in text for k in ["dans la reference", "within reference"]):
        return "within_reference"
    return "not_interpretable"


def normalize_result_status(evidence_item: dict[str, Any]) -> dict[str, str]:
    raw_status = _normalize_status_code(
        str(evidence_item.get("technical_status") or evidence_item.get("status") or ""),
        str(evidence_item.get("technical_status_code") or evidence_item.get("interpretation_status") or ""),
    )
    display_status = _interpretation_fr(raw_status)
    return {"raw_status": raw_status, "display_status": display_status}


def _parse_reference_bounds(reference: str) -> dict[str, Any]:
    raw = str(reference or "").strip()
    if not raw:
        return {
            "metric_available": False,
            "reference_type": "missing",
            "lower_bound": None,
            "upper_bound": None,
        }
    raw_norm = norm_text(raw).replace("≤", "<=").replace("≥", ">=")
    raw_comp = raw.lower().replace(",", ".").replace("≤", "<=").replace("≥", ">=").strip()
    if any(k in raw_norm for k in ["qualitatif", "non disponible", "n a", "na", "negatif", "négatif", "positif"]):
        return {
            "metric_available": False,
            "reference_type": "not_interpretable",
            "lower_bound": None,
            "upper_bound": None,
        }

    upper_match = re.match(r"^\s*(?:<|<=|≤)\s*([0-9]+(?:[.,][0-9]+)?)", raw, flags=re.IGNORECASE)
    if upper_match:
        upper = _value_to_float(upper_match.group(1))
        return {
            "metric_available": upper is not None and upper > 0,
            "reference_type": "upper_threshold",
            "lower_bound": None,
            "upper_bound": upper,
        }

    lower_match = re.match(r"^\s*(?:>|>=|≥)\s*([0-9]+(?:[.,][0-9]+)?)", raw, flags=re.IGNORECASE)
    if lower_match:
        lower = _value_to_float(lower_match.group(1))
        return {
            "metric_available": lower is not None and lower > 0,
            "reference_type": "lower_threshold",
            "lower_bound": lower,
            "upper_bound": None,
        }

    interval_match = re.search(
        r"([0-9]+(?:\.[0-9]+)?)\s*(?:a|à|-|to|jusqu(?:a|à))\s*([0-9]+(?:\.[0-9]+)?)",
        raw_comp,
        flags=re.IGNORECASE,
    )
    if interval_match:
        lo = _value_to_float(interval_match.group(1))
        hi = _value_to_float(interval_match.group(2))
        if lo is not None and hi is not None:
            lower, upper = (lo, hi) if lo <= hi else (hi, lo)
            return {
                "metric_available": lower > 0 and upper > 0,
                "reference_type": "interval",
                "lower_bound": lower,
                "upper_bound": upper,
            }

    nums = re.findall(r"\d+(?:[.,]\d+)?", raw)
    if len(nums) >= 2:
        lo = _value_to_float(nums[0])
        hi = _value_to_float(nums[1])
        if lo is not None and hi is not None:
            lower, upper = (lo, hi) if lo <= hi else (hi, lo)
            return {
                "metric_available": lower > 0 and upper > 0,
                "reference_type": "interval",
                "lower_bound": lower,
                "upper_bound": upper,
            }

    return {
        "metric_available": False,
        "reference_type": "not_interpretable",
        "lower_bound": None,
        "upper_bound": None,
    }


def _deviation_label(deviation: float | None, *, lower_bound: float | None, upper_bound: float | None, metric_available: bool) -> str:
    if not metric_available or deviation is None:
        return "non calculable"
    if abs(deviation) < 1e-12:
        return "dans la référence"
    pct = abs(deviation) * 100.0
    pct_txt = f"{pct:.0f}%"
    if deviation > 0:
        suffix = "de la limite haute" if upper_bound is not None else "du seuil de référence"
        return f"+{pct_txt} au-dessus {suffix}"
    suffix = "de la limite basse" if lower_bound is not None else "du seuil de référence"
    return f"{pct_txt} en dessous {suffix}"


def compute_reference_metric(value: Any, reference: str, status: str) -> dict[str, Any]:
    numeric = _value_to_float(value)
    bounds = _parse_reference_bounds(reference)
    lower_bound = bounds.get("lower_bound")
    upper_bound = bounds.get("upper_bound")
    metric_available = bool(bounds.get("metric_available")) and numeric is not None
    status_code = _normalize_status_code(status)
    deviation: float | None = None

    if metric_available:
        if status_code == "within_reference":
            deviation = 0.0
        elif status_code == "above_reference":
            denom = upper_bound if upper_bound is not None else lower_bound
            if isinstance(denom, float) and denom > 0:
                deviation = (numeric / denom) - 1.0
        elif status_code == "below_reference":
            denom = lower_bound if lower_bound is not None else upper_bound
            if isinstance(denom, float) and denom > 0:
                deviation = (numeric / denom) - 1.0
        else:
            if upper_bound is not None and upper_bound > 0:
                deviation = (numeric / upper_bound) - 1.0
            elif lower_bound is not None and lower_bound > 0:
                deviation = (numeric / lower_bound) - 1.0

    if deviation is None:
        metric_available = False

    return {
        "metric_available": metric_available,
        "reference_type": bounds.get("reference_type"),
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "reference_deviation": (round(float(deviation), 6) if deviation is not None else None),
        "deviation_label": _deviation_label(
            deviation,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            metric_available=metric_available,
        ),
    }


def _build_visualization_data(evidences: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    data: list[dict[str, Any]] = []
    has_deviation = False
    for ev in evidences:
        analyte_norm = str(ev.get("analyte_norm") or "").strip().lower()
        analyte = analyte_display_name(str(ev.get("analyte") or analyte_norm or "analyte"), analyte_norm or None)
        ref = str(ev.get("reference") or ev.get("reference_range") or "").strip()
        raw_value = str(ev.get("current_value") or ev.get("value_raw") or "").strip()
        numeric = _value_to_float(raw_value)
        status_norm = normalize_result_status(ev)
        status_text = str(ev.get("technical_status") or ev.get("status") or status_norm["display_status"]).strip()
        status_code = status_norm["raw_status"]
        metric = compute_reference_metric(raw_value, ref, status_code or status_text)
        if metric.get("metric_available"):
            has_deviation = True

        item: dict[str, Any] = {
            "analyte": analyte,
            "value": numeric if numeric is not None else raw_value,
            "raw_value": raw_value,
            "value_numeric": numeric,
            "unit": str(ev.get("unit") or "").strip(),
            "reference": ref,
            "status": status_text or status_norm["display_status"],
            "status_code": status_code,
            "lower_bound": metric.get("lower_bound"),
            "upper_bound": metric.get("upper_bound"),
            "reference_deviation": metric.get("reference_deviation"),
            "deviation_label": metric.get("deviation_label"),
            "metric_available": bool(metric.get("metric_available")),
            "source_label": str(ev.get("source_label") or ev.get("source") or "").strip(),
        }
        data.append(item)
    return data, has_deviation


def evaluate_visualization_suitability(requested_type: str, evidence_pack: list[dict[str, Any]]) -> dict[str, Any]:
    evidences = list(evidence_pack or [])
    mixed_units = _has_mixed_units(evidences)
    has_time = _has_temporal_axis(evidences)
    data, has_deviation = _build_visualization_data(evidences)
    has_numeric_values = any(isinstance(item.get("value_numeric"), (int, float)) for item in data)
    requested = str(requested_type or "unknown").strip().lower()

    if requested == "unknown":
        return {
            "suitable": False,
            "reason": "Le format demandé est ambigu et ne correspond pas à un type de visualisation reconnu.",
            "recommended_type": "bar" if has_numeric_values else "table",
            "recommendation_reason": "Un graphique en barres reste le format le plus lisible pour comparer plusieurs analytes.",
        }
    if requested == "line":
        if not has_time:
            return {
                "suitable": False,
                "reason": "Ces résultats ne forment pas une série temporelle, donc une courbe n’est pas fiable.",
                "recommended_type": "bar" if has_numeric_values else "table",
                "recommendation_reason": "Un graphique en barres permet une comparaison directe sans suggérer une évolution temporelle.",
            }
        if mixed_units:
            return {
                "suitable": False,
                "reason": "Les résultats utilisent des unités biologiques différentes, ce qui rend une courbe brute trompeuse.",
                "recommended_type": "bar",
                "recommendation_reason": "Une comparaison en barres avec ratio à la référence est plus robuste quand les unités diffèrent.",
            }
    if requested == "radar" and not has_deviation:
        return {
            "suitable": False,
            "reason": "Le profil radar nécessite des valeurs normalisées homogènes, indisponibles pour tous les analytes.",
            "recommended_type": "bar" if has_numeric_values else "table",
            "recommendation_reason": "Un graphique en barres garde une comparaison stable même avec des unités différentes.",
        }
    if requested == "scatter":
        return {
            "suitable": False,
            "reason": "Un nuage de points exige deux axes numériques corrélés par observation, ce qui n’est pas le cas ici.",
            "recommended_type": "bar" if has_numeric_values else "table",
            "recommendation_reason": "La comparaison par analyte en barres correspond mieux à la structure des résultats biologiques.",
        }
    if requested == "heatmap":
        return {
            "suitable": False,
            "reason": "Une heatmap requiert une matrice de valeurs structurée (lignes/colonnes), absente dans ces résultats.",
            "recommended_type": "bar" if has_numeric_values else "table",
            "recommendation_reason": "Le graphique en barres est plus fiable pour comparer des analytes indépendants.",
        }
    return {
        "suitable": True,
        "reason": None,
        "recommended_type": requested if requested in VISUALIZATION_REGISTRY else "bar",
        "recommendation_reason": None,
    }


def build_visualization_payload(
    requested_type: str,
    evidence_pack: list[dict[str, Any]],
    supported_visualizations: list[str],
    *,
    raw_format_phrase: str | None = None,
    source: str = "current_retrieval",
) -> dict[str, Any]:
    requested = str(requested_type or "unknown").strip().lower()
    requested_entry = VISUALIZATION_REGISTRY.get(requested) or VISUALIZATION_REGISTRY["unknown"]
    supported_set = {str(v).strip().lower() for v in supported_visualizations if str(v).strip()}
    supported = requested in supported_set

    suitability = evaluate_visualization_suitability(requested, evidence_pack)
    suitable = bool(suitability.get("suitable"))
    recommendation_type = str(suitability.get("recommended_type") or "bar").strip().lower()
    if recommendation_type not in VISUALIZATION_REGISTRY and recommendation_type != "table":
        recommendation_type = "bar"
    if not supported and recommendation_type == requested:
        if "bar" in supported_set:
            recommendation_type = "bar"
        elif "line" in supported_set:
            recommendation_type = "line"
        else:
            recommendation_type = "table"

    rendered_type: str | None = None
    fallback_reason: str | None = None

    if supported and suitable:
        rendered_type = requested
    else:
        rendered_type = recommendation_type if recommendation_type in supported_set else ("table" if evidence_pack else None)
        if not supported:
            if requested == "unknown":
                phrase = str(raw_format_phrase or "").strip()
                if phrase:
                    fallback_reason = f"Le format « {phrase} » n’est pas directement reconnu par l’interface."
                else:
                    fallback_reason = "Le format demandé n’est pas reconnu par l’interface."
            else:
                fallback_reason = f"{_visualization_label(requested)} n’est pas encore disponible dans l’interface."
        elif not suitable:
            fallback_reason = str(suitability.get("reason") or "").strip() or "Ce format n’est pas adapté aux données disponibles."

    data, has_deviation = _build_visualization_data(evidence_pack)
    y_field = "reference_deviation" if has_deviation else "value_numeric"
    rendered_label = _visualization_label(rendered_type) if rendered_type else None
    metric_label = "Écart normalisé à la référence" if has_deviation else "Valeur mesurée"
    metric_reason = (
        "Les analytes utilisent des unités différentes."
        if _has_mixed_units(evidence_pack)
        else "Les valeurs suivent directement la mesure brute."
    )
    calculable_count = sum(1 for row in data if bool(row.get("metric_available")))
    source_docs = sorted(
        {
            str(ev.get("doc_id") or "").strip().lower()
            for ev in evidence_pack
            if str(ev.get("doc_id") or "").strip()
        }
    )
    title_doc = source_docs[0] if len(source_docs) == 1 else "rapports sélectionnés"

    payload: dict[str, Any] = {
        "requested": True,
        "requested_type": requested,
        "requested_label": str(requested_entry.get("label_fr") or "visualisation demandée"),
        "rendered_type": rendered_type,
        "rendered_label": rendered_label,
        "supported": supported,
        "suitable": suitable,
        "fallback_used": bool(rendered_type != requested),
        "fallback_reason": fallback_reason,
        "recommended_type": recommendation_type,
        "recommendation_reason": str(suitability.get("recommendation_reason") or "").strip() or None,
        "source": source,
        "x_field": "analyte",
        "y_field": y_field,
        "metric_label": metric_label,
        "metric_reason": metric_reason,
        "result_count": len(data),
        "calculable_count": calculable_count,
        "data": data,
        "type": rendered_type,
        "title": f"Écart à la référence — {title_doc}" if rendered_type else "Écart à la référence",
        "reason": fallback_reason or (str(suitability.get("reason") or "").strip() or None),
    }
    return payload


def _humanize_requested_output(query_understanding: QueryUnderstanding) -> str:
    presentation = getattr(query_understanding, "presentation_intent", None)
    requested = str(getattr(presentation, "requested_output", query_understanding.output_format) or "").strip().lower()
    chart_type = str(getattr(presentation, "chart_type", "") or "").strip().lower()
    raw_phrase = str(getattr(presentation, "raw_format_phrase", "") or "").strip()
    raw_norm = norm_text(raw_phrase)
    if raw_phrase and any(k in raw_norm for k in ["bio clinical", "matrix", "comparative", "arithmetic"]):
        return raw_phrase
    if requested == "chart":
        if chart_type == "bar":
            return "graphique en barres"
        if chart_type == "line":
            return "courbe"
        if chart_type == "radar":
            return "graphique radar"
        if chart_type == "scatter":
            return "nuage de points"
        if raw_phrase and raw_phrase.lower() != "chart":
            return raw_phrase
        return "graphique"
    if raw_phrase and raw_phrase.lower() != "chart":
        return raw_phrase
    return requested or "format demandé"


def _ensure_chart_explanation(answer: str, query_understanding: QueryUnderstanding, visualization: dict[str, Any] | None) -> str:
    if str(query_understanding.output_format or "").strip().lower() != "chart":
        return answer
    text = str(answer or "").strip()
    norm = norm_text(text)

    if not visualization:
        return text

    requested_label = str(visualization.get("requested_label") or _humanize_requested_output(query_understanding)).strip()
    rendered_label = str(visualization.get("rendered_label") or _visualization_label(visualization.get("rendered_type"))).strip()
    fallback_used = bool(visualization.get("fallback_used"))
    fallback_reason = str(visualization.get("fallback_reason") or "").strip()
    recommendation_reason = str(visualization.get("recommendation_reason") or "").strip()
    metric_label = str(visualization.get("metric_label") or "écart normalisé à la référence").strip()
    metric_reason = str(visualization.get("metric_reason") or "").strip()
    rendered_type = str(visualization.get("rendered_type") or "").strip().lower()
    requested_type = str(visualization.get("requested_type") or "").strip().lower()
    requested_label_norm = norm_text(requested_label)
    has_visual_words = any(k in norm for k in ["graphique", "visualisation", "visualization", "line-graph", "line graph"])
    if has_visual_words and (not fallback_used or (requested_label_norm and requested_label_norm in norm)):
        return text

    from_previous = str(getattr(query_understanding, "intent", "")).strip().lower() == "response_transform"
    doc_scope = ", ".join(query_understanding.requested_doc_ids or [])
    context_phrase = (
        (f" à partir des résultats de {doc_scope}" if doc_scope else " à partir des résultats précédents")
        if from_previous
        else ""
    )

    if fallback_used:
        base_reason = fallback_reason or "Le format demandé n’est pas disponible tel quel."
        why_alt = recommendation_reason or f"Cette alternative permet une comparaison plus fiable via l’{metric_label.lower()}."
        prefix = (
            f"Vous avez demandé un {requested_label}{context_phrase}. "
            f"Ce format n’est pas rendu tel quel : {base_reason} "
            f"J’affiche donc un {rendered_label}. {why_alt}"
        ).strip()
    elif requested_type == "line" and rendered_type == "line" and not bool(visualization.get("suitable", True)):
        prefix = (
            f"Vous avez demandé une {requested_label}{context_phrase}. "
            "Cette courbe n’est pas affichée telle quelle ; les données sont normalisées pour éviter une comparaison trompeuse."
        ).strip()
    else:
        prefix = f"Voici le {rendered_label} généré à partir des résultats retrouvés{context_phrase}."
        if metric_reason:
            prefix += f" L’axe vertical représente l’{metric_label.lower()} car {metric_reason.lower()}"

    if not text:
        return prefix
    return f"{prefix}\n\n{text}"


def _inject_visualization_payload(
    result: dict[str, Any],
    *,
    query_understanding: QueryUnderstanding,
    displayed_evidences: list[dict[str, Any]],
) -> dict[str, Any]:
    out = dict(result)
    visualization, chart_data = _preview_visualization_payload(query_understanding, displayed_evidences)
    if not visualization:
        out["visualization"] = None
        out["chart_data"] = None
        return out
    out["visualization"] = visualization
    out["chart_data"] = chart_data
    mode = str(out.get("generation_mode") or "")
    answer_text = str(out.get("answer") or "")
    if not (mode.startswith("llm_professional_writer") or mode.startswith("llm_general_conversation") or mode == "specialized_visualization_composer"):
        out["answer"] = _ensure_chart_explanation(answer_text, query_understanding, visualization)
    else:
        out["answer"] = answer_text
    return out


def _preview_visualization_payload(
    query_understanding: QueryUnderstanding,
    evidences: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    presentation = getattr(query_understanding, "presentation_intent", None)
    if not presentation or not bool(getattr(presentation, "user_requested_visualization", False)):
        return None, None

    requested_type = _normalize_requested_visualization_type(
        getattr(presentation, "chart_type", None),
        getattr(presentation, "raw_format_phrase", None),
    )
    source = "previous_evidence_pack" if str(getattr(query_understanding, "intent", "")).strip().lower() == "response_transform" else "current_retrieval"
    visualization = build_visualization_payload(
        requested_type=requested_type,
        evidence_pack=evidences,
        supported_visualizations=[k for k, cfg in VISUALIZATION_REGISTRY.items() if bool(cfg.get("supported"))],
        raw_format_phrase=getattr(presentation, "raw_format_phrase", None),
        source=source,
    )
    chart_data = {
        "type": visualization.get("rendered_type"),
        "x_field": visualization.get("x_field"),
        "y_field": visualization.get("y_field"),
        "metric_label": visualization.get("metric_label"),
        "metric_reason": visualization.get("metric_reason"),
        "data": visualization.get("data"),
        "title": visualization.get("title"),
        "requested_type": visualization.get("requested_type"),
        "rendered_type": visualization.get("rendered_type"),
        "fallback_used": visualization.get("fallback_used"),
    }
    return visualization, chart_data


def _attach_visualization_facts_to_evidence_pack(
    *,
    query_understanding: QueryUnderstanding,
    evidence_pack: dict[str, Any],
    displayed_evidences: list[dict[str, Any]],
) -> dict[str, Any]:
    out = dict(evidence_pack or {})
    visualization, _ = _preview_visualization_payload(query_understanding, displayed_evidences)
    if visualization:
        out["visualization_facts"] = {
            "requested_type": visualization.get("requested_type"),
            "requested_label": visualization.get("requested_label"),
            "rendered_type": visualization.get("rendered_type"),
            "rendered_label": visualization.get("rendered_label"),
            "supported": visualization.get("supported"),
            "suitable": visualization.get("suitable"),
            "fallback_used": visualization.get("fallback_used"),
            "fallback_reason": visualization.get("fallback_reason"),
            "recommendation_reason": visualization.get("recommendation_reason"),
            "metric_label": visualization.get("metric_label"),
            "metric_reason": visualization.get("metric_reason"),
            "result_count": visualization.get("result_count"),
            "raw_format_phrase": getattr(getattr(query_understanding, "presentation_intent", None), "raw_format_phrase", None),
        }
    return out


def _query_understanding_payload(qu: QueryUnderstanding) -> dict[str, Any]:
    presentation = getattr(qu, "presentation_intent", None)
    return {
        "requested_doc_ids": list(qu.requested_doc_ids or []),
        "requested_analytes": list(qu.requested_analytes or []),
        "requested_value": qu.requested_value,
        "requested_unit": qu.requested_unit,
        "comparison_operator": qu.comparison_operator,
        "source_clickable_requested": qu.source_clickable_requested,
        "patient_query": qu.patient_query,
        "intent": qu.intent,
        "output_format": qu.output_format,
        "requested_table_columns": list(qu.requested_table_columns or []),
        "answer_style": qu.answer_style,
        "requires_global_search": qu.requires_global_search,
        "technical_condition": qu.technical_condition,
        "safety_intent": qu.safety_intent,
        "requires_previous_results": qu.requires_previous_results,
        "requires_comparison": qu.requires_comparison,
        "requires_section_summary": qu.requires_section_summary,
        "is_small_talk": qu.is_small_talk,
        "is_response_transform": qu.is_response_transform,
        "language": qu.language,
        "response_strategy": getattr(qu, "response_strategy", "render_table"),
        "response_strategy_reason": getattr(qu, "response_strategy_reason", None),
        "original_user_question": getattr(qu, "original_user_question", ""),
        "raw_user_request": getattr(qu, "raw_user_request", ""),
        "raw_format_phrase": getattr(qu, "raw_format_phrase", None),
        "unhandled_instructions": list(getattr(qu, "unhandled_instructions", []) or []),
        "presentation_confidence": float(getattr(qu, "presentation_confidence", 0.5)),
        "unsupported_presentation_reason": getattr(qu, "unsupported_presentation_reason", None),
        "recommended_alternative_format": getattr(qu, "recommended_alternative_format", None),
        "requested_date_iso": getattr(qu, "requested_date_iso", None),
        "requested_report_type": getattr(qu, "requested_report_type", None),
        "latest_report": bool(getattr(qu, "latest_report", False)),
        "requested_context_type": getattr(qu, "requested_context_type", None),
        "inventory_view_type": getattr(qu, "inventory_view_type", None),
        "qualitative_view_type": getattr(qu, "qualitative_view_type", None),
        "requested_reference_profile": getattr(qu, "requested_reference_profile", None),
        "use_patient_profile": bool(getattr(qu, "use_patient_profile", False)),
        "request_all_reference_ranges": bool(getattr(qu, "request_all_reference_ranges", False)),
        "requested_summary_points": getattr(qu, "requested_summary_points", None),
        "intent_arbitration": build_intent_arbitration_debug(qu),
        "presentation_intent": {
            "requested_output": getattr(presentation, "requested_output", qu.output_format),
            "chart_type": getattr(presentation, "chart_type", None),
            "requested_type": _normalize_requested_visualization_type(
                getattr(presentation, "chart_type", None),
                getattr(presentation, "raw_format_phrase", None),
            ),
            "requested_label": _visualization_label(
                _normalize_requested_visualization_type(
                    getattr(presentation, "chart_type", None),
                    getattr(presentation, "raw_format_phrase", None),
                )
            ),
            "raw_format_phrase": getattr(presentation, "raw_format_phrase", None),
            "wants_clickable_sources": bool(getattr(presentation, "wants_clickable_sources", qu.source_clickable_requested)),
            "wants_intro": bool(getattr(presentation, "wants_intro", True)),
            "wants_conclusion": bool(getattr(presentation, "wants_conclusion", True)),
            "strict_columns": list(getattr(presentation, "strict_columns", qu.requested_table_columns) or []),
            "unsupported_format": bool(getattr(presentation, "unsupported_format", False)),
            "user_requested_visualization": bool(getattr(presentation, "user_requested_visualization", False)),
            "presentation_confidence": float(getattr(presentation, "presentation_confidence", 0.5)),
            "unsupported_reason": getattr(presentation, "unsupported_reason", None),
            "recommended_output": getattr(presentation, "recommended_output", None),
            "unsupported_presentation_reason": getattr(presentation, "unsupported_presentation_reason", None),
            "recommended_alternative_format": getattr(presentation, "recommended_alternative_format", None),
            "unhandled_instructions": list(getattr(presentation, "unhandled_instructions", []) or []),
        },
    }


def _with_resolved_strategy(query_understanding: QueryUnderstanding, evidence_pack: dict[str, Any] | None) -> QueryUnderstanding:
    try:
        strategy = decide_response_strategy(query_understanding, evidence_pack or {})
        return replace(
            query_understanding,
            response_strategy=str(strategy.name or "render_table"),
            response_strategy_reason=str(strategy.reason or "") or None,
        )
    except Exception:
        return query_understanding


def _looks_like_transform_followup(query: str, query_understanding: QueryUnderstanding) -> bool:
    qn = norm_text(query or "")
    if not qn:
        return False
    has_doc_or_analyte = bool(query_understanding.requested_doc_ids or query_understanding.requested_analytes)
    if has_doc_or_analyte:
        return False
    markers = [
        "ok donne moi",
        "ok donne-moi",
        "maintenant donne moi",
        "maintenant donne-moi",
        "donne moi le resultat",
        "donne-moi le resultat",
        "affiche le resultat",
        "mets le resultat",
        "meme resultat",
        "resultat precedent",
        "reponse precedente",
        "meme reponse",
        "sous forme",
        "en graphique",
        "graphique en barres",
        "courbe",
        "line graph",
        "bar chart",
        "json strict",
        "json",
        "tableau",
    ]
    return any(m in qn for m in markers)


def _is_visualization_or_transform_request(query: str, query_understanding: QueryUnderstanding) -> bool:
    presentation = getattr(query_understanding, "presentation_intent", None)
    if bool(getattr(presentation, "user_requested_visualization", False)):
        return True
    if str(query_understanding.intent or "").strip().lower() == "response_transform":
        return True
    if str(query_understanding.output_format or "").strip().lower() == "chart":
        return True
    qn = norm_text(query or "")
    if not qn:
        return False
    viz_markers = [
        "radar",
        "chart",
        "graphique",
        "graphe",
        "visualisation",
        "courbe",
        "line graph",
        "bar chart",
        "diagramme",
    ]
    deictic_markers = ["affiche ca", "mets ca", "visualise ca", "ca ", "ça "]
    return any(m in qn for m in viz_markers) and any(m in qn for m in deictic_markers)


def _explicit_yes_no_requested(query: str) -> bool:
    qn = norm_text(query or "")
    markers = [
        "yes/no",
        "yes or no",
        "yes no",
        "answer only yes",
        "respond only yes",
        "yes ou no",
        "oui/non",
        "oui non",
        "oui ou non",
        "reponds uniquement oui",
        "réponds uniquement oui",
        "est ce que",
        "est-ce que",
    ]
    return any(m in qn for m in markers)


def _build_no_transformable_context_response(
    *,
    request_id: str,
    started: float,
    query: str,
    query_received: str,
    query_used_for_retrieval: str,
    query_used_for_prompt: str,
    top_k: int,
    max_display_results: int,
    show_all_results: bool,
    show_low_quality: bool,
    timeout: int,
    provider: str,
    model: str,
    query_understanding: QueryUnderstanding,
    intents: dict[str, bool],
    exact_analytes: list[str],
    requested_doc_ids: list[str],
    qn: str,
    previous_context_intent: str | None,
) -> dict[str, Any]:
    last_intent = str(previous_context_intent or "").strip().lower()
    if last_intent in {"patient_inventory", "patient_inventory_count"}:
        answer = (
            "Je n’ai pas de résultats biologiques numériques récents à transformer en visualisation. "
            "La dernière réponse concernait un inventaire de patients, pas des valeurs médicales transformables."
        )
    else:
        answer = "Je n’ai pas de résultat précédent exploitable à reformater. Veuillez d’abord demander les résultats à afficher, puis préciser le format souhaité."
    validation = validate_answer(
        query=query,
        answer_text=answer,
        evidence_pack=[],
        displayed_evidences=[],
        source_citations=[],
        generation_mode="deterministic_response_transform",
        retrieval_status="insufficient_context",
        query_received=query_received,
        query_used_for_retrieval=query_used_for_retrieval,
        query_used_for_prompt=query_used_for_prompt,
        query_stored=query,
        detected_analytes=exact_analytes,
        query_intents={**dict(intents or {}), "reference_range_lookup": True},
        output_format_requested=query_understanding.output_format,
        answer_style_requested=query_understanding.answer_style,
        requested_table_columns=query_understanding.requested_table_columns,
        requested_technical_condition=query_understanding.technical_condition,
        source_clickable_requested=bool(query_understanding.source_clickable_requested),
        requested_value=query_understanding.requested_value,
        comparison_operator=query_understanding.comparison_operator,
        raw_format_phrase=getattr(query_understanding, "raw_format_phrase", None),
        unsupported_presentation=bool(getattr(query_understanding.presentation_intent, "unsupported_format", False)),
        user_requested_visualization=bool(getattr(query_understanding.presentation_intent, "user_requested_visualization", False)),
        requested_chart_type=getattr(query_understanding.presentation_intent, "chart_type", None),
        visualization_payload=None,
        chart_data_payload=None,
        transformable_context_available=False,
        previous_intent=last_intent,
    )
    if str(generation_mode or "").strip().lower() == "llm" and str(validation.get("validation_status") or "fail") == "fail":
        final_answer = "Information insuffisante dans les données structurées disponibles pour répondre de façon fiable."
        generation_mode = "deterministic_safety_fallback_after_llm_validation_failure"
        llm_error = None
        validation = validate_answer(
            query=q,
            answer_text=final_answer,
            evidence_pack=evidence_pack,
            displayed_evidences=displayed_evidences,
            source_citations=source_citations,
            exact_analyte=exact_analyte,
            llm_error=None,
            generation_mode=generation_mode,
            retrieval_status=(retrieval_response.answerability or {}).get("status"),
            show_low_quality=show_low_quality,
            max_display_results=max_display_results,
            show_all_results=show_all_results,
            query_received=query_received,
            query_used_for_retrieval=query_used_for_retrieval,
            query_used_for_prompt=query_used_for_prompt,
            query_stored=q,
            detected_analytes=exact_analytes,
            requested_doc_id=requested_doc_id,
            requested_doc_ids=requested_doc_ids,
            missing_requested_doc_ids=missing_requested_doc_ids,
            requested_analytes=exact_analytes,
            found_requested_analytes=found_requested_analyte_norms,
            found_requested_analyte_norms=found_requested_analyte_norms,
            missing_requested_analytes=missing_requested_analytes,
            current_vs_previous_requested=query_understanding.requires_previous_results,
            diagnostic_safety_intent=bool(intents.get("diagnostic_safety_question")),
            query_intents=intents,
            output_format_requested=query_understanding.output_format,
            answer_style_requested=query_understanding.answer_style,
            requested_table_columns=query_understanding.requested_table_columns,
            requested_technical_condition=query_understanding.technical_condition,
            source_clickable_requested=bool(query_understanding.source_clickable_requested),
            requested_value=query_understanding.requested_value,
            comparison_operator=query_understanding.comparison_operator,
            raw_format_phrase=getattr(query_understanding, "raw_format_phrase", None),
            unsupported_presentation=bool(getattr(query_understanding.presentation_intent, "unsupported_format", False)),
            user_requested_visualization=bool(getattr(query_understanding.presentation_intent, "user_requested_visualization", False)),
            requested_chart_type=getattr(query_understanding.presentation_intent, "chart_type", None),
            visualization_payload=_preview_visualization_payload(query_understanding, displayed_evidences)[0],
            chart_data_payload=_preview_visualization_payload(query_understanding, displayed_evidences)[1],
        )
    quality = _quality_report(
        answer=answer,
        validation=validation,
        source_clickable_requested=bool(query_understanding.source_clickable_requested),
        recent_style_history=[],
    )
    intro_text, conclusion_text = _extract_intro_conclusion(answer)
    elapsed = time.perf_counter() - started
    return {
        "request_id": request_id,
        "query": query,
        "query_received": query_received,
        "query_used_for_retrieval": query_used_for_retrieval,
        "query_used_for_prompt": query_used_for_prompt,
        "query_stored": query,
        "normalized_query": query,
        "mode": "response_transform",
        "provider": provider,
        "model": model,
        "top_k": top_k,
        "max_display_results": int(max_display_results),
        "show_all_results": bool(show_all_results),
        "show_low_quality": bool(show_low_quality),
        "timeout": timeout,
        "generation_time_seconds": round(elapsed, 3),
        "answer": answer,
        "citations": [],
        "sources": [],
        "validation": validation,
        "quality_report": quality,
        "llm_error": None,
        "error_type": None,
        "generation_mode": "deterministic_response_transform",
        "detected_analytes": exact_analytes,
        "query_understanding": _query_understanding_payload(query_understanding),
        "structured_evidence_pack": {},
        "evidence_pack": [],
        "displayed_evidences": [],
        "display": {
            "selected_candidates_count": 0,
            "low_quality_evidence_filtered_count": 0,
            "hidden_result_count": 0,
            "requested_multi_result_query": _query_requests_multiple_results(qn),
            "display_notes": [],
        },
        "retrieval": {
            "answerability": {"status": "insufficient_context", "reason": "no_previous_transformable_context"},
            "filters": {"doc_ids": requested_doc_ids, "analytes": exact_analytes},
            "top_results": [],
            "context_chunks": [],
            "sources": [],
        },
        "prompt": "",
        "style_memory_entry": {
            "intro_text": intro_text,
            "conclusion_text": conclusion_text,
            "intent": "response_transform",
            "output_format": query_understanding.output_format,
            "answer_text": answer,
        },
        "debug": {
            "request_id": request_id,
            "generation_mode": "deterministic_response_transform",
            "generation_writer": "professional_fallback",
            "intents": intents,
            "retrieval_skipped_due_to_no_transformable_context": True,
            "visualization_request_detected": True,
        },
        "visualization": None,
        "chart_data": None,
    }


def _build_visualization_recommendation_response(
    *,
    request_id: str,
    started: float,
    query: str,
    query_received: str,
    query_used_for_retrieval: str,
    query_used_for_prompt: str,
    top_k: int,
    max_display_results: int,
    show_all_results: bool,
    show_low_quality: bool,
    timeout: int,
    provider: str,
    model: str,
    query_understanding: QueryUnderstanding,
    intents: dict[str, bool],
    previous_context_intent: str | None,
    previous_data_context_intent: str | None,
    previous_data_context_type: str | None,
    previous_has_patient_inventory: bool,
    has_transformable_context: bool,
) -> dict[str, Any]:
    last_intent = str(previous_context_intent or "").strip().lower()
    data_context_intent = str(previous_data_context_intent or "").strip().lower()
    data_context_type = str(previous_data_context_type or "").strip().lower()
    qn = norm_text(query or "")
    asks_about_comment = any(token in qn for token in ["ce commentaire", "ce comment", "cette note", "note interpretative", "note interprétative"])
    if (
        data_context_type == "medical_qualitative_comment"
        or data_context_intent == "comment_without_measured_value"
        or asks_about_comment
    ):
        answer = (
            "Pour un commentaire médical qualitatif, la présentation la plus fiable est textuelle et sourcée.\n"
            "Je recommande :\n"
            "1. une carte d’information médicale avec le commentaire principal ;\n"
            "2. un bloc commentaire sourcé (document, page, ligne) ;\n"
            "3. un tableau texte : sujet, commentaire, source ;\n"
            "4. un encadré de note interprétative."
        )
    elif (
        data_context_type == "patient_inventory"
        or data_context_intent == "patient_inventory"
        or previous_has_patient_inventory
        or last_intent in {"patient_inventory", "patient_inventory_count", "response_transform_no_context"}
    ):
        answer = (
            "Ces données correspondent à un inventaire de patients et de rapports, pas à des résultats biologiques numériques.\n"
            "La visualisation la plus adaptée est donc une vue d’inventaire plutôt qu’un graphique clinique.\n\n"
            "Je recommande :\n"
            "1. des cartes patient avec le nombre de rapports associés ;\n"
            "2. une liste accordéon pour ouvrir les rapports de chaque patient ;\n"
            "3. une table filtrable par patient, date ou nom de fichier ;\n"
            "4. une timeline documentaire pour suivre l’ordre des rapports.\n\n"
            "Un radar chart, une courbe ou un graphique en barres clinique ne sont pas adaptés ici, car il n’y a pas de valeurs biologiques numériques comparables."
        )
    elif has_transformable_context:
        answer = (
            "Pour des résultats biologiques numériques, je recommande en priorité :\n"
            "1. un graphique en barres pour comparer plusieurs analytes à une même date ;\n"
            "2. une courbe seulement s’il existe une vraie série temporelle ;\n"
            "3. un tableau avec statuts et sources cliquables pour la validation clinique.\n\n"
            "Le radar chart n’est pertinent que si les valeurs sont normalisées et comparables entre elles."
        )
    else:
        answer = (
            "Je peux recommander une visualisation adaptée, mais je n’ai pas de résultats biologiques numériques récents dans ce contexte.\n"
            "Demandez d’abord les résultats à afficher, puis je proposerai le format le plus fiable."
        )

    validation = validate_answer(
        query=query,
        answer_text=answer,
        evidence_pack=[],
        displayed_evidences=[],
        source_citations=[],
        generation_mode="deterministic_visualization_recommendation",
        retrieval_status="not_required",
        query_received=query_received,
        query_used_for_retrieval=query_used_for_retrieval,
        query_used_for_prompt=query_used_for_prompt,
        query_stored=query,
        detected_analytes=[],
        query_intents={**dict(intents or {}), "reference_range_lookup": True},
        output_format_requested="paragraph",
        answer_style_requested="standard",
        requested_table_columns=[],
        requested_technical_condition=None,
        source_clickable_requested=bool(getattr(query_understanding, "source_clickable_requested", False)),
        requested_value=None,
        comparison_operator=None,
        raw_format_phrase=getattr(query_understanding, "raw_format_phrase", None),
        unsupported_presentation=False,
        user_requested_visualization=False,
        requested_chart_type=None,
        visualization_payload=None,
        chart_data_payload=None,
        transformable_context_available=has_transformable_context,
        previous_intent=last_intent,
    )
    quality = _quality_report(
        answer=answer,
        validation=validation,
        source_clickable_requested=False,
        recent_style_history=[],
    )
    intro_text, conclusion_text = _extract_intro_conclusion(answer)
    elapsed = time.perf_counter() - started
    return {
        "request_id": request_id,
        "query": query,
        "query_received": query_received,
        "query_used_for_retrieval": query_used_for_retrieval,
        "query_used_for_prompt": query_used_for_prompt,
        "query_stored": query,
        "normalized_query": query,
        "mode": "visualization_recommendation",
        "provider": provider,
        "model": model,
        "top_k": top_k,
        "max_display_results": int(max_display_results),
        "show_all_results": bool(show_all_results),
        "show_low_quality": bool(show_low_quality),
        "timeout": timeout,
        "generation_time_seconds": round(elapsed, 3),
        "answer": answer,
        "citations": [],
        "sources": [],
        "validation": validation,
        "quality_report": quality,
        "llm_error": None,
        "error_type": None,
        "generation_mode": "deterministic_visualization_recommendation",
        "detected_analytes": [],
        "query_understanding": _query_understanding_payload(query_understanding),
        "structured_evidence_pack": {},
        "evidence_pack": [],
        "displayed_evidences": [],
        "display": {
            "selected_candidates_count": 0,
            "low_quality_evidence_filtered_count": 0,
            "hidden_result_count": 0,
            "requested_multi_result_query": False,
            "display_notes": [],
        },
        "retrieval": {
            "answerability": {"status": "not_required", "reason": "visualization_recommendation_no_retrieval"},
            "filters": {"doc_ids": [], "analytes": []},
            "top_results": [],
            "context_chunks": [],
            "sources": [],
        },
        "prompt": "",
        "style_memory_entry": {
            "intro_text": intro_text,
            "conclusion_text": conclusion_text,
            "intent": "visualization_recommendation",
            "output_format": "paragraph",
            "answer_text": answer,
        },
        "debug": {
            "request_id": request_id,
            "generation_mode": "deterministic_visualization_recommendation",
            "generation_writer": "professional_fallback",
            "intents": intents,
            "retrieval_skipped": True,
            "visualization_request_detected": True,
        },
        "visualization": None,
        "chart_data": None,
    }


def _build_inventory_visualization_render_response(
    *,
    request_id: str,
    started: float,
    query: str,
    query_received: str,
    query_used_for_retrieval: str,
    query_used_for_prompt: str,
    top_k: int,
    max_display_results: int,
    show_all_results: bool,
    show_low_quality: bool,
    timeout: int,
    provider: str,
    model: str,
    query_understanding: QueryUnderstanding,
    intents: dict[str, bool],
    previous_has_patient_inventory: bool,
    previous_patient_inventory: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    patients = list(previous_patient_inventory or [])
    view_type = str(getattr(query_understanding, "inventory_view_type", "") or "patient_cards").strip().lower()
    if view_type not in {"patient_cards", "report_accordion", "filterable_table", "document_timeline"}:
        view_type = "patient_cards"
    if previous_has_patient_inventory and patients:
        qn = norm_text(query or "")
        asks_radar = "radar" in qn
        if view_type == "report_accordion":
            answer = "D’accord. J’affiche l’inventaire sous forme de liste accordéon, afin d’ouvrir les rapports associés à chaque patient."
        elif view_type == "filterable_table":
            answer = "D’accord. J’affiche l’inventaire sous forme de table structurée prête à être filtrée par patient, date ou nom de fichier."
        elif view_type == "document_timeline":
            answer = "Cette vue n’est pas encore implémentée dans l’interface. J’affiche temporairement la vue cartes patient."
        else:
            if asks_radar:
                answer = (
                    "Un inventaire patient n’est pas transformable en radar chart. "
                    "J’affiche plutôt une vue d’inventaire adaptée.\n\n"
                    "D’accord. J’affiche l’inventaire sous forme de cartes patient, avec le nombre de rapports associés pour chaque patient."
                )
            else:
                answer = "D’accord. J’affiche l’inventaire sous forme de cartes patient, avec le nombre de rapports associés pour chaque patient."
    else:
        answer = "Je n’ai pas trouvé d’inventaire patient récent à afficher sous cette forme. Demandez d’abord la liste des patients."
    validation = validate_answer(
        query=query,
        answer_text=answer,
        evidence_pack=[],
        displayed_evidences=[],
        source_citations=[],
        generation_mode="deterministic_inventory_visualization_render",
        retrieval_status="not_required",
        query_received=query_received,
        query_used_for_retrieval=query_used_for_retrieval,
        query_used_for_prompt=query_used_for_prompt,
        query_stored=query,
        detected_analytes=[],
        query_intents=intents,
        output_format_requested="paragraph",
        answer_style_requested="standard",
        requested_table_columns=[],
        requested_technical_condition=None,
        source_clickable_requested=False,
        requested_value=None,
        comparison_operator=None,
        visualization_payload=None,
        chart_data_payload=None,
        transformable_context_available=False,
        previous_intent="patient_inventory" if previous_has_patient_inventory else "",
        patients=patients if patients else None,
        inventory_view={"type": view_type},
    )
    quality = _quality_report(answer=answer, validation=validation, source_clickable_requested=False, recent_style_history=[])
    intro_text, conclusion_text = _extract_intro_conclusion(answer)
    elapsed = time.perf_counter() - started
    return {
        "request_id": request_id,
        "query": query,
        "query_received": query_received,
        "query_used_for_retrieval": query_used_for_retrieval,
        "query_used_for_prompt": query_used_for_prompt,
        "query_stored": query,
        "normalized_query": query,
        "mode": "inventory_visualization_render",
        "provider": provider,
        "model": model,
        "top_k": top_k,
        "max_display_results": int(max_display_results),
        "show_all_results": bool(show_all_results),
        "show_low_quality": bool(show_low_quality),
        "timeout": timeout,
        "generation_time_seconds": round(elapsed, 3),
        "answer": answer,
        "citations": [],
        "sources": [],
        "validation": validation,
        "quality_report": quality,
        "llm_error": None,
        "error_type": None,
        "generation_mode": "deterministic_inventory_visualization_render",
        "detected_analytes": [],
        "query_understanding": _query_understanding_payload(query_understanding),
        "structured_evidence_pack": {},
        "evidence_pack": [],
        "displayed_evidences": [],
        "patients": patients if patients else None,
        "inventory_view": {"type": view_type},
        "prompt": "",
        "style_memory_entry": {
            "intro_text": intro_text,
            "conclusion_text": conclusion_text,
            "intent": "inventory_visualization_render",
            "output_format": "paragraph",
            "answer_text": answer,
        },
        "debug": {
            "request_id": request_id,
            "generation_mode": "deterministic_inventory_visualization_render",
            "generation_writer": "professional_fallback",
            "intents": intents,
            "retrieval_skipped": True,
        },
        "visualization": None,
        "chart_data": None,
    }


def _is_above_reference_query(qn: str) -> bool:
    return any(
        k in qn
        for k in [
            "au dessus de la reference",
            "au-dessus de la reference",
            "superieur a la reference",
            "superieure a la reference",
            "above reference",
            "above_reference",
            "superieur",
            "supérieure",
            "superieure",
            "eleve",
            "elevee",
            "élevé",
            "élevée",
        ]
    )


def _is_normal_or_above_query(qn: str) -> bool:
    return ("normale ou superieure" in qn) or ("normal or above" in qn)


def _is_below_reference_query(qn: str) -> bool:
    return any(
        k in qn
        for k in [
            "inferieur a la reference",
            "inferieure a la reference",
            "en dessous de la reference",
            "below reference",
            "below_reference",
            "inferieur",
            "inférieure",
            "inferieure",
            "bas",
            "basse",
            "sous la reference",
        ]
    )


def _is_previous_result_query(qn: str) -> bool:
    return any(
        k in qn
        for k in [
            "resultat anterieur",
            "resultats anterieurs",
            "previous result",
            "previous results",
            "ancien resultat",
            "anciens resultats",
            "anterieur",
            "anterieurs",
        ]
    )


def _is_compare_query(qn: str) -> bool:
    if not ("compare" in qn or "compar" in qn):
        return False
    if "actuel" in qn and ("anterieur" in qn or "previous" in qn):
        return True
    if "anterieur" in qn or "previous" in qn:
        return True
    return False


def _is_status_query(qn: str) -> bool:
    return "statut technique" in qn or "interpretation technique" in qn


def _is_global_above_reference_query(qn: str, exact_analytes: list[str]) -> bool:
    if exact_analytes:
        return False
    if not _is_above_reference_query(qn):
        return False
    return any(k in qn for k in ["quels resultats", "quelles", "liste", "tous", "resultats sont", "valeur de reference"])


def _query_requests_multiple_results(qn: str) -> bool:
    return any(k in qn for k in ["tous", "toutes", "liste", "retrouves", "retrouvés", "documents"])


def _query_requests_out_of_reference_only(qn: str) -> bool:
    return any(
        k in qn
        for k in [
            "hors reference",
            "hors de la reference",
            "outside reference",
            "out of reference",
            "en dehors de la reference",
            "anomalie biologique",
            "anomalies biologiques",
            "anomalie",
            "anomalies",
            "valeurs anormales",
            "anormaux",
            "anormales",
            "attention technique",
        ]
    ) or (_is_above_reference_query(qn) and "reference" in qn) or (_is_below_reference_query(qn) and "reference" in qn)


def is_valid_analyte_name(analyte: str) -> bool:
    text = str(analyte or "").strip()
    if not text:
        return False
    norm = norm_text(text)
    if len(norm) > 42:
        return False
    if ":" in text and len(text) > 25:
        return False
    if any(
        marker in norm
        for marker in [
            "augmentation de",
            "associes",
            "apres un infarctus",
            "acromegalie",
            "commentaire",
            "interpretation",
            "valeurs de reference",
            "technique",
            "dosage",
            "agenda biochimie",
            "resultats biologiques",
            "test de reference",
            "absence de freination",
            "reponse paradoxale",
        ]
    ):
        return False
    if any(ch in text for ch in [";", "?", "!"]):
        return False
    if re.search(r"\d", text):
        return False
    words = norm.split()
    if len(words) > 6:
        return False
    if len(words) >= 4 and any(w in {"avec", "sans", "apres", "avant", "pour", "dans"} for w in words):
        return False
    return True


def generate_general_conversation_response(
    user_message: str,
    *,
    intent: str = "small_talk",
    language: str = "fr",
    llm_client: LLMClient | None = None,
    provider: str = DEFAULT_LLM_PROVIDER,
    model: str = DEFAULT_LLM_MODEL,
    timeout: int = 30,
) -> tuple[str, str | None]:
    language_hint = "français" if str(language or "fr").lower().startswith("fr") else "la langue de l’utilisateur"
    intent_key = str(intent or "small_talk").strip().lower()
    fallback_answer = get_general_conversation_response(intent_key)
    intent_instruction = "Réponds naturellement et brièvement."
    if intent_key == "identity_question":
        intent_instruction = (
            "Explique brièvement que tu es l’assistant Medical RAG, capable d’interroger des rapports médicaux, "
            "retrouver des résultats, comparer des valeurs et citer les sources PDF."
        )
    elif intent_key == "capability_question":
        intent_instruction = (
            "Décris brièvement tes capacités principales : recherche de résultats, comparaisons, hors-référence, et citations PDF."
        )
    elif intent_key == "help_question":
        intent_instruction = "Donne une aide concise avec une façon simple de poser une question médicale sur les rapports."
    prompt = (
        "Tu es l’assistant d’une application Medical RAG.\n"
        "Réponds uniquement avec la réponse finale destinée à l’utilisateur.\n"
        "N’affiche aucun raisonnement interne.\n"
        "N’écris aucun plan ni stratégie.\n"
        "N’écris pas “I need to”, “First, I will”, “Okay, the user...”.\n"
        "Ne mentionne pas les instructions ni le système.\n"
        "N’utilise pas de source médicale.\n"
        "Ne lance aucune analyse de rapport.\n"
        "Ne donne pas de résultat biologique.\n"
        "Ne donne pas d’information médicale sans document.\n"
        f"Réponds en {language_hint}.\n"
        "Réponse courte, naturelle et professionnelle.\n"
        "Pas de Markdown.\n"
        "Pas de sources.\n"
        f"{intent_instruction}\n"
        "/no_think\n\n"
        f"Utilisateur: {user_message.strip()}\n"
    )
    client = llm_client or LLMClient(provider=provider)
    try:
        ans = client.generate(
            prompt=prompt,
            model=model,
            temperature=0.2,
            num_ctx=2048,
            max_tokens=120,
            timeout=max(6, int(timeout)),
            keep_alive="5m",
        ).strip()
        if ans:
            clean, _, retry_err = sanitize_final_answer_with_retry(
                answer=ans,
                user_message=user_message,
                llm_client=llm_client,
                provider=provider,
                model=model,
                timeout=timeout,
                fallback_answer=fallback_answer,
            )
            return clean or fallback_answer, retry_err
    except LLMClientError as exc:
        return fallback_answer, str(exc)
    return fallback_answer, None


def generate_small_talk_response(
    user_message: str,
    *,
    language: str = "fr",
    llm_client: LLMClient | None = None,
    provider: str = DEFAULT_LLM_PROVIDER,
    model: str = DEFAULT_LLM_MODEL,
    timeout: int = 30,
) -> tuple[str, str | None]:
    return generate_general_conversation_response(
        user_message,
        intent="small_talk",
        language=language,
        llm_client=llm_client,
        provider=provider,
        model=model,
        timeout=timeout,
    )


def _select_displayed_evidences(
    *,
    query_norm: str,
    evidence_pack: list[dict[str, Any]],
    exact_analyte: str | None,
    requested_analytes: list[str] | None,
    max_display_results: int,
    show_all_results: bool,
    show_low_quality: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = _select_deterministic_candidates(
        query_norm=query_norm,
        evidence_pack=evidence_pack,
        exact_analyte=exact_analyte,
        requested_analytes=requested_analytes,
    )
    if not selected and not exact_analyte:
        selected = list(evidence_pack)

    low_quality_filtered_count = 0
    if show_low_quality:
        quality_filtered = selected
    else:
        quality_filtered = [ev for ev in selected if str(ev.get("evidence_display_quality") or "high") != "low"]
        low_quality_filtered_count = max(0, len(selected) - len(quality_filtered))

    if show_all_results:
        displayed = list(quality_filtered)
    else:
        displayed = list(quality_filtered[: max(1, int(max_display_results))])

    hidden_result_count = max(0, len(quality_filtered) - len(displayed))
    notes: list[str] = []
    if hidden_result_count > 0 and not show_all_results:
        notes.append(
            f"Plusieurs résultats existent pour cet analyte ; seuls les {len(displayed)} premiers sont affichés. "
            "Utilisez --show-all-results pour tout afficher."
        )

    return displayed, {
        "selected_candidates_count": len(selected),
        "low_quality_evidence_filtered_count": low_quality_filtered_count,
        "hidden_result_count": hidden_result_count,
        "requested_multi_result_query": _query_requests_multiple_results(query_norm),
        "display_notes": notes,
    }


def _normalize_summary_point(point: str) -> str:
    return " ".join(str(point or "").strip().split())


def deduplicate_summary_points(points: list[str]) -> list[str]:
    out: list[str] = []
    seen: list[str] = []
    for raw in points or []:
        p = _normalize_summary_point(raw)
        if not p:
            continue
        norm = norm_text(p)
        if not norm:
            continue
        duplicate = norm in seen
        if not duplicate:
            for s in seen:
                if len(s) >= 20 and len(norm) >= 20 and (norm in s or s in norm):
                    duplicate = True
                    break
        if duplicate:
            continue
        seen.append(norm)
        out.append(p)
    return out


def _split_text_into_summary_candidates(text: str) -> list[str]:
    compact = " ".join(str(text or "").replace("\n", " ").split()).strip()
    if not compact:
        return []
    candidates: list[str] = []
    # First pass: sentence-like chunks.
    for seg in re.split(r"(?<=[\.\!\?])\s+", compact):
        s = seg.strip(" -•\t")
        if s:
            candidates.append(s)
    # Second pass for one-line comments: split on semantic separators.
    if len(candidates) <= 1:
        for seg in re.split(r"\s+(?=Attention\s*:|Valeur\s+seuil\s*:?)", compact, flags=re.IGNORECASE):
            s = seg.strip(" -•\t")
            if s:
                candidates.append(s)
    if len(candidates) <= 1:
        for seg in re.split(r"\s*;\s*", compact):
            s = seg.strip(" -•\t")
            if s:
                candidates.append(s)
    # Third pass for long one-line qualitative comments: extract semantic blocks.
    if len(candidates) <= 1:
        blocks: list[str] = []
        seuil = re.search(r"(valeur\s+seuil[^:]{0,120}:\s*[^:]+?)(?=\s+attention\s*:|$)", compact, flags=re.IGNORECASE)
        if seuil:
            blocks.append(seuil.group(1).strip(" .;:"))
        alert = re.search(r"(attention\s*:\s*[^:]+)(?::\s*|$)", compact, flags=re.IGNORECASE)
        if alert:
            blocks.append(alert.group(1).strip(" .;:"))
            tail = compact[alert.end() :].strip(" .;:")
            if tail:
                blocks.append(f"Situations mentionnées : {tail}")
        if blocks:
            candidates.extend(blocks)
    return deduplicate_summary_points(candidates)


def _summary_limitation_text(found_points: int) -> str:
    n = max(0, int(found_points))
    return (
        f"Je ne peux extraire que {n} point{'s' if n > 1 else ''} distinct"
        f"{'s' if n > 1 else ''} à partir du contexte disponible."
    )


def _format_summary_answer(
    *,
    context_label: str,
    requested_points: int,
    points: list[str],
    limitation: str | None,
    sources: list[dict[str, Any]],
    clickable_requested: bool,
    include_inline_source: bool = False,
) -> str:
    lines = [f"Résumé en {requested_points} point{'s' if requested_points > 1 else ''} du {context_label}", ""]
    for idx, p in enumerate(points, start=1):
        lines.append(f"{idx}. {p}")
    if limitation:
        lines.append("")
        lines.append(limitation)
    if include_inline_source and sources:
        primary = sources[0]
        label = str(primary.get("label") or "source non disponible").strip()
        md, has_click = format_clickable_source_markdown(
            label,
            str(primary.get("viewer_url") or "").strip() or None,
            str(primary.get("source_url") or primary.get("url") or "").strip() or None,
        )
        source_render = md if has_click else label
        if clickable_requested and not has_click:
            source_render = f"{label} (source non cliquable disponible uniquement en texte)"
        lines.extend(["", f"Source : {source_render}"])
    return "\n".join(lines).strip()


def _collect_sources_from_pack(pack: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(pack, dict):
        return []
    srcs: list[dict[str, Any]] = []
    for src in list(pack.get("sources") or []):
        if isinstance(src, dict):
            srcs.append(normalize_source_for_response(src))
    rows = list(pack.get("evidences") or pack.get("results") or [])
    for row in rows:
        if not isinstance(row, dict):
            continue
        srcs.append(
            normalize_source_for_response(
                {
                    "label": row.get("source_label") or row.get("source"),
                    "source_pdf": row.get("source_pdf"),
                    "doc_id": row.get("doc_id"),
                    "page": row.get("page") if row.get("page") is not None else row.get("page_number"),
                    "line": row.get("line") if row.get("line") is not None else row.get("row"),
                    "viewer_url": row.get("viewer_url"),
                    "source_url": row.get("source_url"),
                }
            )
        )
    deduped = dedup_sources_for_qualitative(srcs)
    # Never expose internal placeholders when a real PDF source exists.
    def _is_internal_source(s: dict[str, Any]) -> bool:
        lbl = norm_text(str(s.get("label") or ""))
        return lbl in {"sqlite deterministic", "sqlite_deterministic", "sqlite"} or "sqlite deterministic" in lbl

    has_pdf_source = any(
        str(s.get("source_pdf") or "").strip()
        or str(s.get("label") or "").strip().lower().endswith(".pdf")
        or ".pdf — page" in str(s.get("label") or "").strip().lower()
        for s in deduped
    )
    if has_pdf_source:
        deduped = [s for s in deduped if not _is_internal_source(s)]
    # Prefer precise citation (page/line) first.
    deduped.sort(
        key=lambda s: (
            0 if str(s.get("source_pdf") or "").strip() else 1,
            0 if isinstance(s.get("page"), int) else 1,
            0 if isinstance(s.get("line"), int) else 1,
            str(s.get("label") or ""),
        )
    )
    return deduped


def _try_llm_grounded_summary(
    *,
    llm_client: LLMClient | None,
    provider: str,
    model: str,
    timeout: int,
    context_type: str,
    subject: str,
    display_text: str,
    evidence_pack: dict[str, Any] | None,
    sources: list[dict[str, Any]],
    requested_summary_points: int,
) -> tuple[list[str] | None, str | None]:
    if llm_client is None:
        return None, "no_llm_client"
    payload = {
        "context_type": context_type,
        "subject": subject,
        "display_text": display_text,
        "evidence_pack": evidence_pack or {},
        "sources": sources,
        "requested_summary_points": requested_summary_points,
    }
    prompt = (
        "Tu es un assistant de synthèse médicale.\n"
        "Tu dois résumer uniquement le CONTEXTE FOURNI.\n"
        "Tu ne dois pas ajouter de connaissance médicale externe.\n"
        "Tu ne dois pas poser de diagnostic.\n"
        "Tu ne dois pas inventer de valeur, source, patient, rapport ou interprétation.\n"
        "Tu dois produire exactement N points si le contexte contient assez d’informations distinctes.\n"
        "Si le contexte ne contient pas assez d’informations distinctes, produis seulement les points fiables et indique la limite.\n"
        "Chaque point doit être court, non redondant, et fondé sur le contexte.\n"
        "Réponds en JSON strict uniquement avec ce schéma:\n"
        '{"title":"...","points":["..."],"limitations":null}\n'
        f"N={requested_summary_points}\n\n"
        f"CONTEXTE:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    client = llm_client or LLMClient(provider=provider)
    try:
        raw = client.generate(
            prompt=prompt,
            model=model,
            temperature=0.0,
            num_ctx=3072,
            max_tokens=420,
            timeout=max(8, int(timeout)),
            keep_alive="5m",
        ).strip()
    except Exception as exc:  # pragma: no cover - network/runtime dependent
        return None, str(exc)
    if not raw:
        return None, "empty_llm_summary"
    data: dict[str, Any] | None = None
    try:
        data = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                data = None
    if not isinstance(data, dict):
        return None, "invalid_llm_summary_json"
    pts = data.get("points")
    if not isinstance(pts, list):
        return None, "invalid_llm_summary_points"
    cleaned = deduplicate_summary_points([str(p) for p in pts if str(p).strip()])[: max(1, int(requested_summary_points))]
    if not cleaned:
        return None, "empty_llm_summary_points"
    limitation = str(data.get("limitations") or "").strip() or None
    return cleaned, limitation


def _build_context_summary_render_response(
    *,
    request_id: str,
    started: float,
    query: str,
    query_received: str,
    query_used_for_retrieval: str,
    query_used_for_prompt: str,
    top_k: int,
    max_display_results: int,
    show_all_results: bool,
    show_low_quality: bool,
    timeout: int,
    provider: str,
    model: str,
    query_understanding: QueryUnderstanding,
    intents: dict[str, bool],
    previous_displayed_context: dict[str, Any] | None,
    previous_qualitative_evidence_pack: dict[str, Any] | None,
    previous_transformable_pack: dict[str, Any] | None,
    previous_patient_inventory: list[dict[str, Any]] | None,
    previous_data_context_type: str | None,
    llm_client: LLMClient | None,
) -> dict[str, Any]:
    requested_points = int(getattr(query_understanding, "requested_summary_points", None) or 3)
    requested_points = max(1, min(10, requested_points))
    wants_clickable = bool(getattr(query_understanding, "source_clickable_requested", False))

    ctx = previous_displayed_context if isinstance(previous_displayed_context, dict) else {}
    context_type = str(ctx.get("context_type") or previous_data_context_type or "none").strip().lower()
    if context_type not in {"medical_qualitative_comment", "biological_numeric_results", "patient_inventory"}:
        context_type = "none"

    points: list[str] = []
    limitation: str | None = None
    answer: str
    sources: list[dict[str, Any]] = []
    structured_pack: dict[str, Any] = {}

    if context_type == "medical_qualitative_comment":
        pack = previous_qualitative_evidence_pack if isinstance(previous_qualitative_evidence_pack, dict) else {}
        structured_pack = pack
        rows = list(pack.get("evidences") or pack.get("results") or [])
        first = rows[0] if rows and isinstance(rows[0], dict) else {}
        subject = resolve_medical_subject(
            query_understanding=query_understanding,
            evidence=first if isinstance(first, dict) else None,
            state_context=ctx,
            query=query,
        )
        comment_text = str(
            first.get("display_comment_text")
            or first.get("comment_text")
            or first.get("current_value")
            or ctx.get("display_text")
            or ""
        ).strip()
        candidates = _split_text_into_summary_candidates(comment_text)
        points = candidates[:requested_points]
        if not points:
            answer = "Je n’ai pas de commentaire qualitatif récent à résumer. Demandez d’abord le commentaire concerné."
            sources = []
        else:
            llm_points, llm_limitation = _try_llm_grounded_summary(
                llm_client=llm_client,
                provider=provider,
                model=model,
                timeout=timeout,
                context_type=context_type,
                subject=subject,
                display_text=comment_text,
                evidence_pack=pack,
                sources=_collect_sources_from_pack(pack),
                requested_summary_points=requested_points,
            )
            if llm_points:
                # Do not degrade quality if LLM returns fewer distinct points than deterministic extraction.
                if len(llm_points) >= len(points):
                    points = llm_points
                    limitation = llm_limitation
            if len(points) < requested_points:
                limitation = _summary_limitation_text(len(points))
            sources = _collect_sources_from_pack(pack)
            answer = _format_summary_answer(
                context_label=f"commentaire sur {subject}",
                requested_points=requested_points,
                points=points,
                limitation=limitation,
                sources=sources,
                clickable_requested=wants_clickable,
                include_inline_source=False,
            )
    elif context_type == "biological_numeric_results":
        pack = previous_transformable_pack if isinstance(previous_transformable_pack, dict) else {}
        structured_pack = pack
        rows = [r for r in list(pack.get("evidences") or pack.get("results") or []) if isinstance(r, dict)]
        if not rows:
            answer = "Je n’ai pas de résultats biologiques récents à résumer. Demandez d’abord les résultats à afficher."
            sources = []
        else:
            if len(rows) == 1:
                row = rows[0]
                analyte = str(row.get("analyte") or row.get("parameter") or "Analyte").strip()
                value = str(row.get("current_value") or row.get("value_raw") or row.get("value_numeric") or "").strip()
                unit = str(row.get("unit") or "").strip()
                reference = str(row.get("reference_range") or row.get("reference") or "").strip()
                status = str(row.get("technical_status") or row.get("status") or row.get("interpretation_status") or "").strip()
                if value:
                    points.append(f"{analyte} = {value}{(' ' + unit) if unit else ''}.")
                if reference:
                    points.append(f"Intervalle de référence : {reference}.")
                if status:
                    points.append(f"Statut technique : {status}.")
            else:
                points.append(f"{len(rows)} résultats biologiques ont été affichés.")
                for row in rows:
                    analyte = str(row.get("analyte") or row.get("parameter") or "").strip()
                    if not analyte:
                        continue
                    value = str(row.get("current_value") or row.get("value_raw") or row.get("value_numeric") or "").strip()
                    status = str(row.get("technical_status") or row.get("status") or row.get("interpretation_status") or "").strip()
                    bits = [analyte]
                    if value:
                        bits.append(value)
                    if status:
                        bits.append(status)
                    points.append(" : ".join([bits[0], " | ".join(bits[1:])]) if len(bits) > 1 else bits[0])
                    if len(points) >= requested_points:
                        break
            points = deduplicate_summary_points(points)[:requested_points]
            if not points:
                answer = "Je n’ai pas assez d’informations numériques exploitables pour produire un résumé fiable."
                sources = []
            else:
                if len(points) < requested_points:
                    limitation = _summary_limitation_text(len(points))
                sources = _collect_sources_from_pack(pack)
                answer = _format_summary_answer(
                    context_label="résultats biologiques affichés",
                    requested_points=requested_points,
                    points=points,
                    limitation=limitation,
                    sources=sources,
                    clickable_requested=wants_clickable,
                    include_inline_source=False,
                )
    elif context_type == "patient_inventory":
        inventory = list(previous_patient_inventory or [])
        if not inventory:
            answer = "Je n’ai pas d’inventaire patient récent à résumer. Demandez d’abord la liste des patients."
            sources = []
        else:
            points.append(f"{len(inventory)} patient{'s' if len(inventory) > 1 else ''} sont présents dans l’inventaire.")
            for patient in inventory:
                pid = str(patient.get("patient_id") or patient.get("label") or "Patient").strip()
                reports = list(patient.get("reports") or [])
                points.append(f"{pid} : {len(reports)} rapport{'s' if len(reports) > 1 else ''} associé{'s' if len(reports) > 1 else ''}.")
                if len(points) >= requested_points:
                    break
            if len(points) < requested_points:
                total_reports = sum(len(list(p.get("reports") or [])) for p in inventory if isinstance(p, dict))
                points.append(f"Total des rapports listés : {total_reports}.")
            points = deduplicate_summary_points(points)[:requested_points]
            if len(points) < requested_points:
                limitation = _summary_limitation_text(len(points))
            answer = _format_summary_answer(
                context_label="inventaire patient",
                requested_points=requested_points,
                points=points,
                limitation=limitation,
                sources=[],
                clickable_requested=wants_clickable,
                include_inline_source=False,
            )
    else:
        answer = (
            "Je n’ai pas de contexte précédent à résumer. "
            "Demandez d’abord des résultats, un commentaire ou un inventaire."
        )

    validation = validate_answer(
        query=query,
        answer_text=answer,
        evidence_pack=[],
        displayed_evidences=[],
        source_citations=sources,
        generation_mode="deterministic_context_summary_render",
        retrieval_status="not_required",
        query_received=query_received,
        query_used_for_retrieval=query_used_for_retrieval,
        query_used_for_prompt=query_used_for_prompt,
        query_stored=query,
        detected_analytes=[],
        query_intents=intents,
        output_format_requested="list",
        answer_style_requested="standard",
        requested_table_columns=[],
        requested_technical_condition=None,
        source_clickable_requested=wants_clickable,
        requested_value=None,
        comparison_operator=None,
        raw_format_phrase=getattr(query_understanding, "raw_format_phrase", None),
        unsupported_presentation=False,
        user_requested_visualization=False,
        requested_chart_type=None,
        visualization_payload=None,
        chart_data_payload=None,
    )
    reason = "context_summary_no_retrieval"
    if context_type == "medical_qualitative_comment":
        reason = "qualitative_comment_summary_no_retrieval"
    elapsed = time.perf_counter() - started
    return {
        "request_id": request_id,
        "query": query,
        "query_received": query_received,
        "query_used_for_retrieval": "",
        "query_used_for_prompt": query_used_for_prompt,
        "query_stored": query,
        "normalized_query": query,
        "mode": "context_summary_render",
        "provider": provider,
        "model": model,
        "top_k": top_k,
        "max_display_results": int(max_display_results),
        "show_all_results": bool(show_all_results),
        "show_low_quality": bool(show_low_quality),
        "timeout": timeout,
        "generation_time_seconds": round(elapsed, 3),
        "answer": answer,
        "citations": [],
        "sources": sources,
        "validation": validation,
        "quality_report": _quality_report(
            answer=answer,
            validation=validation,
            source_clickable_requested=wants_clickable,
            recent_style_history=[],
        ),
        "llm_error": None,
        "error_type": None,
        "generation_mode": "deterministic_context_summary_render",
        "detected_analytes": [],
        "query_understanding": _query_understanding_payload(query_understanding),
        "structured_evidence_pack": structured_pack if isinstance(structured_pack, dict) else {},
        "evidence_pack": [],
        "displayed_evidences": [],
        "retrieval": {"answerability": {"status": "not_required", "reason": reason}},
        "prompt": "",
        "debug": {
            "request_id": request_id,
            "generation_mode": "deterministic_context_summary_render",
            "generation_writer": "deterministic_context_summary",
            "intents": intents,
            "retrieval_skipped": True,
            "context_type": context_type,
            "requested_summary_points": requested_points,
            "produced_summary_points": len(points),
        },
        "visualization": None,
        "chart_data": None,
    }


def _build_qualitative_comment_render_response(
    *,
    request_id: str,
    started: float,
    query: str,
    query_received: str,
    query_used_for_retrieval: str,
    query_used_for_prompt: str,
    top_k: int,
    max_display_results: int,
    show_all_results: bool,
    show_low_quality: bool,
    timeout: int,
    provider: str,
    model: str,
    query_understanding: QueryUnderstanding,
    intents: dict[str, bool],
    previous_qualitative_evidence_pack: dict[str, Any] | None,
    previous_displayed_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pack = previous_qualitative_evidence_pack if isinstance(previous_qualitative_evidence_pack, dict) else {}
    evidences = list(pack.get("evidences") or pack.get("results") or [])
    first = evidences[0] if evidences else {}
    comment_text = str(first.get("display_comment_text") or first.get("comment_text") or first.get("current_value") or "").strip()
    subject = resolve_medical_subject(
        query_understanding=query_understanding,
        evidence=first if isinstance(first, dict) else None,
        state_context=previous_displayed_context if isinstance(previous_displayed_context, dict) else None,
        query=query,
    )

    # Build a user-facing source from structured fields first; never expose internal
    # tags such as "sqlite_deterministic" when a PDF source is available.
    source_candidates: list[dict[str, Any]] = []
    source_from_first = str(first.get("source") or "").strip()
    source_candidates.append(
        {
            "label": str(first.get("source_label") or "").strip(),
            "source_pdf": str(first.get("source_pdf") or "").strip(),
            "doc_id": str(first.get("doc_id") or "").strip(),
            "page": first.get("page") if first.get("page") is not None else first.get("page_number"),
            "line": first.get("line") if first.get("line") is not None else first.get("row"),
            "viewer_url": str(first.get("viewer_url") or "").strip() or None,
            "source_url": str(first.get("source_url") or "").strip() or None,
        }
    )
    if source_from_first and source_from_first.lower() != "sqlite_deterministic":
        source_candidates[0]["label"] = source_candidates[0].get("label") or source_from_first
    for src in list(pack.get("sources") or []):
        if isinstance(src, dict):
            source_candidates.append(dict(src))

    normalized_candidates = [normalize_source_for_response(s) for s in source_candidates if isinstance(s, dict)]
    primary_source = next(
        (
            s
            for s in normalized_candidates
            if str(s.get("source_pdf") or "").strip()
            or str(s.get("label") or "").strip().lower().endswith(".pdf")
        ),
        normalized_candidates[0] if normalized_candidates else normalize_source_for_response({}),
    )
    source_label = str(primary_source.get("label") or "").strip() or "source non disponible"
    viewer_url = str(primary_source.get("viewer_url") or "").strip() or None
    source_url = str(primary_source.get("source_url") or primary_source.get("url") or "").strip() or None
    if source_label.lower() == "sqlite_deterministic":
        # Safety net: avoid exposing internal source identifiers.
        source_label = "source non disponible"
    source_markdown, has_clickable_source = format_clickable_source_markdown(source_label, viewer_url, source_url)
    view_type = str(getattr(query_understanding, "qualitative_view_type", "") or "sourced_comment_block").strip() or "sourced_comment_block"
    wants_clickable = bool(getattr(query_understanding, "source_clickable_requested", False))
    if not comment_text:
        answer = (
            "Je n’ai pas de commentaire médical qualitatif récent à afficher sous cette forme. "
            "Demandez d’abord le commentaire concerné."
        )
        sources: list[dict[str, Any]] = []
    else:
        qn = norm_text(query or "")
        graph_request_on_qualitative = any(k in qn for k in ["graphique", "chart", "courbe", "radar", "bar chart", "line graph"])
        compact_comment = comment_text if len(comment_text) <= 520 else comment_text[:517].rstrip() + "..."
        source_text_for_answer = source_markdown if has_clickable_source else source_label
        if wants_clickable and not has_clickable_source:
            source_text_for_answer = f"{source_label} (source non cliquable disponible uniquement en texte)"
        if view_type == "text_table":
            def _ev_source_cell(ev: dict[str, Any]) -> str:
                ev_source = normalize_source_for_response(
                    {
                        "label": str(ev.get("source_label") or ev.get("source") or "").strip(),
                        "source_pdf": str(ev.get("source_pdf") or "").strip(),
                        "doc_id": str(ev.get("doc_id") or "").strip(),
                        "page": ev.get("page") if ev.get("page") is not None else ev.get("page_number"),
                        "line": ev.get("line") if ev.get("line") is not None else ev.get("row"),
                        "viewer_url": str(ev.get("viewer_url") or "").strip() or None,
                        "source_url": str(ev.get("source_url") or "").strip() or None,
                    }
                )
                ev_label = str(ev_source.get("label") or "").strip() or "source non disponible"
                ev_viewer = str(ev_source.get("viewer_url") or "").strip() or None
                ev_url = str(ev_source.get("source_url") or ev_source.get("url") or "").strip() or None
                md, clickable = format_clickable_source_markdown(ev_label, ev_viewer, ev_url)
                if clickable:
                    return md
                if wants_clickable:
                    return escape_markdown_table_cell(f"{ev_label} (source non cliquable disponible uniquement en texte)")
                return escape_markdown_table_cell(ev_label)

            rows = ["| Sujet | Commentaire | Source |", "|---|---|---|"]
            if len(evidences) > 1:
                for ev in evidences:
                    if not isinstance(ev, dict):
                        continue
                    ev_subject = str(ev.get("subject") or ev.get("analyte") or subject or "Commentaire médical").strip()
                    ev_comment_raw = str(ev.get("display_comment_text") or ev.get("comment_text") or ev.get("current_value") or "").strip()
                    if not ev_comment_raw:
                        continue
                    ev_compact = ev_comment_raw if len(ev_comment_raw) <= 520 else ev_comment_raw[:517].rstrip() + "..."
                    rows.append(
                        f"| {escape_markdown_table_cell(ev_subject)} | {escape_markdown_table_cell(ev_compact)} | {_ev_source_cell(ev)} |"
                    )
                if len(rows) == 2:
                    rows.append(
                        f"| {escape_markdown_table_cell(subject)} | {escape_markdown_table_cell(compact_comment)} | "
                        f"{source_markdown if has_clickable_source else escape_markdown_table_cell(source_text_for_answer)} |"
                    )
            else:
                source_cell = source_markdown if has_clickable_source else escape_markdown_table_cell(source_text_for_answer)
                rows.append(f"| {escape_markdown_table_cell(subject)} | {escape_markdown_table_cell(compact_comment)} | {source_cell} |")
            answer = "\n".join(rows)
        elif view_type == "interpretive_note":
            answer = (
                "Note interprétative sourcée\n\n"
                f"{subject}\n\n"
                f"{comment_text}\n\n"
                f"Source : {source_text_for_answer}"
            )
        elif view_type == "medical_info_card":
            answer = (
                "Carte d’information médicale\n\n"
                f"Sujet : {subject}\n"
                "Type : Commentaire qualitatif\n"
                f"Message principal : {comment_text}\n"
                f"Source : {source_text_for_answer}"
            )
        else:
            view_type = "sourced_comment_block"
            answer = build_sourced_comment_block(
                subject=subject,
                comment_text=comment_text,
                source_label=source_text_for_answer,
            )
        if graph_request_on_qualitative:
            answer = (
                "Ce commentaire est une donnée qualitative textuelle, pas une valeur biologique numérique transformable en graphique. "
                "J’affiche plutôt une vue textuelle sourcée.\n\n"
                f"{answer}"
            )
        # Source is already rendered inline in qualitative views; avoid duplicate
        # "Source + Sources" blocks in the final user message.
        sources = []

    validation = validate_answer(
        query=query,
        answer_text=answer,
        evidence_pack=[],
        displayed_evidences=[],
        source_citations=sources,
        generation_mode="deterministic_qualitative_comment_render",
        retrieval_status="not_required",
        query_received=query_received,
        query_used_for_retrieval=query_used_for_retrieval,
        query_used_for_prompt=query_used_for_prompt,
        query_stored=query,
        detected_analytes=[],
        query_intents=intents,
        output_format_requested="paragraph",
        answer_style_requested="standard",
        requested_table_columns=[],
        requested_technical_condition=None,
        source_clickable_requested=False,
        requested_value=None,
        comparison_operator=None,
        raw_format_phrase=getattr(query_understanding, "raw_format_phrase", None),
        unsupported_presentation=False,
        user_requested_visualization=False,
        requested_chart_type=None,
        visualization_payload=None,
        chart_data_payload=None,
    )
    elapsed = time.perf_counter() - started
    return {
        "request_id": request_id,
        "query": query,
        "query_received": query_received,
        "query_used_for_retrieval": query_used_for_retrieval,
        "query_used_for_prompt": query_used_for_prompt,
        "query_stored": query,
        "normalized_query": query,
        "mode": "qualitative_comment_render",
        "provider": provider,
        "model": model,
        "top_k": top_k,
        "max_display_results": int(max_display_results),
        "show_all_results": bool(show_all_results),
        "show_low_quality": bool(show_low_quality),
        "timeout": timeout,
        "generation_time_seconds": round(elapsed, 3),
        "answer": answer,
        "citations": [],
        "sources": sources,
        "validation": validation,
        "quality_report": _quality_report(
            answer=answer,
            validation=validation,
            source_clickable_requested=False,
            recent_style_history=[],
        ),
        "llm_error": None,
        "error_type": None,
        "generation_mode": "deterministic_qualitative_comment_render",
        "detected_analytes": [],
        "query_understanding": _query_understanding_payload(query_understanding),
        "structured_evidence_pack": pack if comment_text else {},
        "evidence_pack": [],
        "displayed_evidences": [],
        "retrieval": {"answerability": {"status": "not_required", "reason": "qualitative_comment_render_no_retrieval"}},
        "prompt": "",
        "debug": {
            "request_id": request_id,
            "generation_mode": "deterministic_qualitative_comment_render",
            "generation_writer": "professional_fallback",
            "intents": intents,
            "retrieval_skipped": True,
        },
        "visualization": None,
        "chart_data": None,
        "qualitative_view": {"type": view_type},
    }


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None


def _parse_numeric_value(raw: Any) -> float | None:
    """Parse robust numeric values from lab strings (French decimal, attached units, thresholds)."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    s = s.replace(",", ".")
    # Keep first numeric token; supports patterns like "< 5", "1.0 g/L", ">=0.8".
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def _normalize_unit(unit: Any) -> str:
    return norm_text(str(unit or "")).replace(" ", "")


def _units_compatible(unit_a: Any, unit_b: Any) -> bool:
    ua = _normalize_unit(unit_a)
    ub = _normalize_unit(unit_b)
    if not ua or not ub:
        return True
    return ua == ub


def _summarize_reference_for_comparison(reference_raw: str) -> str:
    ref = str(reference_raw or "").strip()
    if not ref:
        return "non disponible"
    # Compact and readable summary for long multi-profile references.
    parts = re.split(r"\s{2,}|\n+", ref)
    compact = " ".join(p.strip() for p in parts if p.strip())
    if len(compact) <= 64:
        return compact
    has_profiles = any(k in norm_text(compact) for k in ["homme", "femme", "adulte", "enfant", "nourrisson", "ans", "jours", "mois", ">"])
    if has_profiles:
        return "Plusieurs plages selon l’âge/profil"
    return compact[:61] + "..."


def _comparison_label(current: Any, previous: Any) -> str:
    cf = _to_float(current)
    pf = _to_float(previous)
    if cf is None or pf is None:
        return "non comparable numériquement"
    if cf > pf:
        return "plus élevée"
    if cf < pf:
        return "plus basse"
    return "égale"


def _load_interpretation_rows(
    *,
    sqlite_path: Path,
    interpretation_status: str,
    limit: int,
    analyte_norm: str | None = None,
    doc_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    if not sqlite_path.exists():
        return []

    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        where = "WHERE lower(m.interpretation_status) = lower(?)"
        params: list[Any] = [interpretation_status]
        if analyte_norm:
            where += " AND lower(m.analyte_norm) = lower(?)"
            params.append(analyte_norm)
        doc_ids_norm = [str(d).strip().lower() for d in (doc_ids or []) if str(d).strip()]
        if doc_ids_norm:
            placeholders = ",".join(["?"] * len(doc_ids_norm))
            where += f" AND lower(c.doc_id) IN ({placeholders})"
            params.extend(doc_ids_norm)
        params.append(int(limit))
        cur.execute(
            f"""
            SELECT
              c.chunk_id,
              c.doc_id,
              c.chunk_type,
              c.parent_chunk_id,
              c.text_for_embedding,
              c.text_for_keyword,
              m.document_type,
              m.sample_type,
              m.patient_token,
              m.sample_token,
              m.report_token,
              m.analyte,
              m.analyte_norm,
              m.parameter,
              m.parameter_norm,
              m.value_raw,
              m.value_numeric,
              m.unit,
              m.reference_range,
              m.interpretation_status,
              m.previous_result_present,
              m.previous_result_value_raw,
              m.previous_result_unit,
              m.section,
              m.section_norm,
              m.source_kind,
              m.source_table_id,
              m.row_index,
              COALESCE(m.source_pdf, o.source_pdf) AS source_pdf,
              COALESCE(m.page_number, o.page_number) AS page_number
            FROM metadata_chunks m
            JOIN chunks c ON c.chunk_id = m.chunk_id
            LEFT JOIN object_references o ON o.chunk_id = c.chunk_id
            {where}
            ORDER BY
              c.doc_id ASC,
              COALESCE(m.page_number, o.page_number, 999999) ASC,
              COALESCE(m.row_index, 999999) ASC,
              c.chunk_id ASC
            LIMIT ?
            """,
            params,
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def _select_deterministic_candidates(
    *,
    query_norm: str,
    evidence_pack: list[dict[str, Any]],
    exact_analyte: str | None,
    requested_analytes: list[str] | None = None,
) -> list[dict[str, Any]]:
    candidates = list(evidence_pack)
    requested = [str(a).strip().lower() for a in (requested_analytes or []) if str(a).strip()]
    if requested:
        req_set = set(requested)
        multi_exact = [
            ev
            for ev in candidates
            if any(
                contains_exact_term(str(ev.get("analyte_norm") or ""), a)
                or contains_exact_term(str(ev.get("analyte") or ""), a)
                for a in req_set
            )
        ]
        if multi_exact:
            candidates = multi_exact

    if exact_analyte:
        exact = [
            ev
            for ev in candidates
            if contains_exact_term(str(ev.get("analyte_norm") or ""), exact_analyte)
            or contains_exact_term(str(ev.get("analyte") or ""), exact_analyte)
        ]
        if exact:
            candidates = exact
        else:
            candidates = []

    if _is_above_reference_query(query_norm) and not _is_normal_or_above_query(query_norm):
        above = [
            ev
            for ev in candidates
            if str(ev.get("interpretation_status") or ev.get("technical_status_code") or "").lower() == "above_reference"
        ]
        if above:
            candidates = above
    elif _is_below_reference_query(query_norm):
        below = [
            ev
            for ev in candidates
            if str(ev.get("interpretation_status") or ev.get("technical_status_code") or "").lower() == "below_reference"
        ]
        if below:
            candidates = below

    if _is_previous_result_query(query_norm) or _is_compare_query(query_norm):
        with_prev = [
            ev
            for ev in candidates
            if int(ev.get("previous_result_present") or 0) == 1 and str(ev.get("previous_result") or "").strip() != ""
        ]
        if with_prev:
            candidates = with_prev

    return candidates


def _build_deterministic_evidence_answer(
    *,
    query: str,
    displayed_evidences: list[dict[str, Any]],
    exact_analyte: str | None,
    display_notes: list[str] | None = None,
) -> str:
    qn = norm_text(query)
    candidates = list(displayed_evidences)
    if not candidates:
        return INSUFFICIENT_CONTEXT_SENTENCE

    lines: list[str] = []
    if _is_compare_query(qn):
        lines.append("J’ai comparé techniquement les résultats actuels et antérieurs retrouvés.")
        for idx, ev in enumerate(candidates, start=1):
            analyte = ev.get("analyte") or ev.get("parameter") or "analyte non précisé"
            cur = ev.get("value_raw") or "non disponible"
            unit = ev.get("unit") or ""
            prev = ev.get("previous_result") or "non disponible"
            relation = _comparison_label(cur, prev)
            lines.append(
                f"{idx}. Pour {ev.get('doc_id')}, {analyte} actuel = {cur} {unit}; "
                f"résultat antérieur = {prev}. La valeur actuelle est {relation} que l'antérieure."
            )
    else:
        lines.append("Voici les résultats techniques retrouvés dans les données indexées.")
        if len(candidates) > 1:
            title = (
                f"{len(candidates)} résultats de {exact_analyte.upper()} ont été retrouvés :"
                if exact_analyte
                else f"{len(candidates)} résultats ont été retrouvés :"
            )
            lines.append(title)
        for idx, ev in enumerate(candidates, start=1):
            analyte = ev.get("analyte_display") or ev.get("analyte") or ev.get("parameter") or "analyte non précisé"
            value = ev.get("value_raw") or "non disponible"
            unit = ev.get("unit") or ""
            ref = ev.get("reference_range") or "non disponible"
            interp = ev.get("interpretation_status") or "non disponible"
            prev = ev.get("previous_result")
            prefix = f"{idx}. " if len(candidates) > 1 else "- "
            part = f"{prefix}{analyte} = {value}"
            if unit:
                part += f" {unit}"
            part += f" (référence: {ref} ; interprétation technique: {interp}"
            if prev not in (None, ""):
                part += f" ; résultat antérieur: {prev}"
            part += ")"
            lines.append(part)

    for note in (display_notes or []):
        lines.append(note)
    lines.append("")
    lines.append("La synthèse ci-dessus reste strictement limitée aux évidences retrouvées.")
    return "\n".join(lines).strip()


def _should_use_deterministic_generation(query: str, evidence_pack: list[dict[str, Any]], exact_analyte: str | None) -> bool:
    if not evidence_pack:
        return False
    qn = norm_text(query)
    if exact_analyte:
        return True
    if len(detect_exact_analytes(query)) >= 2:
        return True
    if _is_above_reference_query(qn) or _is_below_reference_query(qn):
        return True
    if _is_previous_result_query(qn) or _is_compare_query(qn):
        return True
    if _is_status_query(qn):
        return True
    if "quel est le resultat" in qn or "quel est le statut" in qn:
        return True
    return False


def _load_exact_analyte_rows(
    *,
    sqlite_path: Path,
    analyte_norm: str,
    limit: int,
    doc_ids: list[str] | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    if not sqlite_path.exists():
        return 0, []

    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        doc_ids_norm = [str(d).strip().lower() for d in (doc_ids or []) if str(d).strip()]
        where_doc = ""
        params_doc: list[Any] = []
        if doc_ids_norm:
            placeholders = ",".join(["?"] * len(doc_ids_norm))
            where_doc = f" AND lower(c.doc_id) IN ({placeholders})"
            params_doc = list(doc_ids_norm)
        cur.execute(
            f"""
            SELECT COUNT(*) AS c
            FROM metadata_chunks m
            JOIN chunks c ON c.chunk_id = m.chunk_id
            WHERE lower(m.analyte_norm) = lower(?)
            {where_doc}
            """,
            [analyte_norm, *params_doc],
        )
        total = int((cur.fetchone() or {"c": 0})["c"])
        cur.execute(
            f"""
            SELECT
              c.chunk_id,
              c.doc_id,
              c.chunk_type,
              c.parent_chunk_id,
              c.text_for_embedding,
              c.text_for_keyword,
              m.document_type,
              m.sample_type,
              m.patient_token,
              m.sample_token,
              m.report_token,
              m.analyte,
              m.analyte_norm,
              m.parameter,
              m.parameter_norm,
              m.value_raw,
              m.value_numeric,
              m.unit,
              m.reference_range,
              m.interpretation_status,
              m.previous_result_present,
              m.previous_result_value_raw,
              m.previous_result_unit,
              m.section,
              m.section_norm,
              m.source_kind,
              m.source_table_id,
              m.row_index,
              COALESCE(m.source_pdf, o.source_pdf) AS source_pdf,
              COALESCE(m.page_number, o.page_number) AS page_number
            FROM metadata_chunks m
            JOIN chunks c ON c.chunk_id = m.chunk_id
            LEFT JOIN object_references o ON o.chunk_id = c.chunk_id
            WHERE lower(m.analyte_norm) = lower(?)
            {where_doc}
            ORDER BY
              c.doc_id ASC,
              COALESCE(m.page_number, o.page_number, 999999) ASC,
              COALESCE(m.row_index, 999999) ASC,
              c.chunk_id ASC
            LIMIT ?
            """,
            [analyte_norm, *params_doc, int(limit)],
        )
        rows = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()

    return total, rows


def _load_requested_analyte_rows(
    *,
    sqlite_path: Path,
    analyte_norms: list[str],
    limit: int,
    doc_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    requested = [str(a).strip().lower() for a in (analyte_norms or []) if str(a).strip()]
    if not requested:
        return []
    if not sqlite_path.exists():
        return []

    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        analyte_placeholders = ",".join(["?"] * len(requested))
        params: list[Any] = list(requested)
        where_doc = ""
        doc_ids_norm = [str(d).strip().lower() for d in (doc_ids or []) if str(d).strip()]
        if doc_ids_norm:
            doc_placeholders = ",".join(["?"] * len(doc_ids_norm))
            where_doc = f" AND lower(c.doc_id) IN ({doc_placeholders})"
            params.extend(doc_ids_norm)
        params.append(int(limit))

        cur.execute(
            f"""
            SELECT
              c.chunk_id,
              c.doc_id,
              c.chunk_type,
              c.parent_chunk_id,
              c.text_for_embedding,
              c.text_for_keyword,
              m.document_type,
              m.sample_type,
              m.patient_token,
              m.sample_token,
              m.report_token,
              m.analyte,
              m.analyte_norm,
              m.parameter,
              m.parameter_norm,
              m.value_raw,
              m.value_numeric,
              m.unit,
              m.reference_range,
              m.interpretation_status,
              m.previous_result_present,
              m.previous_result_value_raw,
              m.previous_result_unit,
              m.section,
              m.section_norm,
              m.source_kind,
              m.source_table_id,
              m.row_index,
              COALESCE(m.source_pdf, o.source_pdf) AS source_pdf,
              COALESCE(m.page_number, o.page_number) AS page_number
            FROM metadata_chunks m
            JOIN chunks c ON c.chunk_id = m.chunk_id
            LEFT JOIN object_references o ON o.chunk_id = c.chunk_id
            WHERE lower(m.analyte_norm) IN ({analyte_placeholders})
            {where_doc}
            ORDER BY
              c.doc_id ASC,
              COALESCE(m.page_number, o.page_number, 999999) ASC,
              COALESCE(m.row_index, 999999) ASC,
              c.chunk_id ASC
            LIMIT ?
            """,
            params,
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def _filter_rows_by_doc_ids(rows: list[dict[str, Any]], requested_doc_ids: list[str]) -> list[dict[str, Any]]:
    allowed = {str(d).strip().lower() for d in requested_doc_ids if str(d).strip()}
    if not allowed:
        return list(rows)
    return [row for row in rows if str(row.get("doc_id") or "").strip().lower() in allowed]


def _filter_retrieval_response_by_doc_ids(retrieval_response: Any, requested_doc_ids: list[str]) -> None:
    allowed = {str(d).strip().lower() for d in requested_doc_ids if str(d).strip()}
    if not allowed:
        return

    retrieval_response.top_results = [
        r for r in (retrieval_response.top_results or []) if str(getattr(r, "doc_id", "") or "").strip().lower() in allowed
    ]
    retrieval_response.context_chunks = [
        r for r in (retrieval_response.context_chunks or []) if str(getattr(r, "doc_id", "") or "").strip().lower() in allowed
    ]
    retrieval_response.sources = [
        s for s in (retrieval_response.sources or []) if str((s or {}).get("doc_id") or "").strip().lower() in allowed
    ]

    if not retrieval_response.top_results and not retrieval_response.context_chunks:
        retrieval_response.answerability = {
            "status": "insufficient_context",
            "reason": "no_results_for_requested_doc_ids",
            "requested_doc_ids": sorted(allowed),
        }


def _resolve_missing_requested_doc_ids(sqlite_path: Path, requested_doc_ids: list[str]) -> list[str]:
    normalized = [str(d).strip().lower() for d in requested_doc_ids if str(d).strip()]
    if not normalized:
        return []
    if not sqlite_path.exists():
        return list(normalized)

    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        placeholders = ",".join(["?"] * len(normalized))
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT DISTINCT lower(doc_id) AS doc_id
            FROM chunks
            WHERE lower(doc_id) IN ({placeholders})
            """,
            normalized,
        )
        found = {str(row["doc_id"]).strip().lower() for row in cur.fetchall() if row["doc_id"]}
    finally:
        conn.close()

    return sorted(d for d in normalized if d not in found)


def _resolve_doc_ids_by_date(sqlite_path: Path, date_iso: str) -> list[str]:
    if not sqlite_path.exists() or not str(date_iso or "").strip():
        return []
    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT DISTINCT lower(doc_id) AS doc_id
            FROM metadata_chunks
            WHERE doc_id IS NOT NULL
              AND trim(doc_id) != ''
              AND (
                substr(coalesce(report_date, ''), 1, 10) = ?
                OR substr(coalesce(request_date, ''), 1, 10) = ?
              )
            ORDER BY doc_id
            """,
            [date_iso, date_iso],
        )
        return [str(r["doc_id"]).strip().lower() for r in cur.fetchall() if r["doc_id"]]
    finally:
        conn.close()


def _resolve_latest_doc_id(sqlite_path: Path) -> str | None:
    if not sqlite_path.exists():
        return None
    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT lower(doc_id) AS doc_id
            FROM metadata_chunks
            WHERE doc_id IS NOT NULL
              AND trim(doc_id) != ''
            ORDER BY
              coalesce(nullif(substr(report_date, 1, 10), ''), nullif(substr(request_date, 1, 10), ''), '0000-00-00') DESC,
              doc_id DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()
        return str(row["doc_id"]).strip().lower() if row and row["doc_id"] else None
    finally:
        conn.close()


def _clean_analyte_label(value: str | None) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "non précisé"
    cleaned = re.sub(
        r"^(?:(?:µ?g|mg|ng|pg|ui|iu|uu|uiu|mui|mmol|pmol|g|ml|dl|l)\s*/\s*(?:ml|dl|l)\s+){1,3}",
        "",
        raw,
        flags=re.IGNORECASE,
    )
    # Strip embedded reference patterns in analyte labels, e.g. "CRP (0.00mg/l - 5.00mg/l)".
    cleaned = re.sub(r"\(\s*\d+(?:[.,]\d+)?\s*(?:mg|ng|pg|g|ui|iu|uu|mui|mmol|pmol)?\s*/?\s*[a-zA-Z]*\s*-\s*\d+(?:[.,]\d+)?[^)]*\)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -;,:")
    return cleaned or raw


def _extract_reference_from_text(text: str | None) -> str | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    m = re.search(r"\(([^()]*\d[^()]*)\)", raw)
    if not m:
        return None
    ref = str(m.group(1) or "").strip()
    if not ref:
        return None
    ref = re.sub(r"\s+", " ", ref)
    # normalize compact units around bounds (e.g. 0.00mg/l - 5.00mg/l)
    ref = re.sub(r"(\d)\s*(mg|ng|pg|g|ui|iu|uu|mui|mmol|pmol)\s*/\s*([a-zA-Z]+)", r"\1 \2/\3", ref, flags=re.IGNORECASE)
    ref = re.sub(r"\s*-\s*", " - ", ref)
    ref = re.sub(r"\s+", " ", ref).strip()
    nums = re.findall(r"\d+(?:[.,]\d+)?", ref)
    if len(nums) < 2:
        return None
    # normalize "(0.00 - 5.00) mg/l" shape if present
    post = re.search(r"\)\s*([a-zA-Zµ/%]+(?:/[a-zA-Zµ%]+)?)$", raw)
    if post and re.search(r"\(\s*\d+(?:[.,]\d+)?\s*-\s*\d+(?:[.,]\d+)?\s*\)", raw):
        return f"{nums[0]} - {nums[1]} {post.group(1)}".strip()
    return ref


def _canonical_technical_condition(value: str | None) -> str | None:
    tc = str(value or "").strip().lower()
    if tc in {"above_reference_only", "above_reference"}:
        return "above_reference"
    if tc in {"below_reference_only", "below_reference"}:
        return "below_reference"
    if tc == "out_of_reference":
        return "out_of_reference"
    if tc == "within_reference":
        return "within_reference"
    if tc in {"any_result", "not_applicable", ""}:
        return None
    return tc


def _apply_technical_condition_filter(rows: list[dict[str, Any]], technical_condition: str | None) -> list[dict[str, Any]]:
    tc = _canonical_technical_condition(technical_condition)
    if not tc:
        return list(rows)
    if tc == "out_of_reference":
        return [r for r in rows if _status_code(r) in {"above_reference", "below_reference"}]
    if tc in {"above_reference", "below_reference", "within_reference", "not_interpretable"}:
        return [r for r in rows if _status_code(r) == tc]
    return list(rows)


def _canonical_display_name(analyte_norm: str) -> str:
    alias = {
        "t4_libre": "T4 LIBRE",
        "ca_15_3": "CA 15-3",
        "psa_totale": "PSA TOTALE",
        "ckmb": "CKMB",
        "cholesterol_ldl": "CHOLESTEROL LDL",
        "acide_valproique": "ACIDE VALPROIQUE",
        "carbamazepine": "CARBAMAZEPINE",
    }
    if analyte_norm in alias:
        return alias[analyte_norm]
    return analyte_norm.replace("_", " ").upper()


def _interpretation_fr(status: str | None) -> str:
    s = str(status or "").strip().lower()
    if s == "above_reference":
        return "au-dessus de la référence"
    if s == "below_reference":
        return "en dessous de la référence"
    if s == "within_reference":
        return "dans la référence"
    return "non interprétable"


def _is_structured_question_with_fast_path(intents: dict[str, bool], requested_doc_ids: list[str], requested_analytes: list[str]) -> bool:
    if intents.get("is_structured_query"):
        return True
    if requested_doc_ids:
        return True
    if len(requested_analytes) >= 1:
        return True
    return False


def _resolve_analytes_for_query(
    *,
    query: str,
    requested_analytes: list[str],
    sqlite_path: Path,
    max_candidates: int = 5,
) -> tuple[list[str], dict[str, Any]]:
    before = [str(a).strip().lower() for a in (requested_analytes or []) if str(a).strip()]
    debug: dict[str, Any] = {
        "query": query,
        "normalized_query": normalize_analyte_text(query),
        "requested_analytes_before_resolver": before,
        "requested_analytes_after_resolver": list(before),
        "available_analytes_count": 0,
        "resolved_analytes": [],
        "ambiguous_candidates": [],
        "match_reason": None,
        "confidence": None,
        "status": "passthrough" if before else "unresolved",
    }
    if before:
        return before, debug
    available = load_available_analytes(str(sqlite_path))
    debug["available_analytes_count"] = len(available)
    resolved = resolve_requested_analytes(
        query=query,
        available_analytes=available,
        aliases=ANALYTE_ALIAS_GROUPS,
        max_candidates=max_candidates,
    )
    if not resolved:
        debug["status"] = "no_match"
        return [], debug
    first = resolved[0] if isinstance(resolved[0], dict) else {}
    status = str(first.get("status") or "selected")
    selected = [str(r.get("analyte_norm") or "").strip().lower() for r in resolved if str(r.get("analyte_norm") or "").strip()]
    selected = list(dict.fromkeys(selected))
    debug["resolved_analytes"] = selected
    debug["requested_analytes_after_resolver"] = selected
    debug["status"] = status
    debug["match_reason"] = first.get("match_reason")
    debug["confidence"] = first.get("confidence")
    if status == "ambiguous":
        debug["ambiguous_candidates"] = list(first.get("candidates") or [])
    return (selected[:1] if status != "ambiguous" else []), debug


def _build_analyte_terms(analyte_norm: str) -> list[str]:
    base = str(analyte_norm or "").strip().lower()
    if not base:
        return []
    variants = {base, base.replace("_", " ")}
    if base == "acide_valproique":
        variants.update({"valpro", "valporo"})
    if base == "carbamazepine":
        variants.update({"carbamazep"})
    if base == "ckmb":
        variants.update({"ckmb", "cpkmb", "ck mb"})
    if base == "crp":
        variants.update({"crp"})
    if base == "cholesterol_ldl":
        variants.update({"ldl", "cholesterol ldl", "cholestérol ldl"})
    return sorted(v for v in variants if v)


def _fetch_doc_lab_rows(
    *,
    sqlite_path: Path,
    requested_doc_ids: list[str],
    analyte_norms: list[str] | None = None,
    include_text_search_terms: list[str] | None = None,
    limit: int = 300,
) -> list[dict[str, Any]]:
    if not sqlite_path.exists():
        return []
    doc_ids = [str(d).strip().lower() for d in requested_doc_ids if str(d).strip()]
    if not doc_ids:
        return []

    analytes = [str(a).strip().lower() for a in (analyte_norms or []) if str(a).strip()]
    text_terms = [str(t).strip().lower() for t in (include_text_search_terms or []) if str(t).strip()]

    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        doc_placeholders = ",".join(["?"] * len(doc_ids))
        params: list[Any] = list(doc_ids)
        where = [f"lower(c.doc_id) IN ({doc_placeholders})", "c.chunk_type = 'lab_result'"]

        analyte_clauses: list[str] = []
        for analyte in analytes:
            for term in _build_analyte_terms(analyte):
                analyte_clauses.append(
                    "(instr(lower(coalesce(m.analyte_norm,'')), ?) > 0 OR instr(lower(coalesce(m.analyte,'')), ?) > 0)"
                )
                params.extend([term, term])
        for term in text_terms:
            analyte_clauses.append(
                "(instr(lower(coalesce(m.value_raw,'')), ?) > 0 OR instr(lower(coalesce(c.text_for_keyword,'')), ?) > 0)"
            )
            params.extend([term, term])

        if analyte_clauses:
            where.append("(" + " OR ".join(analyte_clauses) + ")")

        params.append(int(limit))
        sql = f"""
            SELECT
              c.chunk_id,
              c.doc_id,
              c.chunk_type,
              c.parent_chunk_id,
              c.text_for_embedding,
              c.text_for_keyword,
              m.document_type,
              m.sample_type,
              m.patient_token,
              m.sample_token,
              m.report_token,
              m.analyte,
              m.analyte_norm,
              m.parameter,
              m.parameter_norm,
              m.value_raw,
              m.value_numeric,
              m.unit,
              m.reference_range,
              m.interpretation_status,
              m.previous_result_present,
              m.previous_result_value_raw,
              m.previous_result_unit,
              m.section,
              m.section_norm,
              m.source_kind,
              m.source_table_id,
              m.row_index,
              COALESCE(m.source_pdf, o.source_pdf) AS source_pdf,
              COALESCE(m.page_number, o.page_number) AS page_number
            FROM metadata_chunks m
            JOIN chunks c ON c.chunk_id = m.chunk_id
            LEFT JOIN object_references o ON o.chunk_id = c.chunk_id
            WHERE {" AND ".join(where)}
            ORDER BY
              c.doc_id ASC,
              COALESCE(m.page_number, o.page_number, 999999) ASC,
              COALESCE(m.row_index, 999999) ASC,
              c.chunk_id ASC
            LIMIT ?
        """
        cur = conn.cursor()
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


try:
    from sort_utils import natural_report_sort_key, build_report_range_label
except Exception:  # pragma: no cover
    from scripts.generation.sort_utils import natural_report_sort_key, build_report_range_label  # type: ignore


def fetch_patient_inventory(sqlite_path: Path) -> list[dict[str, Any]]:
    if not sqlite_path.exists():
        return []
    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT DISTINCT patient_token, doc_id, source_pdf
            FROM metadata_chunks
            WHERE patient_token IS NOT NULL AND patient_token != ''
            ORDER BY patient_token, doc_id;
            """
        )
        rows = [dict(r) for r in cur.fetchall()]
        
        # Group by patient
        inventory: dict[str, dict[str, Any]] = {}
        for row in rows:
            p = row["patient_token"]
            if p not in inventory:
                inventory[p] = {
                    "patient": p,
                    "reports_data": []
                }
            filename = row["source_pdf"].split("/")[-1] if row["source_pdf"] else row["doc_id"]
            
            # Deduplicate reports for the same patient/doc
            if not any(r["doc_id"] == row["doc_id"] for r in inventory[p]["reports_data"]):
                inventory[p]["reports_data"].append({
                    "doc_id": row["doc_id"],
                    "filename": filename,
                    "label": filename,
                    "source_url": f"/api/documents/{row['doc_id']}/pdf",
                    "viewer_url": f"/viewer/pdf?doc_id={row['doc_id']}"
                })
        
        # Convert to sorted lists with natural sort
        result = []
        for p in sorted(inventory.keys()):
            item = inventory[p]
            # Natural sort for reports based on filename
            item["reports"] = sorted(item["reports_data"], key=lambda x: natural_report_sort_key(x["filename"]))
            del item["reports_data"]
            
            # Add summary labels for UI
            count = len(item["reports"])
            item["report_count"] = count
            item["summary_label"] = f"{count} rapport{'s' if count > 1 else ''} associé{'s' if count > 1 else ''}"
            
            filenames = [r["filename"] for r in item["reports"]]
            item["report_range_label"] = build_report_range_label(filenames)
            
            # For backward compatibility or extra safety, we can keep sources as a copy of reports
            item["sources"] = item["reports"]
                
            result.append(item)
        return result
    finally:
        conn.close()


def fetch_patient_count(sqlite_path: Path) -> int:
    if not sqlite_path.exists():
        return 0
    conn = sqlite3.connect(str(sqlite_path))
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(DISTINCT patient_token) FROM metadata_chunks WHERE patient_token IS NOT NULL AND patient_token != '';")
        res = cur.fetchone()
        return int(res[0]) if res else 0
    finally:
        conn.close()


def _fetch_doc_summary_rows(
    *,
    sqlite_path: Path,
    requested_doc_ids: list[str],
    limit: int = 20,
) -> list[dict[str, Any]]:
    if not sqlite_path.exists():
        return []
    doc_ids = [str(d).strip().lower() for d in requested_doc_ids if str(d).strip()]
    if not doc_ids:
        return []
    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        doc_placeholders = ",".join(["?"] * len(doc_ids))
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT
              c.chunk_id,
              c.doc_id,
              c.chunk_type,
              c.text_for_embedding,
              c.text_for_keyword,
              m.analyte,
              m.analyte_norm,
              m.parameter,
              m.value_raw,
              m.value_numeric,
              m.unit,
              m.reference_range,
              m.interpretation_status,
              m.previous_result_present,
              m.previous_result_value_raw,
              m.previous_result_unit,
              m.section,
              m.section_norm,
              m.source_kind,
              m.source_table_id,
              m.row_index,
              COALESCE(m.source_pdf, o.source_pdf) AS source_pdf,
              COALESCE(m.page_number, o.page_number) AS page_number
            FROM chunks c
            LEFT JOIN metadata_chunks m ON m.chunk_id = c.chunk_id
            LEFT JOIN object_references o ON o.chunk_id = c.chunk_id
            WHERE lower(c.doc_id) IN ({doc_placeholders})
              AND c.chunk_type IN ('document_summary', 'exam_section', 'clinical_result')
            ORDER BY c.doc_id ASC, COALESCE(m.page_number, o.page_number, 999999) ASC, COALESCE(m.row_index, 999999) ASC, c.chunk_id ASC
            LIMIT ?
            """,
            [*doc_ids, int(limit)],
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def _fetch_global_lab_rows(
    *,
    sqlite_path: Path,
    analyte_norms: list[str],
    requested_value: str | None = None,
    limit: int = 1200,
) -> list[dict[str, Any]]:
    if not sqlite_path.exists():
        return []
    analytes = [str(a).strip().lower() for a in (analyte_norms or []) if str(a).strip()]
    if not analytes:
        return []
    analyte_terms: list[str] = []
    for analyte in analytes:
        aliases = sorted(get_analyte_aliases(analyte))
        analyte_terms.extend([t for t in aliases if t])
    analyte_terms = sorted(set(analyte_terms))
    if not analyte_terms:
        return []
    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        clauses: list[str] = []
        params: list[Any] = []
        for term in analyte_terms:
            clauses.append(
                "(instr(lower(coalesce(m.analyte_norm,'')), ?) > 0 OR instr(lower(coalesce(m.analyte,'')), ?) > 0)"
            )
            params.extend([term, term])
        where = "(" + " OR ".join(clauses) + ")"
        # Value-level filtering is applied in Python (supports =, >=, <=, <, >)
        # to avoid locale format mismatches (comma vs dot) at SQL LIKE time.
        params.append(int(limit))
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT
              c.chunk_id,
              c.doc_id,
              c.chunk_type,
              c.text_for_embedding,
              c.text_for_keyword,
              m.patient_token,
              m.analyte,
              m.analyte_norm,
              m.value_raw,
              m.value_numeric,
              m.unit,
              m.reference_range,
              m.interpretation_status,
              m.previous_result_present,
              m.previous_result_value_raw,
              m.previous_result_unit,
              m.section,
              m.section_norm,
              m.source_kind,
              m.source_table_id,
              m.row_index,
              COALESCE(m.source_pdf, o.source_pdf) AS source_pdf,
              COALESCE(m.page_number, o.page_number) AS page_number
            FROM metadata_chunks m
            JOIN chunks c ON c.chunk_id = m.chunk_id
            LEFT JOIN object_references o ON o.chunk_id = c.chunk_id
            WHERE c.chunk_type = 'lab_result'
              AND {where}
            ORDER BY lower(c.doc_id) ASC, COALESCE(m.page_number, o.page_number, 999999) ASC, COALESCE(m.row_index, 999999) ASC
            LIMIT ?
            """,
            params,
        )
        rows = [dict(r) for r in cur.fetchall()]
        # Final safeguard using alias matcher on raw fields.
        filtered: list[dict[str, Any]] = []
        for row in rows:
            analyte_field = str(row.get("analyte_norm") or "") + " " + str(row.get("analyte") or "")
            if any(match_analyte(analyte_field, a) for a in analytes):
                filtered.append(row)
        return filtered
    finally:
        conn.close()


def _fetch_global_comment_rows(
    *,
    sqlite_path: Path,
    term: str,
    limit: int = 100,
) -> list[dict[str, Any]]:
    if not sqlite_path.exists():
        return []
    t = str(term or "").strip().lower()
    if not t:
        return []
    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
              c.chunk_id,
              c.doc_id,
              c.chunk_type,
              c.text_for_embedding,
              c.text_for_keyword,
              m.patient_token,
              m.sample_token,
              m.analyte,
              m.analyte_norm,
              m.value_raw,
              m.value_numeric,
              m.unit,
              m.reference_range,
              m.interpretation_status,
              m.section,
              m.section_norm,
              m.row_index,
              COALESCE(m.source_pdf, o.source_pdf) AS source_pdf,
              COALESCE(m.page_number, o.page_number) AS page_number
            FROM chunks c
            LEFT JOIN metadata_chunks m ON m.chunk_id = c.chunk_id
            LEFT JOIN object_references o ON o.chunk_id = c.chunk_id
            WHERE instr(lower(coalesce(c.text_for_keyword,'')), ?) > 0
               OR instr(lower(coalesce(c.text_for_embedding,'')), ?) > 0
               OR instr(lower(coalesce(m.value_raw,'')), ?) > 0
               OR instr(lower(coalesce(m.analyte_norm,'')), ?) > 0
               OR instr(lower(coalesce(m.analyte,'')), ?) > 0
            ORDER BY COALESCE(m.page_number, o.page_number, 999999) ASC, COALESCE(m.row_index, 999999) ASC
            LIMIT ?
            """,
            [t, t, t, t, t, int(limit)],
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def _extract_query_numeric_targets(query: str) -> list[str]:
    q = str(query or "")
    return [m.group(0) for m in re.finditer(r"\b\d+(?:[.,]\d+)?\b", q)]


def _row_matches_any_target_value(row: dict[str, Any], targets: list[str]) -> bool:
    if not targets:
        return True
    value_raw = str(row.get("value_raw") or "").strip()
    value_num = row.get("value_numeric")
    raw_norm = value_raw.replace(",", ".").strip().lower()
    raw_norm_nolead = raw_norm.lstrip("0") or "0"
    vf = _to_float(value_num if value_num not in (None, "") else value_raw)
    for target in targets:
        tn = str(target or "").replace(",", ".").strip().lower()
        tn_nolead = tn.lstrip("0") or "0"
        if raw_norm == tn or raw_norm_nolead == tn_nolead:
            return True
        tf = _to_float(target)
        if tf is not None and vf is not None and abs(tf - vf) <= 1e-9:
            return True
    return False


def _row_matches_value_criterion(row: dict[str, Any], targets: list[str], operator: str | None) -> bool:
    if not targets:
        return True
    op = str(operator or "").strip()
    if op not in {">", ">=", "<", "<=", "="}:
        return _row_matches_any_target_value(row, targets)

    vf = _to_float(row.get("value_numeric"))
    if vf is None:
        vf = _to_float(row.get("value_raw"))
    tf = _to_float(targets[0])
    if vf is None or tf is None:
        return _row_matches_any_target_value(row, targets)

    if op == ">":
        return vf > tf
    if op == ">=":
        return vf >= tf
    if op == "<":
        return vf < tf
    if op == "<=":
        return vf <= tf
    return abs(vf - tf) <= 1e-9


def _rows_to_evidence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidences: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str, str, str]] = set()
    for idx, row in enumerate(rows, start=1):
        previous_raw = row.get("previous_result_value_raw")
        prev_present = row.get("previous_result_present")
        try:
            prev_flag = 1 if int(prev_present or 0) == 1 else 0
        except Exception:
            prev_flag = 0

        excerpt = str(row.get("text_for_keyword") or row.get("text_for_embedding") or "").strip()
        if len(excerpt) > 500:
            excerpt = excerpt[:497] + "..."

        ev = {
                "evidence_id": idx,
                "rank": idx,
                "chunk_id": row.get("chunk_id"),
                "doc_id": row.get("doc_id"),
                "chunk_type": row.get("chunk_type"),
                "analyte": row.get("analyte"),
                "analyte_display": _clean_analyte_label(row.get("analyte") or row.get("parameter")),
                "analyte_norm": row.get("analyte_norm"),
                "parameter": row.get("parameter"),
                "patient_token": row.get("patient_token"),
                "sample_token": row.get("sample_token"),
                "value_raw": row.get("value_raw"),
                "value_numeric": _to_float(row.get("value_numeric")),
                "unit": row.get("unit"),
                "reference_range": row.get("reference_range"),
                "reference_range_raw": row.get("reference_range"),
                "reference_raw": row.get("reference_range"),
                "reference_ranges": parse_reference_ranges(str(row.get("reference_range") or ""), default_unit=str(row.get("unit") or "").strip() or None),
                "interpretation_status": row.get("interpretation_status"),
                "previous_result": previous_raw,
                "previous_result_present": prev_flag,
                "section": row.get("section"),
                "source_kind": row.get("source_kind"),
                "source_table_id": row.get("source_table_id"),
                "source_pdf": row.get("source_pdf"),
                "page_number": row.get("page_number"),
                "row_index": row.get("row_index"),
                "source": "sqlite_deterministic",
                "final_score": None,
                "clinical_rerank_score": None,
                "evidence_display_quality": "high",
                "evidence_display_quality_reasons": [],
                "text_excerpt": excerpt,
            }
        key = (
            str(ev.get("patient_token") or "").strip().lower(),
            str(ev.get("analyte_norm") or ev.get("analyte") or "").strip().lower(),
            str(ev.get("value_numeric") if ev.get("value_numeric") is not None else ev.get("value_raw") or "").strip(),
            str(ev.get("unit") or "").strip().lower(),
            str(ev.get("doc_id") or "").strip().lower(),
            str(ev.get("page_number") or "").strip(),
            str(row.get("sample_token") or "").strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        evidences.append(ev)
    return evidences


def _row_matches_analyte(row: dict[str, Any], analyte_norm: str) -> bool:
    analyte_field = f"{row.get('analyte_norm') or ''} {row.get('analyte') or ''}"
    return match_analyte(analyte_field, analyte_norm)


def _row_matches_excluded(row: dict[str, Any], excluded_analytes: list[str]) -> bool:
    if not excluded_analytes:
        return False
    analyte_field = norm_text(f"{row.get('analyte_norm') or ''} {row.get('analyte') or ''}")
    for excluded in excluded_analytes:
        ex = norm_text(str(excluded or ""))
        if not ex:
            continue
        if ex in {"trak", "anti_tg", "anti_recepteur_tsh"}:
            if ex == "trak" and "trak" in analyte_field:
                return True
            if ex == "anti_tg" and ("anti tg" in analyte_field or "anti-tg" in analyte_field):
                return True
            if ex == "anti_recepteur_tsh" and "recepteur" in analyte_field and "tsh" in analyte_field:
                return True
            continue
        if ex in analyte_field:
            return True
    return False


def _safe_float_pair(current: Any, previous: Any) -> tuple[float | None, float | None]:
    return _to_float(current), _to_float(previous)


def _variation_label(current: Any, previous: Any) -> str:
    cf, pf = _safe_float_pair(current, previous)
    if cf is None or pf is None:
        return "non comparable"
    if cf > pf:
        return "augmenté"
    if cf < pf:
        return "diminué"
    return "stable"


def _missing_doc_answer() -> str:
    return "information non retrouvée dans le document demandé"


def _resolve_reference_scope_doc_ids(
    *,
    sqlite_path: Path,
    requested_doc_ids: list[str],
    requested_date_iso: str | None,
    requested_report_type: str | None,
) -> list[str]:
    explicit = [str(d).strip().lower() for d in (requested_doc_ids or []) if str(d).strip()]
    if explicit:
        return explicit
    if not sqlite_path.exists():
        return []
    report_type = str(requested_report_type or "").strip().lower()
    report_tokens_map: dict[str, list[str]] = {
        "immunoanalyse": ["immuno", "immunoanalyse", "immuno analyse"],
        "biochimie": ["biochimie", "bio chimie", "chimie"],
        "toxicologie": ["toxico", "toxicologie"],
    }
    report_tokens = report_tokens_map.get(report_type, [report_type] if report_type else [])
    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        params: list[Any] = []
        where = []
        if requested_date_iso:
            params.append(str(requested_date_iso).strip())
            where.append("substr(coalesce(report_date, request_date, ''), 1, 10) = ?")
        if report_tokens:
            sub = []
            for tok in report_tokens:
                sub.append(
                    "(instr(lower(coalesce(section_norm,'')), ?) > 0 OR instr(lower(coalesce(section,'')), ?) > 0 OR instr(lower(coalesce(document_type,'')), ?) > 0)"
                )
                params.extend([tok, tok, tok])
            where.append("(" + " OR ".join(sub) + ")")
        if not where:
            return []
        cur.execute(
            f"""
            SELECT DISTINCT lower(doc_id) AS doc_id
            FROM metadata_chunks
            WHERE {' AND '.join(where)}
            ORDER BY doc_id ASC
            """,
            params,
        )
        docs = [str(r["doc_id"]).strip() for r in cur.fetchall() if str(r["doc_id"]).strip()]
        if not requested_date_iso and not explicit and docs:
            # Deterministic behavior for type-only queries: prefer most recent report id.
            docs = sorted(docs, key=_doc_recency_key, reverse=True)
            return [docs[0]]
        return docs
    finally:
        conn.close()


def _resolve_target_analyte_rows(rows: list[dict[str, Any]], requested_analyte: str) -> list[dict[str, Any]]:
    target = norm_text(requested_analyte or "")
    if not target:
        return rows
    exact = [
        r
        for r in rows
        if norm_text(str(r.get("analyte_norm") or r.get("analyte") or "")) == target
        or norm_text(str(r.get("analyte") or "")) == target
    ]
    if exact:
        return exact
    alias = [r for r in rows if _row_matches_analyte(r, target)]
    if len(alias) == 1:
        return alias
    if len(alias) > 1:
        return alias
    return []


def find_reference_range_candidate_rows(
    sqlite_path: Path,
    analyte_names: list[str],
    requested_report_type: str | None = None,
    requested_date_iso: str | None = None,
    requested_doc_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    analytes = [str(a).strip().lower() for a in (analyte_names or []) if str(a).strip()]
    if not analytes or not sqlite_path.exists():
        return []
    report_type = str(requested_report_type or "").strip().lower()
    tokens_map = {
        "immunoanalyse": ["immuno", "immunoanalyse", "immuno analyse"],
        "biochimie": ["biochimie", "bio chimie", "chimie"],
        "toxicologie": ["toxico", "toxicologie"],
    }
    r_tokens = tokens_map.get(report_type, [report_type] if report_type else [])

    # 1) doc_ids + analyte
    explicit = [str(d).strip().lower() for d in (requested_doc_ids or []) if str(d).strip()]
    if explicit:
        rows = _fetch_doc_lab_rows(sqlite_path=sqlite_path, requested_doc_ids=explicit, analyte_norms=analytes, limit=300)
        if rows:
            return rows

    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()

        def _query(where_extra: list[str], params_extra: list[Any]) -> list[dict[str, Any]]:
            analyte_clause = " OR ".join(
                [
                    "(instr(lower(coalesce(m.analyte_norm,'')), ?) > 0 OR instr(lower(coalesce(m.analyte,'')), ?) > 0)"
                    for _ in analytes
                ]
            )
            analyte_params: list[Any] = []
            for a in analytes:
                analyte_params.extend([a, a])
            where = ["c.chunk_type = 'lab_result'", f"({analyte_clause})"]
            where.extend(where_extra)
            sql = f"""
            SELECT
              c.doc_id,
              m.analyte,
              m.analyte_norm,
              m.value_raw,
              m.value_numeric,
              m.unit,
              m.reference_range,
              m.section,
              m.section_norm,
              m.document_type,
              m.row_index,
              COALESCE(m.source_pdf, o.source_pdf) AS source_pdf,
              COALESCE(m.page_number, o.page_number) AS page_number,
              substr(coalesce(m.report_date, m.request_date, ''), 1, 10) AS report_date
            FROM metadata_chunks m
            JOIN chunks c ON c.chunk_id = m.chunk_id
            LEFT JOIN object_references o ON o.chunk_id = c.chunk_id
            WHERE {' AND '.join(where)}
            ORDER BY c.doc_id ASC, COALESCE(m.page_number, o.page_number, 999999) ASC, COALESCE(m.row_index, 999999) ASC
            LIMIT 500
            """
            cur.execute(sql, analyte_params + params_extra)
            return [dict(r) for r in cur.fetchall()]

        # 2) date + type + analyte
        if requested_date_iso:
            where = ["substr(coalesce(m.report_date, m.request_date, ''), 1, 10) = ?"]
            params: list[Any] = [str(requested_date_iso).strip()]
            if r_tokens:
                sub = []
                for t in r_tokens:
                    sub.append("(instr(lower(coalesce(m.section_norm,'')), ?) > 0 OR instr(lower(coalesce(m.section,'')), ?) > 0 OR instr(lower(coalesce(m.document_type,'')), ?) > 0)")
                    params.extend([t, t, t])
                where.append("(" + " OR ".join(sub) + ")")
            rows = _query(where, params)
            if rows:
                return rows

        # 3) type + analyte
        if r_tokens:
            sub = []
            params = []
            for t in r_tokens:
                sub.append("(instr(lower(coalesce(m.section_norm,'')), ?) > 0 OR instr(lower(coalesce(m.section,'')), ?) > 0 OR instr(lower(coalesce(m.document_type,'')), ?) > 0)")
                params.extend([t, t, t])
            rows = _query(["(" + " OR ".join(sub) + ")"], params)
            if rows:
                return rows

        # 4) analyte global fallback
        return _query([], [])
    finally:
        conn.close()


def _build_reference_range_lookup_response(
    *,
    request_id: str,
    started: float,
    query: str,
    query_received: str,
    query_used_for_retrieval: str,
    query_used_for_prompt: str,
    top_k: int,
    max_display_results: int,
    show_all_results: bool,
    show_low_quality: bool,
    timeout: int,
    provider: str,
    model: str,
    query_understanding: QueryUnderstanding,
    intents: dict[str, bool],
    sqlite_path: Path,
    requested_doc_ids: list[str],
    previous_displayed_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    strict_enabled = _is_feature_enabled("REFERENCE_RANGE_STRICT_MODE", default=True)
    llm_rewrite_enabled = _is_feature_enabled("LLM_REWRITE_ENABLED", default=True)
    llm_fallback_non_critical_only = _is_feature_enabled("LLM_FALLBACK_NON_CRITICAL_ONLY", default=True)
    if not strict_enabled:
        answer = (
            "La recherche stricte de plages physiologiques est temporairement désactivée. "
            "Veuillez préciser le rapport ou demander les résultats bruts."
        )
        validation = validate_answer(
            query=query,
            answer_text=answer,
            evidence_pack=[],
            displayed_evidences=[],
            source_citations=[],
            generation_mode="reference_range_lookup_disabled_by_feature_flag",
            retrieval_status="not_required",
            query_received=query_received,
            query_used_for_retrieval=query_used_for_retrieval,
            query_used_for_prompt=query_used_for_prompt,
            query_stored=query,
            detected_analytes=[],
            # Do not run strict reference-range validation constraints while the feature is disabled.
            query_intents={**dict(intents or {}), "reference_range_lookup": False},
            output_format_requested="paragraph",
            answer_style_requested="standard",
            requested_table_columns=[],
            requested_technical_condition=None,
            source_clickable_requested=bool(getattr(query_understanding, "source_clickable_requested", False)),
            requested_value=None,
            comparison_operator=None,
            raw_format_phrase=getattr(query_understanding, "raw_format_phrase", None),
            unsupported_presentation=False,
            user_requested_visualization=False,
            requested_chart_type=None,
            visualization_payload=None,
            chart_data_payload=None,
        )
        validation = dict(validation or {})
        warnings = list(validation.get("warnings") or [])
        warnings.append("skipped_validation_because_feature_disabled")
        validation["warnings"] = warnings
        validation["validation_status"] = "warning"
        return {
            "request_id": request_id,
            "query": query,
            "query_received": query_received,
            "query_used_for_retrieval": query_used_for_retrieval,
            "query_used_for_prompt": query_used_for_prompt,
            "query_stored": query,
            "normalized_query": query,
            "mode": "reference_range_lookup",
            "provider": provider,
            "model": model,
            "top_k": top_k,
            "max_display_results": int(max_display_results),
            "show_all_results": bool(show_all_results),
            "show_low_quality": bool(show_low_quality),
            "timeout": timeout,
            "generation_time_seconds": round(time.perf_counter() - started, 3),
            "answer": answer,
            "citations": [],
            "sources": [],
            "validation": validation,
            "quality_report": _quality_report(answer=answer, validation=validation, source_clickable_requested=False, recent_style_history=[]),
            "llm_error": None,
            "error_type": None,
            "generation_mode": "reference_range_lookup_disabled_by_feature_flag",
            "detected_analytes": [],
            "query_understanding": _query_understanding_payload(query_understanding),
            "structured_evidence_pack": {},
            "evidence_pack": [],
            "displayed_evidences": [],
            "retrieval": {"answerability": {"status": "not_required", "reason": "reference_range_strict_mode_disabled"}},
            "prompt": "",
            "debug": {
                "request_id": request_id,
                "selection_status": "disabled",
                "feature_flags": {"REFERENCE_RANGE_STRICT_MODE": False},
                "reference_range_debug": {
                    "intent": "reference_range_lookup",
                    "requested_analytes": list(query_understanding.requested_analytes or []),
                    "requested_report_type": getattr(query_understanding, "requested_report_type", None),
                    "requested_date_iso": getattr(query_understanding, "requested_date_iso", None),
                    "reason": "strict_mode_disabled",
                },
            },
            "visualization": None,
            "chart_data": None,
        }

    analyte = str((query_understanding.requested_analytes or [""])[0] or "").strip().lower()
    if not analyte:
        answer = "Je n’ai pas détecté l’analyte demandé pour la plage de référence."
        status = "no_match"
        selected = None
        candidates = []
        source_label = "source non disponible"
    else:
        candidate_rows = find_reference_range_candidate_rows(
            sqlite_path=sqlite_path,
            analyte_names=[analyte],
            requested_report_type=getattr(query_understanding, "requested_report_type", None),
            requested_date_iso=getattr(query_understanding, "requested_date_iso", None),
            requested_doc_ids=requested_doc_ids,
        )
        filtered_rows = _resolve_target_analyte_rows(candidate_rows, analyte)
        for r in filtered_rows:
            r["reference_raw"] = r.get("reference_raw") or r.get("reference_range")
            r["page"] = r.get("page") if r.get("page") is not None else r.get("page_number")
            r["row"] = r.get("row") if r.get("row") is not None else r.get("row_index")
        patient_profile = None
        flow = run_reference_range_lookup_from_rows(
            rows=filtered_rows,
            analyte=analyte,
            requested_profile=getattr(query_understanding, "requested_reference_profile", None),
            use_patient_profile=bool(getattr(query_understanding, "use_patient_profile", False)),
            patient_profile=patient_profile,
            request_all_ranges=bool(getattr(query_understanding, "request_all_reference_ranges", False)),
            report_type=getattr(query_understanding, "requested_report_type", None),
            date_iso=getattr(query_understanding, "requested_date_iso", None),
        )
        status = str(flow.get("status") or "no_match")
        answer = str(flow.get("answer") or f"Aucune plage physiologique exploitable n’a été retrouvée pour {analyte_display_name(analyte, analyte)}.")
        debug_extra = dict(flow.get("debug") or {})
        selected_range = debug_extra.get("selected_range")
        flow_sources_raw = list(flow.get("sources") or [])
        flow_sources: list[dict[str, Any]] = []
        for src in flow_sources_raw:
            if not isinstance(src, dict):
                continue
            normalized = normalize_source_for_response(src)
            label = str(normalized.get("label") or "").strip()
            if label.lower().startswith("docs/"):
                normalized["label"] = label[5:].strip()
            flow_sources.append(normalized)
        ref_debug = {
            "intent": "reference_range_lookup",
            "requested_analytes": [analyte],
            "requested_report_type": getattr(query_understanding, "requested_report_type", None),
            "requested_date_iso": getattr(query_understanding, "requested_date_iso", None),
            "requested_reference_profile": getattr(query_understanding, "requested_reference_profile", None),
            "resolved_doc_ids": sorted({str(r.get("doc_id") or "").strip() for r in filtered_rows if str(r.get("doc_id") or "").strip()}),
            "candidate_rows_count": len(filtered_rows),
            "candidate_rows_preview": list(debug_extra.get("candidate_rows_preview") or [])[:8],
            "reference_raw_found": bool(any(str(r.get("reference_raw") or r.get("reference_range") or "").strip() for r in filtered_rows)),
            "parsed_ranges_count": int(debug_extra.get("parsed_ranges_count") or 0),
            "parsed_ranges_preview": list((debug_extra.get("parsed_ranges_preview") or []))[:8],
            "selected_range": debug_extra.get("selected_range"),
            "selector_status": status,
            "failure_reason": debug_extra.get("failure_reason"),
            "followup_resolved": bool(_looks_like_reference_range_followup(query)),
            "previous_analyte": str((((previous_displayed_context or {}) if isinstance(previous_displayed_context, dict) else {}).get("last_reference_range_context") or {}).get("analyte") or ""),
            "new_analyte": analyte,
            "inherited_profile": getattr(query_understanding, "requested_reference_profile", None),
            "resolved_intent": "reference_range_lookup",
            "llm_called": False,
        }
    if 'ref_debug' not in locals():
        ref_debug = {
            "intent": "reference_range_lookup",
            "requested_analytes": [analyte] if analyte else [],
            "requested_report_type": getattr(query_understanding, "requested_report_type", None),
            "requested_date_iso": getattr(query_understanding, "requested_date_iso", None),
            "requested_reference_profile": getattr(query_understanding, "requested_reference_profile", None),
            "resolved_doc_ids": [],
            "candidate_rows_count": 0,
            "candidate_rows_preview": [],
            "reference_raw_found": False,
            "parsed_ranges_count": 0,
            "parsed_ranges_preview": [],
            "selected_range": None,
            "selector_status": status,
            "failure_reason": "no_analyte_detected" if not analyte else None,
        }
    if "flow_sources" not in locals():
        flow_sources = []
    LOGGER.info(
        "reference_range_debug request_id=%s debug=%s",
        request_id,
        json.dumps(ref_debug, ensure_ascii=False, default=str),
    )
    wants_clickable = bool(getattr(query_understanding, "source_clickable_requested", False))
    answer_lines = [ln for ln in str(answer or "").splitlines() if not ln.strip().lower().startswith(("source :", "sources :"))]
    answer = "\n".join(answer_lines).strip()
    if wants_clickable and flow_sources:
        src0 = flow_sources[0]
        source_label = str(src0.get("label") or "").strip() or "source non disponible"
        viewer_url = str(src0.get("viewer_url") or "").strip() or None
        source_url = str(src0.get("url") or "").strip() or None
        source_md, clickable = format_clickable_source_markdown(source_label, viewer_url, source_url)
        if clickable:
            answer += f"\n\nSource : {source_md}"
        else:
            answer += (
                f"\n\nSource : {source_label}\n"
                "Source disponible uniquement en texte ; aucun lien cliquable n’est disponible."
            )
    # Hybrid professional mode: backend always selects medical facts; LLM can rewrite style only.
    # Never let LLM decide values/sources for reference_range_lookup.
    llm_rewrite_attempted = False
    if strict_enabled and llm_rewrite_enabled and not llm_fallback_non_critical_only:
        llm_rewrite_attempted = True
        rr_rows = [r for r in filtered_rows if isinstance(r, dict)]
        rr_primary = rr_rows[0] if rr_rows else {}
        rr_pack = {
            "intent": "reference_range_lookup",
            "requested_doc_ids": [str(rr_primary.get("doc_id") or "").strip()] if str(rr_primary.get("doc_id") or "").strip() else [],
            "requested_analytes": [analyte] if analyte else [],
            "output_format": "paragraph",
            "answer_style": "standard",
            "evidences": [
                {
                    "doc_id": rr_primary.get("doc_id"),
                    "filename": rr_primary.get("source_pdf"),
                    "page": rr_primary.get("page"),
                    "row": rr_primary.get("row"),
                    "analyte": str(rr_primary.get("analyte") or rr_primary.get("analyte_norm") or analyte_display_name(analyte, analyte)),
                    "analyte_norm": str(rr_primary.get("analyte_norm") or analyte),
                    "reference": str(rr_primary.get("reference_raw") or rr_primary.get("reference_range") or ""),
                    "reference_range": str(rr_primary.get("reference_raw") or rr_primary.get("reference_range") or ""),
                    "unit": str(rr_primary.get("unit") or ""),
                    "technical_status": "dans la référence",
                    "technical_status_code": "within_reference",
                }
            ]
            if rr_primary
            else [],
            "missing_items": [],
        }
        try:
            # In critical medical mode, we still allow rewrite, but never free fallback to LLM-only decisions.
            rewrite_mode = "fallback" if llm_fallback_non_critical_only else "auto"
            rewritten = compose_professional_answer(
                user_question=query,
                query_understanding=query_understanding,
                evidence_pack=rr_pack,
                mode=rewrite_mode,
                source_citations=flow_sources,
                llm_client=None,
                provider=provider,
                model=model,
                temperature=0.0,
                num_ctx=DEFAULT_LLM_NUM_CTX,
                max_tokens=DEFAULT_LLM_MAX_TOKENS,
                timeout=timeout,
            )
            candidate = str(rewritten.get("answer") or "").strip()
            # Guardrails: keep deterministic answer unless rewritten text keeps selected numeric facts.
            keep = bool(candidate)
            if keep and status in {"selected", "fallback"} and isinstance(selected_range, dict):
                low = selected_range.get("low")
                high = selected_range.get("high")
                threshold = selected_range.get("threshold")
                op = str(selected_range.get("operator") or "").strip()
                cand_norm = norm_text(candidate)
                if low is not None and high is not None:
                    ltxt = norm_text(str(low).replace(".", ","))
                    htxt = norm_text(str(high).replace(".", ","))
                    if ltxt not in cand_norm and norm_text(str(low)) not in cand_norm:
                        keep = False
                    if htxt not in cand_norm and norm_text(str(high)) not in cand_norm:
                        keep = False
                elif threshold is not None and op in {"<", "<=", ">", ">="}:
                    ttxt = norm_text(str(threshold).replace(".", ","))
                    if ttxt not in cand_norm and norm_text(str(threshold)) not in cand_norm:
                        keep = False
            if keep:
                answer = candidate
                ref_debug["llm_called"] = str(rewritten.get("mode") or "").startswith("llm_")
                ref_debug["llm_rewrite_mode"] = str(rewritten.get("mode") or "")
            else:
                ref_debug["llm_called"] = False
                ref_debug["llm_rewrite_mode"] = "deterministic_retained_after_guardrail"
        except Exception as exc:
            ref_debug["llm_called"] = False
            ref_debug["llm_rewrite_mode"] = f"deterministic_retained_llm_error:{exc}"
    validation = validate_answer(
        query=query,
        answer_text=answer,
        evidence_pack=[
            {
                "analyte": str(r.get("analyte") or r.get("analyte_norm") or analyte),
                "analyte_norm": str(r.get("analyte_norm") or "").strip().lower() or analyte,
                "reference_range": str(r.get("reference_raw") or r.get("reference_range") or "").strip(),
                "reference": str(r.get("reference_raw") or r.get("reference_range") or "").strip(),
                "unit": str(r.get("unit") or "").strip(),
                "doc_id": r.get("doc_id"),
                "source_pdf": r.get("source_pdf"),
                "page_number": r.get("page") if r.get("page") is not None else r.get("page_number"),
                "row_index": r.get("row") if r.get("row") is not None else r.get("row_index"),
            }
            for r in filtered_rows
            if isinstance(r, dict)
        ],
        displayed_evidences=[
            {
                "analyte": str(r.get("analyte") or r.get("analyte_norm") or analyte),
                "analyte_norm": str(r.get("analyte_norm") or "").strip().lower() or analyte,
                "reference_range": str(r.get("reference_raw") or r.get("reference_range") or "").strip(),
                "reference": str(r.get("reference_raw") or r.get("reference_range") or "").strip(),
                "unit": str(r.get("unit") or "").strip(),
                "doc_id": r.get("doc_id"),
                "source_pdf": r.get("source_pdf"),
                "page_number": r.get("page") if r.get("page") is not None else r.get("page_number"),
                "row_index": r.get("row") if r.get("row") is not None else r.get("row_index"),
            }
            for r in filtered_rows
            if isinstance(r, dict)
        ],
        source_citations=flow_sources,
        generation_mode="deterministic_reference_range_lookup",
        retrieval_status="answerable" if status in {"selected", "fallback", "ambiguous", "grouped_options"} else "insufficient_context",
        query_received=query_received,
        query_used_for_retrieval=query_used_for_retrieval,
        query_used_for_prompt=query_used_for_prompt,
        query_stored=query,
        detected_analytes=[],
        query_intents=intents,
        output_format_requested="table" if bool(getattr(query_understanding, "request_all_reference_ranges", False)) else "paragraph",
        answer_style_requested="standard",
        requested_table_columns=[],
        requested_technical_condition=None,
        source_clickable_requested=wants_clickable,
        requested_value=None,
        comparison_operator=None,
        raw_format_phrase=getattr(query_understanding, "raw_format_phrase", None),
        unsupported_presentation=False,
        user_requested_visualization=False,
        requested_chart_type=None,
        visualization_payload=None,
        chart_data_payload=None,
    )
    return {
        "request_id": request_id,
        "query": query,
        "query_received": query_received,
        "query_used_for_retrieval": query_used_for_retrieval,
        "query_used_for_prompt": query_used_for_prompt,
        "query_stored": query,
        "normalized_query": query,
        "mode": "reference_range_lookup",
        "provider": provider,
        "model": model,
        "top_k": top_k,
        "max_display_results": int(max_display_results),
        "show_all_results": bool(show_all_results),
        "show_low_quality": bool(show_low_quality),
        "timeout": timeout,
        "generation_time_seconds": round(time.perf_counter() - started, 3),
        "answer": answer,
        "citations": [],
        "sources": flow_sources,
        "validation": validation,
        "quality_report": _quality_report(answer=answer, validation=validation, source_clickable_requested=wants_clickable, recent_style_history=[]),
        "llm_error": None,
        "error_type": None,
        "generation_mode": "deterministic_reference_range_lookup",
        "detected_analytes": [analyte] if analyte else [],
        "query_understanding": _query_understanding_payload(query_understanding),
        "structured_evidence_pack": {},
        "evidence_pack": [],
        "displayed_evidences": [],
        "retrieval": {"answerability": {"status": "answerable" if status in {"selected", "fallback", "ambiguous", "grouped_options"} else "insufficient_context", "reason": status}},
        "prompt": "",
        "debug": {
            "request_id": request_id,
            "selection_status": status,
            "selected_policy": "deterministic_strict",
            "policy_level": "deterministic_strict",
            "facts_source": "evidence_rows_only",
            "llm_writer_allowed": False,
            "llm_writer_used": bool(ref_debug.get("llm_called", False)),
            "evidence_rows_count": len(filtered_rows),
            "displayed_evidences_count": len(filtered_rows),
            "feature_flags": {
                "REFERENCE_RANGE_STRICT_MODE": True,
                "LLM_REWRITE_ENABLED": llm_rewrite_enabled,
                "LLM_FALLBACK_NON_CRITICAL_ONLY": llm_fallback_non_critical_only,
            },
            "llm_called": bool(ref_debug.get("llm_called", False)),
            "llm_rewrite_attempted": llm_rewrite_attempted,
            "reference_range_debug": ref_debug,
        },
        "visualization": None,
        "chart_data": None,
    }


def _source_label(row: dict[str, Any]) -> str:
    doc_id = str(row.get("doc_id") or "source").strip()
    source_pdf = str(row.get("source_pdf") or "").strip()
    filename = source_pdf.split("/")[-1] if source_pdf else ""
    base = filename or doc_id
    page = row.get("page_number")
    row_index = row.get("row_index")
    label = base
    if isinstance(page, int):
        label += f" — page {page}"
    if isinstance(row_index, int):
        label += f", ligne {row_index}"
    return " ".join(label.split())


def _status_code(row: dict[str, Any]) -> str:
    status = str(row.get("interpretation_status") or "").strip().lower()
    if status in {"above_reference", "below_reference", "within_reference"}:
        return status
    ref = str(row.get("reference_range") or "").strip()
    if not ref:
        ref = str(_extract_reference_from_text(str(row.get("analyte") or "")) or "").strip()
    val = str(row.get("value_raw") or "").strip()
    if not ref:
        return "missing_reference"
    if not val:
        return "not_interpretable"
    cf = _to_float(val)
    if cf is None:
        return "not_interpretable"
    nums = re.findall(r"\d+(?:[.,]\d+)?", ref)
    if not nums:
        return "not_interpretable"
    try:
        if "<" in ref:
            hi = float(nums[0].replace(",", "."))
            return "within_reference" if cf < hi else "above_reference"
        if ">" in ref:
            lo = float(nums[0].replace(",", "."))
            return "within_reference" if cf > lo else "below_reference"
        if len(nums) >= 2:
            lo = float(nums[0].replace(",", "."))
            hi = float(nums[1].replace(",", "."))
            if cf < lo:
                return "below_reference"
            if cf > hi:
                return "above_reference"
            return "within_reference"
    except Exception:
        return "not_interpretable"
    return "not_interpretable"


def _status_fr(status_code: str) -> str:
    mapping = {
        "above_reference": "au-dessus de la référence",
        "below_reference": "en dessous de la référence",
        "within_reference": "dans la référence",
        "missing_reference": "référence manquante",
        "not_interpretable": "non interprétable",
    }
    return mapping.get(status_code, "non interprétable")


_PRIORITY_TEXT_MARKERS = [
    str(t).strip().lower()
    for t in list(((get_priority_scoring_config() or {}).get("priority_scoring") or {}).get("textual_severity_terms") or [])
    if str(t).strip()
]


def _is_complex_reference_text(ref: str) -> bool:
    rn = norm_text(ref or "")
    if not rn:
        return True
    complex_markers = [
        "homme",
        "femme",
        "age",
        "ans",
        "enfant",
        "nourrisson",
        "adulte",
        "nouveau",
        "risque",
        "souhaitable",
        "modere",
        "modéré",
        "eleve",
        "élevé",
        "tres",
        "très",
        ":",
    ]
    nums = re.findall(r"\d+(?:[.,]\d+)?", ref or "")
    if len(nums) != 2:
        return True
    if any(m in rn for m in complex_markers):
        return True
    return False


def _recompute_simple_status(value_raw: str, reference_raw: str) -> str | None:
    val = _to_float(value_raw)
    if val is None:
        return None
    ref = str(reference_raw or "").strip()
    if not ref:
        return None
    if _is_complex_reference_text(ref):
        return None
    nums = re.findall(r"\d+(?:[.,]\d+)?", ref)
    if len(nums) < 2:
        return None
    try:
        lo = float(nums[0].replace(",", "."))
        hi = float(nums[1].replace(",", "."))
    except Exception:
        return None
    if val < lo:
        return "below_reference"
    if val > hi:
        return "above_reference"
    return "within_reference"


def _is_within_any_inclusive_interval(value_raw: str, reference_raw: str) -> bool:
    val = _to_float(value_raw)
    if val is None:
        return False
    ref = str(reference_raw or "")
    # Capture explicit interval fragments in complex references.
    intervals = re.findall(r"(\d+(?:[.,]\d+)?)\s*(?:-|à|a)\s*(\d+(?:[.,]\d+)?)", ref, flags=re.IGNORECASE)
    for lo_s, hi_s in intervals:
        try:
            lo = float(lo_s.replace(",", "."))
            hi = float(hi_s.replace(",", "."))
        except Exception:
            continue
        if lo <= val <= hi:
            return True
    return False


def _severity_category_hit(value_raw: str, reference_raw: str, status_code: str) -> tuple[bool, str]:
    val = _to_float(value_raw)
    if val is None:
        return (False, "")
    ref_raw = str(reference_raw or "").lower()
    refn = unicodedata.normalize("NFKD", ref_raw)
    refn = "".join(ch for ch in refn if not unicodedata.combining(ch))
    refn = re.sub(r"\s+", " ", refn).strip()
    patterns = [
        ("tres haute", r"tres\s+haute\s*[:=]?\s*>\s*(\d+(?:[.,]\d+)?)", "above_reference"),
        ("très haute", r"tr[eè]s\s+haute\s*[:=]?\s*>\s*(\d+(?:[.,]\d+)?)", "above_reference"),
        ("tres eleve", r"tres\s+elev[ée]\s*[:=]?\s*>\s*(\d+(?:[.,]\d+)?)", "above_reference"),
        ("très élevé", r"tr[eè]s\s+[ée]lev[ée]\s*[:=]?\s*>\s*(\d+(?:[.,]\d+)?)", "above_reference"),
        ("tres bas", r"tres\s+bas\s*[:=]?\s*<\s*(\d+(?:[.,]\d+)?)", "below_reference"),
        ("très bas", r"tr[eè]s\s+bas\s*[:=]?\s*<\s*(\d+(?:[.,]\d+)?)", "below_reference"),
        ("tres basse", r"tres\s+basse\s*[:=]?\s*<\s*(\d+(?:[.,]\d+)?)", "below_reference"),
        ("très basse", r"tr[eè]s\s+basse\s*[:=]?\s*<\s*(\d+(?:[.,]\d+)?)", "below_reference"),
    ]
    for label, pat, direction in patterns:
        m = re.search(pat, refn, flags=re.IGNORECASE)
        if not m:
            continue
        try:
            threshold = float(str(m.group(1)).replace(",", "."))
        except Exception:
            continue
        if direction != status_code:
            continue
        if direction == "above_reference" and val > threshold:
            return (True, label)
        if direction == "below_reference" and val < threshold:
            return (True, label)
    return (False, "")


def _priority_family_weight(analyte: str, analyte_norm: str) -> float:
    probe = f"{str(analyte or '').strip()} {str(analyte_norm or '').strip()}".strip()
    families = dict((get_analyte_families_config() or {}).get("families") or {})
    for spec in families.values():
        terms = [norm_text(str(t)) for t in list((spec or {}).get("analytes") or []) if str(t).strip()]
        if any(t and t in norm_text(probe) for t in terms):
            return float((spec or {}).get("weight", 0.15) or 0.15)
    return 0.0


def _compute_priority_fields(ev: dict[str, Any]) -> dict[str, Any]:
    # Heuristique technique non diagnostique. Source de vérité: config + evidence.
    return _compute_priority_score_external(ev)


def _apply_priority_scoring(evidences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for ev in evidences:
        row = dict(ev)
        row.update(_compute_priority_fields(row))
        # Exclude obvious false positives / non-priorisable directions.
        if str(row.get("technical_status_code") or "").strip().lower() not in {"above_reference", "below_reference"}:
            continue
        if str(row.get("priority_level") or "unknown") == "unknown":
            continue
        enriched.append(row)

    level_rank = {"high": 0, "moderate": 1, "low": 2, "unknown": 3}
    enriched.sort(
        key=lambda r: (
            level_rank.get(str(r.get("priority_level") or "unknown"), 9),
            -float(r.get("priority_score") or 0.0),
            str(r.get("analyte") or ""),
        )
    )
    return enriched


def _build_doc_scoped_biological_summary_answer(
    evidences: list[dict[str, Any]],
    *,
    max_lines: int | None,
    no_diagnosis: bool,
) -> str:
    rows = list(evidences or [])
    if not rows:
        return _missing_doc_answer()

    def _status_of(ev: dict[str, Any]) -> str:
        status = str(ev.get("technical_status_code") or ev.get("interpretation_status") or "").strip().lower()
        if status in {"above_reference", "below_reference", "within_reference"}:
            return status
        status_fr = norm_text(str(ev.get("technical_status") or ""))
        if "au dessus" in status_fr or "au-dessus" in status_fr:
            return "above_reference"
        if "en dessous" in status_fr:
            return "below_reference"
        if "dans la reference" in status_fr:
            return "within_reference"
        return "unknown"

    def _severity_rank(ev: dict[str, Any]) -> tuple[int, float]:
        level = str(ev.get("priority_level") or "").strip().lower()
        score = float(ev.get("priority_score") or 0.0)
        if level == "high":
            return (0, -score)
        if level == "moderate":
            return (1, -score)
        if level == "low":
            return (2, -score)
        status = _status_of(ev)
        # Simple severity fallback when priority is unavailable.
        if status == "above_reference":
            return (3, -score)
        if status == "below_reference":
            return (4, -score)
        return (9, -score)

    abnormal = [r for r in rows if _status_of(r) in {"above_reference", "below_reference"}]
    normal = [r for r in rows if _status_of(r) == "within_reference"]
    abnormal_sorted = sorted(abnormal, key=_severity_rank)

    def _fmt_item(ev: dict[str, Any], *, with_status: bool = True) -> str:
        a = str(ev.get("analyte") or "analyte").strip()
        if not with_status:
            return a
        status = _status_of(ev)
        if status == "above_reference":
            return f"{a} (au-dessus)"
        if status == "below_reference":
            return f"{a} (en dessous)"
        return f"{a} (dans la référence)"

    max_l = max(3, min(12, int(max_lines or 6)))
    abnormal_items = ", ".join(_fmt_item(ev) for ev in abnormal_sorted[: max(1, min(6, max_l - 2))]) if abnormal_sorted else "aucune anomalie objectivée dans les lignes exploitables"
    normal_items = ", ".join(_fmt_item(ev, with_status=False) for ev in normal[: max(1, min(4, max_l - 2))]) if normal else "peu de résultats strictement dans la référence parmi les lignes exploitées"
    lines: list[str] = [
        f"Anormaux : {abnormal_items}.",
        f"Normaux / rassurants : {normal_items}.",
    ]
    if no_diagnosis:
        lines.append("Conclusion technique : synthèse limitée aux résultats disponibles, sans conclusion diagnostique.")
    else:
        lines.append("Conclusion technique : synthèse limitée aux résultats disponibles.")
    return "\n".join(lines[:max_l]).strip()


def _structured_record_from_row(row: dict[str, Any], *, requested_doc_id: str | None = None) -> dict[str, Any]:
    value_raw = str(row.get("value_raw") or "").strip()
    unit = str(row.get("unit") or "").strip()
    previous = str(row.get("previous_result_value_raw") or "").strip()
    status_code = _status_code(row)
    variation = "non comparable"
    if previous:
        variation = _variation_label(value_raw, previous)
    analyte_norm = str(row.get("analyte_norm") or "").strip().lower()
    analyte_raw = _clean_analyte_label(str(row.get("analyte") or row.get("parameter") or "non précisé"))
    analyte_human = analyte_display_name(analyte_raw, analyte_norm or None) or analyte_raw
    reference_raw = str(row.get("reference_range") or "").strip()
    if not reference_raw:
        reference_raw = str(_extract_reference_from_text(str(row.get("analyte") or "")) or "").strip()
    return {
        "doc_id": str(row.get("doc_id") or requested_doc_id or ""),
        "document_name": str(row.get("source_pdf") or "").split("/")[-1] if str(row.get("source_pdf") or "").strip() else str(row.get("doc_id") or requested_doc_id or ""),
        "patient_id": str(row.get("patient_token") or "").strip() or None,
        "date": str(row.get("report_date") or row.get("request_date") or "").strip() or None,
        "patient_token": str(row.get("patient_token") or "").strip(),
        "page": row.get("page_number"),
        "row": row.get("row_index"),
        "chunk_id": row.get("chunk_id"),
        "analyte": analyte_human,
        "analyte_norm": analyte_norm,
        "current_value": value_raw,
        "unit": unit,
        "reference": reference_raw,
        "reference_raw": reference_raw,
        "reference_ranges": parse_reference_ranges(reference_raw, default_unit=str(row.get("unit") or "").strip() or None),
        "previous_result": previous,
        "technical_status_code": status_code,
        "technical_status": _status_fr(status_code),
        "variation": variation,
        "source": _source_label(row),
        "source_label": _source_label(row),
        "source_excerpt": " ".join(
            x
            for x in [
                str(analyte_human or "").strip(),
                str(value_raw or "").strip(),
                str(unit or "").strip(),
                str(reference_raw or "").strip(),
            ]
            if x
        ),
    }


def _is_table_markdown(text: str) -> bool:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    for i in range(len(lines) - 1):
        if "|" in lines[i] and re.search(r"^\|?\s*[-:| ]+\s*\|?\s*$", lines[i + 1]):
            return True
    return False


def _table_has_source_column(text: str) -> bool:
    lines = [ln.strip().lower() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return False
    header = lines[0]
    return "|" in header and "source" in header


def _resolve_table_columns(evidence_pack: dict[str, Any], *, for_cohort: bool = False) -> list[str]:
    requested = [str(c).strip().lower() for c in (evidence_pack.get("requested_table_columns") or []) if str(c).strip()]
    if requested:
        return requested
    if for_cohort:
        return ["patient", "report", "analyte", "valeur_actuelle", "reference", "statut", "source"]
    return ["analyte", "valeur_actuelle", "unite", "reference", "statut", "resultat_anterieur", "variation", "source"]


def _finalize_structured_pack(pack: dict[str, Any], query_understanding: QueryUnderstanding) -> dict[str, Any]:
    requested_analytes = list(pack.get("requested_analytes") or [])
    requested_doc_ids = list(pack.get("requested_doc_ids") or [])
    requested_columns = list(pack.get("requested_table_columns") or [])
    constraints = {
        "requested_doc_ids": requested_doc_ids,
        "requested_analytes": requested_analytes,
        "excluded_analytes": [],
        "technical_condition": pack.get("technical_condition"),
        "comparison_operator": query_understanding.comparison_operator,
        "requested_value": query_understanding.requested_value,
        "requested_unit": query_understanding.requested_unit,
        "requested_columns": requested_columns,
        "source_clickable_requested": bool(query_understanding.source_clickable_requested),
        "safety_intent": query_understanding.safety_intent,
    }
    pack["user_question"] = pack.get("question") or ""
    pack["original_user_question"] = getattr(query_understanding, "original_user_question", pack.get("question") or "")
    pack["normalized_intent"] = query_understanding.intent
    pack["response_strategy"] = getattr(query_understanding, "response_strategy", "render_table")
    pack["response_strategy_reason"] = getattr(query_understanding, "response_strategy_reason", None)
    pack["presentation_intent"] = {
        "requested_output": getattr(query_understanding.presentation_intent, "requested_output", query_understanding.output_format),
        "chart_type": getattr(query_understanding.presentation_intent, "chart_type", None),
        "raw_format_phrase": getattr(query_understanding.presentation_intent, "raw_format_phrase", None),
        "wants_clickable_sources": bool(getattr(query_understanding.presentation_intent, "wants_clickable_sources", False)),
        "wants_intro": bool(getattr(query_understanding.presentation_intent, "wants_intro", True)),
        "wants_conclusion": bool(getattr(query_understanding.presentation_intent, "wants_conclusion", True)),
        "strict_columns": list(getattr(query_understanding.presentation_intent, "strict_columns", []) or []),
        "unsupported_format": bool(getattr(query_understanding.presentation_intent, "unsupported_format", False)),
        "user_requested_visualization": bool(getattr(query_understanding.presentation_intent, "user_requested_visualization", False)),
        "presentation_confidence": float(getattr(query_understanding.presentation_intent, "presentation_confidence", 0.5)),
        "unsupported_reason": getattr(query_understanding.presentation_intent, "unsupported_reason", None),
        "recommended_output": getattr(query_understanding.presentation_intent, "recommended_output", None),
        "unhandled_instructions": list(getattr(query_understanding.presentation_intent, "unhandled_instructions", []) or []),
        "unsupported_presentation_reason": getattr(query_understanding.presentation_intent, "unsupported_presentation_reason", None),
        "recommended_alternative_format": getattr(query_understanding.presentation_intent, "recommended_alternative_format", None),
    }
    pack["answer_style"] = pack.get("answer_style") or query_understanding.answer_style
    pack["constraints"] = constraints
    pack["results"] = list(pack.get("evidences") or [])
    pack["warnings"] = list(pack.get("warnings") or [])
    return pack


def _attach_source_fields_to_structured_pack(structured_pack: dict[str, Any], source_citations: list[dict[str, Any]]) -> dict[str, Any]:
    out = dict(structured_pack or {})
    evidences = [dict(ev) for ev in (out.get("evidences") or [])]
    by_key: dict[tuple[str, int | None, int | None], dict[str, Any]] = {}
    for src in source_citations or []:
        doc_id = str(src.get("doc_id") or "").strip().lower()
        if not doc_id:
            continue
        page = src.get("page")
        row = src.get("row")
        key = (doc_id, int(page) if isinstance(page, int) else None, int(row) if isinstance(row, int) else None)
        by_key[key] = src

    for ev in evidences:
        doc_id = str(ev.get("doc_id") or "").strip().lower()
        page = ev.get("page")
        row = ev.get("row")
        key = (
            doc_id,
            int(page) if isinstance(page, int) else None,
            int(row) if isinstance(row, int) else None,
        )
        src = by_key.get(key)
        if not src and doc_id:
            # fallback doc-level match
            for candidate_key, candidate_src in by_key.items():
                if candidate_key[0] == doc_id:
                    src = candidate_src
                    break
        if src:
            ev["source_label"] = src.get("label")
            ev["source_url"] = src.get("url")
            ev["viewer_url"] = src.get("viewer_url")
            ev["filename"] = src.get("filename")

    out["evidences"] = evidences
    out["results"] = list(evidences)
    return out


def _table_header_label(col_key: str) -> str:
    mapping = {
        "analyte": "Analyte",
        "valeur_actuelle": "Valeur actuelle",
        "unite": "Unité",
        "reference": "Référence",
        "statut": "Statut",
        "resultat_anterieur": "Résultat antérieur",
        "variation": "Variation",
        "source": "Source",
        "patient": "Patient",
        "report": "Report",
    }
    return mapping.get(col_key, col_key)


def _table_cell_value(ev: dict[str, Any], col_key: str) -> str:
    key = str(col_key or "").strip().lower()
    if key == "analyte":
        raw = str(ev.get("analyte") or "non précisé")
        norm = str(ev.get("analyte_norm") or "").strip().lower()
        return analyte_display_name(raw, norm or None) or raw
    if key == "valeur_actuelle":
        value = str(ev.get("current_value") or "non disponible")
        unit = str(ev.get("unit") or "").strip()
        if unit:
            return f"{value} {unit}"
        return value
    if key == "unite":
        return str(ev.get("unit") or "")
    if key == "reference":
        return str(ev.get("reference") or "non disponible")
    if key == "statut":
        return str(ev.get("technical_status") or "non interprétable")
    if key == "resultat_anterieur":
        return str(ev.get("previous_result") or "non disponible")
    if key == "variation":
        return str(ev.get("variation") or "non comparable")
    if key == "source":
        return str(ev.get("source") or "")
    if key == "patient":
        return str(ev.get("patient_token") or "non disponible")
    if key == "report":
        return str(ev.get("doc_id") or "")
    return str(ev.get(key) or "")


def render_evidence_pack_deterministic(evidence_pack: dict[str, Any], output_format: str) -> str:
    evidences = list(evidence_pack.get("evidences") or [])
    missing_items = list(evidence_pack.get("missing_items") or [])
    intent = str(evidence_pack.get("intent") or "")
    requested_doc_ids = list(evidence_pack.get("requested_doc_ids") or [])
    requested_analytes = list(evidence_pack.get("requested_analytes") or [])
    requested_doc = requested_doc_ids[0] if requested_doc_ids else "le document demandé"

    if intent == "diagnostic_safety_question":
        qn = norm_text(str(evidence_pack.get("question") or ""))
        if detect_medical_topic(qn) == "thyroid":
            thyroid_norms = {str(a).strip().lower() for a in get_topic_analytes("thyroid")}
            thyroid_rows = [
                ev for ev in evidences
                if str(ev.get("analyte_norm") or "").strip().lower() in thyroid_norms
            ]
            if thyroid_rows:
                details: list[str] = []
                has_tsh_high = False
                has_t3_t4_high = False
                for ev in thyroid_rows:
                    status = str(ev.get("technical_status_code") or "").strip().lower()
                    analyte = str(ev.get("analyte") or "").strip()
                    analyte_norm = str(ev.get("analyte_norm") or "").strip().lower()
                    if status == "above_reference":
                        details.append(f"{analyte} élevée")
                        if analyte_norm in {"tshus", "tsh"}:
                            has_tsh_high = True
                        if analyte_norm in {"t3_libre", "t3libre", "t4_libre", "t4libre"}:
                            has_t3_t4_high = True
                    elif status == "below_reference":
                        details.append(f"{analyte} basse")
                details_txt = ", ".join(details) if details else "anomalies thyroïdiennes"
                discordance = ""
                if has_tsh_high and has_t3_t4_high:
                    discordance = " Ce profil est biologiquement discordant pour une hyperthyroïdie primaire."
                return (
                    f"Le document montre des anomalies thyroïdiennes importantes : {details_txt}. "
                    "Cependant, on ne peut pas conclure seul à un diagnostic thyroïdien à partir de ce document. "
                    f"{discordance} Il faut corréler avec le contexte clinique, les traitements, les interférences analytiques et, si besoin, répéter/compléter le bilan."
                )
        lines = [
            "Non, on ne peut pas conclure à un cancer uniquement à partir de ces marqueurs.",
            "Constat technique sur les marqueurs retrouvés :",
        ]
        if evidences:
            for ev in evidences:
                value = ev.get("current_value") or "non disponible"
                unit = f" {ev.get('unit')}" if ev.get("unit") else ""
                ref = ev.get("reference") or "non disponible"
                lines.append(
                    f"- {ev.get('analyte')}: {value}{unit} | référence: {ref} | statut technique: {ev.get('technical_status')}"
                )
        else:
            lines.append("- Aucun marqueur demandé retrouvé.")
        for analyte in missing_items:
            lines.append(f"- {_canonical_display_name(str(analyte))}: non retrouvé dans {requested_doc}.")
        lines.append("Ces marqueurs biologiques ne suffisent pas à poser un diagnostic ; une interprétation médicale spécialisée est nécessaire.")
        return "\n".join(lines).strip()

    if intent == "comment_without_measured_value":
        comment = str(evidence_pack.get("comment_text") or "").strip()
        rows_for_subject = list(evidence_pack.get("evidences") or evidence_pack.get("results") or [])
        first = rows_for_subject[0] if rows_for_subject and isinstance(rows_for_subject[0], dict) else {}
        subject = str(first.get("subject") or first.get("analyte") or "ce sujet").strip()
        if comment:
            snippet = comment if len(comment) <= 220 else comment[:217] + "..."
            return (
                f"Aucune valeur mesurée n’est retrouvée pour {subject} ; le document contient seulement un commentaire/interprétation "
                f"avec seuil. Extrait: {snippet}"
            )
        return f"Aucun commentaire exploitable n’a été retrouvé pour {subject} dans les données indexées."

    if intent in {"global_patient_lookup", "cohort_search"}:
        if not evidences:
            return "Aucun patient/document ne correspond à ce critère dans la base indexée."
        col_keys = _resolve_table_columns(evidence_pack, for_cohort=True)
        headers = [_table_header_label(c) for c in col_keys]
        rows = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for ev in evidences:
            rows.append(
                "| "
                + " | ".join([_table_cell_value(ev, c) for c in col_keys])
                + " |"
            )
        return "\n".join(rows)
        lines = []
        for ev in evidences:
            lines.append(
                f"- patient {ev.get('patient_token') or 'non disponible'} | {ev.get('doc_id')} | {ev.get('analyte')}: "
                f"{ev.get('current_value') or 'non disponible'} {ev.get('unit') or ''} | source: {ev.get('source')}"
            )
        return "\n".join(lines)

    if intent == "multi_doc_presence_diff":
        if not evidences:
            return _missing_doc_answer()
        rows = [
            "| Analyte | Présent dans | Absent dans | Source |",
            "| --- | --- | --- | --- |",
        ]
        for ev in evidences:
            rows.append(
                "| "
                + " | ".join(
                    [
                        str(ev.get("analyte") or "non précisé"),
                        str(ev.get("present_in") or ""),
                        str(ev.get("absent_in") or ""),
                        str(ev.get("source") or ""),
                    ]
                )
                + " |"
            )
        return "\n".join(rows)

    if intent in {"multi_doc_comparison", "doc_pair_comparison"}:
        doc_ids = requested_doc_ids[:2]
        left = doc_ids[0] if len(doc_ids) >= 1 else "report_a"
        right = doc_ids[1] if len(doc_ids) >= 2 else "report_b"
        grouped: dict[str, dict[str, dict[str, Any]]] = {}
        for ev in evidences:
            analyte_norm = str(ev.get("analyte_norm") or ev.get("analyte") or "").strip().lower()
            side = str(ev.get("comparison_side") or ev.get("doc_id") or "").strip()
            if analyte_norm not in grouped:
                grouped[analyte_norm] = {}
            if side in {left, right}:
                grouped[analyte_norm][side] = ev
        lines: list[str] = []
        requested = list(evidence_pack.get("requested_analytes") or [])
        targets = requested if requested else list(grouped.keys())
        for analyte in targets:
            key = str(analyte).strip().lower()
            label = _canonical_display_name(key)
            side_data = grouped.get(key, {})
            a = side_data.get(left)
            b = side_data.get(right)
            if not a and not b:
                lines.append(f"- {label}: non retrouvé dans {left} ni {right}.")
                continue
            if a and not b:
                lines.append(f"- {label}: présent uniquement dans {left} ({a.get('current_value')} {a.get('unit') or ''}).")
                continue
            if b and not a:
                lines.append(f"- {label}: présent uniquement dans {right} ({b.get('current_value')} {b.get('unit') or ''}).")
                continue
            av = str(a.get("current_value") or "")
            bv = str(b.get("current_value") or "")
            unit = str(a.get("unit") or b.get("unit") or "").strip()
            ref = str(a.get("reference") or b.get("reference") or "non disponible")
            variation = _variation_label(bv, av)
            lines.append(
                f"- {label}: {left}={av}{(' ' + unit) if unit else ''} | {right}={bv}{(' ' + unit) if unit else ''} | "
                f"référence: {ref} | différence technique: {variation}"
            )
        return "\n".join(lines).strip() if lines else _missing_doc_answer()

    if intent in {"doc_scoped_summary", "immunoanalysis_summary"}:
        rows = list(evidence_pack.get("rows") or [])
        question = str(evidence_pack.get("question") or "")
        compare_previous = bool(evidence_pack.get("requires_previous_results"))
        if rows:
            summary = _format_doc_summary_answer(rows=rows, query_norm=norm_text(question), compare_previous=compare_previous)
            if summary.strip():
                return summary
        return "Examens sanguins :\n- non retrouvé\nExamens urinaires :\n- non retrouvé\nSéro-diagnostic :\n- non retrouvé"

    if not evidences and output_format == "yes_no":
        analyte_label = _canonical_display_name(requested_analytes[0]) if requested_analytes else "analyte"
        if _explicit_yes_no_requested(str(evidence_pack.get("question") or "")):
            return f"Non - {analyte_label} non retrouvée dans {requested_doc} ; source : document demandé uniquement."
        return f"{analyte_label} non retrouvée dans {requested_doc} ; source : document demandé uniquement."

    if not evidences:
        return _missing_doc_answer()

    if output_format == "yes_no":
        primary = evidences[0]
        status = str(primary.get("technical_status_code") or "")
        ref = str(primary.get("reference") or "non disponible")
        qn = norm_text(str(evidence_pack.get("question") or ""))
        wants_en_yes_no = any(k in qn for k in ["yes/no", "yes or no", "yes no", "respond only yes", "answer only yes"])
        if not ref or ref.lower() in {"non disponible", "none", "null"}:
            yn = "Cannot determine" if wants_en_yes_no else "Impossible à déterminer"
        else:
            yn = ("Yes" if wants_en_yes_no else "Oui") if status in {"above_reference", "below_reference"} else ("No" if wants_en_yes_no else "Non")
        src = str(primary.get("source") or "")
        analyte = str(primary.get("analyte") or "analyte")
        value = str(primary.get("current_value") or "non disponible")
        if wants_en_yes_no or _explicit_yes_no_requested(str(evidence_pack.get("question") or "")):
            if wants_en_yes_no:
                return f"{yn} - {analyte} = {value} ; reference: {ref} ; source: {src}"
            return f"{yn} - {analyte} = {value} ; référence : {ref} ; source : {src}"
        return f"{analyte} = {value} ; référence : {ref} ; source : {src}"

    if output_format == "table":
        col_keys = _resolve_table_columns(evidence_pack, for_cohort=False)
        headers = [_table_header_label(c) for c in col_keys]
        rows: list[str] = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for ev in evidences:
            rows.append(
                "| "
                + " | ".join([_table_cell_value(ev, c) for c in col_keys])
                + " |"
            )
        table_text = "\n".join(rows)
        if "source" not in set(col_keys):
            srcs = [str(ev.get("source") or "").strip() for ev in evidences if str(ev.get("source") or "").strip()]
            if srcs:
                uniq: list[str] = []
                seen: set[str] = set()
                for src in srcs:
                    if src in seen:
                        continue
                    seen.add(src)
                    uniq.append(src)
                table_text += "\n\nSources :\n" + "\n".join(f"- {s}" for s in uniq)
        return table_text

    lines: list[str] = []
    for ev in evidences:
        line = f"- {ev.get('analyte')}: {ev.get('current_value') or 'non disponible'}"
        if ev.get("unit"):
            line += f" {ev.get('unit')}"
        line += f" | référence: {ev.get('reference') or 'non disponible'} | statut technique: {ev.get('technical_status')}"
        if ev.get("previous_result"):
            line += f" | antérieur: {ev.get('previous_result')}"
            if ev.get("variation"):
                line += f" | variation: {ev.get('variation')}"
        lines.append(line)
    for missing in missing_items:
        lines.append(f"- {_canonical_display_name(str(missing))}: non retrouvé dans {requested_doc}.")
    return "\n".join(lines).strip()


def generate_grounded_response_with_llm(
    *,
    user_question: str,
    query_understanding: QueryUnderstanding,
    evidence_pack: dict[str, Any],
    llm_client: LLMClient | None,
    provider: str,
    model: str,
    temperature: float,
    num_ctx: int,
    max_tokens: int,
    timeout: int,
) -> tuple[str, str, str | None]:
    query_norm = _normalize_query_text(user_question)
    compose_mode = (
        "fallback"
        if _force_deterministic_mode_for_summary_anomalies(query_understanding, query_norm)
        else _hybrid_writer_mode(query_understanding)
    )
    writer_pack = _attach_visualization_facts_to_evidence_pack(
        query_understanding=query_understanding,
        evidence_pack=evidence_pack,
        displayed_evidences=list(evidence_pack.get("evidences") or evidence_pack.get("results") or []),
    )
    composed = compose_professional_answer(
        user_question=user_question,
        query_understanding=query_understanding,
        evidence_pack=writer_pack,
        mode=compose_mode,
        llm_client=llm_client,
        provider=provider,
        model=model,
        temperature=temperature,
        num_ctx=num_ctx,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    answer = str(composed.get("answer") or "").strip()
    mode_out = str(composed.get("mode") or "deterministic_professional_fallback")
    llm_error = composed.get("llm_error")
    if not answer:
        fallback = render_professional_fallback(
            evidence_pack=writer_pack,
            query_understanding=query_understanding,
            user_question=user_question,
        )
        answer = str(fallback.get("answer") or "").strip()
        mode_out = "deterministic_professional_fallback"
    return answer, mode_out, str(llm_error) if llm_error else None


_HALLUCINATION_ERROR_KEYS = {
    "unsupported_value",
    "unsupported_analyte",
    "unsupported_source",
    "unsupported_reference",
    "unsupported_previous_result",
    "unsupported_patient",
    "forbidden_internal_field",
    "diagnostic_safety_violation",
    "hallucinated_diagnosis",
    "llm_hallucination",
}

_STYLE_RETRY_KEYS = {
    "missing_professional_intro",
    "output_format_not_respected",
    "output_columns_not_respected",
    "format_not_respected",
    "source_format_bad",
    "internal_alias_leak",
    "missing_query_criterion_in_intro",
    "clickable_source_missing",
    "missing_conclusion",
    "ugly_pluralization",
    "unsupported_format_silently_ignored",
    "output_format_mismatch",
    "display_name_required",
    "chart_units_warning_missing",
    "no_silent_default_table",
    "forbidden_none_literal",
}


def _should_retry_with_validator(validation: dict[str, Any], generation_mode: str) -> bool:
    if generation_mode not in {"llm_professional_writer", "hybrid_structured_llm_writer"}:
        return False
    errors = {str(e) for e in (validation.get("errors") or [])}
    warnings = {str(w) for w in (validation.get("warnings") or [])}
    if errors & _HALLUCINATION_ERROR_KEYS:
        return False
    return bool((errors | warnings) & _STYLE_RETRY_KEYS)


def _build_validator_retry_feedback(validation: dict[str, Any]) -> str:
    errors = [str(e) for e in (validation.get("errors") or [])]
    warnings = [str(w) for w in (validation.get("warnings") or [])]
    items = errors + warnings
    if not items:
        return ""
    bullets = "\n".join(f"- {i}" for i in items[:12])
    return (
        "Corrige ces points de style/format sans modifier aucune donnée (valeurs, analytes, sources, patients):\n"
        f"{bullets}\n"
        "Conserve strictement les faits et les sources de l'evidence_pack."
    )


def _extract_intro_conclusion(answer: str) -> tuple[str, str]:
    blocks = [b.strip() for b in str(answer or "").split("\n\n") if b.strip()]
    if not blocks:
        return "", ""
    intro = blocks[0]
    conclusion = ""
    for block in reversed(blocks):
        if norm_text(block).startswith("conclusion technique"):
            conclusion = block
            break
    return intro, conclusion


def compute_repetition_score(new_answer: str, recent_answers: list[dict[str, Any]]) -> float:
    if not recent_answers:
        return 0.0
    intro, conclusion = _extract_intro_conclusion(new_answer)
    intro_n = norm_text(intro)
    concl_n = norm_text(conclusion)
    generic_hits = 0
    total = min(20, len(recent_answers))
    same_intro = 0
    same_conclusion = 0
    for item in recent_answers[-20:]:
        prev_intro = norm_text(str(item.get("intro_text") or ""))
        prev_conc = norm_text(str(item.get("conclusion_text") or ""))
        if intro_n and prev_intro and (intro_n == prev_intro or intro_n in prev_intro or prev_intro in intro_n):
            same_intro += 1
        if concl_n and prev_conc and (concl_n == prev_conc or concl_n in prev_conc or prev_conc in concl_n):
            same_conclusion += 1
        for phrase in [
            "j ai recherche les patients",
            "les anomalies techniques ci dessous",
            "aucun resultat exploitable",
            "les resultats ci dessus sont strictement extraits",
        ]:
            if phrase in norm_text(str(item.get("answer_text") or "")):
                generic_hits += 1
    score = (same_intro / max(1, total)) * 0.5 + (same_conclusion / max(1, total)) * 0.3 + min(1.0, generic_hits / max(1, total)) * 0.2
    return round(min(1.0, score), 3)


def _quality_report(
    *,
    answer: str,
    validation: dict[str, Any],
    source_clickable_requested: bool,
    recent_style_history: list[dict[str, Any]],
) -> dict[str, Any]:
    errors = {str(e) for e in (validation.get("errors") or [])}
    warnings = {str(w) for w in (validation.get("warnings") or [])}
    faithfulness = 1.0 if not (errors & {"unsupported_value", "unsupported_analyte", "unsupported_source", "unsupported_reference", "unsupported_previous_result"}) else 0.0
    format_score = 1.0 if not (errors & {"format_not_respected", "output_format_not_respected", "yes_no_not_respected", "strict_json_violation"}) else 0.0
    source_score = 1.0
    if "source_format_bad" in errors or "unsupported_source" in errors:
        source_score = 0.0
    elif source_clickable_requested and "clickable_source_missing" in errors:
        source_score = 0.3
    safety_score = 1.0 if not (errors & {"diagnostic_safety_violation", "hallucinated_diagnosis"}) else 0.0
    readability = 1.0
    if "repeated_generic_style" in warnings or "repeated_generic_sentence" in warnings:
        readability -= 0.3
    if "missing_conclusion" in warnings:
        readability -= 0.2
    if "missing_query_criterion_in_intro" in warnings:
        readability -= 0.2
    if "ugly_pluralization" in warnings:
        readability -= 0.2
    readability = round(max(0.0, readability), 3)
    style_rep = compute_repetition_score(answer, recent_style_history)
    final_status = "pass"
    if faithfulness < 1.0 or safety_score < 1.0 or format_score < 0.95:
        final_status = "fail"
    elif warnings:
        final_status = "warning"
    return {
        "faithfulness_score": faithfulness,
        "format_compliance_score": format_score,
        "readability_score": readability,
        "source_ux_score": source_score,
        "style_repetition_score": style_rep,
        "safety_score": safety_score,
        "final_status": final_status,
    }


def build_structured_evidence_pack(
    *,
    query: str,
    query_understanding: QueryUnderstanding,
    sqlite_path: Path,
) -> dict[str, Any]:
    requested_doc_ids = list(query_understanding.requested_doc_ids or [])
    requested_analytes = list(query_understanding.requested_analytes or [])
    excluded_analytes = list(getattr(query_understanding, "excluded_analytes", []) or [])
    qn = norm_text(query)
    compare_previous = query_understanding.requires_previous_results
    intent = query_understanding.intent
    normalized_intent = intent
    if intent == "global_analyte_abnormal_search":
        normalized_intent = "cohort_search"
    elif intent == "doc_pair_comparison":
        normalized_intent = "multi_doc_comparison"
    elif intent == "doc_scoped_abnormal_results":
        normalized_intent = "doc_scoped_summary"
    elif intent == "single_analyte_lookup":
        normalized_intent = "doc_scoped_results"
    elif intent == "doc_scoped_medical_interpretation_guarded":
        normalized_intent = "diagnostic_safety_question"
    analyte_resolution_debug: dict[str, Any] | None = None

    pack: dict[str, Any] = {
        "question": query,
        "requested_doc_ids": requested_doc_ids,
        "requested_analytes": requested_analytes,
        "excluded_analytes": excluded_analytes,
        "requested_value": query_understanding.requested_value,
        "technical_condition": query_understanding.technical_condition,
        "intent": normalized_intent,
        "original_intent": intent,
        "output_format": query_understanding.output_format,
        "requested_table_columns": list(query_understanding.requested_table_columns or []),
        "answer_style": query_understanding.answer_style,
        "language": query_understanding.language,
        "requires_previous_results": compare_previous,
        "evidences": [],
        "missing_items": [],
        "safety_constraints": [],
        "rows": [],
        "comment_text": "",
    }

    if normalized_intent in {"global_patient_lookup", "cohort_search"}:
        target_values = [str(query_understanding.requested_value)] if query_understanding.requested_value else _extract_query_numeric_targets(query)
        rows = _fetch_global_lab_rows(
            sqlite_path=sqlite_path,
            analyte_norms=requested_analytes,
            requested_value=query_understanding.requested_value,
            limit=2000,
        )
        rows = [
            r
            for r in rows
            if _row_matches_value_criterion(r, target_values, query_understanding.comparison_operator)
        ]
        rows = _apply_technical_condition_filter(rows, query_understanding.technical_condition)
        if excluded_analytes:
            rows = [r for r in rows if not _row_matches_excluded(r, excluded_analytes)]
        evidences = [_structured_record_from_row(r) for r in rows]
        pack["rows"] = rows
        pack["evidences"] = evidences
        return _finalize_structured_pack(pack, query_understanding)

    rows: list[dict[str, Any]] = []

    if normalized_intent == "multi_doc_comparison" and len(requested_doc_ids) >= 2:
        resolved_analytes, analyte_resolution_debug = _resolve_analytes_for_query(
            query=query,
            requested_analytes=requested_analytes,
            sqlite_path=sqlite_path,
            max_candidates=5,
        )
        requested_analytes = list(resolved_analytes)
        pack["requested_analytes"] = list(requested_analytes)
        if analyte_resolution_debug:
            pack["analyte_resolution_debug"] = analyte_resolution_debug
        if not requested_analytes:
            pack["rows"] = []
            pack["evidences"] = []
            pack["missing_items"] = ["analyte_not_identified_for_multi_doc_comparison"]
            return _finalize_structured_pack(pack, query_understanding)
        rows = _fetch_doc_lab_rows(
            sqlite_path=sqlite_path,
            requested_doc_ids=requested_doc_ids,
            analyte_norms=requested_analytes,
            limit=600,
        )
        rows = _apply_technical_condition_filter(rows, query_understanding.technical_condition)
        if excluded_analytes:
            rows = [r for r in rows if not _row_matches_excluded(r, excluded_analytes)]
        if len(requested_doc_ids) > 2:
            rows_by_doc: dict[str, list[dict[str, Any]]] = {
                doc_id: [r for r in rows if str(r.get("doc_id") or "").strip().lower() == doc_id.lower()]
                for doc_id in requested_doc_ids
            }
            evidences: list[dict[str, Any]] = []
            missing: list[str] = []
            for analyte in requested_analytes:
                label = _canonical_display_name(analyte)
                per_doc = {doc_id: _best_row_for_analyte(rows_by_doc.get(doc_id, []), analyte) for doc_id in requested_doc_ids}
                present_rows = [r for r in per_doc.values() if isinstance(r, dict)]
                if not present_rows:
                    missing.append(analyte)
                    continue
                unit = ""
                ref = "non disponible"
                parts: list[str] = []
                missing_docs: list[str] = []
                for doc_id in requested_doc_ids:
                    row = per_doc.get(doc_id)
                    if not row:
                        parts.append(f"{doc_id}=non présent")
                        missing_docs.append(doc_id)
                        continue
                    value = str(row.get("value_raw") or "non disponible").strip()
                    unit = unit or str(row.get("unit") or "").strip()
                    if ref == "non disponible":
                        ref = str(row.get("reference_range") or "non disponible")
                    parts.append(f"{doc_id}={value}{(' ' + unit) if unit else ''}")
                evidences.append(
                    {
                        "doc_id": " vs ".join(requested_doc_ids),
                        "analyte": label,
                        "analyte_norm": analyte,
                        "current_value": " | ".join(parts),
                        "reference": ref,
                        "reference_summary": _summarize_reference_for_comparison(ref),
                        "technical_status_code": "not_interpretable",
                        "technical_status": "comparaison multi-doc",
                        "comparison_status": "non_comparable" if missing_docs else "comparison_ready",
                        "conclusion_fact": (
                            f"Donnée manquante pour {', '.join(missing_docs)}."
                            if missing_docs
                            else "Comparaison multi-doc disponible."
                        ),
                        "doc_ids": list(requested_doc_ids),
                        "source": _source_label(present_rows[0]),
                    }
                )
                if missing_docs:
                    missing.append(analyte)
            pack["evidences"] = evidences
            pack["missing_items"] = sorted(set(missing))
            pack["rows"] = rows
            if analyte_resolution_debug:
                analyte_resolution_debug["structured_rows_found"] = len(rows)
                analyte_resolution_debug["evidences_count"] = len(evidences)
            return _finalize_structured_pack(pack, query_understanding)
        left, right = requested_doc_ids[0], requested_doc_ids[1]
        left_rows = [r for r in rows if str(r.get("doc_id") or "").strip().lower() == left.lower()]
        right_rows = [r for r in rows if str(r.get("doc_id") or "").strip().lower() == right.lower()]
        evidences: list[dict[str, Any]] = []
        missing: list[str] = []
        for analyte in requested_analytes:
            a = _best_row_for_analyte(left_rows, analyte)
            b = _best_row_for_analyte(right_rows, analyte)
            label = _canonical_display_name(analyte)
            doc_a_label = str((a or {}).get("source_pdf") or left).split("/")[-1]
            doc_b_label = str((b or {}).get("source_pdf") or right).split("/")[-1]
            if not a and not b:
                missing.append(analyte)
                evidences.append(
                    {
                        "doc_id": f"{left} vs {right}",
                        "page": None,
                        "row": None,
                        "chunk_id": None,
                        "analyte": label,
                        "analyte_norm": analyte,
                        "current_value": f"{left}=non présent | {right}=non présent",
                        "unit": "",
                        "reference": "non disponible",
                        "reference_summary": "non disponible",
                        "previous_result": "",
                        "technical_status_code": "not_interpretable",
                        "technical_status": "non comparable",
                        "comparison_status": "non_comparable",
                        "conclusion_fact": "Aucune valeur exploitable n’a été retrouvée dans les deux rapports pour cet analyte.",
                        "doc_a": left,
                        "doc_b": right,
                        "doc_a_label": doc_a_label,
                        "doc_b_label": doc_b_label,
                        "variation": "non comparable",
                        "source": "",
                    }
                )
                continue
            av = str((a or {}).get("value_raw") or "non présent").strip()
            bv = str((b or {}).get("value_raw") or "non présent").strip()
            unit_a = str((a or {}).get("unit") or "").strip()
            unit_b = str((b or {}).get("unit") or "").strip()
            unit = unit_a or unit_b
            value_a_num = _parse_numeric_value(av)
            value_b_num = _parse_numeric_value(bv)
            comparison_status = "non_comparable"
            conclusion_fact = "Comparaison non exploitable numériquement."
            delta_abs: float | None = None
            delta_relative_percent: float | None = None
            delta_unit = unit if _units_compatible(unit_a, unit_b) else ""

            if a and not b:
                comparison_status = "missing_in_b"
                conclusion_fact = f"{label} est absent dans {right}."
            elif b and not a:
                comparison_status = "missing_in_a"
                conclusion_fact = f"{label} est absent dans {left}."
            elif value_a_num is not None and value_b_num is not None and _units_compatible(unit_a, unit_b):
                delta_abs = value_b_num - value_a_num
                if abs(delta_abs) <= 1e-12:
                    comparison_status = "identical"
                    delta_abs = 0.0
                    delta_relative_percent = 0.0
                    conclusion_fact = "Aucun écart numérique n’est observé."
                elif delta_abs > 0:
                    comparison_status = "increased"
                    if abs(value_a_num) > 1e-12:
                        delta_relative_percent = (delta_abs / value_a_num) * 100.0
                    conclusion_fact = "Valeur augmentée dans le second rapport."
                else:
                    comparison_status = "decreased"
                    if abs(value_a_num) > 1e-12:
                        delta_relative_percent = (delta_abs / value_a_num) * 100.0
                    conclusion_fact = "Valeur diminuée dans le second rapport."
            elif a and b and (not _units_compatible(unit_a, unit_b)):
                comparison_status = "non_comparable"
                conclusion_fact = "Comparaison non comparable : unités différentes."
            elif a and b:
                comparison_status = "non_comparable"
                conclusion_fact = "Comparaison non comparable : valeur non numérique."

            ref_a = str((a or {}).get("reference_range") or "").strip()
            ref_b = str((b or {}).get("reference_range") or "").strip()
            ref_raw = ref_a or ref_b or "non disponible"
            reference_summary = _summarize_reference_for_comparison(ref_raw)
            evidences.append(
                {
                    "doc_id": f"{left} vs {right}",
                    "page": (a or b or {}).get("page_number"),
                    "row": (a or b or {}).get("row_index"),
                    "page_number": (a or b or {}).get("page_number"),
                    "row_index": (a or b or {}).get("row_index"),
                    "chunk_id": None,
                    "analyte": label,
                    "analyte_norm": analyte,
                    "current_value": f"{left}={av} | {right}={bv}",
                    "value_a_raw": av,
                    "value_b_raw": bv,
                    "value_a_num": value_a_num,
                    "value_b_num": value_b_num,
                    "unit": unit,
                    "unit_a": unit_a,
                    "unit_b": unit_b,
                    "reference": ref_raw,
                    "reference_summary": reference_summary,
                    "previous_result": "",
                    "technical_status_code": "within_reference" if comparison_status == "identical" else "not_interpretable",
                    "technical_status": "valeurs identiques" if comparison_status == "identical" else "comparaison effectuée",
                    "comparison_status": comparison_status,
                    "delta_abs": delta_abs,
                    "delta_unit": delta_unit,
                    "delta_relative_percent": delta_relative_percent,
                    "conclusion_fact": conclusion_fact,
                    "doc_a": left,
                    "doc_b": right,
                    "doc_a_label": doc_a_label,
                    "doc_b_label": doc_b_label,
                    "source_a": {
                        "doc_id": left,
                        "source_pdf": (a or {}).get("source_pdf"),
                        "page": (a or {}).get("page_number"),
                        "line": (a or {}).get("row_index"),
                    } if a else None,
                    "source_b": {
                        "doc_id": right,
                        "source_pdf": (b or {}).get("source_pdf"),
                        "page": (b or {}).get("page_number"),
                        "line": (b or {}).get("row_index"),
                    } if b else None,
                    "comparison_sources": [
                        {
                            "doc_id": left,
                            "source_pdf": (a or {}).get("source_pdf"),
                            "page_number": (a or {}).get("page_number"),
                            "row_index": (a or {}).get("row_index"),
                        }
                    ]
                    + (
                        [
                            {
                                "doc_id": right,
                                "source_pdf": (b or {}).get("source_pdf"),
                                "page_number": (b or {}).get("page_number"),
                                "row_index": (b or {}).get("row_index"),
                            }
                        ]
                        if b
                        else []
                    ),
                    "variation": _variation_label(bv, av) if a and b else "non comparable",
                    "source": _source_label((a or b or {})),
                    "source_pdf": (a or b or {}).get("source_pdf"),
                }
            )
        pack["evidences"] = evidences
        pack["missing_items"] = missing
        pack["rows"] = rows
        if analyte_resolution_debug:
            analyte_resolution_debug["structured_rows_found"] = len(rows)
            analyte_resolution_debug["evidences_count"] = len(evidences)
        return _finalize_structured_pack(pack, query_understanding)

    if intent == "multi_doc_presence_diff" and len(requested_doc_ids) >= 2:
        left, right = requested_doc_ids[0], requested_doc_ids[1]
        rows = _fetch_doc_lab_rows(
            sqlite_path=sqlite_path,
            requested_doc_ids=requested_doc_ids,
            analyte_norms=None,
            limit=2500,
        )
        left_rows = [r for r in rows if str(r.get("doc_id") or "").strip().lower() == left.lower()]
        right_rows = [r for r in rows if str(r.get("doc_id") or "").strip().lower() == right.lower()]
        left_keys: dict[str, dict[str, Any]] = {}
        right_keys: dict[str, dict[str, Any]] = {}
        for row in left_rows:
            key = str(row.get("analyte_norm") or "").strip().lower() or norm_text(str(row.get("analyte") or ""))
            if excluded_analytes and _row_matches_excluded(row, excluded_analytes):
                continue
            if not is_valid_analyte_name(str(row.get("analyte") or row.get("parameter") or key)):
                continue
            if key and key not in left_keys:
                left_keys[key] = row
        for row in right_rows:
            key = str(row.get("analyte_norm") or "").strip().lower() or norm_text(str(row.get("analyte") or ""))
            if excluded_analytes and _row_matches_excluded(row, excluded_analytes):
                continue
            if not is_valid_analyte_name(str(row.get("analyte") or row.get("parameter") or key)):
                continue
            if key and key not in right_keys:
                right_keys[key] = row

        only_left = sorted(set(left_keys.keys()) - set(right_keys.keys()))
        only_right = sorted(set(right_keys.keys()) - set(left_keys.keys()))
        evidences: list[dict[str, Any]] = []
        rows_for_pack: list[dict[str, Any]] = []
        for key in only_left:
            row = left_keys[key]
            evidences.append(
                {
                    "doc_id": left,
                    "analyte": _clean_analyte_label(str(row.get("analyte") or row.get("parameter") or key)),
                    "analyte_norm": key,
                    "present_in": left,
                    "absent_in": right,
                    "source": _source_label(row),
                }
            )
            rows_for_pack.append(row)
        for key in only_right:
            row = right_keys[key]
            evidences.append(
                {
                    "doc_id": right,
                    "analyte": _clean_analyte_label(str(row.get("analyte") or row.get("parameter") or key)),
                    "analyte_norm": key,
                    "present_in": right,
                    "absent_in": left,
                    "source": _source_label(row),
                }
            )
            rows_for_pack.append(row)
        pack["rows"] = rows_for_pack
        pack["evidences"] = evidences
        return _finalize_structured_pack(pack, query_understanding)

    if intent == "comment_without_measured_value":
        qualitative_fetch_called = True
        wants_single_comment = _wants_single_comment(query)
        wants_all_comments = _wants_all_comments_listing(query) and not wants_single_comment
        wants_latest_comment = bool(getattr(query_understanding, "latest_report", False))
        comment_subject_norm, comment_subject_label = _resolve_comment_subject_from_query(
            query_understanding=query_understanding,
            query=query,
        )
        analyte_targets = [comment_subject_norm] if comment_subject_norm else []
        search_term = comment_subject_norm or "commentaire"
        rows = (
            _fetch_doc_lab_rows(
                sqlite_path=sqlite_path,
                requested_doc_ids=requested_doc_ids,
                analyte_norms=analyte_targets,
                include_text_search_terms=[search_term],
                limit=250,
            )
            if requested_doc_ids
            else _fetch_global_comment_rows(sqlite_path=sqlite_path, term=search_term, limit=250)
        )
        # Fallback for "latest report": if scoped latest doc has no comment row,
        # retry globally and keep the latest doc that actually contains a qualitative comment.
        if bool(getattr(query_understanding, "latest_report", False)) and requested_doc_ids:
            global_rows = _fetch_global_comment_rows(sqlite_path=sqlite_path, term=search_term, limit=600)
            candidate_rows = [r for r in global_rows if _row_looks_like_qualitative_comment(r)]
            scoped_comment_rows = [r for r in rows if _row_looks_like_qualitative_comment(r)]
            if candidate_rows and not scoped_comment_rows:
                latest_doc = max(
                    (
                        str(r.get("doc_id") or "").strip()
                        for r in candidate_rows
                        if str(r.get("doc_id") or "").strip()
                    ),
                    key=_doc_recency_key,
                    default="",
                )
                if latest_doc:
                    rows = [r for r in candidate_rows if str(r.get("doc_id") or "").strip().lower() == latest_doc.lower()]
                    pack["requested_doc_ids"] = [latest_doc]
        qualitative_rows_count = len(rows)
        comment_rows = [r for r in rows if _row_looks_like_qualitative_comment(r)]
        measured = [
            r
            for r in rows
            if (
                not comment_subject_norm
                or comment_subject_norm in norm_text(str(r.get("analyte_norm") or ""))
                or comment_subject_norm in norm_text(str(r.get("analyte") or ""))
            )
            and norm_text(str(r.get("analyte") or "")) != "commentaire"
            and str(r.get("value_raw") or "").strip() != ""
            and not _row_looks_like_qualitative_comment(r)
        ]
        if measured:
            pack["evidences"] = [_structured_record_from_row(measured[0])]
            pack["qualitative_debug"] = {
                "enters_comment_without_measured_value_branch": True,
                "qualitative_fetch_called": qualitative_fetch_called,
                "qualitative_rows_count": qualitative_rows_count,
                "qualitative_comment_extracted": False,
                "qualitative_comment_text_length": 0,
                "qualitative_comment_text_preview": "",
            }
        else:
            if (wants_all_comments or wants_single_comment or wants_latest_comment) and not comment_subject_norm:
                seen_fingerprints: set[str] = set()
                best_by_doc: dict[str, dict[str, Any]] = {}
                variants_by_doc: dict[str, list[str]] = {}
                for cr in comment_rows:
                    raw_comment_text = str(cr.get("value_raw") or cr.get("text_for_keyword") or cr.get("text_for_embedding") or "").strip()
                    if not raw_comment_text:
                        continue
                    display_comment_text = clean_qualitative_comment_text(raw_comment_text, "commentaire")
                    if not display_comment_text or _is_low_signal_comment_text(display_comment_text):
                        continue
                    row_subject = str(cr.get("analyte") or cr.get("parameter") or "").strip() or "Commentaire médical"
                    if _is_generic_subject(row_subject):
                        row_subject = "Commentaire médical"
                    row_unit = str(cr.get("unit") or "").strip()
                    display_comment_text = _enrich_comment_with_unit_if_missing(display_comment_text, row_unit)
                    fingerprint = _norm_comment_fingerprint(display_comment_text)
                    if not fingerprint or fingerprint in seen_fingerprints:
                        continue
                    candidate = {
                        "doc_id": str(cr.get("doc_id") or ""),
                        "patient_token": str(cr.get("patient_token") or "").strip(),
                        "page": cr.get("page_number"),
                        "page_number": cr.get("page_number"),
                        "row": cr.get("row_index"),
                        "row_index": cr.get("row_index"),
                        "chunk_id": cr.get("chunk_id"),
                        "analyte": row_subject,
                        "subject": row_subject,
                        "analyte_norm": norm_text(row_subject),
                        "current_value": display_comment_text,
                        "raw_comment_text": raw_comment_text,
                        "display_comment_text": display_comment_text,
                        "comment_text": display_comment_text,
                        "unit": "qualitative",
                        "reference": str(cr.get("reference_range") or "").strip(),
                        "previous_result": "",
                        "technical_status_code": "not_interpretable",
                        "technical_status": "commentaire qualitatif",
                        "variation": "non comparable",
                        "source": _source_label(cr),
                        "source_pdf": str(cr.get("source_pdf") or ""),
                        "result_kind": "comment",
                    }
                    doc_key = str(candidate.get("doc_id") or "").strip().lower() or str(candidate.get("source_pdf") or "").strip().lower()
                    if not doc_key:
                        doc_key = fingerprint[:60]
                    existing = best_by_doc.get(doc_key)
                    variants_by_doc.setdefault(doc_key, []).append(display_comment_text)
                    if existing is None:
                        best_by_doc[doc_key] = candidate
                        seen_fingerprints.add(fingerprint)
                        continue
                    existing_txt = str(existing.get("display_comment_text") or "")
                    existing_precise = existing.get("page") is not None or existing.get("row") is not None
                    candidate_precise = candidate.get("page") is not None or candidate.get("row") is not None
                    if (candidate_precise and not existing_precise) or (len(display_comment_text) > len(existing_txt)):
                        best_by_doc[doc_key] = candidate
                        seen_fingerprints.add(fingerprint)
                for doc_key, best in list(best_by_doc.items()):
                    merged_text = _merge_comment_variants(variants_by_doc.get(doc_key, []))
                    if merged_text:
                        best["display_comment_text"] = merged_text
                        best["comment_text"] = merged_text
                        best["current_value"] = merged_text
                built = sorted(
                    list(best_by_doc.values()),
                    key=lambda ev: (
                        str(ev.get("doc_id") or "").lower(),
                        int(ev.get("page")) if isinstance(ev.get("page"), int) else 999999,
                        int(ev.get("row")) if isinstance(ev.get("row"), int) else 999999,
                    ),
                )
                if (wants_single_comment or wants_latest_comment) and built:
                    built = [max(built, key=lambda ev: _doc_recency_key(str(ev.get("doc_id") or "")))]
                pack["comment_list_mode"] = bool(built)
                pack["evidences"] = built
            else:
                extraction_subject = comment_subject_norm or search_term
                comment_text, cr = extract_comment_text_for_subject(extraction_subject, comment_rows or rows)
                if comment_text and isinstance(cr, dict):
                    raw_comment_text = str(cr.get("value_raw") or cr.get("text_for_keyword") or cr.get("text_for_embedding") or "").strip()
                    display_comment_text = clean_qualitative_comment_text(raw_comment_text or comment_text, extraction_subject) or comment_text
                    row_subject = str(cr.get("analyte") or cr.get("parameter") or "").strip()
                    final_subject = comment_subject_label if comment_subject_norm else (row_subject or comment_subject_label)
                    final_subject_norm = comment_subject_norm or norm_text(final_subject)
                    pack["comment_text"] = display_comment_text
                    pack["raw_comment_text"] = raw_comment_text
                    pack["evidences"] = [
                        {
                            "doc_id": str(cr.get("doc_id") or ""),
                            "patient_token": str(cr.get("patient_token") or "").strip(),
                            "page": cr.get("page_number"),
                            "page_number": cr.get("page_number"),
                            "row": cr.get("row_index"),
                            "row_index": cr.get("row_index"),
                            "chunk_id": cr.get("chunk_id"),
                            "analyte": final_subject,
                            "subject": final_subject,
                            "analyte_norm": final_subject_norm,
                            "current_value": display_comment_text,
                            "raw_comment_text": raw_comment_text,
                            "display_comment_text": display_comment_text,
                            "comment_text": display_comment_text,
                            "unit": "qualitative",
                            "reference": str(cr.get("reference_range") or "").strip(),
                            "previous_result": "",
                            "technical_status_code": "not_interpretable",
                            "technical_status": "commentaire qualitatif",
                            "variation": "non comparable",
                            "source": _source_label(cr),
                            "source_pdf": str(cr.get("source_pdf") or ""),
                            "result_kind": "comment",
                        }
                    ]
            pack["qualitative_debug"] = {
                "enters_comment_without_measured_value_branch": True,
                "qualitative_fetch_called": qualitative_fetch_called,
                "qualitative_rows_count": qualitative_rows_count,
                "qualitative_comment_extracted": bool(pack.get("evidences")),
                "qualitative_comment_text_length": len(str(pack.get("comment_text") or "")),
                "qualitative_comment_text_preview": str(str(pack.get("comment_text") or "")[:80]),
                "qualitative_comment_candidates_count": len(comment_rows),
                "qualitative_comment_list_mode": bool(pack.get("comment_list_mode")),
            }
        pack["rows"] = rows
        return _finalize_structured_pack(pack, query_understanding)

    if not requested_doc_ids:
        return _finalize_structured_pack(pack, query_understanding)

    if intent == "diagnostic_safety_question":
        if requested_analytes:
            safety_analytes = requested_analytes
        else:
            topic = detect_medical_topic(query or "")
            topic_analytes = get_topic_analytes(topic)
            safety_analytes = topic_analytes if topic_analytes else get_topic_analytes("tumor_markers")
            excluded = set(get_topic_exclusions(topic))
            explicit = set(str(a).strip().lower() for a in requested_analytes)
            if excluded:
                safety_analytes = [a for a in safety_analytes if (str(a).strip().lower() not in excluded) or (str(a).strip().lower() in explicit)]
        rows = _fetch_doc_lab_rows(
            sqlite_path=sqlite_path,
            requested_doc_ids=requested_doc_ids,
            analyte_norms=safety_analytes,
            limit=250,
        )
        evidences: list[dict[str, Any]] = []
        missing: list[str] = []
        for analyte in safety_analytes:
            row = _best_row_for_analyte(rows, analyte)
            if row is None:
                missing.append(analyte)
                continue
            evidences.append(_structured_record_from_row(row))
        pack["evidences"] = evidences
        pack["missing_items"] = missing
        pack["safety_constraints"] = ["no_diagnosis_conclusion"]
        pack["rows"] = rows
        return _finalize_structured_pack(pack, query_understanding)

    if intent in {"toxicology_summary", "doc_scoped_summary", "immunoanalysis_summary", "doc_scoped_results", "previous_result_comparison", "unstructured"}:
        analytes = requested_analytes if requested_analytes else None
        rows = _fetch_doc_lab_rows(
            sqlite_path=sqlite_path,
            requested_doc_ids=requested_doc_ids,
            analyte_norms=analytes,
            limit=700,
        )
        target_values = [str(query_understanding.requested_value)] if query_understanding.requested_value else _extract_query_numeric_targets(query)
        if query_understanding.comparison_operator and target_values:
            rows = [
                r
                for r in rows
                if _row_matches_value_criterion(r, target_values, query_understanding.comparison_operator)
            ]
        rows = _apply_technical_condition_filter(rows, query_understanding.technical_condition)
        if intent == "toxicology_summary":
            urine_mode = any(k in qn for k in ["urinaire", "urinaires", "urine"])
            tox_terms = ["ethanol", "acide_valproique", "carbamazepine", "lithium"]
            urine_terms = ["amphetamine", "benzodiazepine", "cocaine", "opiaces", "ecstasy", "phencyclidine"]
            if requested_analytes:
                target_analytes = requested_analytes
                rows = [r for r in rows if any(_row_matches_analyte(r, a) for a in target_analytes)]
            elif urine_mode:
                target_analytes = []
                rows = [
                    r
                    for r in rows
                    if any(
                        t in norm_text(str(r.get("analyte_norm") or "") + " " + str(r.get("analyte") or ""))
                        for t in urine_terms
                    )
                ]
            else:
                target_analytes = tox_terms
                rows = [r for r in rows if any(_row_matches_analyte(r, a) for a in target_analytes)]
            if any(k in qn for k in ["depass", "dépass", "au dessus", "au-dessus"]) and "reference" in qn:
                rows = [r for r in rows if _status_code(r) == "above_reference"]
            if compare_previous:
                rows = [r for r in rows if str(r.get("previous_result_value_raw") or "").strip()]
            requested_analytes = target_analytes

        if excluded_analytes:
            rows = [r for r in rows if not _row_matches_excluded(r, excluded_analytes)]

        if intent != "toxicology_summary" and query_understanding.requires_section_summary and ("urinaire" in qn or "urinaires" in qn or "urine" in qn):
            rows = [r for r in rows if "urina" in norm_text(str(r.get("analyte_norm") or "") + " " + str(r.get("analyte") or ""))]

        if not rows and intent in {"doc_scoped_summary", "immunoanalysis_summary"}:
            summary_rows = _fetch_doc_summary_rows(
                sqlite_path=sqlite_path,
                requested_doc_ids=requested_doc_ids,
                limit=20,
            )
            if summary_rows:
                rows = summary_rows

        evidences: list[dict[str, Any]] = []
        rows_for_pack: list[dict[str, Any]] = []
        missing: list[str] = []
        out_of_reference_only = _query_requests_out_of_reference_only(qn)
        if "priority_level" in {str(c).strip().lower() for c in list(getattr(query_understanding, "requested_table_columns", []) or [])}:
            out_of_reference_only = False
        effective_tc = _canonical_technical_condition(query_understanding.technical_condition)
        yes_no_mode = str(query_understanding.answer_style or "").strip().lower() == "yes_no" or str(query_understanding.output_format or "").strip().lower() == "yes_no"
        if requested_analytes:
            full_rows = list(rows)
            filtered_rows = list(rows)
            if effective_tc == "out_of_reference" or out_of_reference_only:
                filtered_rows = [r for r in rows if _status_code(r) in {"above_reference", "below_reference"}]
            for analyte in requested_analytes:
                row = _best_row_for_analyte(filtered_rows, analyte)
                if row is None and yes_no_mode and out_of_reference_only:
                    row = _best_row_for_analyte(full_rows, analyte)
                if row is None:
                    # Do not mark as missing when analyte exists but is excluded by a user filter.
                    if not _best_row_for_analyte(full_rows, analyte):
                        missing.append(analyte)
                    continue
                record = _structured_record_from_row(row)
                if compare_previous and not record.get("previous_result"):
                    record["variation"] = "non comparable"
                evidences.append(record)
                rows_for_pack.append(row)
        else:
            for row in rows:
                status = _status_code(row)
                if (effective_tc == "out_of_reference" or out_of_reference_only) and status not in {
                    "above_reference",
                    "below_reference",
                }:
                    continue
                evidences.append(_structured_record_from_row(row))
                rows_for_pack.append(row)
        pack["evidences"] = evidences
        pack["missing_items"] = missing
        pack["rows"] = rows_for_pack
        return _finalize_structured_pack(pack, query_understanding)

    return _finalize_structured_pack(pack, query_understanding)


def _filter_rows_for_analyte(rows: list[dict[str, Any]], analyte_norm: str) -> list[dict[str, Any]]:
    return [row for row in rows if _row_matches_analyte(row, analyte_norm)]


def _best_row_for_analyte(rows: list[dict[str, Any]], analyte_norm: str) -> dict[str, Any] | None:
    candidates = _filter_rows_for_analyte(rows, analyte_norm)
    if not candidates:
        return None

    def score(row: dict[str, Any]) -> tuple[int, int]:
        has_value = 1 if str(row.get("value_raw") or "").strip() else 0
        has_ref = 1 if str(row.get("reference_range") or "").strip() else 0
        return (has_value + has_ref, -int(row.get("row_index") or 999999))

    return sorted(candidates, key=score, reverse=True)[0]


def _format_doc_analyte_rows_answer(
    *,
    rows: list[dict[str, Any]],
    requested_doc_id: str,
    requested_analytes: list[str],
    compare_previous: bool,
    include_missing: bool = True,
) -> tuple[str, list[str]]:
    lines: list[str] = []
    missing: list[str] = []
    analytes = requested_analytes or []

    for analyte in analytes:
        row = _best_row_for_analyte(rows, analyte)
        display_name = _canonical_display_name(analyte)
        if row is None:
            if include_missing:
                lines.append(f"- {display_name}: non retrouvé dans {requested_doc_id}.")
            missing.append(analyte)
            continue

        value = str(row.get("value_raw") or "non disponible")
        unit = str(row.get("unit") or "").strip()
        ref = str(row.get("reference_range") or "non disponible")
        status = _interpretation_fr(str(row.get("interpretation_status") or "unknown"))
        previous = str(row.get("previous_result_value_raw") or "").strip()
        core = f"- {display_name}: {value}"
        if unit:
            core += f" {unit}"
        core += f" | référence: {ref} | statut technique: {status}"
        if compare_previous:
            if previous:
                variation = _variation_label(value, previous)
                core += f" | antérieur: {previous} | variation: {variation}"
            else:
                core += " | antérieur: non disponible"
        lines.append(core)

    return "\n".join(lines).strip(), missing


def _format_doc_summary_answer(
    *,
    rows: list[dict[str, Any]],
    query_norm: str,
    compare_previous: bool = False,
) -> str:
    if not rows:
        return _missing_doc_answer()

    wants_above_only = _is_above_reference_query(query_norm) and not _is_normal_or_above_query(query_norm)
    wants_below_only = _is_below_reference_query(query_norm)

    selected: list[dict[str, Any]] = []
    for row in rows:
        status = str(row.get("interpretation_status") or "").lower()
        if wants_above_only and status != "above_reference":
            continue
        if wants_below_only and status != "below_reference":
            continue
        if ("hors reference" in query_norm or "anomal" in query_norm or "attention technique" in query_norm) and status not in {
            "above_reference",
            "below_reference",
        }:
            continue
        selected.append(row)

    if not selected:
        selected = rows

    groups = {"sanguins": [], "urinaires": [], "sero_diagnostic": []}
    for row in selected:
        analyte_norm = norm_text(str(row.get("analyte_norm") or ""))
        analyte = norm_text(str(row.get("analyte") or ""))
        label = _clean_analyte_label(str(row.get("analyte") or row.get("parameter") or "non précisé"))
        status = _interpretation_fr(str(row.get("interpretation_status") or "unknown"))
        value = str(row.get("value_raw") or "non disponible")
        unit = str(row.get("unit") or "").strip()
        ref = str(row.get("reference_range") or "non disponible")
        chunk = f"- {label}: {value}" + (f" {unit}" if unit else "") + f" | référence: {ref} | statut: {status}"
        if compare_previous:
            previous = str(row.get("previous_result_value_raw") or "").strip()
            if previous:
                chunk += f" | antérieur: {previous} | variation: {_variation_label(value, previous)}"
            else:
                chunk += " | antérieur: non disponible"

        if any(k in analyte_norm or k in analyte for k in ["microalbuminurie", "urina", "urinaire", "cocaine", "amphetamine", "benzodiazepine", "opiaces"]):
            groups["urinaires"].append(chunk)
        elif any(k in analyte_norm or k in analyte for k in ["sero", "aslo", "igg", "igm", "ige", "complement", "c3", "c4"]):
            groups["sero_diagnostic"].append(chunk)
        else:
            groups["sanguins"].append(chunk)

    lines: list[str] = []
    for title, key in [("Examens sanguins", "sanguins"), ("Examens urinaires", "urinaires"), ("Séro-diagnostic", "sero_diagnostic")]:
        lines.append(f"{title} :")
        if groups[key]:
            lines.extend(groups[key])
        else:
            lines.append("- non retrouvé")
    return "\n".join(lines).strip()


def _format_multi_doc_comparison_answer(
    *,
    rows: list[dict[str, Any]],
    doc_ids: list[str],
    requested_analytes: list[str],
) -> tuple[str, list[str]]:
    if len(doc_ids) < 2:
        return _missing_doc_answer(), list(requested_analytes)

    rows_by_doc: dict[str, list[dict[str, Any]]] = {
        doc_id: [r for r in rows if str(r.get("doc_id") or "").strip().lower() == doc_id.lower()]
        for doc_id in doc_ids
    }
    missing: list[str] = []
    lines: list[str] = []

    for analyte in requested_analytes:
        label = _canonical_display_name(analyte)
        per_doc_best: dict[str, dict[str, Any] | None] = {
            doc_id: _best_row_for_analyte(rows_by_doc.get(doc_id, []), analyte) for doc_id in doc_ids
        }
        present_rows = [r for r in per_doc_best.values() if isinstance(r, dict)]
        if not present_rows:
            lines.append(f"- {label}: non retrouvé dans {', '.join(doc_ids)}.")
            missing.append(analyte)
            continue

        unit = ""
        ref = "non disponible"
        for row in present_rows:
            unit = unit or str(row.get("unit") or "").strip()
            ref = ref if ref != "non disponible" else str(row.get("reference_range") or "non disponible")
        per_doc_parts: list[str] = []
        for doc_id in doc_ids:
            row = per_doc_best.get(doc_id)
            if not row:
                per_doc_parts.append(f"{doc_id}=non présent")
                continue
            value = str(row.get("value_raw") or "non disponible").strip()
            value_txt = value + (f" {unit}" if unit else "")
            per_doc_parts.append(f"{doc_id}={value_txt}")
        lines.append(f"- {label}: " + " | ".join(per_doc_parts) + f" | référence: {ref}")

    return "\n".join(lines).strip(), missing


def _format_qualitative_comment_answer(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return _missing_doc_answer()

    first = rows[0] if rows and isinstance(rows[0], dict) else {}
    subject = str(first.get("analyte") or first.get("parameter") or first.get("analyte_norm") or "ce paramètre").strip()
    subject_norm = norm_text(subject)
    measured = []
    for r in rows:
        an_norm = norm_text(str(r.get("analyte_norm") or ""))
        an_lbl = norm_text(str(r.get("analyte") or ""))
        if subject_norm and subject_norm not in an_norm and subject_norm not in an_lbl:
            continue
        if norm_text(str(r.get("analyte") or "")) == "commentaire":
            continue
        if str(r.get("value_raw") or "").strip() == "":
            continue
        measured.append(r)
    if measured:
        row = measured[0]
        unit = str(row.get("unit") or "").strip()
        ref = str(row.get("reference_range") or "non disponible")
        return (
            f"Une valeur mesurée est retrouvée pour {subject}: {row.get('value_raw')}"
            + (f" {unit}" if unit else "")
            + f" (référence: {ref})."
        )

    comment_rows = []
    for r in rows:
        raw = norm_text(str(r.get("value_raw") or ""))
        if subject_norm and subject_norm in raw:
            comment_rows.append(r)
    if comment_rows:
        row = comment_rows[0]
        comment = str(row.get("value_raw") or "").strip()
        snippet = comment if len(comment) <= 220 else comment[:217] + "..."
        return (
            f"Aucune valeur mesurée n’est retrouvée pour {subject} ; le document contient seulement un commentaire/interprétation "
            f"avec seuil. Extrait: {snippet}"
        )
    return _missing_doc_answer()


def _format_diagnostic_safety_answer(rows: list[dict[str, Any]], requested_analytes: list[str], requested_doc_id: str | None) -> tuple[str, list[str]]:
    marker_rows = rows
    if requested_analytes:
        filtered: list[dict[str, Any]] = []
        for analyte in requested_analytes:
            best = _best_row_for_analyte(rows, analyte)
            if best:
                filtered.append(best)
        marker_rows = filtered

    lines = [
        "Non, on ne peut pas conclure à un cancer uniquement à partir de ces marqueurs.",
        "Constat technique sur les marqueurs retrouvés :",
    ]
    missing: list[str] = []
    if requested_analytes:
        for analyte in requested_analytes:
            best = _best_row_for_analyte(rows, analyte)
            label = _canonical_display_name(analyte)
            if not best:
                lines.append(f"- {label}: non retrouvé dans {requested_doc_id or 'le document demandé'}.")
                missing.append(analyte)
                continue
            lines.append(
                f"- {label}: {best.get('value_raw')} {best.get('unit') or ''} | "
                f"référence: {best.get('reference_range') or 'non disponible'} | "
                f"statut technique: {_interpretation_fr(best.get('interpretation_status'))}"
            )
    else:
        if not marker_rows:
            lines.append("- Aucun marqueur demandé retrouvé.")
        for row in marker_rows:
            lines.append(
                f"- {_clean_analyte_label(row.get('analyte'))}: {row.get('value_raw')} {row.get('unit') or ''} | "
                f"référence: {row.get('reference_range') or 'non disponible'} | "
                f"statut technique: {_interpretation_fr(row.get('interpretation_status'))}"
            )
    lines.append("Ces marqueurs biologiques ne suffisent pas à poser un diagnostic ; une interprétation médicale spécialisée est nécessaire.")
    return "\n".join(lines).strip(), missing


def _count_displayed_exact_analyte(answer: str, analyte: str) -> int:
    text = norm_text(answer or "")
    a = norm_text(analyte or "")
    if not text or not a:
        return 0
    pattern = re.compile(rf"(?:^|\s){re.escape(a)}\s*(?:=|:)", re.IGNORECASE)
    return len(pattern.findall(text))


def _normalize_transform_evidence_item(ev: dict[str, Any]) -> dict[str, Any]:
    status_info = normalize_result_status(ev)
    status_code = status_info["raw_status"]
    page = ev.get("page")
    if page in (None, ""):
        page = ev.get("page_number")
    row = ev.get("row")
    if row in (None, ""):
        row = ev.get("row_index")
    current_value = ev.get("current_value")
    if current_value in (None, ""):
        current_value = ev.get("value_raw")
    reference = ev.get("reference")
    if reference in (None, ""):
        reference = ev.get("reference_range")
    out = dict(ev)
    out.update(
        {
            "doc_id": ev.get("doc_id"),
            "analyte": ev.get("analyte") or ev.get("parameter"),
            "analyte_norm": ev.get("analyte_norm"),
            "current_value": current_value,
            "value_raw": ev.get("value_raw") if ev.get("value_raw") not in (None, "") else current_value,
            "value_numeric": ev.get("value_numeric"),
            "unit": ev.get("unit"),
            "reference": reference,
            "reference_range": ev.get("reference_range") if ev.get("reference_range") not in (None, "") else reference,
            "previous_result": ev.get("previous_result") or ev.get("previous_result_value_raw"),
            "technical_status_code": status_code or "not_interpretable",
            "interpretation_status": status_code or "not_interpretable",
            "technical_status": str(ev.get("technical_status") or ev.get("status") or status_info["display_status"]),
            "source": ev.get("source"),
            "source_pdf": ev.get("source_pdf"),
            "page": page,
            "page_number": page,
            "row": row,
            "row_index": row,
            "viewer_url": ev.get("viewer_url"),
            "source_url": ev.get("source_url"),
        }
    )
    return out


def _build_response_transform_pack(
    *,
    query: str,
    query_understanding: QueryUnderstanding,
    previous_pack: dict[str, Any],
) -> dict[str, Any]:
    qn = norm_text(query)
    src = dict(previous_pack or {})
    evidences = [_normalize_transform_evidence_item(dict(ev)) for ev in (src.get("evidences") or [])]

    if "au dessus de la reference" in qn or "au-dessus de la reference" in qn or "above reference" in qn:
        evidences = [ev for ev in evidences if str(ev.get("technical_status_code") or "").strip().lower() == "above_reference"]
    elif "en dessous de la reference" in qn or "below reference" in qn:
        evidences = [ev for ev in evidences if str(ev.get("technical_status_code") or "").strip().lower() == "below_reference"]

    requested_columns = list(query_understanding.requested_table_columns or src.get("requested_table_columns") or [])
    if ("sans la colonne source" in qn or "without source" in qn or "without the source column" in qn) and requested_columns:
        requested_columns = [c for c in requested_columns if str(c).strip().lower() != "source"]
    elif ("sans la colonne source" in qn or "without source" in qn or "without the source column" in qn) and not requested_columns:
        if str(src.get("intent") or "") in {"cohort_search", "global_patient_lookup"}:
            requested_columns = ["patient", "report", "analyte", "valeur_actuelle", "reference", "statut"]
        else:
            requested_columns = ["analyte", "valeur_actuelle", "unite", "reference", "statut", "resultat_anterieur", "variation"]

    output_format = str(query_understanding.output_format or "auto")
    if output_format in {"list", "auto"}:
        output_format = str(src.get("output_format") or "list").lower()
    if output_format in {"list", "auto"} and ("json" in qn):
        output_format = "json"
    if output_format in {"list", "auto"} and ("tableau" in qn or "table" in qn):
        output_format = "table"

    return {
        **src,
        "question": query,
        "intent": "response_transform",
        "output_format": output_format,
        "requested_table_columns": requested_columns,
        "answer_style": query_understanding.answer_style or src.get("answer_style") or "standard",
        "evidences": evidences,
        "results": list(evidences),
    }


def run_generation(
    *,
    query: str,
    top_k: int = 5,
    mode: str = "hybrid",
    provider: str = DEFAULT_LLM_PROVIDER,
    model: str = DEFAULT_LLM_MODEL,
    temperature: float = DEFAULT_LLM_TEMPERATURE,
    num_ctx: int = DEFAULT_LLM_NUM_CTX,
    max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
    timeout: int = DEFAULT_LLM_TIMEOUT,
    index_dir: str | Path = "data/indexes",
    collection: str = "medical_chunks",
    search_engine: SearchEngine | None = None,
    llm_client: LLMClient | None = None,
    max_display_results: int = 3,
    show_all_results: bool = False,
    show_low_quality: bool = False,
    previous_structured_evidence_pack: dict[str, Any] | None = None,
    previous_displayed_evidence_pack: dict[str, Any] | None = None,
    previous_displayed_context: dict[str, Any] | None = None,
    previous_context_intent: str | None = None,
    previous_data_context_intent: str | None = None,
    previous_data_context_type: str | None = None,
    previous_doc_scope: list[str] | None = None,
    previous_qualitative_evidence_pack: dict[str, Any] | None = None,
    previous_has_patient_inventory: bool = False,
    previous_patient_inventory: list[dict[str, Any]] | None = None,
    recent_style_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    stage_times_ms: dict[str, float] = {
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
    request_id = str(uuid4())

    query_received = query
    q = normalize_query(query_received)
    query_used_for_retrieval = q
    query_used_for_prompt = q
    qn = norm_text(q)
    style_history = list(recent_style_history or [])
    composed_data = {}
    source_citations_for_response: list[dict[str, Any]] = []
    idx = Path(index_dir)
    sqlite_path = idx / "medical_rag.sqlite"
    t_qu0 = time.perf_counter()
    query_understanding = parse_query_understanding(q)
    query_understanding, llm_qu_debug = _llm_assisted_query_understanding(
        query=q,
        base_qu=query_understanding,
        llm_client=llm_client,
        provider=provider,
        model=model,
        timeout=timeout,
    )
    query_plan = understand_medical_query(
        q,
        llm_client=llm_client,
        model=model,
        timeout=timeout,
    )
    planner_intent_to_qu_intent = {
        "single_analyte_lookup": "single_analyte_lookup",
        "doc_scoped_abnormal_results": "doc_scoped_abnormal_results",
        "doc_scoped_summary": "doc_scoped_summary",
        "doc_pair_comparison": "doc_pair_comparison",
        "global_result_search": "cohort_search",
        "global_abnormal_search": "global_analyte_abnormal_search",
        "guarded_medical_interpretation": "doc_scoped_medical_interpretation_guarded",
        "open_grounded_medical_question": "unstructured",
        "reference_range_lookup": "reference_range_lookup",
        "unsupported_or_insufficient_context": "unstructured",
    }
    planned_intent = planner_intent_to_qu_intent.get(str(getattr(query_plan, "intent", "") or "").strip().lower())
    planned_docs = [str(d).strip() for d in list(getattr(query_plan, "requested_doc_ids", []) or []) if str(d).strip()]
    planned_analytes = [str(a).strip().lower() for a in list(getattr(query_plan, "requested_analytes", []) or []) if str(a).strip()]
    planner_condition_to_qu = {
        "above_reference_only": "above_reference",
        "below_reference_only": "below_reference",
        "out_of_reference": "out_of_reference",
        "within_reference": "within_reference",
        "any_result": None,
        "not_applicable": None,
    }
    planned_condition = planner_condition_to_qu.get(str(getattr(query_plan, "technical_condition", "") or "").strip().lower())
    planner_protected_intents = {
        "response_transform",
        "context_summary_render",
        "inventory_visualization_render",
        "visualization_recommendation",
        "source_followup",
        "comment_without_measured_value",
        "qualitative_comment_render",
        "patient_inventory",
        "patient_inventory_count",
    }
    if planned_intent and str(query_understanding.intent or "").strip().lower() not in planner_protected_intents:
        query_understanding = replace(
            query_understanding,
            intent=planned_intent,
            requested_doc_ids=planned_docs or list(query_understanding.requested_doc_ids or []),
            requested_analytes=planned_analytes or list(query_understanding.requested_analytes or []),
            technical_condition=planned_condition or query_understanding.technical_condition,
        )
    stage_times_ms["query_understanding_ms"] = round((time.perf_counter() - t_qu0) * 1000.0, 3)
    preferred_previous_transformable_pack = (
        previous_displayed_evidence_pack
        if isinstance(previous_displayed_evidence_pack, dict)
        and previous_displayed_evidence_pack
        and evidence_pack_is_transformable(previous_displayed_evidence_pack)
        else previous_structured_evidence_pack
    )
    state_for_resolution = {
        "last_intent": previous_context_intent,
        "last_data_context_type": previous_data_context_type,
        "last_data_context_intent": previous_data_context_intent,
        "last_doc_scope": {"doc_ids": list(previous_doc_scope or [])},
        "last_patient_inventory": previous_patient_inventory if previous_has_patient_inventory else None,
        "last_transformable_evidence_pack": preferred_previous_transformable_pack,
        "last_qualitative_evidence_pack": previous_qualitative_evidence_pack,
        "last_displayed_context": previous_displayed_context if isinstance(previous_displayed_context, dict) else None,
    }
    context_resolution = resolve_context_for_turn(q, query_understanding, state_for_resolution)
    deictic_resolution = resolve_deictic_request(q, query_understanding, state_for_resolution)
    resolution_arbitration = _arbitrate_resolution(
        query_understanding=query_understanding,
        context_resolution=context_resolution,
        deictic_resolution=deictic_resolution,
    )
    if bool(resolution_arbitration.get("conflict")):
        LOGGER.info(
            "routing_resolution_conflict request_id=%s decision=%s",
            request_id,
            json.dumps(
                {
                    "query": q,
                    "base_intent": resolution_arbitration.get("base_intent"),
                    "deictic_intent": resolution_arbitration.get("deictic_intent"),
                    "chosen": resolution_arbitration.get("chosen"),
                    "context_reason": resolution_arbitration.get("context_reason"),
                    "deictic_reason": resolution_arbitration.get("deictic_reason"),
                },
                ensure_ascii=False,
                default=str,
            ),
        )
    qn_deictic = norm_text(q)
    asks_table = any(token in qn_deictic for token in [" table", "table ", "tableau", "tabl", "une table", "dans une table"])
    asks_graph = any(token in qn_deictic for token in ["graphique", "chart", "courbe", "radar", "bar chart", "line graph", "visualise"])
    asks_context_summary = any(token in qn_deictic for token in ["resume", "résume", "resumer", "résumer", "synthese", "synthèse", "synthetise", "synthétise"]) and any(
        token in qn_deictic for token in ["commentaire", "ce commentaire", "cette valeur", "ce resultat", "ce résultat", "ces resultats", "ces résultats", "ce tableau", "ca", "ça", "ceci"]
    )
    is_deictic_render = bool(
        any(token in qn_deictic for token in [" ca", "ça", "ces donnees", "ces données", "affiche", "mets"])
        and (asks_table or asks_graph)
    )
    has_explicit_scope = bool(query_understanding.requested_doc_ids or query_understanding.requested_analytes or getattr(query_understanding, "requested_date_iso", None))
    prev_ctx_type = str(previous_data_context_type or "").strip().lower()
    deictic_no_context = str(deictic_resolution.get("intent") or "") == "deictic_no_context"

    if (is_deictic_render or asks_context_summary or deictic_no_context) and not has_explicit_scope:
        if prev_ctx_type == "patient_inventory":
            if asks_context_summary:
                query_understanding = replace(
                    query_understanding,
                    intent="context_summary_render",
                    is_small_talk=False,
                    is_response_transform=False,
                )
                context_resolution["should_skip_retrieval"] = True
                context_resolution["reason"] = "deictic_summary_reuse_inventory"
            else:
                inventory_view = "filterable_table" if asks_table else (getattr(query_understanding, "inventory_view_type", None) or "patient_cards")
                query_understanding = replace(
                    query_understanding,
                    intent="inventory_visualization_render",
                    inventory_view_type=inventory_view,
                    is_small_talk=False,
                    is_response_transform=False,
                )
        elif prev_ctx_type == "biological_numeric_results" and isinstance(preferred_previous_transformable_pack, dict) and preferred_previous_transformable_pack:
            if asks_context_summary:
                query_understanding = replace(
                    query_understanding,
                    intent="context_summary_render",
                    is_small_talk=False,
                    is_response_transform=False,
                )
            else:
                query_understanding = replace(
                    query_understanding,
                    intent="response_transform",
                    output_format="chart" if asks_graph else "table",
                    is_response_transform=True,
                    is_small_talk=False,
                )
            context_resolution["should_skip_retrieval"] = True
            context_resolution["reason"] = "deictic_summary_reuse_transformable" if asks_context_summary else "deictic_render_reuse_transformable"
        elif prev_ctx_type == "medical_qualitative_comment" and isinstance(previous_qualitative_evidence_pack, dict) and previous_qualitative_evidence_pack:
            if asks_context_summary:
                query_understanding = replace(
                    query_understanding,
                    intent="context_summary_render",
                    is_small_talk=False,
                    is_response_transform=False,
                )
            else:
                query_understanding = replace(
                    query_understanding,
                    intent="qualitative_comment_render",
                    qualitative_view_type="text_table",
                    is_small_talk=False,
                    is_response_transform=False,
                )
            context_resolution["should_skip_retrieval"] = True
            context_resolution["reason"] = "deictic_summary_reuse_qualitative" if asks_context_summary else "deictic_render_reuse_qualitative"
        else:
            if asks_context_summary:
                no_context_answer = "Je n’ai pas de contexte précédent à résumer. Demandez d’abord des résultats, un commentaire ou un inventaire."
            elif str(query_understanding.intent or "").strip().lower() == "inventory_visualization_render":
                no_context_answer = (
                    "Je n’ai pas d’inventaire patient récent à afficher sous cette forme. "
                    "Demandez d’abord la liste des patients."
                )
            elif deictic_no_context:
                no_context_answer = (
                    "Je n’ai pas de contexte précédent à reformater. "
                    "Je n’ai pas de contexte précédent exploitable à afficher sous cette forme. "
                    "Demandez d’abord un résultat, un commentaire ou un inventaire."
                )
            else:
                no_context_answer = (
                    "Je n’ai pas de contexte précédent exploitable à afficher sous cette forme. "
                    "Demandez d’abord les données à présenter."
                )
            no_context_mode = "context_summary_render" if asks_context_summary else "response_transform"
            return {
                "request_id": request_id,
                "query": q,
                "query_received": query_received,
                "query_used_for_retrieval": "",
                "query_used_for_prompt": q,
                "query_stored": q,
                "normalized_query": q,
                "mode": no_context_mode,
                "provider": provider,
                "model": model,
                "top_k": top_k,
                "max_display_results": int(max_display_results),
                "show_all_results": bool(show_all_results),
                "show_low_quality": bool(show_low_quality),
                "timeout": timeout,
                "generation_time_seconds": round(time.perf_counter() - started, 3),
                "answer": no_context_answer,
                "citations": [],
                "sources": [],
                "validation": validate_answer(
                    query=q,
                    answer_text=(
                        no_context_answer
                    ),
                    evidence_pack=[],
                    displayed_evidences=[],
                    source_citations=[],
                    generation_mode="deterministic_response_transform",
                    retrieval_status="insufficient_context",
                    query_received=query_received,
                    query_used_for_retrieval="",
                    query_used_for_prompt=q,
                    query_stored=q,
                    detected_analytes=[],
                    query_intents={"context_summary_render": True} if asks_context_summary else {"response_transform": True},
                    output_format_requested="list" if asks_context_summary else ("chart" if asks_graph else "table"),
                    answer_style_requested="standard",
                    requested_table_columns=[],
                    requested_technical_condition=None,
                    source_clickable_requested=bool(getattr(query_understanding, "source_clickable_requested", False)),
                    requested_value=None,
                    comparison_operator=None,
                    raw_format_phrase=getattr(query_understanding, "raw_format_phrase", None),
                    unsupported_presentation=False,
                    user_requested_visualization=False,
                    requested_chart_type=None,
                    visualization_payload=None,
                    chart_data_payload=None,
                ),
                "quality_report": None,
                "llm_error": None,
                "error_type": None,
                "generation_mode": "deterministic_context_summary_render" if asks_context_summary else "deterministic_response_transform",
                "detected_analytes": [],
                "query_understanding": _query_understanding_payload(query_understanding),
                "structured_evidence_pack": {},
                "evidence_pack": [],
                "displayed_evidences": [],
                "display": {
                    "selected_candidates_count": 0,
                    "low_quality_evidence_filtered_count": 0,
                    "hidden_result_count": 0,
                    "requested_multi_result_query": False,
                    "display_notes": [],
                },
                "retrieval": {
                    "answerability": {
                        "status": "not_required" if asks_context_summary else "insufficient_context",
                        "reason": (
                            "context_summary_no_context"
                            if asks_context_summary
                            else ("deictic_no_context_guard" if deictic_no_context else "deictic_render_no_context")
                        ),
                    },
                    "filters": {"doc_ids": [], "analytes": []},
                    "top_results": [],
                    "context_chunks": [],
                    "sources": [],
                    "retrieval_skipped": True,
                },
                "prompt": "",
                "debug": {
                    "request_id": request_id,
                    "generation_mode": "deterministic_response_transform",
                    "summary_no_context": bool(asks_context_summary),
                    "generation_writer": "professional_fallback",
                    "context_resolution": context_resolution,
                    "deictic_resolution": deictic_resolution,
                    "resolution_arbitration": resolution_arbitration,
                    "retrieval_skipped_due_to_no_transformable_context": True,
                },
                "visualization": None,
                "chart_data": None,
            }

    # Dedicated deictic resolver overrides (same-action / fiche / status follow-up).
    resolved_intent = str(deictic_resolution.get("intent") or "").strip().lower()
    if resolution_arbitration.get("chosen") == "deictic" and deictic_resolution.get("resolved") and resolved_intent not in {"", "deictic_no_context"}:
        if resolved_intent == "doc_scoped_results" and query_understanding.requested_analytes:
            effective_scope = deictic_resolution.get("effective_doc_scope") if isinstance(deictic_resolution.get("effective_doc_scope"), dict) else {}
            scoped_doc_ids = [str(d).strip() for d in (effective_scope.get("doc_ids") or []) if str(d).strip()]
            if scoped_doc_ids:
                query_understanding = replace(
                    query_understanding,
                    intent="doc_scoped_results",
                    requested_doc_ids=scoped_doc_ids,
                    is_small_talk=False,
                    is_response_transform=False,
                )
        elif resolved_intent == "qualitative_comment_render":
            view = str(deictic_resolution.get("render_type") or getattr(query_understanding, "qualitative_view_type", None) or "text_table")
            query_understanding = replace(
                query_understanding,
                intent="qualitative_comment_render",
                qualitative_view_type=view,
                is_small_talk=False,
                is_response_transform=False,
            )
            context_resolution["should_skip_retrieval"] = True
            context_resolution["reason"] = "deictic_qualitative_render"
        elif resolved_intent == "inventory_visualization_render":
            inv_view = str(deictic_resolution.get("render_type") or getattr(query_understanding, "inventory_view_type", None) or "filterable_table")
            query_understanding = replace(
                query_understanding,
                intent="inventory_visualization_render",
                inventory_view_type=inv_view,
                is_small_talk=False,
                is_response_transform=False,
            )
            context_resolution["should_skip_retrieval"] = True
            context_resolution["reason"] = "deictic_inventory_render"
        elif resolved_intent == "context_summary_render":
            query_understanding = replace(
                query_understanding,
                intent="context_summary_render",
                is_small_talk=False,
                is_response_transform=False,
            )
            context_resolution["should_skip_retrieval"] = True
            context_resolution["reason"] = "deictic_context_summary"
        elif resolved_intent == "response_transform":
            query_understanding = replace(
                query_understanding,
                intent="response_transform",
                is_small_talk=False,
                is_response_transform=True,
            )
            context_resolution["should_skip_retrieval"] = True
            context_resolution["reason"] = "deictic_response_transform"
    if (
        str(deictic_resolution.get("intent") or "").strip().lower() == "response_transform"
        and str(deictic_resolution.get("render_type") or "").strip().lower() == "status_check"
        and isinstance(previous_displayed_evidence_pack, dict)
    ):
        prev_rows = [r for r in list(previous_displayed_evidence_pack.get("evidences") or previous_displayed_evidence_pack.get("results") or []) if isinstance(r, dict)]
        if prev_rows:
            ev = prev_rows[0]
            analyte = str(ev.get("analyte") or ev.get("parameter") or "Résultat").strip()
            value = str(ev.get("current_value") or ev.get("value_raw") or ev.get("value_numeric") or "non disponible").strip()
            unit = str(ev.get("unit") or "").strip()
            reference = str(ev.get("reference") or ev.get("reference_range") or "non disponible").strip()
            status = str(ev.get("technical_status") or ev.get("status") or ev.get("interpretation_status") or "non interprétable").strip()
            if any(tok in norm_text(status) for tok in ["above_reference", "au dessus", "au-dessus"]):
                yn = "Oui"
            elif any(tok in norm_text(status) for tok in ["within_reference", "dans la reference", "dans la référence", "normal"]):
                yn = "Non"
            else:
                yn = "Impossible à déterminer"
            src = normalize_source_for_response(
                {
                    "label": ev.get("source_label") or ev.get("source"),
                    "source_pdf": ev.get("source_pdf"),
                    "doc_id": ev.get("doc_id"),
                    "page": ev.get("page") if ev.get("page") is not None else ev.get("page_number"),
                    "line": ev.get("line") if ev.get("line") is not None else ev.get("row"),
                    "viewer_url": ev.get("viewer_url"),
                    "source_url": ev.get("source_url"),
                }
            )
            source_label = str(src.get("label") or "source non disponible").strip()
            answer = (
                f"{yn}. {analyte} est {status} :\n"
                f"{analyte} = {value}{(' ' + unit) if unit else ''} ; référence : {reference}.\n"
                f"Source : {source_label}."
            )
            return {
                "request_id": request_id,
                "query": q,
                "query_received": query_received,
                "query_used_for_retrieval": "",
                "query_used_for_prompt": q,
                "query_stored": q,
                "normalized_query": q,
                "mode": "response_transform",
                "provider": provider,
                "model": model,
                "top_k": top_k,
                "max_display_results": int(max_display_results),
                "show_all_results": bool(show_all_results),
                "show_low_quality": bool(show_low_quality),
                "timeout": timeout,
                "generation_time_seconds": round(time.perf_counter() - started, 3),
                "answer": answer,
                "citations": [],
                "sources": [src],
                "validation": validate_answer(
                    query=q,
                    answer_text=answer,
                    evidence_pack=prev_rows,
                    displayed_evidences=prev_rows,
                    source_citations=[src],
                    generation_mode="deterministic_response_transform",
                    retrieval_status="not_required",
                    query_received=query_received,
                    query_used_for_retrieval="",
                    query_used_for_prompt=q,
                    query_stored=q,
                    detected_analytes=[],
                    query_intents={"response_transform": True},
                    output_format_requested="paragraph",
                    answer_style_requested="standard",
                    requested_table_columns=[],
                    requested_technical_condition=None,
                    source_clickable_requested=bool(getattr(query_understanding, "source_clickable_requested", False)),
                    requested_value=None,
                    comparison_operator=None,
                    raw_format_phrase=getattr(query_understanding, "raw_format_phrase", None),
                    unsupported_presentation=False,
                    user_requested_visualization=False,
                    requested_chart_type=None,
                    visualization_payload=None,
                    chart_data_payload=None,
                ),
                "quality_report": None,
                "llm_error": None,
                "error_type": None,
                "generation_mode": "deterministic_response_transform",
                "detected_analytes": [],
                "query_understanding": _query_understanding_payload(query_understanding),
                "structured_evidence_pack": previous_displayed_evidence_pack,
                "evidence_pack": prev_rows,
                "displayed_evidences": prev_rows,
                "retrieval": {"answerability": {"status": "not_required", "reason": "deictic_status_from_last_displayed_context"}, "retrieval_skipped": True},
                "prompt": "",
                "debug": {"request_id": request_id, "deictic_resolution": deictic_resolution, "context_resolution": context_resolution},
                "visualization": None,
                "chart_data": None,
            }

    # Guardrail: reference-range wording should not drift to global multi-analyte listing.
    # Force deterministic reference_range_lookup when user explicitly asks for norme/plage/référence.
    if _should_force_reference_range_lookup(qn, query_understanding):
        query_understanding = replace(
            query_understanding,
            intent="reference_range_lookup",
            is_small_talk=False,
            is_response_transform=False,
        )

    # Follow-up safeguard for reference ranges: preserve deterministic reference lookup across elliptical turns.
    prev_ctx = previous_displayed_context if isinstance(previous_displayed_context, dict) else {}
    prev_rr_ctx = prev_ctx.get("last_reference_range_context") if isinstance(prev_ctx.get("last_reference_range_context"), dict) else {}
    prev_rr_intent = str(prev_rr_ctx.get("intent") or prev_ctx.get("reference_intent") or previous_context_intent or "").strip().lower()
    curr_profile = (
        dict(getattr(query_understanding, "requested_reference_profile", None) or {})
        if isinstance(getattr(query_understanding, "requested_reference_profile", None), dict)
        else {}
    )
    prev_profile = {}
    if isinstance(prev_rr_ctx.get("requested_reference_profile"), dict):
        prev_profile = dict(prev_rr_ctx.get("requested_reference_profile") or {})
    elif isinstance(prev_ctx.get("reference_profile"), dict):
        prev_profile = dict(prev_ctx.get("reference_profile") or {})
    followup_like = _looks_like_reference_range_followup(q)
    has_curr_analyte = bool(list(getattr(query_understanding, "requested_analytes", None) or []))
    has_curr_profile = _has_any_reference_profile_slot(curr_profile)
    if prev_rr_intent == "reference_range_lookup" and (followup_like or has_curr_analyte or has_curr_profile):
        curr_analytes = [str(a).strip().lower() for a in list(getattr(query_understanding, "requested_analytes", None) or []) if str(a).strip()]
        if not curr_analytes:
            curr_analytes = [str(a).strip().lower() for a in (detect_exact_analytes(q) or []) if str(a).strip()]
        if not curr_analytes:
            prev_analyte = str(prev_rr_ctx.get("analyte") or prev_ctx.get("subject") or "").strip()
            if prev_analyte:
                curr_analytes = [str(a).strip().lower() for a in (detect_exact_analytes(prev_analyte) or []) if str(a).strip()]
        merged_profile = dict(prev_profile)
        for k, v in curr_profile.items():
            if v not in (None, ""):
                merged_profile[k] = v
        # If user explicitly changed analyte only, keep inherited profile for potential specific lookup/fallback messaging.
        final_profile = merged_profile if _has_any_reference_profile_slot(merged_profile) else curr_profile
        if curr_analytes:
            query_understanding = replace(
                query_understanding,
                intent="reference_range_lookup",
                requested_analytes=curr_analytes,
                requested_reference_profile=final_profile if _has_any_reference_profile_slot(final_profile) else None,
                is_small_talk=False,
                is_response_transform=False,
            )

    # Deterministic route normalization for production medical flows.
    selected_route = str(query_understanding.intent or "").strip().lower()
    route_reason = "intent_resolved_by_query_understanding"
    has_explicit_global_scope = any(
        t in qn
        for t in [
            "tous les rapports",
            "rapports disponibles",
            "rapports indexes",
            "rapports indexés",
            "sur l ensemble des rapports",
            "sur l’ensemble des rapports",
            "dans tous les rapports",
            "quels documents",
            "quels rapports",
            "documents",
        ]
    )
    has_short_bio_summary_shape = (
        bool(list(query_understanding.requested_doc_ids or []))
        and any(t in qn for t in ["resume", "résume", "synthese", "synthèse"])
        and any(t in qn for t in ["anomalies", "normaux", "quelques lignes", "partie"])
    )
    has_global_abnormal_shape = (
        has_explicit_global_scope
        and bool(list(query_understanding.requested_analytes or []))
        and any(
            t in qn
            for t in [
                "hors reference",
                "hors norme",
                "hors intervalle",
                "anormal",
                "anomal",
                "elevation",
                "élévation",
                "eleve",
                "élevé",
                "superieure",
                "supérieure",
                "diminuee",
                "diminuée",
                "basse",
                "inferieure",
            ]
        )
    )
    has_priority_shape = (
        bool(list(query_understanding.requested_doc_ids or []))
        and any(
            t in qn
            for t in [
                "hierarchise",
                "hiérarchise",
                "hierarchiser",
                "hiérarchiser",
                "importance technique",
                "ordre d importance",
                "ordre d’importance",
                "classement technique",
            ]
        )
    )
    if selected_route == "unstructured" and has_short_bio_summary_shape:
        selected_route = "doc_scoped_biological_summary"
        route_reason = "heuristic_doc_scoped_biological_summary"
    if selected_route == "unstructured" and has_global_abnormal_shape:
        selected_route = "global_analyte_abnormal_search"
        route_reason = "heuristic_global_analyte_abnormal_search"
    if selected_route == "unstructured" and has_priority_shape:
        selected_route = "doc_scoped_priority_anomalies"
        route_reason = "heuristic_doc_scoped_priority_anomalies"
    if selected_route == "global_analyte_abnormal_search":
        query_understanding = replace(
            query_understanding,
            intent="cohort_search",
            requires_global_search=True,
            requested_doc_ids=[],
            technical_condition=(query_understanding.technical_condition or "out_of_reference"),
        )
        selected_route = "global_analyte_abnormal_search"
        route_reason = "global_scope+analyte+abnormal_wording"
    elif selected_route == "doc_pair_comparison":
        query_understanding = replace(query_understanding, intent="multi_doc_comparison")
        if len(list(query_understanding.requested_doc_ids or [])) >= 3:
            selected_route = "multi_doc_comparison"
            route_reason = "compare+multiple_reports"
        else:
            selected_route = "doc_pair_comparison"
            route_reason = "compare+two_or_more_reports"
    elif selected_route == "multi_doc_comparison":
        query_understanding = replace(query_understanding, intent="multi_doc_comparison")
        selected_route = "multi_doc_comparison"
        route_reason = "compare+multiple_reports"
    elif selected_route == "doc_scoped_abnormal_results":
        query_understanding = replace(
            query_understanding,
            intent="doc_scoped_summary",
            technical_condition=(query_understanding.technical_condition or "out_of_reference"),
        )
        selected_route = "doc_scoped_abnormal_results"
        route_reason = "doc_scope+abnormal_summary_request"
    elif selected_route == "doc_scoped_summary" and _canonical_technical_condition(query_understanding.technical_condition) in {
        "out_of_reference",
        "above_reference",
        "below_reference",
    }:
        selected_route = "doc_scoped_abnormal_results"
        route_reason = "doc_scope_summary+technical_condition_abnormal"
    elif selected_route == "doc_scoped_biological_summary":
        query_understanding = replace(
            query_understanding,
            intent="doc_scoped_summary",
            technical_condition="any_result",
            output_format="paragraph",
        )
        selected_route = "doc_scoped_biological_summary"
        route_reason = "doc_scope+biological_summary_request"
    elif selected_route == "doc_scoped_priority_anomalies":
        query_understanding = replace(
            query_understanding,
            intent="doc_scoped_summary",
            technical_condition="any_result",
            output_format="table",
            requested_table_columns=[
                "priority_level",
                "analyte",
                "valeur_actuelle",
                "reference",
                "statut",
                "priority_reason",
            ],
        )
        selected_route = "doc_scoped_priority_anomalies"
        route_reason = "doc_scope+priority_anomalies_request"
    elif selected_route == "single_analyte_lookup":
        query_understanding = replace(query_understanding, intent="doc_scoped_results")
        selected_route = "single_analyte_lookup"
        route_reason = "single_analyte+doc_scope_lookup"
    elif selected_route == "doc_scoped_medical_interpretation_guarded":
        query_understanding = replace(
            query_understanding,
            intent="diagnostic_safety_question",
            answer_style="standard",
            output_format="paragraph",
        )
        selected_route = "doc_scoped_medical_interpretation_guarded"
        route_reason = "guarded_medical_interpretation_with_doc_scope"
    elif selected_route == "unstructured" and has_explicit_global_scope and any(
        t in qn for t in ["toxique urinaire", "toxiques urinaires", "pharmacotoxicologie", "drogues urinaires"]
    ):
        selected_route = "global_qualitative_toxicology_search"
        route_reason = "global_toxicology_qualitative_search"
    elif selected_route == "unstructured" and has_explicit_global_scope and any(
        t in qn for t in ["pathologie endocrinienne active", "endocrinienne active", "affirmer une pathologie endocrinienne"]
    ):
        selected_route = "open_grounded_medical_question"
        route_reason = "global_endocrine_open_grounded_guarded"

    if selected_route in {"global_analyte_abnormal_search", "global_qualitative_toxicology_search", "open_grounded_medical_question"} or has_explicit_global_scope:
        query_understanding = replace(query_understanding, requested_doc_ids=[])
        requested_doc_ids = []
    else:
        requested_doc_ids = resolve_followup_doc_scope(
            query=q,
            requested_analytes=list(query_understanding.requested_analytes or []),
            requested_doc_ids=list(query_understanding.requested_doc_ids),
            previous_doc_scope=previous_doc_scope,
        )
    if not requested_doc_ids:
        if not (selected_route in {"global_analyte_abnormal_search", "global_qualitative_toxicology_search", "open_grounded_medical_question"} or has_explicit_global_scope):
            effective_scope = context_resolution.get("effective_doc_scope") if isinstance(context_resolution, dict) else None
            if isinstance(effective_scope, dict):
                requested_doc_ids = [str(d).strip() for d in (effective_scope.get("doc_ids") or []) if str(d).strip()]
    requested_date_iso = str(getattr(query_understanding, "requested_date_iso", "") or "").strip()
    latest_report = bool(getattr(query_understanding, "latest_report", False))
    if not requested_doc_ids and requested_date_iso:
        requested_doc_ids = _resolve_doc_ids_by_date(sqlite_path, requested_date_iso)
    if not requested_doc_ids and latest_report:
        latest_doc = _resolve_latest_doc_id(sqlite_path)
        if latest_doc:
            requested_doc_ids = [latest_doc]
    if requested_doc_ids and not query_understanding.requested_doc_ids:
        query_understanding = replace(query_understanding, requested_doc_ids=requested_doc_ids)
    requested_doc_id = requested_doc_ids[0] if len(requested_doc_ids) == 1 else None
    missing_requested_doc_ids = _resolve_missing_requested_doc_ids(sqlite_path, requested_doc_ids)
    sensitive_or_treatment = _query_is_sensitive_or_treatment(q)
    if str(getattr(query_understanding, "requested_context_type", "") or "") == "medical_qualitative_comment":
        LOGGER.info(
            "qa_qualitative_pre current_query=%r detected_intent=%s requested_context_type=%s requested_analytes=%s requested_doc_ids=%s",
            q,
            str(query_understanding.intent or ""),
            str(getattr(query_understanding, "requested_context_type", "") or ""),
            list(query_understanding.requested_analytes or []),
            requested_doc_ids,
        )

    if selected_route in {"global_qualitative_toxicology_search", "open_grounded_medical_question"}:
        early_detected_analytes = list(query_understanding.requested_analytes or [])
        early_policy = _strict_policy_for_route(selected_route)
        answer = "Information insuffisante dans les données structurées disponibles pour répondre de façon fiable."
        if selected_route == "global_qualitative_toxicology_search":
            answer = (
                "Aucune recherche toxicologique urinaire exploitable n’a été retrouvée dans les documents indexés. "
                "Les éléments urinaires non toxicologiques (ex. cristaux) ne sont pas retenus pour cette demande."
            )
        generation_mode = "deterministic_no_evidence_response"
        validation = validate_answer(
            query=q,
            answer_text=answer,
            evidence_pack=[],
            displayed_evidences=[],
            source_citations=[],
            generation_mode=generation_mode,
            retrieval_status="insufficient_context",
            query_received=query_received,
            query_used_for_retrieval=query_used_for_retrieval,
            query_used_for_prompt=query_used_for_prompt,
            query_stored=q,
            detected_analytes=early_detected_analytes,
            query_intents=dict(getattr(query_understanding, "intents", {}) or {}),
            output_format_requested=query_understanding.output_format,
            answer_style_requested=query_understanding.answer_style,
            requested_table_columns=query_understanding.requested_table_columns,
            requested_technical_condition=query_understanding.technical_condition,
            source_clickable_requested=bool(query_understanding.source_clickable_requested),
            requested_value=query_understanding.requested_value,
            comparison_operator=query_understanding.comparison_operator,
            raw_format_phrase=getattr(query_understanding, "raw_format_phrase", None),
            unsupported_presentation=bool(getattr(query_understanding.presentation_intent, "unsupported_format", False)),
            user_requested_visualization=bool(getattr(query_understanding.presentation_intent, "user_requested_visualization", False)),
            requested_chart_type=getattr(query_understanding.presentation_intent, "chart_type", None),
            visualization_payload=None,
            chart_data_payload=None,
        )
        quality = _quality_report(answer=answer, validation=validation, source_clickable_requested=False, recent_style_history=style_history)
        final_safety_check_failed = False
        if str((validation or {}).get("validation_status") or "").strip().lower() == "fail":
            final_safety_check_failed = True
            final_answer = "Je ne peux pas fournir une réponse fiable à partir des données disponibles."
            generation_mode = "deterministic_safe_error_response"
            displayed_evidences = []
            evidence_pack = []
            source_citations = []
            citations = []
            validation = {
                "validation_status": "warning",
                "errors": [],
                "warnings": ["final_safety_check_failed"],
            }
            quality = _quality_report(
                answer=final_answer,
                validation=validation,
                source_clickable_requested=False,
                recent_style_history=style_history,
            )
            stage_times_ms["llm_writer_ms"] = 0.0
            stage_times_ms["repair_ms"] = 0.0

        final_safety_check_failed = False
        if str((validation or {}).get("validation_status") or "").strip().lower() == "fail":
            final_safety_check_failed = True
            final_answer = "Je ne peux pas fournir une réponse fiable à partir des données disponibles."
            generation_mode = "deterministic_safe_error_response"
            displayed_evidences = []
            evidence_pack = []
            source_citations = []
            citations = []
            validation = {
                "validation_status": "warning",
                "errors": [],
                "warnings": ["final_safety_check_failed"],
            }
            quality = _quality_report(
                answer=final_answer,
                validation=validation,
                source_clickable_requested=False,
                recent_style_history=style_history,
            )
            stage_times_ms["llm_writer_ms"] = 0.0
            stage_times_ms["repair_ms"] = 0.0

        elapsed = time.perf_counter() - started
        stage_times_ms["total_ms"] = round(elapsed * 1000.0, 3)
        stage_times_ms["llm_writer_ms"] = 0.0
        stage_times_ms["repair_ms"] = 0.0
        return {
            "request_id": request_id,
            "query": q,
            "query_received": query_received,
            "query_used_for_retrieval": query_used_for_retrieval,
            "query_used_for_prompt": query_used_for_prompt,
            "query_stored": q,
            "normalized_query": q,
            "mode": "sql_deterministic",
            "provider": provider,
            "model": model,
            "top_k": top_k,
            "max_display_results": int(max_display_results),
            "show_all_results": bool(show_all_results),
            "show_low_quality": bool(show_low_quality),
            "timeout": timeout,
            "generation_time_seconds": round(elapsed, 3),
            "answer": answer,
            "citations": [],
            "sources": [],
            "validation": validation,
            "quality_report": quality,
            "llm_error": None,
            "error_type": None,
            "generation_mode": generation_mode,
            "detected_analytes": early_detected_analytes,
            "query_understanding": _query_understanding_payload(query_understanding),
            "structured_evidence_pack": {"evidences": [], "results": [], "sources": [], "intent": query_understanding.intent},
            "evidence_pack": [],
            "displayed_evidences": [],
            "retrieval": {"answerability": {"status": "insufficient_context", "reason": selected_route}},
            "prompt": "",
            "debug": {
                "request_id": request_id,
                "selected_route": selected_route,
                "route_reason": route_reason,
                "selected_policy": early_policy.get("selected_policy"),
                "policy_level": early_policy.get("policy_level"),
                "facts_source": early_policy.get("facts_source"),
                "llm_allowed": bool(early_policy.get("llm_writer_allowed", False)),
                "llm_used": False,
                "timeout_ms": int(early_policy.get("timeout_s") or timeout) * 1000,
                "max_tokens": int(early_policy.get("max_tokens") or max_tokens),
                "validator_policy": str(early_policy.get("validator_policy") or "default"),
                "query_understanding": _query_understanding_payload(query_understanding),
                "stage_timings_ms": dict(stage_times_ms),
            },
            "visualization": None,
            "chart_data": None,
        }

    # 2. DETERMINISTIC ROUTING (Patient Inventory) - Bypasses all retrieval
    if query_understanding.intent in {"patient_inventory", "patient_inventory_count"}:
        elapsed = time.perf_counter() - started
        is_count = query_understanding.intent == "patient_inventory_count"
        if is_count:
            count = fetch_patient_count(sqlite_path)
            composed = compose_patient_inventory_count_answer(count)
        else:
            data = fetch_patient_inventory(sqlite_path)
            composed = compose_patient_inventory_answer(data)
        
        answer = composed["answer"]
        patients_data = composed.get("patients")
        validation = validate_answer(
            query=q,
            answer_text=answer,
            evidence_pack=[],
            displayed_evidences=[],
            source_citations=[],
            generation_mode=composed["mode"],
            retrieval_status="answerable",
            query_received=query_received,
            query_used_for_retrieval="",
            query_used_for_prompt=q,
            query_stored=q,
            detected_analytes=[],
            query_intents=query_understanding.intents or {},
            output_format_requested="table" if not is_count else "paragraph",
            answer_style_requested="standard",
            requested_table_columns=[],
            requested_technical_condition=None,
            source_clickable_requested=False,
            requested_value=None,
            comparison_operator=None,
            raw_format_phrase=getattr(query_understanding, "raw_format_phrase", None),
            unsupported_presentation=False,
            user_requested_visualization=False,
            requested_chart_type=None,
            visualization_payload=None,
            chart_data_payload=None,
            patients=patients_data,
        )
        
        return {
            "request_id": request_id,
            "query": q,
            "query_received": query_received,
            "query_used_for_retrieval": "",
            "query_used_for_prompt": q,
            "query_stored": q,
            "normalized_query": q,
            "mode": composed["mode"],
            "provider": provider,
            "model": model,
            "top_k": top_k,
            "generation_time_seconds": round(elapsed, 3),
            "answer": answer,
            "citations": [],
            "sources": [],
            "patients": patients_data,
            "validation": validation,
            "quality_report": _quality_report(
                answer=answer,
                validation=validation,
                source_clickable_requested=False,
                recent_style_history=style_history
            ),
            "llm_error": None,
            "error_type": None,
            "generation_mode": composed["mode"],
            "detected_analytes": [],
            "query_understanding": _query_understanding_payload(query_understanding),
            "structured_evidence_pack": {},
            "evidence_pack": [],
            "displayed_evidences": [],
            "display": {
                "selected_candidates_count": 0,
                "low_quality_evidence_filtered_count": 0,
                "hidden_result_count": 0,
                "requested_multi_result_query": False,
                "display_notes": [],
            },
            "retrieval": {
                "answerability": {"status": "answerable", "reason": "patient_inventory_metadata"},
                "filters": {"doc_ids": [], "analytes": []},
                "top_results": [],
                "context_chunks": [],
                "sources": [],
                "retrieval_skipped": True,
            },
            "prompt": "",
            "debug": {
                "request_id": request_id,
                "generation_mode": composed["mode"],
                "generation_writer": "deterministic_metadata_query",
                "intents": query_understanding.intents or {},
                "retrieval_skipped": True,
            },
        }

    qdrant_dir = idx / "qdrant"
    source_resolver = DocPdfResolver(index_dir=idx)

    retrieval_filters = RetrievalFilters()
    exact_analytes = list(query_understanding.requested_analytes)
    exact_analyte = exact_analytes[0] if len(exact_analytes) == 1 else None
    if exact_analyte is None and not exact_analytes:
        exact_analyte = detect_exact_analyte(q)
    is_above_reference_query = _is_above_reference_query(qn)
    is_normal_or_above = _is_normal_or_above_query(qn)
    is_below_reference_query = _is_below_reference_query(qn)
    is_global_above_query = _is_global_above_reference_query(qn, exact_analytes)
    intents = dict(query_understanding.intents or detect_query_intents(q, requested_doc_ids=requested_doc_ids, analytes=exact_analytes))
    compare_query = bool(query_understanding.requires_comparison or _is_compare_query(qn))
    compare_previous = bool(query_understanding.requires_previous_results or _is_previous_result_query(qn) or compare_query)
    visualization_or_transform_requested = _is_visualization_or_transform_request(q, query_understanding)

    if str(query_understanding.intent or "").strip().lower() == "visualization_recommendation":
        out = _build_visualization_recommendation_response(
            request_id=request_id,
            started=started,
            query=q,
            query_received=query_received,
            query_used_for_retrieval=query_used_for_retrieval,
            query_used_for_prompt=query_used_for_prompt,
            top_k=top_k,
            max_display_results=max_display_results,
            show_all_results=show_all_results,
            show_low_quality=show_low_quality,
            timeout=timeout,
            provider=provider,
            model=model,
            query_understanding=query_understanding,
            intents=intents,
            previous_context_intent=previous_context_intent,
            previous_data_context_intent=previous_data_context_intent,
            previous_data_context_type=previous_data_context_type,
            previous_has_patient_inventory=previous_has_patient_inventory,
            has_transformable_context=bool(preferred_previous_transformable_pack),
        )
        out.setdefault("debug", {})["context_resolution"] = context_resolution
        out.setdefault("debug", {})["deictic_resolution"] = deictic_resolution
        out.setdefault("debug", {})["resolution_arbitration"] = resolution_arbitration
        out.setdefault("debug", {})["retrieval_skipped_due_to_no_transformable_context"] = bool(
            context_resolution.get("should_skip_retrieval")
        )
        return out
    if str(query_understanding.intent or "").strip().lower() == "inventory_visualization_render":
        out = _build_inventory_visualization_render_response(
            request_id=request_id,
            started=started,
            query=q,
            query_received=query_received,
            query_used_for_retrieval=query_used_for_retrieval,
            query_used_for_prompt=query_used_for_prompt,
            top_k=top_k,
            max_display_results=max_display_results,
            show_all_results=show_all_results,
            show_low_quality=show_low_quality,
            timeout=timeout,
            provider=provider,
            model=model,
            query_understanding=query_understanding,
            intents=intents,
            previous_has_patient_inventory=previous_has_patient_inventory,
            previous_patient_inventory=previous_patient_inventory,
        )
        out.setdefault("debug", {})["context_resolution"] = context_resolution
        out.setdefault("debug", {})["deictic_resolution"] = deictic_resolution
        out.setdefault("debug", {})["resolution_arbitration"] = resolution_arbitration
        out.setdefault("debug", {})["retrieval_skipped_due_to_no_transformable_context"] = bool(
            context_resolution.get("should_skip_retrieval")
        )
        return out
    if str(query_understanding.intent or "").strip().lower() == "qualitative_comment_render":
        out = _build_qualitative_comment_render_response(
            request_id=request_id,
            started=started,
            query=q,
            query_received=query_received,
            query_used_for_retrieval=query_used_for_retrieval,
            query_used_for_prompt=query_used_for_prompt,
            top_k=top_k,
            max_display_results=max_display_results,
            show_all_results=show_all_results,
            show_low_quality=show_low_quality,
            timeout=timeout,
            provider=provider,
            model=model,
            query_understanding=query_understanding,
            intents=intents,
            previous_qualitative_evidence_pack=previous_qualitative_evidence_pack,
            previous_displayed_context=previous_displayed_context,
        )
        out.setdefault("debug", {})["context_resolution"] = context_resolution
        out.setdefault("debug", {})["deictic_resolution"] = deictic_resolution
        out.setdefault("debug", {})["resolution_arbitration"] = resolution_arbitration
        out.setdefault("debug", {})["retrieval_skipped_due_to_no_transformable_context"] = bool(
            context_resolution.get("should_skip_retrieval")
        )
        return out
    if str(query_understanding.intent or "").strip().lower() == "context_summary_render":
        out = _build_context_summary_render_response(
            request_id=request_id,
            started=started,
            query=q,
            query_received=query_received,
            query_used_for_retrieval=query_used_for_retrieval,
            query_used_for_prompt=query_used_for_prompt,
            top_k=top_k,
            max_display_results=max_display_results,
            show_all_results=show_all_results,
            show_low_quality=show_low_quality,
            timeout=timeout,
            provider=provider,
            model=model,
            query_understanding=query_understanding,
            intents=intents,
            previous_displayed_context=previous_displayed_context,
            previous_qualitative_evidence_pack=previous_qualitative_evidence_pack,
            previous_transformable_pack=preferred_previous_transformable_pack if isinstance(preferred_previous_transformable_pack, dict) else None,
            previous_patient_inventory=previous_patient_inventory,
            previous_data_context_type=previous_data_context_type,
            llm_client=llm_client,
        )
        out.setdefault("debug", {})["context_resolution"] = context_resolution
        out.setdefault("debug", {})["deictic_resolution"] = deictic_resolution
        out.setdefault("debug", {})["resolution_arbitration"] = resolution_arbitration
        out.setdefault("debug", {})["retrieval_skipped_due_to_no_transformable_context"] = True
        return out
    if str(query_understanding.intent or "").strip().lower() == "source_followup":
        # Always answer from last displayed context (no retrieval).
        ctx = previous_displayed_context if isinstance(previous_displayed_context, dict) else {}
        ctx_type = str(ctx.get("context_type") or "").strip().lower()
        ctx_subject = str(ctx.get("subject") or "ce commentaire").strip()
        source_candidates = list(ctx.get("sources") or [])
        if (not source_candidates) and isinstance(previous_qualitative_evidence_pack, dict):
            qevs = list(previous_qualitative_evidence_pack.get("evidences") or previous_qualitative_evidence_pack.get("results") or [])
            if qevs and isinstance(qevs[0], dict):
                q0 = qevs[0]
                source_candidates.append(
                    {
                        "label": str(q0.get("source") or q0.get("source_label") or "").strip(),
                        "source_pdf": str(q0.get("source_pdf") or "").strip(),
                        "doc_id": str(q0.get("doc_id") or "").strip(),
                        "page": q0.get("page") if q0.get("page") is not None else q0.get("page_number"),
                        "line": q0.get("line") if q0.get("line") is not None else q0.get("row"),
                        "viewer_url": str(q0.get("viewer_url") or "").strip() or None,
                        "source_url": str(q0.get("source_url") or "").strip() or None,
                    }
                )
        srcs = dedup_sources_for_qualitative(source_candidates)
        src0 = srcs[0] if srcs else {}
        source_label = str(src0.get("label") or "").strip()
        # Never expose internal source tags to users.
        if source_label.lower() in {"sqlite_deterministic", "sqlite", "deterministic"}:
            source_label = ""
        if not source_label:
            src_pdf = str(src0.get("source_pdf") or "").strip()
            src_page = src0.get("page")
            src_line = src0.get("line")
            if src_pdf:
                source_label = src_pdf
                if isinstance(src_page, int):
                    source_label += f" — page {src_page}"
                if isinstance(src_line, int):
                    source_label += f", ligne {src_line}"
        if not source_label:
            source_label = "source non disponible"
        viewer_url = str(src0.get("viewer_url") or "").strip() or None
        source_url = str(src0.get("source_url") or src0.get("url") or "").strip() or None
        source_markdown, has_click = format_clickable_source_markdown(source_label, viewer_url, source_url)
        source_txt = source_markdown if has_click else source_label
        if not has_click:
            source_txt = f"{source_label} (source non cliquable disponible uniquement en texte)"
        if ctx_type == "medical_qualitative_comment":
            subj = str(ctx_subject or "ce sujet").strip()
            if _is_generic_subject(subj):
                subj = "ce commentaire"
            subj_phrase = f"la {subj.lower()}" if subj and not subj.lower().startswith(("le ", "la ", "l'")) else subj.lower()
            if "source exacte" in norm_text(q or ""):
                answer = f"La source exacte du commentaire sur {subj_phrase} est {source_txt}."
            else:
                answer = f"Ce commentaire sur {subj_phrase} provient de {source_txt}."
        else:
            answer = f"La source du dernier élément affiché est {source_txt}."
        validation = validate_answer(
            query=q,
            answer_text=answer,
            evidence_pack=[],
            displayed_evidences=[],
            source_citations=srcs,
            generation_mode="deterministic_source_followup",
            retrieval_status="not_required",
            query_received=query_received,
            query_used_for_retrieval="",
            query_used_for_prompt=q,
            query_stored=q,
            detected_analytes=[],
            query_intents=intents,
            output_format_requested="paragraph",
            answer_style_requested="standard",
            requested_table_columns=[],
            requested_technical_condition=None,
            source_clickable_requested=bool(getattr(query_understanding, "source_clickable_requested", False)),
            requested_value=None,
            comparison_operator=None,
            visualization_payload=None,
            chart_data_payload=None,
        )
        return {
            "request_id": request_id,
            "query": q,
            "query_received": query_received,
            "query_used_for_retrieval": "",
            "query_used_for_prompt": q,
            "query_stored": q,
            "normalized_query": q,
            "mode": "source_followup",
            "provider": provider,
            "model": model,
            "top_k": top_k,
            "max_display_results": int(max_display_results),
            "show_all_results": bool(show_all_results),
            "show_low_quality": bool(show_low_quality),
            "timeout": timeout,
            "generation_time_seconds": round(time.perf_counter() - started, 3),
            "answer": answer,
            "citations": [],
            "sources": srcs,
            "validation": validation,
            "quality_report": _quality_report(answer=answer, validation=validation, source_clickable_requested=False, recent_style_history=[]),
            "llm_error": None,
            "error_type": None,
            "generation_mode": "deterministic_source_followup",
            "detected_analytes": [],
            "query_understanding": _query_understanding_payload(query_understanding),
            "structured_evidence_pack": {},
            "evidence_pack": [],
            "displayed_evidences": [],
            "retrieval": {"answerability": {"status": "not_required", "reason": "source_followup_no_retrieval"}},
            "debug": {
                "request_id": request_id,
                "context_resolution": context_resolution,
                "deictic_resolution": deictic_resolution,
                "resolution_arbitration": resolution_arbitration,
            },
            "visualization": None,
            "chart_data": None,
        }
    if str(query_understanding.intent or "").strip().lower() == "reference_range_lookup":
        out = _build_reference_range_lookup_response(
            request_id=request_id,
            started=started,
            query=q,
            query_received=query_received,
            query_used_for_retrieval=query_used_for_retrieval,
            query_used_for_prompt=query_used_for_prompt,
            top_k=top_k,
            max_display_results=max_display_results,
            show_all_results=show_all_results,
            show_low_quality=show_low_quality,
            timeout=timeout,
            provider=provider,
            model=model,
            query_understanding=query_understanding,
            intents=intents,
            sqlite_path=sqlite_path,
            requested_doc_ids=requested_doc_ids,
            previous_displayed_context=previous_displayed_context,
        )
        out.setdefault("debug", {})["context_resolution"] = context_resolution
        out.setdefault("debug", {})["deictic_resolution"] = deictic_resolution
        out.setdefault("debug", {})["resolution_arbitration"] = resolution_arbitration
        return out

    # Strict guard: after non-transformable contexts (e.g. inventory), do not run retrieval
    # for deictic visualization/transform requests when no transformable context exists.
    if _is_general_conversation_fastpath_candidate(
        query_norm=qn,
        requested_doc_ids=list(query_understanding.requested_doc_ids or []),
        requested_analytes=list(query_understanding.requested_analytes or []),
        requested_value=query_understanding.requested_value,
        comparison_operator=query_understanding.comparison_operator,
    ):
        answer = render_general_conversation_response(
            detect_general_conversation(q) or str(query_understanding.intent or "") or "fallback"
        )
        stage_times_ms["retrieval_ms"] = 0.0
        stage_times_ms["llm_writer_ms"] = 0.0
        stage_times_ms["repair_ms"] = 0.0
        stage_times_ms["validation_ms"] = 0.0
        stage_times_ms["fallback_ms"] = 0.0
        elapsed = time.perf_counter() - started
        stage_times_ms["total_ms"] = round(elapsed * 1000.0, 3)
        validation = validate_answer(
            query=q,
            answer_text=answer,
            evidence_pack=[],
            displayed_evidences=[],
            source_citations=[],
            generation_mode="deterministic_general_conversation",
            retrieval_status="not_required",
            query_received=query_received,
            query_used_for_retrieval="",
            query_used_for_prompt=q,
            query_stored=q,
            detected_analytes=[],
            query_intents={**intents, "general_conversation": True},
            output_format_requested="paragraph",
            answer_style_requested="standard",
            requested_table_columns=[],
            requested_technical_condition=None,
            source_clickable_requested=False,
            requested_value=None,
            comparison_operator=None,
            raw_format_phrase=getattr(query_understanding, "raw_format_phrase", None),
            unsupported_presentation=False,
            user_requested_visualization=False,
            requested_chart_type=None,
            visualization_payload=None,
            chart_data_payload=None,
        )
        return {
            "request_id": request_id,
            "query": q,
            "query_received": query_received,
            "query_used_for_retrieval": "",
            "query_used_for_prompt": q,
            "query_stored": q,
            "normalized_query": q,
            "mode": "general_conversation",
            "provider": provider,
            "model": model,
            "top_k": top_k,
            "max_display_results": int(max_display_results),
            "show_all_results": bool(show_all_results),
            "show_low_quality": bool(show_low_quality),
            "timeout": timeout,
            "generation_time_seconds": round(elapsed, 3),
            "answer": answer,
            "citations": [],
            "sources": [],
            "validation": validation,
            "quality_report": _quality_report(answer=answer, validation=validation, source_clickable_requested=False, recent_style_history=style_history),
            "llm_error": None,
            "error_type": None,
            "generation_mode": "deterministic_general_conversation",
            "detected_analytes": [],
            "query_understanding": _query_understanding_payload(query_understanding),
            "structured_evidence_pack": {},
            "evidence_pack": [],
            "displayed_evidences": [],
            "retrieval": {"answerability": {"status": "not_required", "reason": "general_conversation_fast_path"}, "retrieval_skipped": True},
            "prompt": "",
            "debug": {
                "request_id": request_id,
                "selected_route": "general_conversation",
                "route_reason": "deterministic_general_conversation_fast_path",
                "generation_mode": "deterministic_general_conversation",
                "generation_writer": "deterministic_general_conversation",
                "stage_timings_ms": dict(stage_times_ms),
            },
            "visualization": None,
            "chart_data": None,
        }

    if (
        visualization_or_transform_requested
        and not preferred_previous_transformable_pack
        and str(previous_context_intent or "").strip().lower() in {"patient_inventory", "patient_inventory_count"}
    ):
        out = _build_no_transformable_context_response(
            request_id=request_id,
            started=started,
            query=q,
            query_received=query_received,
            query_used_for_retrieval=query_used_for_retrieval,
            query_used_for_prompt=query_used_for_prompt,
            top_k=top_k,
            max_display_results=max_display_results,
            show_all_results=show_all_results,
            show_low_quality=show_low_quality,
            timeout=timeout,
            provider=provider,
            model=model,
            query_understanding=query_understanding,
            intents=intents,
            exact_analytes=exact_analytes,
            requested_doc_ids=requested_doc_ids,
            qn=qn,
            previous_context_intent=previous_context_intent,
        )
        out.setdefault("debug", {})["context_resolution"] = context_resolution
        out.setdefault("debug", {})["retrieval_skipped_due_to_no_transformable_context"] = True
        return out

    # Follow-up transform priority: if user asks to reformat "this result" without new doc/analyte,
    # reuse previous evidence pack instead of small-talk/retrieval routing.
    if preferred_previous_transformable_pack and _looks_like_transform_followup(q, query_understanding):
        query_understanding = replace(
            query_understanding,
            intent="response_transform",
            is_response_transform=True,
            is_small_talk=False,
            response_strategy="transform_previous_response",
            response_strategy_reason="Follow-up de reformattage détecté sur le contexte précédent.",
        )
        intents["response_transform"] = True
        intents["small_talk"] = False
        intents["general_conversation"] = False

    general_intent = str(query_understanding.intent or "").strip().lower()
    has_medical_payload = bool(requested_doc_ids or exact_analytes or query_understanding.requested_value or query_understanding.comparison_operator)
    if general_intent in GENERAL_CONVERSATION_INTENTS and not has_medical_payload and is_pure_general_conversation(q):
        qn_local = norm_text(q)
        general_answer = render_general_conversation_response(
            detect_general_conversation(qn_local) or general_intent or "fallback"
        )
        general_mode = "deterministic_general_conversation"
        validation = validate_answer(
            query=q,
            answer_text=general_answer,
            evidence_pack=[],
            displayed_evidences=[],
            source_citations=[],
            generation_mode=general_mode,
            retrieval_status="not_required",
            query_received=query_received,
            query_used_for_retrieval="",
            query_used_for_prompt=q,
            query_stored=q,
            detected_analytes=[],
            query_intents={
                **intents,
                "general_conversation": True,
                "small_talk": general_intent == "small_talk",
                "identity_question": general_intent == "identity_question",
                "capability_question": general_intent == "capability_question",
                "help_question": general_intent == "help_question",
            },
            output_format_requested="paragraph",
            answer_style_requested="standard",
            requested_table_columns=[],
            requested_technical_condition=None,
            source_clickable_requested=False,
            requested_value=None,
            comparison_operator=None,
            raw_format_phrase=getattr(query_understanding, "raw_format_phrase", None),
            unsupported_presentation=bool(getattr(query_understanding.presentation_intent, "unsupported_format", False)),
            user_requested_visualization=bool(getattr(query_understanding.presentation_intent, "user_requested_visualization", False)),
            requested_chart_type=getattr(query_understanding.presentation_intent, "chart_type", None),
            visualization_payload=_preview_visualization_payload(query_understanding, [])[0],
            chart_data_payload=_preview_visualization_payload(query_understanding, [])[1],
        )
        intro, conclusion = _extract_intro_conclusion(general_answer)
        quality = _quality_report(
            answer=general_answer,
            validation=validation,
            source_clickable_requested=False,
            recent_style_history=style_history,
        )
        stage_times_ms["retrieval_ms"] = 0.0
        stage_times_ms["llm_writer_ms"] = 0.0
        stage_times_ms["repair_ms"] = 0.0
        stage_times_ms["validation_ms"] = 0.0
        stage_times_ms["fallback_ms"] = 0.0
        elapsed = time.perf_counter() - started
        stage_times_ms["total_ms"] = round(elapsed * 1000.0, 3)
        result = {
            "request_id": request_id,
            "query": q,
            "query_received": query_received,
            "query_used_for_retrieval": "",
            "query_used_for_prompt": q,
            "query_stored": q,
            "normalized_query": q,
            "mode": "general_conversation",
            "provider": provider,
            "model": model,
            "top_k": top_k,
            "max_display_results": int(max_display_results),
            "show_all_results": bool(show_all_results),
            "show_low_quality": bool(show_low_quality),
            "timeout": timeout,
            "generation_time_seconds": round(elapsed, 3),
            "answer": general_answer,
            "citations": [],
            "sources": [],
            "validation": validation,
            "quality_report": quality,
            "llm_error": None,
            "error_type": None,
            "generation_mode": general_mode,
            "detected_analytes": [],
            "query_understanding": _query_understanding_payload(query_understanding),
            "structured_evidence_pack": {},
            "evidence_pack": [],
            "displayed_evidences": [],
            "display": {
                "selected_candidates_count": 0,
                "low_quality_evidence_filtered_count": 0,
                "hidden_result_count": 0,
                "requested_multi_result_query": False,
                "display_notes": [],
            },
            "retrieval": {
                "answerability": {"status": "not_required", "reason": f"{general_intent}_no_retrieval"},
                "filters": {"doc_ids": [], "analytes": []},
                "top_results": [],
                "context_chunks": [],
                "sources": [],
                "retrieval_skipped": True,
            },
            "prompt": "",
            "style_memory_entry": {
                "intro_text": intro,
                "conclusion_text": conclusion,
                "intent": general_intent,
                "output_format": "paragraph",
                "answer_text": general_answer,
            },
            "debug": {
                "request_id": request_id,
                "selected_route": "general_conversation",
                "route_reason": "general_conversation_fast_path",
                "generation_mode": general_mode,
                "generation_writer": "deterministic_general_conversation",
                "intents": intents,
                "stage_timings_ms": dict(stage_times_ms),
            },
        }
        return _inject_visualization_payload(
            result,
            query_understanding=query_understanding,
            displayed_evidences=[],
        )

    if query_understanding.intent == "response_transform":
        if not preferred_previous_transformable_pack:
            return _build_no_transformable_context_response(
                request_id=request_id,
                started=started,
                query=q,
                query_received=query_received,
                query_used_for_retrieval=query_used_for_retrieval,
                query_used_for_prompt=query_used_for_prompt,
                top_k=top_k,
                max_display_results=max_display_results,
                show_all_results=show_all_results,
                show_low_quality=show_low_quality,
                timeout=timeout,
                provider=provider,
                model=model,
                query_understanding=query_understanding,
                intents=intents,
                exact_analytes=exact_analytes,
                requested_doc_ids=requested_doc_ids,
                qn=qn,
            previous_context_intent=previous_context_intent,
            )

        transformed_pack = _build_response_transform_pack(
            query=q,
            query_understanding=query_understanding,
            previous_pack=preferred_previous_transformable_pack,
        )
        transformed_qu = replace(
            query_understanding,
            intent="response_transform",
            output_format=str(transformed_pack.get("output_format") or query_understanding.output_format),
            requested_table_columns=list(
                transformed_pack.get("requested_table_columns") or query_understanding.requested_table_columns or []
            ),
        )
        displayed_evidences = [
            {
                "doc_id": ev.get("doc_id"),
                "chunk_id": ev.get("chunk_id"),
                "page_number": ev.get("page"),
                "row_index": ev.get("row"),
                "source_pdf": ev.get("source_pdf"),
                "analyte_norm": ev.get("analyte_norm"),
                "analyte": ev.get("analyte"),
                "value_raw": ev.get("current_value") or ev.get("value_raw") or ev.get("value"),
                "value_numeric": ev.get("value_numeric"),
                "reference_range": ev.get("reference") or ev.get("reference_range"),
                "unit": ev.get("unit"),
                "previous_result": ev.get("previous_result"),
                "patient_token": ev.get("patient_token"),
                "interpretation_status": ev.get("technical_status_code") or ev.get("interpretation_status"),
                "source": ev.get("source"),
                "source_kind": "sqlite_deterministic",
                "chunk_type": "lab_result",
            }
            for ev in (transformed_pack.get("evidences") or [])
        ]
        source_citations = build_source_citations(displayed_evidences, resolver=source_resolver)
        transformed_pack = _attach_source_fields_to_structured_pack(transformed_pack, source_citations)
        transformed_pack["recent_style_history"] = style_history[-20:]
        transformed_pack = _attach_visualization_facts_to_evidence_pack(
            query_understanding=transformed_qu,
            evidence_pack=transformed_pack,
            displayed_evidences=displayed_evidences,
        )
        transformed_qu = _with_resolved_strategy(transformed_qu, transformed_pack)
        transformed_compose_mode = (
            "fallback"
            if _force_deterministic_mode_for_summary_anomalies(transformed_qu, qn)
            else _hybrid_writer_mode(transformed_qu)
        )
        output_format = str(transformed_pack.get("output_format") or query_understanding.output_format or "list").lower()
        composed = compose_professional_answer(
            user_question=q,
            query_understanding=transformed_qu,
            evidence_pack=transformed_pack,
            mode=transformed_compose_mode,
            source_citations=source_citations,
            llm_client=llm_client,
            provider=provider,
            model=model,
            temperature=temperature,
            num_ctx=num_ctx,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        composed_data = composed
        answer = str(composed.get("answer") or "")
        generation_mode = (
            "deterministic_response_transform_json"
            if output_format == "json"
            else "deterministic_response_transform_professional"
        )
        validation = validate_answer(
            query=q,
            answer_text=answer,
            evidence_pack=displayed_evidences,
            displayed_evidences=displayed_evidences,
            source_citations=source_citations,
            generation_mode=generation_mode,
            retrieval_status="answerable" if displayed_evidences else "insufficient_context",
            query_received=query_received,
            query_used_for_retrieval=query_used_for_retrieval,
            query_used_for_prompt=query_used_for_prompt,
            query_stored=q,
            detected_analytes=exact_analytes,
            query_intents=intents,
            output_format_requested=output_format,
            answer_style_requested=transformed_qu.answer_style,
            requested_table_columns=transformed_pack.get("requested_table_columns")
            or transformed_qu.requested_table_columns,
            requested_technical_condition=transformed_qu.technical_condition,
            source_clickable_requested=bool(transformed_qu.source_clickable_requested),
            requested_value=transformed_qu.requested_value,
            comparison_operator=transformed_qu.comparison_operator,
            raw_format_phrase=getattr(transformed_qu, "raw_format_phrase", None),
            unsupported_presentation=bool(getattr(transformed_qu.presentation_intent, "unsupported_format", False)),
            user_requested_visualization=bool(getattr(transformed_qu.presentation_intent, "user_requested_visualization", False)),
            requested_chart_type=getattr(transformed_qu.presentation_intent, "chart_type", None),
            visualization_payload=_preview_visualization_payload(transformed_qu, displayed_evidences)[0],
            chart_data_payload=_preview_visualization_payload(transformed_qu, displayed_evidences)[1],
            transformable_context_available=True,
            previous_intent=str(previous_context_intent or ""),
        )
        quality = _quality_report(
            answer=answer,
            validation=validation,
            source_clickable_requested=bool(transformed_qu.source_clickable_requested),
            recent_style_history=style_history,
        )
        intro_text, conclusion_text = _extract_intro_conclusion(answer)
        elapsed = time.perf_counter() - started
        return _inject_visualization_payload(
            {
            "request_id": request_id,
            "query": q,
            "query_received": query_received,
            "query_used_for_retrieval": query_used_for_retrieval,
            "query_used_for_prompt": query_used_for_prompt,
            "query_stored": q,
            "normalized_query": q,
            "mode": "response_transform",
            "provider": provider,
            "model": model,
            "top_k": top_k,
            "max_display_results": int(max_display_results),
            "show_all_results": bool(show_all_results),
            "show_low_quality": bool(show_low_quality),
            "timeout": timeout,
            "generation_time_seconds": round(elapsed, 3),
            "answer": answer,
            "citations": [],
            "sources": source_citations_for_response,
            "validation": validation,
            "quality_report": quality,
            "llm_error": None,
            "error_type": None,
            "generation_mode": generation_mode,
            "detected_analytes": exact_analytes,
            "query_understanding": _query_understanding_payload(query_understanding),
            "structured_evidence_pack": transformed_pack,
            "style_memory_entry": {
                "intro_text": intro_text,
                "conclusion_text": conclusion_text,
                "intent": transformed_qu.intent,
                "output_format": transformed_qu.output_format,
                "answer_text": answer,
            },
            "evidence_pack": displayed_evidences,
            "displayed_evidences": displayed_evidences,
            "display": {
                "selected_candidates_count": len(displayed_evidences),
                "low_quality_evidence_filtered_count": 0,
                "hidden_result_count": 0,
                "requested_multi_result_query": _query_requests_multiple_results(qn),
                "display_notes": [],
            },
            "retrieval": {
                "answerability": {"status": "answerable" if displayed_evidences else "insufficient_context", "reason": "response_transform"},
                "filters": {"doc_ids": query_understanding.requested_doc_ids, "analytes": query_understanding.requested_analytes},
                "top_results": [],
                "context_chunks": [],
                "sources": [],
            },
            "prompt": "",
            "debug": {
                "request_id": request_id,
                "query_received": query_received,
                "query_used_for_retrieval": query_used_for_retrieval,
                "query_used_for_prompt": query_used_for_prompt,
                "detected_analytes": exact_analytes,
                "requested_doc_ids": requested_doc_ids,
                "generation_mode": generation_mode,
                "generation_writer": "llm_writer" if str(generation_mode).startswith("llm_") or generation_mode == "hybrid_structured_llm_writer" else "professional_fallback",
                "intents": intents,
            },
            "exact_analyte_coverage": {
                "detected_exact_analyte": exact_analyte,
                "expected_exact_analyte_count": len(exact_analytes),
                "retrieved_exact_analyte_count": len(displayed_evidences),
                "displayed_exact_analyte_count": len(displayed_evidences),
            },
            },
            query_understanding=transformed_qu,
            displayed_evidences=displayed_evidences,
        )

    if _is_structured_question_with_fast_path(intents, requested_doc_ids, exact_analytes) and (
        requested_doc_ids
        or query_understanding.intent
        in {
            "global_patient_lookup",
            "cohort_search",
            "global_analyte_abnormal_search",
            "multi_doc_comparison",
            "comment_without_measured_value",
        }
    ):
        t_route0 = time.perf_counter()
        structured_pack = build_structured_evidence_pack(
            query=q,
            query_understanding=query_understanding,
            sqlite_path=sqlite_path,
        )
        stage_times_ms["routing_ms"] = round((time.perf_counter() - t_route0) * 1000.0, 3)
        stage_times_ms["evidence_build_ms"] = stage_times_ms["routing_ms"]
        if str(query_understanding.intent or "").strip().lower() == "comment_without_measured_value":
            resolved_scope = [
                str(d).strip()
                for d in (structured_pack.get("requested_doc_ids") or [])
                if str(d).strip()
            ]
            if resolved_scope:
                requested_doc_ids = resolved_scope
        structured_rows = list(structured_pack.get("rows") or [])
        structured_evidences = list(structured_pack.get("evidences") or [])
        if selected_route == "doc_scoped_priority_anomalies" and structured_evidences:
            evidence_pack = list(structured_evidences)
        elif str(query_understanding.intent or "").strip().lower() == "comment_without_measured_value":
            evidence_pack = list(structured_pack.get("evidences") or [])
        else:
            evidence_pack = _rows_to_evidence(structured_rows)
        if requested_doc_ids:
            allowed_docs = {d.lower() for d in requested_doc_ids}
            evidence_pack = [ev for ev in evidence_pack if str(ev.get("doc_id") or "").strip().lower() in allowed_docs]
        if selected_route == "doc_scoped_priority_anomalies":
            evidence_pack = _apply_priority_scoring(list(evidence_pack))
            structured_pack["evidences"] = list(evidence_pack)
            structured_pack["results"] = list(evidence_pack)
            structured_pack["rows"] = list(structured_pack.get("rows") or [])
        displayed_evidences = list(evidence_pack)
        citations = build_citations(displayed_evidences)
        source_citations = build_source_citations(displayed_evidences, resolver=source_resolver)
        if str(query_understanding.intent or "").strip().lower() == "comment_without_measured_value":
            source_citations = dedup_sources_for_qualitative(source_citations)
            # For list mode, keep source list in response payload.
            # For single-comment mode, source is rendered inline to avoid duplicates.
            if bool(structured_pack.get("comment_list_mode")):
                source_citations_for_response = source_citations
            else:
                source_citations_for_response = []
        else:
            source_citations_for_response = source_citations
        if selected_route == "doc_scoped_priority_anomalies":
            precise = [s for s in source_citations if isinstance(s, dict) and s.get("page") is not None and s.get("row") is not None]
            if precise:
                source_citations_for_response = precise
        structured_pack = _attach_source_fields_to_structured_pack(structured_pack, source_citations)
        structured_pack["recent_style_history"] = style_history[-20:]
        structured_pack = _attach_visualization_facts_to_evidence_pack(
            query_understanding=query_understanding,
            evidence_pack=structured_pack,
            displayed_evidences=displayed_evidences,
        )
        query_understanding = _with_resolved_strategy(query_understanding, structured_pack)
        found_requested_analytes: list[str] = []
        found_requested_analyte_norms: list[str] = []
        missing_requested_analytes: list[str] = []
        selected_policy = _strict_policy_for_route(selected_route)
        force_llm_writer = str(os.getenv("MEDICAL_RAG_FORCE_LLM_WRITER", "1")).strip().lower() in {"1", "true", "yes", "on"}
        safety_intent_norm = str(getattr(query_understanding, "safety_intent", "") or "").strip().lower()
        llm_cfg = _level2_llm_runtime_config(
            selected_route=selected_route,
            selected_policy=selected_policy,
            force_llm_writer=force_llm_writer,
            safety_intent_norm=safety_intent_norm,
            displayed_evidences=displayed_evidences,
            default_timeout=timeout,
            default_max_tokens=max_tokens,
        )
        llm_writer_allowed = bool(selected_policy.get("llm_writer_allowed", False))
        llm_writer_used = False
        policy_timeout_s = int(llm_cfg.get("timeout_s") or timeout)
        policy_max_tokens = int(llm_cfg.get("max_tokens") or max_tokens)
        validator_policy = str(selected_policy.get("validator_policy") or "default")
        if selected_route == "doc_scoped_biological_summary" and not bool(llm_cfg.get("use_llm", False)):
            no_diag = str(getattr(query_understanding, "safety_intent", "") or "").strip().lower() == "no_diagnosis_constraint"
            final_answer = _build_doc_scoped_biological_summary_answer(
                displayed_evidences,
                max_lines=getattr(query_understanding, "requested_summary_points", None),
                no_diagnosis=no_diag,
            )
            generation_mode = "deterministic_doc_scoped_biological_summary"
            writer_error = None
            llm_prompt_preview = ""
            llm_candidate_answer = final_answer
            retry_used = False
            stage_times_ms["llm_writer_ms"] = 0.0
            stage_times_ms["repair_ms"] = 0.0
            validation = validate_answer(
                query=q,
                answer_text=final_answer,
                evidence_pack=evidence_pack,
                displayed_evidences=displayed_evidences,
                source_citations=source_citations,
                exact_analyte=exact_analyte,
                llm_error=None,
                generation_mode=generation_mode,
                retrieval_status="answerable" if displayed_evidences else "insufficient_context",
                show_low_quality=show_low_quality,
                max_display_results=max_display_results,
                show_all_results=show_all_results,
                query_received=query_received,
                query_used_for_retrieval=query_used_for_retrieval,
                query_used_for_prompt=query_used_for_prompt,
                query_stored=q,
                detected_analytes=exact_analytes,
                requested_doc_id=requested_doc_ids[0] if len(requested_doc_ids) == 1 else None,
                requested_doc_ids=requested_doc_ids,
                missing_requested_doc_ids=missing_requested_doc_ids,
                requested_analytes=exact_analytes,
                found_requested_analytes=found_requested_analytes,
                found_requested_analyte_norms=found_requested_analyte_norms,
                missing_requested_analytes=missing_requested_analytes,
                current_vs_previous_requested=query_understanding.requires_previous_results,
                diagnostic_safety_intent=False,
                query_intents=intents,
                output_format_requested=query_understanding.output_format,
                answer_style_requested=query_understanding.answer_style,
                requested_table_columns=query_understanding.requested_table_columns,
                requested_technical_condition=query_understanding.technical_condition,
                source_clickable_requested=bool(query_understanding.source_clickable_requested),
                requested_value=query_understanding.requested_value,
                comparison_operator=query_understanding.comparison_operator,
                raw_format_phrase=getattr(query_understanding, "raw_format_phrase", None),
                unsupported_presentation=bool(getattr(query_understanding.presentation_intent, "unsupported_format", False)),
                user_requested_visualization=bool(getattr(query_understanding.presentation_intent, "user_requested_visualization", False)),
                requested_chart_type=getattr(query_understanding.presentation_intent, "chart_type", None),
                visualization_payload=_preview_visualization_payload(query_understanding, displayed_evidences)[0],
                chart_data_payload=_preview_visualization_payload(query_understanding, displayed_evidences)[1],
            )
            quality = _quality_report(
                answer=final_answer,
                validation=validation,
                source_clickable_requested=bool(query_understanding.source_clickable_requested),
                recent_style_history=style_history,
            )
            composed_data = {"mode": generation_mode}
            elapsed = time.perf_counter() - started
            stage_times_ms["total_ms"] = round(elapsed * 1000.0, 3)
            return {
                "request_id": request_id,
                "query": q,
                "query_received": query_received,
                "query_used_for_retrieval": query_used_for_retrieval,
                "query_used_for_prompt": query_used_for_prompt,
                "query_stored": q,
                "normalized_query": q,
                "mode": "sql_deterministic",
                "provider": provider,
                "model": model,
                "top_k": top_k,
                "max_display_results": int(max_display_results),
                "show_all_results": bool(show_all_results),
                "show_low_quality": bool(show_low_quality),
                "timeout": timeout,
                "generation_time_seconds": round(elapsed, 3),
                "answer": final_answer,
                "citations": citations,
                "sources": source_citations_for_response,
                "validation": validation,
                "quality_report": quality,
                "llm_error": None,
                "error_type": None,
                "generation_mode": generation_mode,
                "detected_analytes": exact_analytes,
                "query_understanding": _query_understanding_payload(query_understanding),
                "structured_evidence_pack": structured_pack,
                "evidence_pack": evidence_pack,
                "displayed_evidences": displayed_evidences,
                "retrieval": {
                    "answerability": {"status": "answerable" if displayed_evidences else "insufficient_context", "reason": "deterministic_sql_fast_path"},
                    "filters": {"doc_ids": requested_doc_ids, "analytes": exact_analytes},
                    "top_results": [],
                    "context_chunks": [],
                    "sources": [],
                },
                "prompt": "",
                "debug": {
                    "request_id": request_id,
                    "selected_route": selected_route,
                    "route_reason": route_reason,
                    "generation_mode": generation_mode,
                    "generation_writer": "professional_fallback",
                    "selected_policy": selected_policy.get("selected_policy"),
                    "policy_level": selected_policy.get("policy_level"),
                    "facts_source": selected_policy.get("facts_source"),
                    "validator_policy": validator_policy,
                    "llm_allowed": llm_writer_allowed,
                    "llm_used": False,
                    "timeout_ms": policy_timeout_s * 1000,
                    "max_tokens": policy_max_tokens,
                    "query_understanding": _query_understanding_payload(query_understanding),
                    "stage_timings_ms": dict(stage_times_ms),
                },
                "visualization": None,
                "chart_data": None,
            }
        else:
            compose_mode = str(llm_cfg.get("compose_mode") or "fallback")
        t_llm0 = time.perf_counter()
        composed = compose_professional_answer(
            user_question=q,
            query_understanding=query_understanding,
            evidence_pack=structured_pack,
            mode=compose_mode,
            source_citations=source_citations,
            llm_client=llm_client,
            provider=provider,
            model=model,
            temperature=temperature,
            num_ctx=num_ctx,
            max_tokens=policy_max_tokens,
            timeout=policy_timeout_s,
        )
        if compose_mode == "fallback":
            stage_times_ms["llm_writer_ms"] = 0.0
        else:
            llm_writer_used = True
            stage_times_ms["llm_writer_ms"] = round((time.perf_counter() - t_llm0) * 1000.0, 3)
        composed_data = composed
        final_answer = str(composed.get("answer") or "").strip() or _missing_doc_answer()
        llm_prompt_preview = str(composed.get("llm_prompt_preview") or "")[:1200]
        llm_candidate_answer = str(composed.get("llm_candidate_answer") or final_answer)
        retry_used = False
        if str(query_understanding.intent or "").strip().lower() == "multi_doc_comparison" and len(requested_doc_ids) > 2:
            cmp_text, _ = _format_multi_doc_comparison_answer(
                rows=structured_rows,
                doc_ids=requested_doc_ids,
                requested_analytes=list(structured_pack.get("requested_analytes") or exact_analytes or []),
            )
            final_answer = ("Comparaison multi-documents demandée.\n\n" + cmp_text).strip()
            generation_mode = "deterministic_multi_doc_comparison"
        if (
            str(query_understanding.intent or "").strip().lower() == "multi_doc_comparison"
            and "analyte_not_identified_for_multi_doc_comparison" in set(structured_pack.get("missing_items") or [])
            and not (structured_pack.get("evidences") or [])
        ):
            final_answer = "L’analyte demandé n’a pas été identifié pour la comparaison entre les documents demandés."
        if str(query_understanding.intent or "").strip().lower() == "comment_without_measured_value":
            evidences = list(structured_pack.get("evidences") or [])
            if bool(structured_pack.get("comment_list_mode")) and evidences:
                final_answer = _build_multi_comment_answer(evidences)
            else:
                comment_text = str(structured_pack.get("comment_text") or "").strip()
                first_ev = evidences[0] if evidences and isinstance(evidences[0], dict) else {}
                source_label = str(first_ev.get("source") or "").strip()
                subject_label = str(first_ev.get("subject") or first_ev.get("analyte") or "").strip()
                if not subject_label:
                    subject_norm, subject_guess = _resolve_comment_subject_from_query(
                        query_understanding=query_understanding,
                        query=q,
                    )
                    subject_label = subject_guess if subject_guess else (subject_norm or "Commentaire médical")
                final_answer = build_qualitative_comment_answer(
                    subject=subject_label,
                    comment_text=comment_text,
                    source_label=source_label or "source non disponible",
                )
        generation_mode = str(composed.get("mode") or "deterministic_professional_fallback")
        if compose_mode == "fallback":
            pre_validation_mode_overrides = {
                "doc_scoped_abnormal_results": "deterministic_doc_scoped_abnormal_results",
                "doc_scoped_priority_anomalies": "deterministic_doc_scoped_priority_anomalies",
                "global_analyte_abnormal_search": "deterministic_global_analyte_abnormal_search",
                "doc_pair_comparison": "deterministic_doc_pair_comparison",
                "multi_doc_comparison": "deterministic_multi_doc_comparison",
                "doc_scoped_medical_interpretation_guarded": "deterministic_guarded_medical_interpretation",
                "single_analyte_lookup": "deterministic_single_analyte_lookup",
            }
            generation_mode = pre_validation_mode_overrides.get(selected_route, generation_mode)
        if str(query_understanding.intent or "").strip().lower() == "comment_without_measured_value":
            qdbg = structured_pack.get("qualitative_debug") if isinstance(structured_pack.get("qualitative_debug"), dict) else {}
            LOGGER.info(
                "qa_qualitative_post current_query=%r detected_intent=%s requested_context_type=%s enters_comment_without_measured_value_branch=%s qualitative_fetch_called=%s qualitative_rows_count=%s qualitative_comment_extracted=%s qualitative_comment_text_length=%s qualitative_comment_text_preview=%r final_generation_mode=%s final_answer_starts_with_information_insuffisante=%s",
                q,
                str(query_understanding.intent or ""),
                str(getattr(query_understanding, "requested_context_type", "") or ""),
                bool(qdbg.get("enters_comment_without_measured_value_branch")),
                bool(qdbg.get("qualitative_fetch_called")),
                int(qdbg.get("qualitative_rows_count") or 0),
                bool(qdbg.get("qualitative_comment_extracted")),
                int(qdbg.get("qualitative_comment_text_length") or 0),
                str(qdbg.get("qualitative_comment_text_preview") or ""),
                generation_mode,
                str(final_answer or "").strip().lower().startswith("information insuffisante"),
            )
        writer_error = str(composed.get("llm_error") or "") or None
        fallback_reason_debug = writer_error
        if writer_error and "timeout" in writer_error.lower():
            fallback_reason_debug = "llm_timeout"
        if compose_mode == "hybrid_structured_llm_writer" and generation_mode in {
            "llm_writer_error_fallback",
            "llm_writer_format_fallback",
            "llm_writer_quality_fallback",
        }:
            retry_used = True
            fallback_composed = compose_professional_answer(
                user_question=q,
                query_understanding=query_understanding,
                evidence_pack=structured_pack,
                mode="fallback",
                source_citations=source_citations,
            )
            final_answer = str(fallback_composed.get("answer") or final_answer).strip()
            generation_mode = "deterministic_safety_fallback_after_llm_validation_failure"
            writer_error = None
        if _contains_internal_reasoning_leak(final_answer):
            fallback_answer = str(
                (
                    render_professional_fallback(
                        evidence_pack=structured_pack,
                        query_understanding=query_understanding,
                        user_question=q,
                        source_citations=source_citations,
                    )
                    or {}
                ).get("answer")
                or _missing_doc_answer()
            ).strip()
            final_answer, _, sanitize_err = sanitize_final_answer_with_retry(
                answer=final_answer,
                user_message=q,
                llm_client=llm_client,
                provider=provider,
                model=model,
                timeout=timeout,
                fallback_answer=fallback_answer,
            )
            if sanitize_err:
                writer_error = f"{writer_error} | sanitize_retry:{sanitize_err}" if writer_error else f"sanitize_retry:{sanitize_err}"

        qn_local = norm_text(q)
        if (
            safety_intent_norm == "diagnostic_safety_question"
            and any(k in qn_local for k in ["diagnostic", "traitement", "recommandes", "recommander"])
        ):
            refusal = "Je ne peux pas poser ni évoquer un diagnostic à partir de ces résultats seuls."
            if not str(final_answer or "").strip().lower().startswith(refusal.lower()):
                final_answer = f"{refusal}\n\n{final_answer}".strip()
        if safety_intent_norm == "diagnostic_safety_question" and detect_medical_topic(qn_local) == "thyroid":
            tsh_high = False
            t3t4_high = False
            for ev in displayed_evidences:
                an = str(ev.get("analyte_norm") or "").strip().lower()
                st = str(ev.get("technical_status_code") or ev.get("interpretation_status") or "").strip().lower()
                if an in {"tshus", "tsh"} and st == "above_reference":
                    tsh_high = True
                if an in {"t3_libre", "t3libre", "t4_libre", "t4libre"} and st == "above_reference":
                    t3t4_high = True
            discordant_sentence = "Ce profil est biologiquement discordant pour une hyperthyroïdie primaire."
            if tsh_high and t3t4_high and discordant_sentence.lower() not in str(final_answer or "").lower():
                final_answer = f"{final_answer.rstrip()}\n\n{discordant_sentence}"

        missing_requested_doc_ids = _resolve_missing_requested_doc_ids(sqlite_path, requested_doc_ids)
        if not displayed_evidences and requested_doc_ids:
            missing_targets = ", ".join(requested_doc_ids)
            final_answer = (
                f"Information insuffisante dans le contexte fourni. Aucun résultat biologique exploitable n’a été retrouvé pour {missing_targets} dans les documents indexés."
            )
            generation_mode = "deterministic_no_evidence_response"
            writer_error = None
        found_requested_analytes = []
        for analyte in exact_analytes:
            if any(_row_matches_analyte(row, analyte) for row in structured_rows):
                found_requested_analytes.append(analyte)
                continue
            if str(structured_pack.get("comment_text") or "").strip():
                evs_for_subject = list(structured_pack.get("evidences") or [])
                ev0 = evs_for_subject[0] if evs_for_subject and isinstance(evs_for_subject[0], dict) else {}
                ev_norm = norm_text(str(ev0.get("analyte_norm") or ev0.get("analyte") or ev0.get("subject") or ""))
                if ev_norm and (ev_norm == norm_text(analyte) or norm_text(analyte) in ev_norm):
                    found_requested_analytes.append(analyte)
        missing_requested_analytes = sorted(
            {
                str(a).strip().lower()
                for a in (structured_pack.get("missing_items") or [])
                if str(a).strip()
            }
        )
        found_requested_analyte_norms = sorted(
            {
                str(ev.get("analyte_norm") or "").strip().lower()
                for ev in displayed_evidences
                if str(ev.get("analyte_norm") or "").strip()
            }
        )
        if exact_analytes and not missing_requested_analytes:
            missing_requested_analytes = [a for a in exact_analytes if a not in set(found_requested_analytes)]

        t_val0 = time.perf_counter()
        validation = validate_answer(
            query=q,
            answer_text=final_answer,
            evidence_pack=evidence_pack,
            displayed_evidences=displayed_evidences,
            source_citations=source_citations,
            exact_analyte=exact_analyte,
            llm_error=writer_error,
            generation_mode=generation_mode,
            retrieval_status="answerable" if displayed_evidences else "insufficient_context",
            show_low_quality=show_low_quality,
            max_display_results=max_display_results,
            show_all_results=show_all_results,
            query_received=query_received,
            query_used_for_retrieval=query_used_for_retrieval,
            query_used_for_prompt=query_used_for_prompt,
            query_stored=q,
            detected_analytes=exact_analytes,
            requested_doc_id=requested_doc_ids[0] if len(requested_doc_ids) == 1 else None,
            requested_doc_ids=requested_doc_ids,
            missing_requested_doc_ids=missing_requested_doc_ids,
            requested_analytes=exact_analytes,
            found_requested_analytes=found_requested_analytes,
            found_requested_analyte_norms=found_requested_analyte_norms,
            missing_requested_analytes=missing_requested_analytes,
            current_vs_previous_requested=query_understanding.requires_previous_results,
            diagnostic_safety_intent=bool(intents.get("diagnostic_safety_question")),
            query_intents=intents,
            output_format_requested=query_understanding.output_format,
            answer_style_requested=query_understanding.answer_style,
            requested_table_columns=query_understanding.requested_table_columns,
            requested_technical_condition=query_understanding.technical_condition,
            raw_format_phrase=getattr(query_understanding, "raw_format_phrase", None),
            unsupported_presentation=bool(getattr(query_understanding.presentation_intent, "unsupported_format", False)),
            user_requested_visualization=bool(getattr(query_understanding.presentation_intent, "user_requested_visualization", False)),
            requested_chart_type=getattr(query_understanding.presentation_intent, "chart_type", None),
            visualization_payload=_preview_visualization_payload(query_understanding, displayed_evidences)[0],
            chart_data_payload=_preview_visualization_payload(query_understanding, displayed_evidences)[1],
            source_clickable_requested=bool(query_understanding.source_clickable_requested),
            requested_value=query_understanding.requested_value,
            comparison_operator=query_understanding.comparison_operator,
        )
        stage_times_ms["validation_ms"] = round((time.perf_counter() - t_val0) * 1000.0, 3)
        quality = _quality_report(
            answer=final_answer,
            validation=validation,
            source_clickable_requested=bool(query_understanding.source_clickable_requested),
            recent_style_history=style_history,
        )
        validation_errors_set = {str(e) for e in (validation or {}).get("errors") or []}
        hard_gate_triggered = sorted(validation_errors_set.intersection(HARD_GATE_ERRORS))
        if compose_mode == "hybrid_structured_llm_writer" and hard_gate_triggered:
            validation = dict(validation or {})
            validation["validation_status"] = "fail"

        if _should_retry_with_validator(validation, generation_mode):
            t_rep0 = time.perf_counter()
            retry_used = True
            retry_feedback = _build_validator_retry_feedback(validation)
            retry_composed = compose_professional_answer(
                user_question=q,
                query_understanding=query_understanding,
                evidence_pack=structured_pack,
                mode=compose_mode,
                source_citations=source_citations,
                llm_client=llm_client,
                provider=provider,
                model=model,
                temperature=temperature,
                num_ctx=num_ctx,
                max_tokens=policy_max_tokens,
                timeout=policy_timeout_s,
                retry_feedback=retry_feedback,
            )
            retry_answer = str(retry_composed.get("answer") or "").strip()
            retry_mode = str(retry_composed.get("mode") or generation_mode)
            retry_writer_error = str(retry_composed.get("llm_error") or "") or None
            if retry_answer:
                llm_candidate_answer = retry_answer
            if str(retry_composed.get("llm_prompt_preview") or "").strip():
                llm_prompt_preview = str(retry_composed.get("llm_prompt_preview") or "")[:1200]
            if retry_answer:
                retry_validation = validate_answer(
                    query=q,
                    answer_text=retry_answer,
                    evidence_pack=evidence_pack,
                    displayed_evidences=displayed_evidences,
                    source_citations=source_citations,
                    exact_analyte=exact_analyte,
                    llm_error=retry_writer_error,
                    generation_mode=retry_mode,
                    retrieval_status="answerable" if displayed_evidences else "insufficient_context",
                    show_low_quality=show_low_quality,
                    max_display_results=max_display_results,
                    show_all_results=show_all_results,
                    query_received=query_received,
                    query_used_for_retrieval=query_used_for_retrieval,
                    query_used_for_prompt=query_used_for_prompt,
                    query_stored=q,
                    detected_analytes=exact_analytes,
                    requested_doc_id=requested_doc_ids[0] if len(requested_doc_ids) == 1 else None,
                    requested_doc_ids=requested_doc_ids,
                    missing_requested_doc_ids=missing_requested_doc_ids,
                    requested_analytes=exact_analytes,
                    found_requested_analytes=found_requested_analytes,
                    found_requested_analyte_norms=found_requested_analyte_norms,
                    missing_requested_analytes=missing_requested_analytes,
                    current_vs_previous_requested=query_understanding.requires_previous_results,
                    diagnostic_safety_intent=bool(intents.get("diagnostic_safety_question")),
                    query_intents=intents,
                    output_format_requested=query_understanding.output_format,
                    answer_style_requested=query_understanding.answer_style,
                    requested_table_columns=query_understanding.requested_table_columns,
                    requested_technical_condition=query_understanding.technical_condition,
                    source_clickable_requested=bool(query_understanding.source_clickable_requested),
                    requested_value=query_understanding.requested_value,
                    comparison_operator=query_understanding.comparison_operator,
                    raw_format_phrase=getattr(query_understanding, "raw_format_phrase", None),
                    unsupported_presentation=bool(getattr(query_understanding.presentation_intent, "unsupported_format", False)),
                    user_requested_visualization=bool(getattr(query_understanding.presentation_intent, "user_requested_visualization", False)),
                    requested_chart_type=getattr(query_understanding.presentation_intent, "chart_type", None),
                    visualization_payload=_preview_visualization_payload(query_understanding, displayed_evidences)[0],
                    chart_data_payload=_preview_visualization_payload(query_understanding, displayed_evidences)[1],
                )
                if str(retry_validation.get("validation_status") or "fail") != "fail":
                    final_answer = retry_answer
                    generation_mode = retry_mode
                    writer_error = retry_writer_error
                    validation = retry_validation
                else:
                    fallback_composed = compose_professional_answer(
                        user_question=q,
                        query_understanding=query_understanding,
                        evidence_pack=structured_pack,
                        mode="fallback",
                        source_citations=source_citations,
                    )
                    final_answer = str(fallback_composed.get("answer") or final_answer).strip()
                    generation_mode = "deterministic_safety_fallback_after_llm_validation_failure"
                    writer_error = None
                    validation = validate_answer(
                        query=q,
                        answer_text=final_answer,
                        evidence_pack=evidence_pack,
                        displayed_evidences=displayed_evidences,
                        source_citations=source_citations,
                        exact_analyte=exact_analyte,
                        llm_error=None,
                        generation_mode=generation_mode,
                        retrieval_status="answerable" if displayed_evidences else "insufficient_context",
                        show_low_quality=show_low_quality,
                        max_display_results=max_display_results,
                        show_all_results=show_all_results,
                        query_received=query_received,
                        query_used_for_retrieval=query_used_for_retrieval,
                        query_used_for_prompt=query_used_for_prompt,
                        query_stored=q,
                        detected_analytes=exact_analytes,
                        requested_doc_id=requested_doc_ids[0] if len(requested_doc_ids) == 1 else None,
                        requested_doc_ids=requested_doc_ids,
                        missing_requested_doc_ids=missing_requested_doc_ids,
                        requested_analytes=exact_analytes,
                        found_requested_analytes=found_requested_analytes,
                        found_requested_analyte_norms=found_requested_analyte_norms,
                        missing_requested_analytes=missing_requested_analytes,
                        current_vs_previous_requested=query_understanding.requires_previous_results,
                        diagnostic_safety_intent=bool(intents.get("diagnostic_safety_question")),
                        query_intents=intents,
                        output_format_requested=query_understanding.output_format,
                        answer_style_requested=query_understanding.answer_style,
                        requested_table_columns=query_understanding.requested_table_columns,
                        requested_technical_condition=query_understanding.technical_condition,
                        source_clickable_requested=bool(query_understanding.source_clickable_requested),
                        requested_value=query_understanding.requested_value,
                        comparison_operator=query_understanding.comparison_operator,
                        raw_format_phrase=getattr(query_understanding, "raw_format_phrase", None),
                        unsupported_presentation=bool(getattr(query_understanding.presentation_intent, "unsupported_format", False)),
                        user_requested_visualization=bool(getattr(query_understanding.presentation_intent, "user_requested_visualization", False)),
                        requested_chart_type=getattr(query_understanding.presentation_intent, "chart_type", None),
                        visualization_payload=_preview_visualization_payload(query_understanding, displayed_evidences)[0],
                        chart_data_payload=_preview_visualization_payload(query_understanding, displayed_evidences)[1],
                    )
                    quality = _quality_report(
                        answer=final_answer,
                        validation=validation,
                        source_clickable_requested=bool(query_understanding.source_clickable_requested),
                        recent_style_history=style_history,
                    )
            stage_times_ms["repair_ms"] = round((time.perf_counter() - t_rep0) * 1000.0, 3)

        # Hard gate: never return an invalid LLM/hybrid answer to end-users.
        llm_like_mode = (
            str(generation_mode or "").startswith("llm_")
            or str(generation_mode or "") in {"hybrid_structured_llm_writer", "llm"}
        )
        if llm_like_mode and str((validation or {}).get("validation_status") or "fail") == "fail":
            t_fb0 = time.perf_counter()
            fallback_composed = compose_professional_answer(
                user_question=q,
                query_understanding=query_understanding,
                evidence_pack=structured_pack,
                mode="fallback",
                source_citations=source_citations,
            )
            final_answer = str(fallback_composed.get("answer") or final_answer).strip()
            generation_mode = "deterministic_safety_fallback_after_llm_validation_failure"
            writer_error = None
            validation = validate_answer(
                query=q,
                answer_text=final_answer,
                evidence_pack=evidence_pack,
                displayed_evidences=displayed_evidences,
                source_citations=source_citations,
                exact_analyte=exact_analyte,
                llm_error=None,
                generation_mode=generation_mode,
                retrieval_status="answerable" if displayed_evidences else "insufficient_context",
                show_low_quality=show_low_quality,
                max_display_results=max_display_results,
                show_all_results=show_all_results,
                query_received=query_received,
                query_used_for_retrieval=query_used_for_retrieval,
                query_used_for_prompt=query_used_for_prompt,
                query_stored=q,
                detected_analytes=exact_analytes,
                requested_doc_id=requested_doc_ids[0] if len(requested_doc_ids) == 1 else None,
                requested_doc_ids=requested_doc_ids,
                missing_requested_doc_ids=missing_requested_doc_ids,
                requested_analytes=exact_analytes,
                found_requested_analytes=found_requested_analytes,
                found_requested_analyte_norms=found_requested_analyte_norms,
                missing_requested_analytes=missing_requested_analytes,
                current_vs_previous_requested=query_understanding.requires_previous_results,
                diagnostic_safety_intent=bool(intents.get("diagnostic_safety_question")),
                query_intents=intents,
                output_format_requested=query_understanding.output_format,
                answer_style_requested=query_understanding.answer_style,
                requested_table_columns=query_understanding.requested_table_columns,
                requested_technical_condition=query_understanding.technical_condition,
                source_clickable_requested=bool(query_understanding.source_clickable_requested),
                requested_value=query_understanding.requested_value,
                comparison_operator=query_understanding.comparison_operator,
                raw_format_phrase=getattr(query_understanding, "raw_format_phrase", None),
                unsupported_presentation=bool(getattr(query_understanding.presentation_intent, "unsupported_format", False)),
                user_requested_visualization=bool(getattr(query_understanding.presentation_intent, "user_requested_visualization", False)),
                requested_chart_type=getattr(query_understanding.presentation_intent, "chart_type", None),
                visualization_payload=_preview_visualization_payload(query_understanding, displayed_evidences)[0],
                chart_data_payload=_preview_visualization_payload(query_understanding, displayed_evidences)[1],
            )
            quality = _quality_report(
                answer=final_answer,
                validation=validation,
                source_clickable_requested=bool(query_understanding.source_clickable_requested),
                recent_style_history=style_history,
            )
            stage_times_ms["fallback_ms"] = round((time.perf_counter() - t_fb0) * 1000.0, 3)

        route_mode_overrides = {
            "doc_scoped_abnormal_results": "deterministic_doc_scoped_abnormal_results",
            "doc_scoped_biological_summary": "deterministic_doc_scoped_biological_summary",
            "doc_scoped_priority_anomalies": "deterministic_doc_scoped_priority_anomalies",
            "global_analyte_abnormal_search": "deterministic_global_analyte_abnormal_search",
            "doc_pair_comparison": "deterministic_doc_pair_comparison",
            "multi_doc_comparison": "deterministic_multi_doc_comparison",
            "doc_scoped_medical_interpretation_guarded": "deterministic_guarded_medical_interpretation",
            "single_analyte_lookup": "deterministic_single_analyte_lookup",
        }
        if selected_route == "cohort_search" and _canonical_technical_condition(query_understanding.technical_condition) in {
            "above_reference",
            "below_reference",
            "out_of_reference",
        }:
            route_mode_overrides["cohort_search"] = "deterministic_global_analyte_abnormal_search"
        if str(generation_mode or "").startswith("deterministic_") and selected_route in route_mode_overrides:
            generation_mode = route_mode_overrides[selected_route]

        final_safety_check_failed = False
        hard_gate_errors = set(str(e) for e in (validation or {}).get("errors") or [])
        hard_gate_hits = sorted(hard_gate_errors.intersection(HARD_GATE_ERRORS))
        hard_gate_triggered = bool(hard_gate_hits)
        if hard_gate_triggered:
            fallback_mode = _select_hard_gate_fallback_mode(
                hard_gate_errors=hard_gate_errors,
                selected_route=selected_route,
                safety_intent_norm=safety_intent_norm,
                has_evidence=bool(displayed_evidences),
                query_norm=norm_text(q),
            )
            if fallback_mode == "deterministic_general_conversation":
                final_answer = render_general_conversation_response(
                    detect_general_conversation(q) or str(query_understanding.intent or "small_talk")
                )
                selected_route = "general_conversation"
                route_reason = "general_conversation_validator_hard_gate"
                displayed_evidences = []
                evidence_pack = []
                source_citations = []
                citations = []
                stage_times_ms["retrieval_ms"] = 0.0
            else:
                fallback_composed = compose_professional_answer(
                    user_question=q,
                    query_understanding=query_understanding,
                    evidence_pack=structured_pack,
                    mode="fallback",
                    source_citations=source_citations,
                )
                final_answer = str(fallback_composed.get("answer") or final_answer).strip()
                if fallback_mode == "deterministic_diagnostic_refusal_with_technical_summary":
                    refusal = "Je ne peux pas poser ni évoquer un diagnostic à partir de ces résultats seuls."
                    if not final_answer.lower().startswith(refusal.lower()):
                        final_answer = f"{refusal}\n\n{final_answer}".strip()
                elif fallback_mode == "deterministic_treatment_refusal_with_technical_summary":
                    refusal = "Je ne peux pas recommander de traitement à partir de ces résultats seuls."
                    if not final_answer.lower().startswith(refusal.lower()):
                        final_answer = f"{refusal}\n\n{final_answer}".strip()
                elif fallback_mode == "deterministic_no_evidence_response":
                    final_answer = "Information insuffisante dans les données structurées disponibles pour répondre de façon fiable."
                    displayed_evidences = []
                    evidence_pack = []
                    source_citations = []
                    citations = []
            generation_mode = fallback_mode
            writer_error = None
            validation = validate_answer(
                query=q,
                answer_text=final_answer,
                evidence_pack=evidence_pack if fallback_mode != "deterministic_general_conversation" else [],
                displayed_evidences=displayed_evidences if fallback_mode != "deterministic_general_conversation" else [],
                source_citations=source_citations if fallback_mode != "deterministic_general_conversation" else [],
                generation_mode=generation_mode,
                retrieval_status="answerable" if displayed_evidences else "insufficient_context",
                query_received=query_received,
                query_used_for_retrieval=query_used_for_retrieval,
                query_used_for_prompt=query_used_for_prompt,
                query_stored=q,
                detected_analytes=exact_analytes,
                query_intents={**intents, "general_conversation": fallback_mode == "deterministic_general_conversation"},
                output_format_requested=query_understanding.output_format,
                answer_style_requested=query_understanding.answer_style,
                requested_table_columns=query_understanding.requested_table_columns,
                requested_technical_condition=query_understanding.technical_condition,
                source_clickable_requested=bool(query_understanding.source_clickable_requested),
                requested_value=query_understanding.requested_value,
                comparison_operator=query_understanding.comparison_operator,
                raw_format_phrase=getattr(query_understanding, "raw_format_phrase", None),
                unsupported_presentation=False,
                user_requested_visualization=False,
                requested_chart_type=None,
                visualization_payload=None,
                chart_data_payload=None,
            )
            quality = _quality_report(
                answer=final_answer,
                validation=validation,
                source_clickable_requested=bool(query_understanding.source_clickable_requested),
                recent_style_history=style_history,
            )
            stage_times_ms["llm_writer_ms"] = 0.0
            stage_times_ms["repair_ms"] = 0.0

        if str((validation or {}).get("validation_status") or "").strip().lower() == "fail":
            final_safety_check_failed = True
            final_answer = "Je ne peux pas fournir une réponse fiable à partir des données disponibles."
            generation_mode = "deterministic_safe_error_response"
            displayed_evidences = []
            evidence_pack = []
            source_citations = []
            citations = []
            validation = {
                "validation_status": "warning",
                "errors": [],
                "warnings": ["final_safety_check_failed"],
            }
            quality = _quality_report(
                answer=final_answer,
                validation=validation,
                source_clickable_requested=False,
                recent_style_history=style_history,
            )
            stage_times_ms["llm_writer_ms"] = 0.0
            stage_times_ms["repair_ms"] = 0.0

        elapsed = time.perf_counter() - started
        stage_times_ms["total_ms"] = round(elapsed * 1000.0, 3)
        intro_text, conclusion_text = _extract_intro_conclusion(final_answer)
        retrieval_sources = [
            {
                "doc_id": ev.get("doc_id"),
                "page_number": ev.get("page_number"),
                "chunk_id": ev.get("chunk_id"),
                "chunk_type": ev.get("chunk_type"),
            }
            for ev in displayed_evidences
        ]
        return _inject_visualization_payload(
            {
            "request_id": request_id,
            "query": q,
            "query_received": query_received,
            "query_used_for_retrieval": query_used_for_retrieval,
            "query_used_for_prompt": query_used_for_prompt,
            "query_stored": q,
            "normalized_query": q,
            "mode": "sql_deterministic",
            "provider": provider,
            "model": model,
            "top_k": top_k,
            "max_display_results": int(max_display_results),
            "show_all_results": bool(show_all_results),
            "show_low_quality": bool(show_low_quality),
            "timeout": timeout,
            "generation_time_seconds": round(elapsed, 3),
            "answer": final_answer,
            "citations": citations,
            "sources": source_citations,
            "validation": validation,
            "quality_report": quality,
            "llm_error": writer_error,
            "error_type": "llm_writer_error" if writer_error else None,
            "generation_mode": generation_mode,
            "detected_analytes": exact_analytes,
            "query_understanding": _query_understanding_payload(query_understanding),
            "structured_evidence_pack": structured_pack,
            "style_memory_entry": {
                "intro_text": intro_text,
                "conclusion_text": conclusion_text,
                "intent": query_understanding.intent,
                "output_format": query_understanding.output_format,
                "answer_text": final_answer,
            },
            "evidence_pack": evidence_pack,
            "displayed_evidences": displayed_evidences,
            "display": {
                "selected_candidates_count": len(evidence_pack),
                "low_quality_evidence_filtered_count": 0,
                "hidden_result_count": max(0, len(evidence_pack) - len(displayed_evidences)),
                "requested_multi_result_query": _query_requests_multiple_results(qn),
                "display_notes": [],
            },
            "retrieval": {
                "answerability": {"status": "answerable" if displayed_evidences else "insufficient_context", "reason": "deterministic_sql_fast_path"},
                "filters": {"doc_ids": requested_doc_ids, "analytes": exact_analytes},
                "top_results": [],
                "context_chunks": [],
                "sources": retrieval_sources,
            },
            "prompt": "",
            "debug": {
                "request_id": request_id,
                "query_received": query_received,
                "query_used_for_retrieval": query_used_for_retrieval,
                "query_used_for_prompt": query_used_for_prompt,
                "detected_analytes": exact_analytes,
                "requested_doc_ids": requested_doc_ids,
                "generation_mode": generation_mode,
                "selected_route": selected_route,
                "route_reason": route_reason,
                "generation_mode_before_fallback": str(composed_data.get("mode") if isinstance(composed_data, dict) else "") or None,
                "fallback_reason": fallback_reason_debug,
                "generation_writer": "llm_writer" if str(generation_mode).startswith("llm_") or generation_mode == "hybrid_structured_llm_writer" else "professional_fallback",
                "retry_used": retry_used,
                "final_generation_mode": generation_mode,
                "selected_policy": selected_policy.get("selected_policy"),
                "policy_level": selected_policy.get("policy_level"),
                "facts_source": selected_policy.get("facts_source"),
                "validator_policy": validator_policy,
                "llm_allowed": llm_writer_allowed,
                "llm_used": llm_writer_used,
                "timeout_ms": policy_timeout_s * 1000,
                "max_tokens": policy_max_tokens,
                "llm_writer_allowed": llm_writer_allowed,
                "llm_writer_used": llm_writer_used,
                "evidence_rows_count": len(evidence_pack),
                "displayed_evidences_count": len(displayed_evidences),
                "hard_gate_triggered": hard_gate_triggered,
                "hard_gate_errors": hard_gate_hits if 'hard_gate_hits' in locals() else [],
                "hard_gate_policy": "global_non_negotiable",
                "repair_used": retry_used,
                "fallback_generation_mode": generation_mode if hard_gate_triggered else None,
                "final_safety_check_status": "failed" if final_safety_check_failed else "passed",
                "final_safety_check_failed": final_safety_check_failed,
                "llm_prompt_preview": llm_prompt_preview,
                "llm_candidate_answer": llm_candidate_answer,
                "intents": intents,
                "analyte_resolution_debug": structured_pack.get("analyte_resolution_debug"),
                "query_understanding": _query_understanding_payload(query_understanding),
                "query_plan": plan_to_payload(query_plan),
                "evidence_rows_preview": list((structured_pack.get("rows") or [])[:8]),
                "included_rows": list((structured_pack.get("evidences") or [])[:20]),
                "excluded_rows": list(structured_pack.get("excluded_rows") or []),
                "validation": {
                    "errors": list((validation or {}).get("errors") or []),
                    "warnings": list((validation or {}).get("warnings") or []),
                },
                "validation_errors": list((validation or {}).get("errors") or []),
                "stage_timings_ms": dict(stage_times_ms),
            },
            "exact_analyte_coverage": {
                "detected_exact_analyte": exact_analyte,
                "expected_exact_analyte_count": len(exact_analytes),
                "retrieved_exact_analyte_count": len(found_requested_analytes),
                "displayed_exact_analyte_count": len(found_requested_analytes),
            },
            },
            query_understanding=query_understanding,
            displayed_evidences=displayed_evidences,
        )

    if requested_doc_id:
        retrieval_filters.doc_id = requested_doc_id
    if exact_analyte:
        retrieval_filters.analyte_norm = exact_analyte
    if is_above_reference_query and not is_normal_or_above:
        retrieval_filters.interpretation_status = "above_reference"
    elif is_below_reference_query:
        retrieval_filters.interpretation_status = "below_reference"

    retrieval_response: Any
    max_exact_analyte_results = 10
    max_above_reference_results = 10
    exact_analyte_expected_count = 0
    exact_analyte_rows: list[dict[str, Any]] = []
    requested_analyte_rows: list[dict[str, Any]] = []
    supplemental_rows: list[dict[str, Any]] = []
    retrieval_error: str | None = None
    if sensitive_or_treatment:
        retrieval_response = SimpleNamespace(
            answerability={"status": "guardrail_blocked", "reason": "sensitive_or_treatment_query"},
            filters={},
            top_results=[],
            context_chunks=[],
            sources=[],
        )
    else:
        if exact_analyte:
            exact_analyte_expected_count, exact_analyte_rows = _load_exact_analyte_rows(
                sqlite_path=sqlite_path,
                analyte_norm=exact_analyte,
                limit=max(top_k, max_exact_analyte_results),
                doc_ids=requested_doc_ids,
            )
        if len(exact_analytes) >= 2:
            requested_analyte_rows = _load_requested_analyte_rows(
                sqlite_path=sqlite_path,
                analyte_norms=exact_analytes,
                limit=max(top_k, max(8, len(exact_analytes) * 4)),
                doc_ids=requested_doc_ids,
            )
        if is_global_above_query:
            supplemental_rows = _load_interpretation_rows(
                sqlite_path=sqlite_path,
                interpretation_status="above_reference",
                limit=max(top_k, max_above_reference_results),
                doc_ids=requested_doc_ids,
            )
        created_engine = search_engine is None
        engine = search_engine or SearchEngine(
            sqlite_path=sqlite_path,
            qdrant_dir=qdrant_dir,
            collection=collection,
        )
        try:
            retrieval_response = engine.search(
                query=query_used_for_retrieval,
                mode=mode,
                top_k=top_k,
                filters=retrieval_filters,
                expand_context=True,
            )
            if retrieval_filters.analyte_norm and not retrieval_response.top_results:
                relaxed_filters = replace(retrieval_filters)
                relaxed_filters.analyte_norm = None
                retrieval_response = engine.search(
                    query=query_used_for_retrieval,
                    mode=mode,
                    top_k=top_k,
                    filters=relaxed_filters,
                    expand_context=True,
                )
            if retrieval_filters.interpretation_status and not retrieval_response.top_results:
                relaxed_filters = replace(retrieval_filters)
                relaxed_filters.interpretation_status = None
                retrieval_response = engine.search(
                    query=query_used_for_retrieval,
                    mode=mode,
                    top_k=top_k,
                    filters=relaxed_filters,
                    expand_context=True,
                )
        except Exception as exc:
            retrieval_error = str(exc)
            retrieval_response = SimpleNamespace(
                answerability={"status": "retrieval_error", "reason": retrieval_error},
                filters=retrieval_filters.to_dict(),
                top_results=[],
                context_chunks=[],
                sources=[],
            )
        finally:
            if created_engine:
                engine.close()

    if requested_doc_ids:
        _filter_retrieval_response_by_doc_ids(retrieval_response, requested_doc_ids)
        exact_analyte_rows = _filter_rows_by_doc_ids(exact_analyte_rows, requested_doc_ids)
        requested_analyte_rows = _filter_rows_by_doc_ids(requested_analyte_rows, requested_doc_ids)
        supplemental_rows = _filter_rows_by_doc_ids(supplemental_rows, requested_doc_ids)

    if requested_analyte_rows:
        supplemental_rows = list(supplemental_rows) + list(requested_analyte_rows)

    evidence_pack = build_retrieval_evidence_pack(
        retrieval_response,
        query=q,
        max_evidence=(
            max(top_k, max_exact_analyte_results)
            if exact_analyte
            else max(top_k, max_above_reference_results) if is_global_above_query else top_k
        ),
        exact_analyte=exact_analyte,
        exact_analyte_rows=exact_analyte_rows,
        supplemental_rows=supplemental_rows,
        max_exact_analyte_results=max(top_k, max_exact_analyte_results),
    )
    if requested_doc_ids:
        allowed_docs = {d.lower() for d in requested_doc_ids}
        evidence_pack = [ev for ev in evidence_pack if str(ev.get("doc_id") or "").strip().lower() in allowed_docs]
    excluded_analytes = list(getattr(query_understanding, "excluded_analytes", []) or [])
    if excluded_analytes:
        evidence_pack = [ev for ev in evidence_pack if not _row_matches_excluded(ev, excluded_analytes)]
    if _query_requests_out_of_reference_only(qn):
        evidence_pack = [
            ev
            for ev in evidence_pack
            if str(ev.get("interpretation_status") or ev.get("technical_status_code") or "").strip().lower() in {"above_reference", "below_reference"}
        ]

    displayed_evidences, display_meta = _select_displayed_evidences(
        query_norm=qn,
        evidence_pack=evidence_pack,
        exact_analyte=exact_analyte,
        requested_analytes=exact_analytes,
        max_display_results=max(max_display_results, len(exact_analytes)) if len(exact_analytes) >= 2 else max_display_results,
        show_all_results=show_all_results,
        show_low_quality=show_low_quality,
    )

    prompt_evidence_pack = displayed_evidences if displayed_evidences else evidence_pack
    prompt = build_prompt(
        query=query_used_for_prompt,
        evidence_pack=prompt_evidence_pack,
        exact_analyte=exact_analyte,
    )

    llm_answer = ""
    llm_error = None
    generation_mode = "llm"
    error_type: str | None = None

    if sensitive_or_treatment:
        llm_answer = INSUFFICIENT_CONTEXT_SENTENCE
        generation_mode = "guardrail_blocked"
    elif retrieval_error:
        llm_error = f"Retrieval error: {retrieval_error}"
        error_type = "retrieval_error"
        generation_mode = "error"
    elif not evidence_pack:
        llm_answer = INSUFFICIENT_CONTEXT_SENTENCE
        generation_mode = "no_evidence"
    elif not displayed_evidences:
        llm_answer = INSUFFICIENT_CONTEXT_SENTENCE
        generation_mode = "no_displayable_evidence"
    elif _should_use_deterministic_generation(q, evidence_pack, exact_analyte):
        llm_answer = _build_deterministic_evidence_answer(
            query=q,
            displayed_evidences=displayed_evidences,
            exact_analyte=exact_analyte,
            display_notes=display_meta.get("display_notes") or [],
        )
        generation_mode = "deterministic_evidence_template"
    else:
        client = llm_client or LLMClient(provider=provider)
        try:
            llm_answer = client.generate(
                prompt=prompt,
                model=model,
                temperature=temperature,
                num_ctx=num_ctx,
                max_tokens=max_tokens,
                timeout=timeout,
                keep_alive="10m",
            )
            llm_answer = sanitize_final_answer(llm_answer)
            if _contains_internal_reasoning_leak(llm_answer):
                llm_answer, _, sanitize_err = sanitize_final_answer_with_retry(
                    answer=llm_answer,
                    user_message=q,
                    llm_client=llm_client,
                    provider=provider,
                    model=model,
                    timeout=timeout,
                    fallback_answer=_build_structured_fallback_answer(q, displayed_evidences, exact_analyte=exact_analyte),
                )
                if sanitize_err:
                    llm_error = f"{llm_error} | sanitize_retry:{sanitize_err}" if llm_error else f"sanitize_retry:{sanitize_err}"
            if _answer_needs_fallback(llm_answer):
                llm_answer = _build_structured_fallback_answer(q, displayed_evidences, exact_analyte=exact_analyte)
                generation_mode = "llm_fallback_template"
        except LLMClientError as exc:
            llm_error = str(exc)
            if "timeout" in llm_error.lower():
                error_type = "llm_timeout"
            else:
                error_type = "llm_error"
            if displayed_evidences:
                llm_answer = _build_structured_fallback_answer(q, displayed_evidences, exact_analyte=exact_analyte)
                generation_mode = "llm_error_fallback_template"
            else:
                generation_mode = "error"

    citations = build_citations(displayed_evidences)
    source_citations = build_source_citations(displayed_evidences, resolver=source_resolver)

    if llm_error and not llm_answer:
        final_answer = append_source_citations(
            f"Erreur LLM: {llm_error}",
            source_citations,
            fallback_citations=citations,
        )
    else:
        final_answer = append_source_citations(llm_answer, source_citations, fallback_citations=citations)
    final_answer = sanitize_final_answer(final_answer)
    if _contains_internal_reasoning_leak(final_answer):
        cleaned_answer, _, sanitize_err = sanitize_final_answer_with_retry(
            answer=final_answer,
            user_message=q,
            llm_client=llm_client,
            provider=provider,
            model=model,
            timeout=timeout,
            fallback_answer=append_source_citations(
                _build_structured_fallback_answer(q, displayed_evidences, exact_analyte=exact_analyte),
                source_citations,
                fallback_citations=citations,
            ),
        )
        final_answer = cleaned_answer
        if sanitize_err:
            llm_error = f"{llm_error} | sanitize_retry:{sanitize_err}" if llm_error else f"sanitize_retry:{sanitize_err}"

    missing_requested_doc_ids = _resolve_missing_requested_doc_ids(sqlite_path, requested_doc_ids)
    found_requested_analyte_norms = sorted(
        {
            str(ev.get("analyte_norm") or "").strip().lower()
            for ev in displayed_evidences
            if str(ev.get("analyte_norm") or "").strip()
        }
    )
    requested_analytes_norm = sorted({str(a).strip().lower() for a in exact_analytes if str(a).strip()})
    missing_requested_analytes = sorted(a for a in requested_analytes_norm if a not in set(found_requested_analyte_norms))

    validation = validate_answer(
        query=q,
        answer_text=final_answer,
        evidence_pack=evidence_pack,
        displayed_evidences=displayed_evidences,
        source_citations=source_citations,
        exact_analyte=exact_analyte,
        llm_error=llm_error,
        generation_mode=generation_mode,
        retrieval_status=(retrieval_response.answerability or {}).get("status"),
        show_low_quality=show_low_quality,
        max_display_results=max_display_results,
        show_all_results=show_all_results,
        query_received=query_received,
        query_used_for_retrieval=query_used_for_retrieval,
        query_used_for_prompt=query_used_for_prompt,
        query_stored=q,
        detected_analytes=exact_analytes,
        requested_doc_id=requested_doc_id,
        requested_doc_ids=requested_doc_ids,
        missing_requested_doc_ids=missing_requested_doc_ids,
        requested_analytes=exact_analytes,
        found_requested_analytes=found_requested_analyte_norms,
        found_requested_analyte_norms=found_requested_analyte_norms,
        missing_requested_analytes=missing_requested_analytes,
        current_vs_previous_requested=query_understanding.requires_previous_results,
        diagnostic_safety_intent=bool(intents.get("diagnostic_safety_question")),
        query_intents=intents,
        output_format_requested=query_understanding.output_format,
        answer_style_requested=query_understanding.answer_style,
        requested_table_columns=query_understanding.requested_table_columns,
        requested_technical_condition=query_understanding.technical_condition,
        source_clickable_requested=bool(query_understanding.source_clickable_requested),
        requested_value=query_understanding.requested_value,
        comparison_operator=query_understanding.comparison_operator,
        raw_format_phrase=getattr(query_understanding, "raw_format_phrase", None),
        unsupported_presentation=bool(getattr(query_understanding.presentation_intent, "unsupported_format", False)),
        user_requested_visualization=bool(getattr(query_understanding.presentation_intent, "user_requested_visualization", False)),
        requested_chart_type=getattr(query_understanding.presentation_intent, "chart_type", None),
        visualization_payload=_preview_visualization_payload(query_understanding, displayed_evidences)[0],
        chart_data_payload=_preview_visualization_payload(query_understanding, displayed_evidences)[1],
    )
    quality = _quality_report(
        answer=final_answer,
        validation=validation,
        source_clickable_requested=bool(query_understanding.source_clickable_requested),
        recent_style_history=style_history,
    )
    intro_text, conclusion_text = _extract_intro_conclusion(final_answer)

    elapsed = time.perf_counter() - started

    result: dict[str, Any] = {
        "request_id": request_id,
        **composed_data,
        "query": q,
        "query_received": query_received,
        "query_used_for_retrieval": query_used_for_retrieval,
        "query_used_for_prompt": query_used_for_prompt,
        "query_stored": q,
        "normalized_query": q,
        "mode": mode,
        "provider": provider,
        "model": model,
        "top_k": top_k,
        "max_display_results": int(max_display_results),
        "show_all_results": bool(show_all_results),
        "show_low_quality": bool(show_low_quality),
        "timeout": timeout,
        "generation_time_seconds": round(elapsed, 3),
        "answer": final_answer,
        "citations": citations,
        "sources": source_citations,
        "validation": validation,
        "quality_report": quality,
        "llm_error": llm_error,
        "error_type": error_type,
        "generation_mode": generation_mode,
        "detected_analytes": exact_analytes,
        "query_understanding": _query_understanding_payload(query_understanding),
        "structured_evidence_pack": {
            "question": q,
            "original_user_question": getattr(query_understanding, "original_user_question", q),
            "intent": query_understanding.intent,
            "response_strategy": getattr(query_understanding, "response_strategy", "render_table"),
            "response_strategy_reason": getattr(query_understanding, "response_strategy_reason", None),
            "output_format": query_understanding.output_format,
            "answer_style": query_understanding.answer_style,
            "requested_doc_ids": list(query_understanding.requested_doc_ids or []),
            "requested_analytes": list(query_understanding.requested_analytes or []),
            "requested_table_columns": list(query_understanding.requested_table_columns or []),
            "technical_condition": query_understanding.technical_condition,
            "presentation_intent": {
                "requested_output": getattr(query_understanding.presentation_intent, "requested_output", query_understanding.output_format),
                "chart_type": getattr(query_understanding.presentation_intent, "chart_type", None),
                "requested_type": _normalize_requested_visualization_type(
                    getattr(query_understanding.presentation_intent, "chart_type", None),
                    getattr(query_understanding.presentation_intent, "raw_format_phrase", None),
                ),
                "raw_format_phrase": getattr(query_understanding.presentation_intent, "raw_format_phrase", None),
                "unsupported_format": bool(getattr(query_understanding.presentation_intent, "unsupported_format", False)),
                "user_requested_visualization": bool(getattr(query_understanding.presentation_intent, "user_requested_visualization", False)),
                "unsupported_reason": getattr(query_understanding.presentation_intent, "unsupported_reason", None),
                "recommended_output": getattr(query_understanding.presentation_intent, "recommended_output", None),
                "recommended_alternative_format": getattr(query_understanding.presentation_intent, "recommended_alternative_format", None),
            },
            "evidences": list(displayed_evidences),
            "results": list(displayed_evidences),
            "sources": list(source_citations),
        },
        "evidence_pack": evidence_pack,
        "displayed_evidences": displayed_evidences,
        "display": display_meta,
        "retrieval": {
            "answerability": retrieval_response.answerability,
            "filters": retrieval_response.filters,
            "top_results": [r.to_dict() for r in retrieval_response.top_results],
            "context_chunks": [r.to_dict() for r in retrieval_response.context_chunks],
            "sources": retrieval_response.sources,
        },
        "style_memory_entry": {
            "intro_text": intro_text,
            "conclusion_text": conclusion_text,
            "intent": query_understanding.intent,
            "output_format": query_understanding.output_format,
            "answer_text": final_answer,
        },
        "prompt": prompt,
        "debug": {
            "request_id": request_id,
            "query_received": query_received,
            "query_used_for_retrieval": query_used_for_retrieval,
            "query_used_for_prompt": query_used_for_prompt,
            "detected_analytes": exact_analytes,
            "requested_doc_ids": requested_doc_ids,
            "generation_mode": generation_mode,
            "context_resolution": context_resolution,
            "deictic_resolution": deictic_resolution,
            "resolution_arbitration": resolution_arbitration,
            "retrieval_skipped_due_to_no_transformable_context": bool(
                context_resolution.get("reason") == "response_transform_no_transformable_context"
            ),
        },
        "exact_analyte_coverage": {
            "detected_exact_analyte": exact_analyte,
            "expected_exact_analyte_count": exact_analyte_expected_count if exact_analyte else 0,
            "retrieved_exact_analyte_count": sum(
                1
                for ev in displayed_evidences
                if exact_analyte
                and (
                    contains_exact_term(str(ev.get("analyte_norm") or ""), exact_analyte)
                    or contains_exact_term(str(ev.get("analyte") or ""), exact_analyte)
                )
            ),
            "displayed_exact_analyte_count": _count_displayed_exact_analyte(final_answer, exact_analyte or ""),
        },
    }

    return _inject_visualization_payload(
        result,
        query_understanding=query_understanding,
        displayed_evidences=displayed_evidences,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate grounded medical answer using local LLM")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", choices=["keyword", "vector", "hybrid"], default="hybrid")
    parser.add_argument("--provider", default=DEFAULT_LLM_PROVIDER)
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_LLM_TEMPERATURE)
    parser.add_argument("--num-ctx", type=int, default=DEFAULT_LLM_NUM_CTX)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_LLM_MAX_TOKENS)
    parser.add_argument("--timeout", type=int, default=DEFAULT_LLM_TIMEOUT)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--show-context", action="store_true")
    parser.add_argument("--max-display-results", type=int, default=3)
    parser.add_argument("--show-all-results", action="store_true")
    parser.add_argument("--show-low-quality", action="store_true")
    parser.add_argument("--index-dir", default="data/indexes")
    parser.add_argument("--collection", default="medical_chunks")
    return parser.parse_args()


def _print_human(result: dict[str, Any], show_context: bool) -> None:
    print("Réponse :")
    print(result.get("answer") or "")

    validation = result.get("validation") or {}
    print("\nValidation :")
    print(f"- status: {validation.get('validation_status')}")
    print(f"- pii_leak_detected: {validation.get('pii_leak_detected')}")
    print(f"- citation_present: {validation.get('citation_present')}")
    print(f"- insufficient_context_handled: {validation.get('insufficient_context_handled')}")

    if validation.get("warnings"):
        print("- warnings:")
        for w in validation["warnings"]:
            print(f"  - {w}")
    if validation.get("errors"):
        print("- errors:")
        for e in validation["errors"]:
            print(f"  - {e}")

    print(f"\nTemps génération: {result.get('generation_time_seconds')} s")

    if show_context:
        print("\nEvidence pack:")
        print(json.dumps(result.get("evidence_pack") or [], ensure_ascii=False, indent=2))


def main() -> int:
    args = _parse_args()

    try:
        result = run_generation(
            query=args.query,
            top_k=args.top_k,
            mode=args.mode,
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
            num_ctx=args.num_ctx,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            index_dir=args.index_dir,
            collection=args.collection,
            max_display_results=args.max_display_results,
            show_all_results=args.show_all_results,
            show_low_quality=args.show_low_quality,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _print_human(result, show_context=args.show_context)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
