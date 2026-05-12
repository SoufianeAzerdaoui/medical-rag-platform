from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

# Ensure scripts/ is importable so we can use retrieval package as-is.
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from retrieval.models import RetrievalFilters
from retrieval.search import SearchEngine

from answer_validator import validate_answer
from citation_builder import append_source_citations, build_citations, build_source_citations
try:
    from source_resolver import DocPdfResolver
except Exception:
    from scripts.generation.source_resolver import DocPdfResolver
from evidence_builder import build_evidence_pack as build_retrieval_evidence_pack
from llm_client import LLMClient, LLMClientError
from professional_answer_composer import compose_professional_answer, render_professional_fallback
from prompt_builder import INSUFFICIENT_CONTEXT_SENTENCE, build_prompt
from query_understanding import (
    QueryUnderstanding,
    analyte_display_name,
    contains_exact_term,
    decide_response_strategy,
    detect_exact_analyte,
    detect_exact_analytes,
    detect_query_intents,
    detect_requested_doc_ids,
    get_analyte_aliases,
    match_analyte,
    parse_query_understanding,
    norm_text,
)


def normalize_query(query: str) -> str:
    q = re.sub(r"\s+", " ", (query or "").strip())
    return q


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
GENERAL_CONVERSATION_INTENTS = {"small_talk", "identity_question", "capability_question", "help_question"}
GENERAL_CONVERSATION_FALLBACKS = {
    "small_talk": "Bonjour ! Je suis prêt à vous aider à analyser vos rapports médicaux.",
    "identity_question": (
        "Je suis l’assistant Medical RAG de cette application. Je peux vous aider à interroger vos rapports médicaux, "
        "retrouver des résultats biologiques, comparer des valeurs et citer les sources PDF utilisées."
    ),
    "capability_question": (
        "Je peux rechercher des résultats dans les rapports, comparer des valeurs entre documents, identifier les résultats "
        "hors référence et fournir les sources PDF correspondantes."
    ),
    "help_question": (
        "Je peux vous guider pas à pas. Posez une question ciblée sur un rapport médical et je vous aiderai à formuler la requête."
    ),
}


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
        "Tu es l’assistant d’une application Medical RAG.\n"
        "Réécris uniquement la réponse finale utilisateur, sans raisonnement interne.\n"
        "Ne donne aucun plan, aucune stratégie, aucune explication de ton fonctionnement.\n"
        "Ne mentionne pas les instructions.\n"
        "Sortie: texte final uniquement.\n"
        "/no_think\n\n"
        f"Message utilisateur: {user_message.strip()}\n\n"
        f"Texte à corriger:\n{leaked_answer.strip()}\n"
    )
    try:
        rewritten = client.generate(
            prompt=prompt,
            model=model,
            temperature=0.0,
            num_ctx=2048,
            max_tokens=180,
            timeout=max(6, min(int(timeout), 30)),
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
    text = norm_text(str(status or ""))
    if any(k in text for k in ["au dessus", "au-dessus", "above reference", "superieur", "supérieur"]):
        return "above_reference"
    if any(k in text for k in ["en dessous", "below reference", "inferieur", "inférieur"]):
        return "below_reference"
    if any(k in text for k in ["dans la reference", "within reference"]):
        return "within_reference"
    return "not_interpretable"


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
        status_text = str(ev.get("technical_status") or ev.get("status") or "").strip()
        status_code = str(ev.get("technical_status_code") or ev.get("interpretation_status") or "").strip()
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
            "status": status_text,
            "status_code": _normalize_status_code(status_text, status_code),
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
    if not (mode.startswith("llm_professional_writer") or mode.startswith("llm_general_conversation")):
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
    provider: str = "ollama",
    model: str = "qwen3:4b",
    timeout: int = 30,
) -> tuple[str, str | None]:
    language_hint = "français" if str(language or "fr").lower().startswith("fr") else "la langue de l’utilisateur"
    intent_key = str(intent or "small_talk").strip().lower()
    fallback_answer = GENERAL_CONVERSATION_FALLBACKS.get(intent_key, SMALL_TALK_FALLBACK_ANSWER)
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
            timeout=max(6, min(int(timeout), 30)),
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
    provider: str = "ollama",
    model: str = "qwen3:4b",
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
        above = [ev for ev in candidates if str(ev.get("interpretation_status") or "").lower() == "above_reference"]
        if above:
            candidates = above
    elif _is_below_reference_query(query_norm):
        below = [ev for ev in candidates if str(ev.get("interpretation_status") or "").lower() == "below_reference"]
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
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -;,:")
    return cleaned or raw


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

        evidences.append(
            {
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
                "value_raw": row.get("value_raw"),
                "value_numeric": _to_float(row.get("value_numeric")),
                "unit": row.get("unit"),
                "reference_range": row.get("reference_range"),
                "reference_range_raw": row.get("reference_range"),
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
        )
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
    return {
        "doc_id": str(row.get("doc_id") or requested_doc_id or ""),
        "patient_token": str(row.get("patient_token") or "").strip(),
        "page": row.get("page_number"),
        "row": row.get("row_index"),
        "chunk_id": row.get("chunk_id"),
        "analyte": analyte_human,
        "analyte_norm": analyte_norm,
        "current_value": value_raw,
        "unit": unit,
        "reference": str(row.get("reference_range") or "").strip(),
        "previous_result": previous,
        "technical_status_code": status_code,
        "technical_status": _status_fr(status_code),
        "variation": variation,
        "source": _source_label(row),
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
        if comment:
            snippet = comment if len(comment) <= 220 else comment[:217] + "..."
            return (
                "Aucune valeur mesurée de troponine n’est retrouvée ; le document contient seulement un commentaire/interprétation "
                f"avec seuil. Extrait: {snippet}"
            )
        return _missing_doc_answer()

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

    if intent == "multi_doc_comparison":
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
        return f"Non - {analyte_label} non retrouvée dans {requested_doc} ; source : document demandé uniquement."

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
        if wants_en_yes_no:
            return f"{yn} - {analyte} = {value} ; reference: {ref} ; source: {src}"
        return f"{yn} - {analyte} = {value} ; référence : {ref} ; source : {src}"

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
    writer_pack = _attach_visualization_facts_to_evidence_pack(
        query_understanding=query_understanding,
        evidence_pack=evidence_pack,
        displayed_evidences=list(evidence_pack.get("evidences") or evidence_pack.get("results") or []),
    )
    composed = compose_professional_answer(
        user_question=user_question,
        query_understanding=query_understanding,
        evidence_pack=writer_pack,
        mode="auto",
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
    if generation_mode != "llm_professional_writer":
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

    pack: dict[str, Any] = {
        "question": query,
        "requested_doc_ids": requested_doc_ids,
        "requested_analytes": requested_analytes,
        "excluded_analytes": excluded_analytes,
        "requested_value": query_understanding.requested_value,
        "technical_condition": query_understanding.technical_condition,
        "intent": intent,
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

    if intent in {"global_patient_lookup", "cohort_search"}:
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
        technical_condition = str(query_understanding.technical_condition or "").strip().lower()
        if technical_condition in {"above_reference", "below_reference", "within_reference", "not_interpretable"}:
            rows = [r for r in rows if _status_code(r) == technical_condition]
        if excluded_analytes:
            rows = [r for r in rows if not _row_matches_excluded(r, excluded_analytes)]
        evidences = [_structured_record_from_row(r) for r in rows]
        pack["rows"] = rows
        pack["evidences"] = evidences
        return _finalize_structured_pack(pack, query_understanding)

    if not requested_doc_ids:
        return _finalize_structured_pack(pack, query_understanding)

    rows: list[dict[str, Any]] = []

    if intent == "multi_doc_comparison" and len(requested_doc_ids) >= 2:
        rows = _fetch_doc_lab_rows(
            sqlite_path=sqlite_path,
            requested_doc_ids=requested_doc_ids,
            analyte_norms=requested_analytes,
            limit=600,
        )
        left, right = requested_doc_ids[0], requested_doc_ids[1]
        left_rows = [r for r in rows if str(r.get("doc_id") or "").strip().lower() == left.lower()]
        right_rows = [r for r in rows if str(r.get("doc_id") or "").strip().lower() == right.lower()]
        if excluded_analytes:
            left_rows = [r for r in left_rows if not _row_matches_excluded(r, excluded_analytes)]
            right_rows = [r for r in right_rows if not _row_matches_excluded(r, excluded_analytes)]
        evidences: list[dict[str, Any]] = []
        missing: list[str] = []
        for analyte in requested_analytes:
            a = _best_row_for_analyte(left_rows, analyte)
            b = _best_row_for_analyte(right_rows, analyte)
            label = _canonical_display_name(analyte)
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
                        "current_value": f"{left}=non retrouvé | {right}=non retrouvé",
                        "unit": "",
                        "reference": "non disponible",
                        "previous_result": "",
                        "technical_status_code": "not_interpretable",
                        "technical_status": "différence technique",
                        "variation": "non comparable",
                        "source": "",
                    }
                )
                continue
            av = str((a or {}).get("value_raw") or "non retrouvé")
            bv = str((b or {}).get("value_raw") or "non retrouvé")
            unit = str((a or {}).get("unit") or (b or {}).get("unit") or "").strip()
            evidences.append(
                {
                    "doc_id": f"{left} vs {right}",
                    "page": None,
                    "row": None,
                    "chunk_id": None,
                    "analyte": label,
                    "analyte_norm": analyte,
                    "current_value": f"{left}={av} | {right}={bv}",
                    "unit": unit,
                    "reference": str((a or {}).get("reference_range") or (b or {}).get("reference_range") or "non disponible").strip(),
                    "previous_result": "",
                    "technical_status_code": "not_interpretable",
                    "technical_status": "différence technique",
                    "variation": _variation_label(bv, av) if a and b else "non comparable",
                    "source": "",
                }
            )
        pack["evidences"] = evidences
        pack["missing_items"] = missing
        pack["rows"] = rows
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
        rows = _fetch_doc_lab_rows(
            sqlite_path=sqlite_path,
            requested_doc_ids=requested_doc_ids,
            analyte_norms=["troponine"],
            include_text_search_terms=["troponine"],
            limit=250,
        )
        measured = [
            r
            for r in rows
            if ("troponine" in norm_text(str(r.get("analyte_norm") or "")) or "troponine" in norm_text(str(r.get("analyte") or "")))
            and norm_text(str(r.get("analyte") or "")) != "commentaire"
            and str(r.get("value_raw") or "").strip() != ""
        ]
        if measured:
            pack["evidences"] = [_structured_record_from_row(measured[0])]
        else:
            comment_rows = [r for r in rows if "troponine" in norm_text(str(r.get("value_raw") or "") + " " + str(r.get("text_for_keyword") or ""))]
            if comment_rows:
                pack["comment_text"] = str(comment_rows[0].get("value_raw") or "").strip()
        pack["rows"] = rows
        return _finalize_structured_pack(pack, query_understanding)

    if intent == "diagnostic_safety_question":
        safety_analytes = requested_analytes or ["ace", "psa_totale", "ca_15_3"]
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

    if intent in {"toxicology_summary", "doc_scoped_summary", "immunoanalysis_summary", "doc_scoped_results", "previous_result_comparison"}:
        analytes = requested_analytes if requested_analytes else None
        rows = _fetch_doc_lab_rows(
            sqlite_path=sqlite_path,
            requested_doc_ids=requested_doc_ids,
            analyte_norms=analytes,
            limit=700,
        )
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
        yes_no_mode = str(query_understanding.answer_style or "").strip().lower() == "yes_no" or str(query_understanding.output_format or "").strip().lower() == "yes_no"
        if requested_analytes:
            full_rows = list(rows)
            filtered_rows = list(rows)
            if out_of_reference_only:
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
                if out_of_reference_only and status not in {
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

    left, right = doc_ids[0], doc_ids[1]
    left_rows = [r for r in rows if str(r.get("doc_id") or "").lower() == left.lower()]
    right_rows = [r for r in rows if str(r.get("doc_id") or "").lower() == right.lower()]
    missing: list[str] = []
    lines: list[str] = []

    for analyte in requested_analytes:
        label = _canonical_display_name(analyte)
        a = _best_row_for_analyte(left_rows, analyte)
        b = _best_row_for_analyte(right_rows, analyte)
        if not a and not b:
            lines.append(f"- {label}: non retrouvé dans {left} ni {right}.")
            missing.append(analyte)
            continue
        if a and not b:
            lines.append(f"- {label}: présent uniquement dans {left} ({a.get('value_raw')} {a.get('unit') or ''}).")
            missing.append(analyte)
            continue
        if b and not a:
            lines.append(f"- {label}: présent uniquement dans {right} ({b.get('value_raw')} {b.get('unit') or ''}).")
            missing.append(analyte)
            continue

        a_val = str(a.get("value_raw") or "")
        b_val = str(b.get("value_raw") or "")
        unit = str(a.get("unit") or b.get("unit") or "").strip()
        ref = str(a.get("reference_range") or b.get("reference_range") or "non disponible")
        variation = _variation_label(b_val, a_val)
        lines.append(
            f"- {label}: {left}={a_val}{(' ' + unit) if unit else ''} | {right}={b_val}{(' ' + unit) if unit else ''} "
            f"| référence: {ref} | différence technique: {variation}"
        )
    return "\n".join(lines).strip(), missing


def _format_troponine_comment_answer(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return _missing_doc_answer()

    measured = [
        r
        for r in rows
        if ("troponine" in norm_text(str(r.get("analyte_norm") or "")) or "troponine" in norm_text(str(r.get("analyte") or "")))
        and norm_text(str(r.get("analyte") or "")) != "commentaire"
        and str(r.get("value_raw") or "").strip() != ""
    ]
    if measured:
        row = measured[0]
        unit = str(row.get("unit") or "").strip()
        ref = str(row.get("reference_range") or "non disponible")
        return (
            f"Une valeur mesurée de troponine est retrouvée: {row.get('value_raw')}"
            + (f" {unit}" if unit else "")
            + f" (référence: {ref})."
        )

    comment_rows = [r for r in rows if "troponine" in norm_text(str(r.get("value_raw") or ""))]
    if comment_rows:
        row = comment_rows[0]
        comment = str(row.get("value_raw") or "").strip()
        snippet = comment if len(comment) <= 220 else comment[:217] + "..."
        return (
            "Aucune valeur mesurée de troponine n’est retrouvée ; le document contient seulement un commentaire/interprétation "
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


def _build_response_transform_pack(
    *,
    query: str,
    query_understanding: QueryUnderstanding,
    previous_pack: dict[str, Any],
) -> dict[str, Any]:
    qn = norm_text(query)
    src = dict(previous_pack or {})
    evidences = [dict(ev) for ev in (src.get("evidences") or [])]

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
    provider: str = "ollama",
    model: str = "qwen3:4b",
    temperature: float = 0.0,
    num_ctx: int = 4096,
    max_tokens: int = 400,
    timeout: int = 120,
    index_dir: str | Path = "data/indexes",
    collection: str = "medical_chunks",
    search_engine: SearchEngine | None = None,
    llm_client: LLMClient | None = None,
    max_display_results: int = 3,
    show_all_results: bool = False,
    show_low_quality: bool = False,
    previous_structured_evidence_pack: dict[str, Any] | None = None,
    recent_style_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    request_id = str(uuid4())

    query_received = query
    q = normalize_query(query_received)
    query_used_for_retrieval = q
    query_used_for_prompt = q
    qn = norm_text(q)
    style_history = list(recent_style_history or [])
    query_understanding = parse_query_understanding(q)
    requested_doc_ids = list(query_understanding.requested_doc_ids)
    requested_doc_id = requested_doc_ids[0] if len(requested_doc_ids) == 1 else None
    sensitive_or_treatment = _query_is_sensitive_or_treatment(q)
    idx = Path(index_dir)
    sqlite_path = idx / "medical_rag.sqlite"
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

    # Follow-up transform priority: if user asks to reformat "this result" without new doc/analyte,
    # reuse previous evidence pack instead of small-talk/retrieval routing.
    if previous_structured_evidence_pack and _looks_like_transform_followup(q, query_understanding):
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

    if str(query_understanding.intent or "").strip().lower() in GENERAL_CONVERSATION_INTENTS:
        general_intent = str(query_understanding.intent or "small_talk").strip().lower()
        general_answer, general_err = generate_general_conversation_response(
            q,
            intent=general_intent,
            language=query_understanding.language,
            llm_client=llm_client,
            provider=provider,
            model=model,
            timeout=timeout,
        )
        if not general_answer:
            general_answer = GENERAL_CONVERSATION_FALLBACKS.get(general_intent, SMALL_TALK_FALLBACK_ANSWER)
        general_mode = "llm_small_talk" if general_intent == "small_talk" else "llm_general_conversation"
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
        elapsed = time.perf_counter() - started
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
            "llm_error": general_err,
            "error_type": "llm_general_conversation_error" if general_err else None,
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
                "generation_mode": general_mode,
                "generation_writer": "llm_writer" if not general_err else "professional_fallback",
                "intents": intents,
            },
        }
        return _inject_visualization_payload(
            result,
            query_understanding=query_understanding,
            displayed_evidences=[],
        )

    if query_understanding.intent == "response_transform":
        if not previous_structured_evidence_pack:
            elapsed = time.perf_counter() - started
            answer = f"Je n’ai pas de résultat précédent exploitable à reformater. {INSUFFICIENT_CONTEXT_SENTENCE}"
            if bool(getattr(query_understanding.presentation_intent, "user_requested_visualization", False)):
                presentation = getattr(query_understanding, "presentation_intent", None)
                visualization = build_visualization_payload(
                    requested_type=_normalize_requested_visualization_type(
                        getattr(presentation, "chart_type", None),
                        getattr(presentation, "raw_format_phrase", None),
                    ),
                    evidence_pack=[],
                    supported_visualizations=[k for k, cfg in VISUALIZATION_REGISTRY.items() if bool(cfg.get("supported"))],
                    raw_format_phrase=getattr(presentation, "raw_format_phrase", None),
                    source="previous_evidence_pack",
                )
                answer = _ensure_chart_explanation(answer, query_understanding, visualization)
            validation = validate_answer(
                query=q,
                answer_text=answer,
                evidence_pack=[],
                displayed_evidences=[],
                source_citations=[],
                generation_mode="deterministic_response_transform",
                retrieval_status="insufficient_context",
                query_received=query_received,
                query_used_for_retrieval=query_used_for_retrieval,
                query_used_for_prompt=query_used_for_prompt,
                query_stored=q,
                detected_analytes=exact_analytes,
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
                visualization_payload=_preview_visualization_payload(query_understanding, [])[0],
                chart_data_payload=_preview_visualization_payload(query_understanding, [])[1],
            )
            quality = _quality_report(
                answer=answer,
                validation=validation,
                source_clickable_requested=bool(query_understanding.source_clickable_requested),
                recent_style_history=style_history,
            )
            intro_text, conclusion_text = _extract_intro_conclusion(answer)
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
                    "answerability": {"status": "insufficient_context", "reason": "no_previous_response_context"},
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
                "query_received": query_received,
                "query_used_for_retrieval": query_used_for_retrieval,
                "query_used_for_prompt": query_used_for_prompt,
                "detected_analytes": exact_analytes,
                "requested_doc_ids": requested_doc_ids,
                "generation_mode": "deterministic_response_transform",
                "generation_writer": "professional_fallback",
                "intents": intents,
            },
                "exact_analyte_coverage": {
                    "detected_exact_analyte": exact_analyte,
                    "expected_exact_analyte_count": len(exact_analytes),
                    "retrieved_exact_analyte_count": 0,
                    "displayed_exact_analyte_count": 0,
                },
                },
                query_understanding=query_understanding,
                displayed_evidences=[],
            )

        transformed_pack = _build_response_transform_pack(
            query=q,
            query_understanding=query_understanding,
            previous_pack=previous_structured_evidence_pack,
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
                "value_raw": ev.get("current_value"),
                "reference_range": ev.get("reference"),
                "unit": ev.get("unit"),
                "previous_result": ev.get("previous_result"),
                "patient_token": ev.get("patient_token"),
                "interpretation_status": ev.get("technical_status_code"),
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
        output_format = str(transformed_pack.get("output_format") or query_understanding.output_format or "list").lower()
        composed = compose_professional_answer(
            user_question=q,
            query_understanding=transformed_qu,
            evidence_pack=transformed_pack,
            mode="auto",
            source_citations=source_citations,
            llm_client=llm_client,
            provider=provider,
            model=model,
            temperature=temperature,
            num_ctx=num_ctx,
            max_tokens=max_tokens,
            timeout=timeout,
        )
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
            "sources": source_citations,
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
                "generation_writer": "llm_writer" if generation_mode == "llm_professional_writer" else "professional_fallback",
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
        requested_doc_ids or query_understanding.intent in {"global_patient_lookup", "cohort_search"}
    ):
        structured_pack = build_structured_evidence_pack(
            query=q,
            query_understanding=query_understanding,
            sqlite_path=sqlite_path,
        )
        structured_rows = list(structured_pack.get("rows") or [])
        evidence_pack = _rows_to_evidence(structured_rows)
        if requested_doc_ids:
            allowed_docs = {d.lower() for d in requested_doc_ids}
            evidence_pack = [ev for ev in evidence_pack if str(ev.get("doc_id") or "").strip().lower() in allowed_docs]
        displayed_evidences = list(evidence_pack)
        citations = build_citations(displayed_evidences)
        source_citations = build_source_citations(displayed_evidences, resolver=source_resolver)
        structured_pack = _attach_source_fields_to_structured_pack(structured_pack, source_citations)
        structured_pack["recent_style_history"] = style_history[-20:]
        structured_pack = _attach_visualization_facts_to_evidence_pack(
            query_understanding=query_understanding,
            evidence_pack=structured_pack,
            displayed_evidences=displayed_evidences,
        )
        query_understanding = _with_resolved_strategy(query_understanding, structured_pack)
        composed = compose_professional_answer(
            user_question=q,
            query_understanding=query_understanding,
            evidence_pack=structured_pack,
            mode="auto",
            source_citations=source_citations,
            llm_client=llm_client,
            provider=provider,
            model=model,
            temperature=temperature,
            num_ctx=num_ctx,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        final_answer = str(composed.get("answer") or "").strip() or _missing_doc_answer()
        generation_mode = str(composed.get("mode") or "deterministic_professional_fallback")
        writer_error = str(composed.get("llm_error") or "") or None
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

        missing_requested_doc_ids = _resolve_missing_requested_doc_ids(sqlite_path, requested_doc_ids)
        found_requested_analytes = []
        for analyte in exact_analytes:
            if any(_row_matches_analyte(row, analyte) for row in structured_rows):
                found_requested_analytes.append(analyte)
                continue
            if analyte == "troponine" and str(structured_pack.get("comment_text") or "").strip():
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
        quality = _quality_report(
            answer=final_answer,
            validation=validation,
            source_clickable_requested=bool(query_understanding.source_clickable_requested),
            recent_style_history=style_history,
        )

        if _should_retry_with_validator(validation, generation_mode):
            retry_feedback = _build_validator_retry_feedback(validation)
            retry_composed = compose_professional_answer(
                user_question=q,
                query_understanding=query_understanding,
                evidence_pack=structured_pack,
                mode="auto",
                source_citations=source_citations,
                llm_client=llm_client,
                provider=provider,
                model=model,
                temperature=temperature,
                num_ctx=num_ctx,
                max_tokens=max_tokens,
                timeout=timeout,
                retry_feedback=retry_feedback,
            )
            retry_answer = str(retry_composed.get("answer") or "").strip()
            retry_mode = str(retry_composed.get("mode") or generation_mode)
            retry_writer_error = str(retry_composed.get("llm_error") or "") or None
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
                    generation_mode = str(fallback_composed.get("mode") or "deterministic_professional_fallback")
                    writer_error = retry_writer_error or writer_error
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

        elapsed = time.perf_counter() - started
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
                "generation_writer": "llm_writer" if generation_mode == "llm_professional_writer" else "professional_fallback",
                "intents": intents,
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
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--timeout", type=int, default=120)
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
