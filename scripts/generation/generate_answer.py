#!/usr/bin/env python3
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

from retrieval.models import RetrievalFilters, RetrievalResult
from retrieval.search import SearchEngine

from answer_validator import validate_answer
from citation_builder import append_citations, build_citations
from evidence_builder import build_evidence_pack
from llm_client import LLMClient, LLMClientError
from prompt_builder import INSUFFICIENT_CONTEXT_SENTENCE, build_prompt
from query_understanding import (
    contains_exact_term,
    detect_doc_summary_intent,
    detect_exact_analyte,
    detect_exact_analytes,
    norm_text,
)


_ANALYTE_CACHE_BY_SQLITE: dict[str, list[str]] = {}
_DOC_ANALYTE_CACHE: dict[tuple[str, str], list[dict[str, str]]] = {}

_ANALYTE_ALIASES: dict[str, list[str]] = {
    "hdl": ["hdl", "cholesterol hdl", "cholestérol hdl", "cholesterol_hdl", "cholesterol_hdl_direct"],
    "cholesterol_hdl": ["hdl", "cholesterol hdl", "cholestérol hdl", "cholesterol_hdl", "cholestérol-hdl"],
    "c3": ["c3", "complement c3", "complément c3", "complement_c3"],
    "c4": ["c4", "complement c4", "complément c4", "complement_c4"],
    "troponine": ["troponine", "troponin", "troponine i", "troponine t"],
}


def normalize_query(query: str) -> str:
    q = re.sub(r"\s+", " ", (query or "").strip())
    return q


def sanitize_model_answer(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return raw

    # Remove explicit hidden-thought markers if the model emits them.
    raw = re.sub(r"(?is)<think>.*?</think>", "", raw).strip()
    raw = re.sub(r"(?im)^thinking\\s*:\\s*", "", raw).strip()

    # Keep only the expected final format when model emits preamble.
    match = re.search(r"(?im)^réponse\\s*:\\s*", raw)
    if not match:
        match = re.search(r"(?im)^reponse\\s*:\\s*", raw)
    if match:
        return raw[match.start() :].strip()
    return raw


def extract_requested_doc_id(query: str) -> str | None:
    raw = (query or "").strip()
    if not raw:
        return None

    lowered = raw.lower()
    patterns = [
        r"\b(?:report|rapport)\s*[_\-]?\s*\(?\s*(\d{1,6})\s*\)?(?:\s*\.pdf)?\b",
        r"\b(?:report|rapport)\s*\(\s*(\d{1,6})\s*\)(?:\s*\.pdf)?\b",
    ]
    for patt in patterns:
        m = re.search(patt, lowered, flags=re.IGNORECASE)
        if not m:
            continue
        try:
            n = int(m.group(1))
        except Exception:
            continue
        if n < 0:
            continue
        return f"report_{n}"
    return None


def _load_index_analyte_norms(sqlite_path: Path) -> list[str]:
    key = str(sqlite_path.resolve())
    cached = _ANALYTE_CACHE_BY_SQLITE.get(key)
    if cached is not None:
        return cached
    if not sqlite_path.exists():
        _ANALYTE_CACHE_BY_SQLITE[key] = []
        return []

    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT DISTINCT lower(analyte_norm) AS analyte_norm
            FROM metadata_chunks
            WHERE analyte_norm IS NOT NULL AND trim(analyte_norm) <> ''
            """
        )
        rows = [str(r["analyte_norm"]).strip() for r in cur.fetchall() if str(r["analyte_norm"]).strip()]
    finally:
        conn.close()
    _ANALYTE_CACHE_BY_SQLITE[key] = rows
    return rows


def _detect_exact_analytes_with_index(query: str, sqlite_path: Path) -> list[str]:
    detected = detect_exact_analytes(query)
    seen = {norm_text(a) for a in detected}
    qn = norm_text(query)
    for analyte_norm in _load_index_analyte_norms(sqlite_path):
        if contains_exact_term(qn, analyte_norm):
            if norm_text(analyte_norm) not in seen:
                detected.append(analyte_norm)
                seen.add(norm_text(analyte_norm))
    return detected


def _load_doc_analytes(sqlite_path: Path, doc_id: str) -> list[dict[str, str]]:
    key = (str(sqlite_path.resolve()), str(doc_id).strip().lower())
    cached = _DOC_ANALYTE_CACHE.get(key)
    if cached is not None:
        return cached
    if not sqlite_path.exists() or not doc_id:
        _DOC_ANALYTE_CACHE[key] = []
        return []

    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT DISTINCT
              lower(trim(COALESCE(analyte_norm, ''))) AS analyte_norm,
              trim(COALESCE(analyte, '')) AS analyte
            FROM metadata_chunks
            WHERE lower(doc_id) = lower(?)
              AND analyte_norm IS NOT NULL
              AND trim(analyte_norm) <> ''
            ORDER BY analyte_norm
            """,
            (doc_id,),
        )
        rows = [
            {"analyte_norm": str(r["analyte_norm"]).strip(), "analyte": str(r["analyte"]).strip()}
            for r in cur.fetchall()
            if str(r["analyte_norm"]).strip()
        ]
    finally:
        conn.close()
    _DOC_ANALYTE_CACHE[key] = rows
    return rows


def _alias_detected_in_query(query_norm: str, alias: str, canonical: str) -> bool:
    alias_norm = norm_text(alias)
    if not alias_norm:
        return False
    if canonical == "c3":
        return re.search(r"(?<![a-z0-9])c3(?![a-z0-9])", query_norm, flags=re.IGNORECASE) is not None
    if canonical == "c4":
        return re.search(r"(?<![a-z0-9])c4(?![a-z0-9])", query_norm, flags=re.IGNORECASE) is not None
    if len(alias_norm) <= 3:
        return re.search(rf"(?<![a-z0-9]){re.escape(alias_norm)}(?![a-z0-9])", query_norm, flags=re.IGNORECASE) is not None
    return contains_exact_term(query_norm, alias_norm)


def _resolve_requested_analytes_for_doc(
    *,
    query: str,
    sqlite_path: Path,
    requested_doc_id: str,
) -> tuple[list[str], list[str], list[str], dict[str, str], dict[str, str]]:
    query_norm = norm_text(query)
    doc_analytes = _load_doc_analytes(sqlite_path, requested_doc_id)
    doc_norms = [d["analyte_norm"] for d in doc_analytes]
    doc_norm_set = set(doc_norms)
    display_by_norm = {d["analyte_norm"]: (d["analyte"] or d["analyte_norm"]) for d in doc_analytes}

    requested_terms_ordered: list[str] = []
    seen_requested: set[str] = set()

    for canonical, aliases in _ANALYTE_ALIASES.items():
        if any(_alias_detected_in_query(query_norm, alias, canonical) for alias in aliases):
            if canonical not in seen_requested:
                requested_terms_ordered.append(canonical)
                seen_requested.add(canonical)

    for d in doc_analytes:
        dn = d["analyte_norm"]
        if contains_exact_term(query_norm, dn) and dn not in seen_requested:
            requested_terms_ordered.append(dn)
            seen_requested.add(dn)
        analyte_label = norm_text(d.get("analyte") or "")
        if analyte_label and contains_exact_term(query_norm, analyte_label) and dn not in seen_requested:
            requested_terms_ordered.append(dn)
            seen_requested.add(dn)

    for detected in _detect_exact_analytes_with_index(query, sqlite_path):
        if detected not in seen_requested:
            requested_terms_ordered.append(detected)
            seen_requested.add(detected)

    resolved_found: list[str] = []
    missing_requested: list[str] = []
    requested_to_resolved: dict[str, str] = {}

    def _resolve_term(term: str) -> str | None:
        t = norm_text(term)
        if t in doc_norm_set:
            return t
        if t == "hdl":
            for candidate in doc_norms:
                if "hdl" in candidate:
                    return candidate
            return None
        if t == "cholesterol hdl":
            for candidate in doc_norms:
                if "cholesterol_hdl" == candidate or "hdl" in candidate:
                    return candidate
            return None
        if t in {"c3", "c4"} and t in doc_norm_set:
            return t
        for candidate in doc_norms:
            candidate_label = norm_text(display_by_norm.get(candidate, ""))
            if contains_exact_term(candidate, t) or contains_exact_term(candidate_label, t):
                return candidate
        return None

    for requested in requested_terms_ordered:
        resolved = _resolve_term(requested)
        if resolved:
            requested_to_resolved[requested] = resolved
            if resolved not in resolved_found:
                resolved_found.append(resolved)
        else:
            missing_requested.append(requested)
    dedup_requested: list[str] = []
    seen_keys: set[str] = set()
    dedup_mapping: dict[str, str] = {}
    for requested in requested_terms_ordered:
        resolved = requested_to_resolved.get(requested)
        dedup_key = resolved or requested
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)
        dedup_requested.append(requested)
        if resolved:
            dedup_mapping[requested] = resolved

    dedup_missing = [r for r in missing_requested if r in dedup_requested]
    return dedup_requested, resolved_found, dedup_missing, display_by_norm, dedup_mapping


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

    qn = normalize_query(query).lower()
    candidates = evidence_pack

    if exact_analyte:
        exact_candidates = [
            e
            for e in evidence_pack
            if contains_exact_term(str(e.get("analyte_norm") or ""), exact_analyte)
            or contains_exact_term(str(e.get("analyte") or ""), exact_analyte)
        ]
        if exact_candidates:
            candidates = exact_candidates

    if any(k in qn for k in ["supérieur", "superieur", "above_reference"]):
        filtered = [e for e in evidence_pack if str(e.get("interpretation_status") or "").lower() == "above_reference"]
        if filtered:
            candidates = filtered
    elif any(k in qn for k in ["résultat antérieur", "resultat anterieur", "previous result", "ancien résultat", "ancien resultat"]):
        filtered = [
            e
            for e in evidence_pack
            if int(e.get("previous_result_present") or 0) == 1 and str(e.get("previous_result") or "").strip() != ""
        ]
        if filtered:
            candidates = filtered
    elif any(k in qn for k in ["parasite", "parasitologie", "trichuris", "ankylostoma"]):
        filtered = [
            e
            for e in evidence_pack
            if any(
                x in str(
                    (e.get("analyte") or "")
                    + " "
                    + (e.get("parameter") or "")
                    + " "
                    + (e.get("text_excerpt") or "")
                ).lower()
                for x in ["trichuris", "parasite", "ankylostoma"]
            )
        ]
        if filtered:
            candidates = filtered

    lines = ["Réponse :"]
    if len(candidates) > 1 and exact_analyte:
        lines.append(f"Plusieurs résultats de {exact_analyte.upper()} ont été retrouvés :")

    max_items = min(len(candidates), 10 if exact_analyte else 3)
    for idx, ev in enumerate(candidates[:max_items], start=1):
        analyte = ev.get("analyte_display") or ev.get("analyte") or ev.get("parameter") or "non précisé"
        value = ev.get("value_raw")
        unit = ev.get("unit")
        ref = ev.get("reference_range")
        interp = ev.get("interpretation_status")
        prev = ev.get("previous_result")

        prefix = f"{idx}. " if len(candidates) > 1 else "- "
        part = f"{prefix}{analyte}"
        if value not in (None, ""):
            part += f" = {value}"
        if unit not in (None, ""):
            part += f" {unit}"
        details = []
        if ref not in (None, ""):
            details.append(f"référence: {ref}")
        if interp not in (None, ""):
            details.append(f"interprétation technique: {interp}")
        if prev not in (None, ""):
            details.append(f"résultat antérieur: {prev}")
        if details:
            part += " (" + " ; ".join(details) + ")"
        lines.append(part)

    lines.extend(["", "Données utilisées :"])
    for idx, ev in enumerate(candidates[:max_items], start=1):
        lines.extend(
            [
                f"Résultat {idx} :",
                f"- Analyte : {ev.get('analyte_display') or ev.get('analyte') or ev.get('parameter') or 'non précisé'}",
                f"- Valeur : {ev.get('value_raw') or 'non disponible'}",
                f"- Unité : {ev.get('unit') or 'non disponible'}",
                f"- Référence : {ev.get('reference_range') or 'non disponible'}",
                f"- Interprétation technique : {ev.get('interpretation_status') or 'non disponible'}",
                f"- Résultat antérieur : {ev.get('previous_result') or 'non disponible'}",
                (
                    f"- Source : [doc_id={ev.get('doc_id')}, page={ev.get('page_number')}, "
                    f"row={ev.get('row_index')}, chunk_id={ev.get('chunk_id')}]"
                ),
            ]
        )
    return "\n".join(lines).strip()


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
    return any(k in qn for k in ["resultat anterieur", "previous result", "ancien resultat", "antérieur"])


def _is_compare_query(qn: str) -> bool:
    return ("compare" in qn or "compar" in qn) and ("actuel" in qn and ("anterieur" in qn or "previous" in qn))


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


def _select_displayed_evidences(
    *,
    query_norm: str,
    evidence_pack: list[dict[str, Any]],
    exact_analyte: str | None,
    max_display_results: int,
    show_all_results: bool,
    show_low_quality: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = _select_deterministic_candidates(
        query_norm=query_norm,
        evidence_pack=evidence_pack,
        exact_analyte=exact_analyte,
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
) -> list[dict[str, Any]]:
    candidates = list(evidence_pack)
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


def _yes_no_opening_sentence(query_norm: str, candidates: list[dict[str, Any]], exact_analyte: str | None) -> str | None:
    if not candidates:
        return None

    analyte_label = (candidates[0].get("analyte_display") or candidates[0].get("analyte") or exact_analyte or "cet analyte").strip()
    analyte_upper = analyte_label.upper()
    statuses = [str(ev.get("interpretation_status") or "").lower() for ev in candidates]
    count_above = sum(1 for s in statuses if s == "above_reference")
    count_within = sum(1 for s in statuses if s == "within_reference")
    count_below = sum(1 for s in statuses if s == "below_reference")

    if "est il au dessus" in query_norm or "est elle superieure" in query_norm:
        if count_above > 0:
            return f"Oui, {analyte_upper} affiché est au-dessus de la référence technique extraite."
        return f"Non, {analyte_upper} affiché n'est pas au-dessus de la référence technique extraite."

    if "est elle inferieure" in query_norm:
        if count_below > 0:
            return f"Oui, {analyte_upper} affiché se situe en dessous de la référence technique extraite."
        return f"Non, {analyte_upper} affiché ne se situe pas en dessous de la référence technique extraite."

    if "normale ou superieure" in query_norm:
        if count_above > 0 and count_within > 0:
            return f"Certains résultats de {analyte_upper} sont supérieurs à la référence, tandis qu'au moins un résultat est dans la référence."
        if count_above > 0 and count_within == 0 and count_below == 0:
            return f"Oui, les résultats de {analyte_upper} affichés sont supérieurs à la référence technique extraite."
        if count_within > 0 and count_above == 0 and count_below == 0:
            return f"Les résultats de {analyte_upper} affichés sont dans la référence technique extraite."
        if count_below > 0 and count_above == 0:
            return f"Non, les résultats de {analyte_upper} affichés sont inférieurs à la référence technique extraite."
        return f"Les résultats de {analyte_upper} affichés sont mixtes par rapport à la référence technique extraite."

    return None


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

    lines: list[str] = ["Réponse :"]
    opening = _yes_no_opening_sentence(qn, candidates, exact_analyte)
    if opening:
        lines.append(opening)
    if _is_compare_query(qn):
        lines.append("Comparaison technique des résultats actuels et antérieurs :")
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
        if len(candidates) > 1:
            title = f"Plusieurs résultats de {exact_analyte.upper()} ont été retrouvés :" if exact_analyte else "Plusieurs résultats ont été retrouvés :"
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
    lines.append("Données utilisées :")
    for idx, ev in enumerate(candidates, start=1):
        lines.extend(
            [
                f"Résultat {idx} :",
                f"- Analyte : {ev.get('analyte_display') or ev.get('analyte') or ev.get('parameter') or 'non précisé'}",
                f"- Valeur : {ev.get('value_raw') or 'non disponible'}",
                f"- Unité : {ev.get('unit') or 'non disponible'}",
                f"- Référence : {ev.get('reference_range') or 'non disponible'}",
                f"- Interprétation technique : {ev.get('interpretation_status') or 'non disponible'}",
                f"- Résultat antérieur : {ev.get('previous_result') or 'non disponible'}",
                (
                    f"- Source : [doc_id={ev.get('doc_id')}, page={ev.get('page_number')}, "
                    f"row={ev.get('row_index')}, chunk_id={ev.get('chunk_id')}]"
                ),
            ]
        )
    return "\n".join(lines).strip()


def _requested_label(term: str, display_by_norm: dict[str, str]) -> str:
    t = str(term or "").strip()
    if not t:
        return "Analyte non précisé"
    if t in display_by_norm:
        return str(display_by_norm[t] or t)
    if t == "hdl":
        return "Cholestérol HDL"
    if t == "cholesterol_hdl":
        return "Cholestérol HDL"
    if t == "c3":
        return "C3"
    if t == "c4":
        return "C4"
    if t == "troponine":
        return "Troponine"
    return t.replace("_", " ").upper()


def _build_doc_multi_analyte_answer(
    *,
    requested_doc_id: str,
    requested_analytes: list[str],
    found_requested_analytes: list[str],
    missing_requested_analytes: list[str],
    requested_to_resolved: dict[str, str],
    displayed_evidences: list[dict[str, Any]],
    display_by_norm: dict[str, str],
) -> str:
    by_analyte: dict[str, list[dict[str, Any]]] = {}
    for ev in displayed_evidences:
        k = str(ev.get("analyte_norm") or "").strip().lower()
        if not k:
            continue
        by_analyte.setdefault(k, []).append(ev)

    lines: list[str] = ["Réponse :"]
    for requested in requested_analytes:
        label = _requested_label(requested, display_by_norm)
        analyte_norm = requested_to_resolved.get(requested) or (requested if requested in by_analyte else None)
        rows = by_analyte.get(analyte_norm or "", [])
        if rows:
            for ev in rows:
                value = ev.get("value_raw") or "non disponible"
                unit = ev.get("unit") or ""
                ref = ev.get("reference_range") or "non disponible"
                interp = ev.get("interpretation_status") or "non disponible"
                part = f"- {label} = {value}"
                if unit:
                    part += f" {unit}"
                part += f" (référence: {ref} ; interprétation technique: {interp})"
                lines.append(part)
        else:
            lines.append(f"- {label} : information insuffisante dans le contexte fourni pour {requested_doc_id}.")

    if missing_requested_analytes and len(requested_analytes) == 0:
        lines.append(f"Information insuffisante dans le contexte fourni pour {requested_doc_id}.")

    lines.append("")
    lines.append("Données utilisées :")
    idx = 1
    for requested in requested_analytes:
        analyte_norm = requested_to_resolved.get(requested) or (requested if requested in by_analyte else None)
        rows = by_analyte.get(analyte_norm or "", [])
        for ev in rows:
            lines.extend(
                [
                    f"Résultat {idx} :",
                    f"- Analyte : {ev.get('analyte_display') or ev.get('analyte') or ev.get('parameter') or _requested_label(requested, display_by_norm)}",
                    f"- Valeur : {ev.get('value_raw') or 'non disponible'}",
                    f"- Unité : {ev.get('unit') or 'non disponible'}",
                    f"- Référence : {ev.get('reference_range') or 'non disponible'}",
                    f"- Interprétation technique : {ev.get('interpretation_status') or 'non disponible'}",
                    f"- Résultat antérieur : {ev.get('previous_result') or 'non disponible'}",
                    (
                        f"- Source : [doc_id={ev.get('doc_id')}, page={ev.get('page_number')}, "
                        f"row={ev.get('row_index')}, chunk_id={ev.get('chunk_id')}]"
                    ),
                ]
            )
            idx += 1
    if idx == 1:
        lines.append(f"- Aucune evidence disponible pour {requested_doc_id}.")
    return "\n".join(lines).strip()


def _should_use_deterministic_generation(query: str, evidence_pack: list[dict[str, Any]], exact_analyte: str | None) -> bool:
    if not evidence_pack:
        return False
    qn = norm_text(query)
    if exact_analyte:
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
) -> tuple[int, list[dict[str, Any]]]:
    if not sqlite_path.exists():
        return 0, []

    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        analyte_match_sql = """
            (
              lower(m.analyte_norm) = lower(?)
              OR lower(m.parameter_norm) = lower(?)
              OR (
                ' ' || lower(replace(replace(replace(replace(replace(COALESCE(m.analyte_norm, ''), '_', ' '), '-', ' '), '(', ' '), ')', ' '), '/', ' ')) || ' '
              ) LIKE ?
              OR (
                ' ' || lower(replace(replace(replace(replace(replace(COALESCE(m.analyte, ''), '_', ' '), '-', ' '), '(', ' '), ')', ' '), '/', ' ')) || ' '
              ) LIKE ?
              OR (
                ' ' || lower(replace(replace(replace(replace(replace(COALESCE(m.parameter_norm, ''), '_', ' '), '-', ' '), '(', ' '), ')', ' '), '/', ' ')) || ' '
              ) LIKE ?
              OR (
                ' ' || lower(replace(replace(replace(replace(replace(COALESCE(m.parameter, ''), '_', ' '), '-', ' '), '(', ' '), ')', ' '), '/', ' ')) || ' '
              ) LIKE ?
            )
        """
        analyte_like = f"% {str(analyte_norm or '').strip().lower()} %"
        cur.execute(
            """
            SELECT COUNT(*) AS c
            FROM metadata_chunks m
            WHERE
            """
            + analyte_match_sql,
            (analyte_norm, analyte_norm, analyte_like, analyte_like, analyte_like, analyte_like),
        )
        total = int((cur.fetchone() or {"c": 0})["c"])
        cur.execute(
            """
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
            WHERE
            """
            + analyte_match_sql
            + """
            ORDER BY
              CAST(REPLACE(lower(c.doc_id), 'report_', '') AS INTEGER) DESC,
              COALESCE(m.page_number, o.page_number, 999999) ASC,
              COALESCE(m.row_index, 999999) ASC,
              c.chunk_id ASC
            LIMIT ?
            """,
            (
                analyte_norm,
                analyte_norm,
                analyte_like,
                analyte_like,
                analyte_like,
                analyte_like,
                int(limit),
            ),
        )
        rows = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()

    return total, rows


def _row_to_retrieval_result_fast(row: dict[str, Any], reason: str) -> RetrievalResult:
    text = str(row.get("text_for_embedding") or row.get("text_for_keyword") or "").strip()
    text_preview = text if len(text) <= 260 else text[:257] + "..."
    md = dict(row)
    return RetrievalResult(
        chunk_id=str(row.get("chunk_id") or ""),
        doc_id=str(row.get("doc_id") or ""),
        chunk_type=str(row.get("chunk_type") or ""),
        document_type=row.get("document_type"),
        source_pdf=row.get("source_pdf"),
        page_number=int(row.get("page_number")) if row.get("page_number") not in (None, "") else None,
        text=text,
        text_preview=text_preview,
        metadata=md,
        score_keyword=None,
        score_vector=None,
        score_hybrid=None,
        rrf_score=None,
        clinical_rerank_score=None,
        final_score=None,
        retrieval_mode="keyword",
        match_reason=[reason],
    )


def _load_doc_analyte_rows(
    *,
    sqlite_path: Path,
    doc_id: str,
    analyte_norms: list[str],
) -> list[dict[str, Any]]:
    if not sqlite_path.exists() or not doc_id or not analyte_norms:
        return []

    norm_terms = [str(a).strip().lower() for a in analyte_norms if str(a).strip()]
    if not norm_terms:
        return []

    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        analyte_clauses: list[str] = []
        params: list[Any] = [doc_id]
        for term in norm_terms:
            like_term = f"% {term} %"
            analyte_clauses.append(
                """
                (
                  lower(m.analyte_norm) = lower(?)
                  OR lower(m.parameter_norm) = lower(?)
                  OR (
                    ' ' || lower(replace(replace(replace(replace(replace(COALESCE(m.analyte_norm, ''), '_', ' '), '-', ' '), '(', ' '), ')', ' '), '/', ' ')) || ' '
                  ) LIKE ?
                  OR (
                    ' ' || lower(replace(replace(replace(replace(replace(COALESCE(m.analyte, ''), '_', ' '), '-', ' '), '(', ' '), ')', ' '), '/', ' ')) || ' '
                  ) LIKE ?
                  OR (
                    ' ' || lower(replace(replace(replace(replace(replace(COALESCE(m.parameter_norm, ''), '_', ' '), '-', ' '), '(', ' '), ')', ' '), '/', ' ')) || ' '
                  ) LIKE ?
                  OR (
                    ' ' || lower(replace(replace(replace(replace(replace(COALESCE(m.parameter, ''), '_', ' '), '-', ' '), '(', ' '), ')', ' '), '/', ' ')) || ' '
                  ) LIKE ?
                )
                """
            )
            params.extend([term, term, like_term, like_term, like_term, like_term])
        analyte_where = " OR ".join(analyte_clauses)
        cur.execute(
            """
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
            WHERE lower(c.doc_id) = lower(?)
              AND (
            """
            + analyte_where
            + """
              )
            ORDER BY COALESCE(m.page_number, o.page_number, 999999) ASC,
                     COALESCE(m.row_index, 999999) ASC,
                     c.chunk_id ASC
            """,
            params,
        )
        rows = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()
    return rows


def _insufficient_context_for_doc(doc_id: str) -> str:
    return f"Information insuffisante dans le contexte fourni pour {doc_id}."


def _load_doc_summary_rows(
    *,
    sqlite_path: Path,
    doc_id: str,
    immunoanalyse_only: bool,
    above_only: bool,
    below_only: bool,
) -> tuple[list[dict[str, Any]], bool]:
    if not sqlite_path.exists() or not doc_id:
        return [], False

    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        where_clauses: list[str] = [
            "lower(c.doc_id) = lower(?)",
            "c.chunk_type IN ('lab_result', 'clinical_result')",
        ]
        params: list[Any] = [doc_id]
        if above_only:
            where_clauses.append("lower(COALESCE(m.interpretation_status, '')) = 'above_reference'")
        if below_only:
            where_clauses.append("lower(COALESCE(m.interpretation_status, '')) = 'below_reference'")

        immuno_clause = """
            (
              lower(COALESCE(m.section_norm, '')) LIKE '%immunoanalyse%'
              OR lower(COALESCE(m.section, '')) LIKE '%immunoanalyse%'
              OR lower(COALESCE(m.section_norm, '')) LIKE '%immuno analyse%'
              OR lower(COALESCE(m.section, '')) LIKE '%immuno analyse%'
            )
        """
        base_where = " AND ".join(where_clauses)
        rows: list[dict[str, Any]] = []
        section_filter_applied = False
        if immunoanalyse_only:
            cur.execute(
                """
                SELECT COUNT(*) AS c
                FROM metadata_chunks m
                JOIN chunks c ON c.chunk_id = m.chunk_id
                WHERE lower(c.doc_id) = lower(?)
                  AND c.chunk_type IN ('lab_result', 'clinical_result')
                  AND """
                + immuno_clause,
                (doc_id,),
            )
            immuno_count = int((cur.fetchone() or {"c": 0})["c"])
            if immuno_count > 0:
                section_filter_applied = True
                where_sql = f"{base_where} AND {immuno_clause}"
            else:
                where_sql = base_where
        else:
            where_sql = base_where

        cur.execute(
            """
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
            WHERE """
            + where_sql
            + """
            ORDER BY
              CASE lower(COALESCE(m.interpretation_status, ''))
                WHEN 'above_reference' THEN 1
                WHEN 'below_reference' THEN 2
                WHEN 'needs_clinical_context' THEN 3
                WHEN 'within_reference' THEN 4
                ELSE 5
              END,
              COALESCE(m.page_number, o.page_number, 999999) ASC,
              COALESCE(m.row_index, 999999) ASC,
              c.chunk_id ASC
            """,
            params,
        )
        rows = [dict(r) for r in cur.fetchall()]
        return rows, section_filter_applied
    finally:
        conn.close()


def _build_doc_summary_answer(
    *,
    requested_doc_id: str,
    displayed_evidences: list[dict[str, Any]],
    include_within_reference: bool,
    within_reference_hidden_count: int,
    section_filter_applied: bool,
    asked_immunoanalyse: bool,
    show_within_hidden_notice: bool = True,
    needs_context_heading: str = "Résultats nécessitant un contexte clinique :",
    include_technical_disclaimer: bool = False,
) -> str:
    if not displayed_evidences:
        return _insufficient_context_for_doc(requested_doc_id)

    def _row_line(ev: dict[str, Any]) -> str:
        analyte = ev.get("analyte_display") or ev.get("analyte") or ev.get("parameter") or "analyte non précisé"
        value = ev.get("value_raw") or "non disponible"
        unit = ev.get("unit") or ""
        ref = ev.get("reference_range") or "non disponible"
        part = f"- {analyte} = {value}"
        if unit:
            part += f" {unit}"
        part += f" ; référence: {ref} ; interprétation technique: {ev.get('interpretation_status') or 'non disponible'}"
        return part

    grouped: dict[str, list[dict[str, Any]]] = {
        "above_reference": [],
        "below_reference": [],
        "needs_clinical_context": [],
        "within_reference": [],
        "other": [],
    }
    for ev in displayed_evidences:
        status = str(ev.get("interpretation_status") or "").strip().lower()
        if status not in grouped:
            status = "other"
        grouped[status].append(ev)

    lines: list[str] = ["Réponse :", f"Dans {requested_doc_id}, les résultats importants retrouvés sont :"]
    if asked_immunoanalyse:
        if section_filter_applied:
            lines.append("Section utilisée : IMMUNOANALYSE.")
        else:
            lines.append(
                f"Les résultats ci-dessous proviennent des résultats biologiques disponibles pour {requested_doc_id}. "
                "La section exacte n’étant pas explicitement indexée, le résumé est basé sur les résultats biologiques du document."
            )

    if grouped["above_reference"]:
        lines.append("")
        lines.append("Résultats au-dessus de la référence technique :")
        lines.extend(_row_line(ev) for ev in grouped["above_reference"])
    if grouped["below_reference"]:
        lines.append("")
        lines.append("Résultats inférieurs à la référence technique :")
        lines.extend(_row_line(ev) for ev in grouped["below_reference"])
    if grouped["needs_clinical_context"]:
        lines.append("")
        lines.append(needs_context_heading)
        lines.extend(_row_line(ev) for ev in grouped["needs_clinical_context"])
    if include_within_reference and grouped["within_reference"]:
        lines.append("")
        lines.append("Résultats dans la référence technique :")
        lines.extend(_row_line(ev) for ev in grouped["within_reference"])
    elif show_within_hidden_notice and within_reference_hidden_count > 0:
        lines.append("")
        lines.append("Résultats dans la référence :")
        lines.append(
            f"- Masqués par défaut pour garder le résumé court ({within_reference_hidden_count} résultat(s)). "
            "Utilisez --include-within-reference pour les afficher."
        )
    if grouped["other"]:
        lines.append("")
        lines.append("Résultats avec statut technique non standard :")
        lines.extend(_row_line(ev) for ev in grouped["other"])

    lines.append("")
    lines.append("Données utilisées :")
    for idx, ev in enumerate(displayed_evidences, start=1):
        lines.extend(
            [
                f"Résultat {idx} :",
                f"- Analyte : {ev.get('analyte_display') or ev.get('analyte') or ev.get('parameter') or 'non précisé'}",
                f"- Valeur : {ev.get('value_raw') or 'non disponible'}",
                f"- Unité : {ev.get('unit') or 'non disponible'}",
                f"- Référence : {ev.get('reference_range') or 'non disponible'}",
                f"- Interprétation technique : {ev.get('interpretation_status') or 'non disponible'}",
                f"- Résultat antérieur : {ev.get('previous_result') or 'non disponible'}",
                (
                    f"- Source : [doc_id={ev.get('doc_id')}, page={ev.get('page_number')}, "
                    f"row={ev.get('row_index')}, chunk_id={ev.get('chunk_id')}]"
                ),
            ]
        )
    if include_technical_disclaimer:
        lines.append("")
        lines.append("Ces statuts sont des interprétations techniques extraites, pas un diagnostic médical.")
    return "\n".join(lines).strip()


def _keyword_only_search(
    *,
    sqlite_path: Path,
    query: str,
    top_k: int,
    filters: RetrievalFilters,
) -> Any:
    from retrieval.context_builder import ContextBuilder
    from retrieval.filters import MappingResolver
    from retrieval.keyword_search import KeywordSearcher
    from retrieval.sqlite_store import SQLiteStore

    sqlite_store = SQLiteStore(sqlite_path)
    try:
        keyword_searcher = KeywordSearcher(sqlite_store, MappingResolver())
        top_results = keyword_searcher.search(query, top_k=top_k, filters=filters)
        if not top_results:
            return SimpleNamespace(
                answerability={"status": "insufficient_context", "reason": "no_results_keyword_fallback"},
                filters=filters.to_dict(),
                top_results=[],
                context_chunks=[],
                sources=[],
            )

        ctx_builder = ContextBuilder(sqlite_store)
        response = ctx_builder.build(
            query=query,
            mode="keyword",
            top_results=top_results,
            filters=filters,
            max_context_chunks=max(top_k, 8),
            strict_context=True,
            debug_context=False,
        )
        response.filters = filters.to_dict()
        return response
    finally:
        sqlite_store.close()


def _count_displayed_exact_analyte(answer: str, analyte: str) -> int:
    text = norm_text(answer or "")
    a = norm_text(analyte or "")
    if not text or not a:
        return 0
    pattern = re.compile(rf"(?:^|\s){re.escape(a)}\s*(?:=|:)", re.IGNORECASE)
    return len(pattern.findall(text))


def run_generation(
    *,
    query: str,
    top_k: int = 5,
    mode: str = "hybrid",
    provider: str = "ollama",
    model: str = "qwen3:4b",
    temperature: float = 0.0,
    num_ctx: int = 4096,
    max_tokens: int = 800,
    timeout: int = 420,
    index_dir: str | Path = "data/indexes",
    collection: str = "medical_chunks",
    search_engine: SearchEngine | None = None,
    llm_client: LLMClient | None = None,
    max_display_results: int = 3,
    show_all_results: bool = False,
    show_low_quality: bool = False,
    include_within_reference: bool = False,
    max_summary_results: int = 10,
) -> dict[str, Any]:
    started = time.perf_counter()
    request_id = str(uuid4())

    query_received = query
    q = normalize_query(query_received)
    query_used_for_retrieval = q
    query_used_for_prompt = q
    qn = norm_text(q)
    doc_summary_intent = detect_doc_summary_intent(q)
    sensitive_or_treatment = _query_is_sensitive_or_treatment(q)
    idx = Path(index_dir)
    sqlite_path = idx / "medical_rag.sqlite"
    qdrant_dir = idx / "qdrant"

    retrieval_filters = RetrievalFilters()
    requested_doc_id = extract_requested_doc_id(q)
    requested_analytes: list[str] = []
    found_requested_analytes: list[str] = []
    found_requested_analyte_norms: list[str] = []
    missing_requested_analytes: list[str] = []
    doc_analyte_display_by_norm: dict[str, str] = {}
    requested_to_resolved: dict[str, str] = {}
    summary_section_filter_applied = False
    if requested_doc_id:
        requested_analytes, found_requested_analytes, missing_requested_analytes, doc_analyte_display_by_norm, requested_to_resolved = _resolve_requested_analytes_for_doc(
            query=q,
            sqlite_path=sqlite_path,
            requested_doc_id=requested_doc_id,
        )
        exact_analytes = list(requested_analytes)
    else:
        exact_analytes = _detect_exact_analytes_with_index(q, sqlite_path)
    exact_analyte = exact_analytes[0] if exact_analytes else detect_exact_analyte(q)
    is_above_reference_query = _is_above_reference_query(qn)
    is_normal_or_above = _is_normal_or_above_query(qn)
    is_below_reference_query = _is_below_reference_query(qn)
    is_global_above_query = _is_global_above_reference_query(qn, exact_analytes) and not requested_doc_id

    if requested_doc_id:
        retrieval_filters.doc_id = requested_doc_id
    if exact_analyte and (not requested_doc_id or len(exact_analytes) == 1):
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
    supplemental_rows: list[dict[str, Any]] = []
    doc_analyte_fast_path_used = False
    doc_summary_fast_path_used = False
    retrieval_error: str | None = None
    if sensitive_or_treatment:
        retrieval_response = SimpleNamespace(
            answerability={"status": "guardrail_blocked", "reason": "sensitive_or_treatment_query"},
            filters={},
            top_results=[],
            context_chunks=[],
            sources=[],
        )
    elif requested_doc_id and requested_analytes:
        direct_rows = _load_doc_analyte_rows(
            sqlite_path=sqlite_path,
            doc_id=requested_doc_id,
            analyte_norms=found_requested_analytes if found_requested_analytes else requested_analytes,
        )
        doc_analyte_fast_path_used = True
        retrieval_response = SimpleNamespace(
            answerability={
                "status": "answerable" if direct_rows else "insufficient_context",
                "reason": "doc_multi_analyte_sql_exact_match" if direct_rows else "doc_multi_analyte_sql_no_match",
            },
            filters=retrieval_filters.to_dict(),
            top_results=[_row_to_retrieval_result_fast(r, "doc_analyte_sql_fast_path") for r in direct_rows],
            context_chunks=[],
            sources=[],
        )
        exact_analyte_expected_count = len(direct_rows)
        exact_analyte_rows = list(direct_rows)
    elif requested_doc_id and doc_summary_intent.get("is_summary_intent"):
        above_only = bool(doc_summary_intent.get("wants_above_only"))
        below_only = bool(doc_summary_intent.get("wants_below_only"))
        if above_only and below_only:
            above_only = False
            below_only = False
        summary_rows, summary_section_filter_applied = _load_doc_summary_rows(
            sqlite_path=sqlite_path,
            doc_id=requested_doc_id,
            immunoanalyse_only=bool(doc_summary_intent.get("wants_immunoanalyse_section")),
            above_only=above_only,
            below_only=below_only,
        )
        doc_summary_fast_path_used = True
        retrieval_response = SimpleNamespace(
            answerability={
                "status": "answerable" if summary_rows else "insufficient_context",
                "reason": "doc_summary_sql_exact_match" if summary_rows else "doc_summary_sql_no_match",
            },
            filters=retrieval_filters.to_dict(),
            top_results=[_row_to_retrieval_result_fast(r, "doc_summary_sql_fast_path") for r in summary_rows],
            context_chunks=[],
            sources=[],
        )
        exact_analyte_expected_count = len(summary_rows)
        exact_analyte_rows = list(summary_rows)
    else:
        if exact_analyte:
            exact_analyte_expected_count, exact_analyte_rows = _load_exact_analyte_rows(
                sqlite_path=sqlite_path,
                analyte_norm=exact_analyte,
                limit=max(top_k, max_exact_analyte_results),
            )
        if is_global_above_query:
            supplemental_rows = _load_interpretation_rows(
                sqlite_path=sqlite_path,
                interpretation_status="above_reference",
                limit=max(top_k, max_above_reference_results),
            )
        created_engine = search_engine is None
        engine = search_engine
        if engine is None:
            try:
                engine = SearchEngine(
                    sqlite_path=sqlite_path,
                    qdrant_dir=qdrant_dir,
                    collection=collection,
                )
            except Exception as exc:
                # Local Qdrant lock contention can happen in CLI/demo loops.
                # Fallback to keyword-only retrieval to keep the assistant responsive.
                retrieval_error_text = str(exc)
                if "already accessed by another instance of Qdrant client" in retrieval_error_text:
                    retrieval_response = _keyword_only_search(
                        sqlite_path=sqlite_path,
                        query=query_used_for_retrieval,
                        top_k=top_k,
                        filters=retrieval_filters,
                    )
                    retrieval_response.answerability = retrieval_response.answerability or {}
                    retrieval_response.answerability["fallback_mode"] = "keyword_only_due_qdrant_lock"
                    engine = None
                    created_engine = False
                else:
                    raise
        try:
            if engine is not None:
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
                    if requested_doc_id:
                        relaxed_filters = replace(retrieval_filters)
                        relaxed_filters.interpretation_status = None
                        retrieval_response = engine.search(
                            query=query_used_for_retrieval,
                            mode=mode,
                            top_k=top_k,
                            filters=relaxed_filters,
                            expand_context=True,
                        )
                    else:
                        retrieval_response = engine.search(
                            query=query_used_for_retrieval,
                            mode=mode,
                            top_k=top_k,
                            filters=RetrievalFilters(),
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

    evidence_pack = build_evidence_pack(
        retrieval_response,
        query=q,
        max_evidence=(
            max(top_k, 200)
            if doc_summary_fast_path_used
            else (
                max(top_k, max_exact_analyte_results)
                if exact_analyte
                else max(top_k, max_above_reference_results) if is_global_above_query else top_k
            )
        ),
        exact_analyte=exact_analyte,
        exact_analyte_rows=exact_analyte_rows,
        supplemental_rows=supplemental_rows,
        max_exact_analyte_results=max(top_k, max_exact_analyte_results),
    )

    exact_analyte_for_display = exact_analyte
    if requested_doc_id and len(requested_analytes) > 1:
        exact_analyte_for_display = None

    displayed_evidences, display_meta = _select_displayed_evidences(
        query_norm=qn,
        evidence_pack=evidence_pack,
        exact_analyte=exact_analyte_for_display,
        max_display_results=max_display_results,
        show_all_results=show_all_results,
        show_low_quality=show_low_quality,
    )

    if doc_summary_fast_path_used:
        if show_low_quality:
            displayed_evidences = list(evidence_pack)
        else:
            displayed_evidences = [
                ev for ev in evidence_pack if str(ev.get("evidence_display_quality") or "high") != "low"
            ]
        display_meta["selected_candidates_count"] = len(evidence_pack)
        display_meta["hidden_result_count"] = 0
        display_meta["display_notes"] = []

    if requested_doc_id and requested_analytes:
        displayed_norms = {str(ev.get("analyte_norm") or "").strip().lower() for ev in displayed_evidences}
        found_ordered: list[str] = []
        missing_ordered: list[str] = []
        for requested in requested_analytes:
            resolved = requested_to_resolved.get(requested)
            if resolved and resolved in displayed_norms:
                if requested not in found_ordered:
                    found_ordered.append(requested)
                if resolved not in found_requested_analyte_norms:
                    found_requested_analyte_norms.append(resolved)
            else:
                if requested not in missing_ordered:
                    missing_ordered.append(requested)
        found_requested_analytes = found_ordered
        missing_requested_analytes = missing_ordered

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
        llm_answer = _insufficient_context_for_doc(requested_doc_id) if requested_doc_id else INSUFFICIENT_CONTEXT_SENTENCE
        generation_mode = "no_evidence_for_requested_doc" if requested_doc_id else "no_evidence"
    elif not displayed_evidences:
        llm_answer = _insufficient_context_for_doc(requested_doc_id) if requested_doc_id else INSUFFICIENT_CONTEXT_SENTENCE
        generation_mode = "no_displayable_evidence_for_requested_doc" if requested_doc_id else "no_displayable_evidence"
    elif doc_summary_fast_path_used:
        out_of_reference_focus = bool(doc_summary_intent.get("wants_out_of_reference_focus"))
        include_within_reference_effective = (
            bool(include_within_reference)
            or bool(doc_summary_intent.get("wants_complete"))
            or bool(doc_summary_intent.get("wants_grouped"))
        )
        summary_rows_all = list(displayed_evidences)
        summary_rows_for_display = list(summary_rows_all)
        within_reference_hidden_count = 0

        if (doc_summary_intent.get("wants_important") or out_of_reference_focus) and not include_within_reference_effective:
            allowed_statuses = {"above_reference", "below_reference", "needs_clinical_context"}
            if out_of_reference_focus:
                # Keep off-range results first; optional contextual rows are separated in rendering.
                allowed_statuses = {"above_reference", "below_reference", "needs_clinical_context"}
            important_only = []
            for ev in summary_rows_for_display:
                status = str(ev.get("interpretation_status") or "").strip().lower()
                if status in allowed_statuses:
                    important_only.append(ev)
                    continue
                if (not out_of_reference_focus) and int(ev.get("previous_result_present") or 0) == 1:
                    important_only.append(ev)
            within_reference_hidden_count = sum(
                1
                for ev in summary_rows_all
                if str(ev.get("interpretation_status") or "").strip().lower() == "within_reference"
            )
            if important_only:
                summary_rows_for_display = important_only

        display_meta["within_reference_hidden_count"] = within_reference_hidden_count
        summary_rows_for_display = summary_rows_for_display[: max(1, int(max_summary_results))]
        displayed_evidences = summary_rows_for_display
        llm_answer = _build_doc_summary_answer(
            requested_doc_id=requested_doc_id,
            displayed_evidences=displayed_evidences,
            include_within_reference=include_within_reference_effective,
            within_reference_hidden_count=within_reference_hidden_count,
            section_filter_applied=summary_section_filter_applied,
            asked_immunoanalyse=bool(doc_summary_intent.get("wants_immunoanalyse_section")),
            show_within_hidden_notice=not out_of_reference_focus,
            needs_context_heading="À interpréter avec contexte clinique :" if out_of_reference_focus else "Résultats nécessitant un contexte clinique :",
            include_technical_disclaimer=out_of_reference_focus and not bool(doc_summary_intent.get("wants_grouped")),
        )
        generation_mode = "deterministic_doc_summary_sql_template"
    elif _should_use_deterministic_generation(q, evidence_pack, exact_analyte):
        if requested_doc_id and len(requested_analytes) > 1 and doc_analyte_fast_path_used:
            llm_answer = _build_doc_multi_analyte_answer(
                requested_doc_id=requested_doc_id,
                requested_analytes=requested_analytes,
                found_requested_analytes=found_requested_analytes,
                missing_requested_analytes=missing_requested_analytes,
                requested_to_resolved=requested_to_resolved,
                displayed_evidences=displayed_evidences,
                display_by_norm=doc_analyte_display_by_norm,
            )
            generation_mode = "deterministic_doc_multi_analyte_sql_template"
        else:
            llm_answer = _build_deterministic_evidence_answer(
                query=q,
                displayed_evidences=displayed_evidences,
                exact_analyte=exact_analyte,
                display_notes=display_meta.get("display_notes") or [],
            )
            generation_mode = "deterministic_doc_analyte_sql_template" if doc_analyte_fast_path_used else "deterministic_evidence_template"
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
            llm_answer = sanitize_model_answer(llm_answer)
            if _answer_needs_fallback(llm_answer):
                llm_answer = _build_structured_fallback_answer(q, displayed_evidences, exact_analyte=exact_analyte)
                generation_mode = "llm_fallback_template"
        except LLMClientError as exc:
            llm_error = str(exc)
            generation_mode = "error"
            if "timeout" in llm_error.lower():
                error_type = "llm_timeout"
            else:
                error_type = "llm_error"

    citations = build_citations(displayed_evidences)

    if llm_error:
        final_answer = append_citations(f"Erreur LLM: {llm_error}", citations)
    else:
        final_answer = append_citations(llm_answer, citations)

    validation = validate_answer(
        query=q,
        answer_text=final_answer,
        evidence_pack=evidence_pack,
        displayed_evidences=displayed_evidences,
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
        requested_analytes=requested_analytes,
        found_requested_analytes=found_requested_analytes,
        found_requested_analyte_norms=found_requested_analyte_norms,
        missing_requested_analytes=missing_requested_analytes,
        doc_summary_intent=doc_summary_intent,
        summary_section_filter_applied=summary_section_filter_applied,
    )

    elapsed = time.perf_counter() - started

    result: dict[str, Any] = {
        "request_id": request_id,
        "query": q,
        "query_received": query_received,
        "query_used_for_retrieval": query_used_for_retrieval,
        "query_used_for_prompt": query_used_for_prompt,
        "query_stored": q,
        "requested_doc_id": requested_doc_id,
        "requested_analytes": requested_analytes,
        "found_requested_analytes": found_requested_analytes,
        "found_requested_analyte_norms": found_requested_analyte_norms,
        "missing_requested_analytes": missing_requested_analytes,
        "requested_analyte_coverage": {
            "requested_count": len(requested_analytes),
            "found_count": len(found_requested_analytes),
            "missing_count": len(missing_requested_analytes),
        },
        "normalized_query": q,
        "mode": mode,
        "provider": provider,
        "model": model,
        "top_k": top_k,
        "max_display_results": int(max_display_results),
        "show_all_results": bool(show_all_results),
        "show_low_quality": bool(show_low_quality),
        "include_within_reference": bool(include_within_reference),
        "max_summary_results": int(max_summary_results),
        "timeout": timeout,
        "generation_time_seconds": round(elapsed, 3),
        "answer": final_answer,
        "citations": citations,
        "validation": validation,
        "llm_error": llm_error,
        "error_type": error_type,
        "generation_mode": generation_mode,
        "doc_summary_intent": doc_summary_intent,
        "summary_section_filter_applied": summary_section_filter_applied,
        "detected_analytes": exact_analytes,
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
        "prompt": prompt,
        "debug": {
            "request_id": request_id,
            "query_received": query_received,
            "query_used_for_retrieval": query_used_for_retrieval,
            "query_used_for_prompt": query_used_for_prompt,
            "detected_analytes": exact_analytes,
            "requested_doc_id": requested_doc_id,
            "found_requested_analytes": found_requested_analytes,
            "found_requested_analyte_norms": found_requested_analyte_norms,
            "missing_requested_analytes": missing_requested_analytes,
            "doc_summary_intent": doc_summary_intent,
            "summary_section_filter_applied": summary_section_filter_applied,
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

    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate grounded medical answer using local LLM")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", choices=["keyword", "vector", "hybrid"], default="hybrid")
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--timeout", type=int, default=420)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--show-context", action="store_true")
    parser.add_argument("--max-display-results", type=int, default=3)
    parser.add_argument("--show-all-results", action="store_true")
    parser.add_argument("--show-low-quality", action="store_true")
    parser.add_argument("--include-within-reference", action="store_true")
    parser.add_argument("--max-summary-results", type=int, default=10)
    parser.add_argument("--index-dir", default="data/indexes")
    parser.add_argument("--collection", default="medical_chunks")
    return parser.parse_args()


def _print_human(result: dict[str, Any], show_context: bool) -> None:
    print("Réponse :")
    answer = str(result.get("answer") or "").strip()
    answer_no_prefix = answer
    if answer.lower().startswith("réponse :"):
        answer_no_prefix = answer[len("Réponse :") :].lstrip()
    elif answer.lower().startswith("reponse :"):
        answer_no_prefix = answer[len("Reponse :") :].lstrip()
    print(answer_no_prefix)

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
            include_within_reference=args.include_within_reference,
            max_summary_results=args.max_summary_results,
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
