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
from citation_builder import append_citations, build_citations
from evidence_builder import build_evidence_pack as build_retrieval_evidence_pack
from llm_client import LLMClient, LLMClientError
from prompt_builder import INSUFFICIENT_CONTEXT_SENTENCE, build_prompt
from query_understanding import (
    QueryUnderstanding,
    contains_exact_term,
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

    lines: list[str] = []
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
        if requested_value and str(requested_value).strip():
            where += " AND (instr(lower(coalesce(m.value_raw,'')), ?) > 0 OR instr(lower(coalesce(c.text_for_keyword,'')), ?) > 0)"
            value_norm = str(requested_value).strip().lower().replace(",", ".")
            params.extend([value_norm, value_norm])
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
    return f"[doc_id={row.get('doc_id')}, page={row.get('page_number')}, row={row.get('row_index')}]"


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
    return {
        "doc_id": str(row.get("doc_id") or requested_doc_id or ""),
        "patient_token": str(row.get("patient_token") or "").strip(),
        "page": row.get("page_number"),
        "row": row.get("row_index"),
        "chunk_id": row.get("chunk_id"),
        "analyte": _clean_analyte_label(str(row.get("analyte") or row.get("parameter") or "non précisé")),
        "analyte_norm": str(row.get("analyte_norm") or "").strip().lower(),
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
        return str(ev.get("analyte") or "non précisé")
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
    fallback = render_evidence_pack_deterministic(evidence_pack, query_understanding.output_format)
    evidences = list(evidence_pack.get("evidences") or [])
    if not evidences and evidence_pack.get("intent") != "comment_without_measured_value":
        return fallback, "deterministic_structured_renderer", None

    compact_pack = {
        "requested_doc_ids": query_understanding.requested_doc_ids,
        "intent": evidence_pack.get("intent"),
        "output_format": query_understanding.output_format,
        "requested_table_columns": query_understanding.requested_table_columns,
        "answer_style": query_understanding.answer_style,
        "missing_items": evidence_pack.get("missing_items") or [],
        "evidences": [
            {
                "doc_id": ev.get("doc_id"),
                "analyte": ev.get("analyte"),
                "current_value": ev.get("current_value"),
                "unit": ev.get("unit"),
                "reference": ev.get("reference"),
                "technical_status": ev.get("technical_status"),
                "previous_result": ev.get("previous_result"),
                "variation": ev.get("variation"),
            }
            for ev in evidences
        ],
        "comment_text": evidence_pack.get("comment_text"),
    }

    prompt = (
        "Tu es un assistant médical technique pour un système RAG.\n"
        "Tu dois répondre uniquement avec les données JSON fournies ci-dessous.\n"
        "Interdictions strictes: ne pas inventer/analyser médicalement/diagnostiquer, ne pas modifier les valeurs, "
        "ne pas changer de document, ne pas ajouter d'analyte absent.\n"
        "Si output_format=table, réponds uniquement en tableau Markdown.\n"
        "Les sources doivent toujours être visibles: soit en colonne Source, soit en section Sources.\n"
        "Si requested_table_columns est fourni, respecte strictement ces colonnes et leur ordre.\n"
        "Si answer_style=yes_no, commence strictement par Oui, Non ou Impossible à déterminer.\n"
        "Si output_format=json, réponds uniquement en JSON valide sans texte additionnel.\n"
        "Si une donnée manque, indique exactement: non retrouvé dans le document demandé.\n"
        "Question utilisateur:\n"
        f"{user_question.strip()}\n\n"
        "Evidence pack JSON:\n"
        f"{json.dumps(compact_pack, ensure_ascii=False)}\n"
    )

    client = llm_client or LLMClient(provider=provider)
    try:
        llm_answer = client.generate(
            prompt=prompt,
            model=model,
            temperature=0.0 if temperature is None else min(float(temperature), 0.2),
            num_ctx=num_ctx,
            max_tokens=max(160, min(int(max_tokens), 420)),
            timeout=max(4, min(int(timeout), 18)),
            keep_alive="5m",
        )
        llm_answer = sanitize_model_answer(llm_answer)
        if not llm_answer.strip():
            return fallback, "deterministic_structured_renderer", "empty_llm_answer"
        if query_understanding.output_format == "table" and not _is_table_markdown(llm_answer):
            return fallback, "llm_writer_format_fallback", "output_format_not_respected"
        if query_understanding.answer_style == "yes_no":
            prefix = norm_text(llm_answer).strip()
            if not (
                prefix.startswith("oui")
                or prefix.startswith("non")
                or prefix.startswith("yes")
                or prefix.startswith("no")
                or prefix.startswith("impossible a determiner")
                or prefix.startswith("cannot determine")
            ):
                return fallback, "llm_writer_format_fallback", "output_format_not_respected"
        if _answer_needs_fallback(llm_answer):
            return fallback, "llm_writer_quality_fallback", "llm_quality_fallback"
        return llm_answer, "llm_grounded_writer", None
    except LLMClientError as exc:
        return fallback, "llm_writer_error_fallback", str(exc)


def build_structured_evidence_pack(
    *,
    query: str,
    query_understanding: QueryUnderstanding,
    sqlite_path: Path,
) -> dict[str, Any]:
    requested_doc_ids = list(query_understanding.requested_doc_ids or [])
    requested_analytes = list(query_understanding.requested_analytes or [])
    qn = norm_text(query)
    compare_previous = query_understanding.requires_previous_results
    intent = query_understanding.intent

    pack: dict[str, Any] = {
        "question": query,
        "requested_doc_ids": requested_doc_ids,
        "requested_analytes": requested_analytes,
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
        rows = [r for r in rows if _row_matches_any_target_value(r, target_values)]
        technical_condition = str(query_understanding.technical_condition or "").strip().lower()
        if technical_condition in {"above_reference", "below_reference", "within_reference", "not_interpretable"}:
            rows = [r for r in rows if _status_code(r) == technical_condition]
        evidences = [_structured_record_from_row(r) for r in rows]
        pack["rows"] = rows
        pack["evidences"] = evidences
        return pack

    if not requested_doc_ids:
        return pack

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
        evidences: list[dict[str, Any]] = []
        missing: list[str] = []
        for analyte in requested_analytes:
            a = _best_row_for_analyte(left_rows, analyte)
            b = _best_row_for_analyte(right_rows, analyte)
            if not a and not b:
                missing.append(analyte)
                continue
            if a:
                rec = _structured_record_from_row(a)
                rec["comparison_side"] = left
                evidences.append(rec)
            if b:
                rec = _structured_record_from_row(b)
                rec["comparison_side"] = right
                evidences.append(rec)
            if a and b:
                av = str(a.get("value_raw") or "")
                bv = str(b.get("value_raw") or "")
                evidences.append(
                    {
                        "doc_id": f"{left} vs {right}",
                        "page": None,
                        "row": None,
                        "chunk_id": None,
                        "analyte": _canonical_display_name(analyte),
                        "analyte_norm": analyte,
                        "current_value": f"{left}={av} | {right}={bv}",
                        "unit": str(a.get("unit") or b.get("unit") or "").strip(),
                        "reference": str(a.get("reference_range") or b.get("reference_range") or "").strip(),
                        "previous_result": "",
                        "technical_status_code": "not_interpretable",
                        "technical_status": "différence technique",
                        "variation": _variation_label(bv, av),
                        "source": "",
                    }
                )
        pack["evidences"] = evidences
        pack["missing_items"] = missing
        pack["rows"] = rows
        return pack

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
            if key and key not in left_keys:
                left_keys[key] = row
        for row in right_rows:
            key = str(row.get("analyte_norm") or "").strip().lower() or norm_text(str(row.get("analyte") or ""))
            if key and key not in right_keys:
                right_keys[key] = row

        only_left = sorted(set(left_keys.keys()) - set(right_keys.keys()))
        only_right = sorted(set(right_keys.keys()) - set(left_keys.keys()))
        evidences: list[dict[str, Any]] = []
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
        pack["rows"] = rows
        pack["evidences"] = evidences
        return pack

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
        return pack

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
        return pack

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
        missing: list[str] = []
        if requested_analytes:
            for analyte in requested_analytes:
                row = _best_row_for_analyte(rows, analyte)
                if row is None:
                    missing.append(analyte)
                    continue
                record = _structured_record_from_row(row)
                if compare_previous and not record.get("previous_result"):
                    record["variation"] = "non comparable"
                evidences.append(record)
        else:
            for row in rows:
                status = _status_code(row)
                if any(k in qn for k in ["hors reference", "anomal", "attention technique"]) and status not in {
                    "above_reference",
                    "below_reference",
                }:
                    continue
                evidences.append(_structured_record_from_row(row))
        pack["evidences"] = evidences
        pack["missing_items"] = missing
        pack["rows"] = rows
        return pack

    return pack


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

    output_format = query_understanding.output_format
    if output_format == "list":
        output_format = str(src.get("output_format") or "list").lower()
    if output_format == "list" and ("json" in qn):
        output_format = "json"
    if output_format == "list" and ("tableau" in qn or "table" in qn):
        output_format = "table"

    return {
        **src,
        "question": query,
        "intent": "response_transform",
        "output_format": output_format,
        "requested_table_columns": requested_columns,
        "answer_style": query_understanding.answer_style or src.get("answer_style") or "standard",
        "evidences": evidences,
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
) -> dict[str, Any]:
    started = time.perf_counter()
    request_id = str(uuid4())

    query_received = query
    q = normalize_query(query_received)
    query_used_for_retrieval = q
    query_used_for_prompt = q
    qn = norm_text(q)
    query_understanding = parse_query_understanding(q)
    requested_doc_ids = list(query_understanding.requested_doc_ids)
    requested_doc_id = requested_doc_ids[0] if len(requested_doc_ids) == 1 else None
    sensitive_or_treatment = _query_is_sensitive_or_treatment(q)
    idx = Path(index_dir)
    sqlite_path = idx / "medical_rag.sqlite"
    qdrant_dir = idx / "qdrant"

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

    if query_understanding.intent == "response_transform":
        if not previous_structured_evidence_pack:
            elapsed = time.perf_counter() - started
            answer = "Je n’ai pas de réponse précédente exploitable à transformer."
            validation = validate_answer(
                query=q,
                answer_text=answer,
                evidence_pack=[],
                displayed_evidences=[],
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
            )
            return {
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
                "validation": validation,
                "llm_error": None,
                "error_type": None,
                "generation_mode": "deterministic_response_transform",
                "detected_analytes": exact_analytes,
                "query_understanding": {
                    "requested_doc_ids": query_understanding.requested_doc_ids,
                    "requested_analytes": query_understanding.requested_analytes,
                    "requested_value": query_understanding.requested_value,
                    "patient_query": query_understanding.patient_query,
                    "intent": query_understanding.intent,
                    "output_format": query_understanding.output_format,
                    "requested_table_columns": query_understanding.requested_table_columns,
                    "answer_style": query_understanding.answer_style,
                    "requires_global_search": query_understanding.requires_global_search,
                    "technical_condition": query_understanding.technical_condition,
                    "safety_intent": query_understanding.safety_intent,
                    "requires_previous_results": query_understanding.requires_previous_results,
                    "requires_comparison": query_understanding.requires_comparison,
                    "requires_section_summary": query_understanding.requires_section_summary,
                    "language": query_understanding.language,
                },
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
                "debug": {
                    "request_id": request_id,
                    "query_received": query_received,
                    "query_used_for_retrieval": query_used_for_retrieval,
                    "query_used_for_prompt": query_used_for_prompt,
                    "detected_analytes": exact_analytes,
                    "requested_doc_ids": requested_doc_ids,
                    "generation_mode": "deterministic_response_transform",
                    "intents": intents,
                },
                "exact_analyte_coverage": {
                    "detected_exact_analyte": exact_analyte,
                    "expected_exact_analyte_count": len(exact_analytes),
                    "retrieved_exact_analyte_count": 0,
                    "displayed_exact_analyte_count": 0,
                },
            }

        transformed_pack = _build_response_transform_pack(
            query=q,
            query_understanding=query_understanding,
            previous_pack=previous_structured_evidence_pack,
        )
        output_format = str(transformed_pack.get("output_format") or query_understanding.output_format or "list").lower()
        if output_format == "json":
            answer = json.dumps(
                {
                    "intent": transformed_pack.get("intent"),
                    "requested_doc_ids": transformed_pack.get("requested_doc_ids") or [],
                    "requested_analytes": transformed_pack.get("requested_analytes") or [],
                    "evidences": transformed_pack.get("evidences") or [],
                    "missing_items": transformed_pack.get("missing_items") or [],
                },
                ensure_ascii=False,
            )
            generation_mode = "deterministic_response_transform_json"
        else:
            answer = render_evidence_pack_deterministic(transformed_pack, output_format)
            generation_mode = "deterministic_response_transform"

        displayed_evidences = [
            {
                "doc_id": ev.get("doc_id"),
                "chunk_id": ev.get("chunk_id"),
                "page_number": ev.get("page"),
                "row_index": ev.get("row"),
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
        validation = validate_answer(
            query=q,
            answer_text=answer,
            evidence_pack=displayed_evidences,
            displayed_evidences=displayed_evidences,
            generation_mode=generation_mode,
            retrieval_status="answerable" if displayed_evidences else "insufficient_context",
            query_received=query_received,
            query_used_for_retrieval=query_used_for_retrieval,
            query_used_for_prompt=query_used_for_prompt,
            query_stored=q,
            detected_analytes=exact_analytes,
            query_intents=intents,
            output_format_requested=output_format,
            answer_style_requested=query_understanding.answer_style,
            requested_table_columns=transformed_pack.get("requested_table_columns") or query_understanding.requested_table_columns,
            requested_technical_condition=query_understanding.technical_condition,
        )
        elapsed = time.perf_counter() - started
        return {
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
            "validation": validation,
            "llm_error": None,
            "error_type": None,
            "generation_mode": generation_mode,
            "detected_analytes": exact_analytes,
            "query_understanding": {
                "requested_doc_ids": query_understanding.requested_doc_ids,
                "requested_analytes": query_understanding.requested_analytes,
                "requested_value": query_understanding.requested_value,
                "patient_query": query_understanding.patient_query,
                "intent": query_understanding.intent,
                "output_format": query_understanding.output_format,
                "requested_table_columns": query_understanding.requested_table_columns,
                "answer_style": query_understanding.answer_style,
                "requires_global_search": query_understanding.requires_global_search,
                "technical_condition": query_understanding.technical_condition,
                "safety_intent": query_understanding.safety_intent,
                "requires_previous_results": query_understanding.requires_previous_results,
                "requires_comparison": query_understanding.requires_comparison,
                "requires_section_summary": query_understanding.requires_section_summary,
                "language": query_understanding.language,
            },
            "structured_evidence_pack": transformed_pack,
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
                "intents": intents,
            },
            "exact_analyte_coverage": {
                "detected_exact_analyte": exact_analyte,
                "expected_exact_analyte_count": len(exact_analytes),
                "retrieved_exact_analyte_count": len(displayed_evidences),
                "displayed_exact_analyte_count": len(displayed_evidences),
            },
        }

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

        generated_text, generation_mode, writer_error = generate_grounded_response_with_llm(
            user_question=q,
            query_understanding=query_understanding,
            evidence_pack=structured_pack,
            llm_client=llm_client,
            provider=provider,
            model=model,
            temperature=temperature,
            num_ctx=num_ctx,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        if not generated_text.strip():
            generated_text = _missing_doc_answer()

        citations = build_citations(displayed_evidences)
        if query_understanding.output_format == "table" and _table_has_source_column(generated_text):
            final_answer = generated_text.strip()
        elif query_understanding.output_format == "yes_no":
            final_answer = generated_text.strip()
        else:
            final_answer = append_citations(generated_text, citations)

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
        )

        elapsed = time.perf_counter() - started
        retrieval_sources = [
            {
                "doc_id": ev.get("doc_id"),
                "page_number": ev.get("page_number"),
                "chunk_id": ev.get("chunk_id"),
                "chunk_type": ev.get("chunk_type"),
            }
            for ev in displayed_evidences
        ]
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
            "validation": validation,
            "llm_error": writer_error,
            "error_type": "llm_writer_error" if writer_error else None,
            "generation_mode": generation_mode,
            "detected_analytes": exact_analytes,
            "query_understanding": {
                "requested_doc_ids": query_understanding.requested_doc_ids,
                "requested_analytes": query_understanding.requested_analytes,
                "requested_value": query_understanding.requested_value,
                "patient_query": query_understanding.patient_query,
                "intent": query_understanding.intent,
                "output_format": query_understanding.output_format,
                "requested_table_columns": query_understanding.requested_table_columns,
                "answer_style": query_understanding.answer_style,
                "requires_global_search": query_understanding.requires_global_search,
                "technical_condition": query_understanding.technical_condition,
                "safety_intent": query_understanding.safety_intent,
                "requires_previous_results": query_understanding.requires_previous_results,
                "requires_comparison": query_understanding.requires_comparison,
                "requires_section_summary": query_understanding.requires_section_summary,
                "language": query_understanding.language,
            },
            "structured_evidence_pack": structured_pack,
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
                "intents": intents,
            },
            "exact_analyte_coverage": {
                "detected_exact_analyte": exact_analyte,
                "expected_exact_analyte_count": len(exact_analytes),
                "retrieved_exact_analyte_count": len(found_requested_analytes),
                "displayed_exact_analyte_count": len(found_requested_analytes),
            },
        }

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
            llm_answer = sanitize_model_answer(llm_answer)
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

    if llm_error and not llm_answer:
        final_answer = append_citations(f"Erreur LLM: {llm_error}", citations)
    else:
        final_answer = append_citations(llm_answer, citations)

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
    )

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
        "validation": validation,
        "llm_error": llm_error,
        "error_type": error_type,
        "generation_mode": generation_mode,
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
