from __future__ import annotations

from typing import Any

try:
    from medical_entity_resolver import canonicalize_analyte, find_compatible_evidence_rows
except Exception:  # pragma: no cover
    from scripts.generation.medical_entity_resolver import (  # type: ignore
        canonicalize_analyte,
        find_compatible_evidence_rows,
    )


ANSWERABILITY_STATUSES: set[str] = {
    "answerable_exact",
    "answerable_alias",
    "answerable_topic",
    "partially_answerable",
    "not_found",
    "ambiguous",
    "unsafe",
}


def _dedupe_canonical_analytes(requested_analytes: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in list(requested_analytes or []):
        key = canonicalize_analyte(str(raw))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _critical_ambiguity(ambiguity_flags: list[str]) -> bool:
    flags = {str(f).strip().lower() for f in list(ambiguity_flags or []) if str(f).strip()}
    return bool(
        flags.intersection(
            {
                "missing_doc_scope",
                "multiple_doc_scope_ambiguous",
                "topic_vs_specific_analyte_ambiguous",
                "insufficient_clinical_scope",
                "multiple_candidates_clustered",
                "confidence_below_threshold",
            }
        )
    )


def _row_canonical_candidates(row: dict[str, Any]) -> set[str]:
    vals: set[str] = set()
    for field in (
        "analyte_norm",
        "analyte",
        "analyte_label",
        "display_name",
        "source_analyte",
        "parameter",
        "original_analyte",
    ):
        key = canonicalize_analyte(str(row.get(field) or ""))
        if key:
            vals.add(key)
    return vals


def evaluate_answerability(
    *,
    requested_analytes: list[str],
    evidence_rows: list[dict[str, Any]],
    requested_doc_ids: list[str] | None = None,
    safety_intent: str | None = None,
    ambiguity_flags: list[str] | None = None,
) -> dict[str, Any]:
    """
    Deterministic answerability gate used before answer generation.
    """
    safe_intent = str(safety_intent or "").strip().lower()
    if safe_intent in {"diagnostic_safety_question", "treatment_refusal", "treatment_safety_question"}:
        return {
            "status": "unsafe",
            "reason": "treatment_safety_intent" if "treatment" in safe_intent else "diagnostic_safety_intent",
            "matching_strategy": "none",
            "confidence_score": 0.0,
            "found_rows_count": 0,
            "not_found_analytes": _dedupe_canonical_analytes(requested_analytes),
            "matched_doc_ids": [],
            "missing_doc_ids": list(requested_doc_ids or []),
        }

    canonical_requested = _dedupe_canonical_analytes(requested_analytes)
    rows = [dict(r) for r in list(evidence_rows or []) if isinstance(r, dict)]
    req_docs = [str(d).strip() for d in list(requested_doc_ids or []) if str(d).strip()]
    flags = list(ambiguity_flags or [])

    if not canonical_requested:
        if _critical_ambiguity(flags):
            return {
                "status": "ambiguous",
                "reason": "ambiguous_scope_or_intent",
                "matching_strategy": "none",
                "confidence_score": 0.0,
                "found_rows_count": 0,
                "not_found_analytes": [],
                "matched_doc_ids": [],
                "missing_doc_ids": req_docs,
            }
        # No analyte constraint: keep deterministic answerable state neutral.
        return {
            "status": "answerable_exact",
            "reason": "no_requested_analyte_constraint",
            "matching_strategy": "none",
            "confidence_score": 1.0 if rows else 0.0,
            "found_rows_count": len(rows),
            "not_found_analytes": [],
            "matched_doc_ids": [],
            "missing_doc_ids": [],
        }

    compatibility = find_compatible_evidence_rows(
        requested_analytes=canonical_requested,
        evidence_rows=rows,
        scope_doc_ids=req_docs or None,
    )
    found_rows = list(compatibility.get("found_rows") or [])
    strategy = str(compatibility.get("matching_strategy") or "none").strip().lower()
    confidence = float(compatibility.get("confidence_score") or 0.0)
    not_found_analytes = [
        canonicalize_analyte(str(a))
        for a in list(compatibility.get("not_found_analytes") or [])
        if canonicalize_analyte(str(a))
    ]
    not_found_analytes = list(dict.fromkeys(not_found_analytes))

    matched_doc_ids = sorted(
        {
            str(r.get("doc_id") or "").strip()
            for r in found_rows
            if str(r.get("doc_id") or "").strip()
        }
    )
    missing_doc_ids = [d for d in req_docs if d not in set(matched_doc_ids)]

    if not found_rows:
        explicit_constrained_scope = bool(canonical_requested) and bool(req_docs)
        status = "ambiguous" if (_critical_ambiguity(flags) and not explicit_constrained_scope) else "not_found"
        return {
            "status": status,
            "reason": "no_compatible_evidence",
            "matching_strategy": strategy or "none",
            "confidence_score": confidence,
            "found_rows_count": 0,
            "not_found_analytes": not_found_analytes or canonical_requested,
            "matched_doc_ids": matched_doc_ids,
            "missing_doc_ids": missing_doc_ids or req_docs,
        }

    partial_by_analyte = bool(compatibility.get("partially_found"))
    partial_by_doc = bool(req_docs) and bool(missing_doc_ids)
    if partial_by_analyte or partial_by_doc:
        return {
            "status": "partially_answerable",
            "reason": "partial_match",
            "matching_strategy": strategy or "none",
            "confidence_score": confidence,
            "found_rows_count": len(found_rows),
            "not_found_analytes": not_found_analytes,
            "matched_doc_ids": matched_doc_ids,
            "missing_doc_ids": missing_doc_ids,
        }

    requested_set = set(canonical_requested)
    has_strict_exact_match = any(
        bool(_row_canonical_candidates(r).intersection(requested_set))
        for r in found_rows
    )

    if strategy == "topic":
        status = "answerable_topic"
    elif strategy in {"alias", "family", "mixed"} and not has_strict_exact_match:
        status = "answerable_alias"
    else:
        status = "answerable_exact"

    return {
        "status": status,
        "reason": "compatible_evidence_found",
        "matching_strategy": strategy or "exact",
        "confidence_score": confidence if confidence > 0.0 else 1.0,
        "found_rows_count": len(found_rows),
        "not_found_analytes": not_found_analytes,
        "matched_doc_ids": matched_doc_ids,
        "missing_doc_ids": missing_doc_ids,
    }


__all__ = [
    "ANSWERABILITY_STATUSES",
    "evaluate_answerability",
]
