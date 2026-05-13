from __future__ import annotations

from typing import Any

from query_understanding import norm_text


def looks_like_analyte_followup(query: str, requested_analytes: list[str], requested_doc_ids: list[str]) -> bool:
    if requested_doc_ids or not requested_analytes:
        return False
    qn = norm_text(query or "")
    if not qn:
        return False
    starters = ("et ", "et ?", "ok et", "daccord et", "d accord et", "ensuite", "puis")
    return qn.startswith(starters) or qn.endswith("?")


def resolve_followup_doc_scope(
    *,
    query: str,
    requested_analytes: list[str],
    requested_doc_ids: list[str],
    previous_doc_scope: list[str] | None,
) -> list[str]:
    if not looks_like_analyte_followup(query, requested_analytes, requested_doc_ids):
        return list(requested_doc_ids or [])
    if not isinstance(previous_doc_scope, list) or not previous_doc_scope:
        return list(requested_doc_ids or [])
    return [str(d).strip() for d in previous_doc_scope if str(d).strip()]

