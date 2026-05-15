from __future__ import annotations

from typing import Any


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace(",", ".")
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def row_matches_any_target_value(row: dict[str, Any], targets: list[str]) -> bool:
    if not targets:
        return True
    value_raw = str(row.get("value_raw") or "").strip()
    value_num = row.get("value_numeric")
    raw_norm = value_raw.replace(",", ".").strip().lower()
    raw_norm_nolead = raw_norm.lstrip("0") or "0"
    vf = to_float(value_num if value_num not in (None, "") else value_raw)
    for target in targets:
        tn = str(target or "").replace(",", ".").strip().lower()
        tn_nolead = tn.lstrip("0") or "0"
        if raw_norm == tn or raw_norm_nolead == tn_nolead:
            return True
        tf = to_float(target)
        if tf is not None and vf is not None and abs(tf - vf) <= 1e-9:
            return True
    return False


def row_matches_value_criterion(row: dict[str, Any], targets: list[str], operator: str | None) -> bool:
    if not targets:
        return True
    op = str(operator or "").strip()
    if op not in {">", ">=", "<", "<=", "="}:
        return row_matches_any_target_value(row, targets)

    vf = to_float(row.get("value_numeric"))
    if vf is None:
        vf = to_float(row.get("value_raw"))
    tf = to_float(targets[0])
    if vf is None or tf is None:
        return row_matches_any_target_value(row, targets)

    if op == ">":
        return vf > tf
    if op == ">=":
        return vf >= tf
    if op == "<":
        return vf < tf
    if op == "<=":
        return vf <= tf
    return abs(vf - tf) <= 1e-9


def dedup_evidences(evidences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str, str, str]] = set()
    for ev in evidences:
        key = (
            str(ev.get("patient_token") or "").strip().lower(),
            str(ev.get("analyte_norm") or ev.get("analyte") or "").strip().lower(),
            str(ev.get("value_numeric") if ev.get("value_numeric") is not None else ev.get("value_raw") or "").strip(),
            str(ev.get("unit") or "").strip().lower(),
            str(ev.get("doc_id") or "").strip().lower(),
            str(ev.get("page_number") or "").strip(),
            str(ev.get("sample_token") or "").strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(ev)
    return out
