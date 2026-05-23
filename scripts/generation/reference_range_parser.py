from __future__ import annotations

import re
from typing import Any

from query_understanding import norm_text


_UNIT_PATTERN = re.compile(r"\b(?:g/l|mg/l|ug/ml|ug/dl|ui/l|iu/l|iu/ml|mui/l|mmol/l|ng/ml|pg/ml|pmol/l|u/l)\b", re.IGNORECASE)
_RANGE_PATTERN = re.compile(r"([<>]?\s*\d+(?:[.,]\d+)?)\s*(?:-|a|à)\s*([<>]?\s*\d+(?:[.,]\d+)?)", re.IGNORECASE)
_THRESHOLD_PATTERN = re.compile(r"([<>]=?)\s*(\d+(?:[.,]\d+)?)", re.IGNORECASE)
_AGE_BETWEEN_PATTERN = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(j|jour|jours|mois|an|ans|annee|annees)\s*(?:a|à|-)\s*(\d+(?:[.,]\d+)?)\s*(j|jour|jours|mois|an|ans|annee|annees)",
    re.IGNORECASE,
)
_AGE_BETWEEN_SINGLE_UNIT_PATTERN = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(?:-|a|à)\s*(\d+(?:[.,]\d+)?)\s*(j|jour|jours|mois|an|ans|annee|annees)",
    re.IGNORECASE,
)
_AGE_OPERATOR_PATTERN = re.compile(r"(>=|<=|>|<)\s*(\d+(?:[.,]\d+)?)\s*(j|jour|jours|mois|an|ans|annee|annees)?", re.IGNORECASE)
_AGE_BLOCK_PATTERN = re.compile(
    r"-?\s*age\s*\(\s*(\d+(?:[.,]\d+)?)\s*[-–]\s*(\d+(?:[.,]\d+)?)\s*(j|jour|jours|mois|an|ans|annee|annees)\s*\)\s*:\s*([<>]?\s*\d+(?:[.,]\d+)?\s*(?:-|a|à)\s*[<>]?\s*\d+(?:[.,]\d+)?|[<>]=?\s*\d+(?:[.,]\d+)?)\s*([A-Za-z/]+)?",
    re.IGNORECASE,
)
_SEX_CONTEXT_HEADER = re.compile(r"(Homme|Femme[^:]*?)\s*:\s*", re.IGNORECASE)
_SEX_INLINE_MARKER = re.compile(r"\b(Homme|Femme)\b", re.IGNORECASE)
_AGE_CLAUSE_PATTERN = re.compile(
    r"((?:>=|<=|>|<)\s*\d+(?:[.,]\d+)?\s*(?:j|jour|jours|mois|an|ans|annee|annees)|\d+(?:[.,]\d+)?\s*(?:-|a|à)\s*\d+(?:[.,]\d+)?\s*(?:j|jour|jours|mois|an|ans|annee|annees))\s*:\s*([<>]?\s*\d+(?:[.,]\d+)?\s*(?:-|a|à)\s*[<>]?\s*\d+(?:[.,]\d+)?|[<>]=?\s*\d+(?:[.,]\d+)?)\s*([A-Za-z/]+)?",
    re.IGNORECASE,
)


def _parse_contextual_age_blocks(text: str, default_unit: str | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if "-age(" not in text.lower():
        return out
    chunks = list(_SEX_CONTEXT_HEADER.finditer(text))
    if not chunks:
        return out
    for idx, m in enumerate(chunks):
        ctx_label = str(m.group(1) or "").strip()
        start = m.end()
        end = chunks[idx + 1].start() if idx + 1 < len(chunks) else len(text)
        payload = text[start:end].strip()
        parent_ctx = _extract_parent_context(ctx_label)
        sex = parent_ctx.get("sex")
        population = parent_ctx.get("population")
        condition = parent_ctx.get("condition")
        # Parse direct range on header (e.g. "Homme: 4.35-5.35 ng/ml").
        # Ignore ranges located inside age(...) blocks; they are age intervals, not bio ranges.
        payload_without_age_blocks = _AGE_BLOCK_PATTERN.sub(" ", payload)
        mrg_direct = _RANGE_PATTERN.search(payload_without_age_blocks)
        if mrg_direct:
            unit_direct = (
                _UNIT_PATTERN.search(payload_without_age_blocks[mrg_direct.end() : mrg_direct.end() + 24])
                or _UNIT_PATTERN.search(payload_without_age_blocks)
            )
            out.append(
                {
                    "label": ctx_label,
                    "population": population,
                    "sex": sex,
                    "condition": condition,
                    "age_min": None,
                    "age_max": None,
                    "age_operator": None,
                    "age_value": None,
                    "age_unit": None,
                    "low": _to_float(mrg_direct.group(1).replace(">", "").replace("<", "")),
                    "high": _to_float(mrg_direct.group(2).replace(">", "").replace("<", "")),
                    "operator": "range",
                    "threshold": None,
                    "unit": unit_direct.group(0) if unit_direct else (default_unit or None),
                    "raw": f"{ctx_label}: {payload}",
                    "confidence": 0.9,
                }
            )
        # Parse age sub-blocks
        for ab in _AGE_BLOCK_PATTERN.finditer(payload):
            amin = _to_float(ab.group(1))
            amax = _to_float(ab.group(2))
            aunit = _normalize_age_unit(ab.group(3))
            value_blob = str(ab.group(4) or "").strip()
            unit_blob = str(ab.group(5) or "").strip()
            mrg = _RANGE_PATTERN.search(value_blob)
            mth = _THRESHOLD_PATTERN.search(value_blob) if not mrg else None
            if mrg:
                operator = "range"
                low = _to_float(mrg.group(1).replace(">", "").replace("<", ""))
                high = _to_float(mrg.group(2).replace(">", "").replace("<", ""))
                threshold = None
            elif mth:
                operator = mth.group(1)
                low = None
                high = None
                threshold = _to_float(mth.group(2))
            else:
                continue
            out.append(
                {
                    "label": f"{ctx_label} — {int(amin) if amin is not None and amin.is_integer() else amin}-{int(amax) if amax is not None and amax.is_integer() else amax} {ab.group(3)}",
                    "population": population,
                    "sex": sex,
                    "condition": condition,
                    "age_min": amin,
                    "age_max": amax,
                    "age_operator": None,
                    "age_value": None,
                    "age_unit": aunit,
                    "low": low,
                    "high": high,
                    "operator": operator,
                    "threshold": threshold,
                    "unit": unit_blob or (default_unit or None),
                    "raw": f"{ctx_label}: {payload}",
                    "confidence": 0.95,
                }
            )
    return out


def _parse_sex_parent_blocks_general(text: str, default_unit: str | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    marks = list(_SEX_INLINE_MARKER.finditer(text))
    if len(marks) < 1:
        return out
    for idx, m in enumerate(marks):
        sex_label = str(m.group(1) or "").strip()
        start = m.end()
        end = marks[idx + 1].start() if idx + 1 < len(marks) else len(text)
        payload = text[start:end].strip()
        if payload.startswith(":"):
            payload = payload[1:].strip()
        parent_ctx = _extract_parent_context(sex_label)
        clauses = list(_AGE_CLAUSE_PATTERN.finditer(payload))
        if clauses:
            for cl in clauses:
                age_blob = str(cl.group(1) or "").strip()
                value_blob = str(cl.group(2) or "").strip()
                unit_blob = str(cl.group(3) or "").strip()
                item = _base_item(f"{sex_label}: {age_blob}: {value_blob}")
                item["label"] = f"{sex_label} — {age_blob}"
                item["sex"] = parent_ctx.get("sex")
                item["population"] = parent_ctx.get("population")
                item["condition"] = parent_ctx.get("condition")
                m_age_between_single = _AGE_BETWEEN_SINGLE_UNIT_PATTERN.search(age_blob)
                if m_age_between_single:
                    item["age_min"] = _to_float(m_age_between_single.group(1))
                    item["age_max"] = _to_float(m_age_between_single.group(2))
                    item["age_unit"] = _normalize_age_unit(m_age_between_single.group(3))
                else:
                    m_age_op = _AGE_OPERATOR_PATTERN.search(age_blob)
                    if m_age_op:
                        item["age_operator"] = m_age_op.group(1)
                        item["age_value"] = _to_float(m_age_op.group(2))
                        item["age_unit"] = _normalize_age_unit(m_age_op.group(3) or "ans")
                mrg = _RANGE_PATTERN.search(value_blob)
                mth = _THRESHOLD_PATTERN.search(value_blob) if not mrg else None
                if mrg:
                    item["operator"] = "range"
                    item["low"] = _to_float(mrg.group(1).replace(">", "").replace("<", ""))
                    item["high"] = _to_float(mrg.group(2).replace(">", "").replace("<", ""))
                elif mth:
                    item["operator"] = mth.group(1)
                    item["threshold"] = _to_float(mth.group(2))
                else:
                    continue
                item["unit"] = unit_blob or (default_unit or None)
                item["confidence"] = 0.9
                out.append(item)
            continue
    return out


def _to_float(value: str | None) -> float | None:
    s = str(value or "").strip().replace(",", ".")
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _normalize_age_unit(unit: str | None) -> str | None:
    u = norm_text(str(unit or ""))
    if u in {"j", "jour", "jours"}:
        return "days"
    if u in {"mois"}:
        return "months"
    if u in {"an", "ans", "annee", "annees"}:
        return "years"
    return None


def _normalize_population(label: str) -> str | None:
    ln = norm_text(label)
    if any(k in ln for k in ["femme cyclee j2 j4", "j2 j4", "j2-j4"]):
        return "cycled_female_j2_j4"
    if "cordon" in ln:
        return "cord"
    if "nourrisson" in ln:
        return "infant"
    if "nouveau ne" in ln or "nouveaune" in ln:
        return "newborn"
    if "enfant" in ln:
        return "child"
    if "adulte" in ln:
        return "adult"
    return None


def _normalize_condition(label: str) -> str | None:
    ln = norm_text(label)
    if any(k in ln for k in ["j2 j4", "j2-j4", "femme cyclee"]):
        return "cycled_female_j2_j4"
    if "ambulatoire" in ln:
        return "ambulatory"
    if any(k in ln for k in ["alite", "alité"]):
        return "bedridden"
    if "a jeun" in ln:
        return "fasting"
    if "risque majeur" in ln:
        return "risk_major"
    if "souhaitable" in ln:
        return "desirable"
    if "modere" in ln or "modéré" in ln:
        return "moderate"
    if "eleve" in ln or "élevé" in ln:
        return "high"
    # Generic profile labels (e.g., Alpha/Beta) for parent-context grouping.
    if ln and not any(ch.isdigit() for ch in ln):
        if not any(k in ln for k in ["homme", "femme", "adulte", "enfant", "nourrisson", "nouveau ne", "cordon"]):
            token = re.sub(r"[^a-z0-9]+", "_", ln).strip("_")
            if token:
                return token
    return None


def _extract_parent_context(label: str) -> dict[str, Any]:
    return {
        "sex": _normalize_sex(label),
        "population": _normalize_population(label),
        "condition": _normalize_condition(label),
    }


def _normalize_sex(label: str) -> str | None:
    ln = norm_text(label)
    if any(k in ln for k in ["homme", "masculin", "male"]):
        return "male"
    if any(k in ln for k in ["femme", "feminin", "female"]):
        return "female"
    if any(k in ln for k in ["indifferent", "indifferente", "indifferentes", "sexe age indifferent"]):
        return "any"
    return None


def _extract_label(text: str) -> tuple[str, str]:
    parts = text.split(":", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return "", text.strip()


def _split_candidates(reference_raw: str) -> list[str]:
    s = re.sub(r"\s+", " ", str(reference_raw or "").strip())
    if not s:
        return []
    labeled_strict = list(re.finditer(r"([A-ZÀ-Ý0-9><][A-Za-zÀ-ÿ0-9><=\s]*):\s*([^:]+?)(?=(?:[A-ZÀ-Ý0-9><][A-Za-zÀ-ÿ0-9><=\s]*:)|$)", s))
    labeled_wide = list(
        re.finditer(
            r"([A-ZÀ-Ý0-9><][A-Za-zÀ-ÿ0-9><=\s()/_-]*):\s*([^:]+?)(?=(?:[A-ZÀ-Ý0-9><][A-Za-zÀ-ÿ0-9><=\s()/_-]*:)|$)",
            s,
        )
    )
    labeled = labeled_wide if len(labeled_wide) > len(labeled_strict) else labeled_strict
    if labeled:
        return [f"{m.group(1).strip()}: {m.group(2).strip()}".strip(" ;,") for m in labeled if m.group(2).strip()]
    return [x.strip(" ;,") for x in re.split(r"\s{2,}|;\s*", s) if x.strip(" ;,")]


def _base_item(raw: str) -> dict[str, Any]:
    return {
        "label": "",
        "population": None,
        "sex": None,
        "condition": None,
        "age_min": None,
        "age_max": None,
        "age_operator": None,
        "age_value": None,
        "age_unit": None,
        "low": None,
        "high": None,
        "operator": None,
        "threshold": None,
        "unit": None,
        "raw": raw,
        "confidence": 0.5,
    }


def parse_reference_ranges(reference_raw: str, default_unit: str | None = None) -> list[dict]:
    text = str(reference_raw or "").strip()
    if not text:
        return []
    contextual = _parse_contextual_age_blocks(text, default_unit)
    if contextual:
        return contextual
    sex_parent = _parse_sex_parent_blocks_general(text, default_unit)
    qn = norm_text(text)
    if sex_parent and (len(sex_parent) >= 2 or ("homme" in qn and "femme" in qn)):
        return sex_parent
    parsed: list[dict[str, Any]] = []
    for candidate in _split_candidates(text):
        item = _base_item(candidate)
        label, payload = _extract_label(candidate)
        item["label"] = label or payload
        label_blob = f"{label} {payload}".strip()
        item["population"] = _normalize_population(label_blob)
        item["sex"] = _normalize_sex(label_blob)
        item["condition"] = _normalize_condition(label_blob)

        m_age_between = _AGE_BETWEEN_PATTERN.search(label_blob)
        if m_age_between:
            item["age_min"] = _to_float(m_age_between.group(1))
            item["age_max"] = _to_float(m_age_between.group(3))
            item["age_unit"] = _normalize_age_unit(m_age_between.group(4))
            item["confidence"] += 0.15
        else:
            m_age_between_single = _AGE_BETWEEN_SINGLE_UNIT_PATTERN.search(label_blob)
            if m_age_between_single:
                item["age_min"] = _to_float(m_age_between_single.group(1))
                item["age_max"] = _to_float(m_age_between_single.group(2))
                item["age_unit"] = _normalize_age_unit(m_age_between_single.group(3))
                item["confidence"] += 0.15
            m_age_op = _AGE_OPERATOR_PATTERN.search(label_blob)
            if m_age_op and item.get("age_min") is None:
                item["age_operator"] = m_age_op.group(1)
                item["age_value"] = _to_float(m_age_op.group(2))
                item["age_unit"] = _normalize_age_unit(m_age_op.group(3) or "ans")
                item["confidence"] += 0.15

        # Hierarchical age blocks such as:
        # "Femme cyclée J2-J4: -age(20-24 ans): 3.55-4.33 ng/ml -age(25-29 ans): 3.03-3.87 ng/ml"
        age_blocks = list(_AGE_BLOCK_PATTERN.finditer(payload))
        if age_blocks:
            for ab in age_blocks:
                sub = dict(item)
                amin = _to_float(ab.group(1))
                amax = _to_float(ab.group(2))
                aunit = _normalize_age_unit(ab.group(3))
                value_blob = str(ab.group(4) or "").strip()
                unit_blob = str(ab.group(5) or "").strip()
                mrg = _RANGE_PATTERN.search(value_blob)
                mth = _THRESHOLD_PATTERN.search(value_blob) if not mrg else None
                if mrg:
                    sub["operator"] = "range"
                    sub["low"] = _to_float(mrg.group(1).replace(">", "").replace("<", ""))
                    sub["high"] = _to_float(mrg.group(2).replace(">", "").replace("<", ""))
                elif mth:
                    sub["operator"] = mth.group(1)
                    sub["threshold"] = _to_float(mth.group(2))
                else:
                    continue
                sub["age_min"] = amin
                sub["age_max"] = amax
                sub["age_unit"] = aunit
                ctx_label = str(label or "").strip() or "profil"
                sub["label"] = f"{ctx_label} — {int(amin) if amin is not None and amin.is_integer() else amin}-{int(amax) if amax is not None and amax.is_integer() else amax} {ab.group(3)}"
                parent_ctx = _extract_parent_context(ctx_label)
                if sub.get("population") is None:
                    sub["population"] = parent_ctx.get("population")
                if sub.get("sex") is None:
                    sub["sex"] = parent_ctx.get("sex")
                if sub.get("condition") is None:
                    sub["condition"] = parent_ctx.get("condition")
                sub["unit"] = unit_blob or (default_unit or None)
                sub["confidence"] = float(min(0.99, max(0.1, sub["confidence"] + 0.3)))
                parsed.append(sub)
            continue

        range_matches = list(_RANGE_PATTERN.finditer(payload))
        threshold_matches = list(_THRESHOLD_PATTERN.finditer(payload)) if not range_matches else []
        if range_matches:
            for rm in range_matches:
                sub = dict(item)
                sub["low"] = _to_float(rm.group(1).replace(">", "").replace("<", ""))
                sub["high"] = _to_float(rm.group(2).replace(">", "").replace("<", ""))
                sub["operator"] = "range"
                unit_match = _UNIT_PATTERN.search(payload[rm.end() : rm.end() + 24]) or _UNIT_PATTERN.search(payload) or _UNIT_PATTERN.search(label_blob)
                sub["unit"] = unit_match.group(0) if unit_match else (default_unit or None)
                sub["confidence"] = float(min(0.99, max(0.1, sub["confidence"] + 0.25)))
                if sub["sex"] is None and sub["population"] is None and "indifferent" in norm_text(label_blob):
                    sub["sex"] = "any"
                parsed.append(sub)
        elif threshold_matches:
            for tm in threshold_matches:
                sub = dict(item)
                sub["operator"] = tm.group(1)
                sub["threshold"] = _to_float(tm.group(2))
                unit_match = _UNIT_PATTERN.search(payload[tm.end() : tm.end() + 24]) or _UNIT_PATTERN.search(payload) or _UNIT_PATTERN.search(label_blob)
                sub["unit"] = unit_match.group(0) if unit_match else (default_unit or None)
                sub["confidence"] = float(min(0.99, max(0.1, sub["confidence"] + 0.25)))
                if sub["sex"] is None and sub["population"] is None and "indifferent" in norm_text(label_blob):
                    sub["sex"] = "any"
                parsed.append(sub)
    return parsed
