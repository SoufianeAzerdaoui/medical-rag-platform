from __future__ import annotations

import re
import unicodedata
from typing import Any

from config_loader import get_analyte_families_config, get_priority_scoring_config


def _norm(value: str) -> str:
    s = str(value or "").strip().lower().replace("µ", "u")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s)
    return s


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    s = str(value).strip().replace(",", ".")
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def get_textual_severity_terms() -> list[str]:
    cfg = (get_priority_scoring_config() or {}).get("priority_scoring") or {}
    return [str(t).strip().lower() for t in list(cfg.get("textual_severity_terms") or []) if str(t).strip()]


def get_priority_thresholds() -> dict[str, float]:
    cfg = (get_priority_scoring_config() or {}).get("priority_scoring") or {}
    th = dict(cfg.get("thresholds") or {})
    return {
        "high": float(th.get("high", 2.4)),
        "moderate": float(th.get("moderate", 0.9)),
        "low": float(th.get("low", 0.2)),
    }


def get_analyte_family(analyte: str) -> str | None:
    probe = _norm(analyte)
    fam = dict((get_analyte_families_config() or {}).get("families") or {})
    for name, spec in fam.items():
        terms = [_norm(t) for t in list((spec or {}).get("analytes") or []) if str(t).strip()]
        if any(t and t in probe for t in terms):
            return str(name)
    return None


def get_family_weight(analyte: str) -> float:
    fam = dict((get_analyte_families_config() or {}).get("families") or {})
    family = get_analyte_family(analyte)
    if not family:
        return 0.0
    return float(dict(fam.get(family, {}) or {}).get("weight", 0.0) or 0.0)


def _is_complex_reference_text(ref: str) -> bool:
    rn = _norm(ref or "")
    if not rn:
        return True
    complex_markers = ["homme", "femme", "age", "ans", "enfant", "nourrisson", "adulte", "nouveau", "risque", "souhaitable", "modere", "eleve", "tres", ":"]
    nums = re.findall(r"\d+(?:[.,]\d+)?", ref or "")
    return len(nums) != 2 or any(m in rn for m in complex_markers)


def _recompute_simple_status(value_raw: str, reference_raw: str) -> str | None:
    val = _to_float(value_raw)
    if val is None:
        return None
    ref = str(reference_raw or "").strip()
    if not ref or _is_complex_reference_text(ref):
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
    intervals = re.findall(r"(\d+(?:[.,]\d+)?)\s*(?:-|à|a)\s*(\d+(?:[.,]\d+)?)", str(reference_raw or ""), flags=re.IGNORECASE)
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
    refn = _norm(reference_raw)
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
        if not m or direction != status_code:
            continue
        try:
            threshold = float(str(m.group(1)).replace(",", "."))
        except Exception:
            continue
        if direction == "above_reference" and val > threshold:
            return (True, label)
        if direction == "below_reference" and val < threshold:
            return (True, label)
    return (False, "")


def compute_priority_score(ev: dict[str, Any]) -> dict[str, Any]:
    cfg = dict((get_priority_scoring_config() or {}).get("priority_scoring") or {})
    ratio_weight = float(cfg.get("ratio_weight", 1.2))
    textual_severity_bonus = float(cfg.get("textual_severity_bonus", 0.6))

    status_code = str(ev.get("technical_status_code") or "").strip().lower()
    if status_code not in {"above_reference", "below_reference"}:
        return {"priority_score": 0.0, "priority_level": "unknown", "priority_reason": "statut non priorisable."}

    analyte = f"{ev.get('analyte') or ''} {ev.get('analyte_norm') or ''}".strip()
    ref = str(ev.get("reference") or ev.get("reference_raw") or "").strip()
    value_raw = str(ev.get("current_value") or "").strip()
    if not ref:
        return {"priority_score": 0.0, "priority_level": "unknown", "priority_reason": "référence indisponible, priorité non déterminable."}

    score = 0.0
    reasons: list[str] = []
    severity_hit, severity_label = _severity_category_hit(value_raw, ref, status_code)
    simple_status = _recompute_simple_status(value_raw, ref)
    if simple_status is None:
        if _is_within_any_inclusive_interval(value_raw, ref):
            return {"priority_score": 0.0, "priority_level": "unknown", "priority_reason": "valeur incluse dans une plage de référence explicite (référence complexe)."}
        if severity_hit:
            score += 2.6
            reasons.append(f"catégorie textuelle de sévérité détectée ({severity_label}).")
        else:
            reasons.append("référence complexe, priorité à confirmer selon profil patient.")
    else:
        if simple_status != status_code:
            return {"priority_score": 0.0, "priority_level": "unknown", "priority_reason": "direction de référence non confirmée par intervalle simple."}
        v = _to_float(value_raw)
        nums = re.findall(r"\d+(?:[.,]\d+)?", ref)
        lo = float(nums[0].replace(",", "."))
        hi = float(nums[1].replace(",", "."))
        ratio = 1.0
        if status_code == "above_reference" and hi > 0 and v is not None:
            ratio = max(1.0, float(v) / hi)
        elif status_code == "below_reference" and v is not None and v > 0:
            ratio = max(1.0, lo / float(v))
        score += (ratio - 1.0) * ratio_weight
        reasons.append(f"écart relatif ≈ x{ratio:.2f} vs borne.")

    txt = _norm(f"{ref} {ev.get('technical_status') or ''}")
    terms = get_textual_severity_terms()
    if any(t in txt for t in terms) and (simple_status is not None or severity_hit):
        score += textual_severity_bonus
        reasons.append("termes de sévérité détectés dans la référence/interprétation.")

    fam_weight = get_family_weight(analyte)
    if fam_weight > 0:
        score += fam_weight
        reasons.append("poids famille biologique technique.")

    return {
        "priority_score": round(max(0.0, score), 4),
        "priority_level": assign_priority_level(score),
        "priority_reason": get_priority_reason(reasons),
    }


def assign_priority_level(score: float) -> str:
    th = get_priority_thresholds()
    if score >= float(th.get("high", 2.4)):
        return "high"
    if score >= float(th.get("moderate", 0.9)):
        return "moderate"
    if score >= float(th.get("low", 0.2)):
        return "low"
    return "unknown"


def get_priority_reason(reasons: list[str]) -> str:
    return " ".join([str(r).strip() for r in reasons if str(r).strip()]) or "priorité technique calculée."


__all__ = [
    "compute_priority_score",
    "assign_priority_level",
    "get_priority_reason",
    "get_analyte_family",
    "get_family_weight",
    "get_priority_thresholds",
    "get_textual_severity_terms",
]
