from __future__ import annotations

from typing import Any


def _age_to_years(age: float | None, unit: str | None) -> float | None:
    if age is None:
        return None
    u = str(unit or "years").strip().lower()
    if u == "days":
        return age / 365.0
    if u == "months":
        return age / 12.0
    return age


def _matches_age_range(r: dict[str, Any], target_age_years: float | None) -> bool:
    if target_age_years is None:
        return True
    age_min = _age_to_years(r.get("age_min"), r.get("age_unit"))
    age_max = _age_to_years(r.get("age_max"), r.get("age_unit"))
    op = str(r.get("age_operator") or "").strip()
    age_value = _age_to_years(r.get("age_value"), r.get("age_unit"))
    if age_min is not None and age_max is not None:
        return age_min <= target_age_years <= age_max
    if op and age_value is not None:
        if op == ">":
            return target_age_years > age_value
        if op == ">=":
            return target_age_years >= age_value
        if op == "<":
            return target_age_years < age_value
        if op == "<=":
            return target_age_years <= age_value
    return True


def _score_range(r: dict[str, Any], profile: dict[str, Any]) -> float:
    score = 0.0
    req_sex = str(profile.get("sex") or "").strip().lower()
    req_population = str(profile.get("population") or "").strip().lower()
    req_condition = str(profile.get("condition") or "").strip().lower()
    req_age = profile.get("age")
    req_age_min = profile.get("age_min")
    req_age_max = profile.get("age_max")
    req_age_operator = str(profile.get("age_operator") or "").strip()
    req_age_unit = profile.get("age_unit") or "years"
    target_age = _age_to_years(float(req_age), str(req_age_unit)) if req_age not in (None, "") else None
    target_age_min = _age_to_years(float(req_age_min), str(req_age_unit)) if req_age_min not in (None, "") else None
    target_age_max = _age_to_years(float(req_age_max), str(req_age_unit)) if req_age_max not in (None, "") else None

    sex = str(r.get("sex") or "").strip().lower()
    pop = str(r.get("population") or "").strip().lower()
    cond = str(r.get("condition") or "").strip().lower()
    if req_sex:
        if sex == req_sex:
            score += 100.0
        elif sex in {"", "none"}:
            score += 5.0
        elif sex == "any":
            score += 10.0
        else:
            return -999.0
    if req_population:
        if pop == req_population:
            score += 60.0
        elif pop:
            score -= 20.0
    if req_condition:
        if cond == req_condition:
            score += 70.0
        elif cond:
            score -= 25.0
    if target_age is not None:
        if req_age_operator:
            row_op = str(r.get("age_operator") or "").strip()
            row_age_value = _age_to_years(r.get("age_value"), r.get("age_unit"))
            req_age_value = _age_to_years(float(req_age), str(req_age_unit))
            if row_op and row_age_value is not None and req_age_value is not None:
                if row_op == req_age_operator and abs(row_age_value - req_age_value) <= 1e-6:
                    score += 120.0
                else:
                    return -999.0
            elif row_op:
                return -999.0
            elif r.get("age_min") is not None or r.get("age_max") is not None:
                return -999.0
        else:
            if _matches_age_range(r, target_age):
                score += 60.0
            elif r.get("age_min") is not None or r.get("age_max") is not None or r.get("age_operator") is not None:
                return -999.0
    elif target_age_min is not None and target_age_max is not None:
        age_min = _age_to_years(r.get("age_min"), r.get("age_unit"))
        age_max = _age_to_years(r.get("age_max"), r.get("age_unit"))
        if age_min is not None and age_max is not None:
            if not (abs(age_min - target_age_min) < 1e-6 and abs(age_max - target_age_max) < 1e-6):
                return -999.0
            score += 65.0
    if r.get("age_operator") is not None or (r.get("age_min") is not None and r.get("age_max") is not None):
        score += 10.0
    if pop == "adult":
        score += 5.0
    return score


def select_reference_range(
    ranges: list[dict],
    requested_profile: dict | None = None,
    patient_profile: dict | None = None,
    use_patient_profile: bool = False,
) -> dict:
    if not ranges:
        return {
            "status": "no_match",
            "selected": None,
            "candidates": [],
            "fallback": None,
            "reason": "no_ranges_available",
        }
    if len(ranges) == 1 and not requested_profile and not (use_patient_profile and patient_profile):
        return {"status": "selected", "selected": ranges[0], "candidates": [ranges[0]], "fallback": None, "reason": "single_range_only"}

    profile = dict(requested_profile or {})
    if use_patient_profile and isinstance(patient_profile, dict):
        profile = {
            "sex": profile.get("sex") or patient_profile.get("sex"),
            "age": profile.get("age") if profile.get("age") is not None else patient_profile.get("age"),
            "age_unit": profile.get("age_unit") or patient_profile.get("age_unit") or "years",
            "population": profile.get("population") or patient_profile.get("population"),
        }

    has_constraints = any(profile.get(k) not in (None, "") for k in ["sex", "age", "population", "condition", "age_operator", "age_value", "age_min", "age_max"])
    if not has_constraints:
        return {
            "status": "ambiguous",
            "selected": None,
            "candidates": ranges[:8],
            "fallback": None,
            "reason": "multiple_ranges_without_profile",
        }
    req_population = str(profile.get("population") or "").strip().lower()
    req_condition = str(profile.get("condition") or "").strip().lower()
    req_sex = str(profile.get("sex") or "").strip().lower()
    has_age_constraint = any(profile.get(k) not in (None, "") for k in ["age", "age_operator", "age_value", "age_min", "age_max"])
    if (req_population or req_sex or req_condition) and not has_age_constraint:
        filtered: list[dict[str, Any]] = []
        for r in ranges:
            pop = str(r.get("population") or "").strip().lower()
            sex = str(r.get("sex") or "").strip().lower()
            cond = str(r.get("condition") or "").strip().lower()
            if req_population and pop != req_population:
                continue
            if req_sex and sex not in {"", "any", req_sex}:
                continue
            if req_condition and cond != req_condition:
                continue
            filtered.append(r)
        age_specific = [r for r in filtered if (r.get("age_min") is not None and r.get("age_max") is not None) or r.get("age_operator") is not None]
        if (req_population or req_condition or req_sex) and len(age_specific) >= 2:
            return {
                "status": "grouped_options",
                "selected": None,
                "candidates": age_specific[:10],
                "fallback": None,
                "reason": "population_matched_age_required",
            }

    scored = []
    for r in ranges:
        s = _score_range(r, profile)
        if s > -900:
            scored.append((s, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    if scored:
        top_score, top = scored[0]
        if len(scored) > 1 and abs(top_score - scored[1][0]) < 0.001:
            return {
                "status": "ambiguous",
                "selected": None,
                "candidates": [x[1] for x in scored[:6]],
                "fallback": None,
                "reason": "multiple_equivalent_matches",
            }
        return {
            "status": "selected",
            "selected": top,
            "candidates": [x[1] for x in scored[:6]],
            "fallback": None,
            "reason": "best_profile_match",
        }

    fallback = next((r for r in ranges if str(r.get("population") or "").strip().lower() == "adult"), None)
    if fallback is not None:
        return {
            "status": "fallback",
            "selected": fallback,
            "candidates": ranges[:6],
            "fallback": fallback,
            "reason": "no_specific_match_using_adult_fallback",
        }
    return {
        "status": "no_match",
        "selected": None,
        "candidates": ranges[:6],
        "fallback": None,
        "reason": "no_profile_match",
    }
