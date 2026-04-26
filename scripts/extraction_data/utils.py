from __future__ import annotations

import json
import math
import re
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any


FRENCH_MONTHS = {
    "janv": 1,
    "janvier": 1,
    "fev": 2,
    "fevr": 2,
    "fevrier": 2,
    "mars": 3,
    "avr": 4,
    "avril": 4,
    "mai": 5,
    "juin": 6,
    "juil": 7,
    "juillet": 7,
    "aout": 8,
    "sept": 9,
    "septembre": 9,
    "oct": 10,
    "octobre": 10,
    "nov": 11,
    "novembre": 11,
    "dec": 12,
    "decembre": 12,
}

UUID_LIKE_PATTERN = re.compile(
    r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$"
)

KNOWN_RESULT_UNITS = {
    "%",
    "cm",
    "fL",
    "g/dL",
    "kg",
    "mg/dL",
    "mm[Hg]",
    "mmol/L",
    "10*3/uL",
}

PAGE_BOILERPLATE_PATTERNS = (
    "document_synthetique",
    "usage_exclusif_pour_tests_ocr",
    "parsing_retrieval_multimodal",
    "dossier_patient_synthetique_multimodal",
    "communicative_health_care_associates",
)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: Any) -> Path:
    ensure_dir(path.parent)
    path.write_text(json.dumps(sanitize_json_data(data), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return path


def sanitize_json_data(data: Any) -> Any:
    if isinstance(data, dict):
        return {key: sanitize_json_data(value) for key, value in data.items()}
    if isinstance(data, list):
        return [sanitize_json_data(item) for item in data]
    if isinstance(data, tuple):
        return [sanitize_json_data(item) for item in data]
    if isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
        return data
    return data


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_inline_text(text: str) -> str:
    return clean_text(text).replace("\n", " ").strip()


def safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "document"


def page_name(page_number: int) -> str:
    return f"page_{page_number:03d}"


def is_meaningful_text(text: str, minimum_chars: int = 30) -> bool:
    return len(clean_text(text)) >= minimum_chars


def optional_import(module_name: str):
    try:
        return __import__(module_name)
    except ImportError:
        return None


def normalize_label(value: str) -> str:
    value = normalize_inline_text(value).lower()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def parse_iso_date(value: str | None) -> str | None:
    if not value:
        return None
    raw = normalize_inline_text(value).lower().strip(". ")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        return raw

    normalized = unicodedata.normalize("NFKD", raw)
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.replace(".", "")
    match = re.search(r"(\d{1,2})\s+([a-z]+)\s+(\d{4})", normalized)
    if not match:
        return None
    day_text, month_text, year_text = match.groups()
    if not (day_text.isdigit() and year_text.isdigit()):
        return None

    month = FRENCH_MONTHS.get(month_text)
    if month is None:
        return None

    try:
        parsed = date(int(year_text), month, int(day_text))
    except ValueError:
        return None
    return parsed.isoformat()


def parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    match = re.search(r"-?\d+", normalize_inline_text(value))
    return int(match.group(0)) if match else None


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    cleaned = normalize_inline_text(value).replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    return float(match.group(0)) if match else None


def normalize_flag(value: str | None) -> str | None:
    if value is None:
        return None
    flag = normalize_inline_text(value).upper()
    if flag in {"H", "L"}:
        return flag
    if flag in {"", "-", "N", "NAN", "NONE", "NULL", ":" , ";", "."}:
        return None
    return flag


def parse_reference_range(value: str | None) -> dict[str, float | str | None]:
    text = normalize_inline_text(value or "")
    normalized = text.replace(",", ".")
    low = None
    high = None
    range_match = re.search(r"(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)", normalized)
    if range_match:
        low = float(range_match.group(1))
        high = float(range_match.group(2))
    else:
        parts = re.findall(r"-?\d+(?:\.\d+)?", normalized)
        low = float(parts[0]) if len(parts) >= 1 else None
        high = float(parts[1]) if len(parts) >= 2 else None
    return {
        "text": text or None,
        "low": low,
        "high": high,
    }


def normalize_result_unit_text(unit: str | None) -> str | None:
    if not unit:
        return None

    normalized = normalize_inline_text(unit).replace(" ", "")
    replacements = {
        "9": "%",
        "9.": "%",
        "9°": "%",
        "10°3/uL": "10*3/uL",
        "10°3/uL,": "10*3/uL",
        "10°3/uL.": "10*3/uL",
        "mg/d": "mg/dL",
        "mg/dl": "mg/dL",
        "mg/dL.": "mg/dL",
        "mg/dl.": "mg/dL",
        "mm{[Hg]": "mm[Hg]",
        "wml:": "mmol/L",
    }
    return replacements.get(normalized, normalized) or None


def strip_page_boilerplate(text: str | None) -> str:
    if not text:
        return ""
    lines: list[str] = []
    for raw_line in clean_text(text).splitlines():
        line = normalize_inline_text(raw_line)
        if not line:
            continue
        label = normalize_label(line)
        if any(pattern in label for pattern in PAGE_BOILERPLATE_PATTERNS):
            continue
        if re.fullmatch(r"page\s+\d+", line, flags=re.IGNORECASE):
            continue
        lines.append(line)
    return clean_text("\n".join(lines))


def canonicalize_uuid_like(value: str | None) -> str | None:
    if not value:
        return None

    cleaned = normalize_inline_text(value).replace(" ", "").lower()
    cleaned = re.sub(r"[^a-z0-9-]", "", cleaned)
    if not cleaned:
        return None

    raw_parts = [part for part in cleaned.split("-") if part]
    if not raw_parts:
        return None

    expected_lengths = [8, 4, 4, 4, 12]
    normalized_parts: list[str] = []
    for index, part in enumerate(raw_parts):
        expected = expected_lengths[index] if index < len(expected_lengths) else None
        candidate = part.replace("o", "0").replace("i", "1").replace("l", "1")
        if expected is not None and len(candidate) > expected:
            candidate = candidate[-expected:] if index == 0 else candidate[:expected]
        normalized_parts.append(candidate)

    if len(normalized_parts) >= 5:
        merged = [
            normalized_parts[0],
            normalized_parts[1],
            normalized_parts[2],
            normalized_parts[3],
            "".join(normalized_parts[4:]),
        ]
    else:
        merged = normalized_parts

    rebuilt: list[str] = []
    for index, expected in enumerate(expected_lengths):
        if index >= len(merged):
            break
        candidate = merged[index]
        if index == len(expected_lengths) - 1 and len(candidate) < expected:
            break
        if len(candidate) > expected:
            candidate = candidate[-expected:] if index == 0 else candidate[:expected]
        rebuilt.append(candidate)

    if len(rebuilt) == 5 and all(len(part) == expected_lengths[index] for index, part in enumerate(rebuilt)):
        return "-".join(rebuilt)

    return re.sub(r"-+", "-", cleaned).strip("-") or None


def is_valid_uuid_like(value: str | None) -> bool:
    if not value:
        return False
    return bool(UUID_LIKE_PATTERN.fullmatch(value))


def normalize_report_id_text(value: str | None) -> str | None:
    if not value:
        return None
    raw = normalize_inline_text(value).upper()
    cleaned = re.sub(r"\s+", "", raw)
    cleaned = re.sub(r"[^A-Z0-9-]", "", cleaned)
    return cleaned or None


def normalize_sex_text(value: str | None) -> str | None:
    if not value:
        return None
    raw = normalize_label(value)
    mapping = {
        "f": "F",
        "female": "F",
        "feminin": "F",
        "feminin": "F",
        "m": "M",
        "male": "M",
        "masculin": "M",
        "masculine": "M",
    }
    return mapping.get(raw)


def normalize_encounter_type_text(value: str | None) -> str | None:
    if not value:
        return None
    raw = normalize_label(value)
    mapping = {
        "ambulatory": "Ambulatory",
        "wellness": "Wellness",
        "outpatient": "Outpatient",
        "inpatient": "Inpatient",
        "emergency": "Emergency",
        "urgent_care": "Urgent care",
    }
    return mapping.get(raw)


def normalize_specialty_text(value: str | None) -> str | None:
    if not value:
        return None
    normalized = re.sub(r"\s+", " ", normalize_inline_text(value)).strip()
    return normalized.upper() if normalized else None


def _raw_business_value(value: str | None) -> str | None:
    if value is None:
        return None
    raw = normalize_inline_text(str(value))
    return raw or None


def _normalization_status(raw_value: str | None, canonical_value, *, valid: bool) -> str:
    if raw_value in {None, ""}:
        return "missing"
    if not valid:
        return "needs_review"
    if canonical_value == raw_value:
        return "as_extracted"
    return "canonicalized"


def normalize_named_field(field_name: str, value: str | None) -> dict[str, Any]:
    raw_value = _raw_business_value(value)
    canonical_value: Any = raw_value
    valid = raw_value is not None

    if field_name == "patient_id":
        candidate = canonicalize_uuid_like(raw_value)
        valid = is_valid_uuid_like(candidate)
        canonical_value = candidate if valid else raw_value
    elif field_name in {"birth_date", "report_date", "observation_date"}:
        candidate = parse_iso_date(raw_value)
        valid = candidate is not None
        canonical_value = candidate if valid else raw_value
    elif field_name == "report_id":
        candidate = normalize_report_id_text(raw_value)
        valid = bool(candidate and re.fullmatch(r"[A-Z0-9-]+", candidate))
        canonical_value = candidate if valid else raw_value
    elif field_name == "unit":
        candidate = normalize_result_unit_text(raw_value)
        valid = candidate is not None
        canonical_value = candidate if valid else raw_value
    elif field_name == "alert_flag":
        candidate = normalize_flag(raw_value)
        raw_flag = _raw_business_value(value)
        valid = raw_flag in {None, "", "-", "N", "H", "L", "nan", "NaN", "None"}
        canonical_value = candidate
    elif field_name == "sex":
        candidate = normalize_sex_text(raw_value)
        valid = candidate is not None
        canonical_value = candidate if valid else raw_value
    elif field_name == "encounter_type":
        candidate = normalize_encounter_type_text(raw_value)
        valid = candidate is not None
        canonical_value = candidate if valid else raw_value
    elif field_name == "specialty":
        candidate = normalize_specialty_text(raw_value)
        valid = candidate is not None
        canonical_value = candidate if valid else raw_value

    return {
        "raw": raw_value,
        "canonical": canonical_value,
        "normalization_status": _normalization_status(raw_value, canonical_value, valid=valid),
    }


def is_known_result_unit(unit: str | None) -> bool:
    normalized = normalize_result_unit_text(unit)
    return normalized in KNOWN_RESULT_UNITS


def normalize_ocr_analyte_text(text: str | None) -> str:
    if not text:
        return ""

    normalized = normalize_inline_text(text)
    replacements = [
        (r"\(Moles/volume\]", "[Moles/volume]"),
        (r"\bPinema\b", "Plasma"),
        (r"\bP Plasma\b", "Plasma"),
        (r"\bJanv\b", ""),
        (r"\b(?:ae|ys|FANG|earns|seni)\b", ""),
    ]
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

    normalized = re.sub(r"^Automated count\s+", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bra\b$", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s{2,}", " ", normalized)
    normalized = re.sub(r"\s+\]", "]", normalized)
    normalized = re.sub(r"\[\s+", "[", normalized)
    normalized = re.sub(r"\s+([,/])", r"\1", normalized)
    return normalized.strip(" -:;,./")


def _format_repaired_value(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def repair_numeric_with_reference(
    value_raw: str | None,
    reference_range: dict[str, float | str | None] | None,
    *,
    flag: str | None = None,
) -> tuple[str | None, float | None]:
    raw = normalize_inline_text(value_raw or "")
    numeric = parse_float(raw)
    if numeric is None:
        return (raw or None), None

    low = reference_range.get("low") if reference_range else None
    high = reference_range.get("high") if reference_range else None
    if low is None or high is None or high < low:
        return (raw or None), numeric
    if "." in raw or "," in raw:
        return (raw or None), numeric
    if numeric <= high:
        return (raw or None), numeric

    best_value = numeric
    best_score = float("-inf")
    for candidate in (numeric, numeric / 10.0, numeric / 100.0):
        score = 0.0
        if low <= candidate <= high:
            score += 6.0
        elif flag == "H" and candidate > high:
            score += 4.0
        elif flag == "L" and candidate < low:
            score += 4.0
        elif (low * 0.8) <= candidate <= (high * 1.2):
            score += 2.0

        distance = 0.0
        if candidate < low:
            distance = low - candidate
        elif candidate > high:
            distance = candidate - high
        score -= distance / max(high - low, 1.0)

        if candidate == numeric:
            score -= 1.0

        if score > best_score:
            best_score = score
            best_value = candidate

    if best_value != numeric and best_score >= 3.0:
        return _format_repaired_value(best_value), best_value

    return (raw or None), numeric


def deduplicate_dicts(items: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        fingerprint = tuple(item.get(key) for key in keys)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduped.append(item)
    return deduped


def bbox_from_sequence(values: Any) -> dict[str, float]:
    x0, y0, x1, y1 = values
    return {
        "x0": round(float(x0), 2),
        "y0": round(float(y0), 2),
        "x1": round(float(x1), 2),
        "y1": round(float(y1), 2),
    }


def bbox_width(bbox: dict[str, float]) -> float:
    return max(0.0, float(bbox["x1"]) - float(bbox["x0"]))


def bbox_height(bbox: dict[str, float]) -> float:
    return max(0.0, float(bbox["y1"]) - float(bbox["y0"]))


def bbox_area(bbox: dict[str, float]) -> float:
    return bbox_width(bbox) * bbox_height(bbox)


def bbox_union(bboxes: list[dict[str, float]]) -> dict[str, float] | None:
    if not bboxes:
        return None
    return {
        "x0": round(min(bbox["x0"] for bbox in bboxes), 2),
        "y0": round(min(bbox["y0"] for bbox in bboxes), 2),
        "x1": round(max(bbox["x1"] for bbox in bboxes), 2),
        "y1": round(max(bbox["y1"] for bbox in bboxes), 2),
    }


def bbox_vertical_distance(a: dict[str, float], b: dict[str, float]) -> float:
    if a["y1"] < b["y0"]:
        return b["y0"] - a["y1"]
    if b["y1"] < a["y0"]:
        return a["y0"] - b["y1"]
    return 0.0


def bbox_horizontal_overlap_ratio(a: dict[str, float], b: dict[str, float]) -> float:
    overlap = max(0.0, min(a["x1"], b["x1"]) - max(a["x0"], b["x0"]))
    base = max(1.0, min(bbox_width(a), bbox_width(b)))
    return overlap / base


def bbox_contains_y(bbox: dict[str, float], y: float) -> bool:
    return float(bbox["y0"]) <= y <= float(bbox["y1"])
