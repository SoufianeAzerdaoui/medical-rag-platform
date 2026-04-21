from __future__ import annotations

import json
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


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: Any) -> Path:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


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
    parts = normalized.split()
    if len(parts) != 3:
        return None
    day_text, month_text, year_text = parts
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
    if flag in {"", "-", "N"}:
        return None
    return flag


def parse_reference_range(value: str | None) -> dict[str, float | str | None]:
    text = normalize_inline_text(value or "")
    parts = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", "."))
    low = float(parts[0]) if len(parts) >= 1 else None
    high = float(parts[1]) if len(parts) >= 2 else None
    return {
        "text": text or None,
        "low": low,
        "high": high,
    }


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
