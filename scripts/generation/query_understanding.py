from __future__ import annotations

import re
import unicodedata


# canonical analyte_norm -> accepted query/answer aliases
ANALYTE_ALIASES: dict[str, set[str]] = {
    "calcitonine": {"calcitonine"},
    "procalcitonine": {"procalcitonine"},
    "ferritine": {"ferritine"},
    "lithium": {"lithium"},
    "crp": {"crp"},
    "peptide_c": {"peptide c", "peptide_c", "peptide-c"},
    "insuline": {"insuline"},
    "pro_bnp": {"pro bnp", "pro_bnp", "pro-bnp", "probnp"},
    "tshus": {"tshus"},
    "tsh": {"tsh"},
    "acth": {"acth"},
    "vitamine_b12": {"vitamine b12", "vitamine_b12", "vit b12", "b12"},
    "vitamine_d": {"vitamine d", "vitamine_d"},
    "trichuris": {"trichuris", "trichuris trichiura"},
    "ankylostoma": {"ankylostoma"},
}


def norm_text(value: str) -> str:
    s = (value or "").strip().lower().replace("µ", "u").replace("_", " ")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9_\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def contains_exact_term(haystack: str, needle: str) -> bool:
    hay = norm_text(haystack)
    ned = norm_text(needle)
    if not hay or not ned:
        return False
    return f" {ned} " in f" {hay} "


def detect_exact_analytes(query: str) -> list[str]:
    qn = norm_text(query)
    found: list[str] = []
    for canonical, aliases in ANALYTE_ALIASES.items():
        for alias in sorted(aliases, key=len, reverse=True):
            if contains_exact_term(qn, alias):
                found.append(canonical)
                break
    return found


def detect_exact_analyte(query: str) -> str | None:
    found = detect_exact_analytes(query)
    return found[0] if found else None


def find_analyte_mentions(text: str) -> set[str]:
    body = norm_text(text)
    found: set[str] = set()
    for canonical, aliases in ANALYTE_ALIASES.items():
        for alias in aliases:
            if contains_exact_term(body, alias):
                found.add(canonical)
                break
    return found
