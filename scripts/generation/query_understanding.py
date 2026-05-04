from __future__ import annotations

import re
import unicodedata


ANALYTE_LEXICON = {
    "calcitonine",
    "ferritine",
    "lithium",
    "crp",
    "procalcitonine",
    "peptide c",
    "insuline",
    "pro bnp",
    "tshus",
    "acth",
    "tsh",
    "vitamine d",
    "trichuris",
    "ankylostoma",
}


def norm_text(value: str) -> str:
    s = (value or "").strip().lower().replace("µ", "u")
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


def detect_exact_analyte(query: str) -> str | None:
    qn = norm_text(query)
    for analyte in sorted(ANALYTE_LEXICON, key=len, reverse=True):
        if contains_exact_term(qn, analyte):
            return analyte
    tokens = [t for t in qn.split(" ") if t]
    if len(tokens) <= 2 and qn in ANALYTE_LEXICON:
        return qn
    return None


def find_analyte_mentions(text: str) -> set[str]:
    body = norm_text(text)
    found: set[str] = set()
    for analyte in ANALYTE_LEXICON:
        if contains_exact_term(body, analyte):
            found.add(analyte)
    return found

