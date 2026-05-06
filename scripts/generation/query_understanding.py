from __future__ import annotations

import re
import unicodedata


# canonical analyte_norm -> accepted query/answer aliases
ANALYTE_ALIASES: dict[str, set[str]] = {
    "calcitonine": {"calcitonine"},
    "procalcitonine": {"procalcitonine"},
    "ferritine": {"ferritine"},
    "lithium": {"lithium"},
    "c3": {"c3", "complement c3", "complément c3"},
    "c4": {"c4", "complement c4", "complément c4"},
    "cholesterol_hdl": {"hdl", "cholesterol hdl", "cholestérol hdl", "cholesterol_hdl"},
    "crp": {"crp"},
    "peptide_c": {"peptide c", "peptide_c", "peptide-c"},
    "insuline": {"insuline"},
    "pro_bnp": {"pro bnp", "pro_bnp", "pro-bnp", "probnp"},
    "tshus": {"tshus"},
    "tsh": {"tsh"},
    "acth": {"acth"},
    "troponine": {"troponine", "troponine i", "troponine t"},
    "ace": {"ace"},
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


def detect_doc_summary_intent(query: str) -> dict[str, bool]:
    qn = norm_text(query)

    summary_keywords = [
        "resume",
        "synthese",
        "resultats importants",
        "resultats du rapport",
        "section",
        "anomalies",
        "valeurs anormales",
        "hors reference",
        "important",
    ]
    complete_keywords = [
        "tous",
        "toutes",
        "complet",
        "complete",
        "liste tous",
        "tous les resultats",
        "liste complete",
        "liste complete des resultats",
    ]
    immunoanalyse_keywords = [
        "immunoanalyse",
        "immuno analyse",
    ]
    important_keywords = [
        "important",
        "importants",
        "anomalies",
        "hors reference",
        "anormaux",
        "necessitent une attention technique",
        "necessite une attention technique",
        "attention technique",
    ]

    has_summary_intent = any(k in qn for k in summary_keywords)
    wants_complete = any(k in qn for k in complete_keywords)
    wants_important = any(k in qn for k in important_keywords)
    wants_immunoanalyse = any(k in qn for k in immunoanalyse_keywords)
    wants_above_only = (
        any(k in qn for k in ["superieur", "superieure", "au dessus", "above reference", "above_reference"])
        and "reference" in qn
    )
    wants_below_only = (
        any(k in qn for k in ["inferieur", "inferieure", "en dessous", "below reference", "below_reference"])
        and "reference" in qn
    )
    wants_grouped = ("classe" in qn or "classer" in qn) and ("reference" in qn)
    wants_out_of_reference_focus = (
        "hors reference" in qn
        or "anormaux" in qn
        or "attention technique" in qn
        or wants_above_only
        or wants_below_only
    )
    return {
        "is_summary_intent": has_summary_intent or wants_grouped or wants_above_only or wants_below_only or wants_complete,
        "wants_immunoanalyse_section": wants_immunoanalyse,
        "wants_above_only": wants_above_only,
        "wants_below_only": wants_below_only,
        "wants_grouped": wants_grouped,
        "wants_complete": wants_complete,
        "wants_important": wants_important or (has_summary_intent and not wants_complete),
        "wants_out_of_reference_focus": wants_out_of_reference_focus,
    }
