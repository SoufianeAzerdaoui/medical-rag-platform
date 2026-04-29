from __future__ import annotations

import re
import unicodedata


_KNOWN_PARASITE_GENERA: set[str] = {
    "ankylostoma",
    "ancylostoma",
    "trichuris",
    "ascaris",
    "giardia",
    "entamoeba",
    "taenia",
    "hymenolepis",
    "enterobius",
}


def _normalize_text(value: str) -> str:
    text = value.strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text


def _extract_entities_from_result_text(text: str) -> set[str]:
    normalized = _normalize_text(text)
    entities: set[str] = set()

    for match in re.finditer(r"\b([a-z]{3,})\s+([a-z]{3,})\b", normalized):
        genus, species = match.group(1), match.group(2)
        if genus not in _KNOWN_PARASITE_GENERA:
            continue
        entities.add(f"{genus} {species}")

    for genus in _KNOWN_PARASITE_GENERA:
        if re.search(rf"\b{re.escape(genus)}\b", normalized):
            entities.add(genus)

    return entities


def _collect_section_entities(document: dict, section: str) -> list[str]:
    found: set[str] = set()
    for result in document.get("results", []) or []:
        if not isinstance(result, dict):
            continue
        if result.get("section") != section:
            continue
        value = result.get("result")
        if isinstance(value, str) and value.strip():
            found |= _extract_entities_from_result_text(value)
    return sorted(found)


def detect_result_consistency(document: dict) -> dict:
    """
    Analyze results and detect inconsistencies between sections.
    Returns a consistency_checks dict.
    """
    sections = ["staining_exam", "enrichment_exam", "microscopic_exam", "final_result"]
    detected = {section: _collect_section_entities(document, section) for section in sections}

    final_entities = set(detected.get("final_result", []))
    prior_entities: set[str] = set()
    for section in ("staining_exam", "enrichment_exam", "microscopic_exam"):
        prior_entities.update(detected.get(section, []))

    status = "consistent"
    message = "No parasite contradiction detected."
    if final_entities and prior_entities and final_entities.isdisjoint(prior_entities):
        status = "inconsistent"
        message = (
            "Potential contradiction: final_result identifies parasite(s) that differ from parasite(s) "
            "detected in earlier sections."
        )

    return {
        "parasite_consistency": {
            "status": status,
            "detected_in_sections": detected,
            "message": message,
            "check_type": "rule_based_extraction_validation",
        }
    }

