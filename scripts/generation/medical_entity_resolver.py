from __future__ import annotations

import re
import unicodedata
from typing import Any

try:
    from analyte_aliases import ANALYTE_ALIAS_GROUPS
except Exception:  # pragma: no cover
    from scripts.generation.analyte_aliases import ANALYTE_ALIAS_GROUPS  # type: ignore


_ROW_LABEL_FIELDS: tuple[str, ...] = (
    "analyte_label",
    "display_name",
    "source_analyte",
    "parameter",
    "original_analyte",
    "analyte",
    "analyte_norm",
)

_TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "thyroid": ("thyroid", "thyroide", "thyroïde", "thyroidien", "thyroïdien", "tsh", "t4", "t3", "hyperthyro"),
    "toxicology": ("toxicologie", "toxicology", "pharmacotoxicologie", "toxique", "opiac", "benzodiazep", "cocaine", "amphet"),
    "renal": ("renal", "rénal", "renale", "rénale", "creatinine", "créatinine", "uree", "urée", "dfg"),
    "hepatic": ("hepat", "hépat", "foie", "alat", "asat", "ggt", "bilirub"),
    "inflammation": ("inflammation", "crp", "proteine c reactive", "protéine c réactive"),
}


def _norm_text(value: str) -> str:
    txt = str(value or "").strip().lower().replace("µ", "u")
    txt = txt.replace("–", "-").replace("—", "-").replace("−", "-")
    txt = unicodedata.normalize("NFKD", txt)
    txt = "".join(ch for ch in txt if not unicodedata.combining(ch))
    txt = re.sub(r"[^a-z0-9_\-\s]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def _to_alias_key(value: str) -> str:
    txt = _norm_text(value)
    txt = txt.replace("-", " ").replace("_", " ")
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def _to_canonical_key(value: str) -> str:
    txt = _to_alias_key(value)
    if not txt:
        return ""
    txt = txt.replace(" ", "_")
    txt = txt.replace("valporoique", "valproique")
    return txt


def _build_alias_to_canonical() -> dict[str, str]:
    out: dict[str, str] = {}
    for canonical, aliases in dict(ANALYTE_ALIAS_GROUPS or {}).items():
        canon_key = _to_canonical_key(str(canonical))
        if not canon_key:
            continue
        out[_to_alias_key(canon_key)] = canon_key
        out[_to_alias_key(canon_key.replace("_", " "))] = canon_key
        for alias in list(aliases or []):
            alias_key = _to_alias_key(str(alias))
            if alias_key:
                out[alias_key] = canon_key
    # Safe short aliases not consistently present in source data.
    out.setdefault("creat", "creatinine")
    out.setdefault("creatine", "creatinine")
    out.setdefault("creatine", "creatinine")
    out.setdefault("creatininemie", "creatinine")
    out.setdefault("uricemie", "acide_urique")
    out.setdefault("uric acid", "acide_urique")
    out.setdefault("alp", "phosphatase_alcaline")
    out.setdefault("pal", "phosphatase_alcaline")
    out.setdefault("tsh ultra sensible", "tshus")
    out.setdefault("thyroid stimulating hormone", "tsh")
    out.setdefault("thyreostimuline", "tsh")
    out.setdefault("ft3", "t3_libre")
    out.setdefault("ft4", "t4_libre")
    return out


_ALIAS_TO_CANONICAL = _build_alias_to_canonical()

_ANALYTE_FAMILIES: dict[str, str] = {
    "tsh": "thyroid_tsh",
    "tshus": "thyroid_tsh",
    "t3_libre": "thyroid",
    "t4_libre": "thyroid",
    "anti_tg": "thyroid_antibodies",
    "anti_tpo": "thyroid_antibodies",
    "trak": "thyroid_antibodies",
    "crp": "inflammation",
    "creatinine": "renal",
    "acide_urique": "renal_metabolic",
    "phosphatase_alcaline": "hepatic_bone",
    "asat": "hepatic",
    "alat": "hepatic",
    "ggt": "hepatic",
    "bilirubine": "hepatic",
    "ethanol": "toxicology",
    "amphetamine": "toxicology",
    "benzodiazepine": "toxicology",
    "cocaine": "toxicology",
    "opiaces": "toxicology",
    "phencyclidine": "toxicology",
}

_EQUIVALENT_FAMILY_GROUPS: tuple[set[str], ...] = (
    {"tsh", "tshus"},
    {"ckmb", "cpkmb"},
)


def canonicalize_analyte(text: str) -> str:
    alias_key = _to_alias_key(text)
    if not alias_key:
        return ""
    if alias_key in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[alias_key]
    return _to_canonical_key(alias_key)


def get_aliases_for_canonical(analyte: str) -> set[str]:
    canonical = canonicalize_analyte(analyte)
    if not canonical:
        return set()
    out: set[str] = {canonical, canonical.replace("_", " ")}
    aliases = list((ANALYTE_ALIAS_GROUPS or {}).get(canonical, []))
    out.update(_to_alias_key(a) for a in aliases if str(a).strip())
    for alias, canon in _ALIAS_TO_CANONICAL.items():
        if canon == canonical:
            out.add(alias)
    return {x for x in out if x}


def are_equivalent_analytes(left: str, right: str) -> bool:
    l_key = canonicalize_analyte(left)
    r_key = canonicalize_analyte(right)
    if not l_key or not r_key:
        return False
    if l_key == r_key:
        return True
    for group in _EQUIVALENT_FAMILY_GROUPS:
        if l_key in group and r_key in group:
            return True
    return False


def is_analyte_match(requested_analyte: str, evidence_row: dict[str, Any]) -> bool:
    requested_key = canonicalize_analyte(requested_analyte)
    if not requested_key:
        return False
    row_values = [str(evidence_row.get(field) or "") for field in _ROW_LABEL_FIELDS]
    for raw in row_values:
        row_key = canonicalize_analyte(raw)
        if not row_key:
            continue
        if row_key == requested_key or are_equivalent_analytes(requested_key, row_key):
            return True
    return False


def _clean_display_label(value: str) -> str:
    label = str(value or "").strip()
    if not label:
        return ""
    label = re.sub(r"\s+", " ", label).strip(" -;,:")
    return label


def get_display_analyte_label(evidence_row: dict[str, Any]) -> str:
    for field in _ROW_LABEL_FIELDS:
        label = _clean_display_label(str(evidence_row.get(field) or ""))
        if label and label.lower() != "non precise" and label.lower() != "non précisé":
            if "tshus" in _norm_text(label):
                return "TSHus"
            return label
    canonical = canonicalize_analyte(str(evidence_row.get("analyte_norm") or evidence_row.get("analyte") or ""))
    if canonical:
        return canonical.replace("_", " ").upper()
    return "non précisé"


def get_analyte_family(analyte: str) -> str | None:
    canonical = canonicalize_analyte(analyte)
    if not canonical:
        return None
    if canonical in _ANALYTE_FAMILIES:
        return _ANALYTE_FAMILIES[canonical]
    if canonical.startswith("anti_"):
        return "immunology_antibodies"
    return None


def resolve_medical_topic(query: str) -> list[str]:
    qn = _norm_text(query)
    if not qn:
        return []
    topics: list[str] = []
    for topic, markers in _TOPIC_KEYWORDS.items():
        if any(marker in qn for marker in markers):
            topics.append(topic)
    return list(dict.fromkeys(topics))


def find_compatible_evidence_rows(
    requested_analytes: list[str],
    evidence_rows: list[dict],
    scope_doc_ids: list[str] = None,
) -> dict:
    """
    Identifie et retourne les lignes de données (evidence rows) compatibles
    avec les analytes demandés par l'utilisateur.
    
    Le matching utilise une CASCADE DE PRIORITÉS (pas OR parallèle) :
      Priority 1 (CERTAIN) : Exact Match + Alias Match via is_analyte_match()
      Priority 2 (SÛR)     : Family Match (ex: TSH et TSHus = même family)
      Priority 3 (FALLBACK): Topic Match (ex: "bilan rénal" cherche "creatinine", "urée")
    
    Args:
        requested_analytes : list[str]
            Analytes demandés par l'utilisateur (ex: ["creatinine", "tsh"])
        
        evidence_rows : list[dict]
            Toutes les lignes d'evidence disponibles, chacune contenant:
            - "analyte", "analyte_norm", "analyte_label", "display_name", etc.
            - "doc_id" (optionnel, pour scope filtering)
            - "current_value", "unit", etc.
        
        scope_doc_ids : list[str] | None
            OPTIONNEL — Si fourni, ne matcher que sur ces doc_ids.
            Ex: Query "créat report 29" → scope_doc_ids=["report_29"]
            Ignore les matches dans report_10, report_16, etc.
    
    Returns:
        dict {
            "found_rows": list[dict],        # Lignes compatibles trouvées
            "not_found_analytes": list[str], # Analytes NON trouvés
            "partially_found": bool,         # True si some pero not all trouvés
            "confidence_score": float,       # (found / total requested) ratio
            "matching_strategy": str,        # "exact" | "alias" | "family" | "topic" | "mixed"
        }
    """
    
    if not requested_analytes or not evidence_rows:
        return {
            "found_rows": [],
            "not_found_analytes": requested_analytes or [],
            "partially_found": False,
            "confidence_score": 0.0,
            "matching_strategy": "none"
        }
    
    compatible_rows = []
    found_analyte_keys = set()  # Track which requested_analytes were found
    matching_strategies_used = set()
    
    # ============================================================
    # PRIORITY 1 & 2: Exact Match + Alias Match
    # ============================================================
    for req_analyte in requested_analytes:
        found_in_priority_1 = False
        req_canonical = canonicalize_analyte(req_analyte)
        
        for row in evidence_rows:
            # Scope filtering: Skip if doc_id not in scope
            if scope_doc_ids and row.get("doc_id") not in scope_doc_ids:
                continue
            
            # Check all possible analyte label fields (per Phase 1 spec)
            for field in ("analyte_norm", "analyte", "analyte_label", "display_name", 
                         "source_analyte", "parameter", "original_analyte"):
                row_value = row.get(field, "")
                if not row_value:
                    continue
                
                # ⚠️ GUARDRAIL: Ne pas matcher sur parameter seul si analyte principal différent
                if field in ("parameter", "source_analyte", "original_analyte"):
                    main_analyte = row.get("analyte_norm") or row.get("analyte")
                    if main_analyte:
                        main_canon = canonicalize_analyte(main_analyte)
                        if main_canon and main_canon != req_canonical and not are_equivalent_analytes(main_canon, req_canonical):
                            continue
                
                # Strategy 1.A: is_analyte_match() (handles aliases)
                if is_analyte_match(req_analyte, {field: row_value}):
                    compatible_rows.append(row)
                    found_analyte_keys.add(req_analyte)
                    row_canon = canonicalize_analyte(row_value)
                    if row_canon and req_canonical and row_canon != req_canonical and are_equivalent_analytes(req_canonical, row_canon):
                        matching_strategies_used.add("family")
                    else:
                        matching_strategies_used.add("alias")
                    found_in_priority_1 = True
                    break
                
                # Strategy 1.B: Exact canonical match
                row_canonical = canonicalize_analyte(row_value)
                if row_canonical and req_canonical and req_canonical == row_canonical:
                    compatible_rows.append(row)
                    found_analyte_keys.add(req_analyte)
                    matching_strategies_used.add("exact")
                    found_in_priority_1 = True
                    break
            
            if found_in_priority_1:
                break
    
    # ============================================================
    # PRIORITY 2: Family Match
    # ============================================================
    # ONLY for analytes not yet found, to avoid duplicates
    for req_analyte in requested_analytes:
        if req_analyte in found_analyte_keys:
            continue  # Already found in Priority 1, skip
        
        req_family = get_analyte_family(req_analyte)
        if not req_family:
            continue
        
        found_in_priority_2 = False
        for row in evidence_rows:
            # Scope filtering
            if scope_doc_ids and row.get("doc_id") not in scope_doc_ids:
                continue
            
            row_analyte = str(row.get("analyte_norm") or row.get("analyte") or "").strip()
            if not row_analyte:
                continue
            
            row_family = get_analyte_family(row_analyte)
            
            # ⚠️ GUARDRAIL: Family match only if SAME ANALYTE essentially
            if req_family == row_family:
                req_canonical = canonicalize_analyte(req_analyte)
                row_canonical = canonicalize_analyte(row_analyte)
                
                if (req_canonical and row_canonical and 
                    (req_canonical in row_canonical or row_canonical in req_canonical)):
                    compatible_rows.append(row)
                    found_analyte_keys.add(req_analyte)
                    matching_strategies_used.add("family")
                    found_in_priority_2 = True
                    break
        
        if not found_in_priority_2:
            pass
    
    # ============================================================
    # PRIORITY 3: Topic Match (FALLBACK ONLY)
    # ============================================================
    # ONLY if requested is a TOPIC query, not a specific analyte query
    for req_analyte in requested_analytes:
        if req_analyte in found_analyte_keys:
            continue  # Already found
        
        req_topics = resolve_medical_topic(req_analyte)
        if not req_topics:
            continue  # No topic, can't do topic match
        
        found_in_priority_3 = False
        for row in evidence_rows:
            # Scope filtering
            if scope_doc_ids and row.get("doc_id") not in scope_doc_ids:
                continue
            
            row_analyte = str(row.get("analyte_norm") or row.get("analyte") or "").strip()
            if not row_analyte:
                continue
            
            row_topics = resolve_medical_topic(row_analyte)
            
            # Topic match: intersection of topics
            if req_topics and row_topics and set(req_topics).intersection(row_topics):
                if len(req_canonical := canonicalize_analyte(req_analyte)) > 10:
                    compatible_rows.append(row)
                    found_analyte_keys.add(req_analyte)
                    matching_strategies_used.add("topic")
                    found_in_priority_3 = True
                    break
    
    # ============================================================
    # Compute results
    # ============================================================
    not_found = [req for req in requested_analytes if req not in found_analyte_keys]
    partially_found = len(found_analyte_keys) > 0 and len(not_found) > 0
    
    confidence_score = (len(found_analyte_keys) / len(requested_analytes)) if requested_analytes else 0.0
    
    # Choose primary strategy (for debugging)
    if "exact" in matching_strategies_used:
        primary_strategy = "exact"
    elif "alias" in matching_strategies_used:
        primary_strategy = "alias"
    elif "family" in matching_strategies_used:
        primary_strategy = "family"
    elif "topic" in matching_strategies_used:
        primary_strategy = "topic"
    else:
        primary_strategy = "none"
    
    return {
        "found_rows": compatible_rows,
        "not_found_analytes": not_found,
        "partially_found": partially_found,
        "confidence_score": confidence_score,
        "matching_strategy": primary_strategy
    }


__all__ = [
    "are_equivalent_analytes",
    "canonicalize_analyte",
    "get_aliases_for_canonical",
    "get_analyte_family",
    "get_display_analyte_label",
    "is_analyte_match",
    "resolve_medical_topic",
    "find_compatible_evidence_rows"
]
