from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field, replace
from typing import Any

try:
    from analyte_aliases import ANALYTE_ALIAS_GROUPS
    from analyte_resolver import resolve_requested_analytes, normalize_analyte_text
    from medical_entity_resolver import (
        are_equivalent_analytes,
        canonicalize_analyte,
        get_aliases_for_canonical,
        resolve_medical_topic,
    )
except Exception:  # pragma: no cover
    from scripts.generation.analyte_aliases import ANALYTE_ALIAS_GROUPS  # type: ignore
    from scripts.generation.analyte_resolver import resolve_requested_analytes, normalize_analyte_text  # type: ignore
    from scripts.generation.medical_entity_resolver import (  # type: ignore
        are_equivalent_analytes,
        canonicalize_analyte,
        get_aliases_for_canonical,
        resolve_medical_topic,
    )

# canonical analyte_norm -> accepted query/answer aliases
ANALYTE_ALIASES: dict[str, set[str]] = {
    str(k): {str(v) for v in vals}
    for k, vals in ANALYTE_ALIAS_GROUPS.items()
}


ANALYTE_EXCLUSIONS: dict[str, set[str]] = {
    "tshus": {
        "trak",
        "anticorps anti recepteur de la tsh",
        "anti recepteur de la tsh",
        "anti tsh receptor",
        "anti tg",
        "anticorps thyroidiens",
    },
    "tsh": {
        "trak",
        "anticorps anti recepteur de la tsh",
        "anti recepteur de la tsh",
        "anti tsh receptor",
        "anti tg",
        "anticorps thyroidiens",
    },
}


ANALYTE_DISPLAY_NAMES: dict[str, str] = {
    "tshus": "TSHus",
    "tsh": "TSH",
    "acth": "ACTH",
    "t4_libre": "T4 LIBRE",
    "t3_libre": "T3 LIBRE",
    "anti_tg": "ANTI-TG",
    "ckmb": "CKMB",
    "crp": "CRP",
    "ace": "ACE",
    "psa_totale": "PSA TOTALE",
    "ca_15_3": "CA 15-3",
    "acide_valproique": "ACIDE VALPROIQUE (DEPAKINE)",
    "lithium": "LITHIUM",
    "carbamazepine": "CARBAMAZEPINE",
    "insuline": "INSULINE",
    "troponine": "TROPONINE",
    "procalcitonine": "PROCALCITONINE",
    "calcium": "CALCIUM",
    "creatinine": "CRÉATININE",
    "amh": "AMH",
    "pth_intact": "PTH INTACT",
    "haptoglobine": "HAPTOGLOBINE",
    "phosphatase_alcaline": "PHOSPHATASE ALCALINE",
    "acide_urique": "ACIDE URIQUE",
    "cholesterol_hdl": "CHOLESTÉROL HDL",
    "cholesterol_total": "CHOLESTÉROL TOTAL",
    "benzodiazepine": "BENZODIAZÉPINE",
    "amphetamine": "AMPHÉTAMINE",
    "cocaine": "COCAÏNE",
    "opiaces": "OPIACÉS",
    "phencyclidine": "PHENCYCLIDINE",
}

# ============================================================
# PHASE 3 - deterministic shadow scoring constants
# ============================================================
DOC_SCOPED_INTENTS: set[str] = {
    "doc_scoped_results",
    "doc_scoped_analyte_query",
    "doc_scoped_summary",
    "doc_scoped_biological_summary",
    "doc_scoped_abnormal_results",
    "doc_scoped_priority_anomalies",
    "doc_scoped_medical_interpretation_guarded",
    "toxicology_summary",
    "immunoanalysis_summary",
    "doc_scoped_single_analyte_status",
    "reference_ranges_summary",
}

SINGLE_DOC_INTENTS: set[str] = {
    "doc_scoped_results",
    "doc_scoped_analyte_query",
    "doc_scoped_summary",
    "doc_scoped_biological_summary",
    "doc_scoped_abnormal_results",
    "doc_scoped_priority_anomalies",
    "toxicology_summary",
    "immunoanalysis_summary",
    "reference_ranges_summary",
}

ANALYTE_SCOPED_INTENTS: set[str] = {
    "doc_scoped_analyte_query",
    "single_analyte_lookup",
    "multi_analyte_results",
    "previous_result_comparison",
    "doc_scoped_single_analyte_status",
    "reference_range_lookup",
    "global_analyte_abnormal_search",
    "global_patient_lookup",
}

LEXICAL_MARKERS: dict[str, dict[str, Any]] = {
    "doc_scoped_single_analyte_status": {
        "strong": ["creat", "creatinine", "valeur", "est elle", "basse", "elevee", "normal"],
        "medium": ["rapport", "report", "document", "statut"],
        "weak": ["resultat", "mesure"],
        "weight": 1.2,
    },
    "doc_scoped_results": {
        "strong": ["valeur", "resultat", "est elle", "est il", "rapport"],
        "medium": ["analyte", "statut", "reference"],
        "weak": ["mesure"],
        "weight": 1.0,
    },
    "doc_scoped_biological_summary": {
        "strong": ["synthese", "resume", "anormal", "anomalies", "hors reference"],
        "medium": ["resultats", "biologique", "rapport"],
        "weak": ["bilan"],
        "weight": 1.1,
    },
    "doc_scoped_abnormal_results": {
        "strong": ["anormal", "anormaux", "anomalies", "hors reference"],
        "medium": ["resultats", "rapport"],
        "weak": ["statut"],
        "weight": 1.1,
    },
    "doc_scoped_priority_anomalies": {
        "strong": ["priorite", "important", "attention", "classer"],
        "medium": ["anomalies", "anormal"],
        "weak": ["resultats"],
        "weight": 1.0,
    },
    "reference_range_lookup": {
        "strong": ["plage", "norme", "reference", "intervalle", "physiologique", "normal"],
        "medium": ["homme", "femme", "adulte", "enfant"],
        "weak": ["valeur"],
        "weight": 1.0,
    },
    "reference_ranges_summary": {
        "strong": ["valeurs physiologiques", "plages de reference", "plages physiologiques", "types de references", "references selon"],
        "medium": ["seuil", "sexe", "age", "categories", "intervalle"],
        "weak": ["note", "resume", "synthese"],
        "weight": 1.15,
    },
    "global_analyte_abnormal_search": {
        "strong": ["tous les rapports", "quels rapports", "liste", "cohorte", "patients"],
        "medium": ["hors reference", "anormal", "au dessus", "en dessous"],
        "weak": ["recherche"],
        "weight": 1.0,
    },
    "cohort_search": {
        "strong": ["patients", "liste", "quels rapports", "tous les rapports"],
        "medium": ["superieur", "inferieur", "valeur", "strictement"],
        "weak": ["recherche"],
        "weight": 1.0,
    },
    "small_talk": {
        "strong": ["bonjour", "salut", "hello", "merci"],
        "medium": ["ok", "thanks"],
        "weak": [],
        "weight": 1.0,
    },
}

INTENT_TOPIC_RELEVANCE: dict[str, dict[str, float]] = {
    "doc_scoped_single_analyte_status": {"renal": 0.95, "thyroid": 0.85, "cardio": 0.75, "hepatic": 0.7, "toxicology": 0.6},
    "doc_scoped_results": {"renal": 0.85, "thyroid": 0.85, "cardio": 0.8, "hepatic": 0.8, "toxicology": 0.75, "inflammation": 0.75},
    "doc_scoped_biological_summary": {"renal": 0.75, "thyroid": 0.75, "cardio": 0.75, "hepatic": 0.8, "inflammation": 0.75, "toxicology": 0.6},
    "doc_scoped_abnormal_results": {"renal": 0.85, "thyroid": 0.8, "cardio": 0.85, "hepatic": 0.8, "inflammation": 0.8},
    "reference_range_lookup": {"renal": 0.9, "thyroid": 0.9, "cardio": 0.8, "hepatic": 0.8},
    "reference_ranges_summary": {"renal": 0.85, "thyroid": 0.85, "cardio": 0.8, "hepatic": 0.8, "inflammation": 0.75, "toxicology": 0.7},
    "global_analyte_abnormal_search": {"renal": 0.9, "thyroid": 0.9, "cardio": 0.9, "hepatic": 0.85, "inflammation": 0.8, "toxicology": 0.7},
    "cohort_search": {"renal": 0.85, "thyroid": 0.85, "cardio": 0.85, "hepatic": 0.8, "toxicology": 0.75},
}

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "renal": ["creatinine", "creatinine", "uree", "urique", "renal", "rein", "dfg", "filtration", "glomerulaire"],
    "thyroid": ["tsh", "tshus", "t3", "t4", "thyroid", "thyro", "anti tg", "trak", "anti tpo"],
    "cardio": ["troponine", "bnp", "cholesterol", "hdl", "ldl", "triglyceride", "cardiaque", "coeur"],
    "hepatic": ["asat", "alat", "ggt", "bilirubine", "hepatique", "foie", "transaminase", "phosphatase"],
    "inflammation": ["crp", "inflamm", "leucocyte", "lymphocyte", "pmn"],
    "toxicology": ["toxicologie", "benzodiazepine", "amphetamine", "cocaine", "opiaces", "cannabis", "lithium"],
    "general_biology": ["bilan", "biologique", "resultats", "anomalies", "reference"],
}

ANALYTE_TOPIC_MAP: dict[str, list[str]] = {
    "creatinine": ["renal"],
    "uree": ["renal"],
    "acide_urique": ["renal"],
    "tsh": ["thyroid"],
    "tshus": ["thyroid"],
    "t4_libre": ["thyroid"],
    "t3_libre": ["thyroid"],
    "anti_tg": ["thyroid"],
    "crp": ["inflammation"],
    "troponine": ["cardio"],
    "cholesterol_total": ["cardio"],
    "cholesterol_hdl": ["cardio"],
    "phosphatase_alcaline": ["hepatic"],
}

def analyte_display_name(value: str, analyte_norm: str | None = None) -> str:
    key = str(analyte_norm or value or "").strip().lower().replace(" ", "_")
    key = key.replace("-", "_")
    if key in ANALYTE_DISPLAY_NAMES:
        return ANALYTE_DISPLAY_NAMES[key]
    raw = str(value or "").strip()
    if not raw:
        return ""
    if raw.lower() in ANALYTE_DISPLAY_NAMES:
        return ANALYTE_DISPLAY_NAMES[raw.lower()]
    return raw


@dataclass(frozen=True)
class PresentationIntent:
    requested_output: str
    chart_type: str | None
    raw_format_phrase: str | None
    wants_clickable_sources: bool
    wants_intro: bool
    wants_conclusion: bool
    strict_columns: list[str]
    unsupported_format: bool
    user_requested_visualization: bool
    raw_user_request: str
    unhandled_instructions: list[str]
    presentation_confidence: float
    unsupported_reason: str | None
    recommended_output: str | None
    unsupported_presentation_reason: str | None
    recommended_alternative_format: str | None


@dataclass(frozen=True)
class ResponseStrategy:
    name: str
    reason: str | None
    requires_llm_writer: bool
    allow_fallback: bool


@dataclass(frozen=True)
class QueryUnderstanding:
    requested_doc_ids: list[str]
    requested_analytes: list[str]
    excluded_analytes: list[str]
    requested_value: str | None
    requested_unit: str | None
    comparison_operator: str | None
    source_clickable_requested: bool
    patient_query: bool
    intent: str
    output_format: str
    requested_table_columns: list[str]
    answer_style: str
    requires_global_search: bool
    technical_condition: str | None
    safety_intent: str | None
    requires_previous_results: bool
    requires_comparison: bool
    requires_section_summary: bool
    is_small_talk: bool
    is_response_transform: bool
    language: str
    intents: dict[str, bool]
    presentation_intent: PresentationIntent
    response_strategy: str
    response_strategy_reason: str | None
    original_user_question: str
    raw_user_request: str
    raw_format_phrase: str | None
    unhandled_instructions: list[str]
    presentation_confidence: float
    unsupported_presentation_reason: str | None
    recommended_alternative_format: str | None
    inventory_view_type: str | None
    requested_date_iso: str | None
    requested_report_type: str | None
    latest_report: bool
    requested_context_type: str | None
    qualitative_view_type: str | None
    requested_reference_profile: dict[str, Any] | None
    use_patient_profile: bool
    request_all_reference_ranges: bool
    requested_summary_points: int | None = None
    intent_candidates: list[dict[str, Any]] = field(default_factory=list)
    intent_confidence: float = 0.0
    scope_confidence: float = 0.0
    ambiguity_flags: list[str] = field(default_factory=list)
    medical_topics: list[dict[str, Any]] = field(default_factory=list)


def norm_text(value: str) -> str:
    s = (value or "").strip().lower().replace("µ", "u").replace("_", " ")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9_\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_analyte(text: str) -> str:
    # Keep this helper for broader phrase normalization used outside strict analyte
    # canonicalization (presentation/style parsing). Entity canonical keys should go
    # through medical_entity_resolver.canonicalize_analyte.
    return normalize_analyte_text(text).replace("-", " ")


def detect_language(query: str) -> str:
    qn = norm_text(query or "")
    if any(k in qn for k in ["dans report", "rapport", "document", "reponds", "réponds", "oui non", "tableau"]):
        return "fr"
    if any(k in qn for k in ["in report", "answer yes/no", "table format", "json strict"]):
        return "en"
    return "fr"


def get_analyte_aliases(analyte: str) -> set[str]:
    key = canonicalize_analyte(str(analyte or ""))
    aliases = set(get_aliases_for_canonical(key))
    aliases.add(key)
    if key == "acide_valproique":
        aliases.update(
            {
                "acide valporoique",
                "acide valporoique depakine",
                "acide valproique depakine",
                "depakine",
                "depakine chrono",
                "acide valporoique (depakine)",
                "acide valproique (depakine)",
            }
        )
    if key == "ca_15_3":
        aliases.update({"ca 153", "ca153"})
    if key == "t4_libre":
        aliases.update({"t4", "t4 libre", "ft4"})
    return {normalize_analyte(a) for a in aliases if str(a).strip()}


def get_exclusion_aliases(analyte: str) -> set[str]:
    key = str(analyte or "").strip().lower()
    values = ANALYTE_EXCLUSIONS.get(key, set())
    return {normalize_analyte(v) for v in values if str(v).strip()}


def match_analyte(candidate: str, requested: str) -> bool:
    cand = normalize_analyte(candidate)
    req_key = canonicalize_analyte(str(requested or ""))
    cand_key = canonicalize_analyte(cand)
    if req_key and cand_key and (req_key == cand_key or are_equivalent_analytes(req_key, cand_key)):
        return True
    aliases = get_analyte_aliases(req_key)
    exclusions = get_exclusion_aliases(req_key)

    if any(contains_exact_term(cand, ex) or ex in cand for ex in exclusions):
        return False

    for alias in aliases:
        if contains_exact_term(cand, alias):
            return True
    return False


def contains_exact_term(haystack: str, needle: str) -> bool:
    hay = norm_text(haystack)
    ned = norm_text(needle)
    if not hay or not ned:
        return False
    return f" {ned} " in f" {hay} "


def detect_exact_analytes(query: str) -> list[str]:
    available = [{"display_name": analyte_display_name(k, k), "analyte_norm": k} for k in ANALYTE_ALIASES.keys()]
    resolved = resolve_requested_analytes(query=query, available_analytes=available, aliases=ANALYTE_ALIAS_GROUPS, max_candidates=8)
    if not resolved:
        return []
    selected = [str(r.get("analyte_norm") or "").strip().lower() for r in resolved if str(r.get("analyte_norm") or "").strip()]
    if selected:
        return list(dict.fromkeys(selected))
    return []


def detect_exact_analyte(query: str) -> str | None:
    found = detect_exact_analytes(query)
    return found[0] if found else None


def find_analyte_mentions(text: str) -> set[str]:
    body = normalize_analyte(text)
    found: set[str] = set()
    for canonical, aliases in ANALYTE_ALIASES.items():
        all_aliases = get_analyte_aliases(canonical) | {normalize_analyte(a) for a in aliases}
        for alias in all_aliases:
            if contains_exact_term(body, alias):
                found.add(canonical)
                break
    return found


def detect_alias_resolution_used(query: str, requested_analytes: list[str]) -> bool:
    qn = normalize_analyte(query or "")
    for analyte in requested_analytes:
        canonical = str(analyte or "").strip().lower()
        canonical_phrase = normalize_analyte(canonical.replace("_", " "))
        aliases = get_analyte_aliases(canonical)
        if not aliases:
            continue
        alias_hit = any(contains_exact_term(qn, alias) for alias in aliases if alias)
        canonical_hit = contains_exact_term(qn, canonical_phrase)
        if alias_hit and not canonical_hit:
            return True
    return False


_REPORT_ID_PATTERN = re.compile(
    r"\b(?:report|rapport|document)(?:\s*[_\-])?\s*\(?\s*(\d{1,6})\s*\)?\b",
    flags=re.IGNORECASE,
)


def detect_requested_doc_ids(query: str) -> list[str]:
    text = query or ""
    found: list[str] = []
    seen: set[str] = set()
    for match in _REPORT_ID_PATTERN.finditer(text):
        raw_num = (match.group(1) or "").strip()
        if not raw_num:
            continue
        try:
            doc_id = f"report_{int(raw_num)}"
        except Exception:
            continue
        low = doc_id.lower()
        if low in seen:
            continue
        seen.add(low)
        found.append(doc_id)
    # Handle compact plural forms: "reports 16, 19 et 31"
    for m in re.finditer(
        r"\b(?:reports?|rapports?|documents?)\s+((?:\d{1,6}\s*(?:,|et|ou)\s*)+\d{1,6}|\d{1,6})",
        text,
        flags=re.IGNORECASE,
    ):
        block = str(m.group(1) or "")
        for num in re.findall(r"\d{1,6}", block):
            try:
                doc_id = f"report_{int(num)}"
            except Exception:
                continue
            low = doc_id.lower()
            if low in seen:
                continue
            seen.add(low)
            found.append(doc_id)
    return found


def detect_query_intents(query: str, *, requested_doc_ids: list[str] | None = None, analytes: list[str] | None = None) -> dict[str, bool]:
    qn = norm_text(query or "")
    doc_ids = requested_doc_ids if requested_doc_ids is not None else detect_requested_doc_ids(query or "")
    analyte_list = analytes if analytes is not None else detect_exact_analytes(query or "")
    small_talk_markers = {
        "ok",
        "okay",
        "d accord",
        "d'accord",
        "bonjour",
        "bonsoir",
        "salut",
        "hello",
        "hi",
        "hey",
        "ca va",
        "ça va",
        "cava",
        "cv",
        "merci",
        "thank you",
        "thanks",
        "au revoir",
        "bye",
        "bonne journee",
        "bonne journée",
    }
    identity_markers = {
        "tes qui",
        "t es qui",
        "t'es qui",
        "tu es qui",
        "qui es tu",
        "qui es-tu",
        "vous etes qui",
        "vous êtes qui",
        "c est qui toi",
        "c'est qui toi",
        "who are you",
        "what are you",
    }
    capability_markers = {
        "tu peux faire quoi",
        "que peux tu faire",
        "que peux-tu faire",
        "c est quoi ton role",
        "c'est quoi ton rôle",
        "c est quoi ton rôle",
        "ton role",
        "ton rôle",
        "tu sers a quoi",
        "tu sers à quoi",
        "comment tu peux m aider",
        "comment tu peux m'aider",
        "what can you do",
    }
    help_markers = {
        "aide moi",
        "aide",
        "help",
        "comment utiliser",
        "how to use",
        "guide moi",
    }
    patient_inventory_markers = {
        "liste tous les patients",
        "lister tous les patients",
        "liste tous les patients exist",
        "lister tous les patients exist",
        "patients existants",
        "patients indexes",
        "patients indexés",
        "tous les patients avec sources",
        "donne moi les patients",
        "donne-moi les patients",
        "liste des patients",
        "patients et leurs rapports",
        "patients avec documents",
        "patients avec sources",
        "liste patients",
        "lister patients",
    }
    patient_count_markers = {
        "combien de patients",
        "nombre de patients",
        "count patients",
        "combien de patients sont indexes",
        "combien de patients sont indexés",
    }

    has_compare = any(k in qn for k in ["compare", "comparaison", "versus", "vs"])
    has_summary = any(k in qn for k in ["resume", "synthese", "resume les", "fais une synthese", "liste", "anomalies"])
    has_previous = any(k in qn for k in ["resultat anterieur", "resultats anterieurs", "ancien resultat", "ancienne valeur"])
    has_toxicology = any(
        k in qn
        for k in [
            "toxicologie",
            "pharmacotoxicologie",
            "toxico",
            "ethanol",
            "lithium",
            "valpro",
            "carbamazep",
            "opiaces",
            "benzodiazep",
        ]
    )
    has_immuno = any(k in qn for k in ["immunoanalyse", "immuno analyse"])
    has_comment_keyword = any(
        k in qn
        for k in [
            "valeur mesuree",
            "valeur mesurée",
            "seulement un commentaire",
            "commentaire d interpretation",
            "commentaire",
            "note",
            "interpretation",
            "interprétation",
            "valeur seuil",
        ]
    )
    has_generic_comment_request = has_comment_keyword and any(
        k in qn
        for k in [
            "liste",
            "lister",
            "montre",
            "affiche",
            "donne",
            "retourne",
            "tous les commentaires",
            "toutes les notes",
        ]
    )
    has_qualitative_comment_query = has_comment_keyword and (
        len(analyte_list) >= 1
        or "ce commentaire" in qn
        or "ce resultat" in qn
        or "ce résultat" in qn
        or has_generic_comment_request
    )
    has_structured_status_request = any(
        k in qn
        for k in [
            "hors reference",
            "hors de la reference",
            "hors référence",
            "above_reference",
            "below_reference",
            "above reference",
            "below reference",
            "statut technique",
            "parametre",
            "paramètre",
            "valeur",
            "reference",
            "référence",
            "resultats",
            "résultats",
        ]
    )
    has_global_scope_markers = any(
        k in qn
        for k in [
            "tous les rapports",
            "tous les documents",
            "quels documents",
            "quels rapports",
            "y a t il des rapports",
            "y a-t-il des rapports",
            "patients qui ont",
            "rapports disponibles",
            "rapports indexes",
            "rapports indexés",
            "sur l ensemble des rapports",
            "sur l’ensemble des rapports",
            "retrouve tous les cas",
            "dans les documents",
            "dans tous les rapports",
            "dans les rapports disponibles",
        ]
    )
    has_abnormal_wording = any(
        k in qn
        for k in [
            "hors reference",
            "hors de la reference",
            "hors normes",
            "hors norme",
            "hors intervalle",
            "anomalie",
            "anomalies",
            "anormal",
            "anormaux",
            "basse",
            "bas",
            "diminuee",
            "diminuée",
            "inferieure",
            "supérieure",
            "superieure",
            "elevee",
            "élevée",
            "elevation",
            "élévation",
            "au dessus",
            "au-dessus",
            "above_reference",
            "below_reference",
            "above reference",
            "below reference",
        ]
    )
    reference_markers = [
        "plage",
        "norme",
        "normes",
        "valeur normale",
        "valeurs normales",
        "valeurs physiologiques",
        "intervalle de reference",
        "intervalle physiologique",
        "taux normal",
        "fourchette normale",
        "limites normales",
    ]
    profile_markers = ["homme", "femme", "adulte", "enfant", "nourrisson", "nouveau ne", "ans", "mois", "jours", ">"]
    has_reference_semantic = (
        any(k in qn for k in reference_markers)
        or ("reference" in qn and any(k in qn for k in ["homme", "femme", "adulte", "enfant", "nourrisson", "nouveau ne"]))
    )
    has_reference_range_lookup = (
        len(analyte_list) >= 1
        and (has_reference_semantic or "pour ce patient" in qn or "selon ce patient" in qn)
        and (
            any(k in qn for k in profile_markers)
            or "plage du" in qn
            or "plage de" in qn
            or "norme" in qn
            or "pour ce patient" in qn
            or "selon ce patient" in qn
            or "toutes les plages" in qn
        )
    )
    reference_ranges_core_markers = [
        "plage",
        "plages",
        "reference",
        "référence",
        "references",
        "références",
        "intervalle",
        "intervalles",
        "norme",
        "normes",
        "seuil",
        "seuils",
        "types de references",
        "types de références",
    ]
    physiological_stems = ["physiolog", "phisiolog"]
    note_style_markers = [
        "note",
        "resume",
        "résume",
        "synthese",
        "synthèse",
        "classe",
        "classer",
        "categorie",
        "catégorie",
        "categories",
        "catégories",
        "types",
    ]
    profile_axes_markers = [
        "selon sexe",
        "selon age",
        "selon âge",
        "homme",
        "femme",
        "enfant",
        "adulte",
        "nourrisson",
        "nouveau ne",
        "nouveau-né",
    ]
    has_reference_ranges_semantic = (
        any(k in qn for k in reference_ranges_core_markers)
        and (
            any(stem in qn for stem in physiological_stems)
            or any(k in qn for k in profile_axes_markers)
            or "valeurs de reference" in qn
            or "valeurs de référence" in qn
        )
    )
    has_reference_ranges_summary = (
        len(doc_ids) >= 1
        and has_reference_ranges_semantic
        and any(k in qn for k in note_style_markers)
        and len(analyte_list) <= 1
    )
    has_no_diagnostic_constraint = any(
        k in qn
        for k in [
            "ne donne pas de diagnostic",
            "sans diagnostic",
            "pas de diagnostic",
            "sans poser de diagnostic",
        ]
    )
    has_treatment_safety = any(
        k in qn
        for k in [
            "traitement",
            "donne le traitement",
            "donner le traitement",
            "quel traitement",
            "quels traitements",
            "recommande",
            "recommander",
            "prescrire",
            "prescription",
            "posologie",
        ]
    )
    has_diagnostic_safety = (not has_no_diagnostic_constraint) and (("cancer" in qn) or any(
        k in qn
        for k in [
            "peut on conclure",
            "peut-on conclure",
            "conclure a",
            "conclure à",
            "diagnostic definitif",
            "diagnostic définitif",
            "diagnostic",
            "est ce que le patient a",
            "est-ce que le patient a",
            "le patient a quoi",
            "hyperthyroid",
            "hyperthyro",
            "hypothyroid",
            "hypothyro",
        ]
    ))
    has_doc_scope = len(doc_ids) >= 1
    # Guardrail: avoid misrouting structured status queries to qualitative-comment intent
    # when the user says "sans interprétation médicale" or similar phrasing.
    if has_qualitative_comment_query and has_doc_scope and has_structured_status_request:
        has_qualitative_comment_query = False
    has_multi_doc = len(doc_ids) >= 2
    has_doc_pair_comparison = len(doc_ids) == 2 and has_compare
    has_global_analyte_abnormal_search = (
        len(doc_ids) == 0
        and len(analyte_list) >= 1
        and has_global_scope_markers
        and has_abnormal_wording
    )
    has_doc_scoped_abnormal_results = (
        len(doc_ids) >= 1
        and has_abnormal_wording
        and any(k in qn for k in ["uniquement", "anomal", "hors reference", "quels resultats", "quels résultats", "donne", "sans interpretation", "sans interprétation", "résume", "resume"])
        and not any(k in qn for k in ["oui non", "oui/non", "oui ou non", "yes/no", "yes or no", "est ce que", "est-ce que"])
    )
    has_biological_summary_wording = (
        any(k in qn for k in ["resume", "résume", "synthese", "synthèse", "bilan", "medico-biologique", "médico-biologique"])
        and any(
            k in qn
            for k in [
                "separ",
                "sépar",
                "normaux",
                "anormaux",
                "anomalies et",
                "partie anomalies",
                "partie resultats normaux",
                "partie résultats normaux",
                "lignes maximum",
                "lignes max",
                "quelques lignes",
                "lignes",
            ]
        )
    )
    has_short_note_wording = any(
        k in qn
        for k in [
            "note courte",
            "note medicale courte",
            "note médicale courte",
            "note pour un medecin",
            "note pour un médecin",
            "synthese descriptive",
            "synthèse descriptive",
            "resume descriptif court",
            "résumé descriptif court",
            "strictement descriptif",
        ]
    )
    has_biological_summary_wording = has_biological_summary_wording or has_short_note_wording
    has_doc_scoped_biological_summary = (
        len(doc_ids) >= 1
        and has_biological_summary_wording
    )
    has_priority_request = any(
        k in qn
        for k in [
            "hierarchise",
            "hiérarchise",
            "hierarchiser",
            "hiérarchiser",
            "anomalies importantes",
            "priorite technique",
            "priorité technique",
            "importance technique",
            "ordre d importance",
            "ordre d’importance",
            "ordre de priorite",
            "ordre de priorité",
            "classement technique",
            "resultats significatifs",
            "résultats significatifs",
            "attention technique",
            "classer par gravite",
            "classer par gravité",
        ]
    )
    has_doc_scoped_priority_anomalies = (
        len(doc_ids) >= 1
        and has_priority_request
        and (has_abnormal_wording or "anomal" in qn or "important" in qn)
    )
    has_doc_scoped_medical_interpretation_guarded = (
        len(doc_ids) >= 1
        and any(k in qn for k in ["peut on conclure", "peut-on conclure", "conclure a", "conclure à", "hyperthyro", "hypothyro"])
    )
    has_single_analyte_lookup = (
        len(doc_ids) >= 1
        and len(analyte_list) == 1
        and any(k in qn for k in ["valeur de", "resultat de", "résultat de", "donne", "affiche", "montre"])
    )
    has_multi_analyte = len(analyte_list) >= 2
    has_presence_diff = has_multi_doc and (
        ("present" in qn and "absent" in qn)
        or ("présent" in qn and "absent" in qn)
        or "absents dans l autre" in qn
        or "presents dans un rapport mais absents dans l autre" in qn
    )
    response_transform_markers = [
        "reponse precedente",
        "réponse précédente",
        "meme reponse",
        "même réponse",
        "reformule cette reponse",
        "reformule cette réponse",
        "reformule la reponse",
        "reformule la réponse",
        "reformule",
        "convertis",
        "transforme",
        "sans la colonne",
        "ajoute la colonne",
        "retire la colonne",
        "garde seulement",
        "reformate",
        "meme resultat",
        "résultat précédent",
        "resultat precedent",
        "ce resultat",
        "ok donne moi",
        "ok donne-moi",
        "maintenant donne moi",
        "maintenant donne-moi",
    ]
    transform_format_markers = [
        "sous forme",
        "en graphique",
        "graphique en barres",
        "courbe",
        "line graph",
        "bar chart",
        "json strict",
        "tableau",
        "json",
    ]
    has_response_transform = any(k in qn for k in response_transform_markers) or (
        len(doc_ids) == 0
        and len(analyte_list) == 0
        and any(k in qn for k in transform_format_markers)
    )
    measurement_lookup_markers = [
        "donne moi le resultat de",
        "donne-moi le resultat de",
        "affiche le resultat de",
        "montre le resultat de",
        "donne moi la valeur de",
        "affiche la valeur de",
        "montre la valeur de",
    ]
    explicit_measurement_lookup = bool(len(analyte_list) >= 1 and any(k in qn for k in measurement_lookup_markers))
    if explicit_measurement_lookup:
        has_response_transform = False
    visualization_recommendation_markers = [
        "recommande",
        "recommander",
        "quelle visualisation",
        "quel graphique",
        "visualisation adaptee",
        "visualisation adaptée",
        "correspond a ce type de donnees",
        "correspond à ce type de données",
        "comment visualiser",
        "meilleure visualisation",
        "quelle ui",
        "donnees non transformables",
        "données non transformables",
        "pas des valeurs transformables",
        "pas de valeurs transformables",
    ]
    has_visualization_recommendation = (
        len(doc_ids) == 0
        and len(analyte_list) == 0
        and any(k in qn for k in visualization_recommendation_markers)
        and any(k in qn for k in ["visualisation", "graphique", "chart", "ui", "visualiser", "donnees", "données"])
    )
    inventory_visualization_render_markers = [
        "affiche avec des cartes patient",
        "affiche en cartes",
        "montre sous forme de cartes",
        "cartes patient",
        "nombre de rapports associes",
        "nombre de rapports associés",
        "accordeon",
        "accordéon",
        "table filtrable",
        "timeline documentaire",
        "montre l inventaire",
        "montre l’inventaire",
        "affiche cette visualisation",
        "ok affiche comme ca",
        "ok affiche comme ça",
        "utilise cette visualisation",
    ]
    has_inventory_visualization_render = (
        len(doc_ids) == 0
        and len(analyte_list) == 0
        and any(k in qn for k in inventory_visualization_render_markers)
    )
    qualitative_comment_render_markers = [
        "affiche ce commentaire",
        "affiche cette commentaire",
        "ok affiche ce commentaire",
        "ok affiche cette commentaire",
        "affiche cette note",
        "bloc commentaire source",
        "bloc commentaire sourcé",
        "dans un bloc commentaire",
        "encadre note",
        "encadré note",
        "tableau texte",
        "affiche cette interpretation",
        "affiche cette interprétation",
    ]
    has_qualitative_comment_render = (
        len(analyte_list) == 0
        and any(k in qn for k in qualitative_comment_render_markers)
        and any(k in qn for k in ["commentaire", "note", "interpretation", "interprétation", "bloc", "texte"])
    )
    source_followup_markers = [
        "d ou vient",
        "d'où vient",
        "source exacte",
        "source de ce commentaire",
        "donne moi la source",
        "donne-moi la source",
        "quelle page",
        "quel rapport",
        "ouvre la source",
        "source cliquable",
    ]
    has_source_followup = (
        len(doc_ids) == 0
        and any(k in qn for k in source_followup_markers)
        and any(
            k in qn
            for k in [
                "ce commentaire",
                "cette commentaire",
                "cette valeur",
                "ce resultat",
                "ce résultat",
                "cette note",
                "de ce commentaire",
                "de cette valeur",
            ]
        )
    )
    context_summary_markers = [
        "resume",
        "résume",
        "resumer",
        "résumer",
        "synthese",
        "synthèse",
        "synthetise",
        "synthétise",
    ]
    context_summary_targets = [
        "ce commentaire",
        "cette commentaire",
        "ce resultat",
        "ce résultat",
        "cette valeur",
        "ces resultats",
        "ces résultats",
        "ce tableau",
        "cette visualisation",
        "ca",
        "ça",
    ]
    has_context_summary_render = (
        len(doc_ids) == 0
        and any(k in qn for k in context_summary_markers)
        and any(k in qn for k in context_summary_targets)
    )
    has_yes_no_question = (
        qn.startswith("est ce que")
        or qn.startswith("est-ce que")
        or "oui non" in qn
        or "oui/non" in qn
        or "oui ou non" in qn
        or "yes no" in qn
        or "yes/no" in qn
        or "yes or no" in qn
        or "yes ou no" in qn
        or "answer only yes" in qn
        or "respond only yes" in qn
        or "strictly yes/no" in qn
    )
    has_medical_signal = bool(
        len(doc_ids) > 0
        or len(analyte_list) > 0
        or has_reference_range_lookup
        or has_structured_status_request
        or has_global_scope_markers
        or has_abnormal_wording
        or any(
            k in qn
            for k in [
                "plage",
                "norme",
                "reference",
                "référence",
                "valeur",
                "resultat",
                "résultat",
                "rapport",
                "report",
                "document",
            ]
        )
    )
    is_small_talk = (
        len(doc_ids) == 0
        and len(analyte_list) == 0
        and any(m in qn for m in small_talk_markers)
        and not has_medical_signal
        and not has_response_transform
    )
    is_identity_question = (
        len(doc_ids) == 0
        and len(analyte_list) == 0
        and any(m in qn for m in identity_markers)
    )
    is_capability_question = (
        len(doc_ids) == 0
        and len(analyte_list) == 0
        and any(m in qn for m in capability_markers)
    )
    is_help_question = (
        len(doc_ids) == 0
        and len(analyte_list) == 0
        and any(m in qn for m in help_markers)
    )
    is_general_conversation = is_small_talk or is_identity_question or is_capability_question or is_help_question
    is_patient_inventory = (
        len(doc_ids) == 0
        and len(analyte_list) == 0
        and any(m in qn for m in patient_inventory_markers)
    )
    is_patient_count = (
        len(doc_ids) == 0
        and len(analyte_list) == 0
        and any(m in qn for m in patient_count_markers)
    )
    has_global_patient_lookup = (
        len(doc_ids) == 0
        and len(analyte_list) >= 1
        and (
            ("patient" in qn or "patients" in qn)
            or qn.startswith("quels patients")
            or qn.startswith("liste moi tous les patients")
            or qn.startswith("retourne moi tous les patients")
        )
        and (
            any(ch.isdigit() for ch in qn)
            or "au dessus de la reference" in qn
            or "au-dessus de la reference" in qn
            or "en dessous de la reference" in qn
            or "below reference" in qn
            or "above reference" in qn
            or "hors reference" in qn
            or "dans la reference" in qn
        )
    )
    has_global_threshold_without_analyte = (
        len(doc_ids) == 0
        and len(analyte_list) == 0
        and has_global_scope_markers
        and (
            any(ch.isdigit() for ch in qn)
            or any(k in qn for k in ["superieur", "supérieur", "inferieur", "inférieur", "au dessus", "au-dessus", "en dessous", "strictement"])
        )
    )
    has_global_biological_summary = (
        len(doc_ids) == 0
        and has_global_scope_markers
        and any(
            k in qn
            for k in [
                "synthese",
                "synthèse",
                "resume",
                "résumé",
                "note",
                "medico-biologique",
                "médico-biologique",
                "resultats biologiques principaux",
                "résultats biologiques principaux",
                "anomalies biologiques principales",
            ]
        )
    )
    has_global_priority_anomalies_summary = (
        len(doc_ids) == 0
        and has_global_scope_markers
        and any(
            k in qn
            for k in [
                "attention technique",
                "priorite technique",
                "priorité technique",
                "meritent le plus d attention",
                "méritent le plus d’attention",
                "anomalies les plus importantes",
                "resultats importants",
                "résultats importants",
                "classer par importance",
            ]
        )
    )

    intents = {
        "general_conversation": is_general_conversation,
        "small_talk": is_small_talk,
        "identity_question": is_identity_question,
        "capability_question": is_capability_question,
        "help_question": is_help_question,
        "patient_inventory": is_patient_inventory,
        "patient_inventory_count": is_patient_count,
        "response_transform": has_response_transform,
        "visualization_recommendation": has_visualization_recommendation,
        "inventory_visualization_render": has_inventory_visualization_render,
        "qualitative_comment_render": has_qualitative_comment_render,
        "context_summary_render": has_context_summary_render,
        "source_followup": has_source_followup,
        "global_analyte_abnormal_search": has_global_analyte_abnormal_search,
        "doc_pair_comparison": has_doc_pair_comparison,
        "doc_scoped_medical_interpretation_guarded": has_doc_scoped_medical_interpretation_guarded,
        "doc_scoped_biological_summary": has_doc_scoped_biological_summary,
        "reference_ranges_summary": has_reference_ranges_summary,
        "doc_scoped_priority_anomalies": has_doc_scoped_priority_anomalies,
        "doc_scoped_abnormal_results": has_doc_scoped_abnormal_results,
        "single_analyte_lookup": has_single_analyte_lookup,
        "doc_scoped_results": has_doc_scope and (len(analyte_list) >= 1 or not has_summary),
        "doc_scoped_analyte_query": has_doc_scope and len(analyte_list) >= 1,
        "doc_scoped_summary": has_doc_scope and has_summary,
        "multi_analyte_results": has_multi_analyte,
        "previous_result_comparison": has_previous and (has_compare or has_doc_scope),
        "multi_doc_comparison": has_multi_doc and has_compare,
        "toxicology_summary": has_doc_scope and has_toxicology,
        "immunoanalysis_summary": has_doc_scope and has_immuno,
        "comment_without_measured_value": has_qualitative_comment_query,
        "reference_range_lookup": has_reference_range_lookup,
        "diagnostic_safety_question": has_diagnostic_safety,
        "treatment_safety_question": has_treatment_safety,
        "global_patient_lookup": has_global_patient_lookup,
        "global_biological_summary": has_global_biological_summary,
        "global_priority_anomalies_summary": has_global_priority_anomalies_summary,
        "cohort_search": (has_global_patient_lookup or has_global_threshold_without_analyte),
        "multi_doc_presence_diff": has_presence_diff,
        "yes_no_question": has_yes_no_question,
    }
    intents["is_structured_query"] = any(intents.values())
    return intents


def detect_output_format(query: str) -> str:
    return detect_presentation_intent(query).requested_output


def _extract_raw_format_phrase(query: str) -> str | None:
    text = str(query or "")
    patterns = [
        r"(?:sous\s+forme(?:\s+d[eu]|\s+du|\s+des)?\s+)([^?.!\n]+)",
        r"(?:format\s+)([^?.!\n]+)",
        r"(?:as\s+a[n]?\s+)([^?.!\n]+)",
    ]
    for patt in patterns:
        m = re.search(patt, text, flags=re.IGNORECASE)
        if m:
            phrase = str(m.group(1) or "").strip(" .,:;")
            if phrase:
                return phrase
    return None


def _raw_format_instruction_detected(query: str) -> bool:
    text = str(query or "")
    return bool(
        re.search(r"\bsous\s+forme(?:\s+d[eu]|\s+du|\s+des)?\s+[^?.!\n]+", text, flags=re.IGNORECASE)
        or re.search(r"\bformat\s+[^?.!\n]+", text, flags=re.IGNORECASE)
        or re.search(r"\bas\s+a[n]?\s+[^?.!\n]+", text, flags=re.IGNORECASE)
    )


def detect_presentation_intent(query: str) -> PresentationIntent:
    text = str(query or "")
    qn = norm_text(text)
    lower_text = text.lower()

    yes_no_markers = [
        "oui non",
        "oui/non",
        "reponds uniquement oui",
        "réponds uniquement oui",
        "juste oui",
        "juste non",
        "oui ou non",
        "yes no",
        "yes/no",
        "yes or no",
        "answer only yes",
        "respond only yes",
        "only yes or no",
        "yes ou no",
    ]
    table_markers = ["sous forme tableau", "format tableau", "tableau", " table ", "colonnes"]
    list_markers = ["liste", "bullet", "puces"]
    json_markers = ["json", "format json", "json strict"]
    paragraph_markers = ["paragraphe", "paragraph"]
    chart_markers = [
        "graphique",
        "courbe",
        "chart",
        "line graph",
        "line-graph",
        "bar chart",
        "plot",
        "diagramme",
        "visualisation",
        "visualization",
        "radar",
        "scatter",
        "spider",
        "heatmap",
        "carte thermique",
        "matrice",
        "matrix",
        "nuage de points",
        "visualisation comparative",
        "arithmetic line graph",
        "arithmetic line-graph",
    ]
    chart_word_re = re.compile(r"\b(graph|graphique|chart|plot|diagramme|visualisation|visualization|courbe)\b", flags=re.IGNORECASE)

    raw_phrase = _extract_raw_format_phrase(text)
    requested_output = "auto"
    chart_type: str | None = None
    confidence = 0.6
    unsupported = False
    unsupported_reason = None
    recommended = None
    user_visualization = False
    unhandled: list[str] = []

    has_paragraph_request = any(m in qn for m in paragraph_markers)
    has_chart_request = bool(any(m in qn for m in chart_markers) or chart_word_re.search(text))

    if any(m in qn for m in yes_no_markers):
        requested_output = "yes_no"
        confidence = 0.95
    elif any(m in qn for m in json_markers):
        requested_output = "json"
        confidence = 0.95
    elif has_paragraph_request:
        requested_output = "paragraph"
        confidence = 0.85
    elif has_chart_request:
        requested_output = "chart"
        user_visualization = True
        confidence = 0.9
        if any(
            m in qn
            for m in [
                "line graph",
                "line-graph",
                "line chart",
                "courbe",
                "arithmetic line graph",
                "arithmetic line-graph",
            ]
        ):
            chart_type = "line"
        elif any(m in qn for m in ["bar chart", "barres", "barre", "histogramme"]):
            chart_type = "bar"
        elif any(m in qn for m in ["scatter", "nuage de points"]):
            chart_type = "scatter"
        elif any(m in qn for m in ["radar", "spider chart", "spider"]):
            chart_type = "radar"
        elif any(m in qn for m in ["heatmap", "carte thermique", "matrix", "matrice"]):
            chart_type = "heatmap"
        else:
            chart_type = "unknown"
        if "arithmetic line graph" in qn or "arithmetic line-graph" in qn:
            recommended = "line"
        if chart_type in {"radar", "scatter", "heatmap"}:
            recommended = "bar"
        if chart_type == "unknown":
            unsupported = True
            unsupported_reason = "Le type de graphique demandé n’est pas reconnu de manière déterministe."
            recommended = "bar"
    elif any(m in qn for m in table_markers):
        requested_output = "table"
        confidence = 0.9
    elif any(m in qn for m in list_markers):
        requested_output = "list"
        confidence = 0.8
    else:
        raw_format_requested = _raw_format_instruction_detected(text)
        if raw_format_requested and raw_phrase:
            raw_norm = normalize_analyte(raw_phrase)
            if (
                re.search(r"\b(graphique|chart|graph|courbe|radar|scatter|heatmap|matrix|matrice)\b", raw_norm, flags=re.IGNORECASE)
                and not re.search(r"\bparagraphe\b|\bparagraph\b", raw_norm, flags=re.IGNORECASE)
            ):
                requested_output = "chart"
                user_visualization = True
                chart_type = "unknown"
                unsupported = True
                unsupported_reason = "Le format de visualisation demandé est ambigu."
                recommended = "bar"
                confidence = 0.45
                unhandled.append(f"Instruction de présentation non gérée: {raw_phrase}")
            else:
                requested_output = "unknown"
                unsupported = True
                unsupported_reason = "Le format demandé n’est pas reconnu de manière déterministe."
                recommended = "table"
                confidence = 0.4
                unhandled.append(f"Instruction de présentation non gérée: {raw_phrase}")
        else:
            requested_output = "auto"
            confidence = 0.55

    wants_intro = not any(
        k in qn for k in ["sans intro", "no intro", "only table", "uniquement le tableau", "json strict", "yes/no", "oui/non", "oui non"]
    )
    wants_conclusion = not any(k in qn for k in ["sans conclusion", "no conclusion", "json strict"])
    if requested_output == "json" or requested_output == "yes_no":
        wants_intro = False
        if requested_output == "json":
            wants_conclusion = False

    wants_clickable_sources = detect_source_clickable_requested(text)
    strict_columns = extract_requested_table_columns(text)

    if requested_output == "chart":
        requested_phrase = raw_phrase or "graphique"
        if chart_type in {"unknown", None}:
            unhandled.append(f"Type de graphique non précisé: {requested_phrase}")
        if unsupported and unsupported_reason:
            unhandled.append(unsupported_reason)
        if raw_phrase and any(k in normalize_analyte(raw_phrase) for k in ["matrix", "comparative", "bio clinical", "heatmap"]):
            unhandled.append(f"Instruction de présentation complexe à préserver: {raw_phrase}")
    elif raw_phrase and any(k in lower_text for k in ["line-graph", "line graph", "chart", "graphique", "diagramme", "visualisation", "visualization"]):
        unhandled.append(f"Instruction de présentation à préserver: {raw_phrase}")

    return PresentationIntent(
        requested_output=requested_output,
        chart_type=chart_type,
        raw_format_phrase=raw_phrase,
        wants_clickable_sources=wants_clickable_sources,
        wants_intro=wants_intro,
        wants_conclusion=wants_conclusion,
        strict_columns=strict_columns,
        unsupported_format=unsupported,
        user_requested_visualization=user_visualization,
        raw_user_request=text,
        unhandled_instructions=unhandled,
        presentation_confidence=confidence,
        unsupported_reason=unsupported_reason,
        recommended_output=recommended,
        unsupported_presentation_reason=unsupported_reason,
        recommended_alternative_format=recommended,
    )


def interpret_presentation_intent_with_llm(
    *,
    user_question: str,
    current_detected_presentation: PresentationIntent,
    supported_outputs: list[str] | None = None,
    supported_chart_types: list[str] | None = None,
    interpreter: Any | None = None,
) -> PresentationIntent:
    """
    Optional LLM presentation interpreter hook.
    This function is intentionally no-op unless an interpreter callback is provided.
    It never receives medical evidence, only presentation metadata.
    """
    if interpreter is None:
        return current_detected_presentation
    try:
        payload = {
            "user_question": str(user_question or ""),
            "supported_outputs": list(supported_outputs or ["table", "list", "json", "chart", "paragraph", "auto", "unknown"]),
            "supported_chart_types": list(supported_chart_types or ["line", "bar", "scatter", "radar", "heatmap", "unknown"]),
            "current_detected_presentation": {
                "requested_output": current_detected_presentation.requested_output,
                "chart_type": current_detected_presentation.chart_type,
                "raw_format_phrase": current_detected_presentation.raw_format_phrase,
                "unsupported_format": current_detected_presentation.unsupported_format,
                "unsupported_reason": current_detected_presentation.unsupported_reason,
                "recommended_output": current_detected_presentation.recommended_output,
                "presentation_confidence": current_detected_presentation.presentation_confidence,
            },
            "unhandled_instructions": list(current_detected_presentation.unhandled_instructions or []),
        }
        interpreted = interpreter(payload)
        if not isinstance(interpreted, dict):
            return current_detected_presentation
        return PresentationIntent(
            requested_output=str(interpreted.get("requested_output") or current_detected_presentation.requested_output),
            chart_type=str(interpreted.get("chart_type") or current_detected_presentation.chart_type or "unknown"),
            raw_format_phrase=str(interpreted.get("raw_format_phrase") or current_detected_presentation.raw_format_phrase or "") or None,
            wants_clickable_sources=current_detected_presentation.wants_clickable_sources,
            wants_intro=current_detected_presentation.wants_intro,
            wants_conclusion=current_detected_presentation.wants_conclusion,
            strict_columns=list(current_detected_presentation.strict_columns or []),
            unsupported_format=not bool(interpreted.get("is_supported", not current_detected_presentation.unsupported_format)),
            user_requested_visualization=current_detected_presentation.user_requested_visualization,
            raw_user_request=current_detected_presentation.raw_user_request,
            unhandled_instructions=list(current_detected_presentation.unhandled_instructions or []),
            presentation_confidence=float(interpreted.get("confidence") or current_detected_presentation.presentation_confidence),
            unsupported_reason=str(interpreted.get("reason") or current_detected_presentation.unsupported_reason or "") or None,
            recommended_output=str(interpreted.get("recommended_output") or current_detected_presentation.recommended_output or "") or None,
            unsupported_presentation_reason=str(interpreted.get("reason") or current_detected_presentation.unsupported_presentation_reason or "") or None,
            recommended_alternative_format=str(interpreted.get("recommended_output") or current_detected_presentation.recommended_alternative_format or "") or None,
        )
    except Exception:
        return current_detected_presentation


def decide_response_strategy(query_understanding: "QueryUnderstanding", evidence_pack: dict[str, Any] | None = None) -> ResponseStrategy:
    qu = query_understanding
    output = str(qu.output_format or "auto").lower()
    intent = str(qu.intent or "unstructured").lower()
    presentation = getattr(qu, "presentation_intent", None)
    evidences = list((evidence_pack or {}).get("evidences") or (evidence_pack or {}).get("results") or [])

    if qu.is_small_talk or intent in {"small_talk", "identity_question", "capability_question", "help_question"}:
        return ResponseStrategy("small_talk", "Conversation générale sans retrieval médical.", True, True)
    if qu.is_response_transform or intent == "response_transform":
        return ResponseStrategy("transform_previous_response", "Transformation de la réponse précédente.", False, True)
    safety_intent = str(qu.safety_intent or "").strip().lower()
    hard_safety_intents = {
        "diagnosis_refusal",
        "diagnostic_safety_question",
        "treatment_refusal",
        "treatment_safety_question",
        "pii_refusal",
    }
    if safety_intent in hard_safety_intents or intent == "diagnostic_safety_question":
        return ResponseStrategy("safety_response", "Question à risque diagnostique.", True, True)
    if output == "json" or qu.answer_style == "strict_json":
        return ResponseStrategy("render_json", "Format JSON strict demandé.", False, True)
    if output == "yes_no" or qu.answer_style == "yes_no":
        return ResponseStrategy("answer_directly", "Réponse yes/no stricte demandée.", False, True)

    if presentation is not None:
        if presentation.user_requested_visualization and output == "chart" and not presentation.unsupported_format:
            return ResponseStrategy("render_chart_data", "Demande graphique détectée.", True, True)
        if presentation.unsupported_format or output == "unknown":
            return ResponseStrategy(
                "explain_limit_and_provide_data",
                "Format de présentation non supporté ou ambigu ; explication requise.",
                True,
                True,
            )

    if output == "table":
        return ResponseStrategy("render_table", "Tableau explicitement demandé.", True, True)
    if output == "list":
        return ResponseStrategy("answer_directly", "Liste demandée.", True, True)
    if output == "paragraph":
        return ResponseStrategy("answer_directly", "Format paragraphe demandé.", True, True)
    if not evidences and intent in {"unstructured", "doc_scoped_summary"}:
        return ResponseStrategy("ask_clarification", "Contexte insuffisant et demande peu contrainte.", True, True)
    return ResponseStrategy("render_table", "Format structuré fiable par défaut.", True, True)


def extract_requested_table_columns(query: str) -> list[str]:
    text = str(query or "")
    qn = norm_text(text)
    if "colonnes" not in qn and "columns" not in qn:
        return []

    # Capture the segment after "colonnes" / "columns".
    m = re.search(r"(?:colonnes?|columns?)\s*:?\s*(.+)$", text, flags=re.IGNORECASE)
    if not m:
        return []
    tail = m.group(1).strip()
    # Stop on likely sentence boundaries.
    tail = re.split(r"[?.!]\s+", tail, maxsplit=1)[0].strip()
    if not tail:
        return []

    tokens = re.split(r",|\bet\b|\band\b", tail, flags=re.IGNORECASE)
    col_keys: list[str] = []

    for tok in tokens:
        t = norm_text(tok)
        if not t:
            continue
        key = None
        if "analyte" in t or "analyse" in t:
            key = "analyte"
        elif ("valeur" in t and "anterieur" not in t and "antérieur" not in t) or "current value" in t:
            key = "valeur_actuelle"
        elif "reference" in t or "référence" in t:
            key = "reference"
        elif "statut" in t:
            key = "statut"
        elif "anterieur" in t or "antérieur" in t or "previous" in t:
            key = "resultat_anterieur"
        elif "variation" in t or "difference" in t or "différence" in t:
            key = "variation"
        elif "unite" in t or "unité" in t or "unit" in t:
            key = "unite"
        elif "source" in t:
            key = "source"
        elif "patient" in t:
            key = "patient"
        elif "report" in t or "rapport" in t or "document" in t:
            key = "report"
        if key and key not in col_keys:
            col_keys.append(key)

    return col_keys


def detect_answer_style(query: str) -> str:
    qn = norm_text(query or "")
    if (
        qn.startswith("est ce que")
        or qn.startswith("est-ce que")
        or "est ce que" in qn
        or "oui non" in qn
        or "oui/non" in qn
        or "oui ou non" in qn
        or "reponds uniquement oui" in qn
        or "réponds uniquement oui" in qn
        or "yes no" in qn
        or "yes/no" in qn
        or "yes or no" in qn
        or "yes ou no" in qn
        or "answer only yes" in qn
        or "respond only yes" in qn
        or "only yes or no" in qn
    ):
        return "yes_no"
    if any(
        k in qn
        for k in [
            "note medecin",
            "note médecin",
            "note medicale",
            "note médicale",
            "note clinique",
            "note courte",
            "note de synthese",
            "note de synthèse",
            "note documentaire",
        ]
    ):
        return "doctor_note"
    editorial_markers = [
        "synthese editoriale",
        "synthèse éditoriale",
        "resume editorial",
        "résumé éditorial",
        "texte naturel et professionnel",
        "style clinique",
        "phrase d ouverture",
        "phrase d'ouverture",
        "redige un texte",
        "rédige un texte",
        "compte rendu clinique",
        "redaction professionnelle",
        "rédaction professionnelle",
    ]
    if any(marker in qn for marker in editorial_markers):
        return "editorial"
    short_markers = [
        "synthese courte",
        "synthèse courte",
        "resume court",
        "résumé court",
        "version courte",
        "limite toi a 3 a 5 lignes",
        "limite-toi a 3 a 5 lignes",
        "3 a 5 lignes",
        "3-5 lignes",
        "quelques lignes",
        "reste dense",
    ]
    if any(marker in qn for marker in short_markers):
        return "short"
    return "standard"


def detect_technical_condition(query: str) -> str | None:
    qn = norm_text(query or "")
    below_markers = [
        "basse",
        "bas",
        "diminuee",
        "diminuée",
        "diminue",
        "diminué",
        "inferieure",
        "inferieur",
        "en dessous",
        "sous la reference",
        "below",
        "below reference",
        "below_reference",
    ]
    above_markers = [
        "superieure",
        "superieur",
        "elevee",
        "eleve",
        "elevation",
        "élévation",
        "au dessus",
        "au-dessus",
        "above",
        "above reference",
        "above_reference",
    ]
    out_markers = [
        "hors reference",
        "hors norme",
        "anormal",
        "anormaux",
        "anormales",
        "out of range",
        "out_of_reference",
    ]
    if (
        any(k in qn for k in out_markers)
        or "hors de la reference" in qn
        or "resultats anormaux" in qn
        or "résultats anormaux" in qn
        or "anomalies biologiques" in qn
        or (any(k in qn for k in above_markers) and any(k in qn for k in below_markers))
    ):
        return "out_of_reference"
    if any(k in qn for k in above_markers + ["au dessus de la reference", "au-dessus de la reference"]):
        return "above_reference"
    if any(k in qn for k in below_markers + ["en dessous de la reference"]):
        return "below_reference"
    if any(k in qn for k in ["dans la reference", "within reference"]):
        return "within_reference"
    if any(k in qn for k in ["non interpretable", "not interpretable"]):
        return "not_interpretable"
    return None


def extract_requested_value(query: str) -> str | None:
    text = str(query or "")
    patterns = [
        r"(?:=|est|a|à)\s*([<>]?\s*\d+(?:[.,]\d+)?)\b",
        r"\b(?:valeur|value)(?:\s+de)?\s*[:=]?\s*([<>]?\s*\d+(?:[.,]\d+)?)\b",
    ]
    for patt in patterns:
        m = re.search(patt, text, flags=re.IGNORECASE)
        if m:
            value = str(m.group(1) or "").strip()
            if value:
                return value
    return None


def detect_comparison_operator(query: str) -> str | None:
    qn = norm_text(query or "")
    if any(
        k in qn
        for k in [
            "superieure ou egale a",
            "superieur ou egal a",
            "supérieure ou égale à",
            "supérieur ou égal à",
            "superieure ou egale",
            "superieur ou egal",
            "superieurs ou egaux",
            "superieurs ou egaux a",
            "supérieurs ou égaux",
            "supérieurs ou égaux à",
            "au moins",
            "at least",
            ">=",
            "ou plus",
        ]
    ):
        return ">="
    if any(
        k in qn
        for k in [
            "inferieure ou egale a",
            "inferieur ou egal a",
            "inférieure ou égale à",
            "inférieur ou égal à",
            "inferieure ou egale",
            "inferieur ou egal",
            "at most",
            "<=",
            "ou moins",
        ]
    ):
        return "<="
    if any(
        k in qn
        for k in [
            "strictement superieure a",
            "strictement superieur a",
            "strictement supérieure à",
            "strictement supérieur à",
            "strictement superieur",
            "strictement superieure",
            "strictement superieurs",
            "strictement superieurs a",
            "strictement supérieurs",
            "strictement supérieurs à",
            "plus de",
            "superieure a",
            "superieur a",
            "supérieure à",
            "supérieur à",
        ]
    ):
        return ">"
    if any(
        k in qn
        for k in [
            "strictement inferieure a",
            "strictement inferieur a",
            "strictement inférieure à",
            "strictement inférieur à",
            "inferieure a",
            "inferieur a",
            "inférieure à",
            "inférieur à",
            "moins de",
        ]
    ):
        return "<"
    if any(k in qn for k in [">", "plus que"]):
        return ">"
    if any(k in qn for k in ["<", "moins que"]):
        return "<"
    if any(k in qn for k in ["egal a", "égal à", "equal to", "="]):
        return "="
    return None


def detect_excluded_analytes(query: str) -> list[str]:
    qn = norm_text(query or "")
    exclusion_markers = [
        "sans inclure",
        "n inclus pas",
        "n'inclus pas",
        "ne pas inclure",
        "sans",
        "exclude",
        "excluding",
    ]
    if not any(m in qn for m in exclusion_markers):
        return []
    excluded: list[str] = []
    for analyte in detect_exact_analytes(query or ""):
        # Keep only analytes appearing in common exclusion clauses.
        token = analyte.replace("_", " ")
        if any(f"{marker} {token}" in qn for marker in exclusion_markers) or any(
            f"{marker} {alias}" in qn for marker in exclusion_markers for alias in get_analyte_aliases(analyte)
        ):
            excluded.append(analyte)
    # Also handle explicit non-aliased exclusions (TRAK, anti-TG...) for TSHus requests.
    if "trak" in qn and "trak" not in excluded:
        excluded.append("trak")
    if "anti tg" in qn and "anti_tg" not in excluded:
        excluded.append("anti_tg")
    if "anticorps anti recepteur de la tsh" in qn and "anti_recepteur_tsh" not in excluded:
        excluded.append("anti_recepteur_tsh")
    return sorted(set(excluded))


def extract_requested_unit(query: str) -> str | None:
    text = str(query or "")
    m = re.search(
        r"\b\d+(?:[.,]\d+)?\s*(pg/ml|ng/ml|mg/l|ug/ml|µg/ml|uiu?/ml|uu/ml|mui/l|mui/ml|pmol/l|mmol/l|ui/l)\b",
        text,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    return str(m.group(1) or "").strip()


def detect_source_clickable_requested(query: str) -> bool:
    qn = norm_text(query or "")
    markers = [
        "source cliquable",
        "sources cliquables",
        "source clickable",
        "sources clickables",
        "lien source",
        "liens source",
        "avec liens",
        "avec source",
        "ouvrir source",
        "ouvrir les sources",
        "ouvrir pdf",
        "lien pdf",
        "pdf cliquable",
        "document cliquable",
        "cite les sources",
        "citer les sources",
        "source pdf",
    ]
    return any(m in qn for m in markers)


def detect_inventory_view_type(query: str) -> str | None:
    qn = norm_text(query or "")
    if not qn:
        return None
    if any(k in qn for k in ["accordeon", "ouvrir les rapports", "liste accordeon"]):
        return "report_accordion"
    if any(k in qn for k in ["table filtrable", "filtrer par patient", "par date", "nom de fichier", "table"]):
        return "filterable_table"
    if any(k in qn for k in ["timeline", "chronologie", "ordre des rapports"]):
        return "document_timeline"
    if any(k in qn for k in ["cartes patient", "affiche en cartes", "sous forme de cartes", "nombre de rapports associes", "nombre de rapports associés"]):
        return "patient_cards"
    return None


def detect_qualitative_view_type(query: str) -> str | None:
    qn = norm_text(query or "")
    if not qn:
        return None
    if any(contains_exact_term(qn, k) for k in ["tableau texte", "sujet commentaire source", "affiche dans un tableau", "table"]) or ("tabl" in qn):
        return "text_table"
    if any(contains_exact_term(qn, k) for k in ["encadre de note interpretative", "encadre note", "note interpretative", "note"]):
        return "interpretive_note"
    if any(contains_exact_term(qn, k) for k in ["carte d information medicale", "carte medicale", "fiche"]):
        return "medical_info_card"
    if any(contains_exact_term(qn, k) for k in ["bloc commentaire source", "bloc source", "bloc commentaire", "commentaire source"]):
        return "sourced_comment_block"
    return None


def detect_requested_date_iso(query: str) -> str | None:
    text = str(query or "")
    m = re.search(r"\b(\d{2})/(\d{2})/(\d{4})\b", text)
    if not m:
        return None
    dd, mm, yyyy = m.group(1), m.group(2), m.group(3)
    return f"{yyyy}-{mm}-{dd}"


def detect_requested_summary_points(query: str) -> int | None:
    text = str(query or "")
    if not text:
        return None
    qn = norm_text(text)
    if not any(k in qn for k in ["resume", "resumer", "synthese", "synthetise"]):
        return None
    word_to_number = {
        "un": 1,
        "une": 1,
        "deux": 2,
        "trois": 3,
        "quatre": 4,
        "cinq": 5,
        "six": 6,
        "sept": 7,
        "huit": 8,
        "neuf": 9,
        "dix": 10,
    }
    m_digit = re.search(r"\ben\s*(\d{1,2})\s*points?\b", qn)
    if m_digit:
        try:
            return max(1, min(10, int(m_digit.group(1))))
        except Exception:
            return None
    m_word = re.search(r"\ben\s+(un|une|deux|trois|quatre|cinq|six|sept|huit|neuf|dix)\s+points?\b", qn)
    if m_word:
        return max(1, min(10, int(word_to_number.get(m_word.group(1), 3))))
    m_lines_digit = re.search(r"\ben\s*(\d{1,2})\s*lignes?\b", qn)
    if m_lines_digit:
        try:
            return max(1, min(20, int(m_lines_digit.group(1))))
        except Exception:
            return None
    m_lines_word = re.search(r"\ben\s+(un|une|deux|trois|quatre|cinq|six|sept|huit|neuf|dix)\s+lignes?\b", qn)
    if m_lines_word:
        return max(1, min(20, int(word_to_number.get(m_lines_word.group(1), 3))))
    return None


def detect_requested_report_type(query: str) -> str | None:
    qn = norm_text(query or "")
    if "immunoanalyse" in qn or "immuno analyse" in qn:
        return "immunoanalyse"
    if "biochimie" in qn or "bio chimie" in qn:
        return "biochimie"
    if "toxicologie" in qn:
        return "toxicologie"
    return None


def detect_latest_report_flag(query: str) -> bool:
    qn = norm_text(query or "")
    markers = [
        "dernier rapport",
        "dernier bilan",
        "rapport le plus recent",
        "rapport le plus récent",
        "dernier resultat disponible",
        "dernier résultat disponible",
    ]
    return any(m in qn for m in markers)


def detect_requested_context_type(query: str) -> str | None:
    qn = norm_text(query or "")
    # Keep qualitative mode only for explicit qualitative/comment requests.
    # Plain "note médecin" / "résumé" should stay on numerical biological context.
    if any(k in qn for k in ["commentaire", "commentaire source", "valeur seuil", "interpretation qualitative", "interprétation qualitative"]):
        return "medical_qualitative_comment"
    return "biological_numeric_results"


def _extract_requested_reference_profile(query: str) -> dict[str, Any] | None:
    raw = str(query or "")
    qn = norm_text(raw)
    profile: dict[str, Any] = {
        "sex": None,
        "age_operator": None,
        "age": None,
        "age_unit": None,
        "population": None,
        "condition": None,
    }
    if any(contains_exact_term(qn, k) for k in ["homme", "masculin", "male"]):
        profile["sex"] = "male"
    elif any(contains_exact_term(qn, k) for k in ["femme", "feminin", "female"]):
        profile["sex"] = "female"
    context_synonyms: list[tuple[tuple[str, ...], dict[str, str]]] = [
        (("femme cyclee j2 j4", "femme cyclée j2 j4", "j2-j4", "j2 j4"), {"sex": "female", "population": "cycled_female_j2_j4", "condition": "cycled_female_j2_j4"}),
        (("adulte ambulatoire",), {"population": "adult", "condition": "ambulatory"}),
        (("adulte alite", "adulte alité"), {"population": "adult", "condition": "bedridden"}),
        (("a jeun", "à jeun"), {"condition": "fasting"}),
        (("risque majeur",), {"condition": "risk_major"}),
        (("taux souhaitable", "souhaitable"), {"condition": "desirable"}),
        (("taux modere", "taux modéré", "modere"), {"condition": "moderate"}),
        (("taux eleve", "taux élevé", "eleve", "élevé"), {"condition": "high"}),
    ]
    for phrases, attrs in context_synonyms:
        if any(p in qn for p in phrases):
            for k, v in attrs.items():
                if not profile.get(k):
                    profile[k] = v
    if "adulte" in qn:
        profile["population"] = "adult"
    elif "enfant" in qn:
        profile["population"] = "child"
    elif "nourrisson" in qn:
        profile["population"] = "infant"
    elif "nouveau ne" in qn:
        profile["population"] = "newborn"
    elif "cordon" in qn:
        profile["population"] = "cord"

    m = re.search(r"(>=|<=|>|<)\s*(\d+(?:[.,]\d+)?)\s*(j|jour|jours|mois|ans?|années?|annees?)?", raw, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"(plus de|moins de)\s*(\d+(?:[.,]\d+)?)\s*(j|jour|jours|mois|ans?|annees?)", qn)
        if m:
            profile["age_operator"] = ">" if "plus de" in m.group(1) else "<"
            profile["age"] = float(str(m.group(2)).replace(",", "."))
            u = m.group(3)
            profile["age_unit"] = "days" if u.startswith("j") else ("months" if u.startswith("mois") else "years")
    else:
        profile["age_operator"] = m.group(1)
        profile["age"] = float(str(m.group(2)).replace(",", "."))
        u = m.group(3) or "ans"
        uu = norm_text(u)
        profile["age_unit"] = "days" if uu.startswith("j") else ("months" if uu.startswith("mois") else "years")
    m_range = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:-|–|a|à)\s*(\d+(?:[.,]\d+)?)\s*(j|jour|jours|mois|ans?|années?|annees?)", raw, flags=re.IGNORECASE)
    if m_range:
        u = norm_text(m_range.group(3))
        profile["age_min"] = float(str(m_range.group(1)).replace(",", "."))
        profile["age_max"] = float(str(m_range.group(2)).replace(",", "."))
        profile["age_unit"] = "days" if u.startswith("j") else ("months" if u.startswith("mois") else "years")
    if all(profile.get(k) in (None, "") for k in ["sex", "age_operator", "age", "age_unit", "population", "age_min", "age_max"]):
        return None
    return profile


def detect_use_patient_profile(query: str) -> bool:
    qn = norm_text(query or "")
    return any(k in qn for k in ["pour ce patient", "selon ce patient", "selon son age", "selon son sexe", "ce patient"])


def detect_request_all_reference_ranges(query: str) -> bool:
    qn = norm_text(query or "")
    return any(k in qn for k in ["toutes les plages", "tous les intervalles", "toutes les normes", "toutes les references"])


def _resolve_primary_intent(intents: dict[str, bool], *, requested_doc_ids: list[str], requested_analytes: list[str]) -> str:
    # When a concrete data task exists, keep it as primary and downgrade diagnostic safety
    # to a secondary constraint (safety_intent) handled downstream.
    has_data_task = any(
        bool(intents.get(k))
        for k in [
            "reference_range_lookup",
            "reference_ranges_summary",
            "response_transform",
            "global_biological_summary",
            "global_priority_anomalies_summary",
            "global_patient_lookup",
            "comment_without_measured_value",
            "multi_doc_presence_diff",
            "multi_doc_comparison",
            "toxicology_summary",
            "immunoanalysis_summary",
            "previous_result_comparison",
            "doc_scoped_summary",
            "doc_scoped_results",
        ]
    ) or bool(requested_doc_ids)

    if intents.get("patient_inventory_count"):
        return "patient_inventory_count"
    if intents.get("patient_inventory"):
        return "patient_inventory"
    if intents.get("inventory_visualization_render"):
        return "inventory_visualization_render"
    if intents.get("qualitative_comment_render"):
        return "qualitative_comment_render"
    if intents.get("global_priority_anomalies_summary"):
        return "global_priority_anomalies_summary"
    if intents.get("global_biological_summary"):
        return "global_biological_summary"
    if intents.get("context_summary_render"):
        return "context_summary_render"
    if intents.get("source_followup"):
        return "source_followup"
    if intents.get("visualization_recommendation"):
        return "visualization_recommendation"
    if intents.get("identity_question"):
        return "identity_question"
    if intents.get("capability_question"):
        return "capability_question"
    if intents.get("help_question"):
        return "help_question"
    if intents.get("small_talk") or intents.get("general_conversation"):
        return "small_talk"
    if intents.get("multi_doc_presence_diff"):
        return "multi_doc_presence_diff"
    if intents.get("global_analyte_abnormal_search"):
        return "global_analyte_abnormal_search"
    if intents.get("doc_pair_comparison"):
        return "doc_pair_comparison"
    if intents.get("multi_doc_comparison"):
        return "multi_doc_comparison"
    if intents.get("doc_scoped_medical_interpretation_guarded"):
        return "doc_scoped_medical_interpretation_guarded"
    if intents.get("reference_ranges_summary"):
        return "reference_ranges_summary"
    if intents.get("doc_scoped_biological_summary"):
        return "doc_scoped_biological_summary"
    if intents.get("doc_scoped_priority_anomalies"):
        return "doc_scoped_priority_anomalies"
    if intents.get("doc_scoped_abnormal_results"):
        return "doc_scoped_abnormal_results"
    if intents.get("single_analyte_lookup"):
        return "single_analyte_lookup"
    if intents.get("reference_range_lookup"):
        return "reference_range_lookup"
    if intents.get("response_transform"):
        return "response_transform"
    if intents.get("global_patient_lookup"):
        return "cohort_search"
    if intents.get("comment_without_measured_value"):
        return "comment_without_measured_value"
    if intents.get("toxicology_summary"):
        return "toxicology_summary"
    if intents.get("immunoanalysis_summary"):
        return "immunoanalysis_summary"
    if intents.get("previous_result_comparison") and len(requested_analytes) >= 1 and len(requested_doc_ids) == 1:
        return "previous_result_comparison"
    if intents.get("doc_scoped_summary"):
        return "doc_scoped_summary"
    if intents.get("doc_scoped_results") and len(requested_analytes) >= 1:
        return "doc_scoped_results"
    if len(requested_doc_ids) >= 1 and len(requested_analytes) >= 1:
        return "doc_scoped_results"
    if len(requested_doc_ids) >= 1:
        return "doc_scoped_summary"
    if intents.get("treatment_safety_question") and not has_data_task:
        return "unstructured"
    if intents.get("diagnostic_safety_question") and not has_data_task:
        return "diagnostic_safety_question"
    if intents.get("diagnostic_safety_question"):
        return "doc_scoped_summary" if requested_doc_ids else "unstructured"
    return "unstructured"


def compute_lexical_score(candidate_intent: str, query: str) -> float:
    qn = norm_text(query or "")
    cfg = LEXICAL_MARKERS.get(candidate_intent, {})
    strong = list(cfg.get("strong") or [])
    medium = list(cfg.get("medium") or [])
    weak = list(cfg.get("weak") or [])
    weight = float(cfg.get("weight") or 1.0)

    strong_count = sum(1 for marker in strong if contains_exact_term(qn, marker))
    medium_count = sum(1 for marker in medium if contains_exact_term(qn, marker))
    weak_count = sum(1 for marker in weak if contains_exact_term(qn, marker))

    if not (strong or medium or weak):
        return 0.3

    matched = (strong_count * 1.0) + (medium_count * 0.6) + (weak_count * 0.3)
    max_possible = (len(strong) * 1.0) + (len(medium) * 0.6) + (len(weak) * 0.3)
    base = (matched / max(max_possible, 1.0)) if max_possible else 0.0
    return max(0.0, min(1.0, base * weight))


def compute_structural_score(
    candidate_intent: str,
    requested_doc_ids: list[str],
    requested_analytes: list[str],
    technical_condition: str | None,
) -> float:
    doc_scope = 0.0
    if candidate_intent in DOC_SCOPED_INTENTS:
        doc_scope = 1.0 if requested_doc_ids else 0.0
    else:
        doc_scope = 0.5 if requested_doc_ids else 0.7

    analyte_scope = 0.0
    if candidate_intent in ANALYTE_SCOPED_INTENTS:
        analyte_scope = 1.0 if requested_analytes else 0.0
    else:
        analyte_scope = 0.5 if requested_analytes else 0.7

    condition_scope = 1.0 if technical_condition else 0.0
    structural = 0.4 * doc_scope + 0.4 * analyte_scope + 0.2 * condition_scope
    return max(0.0, min(1.0, structural))


def compute_medical_score(candidate_intent: str, medical_topics: list[dict[str, Any]]) -> float:
    if not medical_topics:
        return 0.5
    relevance_map = INTENT_TOPIC_RELEVANCE.get(candidate_intent, {})
    if not relevance_map:
        return 0.5
    best = 0.0
    for topic_dict in medical_topics:
        topic_name = str(topic_dict.get("topic") or "").strip()
        topic_conf = float(topic_dict.get("confidence") or 0.5)
        rel = float(relevance_map.get(topic_name, 0.0))
        best = max(best, topic_conf * rel)
    return max(0.0, min(1.0, best))


def apply_penalties(
    candidate_intent: str,
    requested_doc_ids: list[str],
    requested_analytes: list[str],
    safety_intent: str | None,
    lex_score: float,
    struct_score: float,
) -> float:
    total_penalty = 0.0
    if candidate_intent in DOC_SCOPED_INTENTS and not requested_doc_ids:
        total_penalty += 0.15
    if candidate_intent in ANALYTE_SCOPED_INTENTS and not requested_analytes:
        total_penalty += 0.20
    if str(safety_intent or "").strip().lower() in {"diagnosis_refusal", "diagnostic_safety_question", "treatment_refusal"} and candidate_intent not in {
        "small_talk",
        "identity_question",
        "capability_question",
        "help_question",
        "diagnostic_safety_question",
    }:
        total_penalty += 0.25
    if lex_score < 0.20:
        total_penalty += 0.10
    if struct_score < 0.20:
        total_penalty += 0.05
    return min(0.30, total_penalty)


def resolve_medical_topics_with_confidence(query: str, requested_analytes: list[str]) -> list[dict[str, Any]]:
    topics_raw = list(resolve_medical_topic(query or "") or [])
    if not topics_raw:
        return []
    qn = norm_text(query or "")
    enriched: list[dict[str, Any]] = []
    for topic_name in topics_raw:
        topic = str(topic_name or "").strip()
        if not topic:
            continue
        base_conf = 0.75
        analyte_bonus = 0.0
        for analyte in requested_analytes:
            if topic in ANALYTE_TOPIC_MAP.get(str(analyte), []):
                analyte_bonus += 0.12
        keyword_hits = 0
        for keyword in TOPIC_KEYWORDS.get(topic, []):
            if contains_exact_term(qn, keyword):
                keyword_hits += 1
        if keyword_hits >= 2:
            keyword_bonus = 0.08
        elif keyword_hits == 1:
            keyword_bonus = 0.02
        else:
            keyword_bonus = -0.15
        conf = max(0.30, min(1.0, base_conf + analyte_bonus + keyword_bonus))
        enriched.append({"topic": topic, "confidence": round(conf, 2)})

    enriched = sorted(enriched, key=lambda x: (-float(x.get("confidence") or 0.0), str(x.get("topic") or "")))
    return enriched[:3]


def score_all_intent_candidates(
    query: str,
    requested_doc_ids: list[str],
    requested_analytes: list[str],
    technical_condition: str | None,
    medical_topics: list[dict[str, Any]],
    intents_dict: dict[str, bool],
    safety_intent: str | None,
) -> list[tuple[str, float]]:
    candidates = [intent_name for intent_name, is_true in (intents_dict or {}).items() if is_true and intent_name != "is_structured_query"]
    if not candidates:
        candidates = ["unstructured"]
    scored: dict[str, float] = {}
    for candidate in candidates:
        lex_score = compute_lexical_score(candidate, query)
        struct_score = compute_structural_score(candidate, requested_doc_ids, requested_analytes, technical_condition)
        med_score = compute_medical_score(candidate, medical_topics)
        penalty = apply_penalties(
            candidate,
            requested_doc_ids,
            requested_analytes,
            safety_intent,
            lex_score,
            struct_score,
        )
        raw = 0.45 * lex_score + 0.35 * struct_score + 0.20 * med_score - penalty
        scored[candidate] = round(max(0.0, min(1.0, raw)), 2)
    sorted_candidates = sorted(scored.items(), key=lambda x: (-x[1], x[0]))
    return sorted_candidates[:3]


def _align_candidates_with_legacy_intent(
    legacy_intent: str,
    scored_candidates: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    legacy = str(legacy_intent or "").strip()
    if not legacy:
        return scored_candidates
    if not scored_candidates:
        return [(legacy, 0.0)]
    scores = {name: score for name, score in scored_candidates}
    if legacy not in scores:
        baseline = max(float(scored_candidates[0][1]), 0.60)
        aligned = [(legacy, round(baseline, 2))] + scored_candidates
    else:
        legacy_score = scores.get(legacy, 0.0)
        top_score = float(scored_candidates[0][1])
        promoted_score = round(max(legacy_score, top_score), 2)
        rest = [(name, score) for name, score in scored_candidates if name != legacy]
        aligned = [(legacy, promoted_score)] + rest
    aligned = sorted(aligned, key=lambda x: (-x[1], x[0]))
    if aligned and aligned[0][0] != legacy:
        legacy_score = next((score for name, score in aligned if name == legacy), aligned[0][1])
        aligned = [(legacy, legacy_score)] + [(name, score) for name, score in aligned if name != legacy]
    return aligned[:3]


def compute_ambiguity_flags(
    requested_doc_ids: list[str],
    requested_analytes: list[str],
    detected_intent: str,
    intent_candidates: list[tuple[str, float]],
    safety_intent: str | None,
    medical_topics: list[dict[str, Any]],
    technical_condition: str | None,
    alias_resolved: bool,
) -> list[str]:
    flags: list[str] = []
    if detected_intent in DOC_SCOPED_INTENTS and not requested_doc_ids:
        flags.append("missing_doc_scope")
    if len(requested_doc_ids) > 1 and detected_intent in SINGLE_DOC_INTENTS:
        flags.append("multiple_doc_scope_ambiguous")
    if alias_resolved:
        flags.append("analyte_alias_resolved")
    if medical_topics and not requested_analytes and detected_intent not in {"small_talk", "identity_question", "help_question", "general_conversation"}:
        flags.append("topic_vs_specific_analyte_ambiguous")
    if not requested_doc_ids and not requested_analytes and not technical_condition and not medical_topics:
        flags.append("insufficient_clinical_scope")
    if str(safety_intent or "").strip().lower() in {"diagnosis_refusal", "diagnostic_safety_question", "treatment_refusal"}:
        flags.append("unsafe_diagnosis_request")
    intent_conf = float(intent_candidates[0][1]) if intent_candidates else 0.0
    if intent_conf < 0.60:
        flags.append("confidence_below_threshold")
    if len(intent_candidates) >= 2:
        gap = float(intent_candidates[0][1]) - float(intent_candidates[1][1])
        if gap < 0.10:
            flags.append("multiple_candidates_clustered")
    return list(dict.fromkeys(flags))


def compute_scope_confidence(
    requested_doc_ids: list[str],
    requested_analytes: list[str],
    technical_condition: str | None,
    ambiguity_flags: list[str],
) -> float:
    score = 0.0
    score += 0.55 if requested_doc_ids else 0.10
    score += 0.30 if requested_analytes else 0.05
    score += 0.15 if technical_condition else 0.02
    penalty_map = {
        "missing_doc_scope": -0.15,
        "multiple_doc_scope_ambiguous": -0.20,
        "topic_vs_specific_analyte_ambiguous": -0.10,
        "insufficient_clinical_scope": -0.25,
        "confidence_below_threshold": -0.05,
        "multiple_candidates_clustered": -0.05,
    }
    for flag in ambiguity_flags or []:
        score += penalty_map.get(str(flag), 0.0)
    return round(max(0.0, min(1.0, score)), 2)


def build_intent_arbitration_debug(qu: QueryUnderstanding) -> dict[str, Any]:
    intents = dict(getattr(qu, "intents", {}) or {})
    candidate_intents = [k for k, v in intents.items() if bool(v) and k != "is_structured_query"]
    winner = str(getattr(qu, "intent", "") or "").strip()
    requested_doc_ids = list(getattr(qu, "requested_doc_ids", []) or [])
    requested_analytes = list(getattr(qu, "requested_analytes", []) or [])
    safety = str(getattr(qu, "safety_intent", "") or "").strip()

    if winner == "doc_scoped_summary" and safety == "diagnostic_safety_question":
        reason = (
            "Arbitrage: intent métier doc-scoped prioritaire ; "
            "la contrainte diagnostic est conservée en safety_intent."
        )
    elif winner == "diagnostic_safety_question":
        reason = "Arbitrage: question de sécurité diagnostique sans tâche data prioritaire."
    elif winner == "comment_without_measured_value":
        reason = "Arbitrage: requête qualitative explicite (commentaire/note/interprétation)."
    elif winner == "reference_range_lookup":
        reason = "Arbitrage: requête de plage physiologique détectée."
    elif winner == "reference_ranges_summary":
        reason = "Arbitrage: note/synthèse des types de références physiologiques détectée."
    elif winner == "global_analyte_abnormal_search":
        reason = "Arbitrage: recherche globale/cohorte d’analytes hors référence."
    elif winner == "doc_pair_comparison":
        reason = "Arbitrage: comparaison de deux rapports détectée."
    elif winner == "doc_scoped_medical_interpretation_guarded":
        reason = "Arbitrage: interprétation médicale prudente document-scopée."
    elif winner == "doc_scoped_biological_summary":
        reason = "Arbitrage: résumé médico-biologique court document-scopé."
    elif winner == "doc_scoped_abnormal_results":
        reason = "Arbitrage: résultats anormaux document-scopés."
    elif winner == "doc_scoped_priority_anomalies":
        reason = "Arbitrage: anomalies document-scopées classées par priorité technique."
    elif winner == "single_analyte_lookup":
        reason = "Arbitrage: lookup ciblé d’un analyte dans un rapport."
    elif winner == "multi_doc_comparison":
        reason = "Arbitrage: comparaison multi-documents détectée."
    elif winner == "doc_scoped_results":
        reason = "Arbitrage: extraction de résultats structurés ciblés document."
    elif winner == "doc_scoped_summary":
        reason = "Arbitrage: synthèse document ciblé."
    else:
        reason = "Arbitrage: priorité standard des intents."

    return {
        "candidate_intents": candidate_intents,
        "winner": winner,
        "reason": reason,
        "requested_doc_ids": requested_doc_ids,
        "requested_analytes": requested_analytes,
        "safety_intent": safety or None,
    }


def parse_query_understanding(query: str) -> QueryUnderstanding:
    presentation = detect_presentation_intent(query or "")
    requested_doc_ids = detect_requested_doc_ids(query or "")
    requested_analytes = detect_exact_analytes(query or "")
    qn = norm_text(query or "")
    if (
        "hormones thyroid" in qn
        or "hormones thyro" in qn
        or "thyroidiennes" in qn
        or "thyroidiennes" in qn
        or "parametres thyroid" in qn
        or "paramètres thyro" in qn
        or "profil tsh/t3/t4" in qn
        or "profil tsh t3 t4" in qn
        or "bilan thyroid" in qn
        or "bilan thyro" in qn
        or "bilan thyroïd" in qn
    ):
        for k in ["t4_libre", "t3_libre", "tshus", "anti_tg", "trak", "anti_tpo"]:
            if k not in requested_analytes:
                requested_analytes.append(k)
    requested_analytes = [
        canonicalize_analyte(str(a))
        for a in requested_analytes
        if canonicalize_analyte(str(a))
    ]
    requested_analytes = list(dict.fromkeys(requested_analytes))
    excluded_analytes = detect_excluded_analytes(query or "")
    intents = detect_query_intents(query or "", requested_doc_ids=requested_doc_ids, analytes=requested_analytes)
    requires_previous_results = intents.get("previous_result_comparison", False) or (
        "anterieur" in qn or "antérieur" in qn or "previous" in qn
    )
    requires_comparison = intents.get("multi_doc_comparison", False) or ("compare" in qn or "compar" in qn)
    requires_section_summary = intents.get("doc_scoped_summary", False) or intents.get("immunoanalysis_summary", False)
    answer_style = detect_answer_style(query or "")
    requested_value = extract_requested_value(query or "")
    requested_unit = extract_requested_unit(query or "")
    comparison_operator = detect_comparison_operator(query or "")
    source_clickable_requested = bool(presentation.wants_clickable_sources)
    language = detect_language(query or "")
    no_diagnosis_constraint = any(
        k in qn
        for k in [
            "ne donne pas de diagnostic",
            "sans diagnostic",
            "pas de diagnostic",
            "sans poser de diagnostic",
        ]
    )
    safety_intent = (
        "no_diagnosis_constraint"
        if no_diagnosis_constraint
        else (
            "treatment_refusal"
            if intents.get("treatment_safety_question")
            else ("diagnostic_safety_question" if intents.get("diagnostic_safety_question") else None)
        )
    )
    requested_table_columns = list(presentation.strict_columns or [])
    technical_condition = detect_technical_condition(query or "")
    inventory_view_type = detect_inventory_view_type(query or "")
    requested_date_iso = detect_requested_date_iso(query or "")
    requested_report_type = detect_requested_report_type(query or "")
    latest_report = detect_latest_report_flag(query or "")
    requested_context_type = detect_requested_context_type(query or "")
    qualitative_view_type = detect_qualitative_view_type(query or "")
    requested_reference_profile = _extract_requested_reference_profile(query or "")
    use_patient_profile = detect_use_patient_profile(query or "")
    request_all_reference_ranges = detect_request_all_reference_ranges(query or "")
    requested_summary_points = detect_requested_summary_points(query or "")
    preliminary_intent = _resolve_primary_intent(intents, requested_doc_ids=requested_doc_ids, requested_analytes=requested_analytes)
    if requested_analytes and comparison_operator and preliminary_intent == "unstructured":
        preliminary_intent = "cohort_search"
    if requested_context_type == "medical_qualitative_comment" and "commentaire" in qn and preliminary_intent == "unstructured":
        preliminary_intent = "comment_without_measured_value"
    if latest_report and requested_analytes and preliminary_intent == "unstructured":
        preliminary_intent = "doc_scoped_results"
    if requested_date_iso and preliminary_intent == "unstructured":
        preliminary_intent = "doc_scoped_results"
    direct_measurement_markers = [
        "donne moi le resultat de",
        "donne-moi le resultat de",
        "affiche le resultat de",
        "montre le resultat de",
        "donne moi la valeur de",
        "affiche la valeur de",
        "montre la valeur de",
    ]
    if (
        requested_analytes
        and any(m in qn for m in direct_measurement_markers)
        and preliminary_intent in {"unstructured", "response_transform"}
    ):
        preliminary_intent = "doc_scoped_results"
    medical_topics = resolve_medical_topics_with_confidence(query or "", requested_analytes)
    scored_candidates = score_all_intent_candidates(
        query=query or "",
        requested_doc_ids=requested_doc_ids,
        requested_analytes=requested_analytes,
        technical_condition=technical_condition,
        medical_topics=medical_topics,
        intents_dict=intents,
        safety_intent=safety_intent,
    )
    scored_candidates = _align_candidates_with_legacy_intent(preliminary_intent, scored_candidates)
    alias_resolved = detect_alias_resolution_used(query or "", requested_analytes)
    ambiguity_flags = compute_ambiguity_flags(
        requested_doc_ids=requested_doc_ids,
        requested_analytes=requested_analytes,
        detected_intent=preliminary_intent,
        intent_candidates=scored_candidates,
        safety_intent=safety_intent,
        medical_topics=medical_topics,
        technical_condition=technical_condition,
        alias_resolved=alias_resolved,
    )
    scope_confidence = compute_scope_confidence(
        requested_doc_ids=requested_doc_ids,
        requested_analytes=requested_analytes,
        technical_condition=technical_condition,
        ambiguity_flags=ambiguity_flags,
    )
    intent_candidates = [
        {
            "intent": str(intent_name),
            "confidence": round(float(conf), 2),
        }
        for intent_name, conf in scored_candidates[:3]
    ]
    intent_confidence = round(float(intent_candidates[0]["confidence"]) if intent_candidates else 0.0, 2)
    preview_qu = QueryUnderstanding(
        requested_doc_ids=requested_doc_ids,
        requested_analytes=requested_analytes,
        excluded_analytes=excluded_analytes,
        requested_value=requested_value,
        requested_unit=requested_unit,
        comparison_operator=comparison_operator,
        source_clickable_requested=source_clickable_requested,
        patient_query=bool("patient" in qn or "patients" in qn),
        intent=preliminary_intent,
        output_format=presentation.requested_output,
        requested_table_columns=requested_table_columns,
        answer_style=answer_style,
        requires_global_search=bool(intents.get("global_patient_lookup")),
        technical_condition=technical_condition,
        safety_intent=safety_intent,
        requires_previous_results=requires_previous_results,
        requires_comparison=requires_comparison,
        requires_section_summary=requires_section_summary,
        is_small_talk=bool(intents.get("small_talk")),
        is_response_transform=bool(intents.get("response_transform")),
        language=language,
        intents=intents,
        presentation_intent=presentation,
        response_strategy="render_table",
        response_strategy_reason=None,
        original_user_question=str(query or ""),
        raw_user_request=presentation.raw_user_request,
        raw_format_phrase=presentation.raw_format_phrase,
        unhandled_instructions=list(presentation.unhandled_instructions or []),
        presentation_confidence=float(presentation.presentation_confidence),
        unsupported_presentation_reason=presentation.unsupported_presentation_reason,
        recommended_alternative_format=presentation.recommended_alternative_format,
        inventory_view_type=inventory_view_type,
        requested_date_iso=requested_date_iso,
        requested_report_type=requested_report_type,
        latest_report=latest_report,
        requested_context_type=requested_context_type,
        qualitative_view_type=qualitative_view_type,
        requested_reference_profile=requested_reference_profile,
        use_patient_profile=use_patient_profile,
        request_all_reference_ranges=request_all_reference_ranges,
        requested_summary_points=requested_summary_points,
        intent_candidates=intent_candidates,
        intent_confidence=intent_confidence,
        scope_confidence=scope_confidence,
        ambiguity_flags=ambiguity_flags,
        medical_topics=medical_topics,
    )
    strategy = decide_response_strategy(preview_qu, evidence_pack=None)

    return replace(
        preview_qu,
        response_strategy=strategy.name,
        response_strategy_reason=strategy.reason,
    )


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
