from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any


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
    "tshus": {"tshus", "tsh us", "tsh ultra sensible", "tsh ultrasensible", "tsh"},
    "tsh": {"tsh"},
    "acth": {"acth"},
    "troponine": {"troponine", "troponine i", "troponine t"},
    "ace": {"ace"},
    "psa_totale": {"psa totale", "psa total", "psa"},
    "ca_15_3": {"ca 15-3", "ca 15 3", "ca15-3"},
    "t4_libre": {"t4 libre", "t4l", "ft4", "thyroxine libre", "free t4"},
    "ckmb": {"ckmb", "ck mb", "ck-mb"},
    "triglycerides": {"triglycerides", "triglycérides"},
    "cholesterol_ldl": {"ldl", "cholesterol ldl", "cholestérol ldl", "cholesterol ldl-c", "ldl c"},
    "microalbuminurie": {"microalbuminurie", "micro albuminurie", "microalbumine"},
    "ethanol": {"ethanol", "éthanol", "alcool", "ethyl"},
    "acide_valproique": {"acide valproique", "acide valproïque", "acide valporoique", "valproate", "valproique", "valporoique"},
    "carbamazepine": {"carbamazepine", "carbamazépine"},
    "vitamine_b12": {"vitamine b12", "vitamine_b12", "vit b12", "b12"},
    "vitamine_d": {"vitamine d", "vitamine_d"},
    "trichuris": {"trichuris", "trichuris trichiura"},
    "ankylostoma": {"ankylostoma"},
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


def norm_text(value: str) -> str:
    s = (value or "").strip().lower().replace("µ", "u").replace("_", " ")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9_\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_analyte(text: str) -> str:
    return norm_text(text).replace("-", " ")


def detect_language(query: str) -> str:
    qn = norm_text(query or "")
    if any(k in qn for k in ["dans report", "rapport", "document", "reponds", "réponds", "oui non", "tableau"]):
        return "fr"
    if any(k in qn for k in ["in report", "answer yes/no", "table format", "json strict"]):
        return "en"
    return "fr"


def get_analyte_aliases(analyte: str) -> set[str]:
    key = str(analyte or "").strip().lower()
    aliases = set(ANALYTE_ALIASES.get(key, set()))
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
    req_key = str(requested or "").strip().lower()
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
    qn = normalize_analyte(query)
    found: list[str] = []
    for canonical, aliases in ANALYTE_ALIASES.items():
        all_aliases = get_analyte_aliases(canonical) | {normalize_analyte(a) for a in aliases}
        for alias in sorted(all_aliases, key=len, reverse=True):
            if contains_exact_term(qn, alias):
                found.append(canonical)
                break
    return found


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
    has_troponin_comment = ("troponine" in qn) and any(
        k in qn for k in ["valeur mesuree", "valeur mesurée", "seulement un commentaire", "commentaire d interpretation", "commentaire"]
    )
    has_diagnostic_safety = ("cancer" in qn) or any(
        k in qn
        for k in [
            "peut on conclure",
            "peut-on conclure",
            "conclure a",
            "conclure à",
            "diagnostic definitif",
            "diagnostic définitif",
            "diagnostic",
            "traitement",
            "est ce que le patient a",
            "est-ce que le patient a",
        ]
    )
    has_doc_scope = len(doc_ids) >= 1
    has_multi_doc = len(doc_ids) >= 2
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
        "convertis",
        "transforme",
        "sans la colonne",
        "ajoute la colonne",
        "retire la colonne",
        "garde seulement",
        "reformate",
        "donne moi le resultat",
        "donne-moi le resultat",
        "affiche le resultat",
        "mets le resultat",
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
    is_small_talk = (
        len(doc_ids) == 0
        and len(analyte_list) == 0
        and not any(ch.isdigit() for ch in qn)
        and any(m in qn for m in small_talk_markers)
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

    intents = {
        "general_conversation": is_general_conversation,
        "small_talk": is_small_talk,
        "identity_question": is_identity_question,
        "capability_question": is_capability_question,
        "help_question": is_help_question,
        "response_transform": has_response_transform,
        "doc_scoped_results": has_doc_scope and (len(analyte_list) >= 1 or not has_summary),
        "doc_scoped_analyte_query": has_doc_scope and len(analyte_list) >= 1,
        "doc_scoped_summary": has_doc_scope and has_summary,
        "multi_analyte_results": has_multi_analyte,
        "previous_result_comparison": has_previous and (has_compare or has_doc_scope),
        "multi_doc_comparison": has_multi_doc and has_compare,
        "toxicology_summary": has_doc_scope and has_toxicology,
        "immunoanalysis_summary": has_doc_scope and has_immuno,
        "comment_without_measured_value": has_doc_scope and has_troponin_comment,
        "diagnostic_safety_question": has_diagnostic_safety,
        "global_patient_lookup": has_global_patient_lookup,
        "cohort_search": has_global_patient_lookup,
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
        "graph",
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

    raw_phrase = _extract_raw_format_phrase(text)
    requested_output = "auto"
    chart_type: str | None = None
    confidence = 0.6
    unsupported = False
    unsupported_reason = None
    recommended = None
    user_visualization = False
    unhandled: list[str] = []

    if any(m in qn for m in yes_no_markers):
        requested_output = "yes_no"
        confidence = 0.95
    elif any(m in qn for m in json_markers):
        requested_output = "json"
        confidence = 0.95
    elif any(m in qn for m in chart_markers):
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
    elif any(m in qn for m in paragraph_markers):
        requested_output = "paragraph"
        confidence = 0.8
    elif any(m in qn for m in list_markers):
        requested_output = "list"
        confidence = 0.8
    else:
        raw_format_requested = _raw_format_instruction_detected(text)
        if raw_format_requested and raw_phrase:
            raw_norm = normalize_analyte(raw_phrase)
            if any(k in raw_norm for k in ["graphique", "chart", "graph", "courbe", "radar", "scatter", "heatmap", "matrix", "matrice"]):
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
    if qu.safety_intent or intent == "diagnostic_safety_question":
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
    return "standard"


def detect_technical_condition(query: str) -> str | None:
    qn = norm_text(query or "")
    if any(k in qn for k in ["au dessus de la reference", "au-dessus de la reference", "above reference", "superieur"]):
        return "above_reference"
    if any(k in qn for k in ["en dessous de la reference", "below reference", "inferieur"]):
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
            "strictement superieure a",
            "strictement superieur a",
            "strictement supérieure à",
            "strictement supérieur à",
            "strictement superieur",
            "strictement superieure",
            "plus de",
            "superieure a",
            "superieur a",
            "supérieure à",
            "supérieur à",
        ]
    ):
        return ">"
    if any(k in qn for k in ["ou plus", "superieure ou egale", "superieur ou egal", "at least", ">="]):
        return ">="
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
    if any(k in qn for k in ["ou moins", "inferieure ou egale", "inferieur ou egal", "at most", "<="]):
        return "<="
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
        "lien source",
        "liens source",
        "avec source",
        "cite les sources",
        "citer les sources",
        "source pdf",
    ]
    return any(m in qn for m in markers)


def _resolve_primary_intent(intents: dict[str, bool], *, requested_doc_ids: list[str], requested_analytes: list[str]) -> str:
    if intents.get("identity_question"):
        return "identity_question"
    if intents.get("capability_question"):
        return "capability_question"
    if intents.get("help_question"):
        return "help_question"
    if intents.get("small_talk") or intents.get("general_conversation"):
        return "small_talk"
    if intents.get("response_transform"):
        return "response_transform"
    if intents.get("global_patient_lookup"):
        return "cohort_search"
    if intents.get("diagnostic_safety_question"):
        return "diagnostic_safety_question"
    if intents.get("comment_without_measured_value"):
        return "comment_without_measured_value"
    if intents.get("multi_doc_presence_diff"):
        return "multi_doc_presence_diff"
    if intents.get("multi_doc_comparison"):
        return "multi_doc_comparison"
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
    return "unstructured"


def parse_query_understanding(query: str) -> QueryUnderstanding:
    presentation = detect_presentation_intent(query or "")
    requested_doc_ids = detect_requested_doc_ids(query or "")
    requested_analytes = detect_exact_analytes(query or "")
    excluded_analytes = detect_excluded_analytes(query or "")
    intents = detect_query_intents(query or "", requested_doc_ids=requested_doc_ids, analytes=requested_analytes)

    qn = norm_text(query or "")
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
    safety_intent = "diagnostic_safety_question" if intents.get("diagnostic_safety_question") else None
    requested_table_columns = list(presentation.strict_columns or [])
    technical_condition = detect_technical_condition(query or "")
    preliminary_intent = _resolve_primary_intent(intents, requested_doc_ids=requested_doc_ids, requested_analytes=requested_analytes)
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
    )
    strategy = decide_response_strategy(preview_qu, evidence_pack=None)

    return QueryUnderstanding(
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
        response_strategy=strategy.name,
        response_strategy_reason=strategy.reason,
        original_user_question=str(query or ""),
        raw_user_request=presentation.raw_user_request,
        raw_format_phrase=presentation.raw_format_phrase,
        unhandled_instructions=list(presentation.unhandled_instructions or []),
        presentation_confidence=float(presentation.presentation_confidence),
        unsupported_presentation_reason=presentation.unsupported_presentation_reason,
        recommended_alternative_format=presentation.recommended_alternative_format,
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
