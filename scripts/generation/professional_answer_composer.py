from __future__ import annotations

import json
import re
from typing import Any

from llm_client import LLMClient, LLMClientError
from query_understanding import QueryUnderstanding, norm_text


PROFESSIONAL_WRITER_SYSTEM_PROMPT = """Tu es un rédacteur médical technique dans un système RAG.
Tu dois produire une réponse professionnelle, naturelle et concise à partir de l’evidence_pack fourni.
L’evidence_pack est la seule source de vérité.
Tu n’as pas le droit d’inventer, modifier ou compléter une valeur, une unité, une référence, un patient, un document, un résultat antérieur, une source ou un diagnostic.
Tu ne dois jamais utiliser ta connaissance générale pour ajouter des faits médicaux.
Tu dois respecter strictement l’intention et le format demandé par l’utilisateur.

Règles de style :
- Avant de répondre, organise la réponse en interne, mais ne produis jamais cette organisation. La sortie doit contenir uniquement la réponse finale.
- Si la question est un small talk, réponds naturellement sans source et sans donnée médicale.
- Ne commence pas directement par un tableau, sauf si l’utilisateur demande uniquement un tableau.
- Ajoute une courte introduction utile, adaptée à la question.
- L’introduction doit mentionner le vrai critère recherché, pas une phrase générique.
- Ne répète pas toujours la même phrase d’introduction.
- N’affiche jamais les aliases internes comme “tshus, tsh”.
- Utilise les noms humains des analytes depuis results[].analyte.
- Gère correctement le singulier/pluriel.
- N’écris jamais “résultat(s)” ou “correspondant(s)”.
- Si un seul résultat est trouvé, écris “Un seul résultat correspondant a été retrouvé.”
- Si plusieurs résultats sont trouvés, écris “X résultats correspondants ont été retrouvés.”
- Si aucun résultat est trouvé, écris “Aucun résultat correspondant n’a été retrouvé.”
- Si la question contient un critère de valeur, mentionne ce critère dans l’introduction.
- Si la question demande des sources cliquables, inclure une colonne Source ou afficher des sources cliquables sous la réponse.
- Ajoute une conclusion technique courte quand elle apporte de la valeur.
- Ne sois pas verbeux.
- Ne donne jamais de diagnostic si la question est médicale/diagnostique.

Règles de format :
- Si output_format = table, produis un tableau Markdown propre.
- Si source_clickable_requested = true, le tableau doit inclure une colonne Source avec le label source.
- Si output_format = json ou answer_style = strict_json, retourne uniquement JSON valide, sans texte autour.
- Si answer_style = yes_no, commence par Oui/Non ou Yes/No puis donne valeur, référence et source.
- Pour les sources, utilise source_label.
- N’affiche jamais chunk_id, path local, request_id ou logs techniques.

Règles de grounding :
- Tous les analytes affichés doivent exister dans results.
- Toutes les valeurs affichées doivent exister dans results.
- Toutes les sources affichées doivent exister dans results.
- Ne rajoute aucune ligne au tableau.
- Ne supprime aucune ligne importante fournie dans results, sauf si l’utilisateur demande un filtre."""


_COLD_CONCLUSIONS = {
    "Les résultats ci-dessus sont strictement extraits des données indexées.",
}


def _safe_str(value: Any, default: str = "") -> str:
    text = str(value or "").strip()
    return text if text else default


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def _pick_variant(seed: str, options: list[str]) -> str:
    if not options:
        return ""
    idx = sum(ord(c) for c in (seed or "")) % len(options)
    return options[idx]


def _canonical_analyte_display(alias: str) -> str:
    mapping = {
        "tshus": "TSHus",
        "tsh": "TSH",
        "acth": "ACTH",
        "psa_totale": "PSA totale",
        "ca_15_3": "CA 15-3",
        "t4_libre": "T4 libre",
        "ckmb": "CKMB",
        "crp": "CRP",
        "ace": "ACE",
        "acide_valproique": "Acide valproïque",
    }
    key = norm_text(alias).replace(" ", "_")
    return mapping.get(key, alias.replace("_", " "))


def _analyte_norm_key(value: str) -> str:
    return norm_text(value).replace(" ", "_")


def humanize_analyte_list(analytes: list[str] | None, evidence_pack: dict[str, Any]) -> str:
    requested = [str(a).strip() for a in (analytes or []) if str(a).strip()]
    evidences = list(evidence_pack.get("evidences") or evidence_pack.get("results") or [])

    by_norm: dict[str, str] = {}
    for ev in evidences:
        display = _safe_str(ev.get("analyte"))
        norm_val = _analyte_norm_key(_safe_str(ev.get("analyte_norm")) or display)
        if display and norm_val and norm_val not in by_norm:
            by_norm[norm_val] = display

    # Prefer analytes that are effectively present in evidence rows to avoid alias leakage
    # such as "tshus, tsh" when only TSHus is part of retrieved facts.
    labels: list[str] = []
    if evidences:
        seen_from_evidence: set[str] = set()
        for ev in evidences:
            label = _safe_str(ev.get("analyte"))
            if not label:
                continue
            norm_label = _analyte_norm_key(label)
            if norm_label in seen_from_evidence:
                continue
            seen_from_evidence.add(norm_label)
            labels.append(label)

    if labels:
        if len(labels) == 1:
            return labels[0]
        if len(labels) == 2:
            return f"{labels[0]} et {labels[1]}"
        return ", ".join(labels[:-1]) + f" et {labels[-1]}"

    labels = []
    seen: set[str] = set()
    for raw in requested:
        norm_key = _analyte_norm_key(raw)
        label = by_norm.get(norm_key) or _canonical_analyte_display(raw)
        norm_label = _analyte_norm_key(label)
        if norm_label in seen:
            continue
        seen.add(norm_label)
        labels.append(label)

    if not labels:
        for ev in evidences:
            label = _safe_str(ev.get("analyte"))
            if not label:
                continue
            norm_label = _analyte_norm_key(label)
            if norm_label in seen:
                continue
            seen.add(norm_label)
            labels.append(label)

    if not labels:
        return "les analytes demandés"
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} et {labels[1]}"
    return ", ".join(labels[:-1]) + f" et {labels[-1]}"


def _infer_requested_unit(query_understanding: QueryUnderstanding, evidence_pack: dict[str, Any]) -> str:
    explicit = _safe_str(getattr(query_understanding, "requested_unit", ""))
    if explicit:
        return explicit
    evidences = list(evidence_pack.get("evidences") or evidence_pack.get("results") or [])
    for ev in evidences:
        unit = _safe_str(ev.get("unit"))
        if unit:
            return unit
    return ""


def humanize_condition(query_understanding: QueryUnderstanding, evidence_pack: dict[str, Any]) -> str:
    qn = norm_text(query_understanding.requested_value or "")
    value = _safe_str(query_understanding.requested_value)
    operator = _safe_str(getattr(query_understanding, "comparison_operator", ""))
    requested_unit = _infer_requested_unit(query_understanding, evidence_pack)
    unit_suffix = f" {requested_unit}" if requested_unit else ""
    technical = _safe_str(query_understanding.technical_condition).lower()

    if value:
        if operator == ">":
            return f"strictement supérieure à {value}{unit_suffix}"
        if operator == ">=" or any(k in qn for k in [">=", "ou plus"]):
            return f"supérieure ou égale à {value}{unit_suffix}"
        if operator == "<":
            return f"strictement inférieure à {value}{unit_suffix}"
        if operator == "<=" or any(k in qn for k in ["<=", "ou moins"]):
            return f"inférieure ou égale à {value}{unit_suffix}"
        if operator == "=":
            return f"égale à {value}{unit_suffix}"
        return f"égale à {value}{unit_suffix}"

    if technical == "above_reference":
        return "au-dessus de la référence"
    if technical == "below_reference":
        return "en dessous de la référence"
    if technical == "within_reference":
        return "dans la référence"
    if technical == "not_interpretable":
        return "non interprétable"
    return ""


def format_result_count(n: int) -> str:
    count = max(0, int(n))
    if count == 0:
        return "Aucun résultat correspondant n’a été retrouvé."
    if count == 1:
        return "Un seul résultat correspondant a été retrouvé."
    return f"{count} résultats correspondants ont été retrouvés."


def select_intro_template(intent: str, query_understanding: QueryUnderstanding, evidence_pack: dict[str, Any]) -> str:
    doc_ids = list(query_understanding.requested_doc_ids or [])
    doc_scope = ", ".join(doc_ids) if doc_ids else "les rapports indexés"
    analyte_text = humanize_analyte_list(query_understanding.requested_analytes, evidence_pack)
    condition = humanize_condition(query_understanding, evidence_pack)
    condition_phrase = f" {condition}" if condition else ""
    seed = f"{intent}|{query_understanding.output_format}|{query_understanding.answer_style}|{doc_scope}|{analyte_text}|{condition}"

    if intent in {"cohort_search", "global_patient_lookup"}:
        if condition and query_understanding.requested_value:
            precise = f"J’ai recherché les patients ayant une {analyte_text} {condition}."
        elif condition:
            precise = f"J’ai recherché les patients ayant {analyte_text} {condition}."
        else:
            precise = f"J’ai recherché les patients ayant {analyte_text}."
        opts = [
            precise,
            f"La recherche a été effectuée sur l’ensemble des rapports indexés pour {analyte_text}{condition_phrase}.",
            f"J’ai filtré les rapports indexés pour identifier les patients avec {analyte_text}{condition_phrase}.",
            f"La base a été interrogée pour retrouver les patients répondant au critère : {analyte_text}{condition_phrase}.",
        ]
        return _pick_variant(seed, [o for o in opts if "None" not in o])

    if intent in {"doc_scoped_results", "previous_result_comparison"}:
        return _pick_variant(
            seed,
            [
                f"Dans {doc_scope}, j’ai vérifié les résultats demandés à partir des données extraites du rapport.",
                f"J’ai consulté {doc_scope} pour extraire les valeurs demandées.",
                f"Les résultats suivants proviennent uniquement de {doc_scope}.",
            ],
        )

    if intent in {"multi_doc_comparison", "multi_doc_presence_diff"}:
        return _pick_variant(
            seed,
            [
                "J’ai comparé les deux rapports demandés afin d’identifier les différences techniques.",
                "Voici les éléments présents dans un rapport et absents dans l’autre.",
                "La comparaison ci-dessous se limite aux données structurées extraites des deux documents.",
            ],
        )

    if intent in {"doc_scoped_summary", "immunoanalysis_summary", "toxicology_summary"}:
        return _pick_variant(
            seed,
            [
                f"Voici la synthèse technique des résultats retrouvés dans {doc_scope}.",
                "J’ai regroupé les résultats disponibles par section afin de faciliter la lecture.",
                "Les anomalies techniques ci-dessous sont organisées par section du rapport.",
            ],
        )

    if intent == "diagnostic_safety_question":
        return _pick_variant(
            seed,
            [
                "Non, on ne peut pas conclure à un diagnostic à partir de ces seuls marqueurs.",
                "Non, on ne peut pas conclure à un diagnostic avec ces seuls résultats ; je fournis uniquement une synthèse technique.",
                "Non, on ne peut pas conclure à un cancer sur cette base seule ; la réponse reste strictement technique.",
            ],
        )

    return _pick_variant(
        seed,
        [
            "J’ai vérifié les données retrouvées pour répondre de manière sourcée.",
            "Voici la synthèse des éléments techniques disponibles pour votre demande.",
        ],
    )


def build_professional_intro(query_understanding: QueryUnderstanding, evidence_pack: dict[str, Any]) -> str:
    intent = _safe_str(query_understanding.intent, "unstructured")
    output_format = _safe_str(query_understanding.output_format, "list").lower()
    answer_style = _safe_str(query_understanding.answer_style, "standard").lower()

    if output_format == "json" or answer_style == "yes_no" or output_format == "yes_no":
        return ""

    evidences = list(evidence_pack.get("evidences") or evidence_pack.get("results") or [])
    if not evidences and intent in {"unstructured", "response_transform"}:
        return ""

    if intent == "absence_or_missing_data":
        doc_ids = ", ".join(query_understanding.requested_doc_ids or ["le document demandé"])
        return f"Aucune valeur correspondant à cette demande n’a été retrouvée dans {doc_ids}."

    return select_intro_template(intent, query_understanding, evidence_pack)


def choose_presentation_format(query_understanding: QueryUnderstanding, evidence_pack: dict[str, Any]) -> str:
    output_format = _safe_str(query_understanding.output_format, "list").lower()
    answer_style = _safe_str(query_understanding.answer_style, "standard").lower()
    requested_cols = list(query_understanding.requested_table_columns or [])
    evidences = list(evidence_pack.get("evidences") or evidence_pack.get("results") or [])

    if output_format == "json":
        return "json"
    if answer_style == "yes_no" or output_format == "yes_no":
        return "yes_no"
    if query_understanding.intent in {"cohort_search", "global_patient_lookup"}:
        return "table"
    if output_format == "table" or requested_cols:
        return "table"
    if len(evidences) >= 2:
        homogeneous = all(_safe_str(ev.get("analyte")) for ev in evidences)
        if homogeneous:
            return "table"
    if len(evidences) == 1:
        if query_understanding.intent in {"doc_scoped_results", "diagnostic_safety_question"}:
            return "paragraph"
        return "list"
    if query_understanding.intent in {"doc_scoped_summary", "immunoanalysis_summary", "toxicology_summary"} and len(evidences) >= 2:
        return "table"
    return "list"


def format_source_label(source: dict[str, Any]) -> str:
    filename = _safe_str(source.get("filename"))
    doc_id = _safe_str(source.get("doc_id"), "source")
    page = _safe_int(source.get("page"))
    row = _safe_int(source.get("row"))
    rows = [_safe_int(r) for r in (source.get("rows") or [])]
    rows = [r for r in rows if isinstance(r, int)]

    base = filename or _safe_str(source.get("label")) or doc_id
    base = re.sub(r"\[doc_id=.*?\]", "", base, flags=re.IGNORECASE).strip()
    base = re.sub(r"chunk_id\s*=\s*[^\],\s]+", "", base, flags=re.IGNORECASE).strip()
    base = re.sub(r"/home/[^\s\])]+", "", base).strip()
    base = re.sub(r"[A-Za-z]:\\[^\s\])]+", "", base).strip()
    base = re.sub(r"\bpage\s*(\d+)\s*row\s*(\d+)\b", r"page \1, ligne \2", base, flags=re.IGNORECASE)
    base = re.sub(r"\bpage\s*(\d+)\s*ligne\s*(\d+)\b", r"page \1, ligne \2", base, flags=re.IGNORECASE)
    base = re.sub(r"(ligne\s*\d+)\s*\1\b", r"\1", base, flags=re.IGNORECASE)
    base = re.sub(r"\s{2,}", " ", base).strip()
    has_page = re.search(r"\bpage\s*\d+\b", base, flags=re.IGNORECASE) is not None
    has_line = re.search(r"\bligne(?:s)?\s*\d+", base, flags=re.IGNORECASE) is not None

    if page is not None and not has_page:
        base = f"{base} — page {page}" if base else f"page {page}"

    row_values = sorted(set(rows + ([row] if isinstance(row, int) else [])))
    if row_values and not has_line:
        if len(row_values) == 1:
            base = f"{base}, ligne {row_values[0]}"
        else:
            base = f"{base}, lignes {row_values[0]}–{row_values[-1]}"

    return re.sub(r"\s{2,}", " ", base).strip(" -,")


def deduplicate_sources(sources: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int | None, str], dict[str, Any]] = {}
    for src in sources or []:
        doc_id = _safe_str(src.get("doc_id")).lower()
        if not doc_id:
            continue
        page = int(src.get("page")) if isinstance(src.get("page"), int) else None
        url = _safe_str(src.get("viewer_url") or src.get("url"))
        key = (doc_id, page, url)
        entry = grouped.get(
            key,
            {
                "doc_id": _safe_str(src.get("doc_id")),
                "filename": src.get("filename"),
                "page": page,
                "row": src.get("row"),
                "rows": [],
                "url": src.get("url"),
                "viewer_url": src.get("viewer_url"),
                "label": src.get("label"),
            },
        )
        if isinstance(src.get("row"), int):
            entry["rows"].append(int(src["row"]))
        if isinstance(src.get("rows"), list):
            entry["rows"].extend([int(r) for r in src["rows"] if isinstance(r, int)])
        if not entry.get("filename") and src.get("filename"):
            entry["filename"] = src.get("filename")
        if not entry.get("label") and src.get("label"):
            entry["label"] = src.get("label")
        if not entry.get("url") and src.get("url"):
            entry["url"] = src.get("url")
        if not entry.get("viewer_url") and src.get("viewer_url"):
            entry["viewer_url"] = src.get("viewer_url")
        grouped[key] = entry

    out: list[dict[str, Any]] = []
    for _, entry in sorted(grouped.items(), key=lambda it: (it[1].get("doc_id") or "", it[1].get("page") or 0)):
        entry["rows"] = sorted(set(entry.get("rows") or []))
        entry["label"] = format_source_label(entry)
        out.append(entry)
    return out


def _source_lines(source_citations: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for src in deduplicate_sources(source_citations):
        href = src.get("url") or src.get("viewer_url")
        if href:
            lines.append(f"- [{src.get('label')}]({href})")
        else:
            lines.append(f"- {src.get('label')}")
    return lines


def build_short_conclusion(intent: str, evidence_pack: dict[str, Any], safety_intent: str | None) -> str | None:
    if safety_intent or intent == "diagnostic_safety_question":
        return "Conclusion technique : aucune conclusion diagnostique ne peut être tirée uniquement de ces résultats."

    evidences = list(evidence_pack.get("evidences") or evidence_pack.get("results") or [])
    if not evidences:
        return None

    first = evidences[0]
    status = _safe_str(first.get("technical_status_code")).lower()
    analyte = _safe_str(first.get("analyte"), "l’analyte")
    if len(evidences) == 1:
        if status == "above_reference":
            return f"Conclusion technique : {analyte} est au-dessus de l’intervalle de référence indiqué."
        if status == "below_reference":
            return f"Conclusion technique : {analyte} est en dessous de l’intervalle de référence indiqué."
        if status == "within_reference":
            return "Conclusion technique : le résultat retrouvé respecte le critère demandé."
        return "Conclusion technique : le résultat affiché correspond strictement aux données extraites."

    options_by_intent: dict[str, list[str]] = {
        "cohort_search": [
            "Conclusion technique : la réponse reste limitée aux résultats retrouvés dans les rapports indexés.",
            "Conclusion technique : ces résultats sont basés uniquement sur les données extraites et les sources citées.",
        ],
        "multi_doc_comparison": [
            "Conclusion technique : la comparaison repose uniquement sur les valeurs retrouvées dans les documents demandés.",
            "Conclusion technique : les écarts indiqués reflètent uniquement les mesures disponibles dans les rapports comparés.",
        ],
        "multi_doc_presence_diff": [
            "Conclusion technique : la présence/absence ci-dessus correspond strictement aux données extraites de chaque rapport.",
        ],
        "doc_scoped_results": [
            "Conclusion technique : ces résultats proviennent uniquement du document demandé.",
            "Conclusion technique : la synthèse est strictement fondée sur les données extraites et les sources associées.",
        ],
    }
    options = options_by_intent.get(intent) or []
    if not options:
        return None
    choice = _pick_variant(f"{intent}|{len(evidences)}", options)
    if choice in _COLD_CONCLUSIONS:
        return None
    return choice


def _table(columns: list[str], rows: list[dict[str, Any]]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body: list[str] = [header, sep]
    for row in rows:
        body.append("| " + " | ".join(_safe_str(row.get(c), "non disponible") for c in columns) + " |")
    return "\n".join(body)


def _source_cell(ev: dict[str, Any]) -> str:
    label = _safe_str(ev.get("source_label"))
    if not label:
        filename = _safe_str(ev.get("filename")) or _safe_str(ev.get("doc_id"), "source")
        page = ev.get("page")
        row = ev.get("row")
        label = filename
        if isinstance(page, int):
            label += f" — page {page}"
        if isinstance(row, int):
            label += f", ligne {row}"
    href = _safe_str(ev.get("source_url") or ev.get("viewer_url"))
    if href:
        return f"[{label}]({href})"
    return label


def _writer_intent(intent: str) -> str:
    mapping = {
        "cohort_search": "cohort_search",
        "global_patient_lookup": "cohort_search",
        "doc_scoped_results": "doc_scoped_query",
        "previous_result_comparison": "comparison",
        "multi_doc_comparison": "comparison",
        "multi_doc_presence_diff": "comparison",
        "doc_scoped_summary": "section_summary",
        "immunoanalysis_summary": "section_summary",
        "toxicology_summary": "section_summary",
        "diagnostic_safety_question": "safety",
        "response_transform": "response_transform",
        "absence_or_missing_data": "missing_data",
    }
    return mapping.get(_safe_str(intent), "doc_scoped_query")


def _writer_output_format(query_understanding: QueryUnderstanding, evidence_pack: dict[str, Any]) -> str:
    chosen = choose_presentation_format(query_understanding, evidence_pack)
    if chosen in {"table", "list", "paragraph", "json"}:
        return chosen
    return "auto"


def _normalized_writer_result(ev: dict[str, Any]) -> dict[str, Any]:
    status = _safe_str(ev.get("technical_status") or ev.get("status"))
    status_code = _safe_str(ev.get("technical_status_code") or ev.get("interpretation_status")).lower()
    if not status and status_code == "above_reference":
        status = "au-dessus de la référence"
    elif not status and status_code == "below_reference":
        status = "en dessous de la référence"
    elif not status and status_code == "within_reference":
        status = "dans la référence"
    elif not status:
        status = "non interprétable"

    source_label = _safe_str(ev.get("source_label"))
    if not source_label:
        source_label = format_source_label(
            {
                "filename": ev.get("filename"),
                "doc_id": ev.get("doc_id"),
                "page": ev.get("page"),
                "row": ev.get("row"),
            }
        )
    return {
        "patient": _safe_str(ev.get("patient_token") or ev.get("patient")),
        "doc_id": _safe_str(ev.get("doc_id")),
        "filename": _safe_str(ev.get("filename")),
        "page": _safe_int(ev.get("page")),
        "row": _safe_int(ev.get("row")),
        "analyte": _safe_str(ev.get("analyte")),
        "value": _safe_str(ev.get("current_value") or ev.get("value_raw")),
        "unit": _safe_str(ev.get("unit")),
        "reference": _safe_str(ev.get("reference") or ev.get("reference_range")),
        "status": status,
        "previous_result": ev.get("previous_result"),
        "variation": ev.get("variation"),
        "source_label": source_label,
        "source_url": _safe_str(ev.get("source_url")),
        "viewer_url": _safe_str(ev.get("viewer_url")),
    }


def build_writer_evidence_pack(
    *,
    user_question: str,
    query_understanding: QueryUnderstanding,
    evidence_pack: dict[str, Any],
    source_citations: list[dict[str, Any]],
) -> dict[str, Any]:
    results = [_normalized_writer_result(ev) for ev in (evidence_pack.get("evidences") or evidence_pack.get("results") or [])]
    recent_style_history = list(evidence_pack.get("recent_style_history") or [])
    constraints = {
        "requested_doc_ids": list(query_understanding.requested_doc_ids or []),
        "requested_analytes": list(query_understanding.requested_analytes or []),
        "excluded_analytes": list(getattr(query_understanding, "excluded_analytes", []) or []),
        "technical_condition": query_understanding.technical_condition,
        "comparison_operator": getattr(query_understanding, "comparison_operator", None),
        "requested_value": query_understanding.requested_value,
        "requested_unit": getattr(query_understanding, "requested_unit", None),
        "requested_columns": list(query_understanding.requested_table_columns or []),
        "source_clickable_requested": bool(getattr(query_understanding, "source_clickable_requested", False)),
        "diagnostic_safety": bool(query_understanding.safety_intent),
    }
    return {
        "user_question": user_question,
        "intent": _writer_intent(query_understanding.intent),
        "output_format": _writer_output_format(query_understanding, evidence_pack),
        "answer_style": "yes_no" if query_understanding.answer_style == "yes_no" else "professional",
        "language": _safe_str(query_understanding.language, "fr"),
        "constraints": constraints,
        "results": results,
        "missing_items": list(evidence_pack.get("missing_items") or []),
        "sources": deduplicate_sources(source_citations or []),
        "response_brief": {
            "task_goal": "Répondre à la question utilisateur de manière claire et sourcée.",
            "audience": "Utilisateur non technique consultant des résultats biologiques.",
            "tone": "professionnel, humain, clair, prudent",
            "verbosity": "concise",
            "format": _writer_output_format(query_understanding, evidence_pack),
            "must_include": ["critère utilisateur réel", "résultats extraits", "sources lisibles"],
            "must_not_include": ["chunk_id", "chemin local", "logs techniques", "diagnostic non autorisé", "aliases internes"],
            "grounding_policy": "Toutes les valeurs doivent venir de l’evidence_pack.",
            "style_policy": {
                "avoid_repetitive_intros": True,
                "avoid_generic_sentences": True,
                "vary_intro_and_conclusion": True,
                "no_template_phrasing": True,
            },
        },
        "recent_style_history": recent_style_history[-20:],
        "source_policy": {
            "show_sources": True,
            "clickable_sources": True,
            "group_duplicate_sources": True,
        },
    }


def _build_content_table(
    intent: str,
    evidences: list[dict[str, Any]],
    include_previous: bool,
    requested_columns: list[str] | None = None,
    source_clickable_requested: bool = False,
) -> str:
    requested_cols = [str(c).strip().lower() for c in (requested_columns or []) if str(c).strip()]
    include_source_col = (not requested_cols and source_clickable_requested) or ("source" in set(requested_cols))

    if requested_cols:
        column_map = {
            "patient": "Patient",
            "report": "Report",
            "document": "Document",
            "analyte": "Analyte",
            "valeur_actuelle": "Valeur actuelle",
            "valeur": "Valeur actuelle",
            "unite": "Unité",
            "reference": "Référence",
            "statut": "Statut",
            "resultat_anterieur": "Résultat antérieur",
            "variation": "Variation",
            "source": "Source",
        }
        columns = [column_map[c] for c in requested_cols if c in column_map]
        if columns:
            normalized_rows: list[dict[str, Any]] = []
            for ev in evidences:
                normalized_rows.append(
                    {
                        "Patient": _safe_str(ev.get("patient_token"), "non disponible"),
                        "Report": _safe_str(ev.get("doc_id")),
                        "Document": _safe_str(ev.get("comparison_side") or ev.get("doc_id")),
                        "Analyte": _safe_str(ev.get("analyte"), "non précisé"),
                        "Valeur actuelle": (
                            _safe_str(ev.get("current_value"), "non disponible")
                            + (f" {_safe_str(ev.get('unit'))}" if _safe_str(ev.get("unit")) else "")
                        ).strip(),
                        "Unité": _safe_str(ev.get("unit"), "non disponible"),
                        "Référence": _safe_str(ev.get("reference"), "non disponible"),
                        "Statut": _safe_str(ev.get("technical_status"), "non interprétable"),
                        "Source": _source_cell(ev),
                        "Résultat antérieur": _safe_str(ev.get("previous_result"), "non disponible"),
                        "Variation": _safe_str(ev.get("variation"), "non comparable"),
                    }
                )
            return _table(columns, normalized_rows)

    if intent in {"cohort_search", "global_patient_lookup"}:
        rows = [
            {
                "Patient": _safe_str(ev.get("patient_token"), "non disponible"),
                "Report": _safe_str(ev.get("doc_id")),
                "Analyte": _safe_str(ev.get("analyte"), "non précisé"),
                "Valeur actuelle": (
                    _safe_str(ev.get("current_value"), "non disponible")
                    + (f" {_safe_str(ev.get('unit'))}" if _safe_str(ev.get("unit")) else "")
                ).strip(),
                "Référence": _safe_str(ev.get("reference"), "non disponible"),
                "Statut": _safe_str(ev.get("technical_status"), "non interprétable"),
                **({"Source": _source_cell(ev)} if include_source_col else {}),
            }
            for ev in evidences
        ]
        cols = ["Patient", "Report", "Analyte", "Valeur actuelle", "Référence", "Statut"]
        if include_source_col:
            cols.append("Source")
        return _table(cols, rows)

    if intent == "multi_doc_presence_diff":
        rows = [
            {
                "Analyte": _safe_str(ev.get("analyte"), "non précisé"),
                "Présent dans": _safe_str(ev.get("present_in")),
                "Absent dans": _safe_str(ev.get("absent_in")),
            }
            for ev in evidences
        ]
        return _table(["Analyte", "Présent dans", "Absent dans"], rows)

    rows = []
    for ev in evidences:
        row = {
            "Analyte": _safe_str(ev.get("analyte"), "non précisé"),
            "Valeur actuelle": (
                _safe_str(ev.get("current_value"), "non disponible")
                + (f" {_safe_str(ev.get('unit'))}" if _safe_str(ev.get("unit")) else "")
            ).strip(),
            "Référence": _safe_str(ev.get("reference"), "non disponible"),
            "Statut": _safe_str(ev.get("technical_status"), "non interprétable"),
            **({"Source": _source_cell(ev)} if include_source_col else {}),
        }
        if _safe_str(ev.get("doc_id")):
            row["Document"] = _safe_str(ev.get("comparison_side") or ev.get("doc_id"))
        if include_previous:
            row["Résultat antérieur"] = _safe_str(ev.get("previous_result"), "non disponible")
            row["Variation"] = _safe_str(ev.get("variation"), "non comparable")
        rows.append(row)

    if intent == "response_transform" and requested_cols and rows:
        column_map = {
            "patient": "Patient",
            "report": "Report",
            "document": "Document",
            "analyte": "Analyte",
            "valeur_actuelle": "Valeur actuelle",
            "valeur": "Valeur actuelle",
            "unite": "Unité",
            "reference": "Référence",
            "statut": "Statut",
            "resultat_anterieur": "Résultat antérieur",
            "variation": "Variation",
            "source": "Source",
        }
        normalized_rows: list[dict[str, Any]] = []
        for ev in evidences:
            normalized_rows.append(
                {
                    "Patient": _safe_str(ev.get("patient_token"), "non disponible"),
                    "Report": _safe_str(ev.get("doc_id")),
                    "Document": _safe_str(ev.get("comparison_side") or ev.get("doc_id")),
                    "Analyte": _safe_str(ev.get("analyte"), "non précisé"),
                    "Valeur actuelle": (
                        _safe_str(ev.get("current_value"), "non disponible")
                        + (f" {_safe_str(ev.get('unit'))}" if _safe_str(ev.get("unit")) else "")
                    ).strip(),
                    "Unité": _safe_str(ev.get("unit"), "non disponible"),
                    "Référence": _safe_str(ev.get("reference"), "non disponible"),
                    "Statut": _safe_str(ev.get("technical_status"), "non interprétable"),
                    "Source": _source_cell(ev),
                    "Résultat antérieur": _safe_str(ev.get("previous_result"), "non disponible"),
                    "Variation": _safe_str(ev.get("variation"), "non comparable"),
                }
            )
        columns = [column_map[c] for c in requested_cols if c in column_map]
        if columns:
            return _table(columns, normalized_rows)

    columns = list(rows[0].keys()) if rows else ["Analyte", "Valeur actuelle", "Référence", "Statut"]
    return _table(columns, rows)


def _build_content_list(evidences: list[dict[str, Any]], include_previous: bool) -> str:
    lines: list[str] = []
    for ev in evidences:
        line = (
            f"- {ev.get('analyte')}: {ev.get('current_value') or 'non disponible'}"
            f"{(' ' + _safe_str(ev.get('unit'))) if _safe_str(ev.get('unit')) else ''}"
            f" | référence: {_safe_str(ev.get('reference'), 'non disponible')}"
            f" | statut: {_safe_str(ev.get('technical_status'), 'non interprétable')}"
        )
        if include_previous:
            line += f" | antérieur: {_safe_str(ev.get('previous_result'), 'non disponible')}"
            line += f" | variation: {_safe_str(ev.get('variation'), 'non comparable')}"
        lines.append(line)
    return "\n".join(lines)


def _build_paragraph(evidences: list[dict[str, Any]], query: str) -> str:
    if not evidences:
        return "Aucune donnée mesurée correspondante n’a été retrouvée."

    primary = evidences[0]
    yn = _yn_prefix(
        query,
        _safe_str(primary.get("technical_status_code")),
        _safe_str(primary.get("reference")),
    )
    return (
        f"{yn} — {_safe_str(primary.get('analyte'), 'analyte')} = {_safe_str(primary.get('current_value'), 'non disponible')}"
        f"{(' ' + _safe_str(primary.get('unit'))) if _safe_str(primary.get('unit')) else ''} ; "
        f"référence : {_safe_str(primary.get('reference'), 'non disponible')} ; "
        f"statut technique : {_safe_str(primary.get('technical_status'), 'non interprétable')}."
    )


def _build_json_answer(
    *,
    user_question: str,
    query_understanding: QueryUnderstanding,
    evidence_pack: dict[str, Any],
    source_citations: list[dict[str, Any]],
) -> str:
    def _sanitize_result_record(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "doc_id": item.get("doc_id"),
            "patient": item.get("patient_token") or item.get("patient"),
            "page": item.get("page"),
            "row": item.get("row"),
            "analyte": item.get("analyte"),
            "analyte_norm": item.get("analyte_norm"),
            "value": item.get("current_value") or item.get("value_raw") or item.get("value"),
            "unit": item.get("unit"),
            "reference": item.get("reference") or item.get("reference_range"),
            "status": item.get("technical_status") or item.get("interpretation_status") or item.get("status"),
            "status_code": item.get("technical_status_code"),
            "previous_result": item.get("previous_result"),
            "variation": item.get("variation"),
            "source_label": item.get("source_label") or item.get("source"),
            "source_url": item.get("source_url"),
            "viewer_url": item.get("viewer_url"),
        }

    payload = {
        "question": user_question,
        "intent": query_understanding.intent,
        "output_format": "json",
        "constraints": {
            "requested_doc_ids": list(query_understanding.requested_doc_ids or []),
            "requested_analytes": list(query_understanding.requested_analytes or []),
            "requested_columns": list(query_understanding.requested_table_columns or []),
            "technical_condition": query_understanding.technical_condition,
            "safety_intent": query_understanding.safety_intent,
        },
        "results": [_sanitize_result_record(r) for r in list(evidence_pack.get("evidences") or evidence_pack.get("results") or [])],
        "evidences": [_sanitize_result_record(r) for r in list(evidence_pack.get("evidences") or evidence_pack.get("results") or [])],
        "missing_items": list(evidence_pack.get("missing_items") or []),
        "sources": [
            {
                "label": s.get("label"),
                "doc_id": s.get("doc_id"),
                "page": s.get("page"),
                "row": s.get("row"),
                "url": s.get("url"),
                "viewer_url": s.get("viewer_url"),
            }
            for s in deduplicate_sources(source_citations)
        ],
    }
    return json.dumps(payload, ensure_ascii=False)


def _yn_prefix(query: str, status_code: str | None, ref: str | None) -> str:
    qn = norm_text(query)
    wants_en = any(k in qn for k in ["yes/no", "yes or no", "yes no", "answer only yes", "respond only yes", "yes ou no"])
    if not str(ref or "").strip() or str(ref).strip().lower() == "non disponible":
        return "Cannot determine" if wants_en else "Impossible à déterminer"
    in_range = str(status_code or "").strip().lower() == "within_reference"
    if wants_en:
        return "No" if in_range else "Yes"
    return "Non" if in_range else "Oui"


def render_professional_fallback(
    evidence_pack: dict[str, Any],
    query_understanding: QueryUnderstanding,
    *,
    user_question: str,
    source_citations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    evidences = list(evidence_pack.get("evidences") or evidence_pack.get("results") or [])
    missing_items = list(evidence_pack.get("missing_items") or [])
    sources = deduplicate_sources(source_citations or [])

    presentation = choose_presentation_format(query_understanding, evidence_pack)
    intent = _safe_str(query_understanding.intent, "unstructured")

    if presentation == "json":
        answer = _build_json_answer(
            user_question=user_question,
            query_understanding=query_understanding,
            evidence_pack=evidence_pack,
            source_citations=source_citations or [],
        )
        return {
            "intro": "",
            "content_type": "json",
            "content": answer,
            "conclusion": "",
            "sources": sources,
            "rendering_hints": {"preferred_format": "json", "show_sources": False, "strict_json": True},
            "answer": answer,
            "mode": "deterministic_professional_fallback",
            "llm_error": None,
        }

    if presentation == "yes_no":
        if not evidences:
            doc_scope = ", ".join(query_understanding.requested_doc_ids or ["le document demandé"])
            analyte_text = humanize_analyte_list(query_understanding.requested_analytes, evidence_pack)
            answer = f"Non — {analyte_text} non retrouvé dans {doc_scope}."
        else:
            first = dict(evidences[0] or {})
            analyte = _safe_str(first.get("analyte"), "Analyte")
            value = _safe_str(first.get("current_value"), _safe_str(first.get("value_raw"), "non disponible"))
            unit = _safe_str(first.get("unit"), "")
            reference = _safe_str(first.get("reference"), _safe_str(first.get("reference_range"), "non disponible"))
            status_code = _safe_str(first.get("technical_status_code"), "")
            prefix = _yn_prefix(user_question, status_code, reference)
            value_text = value if not unit else f"{value} {unit}"
            answer = f"{prefix} — {analyte}: {value_text} (référence: {reference})."
        src_lines = _source_lines(source_citations or [])
        if src_lines:
            answer = answer.rstrip() + "\n\nSources :\n" + "\n".join(src_lines)
        return {
            "intro": "",
            "content_type": "yes_no",
            "content": answer,
            "conclusion": "",
            "sources": sources,
            "rendering_hints": {"preferred_format": "yes_no", "show_sources": True, "strict_json": False},
            "answer": answer,
            "mode": "deterministic_professional_fallback",
            "llm_error": None,
        }

    intro = build_professional_intro(query_understanding, evidence_pack)
    count_line = format_result_count(len(evidences))

    include_previous = bool(query_understanding.requires_previous_results)
    if presentation == "table":
        content = (
            _build_content_table(
                intent,
                evidences,
                include_previous,
                query_understanding.requested_table_columns,
                bool(getattr(query_understanding, "source_clickable_requested", False)),
            )
            if evidences
            else "Aucun résultat exploitable."
        )
    elif presentation == "list":
        content = _build_content_list(evidences, include_previous) if evidences else "Aucun résultat exploitable."
    else:
        content = _build_paragraph(evidences[:1], user_question)

    if missing_items and presentation != "yes_no":
        doc_scope = ", ".join(query_understanding.requested_doc_ids or ["le document demandé"])
        miss = "\n".join(f"- {_canonical_analyte_display(str(m))}: non retrouvé dans {doc_scope}." for m in missing_items)
        content = content.rstrip() + "\n\nÉléments non retrouvés :\n" + miss

    conclusion = build_short_conclusion(intent, evidence_pack, query_understanding.safety_intent)
    parts = [p for p in [intro.strip(), count_line.strip(), content.strip(), (conclusion or "").strip()] if p]
    answer = "\n\n".join(parts)

    src_lines = _source_lines(source_citations or [])
    has_source_column_in_table = presentation == "table" and bool(re.search(r"(?im)^\|\s*.*\bsource\b.*\|$", content or ""))
    if src_lines and not has_source_column_in_table:
        answer = answer.rstrip() + "\n\nSources :\n" + "\n".join(src_lines)

    return {
        "intro": intro.strip(),
        "content_type": presentation,
        "content": content.strip(),
        "conclusion": (conclusion or "").strip(),
        "sources": sources,
        "rendering_hints": {
            "preferred_format": presentation,
            "show_sources": True,
            "strict_json": False,
        },
        "answer": answer.strip(),
        "mode": "deterministic_professional_fallback",
        "llm_error": None,
    }


def compose_professional_answer(
    user_question: str,
    query_understanding: QueryUnderstanding,
    evidence_pack: dict[str, Any],
    mode: str = "auto",
    *,
    source_citations: list[dict[str, Any]] | None = None,
    llm_client: LLMClient | None = None,
    provider: str = "ollama",
    model: str = "qwen3:4b",
    temperature: float = 0.0,
    num_ctx: int = 4096,
    max_tokens: int = 420,
    timeout: int = 18,
    retry_feedback: str | None = None,
) -> dict[str, Any]:
    fallback = render_professional_fallback(
        evidence_pack=evidence_pack,
        query_understanding=query_understanding,
        user_question=user_question,
        source_citations=source_citations or [],
    )

    if mode == "fallback":
        return fallback

    presentation = choose_presentation_format(query_understanding, evidence_pack)
    if presentation in {"json", "yes_no"}:
        return fallback

    evidences = list(evidence_pack.get("evidences") or evidence_pack.get("results") or [])
    if not evidences:
        return fallback

    compact_pack = build_writer_evidence_pack(
        user_question=user_question,
        query_understanding=query_understanding,
        evidence_pack=evidence_pack,
        source_citations=source_citations or [],
    )

    prompt = (
        f"{PROFESSIONAL_WRITER_SYSTEM_PROMPT}\n\n"
        "Sortie attendue: réponse finale uniquement.\n"
        "/no_think\n\n"
        "Question utilisateur:\n"
        f"{user_question.strip()}\n\n"
        "evidence_pack JSON:\n"
        f"{json.dumps(compact_pack, ensure_ascii=False)}\n"
    )
    if retry_feedback:
        prompt += (
            "\nCorrections obligatoires:\n"
            "Corrige uniquement le style/format ci-dessous sans modifier aucune donnée factuelle.\n"
            f"{retry_feedback.strip()}\n"
        )

    client = llm_client or LLMClient(provider=provider)
    try:
        llm_answer = client.generate(
            prompt=prompt,
            model=model,
            temperature=0.0 if temperature is None else min(float(temperature), 0.2),
            num_ctx=max(2048, int(num_ctx)),
            max_tokens=max(180, min(int(max_tokens), 520)),
            timeout=max(6, min(int(timeout), 30)),
            keep_alive="5m",
        ).strip()
        if not llm_answer:
            out = dict(fallback)
            out["llm_error"] = "empty_llm_answer"
            return out

        if re.search(r"\brésultat\(s\)|\bcorrespondant\(s\)", llm_answer, flags=re.IGNORECASE):
            out = dict(fallback)
            out["mode"] = "llm_writer_quality_fallback"
            out["llm_error"] = "ugly_pluralization"
            return out

        has_source_col = bool(re.search(r"(?im)^\|\s*.*\bsource\b.*\|$", llm_answer or ""))
        if "sources" not in llm_answer.lower() and not has_source_col:
            src_lines = _source_lines(source_citations or [])
            if src_lines:
                llm_answer = llm_answer.rstrip() + "\n\nSources :\n" + "\n".join(src_lines)

        return {
            "intro": "",
            "content_type": "paragraph",
            "content": llm_answer,
            "conclusion": "",
            "sources": deduplicate_sources(source_citations or []),
            "rendering_hints": {
                "preferred_format": "table" if "|" in llm_answer and "---" in llm_answer else "paragraph",
                "show_sources": True,
                "strict_json": False,
            },
            "answer": llm_answer,
            "mode": "llm_professional_writer",
            "llm_error": None,
        }
    except LLMClientError as exc:
        out = dict(fallback)
        out["mode"] = "llm_writer_error_fallback"
        out["llm_error"] = str(exc)
        return out
