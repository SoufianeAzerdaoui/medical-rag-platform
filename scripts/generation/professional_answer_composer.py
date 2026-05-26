from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, NotRequired, TypedDict

from llm_client import LLMClient, LLMClientError
from model_settings import (
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_NUM_CTX,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LLM_TEMPERATURE,
)
from query_understanding import QueryUnderstanding, analyte_display_name, norm_text
from medical_entity_resolver import canonicalize_analyte, get_display_analyte_label
from source_normalization import dedup_normalized_sources, normalize_source_for_response


LOGGER = logging.getLogger("medical_rag.writer_contract")


class WriterLockedResultRow(TypedDict):
    analyte: str
    analyte_norm: str
    value: str
    unit: str
    reference: str
    status: str
    source_label: str


class WriterScopePayload(TypedDict):
    requested_doc_ids: list[str]
    effective_doc_ids: list[str]
    requested_analytes: list[str]
    effective_analytes: list[str]
    technical_condition: str | None
    row_count: int
    scope_coherent: bool


class WriterEvidenceContractPayload(TypedDict):
    contract_version: str
    rows_filtered: bool
    rows_fact_locked: bool
    scope_coherent: bool
    sources_normalized: bool
    sources_deduplicated: bool
    noise_reduction: list[str]
    canonical_requested_analytes: list[str]
    source_labels_normalized: bool


class WriterEvidencePack(TypedDict):
    original_user_question: str
    user_question: str
    intent: str
    response_strategy: str
    response_strategy_reason: NotRequired[str | None]
    output_format: str
    answer_style: str
    language: str
    presentation_intent: dict[str, Any]
    visualization_facts: dict[str, Any]
    constraints: dict[str, Any]
    results: list[WriterLockedResultRow]
    results_locked: list[WriterLockedResultRow]
    missing_items: list[Any]
    sources: list[dict[str, Any]]
    scope: WriterScopePayload
    evidence_contract: WriterEvidenceContractPayload
    response_brief: dict[str, Any]
    recent_style_history: list[dict[str, Any]]
    source_policy: dict[str, Any]


PROFESSIONAL_WRITER_SYSTEM_PROMPT = """Tu es un rédacteur médical technique intégré à un système RAG.

Mission :
Rédiger une réponse professionnelle, naturelle et concise à partir des faits fournis par le backend.

Source de vérité :
- L’evidence_pack est la seule source autorisée.
- Tu ne dois jamais inventer, modifier ou compléter une valeur, une unité, une référence, un patient, un document, un résultat antérieur, une source ou un diagnostic.
- Tu ne dois jamais utiliser ta connaissance générale pour ajouter des faits médicaux.
- Si une information n’est pas dans l’evidence_pack, indique qu’elle n’est pas disponible.

Rôle exact :
- Le backend sélectionne les valeurs, les références, les sources, les statuts, les tableaux et les visualisations.
- Tu reformules uniquement.
- Tu n'es ni routeur, ni planner, ni answerability gate.
- Tu ne décides jamais si la requête est answerable, ambiguë ou unsafe.
- Tu ne choisis jamais une valeur, une plage physiologique, un résultat antérieur, une source ou un diagnostic.
- Tu ne sélectionnes jamais des lignes toi-même et tu ne recalcules jamais une valeur, un écart ou un statut.
- Tu reformules uniquement les lignes déjà fournies dans results (et results_locked quand présent).
- Tu ne rajoutes aucune ligne au tableau.
- Tu ne supprimes aucune ligne importante fournie dans results, sauf si l’utilisateur demande explicitement un filtre.

Style :
- Réponds en français clair, professionnel et concis.
- Ne sois pas verbeux.
- Ne répète pas toujours la même phrase d’introduction.
- L’introduction doit être spécifique à la question.
- Ne commence pas directement par un tableau sauf si l’utilisateur demande uniquement un tableau.
- N’affiche jamais les aliases internes comme “tshus, tsh”.
- Utilise les noms humains des analytes depuis results[].analyte.
- Gère correctement le singulier/pluriel.
- Évite les formulations mécaniques.
- Si un comptage est utile, formule-le naturellement selon la question.
- Ne donne jamais de diagnostic médical.

Formats :
- Si output_format = table, produis un tableau Markdown propre.
- Si output_format = json ou answer_style = strict_json, retourne uniquement du JSON valide, sans texte autour.
- Utilise Oui/Non uniquement si answer_style = yes_no est explicitement défini par le backend.
- Si output_format = chart, respecte la visualisation demandée ou explique clairement pourquoi une alternative est utilisée.

Sources :
- Utilise uniquement les sources fournies par le backend.
- Pour les sources, utilise source_label.
- Si source_clickable_requested = true et viewer_url/source_url est disponible, affiche la source en Markdown : [source_label](viewer_url ou source_url).
- Si source_clickable_requested = true mais aucun lien n’est disponible, affiche source_label en texte simple et indique que la source est disponible uniquement en texte.
- Ne jamais inventer d’URL.
- N’affiche jamais chunk_id, path local, request_id ou logs techniques.

Grounding :
- Tous les analytes affichés doivent exister dans results.
- Toutes les valeurs affichées doivent exister dans results.
- Toutes les sources affichées doivent exister dans results.
- Ne modifie jamais les décimales, unités ou intervalles.
- Ne transforme jamais une valeur mesurée en plage de référence, ni l’inverse.
- Si une information semble manquante, dis qu’elle est non disponible ; n’infère pas.

Sécurité :
- Ne donne pas de diagnostic.
- Si la question demande une interprétation médicale, limite-toi à une interprétation technique fondée sur les références fournies.
- Si le contexte est insuffisant, dis-le clairement.
- Si la question est ouverte ou ambiguë, n'invente jamais de réponse clinique: reformule seulement les faits disponibles et leurs limites.
"""


PROFESSIONAL_WRITER_VISUALIZATION_RULES = """
Règles de formulation visualisation :
- Le backend fournit visualization_facts avec les faits obligatoires.
- Tu dois produire une introduction NATURELLE et PROFESSIONNELLE.
- Ne pas utiliser de structure rigide de type "Graphique demandé : ... Rendu affiché : ...".
- Si visualization_facts.fallback_used = true :
  1) mentionne le format demandé (ex: Arithmetic Line-Graph) ;
  2) explique pourquoi une alternative (ex: graphique en barres) est utilisée ;
  3) lie cette explication aux contraintes des données (ex: unités différentes, pas de série temporelle).
- INTERDICTION ABSOLUE : ne jamais inclure de labels de données concaténés (ex: INSULINET4LIBRE, TSHusT3) ou de pourcentages d'écarts (ex: 600%) dans ton texte d'introduction. Ces éléments seront rendus séparément par l'interface graphique.
- Ne modifie jamais les valeurs de visualization_facts.
- Ton texte doit être fluide, comme un expert s'adressant à un utilisateur.
"""


def _route_specific_writer_block(query_understanding: QueryUnderstanding) -> str:
    intent = str(getattr(query_understanding, "intent", "") or "").strip().lower()
    if intent == "response_transform":
        return (
            "RÈGLES SPÉCIFIQUES response_transform :\n"
            "- Tu transforms uniquement la forme de la réponse précédente, jamais le fond.\n"
            "- Tu conserves strictement les mêmes résultats, les mêmes sources et le même périmètre documentaire.\n"
            "- Tu n'ajoutes aucune ligne, aucun analyte, aucune source, aucun document et aucun regroupement implicite.\n"
            "- Tu ne fusionnes pas plusieurs résultats en une phrase si cela fait disparaître une information factuelle.\n"
            "- Si le backend demande un tableau, JSON ou résumé, applique seulement ce format à partir de results_locked.\n"
            "- Tu ne réintroduis jamais un contexte ancien ou externe absent de results_locked.\n"
        )
    if intent in {"cohort_search", "global_patient_lookup"}:
        return (
            "RÈGLES SPÉCIFIQUES cohort/global search :\n"
            "- Tu conserves chaque ligne de résultat comme une observation distincte.\n"
            "- Tu ne fusionnes jamais plusieurs patients, documents ou analytes en une seule conclusion globale.\n"
            "- Tu ne transformes jamais un critère numérique en jugement clinique.\n"
            "- Si le backend a sélectionné plusieurs résultats, tu ne réduis pas arbitrairement la liste à un seul cas.\n"
            "- Tu respectes strictement le filtre demandé; tu ne rajoutes aucun patient ou document absent de results_locked.\n"
        )
    if intent in {"multi_doc_comparison", "doc_pair_comparison", "multi_doc_presence_diff"}:
        return (
            "RÈGLES SPÉCIFIQUES comparaison multi-doc :\n"
            "- Tu conserves strictement la séparation par document.\n"
            "- Tu ne permutes jamais report A et report B.\n"
            "- Tu ne recalcules jamais un delta, une tendance ou une amélioration si le backend ne l'a pas déjà fournie.\n"
            "- Tu ne résumes jamais une comparaison en une phrase vague si cela masque les écarts documentaires.\n"
            "- Tu gardes visibles les différences de présence/absence, de valeurs et de sources déjà fournies.\n"
        )
    if intent in {"doc_scoped_summary", "immunoanalysis_summary", "toxicology_summary"}:
        return (
            "RÈGLES SPÉCIFIQUES summary :\n"
            "- Tu distingues clairement les faits anormaux des faits simplement descriptifs.\n"
            "- Tu n'inventes jamais une anomalie, une normalité rassurante ou une hiérarchie clinique.\n"
            "- Tu ne remplaces jamais une référence textuelle complexe par une borne simplifiée.\n"
            "- Tu gardes la synthèse descriptive, sans extrapolation clinique.\n"
        )
    if intent in {"doc_scoped_results", "previous_result_comparison"}:
        return (
            "RÈGLES SPÉCIFIQUES résultats ciblés :\n"
            "- Tu reprends uniquement les mesures explicitement fournies par le backend.\n"
            "- Tu ne transformes jamais une mesure actuelle en résultat antérieur, ni l'inverse.\n"
            "- Tu ne déduis jamais un statut technique si le backend ne l'a pas fourni.\n"
        )
    if intent == "diagnostic_safety_question":
        return (
            "RÈGLES SPÉCIFIQUES safety diagnostic :\n"
            "- Tu conserves un refus technique explicite de diagnostic.\n"
            "- Tu ne reformules jamais le refus en quasi-diagnostic probabiliste.\n"
            "- Tu peux résumer les résultats disponibles, mais sans conclure à une maladie ni proposer une conduite thérapeutique.\n"
        )
    return ""


def _llm_quality_guard_disabled() -> bool:
    v = str(os.getenv("MEDICAL_RAG_DISABLE_LLM_QUALITY_GUARD", "false")).strip().lower()
    return v in {"1", "true", "yes", "on"}


_COLD_CONCLUSIONS = {
    "Ces éléments proviennent uniquement des données indexées.",
    "L’interprétation reste limitée aux informations présentes dans les rapports indexés.",
    "Les valeurs affichées sont issues des données extraites et sourcées.",
}


def _safe_str(value: Any, default: str = "") -> str:
    text = str(value or "").strip()
    return text if text else default


def _strip_html(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"(?is)<[^>]+>", " ", str(text))
    return re.sub(r"\s+", " ", cleaned).strip()


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
    key = norm_text(alias).replace(" ", "_")
    return analyte_display_name(alias.replace("_", " "), key)


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
        return "Aucun résultat exploitable n’a été retrouvé."
    if count == 1:
        return "Une valeur exploitable a été retrouvée."
    return f"{count} valeurs exploitables ont été retrouvées."


def _has_numeric_criterion(query_understanding: QueryUnderstanding, evidence_pack: dict[str, Any]) -> bool:
    value = _safe_str(getattr(query_understanding, "requested_value", "")) or _safe_str(
        (evidence_pack.get("constraints") or {}).get("requested_value") if isinstance(evidence_pack.get("constraints"), dict) else ""
    )
    op = _safe_str(getattr(query_understanding, "comparison_operator", "")) or _safe_str(
        (evidence_pack.get("constraints") or {}).get("comparison_operator") if isinstance(evidence_pack.get("constraints"), dict) else ""
    )
    return bool(value and op in {">", ">=", "<", "<=", "="})


def _should_show_count_line(intent: str, presentation: str, evidences: list[dict[str, Any]]) -> bool:
    if not evidences:
        return False
    if presentation in {"json", "yes_no", "paragraph", "chart"}:
        return False
    return intent in {"cohort_search", "global_patient_lookup", "doc_scoped_summary", "immunoanalysis_summary", "toxicology_summary"}


def select_intro_template(intent: str, query_understanding: QueryUnderstanding, evidence_pack: dict[str, Any]) -> str:
    doc_ids = list(query_understanding.requested_doc_ids or [])
    doc_scope = ", ".join(doc_ids) if doc_ids else "les rapports indexés"
    analyte_text = humanize_analyte_list(query_understanding.requested_analytes, evidence_pack)
    condition = humanize_condition(query_understanding, evidence_pack)
    condition_phrase = f" {condition}" if condition else ""
    seed = f"{intent}|{query_understanding.output_format}|{query_understanding.answer_style}|{doc_scope}|{analyte_text}|{condition}"

    if intent in {"cohort_search", "global_patient_lookup"}:
        has_numeric_criterion = _has_numeric_criterion(query_understanding, evidence_pack)
        if condition and query_understanding.requested_value:
            precise = f"J’ai recherché les patients ayant une {analyte_text} {condition}."
        elif condition:
            precise = f"J’ai recherché les patients ayant {analyte_text} {condition}."
        else:
            precise = f"J’ai recherché les patients ayant {analyte_text}."
        if has_numeric_criterion and condition:
            precise = f"J’ai recherché les patients dont {analyte_text} respecte le critère numérique : {condition}."
        opts = [
            precise,
            (
                f"La recherche a été effectuée sur l’ensemble des rapports indexés pour {analyte_text}{condition_phrase}."
                if not has_numeric_criterion
                else f"La recherche a été effectuée sur l’ensemble des rapports indexés pour {analyte_text} répondant au critère numérique {condition}."
            ),
            (
                f"J’ai filtré les rapports indexés pour identifier les patients avec {analyte_text}{condition_phrase}."
                if not has_numeric_criterion
                else f"J’ai filtré les rapports indexés pour identifier les patients dont {analyte_text} satisfait le critère numérique {condition}."
            ),
            (
                f"La base a été interrogée pour retrouver les patients répondant au critère : {analyte_text}{condition_phrase}."
                if not has_numeric_criterion
                else f"La base a été interrogée pour retrouver les patients répondant au critère numérique : {analyte_text} {condition}."
            ),
        ]
        return _pick_variant(seed, [o for o in opts if "None" not in o])

    if intent in {"doc_scoped_results", "previous_result_comparison"}:
        return _pick_variant(
            seed,
            [
                f"Dans {doc_scope}, les valeurs demandées ont été extraites.",
                f"Voici les mesures demandées dans {doc_scope}.",
                f"Les données ci-dessous proviennent de {doc_scope}.",
            ],
        )

    if intent in {"multi_doc_comparison", "doc_pair_comparison"}:
        return _pick_variant(
            seed,
            [
                "Comparaison des valeurs demandées entre les deux rapports.",
                "Voici la comparaison chiffrée entre les deux documents demandés.",
                "Les écarts ci-dessous sont calculés à partir des valeurs extraites des deux rapports.",
            ],
        )

    if intent == "multi_doc_presence_diff":
        return _pick_variant(
            seed,
            [
                "Voici les éléments présents dans un rapport et absents dans l’autre.",
                "La comparaison ci-dessous se concentre sur la présence/absence entre les deux documents.",
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
            "Voici les éléments techniques disponibles pour votre demande.",
            "Les informations ci-dessous proviennent des données extraites et sourcées.",
        ],
    )


def build_professional_intro(query_understanding: QueryUnderstanding, evidence_pack: dict[str, Any]) -> str:
    intent = _safe_str(query_understanding.intent, "unstructured")
    output_format = _safe_str(query_understanding.output_format, "list").lower()
    answer_style = _safe_str(query_understanding.answer_style, "standard").lower()
    presentation = getattr(query_understanding, "presentation_intent", None)

    if output_format == "json" or answer_style == "yes_no" or output_format == "yes_no":
        return ""
    if bool(getattr(presentation, "unsupported_format", False)):
        raw_phrase = _safe_str(getattr(presentation, "raw_format_phrase", "")) or "format demandé"
        recommended = (
            _safe_str(getattr(presentation, "recommended_output", ""))
            or _safe_str(getattr(presentation, "recommended_alternative_format", ""))
            or "tableau structuré"
        )
        return (
            f"Vous avez demandé un rendu {raw_phrase}. "
            "Ce format n’est pas supporté directement par le système ; "
            f"j’affiche ci-dessous le format alternatif le plus fiable ({recommended})."
        )
    if output_format == "chart":
        raw_phrase = humanize_requested_output(query_understanding)
        viz_facts = dict(evidence_pack.get("visualization_facts") or {})
        from_previous = str(intent) == "response_transform"
        doc_scope = ", ".join(query_understanding.requested_doc_ids or [])
        context_phrase = (
            (f" à partir des résultats de {doc_scope}" if doc_scope else " à partir des résultats précédents")
            if from_previous
            else ""
        )
        metric_label = _safe_str(viz_facts.get("metric_label"), "écart normalisé à la référence")
        metric_reason = _safe_str(viz_facts.get("metric_reason"), "")
        rendered_label = _safe_str(viz_facts.get("rendered_label"), "")
        if bool(viz_facts.get("fallback_used")) and rendered_label:
            return (
                f"Vous avez demandé un {raw_phrase}{context_phrase}. "
                f"Le rendu affiché utilise une alternative en {rendered_label}."
            )
        if metric_reason:
            return (
                f"Voici le {raw_phrase}{context_phrase}. "
                f"L’axe vertical représente l’{metric_label.lower()} car {metric_reason.lower()}."
            )
        return (
            f"Voici le {raw_phrase}{context_phrase}."
        )

    evidences = list(evidence_pack.get("evidences") or evidence_pack.get("results") or [])
    if not evidences and intent in {"unstructured", "response_transform"}:
        return ""

    if intent == "absence_or_missing_data":
        doc_ids = ", ".join(query_understanding.requested_doc_ids or ["le document demandé"])
        return f"Aucune valeur correspondant à cette demande n’a été retrouvée dans {doc_ids}."

    return select_intro_template(intent, query_understanding, evidence_pack)


def choose_presentation_format(query_understanding: QueryUnderstanding, evidence_pack: dict[str, Any]) -> str:
    output_format = _safe_str(query_understanding.output_format, "auto").lower()
    answer_style = _safe_str(query_understanding.answer_style, "standard").lower()
    requested_cols = list(query_understanding.requested_table_columns or [])
    evidences = list(evidence_pack.get("evidences") or evidence_pack.get("results") or [])

    if output_format == "json":
        return "json"
    if answer_style == "yes_no" or output_format == "yes_no":
        return "yes_no"
    if output_format == "unknown":
        return "table"
    if output_format == "chart":
        return "chart"
    if query_understanding.intent in {"multi_doc_comparison", "doc_pair_comparison"}:
        return "table"
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


def humanize_requested_output(query_understanding: QueryUnderstanding) -> str:
    presentation = getattr(query_understanding, "presentation_intent", None)
    requested_output = _safe_str(getattr(presentation, "requested_output", query_understanding.output_format), "auto").lower()
    chart_type = _safe_str(getattr(presentation, "chart_type", ""), "").lower()
    raw_phrase = _safe_str(getattr(presentation, "raw_format_phrase", ""), "")
    raw_norm = norm_text(raw_phrase)
    if raw_phrase and any(k in raw_norm for k in ["bio clinical", "matrix", "comparative", "arithmetic"]):
        return raw_phrase
    if requested_output == "chart":
        if chart_type == "bar":
            return "graphique en barres"
        if chart_type == "line":
            return "courbe"
        if chart_type == "radar":
            return "graphique radar"
        if chart_type == "scatter":
            return "nuage de points"
        if raw_phrase and raw_phrase.lower() != "chart":
            return raw_phrase
        return "graphique"
    if raw_phrase and raw_phrase.lower() != "chart":
        return raw_phrase
    return requested_output or "format demandé"


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
    base = re.sub(r"\bligne\s*(\d+)\s*ligne\s*\1\b", r"ligne \1", base, flags=re.IGNORECASE)
    base = re.sub(r"(,\s*ligne\s*\d+)\s*ligne\s*\d+\b", r"\1", base, flags=re.IGNORECASE)
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
    grouped: dict[tuple[str, int | None], dict[str, Any]] = {}
    for src in sources or []:
        doc_id = _safe_str(src.get("doc_id")).lower()
        if not doc_id:
            continue
        page = int(src.get("page")) if isinstance(src.get("page"), int) else None
        key = (doc_id, page)
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


def _build_sources_block(
    source_citations: list[dict[str, Any]],
    *,
    clickable_requested: bool = False,
) -> str:
    lines = _source_lines(source_citations)
    if not lines:
        return ""
    block = "Sources :\n" + "\n".join(lines)
    has_clickable = any(("](" in ln and "[" in ln) for ln in lines)
    if clickable_requested and not has_clickable:
        block += "\n\nSource disponible uniquement en texte ; aucun lien cliquable n’est disponible."
    return block


def build_short_conclusion(intent: str, evidence_pack: dict[str, Any], safety_intent: str | None) -> str | None:
    # Only force the diagnostic-refusal wording when the intent is explicitly
    # a diagnostic safety question. A generic safety_intent like
    # "no_diagnosis_constraint" (user asked "sans diagnostic") should not
    # trigger the stronger refusal phrasing — prefer the neutral summary.
    if intent == "diagnostic_safety_question" or safety_intent == "diagnostic_safety_question":
        return "Conclusion technique : aucune conclusion diagnostique ne peut être tirée uniquement de ces résultats."

    evidences = list(evidence_pack.get("evidences") or evidence_pack.get("results") or [])
    if not evidences:
        return None

    first = evidences[0]
    status = _safe_str(first.get("technical_status_code")).lower()
    analyte = _display_analyte(first) or "l’analyte"
    constraints = evidence_pack.get("constraints") if isinstance(evidence_pack, dict) else {}
    req_value = _safe_str((constraints or {}).get("requested_value") if isinstance(constraints, dict) else "")
    req_op = _safe_str((constraints or {}).get("comparison_operator") if isinstance(constraints, dict) else "")
    req_unit = _safe_str((constraints or {}).get("requested_unit") if isinstance(constraints, dict) else "")
    if not req_unit:
        req_unit = _safe_str(first.get("unit"))
    unit_suffix = f" {req_unit}" if req_unit else ""
    has_numeric_criterion = intent in {"cohort_search", "global_patient_lookup"} and bool(
        req_value and req_op in {">", ">=", "<", "<=", "="}
    )
    if len(evidences) == 1:
        if intent in {"multi_doc_comparison", "doc_pair_comparison"}:
            cmp_status = _safe_str(first.get("comparison_status")).lower()
            analyte_cmp = (_display_analyte(first) or "analyte").lower()
            if cmp_status == "identical":
                return f"Conclusion technique : aucun écart numérique n’est observé pour le {analyte_cmp}."
            if cmp_status == "increased":
                return f"Conclusion technique : le {analyte_cmp} est plus élevé dans le second rapport."
            if cmp_status == "decreased":
                return f"Conclusion technique : le {analyte_cmp} est plus bas dans le second rapport."
            if cmp_status in {"missing_in_a", "missing_in_b"}:
                return "Conclusion technique : la comparaison est partielle car une valeur manque dans l’un des deux rapports."
            return "Conclusion technique : la comparaison reste non exploitable numériquement."
        if has_numeric_criterion:
            op_label = {
                ">": "strictement supérieur à",
                ">=": "supérieur ou égal à",
                "<": "strictement inférieur à",
                "<=": "inférieur ou égal à",
                "=": "égal à",
            }.get(req_op, "conforme au critère")
            return (
                f"Conclusion technique : {analyte} satisfait le critère numérique demandé "
                f"({op_label} {req_value}{unit_suffix})."
            )
        if status == "above_reference":
            return f"Conclusion technique : {analyte} est au-dessus de l’intervalle de référence indiqué."
        if status == "below_reference":
            return f"Conclusion technique : {analyte} est en dessous de l’intervalle de référence indiqué."
        if status == "within_reference":
            return "Conclusion technique : la valeur est dans l’intervalle de référence indiqué."
        return None

    options_by_intent: dict[str, list[str]] = {
        "cohort_search": [
            "Conclusion technique : la réponse reste limitée aux résultats retrouvés dans les rapports indexés.",
            "Conclusion technique : ces résultats sont basés uniquement sur les données extraites et les sources citées.",
        ],
        "multi_doc_comparison": [
            "Conclusion technique : aucun écart numérique n’est observé lorsque les deux valeurs sont identiques.",
            "Conclusion technique : la comparaison met en évidence un écart chiffré entre les deux rapports.",
        ],
        "doc_pair_comparison": [
            "Conclusion technique : aucun écart numérique n’est observé lorsque les deux valeurs sont identiques.",
            "Conclusion technique : la comparaison met en évidence un écart chiffré entre les deux rapports.",
        ],
        "multi_doc_presence_diff": [
            "Conclusion technique : la présence/absence reflète les données extraites de chaque rapport.",
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


def _display_analyte(ev: dict[str, Any]) -> str:
    raw = _safe_str(ev.get("analyte"))
    norm = _safe_str(ev.get("analyte_norm"))
    return analyte_display_name(raw or norm, norm or None) or raw or "non précisé"


def _single_analyte_display_label(label: str) -> str:
    txt = _safe_str(label)
    if txt and " " in txt and txt.upper() == txt:
        lowered = txt.lower()
        return lowered[:1].upper() + lowered[1:]
    return txt


def _human_report_label(doc_id: str) -> str:
    raw = _safe_str(doc_id)
    if not raw:
        return "document demandé"
    m = re.match(r"(?i)^\s*report[\s_\-\(]*0*([0-9]+)\)?\s*$", raw)
    if m:
        try:
            return f"report {int(m.group(1))}"
        except Exception:
            return f"report {m.group(1)}"
    return raw.replace("_", " ")


def _compact_reference_text(reference: str) -> str:
    ref = _safe_str(reference)
    ref = re.sub(r"\s+", " ", ref).strip(" ;")
    return ref or "non disponible"


def _extract_sex_reference_segment(reference: str, sex_token: str, other_token: str) -> str:
    pattern = re.compile(
        rf"\b{re.escape(sex_token)}\b\s*:?\s*(.+?)(?=(?:\b{re.escape(other_token)}\b\s*:?)|$)",
        re.IGNORECASE,
    )
    m = pattern.search(reference or "")
    if not m:
        return ""
    seg = re.sub(r"\s+", " ", str(m.group(1) or "")).strip(" ;,.-")
    return seg


def _resolve_single_analyte_reference(reference: str, user_question: str) -> tuple[str, str]:
    ref = _compact_reference_text(reference)
    if ref == "non disponible":
        return "Référence disponible", ref
    qn = norm_text(user_question or "")
    wants_female = any(tok in qn for tok in ["femme", "feminin", "féminin", "female"])
    wants_male = any(tok in qn for tok in ["homme", "masculin", "male"])
    if wants_female and not wants_male:
        seg = _extract_sex_reference_segment(ref, "femme", "homme")
        if seg:
            return "Référence applicable", f"Femme {seg}"
    if wants_male and not wants_female:
        seg = _extract_sex_reference_segment(ref, "homme", "femme")
        if seg:
            return "Référence applicable", f"Homme {seg}"
    return "Référence disponible", ref


def _single_analyte_conclusion(status_code: str) -> str:
    sc = _safe_str(status_code).lower()
    if sc in {"above_reference", "below_reference"}:
        return "Conclusion technique : la valeur est hors de l’intervalle de référence indiqué, sans interprétation diagnostique."
    if sc == "within_reference":
        return "Conclusion technique : la valeur est dans l’intervalle de référence indiqué, sans interprétation diagnostique."
    return "Conclusion technique : interprétation technique limitée aux données disponibles, sans interprétation diagnostique."


class ClinicalDeterministicRenderer:
    """Deterministic Python renderer for simple clinical responses.

    Methods:
    - render_compact: compact single-line rendering (not used everywhere yet)
    - render_detailed: multi-line detailed rendering
    - render_not_found: deterministic not-found template that preserves raw doc_id tokens
    """

    def render_compact(self, analyte_label: str, doc_id: str, value: str | None = None) -> str:
        if value:
            return f"{analyte_label}: {value} — source: {doc_id}"
        return f"{analyte_label} non retrouvé dans {doc_id}."

    def render_detailed(self, analyte_label: str, doc_id: str, value_text: str | None, reference: str, status: str, source: str) -> str:
        val = value_text or "non disponible"
        lines = [f"### {analyte_label} — {doc_id}", "", f"- **Valeur** : **{val}**"]
        lines.append(f"- **Référence** : {reference or 'non disponible'}")
        lines.append(f"- **Statut technique** : {status or 'non interprétable'}")
        lines.append(f"- **Source** : {source or doc_id}")
        lines.append("")
        lines.append(_single_analyte_conclusion(status or ""))
        return "\n".join(lines)

    def render_not_found(self, analyte_label: str, raw_doc_id: str, include_explanation: bool = True, canonical_label: str | None = None) -> str:
        # Preserve the raw doc_id token (eg. report_12) so validators and downstream
        # consumers can reliably parse the referenced document.
        doc = raw_doc_id or "le document demandé"
        label = canonical_label or analyte_label
        if include_explanation:
            return (
                f"{label} : non retrouvé dans {doc} parmi les résultats disponibles.\n"
                "Aucune valeur numérique exploitable n’a été identifiée pour cet analyte.\n"
                f"Conclusion technique : aucun résultat correspondant à {label} n’a été retrouvé dans le rapport demandé."
            )
        return f"{label} non retrouvé dans {doc}."


def _writer_intent(intent: str) -> str:
    mapping = {
        "cohort_search": "cohort_search",
        "global_patient_lookup": "cohort_search",
        "doc_scoped_results": "doc_scoped_query",
        "previous_result_comparison": "comparison",
        "multi_doc_comparison": "comparison",
        "doc_pair_comparison": "comparison",
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
    if chosen in {"table", "list", "paragraph", "json", "chart"}:
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

    analyte_norm = canonicalize_analyte(_safe_str(ev.get("analyte_norm") or ev.get("analyte")))
    analyte_raw = _safe_str(ev.get("analyte"))
    analyte_human = _strip_html(get_display_analyte_label(ev)) or analyte_display_name(analyte_raw or analyte_norm, analyte_norm or None) or analyte_raw

    normalized_source = normalize_source_for_response(
        {
            "label": _strip_html(_safe_str(ev.get("source_label"))),
            "filename": ev.get("filename"),
            "doc_id": ev.get("doc_id"),
            "page": ev.get("page"),
            "row": ev.get("row"),
            "viewer_url": ev.get("viewer_url"),
            "source_url": ev.get("source_url"),
            "url": ev.get("url"),
        }
    )
    source_label = _strip_html(_safe_str(normalized_source.get("label")))
    value = _strip_html(_safe_str(ev.get("current_value") or ev.get("value_raw")))
    reference = _strip_html(_safe_str(ev.get("reference") or ev.get("reference_range")))
    return {
        "patient": _safe_str(ev.get("patient_token") or ev.get("patient")),
        "doc_id": _safe_str(ev.get("doc_id")),
        "filename": _safe_str(ev.get("filename")),
        "page": _safe_int(ev.get("page")),
        "row": _safe_int(ev.get("row")),
        "analyte": analyte_human,
        "analyte_norm": analyte_norm,
        "value": value,
        "unit": _safe_str(ev.get("unit")),
        "reference": reference,
        "status": status,
        "previous_result": ev.get("previous_result"),
        "variation": ev.get("variation"),
        "source_label": source_label,
        "source_url": _safe_str(normalized_source.get("source_url")),
        "viewer_url": _safe_str(normalized_source.get("viewer_url")),
    }


def _llm_locked_result_row(result: dict[str, Any]) -> dict[str, Any]:
    """Strict row contract sent to LLM writer: facts only, no rendering/HTML payload."""
    return {
        "analyte": _strip_html(_safe_str(result.get("analyte"), "non précisé")),
        "analyte_norm": _strip_html(_safe_str(result.get("analyte_norm"))),
        "value": _strip_html(_safe_str(result.get("value"), "non disponible")),
        "unit": _strip_html(_safe_str(result.get("unit"))),
        "reference": _strip_html(_safe_str(result.get("reference"), "non disponible")),
        "status": _strip_html(_safe_str(result.get("status"), "non interprétable")),
        "source_label": _strip_html(_safe_str(result.get("source_label"), "source non disponible")),
    }


def _canonical_requested_analytes(values: list[str] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in list(values or []):
        key = canonicalize_analyte(_safe_str(raw))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _writer_row_matches_scope(
    *,
    row: dict[str, Any],
    intent: str,
    requested_doc_ids: list[str],
    requested_analytes: list[str],
) -> bool:
    row_doc_id = _safe_str(row.get("doc_id"))
    row_analyte_norm = canonicalize_analyte(_safe_str(row.get("analyte_norm") or row.get("analyte")))
    comparison_like = intent in {"multi_doc_comparison", "doc_pair_comparison", "multi_doc_presence_diff"}
    if requested_doc_ids and not comparison_like and row_doc_id and row_doc_id not in requested_doc_ids:
        return False
    if requested_analytes and row_analyte_norm and row_analyte_norm not in requested_analytes:
        return False
    return True


def _filter_writer_results_for_contract(
    *,
    query_understanding: QueryUnderstanding,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    requested_doc_ids = [d for d in (query_understanding.requested_doc_ids or []) if _safe_str(d)]
    requested_analytes = _canonical_requested_analytes(list(query_understanding.requested_analytes or []))
    intent = _safe_str(query_understanding.intent).lower()
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if not _safe_str(row.get("analyte")):
            continue
        if not _writer_row_matches_scope(
            row=row,
            intent=intent,
            requested_doc_ids=requested_doc_ids,
            requested_analytes=requested_analytes,
        ):
            continue
        filtered.append(row)
    return filtered


def _build_writer_scope(query_understanding: QueryUnderstanding, results_locked: list[dict[str, Any]]) -> dict[str, Any]:
    requested_doc_ids = [d for d in (query_understanding.requested_doc_ids or []) if _safe_str(d)]
    canonical_requested = _canonical_requested_analytes(list(query_understanding.requested_analytes or []))
    intent = _safe_str(query_understanding.intent).lower()
    comparison_like = intent in {"multi_doc_comparison", "doc_pair_comparison", "multi_doc_presence_diff"}
    row_doc_ids = sorted({_safe_str(r.get("doc_id")) for r in results_locked if _safe_str(r.get("doc_id"))})
    row_analytes = sorted(
        {
            canonicalize_analyte(_safe_str(r.get("analyte_norm") or r.get("analyte")))
            for r in results_locked
            if canonicalize_analyte(_safe_str(r.get("analyte_norm") or r.get("analyte")))
        }
    )
    return {
        "requested_doc_ids": requested_doc_ids,
        "effective_doc_ids": row_doc_ids,
        "requested_analytes": canonical_requested,
        "effective_analytes": row_analytes,
        "technical_condition": query_understanding.technical_condition,
        "row_count": len(results_locked),
        "scope_coherent": (
            (comparison_like or not requested_doc_ids or all(doc in requested_doc_ids for doc in row_doc_ids))
            and (not canonical_requested or all(analyte in canonical_requested for analyte in row_analytes))
        ),
    }


def _validate_writer_evidence_pack_contract(
    writer_pack: dict[str, Any],
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not isinstance(writer_pack, dict):
        return False, ["writer_pack_not_dict"]

    results_locked = writer_pack.get("results_locked")
    if not isinstance(results_locked, list):
        errors.append("results_locked_missing")
        results_locked = []
    elif not results_locked:
        errors.append("results_locked_empty")

    required_row_keys = {"analyte", "analyte_norm", "value", "unit", "reference", "status", "source_label"}
    for idx, row in enumerate(results_locked):
        if not isinstance(row, dict):
            errors.append(f"results_locked_row_not_dict:{idx}")
            continue
        missing = sorted(required_row_keys - set(row.keys()))
        if missing:
            errors.append(f"results_locked_row_missing_keys:{idx}:{','.join(missing)}")
        analyte = _safe_str(row.get("analyte"))
        analyte_norm = canonicalize_analyte(_safe_str(row.get("analyte_norm")))
        source_label = _safe_str(row.get("source_label"))
        if not analyte:
            errors.append(f"results_locked_row_empty_analyte:{idx}")
        if not analyte_norm:
            errors.append(f"results_locked_row_empty_analyte_norm:{idx}")
        if "<" in source_label or "chunk_id" in source_label.lower() or "doc_id=" in source_label.lower():
            errors.append(f"results_locked_row_bad_source_label:{idx}")

    sources = writer_pack.get("sources")
    if not isinstance(sources, list):
        errors.append("sources_missing")
        sources = []
    for idx, src in enumerate(sources):
        if not isinstance(src, dict):
            errors.append(f"sources_row_not_dict:{idx}")
            continue
        if not _safe_str(src.get("label")):
            errors.append(f"sources_label_missing:{idx}")

    scope = writer_pack.get("scope")
    if not isinstance(scope, dict):
        errors.append("scope_missing")
    else:
        if not isinstance(scope.get("scope_coherent"), bool):
            errors.append("scope_coherent_missing")
        elif not bool(scope.get("scope_coherent")):
            errors.append("scope_incoherent")

    contract = writer_pack.get("evidence_contract")
    if not isinstance(contract, dict):
        errors.append("evidence_contract_missing")
    else:
        if str(contract.get("contract_version") or "") != "v1":
            errors.append("evidence_contract_version_invalid")
        if contract.get("rows_fact_locked") is not True:
            errors.append("evidence_contract_rows_fact_locked_false")
        if contract.get("sources_normalized") is not True:
            errors.append("evidence_contract_sources_not_normalized")

    constraints = writer_pack.get("constraints") if isinstance(writer_pack.get("constraints"), dict) else {}
    requested_analytes = {
        canonicalize_analyte(_safe_str(item))
        for item in list(constraints.get("requested_analytes") or [])
        if canonicalize_analyte(_safe_str(item))
    }
    effective_analytes = set()
    for row in results_locked:
        if isinstance(row, dict):
            norm_val = canonicalize_analyte(_safe_str(row.get("analyte_norm") or row.get("analyte")))
            if norm_val:
                effective_analytes.add(norm_val)
    if requested_analytes and effective_analytes and not effective_analytes.issubset(requested_analytes):
        errors.append("requested_analytes_effective_analytes_mismatch")

    return not errors, errors


def build_writer_evidence_pack(
    *,
    user_question: str,
    query_understanding: QueryUnderstanding,
    evidence_pack: dict[str, Any],
    source_citations: list[dict[str, Any]],
) -> WriterEvidencePack:
    presentation = getattr(query_understanding, "presentation_intent", None)
    raw_rows = [_normalized_writer_result(ev) for ev in (evidence_pack.get("evidences") or evidence_pack.get("results") or [])]
    results_full = _filter_writer_results_for_contract(
        query_understanding=query_understanding,
        rows=raw_rows,
    )
    results_locked = [_llm_locked_result_row(r) for r in results_full]
    recent_style_history = list(evidence_pack.get("recent_style_history") or [])
    normalized_sources = dedup_normalized_sources(list(source_citations or []))
    scope_payload = _build_writer_scope(query_understanding, results_full)
    constraints = {
        "requested_doc_ids": list(query_understanding.requested_doc_ids or []),
        "requested_analytes": _canonical_requested_analytes(list(query_understanding.requested_analytes or [])),
        "excluded_analytes": list(getattr(query_understanding, "excluded_analytes", []) or []),
        "technical_condition": query_understanding.technical_condition,
        "comparison_operator": getattr(query_understanding, "comparison_operator", None),
        "requested_value": query_understanding.requested_value,
        "requested_unit": getattr(query_understanding, "requested_unit", None),
        "requested_columns": list(query_understanding.requested_table_columns or []),
        "source_clickable_requested": bool(getattr(query_understanding, "source_clickable_requested", False)),
        "diagnostic_safety": bool(query_understanding.safety_intent),
    }
    visualization_facts = dict(evidence_pack.get("visualization_facts") or {})
    return {
        "original_user_question": _safe_str(getattr(query_understanding, "original_user_question", "") or user_question),
        "user_question": user_question,
        "intent": _writer_intent(query_understanding.intent),
        "response_strategy": _safe_str(getattr(query_understanding, "response_strategy", "render_table"), "render_table"),
        "response_strategy_reason": getattr(query_understanding, "response_strategy_reason", None),
        "output_format": _writer_output_format(query_understanding, evidence_pack),
        "answer_style": "yes_no" if query_understanding.answer_style == "yes_no" else "professional",
        "language": _safe_str(query_understanding.language, "fr"),
        "presentation_intent": {
            "requested_output": _safe_str(getattr(presentation, "requested_output", query_understanding.output_format), "auto"),
            "chart_type": getattr(presentation, "chart_type", None),
            "raw_format_phrase": getattr(presentation, "raw_format_phrase", None),
            "wants_clickable_sources": bool(getattr(presentation, "wants_clickable_sources", False)),
            "wants_intro": bool(getattr(presentation, "wants_intro", True)),
            "wants_conclusion": bool(getattr(presentation, "wants_conclusion", True)),
            "strict_columns": list(getattr(presentation, "strict_columns", []) or []),
            "unsupported_format": bool(getattr(presentation, "unsupported_format", False)),
            "user_requested_visualization": bool(getattr(presentation, "user_requested_visualization", False)),
            "presentation_confidence": float(getattr(presentation, "presentation_confidence", 0.5)),
            "unsupported_reason": getattr(presentation, "unsupported_reason", None),
            "recommended_output": getattr(presentation, "recommended_output", None),
            "unsupported_presentation_reason": getattr(presentation, "unsupported_presentation_reason", None),
            "recommended_alternative_format": getattr(presentation, "recommended_alternative_format", None),
            "unhandled_instructions": list(getattr(presentation, "unhandled_instructions", []) or []),
        },
        "visualization_facts": {
            "requested_type": visualization_facts.get("requested_type"),
            "requested_label": visualization_facts.get("requested_label"),
            "rendered_type": visualization_facts.get("rendered_type"),
            "rendered_label": visualization_facts.get("rendered_label"),
            "supported": visualization_facts.get("supported"),
            "suitable": visualization_facts.get("suitable"),
            "fallback_used": visualization_facts.get("fallback_used"),
            "fallback_reason": visualization_facts.get("fallback_reason"),
            "recommendation_reason": visualization_facts.get("recommendation_reason"),
            "metric_label": visualization_facts.get("metric_label"),
            "metric_reason": visualization_facts.get("metric_reason"),
            "result_count": visualization_facts.get("result_count"),
            "raw_format_phrase": visualization_facts.get("raw_format_phrase"),
        },
        "constraints": constraints,
        "results": results_locked,
        "results_locked": results_locked,
        "missing_items": list(evidence_pack.get("missing_items") or []),
        "sources": normalized_sources,
        "scope": scope_payload,
        "evidence_contract": {
            "contract_version": "v1",
            "rows_filtered": True,
            "rows_fact_locked": True,
            "scope_coherent": scope_payload.get("scope_coherent"),
            "sources_normalized": True,
            "sources_deduplicated": True,
            "noise_reduction": [
                "results_locked_only",
                "sources_normalized",
                "sources_deduplicated",
                "style_history_capped_20",
            ],
            "canonical_requested_analytes": _canonical_requested_analytes(list(query_understanding.requested_analytes or [])),
            "source_labels_normalized": all(_safe_str(src.get("label")) for src in normalized_sources),
        },
        "response_brief": {
            "task_goal": "Répondre à la question utilisateur de manière claire et sourcée.",
            "audience": "Utilisateur non technique consultant des résultats biologiques.",
            "tone": "professionnel, humain, clair, prudent",
            "verbosity": "concise",
            "format": _writer_output_format(query_understanding, evidence_pack),
            "must_include": ["critère utilisateur réel", "résultats extraits", "sources lisibles"],
            "must_not_include": ["chunk_id", "chemin local", "logs techniques", "diagnostic non autorisé", "aliases internes"],
            "grounding_policy": "Toutes les valeurs doivent venir de l’evidence_pack.",
            "immutable_visualization_facts": [
                "requested_type",
                "rendered_type",
                "fallback_reason",
                "recommendation_reason",
                "metric_reason",
            ],
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
            "priorite": "Priorité",
            "priority_level": "Priorité",
            "priority_score": "Score priorité",
            "priority_reason": "Raison technique",
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
                        "Priorité": _safe_str(ev.get("priority_level"), "unknown"),
                        "Score priorité": _safe_str(ev.get("priority_score"), "0"),
                        "Raison technique": _safe_str(ev.get("priority_reason"), "non disponible"),
                        "Analyte": _display_analyte(ev),
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

    if intent in {"multi_doc_comparison", "doc_pair_comparison"}:
        def _render_comparison_value(raw_value: str, unit: str) -> str:
            value_norm = _safe_str(raw_value).strip().lower()
            if value_norm in {"non présent", "non present", "non disponible", "non retrouvé", "non retrouve"}:
                return "non présent" if "présent" in value_norm or "present" in value_norm else "non disponible"
            return (raw_value + (f" {unit}" if unit else "")).strip()

        rows = []
        for ev in evidences:
            analyte = _display_analyte(ev)
            doc_a = _safe_str(ev.get("doc_a"), "report A")
            doc_b = _safe_str(ev.get("doc_b"), "report B")
            value_a = _safe_str(ev.get("value_a_raw") or ev.get("value_a"), "non disponible")
            value_b = _safe_str(ev.get("value_b_raw") or ev.get("value_b"), "non disponible")
            unit_a = _safe_str(ev.get("unit_a") or ev.get("unit"))
            unit_b = _safe_str(ev.get("unit_b") or ev.get("unit"))
            status = _safe_str(ev.get("comparison_status")).lower()
            delta_abs = ev.get("delta_abs")
            delta_unit = _safe_str(ev.get("delta_unit") or ev.get("unit"))
            if status == "identical":
                delta_label = f"0{(' ' + delta_unit) if delta_unit else ''}"
                conclusion = "Valeurs identiques"
            elif status == "increased":
                if isinstance(delta_abs, (int, float)):
                    delta_label = f"+{delta_abs:g}{(' ' + delta_unit) if delta_unit else ''}"
                else:
                    delta_label = "augmentation"
                conclusion = "Augmentation"
            elif status == "decreased":
                if isinstance(delta_abs, (int, float)):
                    delta_label = f"{delta_abs:g}{(' ' + delta_unit) if delta_unit else ''}"
                else:
                    delta_label = "diminution"
                conclusion = "Diminution"
            elif status == "missing_in_a":
                delta_label = "non calculable"
                conclusion = f"Absent dans {doc_a}"
            elif status == "missing_in_b":
                delta_label = "non calculable"
                conclusion = f"Absent dans {doc_b}"
            else:
                delta_label = "non comparable"
                conclusion = "Non comparable"
            rows.append(
                {
                    "Analyte": analyte,
                    doc_a: _render_comparison_value(value_a, unit_a),
                    doc_b: _render_comparison_value(value_b, unit_b),
                    "Écart": delta_label,
                    "Référence": _safe_str(ev.get("reference_summary") or ev.get("reference") or "non disponible"),
                    "Conclusion": conclusion,
                }
            )
        if rows:
            columns = list(rows[0].keys())
            return _table(columns, rows)

    if intent in {"cohort_search", "global_patient_lookup"}:
        rows = [
            {
                "Patient": _safe_str(ev.get("patient_token"), "non disponible"),
                "Report": _safe_str(ev.get("doc_id")),
                "Analyte": _display_analyte(ev),
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
                "Analyte": _display_analyte(ev),
                "Présent dans": _safe_str(ev.get("present_in")),
                "Absent dans": _safe_str(ev.get("absent_in")),
            }
            for ev in evidences
        ]
        return _table(["Analyte", "Présent dans", "Absent dans"], rows)

    rows = []
    for ev in evidences:
        row = {
            "Analyte": _display_analyte(ev),
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
                    "Analyte": _display_analyte(ev),
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
            f"- {_display_analyte(ev)}: {ev.get('current_value') or 'non disponible'}"
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
    body = (
        f"{_display_analyte(primary)} = {_safe_str(primary.get('current_value'), 'non disponible')}"
        f"{(' ' + _safe_str(primary.get('unit'))) if _safe_str(primary.get('unit')) else ''} ; "
        f"référence : {_safe_str(primary.get('reference'), 'non disponible')} ; "
        f"statut technique : {_safe_str(primary.get('technical_status'), 'non interprétable')}."
    )
    if not _explicit_yes_no_requested(query):
        return body
    yn = _yn_prefix(
        query,
        _safe_str(primary.get("technical_status_code")),
        _safe_str(primary.get("reference")),
    )
    return f"{yn} — {body}"


def _extract_doc_values_from_current_value(current_value: str) -> tuple[tuple[str, str] | None, tuple[str, str] | None]:
    txt = _safe_str(current_value)
    m = re.findall(r"([A-Za-z0-9_()\- ]+)\s*=\s*([^|]+)", txt)
    if len(m) >= 2:
        left = (m[0][0].strip(), m[0][1].strip())
        right = (m[1][0].strip(), m[1][1].strip())
        return left, right
    return None, None


def _to_float_local(v: str) -> float | None:
    try:
        return float(str(v).replace(",", ".").strip())
    except Exception:
        return None


def _build_multi_doc_comparison_narrative(evidences: list[dict[str, Any]]) -> str:
    if not evidences:
        return "Aucun résultat exploitable."
    lines: list[str] = []
    for ev in evidences:
        analyte = _display_analyte(ev).lower()
        unit = _safe_str(ev.get("unit"))
        comparison_status = _safe_str(ev.get("comparison_status")).lower()
        doc_a = _safe_str(ev.get("doc_a"))
        doc_b = _safe_str(ev.get("doc_b"))
        val_a = _safe_str(ev.get("value_a"))
        val_b = _safe_str(ev.get("value_b"))
        if not (doc_a and doc_b and val_a and val_b):
            left, right = _extract_doc_values_from_current_value(_safe_str(ev.get("current_value")))
            if left and right:
                doc_a, val_a = left
                doc_b, val_b = right
        if not (doc_a and doc_b and val_a and val_b):
            lines.append(
                f"Pour {analyte}, les données ne permettent pas une comparaison numérique fiable."
            )
            continue
        f_a = _to_float_local(val_a)
        f_b = _to_float_local(val_b)
        unit_suffix = f" {unit}" if unit else ""
        if comparison_status == "missing_in_a":
            lines.append(f"Pour {analyte}, la valeur est absente dans {doc_a}.")
            continue
        if comparison_status == "missing_in_b":
            lines.append(f"Pour {analyte}, la valeur est absente dans {doc_b}.")
            continue

        if f_a is not None and f_b is not None:
            if comparison_status == "identical" or abs(f_a - f_b) <= 1e-12:
                lines.append(
                    f"Le {analyte} présente la même valeur dans les deux rapports : "
                    f"{val_a}{unit_suffix} dans {doc_a} et {val_b}{unit_suffix} dans {doc_b}. "
                    "Aucun écart numérique n’est observé."
                )
            else:
                delta = f_b - f_a
                if comparison_status == "increased":
                    trend = "plus élevé"
                elif comparison_status == "decreased":
                    trend = "plus faible"
                else:
                    trend = "plus élevé" if delta > 0 else "plus faible"
                signed_delta = f"{delta:g}"
                lines.append(
                    f"Le {analyte} est {trend} dans {doc_b} que dans {doc_a} : "
                    f"{val_b}{unit_suffix} contre {val_a}{unit_suffix}, soit un écart de {signed_delta}{unit_suffix}."
                )
        else:
            lines.append(
                f"Pour {analyte}, comparaison non numérique : {doc_a}={val_a}{unit_suffix} ; {doc_b}={val_b}{unit_suffix}."
            )
    return "\n".join(lines).strip()


def _extract_locked_facts(evidences: list[dict[str, Any]]) -> dict[str, set[str]]:
    analytes: set[str] = set()
    values: set[str] = set()
    refs: set[str] = set()
    sources: set[str] = set()
    for ev in evidences:
        analyte = _safe_str(_display_analyte(ev))
        if analyte:
            analytes.add(norm_text(analyte))
        current_value = _safe_str(ev.get("current_value") or ev.get("value_raw"))
        if current_value:
            values.add(current_value.replace(",", "."))
        reference = _safe_str(ev.get("reference") or ev.get("reference_range"))
        if reference:
            refs.add(reference.replace(",", "."))
        src = _safe_str(ev.get("source_label"))
        if not src:
            src = format_source_label(
                {
                    "filename": ev.get("filename"),
                    "doc_id": ev.get("doc_id"),
                    "page": ev.get("page"),
                    "row": ev.get("row"),
                }
            )
        if src:
            sources.add(norm_text(src))
    return {"analytes": analytes, "values": values, "references": refs, "sources": sources}


def _llm_writer_preserves_facts(
    *,
    llm_answer: str,
    evidences: list[dict[str, Any]],
    source_citations: list[dict[str, Any]],
) -> tuple[bool, str | None]:
    if not llm_answer.strip():
        return False, "empty_llm_answer"
    locked = _extract_locked_facts(evidences)
    llm_norm = norm_text(llm_answer)
    llm_numeric = llm_answer.replace(",", ".")
    for analyte in sorted(locked["analytes"]):
        if analyte and analyte not in llm_norm:
            return False, "llm_missing_analyte"
    for val in sorted(locked["values"]):
        if val and val not in llm_numeric:
            return False, "llm_modified_values"
    # Source labels must be preserved only when sources are displayed.
    if source_citations:
        for src in deduplicate_sources(source_citations):
            label = norm_text(_safe_str(src.get("label")))
            if label and label not in llm_norm:
                return False, "llm_missing_source_label"
    return True, None


def _build_chart_explanation(
    query_understanding: QueryUnderstanding,
    evidences: list[dict[str, Any]],
    evidence_pack: dict[str, Any] | None = None,
) -> str:
    presentation = getattr(query_understanding, "presentation_intent", None)
    raw_phrase = humanize_requested_output(query_understanding)
    viz_facts = dict((evidence_pack or {}).get("visualization_facts") or {})
    from_previous = str(getattr(query_understanding, "intent", "")).strip().lower() == "response_transform"
    doc_scope = ", ".join(query_understanding.requested_doc_ids or [])
    context_phrase = (f" à partir des résultats de {doc_scope}" if doc_scope else " à partir des résultats précédents") if from_previous else ""
    units = sorted({_safe_str(ev.get("unit")).lower() for ev in evidences if _safe_str(ev.get("unit"))})
    mixed_units = len(set(units)) > 1
    metric_label = _safe_str(viz_facts.get("metric_label"), "écart normalisé à la référence")
    metric_reason = _safe_str(viz_facts.get("metric_reason"), "les analytes utilisent des unités différentes")
    requested_type = _safe_str(getattr(presentation, "chart_type", "unknown"), "unknown").lower()
    if requested_type in {"radar", "scatter", "heatmap", "unknown"}:
        return (
            f"Vous avez demandé un {raw_phrase}{context_phrase}. "
            "Ce type de visualisation n’est pas disponible tel quel dans l’interface ; "
            f"j’affiche donc une alternative en graphique en barres basée sur l’{metric_label.lower()}."
        )
    if requested_type == "line" and mixed_units:
        return (
            f"Vous avez demandé une {raw_phrase}{context_phrase}. "
            "Cette courbe n’est pas affichée telle quelle, car les unités biologiques sont différentes ; "
            f"j’affiche une alternative en graphique en barres avec l’{metric_label.lower()}."
        )
    return (
        f"Voici le {raw_phrase} généré à partir des résultats retrouvés{context_phrase}. "
        f"L’axe vertical représente l’{metric_label.lower()} car {metric_reason.lower()}."
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


def _explicit_yes_no_requested(query: str) -> bool:
    qn = norm_text(query)
    markers = [
        "yes/no",
        "yes or no",
        "yes no",
        "answer only yes",
        "respond only yes",
        "yes ou no",
        "oui/non",
        "oui non",
        "oui ou non",
        "reponds uniquement oui",
        "réponds uniquement oui",
        "est ce que",
        "est-ce que",
    ]
    return any(m in qn for m in markers)


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
    requested_docs = [d for d in (query_understanding.requested_doc_ids or []) if str(d).strip()]
    requested_analytes = [a for a in (query_understanding.requested_analytes or []) if str(a).strip()]

    # UX-focused deterministic template for single analyte in a single report.
    if (
        intent in {"doc_scoped_results", "single_analyte_lookup"}
        and len(evidences) == 1
        and len(requested_docs) == 1
        and len(requested_analytes) == 1
        and not missing_items
        and presentation in {"list", "paragraph"}
    ):
        ev = dict(evidences[0] or {})
        analyte = _single_analyte_display_label(_display_analyte(ev))
        report_label = _human_report_label(_safe_str(ev.get("doc_id")) or requested_docs[0])
        value = _safe_str(ev.get("current_value"), _safe_str(ev.get("value_raw"), "non disponible"))
        unit = _safe_str(ev.get("unit"))
        value_text = value if not unit else f"{value} {unit}"
        reference = _safe_str(ev.get("reference"), _safe_str(ev.get("reference_range"), "non disponible"))
        ref_label, ref_value = _resolve_single_analyte_reference(reference, user_question)
        status = _safe_str(ev.get("technical_status"), "non interprétable")
        source = _safe_str(ev.get("source_label"))
        if not source:
            source = format_source_label(
                {
                    "filename": ev.get("filename"),
                    "doc_id": ev.get("doc_id"),
                    "page": ev.get("page"),
                    "row": ev.get("row"),
                }
            )
        conclusion = _single_analyte_conclusion(_safe_str(ev.get("technical_status_code"), _safe_str(ev.get("interpretation_status"))))
        answer = (
            f"### {analyte} — {report_label}\n\n"
            f"- **Valeur** : **{value_text}**\n"
            f"- **{ref_label}** : {ref_value}\n"
            f"- **Statut technique** : {status}\n"
            f"- **Source** : {source}\n\n"
            f"{conclusion}"
        )
        return {
            "intro": "",
            "content_type": "list",
            "content": answer.strip(),
            "conclusion": conclusion,
            "sources": sources,
            "rendering_hints": {"preferred_format": "list", "show_sources": True, "strict_json": False},
            "answer": answer.strip(),
            "mode": "deterministic_professional_fallback",
            "llm_error": None,
        }

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
        explicit_yn = _explicit_yes_no_requested(user_question)
        if not evidences:
            doc_scope = ", ".join(query_understanding.requested_doc_ids or ["le document demandé"])
            analyte_text = humanize_analyte_list(query_understanding.requested_analytes, evidence_pack)
            if explicit_yn:
                answer = f"Non — {analyte_text} non retrouvé dans {doc_scope}."
            else:
                answer = f"{analyte_text} non retrouvé dans {doc_scope}."
        else:
            first = dict(evidences[0] or {})
            analyte = _display_analyte(first)
            value = _safe_str(first.get("current_value"), _safe_str(first.get("value_raw"), "non disponible"))
            unit = _safe_str(first.get("unit"), "")
            reference = _safe_str(first.get("reference"), _safe_str(first.get("reference_range"), "non disponible"))
            status_code = _safe_str(first.get("technical_status_code"), "")
            value_text = value if not unit else f"{value} {unit}"
            if explicit_yn:
                prefix = _yn_prefix(user_question, status_code, reference)
                answer = f"{prefix} — {analyte}: {value_text} (référence: {reference})."
            else:
                answer = f"{analyte}: {value_text} (référence: {reference})."
        source_block = _build_sources_block(
            source_citations or [],
            clickable_requested=bool(getattr(query_understanding, "source_clickable_requested", False)),
        )
        if source_block:
            answer = answer.rstrip() + "\n\n" + source_block
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

    if presentation == "chart":
        chart_intro = _build_chart_explanation(query_understanding, evidences, evidence_pack)
        content = (
            _build_content_table(
                intent,
                evidences,
                include_previous=bool(query_understanding.requires_previous_results),
                requested_columns=query_understanding.requested_table_columns,
                source_clickable_requested=bool(getattr(query_understanding, "source_clickable_requested", False)),
            )
            if evidences
            else "Aucune donnée structurée exploitable pour visualisation."
        )
        answer_parts = [chart_intro]
        if evidences:
            answer_parts.append("Données utilisées")
        answer_parts.append(content)
        answer = "\n\n".join([p for p in answer_parts if p.strip()])
        source_block = _build_sources_block(
            source_citations or [],
            clickable_requested=bool(getattr(query_understanding, "source_clickable_requested", False)),
        )
        if source_block:
            answer = answer.rstrip() + "\n\n" + source_block
        return {
            "intro": chart_intro,
            "content_type": "chart",
            "content": content.strip(),
            "conclusion": "",
            "sources": sources,
            "rendering_hints": {"preferred_format": "chart", "show_sources": True, "strict_json": False},
            "answer": answer.strip(),
            "mode": "deterministic_professional_fallback",
            "llm_error": None,
        }

    intro = build_professional_intro(query_understanding, evidence_pack)
    count_line = format_result_count(len(evidences)) if _should_show_count_line(intent, presentation, evidences) else ""

    include_previous = bool(query_understanding.requires_previous_results)
    if intent == "multi_doc_presence_diff" and not evidences:
        doc_a = requested_docs[0] if len(requested_docs) >= 1 else "report_A"
        doc_b = requested_docs[1] if len(requested_docs) >= 2 else "report_B"
        content = (
            f"Présents uniquement dans {doc_a} :\n"
            "- Aucun analyte distinct retrouvé.\n\n"
            f"Présents uniquement dans {doc_b} :\n"
            "- Aucun analyte distinct retrouvé."
        )
    elif intent in {"multi_doc_comparison", "doc_pair_comparison"}:
        content = (
            _build_content_table(
                intent,
                evidences,
                include_previous=False,
                requested_columns=query_understanding.requested_table_columns,
                source_clickable_requested=bool(getattr(query_understanding, "source_clickable_requested", False)),
            )
            if evidences
            else "Aucun résultat exploitable."
        )
    elif presentation == "table":
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

    if missing_items and presentation != "yes_no" and intent not in {"multi_doc_comparison", "doc_pair_comparison"}:
        doc_scope = ", ".join(query_understanding.requested_doc_ids or ["le document demandé"])
        miss = "\n".join(f"- {_canonical_analyte_display(str(m))}: non retrouvé dans {doc_scope}." for m in missing_items)
        content = content.rstrip() + "\n\nÉléments non retrouvés :\n" + miss

    conclusion = build_short_conclusion(intent, evidence_pack, query_understanding.safety_intent)
    if intent == "multi_doc_presence_diff" and not (conclusion or "").strip():
        conclusion = "Conclusion technique : aucun analyte distinct n’a été retrouvé entre les deux documents demandés."
    parts = [p for p in [intro.strip(), count_line.strip(), content.strip(), (conclusion or "").strip()] if p]
    answer = "\n\n".join(parts)

    source_block = _build_sources_block(
        source_citations or [],
        clickable_requested=bool(getattr(query_understanding, "source_clickable_requested", False)),
    )
    has_source_column_in_table = presentation == "table" and bool(re.search(r"(?im)^\|\s*.*\bsource\b.*\|$", content or ""))
    if source_block and not has_source_column_in_table:
        answer = answer.rstrip() + "\n\n" + source_block

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


def compose_visualization_answer(
    user_question: str,
    query_understanding: QueryUnderstanding,
    evidence_pack: dict[str, Any],
    llm_client: LLMClient | None = None,
    provider: str = DEFAULT_LLM_PROVIDER,
    model: str = DEFAULT_LLM_MODEL,
) -> dict[str, Any]:
    """ Specialized composer for visualization requests to ensure clean separation. """
    viz_facts = dict(evidence_pack.get("visualization_facts") or {})
    evidences = list(evidence_pack.get("evidences") or evidence_pack.get("results") or [])
    
    # Build a prompt to get ONLY the intro text
    compact_pack = build_writer_evidence_pack(
        user_question=user_question,
        query_understanding=query_understanding,
        evidence_pack=evidence_pack,
        source_citations=[],
    )
    
    prompt = (
        f"{PROFESSIONAL_WRITER_SYSTEM_PROMPT}\n\n{PROFESSIONAL_WRITER_VISUALIZATION_RULES}\n\n"
        "TASK: Rédige une introduction factuelle, naturelle et concise pour présenter les données et la visualisation ci-dessous.\n"
        "RECO: Si c'est un fallback, explique-le avec fluidité (ex: 'Compte tenu de la disparité des unités...').\n"
        "IMPORTANT: Ta réponse doit être UNIQUEMENT le texte de l'introduction. Ne génère JAMAIS de tableau, de liste de sources ou de labels techniques Recharts concaténés ici.\n"
        "/no_think\n\n"
        "evidence_pack JSON:\n"
        f"{json.dumps(compact_pack, ensure_ascii=False)}\n"
    )
    
    client = llm_client or LLMClient(provider=provider)
    intro = client.generate(prompt=prompt, model=model).strip()
    
    # Post-generation sanitizer for Recharts leakage
    # We remove patterns of uppercase joined words and percentage labels that often leak from Recharts DOM
    intro = re.sub(r'[A-ZÀ-ÿ]{4,}(?=[A-ZÀ-ÿ][a-z])', '', intro) # Join between words
    intro = re.sub(r'\b\d+%\b', '', intro) # 600% etc
    intro = re.sub(r'Écart normalisé.*', '', intro) # Common leaked footer
    intro = intro.strip()
    
    # Build the data table for display alongside the chart
    data_table = []
    for ev in evidences:
        data_table.append({
            "analyte": ev.get("analyte_display") or ev.get("analyte"),
            "value": ev.get("value_raw"),
            "unit": ev.get("unit"),
            "reference": ev.get("reference_range"),
            "status": ev.get("interpretation_status"),
            "doc_id": ev.get("doc_id"),
            "source": ev.get("source_label")
        })

    # Clean sources
    sources = deduplicate_sources(evidence_pack.get("sources") or [])
    
    # The final 'answer' for the UI (Markdown)
    answer = intro
    if sources:
        answer += "\n\n**Sources consultées :**\n" + "\n".join(_source_lines(sources))

    return {
        "intro": intro,
        "visualization": viz_facts,
        "data_table": data_table,
        "sources": sources,
        "conclusion": None,
        "answer": answer,
        "mode": "specialized_visualization_composer",
    }


def compose_patient_inventory_count_answer(count: int) -> dict[str, Any]:
    """Composes deterministic count-only answer for patient metadata inventory."""
    safe_count = int(count or 0)
    msg = f"L’analyse des métadonnées identifie {safe_count} patient{'s' if safe_count > 1 else ''} distinct{'s' if safe_count > 1 else ''} indexé{'s' if safe_count > 1 else ''} dans la base."
    return {
        "answer": msg,
        "count": safe_count,
        "mode": "deterministic_patient_count",
        "content_type": "text",
    }


def compose_patient_inventory_answer(
    inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    """ 
    Composes a professional answer for patient inventory requests.
    Returns a structured object for the Frontend PatientInventoryRenderer.
    """
    if not inventory:
        return {
            "answer": "Aucun patient n'est actuellement répertorié dans le système.",
            "patients": [],
            "mode": "deterministic_patient_inventory",
            "content_type": "text"
        }

    patient_count = len(inventory)
    intro = f"Les patients indexés dans la base sont listés ci-dessous avec leurs rapports associés ({patient_count} patient{'s' if patient_count > 1 else ''} trouvé{'s' if patient_count > 1 else ''})."
    
    # Build Markdown table fallback with clickable sources.
    table = "| Patient | Rapports | Aperçu | Sources |\n| :--- | :---: | :--- | :--- |\n"
    
    for item in inventory:
        p_token = item["patient"]
        count = item["report_count"]
        range_label = item["report_range_label"]
        
        reports = list(item.get("reports") or [])
        clickable = []
        for rep in reports[:4]:
            href = str(rep.get("source_url") or rep.get("viewer_url") or "").strip()
            label = str(rep.get("label") or rep.get("filename") or rep.get("doc_id") or "rapport").strip()
            if href:
                clickable.append(f"[{label}]({href})")
        suffix = ", …" if len(reports) > 4 else ""
        source_cell = ", ".join(clickable) + suffix if clickable else "non disponible"
        table += f"| **{p_token}** | {count} | {range_label} | {source_cell} |\n"

    # For the fallback Markdown answer, we include the intro + table.
    # We do NOT add the global source citations block here if we expect the UI to handle it,
    # but we provide it in the 'sources' field for the renderer.
    
    global_sources = []
    for item in inventory:
        for src in item["sources"]:
            if not any(gs["doc_id"] == src["doc_id"] for gs in global_sources):
                global_sources.append({
                    "doc_id": src["doc_id"],
                    "label": src["label"],
                    "url": src["source_url"],
                    "viewer_url": src["viewer_url"]
                })

    answer = f"{intro}\n\n{table}"
    
    return {
        "intro": intro,
        "patients": inventory, 
        "sources": global_sources,
        "answer": answer,
        "mode": "deterministic_patient_inventory",
        "content_type": "patient_inventory"
    }


def compose_professional_answer(
    user_question: str,
    query_understanding: QueryUnderstanding,
    evidence_pack: dict[str, Any],
    mode: str = "auto",
    *,
    source_citations: list[dict[str, Any]] | None = None,
    llm_client: LLMClient | None = None,
    provider: str = DEFAULT_LLM_PROVIDER,
    model: str = DEFAULT_LLM_MODEL,
    temperature: float = DEFAULT_LLM_TEMPERATURE,
    num_ctx: int = DEFAULT_LLM_NUM_CTX,
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

    critical_deterministic_intents = {
        "reference_range_lookup",
    }
    if str(getattr(query_understanding, "intent", "")).strip().lower() in critical_deterministic_intents:
        return fallback

    if bool(getattr(query_understanding.presentation_intent, "user_requested_visualization", False)):
        try:
            return compose_visualization_answer(
                user_question=user_question,
                query_understanding=query_understanding,
                evidence_pack=evidence_pack,
                llm_client=llm_client,
                provider=provider,
                model=model,
            )
        except Exception as e:
            # Fallback to standard answer if visualization generation fails
            fb = dict(fallback)
            fb["llm_error"] = str(e)
            return fb

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
    contract_ok, contract_errors = _validate_writer_evidence_pack_contract(compact_pack)
    if not contract_ok:
        LOGGER.warning(
            "contract_violation %s",
            json.dumps(
                {
                    "event": "contract_violation",
                    "component": "professional_writer",
                    "intent": str(getattr(query_understanding, "intent", "") or "") or None,
                    "mode": mode,
                    "errors": contract_errors,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
        out = dict(fallback)
        out["mode"] = "writer_contract_violation_fallback"
        out["llm_error"] = "writer_evidence_contract_violation"
        out["contract_violation"] = contract_errors
        return out
    route_specific_block = _route_specific_writer_block(query_understanding)

    prompt = (
        f"{PROFESSIONAL_WRITER_SYSTEM_PROMPT}\n\n{PROFESSIONAL_WRITER_VISUALIZATION_RULES}\n\n"
        "RÈGLE STRICTE: reformule uniquement les facts de results_locked.\n"
        "INTERDIT: ajouter/supprimer/modifier analyte, valeur, unité, référence, statut ou source.\n"
        "INTERDIT: recalculer, diagnostiquer, proposer un traitement, utiliser un résultat antérieur comme valeur actuelle.\n"
        "Si une donnée manque, écrire 'non présent' ou 'non disponible'.\n"
        f"{route_specific_block}\n"
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
            timeout=max(6, int(timeout)),
            keep_alive=str(os.getenv("MEDICAL_RAG_OLLAMA_KEEP_ALIVE", "10m")).strip() or "10m",
        ).strip()
        if not llm_answer:
            out = dict(fallback)
            out["llm_error"] = "empty_llm_answer"
            out["llm_prompt_preview"] = prompt[:1200]
            return out

        guard_disabled = _llm_quality_guard_disabled()

        if (not guard_disabled) and re.search(r"\brésultat\(s\)|\bcorrespondant\(s\)", llm_answer, flags=re.IGNORECASE):
            out = dict(fallback)
            out["mode"] = "llm_writer_quality_fallback"
            out["llm_error"] = "ugly_pluralization"
            out["llm_prompt_preview"] = prompt[:1200]
            out["llm_candidate_answer"] = llm_answer
            return out
        if (not guard_disabled) and re.search(r"\bchart\b", norm_text(llm_answer), flags=re.IGNORECASE):
            out = dict(fallback)
            out["mode"] = "llm_writer_quality_fallback"
            out["llm_error"] = "internal_chart_term_visible"
            out["llm_prompt_preview"] = prompt[:1200]
            out["llm_candidate_answer"] = llm_answer
            return out

        viz = dict(compact_pack.get("visualization_facts") or {})
        if (not guard_disabled) and bool(viz.get("fallback_used")):
            requested_label = _safe_str(viz.get("requested_label")).lower()
            rendered_label = _safe_str(viz.get("rendered_label")).lower()
            fallback_reason = _safe_str(viz.get("fallback_reason")).lower()
            ans_norm = norm_text(llm_answer)
            if requested_label and norm_text(requested_label) not in ans_norm:
                out = dict(fallback)
                out["mode"] = "llm_writer_quality_fallback"
                out["llm_error"] = "requested_visualization_not_respected"
                out["llm_prompt_preview"] = prompt[:1200]
                out["llm_candidate_answer"] = llm_answer
                return out
            if rendered_label and norm_text(rendered_label) not in ans_norm:
                out = dict(fallback)
                out["mode"] = "llm_writer_quality_fallback"
                out["llm_error"] = "rendered_visualization_not_mentioned"
                out["llm_prompt_preview"] = prompt[:1200]
                out["llm_candidate_answer"] = llm_answer
                return out
            if fallback_reason:
                key_terms = [tok for tok in re.findall(r"[a-zA-ZÀ-ÿ]{5,}", fallback_reason) if tok not in {"dans", "pour", "avec", "encore"}]
                if key_terms and not any(norm_text(term) in ans_norm for term in key_terms[:4]):
                    out = dict(fallback)
                    out["mode"] = "llm_writer_quality_fallback"
                out["llm_error"] = "fallback_reason_missing"
                out["llm_prompt_preview"] = prompt[:1200]
                out["llm_candidate_answer"] = llm_answer
                return out

        if not guard_disabled:
            preserves_facts, fact_error = _llm_writer_preserves_facts(
                llm_answer=llm_answer,
                evidences=evidences,
                source_citations=source_citations or [],
            )
            if not preserves_facts:
                out = dict(fallback)
                out["mode"] = "llm_writer_quality_fallback"
                out["llm_error"] = fact_error or "llm_modified_facts"
                out["llm_prompt_preview"] = prompt[:1200]
                out["llm_candidate_answer"] = llm_answer
                return out

        has_source_col = bool(re.search(r"(?im)^\|\s*.*\bsource\b.*\|$", llm_answer or ""))
        if "sources" not in llm_answer.lower() and not has_source_col:
            source_block = _build_sources_block(
                source_citations or [],
                clickable_requested=bool(getattr(query_understanding, "source_clickable_requested", False)),
            )
            if source_block:
                llm_answer = llm_answer.rstrip() + "\n\n" + source_block

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
            "mode": "hybrid_structured_llm_writer" if mode == "hybrid_structured_llm_writer" else "llm_professional_writer",
            "llm_error": "quality_guard_disabled_debug_mode" if guard_disabled else None,
            "llm_prompt_preview": prompt[:1200],
            "llm_candidate_answer": llm_answer,
        }
    except LLMClientError as exc:
        out = dict(fallback)
        out["mode"] = "llm_writer_error_fallback"
        out["llm_error"] = str(exc)
        out["llm_prompt_preview"] = prompt[:1200]
        return out


# =============================================================================
# ClinicalDeterministicRenderer — Phase 3 / Option A
# Deterministic, evidence-only renderer for single_analyte_lookup &
# doc_scoped_results routes. Zero LLM, zero hallucination.
# =============================================================================

class ClinicalDeterministicRenderer:
    """
    Renders clinical evidence rows into safe, formatted outputs for physician UX.

    Hard constraints:
    - Uses ONLY displayed_evidences as facts. Never invents values.
    - Never diagnoses, never recommends treatment, never exposes PII.
    - Emits debug payload ONLY when debug=True.
    - If compatible evidence rows exist (> 0), never returns "non retrouvé".

    Output format:
        {
            "text": str,                   # Human-readable markdown
            "table": list[dict] | None,    # Structured rows for detailed mode
            "sources": list[str],          # doc_id citations
            "conclusion": str,             # Single factual conclusion line
            "debug": dict | None,          # Machine payload, None unless debug=True
            "mode": str,                   # "compact" | "detailed" | "not_found"
        }
    """

    # ------------------------------------------------------------------ #
    #  Status mapping                                                      #
    # ------------------------------------------------------------------ #
    _STATUS_MAP: dict[str, str] = {
        "above_reference": "au-dessus de la référence",
        "below_reference": "en dessous de la référence",
        "within_reference": "dans la référence",
        "not_interpretable": "valeur non numériquement exploitable",
        "not_numeric": "valeur non numériquement exploitable",
    }

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_display_label(row: dict) -> str:
        """Return the best human-readable analyte label for a row."""
        try:
            from medical_entity_resolver import get_display_analyte_label
            label = get_display_analyte_label(row)
            if label and label.lower() not in ("non précisé", "non precise"):
                return label
        except Exception:
            pass
        for field in ("analyte_label", "display_name", "analyte", "analyte_norm"):
            val = str(row.get(field) or "").strip()
            if val and val.lower() not in ("non précisé", "non precise"):
                return val
        return "Analyte inconnu"

    @staticmethod
    def _get_value_str(row: dict) -> str:
        """Return the numeric value string from a row."""
        val = str(row.get("current_value") or row.get("value_raw") or row.get("value_numeric") or "").strip()
        return val if val else "–"

    @staticmethod
    def _get_unit(row: dict) -> str:
        return str(row.get("unit") or "").strip()

    @staticmethod
    def _get_status_fr(row: dict) -> str:
        """Map technical_status / status to a French display label."""
        raw = str(
            row.get("technical_status_code")
            or row.get("interpretation_status")
            or row.get("technical_status")
            or row.get("status")
            or ""
        ).strip().lower().replace("-", "_")
        # Normalise common variants
        if "above" in raw:
            raw = "above_reference"
        elif "below" in raw or "under" in raw:
            raw = "below_reference"
        elif "within" in raw or "normal" in raw or "dans" in raw:
            raw = "within_reference"
        return ClinicalDeterministicRenderer._STATUS_MAP.get(raw, "valeur non numériquement exploitable")

    @staticmethod
    def _get_ref_concise(row: dict) -> str:
        """Return a concise ≤140-char reference string."""
        ref = str(row.get("reference") or row.get("reference_range") or "").strip()
        unit = ClinicalDeterministicRenderer._get_unit(row)
        if not ref:
            return "Réf: voir détail"
        full = f"Réf: {ref}"
        if unit and unit not in ref:
            full = f"Réf: {ref} {unit}"
        return full[:140]

    @staticmethod
    def _get_doc_id(row: dict) -> str:
        return str(row.get("doc_id") or "").strip()

    @staticmethod
    def _get_source_label(row: dict) -> str:
        """Return a short source citation string."""
        doc_id = ClinicalDeterministicRenderer._get_doc_id(row)
        page = row.get("page") or row.get("page_number")
        row_idx = row.get("row") or row.get("row_index")
        label = doc_id or "source inconnue"
        if page:
            label += f" — page {page}"
            if row_idx:
                label += f", ligne {row_idx}"
        return label

    @staticmethod
    def _scope_label(query_understanding: dict) -> str:
        doc_ids = list(query_understanding.get("requested_doc_ids") or [])
        if not doc_ids:
            return "global"
        if len(doc_ids) == 1:
            return str(doc_ids[0])
        return ", ".join(str(d) for d in doc_ids[:3]) + ("…" if len(doc_ids) > 3 else "")

    @staticmethod
    def _user_query(query_understanding: dict) -> str:
        return str(
            query_understanding.get("original_user_question")
            or query_understanding.get("raw_user_request")
            or "bilan demandé"
        ).strip()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def render_compact(
        self,
        displayed_evidences: list[dict],
        query_understanding: dict,
        debug: bool = False,
    ) -> dict:
        """
        Compact mode — ≤ 5 lines (excluding sources block and conclusion).
        Returns structured dict with text, sources, conclusion, and optional debug.
        """
        if not displayed_evidences:
            analytes = list(query_understanding.get("requested_analytes") or [])
            doc_ids = list(query_understanding.get("requested_doc_ids") or [])
            analyte = analytes[0] if analytes else "analyte inconnu"
            doc_id = doc_ids[0] if doc_ids else None
            return self.render_not_found(analyte, doc_id, debug_info={"requested_analytes": analytes, "requested_doc_ids": doc_ids} if debug else None)

        scope = self._scope_label(query_understanding)
        user_q = self._user_query(query_understanding)
        total = len(displayed_evidences)

        # Count anomalies (above + below reference)
        anomaly_rows = [r for r in displayed_evidences if self._get_status_fr(r) in ("au-dessus de la référence", "en dessous de la référence")]
        normal_rows = [r for r in displayed_evidences if r not in anomaly_rows]

        # Header line
        lines: list[str] = [f"**Bilan demandé :** {user_q} — {scope}"]

        # Summary line
        n_anom = len(anomaly_rows)
        if n_anom:
            lines.append(f"{total} valeur(s) exploitable(s) retrouvée(s). {n_anom} valeur(s) hors référence.")
        else:
            lines.append(f"{total} valeur(s) exploitable(s) retrouvée(s). Aucune anomalie.")

        # Up to 3 key rows (anomalies first, then normal)
        key_rows = (anomaly_rows + normal_rows)[:3]
        for row in key_rows:
            label = self._get_display_label(row)
            value = self._get_value_str(row)
            unit = self._get_unit(row)
            status = self._get_status_fr(row)
            doc = self._get_doc_id(row)
            val_str = f"{value} {unit}".strip()
            lines.append(f"- **{label}** = {val_str} ({status}) — {doc}")

        # Sources block
        sources = sorted({self._get_source_label(r) for r in displayed_evidences})

        # Conclusion technique
        if n_anom:
            top_analytes = ", ".join(
                f"{self._get_display_label(r)} — {self._get_doc_id(r)}"
                for r in anomaly_rows[:2]
            )
            conclusion = f"Conclusion technique : {n_anom} valeur(s) hors référence ({top_analytes})."
        else:
            conclusion = f"Conclusion technique : {total} résultat(s) dans les limites de référence rapportées."

        text = "\n".join(lines)

        result: dict = {
            "text": text,
            "table": None,
            "sources": sources,
            "conclusion": conclusion,
            "mode": "compact",
            "debug": None,
        }
        if debug:
            result["debug"] = self._build_debug(displayed_evidences, query_understanding)
        return result

    def render_detailed(
        self,
        displayed_evidences: list[dict],
        query_understanding: dict,
        debug: bool = False,
    ) -> dict:
        """
        Detailed mode — markdown table with columns:
        Analyte | Valeur (unit source) | Statut | Réf concise | Document (doc_id)
        """
        if not displayed_evidences:
            analytes = list(query_understanding.get("requested_analytes") or [])
            doc_ids = list(query_understanding.get("requested_doc_ids") or [])
            analyte = analytes[0] if analytes else "analyte inconnu"
            doc_id = doc_ids[0] if doc_ids else None
            return self.render_not_found(analyte, doc_id, debug_info={"requested_analytes": analytes, "requested_doc_ids": doc_ids} if debug else None)

        scope = self._scope_label(query_understanding)
        user_q = self._user_query(query_understanding)

        header = f"**Bilan demandé :** {user_q} — {scope}\n"

        # Build structured table rows
        table_rows: list[dict] = []
        for row in displayed_evidences:
            label = self._get_display_label(row)
            value = self._get_value_str(row)
            unit = self._get_unit(row)
            val_str = f"{value} {unit}".strip()
            status = self._get_status_fr(row)
            ref_c = self._get_ref_concise(row)
            doc_id = self._get_doc_id(row)
            table_rows.append({
                "Analyte": label,
                "Valeur (unit source)": val_str,
                "Statut": status,
                "Réf concise": ref_c,
                "Document (doc_id)": doc_id,
            })

        # Markdown table
        md_lines = [
            header,
            "| Analyte | Valeur (unit source) | Statut | Réf concise | Document (doc_id) |",
            "|---------|----------------------|--------|-------------|-------------------|",
        ]
        for tr in table_rows:
            md_lines.append(
                f"| {tr['Analyte']} | {tr['Valeur (unit source)']} | {tr['Statut']} | {tr['Réf concise']} | {tr['Document (doc_id)']} |"
            )

        text = "\n".join(md_lines)

        sources = sorted({self._get_source_label(r) for r in displayed_evidences})

        # Conclusion
        anomaly_count = sum(
            1 for r in displayed_evidences
            if self._get_status_fr(r) in ("au-dessus de la référence", "en dessous de la référence")
        )
        if anomaly_count:
            top = next(
                (f"{self._get_display_label(r)} — {self._get_doc_id(r)}"
                 for r in displayed_evidences
                 if self._get_status_fr(r) in ("au-dessus de la référence", "en dessous de la référence")),
                ""
            )
            conclusion = f"Conclusion technique : {anomaly_count} valeur(s) hors référence (ex: {top})."
        else:
            conclusion = f"Conclusion technique : {len(displayed_evidences)} résultat(s) dans les limites de référence rapportées."

        result: dict = {
            "text": text,
            "table": table_rows,
            "sources": sources,
            "conclusion": conclusion,
            "mode": "detailed",
            "debug": None,
        }
        if debug:
            result["debug"] = self._build_debug(displayed_evidences, query_understanding)
        return result

    def render_not_found(
        self,
        analyte: str,
        doc_id: str | None = None,
        debug_info: dict | None = None,
    ) -> dict:
        """
        Exact not-found template per clinical spec.
        Emits CTA for physician to refine query.
        """
        analyte_upper = str(analyte or "analyte").upper().strip()
        if doc_id:
            doc_token = str(doc_id).strip()
            title = f"{analyte_upper} — {doc_token}"
            body = (
                f"Aucune valeur numérique de {analyte_upper} indexée trouvée dans {doc_token} "
                f"parmi les résultats disponibles. "
                f"Souhaitez‑vous : (1) rechercher globalement, (2) vérifier alias/orthographe, "
                f"(3) préciser un autre rapport ?"
            )
            conclusion = (
                f"Conclusion technique : aucune valeur exploitable identifiée pour "
                f"{analyte_upper} dans le rapport demandé."
            )
        else:
            title = analyte_upper
            body = (
                f"Aucune valeur numérique de {analyte_upper} n'a été identifiée dans les rapports indexés. "
                f"Voulez‑vous une recherche étendue ?"
            )
            conclusion = (
                f"Conclusion technique : aucune valeur exploitable identifiée pour "
                f"{analyte_upper} dans les rapports disponibles."
            )

        text = f"### {title}\n\n{body}\n\n{conclusion}"

        result: dict = {
            "text": text,
            "table": None,
            "sources": [],
            "conclusion": conclusion,
            "mode": "not_found",
            "debug": debug_info,
        }
        return result

    def render(
        self,
        displayed_evidences: list[dict],
        query_understanding: dict,
        answer_style: str = "compact",
        debug: bool = False,
    ) -> dict:
        """
        Dispatcher: routes to compact / detailed / not_found based on answer_style
        and presence of evidence rows.
        """
        style = str(answer_style or "compact").strip().lower()
        if style in ("detailed", "table", "strict_json"):
            return self.render_detailed(displayed_evidences, query_understanding, debug=debug)
        return self.render_compact(displayed_evidences, query_understanding, debug=debug)

    # ------------------------------------------------------------------ #
    #  Debug payload builder                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_debug(displayed_evidences: list[dict], query_understanding: dict) -> dict:
        return {
            "requested_analytes": list(query_understanding.get("requested_analytes") or []),
            "requested_doc_ids": list(query_understanding.get("requested_doc_ids") or []),
            "found_count": len(displayed_evidences),
            "displayed_evidences_count": len(displayed_evidences),
            "matching_strategy": str(query_understanding.get("_matching_strategy") or "unknown"),
            "confidence_score": float(query_understanding.get("_confidence_score") or 0.0),
        }


# Singleton for import convenience
clinical_renderer = ClinicalDeterministicRenderer()
