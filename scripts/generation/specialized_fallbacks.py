from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config_loader import get_assistant_messages_config


FALLBACK_KINDS: set[str] = {
    "single_analyte_not_found",
    "topic_not_found",
    "document_not_found",
    "ambiguous_analyte",
    "ambiguous_document_scope",
    "diagnosis_refusal",
    "treatment_refusal",
    "pii_refusal",
    "partial_answer",
    "insufficient_evidence",
}


@dataclass(frozen=True)
class SpecializedFallback:
    kind: str
    answer: str
    generation_mode: str
    warning_code: str


def _assistant_message(path: list[str], default: str) -> str:
    cfg: Any = get_assistant_messages_config() or {}
    node: Any = cfg
    for key in path:
        if not isinstance(node, dict):
            return default
        node = node.get(key)
    value = str(node or "").strip()
    return value or default


def _label_from_doc_id(doc_id: str) -> str:
    txt = str(doc_id or "").strip().replace("_", " ")
    return txt or "document demandé"


def _join_doc_labels(doc_ids: list[str]) -> str:
    labels = [_label_from_doc_id(d) for d in list(doc_ids or []) if str(d).strip()]
    if not labels:
        return "les documents demandés"
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} et {labels[1]}"
    return f"{', '.join(labels[:-1])} et {labels[-1]}"


def _safe_format(template: str, **kwargs: Any) -> str:
    try:
        return str(template).format(**kwargs).strip()
    except Exception:
        return str(template).strip()


def build_specialized_fallback(
    *,
    kind: str,
    requested_analytes: list[str] | None = None,
    requested_doc_ids: list[str] | None = None,
    matched_doc_ids: list[str] | None = None,
    missing_doc_ids: list[str] | None = None,
    requested_value: str | None = None,
    comparison_operator: str | None = None,
) -> SpecializedFallback:
    fallback_kind = str(kind or "").strip().lower()
    if fallback_kind not in FALLBACK_KINDS:
        fallback_kind = "insufficient_evidence"

    analytes = [str(a).strip() for a in list(requested_analytes or []) if str(a).strip()]
    docs = [str(d).strip() for d in list(requested_doc_ids or []) if str(d).strip()]
    matched_docs = [str(d).strip() for d in list(matched_doc_ids or []) if str(d).strip()]
    missing_docs = [str(d).strip() for d in list(missing_doc_ids or []) if str(d).strip()]
    analyte_label = analytes[0] if analytes else "cet analyte"
    doc_labels = _join_doc_labels(docs)
    matched_labels = _join_doc_labels(matched_docs)
    missing_labels = _join_doc_labels(missing_docs)

    op_norm = str(comparison_operator or "").strip().lower()
    val_txt = str(requested_value or "").strip()
    criterion = ""
    if op_norm in {">", ">="} and val_txt:
        criterion = f" strictement supérieur à {val_txt}"
    elif op_norm in {"<", "<="} and val_txt:
        criterion = f" strictement inférieur à {val_txt}"
    elif op_norm in {"=", "=="} and val_txt:
        criterion = f" égal à {val_txt}"
    elif val_txt:
        criterion = f" correspondant au critère {val_txt}"

    if fallback_kind == "single_analyte_not_found":
        template = _assistant_message(
            ["fallbacks", "single_analyte_not_found_template"],
            (
                "### {analyte_label} — {doc_labels}\n\n"
                "Aucun résultat correspondant à {analyte_label} n’a été retrouvé dans {doc_labels} parmi les résultats disponibles.\n\n"
                "Conclusion technique : aucune valeur numérique exploitable n’a été identifiée pour cet analyte dans le rapport demandé."
            ),
        )
        answer = _safe_format(template, analyte_label=analyte_label, doc_labels=doc_labels)
        return SpecializedFallback(
            kind=fallback_kind,
            answer=answer,
            generation_mode="deterministic_single_analyte_not_found",
            warning_code="specialized_fallback_single_analyte_not_found",
        )

    if fallback_kind == "topic_not_found":
        template = _assistant_message(
            ["fallbacks", "topic_not_found_template"],
            (
                "Aucun résultat correspondant au thème {analyte_label} n’a été retrouvé dans {doc_labels}.\n\n"
                "Conclusion technique : aucune donnée exploitable n’a été identifiée pour ce thème dans le périmètre demandé."
            ),
        )
        answer = _safe_format(template, analyte_label=analyte_label, doc_labels=doc_labels)
        return SpecializedFallback(
            kind=fallback_kind,
            answer=answer,
            generation_mode="deterministic_no_evidence_response",
            warning_code="specialized_fallback_topic_not_found",
        )

    if fallback_kind == "document_not_found":
        template = _assistant_message(
            ["fallbacks", "document_not_found_template"],
            (
                "Aucun résultat biologique exploitable n’a été retrouvé dans {doc_labels} pour la demande formulée.\n\n"
                "Conclusion technique : le périmètre documentaire demandé ne contient pas de données compatibles."
            ),
        )
        answer = _safe_format(template, doc_labels=doc_labels)
        return SpecializedFallback(
            kind=fallback_kind,
            answer=answer,
            generation_mode="deterministic_no_evidence_response",
            warning_code="specialized_fallback_document_not_found",
        )

    if fallback_kind == "ambiguous_analyte":
        template = _assistant_message(
            ["fallbacks", "ambiguous_analyte_template"],
            (
                "La demande nécessite de préciser l’analyte ciblé.\n\n"
                "Précisez l’analyte exact et, si besoin, le rapport visé.\n\n"
                "Conclusion technique : clarification d’analyte requise avant extraction déterministe."
            ),
        )
        answer = _safe_format(template, analyte_label=analyte_label)
        return SpecializedFallback(
            kind=fallback_kind,
            answer=answer,
            generation_mode="deterministic_no_evidence_response",
            warning_code="specialized_fallback_ambiguous_analyte",
        )

    if fallback_kind == "ambiguous_document_scope":
        template = _assistant_message(
            ["fallbacks", "ambiguous_document_scope_template"],
            (
                "La demande nécessite un périmètre documentaire explicite.\n"
                "Précisez un rapport (ex: report 24) ou confirmez une recherche globale.\n\n"
                "Conclusion technique : clarification de périmètre requise avant extraction déterministe."
            ),
        )
        answer = _safe_format(template, doc_labels=doc_labels)
        return SpecializedFallback(
            kind=fallback_kind,
            answer=answer,
            generation_mode="deterministic_no_evidence_response",
            warning_code="specialized_fallback_ambiguous_document_scope",
        )

    if fallback_kind == "diagnosis_refusal":
        template = _assistant_message(
            ["fallbacks", "diagnosis_refusal_template"],
            (
                "Je ne peux pas poser de diagnostic à partir de ces résultats.\n\n"
                "Conclusion technique : refus diagnostique de sécurité, sans interprétation clinique."
            ),
        )
        return SpecializedFallback(
            kind=fallback_kind,
            answer=_safe_format(template),
            generation_mode="deterministic_diagnostic_safety_refusal",
            warning_code="specialized_fallback_diagnosis_refusal",
        )

    if fallback_kind == "treatment_refusal":
        template = _assistant_message(
            ["fallbacks", "treatment_refusal_template"],
            (
                "Je ne peux pas recommander de traitement à partir de ces résultats seuls.\n\n"
                "Conclusion technique : restitution factuelle uniquement, sans recommandation thérapeutique."
            ),
        )
        return SpecializedFallback(
            kind=fallback_kind,
            answer=_safe_format(template),
            generation_mode="deterministic_treatment_refusal_with_technical_summary",
            warning_code="specialized_fallback_treatment_refusal",
        )

    if fallback_kind == "pii_refusal":
        template = _assistant_message(
            ["fallbacks", "pii_refusal_template"],
            (
                "Je ne peux pas divulguer de données personnelles identifiantes.\n\n"
                "Conclusion technique : refus de sécurité PII."
            ),
        )
        return SpecializedFallback(
            kind=fallback_kind,
            answer=_safe_format(template),
            generation_mode="deterministic_pii_refusal",
            warning_code="specialized_fallback_pii_refusal",
        )

    if fallback_kind == "partial_answer":
        template = _assistant_message(
            ["fallbacks", "partial_answer_template"],
            (
                "Une réponse partielle est disponible.\n"
                "Documents avec résultat : {matched_labels}.\n"
                "Documents sans résultat : {missing_labels}.\n\n"
                "Conclusion technique : réponse limitée aux données compatibles retrouvées."
            ),
        )
        answer = _safe_format(
            template,
            matched_labels=matched_labels,
            missing_labels=missing_labels,
            doc_labels=doc_labels,
        )
        return SpecializedFallback(
            kind=fallback_kind,
            answer=answer,
            generation_mode="deterministic_no_evidence_response",
            warning_code="specialized_fallback_partial_answer",
        )

    template = _assistant_message(
        ["fallbacks", "insufficient_evidence_template"],
        (
            "Information insuffisante dans les données structurées disponibles pour répondre de façon fiable.\n\n"
            "Conclusion technique : aucun résultat exploitable n’a été identifié pour {analyte_label}{criterion} dans {doc_labels}."
        ),
    )
    answer = _safe_format(
        template,
        analyte_label=analyte_label,
        criterion=criterion,
        doc_labels=doc_labels,
    )
    return SpecializedFallback(
        kind="insufficient_evidence",
        answer=answer,
        generation_mode="deterministic_no_evidence_response",
        warning_code="specialized_fallback_insufficient_evidence",
    )


def infer_specialized_fallback_kind(
    *,
    answerability_status: str,
    answerability_reason: str,
    safety_intent: str,
    requested_analytes: list[str] | None,
    requested_doc_ids: list[str] | None,
    ambiguity_flags: list[str] | None = None,
) -> str:
    status = str(answerability_status or "").strip().lower()
    reason = str(answerability_reason or "").strip().lower()
    safety = str(safety_intent or "").strip().lower()
    analytes = [str(a).strip() for a in list(requested_analytes or []) if str(a).strip()]
    docs = [str(d).strip() for d in list(requested_doc_ids or []) if str(d).strip()]
    flags = {str(f).strip().lower() for f in list(ambiguity_flags or []) if str(f).strip()}

    if "pii" in safety:
        return "pii_refusal"
    if "treatment" in safety:
        return "treatment_refusal"
    if "diagnostic" in safety or status == "unsafe":
        return "diagnosis_refusal"

    if status == "partially_answerable":
        return "partial_answer"

    if status == "ambiguous":
        if "missing_doc_scope" in flags or "multiple_doc_scope_ambiguous" in flags:
            return "ambiguous_document_scope"
        if analytes:
            return "ambiguous_analyte"
        return "ambiguous_document_scope"

    if status == "not_found":
        if analytes and len(analytes) == 1 and len(docs) == 1:
            return "single_analyte_not_found"
        if docs:
            return "document_not_found"
        if analytes:
            return "topic_not_found" if ("topic" in reason) else "insufficient_evidence"
        return "insufficient_evidence"

    if docs and not analytes:
        return "document_not_found"
    return "insufficient_evidence"


__all__ = [
    "FALLBACK_KINDS",
    "SpecializedFallback",
    "build_specialized_fallback",
    "infer_specialized_fallback_kind",
]
