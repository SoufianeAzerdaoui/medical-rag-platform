from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("medical_rag.generation.config")

try:  # pragma: no cover
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


_DEFAULT_MEDICAL_TOPICS: dict[str, Any] = {
    "topics": {
        "thyroid": {
            "triggers": ["hyperthyro", "hypothyro", "thyroide", "thyroid", "tsh", "t3", "t4"],
            "analytes": [
                "tsh",
                "tshus",
                "t4_libre",
                "t4libre",
                "t3_libre",
                "t3libre",
                "anti_tg",
                "anti tg",
                "anti_tpo",
                "anti tpo",
                "trak",
            ],
            "excluded_unless_explicit": ["acth", "insuline", "pth", "gh"],
            "rules": {
                "high_groups": {
                    "tsh": ["tsh", "tshus"],
                    "t3_t4": ["t3_libre", "t3libre", "t4_libre", "t4libre"],
                }
            },
        },
        "tumor_markers": {
            "triggers": ["marqueur tumoral", "cancer", "tumeur", "oncologie"],
            "analytes": ["ace", "psa_totale", "ca_15_3"],
            "excluded_unless_explicit": [],
        },
    }
}

_DEFAULT_ANALYTE_FAMILIES: dict[str, Any] = {
    "families": {
        "inflammation": {"weight": 0.15, "analytes": ["crp", "procalcitonine", "pct"]},
        "cardio_muscle": {"weight": 0.15, "analytes": ["troponine", "ckmb", "cpkmb", "ck", "ldh"]},
        "renal": {"weight": 0.15, "analytes": ["creatinine", "uree", "dfg", "clairance creatinine"]},
        "lipid": {"weight": 0.15, "analytes": ["triglycerides", "cholesterol ldl", "cholesterol hdl", "apo b", "apo a1"]},
        "protein_nutrition": {"weight": 0.15, "analytes": ["albumine", "proteines totales"]},
        "hepatic": {"weight": 0.15, "analytes": ["asat", "alat", "ggt", "bilirubine"]},
        "endocrino": {"weight": 0.15, "analytes": ["tsh", "tshus", "t3", "t4", "t3_libre", "t4_libre", "insuline"]},
        "vitamines": {"weight": 0.15, "analytes": ["vitamine d", "vitamine b12", "b12", "ferritine"]},
    }
}

_DEFAULT_PRIORITY_SCORING: dict[str, Any] = {
    "priority_scoring": {
        "ratio_weight": 1.2,
        "textual_severity_bonus": 0.6,
        "family_bonus": 0.15,
        "thresholds": {"high": 2.4, "moderate": 0.9, "low": 0.2},
        "textual_severity_terms": [
            "tres haute",
            "très haute",
            "tres bas",
            "très bas",
            "critique",
            "severe",
            "sévère",
            "majeur",
            "urgent",
            "pathologique",
            "tres eleve",
            "très élevé",
            "tres basse",
            "très basse",
        ],
    }
}

_DEFAULT_ASSISTANT_MESSAGES: dict[str, Any] = {
    "general_conversation": {
        "small_talk": "Bonjour. Je peux vous aider à analyser les rapports médicaux indexés : anomalies biologiques, comparaisons entre rapports, valeurs hors référence ou synthèses descriptives sans diagnostic.",
        "identity_question": "Je suis un assistant RAG médical. Mon rôle est d’aider à interroger les rapports médicaux indexés en restant fidèle aux données extraites, avec sources, sans poser de diagnostic ni recommander de traitement.",
        "capabilities": "Vous pouvez me demander de lister les anomalies d’un rapport, comparer deux rapports, rechercher des valeurs hors référence dans tous les documents, ou produire une synthèse descriptive sans diagnostic.",
        "help_question": "Vous pouvez me demander de lister les anomalies d’un rapport, comparer deux rapports, rechercher des valeurs hors référence dans tous les documents, ou produire une synthèse descriptive sans diagnostic.",
        "fallback": "Je peux vous aider à interroger les rapports médicaux indexés.",
    },
    "diagnostic_safety": {
        "generic": {
            "cancer_refusal": "Non, on ne peut pas conclure à un cancer uniquement à partir de ces marqueurs.",
            "markers_intro": "Constat technique sur les marqueurs retrouvés :",
            "closing": "Ces marqueurs biologiques ne suffisent pas à poser un diagnostic ; une interprétation médicale spécialisée est nécessaire.",
        },
        "thyroid": {
            "detail_fallback": "anomalies thyroïdiennes",
            "discordance_sentence": "Ce profil est biologiquement discordant pour une hyperthyroïdie primaire.",
            "no_diagnostic_sentence": "Cependant, on ne peut pas conclure seul à un diagnostic thyroïdien à partir de ce document.",
            "correlation_sentence": "Il faut corréler avec le contexte clinique, les traitements, les interférences analytiques et, si besoin, répéter/compléter le bilan.",
            "summary_template": "Le document montre des anomalies thyroïdiennes importantes : {details_txt}. {no_diagnostic_sentence} {discordance} {correlation_sentence}",
        },
    },
    "clarifications": {
        "global_summary_no_scope": (
            "Je dois connaître le document, le patient ou le périmètre à résumer. "
            "Précisez un rapport ou demandez une synthèse sur l’ensemble des rapports disponibles."
        ),
        "abnormal_without_scope": (
            "La demande « résultats anormaux » nécessite un périmètre explicite. "
            "Précisez un rapport (ex: report 24) ou confirmez une recherche globale sur tous les rapports."
        ),
        "abnormal_without_scope_conclusion": (
            "Conclusion technique : clarification de périmètre requise avant extraction déterministe des anomalies."
        ),
    },
    "fallbacks": {
        "single_analyte_not_found_template": (
            "### {analyte_label} — {doc_labels}\n\n"
            "Aucun résultat correspondant à {analyte_label} n’a été retrouvé dans {doc_labels} parmi les résultats disponibles.\n\n"
            "Conclusion technique : aucune valeur numérique exploitable n’a été identifiée pour cet analyte dans le rapport demandé."
        ),
        "topic_not_found_template": (
            "Aucun résultat correspondant au thème {analyte_label} n’a été retrouvé dans {doc_labels}.\n\n"
            "Conclusion technique : aucune donnée exploitable n’a été identifiée pour ce thème dans le périmètre demandé."
        ),
        "document_not_found_template": (
            "Aucune donnée structurée exploitable n’a été extraite de {doc_pdf_labels} pour la demande formulée.\n\n"
            "Source documentaire : {doc_pdf_labels}.\n"
            "Conclusion technique : le périmètre documentaire demandé ne contient pas de données compatibles."
        ),
        "ambiguous_analyte_template": (
            "La demande nécessite de préciser l’analyte ciblé.\n\n"
            "Conclusion technique : clarification d’analyte requise avant extraction déterministe."
        ),
        "ambiguous_document_scope_template": (
            "La demande nécessite un périmètre documentaire explicite.\n"
            "Précisez un rapport (ex: report 24) ou confirmez une recherche globale.\n\n"
            "Conclusion technique : clarification de périmètre requise avant extraction déterministe."
        ),
        "diagnosis_refusal_template": (
            "Je ne peux pas poser de diagnostic à partir de ces résultats.\n\n"
            "Conclusion technique : refus diagnostique de sécurité, sans interprétation clinique."
        ),
        "treatment_refusal_template": (
            "Je ne peux pas recommander de traitement à partir de ces résultats seuls.\n\n"
            "Conclusion technique : restitution factuelle uniquement, sans recommandation thérapeutique."
        ),
        "pii_refusal_template": (
            "Je ne peux pas divulguer de données personnelles identifiantes.\n\n"
            "Conclusion technique : refus de sécurité PII."
        ),
        "partial_answer_template": (
            "Une réponse partielle est disponible.\n"
            "Documents avec résultat : {matched_labels}.\n"
            "Documents sans résultat : {missing_labels}.\n\n"
            "Conclusion technique : réponse limitée aux données compatibles retrouvées."
        ),
        "insufficient_evidence_template": (
            "Aucune donnée structurée exploitable n’a été identifiée pour répondre de façon fiable.\n\n"
            "Source documentaire : {doc_pdf_labels}.\n"
            "Conclusion technique : aucun résultat exploitable n’a été identifié pour {analyte_label}{criterion} dans {doc_labels}."
        ),
    },
}

_DEFAULT_SAFETY_GUARDRAILS: dict[str, Any] = {
    "diagnostic_safety": {
        "thyroid_topic_keywords": ["hyperthyro", "hypothyro", "thyroid", "thyroide", "thyroïde"],
        "strong_suggestion_patterns": [
            r"\bsugg[eè]re\s+une?\s+hyperthyro",
            r"\bcompatible\s+avec\s+une?\s+hyperthyro",
            r"\b[eé]voque\s+une?\s+hyperthyro",
            r"\bindique\s+une?\s+hyperthyro",
            r"\ben\s+faveur\s+d['’]une?\s+hyperthyro",
        ],
        "explicit_negation_markers": [
            "ne permet pas de conclure",
            "n est pas suffisant pour conclure",
            "n'est pas suffisant pour conclure",
            "on ne peut pas conclure",
        ],
        "forbidden_clinical_style_patterns": [
            r"(?im)^.*il est essentiel de.*$",
            r"(?im)^.*examens compl[eé]mentaires.*$",
            r"(?im)^.*[eé]valuation compl[eè]te.*$",
            r"(?im)^.*cause sous[-\\s]jacente.*$",
            r"(?im)^.*prendre en compte les autres facteurs cliniques.*$",
            r"(?im)^.*confirmer ou [eé]liminer le diagnostic.*$",
            r"(?im)^.*\\bconsulter\\b.*$",
        ],
        "limitation_sentence": "L’interprétation reste limitée aux données biologiques fournies.",
        "discordance_replacement": "profil biologique discordant pour une hyperthyroïdie primaire",
    }
}

_DEFAULT_MEDICAL_ENTITY_RESOLVER: dict[str, Any] = {
    "medical_entity_resolver": {
        "topic_keywords": {
            "thyroid": ["thyroid", "thyroide", "thyroïde", "thyroidien", "thyroïdien", "tsh", "t4", "t3", "hyperthyro"],
            "toxicology": ["toxicologie", "toxicology", "pharmacotoxicologie", "toxique", "opiac", "benzodiazep", "cocaine", "amphet"],
            "renal": ["renal", "rénal", "renale", "rénale", "creatinine", "créatinine", "uree", "urée", "dfg"],
            "hepatic": ["hepat", "hépat", "foie", "alat", "asat", "ggt", "bilirub"],
            "inflammation": ["inflammation", "crp", "proteine c reactive", "protéine c réactive"],
        },
        "analyte_families": {
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
        },
        "equivalent_groups": [
            ["tsh", "tshus"],
            ["ckmb", "cpkmb"],
        ],
        "safe_extra_aliases": {
            "creat": "creatinine",
            "creatininemie": "creatinine",
            "uricemie": "acide_urique",
            "uric acid": "acide_urique",
            "alp": "phosphatase_alcaline",
            "pal": "phosphatase_alcaline",
            "tsh ultra sensible": "tshus",
            "thyroid stimulating hormone": "tsh",
            "thyreostimuline": "tsh",
            "ft3": "t3_libre",
            "ft4": "t4_libre",
        },
    }
}

_DEFAULT_GENERATION_ROUTING: dict[str, Any] = {
    "generation_routing": {
        "abnormal_without_scope": {
            "global_scope_markers": [
                "tous les rapports",
                "rapports disponibles",
                "ensemble des rapports",
                "dans tous les rapports",
                "quels rapports",
            ],
            "abnormal_hint_patterns": [
                r"\banomal\w*",
                r"\bhors\s+(?:de\s+la\s+)?(?:reference|norme|intervalle)\b",
                r"\b(?:resultat|résultat|resultats|résultats|valeur|valeurs|taux)\b",
                r"\bbiolog\w*\b",
            ],
        }
    }
}


_DEFAULT_GENERATION_TEMPLATES: dict[str, Any] = {
    "reference_ranges_summary": {
        "interpretive_markers": [],
        "target_counts": {
            "minmax": 8,
            "sex_age": 11,
            "threshold": 7,
            "interpretive": 4,
        },
        "priority_order": [],
        "line_labels": {
            "title": "Note sur les valeurs physiologiques",
            "document_prefix": "Document analysé :",
            "minmax_prefix": "Plages min-max :",
            "sex_age_prefix": "Références selon âge/sexe :",
            "threshold_prefix": "Seuils et catégories interprétatives (incluant seuils interprétatifs de risque) :",
            "source_prefix": "Source :",
        },
        "narrative_intro": (
            "Le rapport contient plusieurs formats de valeurs physiologiques : "
            "plages min-max, seuils numériques, références selon l’âge, selon le sexe "
            "et catégories interprétatives."
        ),
        "conclusion_no_diagnosis": (
            "Conclusion technique : note descriptive uniquement, sans diagnostic médical."
        ),
        "conclusion_default": "Conclusion technique : note descriptive uniquement, sans diagnostic médical.",
        "line_limits": {
            "min": 7,
            "max": 8,
            "default": 7,
            "llm_note_min": 5,
            "llm_note_max": 7,
        },
        "llm_intro_filters": {
            "max_colon_line_length": 140,
            "max_line_length": 220,
            "excluded_prefixes": [
                "note sur les valeurs physiologiques",
                "plages min max",
                "seuils",
                "references selon",
                "categories interpretatives",
                "conclusion technique",
                "source",
            ],
        },
        "llm_style_guard": {
            "banned_line_prefixes": [
                "Plages min-max :",
                "Références selon âge/sexe :",
                "Seuils et catégories interprétatives :",
                "Catégories interprétatives :",
            ],
            "required_narrative_markers": [
                "Le rapport contient plusieurs formats de valeurs physiologiques",
                "Les plages min-max concernent",
                "Certaines références varient",
                "D'autres paramètres utilisent",
                "Ces références servent à",
                "références selon l’âge",
                "références selon le sexe",
                "catégories interprétatives",
            ],
            "min_narrative_marker_hits": 2,
            "min_sentence_count": 3,
            "min_narrative_verb_hits": 2,
        },
    }
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(dict(out[k]), v)
        else:
            out[k] = v
    return out


def load_yaml_config(path: str | Path, default: dict[str, Any]) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        LOGGER.warning("Config file missing: %s; using defaults.", p)
        return dict(default)
    try:
        raw = p.read_text(encoding="utf-8")
    except Exception:
        LOGGER.warning("Config file unreadable: %s; using defaults.", p)
        return dict(default)

    loaded: dict[str, Any] | None = None
    if yaml is not None:
        try:
            parsed = yaml.safe_load(raw)
            if isinstance(parsed, dict):
                loaded = parsed
        except Exception:
            loaded = None
    if loaded is None:
        try:
            parsed_json = json.loads(raw)
            if isinstance(parsed_json, dict):
                loaded = parsed_json
        except Exception:
            loaded = None

    if loaded is None:
        LOGGER.warning("Config parse failed: %s; using defaults.", p)
        return dict(default)
    return _deep_merge(dict(default), loaded)


@lru_cache(maxsize=1)
def get_medical_topics_config() -> dict[str, Any]:
    return load_yaml_config(CONFIG_DIR / "medical_topics.yml", _DEFAULT_MEDICAL_TOPICS)


@lru_cache(maxsize=1)
def get_analyte_families_config() -> dict[str, Any]:
    return load_yaml_config(CONFIG_DIR / "analyte_families.yml", _DEFAULT_ANALYTE_FAMILIES)


@lru_cache(maxsize=1)
def get_priority_scoring_config() -> dict[str, Any]:
    return load_yaml_config(CONFIG_DIR / "priority_scoring.yml", _DEFAULT_PRIORITY_SCORING)


@lru_cache(maxsize=1)
def get_assistant_messages_config() -> dict[str, Any]:
    return load_yaml_config(CONFIG_DIR / "assistant_messages.yml", _DEFAULT_ASSISTANT_MESSAGES)


@lru_cache(maxsize=1)
def get_safety_guardrails_config() -> dict[str, Any]:
    return load_yaml_config(CONFIG_DIR / "safety_guardrails.yml", _DEFAULT_SAFETY_GUARDRAILS)


@lru_cache(maxsize=1)
def get_medical_entity_resolver_config() -> dict[str, Any]:
    return load_yaml_config(CONFIG_DIR / "medical_entity_resolver.yml", _DEFAULT_MEDICAL_ENTITY_RESOLVER)

@lru_cache(maxsize=1)
def get_generation_routing_config() -> dict[str, Any]:
    return load_yaml_config(CONFIG_DIR / "generation_routing.yml", _DEFAULT_GENERATION_ROUTING)


@lru_cache(maxsize=1)
def get_generation_templates_config() -> dict[str, Any]:
    return load_yaml_config(CONFIG_DIR / "generation_templates.yml", _DEFAULT_GENERATION_TEMPLATES)


__all__ = [
    "load_yaml_config",
    "get_medical_topics_config",
    "get_analyte_families_config",
    "get_priority_scoring_config",
    "get_assistant_messages_config",
    "get_safety_guardrails_config",
    "get_medical_entity_resolver_config",
    "get_generation_routing_config",
    "get_generation_templates_config",
]
