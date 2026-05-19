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


__all__ = [
    "load_yaml_config",
    "get_medical_topics_config",
    "get_analyte_families_config",
    "get_priority_scoring_config",
    "get_assistant_messages_config",
]
