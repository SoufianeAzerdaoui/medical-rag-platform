from __future__ import annotations

import re
import unicodedata
from typing import Any

from config_loader import get_medical_topics_config


def _norm(value: str) -> str:
    s = str(value or "").strip().lower().replace("µ", "u").replace("_", " ")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s)
    return s


def detect_medical_topic(query: str) -> str | None:
    qn = _norm(query)
    topics = dict((get_medical_topics_config() or {}).get("topics") or {})
    for topic_name, spec in topics.items():
        triggers = [_norm(t) for t in list((spec or {}).get("triggers") or []) if str(t).strip()]
        if any(t and t in qn for t in triggers):
            return str(topic_name)
    return None


def get_topic_analytes(topic: str | None) -> list[str]:
    if not topic:
        return []
    topics = dict((get_medical_topics_config() or {}).get("topics") or {})
    spec = dict(topics.get(str(topic), {}) or {})
    return [str(a).strip().lower() for a in list(spec.get("analytes") or []) if str(a).strip()]


def get_topic_exclusions(topic: str | None) -> list[str]:
    if not topic:
        return []
    topics = dict((get_medical_topics_config() or {}).get("topics") or {})
    spec = dict(topics.get(str(topic), {}) or {})
    return [str(a).strip().lower() for a in list(spec.get("excluded_unless_explicit") or []) if str(a).strip()]


def get_topic_rules(topic: str | None) -> dict[str, Any]:
    if not topic:
        return {}
    topics = dict((get_medical_topics_config() or {}).get("topics") or {})
    spec = dict(topics.get(str(topic), {}) or {})
    rules = spec.get("rules") or {}
    return dict(rules) if isinstance(rules, dict) else {}


__all__ = ["detect_medical_topic", "get_topic_analytes", "get_topic_exclusions", "get_topic_rules"]
