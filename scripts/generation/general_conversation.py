from __future__ import annotations

import re
import unicodedata

from config_loader import get_assistant_messages_config


def _norm(value: str) -> str:
    s = str(value or "").strip().lower().replace("µ", "u")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s)
    return s


def get_assistant_message(key: str) -> str:
    cfg = dict((get_assistant_messages_config() or {}).get("general_conversation") or {})
    if key in cfg and str(cfg.get(key) or "").strip():
        return str(cfg.get(key)).strip()
    return str(cfg.get("fallback") or "Je peux vous aider à interroger les rapports médicaux indexés.")


def get_general_conversation_response(intent: str) -> str:
    intent_key = str(intent or "").strip().lower()
    if intent_key in {"capability_question", "capabilities"}:
        return get_assistant_message("capabilities")
    if intent_key in {"help_question", "identity_question", "small_talk"}:
        return get_assistant_message(intent_key)
    return get_assistant_message("fallback")


def detect_general_conversation(query: str) -> str | None:
    qn = _norm(query)
    if any(m in qn for m in ["t es qui", "t'es qui", "tes qui", "qui es tu", "qui es-tu", "tu es qui"]):
        return "identity_question"
    if any(m in qn for m in ["tu peux faire quoi", "que peux tu faire", "qu est ce que tu peux faire", "qu'est-ce que tu peux faire"]):
        return "capabilities"
    if any(m in qn for m in ["aide moi", "aide-moi", "help", "comment tu fonctionnes"]):
        return "help_question"
    if any(m in qn for m in ["merci", "thanks", "thank you"]):
        return "thanks"
    if any(m in qn for m in ["bonjour", "bonsoir", "salut", "merci", "ok"]):
        return "small_talk"
    return None


def is_pure_general_conversation(query: str) -> bool:
    qn = _norm(query)
    if not qn:
        return False
    # Mixed asks must stay medical (ex: "Bonjour, résume le report 16").
    medical_markers = [
        "report",
        "rapport",
        "resultat",
        "résultat",
        "anomal",
        "reference",
        "référence",
        "doc_scoped",
        "insuline",
        "crp",
        "tsh",
        "glucose",
        "resumer",
        "résumer",
        "resume",
        "résume",
        "comparer",
        "compare",
    ]
    if any(m in qn for m in medical_markers):
        return False
    return detect_general_conversation(query) is not None


def render_general_conversation_response(kind: str) -> str:
    key = str(kind or "").strip().lower()
    if key in {"capability_question", "capabilities"}:
        return get_assistant_message("capabilities")
    if key == "thanks":
        return get_assistant_message("fallback")
    if key in {"small_talk", "identity_question", "help_question"}:
        return get_assistant_message(key)
    return get_assistant_message("fallback")


__all__ = [
    "detect_general_conversation",
    "get_assistant_message",
    "get_general_conversation_response",
    "is_pure_general_conversation",
    "render_general_conversation_response",
]
