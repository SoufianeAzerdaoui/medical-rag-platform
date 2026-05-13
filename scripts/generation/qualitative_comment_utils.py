from __future__ import annotations

import re
from typing import Any


def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def extract_comment_text_for_subject(subject: str, rows: list[dict[str, Any]]) -> tuple[str | None, dict[str, Any] | None]:
    subj = _norm(subject)
    if not subj or not rows:
        return None, None

    best_row: dict[str, Any] | None = None
    best_text = ""
    best_score = -1
    for row in rows:
        merged = " ".join(
            [
                str(row.get("value_raw") or ""),
                str(row.get("text_for_keyword") or ""),
                str(row.get("text_for_embedding") or ""),
            ]
        ).strip()
        if not merged:
            continue
        low = _norm(merged)
        if subj not in low:
            continue
        score = 0
        if "commentaire" in low:
            score += 3
        if "valeur seuil" in low:
            score += 3
        if "attention" in low:
            score += 2
        if "troponine" in low:
            score += 2
        score += min(len(merged) // 120, 4)
        if score > best_score:
            best_score = score
            best_row = row
            best_text = merged

    if not best_row or not best_text:
        return None, None

    txt = best_text
    low = _norm(txt)
    start = 0
    for marker in ["commentaire", "valeur seuil", "attention"]:
        pos = low.find(marker)
        if pos >= 0:
            start = pos
            break
    trimmed = txt[start:].strip() if start > 0 else txt.strip()
    trimmed = re.sub(r"\s+", " ", trimmed)
    if len(trimmed) > 1200:
        trimmed = trimmed[:1197].rstrip() + "..."
    return trimmed, best_row


def build_qualitative_comment_answer(*, subject: str, comment_text: str, source_label: str) -> str:
    sub = (subject or "Commentaire médical").strip()
    comment = (comment_text or "").strip()
    source = (source_label or "source non disponible").strip()
    if not comment:
        return "Aucun commentaire troponine exploitable n’a été retrouvé dans les données indexées."
    return (
        f"Voici le commentaire retrouvé sur la {sub.lower()} :\n\n"
        f"{comment}\n\n"
        f"Source : {source}."
    )


def build_sourced_comment_block(*, subject: str, comment_text: str, source_label: str) -> str:
    sub = (subject or "Commentaire médical").strip()
    comment = (comment_text or "").strip()
    source = (source_label or "source non disponible").strip()
    if not comment:
        return (
            "Je n’ai pas de commentaire médical qualitatif récent à afficher sous cette forme. "
            "Demandez d’abord le commentaire concerné (par exemple la troponine)."
        )
    return (
        "Bloc commentaire sourcé\n"
        f"Sujet : {sub}\n"
        "Commentaire :\n"
        f"{comment}\n"
        f"Source : {source}"
    )
