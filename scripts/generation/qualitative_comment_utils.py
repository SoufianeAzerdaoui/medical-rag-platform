from __future__ import annotations

import re
import unicodedata
from typing import Any

try:
    from source_normalization import dedup_normalized_sources
except Exception:  # pragma: no cover
    from scripts.generation.source_normalization import dedup_normalized_sources

def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _norm_no_accents(s: str) -> str:
    low = (s or "").strip().lower()
    low = unicodedata.normalize("NFKD", low)
    low = "".join(ch for ch in low if not unicodedata.combining(ch))
    return " ".join(low.split())


def clean_qualitative_comment_text(raw_text: str, subject: str) -> str:
    txt = str(raw_text or "").strip()
    if not txt:
        return ""

    # Keep the useful medical part first.
    low = _norm_no_accents(txt)
    start_idx = 0
    for marker in ["valeur seuil", "attention", "commentaire", _norm_no_accents(subject)]:
        pos = low.find(marker)
        if pos >= 0:
            start_idx = pos
            break
    if start_idx > 0:
        txt = txt[start_idx:]

    # Remove index/debug metadata fragments.
    metadata_patterns = [
        r"\bqualitative\b",
        r"\bcomplete qualitative\b",
        r"\bunknown\b",
        r"valeurs?\s+de\s+r[eé]f[eé]rence\s+brutes?.*",
        r"sexe\s*:\s*[^\n]+",
        r"[aâ]ge\s+calcul[eé]\s*:\s*[^\n]+",
        r"type\s+de\s+pr[eé]l[eè]vement\s*:\s*[^\n]+",
        r"date\s+d[’']observation\s*:\s*[^\n]+",
        r"section\s*:\s*[^\n]+",
        r"r[eé]sultats?\s+biologiques?.*",
        r"resultat\s+de\s+laboratoire\s*:\s*",
        r"commentaire\s*=\s*",
    ]
    for patt in metadata_patterns:
        txt = re.sub(patt, " ", txt, flags=re.IGNORECASE)

    txt = re.sub(r"\bcommentaire\s+commentaire\b", "Commentaire", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\s+", " ", txt).strip(" .;:")

    # Deduplicate repeated accent/non-accent versions by sentence normalization.
    chunks = re.split(r"(?<=[\.\!\?])\s+", txt)
    deduped: list[str] = []
    seen: set[str] = set()
    for c in chunks:
        c = c.strip()
        if not c:
            continue
        key = _norm_no_accents(c)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    txt = " ".join(deduped).strip()

    # If a complete version exists, avoid ellipsis-truncated fallback.
    if txt.endswith("...") and len(txt) > 220:
        txt = txt[:-3].rstrip()

    # Gentle readability normalization.
    txt = re.sub(r"\s*:\s*", " : ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


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
        if subj in low:
            score += 2
        score += min(len(merged) // 120, 4)
        if score > best_score:
            best_score = score
            best_row = row
            best_text = merged

    if not best_row or not best_text:
        return None, None

    trimmed = clean_qualitative_comment_text(best_text, subject)
    if len(trimmed) > 1600:
        trimmed = trimmed[:1597].rstrip() + "..."
    return trimmed, best_row


def build_qualitative_comment_answer(*, subject: str, comment_text: str, source_label: str) -> str:
    sub = (subject or "Commentaire médical").strip()
    comment = (comment_text or "").strip()
    source = (source_label or "source non disponible").strip()
    if not comment:
        return f"Aucun commentaire exploitable n’a été retrouvé pour {sub} dans les données indexées."
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
            "Demandez d’abord le commentaire concerné."
        )
    return (
        "Bloc commentaire sourcé\n"
        f"Sujet : {sub}\n"
        "Commentaire :\n"
        f"{comment}\n"
        f"Source : {source}"
    )


def dedup_sources_for_qualitative(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for s in sources or []:
        item = dict(s or {})
        label = str(item.get("label") or "").strip()
        # Normalize inconsistent prefixes such as "docs/report (...).pdf".
        if label.lower().startswith("docs/"):
            item["label"] = label[5:]
        source_pdf = str(item.get("source_pdf") or item.get("filename") or "").strip()
        if source_pdf.lower().startswith("docs/"):
            item["source_pdf"] = source_pdf[5:]
        if item.get("line") is None and item.get("row") is not None:
            item["line"] = item.get("row")
        cleaned.append(item)
    deduped = dedup_normalized_sources(cleaned)
    # Prefer precise source (page/line) over coarse "PDF-only" duplicate.
    precise_by_pdf = {
        str(s.get("source_pdf") or "").strip().lower()
        for s in deduped
        if str(s.get("source_pdf") or "").strip()
        and (isinstance(s.get("page"), int) or isinstance(s.get("line"), int))
    }
    if not precise_by_pdf:
        return deduped
    filtered: list[dict[str, Any]] = []
    for s in deduped:
        pdf = str(s.get("source_pdf") or "").strip().lower()
        has_precision = isinstance(s.get("page"), int) or isinstance(s.get("line"), int)
        if pdf in precise_by_pdf and not has_precision:
            continue
        filtered.append(s)
    return filtered


def format_clickable_source_markdown(label: str, viewer_url: str | None, source_url: str | None) -> tuple[str, bool]:
    lbl = (label or "source").strip()
    href = (viewer_url or source_url or "").strip()
    if href:
        return f"[{lbl}]({href})", True
    return lbl, False


def escape_markdown_table_cell(text: str) -> str:
    return (text or "").replace("|", "\\|").replace("\n", "<br>")
