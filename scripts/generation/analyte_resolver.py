from __future__ import annotations

import re
import sqlite3
import unicodedata
from typing import Any

try:
    from analyte_aliases import ANALYTE_ALIAS_GROUPS
except Exception:  # pragma: no cover
    from scripts.generation.analyte_aliases import ANALYTE_ALIAS_GROUPS  # type: ignore


def normalize_analyte_text(text: str) -> str:
    s = str(text or "").strip().lower()
    s = s.replace("µ", "u")
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    s = s.replace("/", " ").replace("'", " ").replace("’", " ")
    s = s.replace("(", " ").replace(")", " ")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    # Keep letters, digits, spaces and hyphen for analytes like CA 15-3.
    s = re.sub(r"[^a-z0-9\-\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _to_norm_key(text: str) -> str:
    return normalize_analyte_text(text).replace("-", " ").replace(" ", "_")


def load_available_analytes(index_dir_or_db_path: str) -> list[dict[str, str]]:
    path = str(index_dir_or_db_path or "")
    if not path:
        return []
    if not path.endswith(".sqlite"):
        path = path.rstrip("/") + "/medical_rag.sqlite"
    try:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT DISTINCT
              lower(trim(coalesce(m.analyte_norm,''))) AS analyte_norm,
              trim(coalesce(m.analyte,'')) AS analyte
            FROM metadata_chunks m
            JOIN chunks c ON c.chunk_id = m.chunk_id
            WHERE c.chunk_type='lab_result'
            """
        )
        rows = [dict(r) for r in cur.fetchall()]
    except Exception:
        rows = []
    finally:
        try:
            conn.close()  # type: ignore[name-defined]
        except Exception:
            pass

    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for r in rows:
        norm_val = str(r.get("analyte_norm") or "").strip().lower()
        disp = str(r.get("analyte") or "").strip()
        if not norm_val and disp:
            norm_val = _to_norm_key(disp)
        if not norm_val or norm_val in seen:
            continue
        seen.add(norm_val)
        out.append({"display_name": disp or norm_val.replace("_", " ").upper(), "analyte_norm": norm_val})
    return out


def _token_set(text: str) -> set[str]:
    return {t for t in normalize_analyte_text(text).split() if t}


def resolve_requested_analytes(
    query: str,
    available_analytes: list[dict] | None = None,
    aliases: dict | None = None,
    max_candidates: int = 5,
) -> list[dict[str, Any]]:
    q_norm = normalize_analyte_text(query)
    if not q_norm:
        return []
    q_parts = [q_norm]
    # Parentheses split and punctuation variants are already normalized; include compact as helper.
    q_compact = q_norm.replace(" ", "")
    if q_compact and q_compact not in q_parts:
        q_parts.append(q_compact)

    alias_groups: dict[str, list[str]] = dict(ANALYTE_ALIAS_GROUPS)
    if isinstance(aliases, dict):
        for k, vals in aliases.items():
            kk = str(k).strip().lower()
            if not kk:
                continue
            existing = alias_groups.get(kk, [])
            merged = list(existing) + [str(v) for v in (vals or [])]
            alias_groups[kk] = merged

    dynamic = list(available_analytes or [])
    canonical_pool: dict[str, str] = {}
    for item in dynamic:
        n = str((item or {}).get("analyte_norm") or "").strip().lower()
        d = str((item or {}).get("display_name") or n).strip()
        if n:
            canonical_pool.setdefault(n, d)
    for k in alias_groups.keys():
        canonical_pool.setdefault(str(k).strip().lower(), str(k).replace("_", " ").upper())

    candidates: list[dict[str, Any]] = []
    for canon, display in canonical_pool.items():
        canon_norm = normalize_analyte_text(canon.replace("_", " "))
        all_aliases = [canon_norm] + [normalize_analyte_text(a) for a in alias_groups.get(canon, [])]
        all_aliases = [a for a in all_aliases if a]
        best_score = 0.0
        best_reason = ""
        best_match = ""
        for a in all_aliases:
            # Priority 1/2 exact contains on normalized phrase
            if re.search(rf"(?<![a-z0-9]){re.escape(a)}(?![a-z0-9])", q_norm):
                token_count = len([t for t in a.split() if t])
                if a == canon_norm:
                    if token_count == 1 and len(a) <= 4:
                        score = 0.94
                    else:
                        score = 0.99
                    reason = "exact_analyte_norm"
                else:
                    # Penalize very short generic aliases (e.g. "tsh") to avoid overpowering
                    # more specific aliases like "tsh ultrasensible".
                    if token_count == 1 and len(a) <= 4:
                        score = 0.93
                    else:
                        score = 0.985
                    reason = "alias_exact"
            elif (" " in a or "-" in a) and a.replace(" ", "") in q_compact and len(a) >= 3:
                score = 0.95
                reason = "alias_parentheses_exact"
            else:
                # Priority 4 token overlap controlled
                qa = _token_set(a)
                qq = _token_set(q_norm)
                inter = len(qa & qq)
                union = len(qa | qq) or 1
                jaccard = inter / union
                if inter and jaccard >= 0.6:
                    score = 0.90
                    reason = "token_overlap_controlled"
                else:
                    score = 0.0
                    reason = ""
            if score > best_score:
                best_score = score
                best_reason = reason
                best_match = a
        if best_score >= 0.90:
            candidates.append(
                {
                    "display_name": display,
                    "analyte_norm": canon,
                    "matched_text": best_match,
                    "confidence": round(best_score, 4),
                    "match_reason": best_reason,
                }
            )

    candidates.sort(key=lambda x: (-float(x.get("confidence") or 0), str(x.get("analyte_norm") or "")))
    top = candidates[: max(1, int(max_candidates or 5))]
    if not top:
        return []
    # Ambiguity guard: if top two are close and not exact-high, keep both as candidates for caller policy.
    if len(top) >= 2:
        c1 = float(top[0].get("confidence") or 0.0)
        c2 = float(top[1].get("confidence") or 0.0)
        if c1 < 0.96 and (c1 - c2) < 0.03:
            for item in top:
                item["status"] = "ambiguous"
                item["candidates"] = [str(t.get("display_name") or t.get("analyte_norm") or "") for t in top]
            return top
    top[0]["status"] = "selected"
    top[0]["candidates"] = [str(top[0].get("display_name") or top[0].get("analyte_norm") or "")]
    return top
