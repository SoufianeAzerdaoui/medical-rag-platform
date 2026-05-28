#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.config import APP_DB_PATH
from backend.database import init_schema


DOC_ID_RE = re.compile(r"\breport[_\s-]?(\d+)\b", re.IGNORECASE)
PAGE_RE = re.compile(r"\bpage[s]?\s*(\d+)(?:\s*[-–]\s*(\d+))?\b", re.IGNORECASE)
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
VIEWER_URL_RE = re.compile(r"(/viewer/pdf\?[^)\s]+)")


def _load_json(raw: str | None, default: Any) -> Any:
    if not raw:
        return default
    try:
        out = json.loads(raw)
        return out if out is not None else default
    except Exception:
        return default


def _normalize_doc_id(raw: str) -> str | None:
    txt = str(raw or "").strip().lower().replace(" ", "_").replace("-", "_")
    if not txt:
        return None
    m = DOC_ID_RE.search(txt)
    if m:
        return f"report_{m.group(1)}"
    if txt.startswith("report_"):
        return txt
    return None


def _extract_doc_id_and_page_from_plain_source(content: str) -> tuple[str | None, int | None]:
    doc_id = None
    m_doc = DOC_ID_RE.search(content or "")
    if m_doc:
        doc_id = f"report_{m_doc.group(1)}"

    page = None
    m_page = PAGE_RE.search(content or "")
    if m_page:
        try:
            page = int(m_page.group(1))
        except Exception:
            page = None
    return doc_id, page


def _source_from_url(label: str, href: str, idx: int) -> dict[str, Any]:
    parsed = urlparse(href)
    query = parse_qs(parsed.query or "")
    doc_id = _normalize_doc_id((query.get("doc_id") or [None])[0] or "")
    page_raw = (query.get("page") or [None])[0]
    page = None
    if page_raw is not None:
        try:
            page = int(str(page_raw))
        except Exception:
            page = None
    safe_label = str(label or "").strip() or (f"{doc_id} — page {page}" if doc_id and page else "Source PDF")
    return {
        "id": f"source-backfill-{idx}",
        "documentName": safe_label,
        "documentId": doc_id,
        "doc_id": doc_id,
        "filename": None,
        "page": page,
        "row": None,
        "label": safe_label,
        "url": href,
        "viewer_url": href if "/viewer/pdf" in href else None,
        "type": "pdf_source",
    }


def _dedup_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for src in sources:
        doc_id = str(src.get("doc_id") or src.get("documentId") or "").strip().lower()
        page = str(src.get("page") or "").strip()
        href = str(src.get("viewer_url") or src.get("url") or "").strip()
        key = (doc_id, page, href)
        if key in seen:
            continue
        seen.add(key)
        out.append(src)
    return out


def _extract_sources_from_message(content: str) -> list[dict[str, Any]]:
    text = str(content or "")
    out: list[dict[str, Any]] = []
    idx = 0

    for m in MD_LINK_RE.finditer(text):
        label = str(m.group(1) or "").strip()
        href = str(m.group(2) or "").strip()
        if not href:
            continue
        if "/viewer/pdf" in href or "http" in href:
            idx += 1
            out.append(_source_from_url(label=label, href=href, idx=idx))

    if not out:
        for m in VIEWER_URL_RE.finditer(text):
            href = str(m.group(1) or "").strip()
            if not href:
                continue
            idx += 1
            out.append(_source_from_url(label="Source PDF", href=href, idx=idx))

    if not out:
        doc_id, page = _extract_doc_id_and_page_from_plain_source(text)
        if doc_id:
            idx += 1
            href = f"/viewer/pdf?doc_id={doc_id}"
            if page:
                href = f"{href}&page={page}"
            out.append(
                {
                    "id": f"source-backfill-{idx}",
                    "documentName": f"{doc_id}{f' — page {page}' if page else ''}",
                    "documentId": doc_id,
                    "doc_id": doc_id,
                    "filename": None,
                    "page": page,
                    "row": None,
                    "label": f"{doc_id}{f' — page {page}' if page else ''}",
                    "url": href,
                    "viewer_url": href,
                    "type": "pdf_source",
                }
            )
    return _dedup_sources(out)


def _infer_diagnostics(content: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    text = str(content or "")
    n = text.lower()
    out: dict[str, Any] = {
        "backfilled": True,
        "backfilled_version": "v1",
    }

    if text.strip().lower().startswith("note de synthèse médicale") or text.strip().lower().startswith("note médicale"):
        out["intent"] = "doc_scoped_summary"
        out["selected_route"] = "doc_scoped_biological_summary"
        out["generation_mode"] = "hybrid_structured_llm_writer"
        out["generation_writer"] = "llm_writer"
    elif "anormaux :" in n and "conclusion technique" in n:
        out["intent"] = "doc_scoped_summary"
        out["selected_route"] = "doc_scoped_biological_summary"
        out["generation_mode"] = "hybrid_structured_llm_writer"
        out["generation_writer"] = "llm_writer"
    elif "je ne peux pas poser ni évoquer un diagnostic" in n:
        out["intent"] = "doc_scoped_medical_interpretation_guarded"
        out["selected_route"] = "doc_scoped_medical_interpretation_guarded"
        out["generation_mode"] = "deterministic_guarded_medical_interpretation"
        out["generation_writer"] = "professional_fallback"

    requested_doc_ids: list[str] = []
    for src in sources:
        doc_id = _normalize_doc_id(str(src.get("doc_id") or src.get("documentId") or ""))
        if doc_id and doc_id not in requested_doc_ids:
            requested_doc_ids.append(doc_id)
    if requested_doc_ids:
        out["requested_doc_ids"] = requested_doc_ids
    return out


def _needs_sources_backfill(existing_sources: Any) -> bool:
    if not isinstance(existing_sources, list) or not existing_sources:
        return True
    for src in existing_sources:
        if not isinstance(src, dict):
            continue
        if str(src.get("viewer_url") or src.get("url") or "").strip():
            return False
    return True


def _needs_diag_backfill(existing_diag: Any) -> bool:
    if not isinstance(existing_diag, dict) or not existing_diag:
        return True
    key_fields = ("selected_route", "intent", "generation_mode")
    return not any(str(existing_diag.get(k) or "").strip() for k in key_fields)


def run_backfill(
    *,
    db_path: Path,
    apply_changes: bool,
    conversation_id: str | None,
    limit: int | None,
    only_missing: bool,
) -> int:
    init_schema()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    sql = """
        SELECT id, conversation_id, role, content, sources_json, diagnostics_json
        FROM messages
        WHERE role = 'assistant'
    """
    params: list[Any] = []
    if conversation_id:
        sql += " AND conversation_id = ?"
        params.append(str(conversation_id))
    sql += " ORDER BY datetime(created_at) ASC"
    if limit and limit > 0:
        sql += " LIMIT ?"
        params.append(int(limit))

    rows = conn.execute(sql, tuple(params)).fetchall()
    processed = 0
    candidate_messages = 0
    updated = 0
    source_updates = 0
    diag_updates = 0
    sample_logs = 0

    try:
        for row in rows:
            processed += 1
            msg_id = str(row["id"])
            content = str(row["content"] or "")
            existing_sources = _load_json(row["sources_json"], [])
            existing_diag = _load_json(row["diagnostics_json"], {})
            need_sources = _needs_sources_backfill(existing_sources)
            need_diag = _needs_diag_backfill(existing_diag)
            if not need_sources and not need_diag:
                continue
            candidate_messages += 1

            inferred_sources = _extract_sources_from_message(content) if need_sources else list(existing_sources or [])
            inferred_diag = _infer_diagnostics(content, inferred_sources) if need_diag else dict(existing_diag or {})

            next_sources = list(existing_sources or [])
            sources_changed = False
            if need_sources and inferred_sources:
                normalized_existing_sources = json.dumps(existing_sources or [], ensure_ascii=False, sort_keys=True)
                normalized_inferred_sources = json.dumps(inferred_sources or [], ensure_ascii=False, sort_keys=True)
                if normalized_existing_sources != normalized_inferred_sources:
                    next_sources = inferred_sources
                    sources_changed = True
            next_diag = dict(existing_diag or {})
            diag_changed = False
            if need_diag:
                for key, value in inferred_diag.items():
                    if key not in next_diag or next_diag.get(key) in (None, "", [], {}):
                        next_diag[key] = value
                        diag_changed = True

            if only_missing:
                # Guardrail mode: never overwrite existing non-empty values.
                pass
            else:
                # In non-strict mode, allow replacing stale empty containers with inferred values.
                if need_sources and inferred_sources and not sources_changed:
                    if not isinstance(existing_sources, list) or not existing_sources:
                        next_sources = inferred_sources
                        sources_changed = True

            if not sources_changed and not diag_changed:
                continue

            if sources_changed:
                source_updates += 1
            if diag_changed:
                diag_updates += 1

            if sample_logs < 10:
                print(
                    f"[sample] id={msg_id} conv={row['conversation_id']} "
                    f"sources:{len(existing_sources or [])}->{len(next_sources or [])} "
                    f"diag_keys:{len((existing_diag or {}).keys()) if isinstance(existing_diag, dict) else 0}->{len(next_diag.keys())}"
                )
                sample_logs += 1

            if apply_changes:
                conn.execute(
                    """
                    UPDATE messages
                    SET sources_json = ?, diagnostics_json = ?
                    WHERE id = ?
                    """,
                    (
                        json.dumps(next_sources, ensure_ascii=False),
                        json.dumps(next_diag, ensure_ascii=False),
                        msg_id,
                    ),
                )
            updated += 1

        if apply_changes:
            conn.commit()
    finally:
        conn.close()

    print(
        json.dumps(
            {
                "db_path": str(db_path),
                "conversation_id": conversation_id,
                "apply": apply_changes,
                "processed_messages": processed,
                "candidate_messages": candidate_messages,
                "updated_messages": updated,
                "sources_updated_messages": source_updates,
                "diagnostics_updated_messages": diag_updates,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Best-effort backfill for assistant messages (sources_json + diagnostics_json) from stored content."
    )
    parser.add_argument(
        "--db-path",
        default=str(APP_DB_PATH),
        help="Path to app_state sqlite database (default: backend.config.APP_DB_PATH).",
    )
    parser.add_argument(
        "--conversation-id",
        default=None,
        help="Optional conversation id filter.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max messages to process.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply updates. Without this flag, script runs as dry-run (no DB writes).",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Only fill missing metadata fields (default behavior is best-effort fill as well).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    db_path = Path(str(args.db_path)).resolve()
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return 1
    return run_backfill(
        db_path=db_path,
        apply_changes=bool(args.apply),
        conversation_id=args.conversation_id,
        limit=args.limit,
        only_missing=bool(args.only_missing),
    )


if __name__ == "__main__":
    raise SystemExit(main())
