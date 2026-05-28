from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from backend.database import db_connect, now_iso


def save_message(
    conversation_id: str,
    role: str,
    content: str,
    *,
    sources: list[dict[str, Any]] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    message_id = f"msg_{uuid4()}"
    created_at = now_iso()
    sources_json = json.dumps(list(sources or []), ensure_ascii=False)
    diagnostics_json = json.dumps(dict(diagnostics or {}), ensure_ascii=False) if diagnostics is not None else None

    conn = db_connect()
    try:
        conn.execute(
            """
            INSERT INTO messages (id, conversation_id, role, content, sources_json, diagnostics_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                str(conversation_id),
                str(role),
                str(content),
                sources_json,
                diagnostics_json,
                created_at,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "id": message_id,
        "conversation_id": str(conversation_id),
        "role": str(role),
        "content": str(content),
        "sources": list(sources or []),
        "diagnostics": dict(diagnostics or {}) if diagnostics is not None else None,
        "created_at": created_at,
    }


def list_messages(conversation_id: str) -> list[dict[str, Any]]:
    conn = db_connect()
    try:
        rows = conn.execute(
            """
            SELECT id, conversation_id, role, content, sources_json, diagnostics_json, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY datetime(created_at) ASC
            """,
            (str(conversation_id),),
        ).fetchall()
    finally:
        conn.close()

    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        sources_raw = item.pop("sources_json", None)
        diagnostics_raw = item.pop("diagnostics_json", None)
        try:
            parsed_sources = json.loads(str(sources_raw)) if sources_raw else []
            item["sources"] = parsed_sources if isinstance(parsed_sources, list) else []
        except Exception:
            item["sources"] = []
        try:
            parsed_diagnostics = json.loads(str(diagnostics_raw)) if diagnostics_raw else None
            item["diagnostics"] = parsed_diagnostics if isinstance(parsed_diagnostics, dict) else None
        except Exception:
            item["diagnostics"] = None
        out.append(item)
    return out
