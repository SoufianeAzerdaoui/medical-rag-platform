from __future__ import annotations

import json
from typing import Any

from backend.database import db_connect, now_iso
from backend.services import security_service


def log_event(
    *,
    event_type: str,
    actor_user_id: str | None,
    actor_email: str | None,
    target_type: str | None = None,
    target_id: str | None = None,
    status: str = "success",
    payload: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
) -> None:
    conn = db_connect()
    try:
        conn.execute(
            """
            INSERT INTO audit_events (
                event_type,
                actor_user_id,
                actor_email,
                target_type,
                target_id,
                status,
                payload_json,
                result_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(event_type),
                str(actor_user_id or "") or None,
                str(actor_email or "") or None,
                str(target_type or "") or None,
                str(target_id or "") or None,
                str(status),
                security_service.encrypt_json(payload or {}),
                security_service.encrypt_json(result or {}),
                now_iso(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def list_events(*, target_type: str | None = None, target_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
    safe_limit = max(1, min(int(limit), 1000))
    where: list[str] = []
    params: list[Any] = []
    if target_type:
        where.append("target_type = ?")
        params.append(str(target_type))
    if target_id:
        where.append("target_id = ?")
        params.append(str(target_id))
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    conn = db_connect()
    try:
        rows = conn.execute(
            f"""
            SELECT
                id,
                event_type,
                actor_user_id,
                actor_email,
                target_type,
                target_id,
                status,
                payload_json,
                result_json,
                created_at
            FROM audit_events
            {where_sql}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (*params, safe_limit),
        ).fetchall()
    finally:
        conn.close()

    out: list[dict[str, Any]] = []
    for row in rows:
        payload_raw = str(row["payload_json"] or "")
        result_raw = str(row["result_json"] or "")
        out.append(
            {
                "id": int(row["id"]),
                "event_type": str(row["event_type"] or ""),
                "actor_user_id": str(row["actor_user_id"] or ""),
                "actor_email": str(row["actor_email"] or ""),
                "target_type": str(row["target_type"] or ""),
                "target_id": str(row["target_id"] or ""),
                "status": str(row["status"] or ""),
                "payload": security_service.decrypt_json(payload_raw),
                "result": security_service.decrypt_json(result_raw),
                "created_at": str(row["created_at"] or ""),
            }
        )
    return out
