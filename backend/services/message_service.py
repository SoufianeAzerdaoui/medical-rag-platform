from __future__ import annotations

from typing import Any
from uuid import uuid4

from backend.database import db_connect, now_iso


def save_message(conversation_id: str, role: str, content: str) -> dict[str, Any]:
    message_id = f"msg_{uuid4()}"
    created_at = now_iso()

    conn = db_connect()
    try:
        conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (message_id, str(conversation_id), str(role), str(content), created_at),
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "id": message_id,
        "conversation_id": str(conversation_id),
        "role": str(role),
        "content": str(content),
        "created_at": created_at,
    }


def list_messages(conversation_id: str) -> list[dict[str, Any]]:
    conn = db_connect()
    try:
        rows = conn.execute(
            """
            SELECT id, conversation_id, role, content, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY datetime(created_at) ASC
            """,
            (str(conversation_id),),
        ).fetchall()
    finally:
        conn.close()

    return [dict(row) for row in rows]
