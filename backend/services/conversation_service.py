from __future__ import annotations

from typing import Any
from uuid import uuid4

from fastapi import HTTPException, status

from backend.database import db_connect, now_iso


def create_conversation_record(*, user_id: str, title: str | None = None, conversation_id: str | None = None) -> dict[str, Any]:
    now = now_iso()
    conv_id = str(conversation_id or f"conv_{uuid4()}")
    safe_title = (str(title or "").strip() or "Nouvelle conversation")[:240]
    title_source = "manual" if str(title or "").strip() else "auto"

    conn = db_connect()
    try:
        conn.execute(
            "INSERT INTO conversations (id, user_id, title, title_source, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (conv_id, str(user_id), safe_title, title_source, now, now),
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "id": conv_id,
        "user_id": str(user_id),
        "title": safe_title,
        "created_at": now,
        "updated_at": now,
    }


def list_conversations_for_user(user_id: str) -> list[dict[str, Any]]:
    conn = db_connect()
    try:
        rows = conn.execute(
            """
            SELECT id, user_id, title, created_at, updated_at
            FROM conversations
            WHERE user_id = ?
            ORDER BY datetime(updated_at) DESC
            """,
            (str(user_id),),
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def get_conversation_for_user(conversation_id: str, user_id: str) -> dict[str, Any] | None:
    conn = db_connect()
    try:
        row = conn.execute(
            "SELECT id, user_id, title, created_at, updated_at FROM conversations WHERE id = ? AND user_id = ?",
            (str(conversation_id), str(user_id)),
        ).fetchone()
    finally:
        conn.close()
    return dict(row) if row is not None else None


def get_conversation_any_owner(conversation_id: str) -> dict[str, Any] | None:
    conn = db_connect()
    try:
        row = conn.execute(
            "SELECT id, user_id, title, created_at, updated_at FROM conversations WHERE id = ?",
            (str(conversation_id),),
        ).fetchone()
    finally:
        conn.close()
    return dict(row) if row is not None else None


def require_owned_conversation(conversation_id: str, user_id: str) -> dict[str, Any]:
    conversation = get_conversation_any_owner(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation introuvable")
    if str(conversation.get("user_id") or "") != str(user_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Conversation interdite")
    return conversation


def touch_conversation(conversation_id: str) -> None:
    conn = db_connect()
    try:
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now_iso(), str(conversation_id)),
        )
        conn.commit()
    finally:
        conn.close()


def delete_conversation(conversation_id: str, user_id: str) -> None:
    conn = db_connect()
    try:
        conn.execute(
            "DELETE FROM conversations WHERE id = ? AND user_id = ?",
            (str(conversation_id), str(user_id)),
        )
        conn.commit()
    finally:
        conn.close()
