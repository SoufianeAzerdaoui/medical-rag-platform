from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from backend.database import db_connect, now_iso
from scripts.generation.conversation_state_store import InMemoryConversationStateStore
from scripts.generation.conversation_state_utils import (
    evidence_pack_is_transformable,
    get_transformable_context,
    update_conversation_state,
)


class ConversationStateService:
    def __init__(self) -> None:
        self._legacy_state: dict[str, dict[str, Any]] = defaultdict(dict)
        self._store = InMemoryConversationStateStore(self._legacy_state)

    @property
    def legacy_state(self) -> dict[str, dict[str, Any]]:
        return self._legacy_state

    @property
    def memory_store(self) -> InMemoryConversationStateStore:
        return self._store

    def cleanup_expired(self) -> int:
        return self._store.cleanup_expired()

    def load(self, conversation_id: str) -> dict[str, Any]:
        state = self._store.load(conversation_id)
        self._legacy_state[conversation_id] = state
        return state

    def save(self, conversation_id: str, state: dict[str, Any]) -> None:
        self._store.save(conversation_id, state)
        self._legacy_state[conversation_id] = self._store.load(conversation_id)

    def delete(self, conversation_id: str) -> None:
        self._store.delete(conversation_id)
        self._legacy_state.pop(str(conversation_id), None)

    def load_from_db(self, conversation_id: str) -> dict[str, Any] | None:
        conn = db_connect()
        try:
            row = conn.execute(
                "SELECT state_json FROM conversation_states WHERE conversation_id = ?",
                (str(conversation_id),),
            ).fetchone()
        finally:
            conn.close()

        if row is None:
            return None

        try:
            loaded = json.loads(str(row["state_json"]))
        except Exception:
            return None
        return loaded if isinstance(loaded, dict) else None

    def save_to_db(self, conversation_id: str, state: dict[str, Any]) -> None:
        state_json = json.dumps(state, ensure_ascii=False)
        updated_at = now_iso()

        conn = db_connect()
        try:
            conn.execute(
                """
                INSERT INTO conversation_states (conversation_id, state_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(conversation_id)
                DO UPDATE SET state_json=excluded.state_json, updated_at=excluded.updated_at
                """,
                (str(conversation_id), state_json, updated_at),
            )
            conn.commit()
        finally:
            conn.close()

    def delete_db_state(self, conversation_id: str) -> None:
        conn = db_connect()
        try:
            conn.execute("DELETE FROM conversation_states WHERE conversation_id = ?", (str(conversation_id),))
            conn.commit()
        finally:
            conn.close()
        self.delete(conversation_id)

    def hydrate_from_db_if_present(self, conversation_id: str) -> None:
        persisted_state = self.load_from_db(conversation_id)
        if isinstance(persisted_state, dict) and persisted_state:
            self.save(conversation_id, persisted_state)

    def update_from_generation(self, *, conversation_id: str, state: dict[str, Any], generation: dict[str, Any], user_message: str) -> None:
        update_conversation_state(
            state_store=self._legacy_state,
            chat_id=conversation_id,
            state=state,
            generation=generation,
            user_message=user_message,
        )
        self.save(conversation_id, self._legacy_state.get(conversation_id) or {})


def evidence_pack_transformable(pack: Any) -> bool:
    return evidence_pack_is_transformable(pack)


def transformable_context(state: dict[str, Any], *, requested_doc_ids: list[str] | None = None) -> dict[str, Any] | None:
    return get_transformable_context(state, requested_doc_ids=requested_doc_ids)
