from __future__ import annotations

import os
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Any

from conversation_state_utils import ConversationState, create_empty_conversation_state, migrate_conversation_state


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class ConversationStateStore(ABC):
    @abstractmethod
    def load(self, conversation_id: str) -> ConversationState:
        raise NotImplementedError

    @abstractmethod
    def save(self, conversation_id: str, state: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, conversation_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def touch(self, conversation_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def cleanup_expired(self) -> int:
        raise NotImplementedError


class InMemoryConversationStateStore(ConversationStateStore):
    def __init__(
        self,
        backing: dict[str, dict[str, Any]] | None = None,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}
        self._legacy_state: dict[str, dict[str, Any]] = backing if backing is not None else {}
        ttl_env = os.getenv("CONVERSATION_TTL_SECONDS")
        default_ttl = 60 * 60
        if ttl_seconds is not None:
            self._ttl_seconds = max(60, int(ttl_seconds))
        elif ttl_env and ttl_env.strip().isdigit():
            self._ttl_seconds = max(60, int(ttl_env.strip()))
        else:
            self._ttl_seconds = default_ttl
        self._locks: set[str] = set()

    @property
    def sessions(self) -> dict[str, dict[str, Any]]:
        return self._sessions

    @property
    def data(self) -> dict[str, dict[str, Any]]:
        return self._legacy_state

    @property
    def ttl_seconds(self) -> int:
        return self._ttl_seconds

    def _new_record(self, conversation_id: str, state: dict[str, Any]) -> dict[str, Any]:
        now = _utc_now()
        expires = now + timedelta(seconds=self._ttl_seconds)
        migrated = migrate_conversation_state(state, conversation_id=conversation_id)
        return {
            "state": migrated,
            "created_at": _iso(now),
            "updated_at": _iso(now),
            "expires_at": _iso(expires),
        }

    def _sync_legacy(self, conversation_id: str, state: dict[str, Any]) -> None:
        self._legacy_state[conversation_id] = state

    def _is_expired(self, record: dict[str, Any], now: datetime) -> bool:
        expires_at = _parse_iso(str(record.get("expires_at") or ""))
        if expires_at is None:
            return False
        return expires_at <= now

    def _hydrate_from_legacy(self, conversation_id: str) -> dict[str, Any] | None:
        raw = self._legacy_state.get(conversation_id)
        if not isinstance(raw, dict) or not raw:
            return None
        if isinstance(raw.get("state"), dict):
            legacy_state = raw.get("state") or {}
        else:
            legacy_state = raw
        record = self._new_record(conversation_id, legacy_state)
        self._sessions[conversation_id] = record
        self._sync_legacy(conversation_id, record["state"])
        return record

    def load(self, conversation_id: str) -> ConversationState:
        conv_id = str(conversation_id or "").strip()
        if not conv_id:
            raise ValueError("conversation_id is required")

        now = _utc_now()
        record = self._sessions.get(conv_id)
        if record and self._is_expired(record, now):
            self.delete(conv_id)
            record = None

        if record is None:
            record = self._hydrate_from_legacy(conv_id)

        if record is None:
            record = self._new_record(conv_id, create_empty_conversation_state(conv_id))
            self._sessions[conv_id] = record

        state = migrate_conversation_state(record.get("state") if isinstance(record.get("state"), dict) else {}, conversation_id=conv_id)
        record["state"] = state
        self.touch(conv_id)
        self._sync_legacy(conv_id, state)
        return state

    def save(self, conversation_id: str, state: dict[str, Any]) -> None:
        conv_id = str(conversation_id or "").strip()
        if not conv_id:
            raise ValueError("conversation_id is required")

        now = _utc_now()
        migrated = migrate_conversation_state(state, conversation_id=conv_id)
        existing = self._sessions.get(conv_id)
        created_at = _iso(now)
        if existing and str(existing.get("created_at") or "").strip():
            created_at = str(existing.get("created_at"))

        self._sessions[conv_id] = {
            "state": migrated,
            "created_at": created_at,
            "updated_at": _iso(now),
            "expires_at": _iso(now + timedelta(seconds=self._ttl_seconds)),
        }
        self._sync_legacy(conv_id, migrated)

    def delete(self, conversation_id: str) -> None:
        conv_id = str(conversation_id or "").strip()
        if not conv_id:
            return
        self._sessions.pop(conv_id, None)
        self._legacy_state.pop(conv_id, None)
        self._locks.discard(conv_id)

    def touch(self, conversation_id: str) -> None:
        conv_id = str(conversation_id or "").strip()
        if not conv_id:
            return
        record = self._sessions.get(conv_id)
        if record is None:
            record = self._new_record(conv_id, create_empty_conversation_state(conv_id))
            self._sessions[conv_id] = record
            self._sync_legacy(conv_id, record["state"])
            return
        now = _utc_now()
        record["updated_at"] = _iso(now)
        record["expires_at"] = _iso(now + timedelta(seconds=self._ttl_seconds))

    def cleanup_expired(self) -> int:
        now = _utc_now()
        expired_ids = [conv_id for conv_id, record in self._sessions.items() if self._is_expired(record, now)]
        for conv_id in expired_ids:
            self.delete(conv_id)
        return len(expired_ids)

    # Backward-compatible helpers used by existing callers.
    def expire(self, conversation_id: str) -> None:
        self.delete(conversation_id)

    def acquire_lock(self, conversation_id: str) -> bool:
        conv_id = str(conversation_id or "").strip()
        if not conv_id:
            return False
        if conv_id in self._locks:
            return False
        self._locks.add(conv_id)
        return True

    def release_lock(self, conversation_id: str) -> None:
        self._locks.discard(str(conversation_id or "").strip())
