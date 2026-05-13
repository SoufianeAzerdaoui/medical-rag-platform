from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from conversation_state_utils import ConversationState, migrate_conversation_state


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
    def expire(self, conversation_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def acquire_lock(self, conversation_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def release_lock(self, conversation_id: str) -> None:
        raise NotImplementedError


class InMemoryConversationStateStore(ConversationStateStore):
    def __init__(self, backing: dict[str, dict[str, Any]] | None = None) -> None:
        self._data: dict[str, dict[str, Any]] = backing if backing is not None else {}
        self._locks: set[str] = set()

    @property
    def data(self) -> dict[str, dict[str, Any]]:
        return self._data

    def load(self, conversation_id: str) -> ConversationState:
        raw = self._data.get(conversation_id) or {}
        migrated = migrate_conversation_state(raw, conversation_id=conversation_id)
        self._data[conversation_id] = migrated
        return migrated

    def save(self, conversation_id: str, state: dict[str, Any]) -> None:
        self._data[conversation_id] = migrate_conversation_state(state, conversation_id=conversation_id)

    def delete(self, conversation_id: str) -> None:
        self._data.pop(conversation_id, None)
        self._locks.discard(conversation_id)

    def expire(self, conversation_id: str) -> None:
        self.delete(conversation_id)

    def acquire_lock(self, conversation_id: str) -> bool:
        if conversation_id in self._locks:
            return False
        self._locks.add(conversation_id)
        return True

    def release_lock(self, conversation_id: str) -> None:
        self._locks.discard(conversation_id)

