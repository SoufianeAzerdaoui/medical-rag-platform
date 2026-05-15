from __future__ import annotations

import importlib
import os
import sys
import unittest
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestUserHistoryIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db_path = Path("/tmp") / f"medical_rag_auth_test_{uuid4().hex}.sqlite3"
        os.environ["APP_DB_PATH"] = str(cls.db_path)

        if "backend_api" in sys.modules:
            del sys.modules["backend_api"]

        cls.backend = importlib.import_module("backend_api")

        def fake_run_generation(**kwargs):
            message = str(kwargs.get("query") or "")
            return {
                "answer": f"assistant::{message}",
                "sources": [],
                "generation_time_seconds": 0.01,
                "validation": {"validation_status": "pass"},
                "generation_mode": "deterministic_test",
                "debug": {"generation_writer": "professional_fallback"},
                "query_understanding": {"intent": "doc_scoped_results"},
                "structured_evidence_pack": {},
                "displayed_evidences": [],
                "quality_report": None,
                "visualization": None,
                "chart_data": None,
                "patients": None,
                "inventory_view": None,
            }

        cls.backend._run_generation = fake_run_generation

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            if cls.db_path.exists():
                cls.db_path.unlink()
        except Exception:
            pass

    def setUp(self) -> None:
        conn = self.backend._db_connect()
        try:
            conn.execute("DELETE FROM messages")
            conn.execute("DELETE FROM conversation_states")
            conn.execute("DELETE FROM conversations")
            conn.execute("DELETE FROM users")
            conn.commit()
        finally:
            conn.close()
        self.backend._CONVERSATION_STATE.clear()
        self.backend._STATE_STORE.sessions.clear()

    def _register(self, email: str, password: str = "StrongPass123") -> dict:
        payload = self.backend.AuthRegisterRequest(email=email, password=password)
        response = self.backend.auth_register(payload)
        return {
            "token": response.access_token,
            "user": {"id": response.user.id, "email": response.user.email, "created_at": response.user.created_at},
        }

    def test_a_user_a_history_isolated(self) -> None:
        reg_a = self._register("user.a@example.com")
        user_a = reg_a["user"]

        created = self.backend.create_conversation(
            self.backend.ConversationCreateRequest(title="A chat"),
            current_user=user_a,
        )
        conv_id = created.id

        chat_resp = self.backend.chat(
            self.backend.ChatRequest(conversation_id=conv_id, message="montre ACTH du dernier rapport", history=[], mode="general"),
            current_user=user_a,
        )
        self.assertEqual(chat_resp.conversation_id, conv_id)

        listed = self.backend.list_conversations(current_user=user_a)
        ids = [row.id for row in listed]
        self.assertIn(conv_id, ids)

    def test_b_user_b_does_not_see_user_a(self) -> None:
        reg_a = self._register("user.a2@example.com")
        user_a = reg_a["user"]
        created = self.backend.create_conversation(
            self.backend.ConversationCreateRequest(title="A chat"),
            current_user=user_a,
        )

        reg_b = self._register("user.b2@example.com")
        user_b = reg_b["user"]
        listed_b = self.backend.list_conversations(current_user=user_b)
        ids_b = [row.id for row in listed_b]
        self.assertNotIn(created.id, ids_b)

    def test_c_forbidden_cross_user_chat(self) -> None:
        user_a = self._register("user.a3@example.com")["user"]
        user_b = self._register("user.b3@example.com")["user"]

        created = self.backend.create_conversation(
            self.backend.ConversationCreateRequest(title="A private chat"),
            current_user=user_a,
        )
        conv_a = created.id

        with self.assertRaises(HTTPException) as exc:
            self.backend.chat(
                self.backend.ChatRequest(conversation_id=conv_a, message="affiche ça en tableau", history=[], mode="general"),
                current_user=user_b,
            )
        self.assertEqual(exc.exception.status_code, 403)

    def test_e_new_user_empty_history(self) -> None:
        user_new = self._register("user.new@example.com")["user"]
        listed = self.backend.list_conversations(current_user=user_new)
        self.assertEqual(listed, [])


if __name__ == "__main__":
    unittest.main()
