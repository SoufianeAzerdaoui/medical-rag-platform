from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class TestMessageServicePersistence(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.db_path = Path(self._tmpdir.name) / "app_state_test.sqlite3"

        import backend.config as backend_config
        import backend.database as backend_database
        from backend.database import init_schema

        backend_config.APP_DB_PATH = self.db_path
        backend_database.APP_DB_PATH = self.db_path
        init_schema()

    def test_save_and_load_sources_and_diagnostics(self) -> None:
        from backend.database import db_connect, now_iso
        from backend.services import conversation_service
        from backend.services import message_service

        conn = db_connect()
        try:
            conn.execute(
                "INSERT INTO users (id, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                ("u_test_1", "test@example.com", "hash", now_iso()),
            )
            conn.commit()
        finally:
            conn.close()
        conversation_service.create_conversation_record(
            user_id="u_test_1",
            conversation_id="conv_test_1",
            title="Test",
        )

        sources = [
            {
                "id": "source-1",
                "documentName": "report (12).pdf",
                "documentId": "report_12",
                "page": 1,
                "doc_id": "report_12",
                "filename": "report (12).pdf",
                "label": "report (12).pdf — page 1, lignes 1-10",
                "url": "/viewer/pdf?doc_id=report_12&page=1",
                "viewer_url": "/viewer/pdf?doc_id=report_12&page=1",
            }
        ]
        diagnostics = {
            "selected_route": "doc_scoped_biological_summary",
            "generation_mode": "hybrid_structured_llm_writer",
            "validation_status": "pass",
        }

        message_service.save_message(
            "conv_test_1",
            "assistant",
            "Réponse test",
            sources=sources,
            diagnostics=diagnostics,
        )

        rows = message_service.list_messages("conv_test_1")
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["role"], "assistant")
        self.assertEqual(row["content"], "Réponse test")
        self.assertEqual(row.get("sources"), sources)
        self.assertEqual(row.get("diagnostics"), diagnostics)


if __name__ == "__main__":
    unittest.main()
