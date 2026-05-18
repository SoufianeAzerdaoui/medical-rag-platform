from __future__ import annotations

import sys
import unittest
from pathlib import Path
from uuid import uuid4

try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover - optional dependency in some envs
    TestClient = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
GENERATION_ROOT = SCRIPTS_ROOT / "generation"
for root in (PROJECT_ROOT, SCRIPTS_ROOT, GENERATION_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


class TestApiDocSummaryAnomaliesValidation(unittest.TestCase):
    def test_chat_anomalies_report16_not_fail_when_facts_correct(self) -> None:
        if TestClient is None:
            self.skipTest("fastapi/testclient non disponible")

        import backend_api

        client = TestClient(backend_api.app)
        user = {"id": "ut-user-anomalies", "email": "ut-anomalies@example.com"}

        backend_api.app.dependency_overrides[backend_api.get_current_user] = lambda: user
        try:
            conv_id = f"conv_ut_anomalies_{uuid4().hex[:10]}"
            backend_api.conversation_service.create_conversation_record(
                user_id=str(user["id"]),
                title="UT anomalies report16",
                conversation_id=conv_id,
            )

            response = client.post(
                "/chat",
                json={
                    "conversation_id": conv_id,
                    "message": "Résume uniquement les anomalies biologiques du report (16), sans poser de diagnostic.",
                    "history": [],
                    "mode": "general",
                },
            )

            self.assertEqual(response.status_code, 200, msg=response.text)
            payload = response.json()

            self.assertTrue(str(payload.get("answer") or "").strip())
            self.assertIn(payload.get("validation_status"), {"pass", "warning"})
            self.assertNotEqual(payload.get("validation_status"), "fail")
        finally:
            backend_api.app.dependency_overrides.clear()


if __name__ == "__main__":
    unittest.main()

