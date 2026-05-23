from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATION_ROOT = PROJECT_ROOT / "scripts" / "generation"
if str(GENERATION_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERATION_ROOT))

from general_conversation import (
    detect_general_conversation,
    get_general_conversation_response,
    is_pure_general_conversation,
)
from query_understanding import parse_query_understanding


class TestGeneralConversationConfig(unittest.TestCase):
    def test_bonjour_uses_config_message(self) -> None:
        qu = parse_query_understanding("Bonjour.")
        self.assertEqual(qu.intent, "small_talk")
        ans = get_general_conversation_response("small_talk")
        self.assertTrue("rapports médicaux" in ans or "rapports medicaux" in ans)

    def test_detect_identity_capabilities_and_thanks(self) -> None:
        self.assertEqual(detect_general_conversation("t'es qui ?"), "identity_question")
        self.assertEqual(detect_general_conversation("qu'est-ce que tu peux faire ?"), "capabilities")
        self.assertEqual(detect_general_conversation("Merci"), "thanks")

    def test_mixed_greeting_and_medical_ask_is_not_pure_general_conversation(self) -> None:
        self.assertFalse(is_pure_general_conversation("Bonjour, peux-tu résumer le report 16 ?"))


if __name__ == "__main__":
    unittest.main()
