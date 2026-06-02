from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATION_DIR = PROJECT_ROOT / "scripts" / "generation"
if str(GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATION_DIR))

from llm_client import LLMClient


class TestLlmClientStructuredMessages(unittest.TestCase):
    def test_build_message_payload_separates_system_and_user(self) -> None:
        messages, preview = LLMClient._build_message_payload(  # type: ignore[attr-defined]
            prompt=None,
            system_prompt="System rules",
            user_prompt="User question",
            messages=None,
        )
        self.assertEqual(messages, [{"role": "system", "content": "System rules"}, {"role": "user", "content": "User question"}])
        self.assertIn("[system] System rules", preview)
        self.assertIn("[user] User question", preview)

    def test_build_message_payload_prefers_explicit_messages(self) -> None:
        messages, preview = LLMClient._build_message_payload(  # type: ignore[attr-defined]
            prompt="ignored",
            system_prompt="ignored",
            user_prompt="ignored",
            messages=[
                {"role": "system", "content": "Keep this"},
                {"role": "assistant", "content": "History"},
                {"role": "user", "content": "Question"},
            ],
        )
        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "Keep this"},
                {"role": "assistant", "content": "History"},
                {"role": "user", "content": "Question"},
            ],
        )
        self.assertIn("[assistant] History", preview)
        self.assertIn("[user] Question", preview)


if __name__ == "__main__":
    unittest.main()
