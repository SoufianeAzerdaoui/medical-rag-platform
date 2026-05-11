from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATION_SCRIPT_ROOT = PROJECT_ROOT / "scripts" / "generation"
if str(GENERATION_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERATION_SCRIPT_ROOT))

from generate_answer import (  # noqa: E402
    SMALL_TALK_FALLBACK_ANSWER,
    generate_general_conversation_response,
    generate_small_talk_response,
    sanitize_final_answer,
)


class _FakeLLM:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.i = 0

    def generate(self, **_: object) -> str:
        if self.i >= len(self.outputs):
            return self.outputs[-1] if self.outputs else ""
        out = self.outputs[self.i]
        self.i += 1
        return out


class TestSmallTalkSanitization(unittest.TestCase):
    def test_01_bonjour_final_only(self) -> None:
        llm = _FakeLLM(["Bonjour ! Je suis prêt à vous aider à analyser vos rapports médicaux."])
        ans, err = generate_small_talk_response("bonjour", llm_client=llm, model="qwen3:4b")
        self.assertIsNone(err)
        self.assertIn("Bonjour", ans)
        self.assertNotIn("Okay, the user", ans)
        self.assertNotIn("I need to", ans)

    def test_02_salut_ca_va_short(self) -> None:
        llm = _FakeLLM(["Salut ! Oui, ça va bien. Vous pouvez me poser une question sur un rapport biologique."])
        ans, _ = generate_small_talk_response("salut ça va ?", llm_client=llm, model="qwen3:4b")
        self.assertIn("Salut", ans)
        self.assertNotIn("source", ans.lower())
        self.assertNotIn("I need to", ans)

    def test_03_merci_polite(self) -> None:
        llm = _FakeLLM(["Avec plaisir ! Je reste disponible si vous avez une autre question."])
        ans, _ = generate_small_talk_response("merci", llm_client=llm, model="qwen3:4b")
        self.assertIn("plaisir", ans.lower())
        self.assertNotIn("Okay, the user", ans)

    def test_04_leak_simulation_fallback(self) -> None:
        leak = "Okay, the user said bonjour. I need to respond. First, I'll greet."
        llm = _FakeLLM([leak, leak])
        ans, _ = generate_small_talk_response("bonjour", llm_client=llm, model="qwen3:4b")
        self.assertEqual(ans, SMALL_TALK_FALLBACK_ANSWER)

    def test_05_sanitize_think_tags(self) -> None:
        raw = "<think>I need to plan</think>\nRéponse: Bonjour !"
        self.assertEqual(sanitize_final_answer(raw), "Bonjour !")

    def test_06_identity_response(self) -> None:
        llm = _FakeLLM(
            [
                "Je suis l’assistant Medical RAG de cette application. Je peux aider à interroger des rapports médicaux et citer les sources PDF.",
            ]
        )
        ans, _ = generate_general_conversation_response("t es qui", intent="identity_question", llm_client=llm, model="qwen3:4b")
        self.assertIn("Medical RAG", ans)
        self.assertNotIn("report_", ans.lower())
        self.assertNotIn("source :", ans.lower())

    def test_07_capability_response(self) -> None:
        llm = _FakeLLM(
            [
                "Je peux rechercher des résultats, comparer des valeurs entre rapports et fournir les sources PDF associées.",
            ]
        )
        ans, _ = generate_general_conversation_response(
            "tu peux faire quoi",
            intent="capability_question",
            llm_client=llm,
            model="qwen3:4b",
        )
        self.assertIn("comparer", ans.lower())
        self.assertNotIn("Okay, the user", ans)


if __name__ == "__main__":
    unittest.main()
