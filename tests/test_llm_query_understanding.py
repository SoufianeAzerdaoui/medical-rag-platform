from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATION_DIR = PROJECT_ROOT / "scripts" / "generation"
if str(GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATION_DIR))

from generate_answer import _llm_assisted_query_understanding
from query_understanding import parse_query_understanding


class _FakeLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, **_: object) -> str:
        return self.response


class TestLlmQueryUnderstanding(unittest.TestCase):
    def test_disabled_flag_keeps_deterministic_qu(self) -> None:
        base = parse_query_understanding("donne moi le resultat de AMH")
        with patch("generate_answer._is_feature_enabled", return_value=False):
            merged, dbg = _llm_assisted_query_understanding(
                query="donne moi le resultat de AMH",
                base_qu=base,
                llm_client=_FakeLLMClient('{"intent":"reference_range_lookup"}'),
                provider="ollama",
                model="llama3.2:latest",
                timeout=20,
            )
        self.assertEqual(merged.intent, base.intent)
        self.assertFalse(bool(dbg.get("enabled")))

    def test_enabled_flag_uses_llm_json(self) -> None:
        base = parse_query_understanding("et pour TSHus ?")
        llm_json = (
            '{"intent":"reference_range_lookup","requested_analytes":["tshus"],'
            '"requested_report_type":"immunoanalyse","requested_date_iso":"2024-07-19",'
            '"requested_reference_profile":{"sex":"male","age_operator":">","age":60,"age_unit":"years"},'
            '"use_patient_profile":false,"request_all_reference_ranges":false,'
            '"output_format":"paragraph","answer_style":"standard"}'
        )
        with patch("generate_answer._is_feature_enabled", return_value=True):
            merged, dbg = _llm_assisted_query_understanding(
                query="et pour TSHus ?",
                base_qu=base,
                llm_client=_FakeLLMClient(llm_json),
                provider="ollama",
                model="deepseek-r1:8b",
                timeout=20,
            )
        self.assertEqual(merged.intent, "reference_range_lookup")
        self.assertIn("tshus", merged.requested_analytes)
        self.assertEqual(merged.requested_report_type, "immunoanalyse")
        self.assertEqual(merged.requested_date_iso, "2024-07-19")
        self.assertTrue(bool(dbg.get("used")))

    def test_llm_does_not_override_reference_range_selection(self) -> None:
        base = parse_query_understanding("plage calcium pour homme > 60 ans")
        llm_json = (
            '{"intent":"reference_range_lookup","requested_analytes":["calcium"],'
            '"requested_reference_profile":{"sex":"male","age_operator":">","age":60,"age_unit":"years"},'
            '"use_patient_profile":false,"request_all_reference_ranges":true,'
            '"output_format":"table","answer_style":"standard"}'
        )
        with patch("generate_answer._is_feature_enabled", return_value=True):
            merged, dbg = _llm_assisted_query_understanding(
                query="plage calcium pour homme > 60 ans",
                base_qu=base,
                llm_client=_FakeLLMClient(llm_json),
                provider="ollama",
                model="llama3.2:latest",
                timeout=20,
            )
        self.assertEqual(merged.intent, "reference_range_lookup")
        self.assertFalse(merged.request_all_reference_ranges)
        self.assertEqual(merged.requested_reference_profile, base.requested_reference_profile)
        self.assertTrue(bool(dbg.get("used")))

    def test_enabled_flag_invalid_json_fallback(self) -> None:
        base = parse_query_understanding("donne moi le resultat de AMH")
        with patch("generate_answer._is_feature_enabled", return_value=True):
            merged, dbg = _llm_assisted_query_understanding(
                query="donne moi le resultat de AMH",
                base_qu=base,
                llm_client=_FakeLLMClient("not-json"),
                provider="ollama",
                model="deepseek-r1:8b",
                timeout=20,
            )
        self.assertEqual(merged.intent, base.intent)
        self.assertEqual(str(dbg.get("error") or ""), "invalid_json")


if __name__ == "__main__":
    unittest.main()

