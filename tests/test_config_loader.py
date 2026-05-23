from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATION_ROOT = PROJECT_ROOT / "scripts" / "generation"
if str(GENERATION_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERATION_ROOT))

import config_loader as cl


class TestConfigLoader(unittest.TestCase):
    def test_load_existing_configs(self) -> None:
        self.assertIn("topics", cl.get_medical_topics_config())
        self.assertIn("families", cl.get_analyte_families_config())
        self.assertIn("priority_scoring", cl.get_priority_scoring_config())
        self.assertIn("general_conversation", cl.get_assistant_messages_config())

    def test_missing_file_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "missing.yml"
            out = cl.load_yaml_config(p, {"x": {"y": 1}})
            self.assertEqual(out["x"]["y"], 1)


if __name__ == "__main__":
    unittest.main()
