from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATION_ROOT = PROJECT_ROOT / "scripts" / "generation"
if str(GENERATION_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERATION_ROOT))

from medical_topics import detect_medical_topic, get_topic_analytes, get_topic_exclusions


class TestMedicalTopicsConfig(unittest.TestCase):
    def test_detect_thyroid_topic(self) -> None:
        topic = detect_medical_topic("Peux-tu analyser ce profil hyperthyroïdie ?")
        self.assertEqual(topic, "thyroid")

    def test_thyroid_analytes_and_exclusions(self) -> None:
        analytes = set(get_topic_analytes("thyroid"))
        self.assertTrue({"tshus", "t4_libre", "t3_libre", "anti_tg", "anti_tpo", "trak"}.issubset(analytes))
        exclusions = set(get_topic_exclusions("thyroid"))
        self.assertIn("acth", exclusions)
        self.assertIn("insuline", exclusions)


if __name__ == "__main__":
    unittest.main()
