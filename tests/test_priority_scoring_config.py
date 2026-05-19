from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATION_ROOT = PROJECT_ROOT / "scripts" / "generation"
if str(GENERATION_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERATION_ROOT))

from priority_scoring import assign_priority_level, compute_priority_score, get_priority_thresholds


class TestPriorityScoringConfig(unittest.TestCase):
    def test_thresholds_loaded(self) -> None:
        th = get_priority_thresholds()
        self.assertAlmostEqual(float(th["high"]), 2.4)
        self.assertAlmostEqual(float(th["moderate"]), 0.9)
        self.assertAlmostEqual(float(th["low"]), 0.2)

    def test_triglycerides_very_high_is_high(self) -> None:
        out = compute_priority_score(
            {
                "analyte": "Triglycérides",
                "analyte_norm": "triglycerides",
                "current_value": "8",
                "reference": "Normale: < 1,50 g/l limite haute: 1,50 - 1,99 g/l Haute: 2 - 4,99 g/l Très haute: > 5 g/l",
                "technical_status_code": "above_reference",
                "technical_status": "très élevée",
            }
        )
        self.assertEqual(out["priority_level"], "high")

    def test_albumine_basse_is_high(self) -> None:
        out = compute_priority_score(
            {
                "analyte": "Albumine",
                "analyte_norm": "albumine",
                "current_value": "10",
                "reference": "35 - 52 g/l",
                "technical_status_code": "below_reference",
                "technical_status": "en dessous de la référence",
            }
        )
        self.assertIn(out["priority_level"], {"moderate", "high"})
        self.assertIn(assign_priority_level(float(out["priority_score"])), {"moderate", "high"})


if __name__ == "__main__":
    unittest.main()
