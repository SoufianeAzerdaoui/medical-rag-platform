from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
GENERATION_ROOT = SCRIPTS_ROOT / "generation"
for root in (SCRIPTS_ROOT, GENERATION_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from generate_answer import _compute_priority_fields


def _ev(
    *,
    analyte: str = "TEST_X",
    analyte_norm: str = "test_x",
    value: str = "1",
    reference: str = "0 - 10 g/l",
    status: str = "above_reference",
    technical_status: str = "au-dessus de la référence",
) -> dict:
    return {
        "analyte": analyte,
        "analyte_norm": analyte_norm,
        "current_value": value,
        "reference": reference,
        "technical_status_code": status,
        "technical_status": technical_status,
    }


class TestTechnicalPriorityScoring(unittest.TestCase):
    def test_above_reference_simple_ratio_high(self) -> None:
        out = _compute_priority_fields(
            _ev(value="30", reference="0 - 10 g/l", status="above_reference")
        )
        self.assertEqual(out["priority_level"], "high")
        self.assertGreaterEqual(float(out["priority_score"]), 2.4)
        self.assertIn("écart relatif", str(out["priority_reason"]).lower())

    def test_above_reference_moderate_lower_score_than_ratio3(self) -> None:
        high = _compute_priority_fields(
            _ev(value="30", reference="0 - 10 g/l", status="above_reference")
        )
        moderate = _compute_priority_fields(
            _ev(value="14", reference="0 - 10 g/l", status="above_reference")
        )
        self.assertIn(moderate["priority_level"], {"low", "moderate"})
        self.assertLess(float(moderate["priority_score"]), float(high["priority_score"]))

    def test_below_reference_simple_high(self) -> None:
        out = _compute_priority_fields(
            _ev(value="2", reference="10 - 20 g/l", status="below_reference", technical_status="en dessous de la référence")
        )
        self.assertEqual(out["priority_level"], "high")
        self.assertGreaterEqual(float(out["priority_score"]), 2.4)
        self.assertIn("écart relatif", str(out["priority_reason"]).lower())

    def test_within_reference_not_priority(self) -> None:
        out = _compute_priority_fields(
            _ev(value="40", reference="40 - 150 UI/L", status="within_reference", technical_status="dans la référence")
        )
        self.assertEqual(out["priority_level"], "unknown")
        self.assertEqual(float(out["priority_score"]), 0.0)

    def test_complex_qualitative_reference_is_conservative(self) -> None:
        out = _compute_priority_fields(
            _ev(
                analyte="Cholestérol HDL",
                analyte_norm="cholesterol_hdl",
                value="0.50",
                reference="Risque maladies cardiaques Majeur : <0,4 g/l ; Négatif : >0,60 g/l",
                status="above_reference",
            )
        )
        self.assertIn(out["priority_level"], {"unknown", "low", "moderate"})
        self.assertNotEqual(out["priority_level"], "high")
        self.assertIn("référence complexe", str(out["priority_reason"]).lower())

    def test_textual_critical_marker_boosts_priority(self) -> None:
        out = _compute_priority_fields(
            _ev(
                analyte="Triglycérides",
                analyte_norm="triglycerides",
                value="8",
                reference="Normale: < 1,50 g/l limite haute: 1,50 - 1,99 g/l Haute: 2 - 4,99 g/l Très haute: > 5 g/l",
                status="above_reference",
                technical_status="très élevée",
            )
        )
        self.assertEqual(out["priority_level"], "high")
        self.assertIn("sévérité", str(out["priority_reason"]).lower())

    def test_phosphatase_at_inclusive_bound_is_not_above(self) -> None:
        out = _compute_priority_fields(
            _ev(
                analyte="PHOSPHATASE ALCALINE",
                analyte_norm="phosphatase_alcaline",
                value="40",
                reference="Femme : 1 à 12 ans: < 500 UI/L > 15 ans: 40 - 150 UI/L Homme : >20 ans: 40 - 150 UI/L",
                status="above_reference",
            )
        )
        self.assertEqual(out["priority_level"], "unknown")
        self.assertEqual(float(out["priority_score"]), 0.0)
        self.assertIn("plage de référence explicite", str(out["priority_reason"]).lower())

    def test_family_bonus_is_light_and_does_not_create_anomaly(self) -> None:
        crp = _compute_priority_fields(
            _ev(analyte="CRP", analyte_norm="crp", value="10", reference="0 - 5 mg/l", status="above_reference")
        )
        unk = _compute_priority_fields(
            _ev(analyte="TEST_X", analyte_norm="test_x", value="10", reference="0 - 5 mg/l", status="above_reference")
        )
        self.assertGreater(float(crp["priority_score"]), float(unk["priority_score"]))

        within = _compute_priority_fields(
            _ev(analyte="CRP", analyte_norm="crp", value="3", reference="0 - 5 mg/l", status="within_reference", technical_status="dans la référence")
        )
        self.assertEqual(within["priority_level"], "unknown")
        self.assertEqual(float(within["priority_score"]), 0.0)

    def test_missing_reference_returns_unknown(self) -> None:
        out = _compute_priority_fields(
            _ev(value="8", reference="", status="above_reference")
        )
        self.assertEqual(out["priority_level"], "unknown")
        self.assertEqual(float(out["priority_score"]), 0.0)
        self.assertIn("référence", str(out["priority_reason"]).lower())


if __name__ == "__main__":
    unittest.main()
