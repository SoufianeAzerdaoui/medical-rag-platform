from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATION_DIR = PROJECT_ROOT / "scripts" / "generation"
if str(GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATION_DIR))

from analyte_aliases import ANALYTE_ALIAS_GROUPS
from analyte_resolver import resolve_requested_analytes


class TestAnalyteResolver(unittest.TestCase):
    def _resolve_one(self, query: str, available: list[dict] | None = None) -> dict:
        resolved = resolve_requested_analytes(
            query=query,
            available_analytes=available or [{"display_name": k.upper(), "analyte_norm": k} for k in ANALYTE_ALIAS_GROUPS.keys()],
            aliases=ANALYTE_ALIAS_GROUPS,
            max_candidates=5,
        )
        self.assertTrue(resolved)
        return resolved[0]

    def test_parentheses_glucose(self) -> None:
        r = self._resolve_one("Glycémie (Glucose)")
        self.assertEqual(r.get("analyte_norm"), "glucose")

    def test_glycemie_without_accent(self) -> None:
        r = self._resolve_one("glycemie")
        self.assertEqual(r.get("analyte_norm"), "glucose")

    def test_tsh_variants(self) -> None:
        r1 = self._resolve_one("TSH ultrasensible")
        r2 = self._resolve_one("TSH us")
        r3 = self._resolve_one("TSH-us")
        self.assertEqual(r1.get("analyte_norm"), "tshus")
        self.assertEqual(r2.get("analyte_norm"), "tshus")
        self.assertEqual(r3.get("analyte_norm"), "tshus")

    def test_ca_15_3_numeric_kept(self) -> None:
        r = self._resolve_one("CA 15-3")
        self.assertEqual(r.get("analyte_norm"), "ca_15_3")

    def test_pth_intacte(self) -> None:
        r = self._resolve_one("PTH intacte")
        self.assertEqual(r.get("analyte_norm"), "pth_intact")

    def test_gh_sth(self) -> None:
        r1 = self._resolve_one("GH")
        r2 = self._resolve_one("STH")
        self.assertEqual(r1.get("analyte_norm"), "gh_ou_sth")
        self.assertEqual(r2.get("analyte_norm"), "gh_ou_sth")

    def test_phosphatase_alias(self) -> None:
        r1 = self._resolve_one("PAL")
        r2 = self._resolve_one("phosphatase alcaline")
        self.assertEqual(r1.get("analyte_norm"), "phosphatase_alcaline")
        self.assertEqual(r2.get("analyte_norm"), "phosphatase_alcaline")

    def test_dynamic_vocab_without_alias(self) -> None:
        available = [{"display_name": "TEST X BETA", "analyte_norm": "test_x_beta"}]
        r = self._resolve_one("compare TEST X beta", available=available)
        self.assertEqual(r.get("analyte_norm"), "test_x_beta")

    def test_acide_urique_aliases(self) -> None:
        for q in ["acide urique", "uric acid", "urate", "uricémie"]:
            r = self._resolve_one(q)
            self.assertEqual(r.get("analyte_norm"), "acide_urique")

    def test_hdl_aliases(self) -> None:
        for q in ["cholestérol hdl", "hdl cholesterol", "chol hdl"]:
            r = self._resolve_one(q)
            self.assertEqual(r.get("analyte_norm"), "cholesterol_hdl")


if __name__ == "__main__":
    unittest.main()
