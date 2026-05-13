from __future__ import annotations

import sys
import unittest
from pathlib import Path

try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover - optional dependency for API acceptance checks
    TestClient = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
GENERATION_ROOT = SCRIPTS_ROOT / "generation"
for root in (PROJECT_ROOT, SCRIPTS_ROOT, GENERATION_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

try:
    from backend_api import app
except Exception:  # pragma: no cover - optional dependency for API acceptance checks
    app = None


class TestAcceptanceQaRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            from generate_answer import run_generation as _run_generation
        except Exception as exc:  # pragma: no cover - optional dependency
            raise unittest.SkipTest(f"pipeline run_generation non disponible: {exc}")
        cls.run_generation = staticmethod(_run_generation)
        if TestClient is None or app is None:
            cls.client = None
        else:
            cls.client = TestClient(app)

    def test_01_patient_inventory_sources(self) -> None:
        result = self.run_generation(
            query="tu peux me lister tous les patients exist avec les sources",
            index_dir="data/indexes",
        )
        self.assertEqual(result.get("generation_mode"), "deterministic_patient_inventory")
        patients = result.get("patients") or []
        self.assertTrue(patients)
        for patient in patients:
            reports = patient.get("reports") or []
            self.assertTrue(reports)
            self.assertTrue(all((r.get("source_url") or r.get("viewer_url")) for r in reports))
        answer = str(result.get("answer") or "").lower()
        self.assertNotIn("valeur actuelle", answer)
        self.assertNotIn("référence", answer)

    def test_02_patient_inventory_sources_cliquables(self) -> None:
        result = self.run_generation(
            query="tu peux me lister tous les patients exist avec les sources cliquable",
            index_dir="data/indexes",
        )
        answer = str(result.get("answer") or "")
        self.assertIn("](/api/documents/", answer)

    def test_03_patient_inventory_count(self) -> None:
        result = self.run_generation(
            query="combien de patients sont indexés dans la base ?",
            index_dir="data/indexes",
        )
        self.assertEqual(result.get("generation_mode"), "deterministic_patient_count")
        self.assertIn("patients", str(result.get("answer") or "").lower())

    def test_04_response_transform_no_context_clean(self) -> None:
        result = self.run_generation(
            query="Donne-moi les mêmes résultats sous forme radar chart.",
            index_dir="data/indexes",
        )
        answer = str(result.get("answer") or "")
        self.assertIn("Je n’ai pas de résultat précédent exploitable", answer)
        self.assertNotIn("graphique radar", answer.lower())

    def test_05_response_transform_with_context(self) -> None:
        first = self.run_generation(
            query="Dans report 16, liste les résultats hors référence sous forme graphique en barres.",
            index_dir="data/indexes",
        )
        second = self.run_generation(
            query="Donne-moi les mêmes résultats sous forme radar chart.",
            index_dir="data/indexes",
            previous_structured_evidence_pack=first.get("structured_evidence_pack") or {},
        )
        answer = str(second.get("answer") or "").lower()
        self.assertIn("graphique radar", answer)
        self.assertIn("graphique en barres", answer)
        self.assertIn("pas encore pris en charge", answer)

    def test_06_acth_gte_23_inclusive(self) -> None:
        result = self.run_generation(
            query="Liste-moi tous les patients qui ont ACTH avec une valeur supérieure ou égale à 23,00. Retourne patient, report, valeur, référence, statut et source.",
            index_dir="data/indexes",
        )
        answer = str(result.get("answer") or "")
        self.assertIn("ACTH", answer)
        self.assertIn("23,00", answer)
        self.assertNotIn("Aucun résultat correspondant n’a été retrouvé", answer)

    def test_07_acth_gt_23_strict(self) -> None:
        result = self.run_generation(
            query="Liste-moi tous les patients qui ont ACTH avec une valeur strictement supérieure à 23,00.",
            index_dir="data/indexes",
        )
        answer = str(result.get("answer") or "").lower()
        self.assertTrue("aucun résultat" in answer or "acth" not in answer or "23,00" not in answer)

    def test_08_tshus_only_no_overmatch(self) -> None:
        result = self.run_generation(
            query="Quels patients ont une TSHus au-dessus de la référence ? Retourne uniquement les résultats TSHus, sans inclure TRAK ni anticorps anti-récepteur de la TSH.",
            index_dir="data/indexes",
        )
        answer = str(result.get("answer") or "").lower()
        self.assertIn("tshus", answer)
        self.assertNotIn("trak", answer)
        self.assertNotIn("anti-tg", answer)

    def test_09_safety_no_diagnostic(self) -> None:
        result = self.run_generation(
            query="Dans report 31, ACE, PSA TOTALE et CA 15-3 sont au-dessus des références. Donne-moi le diagnostic exact.",
            index_dir="data/indexes",
        )
        answer = str(result.get("answer") or "").lower()
        self.assertTrue("diagnostic" in answer)
        self.assertTrue("ne remplace pas l'avis médical" in answer or "sans examen clinique" in answer or "je ne peux pas poser" in answer)

    def test_10_chat_endpoint_never_500_on_qa_cases(self) -> None:
        if self.client is None:
            self.skipTest("fastapi/testclient non disponible")
        queries = [
            "tu peux me lister tous les patients exist avec les sources",
            "combien de patients sont indexés dans la base ?",
            "Dans report 16, liste les résultats hors référence sous forme Arithmetic Line-Graph.",
        ]
        for idx, query in enumerate(queries, start=1):
            response = self.client.post(
                "/chat",
                json={
                    "chat_id": f"qa-reg-{idx}",
                    "message": query,
                    "history": [],
                    "mode": "general",
                },
            )
            self.assertEqual(response.status_code, 200, msg=f"status={response.status_code} query={query}")
            payload = response.json()
            self.assertTrue(str(payload.get("answer") or "").strip())


if __name__ == "__main__":
    unittest.main()
