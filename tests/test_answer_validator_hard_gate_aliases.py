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

from answer_validator import validate_answer


def _base_evidence() -> list[dict[str, object]]:
    return [
        {
            "doc_id": "report_12",
            "chunk_id": "chk_report_12_crp",
            "analyte": "CRP",
            "analyte_norm": "crp",
            "value_raw": "20",
            "current_value": "20",
            "unit": "mg/l",
            "reference_range": "0 - 5 mg/l",
            "technical_status_code": "above_reference",
            "interpretation_status": "above_reference",
            "source_pdf": "report (12).pdf",
            "page_number": 1,
            "row_index": 10,
        }
    ]


def _base_sources() -> list[dict[str, object]]:
    return [
        {
            "doc_id": "report_12",
            "chunk_id": "chk_report_12_crp",
            "label": "report (12).pdf — page 1, ligne 10",
            "source_pdf": "report (12).pdf",
            "page": 1,
            "row": 10,
            "viewer_url": "/viewer/pdf?doc_id=report_12&page=1",
        }
    ]


class TestAnswerValidatorHardGateAliases(unittest.TestCase):
    def test_source_mismatch_alias_emitted(self) -> None:
        validation = validate_answer(
            query="CRP report 12 ?",
            answer_text="CRP: 20 mg/l.\nSources:\n- [report (99).pdf — page 1, ligne 1](/viewer/pdf?doc_id=report_99&page=1)",
            evidence_pack=_base_evidence(),
            displayed_evidences=_base_evidence(),
            source_citations=[{**_base_sources()[0], "doc_id": "report_99"}],
            generation_mode="hybrid_structured_llm_writer",
        )
        self.assertEqual(str(validation.get("validation_status") or ""), "fail")
        self.assertIn("source_mismatch", list(validation.get("errors") or []))

    def test_raw_internal_source_alias_emitted(self) -> None:
        validation = validate_answer(
            query="CRP report 12 ?",
            answer_text='CRP: 20 mg/l.\nSources:\n- [doc_id=report_12, chunk_id="chk_report_12_crp"]',
            evidence_pack=_base_evidence(),
            displayed_evidences=_base_evidence(),
            source_citations=_base_sources(),
            generation_mode="hybrid_structured_llm_writer",
        )
        self.assertEqual(str(validation.get("validation_status") or ""), "fail")
        self.assertIn("raw_internal_source", list(validation.get("errors") or []))

    def test_pii_exposure_alias_emitted(self) -> None:
        validation = validate_answer(
            query="le patient a quoi ?",
            answer_text="Patient test1: CRP 20 mg/l.",
            evidence_pack=_base_evidence(),
            displayed_evidences=_base_evidence(),
            source_citations=_base_sources(),
            generation_mode="hybrid_structured_llm_writer",
        )
        self.assertEqual(str(validation.get("validation_status") or ""), "fail")
        self.assertIn("pii_exposure", list(validation.get("errors") or []))

    def test_value_changed_and_unit_mismatch_aliases_emitted(self) -> None:
        validation = validate_answer(
            query="CRP report 12 ?",
            answer_text="CRP: 999 mmol/l.\nSources:\n- [report (12).pdf — page 1, ligne 10](/viewer/pdf?doc_id=report_12&page=1)",
            evidence_pack=_base_evidence(),
            displayed_evidences=_base_evidence(),
            source_citations=_base_sources(),
            generation_mode="deterministic_doc_scoped_results",
        )
        self.assertEqual(str(validation.get("validation_status") or ""), "fail")
        self.assertIn("value_changed", list(validation.get("errors") or []))
        self.assertIn("unit_mismatch", list(validation.get("errors") or []))


if __name__ == "__main__":
    unittest.main()
