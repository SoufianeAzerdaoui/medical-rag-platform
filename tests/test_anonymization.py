from __future__ import annotations

import sys
import unittest
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANONYMIZATION_SCRIPT_ROOT = PROJECT_ROOT / "scripts" / "anonymization"
if str(ANONYMIZATION_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(ANONYMIZATION_SCRIPT_ROOT))

from anonymize_chunks import MappingStore, anonymize_chunk, compute_content_hash, validate_anonymized


class TestAnonymization(unittest.TestCase):
    def _raw_parent_chunk(self) -> dict:
        chunk = {
            "chunk_id": "chk_report_test_document_summary",
            "parent_chunk_id": None,
            "doc_id": "report_test",
            "chunk_type": "document_summary",
            "modality": "text",
            "schema_version": "clinical_chunk_raw_v2",
            "text_for_embedding": "Résumé du document patient 53.",
            "text_for_keyword": "document summary patient 53",
            "metadata": {
                "patient_name": "PATIENT TEST1",
                "patient_id": "53",
                "patient_id_raw": "53",
                "ip_patient": "53",
                "sample_id": "240601915",
                "report_id": "RPT-ABC-001",
                "patient_birth_date": "1997-01-01",
                "patient_birth_date_raw": "01/01/1997",
                "prescriber": "Dr. X",
                "validated_by": "Dr. Y",
                "edited_by": "Dr. Z",
                "printed_by": "Dr. K",
                "phone": "0600000000",
                "fax": "0500000000",
                "section": "Résumé",
            },
            "provenance": {
                "source_pdf": "docs/report (1).pdf",
                "extraction_json": "data/extraction/report_1/document.json",
            },
            "quality": {"confidence": "high", "confidence_score": 0.9},
            "routing": {"vector_index": True, "keyword_index": True, "metadata_index": True},
        }
        chunk["content_hash"] = compute_content_hash(chunk)
        return chunk

    def _raw_lab_chunk(self) -> dict:
        chunk = {
            "chunk_id": "chk_report_test_lab_result_1",
            "parent_chunk_id": "chk_report_test_document_summary",
            "doc_id": "report_test",
            "chunk_type": "lab_result",
            "modality": "text",
            "schema_version": "clinical_chunk_raw_v2",
            "text_for_embedding": (
                "Résultat de laboratoire: CALCITONINE = 3,00 pg/mL. "
                "IP Patient: 53. Né(e) le: 01/01/1997. Validé(e) par: Dr.Y."
            ),
            "text_for_keyword": (
                "CALCITONINE 3,00 pg/mL patient_id 53 1997-01-01 Validé(e) par Dr.Y"
            ),
            "metadata": {
                "patient_name": "PATIENT TEST1",
                "patient_id": "53",
                "patient_id_raw": "53",
                "ip_patient": "53",
                "sample_id": "240601915",
                "sample_id_raw": "240601915",
                "report_id": "RPT-ABC-001",
                "report_id_raw": "RPT-ABC-001",
                "patient_birth_date": "1997-01-01",
                "patient_birth_date_raw": "01/01/1997",
                "prescriber": "Dr.X",
                "validated_by": "Dr.Y",
                "edited_by": "Dr.Z",
                "printed_by": "Dr.K",
                "phone": "0600000000",
                "fax": "0500000000",
                "print_date": "2024-07-19",
                "analyte": "CALCITONINE",
                "analyte_norm": "calcitonine",
                "value_raw": "3,00",
                "value_numeric": 3.0,
                "unit": "pg/mL",
                "reference_range": "< 11,80 pg/ml",
                "reference_range_text": "< 11,80 pg/ml",
                "reference_operator": "<",
                "reference_separator": None,
                "reference_unit": "pg/ml",
                "reference_low": None,
                "reference_high": 11.8,
                "previous_result": {"value_raw": "5,30", "value_numeric": 5.3, "unit": None},
                "row_index": 10,
                "section": "Résultats biologiques",
                "result_kind": "numeric",
                "interpretation_status": "within_reference",
                "page_number": 1,
            },
            "provenance": {
                "source_pdf": "docs/report (1).pdf",
                "extraction_json": "data/extraction/report_1/document.json",
                "source_kind": "chu_text_fallback",
                "row_index": 10,
            },
            "quality": {"confidence": "high", "confidence_score": 0.9},
            "routing": {"vector_index": True, "keyword_index": True, "metadata_index": True},
        }
        chunk["content_hash"] = compute_content_hash(chunk)
        return chunk

    def test_anonymization_removes_pii_and_preserves_medical_fields(self) -> None:
        raw_chunk = self._raw_lab_chunk()
        raw_hash = raw_chunk["content_hash"]

        mapping = MappingStore()
        counters = defaultdict(int)
        anonymized, sensitive_values = anonymize_chunk(raw_chunk, mapping, counters)

        # IDs and links must stay stable.
        self.assertEqual(anonymized["chunk_id"], raw_chunk["chunk_id"])
        self.assertEqual(anonymized["parent_chunk_id"], raw_chunk["parent_chunk_id"])

        metadata = anonymized["metadata"]
        for field in ("patient_id", "patient_id_raw", "ip_patient"):
            self.assertNotIn(field, metadata)
        self.assertTrue(metadata.get("patient_token"))
        self.assertEqual(metadata.get("patient_name"), "PATIENT_ANON")
        self.assertNotIn("patient_birth_date", metadata)
        self.assertNotIn("patient_birth_date_raw", metadata)
        for field in ("prescriber", "validated_by", "edited_by", "printed_by", "phone", "fax"):
            self.assertNotIn(field, metadata)

        text_blob = f"{anonymized['text_for_embedding']} {anonymized['text_for_keyword']}"
        self.assertNotIn("01/01/1997", text_blob)
        self.assertNotIn("1997-01-01", text_blob)
        self.assertNotIn("Dr.", text_blob)
        self.assertNotIn("Validé(e) par", text_blob)

        # Sensitive originals tracked by mapping must not leak in anonymized text.
        for value in set(v for v in sensitive_values if v):
            if len(value) >= 4:
                self.assertNotIn(value, text_blob)

        # Medical fields must stay intact.
        for field in (
            "analyte",
            "value_raw",
            "value_numeric",
            "unit",
            "reference_range",
            "previous_result",
            "row_index",
            "section",
        ):
            self.assertEqual(metadata.get(field), raw_chunk["metadata"].get(field))

        # content_hash must be recomputed from final anonymized chunk, deterministic.
        self.assertEqual(anonymized["content_hash"], compute_content_hash(anonymized))
        self.assertNotEqual(anonymized["content_hash"], raw_hash)

    def test_parent_link_valid_and_hash_stable_with_same_mapping(self) -> None:
        parent_raw = self._raw_parent_chunk()
        child_raw = self._raw_lab_chunk()

        mapping = MappingStore()

        anon_parent_1, sens_parent_1 = anonymize_chunk(parent_raw, mapping, defaultdict(int))
        anon_child_1, sens_child_1 = anonymize_chunk(child_raw, mapping, defaultdict(int))
        errors_1, _warn_1, _corrupt_1 = validate_anonymized(
            [anon_parent_1, anon_child_1],
            {
                anon_parent_1["chunk_id"]: sens_parent_1,
                anon_child_1["chunk_id"]: sens_child_1,
            },
        )
        broken_parent_errors = [e for e in errors_1 if e.get("issue") == "broken_parent_chunk_id"]
        self.assertEqual(broken_parent_errors, [])

        # Re-run with same mapping: tokens and hash must stay stable.
        anon_parent_2, _sens_parent_2 = anonymize_chunk(parent_raw, mapping, defaultdict(int))
        anon_child_2, _sens_child_2 = anonymize_chunk(child_raw, mapping, defaultdict(int))

        self.assertEqual(
            anon_child_1["metadata"].get("patient_token"),
            anon_child_2["metadata"].get("patient_token"),
        )
        self.assertEqual(
            anon_child_1["metadata"].get("sample_token"),
            anon_child_2["metadata"].get("sample_token"),
        )
        self.assertEqual(
            anon_child_1["metadata"].get("report_token"),
            anon_child_2["metadata"].get("report_token"),
        )
        self.assertEqual(anon_parent_1["content_hash"], anon_parent_2["content_hash"])
        self.assertEqual(anon_child_1["content_hash"], anon_child_2["content_hash"])


if __name__ == "__main__":
    unittest.main()
