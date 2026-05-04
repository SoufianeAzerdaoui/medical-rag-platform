from __future__ import annotations

import sys
import unittest
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANONYMIZATION_SCRIPT_ROOT = PROJECT_ROOT / "scripts" / "anonymization"
if str(ANONYMIZATION_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(ANONYMIZATION_SCRIPT_ROOT))

from anonymize_chunks import MappingStore, anonymize_chunk, compute_content_hash


class TestAnonymizationContentHash(unittest.TestCase):
    def _raw_chunk(self) -> dict:
        chunk = {
            "chunk_id": "chk_report_test_lab_result_1",
            "parent_chunk_id": "chk_report_test_exam_section_1",
            "doc_id": "report_test",
            "chunk_type": "lab_result",
            "modality": "text",
            "schema_version": "clinical_chunk_raw_v2",
            "text_for_embedding": (
                "Résultat de laboratoire: CALCITONINE = 3,00 pg/mL. "
                "IP Patient: 53. Edité(e) par: Dr.X."
            ),
            "text_for_keyword": (
                "CALCITONINE 3,00 pg/mL IP Patient 53 Edité(e) par Dr.X"
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

    def test_content_hash_recomputed_and_stable(self) -> None:
        raw_chunk = self._raw_chunk()
        raw_hash = raw_chunk["content_hash"]

        mapping = MappingStore()
        counters = defaultdict(int)

        anonymized_1, _sensitive_1 = anonymize_chunk(raw_chunk, mapping, counters)
        self.assertEqual(anonymized_1["chunk_id"], raw_chunk["chunk_id"])
        self.assertEqual(anonymized_1["parent_chunk_id"], raw_chunk["parent_chunk_id"])

        # Hash must represent final anonymized content.
        self.assertEqual(anonymized_1["content_hash"], compute_content_hash(anonymized_1))
        self.assertNotEqual(anonymized_1["content_hash"], raw_hash)

        # Re-running anonymization with same mapping should stay stable.
        counters_2 = defaultdict(int)
        anonymized_2, _sensitive_2 = anonymize_chunk(raw_chunk, mapping, counters_2)
        self.assertEqual(
            anonymized_1["metadata"].get("patient_token"),
            anonymized_2["metadata"].get("patient_token"),
        )
        self.assertEqual(
            anonymized_1["metadata"].get("sample_token"),
            anonymized_2["metadata"].get("sample_token"),
        )
        self.assertEqual(
            anonymized_1["metadata"].get("report_token"),
            anonymized_2["metadata"].get("report_token"),
        )
        self.assertEqual(anonymized_1["content_hash"], anonymized_2["content_hash"])


if __name__ == "__main__":
    unittest.main()
