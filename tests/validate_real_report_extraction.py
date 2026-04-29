from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTRACTION_SCRIPT_ROOT = PROJECT_ROOT / "scripts" / "extraction_data"
if str(EXTRACTION_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTRACTION_SCRIPT_ROOT))

from run_extraction import run_pipeline


REPORT_PDF = PROJECT_ROOT / "docs" / "report.pdf"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "extraction"


def load_real_report() -> dict:
    output_dir = run_pipeline(REPORT_PDF, OUTPUT_ROOT)
    return json.loads((output_dir / "document.json").read_text(encoding="utf-8"))


def add_issue(issues: list[str], condition: bool, message: str) -> None:
    if not condition:
        issues.append(message)


def find_result(document: dict, parameter: str, result: str) -> dict | None:
    for item in document.get("results", []):
        if item.get("parameter") == parameter and item.get("result") == result:
            return item
    return None


def find_block(document: dict, block_type: str) -> dict | None:
    return next((block for block in document.get("blocks", []) if block.get("block_type") == block_type), None)


def main() -> int:
    document = load_real_report()
    issues: list[str] = []

    facility = document.get("facility", {})
    patient = document.get("patient", {})
    report = document.get("report", {})
    validation = document.get("validation", {})
    validation_report = document.get("validation_report", {})
    results = document.get("results", [])
    tables = document.get("tables", [])

    patient_block = find_block(document, "patient_info_block")
    report_block = find_block(document, "report_metadata_block")
    sample_block = find_block(document, "sample_block")
    staining_block = find_block(document, "staining_exam_block")
    final_block = find_block(document, "final_result_block")
    edition_block = find_block(document, "edition_block")

    add_issue(issues, document.get("document_type") == "parasitology_stool_report", "document_type should be parasitology_stool_report")
    add_issue(
        issues,
        isinstance(document.get("source_pdf"), str) and not str(document.get("source_pdf")).startswith("/home/"),
        "source_pdf should not be an absolute /home/... path",
    )
    add_issue(
        issues,
        isinstance(document.get("output_dir"), str) and not str(document.get("output_dir")).startswith("/home/"),
        "output_dir should not be an absolute /home/... path",
    )

    add_issue(issues, patient.get("name") == "PYXIS TEST", "patient.name should be PYXIS TEST")
    add_issue(issues, "address" not in patient, "patient should not contain address")
    add_issue(issues, "reported_age" in patient, "patient should contain reported_age")
    add_issue(issues, patient.get("age_consistency_status") == "inconsistent_with_birth_date", "patient.age_consistency_status mismatch")

    add_issue(issues, "report_id" not in report, "report should not contain report_id")
    add_issue(issues, "report_id_raw" not in report, "report should not contain report_id_raw")
    add_issue(issues, report.get("exam_name") == "EXAMEN PARASITOLOGIQUE DES SELLES", "report.exam_name mismatch")
    add_issue(issues, report.get("sample_id") == "240601915", "report.sample_id should be 240601915")
    add_issue(issues, report.get("sample_id_raw") == "240601915", "report.sample_id_raw should be 240601915")
    add_issue(issues, report.get("prescriber") == "Dr.CHAYMAE EL-MEJAHED", "report.prescriber mismatch")
    add_issue(issues, report.get("sample_type") == "SELLES", "report.sample_type should be SELLES")
    add_issue(issues, report.get("sample_label") == "SELLES N°1", "report.sample_label mismatch")

    add_issue(issues, facility.get("website") == "www.chuoujda.ma", "facility.website should be www.chuoujda.ma")
    add_issue(issues, "address" not in facility, "facility should not contain address")
    add_issue(
        issues,
        "Centre Hospitalo-Universitaire Mohammed VI - Oujda" in (facility.get("organization") or ""),
        "facility.organization should contain CHU Mohammed VI Oujda",
    )

    add_issue(issues, document.get("ocr_available") is False, "ocr_available should be false for native-text report")
    add_issue(issues, "ocr_visuals" not in document, "ocr_visuals should be omitted when ocr_available is false")

    add_issue(issues, len(results) == 15, f"results should contain 15 rows, got {len(results)}")
    add_issue(issues, all("parameter" in result for result in results), "each result should contain parameter")
    add_issue(issues, all("result" in result for result in results), "each result should contain result")
    for forbidden in [
        "reference_range",
        "unit",
        "unit_raw",
        "value_numeric",
        "observation_date_raw",
        "ocr_correction_applied",
        "field_confidence",
    ]:
        add_issue(issues, all(forbidden not in result for result in results), f"results should not contain {forbidden}")

    final_result = find_result(document, "RÉSULTAT FINAL", "TRICHURIS TRICHIURA")
    add_issue(issues, final_result is not None, "final parasitology result should exist")
    add_issue(issues, final_result is not None and final_result.get("result_kind") == "pathogen_identification", "final result_kind mismatch")

    logical_table = next((table for table in tables if table.get("table_id") == "logical_results_p001_01"), None)
    add_issue(issues, logical_table is not None, "logical_results_p001_01 should exist")
    if logical_table:
        add_issue(issues, "parameter" in logical_table.get("columns", []), "logical table should use parameter")
        add_issue(issues, "result" in logical_table.get("columns", []), "logical table should use result")
        add_issue(issues, "analyte" not in logical_table.get("columns", []), "logical table should not expose analyte")
        add_issue(issues, "value" not in logical_table.get("columns", []), "logical table should not expose value")
        add_issue(
            issues,
            all("parameter" in record and "result" in record for record in logical_table.get("records", [])),
            "logical table records should use parameter/result",
        )
        add_issue(
            issues,
            all("analyte" not in record and "value" not in record for record in logical_table.get("records", [])),
            "logical table records should not expose analyte/value",
        )

    add_issue(issues, validation.get("edited_by") == "BASMA HASSANI", "validation.edited_by mismatch")
    add_issue(issues, validation.get("printed_by") == "M.REHALI", "validation.printed_by mismatch")
    add_issue(issues, validation.get("validated_by") is None, "validation should not set validated_by")

    add_issue(issues, patient_block is not None, "patient_info_block should exist")
    if patient_block:
        add_issue(issues, "المملكة المغربية" not in patient_block.get("text", ""), "patient_info_block should not contain Arabic institutional text")
        add_issue(issues, "Centre Hospitalo-Universitaire" not in patient_block.get("text", ""), "patient_info_block should not contain facility text")
        add_issue(
            issues,
            set((patient_block.get("structured_fields") or {}).keys()) > {"source"},
            "patient_info_block structured_fields should contain business fields",
        )

    add_issue(issues, report_block is not None, "report_metadata_block should exist")
    if report_block:
        add_issue(issues, "Dr.CHAYMAE EL-MEJAHED" in report_block.get("text", ""), "report_metadata_block should contain prescriber")
        add_issue(
            issues,
            (report_block.get("structured_fields") or {}).get("exam_name") == "EXAMEN PARASITOLOGIQUE DES SELLES",
            "report_metadata_block.structured_fields.exam_name mismatch",
        )

    add_issue(issues, sample_block is not None, "sample_block should exist")
    if sample_block:
        add_issue(
            issues,
            (sample_block.get("structured_fields") or {}).get("exam_name") == "EXAMEN PARASITOLOGIQUE DES SELLES",
            "sample_block.structured_fields.exam_name mismatch",
        )

    add_issue(issues, staining_block is not None, "staining_exam_block should exist")
    if staining_block:
        add_issue(issues, "RÉSULAT FINAL" not in staining_block.get("text", ""), "staining_exam_block should not contain RÉSULAT FINAL")
        add_issue(issues, "RÉSULTAT FINAL" not in staining_block.get("text", ""), "staining_exam_block should not contain RÉSULTAT FINAL")
        add_issue(issues, "RÉSULAT FINAL" not in staining_block.get("index_text", ""), "staining_exam_block.index_text should be clean")

    add_issue(issues, final_block is not None, "final_result_block should exist")
    if final_block:
        add_issue(issues, final_block.get("text") == "RÉSULTAT FINAL : TRICHURIS TRICHIURA", "final_result_block.text mismatch")
        add_issue(issues, "Edité(e) par" not in final_block.get("text", ""), "final_result_block should not contain edition text")
        add_issue(issues, "Page 1 sur 1" not in final_block.get("text", ""), "final_result_block should not contain page footer")
        add_issue(issues, "04/06/2024" not in final_block.get("index_text", ""), "final_result_block.index_text should not contain edit date")
        add_issue(
            issues,
            set((final_block.get("structured_fields") or {}).keys()) > {"source"},
            "final_result_block structured_fields should contain business fields",
        )

    add_issue(issues, edition_block is not None, "edition_block should exist")
    if edition_block:
        add_issue(issues, "BASMA HASSANI" in edition_block.get("text", ""), "edition_block should contain BASMA HASSANI")
        add_issue(
            issues,
            set((edition_block.get("structured_fields") or {}).keys()) > {"source"},
            "edition_block structured_fields should contain business fields",
        )

    add_issue(issues, validation_report.get("canonical_result_rows") == 15, "canonical_result_rows should stay at 15")
    add_issue(
        issues,
        validation_report.get("computed_age_reference_date") == "request_date",
        "validation_report.computed_age_reference_date should be request_date",
    )

    if issues:
        print("FAIL real report extraction")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print("PASS real report extraction")
    print("canonical_result_rows=15")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
