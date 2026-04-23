from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTRACTION_SCRIPT_ROOT = PROJECT_ROOT / "scripts" / "extraction_data"
if str(EXTRACTION_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTRACTION_SCRIPT_ROOT))

from utils import PAGE_BOILERPLATE_PATTERNS, is_valid_uuid_like, normalize_label


DEFAULT_EXTRACTION_ROOT = PROJECT_ROOT / "data" / "extraction"
DEFAULT_REPORT_PATH = DEFAULT_EXTRACTION_ROOT / "qa_validation_report.json"
REQUIRED_TOP_LEVEL_SECTIONS = {
    "patient",
    "report",
    "results",
    "validation_report",
    "pages",
    "tables",
    "blocks",
    "images",
}
ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def load_document(document_path: Path) -> dict:
    return json.loads(document_path.read_text(encoding="utf-8"))


def discover_documents(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("*/document.json") if path.is_file())


def make_issue(severity: str, code: str, message: str) -> dict[str, str]:
    return {"severity": severity, "code": code, "message": message}


def add_issue(issues: list[dict], severity: str, code: str, message: str) -> None:
    issues.append(make_issue(severity, code, message))


def has_boilerplate(text: str) -> bool:
    normalized = normalize_label(text)
    return any(pattern in normalized for pattern in PAGE_BOILERPLATE_PATTERNS)


def is_iso_date(value: object) -> bool:
    return isinstance(value, str) and bool(ISO_DATE_PATTERN.fullmatch(value))


def evaluate_status(issues: list[dict]) -> str:
    severities = {issue["severity"] for issue in issues}
    if "FAIL" in severities:
        return "FAIL"
    if "WARNING" in severities:
        return "WARNING"
    return "PASS"


def validate_required_sections(document: dict, issues: list[dict]) -> None:
    missing = sorted(section for section in REQUIRED_TOP_LEVEL_SECTIONS if section not in document)
    if missing:
        add_issue(issues, "FAIL", "missing_top_level_sections", f"Missing sections: {', '.join(missing)}")


def validate_patient(document: dict, issues: list[dict]) -> None:
    patient = document.get("patient", {})
    patient_id = patient.get("patient_id")
    if not patient_id:
        add_issue(issues, "FAIL", "missing_patient_id", "patient.patient_id is missing")
    elif not is_valid_uuid_like(patient_id):
        add_issue(issues, "FAIL", "invalid_patient_id", f"patient.patient_id is not canonical UUID-like: {patient_id}")


def validate_report(document: dict, issues: list[dict]) -> None:
    report = document.get("report", {})
    report_date = report.get("report_date")
    if not report_date:
        add_issue(issues, "FAIL", "missing_report_date", "report.report_date is missing")
    elif not is_iso_date(report_date):
        add_issue(issues, "FAIL", "invalid_report_date", f"report.report_date is not ISO format: {report_date}")


def validate_result_fields(document: dict, issues: list[dict]) -> None:
    for index, result in enumerate(document.get("results", []), start=1):
        missing = [
            field
            for field in ("analyte", "value_raw", "value_numeric", "unit", "observation_date")
            if result.get(field) in {None, ""}
        ]
        if missing:
            add_issue(
                issues,
                "FAIL",
                "missing_result_fields",
                f"Result #{index} missing fields: {', '.join(missing)}",
            )
        observation_date = result.get("observation_date")
        if observation_date and not is_iso_date(observation_date):
            add_issue(
                issues,
                "FAIL",
                "invalid_observation_date",
                f"Result #{index} has non-ISO observation_date: {observation_date}",
            )


def validate_result_consistency(document: dict, issues: list[dict]) -> None:
    for index, result in enumerate(document.get("results", []), start=1):
        reference = result.get("reference_range") or {}
        low = reference.get("low")
        high = reference.get("high")
        value = result.get("value_numeric")
        alert_flag = result.get("alert_flag")
        is_abnormal = result.get("is_abnormal")

        if low is not None and high is not None and low > high:
            add_issue(
                issues,
                "FAIL",
                "invalid_reference_range",
                f"Result #{index} has low > high: {low} > {high}",
            )
            continue

        if value is None or low is None or high is None:
            continue

        if value > high:
            if is_abnormal is not True:
                add_issue(issues, "FAIL", "abnormal_flag_missing_high", f"Result #{index} should be abnormal above range")
            if alert_flag not in {"H", "HH"}:
                add_issue(issues, "FAIL", "alert_flag_inconsistent_high", f"Result #{index} alert_flag should indicate high")
        elif value < low:
            if is_abnormal is not True:
                add_issue(issues, "FAIL", "abnormal_flag_missing_low", f"Result #{index} should be abnormal below range")
            if alert_flag not in {"L", "LL"}:
                add_issue(issues, "FAIL", "alert_flag_inconsistent_low", f"Result #{index} alert_flag should indicate low")
        else:
            if is_abnormal is not False:
                add_issue(issues, "FAIL", "abnormal_flag_inconsistent_normal", f"Result #{index} should not be abnormal in range")
            if alert_flag not in {None, "", "-"}:
                add_issue(issues, "FAIL", "alert_flag_inconsistent_normal", f"Result #{index} alert_flag should be empty/inactive in range")


def validate_deduplication(document: dict, issues: list[dict]) -> None:
    results = document.get("results", [])
    validation_report = document.get("validation_report", {})
    dedup_keys = [result.get("dedup_key") for result in results if result.get("is_canonical")]
    if len(dedup_keys) != len(set(dedup_keys)):
        add_issue(issues, "FAIL", "duplicate_dedup_keys", "Canonical results share duplicate dedup_key values")

    raw_rows = validation_report.get("raw_result_rows")
    canonical_rows = validation_report.get("canonical_result_rows")
    removed = validation_report.get("duplicate_rows_removed")
    if all(isinstance(value, int) for value in (raw_rows, canonical_rows, removed)):
        if removed != raw_rows - canonical_rows:
            add_issue(
                issues,
                "FAIL",
                "invalid_dedup_counts",
                f"duplicate_rows_removed={removed} but raw-canonical={raw_rows - canonical_rows}",
            )
    else:
        add_issue(issues, "FAIL", "missing_dedup_counts", "validation_report deduplication counts are incomplete")


def validate_validation_report(document: dict, issues: list[dict]) -> None:
    validation_report = document.get("validation_report", {})
    result_consistency_issues = validation_report.get("result_consistency_issues") or []
    if result_consistency_issues:
        add_issue(
            issues,
            "WARNING",
            "validation_report_consistency_issues",
            f"validation_report contains {len(result_consistency_issues)} consistency issue(s)",
        )

    structured_vs_raw = validation_report.get("structured_vs_raw_tables") or {}
    missing_from_raw = structured_vs_raw.get("missing_from_raw_tables") or []
    if missing_from_raw:
        add_issue(
            issues,
            "WARNING",
            "missing_from_raw_tables",
            f"{len(missing_from_raw)} structured result(s) missing from raw tables",
        )


def detect_uniform_confidence(document: dict, issues: list[dict]) -> None:
    scores: list[float] = []
    for result in document.get("results", []):
        score = result.get("confidence_score")
        if isinstance(score, (int, float)):
            scores.append(round(float(score), 3))
    for block in document.get("blocks", []):
        score = block.get("confidence_score")
        if isinstance(score, (int, float)):
            scores.append(round(float(score), 3))
    if len(scores) >= 6 and len(set(scores)) <= 2:
        add_issue(issues, "WARNING", "uniform_confidence_scores", "Confidence scores are suspiciously uniform")


def detect_boilerplate(document: dict, issues: list[dict]) -> None:
    for page in document.get("pages", []):
        if has_boilerplate(page.get("final_text", "")):
            add_issue(
                issues,
                "WARNING",
                "page_boilerplate_present",
                f"Boilerplate still present in page {page.get('page_number')} final_text",
            )


def detect_repeated_indexable_block_text(document: dict, issues: list[dict]) -> None:
    seen: dict[str, list[str]] = {}
    for block in document.get("blocks", []):
        if not block.get("is_indexable"):
            continue
        text = block.get("text", "")
        normalized = normalize_label(text)
        if not normalized:
            continue
        block_ref = f"{block.get('block_type')}@p{block.get('page_number')}"
        seen.setdefault(normalized, []).append(block_ref)

    repeated = {text: refs for text, refs in seen.items() if len(refs) > 1}
    if repeated:
        sample_refs = next(iter(repeated.values()))
        add_issue(
            issues,
            "WARNING",
            "repeated_indexable_blocks",
            f"Repeated indexable block text detected across {', '.join(sample_refs)}",
        )


def validate_document(document_path: Path) -> dict:
    document = load_document(document_path)
    issues: list[dict] = []

    validate_required_sections(document, issues)
    validate_patient(document, issues)
    validate_report(document, issues)
    validate_result_fields(document, issues)
    validate_result_consistency(document, issues)
    validate_deduplication(document, issues)
    validate_validation_report(document, issues)
    detect_uniform_confidence(document, issues)
    detect_boilerplate(document, issues)
    detect_repeated_indexable_block_text(document, issues)

    status = evaluate_status(issues)
    return {
        "doc_id": document.get("doc_id", document_path.parent.name),
        "document_path": str(document_path),
        "status": status,
        "issue_count": len(issues),
        "issues": issues,
    }


def summarize(results: list[dict]) -> dict:
    counts = {"total_pass": 0, "total_warning": 0, "total_fail": 0}
    for result in results:
        status = result["status"]
        if status == "PASS":
            counts["total_pass"] += 1
        elif status == "WARNING":
            counts["total_warning"] += 1
        else:
            counts["total_fail"] += 1
    return counts


def print_summary(results: list[dict], counts: dict) -> None:
    print(f"{'STATUS':<10} {'DOC_ID':<60} {'ISSUES':>6}")
    print("-" * 80)
    for result in results:
        print(f"{result['status']:<10} {result['doc_id']:<60} {result['issue_count']:>6}")
    print("-" * 80)
    print(
        f"PASS={counts['total_pass']} WARNING={counts['total_warning']} FAIL={counts['total_fail']}"
    )


def write_report(report_path: Path, results: list[dict], counts: dict) -> Path:
    payload = {
        "root": str(report_path.parent),
        "summary": counts,
        "documents": results,
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QA validation for medical extraction outputs.")
    parser.add_argument(
        "--root",
        default=str(DEFAULT_EXTRACTION_ROOT),
        help="Root extraction folder containing per-document subfolders with document.json",
    )
    parser.add_argument(
        "--report-json",
        default=str(DEFAULT_REPORT_PATH),
        help="Path to write the machine-readable QA JSON report",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.root).expanduser().resolve()
    report_path = Path(args.report_json).expanduser().resolve()

    document_paths = discover_documents(root)
    results = [validate_document(path) for path in document_paths]
    counts = summarize(results)
    print_summary(results, counts)
    written_path = write_report(report_path, results, counts)
    print(f"\nQA report written to: {written_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
