from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
DOCS_DIR = PROJECT_ROOT / "docs"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "extraction"

EXTRACTION_SCRIPT_ROOT = PROJECT_ROOT / "scripts" / "extraction_data"

for path in (TESTS_DIR, EXTRACTION_SCRIPT_ROOT, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from run_extraction import run_pipeline
from validate_extraction_outputs import evaluate_status, make_issue, validate_document

def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _extra_checks(document: dict) -> list[dict]:
    """
    Additional validation rules on top of generic validation.
    """
    issues: list[dict] = []
    doc_type = document.get("document_type")

    # -------------------------------
    # Global critical checks
    # -------------------------------
    if not document.get("results"):
        issues.append(
            make_issue(
                "FAIL",
                "missing_results",
                "No structured results found in document",
            )
        )

    if not document.get("blocks"):
        issues.append(
            make_issue(
                "FAIL",
                "missing_blocks",
                "No blocks found in document",
            )
        )

    # -------------------------------
    # Parasitology-specific checks
    # -------------------------------
    if doc_type == "parasitology_stool_report":
        top_checks = document.get("consistency_checks")
        report_checks = document.get("validation_report", {}).get("consistency_checks")

        if not isinstance(report_checks, dict):
            issues.append(
                make_issue(
                    "WARNING",
                    "missing_validation_report_consistency_checks",
                    "validation_report.consistency_checks is missing for parasitology_stool_report",
                )
            )

        if not isinstance(top_checks, dict):
            issues.append(
                make_issue(
                    "WARNING",
                    "missing_consistency_checks",
                    "top-level consistency_checks is missing for parasitology_stool_report",
                )
            )

        if isinstance(top_checks, dict) and isinstance(report_checks, dict):
            if top_checks != report_checks:
                issues.append(
                    make_issue(
                        "FAIL",
                        "consistency_checks_mismatch",
                        "Top-level consistency_checks does not match validation_report.consistency_checks",
                    )
                )

        checks = report_checks if isinstance(report_checks, dict) else top_checks

        if not isinstance(checks, dict):
            return issues

        parasite = checks.get("parasite_consistency")

        if not isinstance(parasite, dict):
            issues.append(
                make_issue(
                    "WARNING",
                    "missing_parasite_consistency",
                    "parasite_consistency is missing",
                )
            )
            return issues

        status = parasite.get("status")
        detected = parasite.get("detected_in_sections")

        allowed_statuses = {
            "consistent",
            "needs_review",
            "partial_overlap",
            "discordant_findings",
        }

        if status == "inconsistent":
            issues.append(
                make_issue(
                    "FAIL",
                    "legacy_parasite_consistency_status",
                    "Legacy status 'inconsistent' should be replaced by 'discordant_findings', 'partial_overlap', or 'needs_review'",
                )
            )
        elif status not in allowed_statuses:
            issues.append(
                make_issue(
                    "FAIL",
                    "invalid_parasite_consistency_status",
                    f"Invalid parasite consistency status: {status}",
                )
            )

        if not isinstance(detected, dict) or not detected:
            issues.append(
                make_issue(
                    "WARNING",
                    "missing_detected_sections",
                    "detected_in_sections is missing or empty",
                )
            )
            return issues

        expected_sections = {
            "staining_exam",
            "enrichment_exam",
            "microscopic_exam",
            "final_result",
        }

        missing_sections = expected_sections - set(detected.keys())
        if missing_sections:
            issues.append(
                make_issue(
                    "WARNING",
                    "missing_detected_section_keys",
                    "detected_in_sections is missing keys: "
                    + ", ".join(sorted(missing_sections)),
                )
            )

        for section, values in detected.items():
            if not isinstance(values, list):
                issues.append(
                    make_issue(
                        "WARNING",
                        "invalid_detected_section_values",
                        f"detected_in_sections.{section} must be a list",
                    )
                )
                continue

            normalized_values = [
                value.lower().strip()
                for value in values
                if isinstance(value, str)
            ]

            if len(normalized_values) != len(set(normalized_values)):
                issues.append(
                    make_issue(
                        "WARNING",
                        "duplicate_detected_entities",
                        f"Duplicate parasite entities found in detected_in_sections.{section}",
                    )
                )

        final_values = {
            value.lower().strip()
            for value in detected.get("final_result", [])
            if isinstance(value, str)
        }

        prior_values: set[str] = set()
        for section in ("staining_exam", "enrichment_exam", "microscopic_exam"):
            prior_values.update(
                value.lower().strip()
                for value in detected.get(section, [])
                if isinstance(value, str)
            )

        if status == "discordant_findings" and final_values.intersection(prior_values):
            issues.append(
                make_issue(
                    "WARNING",
                    "discordance_has_overlap",
                    "Status is discordant_findings but final_result overlaps with previous parasite findings",
                )
            )

        if status == "consistent" and final_values and prior_values and final_values != prior_values:
            issues.append(
                make_issue(
                    "WARNING",
                    "false_consistent_parasite_status",
                    "Status is consistent but final_result differs from previous parasite findings",
                )
            )

    return issues


def _compute_quality_score(issues: list[dict]) -> int:
    """
    Compute a simple quality score based on issues.
    """
    score = 100

    for issue in issues:
        severity = issue.get("severity")

        if severity == "FAIL":
            score -= 20
        elif severity == "WARNING":
            score -= 10
        elif severity == "INFO":
            score -= 2

    return max(score, 0)


def main() -> int:
    pdfs = sorted(DOCS_DIR.glob("*.pdf"))

    if not pdfs:
        print(f"No PDFs found in {DOCS_DIR}")
        return 1

    results: list[dict] = []

    for pdf in pdfs:
        try:
            out_dir = run_pipeline(pdf, OUTPUT_ROOT)
            document_path = Path(out_dir) / "document.json"

            qa = validate_document(document_path)
            doc = _load_json(document_path)

            qa["source_pdf"] = doc.get("source_pdf")
            qa["document_type"] = doc.get("document_type")
            qa["document_path"] = str(document_path)

            # Apply additional validation rules
            extra_issues = _extra_checks(doc)
            qa["issues"].extend(extra_issues)

            qa["issue_count"] = len(qa["issues"])
            qa["status"] = evaluate_status(qa["issues"])
            qa["quality_score"] = _compute_quality_score(qa["issues"])

            results.append(qa)

        except Exception:
            results.append(
                {
                    "doc_id": pdf.stem,
                    "document_path": "",
                    "source_pdf": str(pdf),
                    "document_type": None,
                    "status": "FAIL",
                    "issue_count": 1,
                    "quality_score": 0,
                    "issues": [
                        make_issue(
                            "FAIL",
                            "extraction_exception",
                            "Extraction crashed:\n"
                            + traceback.format_exc(limit=20),
                        )
                    ],
                }
            )

    # -------------------------------
    # Compact report
    # -------------------------------
    print(
        f"{'STATUS':<10} {'PDF':<35} {'DOC_TYPE':<28} {'ISSUES':>6} {'SCORE':>6}"
    )
    print("-" * 100)

    for item in results:
        pdf_name = Path(
            str(item.get("source_pdf") or item.get("doc_id") or "")
        ).name
        doc_type = str(item.get("document_type") or "-")
        score = item.get("quality_score", 0)

        print(
            f"{item['status']:<10} {pdf_name:<35} {doc_type:<28} "
            f"{item['issue_count']:>6} {score:>6}"
        )

    print("-" * 100)

    failures = [item for item in results if item["status"] == "FAIL"]
    warnings = [item for item in results if item["status"] == "WARNING"]

    print(
        f"PASS={len(results) - len(failures) - len(warnings)} "
        f"WARNING={len(warnings)} FAIL={len(failures)}"
    )

    # -------------------------------
    # Detailed issues
    # -------------------------------
    if failures or warnings:
        print("\nDetails (first 8 issues per document):")

        for item in results:
            if item["status"] == "PASS":
                continue

            pdf_name = Path(
                str(item.get("source_pdf") or item.get("doc_id") or "")
            ).name

            print(
                f"\n[{item['status']}] {pdf_name} "
                f"({item.get('document_type')})"
            )
            print(f"document_path={item.get('document_path')}")

            for issue in (item.get("issues") or [])[:8]:
                print(
                    f"- {issue.get('severity')} "
                    f"{issue.get('code')}: {issue.get('message')}"
                )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())