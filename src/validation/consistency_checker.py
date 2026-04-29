from __future__ import annotations


def detect_result_consistency(document: dict) -> dict:
    """
    Return canonical consistency checks for the document.

    The clinical extraction layer is the source of truth and writes
    consistency checks inside:

        validation_report.consistency_checks

    This function must not recompute parasite consistency, otherwise the
    top-level consistency_checks can diverge from validation_report.
    """
    validation_report = document.get("validation_report", {})
    report_checks = validation_report.get("consistency_checks")

    if isinstance(report_checks, dict) and report_checks:
        return report_checks

    return {}