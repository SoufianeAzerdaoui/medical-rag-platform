from typing import Any
import re

def natural_report_sort_key(value: str) -> tuple[list[Any], str]:
    """
    Improved natural sort key that handles extensions and ensures base filenames 
    come before versions/copies.
    Example: report.pdf < report (1).pdf < report (10).pdf
    """
    # Remove extension for base comparison if present
    base = str(value)
    ext = ""
    if "." in base:
        parts = base.rsplit(".", 1)
        base = parts[0]
        ext = parts[1]
    
    # Split by digits
    chunks = [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', base)]
    
    # Returning a tuple (chunks, ext) ensures consistent sorting
    return chunks, ext

def build_report_range_label(report_filenames: list[str]) -> str:
    """
    Builds a human-friendly range label for a list of report filenames.
    - 0 report: "Aucun rapport"
    - 1 report: "filename"
    - 2 reports: "A, B"
    - 3+ reports: "First → Last"
    """
    if not report_filenames:
        return "Aucun rapport"
    
    # Sort reports naturally
    sorted_reports = sorted(report_filenames, key=natural_report_sort_key)
    count = len(sorted_reports)
    
    if count == 1:
        return sorted_reports[0]
    if count == 2:
        return f"{sorted_reports[0]}, {sorted_reports[1]}"
    
    # 3 or more: First → Last
    return f"{sorted_reports[0]} → {sorted_reports[-1]}"
