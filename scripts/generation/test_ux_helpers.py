import re
from typing import Any

def natural_report_sort_key(value: str) -> list[Any]:
    """
    Improved natural sort key that handles extensions and ensures base filenames 
    come before versions/copies.
    Example: report.pdf < report (1).pdf < report (10).pdf
    """
    # Remove extension for base comparison if present
    base = value
    ext = ""
    if "." in value:
        parts = value.rsplit(".", 1)
        base = parts[0]
        ext = parts[1]
    
    # Split by digits
    chunks = [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', base)]
    
    # Add a penalty for non-alphanumeric separators to ensure 'report' < 'report (1)'
    # and 'report' < 'report_1'
    # We use a tuple (chunks, ext) for sorting
    return chunks, ext

def build_report_range_label(reports: list[str]) -> str:
    """
    Builds a human-friendly range label for a list of reports.
    """
    if not reports:
        return "Aucun rapport"
    
    # Sort reports naturally
    sorted_reports = sorted(reports, key=natural_report_sort_key)
    count = len(sorted_reports)
    
    if count == 1:
        return sorted_reports[0]
    if count == 2:
        return f"{sorted_reports[0]}, {sorted_reports[1]}"
    
    # 3 or more: First → Last
    return f"{sorted_reports[0]} → {sorted_reports[-1]}"

# Tests
def test_sort():
    input_list = ["report (1).pdf", "report (10).pdf", "report (2).pdf", "report.pdf"]
    expected = ["report.pdf", "report (1).pdf", "report (2).pdf", "report (10).pdf"]
    actual = sorted(input_list, key=natural_report_sort_key)
    print(f"Sort Test: {'PASS' if actual == expected else 'FAIL'}")
    print(f"  Actual: {actual}")

def test_range():
    # Test 1
    reports1 = ["report.pdf"] + [f"report ({i}).pdf" for i in range(1, 16)]
    expected1 = "report.pdf → report (15).pdf"
    actual1 = build_report_range_label(reports1)
    print(f"Range Test 1: {'PASS' if actual1 == expected1 else 'FAIL'}")
    print(f"  Actual: {actual1}")

    # Test 2
    reports2 = [f"report ({i}).pdf" for i in range(16, 32)]
    expected2 = "report (16).pdf → report (31).pdf"
    actual2 = build_report_range_label(reports2)
    print(f"Range Test 2: {'PASS' if actual2 == expected2 else 'FAIL'}")
    print(f"  Actual: {actual2}")

if __name__ == "__main__":
    test_sort()
    test_range()
