#!/usr/bin/env python3
"""
Test Results Analysis & Dashboard
Helps track RAG system quality over time and identify trends.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Any

def load_test_report(path: Path) -> dict[str, Any]:
    """Load test report JSON."""
    return json.loads(path.read_text(encoding="utf-8"))

def print_header(text: str, width: int = 80) -> None:
    """Print formatted header."""
    print(f"\n{'='*width}")
    print(f"{text.center(width)}")
    print(f"{'='*width}\n")

def print_section(text: str, width: int = 80) -> None:
    """Print formatted section."""
    print(f"\n{'-'*width}")
    print(f"{text}")
    print(f"{'-'*width}\n")

def analyze_single_report(report: dict[str, Any]) -> None:
    """Analyze and display single test report."""
    
    generated_at = report.get("generated_at", "unknown")
    summary = report.get("summary", {})
    by_suite = report.get("by_suite", {})
    failures = report.get("failures", [])
    recommendations = report.get("recommendations", [])
    
    # Overall summary
    print_header("RAG SYSTEM TEST RESULTS")
    print(f"Generated: {generated_at}\n")
    
    total = summary.get("total_tests", 0)
    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    pass_rate = summary.get("pass_rate_percent", 0)
    avg_score = summary.get("average_score", 0)
    avg_time = summary.get("average_response_time_ms", 0)
    
    # Color coded results
    if pass_rate >= 90:
        status_color = "🟢"
    elif pass_rate >= 75:
        status_color = "🟡"
    else:
        status_color = "🔴"
    
    print(f"Overall Status: {status_color}")
    print(f"  Total Tests:      {total}")
    print(f"  Passed:           {passed}")
    print(f"  Failed:           {failed}")
    print(f"  Pass Rate:        {pass_rate}%")
    print(f"  Average Score:    {avg_score}/100")
    print(f"  Response Time:    {avg_time:.0f}ms")
    
    # Per-suite breakdown
    print_section("BREAKDOWN BY SUITE")
    
    suite_data = []
    for suite_name, suite_stats in by_suite.items():
        suite_total = suite_stats.get("total", 0)
        suite_passed = suite_stats.get("passed", 0)
        suite_rate = suite_stats.get("pass_rate_percent", 0)
        suite_score = suite_stats.get("average_score", 0)
        
        if suite_rate >= 95:
            indicator = "✅"
        elif suite_rate >= 80:
            indicator = "⚠️"
        else:
            indicator = "❌"
        
        suite_data.append({
            "name": suite_name.replace("suite_", "").split("_")[0],
            "total": suite_total,
            "passed": suite_passed,
            "rate": suite_rate,
            "score": suite_score,
            "indicator": indicator,
            "full_name": suite_name
        })
    
    # Sort by pass rate
    suite_data.sort(key=lambda x: x["rate"])
    
    print(f"{'Suite':<20} {'Total':>6} {'Passed':>6} {'Rate':>6} {'Score':>6} {'Status'}")
    print("-" * 58)
    
    for suite in suite_data:
        print(
            f"{suite['name']:<20} {suite['total']:>6} {suite['passed']:>6} "
            f"{suite['rate']:>5}% {suite['score']:>6.1f}  {suite['indicator']}"
        )
    
    # Failed tests
    if failures:
        print_section(f"FAILURES ({len(failures)} tests)")
        
        for i, failure in enumerate(failures[:10], 1):  # Show first 10
            test_id = failure.get("test_id", "unknown")
            suite = failure.get("suite", "unknown")
            query = failure.get("query", "")
            issues = failure.get("issues", [])
            
            print(f"{i}. [{test_id}] {suite}")
            print(f"   Query: {query[:60]}...")
            print(f"   Issues:")
            for issue in issues:
                print(f"     - {issue}")
            print()
        
        if len(failures) > 10:
            print(f"... and {len(failures) - 10} more failures\n")
    
    # Recommendations
    if recommendations:
        print_section("RECOMMENDATIONS")
        for rec in recommendations:
            print(f"• {rec}")
        print()
    
    # Key metrics interpretation
    print_section("KEY METRICS INTERPRETATION")
    
    print("Response Time Performance:")
    if avg_time < 30000:
        print("  ✅ Excellent - System responds quickly")
    elif avg_time < 60000:
        print("  ⚠️ Acceptable - Within target range")
    else:
        print("  ❌ Slow - Exceeds 60s target, investigate bottlenecks")
    
    print("\nAccuracy Assessment:")
    if pass_rate >= 95:
        print("  ✅ Production-ready - High accuracy")
    elif pass_rate >= 85:
        print("  ⚠️ Good - Ready for limited production with monitoring")
    elif pass_rate >= 75:
        print("  ❌ Needs work - Not production-ready")
    else:
        print("  ❌ Critical issues - Requires significant fixes")
    
    print("\nQuality Score Meaning:")
    if avg_score >= 85:
        print("  ✅ High quality - Consistent good answers")
    elif avg_score >= 70:
        print("  ⚠️ Moderate - Some quality issues to address")
    else:
        print("  ❌ Low quality - Systematic improvements needed")

def compare_reports(old_path: Path, new_path: Path) -> None:
    """Compare two test reports to show improvement/regression."""
    
    old_report = load_test_report(old_path)
    new_report = load_test_report(new_path)
    
    print_header("TEST RESULTS COMPARISON")
    print(f"Previous: {old_report.get('generated_at', 'unknown')}")
    print(f"Current:  {new_report.get('generated_at', 'unknown')}\n")
    
    old_summary = old_report.get("summary", {})
    new_summary = new_report.get("summary", {})
    
    metrics = [
        ("Total Tests", "total_tests", ""),
        ("Passed", "passed", ""),
        ("Pass Rate %", "pass_rate_percent", "%"),
        ("Average Score", "average_score", "/100"),
        ("Response Time (ms)", "average_response_time_ms", "ms"),
    ]
    
    print(f"{'Metric':<25} {'Previous':>12} {'Current':>12} {'Change':>12}")
    print("-" * 65)
    
    for metric_name, key, unit in metrics:
        old_val = old_summary.get(key, 0)
        new_val = new_summary.get(key, 0)
        
        # Calculate change
        if old_val != 0:
            if key == "average_response_time_ms":
                change = new_val - old_val  # Lower is better
                change_pct = (change / old_val) * 100
                change_indicator = "↓" if change < 0 else "↑"
            else:
                change_pct = ((new_val - old_val) / old_val) * 100
                change_indicator = "↑" if change_pct > 0 and key != "average_response_time_ms" else "↓"
        else:
            change_pct = 0
            change_indicator = "-"
        
        # Format for display
        old_str = f"{old_val:.1f}{unit}" if isinstance(old_val, float) else f"{old_val}{unit}"
        new_str = f"{new_val:.1f}{unit}" if isinstance(new_val, float) else f"{new_val}{unit}"
        change_str = f"{change_indicator} {change_pct:+.1f}%"
        
        print(f"{metric_name:<25} {old_str:>12} {new_str:>12} {change_str:>12}")
    
    # Suite-level comparison
    print_section("SUITE-LEVEL CHANGES")
    
    old_suites = old_report.get("by_suite", {})
    new_suites = new_report.get("by_suite", {})
    
    improved = []
    regressed = []
    maintained = []
    
    for suite_name in new_suites:
        new_rate = new_suites[suite_name].get("pass_rate_percent", 0)
        old_rate = old_suites.get(suite_name, {}).get("pass_rate_percent", 0)
        
        change = new_rate - old_rate
        
        if change > 5:
            improved.append((suite_name, old_rate, new_rate, change))
        elif change < -5:
            regressed.append((suite_name, old_rate, new_rate, change))
        else:
            maintained.append((suite_name, old_rate, new_rate, change))
    
    if improved:
        print("✅ IMPROVED:")
        for name, old, new, change in improved:
            print(f"  {name}: {old:.0f}% → {new:.0f}% (+{change:.0f}%)")
    
    if maintained:
        print("\n➡️  MAINTAINED:")
        for name, old, new, change in maintained[:5]:  # Show first 5
            print(f"  {name}: {old:.0f}%")
    
    if regressed:
        print("\n❌ REGRESSED:")
        for name, old, new, change in regressed:
            print(f"  {name}: {old:.0f}% → {new:.0f}% ({change:.0f}%)")

def main() -> int:
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Test Results Analyzer")
    parser.add_argument(
        "report",
        type=Path,
        nargs="?",
        default=Path("reports/rag_test_report.json"),
        help="Test report JSON file"
    )
    parser.add_argument(
        "--compare",
        type=Path,
        help="Compare with another report"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    
    args = parser.parse_args()
    
    # Check if report exists
    if not args.report.exists():
        print(f"[ERROR] Report not found: {args.report}")
        print("\nRun tests first:")
        print("  python3 scripts/evaluation/comprehensive_rag_tester.py")
        return 1
    
    try:
        if args.compare:
            if not args.compare.exists():
                print(f"[ERROR] Comparison report not found: {args.compare}")
                return 1
            compare_reports(args.compare, args.report)
        else:
            report = load_test_report(args.report)
            if args.json:
                print(json.dumps(report, indent=2))
            else:
                analyze_single_report(report)
    
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
