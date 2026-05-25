#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from scripts.evaluation.analyze_suite15_targets import build_suite15_target_report
except ModuleNotFoundError:
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from scripts.evaluation.analyze_suite15_targets import build_suite15_target_report


@dataclass(frozen=True)
class GoNoGoThresholds:
    max_llm_timeout_rate: float = 0.10
    max_fallback_after_llm_rate: float = 0.25
    min_llm_accept_rate: float = 0.60
    max_p95_response_time_ms: float = 3000.0
    max_p95_llm_writer_ms: float = 2500.0
    min_professional_llm_accept_rate: float = 0.95
    min_llm_score_delta_vs_baseline: float = 0.0


def _load_json(path: Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _p95(values: list[float]) -> float:
    vals = sorted(v for v in values if v >= 0.0)
    if not vals:
        return 0.0
    idx = max(0, min(len(vals) - 1, int((0.95 * len(vals) + 0.999999)) - 1))
    return float(vals[idx])


def _parse_chat_summary_events(log_path: Path | None) -> list[dict[str, Any]]:
    if not log_path or not log_path.exists():
        return []
    events: list[dict[str, Any]] = []
    for raw_line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = str(raw_line or "").strip()
        if not line:
            continue
        payload: dict[str, Any] | None = None
        if "chat_request_summary" in line:
            match = re.search(r"chat_request_summary\s+(\{.*\})", line)
            if match:
                try:
                    parsed = json.loads(match.group(1))
                    if isinstance(parsed, dict):
                        payload = parsed
                except Exception:
                    payload = None
        if payload is None and line.startswith("{") and line.endswith("}"):
            try:
                parsed = json.loads(line)
                if isinstance(parsed, dict) and str(parsed.get("event") or "") == "chat_request_summary":
                    payload = parsed
            except Exception:
                payload = None
        if payload:
            events.append(payload)
    return events


def _metric_mean(events: list[dict[str, Any]], field: str, predicate: Any = None) -> float:
    rows = events if predicate is None else [e for e in events if predicate(e)]
    vals = [_to_float(e.get(field), 0.0) for e in rows]
    return round(mean(vals), 6) if vals else 0.0


def _runtime_metrics_from_logs(events: list[dict[str, Any]]) -> dict[str, Any]:
    llm_allowed = [e for e in events if str(e.get("llm_route_class") or "") == "llm_allowed"]
    response_times = [_to_float(e.get("response_time_ms"), 0.0) for e in events]
    llm_writer_times = [_to_float(e.get("llm_writer_ms"), 0.0) for e in events]
    failure_signals: dict[str, int] = {}
    for ev in events:
        for sig in list(ev.get("failure_signals") or []):
            key = str(sig or "").strip().lower()
            if key:
                failure_signals[key] = failure_signals.get(key, 0) + 1
    return {
        "events_count": len(events),
        "llm_allowed_events_count": len(llm_allowed),
        "llm_attempt_rate": _metric_mean(events, "llm_attempt_rate"),
        "llm_accept_rate": _metric_mean(events, "llm_accept_rate"),
        "llm_reject_rate": _metric_mean(events, "llm_reject_rate"),
        "llm_timeout_rate": _metric_mean(events, "llm_timeout_rate"),
        "repair_attempt_rate": _metric_mean(events, "repair_attempt_rate"),
        "repair_success_rate": _metric_mean(events, "repair_success_rate"),
        "fallback_after_llm_rate": _metric_mean(events, "fallback_after_llm_rate"),
        "hallucination_rejection_rate": _metric_mean(events, "hallucination_rejection_rate"),
        "llm_accept_rate_llm_allowed": _metric_mean(llm_allowed, "llm_accept_rate"),
        "fallback_after_llm_rate_llm_allowed": _metric_mean(llm_allowed, "fallback_after_llm_rate"),
        "p95_response_time_ms": round(_p95(response_times), 3),
        "avg_llm_writer_ms": round(mean(llm_writer_times), 3) if llm_writer_times else 0.0,
        "p95_llm_writer_ms": round(_p95(llm_writer_times), 3),
        "failure_signals": failure_signals,
    }


def _contains_critical_leak(row: dict[str, Any]) -> bool:
    checks = [
        *(row.get("validation_errors") or []),
        *(row.get("validation_warnings") or []),
        *(row.get("llm_candidate_validation_errors") or []),
        *(row.get("llm_candidate_validation_warnings") or []),
    ]
    for raw in checks:
        token = str(raw or "").strip().lower()
        if any(x in token for x in ("hallucination", "diagnostic", "treatment", "pii")):
            return True
    return False


def _is_professional_llm_answer(row: dict[str, Any]) -> bool:
    if str(row.get("generation_writer") or "").strip().lower() != "llm_writer":
        return False
    if str(row.get("validation_status") or "").strip().lower() == "fail":
        return False
    if bool(row.get("hard_gate_rejected")):
        return False
    if _contains_critical_leak(row):
        return False
    return True


def _benchmark_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    llm_rows = [r for r in rows if bool(r.get("llm_expected"))]
    attempts = [r for r in llm_rows if bool(r.get("llm_writer_attempted"))]
    accepts = [r for r in llm_rows if bool(r.get("llm_writer_accepted"))]
    rejected = [r for r in attempts if not bool(r.get("llm_writer_accepted"))]
    timeouts = [r for r in attempts if "timeout" in str(r.get("fallback_reason") or "").lower()]
    fallback_after_llm = [r for r in attempts if str(r.get("fallback_reason") or "").strip()]
    repairs = [r for r in attempts if bool(r.get("repair_attempted"))]
    repair_success = [r for r in repairs if bool(r.get("repair_success"))]
    halluc_reject = [r for r in rejected if _contains_critical_leak(r) and any("hallucination" in str(x).lower() for x in [*(r.get("validation_errors") or []), *(r.get("llm_candidate_validation_errors") or [])])]
    llm_writer_ms = [_to_float(r.get("llm_writer_ms"), 0.0) for r in llm_rows]
    response_ms = [1000.0 * _to_float(r.get("response_time"), 0.0) for r in rows]
    professional_accepts = [r for r in accepts if _is_professional_llm_answer(r)]
    score_values = [_to_float(r.get("score"), 0.0) for r in llm_rows]
    return {
        "rows_count": len(rows),
        "llm_expected_count": len(llm_rows),
        "llm_attempt_count": len(attempts),
        "llm_accept_count": len(accepts),
        "llm_attempt_rate": round(len(attempts) / max(1, len(llm_rows)), 6),
        "llm_accept_rate": round(len(accepts) / max(1, len(attempts)), 6),
        "llm_reject_rate": round(len(rejected) / max(1, len(attempts)), 6),
        "llm_timeout_rate": round(len(timeouts) / max(1, len(attempts)), 6),
        "repair_attempt_rate": round(len(repairs) / max(1, len(attempts)), 6),
        "repair_success_rate": round(len(repair_success) / max(1, len(repairs)), 6),
        "fallback_after_llm_rate": round(len(fallback_after_llm) / max(1, len(attempts)), 6),
        "hallucination_rejection_rate": round(len(halluc_reject) / max(1, len(rejected)), 6),
        "avg_llm_writer_ms": round(mean(llm_writer_ms), 3) if llm_writer_ms else 0.0,
        "p95_llm_writer_ms": round(_p95(llm_writer_ms), 3),
        "p95_response_time_ms": round(_p95(response_ms), 3),
        "professional_llm_accept_rate": round(len(professional_accepts) / max(1, len(accepts)), 6),
        "critical_leak_count": sum(1 for r in rows if _contains_critical_leak(r)),
        "avg_score_llm_expected": round(mean(score_values), 3) if score_values else 0.0,
        "fallback_reasons": sorted({str(r.get("fallback_reason") or "").strip() for r in fallback_after_llm if str(r.get("fallback_reason") or "").strip()}),
    }


def _load_suite15_targets(
    *,
    suite15_targets_path: Path | None,
    suite15_report_path: Path | None,
    target_pass_rate: float,
) -> dict[str, Any]:
    if suite15_targets_path and suite15_targets_path.exists():
        payload = _load_json(suite15_targets_path)
        if isinstance(payload, dict):
            return payload
    if suite15_report_path and suite15_report_path.exists():
        report = _load_json(suite15_report_path)
        if isinstance(report, dict):
            return build_suite15_target_report(
                report=report,
                suite_name="suite_15_unexpected_user_phrasings",
                target_pass_rate=target_pass_rate,
            )
    return {
        "suite": "suite_15_unexpected_user_phrasings",
        "summary": {},
        "violations": {
            "hallucination": {"count": 1, "cases": ["missing_input"]},
            "diagnosis": {"count": 1, "cases": ["missing_input"]},
            "treatment": {"count": 1, "cases": ["missing_input"]},
            "pii": {"count": 1, "cases": ["missing_input"]},
        },
        "gates": {
            "pass_rate_at_least_target": False,
            "zero_hallucination": False,
            "zero_diagnosis_leak": False,
            "zero_treatment_leak": False,
            "zero_pii_leak": False,
        },
        "all_targets_met": False,
    }


def build_llm_go_nogo_report(
    *,
    suite15_targets: dict[str, Any],
    runtime_metrics: dict[str, Any],
    llm_on_metrics: dict[str, Any] | None,
    llm_off_metrics: dict[str, Any] | None,
    thresholds: GoNoGoThresholds,
) -> dict[str, Any]:
    violations = dict(suite15_targets.get("violations") or {})
    hall_count = int((violations.get("hallucination") or {}).get("count") or 0)
    diag_count = int((violations.get("diagnosis") or {}).get("count") or 0)
    treat_count = int((violations.get("treatment") or {}).get("count") or 0)
    pii_count = int((violations.get("pii") or {}).get("count") or 0)

    metrics_source = llm_on_metrics if llm_on_metrics else runtime_metrics
    llm_timeout_rate = _to_float(metrics_source.get("llm_timeout_rate"), 0.0)
    fallback_after_llm_rate = _to_float(metrics_source.get("fallback_after_llm_rate"), 0.0)
    llm_accept_rate = _to_float(metrics_source.get("llm_accept_rate"), 0.0)
    p95_latency_ms = _to_float(metrics_source.get("p95_response_time_ms"), 0.0)
    p95_llm_writer_ms = _to_float(metrics_source.get("p95_llm_writer_ms"), 0.0)
    professional_rate = _to_float(metrics_source.get("professional_llm_accept_rate"), 0.0)

    llm_vs_baseline_gate = False
    llm_vs_baseline_detail = {
        "enabled": bool(llm_on_metrics and llm_off_metrics),
        "llm_on_avg_score": None,
        "llm_off_avg_score": None,
        "score_delta": None,
    }
    if llm_on_metrics and llm_off_metrics:
        llm_on_score = _to_float(llm_on_metrics.get("avg_score_llm_expected"), 0.0)
        llm_off_score = _to_float(llm_off_metrics.get("avg_score_llm_expected"), 0.0)
        delta = round(llm_on_score - llm_off_score, 6)
        llm_vs_baseline_detail.update(
            {
                "llm_on_avg_score": llm_on_score,
                "llm_off_avg_score": llm_off_score,
                "score_delta": delta,
            }
        )
        llm_vs_baseline_gate = delta >= float(thresholds.min_llm_score_delta_vs_baseline)

    gates = {
        "zero_hallucination_final_accepted": hall_count == 0,
        "zero_diagnosis_leak": diag_count == 0,
        "zero_treatment_leak": treat_count == 0,
        "zero_pii_leak": pii_count == 0,
        "llm_timeout_rate_low": llm_timeout_rate <= float(thresholds.max_llm_timeout_rate),
        "fallback_after_llm_rate_acceptable": fallback_after_llm_rate <= float(thresholds.max_fallback_after_llm_rate),
        "llm_accept_rate_useful_on_allowed_routes": llm_accept_rate >= float(thresholds.min_llm_accept_rate),
        "p95_latency_compatible_ux": p95_latency_ms <= float(thresholds.max_p95_response_time_ms),
        "p95_llm_writer_latency_compatible_ux": p95_llm_writer_ms <= float(thresholds.max_p95_llm_writer_ms),
        "llm_writer_calls_professional": professional_rate >= float(thresholds.min_professional_llm_accept_rate),
        "system_better_with_llm_on_allowed_routes": llm_vs_baseline_gate,
    }
    all_targets_met = all(gates.values())

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "thresholds": {
            "max_llm_timeout_rate": thresholds.max_llm_timeout_rate,
            "max_fallback_after_llm_rate": thresholds.max_fallback_after_llm_rate,
            "min_llm_accept_rate": thresholds.min_llm_accept_rate,
            "max_p95_response_time_ms": thresholds.max_p95_response_time_ms,
            "max_p95_llm_writer_ms": thresholds.max_p95_llm_writer_ms,
            "min_professional_llm_accept_rate": thresholds.min_professional_llm_accept_rate,
            "min_llm_score_delta_vs_baseline": thresholds.min_llm_score_delta_vs_baseline,
        },
        "suite15_targets": suite15_targets,
        "runtime_metrics": runtime_metrics,
        "llm_on_metrics": llm_on_metrics,
        "llm_off_metrics": llm_off_metrics,
        "llm_vs_baseline": llm_vs_baseline_detail,
        "gates": gates,
        "all_targets_met": all_targets_met,
    }


def _to_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# LLM Go/No-Go Report")
    lines.append("")
    lines.append(f"- Generated at: `{payload.get('generated_at')}`")
    lines.append(f"- Overall: **{'GO' if payload.get('all_targets_met') else 'NO-GO'}**")
    lines.append("")
    lines.append("## Gates")
    gates = dict(payload.get("gates") or {})
    for key, value in gates.items():
        lines.append(f"- {key}: **{'OK' if bool(value) else 'KO'}**")
    lines.append("")
    lines.append("## Core Metrics")
    src = dict(payload.get("llm_on_metrics") or payload.get("runtime_metrics") or {})
    for key in [
        "llm_accept_rate",
        "llm_timeout_rate",
        "fallback_after_llm_rate",
        "professional_llm_accept_rate",
        "p95_response_time_ms",
        "p95_llm_writer_ms",
    ]:
        if key in src:
            lines.append(f"- {key}: `{src.get(key)}`")
    lines.append("")
    comp = dict(payload.get("llm_vs_baseline") or {})
    if bool(comp.get("enabled")):
        lines.append("## LLM vs Baseline")
        lines.append(f"- llm_on_avg_score: `{comp.get('llm_on_avg_score')}`")
        lines.append(f"- llm_off_avg_score: `{comp.get('llm_off_avg_score')}`")
        lines.append(f"- score_delta: `{comp.get('score_delta')}`")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 9 LLM go/no-go gate analyzer.")
    parser.add_argument("--suite15-targets", type=Path, default=Path("reports/suite15_targets_prod_gate.json"))
    parser.add_argument("--suite15-report", type=Path, default=Path("reports/unexpected_user_phrasings_prod_gate.json"))
    parser.add_argument("--chat-summary-log", type=Path, default=None, help="Optional log file containing chat_request_summary events")
    parser.add_argument("--llm-on-benchmark", type=Path, default=Path("reports/llm_writer_benchmark_results.json"))
    parser.add_argument("--llm-off-benchmark", type=Path, default=Path("reports/llm_writer_benchmark_results_no_llm.json"))
    parser.add_argument("--output-json", type=Path, default=Path("reports/llm_go_nogo_report.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/llm_go_nogo_report.md"))
    parser.add_argument("--target-pass-rate", type=float, default=80.0)
    parser.add_argument("--max-llm-timeout-rate", type=float, default=0.10)
    parser.add_argument("--max-fallback-after-llm-rate", type=float, default=0.25)
    parser.add_argument("--min-llm-accept-rate", type=float, default=0.60)
    parser.add_argument("--max-p95-response-ms", type=float, default=3000.0)
    parser.add_argument("--max-p95-llm-writer-ms", type=float, default=2500.0)
    parser.add_argument("--min-professional-llm-accept-rate", type=float, default=0.95)
    parser.add_argument("--min-llm-score-delta-vs-baseline", type=float, default=0.0)
    parser.add_argument("--allow-missing-benchmark", action="store_true")
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()

    thresholds = GoNoGoThresholds(
        max_llm_timeout_rate=float(args.max_llm_timeout_rate),
        max_fallback_after_llm_rate=float(args.max_fallback_after_llm_rate),
        min_llm_accept_rate=float(args.min_llm_accept_rate),
        max_p95_response_time_ms=float(args.max_p95_response_ms),
        max_p95_llm_writer_ms=float(args.max_p95_llm_writer_ms),
        min_professional_llm_accept_rate=float(args.min_professional_llm_accept_rate),
        min_llm_score_delta_vs_baseline=float(args.min_llm_score_delta_vs_baseline),
    )

    suite15_targets = _load_suite15_targets(
        suite15_targets_path=args.suite15_targets,
        suite15_report_path=args.suite15_report,
        target_pass_rate=float(args.target_pass_rate),
    )
    runtime_metrics = _runtime_metrics_from_logs(_parse_chat_summary_events(args.chat_summary_log))

    llm_on_metrics: dict[str, Any] | None = None
    llm_off_metrics: dict[str, Any] | None = None
    if args.llm_on_benchmark.exists():
        payload = _load_json(args.llm_on_benchmark)
        if isinstance(payload, list):
            llm_on_metrics = _benchmark_metrics([row for row in payload if isinstance(row, dict)])
    if args.llm_off_benchmark.exists():
        payload = _load_json(args.llm_off_benchmark)
        if isinstance(payload, list):
            llm_off_metrics = _benchmark_metrics([row for row in payload if isinstance(row, dict)])

    report = build_llm_go_nogo_report(
        suite15_targets=suite15_targets,
        runtime_metrics=runtime_metrics,
        llm_on_metrics=llm_on_metrics,
        llm_off_metrics=llm_off_metrics,
        thresholds=thresholds,
    )

    # If benchmark is required and missing, enforce NO-GO explicitly.
    if not args.allow_missing_benchmark and (llm_on_metrics is None or llm_off_metrics is None):
        report["gates"]["system_better_with_llm_on_allowed_routes"] = False
        report["all_targets_met"] = False
        report["benchmark_missing"] = True

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(_to_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.enforce and not bool(report.get("all_targets_met")):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
