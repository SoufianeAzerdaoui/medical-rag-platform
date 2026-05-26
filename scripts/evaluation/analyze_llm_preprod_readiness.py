#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReadinessThresholds:
    max_temperature: float = 0.2
    min_offline_useful_score: float = 7.0
    min_required_llm_routes: int = 1
    max_allowed_llm_routes: int = 6


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    payload = _load_json(path)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        rows = payload.get("rows") or payload.get("results") or []
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "on"}


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


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _load_llm_route_policy() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    scripts_dir = root / "scripts"
    generation_dir = scripts_dir / "generation"
    for p in (str(root), str(scripts_dir), str(generation_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)
    module = importlib.import_module("policy_matrix")
    llm_routes = sorted(str(x) for x in set(getattr(module, "LLM_ALLOWED_ROUTES", set())))
    deterministic_only = sorted(str(x) for x in set(getattr(module, "DETERMINISTIC_ONLY_ROUTES", set())))
    return {
        "llm_allowed_routes": llm_routes,
        "deterministic_only_routes": deterministic_only,
    }


def _load_feature_flags_seed() -> set[str]:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module = importlib.import_module("backend.services.feature_flag_service")
    defaults = getattr(module, "DEFAULT_FLAGS", tuple())
    names: set[str] = set()
    for item in defaults:
        if isinstance(item, tuple) and item:
            names.add(str(item[0]))
    return names


def build_preprod_readiness_report(
    *,
    go_nogo_report: dict[str, Any],
    llm_on_rows: list[dict[str, Any]],
    llm_off_rows: list[dict[str, Any]],
    thresholds: ReadinessThresholds,
) -> dict[str, Any]:
    go_nogo_gates = dict(go_nogo_report.get("gates") or {})
    go_nogo_thresholds = dict(go_nogo_report.get("thresholds") or {})
    suite15 = dict(go_nogo_report.get("suite15_targets") or {})
    suite15_gates = dict(suite15.get("gates") or {})
    llm_on_metrics = dict(go_nogo_report.get("llm_on_metrics") or {})
    llm_off_metrics = dict(go_nogo_report.get("llm_off_metrics") or {})
    llm_vs_baseline = dict(go_nogo_report.get("llm_vs_baseline") or {})

    env_checks = {
        "model_configured_by_env": bool(str(os.getenv("MEDICAL_RAG_LLM_MODEL", "")).strip()),
        "timeout_configured_by_env": bool(str(os.getenv("MEDICAL_RAG_LLM_TIMEOUT", "")).strip()),
        "max_tokens_configured_by_env": bool(str(os.getenv("MEDICAL_RAG_LLM_MAX_TOKENS", "")).strip()),
        "temperature_configured_by_env": bool(str(os.getenv("MEDICAL_RAG_LLM_TEMPERATURE", "")).strip()),
    }
    llm_temperature = _to_float(os.getenv("MEDICAL_RAG_LLM_TEMPERATURE"), default=0.0)
    temperature_low_stable = llm_temperature <= float(thresholds.max_temperature)
    retry_policy_explicit = bool(str(os.getenv("MEDICAL_RAG_LLM_MAX_RETRY_ATTEMPTS", "")).strip())
    timeout_circuit_explicit = _to_bool(os.getenv("MEDICAL_RAG_LLM_TIMEOUT_CIRCUIT_ENABLED", "1"))

    required_observability_fields = {
        "selected_route",
        "generation_mode",
        "generation_writer",
        "validation_status",
        "quality_final_status",
        "llm_expected",
        "llm_writer_attempted",
        "llm_writer_accepted",
        "fallback_reason",
        "response_time",
        "sources_count",
    }
    all_rows = list(llm_on_rows or []) + list(llm_off_rows or [])
    observability_explicit = bool(
        all_rows
        and all(required_observability_fields.issubset(set(row.keys())) for row in all_rows)
    )

    feature_flags_seed = _load_feature_flags_seed()
    rollback_to_deterministic_only = "LLM_GLOBAL_ENABLED" in feature_flags_seed

    policy = _load_llm_route_policy()
    llm_allowed_routes = list(policy.get("llm_allowed_routes") or [])
    deterministic_only_routes = set(policy.get("deterministic_only_routes") or [])
    llm_routes_limited_explicit = (
        len(llm_allowed_routes) >= int(thresholds.min_required_llm_routes)
        and len(llm_allowed_routes) <= int(thresholds.max_allowed_llm_routes)
    )
    critical_routes_kept_deterministic = (
        "doc_scoped_single_analyte_status" in deterministic_only_routes
        and "reference_range_lookup" in deterministic_only_routes
    )

    off_validation_ok = [
        row for row in llm_off_rows
        if str(row.get("validation_status") or "").strip().lower() in {"pass", "warning"}
    ]
    off_answers_non_empty = [row for row in llm_off_rows if str(row.get("answer") or "").strip()]
    off_scores = [_to_float(row.get("score"), 0.0) for row in llm_off_rows]
    useful_without_llm = bool(
        llm_off_rows
        and len(off_validation_ok) == len(llm_off_rows)
        and len(off_answers_non_empty) == len(llm_off_rows)
        and _safe_mean(off_scores) >= float(thresholds.min_offline_useful_score)
    )

    on_critical_leak_count = sum(1 for row in llm_on_rows if _contains_critical_leak(row))
    off_critical_leak_count = sum(1 for row in llm_off_rows if _contains_critical_leak(row))
    llm_adds_style_not_truth_risk = on_critical_leak_count == 0 and off_critical_leak_count == 0

    llm_reject_rows = [
        row for row in llm_on_rows
        if bool(row.get("llm_expected"))
        and bool(str(row.get("fallback_reason") or "").strip())
    ]
    ux_coherent_on_llm_reject = bool(
        (not llm_reject_rows)
        or all(
            str(row.get("validation_status") or "").strip().lower() in {"pass", "warning"}
            and bool(str(row.get("answer") or "").strip())
            for row in llm_reject_rows
        )
    )

    sources_clickable_preserved = bool(
        llm_on_rows
        and all(
            (int(row.get("displayed_count") or 0) <= 0)
            or (int(row.get("sources_count") or 0) > 0)
            for row in llm_on_rows
        )
    )

    frontend_format_required_fields = {
        "question_id",
        "answer",
        "generation_mode",
        "generation_writer",
        "validation_status",
        "quality_final_status",
        "selected_route",
        "response_time",
    }
    frontend_format_stable = bool(
        llm_on_rows
        and all(frontend_format_required_fields.issubset(set(row.keys())) for row in llm_on_rows)
    )

    metrics_visible = bool(llm_on_metrics) and all(
        k in llm_on_metrics
        for k in [
            "llm_expected_count",
            "llm_attempt_rate",
            "llm_accept_rate",
            "llm_timeout_rate",
            "fallback_after_llm_rate",
            "avg_llm_writer_ms",
            "p95_llm_writer_ms",
            "p95_response_time_ms",
        ]
    )

    safety_tests_100 = all(
        bool(suite15_gates.get(k))
        for k in [
            "zero_hallucination",
            "zero_diagnosis_leak",
            "zero_treatment_leak",
            "zero_pii_leak",
        ]
    )

    system_stable_when_llm_fails = bool(go_nogo_gates.get("fallback_after_llm_rate_acceptable")) and ux_coherent_on_llm_reject
    quality_added_on_allowed_routes = bool(go_nogo_gates.get("system_better_with_llm_on_allowed_routes"))

    technical_checks = {
        **env_checks,
        "temperature_low_stable": temperature_low_stable,
        "retry_policy_explicit": retry_policy_explicit,
        "timeout_circuit_explicit": timeout_circuit_explicit,
        "no_implicit_unobserved_behavior": observability_explicit,
        "rollback_to_deterministic_only_available": rollback_to_deterministic_only,
    }
    product_checks = {
        "useful_response_without_llm": useful_without_llm,
        "llm_improves_style_not_truth_risk": llm_adds_style_not_truth_risk,
        "ux_coherent_when_llm_rejected": ux_coherent_on_llm_reject,
        "clickable_sources_preserved": sources_clickable_preserved,
        "frontend_format_stable": frontend_format_stable,
    }
    final_checks = {
        "llm_routes_limited_and_explicit": llm_routes_limited_explicit,
        "critical_routes_kept_deterministic": critical_routes_kept_deterministic,
        "guardrails_block_critical_drift": all(
            bool(go_nogo_gates.get(k))
            for k in [
                "zero_hallucination_final_accepted",
                "zero_diagnosis_leak",
                "zero_treatment_leak",
                "zero_pii_leak",
            ]
        ),
        "metrics_visible": metrics_visible,
        "safety_tests_100_percent": safety_tests_100,
        "stable_when_llm_fails": system_stable_when_llm_fails,
        "quality_added_by_llm_on_allowed_routes": quality_added_on_allowed_routes,
    }

    technical_pass = all(technical_checks.values())
    product_pass = all(product_checks.values())
    final_pass = all(final_checks.values())
    overall_ready = technical_pass and product_pass and final_pass

    blockers: list[str] = []
    for section_name, checks in (
        ("technical", technical_checks),
        ("product", product_checks),
        ("final", final_checks),
    ):
        for key, value in checks.items():
            if not bool(value):
                blockers.append(f"{section_name}:{key}")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "thresholds": {
            "max_temperature": thresholds.max_temperature,
            "min_offline_useful_score": thresholds.min_offline_useful_score,
            "min_required_llm_routes": thresholds.min_required_llm_routes,
            "max_allowed_llm_routes": thresholds.max_allowed_llm_routes,
            "go_nogo_thresholds": go_nogo_thresholds,
        },
        "inputs": {
            "llm_on_rows_count": len(llm_on_rows),
            "llm_off_rows_count": len(llm_off_rows),
            "llm_allowed_routes": llm_allowed_routes,
            "env": {
                "MEDICAL_RAG_LLM_MODEL": str(os.getenv("MEDICAL_RAG_LLM_MODEL", "")),
                "MEDICAL_RAG_LLM_TIMEOUT": str(os.getenv("MEDICAL_RAG_LLM_TIMEOUT", "")),
                "MEDICAL_RAG_LLM_MAX_TOKENS": str(os.getenv("MEDICAL_RAG_LLM_MAX_TOKENS", "")),
                "MEDICAL_RAG_LLM_TEMPERATURE": str(os.getenv("MEDICAL_RAG_LLM_TEMPERATURE", "")),
                "MEDICAL_RAG_LLM_MAX_RETRY_ATTEMPTS": str(os.getenv("MEDICAL_RAG_LLM_MAX_RETRY_ATTEMPTS", "")),
                "MEDICAL_RAG_LLM_TIMEOUT_CIRCUIT_ENABLED": str(os.getenv("MEDICAL_RAG_LLM_TIMEOUT_CIRCUIT_ENABLED", "")),
            },
        },
        "technical_readiness": {
            "checks": technical_checks,
            "all_pass": technical_pass,
        },
        "product_readiness": {
            "checks": product_checks,
            "all_pass": product_pass,
            "supporting_metrics": {
                "llm_on_metrics": llm_on_metrics,
                "llm_off_metrics": llm_off_metrics,
                "llm_vs_baseline": llm_vs_baseline,
                "llm_rejected_rows_count": len(llm_reject_rows),
                "on_critical_leak_count": on_critical_leak_count,
                "off_critical_leak_count": off_critical_leak_count,
            },
        },
        "final_decision": {
            "checks": final_checks,
            "all_pass": final_pass,
            "ready_for_preprod": overall_ready,
            "blocking_items": blockers,
        },
    }


def _to_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# LLM Pre-Production Readiness")
    lines.append("")
    lines.append(f"- Generated at: `{report.get('generated_at')}`")
    ready = bool((((report.get("final_decision") or {}).get("ready_for_preprod"))))
    lines.append(f"- Overall: **{'READY' if ready else 'NOT READY'}**")
    lines.append("")
    for section_key, title in (
        ("technical_readiness", "Technical Readiness"),
        ("product_readiness", "Product Readiness"),
        ("final_decision", "Final Decision"),
    ):
        section = dict(report.get(section_key) or {})
        checks = dict(section.get("checks") or {})
        lines.append(f"## {title}")
        for key, value in checks.items():
            lines.append(f"- {key}: **{'OK' if bool(value) else 'KO'}**")
        if section_key == "final_decision":
            blockers = list(section.get("blocking_items") or [])
            if blockers:
                lines.append("- blockers:")
                for item in blockers:
                    lines.append(f"  - `{item}`")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM pre-production readiness analyzer (technical + product + final decision).")
    parser.add_argument("--go-nogo-report", type=Path, default=Path("reports/llm_go_nogo_report.json"))
    parser.add_argument("--llm-on-benchmark", type=Path, default=Path("reports/llm_writer_benchmark_results.json"))
    parser.add_argument("--llm-off-benchmark", type=Path, default=Path("reports/llm_writer_benchmark_results_no_llm.json"))
    parser.add_argument("--output-json", type=Path, default=Path("reports/llm_preprod_readiness_report.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/llm_preprod_readiness_report.md"))
    parser.add_argument("--max-temperature", type=float, default=0.2)
    parser.add_argument("--min-offline-useful-score", type=float, default=7.0)
    parser.add_argument("--min-required-llm-routes", type=int, default=1)
    parser.add_argument("--max-allowed-llm-routes", type=int, default=6)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()

    if not args.go_nogo_report.exists():
        raise SystemExit(f"Missing go/no-go report: {args.go_nogo_report}")

    report = build_preprod_readiness_report(
        go_nogo_report=_load_json(args.go_nogo_report),
        llm_on_rows=_load_rows(args.llm_on_benchmark),
        llm_off_rows=_load_rows(args.llm_off_benchmark),
        thresholds=ReadinessThresholds(
            max_temperature=float(args.max_temperature),
            min_offline_useful_score=float(args.min_offline_useful_score),
            min_required_llm_routes=int(args.min_required_llm_routes),
            max_allowed_llm_routes=int(args.max_allowed_llm_routes),
        ),
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(_to_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.enforce and not bool((((report.get("final_decision") or {}).get("ready_for_preprod")))):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
