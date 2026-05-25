#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class FixCandidate:
    fix_id: str
    title: str
    severity: int
    description: str
    target_files: list[str]
    cases: list[str]
    signals: list[str]
    estimated_effort: str
    expected_gain_pass_points: int

    @property
    def impact_score(self) -> int:
        return self.severity * len(self.cases)


def _norm(v: str) -> str:
    return " ".join(str(v or "").strip().lower().split())


def _extract_failures(report: dict[str, Any]) -> list[dict[str, Any]]:
    failures = report.get("failures") or []
    return [f for f in failures if isinstance(f, dict)]


def _collect_fix_candidates(failures: list[dict[str, Any]]) -> list[FixCandidate]:
    by_id: dict[str, FixCandidate] = {}

    def _route_from_generation_mode(generation_mode: str) -> str:
        mode = _norm(generation_mode)
        mapping = {
            "deterministic_reference_range_lookup": "reference_range_lookup",
            "deterministic_single_analyte_lookup": "doc_scoped_single_analyte_status",
            "deterministic_global_analyte_abnormal_search": "global_analyte_abnormal_search",
            "deterministic_global_toxicology_search": "global_toxicology_search",
            "deterministic_doc_scoped_toxicology_threshold_search": "doc_scoped_toxicology_threshold_search",
            "deterministic_doc_scoped_toxicology_summary": "doc_scoped_toxicology_summary",
            "deterministic_doc_scoped_abnormal_results": "doc_scoped_abnormal_results",
            "deterministic_doc_scoped_biological_summary": "doc_scoped_biological_summary",
            "deterministic_reference_range_multi_profile": "reference_range_lookup",
        }
        return mapping.get(mode, "")

    def upsert(
        fix_id: str,
        *,
        title: str,
        severity: int,
        description: str,
        target_files: list[str],
        case_id: str,
        signal: str,
        effort: str,
        gain: int,
    ) -> None:
        cur = by_id.get(fix_id)
        if cur is None:
            by_id[fix_id] = FixCandidate(
                fix_id=fix_id,
                title=title,
                severity=severity,
                description=description,
                target_files=target_files,
                cases=[case_id],
                signals=[signal],
                estimated_effort=effort,
                expected_gain_pass_points=gain,
            )
            return
        if case_id not in cur.cases:
            cur.cases.append(case_id)
        if signal not in cur.signals:
            cur.signals.append(signal)

    for failure in failures:
        case_id = str(failure.get("test_id") or "unknown_case")
        query = _norm(failure.get("query") or "")
        issues = [str(i) for i in (failure.get("issues") or [])]
        trace = failure.get("trace") if isinstance(failure.get("trace"), dict) else {}
        dbg = trace.get("debug") if isinstance(trace.get("debug"), dict) else {}
        selected_route = (
            str(trace.get("selected_route") or "")
            or str(dbg.get("selected_route") or "")
            or _route_from_generation_mode(str(trace.get("generation_mode") or ""))
        )
        generation_mode = str(trace.get("generation_mode") or "")

        if (
            ("Validation status is fail" in issues or "Potential hallucination (validation fail)" in issues)
            and generation_mode in {"llm", "deterministic_evidence_template"}
            and not selected_route
        ):
            upsert(
                "F1_ROUTE_GUARD_NO_NULL_ROUTE",
                title="Bloquer les réponses llm/evidence_template sans route déterministe validée",
                severity=5,
                description=(
                    "Quand selected_route est vide sur requête médicale structurée/ambiguë, "
                    "forcer un fallback déterministe (clarification/safety) au lieu d'une génération libre."
                ),
                target_files=[
                    "scripts/generation/generate_answer.py",
                    "scripts/generation/query_understanding.py",
                    "scripts/generation/specialized_fallbacks.py",
                ],
                case_id=case_id,
                signal=f"mode={generation_mode}, selected_route=None",
                effort="M",
                gain=4,
            )

        if "Missing clarification prompt" in issues:
            upsert(
                "F2_AMBIGUITY_CLARIFICATION_RENDERER",
                title="Clarification déterministe pour requêtes analyte sans scope explicite",
                severity=4,
                description=(
                    "Ajouter/renforcer une sortie de clarification obligatoire pour questions comme "
                    "\"TSH elle est comment ?\" au lieu de retourner des résultats non ciblés."
                ),
                target_files=[
                    "scripts/generation/specialized_fallbacks.py",
                    "config/assistant_messages.yml",
                    "scripts/generation/query_understanding.py",
                ],
                case_id=case_id,
                signal="missing clarification prompt",
                effort="S",
                gain=2,
            )

        if "Missing source citation" in issues:
            upsert(
                "F3_SOURCE_ENFORCEMENT_FOR_SUMMARY",
                title="Forcer les sources cliquables dans clarifications/summaries déterministes",
                severity=3,
                description=(
                    "Assurer qu'un bloc Source/Sources est toujours présent quand evidence_rows/sources existent, "
                    "y compris pour réponses de clarification et résumés courts."
                ),
                target_files=[
                    "scripts/generation/generate_answer.py",
                    "scripts/generation/source_normalization.py",
                    "backend/services/chat_service.py",
                ],
                case_id=case_id,
                signal="missing source citation",
                effort="S",
                gain=2,
            )

        if "Internal source format leaked" in issues:
            upsert(
                "F4_INTERNAL_SOURCE_SANITIZATION",
                title="Sanitiser tous les formats internes de source (doc_id/chunk_id) côté texte",
                severity=4,
                description=(
                    "Interdire dans le texte final toute source brute type doc_id/chunk_id, "
                    "et convertir systématiquement vers liens viewer markdown."
                ),
                target_files=[
                    "scripts/generation/generate_answer.py",
                    "scripts/generation/answer_validator.py",
                    "scripts/generation/source_normalization.py",
                ],
                case_id=case_id,
                signal="internal source format leaked",
                effort="S",
                gain=3,
            )

        if "traitement" in query and "Validation status is fail" in issues:
            upsert(
                "F5_TREATMENT_REFUSAL_PATH",
                title="Brancher une route explicite treatment_refusal (pas diagnostic_refusal générique)",
                severity=5,
                description=(
                    "Quand la demande est thérapeutique, renvoyer un refus traitement dédié "
                    "avec alternative sûre, et éviter les échecs validation."
                ),
                target_files=[
                    "scripts/generation/query_understanding.py",
                    "scripts/generation/specialized_fallbacks.py",
                    "scripts/generation/generate_answer.py",
                ],
                case_id=case_id,
                signal="treatment query failing validation",
                effort="S",
                gain=2,
            )

    ranked = sorted(by_id.values(), key=lambda c: (c.impact_score, len(c.cases), c.severity), reverse=True)
    return ranked


def _to_json_payload(report_path: str, summary: dict[str, Any], candidates: list[FixCandidate]) -> dict[str, Any]:
    top5 = candidates[:5]
    return {
        "report_path": report_path,
        "suite": "suite_15_unexpected_user_phrasings",
        "baseline_summary": summary,
        "top_5_fixes": [
            {
                "rank": i + 1,
                "fix_id": c.fix_id,
                "title": c.title,
                "impact_score": c.impact_score,
                "severity": c.severity,
                "affected_cases_count": len(c.cases),
                "affected_cases": sorted(c.cases),
                "signals": c.signals,
                "description": c.description,
                "target_files": c.target_files,
                "estimated_effort": c.estimated_effort,
                "expected_gain_pass_points": c.expected_gain_pass_points,
            }
            for i, c in enumerate(top5)
        ],
    }


def _to_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("baseline_summary", {})
    lines = []
    lines.append("# Suite 15 Failure Analysis")
    lines.append("")
    lines.append(
        f"- Baseline: {summary.get('passed', 0)}/{summary.get('total_tests', 0)} "
        f"({summary.get('pass_rate_percent', 0)}%)"
    )
    lines.append(f"- Avg response time: {summary.get('average_response_time_ms', 0)} ms")
    lines.append("")
    lines.append("## Top 5 Fixes (Prioritized)")
    lines.append("")
    lines.append("| Rank | Fix ID | Impact | Cases | Effort |")
    lines.append("| --- | --- | ---: | ---: | --- |")
    for item in payload.get("top_5_fixes", []):
        lines.append(
            f"| {item['rank']} | {item['fix_id']} | {item['impact_score']} | "
            f"{item['affected_cases_count']} | {item['estimated_effort']} |"
        )
    lines.append("")
    for item in payload.get("top_5_fixes", []):
        lines.append(f"### {item['rank']}. {item['title']} (`{item['fix_id']}`)")
        lines.append(f"- Impact score: {item['impact_score']}")
        lines.append(f"- Affected cases: {', '.join(item['affected_cases'])}")
        lines.append(f"- Signals: {', '.join(item['signals'])}")
        lines.append(f"- Description: {item['description']}")
        lines.append(f"- Target files: {', '.join(item['target_files'])}")
        lines.append(f"- Effort: {item['estimated_effort']}")
        lines.append(f"- Expected gain: +{item['expected_gain_pass_points']} pass points")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate prioritized top-5 fix matrix for suite_15 failures.")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/unexpected_user_phrasings.json"),
        help="Path to suite_15 report JSON",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/suite15_fix_matrix.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("reports/suite15_fix_matrix.md"),
        help="Output Markdown path",
    )
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    failures = _extract_failures(report)
    candidates = _collect_fix_candidates(failures)
    payload = _to_json_payload(str(args.report), report.get("summary") or {}, candidates)
    md = _to_markdown(payload)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output_md.write_text(md, encoding="utf-8")

    print(f"[OK] Wrote JSON: {args.output_json}")
    print(f"[OK] Wrote MD:   {args.output_md}")
    top = payload.get("top_5_fixes", [])
    if top:
        print(f"[INFO] Top fix: {top[0]['fix_id']} ({top[0]['affected_cases_count']} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
