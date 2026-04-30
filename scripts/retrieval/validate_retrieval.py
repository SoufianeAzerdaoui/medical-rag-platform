#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .config import DEFAULT_TEST_CASES_PATH, VALIDATION_REPORT_PATH
    from .models import RetrievalFilters, RetrievalResult
    from .search import SearchEngine
    from .context_builder import detect_exact_medical_entities
except ImportError:
    from config import DEFAULT_TEST_CASES_PATH, VALIDATION_REPORT_PATH
    from models import RetrievalFilters, RetrievalResult
    from search import SearchEngine
    from context_builder import detect_exact_medical_entities


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Test cases not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("retrieval test cases must be a non-empty JSON list")
    return data


def _build_filters(case: dict[str, Any]) -> RetrievalFilters:
    f = case.get("filters") or {}
    return RetrievalFilters(
        document_type=f.get("document_type"),
        doc_id=f.get("doc_id"),
        patient_id=f.get("patient_id"),
        sample_id=f.get("sample_id"),
        sample_type=f.get("sample_type"),
        request_date=f.get("request_date"),
        chunk_type=f.get("chunk_type"),
        source_pdf=f.get("source_pdf"),
    )


def _match_expected(case: dict[str, Any], result: RetrievalResult) -> bool:
    expected_doc_id = case.get("expected_doc_id")
    expected_chunk_contains = case.get("expected_chunk_contains")
    expected_chunk_type = case.get("expected_chunk_type")
    expected_doc_id_any = bool(case.get("expected_doc_id_any"))

    if expected_doc_id_any:
        return True

    if expected_doc_id and result.doc_id != expected_doc_id:
        return False

    if expected_chunk_type and result.chunk_type != expected_chunk_type:
        return False

    if expected_chunk_contains:
        blob = f"{result.text_preview}\n{result.text}".lower()
        if str(expected_chunk_contains).lower() not in blob:
            return False

    return True


def _first_relevant_rank(case: dict[str, Any], results: list[RetrievalResult], k: int) -> int | None:
    for i, r in enumerate(results[:k], start=1):
        if _match_expected(case, r):
            return i
    return None


def _compute_metrics(cases: list[dict[str, Any]], per_case_results: list[list[RetrievalResult]], top_k: int) -> dict[str, float]:
    n = len(cases)
    hit1 = 0
    hit3 = 0
    hit5 = 0
    mrr = 0.0

    for case, results in zip(cases, per_case_results):
        r1 = _first_relevant_rank(case, results, 1)
        r3 = _first_relevant_rank(case, results, min(3, top_k))
        r5 = _first_relevant_rank(case, results, min(5, top_k))
        rall = _first_relevant_rank(case, results, top_k)

        if r1 is not None:
            hit1 += 1
        if r3 is not None:
            hit3 += 1
        if r5 is not None:
            hit5 += 1
        if rall is not None:
            mrr += 1.0 / rall

    return {
        "recall_at_1": hit1 / n if n else 0.0,
        "recall_at_3": hit3 / n if n else 0.0,
        "recall_at_5": hit5 / n if n else 0.0,
        "mrr": mrr / n if n else 0.0,
    }


def _run_mode(
    engine: SearchEngine,
    *,
    mode: str,
    cases: list[dict[str, Any]],
    top_k: int,
) -> tuple[dict[str, float], list[dict[str, Any]], list[list[RetrievalResult]], list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    all_results: list[list[RetrievalResult]] = []
    eval_cases: list[dict[str, Any]] = []
    eval_results: list[list[RetrievalResult]] = []

    for i, case in enumerate(cases, start=1):
        query = str(case.get("query") or "").strip()
        query_type = str(case.get("query_type") or "").strip().lower() or "exact_entity"
        if not query:
            failures.append({"index": i, "query": "", "error": "empty query in test case"})
            all_results.append([])
            continue
        if mode == "keyword" and query_type == "semantic":
            skipped.append(
                {
                    "index": i,
                    "id": case.get("id"),
                    "query": query,
                    "reason": "semantic_query_skipped_for_keyword_mode",
                }
            )
            all_results.append([])
            continue

        try:
            response = engine.search(
                query=query,
                mode=mode,
                top_k=top_k,
                filters=_build_filters(case),
                expand_context=False,
            )
            results = response.top_results
            all_results.append(results)
            eval_cases.append(case)
            eval_results.append(results)

            if _first_relevant_rank(case, results, top_k) is None:
                failures.append(
                    {
                        "index": i,
                        "query": query,
                        "filters": case.get("filters") or {},
                        "expected": {
                            "expected_doc_id": case.get("expected_doc_id"),
                            "expected_chunk_contains": case.get("expected_chunk_contains"),
                            "expected_chunk_type": case.get("expected_chunk_type"),
                            "expected_doc_id_any": bool(case.get("expected_doc_id_any")),
                        },
                        "top_results": [
                            {
                                "rank": idx,
                                "chunk_id": r.chunk_id,
                                "doc_id": r.doc_id,
                                "chunk_type": r.chunk_type,
                                "preview": r.text_preview,
                            }
                            for idx, r in enumerate(results[:top_k], start=1)
                        ],
                    }
                )
        except Exception as exc:
            failures.append(
                {
                    "index": i,
                    "query": query,
                    "filters": case.get("filters") or {},
                    "error": str(exc),
                }
            )
            all_results.append([])

    metrics = _compute_metrics(eval_cases, eval_results, top_k=top_k)
    if skipped:
        metrics["evaluated_cases"] = float(len(eval_cases))
        metrics["skipped_cases"] = float(len(skipped))
    return metrics, failures, all_results, skipped


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate retrieval layer with test cases")
    parser.add_argument("--test-cases", type=Path, default=DEFAULT_TEST_CASES_PATH)
    parser.add_argument("--mode", choices=["keyword", "vector", "hybrid", "all"], default="all")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--index-dir", default="data/indexes")
    parser.add_argument("--collection", default="medical_chunks")
    parser.add_argument("--report", type=Path, default=VALIDATION_REPORT_PATH)
    return parser.parse_args()


def _contains_text(results: list[RetrievalResult], needle: str) -> bool:
    n = str(needle or "").lower().strip()
    if not n:
        return False
    for r in results:
        blob = f"{r.text}\n{r.text_preview}".lower()
        if n in blob:
            return True
    return False


def _run_context_checks(engine: SearchEngine, cases: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    exact_cases = 0
    explicit_entity_cases = 0
    precision_hits = 0
    precision_total = 0
    forbidden_leakage = 0
    entity_covered = 0
    related_evidence_cases = 0
    related_evidence_covered = 0
    required_chunk_type_cases = 0
    required_chunk_type_covered = 0
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    for i, case in enumerate(cases, start=1):
        query = str(case.get("query") or "").strip()
        strict_context = bool(case.get("strict_context", True))
        query_type = str(case.get("query_type") or "").strip().lower() or "exact_entity"
        exact_entities = detect_exact_medical_entities(query)
        expected_context_doc_ids = set(case.get("expected_context_doc_ids") or [])
        forbidden_context_doc_ids = set(case.get("forbidden_context_doc_ids") or [])
        expected_context_contains = case.get("expected_context_contains")
        expected_related_context_contains = case.get("expected_related_context_contains") or []
        expected_context_chunk_types = [str(x).lower() for x in (case.get("expected_context_chunk_types") or [])]

        if not query:
            continue
        if not strict_context:
            continue

        response = engine.search(
            query=query,
            mode="hybrid",
            top_k=top_k,
            filters=_build_filters(case),
            expand_context=True,
            strict_context=True,
            debug_context=False,
        )

        context = response.context_chunks
        context_doc_ids = [c.doc_id for c in context]

        if query_type == "exact_entity" and exact_entities:
            exact_cases += 1
            if expected_context_doc_ids:
                for c in context:
                    precision_total += 1
                    if c.doc_id in expected_context_doc_ids:
                        precision_hits += 1

            leaked = [d for d in context_doc_ids if d in forbidden_context_doc_ids]
            if leaked:
                forbidden_leakage += len(leaked)
                failures.append(
                    {
                        "index": i,
                        "query": query,
                        "error": "forbidden_context_doc_leakage",
                        "leaked_doc_ids": leaked,
                    }
                )

            if expected_context_contains:
                explicit_entity_cases += 1
                covered = _contains_text(context, str(expected_context_contains))
                if covered:
                    entity_covered += 1
                else:
                    failures.append(
                        {
                            "index": i,
                            "query": query,
                            "error": "explicit_entity_not_found_in_context",
                            "expected_context_contains": expected_context_contains,
                        }
                    )

            if expected_related_context_contains:
                related_evidence_cases += 1
                all_ok = True
                for token in expected_related_context_contains:
                    if not _contains_text(context, str(token)):
                        all_ok = False
                        failures.append(
                            {
                                "index": i,
                                "query": query,
                                "error": "related_evidence_not_found_in_context",
                                "missing_related_token": token,
                            }
                        )
                if all_ok:
                    related_evidence_covered += 1

            if expected_context_chunk_types:
                required_chunk_type_cases += 1
                present_types = {str(c.chunk_type or "").lower() for c in context}
                missing_types: list[str] = []
                for t in expected_context_chunk_types:
                    if t in present_types:
                        continue
                    # validation_status can be satisfied by explicit warning chunks in some corpora
                    if t == "validation_status" and "warning" in present_types:
                        continue
                    missing_types.append(t)
                if missing_types:
                    failures.append(
                        {
                            "index": i,
                            "query": query,
                            "error": "required_context_chunk_type_missing",
                            "missing_chunk_types": missing_types,
                            "present_chunk_types": sorted(present_types),
                        }
                    )
                else:
                    required_chunk_type_covered += 1

            # warning when raw top_results noisy but context clean
            raw_noisy = any(
                ("vector_match" in r.match_reason and "keyword_match" not in r.match_reason and r.doc_id in forbidden_context_doc_ids)
                for r in response.top_results
            )
            if raw_noisy and not leaked:
                warnings.append(
                    {
                        "index": i,
                        "query": query,
                        "warning": "raw_top_results_include_vector_only_unrelated_docs_but_context_is_clean",
                    }
                )

    context_precision = (precision_hits / precision_total) if precision_total else 1.0
    entity_coverage = (entity_covered / explicit_entity_cases) if explicit_entity_cases else 1.0
    related_coverage = (related_evidence_covered / related_evidence_cases) if related_evidence_cases else 1.0
    chunk_type_coverage = (required_chunk_type_covered / required_chunk_type_cases) if required_chunk_type_cases else 1.0

    return {
        "metrics": {
            "ContextPrecision": context_precision,
            "ForbiddenDocLeakage": forbidden_leakage,
            "ExplicitEntityCoverage": entity_coverage,
            "RelatedEvidenceCoverage": related_coverage,
            "RequiredChunkTypeCoverage": chunk_type_coverage,
        },
        "failures": failures,
        "warnings": warnings,
        "exact_entity_cases": exact_cases,
        "explicit_entity_cases": explicit_entity_cases,
        "related_evidence_cases": related_evidence_cases,
        "required_chunk_type_cases": required_chunk_type_cases,
    }


def main() -> int:
    args = _parse_args()
    cases = _load_cases(args.test_cases)

    index_dir = Path(args.index_dir)
    sqlite_path = index_dir / "medical_rag.sqlite"
    qdrant_dir = index_dir / "qdrant"

    modes = [args.mode] if args.mode != "all" else ["keyword", "vector", "hybrid"]

    report: dict[str, Any] = {
        "generated_at": utc_now_iso(),
        "test_cases_path": str(args.test_cases),
        "total_test_cases": len(cases),
        "top_k": args.top_k,
        "modes": {},
        "context_quality": {},
        "final_status": "FAIL",
        "notes": [],
    }

    engine = SearchEngine(
        sqlite_path=sqlite_path,
        qdrant_dir=qdrant_dir,
        collection=args.collection,
    )
    try:
        for mode in modes:
            metrics, failures, _, skipped = _run_mode(engine, mode=mode, cases=cases, top_k=args.top_k)
            report["modes"][mode] = {
                "metrics": metrics,
                "failures": failures,
                "failed_cases": len(failures),
                "skipped": skipped,
                "skipped_cases": len(skipped),
            }

        # Context quality checks are evaluated on hybrid strict context
        context_quality = _run_context_checks(engine, cases, args.top_k)
        report["context_quality"] = context_quality

        hybrid_metrics = report["modes"].get("hybrid", {}).get("metrics", {})
        recall3 = float(hybrid_metrics.get("recall_at_3", 0.0))
        recall5 = float(hybrid_metrics.get("recall_at_5", 0.0))
        forbidden_leakage = int(report["context_quality"].get("metrics", {}).get("ForbiddenDocLeakage", 0))
        explicit_coverage = float(report["context_quality"].get("metrics", {}).get("ExplicitEntityCoverage", 0.0))
        related_coverage = float(report["context_quality"].get("metrics", {}).get("RelatedEvidenceCoverage", 0.0))
        chunk_type_coverage = float(report["context_quality"].get("metrics", {}).get("RequiredChunkTypeCoverage", 0.0))

        if args.mode != "all" and args.mode != "hybrid":
            report["final_status"] = "PASS"
            report["notes"].append("Hybrid thresholds are evaluated only when hybrid mode runs.")
        else:
            if recall5 < 0.90:
                report["final_status"] = "FAIL"
            elif forbidden_leakage > 0:
                report["final_status"] = "FAIL"
            elif explicit_coverage < 1.0:
                report["final_status"] = "FAIL"
            elif related_coverage < 1.0:
                report["final_status"] = "FAIL"
            elif chunk_type_coverage < 1.0:
                report["final_status"] = "FAIL"
            elif recall3 < 0.85:
                report["final_status"] = "WARNING"
            else:
                critical_errors = any(
                    isinstance(f, dict) and "error" in f
                    for f in report["modes"].get("hybrid", {}).get("failures", [])
                )
                context_errors = bool(report["context_quality"].get("failures"))
                report["final_status"] = "FAIL" if (critical_errors or context_errors) else "PASS"

        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"FINAL_STATUS: {report['final_status']}")
        if "hybrid" in report["modes"]:
            hm = report["modes"]["hybrid"]["metrics"]
            print(f"Hybrid Recall@1: {hm['recall_at_1']:.3f}")
            print(f"Hybrid Recall@3: {hm['recall_at_3']:.3f}")
            print(f"Hybrid Recall@5: {hm['recall_at_5']:.3f}")
            print(f"Hybrid MRR: {hm['mrr']:.3f}")
        cq = report.get("context_quality", {}).get("metrics", {})
        if cq:
            print(f"ContextPrecision: {cq.get('ContextPrecision', 0.0):.3f}")
            print(f"ForbiddenDocLeakage: {int(cq.get('ForbiddenDocLeakage', 0))}")
            print(f"ExplicitEntityCoverage: {cq.get('ExplicitEntityCoverage', 0.0):.3f}")
            print(f"RelatedEvidenceCoverage: {cq.get('RelatedEvidenceCoverage', 0.0):.3f}")
            print(f"RequiredChunkTypeCoverage: {cq.get('RequiredChunkTypeCoverage', 0.0):.3f}")

        for mode in modes:
            failures = report["modes"][mode]["failures"]
            if failures:
                print(f"\n[{mode}] failures: {len(failures)}")
                for f in failures[:10]:
                    print(json.dumps(f, ensure_ascii=False))
            skipped = report["modes"][mode].get("skipped", [])
            if skipped:
                print(f"\n[{mode}] skipped: {len(skipped)}")
                for s in skipped[:10]:
                    print(json.dumps(s, ensure_ascii=False))
        cfail = report.get("context_quality", {}).get("failures", [])
        if cfail:
            print(f"\n[context_quality] failures: {len(cfail)}")
            for f in cfail[:10]:
                print(json.dumps(f, ensure_ascii=False))
        cwarn = report.get("context_quality", {}).get("warnings", [])
        if cwarn:
            print(f"\n[context_quality] warnings: {len(cwarn)}")
            for w in cwarn[:10]:
                print(json.dumps(w, ensure_ascii=False))

        return 0 if report["final_status"] in {"PASS", "WARNING"} else 1
    finally:
        engine.close()


if __name__ == "__main__":
    raise SystemExit(main())
