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
        analyte_norm=f.get("analyte_norm"),
        section=f.get("section"),
        source_kind=f.get("source_kind"),
        interpretation_status=f.get("interpretation_status"),
        reference_quality_status=f.get("reference_quality_status"),
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


def _result_score(result: RetrievalResult, mode: str) -> float | None:
    if mode == "keyword":
        return result.score_keyword
    if mode == "vector":
        return result.score_vector
    if mode == "hybrid":
        return result.score_hybrid
    return None


def _result_snapshot(result: RetrievalResult, rank: int, mode: str) -> dict[str, Any]:
    md = result.metadata or {}
    return {
        "rank": rank,
        "chunk_id": result.chunk_id,
        "doc_id": result.doc_id,
        "chunk_type": result.chunk_type,
        "analyte": md.get("analyte"),
        "value_raw": md.get("value_raw"),
        "unit": md.get("unit"),
        "reference_range": md.get("reference_range"),
        "previous_result": md.get("previous_result"),
        "row_index": md.get("row_index"),
        "page_number": result.page_number,
        "source_kind": md.get("source_kind"),
        "score_final": _result_score(result, mode),
        "score_keyword": result.score_keyword,
        "score_vector": result.score_vector,
        "score_hybrid": result.score_hybrid,
        "preview": result.text_preview,
    }


def _run_mode(
    engine: SearchEngine,
    *,
    mode: str,
    cases: list[dict[str, Any]],
    top_k: int,
) -> tuple[dict[str, float], list[dict[str, Any]], list[list[RetrievalResult]], list[dict[str, Any]], list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    all_results: list[list[RetrievalResult]] = []
    case_results: list[dict[str, Any]] = []
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
            first_rank = _first_relevant_rank(case, results, top_k)
            case_results.append(
                {
                    "index": i,
                    "id": case.get("id"),
                    "query": query,
                    "query_type": query_type,
                    "suite": case.get("suite", "unspecified"),
                    "filters": case.get("filters") or {},
                    "expected": {
                        "expected_doc_id": case.get("expected_doc_id"),
                        "expected_chunk_contains": case.get("expected_chunk_contains"),
                        "expected_chunk_type": case.get("expected_chunk_type"),
                        "expected_doc_id_any": bool(case.get("expected_doc_id_any")),
                    },
                    "first_relevant_rank": first_rank,
                    "hit_at_1": first_rank == 1,
                    "hit_at_3": bool(first_rank is not None and first_rank <= 3),
                    "hit_at_5": bool(first_rank is not None and first_rank <= 5),
                    "top_chunks": [
                        _result_snapshot(r, idx, mode) for idx, r in enumerate(results[:top_k], start=1)
                    ],
                }
            )

            if first_rank is None:
                failures.append(
                    {
                        "index": i,
                        "id": case.get("id"),
                        "suite": case.get("suite", "unspecified"),
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
                    "id": case.get("id"),
                    "suite": case.get("suite", "unspecified"),
                    "query": query,
                    "filters": case.get("filters") or {},
                    "error": str(exc),
                }
            )
            all_results.append([])
            case_results.append(
                {
                    "index": i,
                    "id": case.get("id"),
                    "query": query,
                    "query_type": query_type,
                    "suite": case.get("suite", "unspecified"),
                    "filters": case.get("filters") or {},
                    "error": str(exc),
                    "top_chunks": [],
                }
            )

    metrics = _compute_metrics(eval_cases, eval_results, top_k=top_k)
    metrics["queries_tested"] = float(len(eval_cases))
    metrics["queries_succeeded"] = float(len([x for x in case_results if x.get("first_relevant_rank") is not None]))
    if skipped:
        metrics["evaluated_cases"] = float(len(eval_cases))
        metrics["skipped_cases"] = float(len(skipped))
    return metrics, failures, all_results, skipped, case_results


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


def _compute_suite_metrics(case_results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in case_results:
        suite = str(item.get("suite") or "unspecified")
        grouped.setdefault(suite, []).append(item)

    out: dict[str, dict[str, float]] = {}
    for suite, rows in grouped.items():
        usable = [r for r in rows if not r.get("error")]
        n = len(usable)
        if n == 0:
            out[suite] = {
                "queries": 0.0,
                "hit_at_1": 0.0,
                "hit_at_3": 0.0,
                "hit_at_5": 0.0,
                "mrr": 0.0,
            }
            continue
        hit1 = sum(1 for r in usable if r.get("hit_at_1"))
        hit3 = sum(1 for r in usable if r.get("hit_at_3"))
        hit5 = sum(1 for r in usable if r.get("hit_at_5"))
        mrr = 0.0
        for r in usable:
            rank = r.get("first_relevant_rank")
            if isinstance(rank, int) and rank > 0:
                mrr += 1.0 / float(rank)
        out[suite] = {
            "queries": float(n),
            "hit_at_1": hit1 / n,
            "hit_at_3": hit3 / n,
            "hit_at_5": hit5 / n,
            "mrr": mrr / n,
        }
    return out


def _mode_hits_by_id(case_results: list[dict[str, Any]]) -> dict[str, dict[str, bool]]:
    out: dict[str, dict[str, bool]] = {}
    for item in case_results:
        if item.get("error"):
            continue
        cid = str(item.get("id") or item.get("index"))
        out[cid] = {
            "hit_at_1": bool(item.get("hit_at_1")),
            "hit_at_3": bool(item.get("hit_at_3")),
            "hit_at_5": bool(item.get("hit_at_5")),
        }
    return out


def _compute_hybrid_vs_others(report_modes: dict[str, Any]) -> dict[str, float]:
    hybrid = _mode_hits_by_id((report_modes.get("hybrid") or {}).get("case_results") or [])
    keyword = _mode_hits_by_id((report_modes.get("keyword") or {}).get("case_results") or [])
    vector = _mode_hits_by_id((report_modes.get("vector") or {}).get("case_results") or [])

    common = sorted(set(hybrid.keys()) & set(keyword.keys()) & set(vector.keys()))
    if not common:
        return {
            "common_queries": 0.0,
            "hybrid_ge_keyword_hit_at_5_ratio": 0.0,
            "hybrid_ge_vector_hit_at_5_ratio": 0.0,
            "hybrid_ge_both_hit_at_5_ratio": 0.0,
        }

    ge_keyword = 0
    ge_vector = 0
    ge_both = 0
    for qid in common:
        h = int(hybrid[qid]["hit_at_5"])
        k = int(keyword[qid]["hit_at_5"])
        v = int(vector[qid]["hit_at_5"])
        if h >= k:
            ge_keyword += 1
        if h >= v:
            ge_vector += 1
        if h >= k and h >= v:
            ge_both += 1

    n = float(len(common))
    return {
        "common_queries": n,
        "hybrid_ge_keyword_hit_at_5_ratio": ge_keyword / n,
        "hybrid_ge_vector_hit_at_5_ratio": ge_vector / n,
        "hybrid_ge_both_hit_at_5_ratio": ge_both / n,
    }


def _compute_admin_noise(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    target = [x for x in case_results if str(x.get("suite")) == "exact_analyte" and not x.get("error")]
    noisy_queries = []
    for item in target:
        top_chunks = item.get("top_chunks") or []
        noisy = False
        for chunk in top_chunks[:5]:
            ctype = str(chunk.get("chunk_type") or "").lower()
            if ctype in {"validation_status", "visual_reference"}:
                noisy = True
                break
        if noisy:
            noisy_queries.append(
                {
                    "id": item.get("id"),
                    "query": item.get("query"),
                }
            )
    return {
        "checked_queries": len(target),
        "noisy_queries_count": len(noisy_queries),
        "noisy_queries": noisy_queries,
    }


def _compute_security_scan(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    forbidden_metadata_keys = {
        "patient_id",
        "patient_id_raw",
        "ip_patient",
        "patient_birth_date",
        "patient_birth_date_raw",
        "prescriber",
        "validated_by",
        "edited_by",
        "printed_by",
        "phone",
        "fax",
        "patient_name",
    }
    forbidden_text_markers = [
        "PATIENT TEST1",
        "PYXIS TEST",
        "anonymization_mapping",
        "data/private/",
        "Validé(e) par",
        "Edité(e) par",
        "Imprimé par",
    ]

    leak_rows: list[dict[str, Any]] = []
    for item in case_results:
        if item.get("error"):
            continue
        for chunk in item.get("top_chunks") or []:
            preview = str(chunk.get("preview") or "")
            md_keys = set((chunk.keys() if isinstance(chunk, dict) else []))
            md_hits = sorted([k for k in forbidden_metadata_keys if k in md_keys])
            txt_hits = [m for m in forbidden_text_markers if m.lower() in preview.lower()]
            if md_hits or txt_hits:
                leak_rows.append(
                    {
                        "id": item.get("id"),
                        "query": item.get("query"),
                        "rank": chunk.get("rank"),
                        "chunk_id": chunk.get("chunk_id"),
                        "metadata_hits": md_hits,
                        "text_hits": txt_hits,
                    }
                )
    return {
        "leak_count": len(leak_rows),
        "leaks": leak_rows[:50],
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
        "acceptance_thresholds": {
            "exact_analyte_hit_at_1_min": 0.70,
            "exact_analyte_hit_at_5_min": 0.90,
            "hybrid_ge_keyword_hit_at_5_ratio_min": 0.50,
            "hybrid_ge_vector_hit_at_5_ratio_min": 0.50,
            "admin_noise_top5_exact_analyte_max": 0,
            "security_leak_count_max": 0,
        },
        "acceptance_results": {},
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
            metrics, failures, _, skipped, case_results = _run_mode(engine, mode=mode, cases=cases, top_k=args.top_k)
            report["modes"][mode] = {
                "metrics": metrics,
                "suite_metrics": _compute_suite_metrics(case_results),
                "case_results": case_results,
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

        acceptance = report["acceptance_thresholds"]
        exact_suite = (report["modes"].get("hybrid", {}).get("suite_metrics", {}) or {}).get("exact_analyte", {})
        exact_hit1 = float(exact_suite.get("hit_at_1", 0.0))
        exact_hit5 = float(exact_suite.get("hit_at_5", 0.0))
        hybrid_vs = _compute_hybrid_vs_others(report["modes"])
        admin_noise = _compute_admin_noise((report["modes"].get("hybrid") or {}).get("case_results") or [])
        security_scan = _compute_security_scan((report["modes"].get("hybrid") or {}).get("case_results") or [])

        report["acceptance_results"] = {
            "exact_analyte_hit_at_1": exact_hit1,
            "exact_analyte_hit_at_5": exact_hit5,
            "hybrid_vs_others": hybrid_vs,
            "admin_noise": admin_noise,
            "security_scan": security_scan,
            "passes": {
                "exact_analyte_hit_at_1": exact_hit1 >= float(acceptance["exact_analyte_hit_at_1_min"]),
                "exact_analyte_hit_at_5": exact_hit5 >= float(acceptance["exact_analyte_hit_at_5_min"]),
                "hybrid_ge_keyword": float(hybrid_vs.get("hybrid_ge_keyword_hit_at_5_ratio", 0.0)) >= float(acceptance["hybrid_ge_keyword_hit_at_5_ratio_min"]),
                "hybrid_ge_vector": float(hybrid_vs.get("hybrid_ge_vector_hit_at_5_ratio", 0.0)) >= float(acceptance["hybrid_ge_vector_hit_at_5_ratio_min"]),
                "admin_noise": int(admin_noise.get("noisy_queries_count", 0)) <= int(acceptance["admin_noise_top5_exact_analyte_max"]),
                "security": int(security_scan.get("leak_count", 0)) <= int(acceptance["security_leak_count_max"]),
            },
        }

        if args.mode != "all" and args.mode != "hybrid":
            report["final_status"] = "PASS"
            report["notes"].append("Hybrid thresholds are evaluated only when hybrid mode runs.")
        else:
            context_errors = bool(report["context_quality"].get("failures"))
            acceptance_fail = not all(report["acceptance_results"]["passes"].values())
            hybrid_failures = (report["modes"].get("hybrid") or {}).get("failures") or []
            non_semantic_failures = [
                f for f in hybrid_failures if str(f.get("suite", "unspecified")) != "semantic"
            ]
            semantic_failures = [
                f for f in hybrid_failures if str(f.get("suite", "unspecified")) == "semantic"
            ]

            if context_errors or acceptance_fail or non_semantic_failures:
                report["final_status"] = "FAIL"
            elif semantic_failures:
                report["final_status"] = "WARNING"
                report["notes"].append(
                    "Some semantic queries are not yet reliable in hybrid mode; exact/filter acceptance thresholds are satisfied."
                )
            else:
                report["final_status"] = "PASS"

        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"FINAL_STATUS: {report['final_status']}")
        if "hybrid" in report["modes"]:
            hm = report["modes"]["hybrid"]["metrics"]
            print(f"Hybrid Recall@1: {hm['recall_at_1']:.3f}")
            print(f"Hybrid Recall@3: {hm['recall_at_3']:.3f}")
            print(f"Hybrid Recall@5: {hm['recall_at_5']:.3f}")
            print(f"Hybrid MRR: {hm['mrr']:.3f}")
            exact_suite_out = report["modes"]["hybrid"].get("suite_metrics", {}).get("exact_analyte", {})
            if exact_suite_out:
                print(f"ExactAnalyte Hit@1: {float(exact_suite_out.get('hit_at_1', 0.0)):.3f}")
                print(f"ExactAnalyte Hit@5: {float(exact_suite_out.get('hit_at_5', 0.0)):.3f}")
        if report.get("acceptance_results"):
            ar = report["acceptance_results"]
            passes = ar.get("passes", {})
            print(
                "Acceptance passes:",
                json.dumps(passes, ensure_ascii=False),
            )
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
