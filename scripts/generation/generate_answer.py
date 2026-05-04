#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Ensure scripts/ is importable so we can use retrieval package as-is.
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from retrieval.models import RetrievalFilters
from retrieval.search import SearchEngine

from answer_validator import validate_answer
from citation_builder import append_citations, build_citations
from evidence_builder import build_evidence_pack
from llm_client import LLMClient, LLMClientError
from prompt_builder import INSUFFICIENT_CONTEXT_SENTENCE, build_prompt
from query_understanding import contains_exact_term, detect_exact_analyte, norm_text


def normalize_query(query: str) -> str:
    q = re.sub(r"\s+", " ", (query or "").strip())
    return q


def sanitize_model_answer(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return raw

    # Remove explicit hidden-thought markers if the model emits them.
    raw = re.sub(r"(?is)<think>.*?</think>", "", raw).strip()
    raw = re.sub(r"(?im)^thinking\\s*:\\s*", "", raw).strip()

    # Keep only the expected final format when model emits preamble.
    match = re.search(r"(?im)^réponse\\s*:\\s*", raw)
    if not match:
        match = re.search(r"(?im)^reponse\\s*:\\s*", raw)
    if match:
        return raw[match.start() :].strip()
    return raw


def _query_is_sensitive_or_treatment(query: str) -> bool:
    q = normalize_query(query).lower()
    sensitive_markers = [
        "nom du patient",
        "date de naissance",
        "prescripteur",
        "telephone",
        "numéro",
        "numero",
        "patient id",
        "patient_id",
    ]
    treatment_markers = [
        "traitement",
        "prescrire",
        "posologie",
        "dose",
        "medicament",
        "médicament",
    ]
    return any(k in q for k in (sensitive_markers + treatment_markers))


def _build_structured_fallback_answer(query: str, evidence_pack: list[dict[str, Any]], exact_analyte: str | None = None) -> str:
    if not evidence_pack:
        return INSUFFICIENT_CONTEXT_SENTENCE

    qn = normalize_query(query).lower()
    candidates = evidence_pack

    if exact_analyte:
        exact_candidates = [
            e
            for e in evidence_pack
            if contains_exact_term(str(e.get("analyte_norm") or ""), exact_analyte)
            or contains_exact_term(str(e.get("analyte") or ""), exact_analyte)
        ]
        if exact_candidates:
            candidates = exact_candidates

    if any(k in qn for k in ["supérieur", "superieur", "above_reference"]):
        filtered = [e for e in evidence_pack if str(e.get("interpretation_status") or "").lower() == "above_reference"]
        if filtered:
            candidates = filtered
    elif any(k in qn for k in ["résultat antérieur", "resultat anterieur", "previous result", "ancien résultat", "ancien resultat"]):
        filtered = [
            e
            for e in evidence_pack
            if int(e.get("previous_result_present") or 0) == 1 and str(e.get("previous_result") or "").strip() != ""
        ]
        if filtered:
            candidates = filtered
    elif any(k in qn for k in ["parasite", "parasitologie", "trichuris", "ankylostoma"]):
        filtered = [
            e
            for e in evidence_pack
            if any(
                x in str(
                    (e.get("analyte") or "")
                    + " "
                    + (e.get("parameter") or "")
                    + " "
                    + (e.get("text_excerpt") or "")
                ).lower()
                for x in ["trichuris", "parasite", "ankylostoma"]
            )
        ]
        if filtered:
            candidates = filtered

    lines = ["Réponse :"]
    if len(candidates) > 1 and exact_analyte:
        lines.append(f"Plusieurs résultats de {exact_analyte.upper()} ont été retrouvés :")

    max_items = min(len(candidates), 10 if exact_analyte else 3)
    for idx, ev in enumerate(candidates[:max_items], start=1):
        analyte = ev.get("analyte") or ev.get("parameter") or "non précisé"
        value = ev.get("value_raw")
        unit = ev.get("unit")
        ref = ev.get("reference_range")
        interp = ev.get("interpretation_status")
        prev = ev.get("previous_result")

        prefix = f"{idx}. " if len(candidates) > 1 else "- "
        part = f"{prefix}{analyte}"
        if value not in (None, ""):
            part += f" = {value}"
        if unit not in (None, ""):
            part += f" {unit}"
        details = []
        if ref not in (None, ""):
            details.append(f"référence: {ref}")
        if interp not in (None, ""):
            details.append(f"interprétation technique: {interp}")
        if prev not in (None, ""):
            details.append(f"résultat antérieur: {prev}")
        if details:
            part += " (" + " ; ".join(details) + ")"
        lines.append(part)

    lines.extend(["", "Données utilisées :"])
    for idx, ev in enumerate(candidates[:max_items], start=1):
        lines.extend(
            [
                f"Résultat {idx} :",
                f"- Analyte : {ev.get('analyte') or ev.get('parameter') or 'non précisé'}",
                f"- Valeur : {ev.get('value_raw') or 'non disponible'}",
                f"- Unité : {ev.get('unit') or 'non disponible'}",
                f"- Référence : {ev.get('reference_range') or 'non disponible'}",
                f"- Interprétation technique : {ev.get('interpretation_status') or 'non disponible'}",
                f"- Résultat antérieur : {ev.get('previous_result') or 'non disponible'}",
                (
                    f"- Source : [doc_id={ev.get('doc_id')}, page={ev.get('page_number')}, "
                    f"row={ev.get('row_index')}, chunk_id={ev.get('chunk_id')}]"
                ),
            ]
        )
    return "\n".join(lines).strip()


def _answer_needs_fallback(text: str) -> bool:
    if not text.strip():
        return True
    low = text.lower()
    noisy_markers = [
        "okay,",
        "let's",
        "i need to",
        "the user is asking",
        "let me",
        "first,",
    ]
    if any(m in low for m in noisy_markers):
        return True
    if len(text) > 2200:
        return True
    return False


def _load_exact_analyte_rows(
    *,
    sqlite_path: Path,
    analyte_norm: str,
    limit: int,
) -> tuple[int, list[dict[str, Any]]]:
    if not sqlite_path.exists():
        return 0, []

    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT COUNT(*) AS c
            FROM metadata_chunks
            WHERE lower(analyte_norm) = lower(?)
            """,
            (analyte_norm,),
        )
        total = int((cur.fetchone() or {"c": 0})["c"])
        cur.execute(
            """
            SELECT
              c.chunk_id,
              c.doc_id,
              c.chunk_type,
              c.parent_chunk_id,
              c.text_for_embedding,
              c.text_for_keyword,
              m.document_type,
              m.sample_type,
              m.patient_token,
              m.sample_token,
              m.report_token,
              m.analyte,
              m.analyte_norm,
              m.parameter,
              m.parameter_norm,
              m.value_raw,
              m.value_numeric,
              m.unit,
              m.reference_range,
              m.interpretation_status,
              m.previous_result_present,
              m.previous_result_value_raw,
              m.previous_result_unit,
              m.section,
              m.section_norm,
              m.source_kind,
              m.source_table_id,
              m.row_index,
              COALESCE(m.source_pdf, o.source_pdf) AS source_pdf,
              COALESCE(m.page_number, o.page_number) AS page_number
            FROM metadata_chunks m
            JOIN chunks c ON c.chunk_id = m.chunk_id
            LEFT JOIN object_references o ON o.chunk_id = c.chunk_id
            WHERE lower(m.analyte_norm) = lower(?)
            ORDER BY
              c.doc_id ASC,
              COALESCE(m.page_number, o.page_number, 999999) ASC,
              COALESCE(m.row_index, 999999) ASC,
              c.chunk_id ASC
            LIMIT ?
            """,
            (analyte_norm, int(limit)),
        )
        rows = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()

    return total, rows


def _count_displayed_exact_analyte(answer: str, analyte: str) -> int:
    text = norm_text(answer or "")
    a = norm_text(analyte or "")
    if not text or not a:
        return 0
    pattern = re.compile(rf"(?:^|\s){re.escape(a)}\s*(?:=|:)", re.IGNORECASE)
    return len(pattern.findall(text))


def run_generation(
    *,
    query: str,
    top_k: int = 5,
    mode: str = "hybrid",
    provider: str = "ollama",
    model: str = "qwen3:4b",
    temperature: float = 0.0,
    num_ctx: int = 4096,
    max_tokens: int = 800,
    timeout: int = 420,
    index_dir: str | Path = "data/indexes",
    collection: str = "medical_chunks",
) -> dict[str, Any]:
    started = time.perf_counter()

    q = normalize_query(query)
    sensitive_or_treatment = _query_is_sensitive_or_treatment(q)
    idx = Path(index_dir)
    sqlite_path = idx / "medical_rag.sqlite"
    qdrant_dir = idx / "qdrant"

    retrieval_filters = RetrievalFilters()
    ql = q.lower()
    exact_analyte = detect_exact_analyte(q)
    if exact_analyte:
        retrieval_filters.analyte_norm = exact_analyte
    if any(k in ql for k in ["supérieur", "superieur", "above reference", "au dessus de la référence", "au dessus de la reference"]):
        retrieval_filters.interpretation_status = "above_reference"

    retrieval_response: Any
    max_exact_analyte_results = 10
    exact_analyte_expected_count = 0
    exact_analyte_rows: list[dict[str, Any]] = []
    if sensitive_or_treatment:
        retrieval_response = SimpleNamespace(
            answerability={"status": "guardrail_blocked", "reason": "sensitive_or_treatment_query"},
            filters={},
            top_results=[],
            context_chunks=[],
            sources=[],
        )
    else:
        if exact_analyte:
            exact_analyte_expected_count, exact_analyte_rows = _load_exact_analyte_rows(
                sqlite_path=sqlite_path,
                analyte_norm=exact_analyte,
                limit=max(top_k, max_exact_analyte_results),
            )
        engine = SearchEngine(
            sqlite_path=sqlite_path,
            qdrant_dir=qdrant_dir,
            collection=collection,
        )
        try:
            retrieval_response = engine.search(
                query=q,
                mode=mode,
                top_k=top_k,
                filters=retrieval_filters,
                expand_context=True,
            )
            if retrieval_filters.analyte_norm and not retrieval_response.top_results:
                relaxed_filters = replace(retrieval_filters)
                relaxed_filters.analyte_norm = None
                retrieval_response = engine.search(
                    query=q,
                    mode=mode,
                    top_k=top_k,
                    filters=relaxed_filters,
                    expand_context=True,
                )
            if retrieval_filters.interpretation_status and not retrieval_response.top_results:
                retrieval_response = engine.search(
                    query=q,
                    mode=mode,
                    top_k=top_k,
                    filters=RetrievalFilters(),
                    expand_context=True,
                )
        finally:
            engine.close()

    evidence_pack = build_evidence_pack(
        retrieval_response,
        query=q,
        max_evidence=max(top_k, max_exact_analyte_results) if exact_analyte else top_k,
        exact_analyte=exact_analyte,
        exact_analyte_rows=exact_analyte_rows,
        max_exact_analyte_results=max(top_k, max_exact_analyte_results),
    )

    prompt = build_prompt(
        query=q,
        evidence_pack=evidence_pack,
        exact_analyte=exact_analyte,
    )

    llm_answer = ""
    llm_error = None

    if sensitive_or_treatment:
        llm_answer = INSUFFICIENT_CONTEXT_SENTENCE
    elif not evidence_pack:
        llm_answer = INSUFFICIENT_CONTEXT_SENTENCE
    else:
        client = LLMClient(provider=provider)
        try:
            llm_answer = client.generate(
                prompt=prompt,
                model=model,
                temperature=temperature,
                num_ctx=num_ctx,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            llm_answer = sanitize_model_answer(llm_answer)
            if _answer_needs_fallback(llm_answer):
                llm_answer = _build_structured_fallback_answer(q, evidence_pack, exact_analyte=exact_analyte)
        except LLMClientError as exc:
            llm_error = str(exc)

    citations = build_citations(evidence_pack)

    if llm_error:
        final_answer = append_citations(
            f"{INSUFFICIENT_CONTEXT_SENTENCE}\n\nErreur LLM: {llm_error}",
            citations,
        )
    else:
        final_answer = append_citations(llm_answer, citations)

    validation = validate_answer(
        query=q,
        answer_text=final_answer,
        evidence_pack=evidence_pack,
        exact_analyte=exact_analyte,
    )

    elapsed = time.perf_counter() - started

    result: dict[str, Any] = {
        "query": query,
        "normalized_query": q,
        "mode": mode,
        "provider": provider,
        "model": model,
        "top_k": top_k,
        "timeout": timeout,
        "generation_time_seconds": round(elapsed, 3),
        "answer": final_answer,
        "citations": citations,
        "validation": validation,
        "llm_error": llm_error,
        "evidence_pack": evidence_pack,
        "retrieval": {
            "answerability": retrieval_response.answerability,
            "filters": retrieval_response.filters,
            "top_results": [r.to_dict() for r in retrieval_response.top_results],
            "context_chunks": [r.to_dict() for r in retrieval_response.context_chunks],
            "sources": retrieval_response.sources,
        },
        "prompt": prompt,
        "exact_analyte_coverage": {
            "detected_exact_analyte": exact_analyte,
            "expected_exact_analyte_count": exact_analyte_expected_count if exact_analyte else 0,
            "retrieved_exact_analyte_count": sum(
                1
                for ev in evidence_pack
                if exact_analyte
                and (
                    contains_exact_term(str(ev.get("analyte_norm") or ""), exact_analyte)
                    or contains_exact_term(str(ev.get("analyte") or ""), exact_analyte)
                )
            ),
            "displayed_exact_analyte_count": _count_displayed_exact_analyte(final_answer, exact_analyte or ""),
        },
    }

    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate grounded medical answer using local LLM")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", choices=["keyword", "vector", "hybrid"], default="hybrid")
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--timeout", type=int, default=420)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--show-context", action="store_true")
    parser.add_argument("--index-dir", default="data/indexes")
    parser.add_argument("--collection", default="medical_chunks")
    return parser.parse_args()


def _print_human(result: dict[str, Any], show_context: bool) -> None:
    print("Réponse :")
    print(result.get("answer") or "")

    validation = result.get("validation") or {}
    print("\nValidation :")
    print(f"- status: {validation.get('validation_status')}")
    print(f"- pii_leak_detected: {validation.get('pii_leak_detected')}")
    print(f"- citation_present: {validation.get('citation_present')}")
    print(f"- insufficient_context_handled: {validation.get('insufficient_context_handled')}")

    if validation.get("warnings"):
        print("- warnings:")
        for w in validation["warnings"]:
            print(f"  - {w}")
    if validation.get("errors"):
        print("- errors:")
        for e in validation["errors"]:
            print(f"  - {e}")

    print(f"\nTemps génération: {result.get('generation_time_seconds')} s")

    if show_context:
        print("\nEvidence pack:")
        print(json.dumps(result.get("evidence_pack") or [], ensure_ascii=False, indent=2))


def main() -> int:
    args = _parse_args()

    try:
        result = run_generation(
            query=args.query,
            top_k=args.top_k,
            mode=args.mode,
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
            num_ctx=args.num_ctx,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            index_dir=args.index_dir,
            collection=args.collection,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _print_human(result, show_context=args.show_context)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
