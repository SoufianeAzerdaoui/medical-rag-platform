from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .models import RetrievalFilters, RetrievalResult, SearchResponse
    from .sqlite_store import SQLiteStore
except ImportError:
    from models import RetrievalFilters, RetrievalResult, SearchResponse
    from sqlite_store import SQLiteStore


def _preview(text: str, max_len: int = 220) -> str:
    t = (text or "").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 3] + "..."


def _normalize_text(value: str) -> str:
    s = (value or "").strip().lower().replace("µ", "u")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9./\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _contains_text(haystack: str, needle: str) -> bool:
    return _normalize_text(needle).replace(" ", "") in _normalize_text(haystack).replace(" ", "")


_EXACT_MEDICAL_ENTITY_PATTERNS = [
    r"\btrichuris\s+trichiura\b",
    r"\btrichuris\b",
    r"\bankylostoma\s+duodenale\b",
    r"\bankylostoma\b",
    r"\bferritine\b",
    r"\balbumine\b",
    r"\bammonium\b",
    r"\bacide\s+urique\b",
    r"\bcalcium\b",
    r"\bcrp\b",
    r"\bacth\b",
    r"\btshus\b",
    r"\btsh\b",
    r"\blithium\b",
]


def detect_exact_medical_entities(query: str) -> list[str]:
    norm = _normalize_text(query)
    found: list[str] = []
    for patt in _EXACT_MEDICAL_ENTITY_PATTERNS:
        m = re.search(patt, norm)
        if m:
            found.append(m.group(0))
    for m in re.findall(r"\b\d{5,}\b", norm):
        found.append(m)
    for m in re.findall(r"\b\d+(?:[.,]\d+)?\s*(?:g/l|mg/l|ug/dl|mui/l|pg/ml|mmol/l|ng/ml)\b", norm):
        found.append(m.replace(" ", ""))
    out: list[str] = []
    seen: set[str] = set()
    for item in found:
        k = _normalize_text(item)
        if k and k not in seen:
            seen.add(k)
            out.append(item)
    return out


def _contains_any_entity(text: str, entities: list[str]) -> bool:
    return any(_contains_text(text, e) for e in entities)


def _load_policy() -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "exact_entity": {
            "strict_context": True,
            "allow_other_docs": False,
            "require_explicit_entity_match": True,
            "min_evidence_score": 0.45,
            "max_context_chunks": 8,
            "must_include_same_doc": ["document_summary", "validation_status"],
            "must_include_parasitology_related_sections": True,
            "weights": {
                "exact_entity_match": 0.45,
                "keyword_match": 0.2,
                "vector_match": 0.1,
                "same_main_doc": 0.15,
                "document_summary": 0.15,
                "validation_status": 0.25,
                "warning": 0.3,
                "related_section": 0.2,
                "parasitology_related_section": 0.2,
                "final_result_related": 0.2,
                "staining_exam_related": 0.2,
                "parent_or_sibling": 0.1,
                "other_doc_penalty": -0.5,
                "vector_only_penalty": -0.25,
            },
        },
        "semantic": {
            "strict_context": True,
            "allow_other_docs": True,
            "require_explicit_entity_match": False,
            "min_evidence_score": 0.35,
            "max_context_chunks": 8,
            "weights": {
                "keyword_match": 0.15,
                "vector_match": 0.25,
                "same_main_doc": 0.2,
                "document_summary": 0.1,
                "validation_status": 0.1,
                "warning": 0.15,
                "related_section": 0.15,
                "parent_or_sibling": 0.1,
                "other_doc_penalty": -0.15,
                "vector_only_penalty": -0.05,
            },
        },
    }
    path = Path(__file__).resolve().parent / "context_policy.json"
    if not path.exists():
        return defaults
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            return defaults
        out = dict(defaults)
        out.update(loaded)
        return out
    except Exception:
        return defaults


def _to_result_from_row(row: dict[str, Any], mode: str, reason: str) -> RetrievalResult:
    text = (row.get("text_for_embedding") or row.get("text_for_keyword") or "").strip()
    metadata = {
        k: row.get(k)
        for k in [
            "document_type",
            "sample_type",
            "request_date",
            "patient_token",
            "sample_token",
            "report_token",
            "validation_status",
            "reference_quality_status",
            "age_consistency_status",
            "section",
            "section_norm",
            "analyte",
            "value_raw",
            "unit",
            "source_pdf",
            "page_number",
            "parent_chunk_id",
        ]
        if k in row
    }
    return RetrievalResult(
        chunk_id=str(row.get("chunk_id") or ""),
        doc_id=str(row.get("doc_id") or ""),
        chunk_type=str(row.get("chunk_type") or ""),
        document_type=row.get("document_type"),
        source_pdf=row.get("source_pdf") or row.get("object_source_pdf"),
        page_number=row.get("page_number") or row.get("object_page_number"),
        text=text,
        text_preview=_preview(text),
        metadata=metadata,
        retrieval_mode=mode,
        match_reason=[reason],
    )


class ContextBuilder:
    def __init__(self, sqlite_store: SQLiteStore) -> None:
        self.sqlite_store = sqlite_store
        self.policy = _load_policy()

    def _is_warning_row(self, row: dict[str, Any]) -> bool:
        chunk_type = str(row.get("chunk_type") or "").lower()
        if chunk_type == "validation_status" or "warning" in chunk_type or "consistency" in chunk_type:
            return True
        validation_status = str(row.get("validation_status") or "").lower()
        return validation_status in {"rejected", "failed"}

    def _section_tag(self, row_or_result: dict[str, Any] | RetrievalResult) -> str:
        if isinstance(row_or_result, RetrievalResult):
            return str((row_or_result.metadata or {}).get("section_norm") or "").lower()
        return str(row_or_result.get("section_norm") or "").lower()

    def build(
        self,
        *,
        query: str,
        mode: str,
        top_results: list[RetrievalResult],
        filters: RetrievalFilters | None = None,
        max_context_chunks: int = 8,
        strict_context: bool = True,
        debug_context: bool = False,
    ) -> SearchResponse:
        filters = filters or RetrievalFilters()
        if not top_results:
            return SearchResponse(
                query=query,
                mode=mode,
                filters={},
                top_results=[],
                context_chunks=[],
                sources=[],
                excluded_context_candidates=[],
                answerability={
                    "status": "insufficient_context",
                    "reason": "no_results",
                    "confidence": 0.0,
                    "evidence_count": 0,
                    "explicit_entity_match_found": False,
                    "main_doc_ids": [],
                    "missing_evidence": ["no_results"],
                },
            )

        exact_entities = detect_exact_medical_entities(query)
        query_type = "exact_entity" if exact_entities else "semantic"
        policy = self.policy.get(query_type, {})
        weights = policy.get("weights", {}) if isinstance(policy.get("weights"), dict) else {}
        policy_strict = bool(policy.get("strict_context", True))
        strict = strict_context and policy_strict
        allow_other_docs = bool(policy.get("allow_other_docs", query_type == "semantic"))
        require_explicit = bool(policy.get("require_explicit_entity_match", query_type == "exact_entity"))
        min_score = float(policy.get("min_evidence_score", 0.45 if query_type == "exact_entity" else 0.35))
        ctx_cap = min(max_context_chunks, int(policy.get("max_context_chunks", max_context_chunks)))
        must_include_same_doc = {str(x).lower() for x in (policy.get("must_include_same_doc") or [])}
        must_include_para = bool(policy.get("must_include_parasitology_related_sections", True))

        rows_by_doc: dict[str, list[dict[str, Any]]] = {}
        for r in top_results:
            if r.doc_id not in rows_by_doc:
                rows_by_doc[r.doc_id] = self.sqlite_store.get_doc_chunks(r.doc_id)

        strong_matches: list[RetrievalResult] = []
        for r in top_results:
            blob = f"{r.text} {r.text_preview}"
            has_entity = _contains_any_entity(blob, exact_entities) if exact_entities else False
            reasons = set(r.match_reason or [])
            if (
                ("keyword_match" in reasons and "vector_match" in reasons)
                or ("keyword_match" in reasons and has_entity)
                or (r.score_keyword is not None and (r.rank_keyword or 999) <= 5)
                or ((r.rank_hybrid or 999) <= 2 and has_entity)
            ):
                strong_matches.append(r)

        main_doc_ids = {r.doc_id for r in strong_matches}
        if not main_doc_ids:
            main_doc_ids = {top_results[0].doc_id}

        candidates_by_id: dict[str, RetrievalResult] = {}

        def add_candidate(item: RetrievalResult, reason: str) -> None:
            cid = item.chunk_id
            if not cid:
                return
            cur = candidates_by_id.get(cid)
            if cur is None:
                if reason not in item.match_reason:
                    item.match_reason.append(reason)
                candidates_by_id[cid] = item
                return
            if reason not in cur.match_reason:
                cur.match_reason.append(reason)

        # Include top results candidates
        for r in top_results:
            add_candidate(r, "hybrid_match")

        # Expand from main docs only
        for doc_id in main_doc_ids:
            rows = rows_by_doc.get(doc_id, [])
            row_by_id = {str(x.get("chunk_id") or ""): x for x in rows}

            # Base summary/validation/warnings/related sections
            for row in rows:
                cid = str(row.get("chunk_id") or "")
                if not cid:
                    continue
                ctype = str(row.get("chunk_type") or "").lower()
                sec = self._section_tag(row)
                if ctype == "document_summary":
                    add_candidate(_to_result_from_row(row, mode, "context_expansion:document_summary"), "context_expansion:document_summary")
                elif ctype == "validation_status":
                    add_candidate(_to_result_from_row(row, mode, "context_expansion:validation_status"), "context_expansion:validation_status")
                elif self._is_warning_row(row):
                    add_candidate(_to_result_from_row(row, mode, "context_expansion:warning"), "context_expansion:warning")
                elif sec in {"resultat_final", "examen_apres_coloration", "examen_apres_enrichissement", "examen_microscopique"}:
                    add_candidate(_to_result_from_row(row, mode, "context_expansion:related_section"), "context_expansion:related_section")

            # parent/siblings for strong chunks
            by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                p = str(row.get("parent_chunk_id") or "")
                if p:
                    by_parent[p].append(row)
            for item in [r for r in strong_matches if r.doc_id == doc_id]:
                parent = str((item.metadata or {}).get("parent_chunk_id") or "")
                if parent and parent in row_by_id:
                    add_candidate(_to_result_from_row(row_by_id[parent], mode, "context_expansion:related_section"), "context_expansion:parent_or_sibling")
                if parent:
                    for sib in by_parent.get(parent, []):
                        sctype = str(sib.get("chunk_type") or "").lower()
                        if sctype in {"lab_result", "clinical_result"}:
                            add_candidate(_to_result_from_row(sib, mode, "context_expansion:sibling_result"), "context_expansion:parent_or_sibling")

            # must-include parasitology related sections
            doc_type = str((rows[0].get("document_type") if rows else "") or "")
            if doc_type == "parasitology_stool_report" and must_include_para:
                trigger = False
                qn = _normalize_text(query)
                if any(x in qn for x in ["trichuris", "ankylostoma", "resultat final", "examen apres coloration"]):
                    trigger = True
                if any(self._section_tag(r) in {"resultat_final", "examen_apres_coloration"} for r in strong_matches if r.doc_id == doc_id):
                    trigger = True
                if trigger:
                    for row in rows:
                        sec = self._section_tag(row)
                        ctype = str(row.get("chunk_type") or "").lower()
                        if ctype == "document_summary":
                            add_candidate(
                                _to_result_from_row(row, mode, "context_expansion:document_summary"),
                                "must_include_same_doc",
                            )
                        if ctype == "validation_status" or self._is_warning_row(row):
                            add_candidate(
                                _to_result_from_row(row, mode, "context_expansion:validation_status" if ctype == "validation_status" else "context_expansion:warning"),
                                "must_include_same_doc",
                            )
                        if sec in {"resultat_final", "examen_apres_coloration", "examen_apres_enrichissement", "examen_microscopique"}:
                            add_candidate(
                                _to_result_from_row(row, mode, "context_expansion:parasitology_related_section"),
                                "must_include_parasitology_related_section",
                            )

        included: list[RetrievalResult] = []
        excluded: list[dict[str, Any]] = []

        def score_item(item: RetrievalResult) -> tuple[float, list[str], bool]:
            reasons = set(item.match_reason or [])
            blob = f"{item.text} {item.text_preview}"
            has_entity = _contains_any_entity(blob, exact_entities) if exact_entities else False
            same_doc = item.doc_id in main_doc_ids
            ctype = (item.chunk_type or "").lower()
            sec = self._section_tag(item)
            ev_reasons: list[str] = []
            score = 0.0

            if has_entity:
                score += float(weights.get("exact_entity_match", 0.0))
                ev_reasons.append("exact_entity_match")
            if "keyword_match" in reasons:
                score += float(weights.get("keyword_match", 0.0))
                ev_reasons.append("keyword_match")
            if "vector_match" in reasons:
                score += float(weights.get("vector_match", 0.0))
                ev_reasons.append("vector_match")
            if same_doc:
                score += float(weights.get("same_main_doc", 0.0))
                ev_reasons.append("same_main_doc")
            if ctype == "document_summary":
                score += float(weights.get("document_summary", 0.0))
                ev_reasons.append("document_summary")
            if ctype == "validation_status":
                score += float(weights.get("validation_status", 0.0))
                ev_reasons.append("validation_status")
            if "context_expansion:warning" in reasons:
                score += float(weights.get("warning", 0.0))
                ev_reasons.append("warning")
            if "context_expansion:related_section" in reasons:
                score += float(weights.get("related_section", 0.0))
                ev_reasons.append("related_section")
            if "context_expansion:parasitology_related_section" in reasons:
                score += float(weights.get("parasitology_related_section", 0.0))
                ev_reasons.append("parasitology_related_section")
            if sec == "resultat_final":
                score += float(weights.get("final_result_related", 0.0))
                ev_reasons.append("final_result_related")
            if sec == "examen_apres_coloration":
                score += float(weights.get("staining_exam_related", 0.0))
                ev_reasons.append("staining_exam_related")
            if "context_expansion:parent_or_sibling" in reasons or "context_expansion:sibling_result" in reasons:
                score += float(weights.get("parent_or_sibling", 0.0))
                ev_reasons.append("parent_or_sibling")

            is_other_doc = item.doc_id not in main_doc_ids
            vector_only = ("vector_match" in reasons and "keyword_match" not in reasons)
            if is_other_doc:
                score += float(weights.get("other_doc_penalty", 0.0))
                ev_reasons.append("other_doc_penalty")
            if vector_only:
                score += float(weights.get("vector_only_penalty", 0.0))
                ev_reasons.append("vector_only_penalty")

            must_include = False
            if same_doc and ctype in must_include_same_doc:
                must_include = True
                ev_reasons.append("must_include_same_doc")
            if "must_include_parasitology_related_section" in reasons:
                must_include = True
                ev_reasons.append("must_include_parasitology_related_section")
            return min(1.0, max(0.0, score)), ev_reasons, must_include

        for item in candidates_by_id.values():
            score, ev_reasons, must_include = score_item(item)
            item.metadata["evidence_score"] = round(score, 4)
            item.metadata["evidence_reasons"] = ev_reasons

            decision = "include"
            reason = ""
            if strict and not allow_other_docs and item.doc_id not in main_doc_ids:
                decision = "exclude"
                reason = "other_doc_vector_only_without_entity_match"
            elif score < min_score and not must_include:
                decision = "exclude"
                reason = "low_evidence_score"

            if decision == "include":
                included.append(item)
            elif debug_context:
                excluded.append(
                    {
                        "chunk_id": item.chunk_id,
                        "doc_id": item.doc_id,
                        "chunk_type": item.chunk_type,
                        "match_reason": item.match_reason,
                        "evidence_score": round(score, 4),
                        "min_evidence_score": min_score,
                        "evidence_reasons": ev_reasons,
                        "decision": "exclude",
                        "reason": reason,
                    }
                )

        def priority(it: RetrievalResult) -> tuple[int, int]:
            rs = set(it.match_reason or [])
            if "keyword_match" in rs and "vector_match" in rs:
                return (0, it.rank_hybrid or 999)
            if "keyword_match" in rs:
                return (1, it.rank_hybrid or 999)
            if "context_expansion:document_summary" in rs:
                return (2, 999)
            if "context_expansion:validation_status" in rs or "context_expansion:warning" in rs:
                return (3, 999)
            return (4, it.rank_hybrid or 999)

        included.sort(key=priority)
        context = included[:ctx_cap]

        explicit_found = any(_contains_any_entity(f"{c.text} {c.text_preview}", exact_entities) for c in context) if exact_entities else bool(context)
        evidence_scores = sorted([float((c.metadata or {}).get("evidence_score", 0.0)) for c in context], reverse=True)
        if evidence_scores:
            head = evidence_scores[: min(4, len(evidence_scores))]
            confidence = (sum(head) / len(head)) + (0.08 if explicit_found else 0.0)
            confidence = min(1.0, confidence)
        else:
            confidence = 0.0
        missing_evidence: list[str] = []
        if require_explicit and not explicit_found:
            missing_evidence.append("explicit_entity_match")

        ans_status = "answerable"
        ans_reason = "explicit_entity_match_found" if explicit_found and exact_entities else "context_available"
        if missing_evidence:
            ans_status = "insufficient_context"
            ans_reason = "no_explicit_entity_match"

        # Discordance warning for parasitology
        warning_flags: list[str] = []
        if context:
            blob_all = "\n".join([f"{c.text}\n{c.text_preview}" for c in context])
            has_trich = _contains_text(blob_all, "trichuris")
            has_anky = _contains_text(blob_all, "ankylostoma")
            if has_trich and has_anky:
                warning_flags.append("discordant_parasitology_findings_present")

        if debug_context:
            for c in context:
                excluded.append(
                    {
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "chunk_type": c.chunk_type,
                        "match_reason": c.match_reason,
                        "evidence_score": c.metadata.get("evidence_score"),
                        "min_evidence_score": min_score,
                        "evidence_reasons": c.metadata.get("evidence_reasons"),
                        "decision": "include",
                        "reason": "",
                    }
                )

        sources = [
            {
                "doc_id": r.doc_id,
                "source_pdf": r.source_pdf,
                "page_number": r.page_number,
                "chunk_id": r.chunk_id,
                "chunk_type": r.chunk_type,
                "text_preview": r.text_preview,
            }
            for r in context
        ]

        return SearchResponse(
            query=query,
            mode=mode,
            filters={},
            top_results=top_results,
            context_chunks=context,
            sources=sources,
            excluded_context_candidates=excluded,
            answerability={
                "status": ans_status,
                "reason": ans_reason,
                "confidence": round(confidence, 4),
                "evidence_count": len(context),
                "explicit_entity_match_found": bool(explicit_found),
                "main_doc_ids": sorted(main_doc_ids),
                "missing_evidence": missing_evidence,
                "warnings": warning_flags,
            },
        )
