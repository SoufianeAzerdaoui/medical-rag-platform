from __future__ import annotations

import re
import unicodedata
from dataclasses import replace
from typing import Any

try:
    from .models import RetrievalFilters, RetrievalResult
    from .keyword_search import KeywordSearcher
    from .vector_search import VectorSearcher
except ImportError:
    from models import RetrievalFilters, RetrievalResult
    from keyword_search import KeywordSearcher
    from vector_search import VectorSearcher


def _norm(value: str) -> str:
    s = (value or "").strip().lower().replace("µ", "u")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9_\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


_ANALYTE_LEXICON = {
    "calcitonine",
    "ferritine",
    "lithium",
    "crp",
    "procalcitonine",
    "peptide c",
    "insuline",
    "pro bnp",
    "tshus",
    "acth",
    "tsh",
    "vitamine d",
    "trichuris",
    "ankylostoma",
}

_HORMONAL_EXPANSION = [
    "tsh",
    "tshus",
    "t3",
    "t4",
    "calcitonine",
    "insuline",
    "peptide c",
    "pro bnp",
    "acth",
    "immunoanalyse",
    "hormone",
]

_PARASITO_EXPANSION = [
    "parasitologie",
    "positif",
    "presence",
    "parasite",
    "oeufs",
    "kyste",
    "trichuris",
    "ankylostoma",
    "resultat final",
    "examen microscopique",
]

_ABOVE_REF_EXPANSION = [
    "valeur superieure",
    "reference",
    "above_reference",
    "high",
    "abnormal",
    "au dessus",
]


class QueryIntent:
    def __init__(self, query: str) -> None:
        qn = _norm(query)
        self.query = query
        self.query_norm = qn
        self.is_validation_request = any(x in qn for x in ["validation", "valide", "consistency", "coherence"])
        self.is_visual_request = any(x in qn for x in ["image", "visuel", "visual", "scan", "capture"])
        self.is_above_reference = any(
            x in qn for x in ["superieure a la reference", "au dessus de la reference", "above reference", "high"]
        )
        self.is_without_unit = any(x in qn for x in ["sans unite", "without unit", "non specifiee"])
        self.is_previous_result = any(x in qn for x in ["resultat anterieur", "previous result", "anterieur"])
        self.is_hormonal_reference = any(
            x in qn for x in ["hormonal", "hormone", "immunoanalyse", "thyroid", "thyroidienne"]
        ) and any(x in qn for x in ["reference", "valeur"])
        self.is_parasitology_positive = any(x in qn for x in ["parasitologie", "parasite", "trichuris", "ankylostoma"]) and any(
            x in qn for x in ["positif", "presence", "resultat", "pathogene"]
        )

        analyte = None
        for a in sorted(_ANALYTE_LEXICON, key=len, reverse=True):
            if a in qn:
                analyte = a
                break
        tokens = [t for t in qn.split(" ") if t]
        self.exact_analyte = analyte
        self.is_exact_analyte = analyte is not None or (len(tokens) <= 2 and qn in _ANALYTE_LEXICON)
        self.is_broad_semantic = not self.is_exact_analyte

    def expanded_query(self) -> str:
        terms: list[str] = [self.query]
        if self.is_hormonal_reference:
            terms.extend(_HORMONAL_EXPANSION)
        if self.is_parasitology_positive:
            terms.extend(_PARASITO_EXPANSION)
        if self.is_above_reference:
            terms.extend(_ABOVE_REF_EXPANSION)
        if self.is_without_unit:
            terms.extend(["sans unite", "unit missing", "non specifiee", "qualitative"])
        if self.is_previous_result:
            terms.extend(["resultat anterieur", "previous result", "comparatif"])
        out: list[str] = []
        seen: set[str] = set()
        for t in terms:
            k = _norm(t)
            if not k or k in seen:
                continue
            seen.add(k)
            out.append(t)
        return " ".join(out)

    def strict_filters(self, base: RetrievalFilters) -> RetrievalFilters:
        f = replace(base)
        if self.is_above_reference and not f.interpretation_status:
            f.interpretation_status = "above_reference"
        if self.is_without_unit and not f.result_quality_status:
            f.result_quality_status = "unit_missing"
        if self.is_previous_result and f.previous_result_present is None:
            f.previous_result_present = 1
        if self.is_parasitology_positive and not f.document_type:
            f.document_type = "parasitology_stool_report"
        return f


class HybridSearcher:
    def __init__(self, keyword_searcher: KeywordSearcher, vector_searcher: VectorSearcher) -> None:
        self.keyword_searcher = keyword_searcher
        self.vector_searcher = vector_searcher

    @staticmethod
    def _has_value(value: Any) -> bool:
        return value is not None and str(value).strip() != ""

    def _clinical_rerank_score(
        self,
        item: RetrievalResult,
        *,
        intent: QueryIntent,
        filters: RetrievalFilters,
        has_strong_exact_lab: bool,
    ) -> tuple[float, list[str]]:
        md = item.metadata or {}
        reasons: list[str] = []
        score = 0.0
        ctype = (item.chunk_type or "").lower()
        text_blob = f"{item.text} {item.text_preview}".lower()

        # Base chunk type boosts
        if ctype == "lab_result":
            score += 0.20
            reasons.append("chunk_type:lab_result")
        elif ctype == "clinical_result":
            score += 0.15
            reasons.append("chunk_type:clinical_result")

        analyte_norm = str(md.get("analyte_norm") or "").strip().lower()
        analyte_text = str(md.get("analyte") or "").strip().lower()
        if intent.is_exact_analyte and intent.exact_analyte:
            exact_blob = f"{analyte_norm} {analyte_text} {text_blob}"
            if intent.exact_analyte in exact_blob:
                score += 0.35
                reasons.append("exact_analyte_match")
            else:
                score -= 0.40
                reasons.append("penalty:exact_analyte_mismatch")

        if self._has_value(md.get("value_raw")):
            score += 0.05
            reasons.append("value_raw_present")
        if self._has_value(md.get("unit")):
            score += 0.05
            reasons.append("unit_present")
        if self._has_value(md.get("reference_range")):
            score += 0.05
            reasons.append("reference_range_present")

        if intent.is_previous_result:
            prev_present_flag = str(md.get("previous_result_present") or "").strip()
            prev_value = md.get("previous_result") or md.get("previous_result_value_raw")
            has_prev = False
            try:
                has_prev = int(prev_present_flag or 0) == 1
            except Exception:
                has_prev = False
            has_prev = has_prev or self._has_value(prev_value)
            if has_prev or "resultat anterieur" in text_blob:
                score += 0.30
                reasons.append("previous_result_match")
            else:
                score -= 0.20
                reasons.append("penalty:previous_result_missing")

        if intent.is_above_reference:
            if str(md.get("interpretation_status") or "").strip().lower() == "above_reference":
                score += 0.25
                reasons.append("above_reference_match")
            if self._has_value(md.get("value_numeric")) and self._has_value(md.get("reference_range")):
                score += 0.05
                reasons.append("numeric_with_reference")

        if intent.is_without_unit:
            unit = str(md.get("unit") or "").strip().lower()
            rqs = str(md.get("result_quality_status") or "").strip().lower()
            if rqs == "unit_missing" or unit in {"non specifiee", "non spécifiée", "qualitative", ""}:
                score += 0.25
                reasons.append("unit_missing_match")

        if intent.is_hormonal_reference:
            section = str(md.get("section") or "").lower()
            section_norm = str(md.get("section_norm") or "").lower()
            hormone_hit = any(h in text_blob for h in ["tsh", "t4", "t3", "acth", "insuline", "peptide c", "pro bnp", "calcitonine"])
            if "immunoanalyse" in section or "immunoanalyse" in section_norm or hormone_hit:
                score += 0.15
                reasons.append("hormonal_match")
            if self._has_value(md.get("reference_range")):
                score += 0.05
                reasons.append("hormonal_reference_present")

        if intent.is_parasitology_positive:
            doc_type = str(item.document_type or md.get("document_type") or "").lower()
            rkind = str(md.get("result_kind") or "").lower()
            if doc_type == "parasitology_stool_report":
                score += 0.15
                reasons.append("parasitology_document_type")
            if rkind in {"pathogen_identification", "microscopy_finding"} or any(
                tok in text_blob for tok in ["parasite", "trichuris", "ankylostoma", "oeufs", "kyste", "positif"]
            ):
                score += 0.15
                reasons.append("parasitology_positive_signal")

        # Metadata filter match boost
        filter_boost_fields = [
            ("document_type", filters.document_type),
            ("section", filters.section),
            ("section_norm", filters.section_norm),
            ("source_kind", filters.source_kind),
            ("source_table_id", filters.source_table_id),
            ("analyte_norm", filters.analyte_norm),
            ("interpretation_status", filters.interpretation_status),
            ("reference_quality_status", filters.reference_quality_status),
            ("result_quality_status", filters.result_quality_status),
        ]
        for field, expected in filter_boost_fields:
            if not expected:
                continue
            got = str(md.get(field) or getattr(item, field, "") or "")
            if got == str(expected):
                score += 0.15
                reasons.append(f"metadata_match:{field}")
                break

        # Admin penalties (except explicit intent)
        if ctype == "validation_status" and not intent.is_validation_request:
            score -= 0.50
            reasons.append("penalty:validation_status")
        if ctype == "visual_reference" and not intent.is_visual_request:
            score -= 0.50
            reasons.append("penalty:visual_reference")
        if intent.is_exact_analyte and ctype == "document_summary":
            score -= 0.20
            reasons.append("penalty:summary_for_exact_analyte")
        if intent.is_exact_analyte and has_strong_exact_lab and ctype == "exam_section":
            score -= 0.10
            reasons.append("penalty:exam_section_when_exact_lab_exists")

        return score, reasons

    def _has_previous_result_signal(self, item: RetrievalResult) -> bool:
        md = item.metadata or {}
        prev_flag = str(md.get("previous_result_present") or "").strip()
        prev_value = md.get("previous_result") or md.get("previous_result_value_raw")
        text_blob = f"{item.text} {item.text_preview}".lower()
        try:
            if int(prev_flag or 0) == 1:
                return True
        except Exception:
            pass
        return self._has_value(prev_value) or ("resultat anterieur" in text_blob)

    def _enforce_previous_result_presence(
        self,
        *,
        top: list[RetrievalResult],
        merged: list[RetrievalResult],
        top_k: int,
        query: str,
        expanded_query: str,
        keyword_internal: int,
        vector_internal: int,
        rrf_k: int,
        filters: RetrievalFilters,
        intent: QueryIntent,
        has_strong_exact_lab: bool,
    ) -> list[RetrievalResult]:
        if any(self._has_previous_result_signal(item) for item in top):
            return top

        # First try to promote an already-ranked candidate outside current top-k.
        for candidate in merged[top_k:]:
            if self._has_previous_result_signal(candidate):
                if top:
                    top = top[:-1] + [candidate]
                else:
                    top = [candidate]
                candidate.match_reason.append("previous_result_guard_promoted")
                return top

        # Then force a dedicated fallback query with previous_result filter.
        forced_filters = replace(filters)
        forced_filters.previous_result_present = 1
        kw = self.keyword_searcher.search(expanded_query, top_k=keyword_internal, filters=forced_filters)
        vec = self.vector_searcher.search(expanded_query, top_k=vector_internal, filters=forced_filters)

        best: RetrievalResult | None = None
        if vec:
            best = RetrievalResult(**vec[0].to_dict())
        elif kw:
            best = RetrievalResult(**kw[0].to_dict())
        if best is None:
            return top

        rk = best.rank_keyword
        rv = best.rank_vector
        rrf_score = 0.0
        if rk is not None:
            rrf_score += 1.0 / (rrf_k + rk)
            if "keyword_match" not in best.match_reason:
                best.match_reason.append("keyword_match")
        if rv is not None:
            rrf_score += 1.0 / (rrf_k + rv)
            if "vector_match" not in best.match_reason:
                best.match_reason.append("vector_match")
        best.rrf_score = rrf_score
        clinical_score, reasons = self._clinical_rerank_score(
            best,
            intent=intent,
            filters=filters,
            has_strong_exact_lab=has_strong_exact_lab,
        )
        best.clinical_rerank_score = clinical_score
        best.final_score = (best.rrf_score or 0.0) + clinical_score
        best.score_hybrid = best.final_score
        best.retrieval_mode = "hybrid"
        best.match_reason.append("previous_result_guard_injected")
        if reasons:
            best.metadata["rerank_reasons"] = reasons
        best.metadata["rrf_score"] = round(best.rrf_score or 0.0, 6)
        best.metadata["clinical_rerank_score"] = round(best.clinical_rerank_score or 0.0, 6)
        best.metadata["final_score"] = round(best.final_score or 0.0, 6)

        existing_ids = {x.chunk_id for x in top}
        if best.chunk_id in existing_ids:
            return top
        if top:
            top = top[:-1] + [best]
        else:
            top = [best]
        return top

    def search(
        self,
        query: str,
        *,
        top_k: int,
        keyword_top_k: int,
        vector_top_k: int,
        rrf_k: int,
        filters: RetrievalFilters,
    ) -> list[RetrievalResult]:
        if not (query or "").strip():
            raise ValueError("query is empty")

        intent = QueryIntent(query)
        expanded_query = intent.expanded_query()
        strict_filters = intent.strict_filters(filters)

        if intent.is_broad_semantic:
            keyword_internal = max(keyword_top_k, top_k * 5, 30)
            vector_internal = max(vector_top_k, top_k * 5, 30)
        else:
            keyword_internal = max(keyword_top_k, top_k)
            vector_internal = max(vector_top_k, top_k)

        def run_once(active_filters: RetrievalFilters) -> tuple[list[RetrievalResult], list[RetrievalResult]]:
            kw = self.keyword_searcher.search(
                expanded_query,
                top_k=keyword_internal,
                filters=active_filters,
            )
            vec = self.vector_searcher.search(
                expanded_query,
                top_k=vector_internal,
                filters=active_filters,
            )
            return kw, vec

        keyword_results, vector_results = run_once(strict_filters)
        strict_changed = strict_filters.to_dict() != filters.to_dict()
        used_fallback = False
        if strict_changed and not keyword_results and not vector_results:
            keyword_results, vector_results = run_once(filters)
            used_fallback = True

        by_chunk_id: dict[str, RetrievalResult] = {}

        for r in keyword_results:
            clone = RetrievalResult(**r.to_dict())
            by_chunk_id[r.chunk_id] = clone

        for r in vector_results:
            if r.chunk_id in by_chunk_id:
                cur = by_chunk_id[r.chunk_id]
                cur.score_vector = r.score_vector
                cur.rank_vector = r.rank_vector
                cur.vector_rank = r.rank_vector
                if "vector_match" not in cur.match_reason:
                    cur.match_reason.append("vector_match")
            else:
                clone = RetrievalResult(**r.to_dict())
                by_chunk_id[r.chunk_id] = clone

        merged: list[RetrievalResult] = []
        for item in by_chunk_id.values():
            rk = item.rank_keyword
            rv = item.rank_vector
            rrf_score = 0.0
            if rk is not None:
                rrf_score += 1.0 / (rrf_k + rk)
            if rv is not None:
                rrf_score += 1.0 / (rrf_k + rv)
            if item.rank_keyword is not None and "keyword_match" not in item.match_reason:
                item.match_reason.append("keyword_match")
            if item.rank_vector is not None and "vector_match" not in item.match_reason:
                item.match_reason.append("vector_match")
            if item.rank_keyword is None and item.rank_vector is None:
                continue
            item.rrf_score = rrf_score
            item.retrieval_mode = "hybrid"
            merged.append(item)

        has_strong_exact_lab = False
        if intent.is_exact_analyte and intent.exact_analyte:
            for item in merged:
                md = item.metadata or {}
                analyte_blob = f"{md.get('analyte_norm') or ''} {md.get('analyte') or ''}".lower()
                if item.chunk_type == "lab_result" and intent.exact_analyte in analyte_blob:
                    has_strong_exact_lab = True
                    break

        for item in merged:
            clinical_score, reasons = self._clinical_rerank_score(
                item,
                intent=intent,
                filters=filters,
                has_strong_exact_lab=has_strong_exact_lab,
            )
            item.clinical_rerank_score = clinical_score
            item.final_score = (item.rrf_score or 0.0) + clinical_score
            item.score_hybrid = item.final_score
            if strict_changed:
                if used_fallback:
                    if "auto_filter_fallback" not in item.match_reason:
                        item.match_reason.append("auto_filter_fallback")
                else:
                    if "auto_filter_applied" not in item.match_reason:
                        item.match_reason.append("auto_filter_applied")
            item.metadata["rrf_score"] = round(item.rrf_score or 0.0, 6)
            item.metadata["clinical_rerank_score"] = round(item.clinical_rerank_score or 0.0, 6)
            item.metadata["final_score"] = round(item.final_score or 0.0, 6)
            if reasons:
                item.metadata["rerank_reasons"] = reasons

        merged.sort(key=lambda x: x.final_score or 0.0, reverse=True)
        top = merged[:top_k]
        if intent.is_previous_result:
            top = self._enforce_previous_result_presence(
                top=top,
                merged=merged,
                top_k=top_k,
                query=query,
                expanded_query=expanded_query,
                keyword_internal=keyword_internal,
                vector_internal=vector_internal,
                rrf_k=rrf_k,
                filters=filters,
                intent=intent,
                has_strong_exact_lab=has_strong_exact_lab,
            )
            # Keep final ordering deterministic after potential replacement.
            top.sort(key=lambda x: x.final_score or 0.0, reverse=True)
        for i, item in enumerate(top, start=1):
            item.rank_hybrid = i
            item.hybrid_rank = i
            if item.rank_keyword is not None:
                item.keyword_rank = item.rank_keyword
            if item.rank_vector is not None:
                item.vector_rank = item.rank_vector
        return top
