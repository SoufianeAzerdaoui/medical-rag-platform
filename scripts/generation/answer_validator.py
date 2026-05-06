from __future__ import annotations

import re
import unicodedata
from typing import Any

from prompt_builder import INSUFFICIENT_CONTEXT_SENTENCE
from query_understanding import contains_exact_term, detect_exact_analyte, detect_exact_analytes, find_analyte_mentions


_PII_PATTERNS = [
    r"\bpyxis\s*test\b",
    r"\bpatient\s*test1\b",
    r"\bdr\.\b",
    r"\bprescripteur\b",
    r"\bvalide\(e\)\s*par\b",
    r"\bimprime\(e\)\s*par\b",
    r"\banonymization_mapping\b",
    r"\bdata/private\b",
    r"\bchunks\.raw\.jsonl\b",
]

_FORBIDDEN_LEAK_MARKERS = [
    "anonymization_mapping",
    "data/private",
    "chunks.raw.jsonl",
]

_TREATMENT_PATTERNS = [
    r"\btraitement\s+recommande\b",
    r"\bprescrire\b",
    r"\bposologie\b",
    r"\bdose\b",
    r"\bprenez\b",
    r"\badministrer\b",
]

_DIAGNOSIS_PATTERNS = [
    r"\bdiagnostic\s+definitif\b",
    r"\bdiagnostic\s+confirm\w+\b",
    r"\bvous\s+avez\b",
]

_UNIT_REGEX = re.compile(
    r"\b(?:g/l|mg/l|ug/dl|ui/l|mui/l|mmol/l|ng/ml|pg/ml|uui?/ml|uu/ml|pmol/l|mui/ml|iu/l)\b",
    re.IGNORECASE,
)


def _norm(value: str) -> str:
    s = (value or "").strip().lower().replace("µ", "u")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s)
    return s


def _is_true(value: Any) -> bool:
    try:
        return int(value or 0) == 1
    except Exception:
        return bool(value)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None


def _num_eq(a: Any, b: Any, tol: float = 1e-9) -> bool:
    af = _to_float(a)
    bf = _to_float(b)
    if af is None or bf is None:
        return False
    return abs(af - bf) <= tol


def _extract_numeric_tokens_for_validation(core_text: str) -> list[str]:
    tokens: list[str] = []
    for raw_line in (core_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        low = _norm(line)
        if any(m in low for m in ["doc_id", "chunk_id", "page=", "row=", "source :"]):
            continue
        if "seuls les" in low and "premiers" in low:
            continue
        if "utilisez --show-all-results" in low:
            continue

        # Remove list ordinals and result section ordinals (non-medical numbering).
        line = re.sub(r"^\s*\d+\.\s*", "", line)
        line = re.sub(r"(?im)^\s*résultat\s+\d+\s*:\s*", "", line)
        line = re.sub(r"(?im)^\s*resultat\s+\d+\s*:\s*", "", line)
        line = re.sub(r"\breport[_\-]?\d+\b", " ", line, flags=re.IGNORECASE)

        for t in re.findall(r"\d+(?:[.,]\d+)?", line):
            tokens.append(t)
    return tokens


def _value_supported_by_evidence(value: str, allowed: dict[str, set[str]], evidence_pack: list[dict[str, Any]]) -> bool:
    v_norm = _norm(value)
    if v_norm in allowed["values"]:
        return True
    vf = _to_float(value)
    if vf is not None:
        for evf in allowed.get("numeric_values", set()):
            if abs(vf - evf) <= 1e-9:
                return True
    # Fallback: explicit textual evidence in excerpt.
    for ev in evidence_pack:
        excerpt = str(ev.get("text_excerpt") or "")
        if not excerpt:
            continue
        if _norm(value) in _norm(excerpt):
            return True
        if vf is not None:
            for t in re.findall(r"\d+(?:[.,]\d+)?", excerpt):
                if _num_eq(vf, t):
                    return True
    return False


def _extract_allowed_sets(evidence_pack: list[dict[str, Any]]) -> dict[str, set[str]]:
    analytes: set[str] = set()
    values: set[str] = set()
    units: set[str] = set()
    previous_values: set[str] = set()
    numeric_values: set[float] = set()

    for ev in evidence_pack:
        for key in ("analyte", "parameter"):
            v = str(ev.get(key) or "").strip()
            if v:
                analytes.add(_norm(v))

        for key in ("value_raw", "previous_result"):
            v = str(ev.get(key) or "").strip()
            if v:
                values.add(_norm(v))
                values.add(_norm(v.replace(",", ".")))
                vf = _to_float(v)
                if vf is not None:
                    numeric_values.add(vf)

        numeric = ev.get("value_numeric")
        if numeric is not None:
            try:
                f = float(numeric)
                values.add(_norm(str(f)))
                values.add(_norm(str(f).replace(".", ",")))
                numeric_values.add(float(f))
            except Exception:
                pass

        prev_num = ev.get("previous_result")
        if prev_num not in (None, ""):
            previous_values.add(_norm(str(prev_num)))
            previous_values.add(_norm(str(prev_num).replace(",", ".")))
            pf = _to_float(prev_num)
            if pf is not None:
                numeric_values.add(pf)

        unit = str(ev.get("unit") or "").strip()
        if unit:
            units.add(_norm(unit))

        ref = str(ev.get("reference_range") or "").strip()
        if ref:
            values.add(_norm(ref))
            for found_unit in _UNIT_REGEX.findall(_norm(ref)):
                units.add(_norm(found_unit))
            for num in re.findall(r"\d+(?:[.,]\d+)?", ref):
                values.add(_norm(num))
                values.add(_norm(num.replace(",", ".")))
                nf = _to_float(num)
                if nf is not None:
                    numeric_values.add(nf)

        # Keep support for numbers present in raw chunk excerpt (e.g., previous_result textual traces).
        excerpt = str(ev.get("text_excerpt") or "")
        for num in re.findall(r"\d+(?:[.,]\d+)?", excerpt):
            values.add(_norm(num))
            values.add(_norm(num.replace(",", ".")))
            nf = _to_float(num)
            if nf is not None:
                numeric_values.add(nf)

    return {
        "analytes": analytes,
        "values": values,
        "units": units,
        "previous_values": previous_values,
        "numeric_values": numeric_values,
    }


def _split_answer_core(answer_text: str) -> str:
    lower = answer_text.lower()
    idx = lower.find("sources")
    if idx == -1:
        return answer_text
    return answer_text[:idx]


def _extract_main_response_block(answer_text: str) -> str:
    text = answer_text or ""
    low = text.lower()
    start = low.find("réponse")
    if start == -1:
        start = low.find("reponse")
    if start == -1:
        start = 0

    end = len(text)
    for marker in ("données utilisées", "donnees utilisees", "sources"):
        pos = low.find(marker, start)
        if pos != -1:
            end = min(end, pos)
    return text[start:end]


def _extract_source_chunk_ids(answer_text: str) -> list[str]:
    return re.findall(r"chunk_id=([^\],\s]+)", answer_text or "", flags=re.IGNORECASE)


def _extract_source_doc_ids(answer_text: str) -> list[str]:
    return re.findall(r"doc_id=([^\],\s]+)", answer_text or "", flags=re.IGNORECASE)


def _is_simple_question(query: str) -> bool:
    qn = _norm(query)
    if any(k in qn for k in ["tous", "toutes", "liste", "retrouves", "retrouvés", "documents"]):
        return False
    return qn.startswith("quel est") or qn.startswith("quelle est")


def validate_answer(
    *,
    query: str,
    answer_text: str,
    evidence_pack: list[dict[str, Any]],
    displayed_evidences: list[dict[str, Any]] | None = None,
    exact_analyte: str | None = None,
    llm_error: str | None = None,
    generation_mode: str | None = None,
    retrieval_status: str | None = None,
    show_low_quality: bool = False,
    max_display_results: int = 3,
    show_all_results: bool = False,
    query_received: str | None = None,
    query_used_for_retrieval: str | None = None,
    query_used_for_prompt: str | None = None,
    query_stored: str | None = None,
    detected_analytes: list[str] | None = None,
    requested_doc_id: str | None = None,
    requested_doc_ids: list[str] | None = None,
    missing_requested_doc_ids: list[str] | None = None,
    requested_analytes: list[str] | None = None,
    found_requested_analytes: list[str] | None = None,
    found_requested_analyte_norms: list[str] | None = None,
    missing_requested_analytes: list[str] | None = None,
    doc_summary_intent: dict[str, bool] | None = None,
    summary_section_filter_applied: bool = False,
    current_vs_previous_requested: bool = False,
    diagnostic_safety_intent: bool = False,
    allow_low_quality_display: bool = False,
) -> dict[str, Any]:
    text = (answer_text or "").strip()
    text_norm = _norm(text)
    core_text = _split_answer_core(text)
    core_norm = _norm(core_text)
    displayed = displayed_evidences or []
    source_chunk_ids = _extract_source_chunk_ids(text)
    source_doc_ids = _extract_source_doc_ids(text)
    displayed_chunk_ids = [str(ev.get("chunk_id") or "") for ev in displayed if ev.get("chunk_id")]
    displayed_doc_ids = [str(ev.get("doc_id") or "") for ev in displayed if ev.get("doc_id")]

    errors: list[str] = []
    warnings: list[str] = []
    unsupported_claims: list[str] = []

    # Security / PII leaks
    pii_hits: list[str] = []
    for patt in _PII_PATTERNS:
        if re.search(patt, text_norm, flags=re.IGNORECASE):
            pii_hits.append(patt)
    for marker in _FORBIDDEN_LEAK_MARKERS:
        if marker in text_norm:
            pii_hits.append(marker)

    pii_leak_detected = len(pii_hits) > 0
    if pii_leak_detected:
        errors.append("PII/PHI or forbidden marker detected in generated answer.")

    # Thinking exposure
    if "<think>" in text_norm or "thinking:" in text_norm:
        errors.append("Model thinking content exposed in final answer.")

    # Generation/retrieval hard errors
    if llm_error:
        errors.append(f"LLM error detected: {llm_error}")
    if "erreur generation" in text_norm or "erreur génération" in text_norm:
        errors.append("Generation error exposed in final answer.")
    if "ollama timeout" in text_norm or "timeout" in text_norm and "erreur llm" in text_norm:
        errors.append("Timeout error detected in final answer.")
    if "no such column" in text_norm or "sql" in text_norm and "erreur" in text_norm:
        errors.append("SQL error detected in final answer.")
    if retrieval_status == "retrieval_error":
        errors.append("Retrieval error status detected.")

    citation_present = "[doc_id=" in text
    if displayed and not citation_present:
        errors.append("Missing citations while evidence exists.")

    allowed = _extract_allowed_sets(displayed if displayed else evidence_pack)

    # Unsupported numerics
    unsupported_numeric: list[str] = []
    for token in _extract_numeric_tokens_for_validation(core_text):
        if _norm(token) in {"0", "1"}:
            continue
        if not _value_supported_by_evidence(token, allowed, displayed if displayed else evidence_pack):
            unsupported_numeric.append(token)
    if unsupported_numeric:
        unsupported_claims.append(f"Unsupported numeric values: {sorted(set(unsupported_numeric))}")
        warnings.append("Some numeric values were not found in evidence.")

    # Unsupported units
    unsupported_units: list[str] = []
    for unit in _UNIT_REGEX.findall(core_text):
        if _norm(unit) not in allowed["units"]:
            unsupported_units.append(unit)
    if unsupported_units:
        unsupported_claims.append(f"Unsupported units: {sorted(set(unsupported_units))}")
        warnings.append("Some units were not found in evidence.")

    # Previous result claim validation
    qn = _norm(query)
    exact_analyte = exact_analyte or detect_exact_analyte(query)
    requested_analyte_list = [str(a).strip().lower() for a in (requested_analytes or []) if str(a).strip()]
    found_requested = [str(a).strip().lower() for a in (found_requested_analytes or []) if str(a).strip()]
    found_requested_norms = [str(a).strip().lower() for a in (found_requested_analyte_norms or []) if str(a).strip()]
    missing_requested = [str(a).strip().lower() for a in (missing_requested_analytes or []) if str(a).strip()]
    requested_doc_ids_norm = [str(d).strip().lower() for d in (requested_doc_ids or []) if str(d).strip()]
    multi_doc_requested = len(requested_doc_ids_norm) >= 2

    if exact_analyte and not requested_analyte_list and not multi_doc_requested:
        detected = find_analyte_mentions(core_text)
        irrelevant = sorted(a for a in detected if a != exact_analyte)
        if irrelevant:
            errors.append("irrelevant_analyte_in_answer")
            unsupported_claims.append(
                f"Irrelevant analyte mentions for exact query '{exact_analyte}': {irrelevant}"
            )

        bad_evidence_ids: list[str] = []
        for ev in evidence_pack:
            analyte_norm = str(ev.get("analyte_norm") or "")
            analyte = str(ev.get("analyte") or "")
            if not (contains_exact_term(analyte_norm, exact_analyte) or contains_exact_term(analyte, exact_analyte)):
                bad_evidence_ids.append(str(ev.get("chunk_id") or "unknown_chunk"))
        if bad_evidence_ids:
            warnings.append("non_exact_analyte_evidence_present")
            unsupported_claims.append(
                f"Evidence contains non-exact analyte entries for '{exact_analyte}': {bad_evidence_ids}"
            )

    if requested_analyte_list:
        requested_set = set(requested_analyte_list)
        coverage_set = set(found_requested) | set(missing_requested)
        uncovered = sorted(a for a in requested_set if a not in coverage_set)
        if uncovered:
            errors.append("requested_analyte_coverage_incomplete")
            unsupported_claims.append(f"Requested analytes without found/missing status: {uncovered}")
        if missing_requested and generation_mode != "deterministic_measured_value_vs_comment_sql_template":
            warnings.append("controlled_warning_missing_requested_analytes")

        displayed_norms = {str(ev.get("analyte_norm") or "").strip().lower() for ev in displayed if ev.get("analyte_norm")}
        if found_requested_norms:
            allowed_norms = set(found_requested_norms)
            bad_displayed = sorted(n for n in displayed_norms if n not in allowed_norms)
            if bad_displayed:
                errors.append("non_exact_analyte_evidence_present")
                unsupported_claims.append(
                    f"Displayed analytes not in found_requested_analytes: displayed={sorted(displayed_norms)}, found={sorted(allowed_norms)}"
                )

    # Multi-result source consistency
    main_block = _extract_main_response_block(answer_text)
    result_line_count = len(
        [
            ln
            for ln in main_block.splitlines()
            if re.match(r"^\s*(?:[-*]|\d+\.)\s+", ln)
        ]
    )
    source_count = len(re.findall(r"\[doc_id=", answer_text or "", flags=re.IGNORECASE))
    relaxed_line_source_modes = {
        "deterministic_doc_summary_sql_template",
        "deterministic_section_grouped_summary_sql_template",
        "deterministic_multi_doc_analyte_comparison_sql_template",
        "deterministic_measured_value_vs_comment_sql_template",
    }
    if generation_mode not in relaxed_line_source_modes and result_line_count >= 2 and source_count < result_line_count:
        warnings.append("multi_result_missing_structured_details")

    prev_mentions = list(
        re.finditer(r"(?im)^\s*-?\s*résultat\s*antérieur\s*:\s*([^\n]+)", core_text)
    ) + list(
        re.finditer(r"(?im)^\s*-?\s*resultat\s*anterieur\s*:\s*([^\n]+)", core_text)
    )
    for match in prev_mentions:
        raw_prev = match.group(1).strip()
        prev_val = _norm(raw_prev)
        if prev_val and prev_val not in {"non", "aucun", "none", "null", "n/a", "non disponible"}:
            if not _value_supported_by_evidence(raw_prev, allowed, displayed if displayed else evidence_pack):
                unsupported_claims.append(f"Unsupported previous result: {raw_prev}")
                warnings.append("Previous result in answer not found in evidence.")

    # Treatment / diagnosis guardrails
    lower_core = core_norm
    if any(re.search(p, lower_core) for p in _TREATMENT_PATTERNS):
        if "ne peux pas proposer de traitement" not in lower_core and INSUFFICIENT_CONTEXT_SENTENCE.lower() not in lower_core:
            errors.append("Treatment recommendation detected.")
    if any(re.search(p, lower_core) for p in _DIAGNOSIS_PATTERNS):
        errors.append("Definitive diagnosis detected.")

    # Insufficient context handling
    insufficient_context_handled = False
    no_evidence = len(displayed if displayed else evidence_pack) == 0
    sensitive_query = any(k in qn for k in ["nom du patient", "patient", "date de naissance", "prescripteur"]) or any(
        k in qn for k in ["traitement", "prescrire", "posologie"]
    )
    is_guardrail_mode = generation_mode == "guardrail_blocked"
    has_insufficient_sentence = INSUFFICIENT_CONTEXT_SENTENCE.lower() in text_norm or (
        "information insuffisante dans le contexte fourni" in text_norm
    )

    if no_evidence:
        insufficient_context_handled = has_insufficient_sentence
        if not insufficient_context_handled:
            errors.append("No evidence available but answer did not report insufficient context.")
    elif sensitive_query:
        insufficient_context_handled = (
            has_insufficient_sentence
            or "anonym" in text_norm
            or "non disponible" in text_norm
            or "je ne peux pas" in text_norm
        )
        if not insufficient_context_handled:
            warnings.append("Sensitive query should generally return anonymized or insufficient-context response.")
    else:
        insufficient_context_handled = has_insufficient_sentence
        if (displayed or evidence_pack) and has_insufficient_sentence and not is_guardrail_mode and not missing_requested:
            errors.append("Insufficient-context answer returned despite available evidence.")

    requested_analytes = detected_analytes or detect_exact_analytes(query)
    mentioned_analytes = find_analyte_mentions(core_text)
    if requested_analytes:
        allowed_mentions = set(str(a).strip().lower() for a in requested_analytes if str(a).strip())
        allowed_mentions.update(found_requested_norms)
        if "hdl" in allowed_mentions:
            allowed_mentions.add("cholesterol_hdl")
        bad_mentions = sorted(a for a in mentioned_analytes if a not in allowed_mentions)
        if bad_mentions:
            errors.append("query_answer_alignment_mismatch")
            unsupported_claims.append(
                f"Answer mentions analytes not requested: requested={sorted(allowed_mentions)}, mentioned={sorted(mentioned_analytes)}"
            )

    source_alignment_pass = set(source_chunk_ids) == set(displayed_chunk_ids)
    if not source_alignment_pass:
        errors.append("source_alignment_mismatch")
        unsupported_claims.append(
            f"source_chunk_ids={sorted(set(source_chunk_ids))}, displayed_evidence_chunk_ids={sorted(set(displayed_chunk_ids))}"
        )
    citation_coverage = 1.0
    if displayed_chunk_ids:
        citation_coverage = len(set(source_chunk_ids)) / max(1, len(set(displayed_chunk_ids)))

    if not show_low_quality and not allow_low_quality_display:
        low_displayed = [
            str(ev.get("chunk_id") or "unknown_chunk")
            for ev in displayed
            if str(ev.get("evidence_display_quality") or "high") == "low"
        ]
        if low_displayed:
            errors.append("low_quality_display_without_opt_in")
            unsupported_claims.append(f"low_quality_displayed_chunk_ids={low_displayed}")

    if _is_simple_question(query) and (not show_all_results) and len(displayed) > max(1, int(max_display_results)):
        warnings.append("max_display_results_exceeded_for_simple_query")

    stale_query = False
    q_ref = _norm(query_received or query)
    for candidate in [query_used_for_retrieval, query_used_for_prompt, query_stored]:
        if candidate is None:
            continue
        if _norm(candidate) != q_ref:
            stale_query = True
            break
    if stale_query:
        errors.append("stale_response_detection")

    requested_doc_id_norm = str(requested_doc_id or "").strip().lower()
    requested_doc_id_mismatch = False
    if requested_doc_id_norm:
        bad_display_docs = sorted({d for d in displayed_doc_ids if str(d).strip().lower() != requested_doc_id_norm})
        bad_source_docs = sorted({d for d in source_doc_ids if str(d).strip().lower() != requested_doc_id_norm})
        if bad_display_docs or bad_source_docs:
            requested_doc_id_mismatch = True
            errors.append("requested_doc_id_mismatch")
            unsupported_claims.append(
                f"requested_doc_id={requested_doc_id_norm}, displayed_doc_ids={sorted(set(displayed_doc_ids))}, source_doc_ids={sorted(set(source_doc_ids))}"
            )

    missing_doc_ids = [str(d).strip().lower() for d in (missing_requested_doc_ids or []) if str(d).strip()]
    requested_doc_ids_incomplete = False
    if len(requested_doc_ids_norm) >= 2:
        represented_docs = {str(d).strip().lower() for d in displayed_doc_ids + source_doc_ids if str(d).strip()}
        represented_or_missing = represented_docs | set(missing_doc_ids)
        not_covered = sorted(d for d in requested_doc_ids_norm if d not in represented_or_missing)
        if not_covered:
            requested_doc_ids_incomplete = True
            errors.append("requested_doc_ids_incomplete")
            unsupported_claims.append(
                f"requested_doc_ids={requested_doc_ids_norm}, represented_docs={sorted(represented_docs)}, missing_doc_ids={sorted(missing_doc_ids)}, not_covered={not_covered}"
            )

    if generation_mode == "deterministic_doc_summary_sql_template":
        if llm_error:
            errors.append("doc_summary_mode_llm_error_present")
        if citation_coverage < 1.0:
            errors.append("doc_summary_mode_citation_coverage_not_full")
        intent = doc_summary_intent or {}
        if intent.get("wants_immunoanalyse_section") and summary_section_filter_applied:
            bad_sections = []
            for ev in displayed:
                section = _norm(str(ev.get("section_norm") or ev.get("section") or ""))
                if "immunoanalyse" not in section and "immuno analyse" not in section:
                    bad_sections.append(str(ev.get("chunk_id") or "unknown_chunk"))
            if bad_sections:
                errors.append("doc_summary_immunoanalyse_section_mismatch")
                unsupported_claims.append(
                    f"Non-immunoanalyse chunks displayed while section filter applied: {bad_sections}"
                )

    if current_vs_previous_requested:
        if not any(k in core_norm for k in ["comparaison", "valeur actuelle", "plus elevee", "plus basse", "egale"]):
            errors.append("missing_current_vs_previous_comparison")

    if diagnostic_safety_intent:
        if "on ne peut pas conclure a un diagnostic" not in core_norm and "on ne peut pas conclure à un diagnostic" not in core_norm:
            errors.append("missing_diagnostic_safety_refusal")

    value_accuracy = len(unsupported_numeric) == 0
    unit_accuracy = len(unsupported_units) == 0

    if errors:
        validation_status = "fail"
    elif warnings:
        validation_status = "warning"
    else:
        validation_status = "pass"

    return {
        "validation_status": validation_status,
        "warnings": warnings,
        "errors": errors,
        "pii_leak_detected": pii_leak_detected,
        "pii_hits": pii_hits,
        "unsupported_claims": unsupported_claims,
        "citation_present": citation_present,
        "insufficient_context_handled": insufficient_context_handled,
        "value_accuracy": value_accuracy,
        "unit_accuracy": unit_accuracy,
        "evidence_count": len(evidence_pack),
        "displayed_evidence_count": len(displayed),
        "source_chunk_ids": source_chunk_ids,
        "displayed_evidence_chunk_ids": displayed_chunk_ids,
        "source_doc_ids": source_doc_ids,
        "displayed_evidence_doc_ids": displayed_doc_ids,
        "source_alignment_pass": source_alignment_pass,
        "citation_coverage": round(float(citation_coverage), 3),
        "query_answer_alignment_pass": "query_answer_alignment_mismatch" not in errors,
        "stale_response_detected": stale_query,
        "requested_doc_id": requested_doc_id,
        "requested_doc_id_mismatch": requested_doc_id_mismatch,
        "requested_doc_ids": requested_doc_ids_norm,
        "missing_requested_doc_ids": missing_doc_ids,
        "requested_doc_ids_incomplete": requested_doc_ids_incomplete,
        "found_requested_analytes": found_requested,
        "found_requested_analyte_norms": found_requested_norms,
        "missing_requested_analytes": missing_requested,
        "requested_analyte_coverage": {
            "requested_count": len(requested_analyte_list),
            "found_count": len(found_requested),
            "missing_count": len(missing_requested),
            "uncovered_count": max(0, len(set(requested_analyte_list) - (set(found_requested) | set(missing_requested)))),
        },
    }
