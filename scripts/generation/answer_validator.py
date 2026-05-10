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

_FORBIDDEN_INTERNAL_MARKERS = [
    "chunk_id=",
    "request_id",
    "query_used_for_retrieval",
    "traceback",
    "exception",
    "loading weights",
    "inference embeddings",
    "pre tokenize",
]

_GENERIC_COLD_SENTENCES = [
    "les resultats ci dessus sont strictement extraits des donnees indexees",
    "les résultats ci-dessus sont strictement extraits des données indexées",
]

_INTERNAL_REASONING_LEAK_PATTERNS = [
    "okay, the user",
    "the user said",
    "the user wants",
    "i need to",
    "i should",
    "first, i'll",
    "first i ll",
    "first, i will",
    "first i will",
    "i will",
    "let me",
    "je dois répondre",
    "je vais répondre",
    "<think>",
    "</think>",
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
        if re.search(r"\b\d+\s+r[ée]sultat(?:s)?\b", low):
            continue

        # Remove list ordinals and result section ordinals (non-medical numbering).
        line = re.sub(r"^\s*\d+\.\s*", "", line)
        line = re.sub(r"(?im)^\s*résultat\s+\d+\s*:\s*", "", line)
        line = re.sub(r"(?im)^\s*resultat\s+\d+\s*:\s*", "", line)
        line = re.sub(r"\breport[_\-]?\d+\b", " ", line, flags=re.IGNORECASE)
        line = re.sub(r"\bpat[_\-]?\d+\b", " ", line, flags=re.IGNORECASE)

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

        for key in ("value_raw", "current_value", "value", "previous_result"):
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
    return re.findall(r"doc_id=([^\],\s&#?]+)", answer_text or "", flags=re.IGNORECASE)


def _extract_link_labels(answer_text: str) -> list[str]:
    labels: list[str] = []
    for m in re.finditer(r"\[([^\]]+)\]\((/api/documents/[^)]+|/viewer/[^)]+)\)", answer_text or "", flags=re.IGNORECASE):
        labels.append((m.group(1) or "").strip())
    return labels


def _extract_sources_block_labels(answer_text: str) -> list[str]:
    text = answer_text or ""
    low = text.lower()
    start = low.find("sources")
    if start == -1:
        return []
    block = text[start:]
    labels: list[str] = []
    for line in block.splitlines():
        ln = line.strip()
        if not ln.startswith("- "):
            continue
        content = ln[2:].strip()
        if not content:
            continue
        md = re.match(r"\[([^\]]+)\]\(([^)]+)\)", content)
        if md:
            labels.append((md.group(1) or "").strip())
        else:
            labels.append(content.split(" : ")[0].strip())
    return labels


def _parse_source_label(label: str) -> tuple[str, int | None, tuple[int, ...] | None]:
    text = str(label or "").strip()
    if not text:
        return ("", None, None)
    normalized = _norm(text)
    filename = normalized.split("—")[0].strip() if "—" in normalized else normalized.split("- page")[0].strip()
    page_match = re.search(r"\bpage\s*(\d+)\b", normalized)
    page = int(page_match.group(1)) if page_match else None
    line_match = re.search(r"\blignes?\s*(\d+)(?:\s*[–-]\s*(\d+))?\b", normalized)
    if not line_match:
        return (filename, page, None)
    start = int(line_match.group(1))
    end = int(line_match.group(2) or start)
    if end < start:
        start, end = end, start
    return (filename, page, tuple(range(start, end + 1)))


def _source_label_supported(mentioned_label: str, allowed_labels: set[str]) -> bool:
    mn = _norm(mentioned_label)
    if mn in allowed_labels:
        return True
    m_file, m_page, m_rows = _parse_source_label(mentioned_label)
    if not m_file:
        return False
    candidates: list[tuple[str, int | None, tuple[int, ...] | None]] = []
    for allowed in allowed_labels:
        candidates.append(_parse_source_label(allowed))
    same_file_page = [c for c in candidates if c[0] == m_file and (m_page is None or c[1] == m_page)]
    if not same_file_page:
        return False
    if m_rows is None:
        return True
    allowed_rows = {
        row
        for _, _, rows in same_file_page
        if rows is not None
        for row in rows
    }
    if not allowed_rows:
        return False
    return bool(set(m_rows) & allowed_rows)


def _table_has_source_column(answer_text: str) -> bool:
    lines = [ln.strip() for ln in (answer_text or "").splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    for i in range(len(lines) - 1):
        if "|" in lines[i] and re.search(r"^\|?\s*[-:| ]+\s*\|?\s*$", lines[i + 1]):
            return "source" in _norm(lines[i])
    return False


def _is_small_talk_query(query: str) -> bool:
    qn = _norm(query)
    markers = [
        "bonjour",
        "bonsoir",
        "salut",
        "hello",
        "hi",
        "hey",
        "ca va",
        "ça va",
        "cava",
        "cv",
        "merci",
        "au revoir",
        "bonne journee",
        "bonne journée",
    ]
    if any(ch.isdigit() for ch in qn):
        return False
    if any(m in qn for m in markers):
        if any(k in qn for k in ["report", "rapport", "resultat", "valeur", "analyte", "patient"]):
            return False
        return True
    return False


def _detect_general_conversation_intent(query: str, query_intents: dict[str, Any] | None) -> str | None:
    intents = query_intents or {}
    if intents.get("identity_question"):
        return "identity_question"
    if intents.get("capability_question"):
        return "capability_question"
    if intents.get("help_question"):
        return "help_question"
    if intents.get("small_talk") or intents.get("general_conversation"):
        return "small_talk"

    qn = _norm(query)
    if any(m in qn for m in ["t es qui", "tu es qui", "qui es tu", "who are you", "what are you", "vous etes qui", "c est qui toi"]):
        return "identity_question"
    if any(
        m in qn
        for m in [
            "tu peux faire quoi",
            "que peux tu faire",
            "c est quoi ton role",
            "ton role",
            "tu sers a quoi",
            "comment tu peux m aider",
            "what can you do",
        ]
    ):
        return "capability_question"
    if any(m in qn for m in ["aide moi", "help", "comment utiliser", "how to use", "guide moi"]):
        return "help_question"
    if _is_small_talk_query(query):
        return "small_talk"
    return None


def check_internal_reasoning_leak(answer: str) -> bool:
    low = _norm(answer or "")
    if not low:
        return False
    return any(_norm(p) in low for p in _INTERNAL_REASONING_LEAK_PATTERNS)


def _is_valid_presence_analyte_label(label: str) -> bool:
    text = str(label or "").strip()
    n = _norm(text)
    if not text:
        return False
    if len(n) > 42 or len(n.split()) > 6:
        return False
    if ":" in text and len(text) > 25:
        return False
    if any(
        m in n
        for m in [
            "augmentation de",
            "associes",
            "acromegalie",
            "commentaire",
            "interpretation",
            "apres un infarctus",
            "valeurs de reference",
        ]
    ):
        return False
    return True


def _contains_medical_like(text_norm: str) -> bool:
    return any(
        k in text_norm
        for k in [
            "analyte",
            "statut technique",
            "report_",
            "doc_id",
            "pg/ml",
            "m ui",
            "mmol/l",
        ]
    )


def _is_simple_question(query: str) -> bool:
    qn = _norm(query)
    if any(k in qn for k in ["tous", "toutes", "liste", "retrouves", "retrouvés", "documents"]):
        return False
    return qn.startswith("quel est") or qn.startswith("quelle est")


def _parse_markdown_table_header_keys(text: str) -> list[str]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) < 2:
        return []
    header = None
    for i in range(len(lines) - 1):
        if "|" in lines[i] and re.search(r"^\|?\s*[-:| ]+\s*\|?\s*$", lines[i + 1]):
            header = lines[i]
            break
    if header is None:
        return []
    cols = [c.strip().lower() for c in header.strip("|").split("|")]
    keys: list[str] = []
    for col in cols:
        c = _norm(col)
        key = c
        if "analyte" in c:
            key = "analyte"
        elif "valeur actuelle" in c or c == "valeur":
            key = "valeur_actuelle"
        elif "unite" in c:
            key = "unite"
        elif "reference" in c:
            key = "reference"
        elif "statut" in c:
            key = "statut"
        elif "resultat anterieur" in c:
            key = "resultat_anterieur"
        elif "variation" in c:
            key = "variation"
        elif "source" in c:
            key = "source"
        elif "patient" in c:
            key = "patient"
        elif "report" in c or "rapport" in c or "document" in c:
            key = "report"
        keys.append(key)
    return keys


def _canonical_analyte_key(value: str) -> str:
    key = _norm(value).replace("_", " ")
    key = key.replace("valporoique", "valproique")
    key = re.sub(r"\bdepakine\b", "", key)
    key = re.sub(r"\s+", " ", key).strip()
    return key


def validate_answer(
    *,
    query: str,
    answer_text: str,
    evidence_pack: list[dict[str, Any]],
    displayed_evidences: list[dict[str, Any]] | None = None,
    source_citations: list[dict[str, Any]] | None = None,
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
    query_intents: dict[str, bool] | None = None,
    output_format_requested: str | None = None,
    answer_style_requested: str | None = None,
    requested_table_columns: list[str] | None = None,
    requested_technical_condition: str | None = None,
    source_clickable_requested: bool = False,
    requested_value: str | None = None,
    comparison_operator: str | None = None,
) -> dict[str, Any]:
    text = (answer_text or "").strip()
    text_norm = _norm(text)
    core_text = _split_answer_core(text)
    core_norm = _norm(core_text)
    displayed = displayed_evidences or []
    structured_sources = source_citations or []
    source_chunk_ids = _extract_source_chunk_ids(text)
    source_doc_ids = _extract_source_doc_ids(text)
    structured_source_doc_ids = [
        str(s.get("doc_id") or "").strip()
        for s in structured_sources
        if str(s.get("doc_id") or "").strip()
    ]
    displayed_chunk_ids = [str(ev.get("chunk_id") or "") for ev in displayed if ev.get("chunk_id")]
    displayed_doc_ids = [str(ev.get("doc_id") or "") for ev in displayed if ev.get("doc_id")]

    errors: list[str] = []
    warnings: list[str] = []
    unsupported_claims: list[str] = []
    intents = query_intents or {}
    general_conversation_intent = _detect_general_conversation_intent(query, intents)
    small_talk_query = general_conversation_intent == "small_talk"
    is_general_conversation_query = general_conversation_intent in {"small_talk", "identity_question", "capability_question", "help_question"}

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
    if check_internal_reasoning_leak(text):
        errors.append("internal_reasoning_leak")

    forbidden_internal_hits = [m for m in _FORBIDDEN_INTERNAL_MARKERS if m in text_norm]
    if forbidden_internal_hits or "/home/" in text or "\\home\\" in text or re.search(r"[A-Za-z]:\\", text):
        errors.append("forbidden_internal_field")
    if re.search(r"résultat\(s\)|correspondant\(s\)", text, flags=re.IGNORECASE):
        warnings.append("ugly_pluralization")
    if (
        re.search(r"page\s*\d+\s*row\s*\d+", text_norm, flags=re.IGNORECASE)
        or "chunk_id" in text_norm
        or "/home/" in text
        or "\\home\\" in text
        or re.search(r"[A-Za-z]:\\", text)
    ):
        errors.append("source_format_bad")
    if re.search(r"(?:^|[^a-z])page\s*\d+row\s*\d+", text_norm, flags=re.IGNORECASE):
        errors.append("source_format_bad")
    if any(sentence in text_norm for sentence in _GENERIC_COLD_SENTENCES):
        warnings.append("repeated_generic_sentence")

    # Generation/retrieval hard errors
    if llm_error:
        fallback_modes = {
            "llm_writer_error_fallback",
            "llm_writer_format_fallback",
            "llm_writer_quality_fallback",
            "deterministic_structured_renderer",
            "llm_error_fallback_template",
        }
        if str(generation_mode or "") in fallback_modes:
            warnings.append(f"llm_fallback_used:{llm_error}")
        else:
            errors.append(f"LLM error detected: {llm_error}")
    if "erreur generation" in text_norm or "erreur génération" in text_norm:
        errors.append("Generation error exposed in final answer.")
    if "ollama timeout" in text_norm or "timeout" in text_norm and "erreur llm" in text_norm:
        errors.append("Timeout error detected in final answer.")
    if "no such column" in text_norm or "sql" in text_norm and "erreur" in text_norm:
        errors.append("SQL error detected in final answer.")
    if retrieval_status == "retrieval_error":
        errors.append("Retrieval error status detected.")
    if any(m in text_norm for m in ["traceback", "exception", "ollama timeout", "chunk_id="]):
        errors.append("raw_error_visible")

    citation_present = ("[doc_id=" in text) or ("doc_id=" in text) or bool(structured_sources)
    if displayed and not citation_present:
        errors.append("Missing citations while evidence exists.")

    evidence_doc_set = {
        str(ev.get("doc_id") or "").strip().lower()
        for ev in (displayed if displayed else evidence_pack)
        if str(ev.get("doc_id") or "").strip()
    }
    for src in structured_sources:
        src_doc = str(src.get("doc_id") or "").strip().lower()
        src_url = src.get("url")
        src_viewer = src.get("viewer_url")
        src_label = str(src.get("label") or "")
        if src_doc and evidence_doc_set and src_doc not in evidence_doc_set:
            errors.append("source_evidence_doc_mismatch")
        if "chunk_id=" in _norm(src_label):
            warnings.append("source_label_contains_chunk_id")
        if src_url:
            u = str(src_url).strip()
            if not u.startswith("/api/documents/"):
                errors.append("source_url_invalid_prefix")
            if "../" in u or "..\\" in u:
                errors.append("source_url_path_traversal")
            if "/home/" in u or "\\home\\" in u or re.search(r"^[a-zA-Z]:[\\/]", u):
                errors.append("source_url_local_path_leak")
            m = re.match(r"^/api/documents/([^/?#]+)/pdf(?:[?#].*)?$", u)
            if m:
                url_doc = str(m.group(1) or "").strip().lower()
                if src_doc and url_doc != src_doc:
                    errors.append("source_url_docid_mismatch")
            else:
                errors.append("source_url_pattern_invalid")
        if src_viewer:
            v = str(src_viewer).strip()
            if not v.startswith("/viewer/"):
                errors.append("viewer_url_invalid_prefix")
            if "../" in v or "..\\" in v:
                errors.append("viewer_url_path_traversal")
            if "/home/" in v or "\\home\\" in v or re.search(r"^[a-zA-Z]:[\\/]", v):
                errors.append("viewer_url_local_path_leak")

    allowed_source_labels = {
        _norm(str(s.get("label") or ""))
        for s in structured_sources
        if str(s.get("label") or "").strip()
    }
    for ev in (displayed if displayed else evidence_pack):
        src_label = str(ev.get("source_label") or ev.get("source") or "").strip()
        if src_label and "doc_id=" not in _norm(src_label):
            allowed_source_labels.add(_norm(src_label))
    mentioned_source_labels = _extract_link_labels(text) + _extract_sources_block_labels(text)
    unsupported_source_labels = []
    for lbl in mentioned_source_labels:
        if not lbl.strip():
            continue
        if "page " not in _norm(lbl) and ".pdf" not in _norm(lbl):
            continue
        if allowed_source_labels and not _source_label_supported(lbl, allowed_source_labels):
            unsupported_source_labels.append(lbl)
    if unsupported_source_labels:
        errors.append("unsupported_source")
        unsupported_claims.append(f"Unsupported source labels: {sorted(set(unsupported_source_labels))}")

    allowed = _extract_allowed_sets(displayed if displayed else evidence_pack)

    allowed_analytes_from_evidence = {
        _norm(str(ev.get("analyte_norm") or ev.get("analyte") or ev.get("parameter") or ""))
        for ev in (displayed if displayed else evidence_pack)
        if str(ev.get("analyte_norm") or ev.get("analyte") or ev.get("parameter") or "").strip()
    }
    mentioned_analytes_global = find_analyte_mentions(core_text)
    is_presence_diff = bool((query_intents or {}).get("multi_doc_presence_diff") or (query_intents or {}).get("multi_doc_comparison"))
    if mentioned_analytes_global and allowed_analytes_from_evidence and not is_presence_diff:
        allowed_canonical = {_canonical_analyte_key(a) for a in allowed_analytes_from_evidence}
        bad_analytes = sorted(
            a
            for a in mentioned_analytes_global
            if _canonical_analyte_key(a) not in allowed_canonical and a not in {"tsh", "tshus"}
        )
        if bad_analytes:
            errors.append("unsupported_analyte")
            unsupported_claims.append(f"Unsupported analytes: {bad_analytes}")

    # Unsupported numerics
    unsupported_numeric: list[str] = []
    unsupported_units: list[str] = []
    if not is_general_conversation_query:
        requested_value_norm = _norm(str(requested_value or "")).replace(".", ",")
        requested_value_alt = _norm(str(requested_value or "")).replace(",", ".")
        for token in _extract_numeric_tokens_for_validation(core_text):
            if _norm(token) in {"0", "1"}:
                continue
            token_norm = _norm(token)
            if requested_value and token_norm in {requested_value_norm, requested_value_alt}:
                continue
            if not _value_supported_by_evidence(token, allowed, displayed if displayed else evidence_pack):
                unsupported_numeric.append(token)
        if unsupported_numeric:
            unsupported_claims.append(f"Unsupported numeric values: {sorted(set(unsupported_numeric))}")
            warnings.append("Some numeric values were not found in evidence.")
            errors.append("unsupported_value")

        # Unsupported units
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
    source_count = max(
        len(re.findall(r"\[doc_id=", answer_text or "", flags=re.IGNORECASE)),
        len(re.findall(r"\bdoc_id=", answer_text or "", flags=re.IGNORECASE)),
        len(structured_sources),
    )
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
        errors.append("hallucinated_diagnosis")

    # Insufficient context handling
    insufficient_context_handled = False
    no_evidence = len(displayed if displayed else evidence_pack) == 0
    sensitive_query = any(k in qn for k in ["nom du patient", "patient", "date de naissance", "prescripteur"]) or any(
        k in qn for k in ["traitement", "prescrire", "posologie"]
    )
    is_guardrail_mode = generation_mode == "guardrail_blocked"
    has_insufficient_sentence = INSUFFICIENT_CONTEXT_SENTENCE.lower() in text_norm or (
        "information insuffisante dans le contexte fourni" in text_norm
    ) or (
        "information non retrouvee" in text_norm
    ) or (
        "information non retrouvée" in text_norm
    ) or (
        "aucun resultat correspondant n a ete retrouve" in text_norm
    ) or (
        "aucun resultat correspondant n'a ete retrouve" in text_norm
    ) or (
        "aucun resultat correspondant n a été retrouve" in text_norm
    ) or (
        "aucun résultat correspondant n’a été retrouvé" in text_norm
    ) or (
        "aucun resultat correspondant" in text_norm and "retrouve" in text_norm
    )

    if no_evidence:
        if (answer_style_requested or "").strip().lower() == "yes_no":
            compact = core_norm.strip()
            insufficient_context_handled = compact.startswith("non") or compact.startswith("no") or has_insufficient_sentence
        elif is_general_conversation_query:
            insufficient_context_handled = True
        else:
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

    if source_chunk_ids:
        source_alignment_pass = set(source_chunk_ids) == set(displayed_chunk_ids)
        if not source_alignment_pass:
            errors.append("source_alignment_mismatch")
            errors.append("unsupported_source")
            unsupported_claims.append(
                f"source_chunk_ids={sorted(set(source_chunk_ids))}, displayed_evidence_chunk_ids={sorted(set(displayed_chunk_ids))}"
            )
    else:
        displayed_doc_set = {str(d).strip().lower() for d in displayed_doc_ids if str(d).strip()}
        source_doc_set = {str(d).strip().lower() for d in (source_doc_ids + structured_source_doc_ids) if str(d).strip()}
        source_alignment_pass = (not source_doc_set) or (source_doc_set == displayed_doc_set)
        if not source_alignment_pass:
            errors.append("source_alignment_mismatch_doc_level")
            errors.append("unsupported_source")
            unsupported_claims.append(
                f"source_doc_ids={sorted(source_doc_set)}, displayed_doc_ids={sorted(displayed_doc_set)}"
            )
    citation_coverage = 1.0
    if source_chunk_ids and displayed_chunk_ids:
        citation_coverage = len(set(source_chunk_ids)) / max(1, len(set(displayed_chunk_ids)))
    elif (source_doc_ids or structured_source_doc_ids) and displayed_doc_ids:
        citation_coverage = len({str(d).strip().lower() for d in (source_doc_ids + structured_source_doc_ids) if str(d).strip()}) / max(
            1, len({str(d).strip().lower() for d in displayed_doc_ids if str(d).strip()})
        )

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
    if stale_query and not is_general_conversation_query:
        errors.append("stale_response_detection")

    requested_doc_id_norm = str(requested_doc_id or "").strip().lower()
    requested_doc_id_mismatch = False
    if requested_doc_id_norm:
        bad_display_docs = sorted({d for d in displayed_doc_ids if str(d).strip().lower() != requested_doc_id_norm})
        all_source_docs = list(source_doc_ids) + list(structured_source_doc_ids)
        bad_source_docs = sorted({d for d in all_source_docs if str(d).strip().lower() != requested_doc_id_norm})
        if bad_display_docs or bad_source_docs:
            requested_doc_id_mismatch = True
            errors.append("requested_doc_id_mismatch")
            errors.append("doc_id_mismatch")
            unsupported_claims.append(
                f"requested_doc_id={requested_doc_id_norm}, displayed_doc_ids={sorted(set(displayed_doc_ids))}, source_doc_ids={sorted(set(all_source_docs))}"
            )

    missing_doc_ids = [str(d).strip().lower() for d in (missing_requested_doc_ids or []) if str(d).strip()]
    requested_doc_ids_incomplete = False
    if len(requested_doc_ids_norm) >= 2:
        represented_docs = {
            str(d).strip().lower()
            for d in displayed_doc_ids + source_doc_ids + structured_source_doc_ids
            if str(d).strip()
        }
        represented_or_missing = represented_docs | set(missing_doc_ids)
        not_covered = sorted(d for d in requested_doc_ids_norm if d not in represented_or_missing)
        if not_covered:
            # Multi-doc comparison can legitimately return data from one doc only,
            # as long as the answer explicitly states missing/present-only status.
            if not ("present uniquement" in core_norm or "présent uniquement" in core_norm or "non retrouve" in core_norm or "non retrouvé" in core_norm):
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
        if not any(
            k in core_norm
            for k in [
                "comparaison",
                "valeur actuelle",
                "plus elevee",
                "plus basse",
                "egale",
                "variation",
                "augmente",
                "diminue",
                "stable",
            ]
        ):
            errors.append("missing_current_vs_previous_comparison")

    if diagnostic_safety_intent:
        if "on ne peut pas conclure a un diagnostic" not in core_norm and "on ne peut pas conclure à un diagnostic" not in core_norm:
            if "on ne peut pas conclure a un cancer" not in core_norm and "on ne peut pas conclure à un cancer" not in core_norm:
                errors.append("missing_diagnostic_safety_refusal")
        if any(k in core_norm for k in ["oui", "certain", "confirme", "confirmé"]) and "cancer" in core_norm:
            errors.append("diagnostic_claim_detected")
            errors.append("diagnostic_safety_violation")

    if (output_format_requested or "").strip().lower() == "table":
        lines = [ln.strip() for ln in (core_text or "").splitlines() if ln.strip()]
        has_table = False
        for i in range(len(lines) - 1):
            if "|" in lines[i] and re.search(r"^\|?\s*[-:| ]+\s*\|?\s*$", lines[i + 1]):
                has_table = True
                break
        if not has_table:
            errors.append("output_format_not_respected")
        elif requested_table_columns:
            header_keys = _parse_markdown_table_header_keys(core_text)
            req_keys = [str(c).strip().lower() for c in requested_table_columns if str(c).strip()]
            if header_keys and req_keys and header_keys != req_keys:
                errors.append("output_columns_not_respected")
                errors.append("exact_columns_not_respected")
        if source_clickable_requested:
            has_source_col = _table_has_source_column(core_text)
            has_clickable_structured = any(str(s.get("url") or s.get("viewer_url") or "").strip() for s in structured_sources)
            if not has_source_col and not has_clickable_structured:
                errors.append("clickable_source_missing")
    if (output_format_requested or "").strip().lower() == "yes_no":
        compact = core_norm.strip()
        if not (
            compact.startswith("oui")
            or compact.startswith("non")
            or compact.startswith("yes")
            or compact.startswith("no")
            or compact.startswith("impossible a determiner")
            or compact.startswith("cannot determine")
        ):
            errors.append("output_format_not_respected")
            errors.append("yes_no_not_respected")
    if (answer_style_requested or "").strip().lower() == "yes_no":
        compact = core_norm.strip()
        if not (
            compact.startswith("oui")
            or compact.startswith("non")
            or compact.startswith("yes")
            or compact.startswith("no")
            or compact.startswith("impossible a determiner")
            or compact.startswith("cannot determine")
        ):
            errors.append("yes_no_not_respected")

    if intents.get("is_structured_query") and (answer_style_requested or "").strip().lower() != "yes_no" and (
        output_format_requested or ""
    ).strip().lower() != "json":
        core_lines = [ln for ln in (core_text or "").splitlines()]
        non_empty = [ln for ln in core_lines if ln.strip()]
        has_table = False
        table_idx = -1
        for i in range(max(0, len(non_empty) - 1)):
            if "|" in non_empty[i] and re.search(r"^\|?\s*[-:| ]+\s*\|?\s*$", non_empty[i + 1].strip()):
                has_table = True
                table_idx = i
                break
        if has_table and table_idx == 0:
            errors.append("missing_professional_intro")
        elif non_empty and re.match(r"^\s*[-*]\s+", non_empty[0]) and len(non_empty) > 2:
            # Structured list answer should start with a short context sentence.
            errors.append("missing_professional_intro")
        if has_table and "conclusion technique" not in core_norm:
            warnings.append("missing_conclusion")

    if source_clickable_requested:
        has_source_col_any = _table_has_source_column(text)
        has_clickable_structured = any(str(s.get("url") or s.get("viewer_url") or "").strip() for s in structured_sources)
        has_clickable_markdown = bool(re.search(r"\[[^\]]+\]\((/api/documents/|/viewer/)[^)]+\)", text, flags=re.IGNORECASE))
        if not has_source_col_any and not has_clickable_structured and not has_clickable_markdown:
            errors.append("clickable_source_missing")

    intro_block = (core_text or "").split("\n\n")[0].strip() if (core_text or "").strip() else ""
    intro_sentences = [s for s in re.split(r"[.!?]+", intro_block) if s.strip()]
    if (
        (answer_style_requested or "").strip().lower() != "yes_no"
        and (output_format_requested or "").strip().lower() != "json"
        and not is_general_conversation_query
    ):
        if len(intro_sentences) > 2:
            warnings.append("over_verbose_intro")
        if requested_value:
            intro_norm = _norm(intro_block)
            rv = _norm(str(requested_value))
            op = _norm(str(comparison_operator or ""))
            has_value = bool(rv and rv in intro_norm)
            has_operator_hint = any(
                k in intro_norm for k in ["ou plus", "ou moins", "superieur", "supérieur", "inferieur", "inférieur", "egal", "égal"]
            )
            if not has_value or (op and not has_operator_hint):
                warnings.append("missing_query_criterion_in_intro")
            if op == ">" and ("superieure ou egale" in intro_norm or "supérieure ou égale" in intro_norm or "ou plus" in intro_norm):
                errors.append("numeric_operator_mismatch")
            if op == "<" and ("inferieure ou egale" in intro_norm or "inférieure ou égale" in intro_norm or "ou moins" in intro_norm):
                errors.append("numeric_operator_mismatch")
        if "conclusion technique" not in core_norm:
            warnings.append("missing_conclusion")
    if re.search(r"\btshus\s*,\s*tsh\b|\btsh\s*,\s*tshus\b", intro_block, flags=re.IGNORECASE):
        errors.append("internal_alias_leak")

    if str(output_format_requested or "").strip().lower() == "json":
        stripped = (answer_text or "").strip()
        if not (stripped.startswith("{") or stripped.startswith("[")):
            errors.append("format_not_respected")
            errors.append("strict_json_violation")
        try:
            import json

            json.loads(stripped)
        except Exception:
            errors.append("strict_json_violation")
        if "sources :" in _norm(stripped) or "réponse :" in _norm(stripped) or "reponse :" in _norm(stripped):
            errors.append("strict_json_violation")
    if "output_format_not_respected" in errors:
        errors.append("format_not_respected")

    result_count_match = re.search(r"\b(\d+)\s+r[ée]sultat(?:s)?\b", core_text or "", flags=re.IGNORECASE)
    if result_count_match:
        declared = int(result_count_match.group(1))
        actual = len(displayed if displayed else evidence_pack)
        if declared != actual:
            errors.append("wrong_result_count")
    if intents.get("is_structured_query") and generation_mode in {"llm", "llm_fallback_template"}:
        warnings.append("llm_used_for_structured_query")

    if intents.get("comment_without_measured_value"):
        has_troponine_comment = any("troponine" in _norm(str(ev.get("text_excerpt") or "")) for ev in (displayed if displayed else evidence_pack))
        if has_troponine_comment and "information insuffisante" in core_norm:
            errors.append("insufficient_but_available")

    if intents.get("global_patient_lookup"):
        allowed_patients = {
            _norm(str(ev.get("patient_token") or ""))
            for ev in (displayed if displayed else evidence_pack)
            if str(ev.get("patient_token") or "").strip()
        }
        mentioned_patients = {
            _norm(m)
            for m in re.findall(r"\bPAT[_\-][A-Z0-9]+\b", answer_text or "", flags=re.IGNORECASE)
            if str(m).strip()
        }
        bad_patients = sorted(p for p in mentioned_patients if p not in allowed_patients)
        if bad_patients:
            errors.append("unsupported_patient")
            unsupported_claims.append(f"Unsupported patients: {bad_patients}")

        has_evidence = len(displayed if displayed else evidence_pack) > 0
        if has_evidence and any(k in core_norm for k in ["information insuffisante", "information non retrouvee", "information non retrouvée"]):
            errors.append("cohort_search_empty_but_evidence_exists")
    if intents.get("cohort_search") and requested_technical_condition:
        expected = str(requested_technical_condition).strip().lower()
        for ev in (displayed if displayed else evidence_pack):
            got = _norm(str(ev.get("interpretation_status") or ev.get("technical_status_code") or ""))
            if got and expected and got != expected:
                errors.append("cohort_condition_not_applied")
                break

    if requested_analyte_list and "tshus" in requested_analyte_list:
        if any(k in core_norm for k in ["trak", "anticorps anti recepteur de la tsh", "anti recepteur de la tsh"]):
            errors.append("analyte_overmatch")

    if any(k in qn for k in ["hors reference", "hors de la reference", "outside reference", "out of reference"]):
        if "dans la reference" in core_norm or "within_reference" in core_norm:
            errors.append("filter_violation_hors_reference")

    if "non retrouve" in core_norm or "non retrouvé" in core_norm:
        for analyte in requested_analyte_list:
            if analyte in found_requested_norms:
                errors.append("false_missing_item")
                break

    if (query_intents or {}).get("comment_without_measured_value"):
        if any(k in core_norm for k in ["aucun resultat exploitable", "information non retrouvee", "information non retrouvée"]):
            errors.append("comment_only_misclassified_as_no_result")

    if is_general_conversation_query:
        if (displayed if displayed else evidence_pack) or structured_sources or _contains_medical_like(core_norm):
            errors.append("general_conversation_no_retrieval_violation")
            if small_talk_query:
                errors.append("small_talk_triggered_retrieval")
        if any(k in core_norm for k in ["doc_id", "report_", "sources :", "/api/documents/", "/viewer/", "| --- |"]):
            errors.append("general_conversation_no_retrieval_violation")
            if small_talk_query:
                errors.append("small_talk_content_violation")
        if find_analyte_mentions(core_text):
            errors.append("general_conversation_no_retrieval_violation")
            if small_talk_query:
                errors.append("small_talk_content_violation")
        if re.search(
            r"\b\d+(?:[.,]\d+)?\s*(pg/ml|ng/ml|mg/l|ug/ml|uui?/ml|uu/ml|mui/l|mui/ml|pmol/l|mmol/l|ui/l)\b",
            core_text or "",
            flags=re.IGNORECASE,
        ):
            errors.append("general_conversation_no_retrieval_violation")
            if small_talk_query:
                errors.append("small_talk_content_violation")
        if check_internal_reasoning_leak(answer_text or ""):
            errors.append("internal_reasoning_leak")
        if general_conversation_intent == "identity_question":
            if not any(
                k in core_norm
                for k in [
                    "assistant medical rag",
                    "assistant medical",
                    "medical rag",
                    "rapports medicaux",
                    "rapports biologiques",
                    "sources pdf",
                ]
            ):
                errors.append("identity_answer_required")

    if (query_intents or {}).get("response_transform"):
        if "pas de réponse précédente" in (answer_text or "").lower() or "pas de reponse precedente" in (answer_text or "").lower():
            errors.append("response_transform_missing_context")

    if (query_intents or {}).get("multi_doc_comparison"):
        if len(requested_doc_ids_norm) >= 2:
            if not all(d in core_norm for d in requested_doc_ids_norm[:2]):
                errors.append("multi_doc_comparison_not_performed")

    if (query_intents or {}).get("multi_doc_presence_diff"):
        for line in (core_text or "").splitlines():
            if "|" in line and "analyte" in _norm(line):
                continue
            if line.strip().startswith("- "):
                label = line.strip()[2:].split("|")[0].strip()
                if label and not _is_valid_presence_analyte_label(label):
                    warnings.append("presence_diff_noise_analyte")
                    break

    if any(sentence in text_norm for sentence in _GENERIC_COLD_SENTENCES):
        warnings.append("repeated_generic_style")

    if (answer_style_requested or "").strip().lower() == "yes_no" and missing_requested:
        compact = core_norm.strip()
        if not (compact.startswith("non") or compact.startswith("no")):
            errors.append("absent_analyte_yes_no_format")

    if requested_analyte_list:
        for missing_analyte in missing_requested:
            warnings.append(f"missing_requested_analyte:{missing_analyte}")

    # Reference support strictness: each displayed reference should appear in evidence.
    displayed_refs = []
    for line in (core_text or "").splitlines():
        m = re.search(r"référence\s*:\s*([^|;\n]+)", line, flags=re.IGNORECASE)
        if m:
            raw_ref = re.sub(r"[)\].,;:]+$", "", m.group(1).strip())
            ref_value = _norm(raw_ref)
            if ref_value in {"", "non disponible", "n a", "na"}:
                continue
            displayed_refs.append(ref_value)
    allowed_refs = {
        _norm(str(ev.get("reference_range") or ev.get("reference") or ""))
        for ev in (displayed if displayed else evidence_pack)
        if str(ev.get("reference_range") or ev.get("reference") or "").strip()
    }
    bad_refs = [r for r in displayed_refs if r and r not in allowed_refs]
    if bad_refs:
        errors.append("unsupported_reference")
        unsupported_claims.append(f"Unsupported references: {sorted(set(bad_refs))}")

    # Previous result support strictness.
    prev_mentions_inline = re.findall(r"(?:antérieur|anterieur)\s*:\s*([^|;\n]+)", core_text, flags=re.IGNORECASE)
    allowed_prev = {_norm(str(ev.get("previous_result") or "")) for ev in (displayed if displayed else evidence_pack) if str(ev.get("previous_result") or "").strip()}
    bad_prev = [
        v
        for v in prev_mentions_inline
        if _norm(v)
        and _norm(v) not in {"non disponible", "n a", "na", "none", "null"}
        and _norm(v) not in allowed_prev
    ]
    if bad_prev:
        errors.append("unsupported_previous_result")
        unsupported_claims.append(f"Unsupported previous results: {sorted(set(_norm(v) for v in bad_prev))}")

    if generation_mode and generation_mode.startswith("llm") and (
        "unsupported_value" in errors or "unsupported_reference" in errors or "unsupported_previous_result" in errors
    ):
        errors.append("llm_hallucination")

    if not (displayed if displayed else evidence_pack):
        if re.search(r"\|\s*[^|\n]+\s*\|", core_text or "") or re.search(r"\bPAT[_\-]\w+\b", answer_text or "", flags=re.IGNORECASE):
            errors.append("no_evidence_hallucination")

    if structured_sources:
        source_keys = [
            (
                str(s.get("doc_id") or "").strip().lower(),
                s.get("page"),
                s.get("row"),
            )
            for s in structured_sources
        ]
        if len(source_keys) != len(set(source_keys)):
            warnings.append("source_duplication")

    if any(
        marker in _norm(answer_text)
        for marker in [
            "pre tokenize",
            "inference embeddings",
            "loading weights",
            "fetching",
            "warning you are sending unauthenticated",
        ]
    ):
        errors.append("raw_logs_visible")

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
        "source_doc_ids": sorted(
            {
                str(d).strip()
                for d in (source_doc_ids + structured_source_doc_ids)
                if str(d).strip()
            }
        ),
        "displayed_evidence_doc_ids": displayed_doc_ids,
        "source_citation_count": len(structured_sources),
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
