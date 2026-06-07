from __future__ import annotations

import re
import unicodedata
from typing import Any

from config_loader import get_safety_guardrails_config
from prompt_builder import INSUFFICIENT_CONTEXT_SENTENCE
from query_understanding import contains_exact_term, detect_exact_analyte, detect_exact_analytes, find_analyte_mentions
try:
    from medical_entity_resolver import (
        canonicalize_analyte,
        are_equivalent_analytes,
        get_aliases_for_canonical,
        get_analyte_family,
        is_analyte_match,
    )
except Exception:  # pragma: no cover
    from scripts.generation.medical_entity_resolver import (  # type: ignore
        canonicalize_analyte,
        are_equivalent_analytes,
        get_aliases_for_canonical,
        get_analyte_family,
        is_analyte_match,
    )


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
    "total_results_count",
    "abnormal_rows_count",
    "within_reference_rows_count",
    "ambiguous_rows_count",
    "evidence_rows_count",
    "llm_evidence_rows_count",
    "raw_debug",
    "debug.",
    "fallback_reason",
    "validation_status",
    "generation_mode",
    "generation_writer",
    "selected_route",
    "technical_condition",
    "requested_doc_ids",
    "requested_analytes",
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

_CHART_LEAK_PATTERNS = [
    r"INSULINE[A-Z0-9]+",
    r"TSHus[A-Z0-9]+",
    r"ANTI-TG-[0-9]+%",
    r"[0-9]+%Écart",
    r"Écart normalisé à la référence",
]

_ROBOTIC_VIZ_PATTERNS = [
    r"Graphique demandé\s*:",
    r"Rendu affiché\s*:",
    r"Raison\s*:",
]

_GENERIC_VIZ_CONCLUSIONS = [
    "données structurées fournies pour visualisation côté interface",
    "vérifiez le backend",
    "impossible de générer",
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


def _diagnostic_guardrails() -> dict[str, Any]:
    cfg = dict(get_safety_guardrails_config() or {})
    diag = cfg.get("diagnostic_safety") or {}
    return dict(diag) if isinstance(diag, dict) else {}


def _thyroid_topic_keywords() -> list[str]:
    diag = _diagnostic_guardrails()
    vals = [str(v).strip().lower() for v in list(diag.get("thyroid_topic_keywords") or []) if str(v).strip()]
    return vals or ["hyperthyro", "hypothyro", "thyroid", "thyroide", "thyroïde"]


def _diagnostic_strong_suggestion_patterns() -> list[str]:
    diag = _diagnostic_guardrails()
    vals = [str(v).strip() for v in list(diag.get("strong_suggestion_patterns") or []) if str(v).strip()]
    return vals or [
        r"\bsugg[eè]re\s+une?\s+hyperthyro",
        r"\bcompatible\s+avec\s+une?\s+hyperthyro",
        r"\b[eé]voque\s+une?\s+hyperthyro",
        r"\bindique\s+une?\s+hyperthyro",
        r"\ben\s+faveur\s+d['’]une?\s+hyperthyro",
    ]


def _diagnostic_explicit_negation_markers() -> list[str]:
    diag = _diagnostic_guardrails()
    vals = [str(v).strip().lower() for v in list(diag.get("explicit_negation_markers") or []) if str(v).strip()]
    return vals or [
        "ne permet pas de conclure",
        "n est pas suffisant pour conclure",
        "n'est pas suffisant pour conclure",
        "on ne peut pas conclure",
    ]


def _reason_present_in_answer(reason: str, answer_norm: str) -> bool:
    base = _norm(reason or "")
    if not base:
        return True
    if base in answer_norm:
        return True
    tokens = [
        tok
        for tok in re.findall(r"[a-z0-9]+", base)
        if len(tok) >= 5
        and tok
        not in {
            "dans",
            "avec",
            "pour",
            "cette",
            "comme",
            "donnee",
            "donnees",
            "graphique",
            "radar",
            "barres",
            "courbe",
            "nuage",
            "points",
            "heatmap",
        }
    ]
    if not tokens:
        return False
    window = tokens[:6]
    hit_count = sum(1 for tok in window if tok in answer_norm)
    min_hits = 2 if len(window) >= 3 else 1
    return hit_count >= min_hits


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
        # Remove analyte-like labels that include numeric tokens (e.g. T3, T4, CA 15-3).
        line = re.sub(r"\bT\d+\b", " ", line, flags=re.IGNORECASE)
        line = re.sub(r"\bCA\s*\d+(?:\s*[-/]\s*\d+)+\b", " ", line, flags=re.IGNORECASE)
        line = re.sub(r"\breport[_\-]?\d+\b", " ", line, flags=re.IGNORECASE)
        line = re.sub(r"\breport\s+\d+\b", " ", line, flags=re.IGNORECASE)
        line = re.sub(r"\breport\s*\(\s*\d+\s*\)", " ", line, flags=re.IGNORECASE)
        line = re.sub(r"\bpage\s+\d+\b", " ", line, flags=re.IGNORECASE)
        line = re.sub(r"\bligne\s+\d+\b", " ", line, flags=re.IGNORECASE)
        line = re.sub(r"\bline\s+\d+\b", " ", line, flags=re.IGNORECASE)
        line = re.sub(r"\brow\s+\d+\b", " ", line, flags=re.IGNORECASE)
        line = re.sub(r"\bpat[_\-]?\d+\b", " ", line, flags=re.IGNORECASE)

        # Normalize spaces around decimal separators to keep "0,45" as one token.
        line = re.sub(r"(\d)\s*([.,])\s*(\d)", r"\1\2\3", line)
        for t in re.findall(r"(?<!\d)[-+]?\d+(?:[.,]\d+)?(?!\d)", line):
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
    if any(m in qn for m in ["tes qui", "t es qui", "tu es qui", "qui es tu", "who are you", "what are you", "vous etes qui", "c est qui toi"]):
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
        elif "priorite" in c or "priorité" in c:
            key = "priorite"
        elif "raison technique" in c:
            key = "raison_technique"
        elif "patient" in c:
            key = "patient"
        elif "report" in c or "rapport" in c or "document" in c:
            key = "report"
        keys.append(key)
    return keys


def _is_markdown_table(text: str) -> bool:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    for i in range(len(lines) - 1):
        if "|" in lines[i] and re.search(r"^\|?\s*[-:| ]+\s*\|?\s*$", lines[i + 1]):
            return True
    return False


def _extract_reference_only_section(answer_text: str) -> str:
    txt = str(answer_text or "")
    if not txt.strip():
        return ""
    lines = txt.splitlines()
    start_idx = -1
    start_re = re.compile(
        r"(?im)^\s*(?:[-*]\s*)?"
        r"(?:resultats?\s+dans\s+la\s+reference(?:\s+uniquement)?|"
        r"résultats?\s+dans\s+la\s+référence(?:\s+uniquement)?|"
        r"resultats?\s+strictement\s+dans\s+la\s+reference|"
        r"résultats?\s+strictement\s+dans\s+la\s+référence|"
        r"normaux|rassurants|within reference)\s*:"
    )
    stop_re = re.compile(
        r"(?im)^\s*(?:[-*]\s*)?"
        r"(?:anormaux|anomalies|conclusion(?:\s+technique)?|sources?|priorit[eé])\s*:"
    )
    for idx, line in enumerate(lines):
        if start_re.search(line):
            start_idx = idx
            break
    if start_idx < 0:
        return ""
    out: list[str] = []
    for idx in range(start_idx, len(lines)):
        line = lines[idx].strip()
        if idx > start_idx and stop_re.search(line):
            break
        out.append(line)
    return "\n".join(out).strip()


def _extract_section_block(answer_text: str, section_title_norm: str) -> str:
    txt = str(answer_text or "")
    if not txt:
        return ""
    lines = txt.splitlines()
    start = -1
    for idx, line in enumerate(lines):
        ln = _norm(line)
        if section_title_norm in ln and ":" in ln:
            start = idx
            break
    if start < 0:
        return ""
    out: list[str] = []
    for idx in range(start, len(lines)):
        ln = lines[idx].strip()
        if idx > start and re.search(r"(?im)^(priorit[eé]|conclusion technique)\s*:", ln):
            break
        out.append(ln)
    return "\n".join(out).strip()


def _canonical_analyte_key(value: str) -> str:
    canonical = canonicalize_analyte(str(value or ""))
    if canonical:
        return canonical.replace("_", " ")
    key = _norm(value).replace("_", " ")
    key = re.sub(r"\s+", " ", key).strip()
    return key


def _multi_doc_analyte_represented(core_norm: str, analyte: str) -> bool:
    a = _canonical_analyte_key(analyte)
    alias_groups = {
        "anti tg": {"anti tg", "anti-tg", "anti_tg"},
        "tshus": {"tshus", "tsh us", "tsh_us"},
        "t4 libre": {"t4 libre", "t4_libre", "ft4"},
        "t3 libre": {"t3 libre", "t3_libre", "ft3"},
    }
    aliases = alias_groups.get(a, {a})
    try:
        dyn_aliases = {str(x or "").strip() for x in (get_aliases_for_canonical(analyte) or set())}
    except Exception:
        dyn_aliases = set()
    for raw in dyn_aliases:
        k = _canonical_analyte_key(raw)
        if k and len(k) >= 3:
            aliases.add(k)
            aliases.add(k.replace("_", " ").strip())
    aliases = {al for al in aliases if isinstance(al, str) and len(al.strip()) >= 3}
    return any(alias in core_norm for alias in aliases)


def _thyroid_equivalent_requested_keys(analyte: str) -> set[str]:
    key = _canonical_analyte_key(analyte)
    if key in {"tsh", "tshus"} or are_equivalent_analytes(key, "tsh"):
        return {"tsh", "tshus"}
    return {key}


def _analytes_equivalent_or_same_family(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    if are_equivalent_analytes(left, right):
        return True
    l_family = get_analyte_family(left)
    r_family = get_analyte_family(right)
    return bool(l_family and r_family and l_family == r_family)


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
    raw_format_phrase: str | None = None,
    unsupported_presentation: bool = False,
    user_requested_visualization: bool = False,
    requested_chart_type: str | None = None,
    visualization_payload: dict[str, Any] | None = None,
    chart_data_payload: dict[str, Any] | None = None,
    patients: list[dict[str, Any]] | None = None,
    inventory_view: dict[str, Any] | None = None,
    transformable_context_available: bool | None = None,
    previous_intent: str | None = None,
) -> dict[str, Any]:
    text = (answer_text or "").strip()
    text_norm = _norm(text)
    qn_query = _norm(query or "")
    core_text = _split_answer_core(text)
    core_norm = _norm(core_text)
    displayed = displayed_evidences or []
    generation_mode_norm = str(generation_mode or "").strip().lower()
    structured_first_mode = generation_mode_norm.startswith("llm_") or generation_mode_norm in {"llm_professional_writer", "hybrid_structured_llm_writer"}
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
    if (
        (output_format_requested or "").strip().lower() != "json"
        and re.search(r"(?<![A-Za-z])None(?![A-Za-z])", text)
    ):
        errors.append("forbidden_none_literal")
    if re.search(r"(?im)^\s*(?:[-*]\s*)?(?:priorit[eé]\s+[^\n:]+|conclusion technique|anormaux|r[eé]sultats?\s+dans\s+la\s+r[eé]f[ée]rence[^\n:]*)\s*:\s*\.\.\.\s*$", core_text or ""):
        errors.append("output_contains_placeholder_ellipsis")

    forbidden_internal_hits = [m for m in _FORBIDDEN_INTERNAL_MARKERS if m in text_norm]
    if forbidden_internal_hits or "/home/" in text or "\\home\\" in text or re.search(r"[A-Za-z]:\\", text):
        errors.append("forbidden_internal_field")
    if any(
        marker in text_norm
        for marker in [
            "total_results_count",
            "abnormal_rows_count",
            "within_reference_rows_count",
            "ambiguous_rows_count",
            "evidence_rows_count",
            "llm_evidence_rows_count",
            "raw_debug",
            "debug.",
            "fallback_reason",
            "validation_status",
            "generation_mode",
            "generation_writer",
            "selected_route",
            "technical_condition",
            "requested_doc_ids",
            "requested_analytes",
        ]
    ):
        errors.append("internal_debug_leak")
    if re.search(r"résultat\(s\)|correspondant\(s\)", text, flags=re.IGNORECASE):
        warnings.append("ugly_pluralization")
    if (
        re.search(r"page\s*\d+\s*row\s*\d+", text_norm, flags=re.IGNORECASE)
        or re.search(r"ligne\s*\d+\s*ligne\s*\d+", text_norm, flags=re.IGNORECASE)
        or "chunk_id" in text_norm
        or "/home/" in text
        or "\\home\\" in text
        or re.search(r"[A-Za-z]:\\", text)
    ):
        errors.append("source_format_bad")
    if re.search(r"\brendu\s+chart\b", text_norm, flags=re.IGNORECASE):
        errors.append("render_internal_term_leak")
    if re.search(r"\btype\s+chart\b", text_norm, flags=re.IGNORECASE):
        errors.append("internal_chart_term_visible")
    if re.search(r"\bchart\b", text_norm, flags=re.IGNORECASE):
        errors.append("internal_chart_term_visible")
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
            "deterministic_safety_fallback_after_llm_validation_failure",
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
            if not (u.startswith("/api/documents/") or u.startswith("/viewer/")):
                errors.append("source_url_invalid_prefix")
            if "../" in u or "..\\" in u:
                errors.append("source_url_path_traversal")
            if "/home/" in u or "\\home\\" in u or re.search(r"^[a-zA-Z]:[\\/]", u):
                errors.append("source_url_local_path_leak")
            if u.startswith("/api/documents/"):
                m = re.match(r"^/api/documents/([^/?#]+)/pdf(?:[?#].*)?$", u)
                if m:
                    url_doc = str(m.group(1) or "").strip().lower()
                    if src_doc and url_doc != src_doc:
                        errors.append("source_url_docid_mismatch")
                else:
                    errors.append("source_url_pattern_invalid")
            elif u.startswith("/viewer/"):
                # viewer links are valid user-facing sources for deterministic reference-range answers
                pass
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
    is_toxicology_deterministic_mode = generation_mode_norm in {
        "deterministic_global_toxicology_search",
        "deterministic_doc_scoped_toxicology_threshold_search",
        "deterministic_doc_scoped_toxicology_summary",
    }
    is_summary_deterministic_mode = generation_mode_norm in {
        "deterministic_doc_scoped_abnormal_results",
        "deterministic_doc_scoped_biological_summary",
        "deterministic_doc_scoped_priority_anomalies",
    }
    is_doc_scoped_summary_like = bool(
        (query_intents or {}).get("doc_scoped_summary")
        or (query_intents or {}).get("doc_scoped_biological_summary")
        or (query_intents or {}).get("doc_scoped_abnormal_results")
        or (query_intents or {}).get("doc_scoped_priority_anomalies")
    )
    explicit_query_analytes = [str(a).strip().lower() for a in (detect_exact_analytes(query) or []) if str(a).strip()]
    if (not structured_first_mode) and mentioned_analytes_global and allowed_analytes_from_evidence and not is_presence_diff:
        allowed_canonical = {_canonical_analyte_key(a) for a in allowed_analytes_from_evidence}
        bad_analytes = sorted(
            a
            for a in mentioned_analytes_global
            if _canonical_analyte_key(a) not in allowed_canonical and a not in {"tsh", "tshus"}
        )
        is_global_abnormal = bool((query_intents or {}).get("cohort_search")) and any(
            k in _norm(query)
            for k in ["rapports disponibles", "tous les rapports", "ensemble des rapports", "quels documents", "documents"]
        )
        # Summary-like routes without an explicit analyte request are allowed to mention
        # heterogeneous analytes from evidence rows; avoid false unsupported_analyte flags.
        skip_summary_mismatch = (is_doc_scoped_summary_like or is_summary_deterministic_mode) and not explicit_query_analytes
        if bad_analytes and not is_global_abnormal and not is_toxicology_deterministic_mode and not skip_summary_mismatch:
            errors.append("unsupported_analyte")
            unsupported_claims.append(f"Unsupported analytes: {bad_analytes}")

    # Unsupported numerics
    unsupported_numeric: list[str] = []
    unsupported_units: list[str] = []
    if (not structured_first_mode) and (not is_general_conversation_query):
        requested_value_norm = _norm(str(requested_value or "")).replace(".", ",")
        requested_value_alt = _norm(str(requested_value or "")).replace(",", ".")
        if generation_mode_norm not in {"deterministic_global_toxicology_search", "deterministic_doc_scoped_toxicology_summary"}:
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
        irrelevant = sorted(
            a
            for a in detected
            if not _analytes_equivalent_or_same_family(_canonical_analyte_key(a), _canonical_analyte_key(exact_analyte))
        )
        if irrelevant:
            errors.append("irrelevant_analyte_in_answer")
            unsupported_claims.append(
                f"Irrelevant analyte mentions for exact query '{exact_analyte}': {irrelevant}"
            )

        bad_evidence_ids: list[str] = []
        for ev in evidence_pack:
            if not is_analyte_match(str(exact_analyte or ""), ev):
                bad_evidence_ids.append(str(ev.get("chunk_id") or "unknown_chunk"))
        if bad_evidence_ids:
            warnings.append("non_exact_analyte_evidence_present")
            unsupported_claims.append(
                f"Evidence contains non-exact analyte entries for '{exact_analyte}': {bad_evidence_ids}"
            )

    if requested_analyte_list:
        requested_set = set(requested_analyte_list)
        effective_missing_requested = list(missing_requested)
        # Guarded thyroid asks should only require analytes that are actually present in evidence rows.
        if diagnostic_safety_intent and any(k in qn_query for k in _thyroid_topic_keywords()):
            evidence_norms = {
                _canonical_analyte_key(str(ev.get("analyte_norm") or ev.get("analyte") or ""))
                for ev in (displayed if displayed else evidence_pack)
                if str(ev.get("analyte_norm") or ev.get("analyte") or "").strip()
            }
            requested_restricted: set[str] = set()
            for req in requested_set:
                eqs = _thyroid_equivalent_requested_keys(req)
                if evidence_norms.intersection(eqs):
                    requested_restricted.add(req)
            if requested_restricted:
                requested_set = requested_restricted
                filtered_missing: list[str] = []
                for item in effective_missing_requested:
                    item_key = _canonical_analyte_key(item)
                    item_equivs = _thyroid_equivalent_requested_keys(item_key)
                    if item_equivs.intersection(requested_set):
                        filtered_missing.append(item)
                effective_missing_requested = filtered_missing
        coverage_set = set(found_requested) | set(missing_requested)
        coverage_with_equiv: set[str] = set(coverage_set)
        for item in coverage_set:
            coverage_with_equiv.update(_thyroid_equivalent_requested_keys(item))
        uncovered = sorted(a for a in requested_set if a not in coverage_with_equiv)
        if uncovered:
            errors.append("requested_analyte_coverage_incomplete")
            unsupported_claims.append(f"Requested analytes without found/missing status: {uncovered}")
        if effective_missing_requested and generation_mode != "deterministic_measured_value_vs_comment_sql_template":
            warnings.append("controlled_warning_missing_requested_analytes")

        displayed_norms = {str(ev.get("analyte_norm") or "").strip().lower() for ev in displayed if ev.get("analyte_norm")}
        if found_requested_norms:
            allowed_norms = {_canonical_analyte_key(n) for n in set(found_requested_norms)}
            bad_displayed = sorted(
                n
                for n in displayed_norms
                if not any(_analytes_equivalent_or_same_family(_canonical_analyte_key(n), allowed) for allowed in allowed_norms)
            )
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
        "deterministic_single_analyte_lookup",
    }
    if (
        generation_mode not in relaxed_line_source_modes
        and result_line_count >= 2
        and source_count < result_line_count
        and not diagnostic_safety_intent
    ):
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
            errors.append("treatment_recommendation")
    if any(re.search(p, lower_core) for p in _DIAGNOSIS_PATTERNS):
        errors.append("Definitive diagnosis detected.")
        errors.append("hallucinated_diagnosis")
        errors.append("diagnostic_affirmation")

    # Section consistency for LLM summaries.
    ref_only_section = _extract_reference_only_section(core_text)
    if structured_first_mode and ref_only_section:
        ref_only_norm = _norm(ref_only_section)
        if any(
            k in ref_only_norm
            for k in [
                "au-dessus",
                "au dessus",
                "above_reference",
                "below_reference",
                "en dessous",
                "sous reference",
                "sous référence",
            ]
        ):
            errors.append("section_status_mismatch")
        abnormal_analytes = {
            _norm(str(ev.get("analyte") or ev.get("parameter") or ""))
            for ev in (displayed if displayed else evidence_pack)
            if str(ev.get("analyte") or ev.get("parameter") or "").strip()
            and _norm(str(ev.get("interpretation_status") or ev.get("technical_status_code") or "")) in {"above_reference", "below_reference"}
        }
        if abnormal_analytes and any(a and a in ref_only_norm for a in abnormal_analytes):
            errors.append("abnormal_in_reassuring_section")

    # Guardrail for biological summaries: do not claim "no abnormalities" when evidence contains abnormalities.
    summary_modes = {"hybrid_structured_llm_writer", "deterministic_doc_scoped_biological_summary"}
    if generation_mode_norm in summary_modes and (
        (query_intents or {}).get("doc_scoped_summary")
        or (query_intents or {}).get("doc_scoped_biological_summary")
    ):
        summary_evidences = list(evidence_pack if evidence_pack else displayed)
        abnormal_rows = [
            ev
            for ev in summary_evidences
            if _norm(str(ev.get("technical_status_code") or ev.get("interpretation_status") or "")) in {"above_reference", "below_reference", "out_of_reference"}
        ]
        if abnormal_rows:
            no_abnormal_markers = [
                "anormaux : aucun",
                "anormaux: aucun",
                "aucun fait anormal",
                "aucune anomalie",
                "aucun resultat anormal",
                "aucun résultat anormal",
            ]
            if any(m in core_norm for m in no_abnormal_markers):
                errors.append("false_no_abnormal_summary")
            abnormal_analytes = {
                _norm(str(ev.get("analyte") or ev.get("parameter") or ""))
                for ev in abnormal_rows
                if str(ev.get("analyte") or ev.get("parameter") or "").strip()
            }
            if abnormal_analytes:
                abnormal_block_norm = _norm(_extract_section_block(core_text, "anormaux"))
                mention_space = abnormal_block_norm or core_norm
                mentioned = any(
                    _multi_doc_analyte_represented(mention_space, analyte)
                    for analyte in abnormal_analytes
                    if analyte
                )
                if not mentioned and abnormal_block_norm:
                    mentioned = any(
                        marker in abnormal_block_norm
                        for marker in [
                            "au dessus",
                            "au-dessus",
                            "en dessous",
                            "above_reference",
                            "below_reference",
                        ]
                    )
                if not mentioned:
                    errors.append("summary_missing_abnormal_coverage")

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
        "information insuffisante dans les donnees structurees disponibles" in text_norm
    ) or (
        "information insuffisante dans les données structurées disponibles" in text_norm
    ) or (
        "information non retrouvee" in text_norm
    ) or (
        "information non retrouvée" in text_norm
    ) or (
        "aucune recherche" in text_norm and "exploitable" in text_norm and "retrouve" in text_norm
    ) or (
        "aucun resultat" in text_norm and "exploitable" in text_norm and "retrouve" in text_norm
    ) or (
        "aucun résultat" in text_norm and "exploitable" in text_norm and "retrouve" in text_norm
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
    ) or (
        "clarification d analyte requise" in text_norm
    ) or (
        "clarification d’analyte requise" in core_norm
    ) or (
        "clarification de perimetre requise" in text_norm
    ) or (
        "clarification de périmètre requise" in core_norm
    )

    if no_evidence:
        safety_refusal_modes = {
            "deterministic_treatment_refusal_with_technical_summary",
            "deterministic_diagnostic_safety_refusal",
            "deterministic_diagnostic_refusal_with_technical_summary",
            "deterministic_pii_refusal",
        }
        if (answer_style_requested or "").strip().lower() == "yes_no":
            compact = core_norm.strip()
            insufficient_context_handled = compact.startswith("non") or compact.startswith("no") or has_insufficient_sentence
        elif is_general_conversation_query:
            insufficient_context_handled = True
        elif generation_mode_norm in safety_refusal_modes:
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
        has_alias_compatible_evidence = bool(
            requested_analyte_list
            and (displayed or evidence_pack)
            and any(
                any(is_analyte_match(req, ev) for req in requested_analyte_list)
                for ev in (displayed if displayed else evidence_pack)
            )
        )
        if (
            (displayed or evidence_pack)
            and has_insufficient_sentence
            and not is_guardrail_mode
            and not missing_requested
            and not has_alias_compatible_evidence
        ):
            errors.append("Insufficient-context answer returned despite available evidence.")

    requested_analytes = detected_analytes or detect_exact_analytes(query)
    mentioned_analytes = find_analyte_mentions(core_text)
    if requested_analytes and (not structured_first_mode):
        allowed_mentions = {_canonical_analyte_key(str(a).strip().lower()) for a in requested_analytes if str(a).strip()}
        allowed_mentions.update({_canonical_analyte_key(a) for a in found_requested_norms})
        if "hdl" in allowed_mentions:
            allowed_mentions.add("cholesterol_hdl")
        bad_mentions = sorted(
            a
            for a in mentioned_analytes
            if not any(
                _analytes_equivalent_or_same_family(_canonical_analyte_key(a), allowed)
                for allowed in allowed_mentions
            )
        )
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
        if generation_mode_norm == "deterministic_doc_scoped_priority_anomalies":
            bad_source_docs = []
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
        has_refusal = any(
            phrase in core_norm
            for phrase in [
                "on ne peut pas conclure a un diagnostic",
                "on ne peut pas conclure à un diagnostic",
                "on ne peut pas conclure a un cancer",
                "on ne peut pas conclure à un cancer",
                "je ne peux pas poser ni evoquer un diagnostic",
                "je ne peux pas poser ni évoquer un diagnostic",
                "je ne peux pas poser de diagnostic",
            ]
        )
        if not has_refusal:
                errors.append("missing_diagnostic_safety_refusal")
        if any(k in core_norm for k in ["oui", "certain", "confirme", "confirmé"]) and "cancer" in core_norm:
            errors.append("diagnostic_claim_detected")
            errors.append("diagnostic_safety_violation")
        qn_diag = _norm(query or "")
        if any(k in qn_diag for k in _thyroid_topic_keywords()):
            has_thyroid_block = any(k in core_norm for k in ["t4 libre", "t3 libre", "tshus", "anti tg"])
            if not has_thyroid_block:
                errors.append("guarded_thyroid_interpretation_missing_thyroid_facts")
            if "acth" in core_norm and not has_thyroid_block:
                errors.append("guarded_thyroid_interpretation_acth_misfocus")
            has_explicit_negation = any(
                k in core_norm
                for k in _diagnostic_explicit_negation_markers()
            )
            if not has_explicit_negation:
                for patt in _diagnostic_strong_suggestion_patterns():
                    if re.search(patt, core_norm, flags=re.IGNORECASE):
                        errors.append("diagnostic_suggestion_too_strong")
                        break

    if (output_format_requested or "").strip().lower() == "table":
        priority_structured_prose = bool(
            (query_intents or {}).get("doc_scoped_priority_anomalies")
            and all(
                k in core_norm
                for k in ["priorite elevee", "priorite moderee", "conclusion technique"]
            )
        ) or bool(
            (query_intents or {}).get("doc_scoped_priority_anomalies")
            and all(
                k in core_norm
                for k in ["priorité élevée", "priorité modérée", "conclusion technique"]
            )
        )
        if priority_structured_prose:
            # Priority route accepts concise prose blocks as equivalent to table intent.
            pass
        else:
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
    if generation_mode_norm == "deterministic_doc_scoped_priority_anomalies":
        header_keys = _parse_markdown_table_header_keys(core_text)
        required_priority_keys = {"priorite", "analyte", "valeur_actuelle", "reference", "statut", "raison_technique"}
        if header_keys and required_priority_keys.issubset(set(header_keys)):
            errors = [e for e in errors if e not in {"output_columns_not_respected", "exact_columns_not_respected"}]
        else:
            # Accept structured prose for priority route when no markdown table is produced.
            has_structured_prose = all(
                k in core_norm
                for k in ["priorite elevee", "priorite moderee", "conclusion technique"]
            ) or all(
                k in core_norm
                for k in ["priorité élevée", "priorité modérée", "conclusion technique"]
            )
            if has_structured_prose:
                errors = [e for e in errors if e not in {"output_format_not_respected", "format_not_respected", "output_columns_not_respected", "exact_columns_not_respected"}]
                # Ensure backend priority_level is respected in prose sections.
                high_block = _norm(_extract_section_block(core_text, "priorite elevee"))
                moderate_block = _norm(_extract_section_block(core_text, "priorite moderee"))
                high_analytes = {
                    _canonical_analyte_key(str(ev.get("analyte") or ev.get("analyte_norm") or ""))
                    for ev in (displayed if displayed else evidence_pack)
                    if str(ev.get("priority_level") or "").strip().lower() == "high"
                }
                moderate_low_analytes = {
                    _canonical_analyte_key(str(ev.get("analyte") or ev.get("analyte_norm") or ""))
                    for ev in (displayed if displayed else evidence_pack)
                    if str(ev.get("priority_level") or "").strip().lower() in {"moderate", "low"}
                }
                mismatch = False
                for analyte in high_analytes:
                    if analyte and analyte in moderate_block:
                        mismatch = True
                        break
                if not mismatch:
                    for analyte in moderate_low_analytes:
                        if analyte and analyte in high_block:
                            mismatch = True
                            break
                if mismatch:
                    errors.append("priority_level_mismatch")
                # Ensure priority sections cover all provided backend facts.
                coverage_missing = False
                for analyte in high_analytes:
                    if analyte and analyte not in high_block:
                        coverage_missing = True
                        break
                if not coverage_missing:
                    for analyte in moderate_low_analytes:
                        if analyte and analyte not in moderate_block:
                            coverage_missing = True
                            break
                if coverage_missing:
                    errors.append("section_coverage_missing")
                if re.search(r"\b(\d+(?:[.,]\d+)?)\s*-\s*\1\b", core_text or ""):
                    errors.append("suspicious_reference_collapse")
        if source_clickable_requested:
            has_source_col = _table_has_source_column(core_text)
            has_clickable_structured = any(str(s.get("url") or s.get("viewer_url") or "").strip() for s in structured_sources)
            if not has_source_col and not has_clickable_structured:
                errors.append("clickable_source_missing")
    if (query_intents or {}).get("doc_scoped_priority_anomalies") and structured_first_mode:
        high_block = _norm(_extract_section_block(core_text, "priorite elevee"))
        moderate_block = _norm(_extract_section_block(core_text, "priorite moderee"))
        high_analytes = {
            _canonical_analyte_key(str(ev.get("analyte") or ev.get("analyte_norm") or ""))
            for ev in (displayed if displayed else evidence_pack)
            if str(ev.get("priority_level") or "").strip().lower() == "high"
        }
        moderate_low_analytes = {
            _canonical_analyte_key(str(ev.get("analyte") or ev.get("analyte_norm") or ""))
            for ev in (displayed if displayed else evidence_pack)
            if str(ev.get("priority_level") or "").strip().lower() in {"moderate", "low"}
        }
        mismatch = False
        for analyte in high_analytes:
            if analyte and analyte in moderate_block:
                mismatch = True
                break
        if not mismatch:
            for analyte in moderate_low_analytes:
                if analyte and analyte in high_block:
                    mismatch = True
                    break
        if mismatch:
            errors.append("priority_level_mismatch")
        coverage_missing = False
        for analyte in high_analytes:
            if analyte and analyte not in high_block:
                coverage_missing = True
                break
        if not coverage_missing:
            for analyte in moderate_low_analytes:
                if analyte and analyte not in moderate_block:
                    coverage_missing = True
                    break
        if coverage_missing:
            errors.append("section_coverage_missing")
        if re.search(r"\b(\d+(?:[.,]\d+)?)\s*-\s*\1\b", core_text or ""):
            errors.append("suspicious_reference_collapse")
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
    if (output_format_requested or "").strip().lower() == "chart":
        viz = visualization_payload or {}
        requested_type = str(viz.get("requested_type") or requested_chart_type or "").strip().lower()
        rendered_type = str(viz.get("rendered_type") or (chart_data_payload.get("rendered_type") if chart_data_payload else "")).strip().lower()
        requested_label = _norm(str(viz.get("requested_label") or ""))
        rendered_label = _norm(str(viz.get("rendered_label") or ""))
        fallback_used = bool(viz.get("fallback_used"))
        fallback_reason = str(viz.get("fallback_reason") or "").strip()
        supported = bool(viz.get("supported")) if viz else None
        suitable = bool(viz.get("suitable")) if viz else None
        viz_data = list(viz.get("data") or (chart_data_payload.get("data") if chart_data_payload else []) or [])

        if user_requested_visualization and not viz:
            errors.append("visualization_payload_missing")
        if "reference_ratio" in core_norm or _norm(str(viz.get("y_field") or "")) == "reference_ratio" or _norm(
            str((chart_data_payload or {}).get("y_field") or "")
        ) == "reference_ratio":
            errors.append("bad_metric_label")
        if requested_label and requested_label not in core_norm:
            errors.append("requested_visualization_not_respected")
        if fallback_used and rendered_type and requested_type and requested_type != rendered_type:
            has_fallback_explainer = any(
                k in core_norm
                for k in [
                    "vous avez demande",
                    "vous avez demandé",
                    "rendu affiche",
                    "rendu affiché",
                    "j affiche donc",
                    "j’affiche donc",
                    "alternative",
                    "raison",
                ]
            )
            if not has_fallback_explainer:
                errors.append("silent_visualization_fallback")
            if rendered_label and rendered_label not in core_norm:
                errors.append("fallback_alternative_not_mentioned")
            if fallback_reason and not _reason_present_in_answer(fallback_reason, core_norm):
                errors.append("fallback_reason_missing_in_answer")
        if supported is False and fallback_used and not fallback_reason:
            errors.append("unsupported_visualization_without_reason")
        if requested_type == "bar" and supported is True and suitable is True and rendered_type and rendered_type != "bar":
            errors.append("wrong_rendered_type")

        if viz_data:
            missing_tooltip_rows = []
            for idx, row in enumerate(viz_data):
                if not isinstance(row, dict):
                    continue
                required = ["raw_value", "unit", "reference", "status"]
                if any(k not in row for k in required):
                    missing_tooltip_rows.append(idx)
            if missing_tooltip_rows:
                errors.append("tooltip_fields_missing")

            for row in viz_data:
                if not isinstance(row, dict):
                    continue
                analyte_norm = _norm(str(row.get("analyte") or ""))
                reference_text = str(row.get("reference") or "")
                metric_available = bool(row.get("metric_available"))
                deviation = row.get("reference_deviation")
                status_row = _norm(str(row.get("status_code") or row.get("status") or ""))

                if ("anti tg" in analyte_norm or "anti-tg" in analyte_norm) and reference_text.strip().startswith("<"):
                    if not metric_available or deviation in (None, ""):
                        errors.append("antitg_ratio_missing")
                        break

                if ("below_reference" in status_row or "en dessous" in status_row) and metric_available and isinstance(deviation, (int, float)):
                    if float(deviation) >= 0:
                        errors.append("below_reference_negative_deviation")
                        break

        has_chart_explanation = any(
            k in core_norm
            for k in [
                "graphique",
                "visualisation",
                "visualization",
                "graph",
                "non disponible",
                "pas adapte",
                "pas adapté",
                "donnees structurees",
                "données structurées",
                "rendu cote interface",
                "rendu côté interface",
                "barres",
                "ratio",
                "ecart normalise",
                "écart normalisé",
            ]
        )
        has_table_only = _is_markdown_table(core_text)
        if not has_chart_explanation:
            errors.append("unsupported_format_silently_ignored")
        if has_table_only and not has_chart_explanation:
            errors.append("output_format_mismatch")
        units = {str(ev.get("unit") or "").strip().lower() for ev in (displayed if displayed else evidence_pack) if str(ev.get("unit") or "").strip()}
        if len(units) > 1 and not any(k in core_norm for k in ["unites differentes", "unités différentes", "ratio", "barres", "bar chart"]):
            warnings.append("chart_units_warning_missing")
        if str(requested_chart_type or "").strip().lower() == "line" and len(units) > 1:
            if not any(k in core_norm for k in ["unites differentes", "unités différentes", "ratio", "normalis"]):
                errors.append("unsuitable_chart_without_warning")
        if str(requested_chart_type or "").strip().lower() == "bar" and "graphique en barres" not in core_norm:
            errors.append("bar_chart_phrase_missing")

        repeated_explanation_markers = [
            "unites biologiques",
            "unités biologiques",
            "ecart normalise",
            "écart normalisé",
        ]
        for marker in repeated_explanation_markers:
            if core_norm.count(_norm(marker)) > 1:
                warnings.append("duplicate_explanation")
                break
        if "donnees structurees fournies pour visualisation cote interface" in core_norm:
            warnings.append("generic_conclusion")
    raw_format_norm = _norm(str(raw_format_phrase or ""))
    if unsupported_presentation and raw_format_norm:
        mentions_requested_format = raw_format_norm in core_norm
        has_limit_explanation = any(
            k in core_norm
            for k in [
                "non supporte",
                "non supporté",
                "pas supporte",
                "pas supporté",
                "necessite un composant graphique",
                "nécessite un composant graphique",
                "format alternatif",
                "recommandation",
                "visualisation",
                "visualization",
            ]
        )
        if not (mentions_requested_format or has_limit_explanation):
            errors.append("no_silent_default_table")
    if user_requested_visualization and (output_format_requested or "").strip().lower() != "chart":
        has_visualization_explainer = any(
            k in core_norm for k in ["graphique", "visualisation", "visualization", "format alternatif", "recommandation"]
        )
        if not has_visualization_explainer:
            errors.append("unsupported_format_silently_ignored")

    if re.search(r"\b(chart\.js|chartjs|html|javascript|python)\b", core_norm, flags=re.IGNORECASE) and any(
        k in core_norm for k in ["genere", "génère", "execute", "exécute", "copie ce code", "script"]
    ):
        errors.append("no_code_execution")
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
        no_evidence_like = (
            "information insuffisante" in core_norm
            or "aucun resultat biologique exploitable" in core_norm
            or "aucune recherche" in core_norm
        )
        has_conclusion_header = ("conclusion technique" in core_norm) or ("conclusion de prudence" in core_norm)
        if has_table and not has_conclusion_header and not no_evidence_like and generation_mode != "deterministic_reference_range_lookup" and not (query_intents or {}).get("reference_range_lookup"):
            warnings.append("missing_conclusion")

    if source_clickable_requested:
        has_source_col_any = _table_has_source_column(text)
        has_clickable_structured = any(str(s.get("url") or s.get("viewer_url") or "").strip() for s in structured_sources)
        has_clickable_markdown = bool(re.search(r"\[[^\]]+\]\((/api/documents/|/viewer/)[^)]+\)", text, flags=re.IGNORECASE))
        if not has_source_col_any and not has_clickable_structured and not has_clickable_markdown:
            errors.append("clickable_source_missing")

    intro_block = (core_text or "").split("\n\n")[0].strip() if (core_text or "").strip() else ""
    intro_sentences = [s for s in re.split(r"[.!?]+", intro_block) if s.strip()]
    section_intro_ok = bool(
        re.match(
            r"(?is)^\s*(anormaux|résultats?\s+dans\s+la\s+référence\s+uniquement|resultats?\s+dans\s+la\s+reference\s+uniquement|priorité\s+élevée|priorite\s+elevee|priorité\s+modérée/faible|priorite\s+moderee/faible|faits\s+techniques\s+observés|faits\s+techniques\s+observes|limites|conclusion\s+technique|synthèse\s+toxicologique\s+technique|synthese\s+toxicologique\s+technique|résultat\s+correspondant|resultat\s+correspondant|note\s+de\s+synth[eè]se\s+m[ée]dicale|note\s+m[ée]dicale)\s*[:—-]?",
            intro_block or "",
            flags=re.IGNORECASE,
        )
    )
    if (
        (answer_style_requested or "").strip().lower() != "yes_no"
        and (output_format_requested or "").strip().lower() != "json"
        and not is_general_conversation_query
    ):
        is_multi_doc_single_analyte_deterministic = (
            generation_mode_norm == "deterministic_single_analyte_lookup"
            and len(requested_doc_ids_norm) >= 2
            and len(requested_analyte_list) == 1
        )
        if len(intro_sentences) > 2 and not section_intro_ok and not is_multi_doc_single_analyte_deterministic:
            status_counts = {
                "above": 0,
                "below": 0,
                "within": 0,
                "context": 0,
            }
            for ev in displayed:
                status = str(ev.get("technical_status_code") or ev.get("interpretation_status") or ev.get("status") or "").strip().lower()
                if status == "above_reference":
                    status_counts["above"] += 1
                elif status == "below_reference":
                    status_counts["below"] += 1
                elif status == "within_reference":
                    status_counts["within"] += 1
                elif status == "needs_clinical_context":
                    status_counts["context"] += 1
            rich_biological_summary = (
                generation_mode_norm in {"llm_professional_writer", "hybrid_structured_llm_writer"}
                and len(displayed) >= 5
                and (status_counts["above"] + status_counts["below"] + status_counts["within"]) >= 5
                and status_counts["within"] >= 1
            )
            if not rich_biological_summary:
                warnings.append("over_verbose_intro")
        if requested_value and (answer_style_requested or "").strip().lower() not in {"doctor_note"}:
            intro_norm = _norm(intro_block)
            rv = _norm(str(requested_value))
            op = _norm(str(comparison_operator or ""))
            directional_status_query = any(
                token in qn_query
                for token in [
                    "hors reference",
                    "hors norme",
                    "est il bas",
                    "est-il bas",
                    "est elle basse",
                    "est-elle basse",
                    "est il haut",
                    "est-il haut",
                    "est elle haute",
                    "est-elle haute",
                    "dans la reference",
                    "dans la norme",
                ]
            )
            has_value = bool(rv and rv in intro_norm)
            has_operator_hint = any(
                k in intro_norm for k in ["ou plus", "ou moins", "superieur", "supérieur", "inferieur", "inférieur", "egal", "égal"]
            )
            if (not directional_status_query) and (not has_value or (op and not has_operator_hint)):
                warnings.append("missing_query_criterion_in_intro")
            if op == ">" and ("superieure ou egale" in intro_norm or "supérieure ou égale" in intro_norm or "ou plus" in intro_norm):
                errors.append("numeric_operator_mismatch")
            if op == "<" and ("inferieure ou egale" in intro_norm or "inférieure ou égale" in intro_norm or "ou moins" in intro_norm):
                errors.append("numeric_operator_mismatch")
        no_evidence_like = (
            "information insuffisante" in core_norm
            or "aucun resultat biologique exploitable" in core_norm
            or "aucune recherche" in core_norm
        )
        has_conclusion_header = ("conclusion technique" in core_norm) or ("conclusion de prudence" in core_norm)
        if not has_conclusion_header and not no_evidence_like and generation_mode != "deterministic_reference_range_lookup" and not (query_intents or {}).get("reference_range_lookup"):
            warnings.append("missing_conclusion")
    if re.search(r"\btshus\s*,\s*tsh\b|\btsh\s*,\s*tshus\b", intro_block, flags=re.IGNORECASE):
        errors.append("internal_alias_leak")
    if (output_format_requested or "").strip().lower() != "json":
        if re.search(
            r"\b(?:t4_libre|psa_totale|ca_15_3|acide_valproique|cholesterol_ldl|cholesterol_hdl|pro_bnp|peptide_c)\b",
            core_norm,
            flags=re.IGNORECASE,
        ):
            errors.append("display_name_required")

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
        has_qualitative_comment = any(
            any(marker in _norm(str(ev.get("text_excerpt") or "")) for marker in ["commentaire", "valeur seuil", "attention"])
            for ev in (displayed if displayed else evidence_pack)
        )
        if has_qualitative_comment and "information insuffisante" in core_norm:
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

    if requested_analyte_list and "tshus" in requested_analyte_list and generation_mode_norm != "deterministic_multi_doc_comparison":
        if any(k in core_norm for k in ["trak", "anticorps anti recepteur de la tsh", "anti recepteur de la tsh"]):
            errors.append("analyte_overmatch")

    if any(k in qn for k in ["hors reference", "hors de la reference", "outside reference", "out of reference"]):
        doc_scoped_single_status_query = bool(requested_doc_ids) and len(requested_analyte_list) == 1 and any(
            token in qn
            for token in ["est il", "est-il", "est elle", "est-elle", "la valeur", "donne", "quelle est la valeur"]
        )
        if (
            ("dans la reference" in core_norm or "within_reference" in core_norm)
            and not doc_scoped_single_status_query
        ):
            errors.append("filter_violation_hors_reference")

    if "non retrouve" in core_norm or "non retrouvé" in core_norm:
        if generation_mode_norm == "deterministic_multi_doc_comparison":
            # In multi-doc comparison, missing in one document is expected.
            # Flag only if an analyte is globally missing and also not represented in answer lines.
            for analyte in requested_analyte_list:
                if analyte in found_requested_norms and not _multi_doc_analyte_represented(core_norm, analyte):
                    errors.append("false_missing_item")
                    break
        else:
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

    if (query_intents or {}).get("response_transform") and transformable_context_available is False:
        no_context_markers = [
            "pas de résultat précédent exploitable",
            "pas de resultats biologiques numeriques recents a transformer",
            "pas des valeurs médicales transformables",
            "veuillez d’abord demander les résultats",
        ]
        if not any(m in core_norm for m in no_context_markers):
            errors.append("response_transform_no_context_clean_missing")
        if isinstance(visualization_payload, dict):
            if str(visualization_payload.get("rendered_type") or "").strip():
                errors.append("response_transform_no_context_visualization_forbidden")
        if any(k in core_norm for k in ["ace", "psa totale", "ca 15-3"]) and str(previous_intent or "").strip().lower() in {"patient_inventory", "patient_inventory_count"}:
            errors.append("response_transform_old_medical_context_leak")

    if (query_intents or {}).get("multi_doc_comparison"):
        if len(requested_doc_ids_norm) >= 2:
            if not all(d in core_norm for d in requested_doc_ids_norm[:2]):
                errors.append("multi_doc_comparison_not_performed")
        forbidden = [
            "présents dans un rapport et absents dans l’autre",
            "présent dans un rapport et absent dans l’autre",
            "différence technique",
        ]
        if any(f in core_norm for f in forbidden):
            errors.append("multi_doc_comparison_generic_or_wrong_wording")
        if "aucun écart numérique" in core_norm and ("+0" in core_norm or "-0" in core_norm):
            warnings.append("multi_doc_comparison_zero_delta_sign_noise")

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

    # Structured-first validation for llm writer:
    # keep style flexible, but reject unsupported factual medical values.
    if structured_first_mode and displayed:
        allowed_values = set()
        for ev in displayed:
            for key in ("current_value", "value_raw", "value"):
                raw_v = str(ev.get(key) or "").strip()
                if raw_v:
                    allowed_values.add(_norm(raw_v))
                    allowed_values.add(_norm(raw_v.replace(",", ".")))
        factual_numbers: list[str] = []
        for m in re.finditer(
            r"(\d+(?:[.,]\d+)?)\s*(pg/ml|ng/ml|mg/l|ug/ml|ug/dl|uui?/ml|uu/ml|mui/l|mui/ml|pmol/l|mmol/l|ui/l)\b",
            core_text or "",
            flags=re.IGNORECASE,
        ):
            factual_numbers.append(m.group(1))
        for m in re.finditer(
            r"(\d+(?:[.,]\d+)?)\s*[–-]\s*(\d+(?:[.,]\d+)?)\s*(pg/ml|ng/ml|mg/l|ug/ml|ug/dl|uui?/ml|uu/ml|mui/l|mui/ml|pmol/l|mmol/l|ui/l)\b",
            core_text or "",
            flags=re.IGNORECASE,
        ):
            factual_numbers.extend([m.group(1), m.group(2)])
        unsupported = []
        for tok in factual_numbers:
            n = _norm(tok)
            if n in {"0", "1"}:
                continue
            if n not in allowed["values"] and n not in allowed_values:
                unsupported.append(tok)
        if unsupported:
            errors.append("unsupported_value")
            unsupported_claims.append(f"Unsupported numeric values (structured-first): {sorted(set(unsupported))}")

    # In structured-first mode, style/wording issues should not fail validation
    # when factual grounding is preserved by structured evidence checks.
    if structured_first_mode and errors:
        downgradable_exact = {
            "missing_professional_intro",
            "bar_chart_phrase_missing",
            "over_verbose_intro",
        }
        retained_errors: list[str] = []
        for err in errors:
            if err in downgradable_exact:
                warnings.append(f"style_issue:{err}")
                continue
            retained_errors.append(err)
        errors = retained_errors

    if (answer_style_requested or "").strip().lower() == "yes_no" and missing_requested:
        compact = core_norm.strip()
        if not (compact.startswith("non") or compact.startswith("no")):
            errors.append("absent_analyte_yes_no_format")

    if requested_analyte_list and not (query_intents or {}).get("multi_doc_comparison"):
        suppress_missing_requested_warning = bool(
            diagnostic_safety_intent
            and any(k in qn_query for k in _thyroid_topic_keywords())
        )
        for missing_analyte in missing_requested:
            if suppress_missing_requested_warning:
                continue
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

    if not (displayed if displayed else evidence_pack) and generation_mode not in {"deterministic_patient_inventory", "deterministic_patient_count"}:
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

    # Patient Inventory Specific Validation
    if generation_mode in {"deterministic_patient_inventory", "deterministic_patient_count"}:
        forbidden_inventory_markers = [
            "non precise", "non-precise", "non précisé", "non-précisé",
            "non disponible", "n/a", "reference non disponible", "référence non disponible",
            "statut non interpretable", "statut non interprétable",
            "analyte", "valeur actuelle", "reference", "référence", "statut"
        ]
        low_answer = _norm(answer_text)
        found_forbidden = [m for m in forbidden_inventory_markers if m in low_answer]
        if found_forbidden:
            errors.append(f"patient_inventory_forbidden_markers:{found_forbidden}")
        
        if generation_mode == "deterministic_patient_inventory":
            if "patient" not in low_answer:
                errors.append("patient_inventory_requires_patient_column")
            if "sources" not in low_answer:
                errors.append("patient_inventory_requires_sources")

    # Visualization recommendation must stay advisory-only (no rendered chart payload).
    if (query_intents or {}).get("visualization_recommendation"):
        if isinstance(visualization_payload, dict) and str(visualization_payload.get("rendered_type") or "").strip():
            errors.append("visualization_recommendation_must_not_render_chart")
        if isinstance(chart_data_payload, dict) and str(chart_data_payload.get("rendered_type") or "").strip():
            errors.append("visualization_recommendation_chart_data_forbidden")
        forbidden = [
            "format visuel demande",
            "j affiche donc un format visuel demande",
            "bonjour ! je suis pret",
        ]
        for marker in forbidden:
            if marker in core_norm:
                errors.append(f"visualization_recommendation_placeholder_or_greeting:{marker}")
        if str(previous_intent or "").strip().lower() in {"patient_inventory", "patient_inventory_count"}:
            required_any = ["inventaire", "patients", "rapports"]
            if not all(k in core_norm for k in required_any):
                warnings.append("visualization_recommendation_inventory_context_missing")
        if str(previous_intent or "").strip().lower() in {"comment_without_measured_value", "qualitative_comment_render"}:
            forbidden_inventory = ["cartes patient", "accordeon", "accordéon", "timeline documentaire"]
            if any(k in core_norm for k in forbidden_inventory):
                errors.append("visualization_recommendation_wrong_context_inventory_after_qualitative")
            if any(k in core_norm for k in ["graphique en barres", "radar chart", "courbe", "line chart"]):
                errors.append("visualization_recommendation_wrong_numeric_chart_after_qualitative")

    if (query_intents or {}).get("inventory_visualization_render"):
        requested_inventory_view = str((inventory_view or {}).get("type") or "").strip().lower()
        if "bonjour" in core_norm or "je suis pret a vous accompagner" in core_norm:
            errors.append("inventory_visualization_render_smalltalk_leak")
        if isinstance(visualization_payload, dict) and str(visualization_payload.get("rendered_type") or "").strip():
            errors.append("inventory_visualization_render_chart_forbidden")
        if isinstance(chart_data_payload, dict) and str(chart_data_payload.get("rendered_type") or "").strip():
            errors.append("inventory_visualization_render_chart_data_forbidden")
        if str(previous_intent or "").strip().lower() in {"patient_inventory", "response_transform_no_context"} and not patients:
            errors.append("inventory_visualization_render_missing_patients_payload")
        if requested_inventory_view == "report_accordion" and "cartes patient" in core_norm:
            errors.append("inventory_visualization_render_wrong_copy_for_accordion")
        if requested_inventory_view == "filterable_table" and "cartes patient" in core_norm:
            errors.append("inventory_visualization_render_wrong_copy_for_table")
    if (
        any(tok in qn_query for tok in ["ca", "ça", "ces donnees", "ces données"])
        and any(tok in qn_query for tok in ["table", "tableau"])
        and str(previous_intent or "").strip().lower() in {"patient_inventory", "inventory_visualization_render", "visualization_recommendation", "response_transform_no_context"}
    ):
        if any(tok in core_norm for tok in ["pus", "residus alimentaires", "résidus alimentaires", "analyte", "valeur actuelle"]):
            errors.append("deictic_inventory_table_medical_leak")
        if not patients:
            errors.append("deictic_inventory_table_missing_patients")

    if (query_intents or {}).get("qualitative_comment_render"):
        if "bonjour" in core_norm or "je suis pret a vous accompagner" in core_norm:
            errors.append("qualitative_comment_render_smalltalk_leak")
        if isinstance(visualization_payload, dict) and str(visualization_payload.get("rendered_type") or "").strip():
            errors.append("qualitative_comment_render_chart_forbidden")
        if isinstance(chart_data_payload, dict) and str(chart_data_payload.get("rendered_type") or "").strip():
            errors.append("qualitative_comment_render_chart_data_forbidden")
        # If context exists, response should look like a sourced comment block.
        if transformable_context_available is False and "bloc commentaire source" not in core_norm and "bloc commentaire sourc" not in core_norm:
            warnings.append("qualitative_comment_render_block_copy_missing")

    if (query_intents or {}).get("reference_range_lookup"):
        has_structured_source = bool(structured_sources)
        if (
            "source :" not in core_norm
            and "| source |" not in core_norm
            and "source non disponible" not in core_norm
            and not has_structured_source
        ):
            errors.append("reference_range_source_required")
        has_reference_keyword = any(k in core_norm for k in ["plage", "intervalle", "norme", "reference", "référence"])
        has_profile_markers = any(
            marker in core_norm
            for marker in [
                "sous-profils",
                "sous profils",
                "homme",
                "femme",
                "age",
                "ans",
            ]
        )
        has_numeric_range = bool(
            re.search(r"\b\d+(?:[.,]\d+)?\s*(?:-|–|a|à)\s*\d+(?:[.,]\d+)?\s*(?:ng/ml|mg/l|iu/ml|ui/l|pmol/l|mmol/l)?\b", core_norm)
            or re.search(r"\b[<>]\s*\d+(?:[.,]\d+)?\s*(?:ng/ml|mg/l|iu/ml|ui/l|pmol/l|mmol/l)?\b", core_norm)
        )
        if not (has_reference_keyword or (has_profile_markers and has_numeric_range)):
            errors.append("reference_range_missing_main_fact")
        if "resultats correspondants ont ete retrouves" in core_norm or "résultats correspondants ont été retrouvés" in core_norm:
            errors.append("reference_range_forbidden_bulk_listing")
        if any(tok in core_norm for tok in ["| analyte |", "| valeur actuelle |", "| statut | document |"]):
            errors.append("reference_range_forbidden_multi_analyte_table")
        if "valeur actuelle" in core_norm:
            errors.append("reference_range_forbidden_current_value_render")
        if "fallback" in core_norm and not any(k in core_norm for k in ["fallback", "pas trouve de plage specifique", "pas trouvé de plage spécifique"]):
            errors.append("reference_range_fallback_not_explicit")
        has_internal_docid_leak = bool(re.search(r"(?<![?&])doc_id=", core_norm))
        if "chunk_id" in core_norm or has_internal_docid_leak or "sqlite_deterministic" in core_norm:
            errors.append("reference_range_internal_source_leak")

    if generation_mode_norm in {
        "deterministic_global_toxicology_search",
        "deterministic_doc_scoped_toxicology_threshold_search",
        "deterministic_doc_scoped_toxicology_summary",
    }:
        if "chunk_id" in core_norm or "sqlite_deterministic" in core_norm or "doc_id=" in core_norm:
            errors.append("toxicology_internal_source_leak")
        if any(x in core_norm for x in ["cristaux d acide urique", "ecbu", "cytologie urinaire"]):
            errors.append("toxicology_non_target_urine_confusion")
        if generation_mode_norm == "deterministic_doc_scoped_toxicology_threshold_search":
            for ev in displayed:
                st = str(ev.get("technical_status_code") or ev.get("interpretation_status") or "").strip().lower()
                if st != "above_reference":
                    errors.append("toxicology_threshold_non_above_result_present")
                    break
        toxicology_query = any(
            t in qn_query
            for t in ["toxiques urinaires", "toxicologie urinaire", "pharmacotoxicologie", "toxiques sanguins", "toxicologie sanguine"]
        )
        if toxicology_query and not displayed and "aucune donnee" not in core_norm and "aucune donnée" not in core_norm:
            warnings.append("toxicology_empty_evidence_pack")
    # Semantic guardrail: even if intent classification drifts, reference-range wording in query
    # must never return a global multi-analyte bulk listing.
    reference_semantic_query = any(
        token in qn_query
        for token in [
            "plage normale",
            "plage",
            "norme",
            "reference",
            "référence",
            "valeur normale",
            "valeurs physiologiques",
            "intervalle de reference",
            "intervalle de référence",
            "plage de reference",
            "plage de référence",
        ]
    )
    directional_status_query = any(
        token in qn_query
        for token in [
            "above_reference",
            "above reference",
            "au dessus",
            "au-dessus",
            "supérieure",
            "superieure",
            "supérieur",
            "superieur",
            "below_reference",
            "below reference",
            "en dessous",
            "inférieure",
            "inferieure",
            "inférieur",
            "inferieur",
            "basse",
            "bas",
            "hors reference",
            "anormal",
            "anormaux",
            "out_of_reference",
            "out of range",
        ]
    )
    if reference_semantic_query and not directional_status_query:
        if "resultats correspondants ont ete retrouves" in core_norm or "résultats correspondants ont été retrouvés" in core_norm:
            errors.append("reference_semantic_forbidden_bulk_listing")
        if any(tok in core_norm for tok in ["| analyte |", "| valeur actuelle |", "| statut | document |"]):
            errors.append("reference_semantic_forbidden_multi_analyte_table")

    if len(requested_analyte_list) == 1 and len(requested_doc_ids_norm) == 1 and displayed:
        requested_norm = _canonical_analyte_key(str(requested_analyte_list[0] or ""))
        displayed_norms = {
            _canonical_analyte_key(str(ev.get("analyte_norm") or ev.get("analyte") or ""))
            for ev in displayed
            if str(ev.get("analyte_norm") or ev.get("analyte") or "").strip()
        }
        extra = sorted([a for a in displayed_norms if a and a != requested_norm])
        if extra:
            errors.append("single_analyte_over_display")
            unsupported_claims.append(f"Single-analyte query displayed extra analytes: {extra}")

    toxicology_global_query = any(
        token in qn_query
        for token in [
            "toxiques urinaires",
            "toxicologie urinaire",
            "pharmacotoxicologie",
            "toxiques sanguins",
            "toxicologie sanguine",
            "recherche de toxiques",
        ]
    ) and any(token in qn_query for token in ["quels rapports", "quels documents", "tous les rapports", "rapports disponibles"])
    if toxicology_global_query and any(k in core_norm for k in ["aucune recherche", "aucun resultat", "aucun résultat"]):
        if displayed_evidences:
            errors.append("toxicology_false_no_evidence")

    # Global business-flow guard: no greeting small-talk leak in deterministic business intents.
    business_intent_keys = {
        "response_transform",
        "visualization_recommendation",
        "inventory_visualization_render",
        "qualitative_comment_render",
        "context_summary_render",
        "doc_scoped_results",
        "cohort_search",
        "comment_without_measured_value",
        "reference_range_lookup",
    }
    if any((query_intents or {}).get(k) for k in business_intent_keys):
        if "bonjour ! je suis pret" in core_norm or "je suis pret a vous accompagner" in core_norm:
            errors.append("business_intent_smalltalk_leak")

    # ... (existing return dict building)
    prod_checks = validate_production_ux(
        answer_text=answer_text,
        patients=patients,
        user_requested_visualization=user_requested_visualization
    )
    
    for pc in prod_checks:
        if pc["status"] == "fail":
            errors.append(f"{pc['id']}:{pc['message']}")
        elif pc["status"] == "warning":
            warnings.append(f"{pc['id']}:{pc['message']}")
    if generation_mode_norm == "deterministic_doc_scoped_priority_anomalies":
        warnings = [
            w
            for w in warnings
            if not str(w).startswith("patient_inventory_long_cell:")
        ]

    deterministic_fact_modes = {
        "deterministic_professional_fallback",
        "deterministic_evidence_template",
        "deterministic_doc_scoped_abnormal_results",
        "deterministic_doc_scoped_biological_summary",
        "deterministic_doc_scoped_priority_anomalies",
        "deterministic_anomaly_summary",
        "deterministic_global_analyte_abnormal_search",
        "deterministic_doc_pair_comparison",
        "deterministic_guarded_medical_interpretation",
        "deterministic_single_analyte_lookup",
        "deterministic_safety_fallback_after_llm_validation_failure",
    }
    if generation_mode_norm in deterministic_fact_modes and displayed:
        fact_errors = {
            "unsupported_value",
            "unsupported_reference",
            "unsupported_previous_result",
            "unsupported_source",
            "source_alignment_mismatch",
            "source_alignment_mismatch_doc_level",
            "requested_doc_id_mismatch",
            "requested_doc_ids_incomplete",
            "non_exact_analyte_evidence_present",
            "abnormal_in_reassuring_section",
            "section_status_mismatch",
            "false_no_abnormal_summary",
            "summary_missing_abnormal_coverage",
            "reference_semantic_forbidden_bulk_listing",
            "reference_range_forbidden_multi_analyte_table",
            "reference_range_forbidden_current_value_render",
        }
        retained_errors: list[str] = []
        for err in errors:
            if err in fact_errors or err.startswith("patient_inventory_"):
                retained_errors.append(err)
            else:
                warnings.append(f"downgraded_non_fact_error:{err}")
        errors = retained_errors

    if (
        generation_mode_norm in deterministic_fact_modes
        and displayed
        and bool((query_intents or {}).get("doc_scoped_numeric_result_lookup"))
        and not [str(a).strip() for a in (requested_analytes or []) if str(a).strip()]
    ):
        relaxed_numeric_lookup_errors = {
            "unsupported_value",
            "unsupported_analyte",
            "unsupported_source",
            "source_alignment_mismatch_doc_level",
            "reference_semantic_forbidden_multi_analyte_table",
            "reference_range_forbidden_multi_analyte_table",
        }
        retained: list[str] = []
        for err in errors:
            if err in relaxed_numeric_lookup_errors:
                warnings.append(f"downgraded_non_fact_error:{err}")
                continue
            retained.append(err)
        errors = retained

    if generation_mode_norm in {"deterministic_doc_pair_comparison", "deterministic_multi_doc_comparison"} and errors:
        kept: list[str] = []
        for err in errors:
            if err == "unsupported_value" and (
                "non présent" in core_text.lower() or "non present" in core_text.lower() or "non retrouvé" in core_text.lower()
            ):
                warnings.append("downgraded_non_fact_error:unsupported_value_missing_comparison_item")
                continue
            kept.append(err)
        errors = kept

    if generation_mode_norm == "deterministic_general_conversation" and errors:
        downgradable_general = {
            "general_conversation_no_retrieval_violation",
            "small_talk_content_violation",
            "small_talk_triggered_retrieval",
            "stale_response_detection",
        }
        kept: list[str] = []
        for err in errors:
            if err in downgradable_general:
                warnings.append(f"downgraded_non_fact_error:{err}")
                continue
            kept.append(err)
        errors = kept

    if generation_mode_norm == "deterministic_doc_scoped_priority_anomalies" and errors:
        downgradable_for_priority = {
            "unsupported_value",
            "requested_doc_id_mismatch",
            "source_alignment_mismatch_doc_level",
            "unsupported_source",
        }
        kept: list[str] = []
        for err in errors:
            if err in downgradable_for_priority:
                warnings.append(f"downgraded_non_fact_error:{err}")
            else:
                kept.append(err)
        errors = kept

    if generation_mode_norm in {
        "deterministic_global_toxicology_search",
        "deterministic_doc_scoped_toxicology_threshold_search",
        "deterministic_doc_scoped_toxicology_summary",
    } and errors:
        kept: list[str] = []
        for err in errors:
            if err == "unsupported_analyte":
                warnings.append("downgraded_non_fact_error:unsupported_analyte_toxicology_family_labels")
                continue
            kept.append(err)
        errors = kept

    guarded_style_only_warnings = {
        "missing_conclusion",
        "over_verbose_intro",
        "multi_result_missing_structured_details",
    }
    guarded_thyroid_contract_pass = bool(
        diagnostic_safety_intent
        and any(k in qn_query for k in _thyroid_topic_keywords())
        and not errors
        and any(
            phrase in core_norm
            for phrase in [
                "on ne peut pas conclure a un diagnostic",
                "on ne peut pas conclure à un diagnostic",
                "aucune conclusion diagnostique ne peut etre posee",
                "aucune conclusion diagnostique ne peut être posée",
            ]
        )
        and any(k in core_norm for k in ["t4 libre", "t3 libre", "tshus", "anti tg"])
        and any(k in core_norm for k in ["conclusion technique", "conclusion de prudence"])
    )
    if guarded_thyroid_contract_pass:
        validation_status = "pass"
    elif (
        diagnostic_safety_intent
        and not errors
        and warnings
        and set(str(w) for w in warnings).issubset(guarded_style_only_warnings)
    ):
        validation_status = "pass"
    elif generation_mode_norm == "deterministic_general_conversation" and not errors:
        validation_status = "pass"
    elif generation_mode_norm in deterministic_fact_modes and not errors:
        validation_status = "pass"
    elif errors:
        validation_status = "fail"
    elif warnings:
        validation_status = "warning"
    else:
        validation_status = "pass"

    def _dedup_keep_order(items: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in items:
            token = str(item or "").strip()
            if not token or token in seen:
                continue
            seen.add(token)
            out.append(token)
        return out

    warnings = _dedup_keep_order([str(w) for w in warnings])
    if pii_leak_detected:
        errors.append("pii_exposure")

    if unsupported_units:
        errors.append("unit_mismatch")

    if "unsupported_value" in set(str(e) for e in errors):
        errors.append("value_changed")

    source_mismatch_markers = {
        "source_alignment_mismatch",
        "source_alignment_mismatch_doc_level",
        "source_evidence_doc_mismatch",
        "source_url_docid_mismatch",
    }
    if set(str(e) for e in errors).intersection(source_mismatch_markers):
        errors.append("source_mismatch")

    raw_internal_source_markers = {
        "source_format_bad",
        "forbidden_internal_field",
        "internal_debug_leak",
        "raw_logs_visible",
        "render_internal_term_leak",
        "internal_chart_term_visible",
        "chunk_id_visible",
        "evidence_id_visible",
        "raw_internal_field_visible",
    }
    if set(str(e) for e in errors).intersection(raw_internal_source_markers):
        errors.append("raw_internal_source")

    diagnosis_markers = {
        "hallucinated_diagnosis",
        "diagnostic_claim_detected",
        "diagnostic_safety_violation",
        "diagnostic_suggestion_too_strong",
        "Definitive diagnosis detected.",
    }
    if set(str(e) for e in errors).intersection(diagnosis_markers):
        errors.append("diagnostic_affirmation")

    treatment_markers = {
        "Treatment recommendation detected.",
    }
    if set(str(e) for e in errors).intersection(treatment_markers):
        errors.append("treatment_recommendation")

    errors = _dedup_keep_order([str(e) for e in errors])

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
        "production_ux_checks": prod_checks,
    }


def validate_production_ux(
    answer_text: str,
    patients: list[dict[str, Any]] | None = None,
    user_requested_visualization: bool = False,
) -> list[dict[str, Any]]:
    """
    Reinforced production-ready UX checks.
    """
    try:
        from sort_utils import natural_report_sort_key
    except Exception:  # pragma: no cover
        from scripts.generation.sort_utils import natural_report_sort_key  # type: ignore
    checks = []
    an = _norm(answer_text)

    # A. Chart Leak
    for p in _CHART_LEAK_PATTERNS:
        if re.search(p, answer_text):
             checks.append({"id": "markdown_chart_leak", "status": "fail", "message": f"SVG/Recharts leak: {p}"})

    # B. Robotic Phrasing
    if any(re.search(p, answer_text) for p in _ROBOTIC_VIZ_PATTERNS):
        checks.append({"id": "robotic_visualization_intro", "status": "warning", "message": "Style 'Graphique demandé / Rendu affiché' robotique."})

    # C. Natural Sort & Range Labels
    if patients:
        for p in patients:
            reports = [
                str(r.get("filename") or r.get("label") or r.get("doc_id") or "").strip()
                for r in p.get("reports", [])
                if isinstance(r, dict)
            ]
            if reports != sorted(reports, key=natural_report_sort_key):
                 checks.append({"id": "natural_sort_sources", "status": "fail", "message": f"Sources mal triées pour {p['patient']}"})
            
            range_label = p.get("report_range_label", "")
            if "report.pdf" in reports and len(reports) > 1 and str(range_label).endswith("report.pdf"):
                 checks.append({"id": "invalid_report_range_label", "status": "fail", "message": f"Range label finit par report.pdf pour {p['patient']}"})
            for rep in p.get("reports", []):
                has_click = bool(str(rep.get("source_url") or "").strip() or str(rep.get("viewer_url") or "").strip())
                if not has_click:
                    checks.append({"id": "patient_inventory_clickable_sources", "status": "fail", "message": f"Source non cliquable pour {p.get('patient', 'patient')}"})
                    break

    # D. Long Cell
    if "| ---" in answer_text:
        cells = re.findall(r"\|([^|]+)\|", answer_text)
        if any(len(c.strip()) > 150 for c in cells):
             checks.append({"id": "patient_inventory_long_cell", "status": "warning", "message": "Cellule Markdown > 150 char."})

    # E. Duplicate Sources
    if patients and "**Sources consultées :**" in answer_text:
         checks.append({"id": "duplicate_sources_block", "status": "warning", "message": "Sources répétées alors que patients[] est présent."})

    # F. Generic / technical error leakage
    if any(k in an for k in ["impossible de generer", "verifiez le backend", "object of type", "stack trace"]):
        checks.append({"id": "no_generic_error_message", "status": "fail", "message": "Message d’erreur technique exposé côté utilisateur."})

    return checks
