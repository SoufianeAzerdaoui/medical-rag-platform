from __future__ import annotations

import re
from typing import Any, Callable


def thyroid_high_groups_from_rules(rules: dict[str, Any] | None) -> tuple[set[str], set[str]]:
    cfg = dict(rules or {})
    groups = dict(cfg.get("high_groups") or {})
    tsh_group = {
        str(a).strip().lower()
        for a in list(groups.get("tsh") or ["tsh", "tshus"])
        if str(a).strip()
    }
    t3_t4_group = {
        str(a).strip().lower()
        for a in list(groups.get("t3_t4") or ["t3_libre", "t3libre", "t4_libre", "t4libre"])
        if str(a).strip()
    }
    return tsh_group, t3_t4_group


def build_thyroid_diagnostic_safety_answer(
    thyroid_rows: list[dict[str, Any]],
    *,
    detail_fallback: str,
    discordance_sentence: str,
    no_diagnostic_sentence: str,
    correlation_sentence: str,
    summary_template: str,
    tsh_group: set[str],
    t3_t4_group: set[str],
    normalize_status_code: Callable[[str | None, str | None], str],
) -> str:
    if not thyroid_rows:
        return ""
    details: list[str] = []
    has_tsh_high = False
    has_t3_t4_high = False
    for ev in thyroid_rows:
        status = normalize_status_code(
            str(ev.get("technical_status") or ev.get("status") or ""),
            str(ev.get("technical_status_code") or ev.get("interpretation_status") or ""),
        )
        analyte = str(ev.get("analyte") or "").strip()
        analyte_norm = str(ev.get("analyte_norm") or "").strip().lower()
        if status == "above_reference":
            details.append(f"{analyte} elevee")
            if analyte_norm in tsh_group:
                has_tsh_high = True
            if analyte_norm in t3_t4_group:
                has_t3_t4_high = True
        elif status == "below_reference":
            details.append(f"{analyte} basse")
    details_txt = ", ".join(details) if details else detail_fallback
    discordance = discordance_sentence if (has_tsh_high and has_t3_t4_high) else ""
    return summary_template.format(
        details_txt=details_txt,
        no_diagnostic_sentence=no_diagnostic_sentence,
        discordance=discordance,
        correlation_sentence=correlation_sentence,
    ).strip()


def ensure_guarded_thyroid_conclusion(
    answer: str,
    *,
    strong_patterns: list[str],
    limitation_sentence: str,
    discordance_replacement: str,
    clinical_style_patterns: list[str],
    norm_text: Callable[[str], str],
) -> str:
    text = str(answer or "").strip()
    if not text:
        return text
    text = re.sub(r"(?im)^conclusion de prudence\s*:?", "Conclusion technique :", text)
    for patt in list(strong_patterns or []):
        text = re.sub(
            str(patt),
            str(discordance_replacement),
            text,
            flags=re.IGNORECASE,
        )
    for patt in list(clinical_style_patterns or []):
        text = re.sub(str(patt), str(limitation_sentence), text)
    lim_esc = re.escape(str(limitation_sentence))
    text = re.sub(rf"(?im)^({lim_esc}\s*){{2,}}$", str(limitation_sentence), text)

    dedup_lines: list[str] = []
    seen_limitation = False
    seen_discordance = False
    discordance_key = "discordant pour une hyperthyroidie primaire"
    for ln in [x.strip() for x in text.splitlines() if x.strip()]:
        if norm_text(ln) == norm_text(str(limitation_sentence)):
            if seen_limitation:
                continue
            seen_limitation = True
        if (norm_text(str(discordance_replacement)) in norm_text(ln)) or (discordance_key in norm_text(ln)):
            if seen_discordance:
                continue
            seen_discordance = True
        dedup_lines.append(ln)

    dedup_lines = [
        ln
        for ln in dedup_lines
        if (
            norm_text(str(discordance_replacement)) not in norm_text(ln)
            and discordance_key not in norm_text(ln)
        )
        or norm_text(ln).startswith("conclusion technique")
    ]
    text = "\n".join(dedup_lines).strip()
    required = (
        f"Conclusion technique : {discordance_replacement} ; "
        "aucune conclusion diagnostique ne peut être posée à partir de ces seuls éléments."
    )
    if "conclusion technique" in norm_text(text):
        text = re.sub(r"(?im)^conclusion technique\s*:.*$", "", text).strip()
        return f"{text}\n{required}".strip()
    return f"{text}\n{required}".strip()


def enforce_guarded_thyroid_display_labels(
    answer: str,
    evidences: list[dict[str, Any]],
    *,
    clean_analyte_label: Callable[[str | None], str],
) -> str:
    text = str(answer or "")
    if not text:
        return text
    rows = list(evidences or [])
    tshus_label = ""
    for ev in rows:
        an_norm = str(ev.get("analyte_norm") or "").strip().lower()
        if an_norm != "tshus":
            continue
        preferred = clean_analyte_label(
            str(
                ev.get("analyte")
                or ev.get("analyte_label")
                or ev.get("display_name")
                or ev.get("source_analyte")
                or ""
            )
        )
        tshus_label = preferred or "TSHus"
        break
    if not tshus_label:
        return text
    text = re.sub(
        r"(?im)^(\s*[-•]\s*)tsh(\s*:)",
        rf"\1{tshus_label}\2",
        text,
    )
    text = re.sub(r"(?i)\btsh\b", tshus_label, text)
    return text


def maybe_rebuild_guarded_thyroid_answer(
    *,
    question: str,
    answer: str,
    evidences: list[dict[str, Any]],
    is_thyroid_topic: Callable[[str], bool],
    thyroid_analyte_norms: set[str],
    build_thyroid_answer: Callable[[list[dict[str, Any]]], str],
    discordance_replacement: str,
    enforce_display_labels: Callable[[str, list[dict[str, Any]]], str],
    norm_text: Callable[[str], str],
) -> str:
    qn = norm_text(question or "")
    if not is_thyroid_topic(qn):
        return str(answer or "").strip()
    thyroid_norms = {str(a).strip().lower() for a in set(thyroid_analyte_norms or set()) if str(a).strip()}
    thyroid_rows = [
        dict(ev)
        for ev in list(evidences or [])
        if str(ev.get("analyte_norm") or "").strip().lower() in thyroid_norms
    ]
    if not thyroid_rows:
        return str(answer or "").strip()

    rebuilt = str(build_thyroid_answer(thyroid_rows) or "").strip()
    if not rebuilt:
        return str(answer or "").strip()

    required_conclusion = (
        f"Conclusion technique : {discordance_replacement} ; "
        "aucune conclusion diagnostique ne peut être posée à partir de ces seuls éléments."
    )
    if any(k in qn for k in ["diagnostic", "traitement", "recommandes", "recommander"]):
        refusal = "Je ne peux pas poser ni évoquer un diagnostic à partir de ces résultats seuls."
        if not rebuilt.lower().startswith(refusal.lower()):
            rebuilt = f"{refusal}\n\n{rebuilt}".strip()

    discordance_key = norm_text(discordance_replacement)
    discordance_variant_key = "discordant pour une hyperthyroidie primaire"
    if discordance_key or discordance_variant_key:
        kept_parts: list[str] = []
        for part in re.split(r"(?<=[\.\!\?])\s+|\n+", rebuilt):
            p = str(part or "").strip()
            if not p:
                continue
            pn = norm_text(p)
            if (discordance_key and discordance_key in pn) or (discordance_variant_key in pn):
                continue
            kept_parts.append(p)
        rebuilt = " ".join(kept_parts).strip()

    if norm_text(required_conclusion) not in norm_text(rebuilt):
        rebuilt = f"{rebuilt.rstrip()}\n\n{required_conclusion}".strip()
    rebuilt = enforce_display_labels(rebuilt, thyroid_rows)
    return rebuilt.strip()


def ensure_diagnostic_refusal_prefix(
    *,
    question: str,
    safety_intent: str,
    answer: str,
    norm_text: Callable[[str], str],
) -> str:
    safety_norm = str(safety_intent or "").strip().lower()
    if safety_norm != "diagnostic_safety_question":
        return str(answer or "").strip()
    qn = norm_text(question or "")
    if not any(k in qn for k in ["diagnostic", "traitement", "recommandes", "recommander"]):
        return str(answer or "").strip()
    refusal = "Je ne peux pas poser ni évoquer un diagnostic à partir de ces résultats seuls."
    text = str(answer or "").strip()
    if norm_text(text).startswith(norm_text(refusal)):
        return text
    return f"{refusal}\n\n{text}".strip() if text else refusal
