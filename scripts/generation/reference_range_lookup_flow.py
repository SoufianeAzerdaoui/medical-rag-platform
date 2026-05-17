from __future__ import annotations

import os
from typing import Any

from query_understanding import analyte_display_name, normalize_analyte
from reference_range_parser import parse_reference_ranges
from reference_range_selector import select_reference_range


def _range_signature(r: dict[str, Any]) -> str:
    if str(r.get("operator") or "") == "range":
        return f"range:{r.get('low')}:{r.get('high')}:{str(r.get('unit') or '').lower()}"
    return f"th:{r.get('operator')}:{r.get('threshold')}:{str(r.get('unit') or '').lower()}"


def _range_text(r: dict[str, Any]) -> str:
    def _fmt_num(v: Any) -> str:
        if isinstance(v, float):
            txt = f"{v:.2f}".rstrip("0").rstrip(".")
            return txt.replace(".", ",")
        return str(v)
    if str(r.get("operator") or "") == "range":
        return f"{_fmt_num(r.get('low'))}–{_fmt_num(r.get('high'))}{(' ' + str(r.get('unit') or '').strip()) if str(r.get('unit') or '').strip() else ''}"
    return f"{r.get('operator') or ''}{_fmt_num(r.get('threshold') or '')}{(' ' + str(r.get('unit') or '').strip()) if str(r.get('unit') or '').strip() else ''}"


def _article_for_analyte(label: str) -> str:
    s = str(label or "").strip()
    if not s:
        return "le"
    first = s[0].lower()
    # Common acronym vowels in French pronunciation and standard vowels.
    if first in {"a", "e", "i", "o", "u", "y", "h"}:
        return "l’"
    return "le"


def _format_analyte_for_sentence(analyte: str) -> tuple[str, str]:
    raw = str(analyte_display_name(analyte, analyte) or analyte or "").strip()
    key = normalize_analyte(raw)
    pretty_map = {
        "phosphatase alcaline": ("la", "phosphatase alcaline"),
        "calcium": ("le", "calcium"),
        "pth intact": ("la", "PTH intacte"),
        "haptoglobine": ("l’", "haptoglobine"),
        "amh": ("l’", "AMH"),
    }
    if key in pretty_map:
        return pretty_map[key]
    lowered = raw.lower()
    if raw.isupper() and " " in raw:
        lowered = lowered.capitalize()
    return _article_for_analyte(lowered), lowered


def _looks_like_range_label(label: str) -> bool:
    txt = str(label or "").strip().lower()
    if not txt:
        return True
    numeric = bool(any(ch.isdigit() for ch in txt))
    has_range_sep = any(k in txt for k in ["-", "–", "<", ">"])
    has_unit = any(k in txt for k in ["pg/ml", "ng/ml", "mg/l", "pmol/l", "g/l", "ui/l", "iu/l"])
    return bool(numeric and (has_range_sep or has_unit))


def _clean_profile_label(selected: dict[str, Any]) -> str:
    label = str(selected.get("label") or "").strip()
    if label and not _looks_like_range_label(label):
        return label
    if str(selected.get("sex") or "").strip().lower() == "female":
        if selected.get("age_min") is not None and selected.get("age_max") is not None:
            return f"Femme — {int(selected['age_min']) if float(selected['age_min']).is_integer() else selected['age_min']}-{int(selected['age_max']) if float(selected['age_max']).is_integer() else selected['age_max']} ans"
        return "Femme"
    if str(selected.get("sex") or "").strip().lower() == "male":
        if str(selected.get("age_operator") or "").strip() and selected.get("age_value") is not None:
            return f"Homme {selected.get('age_operator')} {int(selected['age_value']) if float(selected['age_value']).is_integer() else selected['age_value']} ans"
        return "Homme"
    pop = str(selected.get("population") or "").strip()
    return pop if pop else "profil demandé"


def _profile_phrase(selected: dict[str, Any]) -> str:
    sex = str(selected.get("sex") or "").strip().lower()
    age_min = selected.get("age_min")
    age_max = selected.get("age_max")
    age_op = str(selected.get("age_operator") or "").strip()
    age_val = selected.get("age_value")
    if sex == "female":
        if age_min is not None and age_max is not None:
            mn = int(age_min) if isinstance(age_min, (int, float)) and float(age_min).is_integer() else age_min
            mx = int(age_max) if isinstance(age_max, (int, float)) and float(age_max).is_integer() else age_max
            return f"une femme de {mn}–{mx} ans"
        if age_op and age_val is not None:
            av = int(age_val) if isinstance(age_val, (int, float)) and float(age_val).is_integer() else age_val
            return f"une femme {age_op} {av} ans"
        return "une femme"
    if sex == "male":
        if age_min is not None and age_max is not None:
            mn = int(age_min) if isinstance(age_min, (int, float)) and float(age_min).is_integer() else age_min
            mx = int(age_max) if isinstance(age_max, (int, float)) and float(age_max).is_integer() else age_max
            return f"un homme de {mn}–{mx} ans"
        if age_op and age_val is not None:
            av = int(age_val) if isinstance(age_val, (int, float)) and float(age_val).is_integer() else age_val
            return f"un homme {age_op} {av} ans"
        return "un homme"
    pop = str(selected.get("population") or "").strip()
    cond = str(selected.get("condition") or "").strip()
    if cond == "cycled_female_j2_j4":
        return "une femme cyclée J2-J4"
    if pop:
        return pop
    return "le profil demandé"


def _stable_pick(templates: list[str], key: str) -> str:
    if not templates:
        return ""
    fingerprint = sum(ord(ch) for ch in str(key or ""))
    return templates[fingerprint % len(templates)]


def _profile_signature(r: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(r.get("sex") or "").strip().lower() or None,
        str(r.get("population") or "").strip().lower() or None,
        r.get("age_min"),
        r.get("age_max"),
        str(r.get("age_operator") or "").strip() or None,
        r.get("age_value"),
        str(r.get("age_unit") or "").strip().lower() or None,
    )


def _clean_source_pdf_name(raw: str) -> str:
    txt = str(raw or "").strip().replace("\\", "/")
    if not txt:
        return ""
    if txt.lower().startswith("docs/"):
        txt = txt[5:]
    return os.path.basename(txt)


def _source_item_from_selected(s: dict[str, Any]) -> dict[str, Any]:
    source_pdf = _clean_source_pdf_name(str(s.get("_source_pdf") or ""))
    doc_id = str(s.get("_doc_id") or "").strip()
    page = s.get("_page")
    line = s.get("_row")
    label_base = source_pdf or doc_id or "source"
    label = label_base + (f" — page {page}" if page is not None else "")
    if line is not None:
        label += f", ligne {line}"
    return {
        "label": label,
        "source_pdf": source_pdf or None,
        "doc_id": doc_id or None,
        "page": page,
        "line": line,
        "url": s.get("_source_url"),
        "viewer_url": s.get("_viewer_url"),
    }


def _age_band_text(r: dict[str, Any]) -> str:
    age_min = r.get("age_min")
    age_max = r.get("age_max")
    if age_min is not None and age_max is not None:
        mn = int(age_min) if isinstance(age_min, (int, float)) and float(age_min).is_integer() else age_min
        mx = int(age_max) if isinstance(age_max, (int, float)) and float(age_max).is_integer() else age_max
        return f"{mn}–{mx} ans"
    op = str(r.get("age_operator") or "").strip()
    val = r.get("age_value")
    if op and val is not None:
        vv = int(val) if isinstance(val, (int, float)) and float(val).is_integer() else val
        if op in {">", "<"}:
            return f"{op} {vv} ans"
        if op == ">=":
            return f"≥{vv} ans"
        if op == "<=":
            return f"≤{vv} ans"
        return f"{op} {vv} ans"
    return "âge non précisé"


def run_reference_range_lookup_from_rows(
    *,
    rows: list[dict[str, Any]],
    analyte: str,
    requested_profile: dict[str, Any] | None,
    use_patient_profile: bool = False,
    patient_profile: dict[str, Any] | None = None,
    request_all_ranges: bool = False,
    report_type: str | None = None,
    date_iso: str | None = None,
) -> dict[str, Any]:
    article, analyte_label = _format_analyte_for_sentence(analyte)
    per_doc_selected: list[dict[str, Any]] = []
    all_ranges: list[dict[str, Any]] = []
    parsed_ranges_preview: list[dict[str, Any]] = []
    debug_preview: list[dict[str, Any]] = []
    grouped_age_candidates: list[dict[str, Any]] = []
    grouped_sources: list[dict[str, Any]] = []
    grouped_profile_candidates: list[dict[str, Any]] = []
    for row in rows:
        rr = row.get("reference_ranges")
        if not isinstance(rr, list) or not rr:
            rr = parse_reference_ranges(str(row.get("reference_raw") or row.get("reference_range") or ""), default_unit=str(row.get("unit") or "").strip() or None)
        all_ranges.extend(rr)
        for r in rr[:6]:
            parsed_ranges_preview.append(
                {
                    "label": r.get("label"),
                    "sex": r.get("sex"),
                    "age_min": r.get("age_min"),
                    "age_max": r.get("age_max"),
                    "age_operator": r.get("age_operator"),
                    "age_value": r.get("age_value"),
                    "unit": r.get("unit"),
                    "low": r.get("low"),
                    "high": r.get("high"),
                    "operator": r.get("operator"),
                    "threshold": r.get("threshold"),
                }
            )
        debug_preview.append(
            {
                "doc_id": row.get("doc_id"),
                "analyte": row.get("analyte"),
                "section": row.get("section") or row.get("section_norm"),
                "reference_raw_preview": str(row.get("reference_raw") or row.get("reference_range") or "")[:140],
            }
        )
        if not rr:
            continue
        sel = select_reference_range(rr, requested_profile=requested_profile, patient_profile=patient_profile, use_patient_profile=use_patient_profile)
        if str(sel.get("status") or "") in {"selected", "fallback"} and isinstance(sel.get("selected"), dict):
            s = dict(sel["selected"])
            s["_doc_id"] = row.get("doc_id")
            s["_source_pdf"] = row.get("source_pdf")
            s["_page"] = row.get("page") if row.get("page") is not None else row.get("page_number")
            s["_row"] = row.get("row") if row.get("row") is not None else row.get("row_index")
            s["_viewer_url"] = row.get("viewer_url")
            s["_source_url"] = row.get("source_url")
            per_doc_selected.append(s)
        elif str(sel.get("status") or "") == "grouped_options":
            for cand in list(sel.get("candidates") or []):
                if not isinstance(cand, dict):
                    continue
                cc = dict(cand)
                cc["_doc_id"] = row.get("doc_id")
                cc["_source_pdf"] = row.get("source_pdf")
                cc["_page"] = row.get("page") if row.get("page") is not None else row.get("page_number")
                cc["_row"] = row.get("row") if row.get("row") is not None else row.get("row_index")
                cc["_viewer_url"] = row.get("viewer_url")
                cc["_source_url"] = row.get("source_url")
                grouped_age_candidates.append(cc)
            src = _source_item_from_selected(
                {
                    "_source_pdf": row.get("source_pdf"),
                    "_doc_id": row.get("doc_id"),
                    "_page": row.get("page") if row.get("page") is not None else row.get("page_number"),
                    "_row": row.get("row") if row.get("row") is not None else row.get("row_index"),
                    "_viewer_url": row.get("viewer_url"),
                    "_source_url": row.get("source_url"),
                }
            )
            if src.get("source_pdf") or src.get("doc_id"):
                if not any(x.get("label") == src.get("label") and x.get("doc_id") == src.get("doc_id") for x in grouped_sources):
                    grouped_sources.append(src)
        elif str(sel.get("status") or "") == "ambiguous":
            for cand in list(sel.get("candidates") or []):
                if not isinstance(cand, dict):
                    continue
                cc = dict(cand)
                cc["_doc_id"] = row.get("doc_id")
                cc["_source_pdf"] = row.get("source_pdf")
                cc["_page"] = row.get("page") if row.get("page") is not None else row.get("page_number")
                cc["_row"] = row.get("row") if row.get("row") is not None else row.get("row_index")
                cc["_viewer_url"] = row.get("viewer_url")
                cc["_source_url"] = row.get("source_url")
                grouped_profile_candidates.append(cc)
            src = _source_item_from_selected(
                {
                    "_source_pdf": row.get("source_pdf"),
                    "_doc_id": row.get("doc_id"),
                    "_page": row.get("page") if row.get("page") is not None else row.get("page_number"),
                    "_row": row.get("row") if row.get("row") is not None else row.get("row_index"),
                    "_viewer_url": row.get("viewer_url"),
                    "_source_url": row.get("source_url"),
                }
            )
            if src.get("source_pdf") or src.get("doc_id"):
                if not any(x.get("label") == src.get("label") and x.get("doc_id") == src.get("doc_id") for x in grouped_sources):
                    grouped_sources.append(src)
    if request_all_ranges:
        lines = ["| Profil | Plage | Unité |", "|---|---|---|"]
        for r in all_ranges[:25]:
            lines.append(f"| {str(r.get('label') or r.get('population') or r.get('sex') or 'profil non précisé')} | {_range_text(r)} | {str(r.get('unit') or '')} |")
        return {
            "status": "selected" if all_ranges else "no_match",
            "answer": "\n".join(lines) if all_ranges else f"Aucune plage physiologique exploitable n’a été retrouvée pour {analyte_label}.",
            "debug": {"candidate_rows_count": len(rows), "candidate_rows_preview": debug_preview[:8], "parsed_ranges_count": len(all_ranges), "parsed_ranges_preview": parsed_ranges_preview[:8]},
        }
    if not per_doc_selected:
        if all_ranges:
            # Multi-unit single-profile case (e.g. "15-65 pg/ml (1.6-6.9 pmol/l)"):
            # do not treat as ambiguous; answer directly with primary + secondary unit.
            profile_signatures = {_profile_signature(r) for r in all_ranges if isinstance(r, dict)}
            if len(profile_signatures) == 1 and len(rows) >= 1:
                primary = all_ranges[0]
                secondary = None
                for r in all_ranges[1:]:
                    if (
                        str(r.get("operator") or "") == str(primary.get("operator") or "")
                        and (r.get("low"), r.get("high")) != (primary.get("low"), primary.get("high"))
                        and str(r.get("unit") or "").strip().lower() != str(primary.get("unit") or "").strip().lower()
                    ):
                        secondary = r
                        break
                prefix = f"Pour {article}{analyte_label}" if article == "l’" else f"Pour {article} {analyte_label}"
                answer = f"{prefix}, la plage normale est {_range_text(primary)}"
                if secondary:
                    answer += f", soit {_range_text(secondary)}."
                else:
                    answer += "."
                first_row = rows[0] if rows else {}
                src = _source_item_from_selected(
                    {
                        "_source_pdf": first_row.get("source_pdf"),
                        "_doc_id": first_row.get("doc_id"),
                        "_page": first_row.get("page") if first_row.get("page") is not None else first_row.get("page_number"),
                        "_row": first_row.get("row") if first_row.get("row") is not None else first_row.get("row_index"),
                        "_viewer_url": first_row.get("viewer_url"),
                        "_source_url": first_row.get("source_url"),
                    }
                )
                return {
                    "status": "selected",
                    "answer": answer,
                    "sources": [src] if (src.get("source_pdf") or src.get("doc_id")) else [],
                    "debug": {"candidate_rows_count": len(rows), "candidate_rows_preview": debug_preview[:8], "parsed_ranges_count": len(all_ranges), "parsed_ranges_preview": parsed_ranges_preview[:8], "selected_range": primary},
                }
        if grouped_age_candidates:
            profile_phrase = "ce profil"
            if isinstance(requested_profile, dict):
                profile_phrase = _profile_phrase(requested_profile)
            dedup: list[dict[str, Any]] = []
            seen_keys: set[tuple[Any, ...]] = set()
            for c in grouped_age_candidates:
                key = (
                    c.get("age_min"),
                    c.get("age_max"),
                    c.get("age_operator"),
                    c.get("age_value"),
                    c.get("low"),
                    c.get("high"),
                    c.get("operator"),
                    c.get("threshold"),
                    str(c.get("unit") or "").strip().lower(),
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                dedup.append(c)
            dedup.sort(key=lambda x: ((x.get("age_min") if x.get("age_min") is not None else 10**9), (x.get("age_max") if x.get("age_max") is not None else 10**9)))
            lines = [
                f"Pour {article}{analyte_label}" if article == "l’" else f"Pour {article} {analyte_label}",
            ]
            lines[0] += f" chez {profile_phrase}, la norme dépend de la tranche d’âge."
            lines += [
                "Tranches disponibles :",
            ]
            for c in dedup[:8]:
                lines.append(f"- {_age_band_text(c)} : {_range_text(c)}")
            lines.append("Donnez votre tranche d’âge (ex: 30–34 ans).")
            return {
                "status": "grouped_options",
                "answer": "\n".join(lines),
                "sources": grouped_sources,
                "debug": {"candidate_rows_count": len(rows), "candidate_rows_preview": debug_preview[:8], "parsed_ranges_count": len(all_ranges), "parsed_ranges_preview": parsed_ranges_preview[:8], "failure_reason": "population_needs_age"},
            }
        if grouped_profile_candidates:
            dedup: list[dict[str, Any]] = []
            seen_keys: set[tuple[Any, ...]] = set()
            for c in grouped_profile_candidates:
                key = (
                    str(c.get("label") or "").strip().lower(),
                    c.get("age_min"),
                    c.get("age_max"),
                    c.get("age_operator"),
                    c.get("age_value"),
                    c.get("low"),
                    c.get("high"),
                    c.get("operator"),
                    c.get("threshold"),
                    str(c.get("unit") or "").strip().lower(),
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                dedup.append(c)
            lines = [
                f"Pour {analyte_label}, plusieurs sous-profils valides existent pour la demande.",
                "Sous-profils disponibles :",
            ]
            for c in dedup[:8]:
                age_txt = _age_band_text(c) if (c.get("age_min") is not None or c.get("age_operator") is not None) else "âge non précisé"
                label = _clean_profile_label(c)
                lines.append(f"- {label} — {age_txt} : {_range_text(c)}")
            return {
                "status": "ambiguous",
                "answer": "\n".join(lines),
                "sources": grouped_sources,
                "debug": {"candidate_rows_count": len(rows), "candidate_rows_preview": debug_preview[:8], "parsed_ranges_count": len(all_ranges), "parsed_ranges_preview": parsed_ranges_preview[:8], "failure_reason": "profile_subgroups_ambiguous"},
            }
        if all_ranges:
            options: list[str] = []
            for r in all_ranges[:4]:
                profile = _clean_profile_label(r)
                options.append(f"- {profile}: {_range_text(r)}")
            options_block = ("\nOptions :\n" + "\n".join(options)) if options else ""
            return {
                "status": "ambiguous",
                "answer": (
                    f"Plusieurs plages valides existent pour {analyte_label}. "
                    f"Précisez le profil (sexe/âge/population) ou demandez toutes les plages."
                    f"{options_block}"
                ),
                "debug": {"candidate_rows_count": len(rows), "candidate_rows_preview": debug_preview[:8], "parsed_ranges_count": len(all_ranges), "parsed_ranges_preview": parsed_ranges_preview[:8]},
            }
        return {
            "status": "no_match",
            "answer": f"Aucune plage physiologique exploitable n’a été retrouvée pour {analyte_label}.",
            "debug": {"candidate_rows_count": len(rows), "candidate_rows_preview": debug_preview[:8], "parsed_ranges_count": 0, "parsed_ranges_preview": []},
        }
    groups: dict[str, list[dict[str, Any]]] = {}
    for s in per_doc_selected:
        groups.setdefault(_range_signature(s), []).append(s)
    if len(groups) == 1:
        selected = per_doc_selected[0]
        profile_label = _profile_phrase(selected)
        selected_templates = [
            "{prefix}, la plage physiologique indiquée pour {profile} est {range_txt}.",
            "{prefix}, l’intervalle de référence pour {profile} est {range_txt}.",
            "{prefix}, la norme indiquée pour {profile} est {range_txt}.",
            "{prefix}, la plage de référence retenue pour {profile} est {range_txt}.",
            "{prefix}, la valeur de référence pour {profile} est {range_txt}.",
        ]
        prefix = f"Pour {article}{analyte_label}" if article == "l’" else f"Pour {article} {analyte_label}"
        tmpl = _stable_pick(
            selected_templates,
            key=f"{analyte_label}|{profile_label}|{selected.get('low')}|{selected.get('high')}|{report_type}|{date_iso}",
        )
        if article == "l’":
            answer = tmpl.format(prefix=prefix, profile=profile_label, range_txt=_range_text(selected))
        else:
            answer = tmpl.format(prefix=prefix, profile=profile_label, range_txt=_range_text(selected))
        source_items: list[dict[str, Any]] = []
        for s in per_doc_selected:
            item = _source_item_from_selected(s)
            if not (item.get("source_pdf") or item.get("doc_id")):
                continue
            if any(
                x.get("label") == item.get("label")
                and x.get("doc_id") == item.get("doc_id")
                and x.get("page") == item.get("page")
                for x in source_items
            ):
                continue
            source_items.append(item)
        if len({str(s.get('_doc_id') or '') for s in per_doc_selected}) > 1:
            answer += "\n\nCette plage a été retrouvée dans plusieurs rapports"
            if report_type:
                answer += f" de {report_type}"
            answer += ". Précisez une date si vous voulez une source documentaire unique."
        return {
            "status": "selected",
            "answer": answer,
            "sources": source_items,
            "debug": {"candidate_rows_count": len(rows), "candidate_rows_preview": debug_preview[:8], "parsed_ranges_count": len(all_ranges), "parsed_ranges_preview": parsed_ranges_preview[:8], "selected_range": selected},
        }
    options: list[str] = []
    for _, vals in list(groups.items())[:3]:
        v = vals[0]
        src = str(v.get("_source_pdf") or v.get("_doc_id") or "").strip()
        page = v.get("_page")
        location = f"{src} — page {page}" if src and page is not None else (src or str(v.get("_doc_id") or "source"))
        options.append(f"- {location}: {_range_text(v)}")
    prefix = "Plusieurs plages différentes ont été retrouvées"
    if report_type:
        prefix += f" en {report_type}"
    if date_iso:
        prefix += f" à la date {date_iso}"
    answer = prefix + ". Précisez la date/le document voulu.\n" + "\n".join(options)
    amb_sources: list[dict[str, Any]] = []
    for _, vals in list(groups.items())[:3]:
        if vals:
            it = _source_item_from_selected(vals[0])
            if it.get("source_pdf") or it.get("doc_id"):
                amb_sources.append(it)
    return {
        "status": "ambiguous",
        "answer": answer,
        "sources": amb_sources,
        "debug": {"candidate_rows_count": len(rows), "candidate_rows_preview": debug_preview[:8], "parsed_ranges_count": len(all_ranges), "parsed_ranges_preview": parsed_ranges_preview[:8], "failure_reason": "multiple_distinct_ranges"},
    }
