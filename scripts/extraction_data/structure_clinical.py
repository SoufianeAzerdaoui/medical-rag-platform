from __future__ import annotations

import re
from collections import OrderedDict

from utils import (
    is_known_result_unit,
    normalize_named_field,
    normalize_flag,
    normalize_inline_text,
    normalize_label,
    normalize_ocr_analyte_text,
    parse_float,
    parse_int,
    parse_iso_date,
    parse_reference_range,
    repair_numeric_with_reference,
    normalize_result_unit_text,
)


def _find_table(tables: list, role: str):
    return next((table for table in tables if table.table_role == role), None)


def _find_block(blocks: list, block_type: str):
    return next((block for block in blocks if block.block_type == block_type), None)


def _score_band(score: float) -> str:
    if score >= 0.85:
        return "high"
    if score >= 0.6:
        return "medium"
    return "low"


def _round_score(score: float) -> float:
    return round(max(0.0, min(1.0, score)), 3)


def _source_score(table=None, block=None) -> float:
    if table is not None and getattr(table, "bbox", None):
        return 0.97
    if table is not None:
        return 0.92
    if block is not None:
        return float(getattr(block, "confidence_score", 0.68))
    return 0.0


def _field_confidence(value, source_score: float, *, parser_applied: bool = False) -> dict[str, object]:
    if value is None or value == "":
        return {"label": "low", "score": 0.0}
    score = source_score - (0.03 if parser_applied else 0.0)
    score = _round_score(score)
    return {"label": _score_band(score), "score": score}


def _aggregate_field_confidence(field_confidence: dict[str, dict[str, object]]) -> tuple[str, float]:
    scores = [float(meta["score"]) for meta in field_confidence.values() if float(meta["score"]) > 0]
    if not scores:
        return ("low", 0.0)
    score = _round_score(sum(scores) / len(scores))
    return (_score_band(score), score)


def _table_to_key_values(table) -> dict[str, str]:
    if table is None or len(table.columns) < 2:
        return {}

    left_key = table.columns[0]
    right_key = table.columns[1]
    if normalize_label(left_key) == "field" and normalize_label(right_key) == "value":
        values: dict[str, str] = {}
        for record in table.records:
            key = normalize_label(str(record.get(left_key, "")))
            value = normalize_inline_text(str(record.get(right_key, "")))
            if key and value:
                values[key] = value
        return values

    values = {normalize_label(left_key): normalize_inline_text(right_key)}
    for record in table.records:
        key = normalize_label(str(record.get(left_key, "")))
        values[key] = normalize_inline_text(str(record.get(right_key, "")))
    return values


def _extract_labeled_fields(text: str, labels: list[str]) -> dict[str, str]:
    normalized = normalize_inline_text(text)
    positions: list[tuple[int, int, str]] = []
    for label in labels:
        match = re.search(re.escape(label), normalized, flags=re.IGNORECASE)
        if match:
            positions.append((match.start(), match.end(), label))
    positions.sort()
    values: dict[str, str] = {}
    for index, (_, end, label) in enumerate(positions):
        next_start = positions[index + 1][0] if index + 1 < len(positions) else len(normalized)
        values[normalize_label(label)] = normalize_inline_text(normalized[end:next_start])
    return values


def _extract_raw_fields_from_first_page(page_text_data: list[dict], labels: list[str]) -> dict[str, str]:
    if not page_text_data:
        return {}
    first_page = page_text_data[0]
    text = first_page.get("native_text") or first_page.get("final_text") or ""
    return _extract_labeled_fields(text, labels)


def _table_field_values(table) -> dict[str, str]:
    return _table_to_key_values(table) if table is not None else {}


def detect_document_type(page_text_data: list[dict]) -> str:
    if not page_text_data:
        return "medical_report"
    text = (page_text_data[0].get("final_text") or page_text_data[0].get("native_text", "")).lower()
    if "analyses biologiques" in text:
        return "biology_report"
    if "imagerie" in text:
        return "imaging_report"
    return "medical_report"


def extract_facility_info(page_text_data: list[dict]) -> dict:
    if not page_text_data:
        return {}
    first_page = page_text_data[0]
    for block in first_page.get("text_blocks", []):
        text = normalize_inline_text(block["text"])
        if "Tel." in text or "Tel " in text:
            match = re.match(r"^(.*?)\s*-\s*Tel\.?\s*(.*)$", text)
            if not match:
                return {"raw_text": text}
            return {
                "raw_text": text,
                "address": normalize_inline_text(match.group(1)),
                "phone": normalize_inline_text(match.group(2)),
            }
    return {}


def extract_patient_info(tables: list, blocks: list, page_text_data: list[dict]) -> dict:
    table = _find_table(tables, "patient_info_table")
    block = _find_block(blocks, "patient_info_block")
    values = _table_to_key_values(table)
    if not values:
        values = _extract_labeled_fields(
            block.text if block else "",
            ["Patient", "Identifiant", "Date de naissance", "Age", "Sexe", "Adresse"],
        )
    raw_values = (
        _table_field_values(table)
        if table is not None and getattr(table, "bbox", None) is None
        else _extract_raw_fields_from_first_page(
            page_text_data,
            ["Patient", "Identifiant", "Date de naissance", "Age", "Sexe", "Adresse"],
        )
    )
    patient_id_meta = normalize_named_field("patient_id", raw_values.get("identifiant") or values.get("identifiant"))
    birth_date_meta = normalize_named_field("birth_date", raw_values.get("date_de_naissance") or values.get("date_de_naissance"))
    sex_meta = normalize_named_field("sex", raw_values.get("sexe") or values.get("sexe"))
    source_score = _source_score(table, block)
    field_confidence = {
        "name": _field_confidence(values.get("patient"), source_score),
        "patient_id": _field_confidence(patient_id_meta["canonical"], source_score),
        "birth_date": _field_confidence(birth_date_meta["canonical"], source_score, parser_applied=True),
        "age": _field_confidence(parse_int(values.get("age")), source_score, parser_applied=True),
        "sex": _field_confidence(sex_meta["canonical"], source_score),
        "address": _field_confidence(values.get("adresse"), source_score),
    }
    confidence, confidence_score = _aggregate_field_confidence(field_confidence)
    return {
        "name": values.get("patient"),
        "patient_id_raw": patient_id_meta["raw"],
        "patient_id": patient_id_meta["canonical"],
        "patient_id_normalization_status": patient_id_meta["normalization_status"],
        "birth_date_raw": birth_date_meta["raw"],
        "birth_date": birth_date_meta["canonical"],
        "birth_date_normalization_status": birth_date_meta["normalization_status"],
        "age": parse_int(values.get("age")),
        "sex_raw": sex_meta["raw"],
        "sex": sex_meta["canonical"],
        "sex_normalization_status": sex_meta["normalization_status"],
        "address": values.get("adresse"),
        "confidence": confidence,
        "confidence_score": confidence_score,
        "field_confidence": field_confidence,
    }


def extract_report_metadata(tables: list, blocks: list, page_text_data: list[dict]) -> dict:
    table = _find_table(tables, "report_info_table")
    block = _find_block(blocks, "report_info_block")
    values = _table_to_key_values(table)
    if not values:
        values = _extract_labeled_fields(
            block.text if block else "",
            ["Numero de rapport", "Date du document", "Type de rencontre", "Prescripteur", "Specialite", "Statut"],
        )
    raw_values = (
        _table_field_values(table)
        if table is not None and getattr(table, "bbox", None) is None
        else _extract_raw_fields_from_first_page(
            page_text_data,
            [
                "Numero de rapport",
                "Date du document",
                "Identifiant",
                "Type de rencontre",
                "Prescripteur",
                "Specialite",
                "Statut",
            ],
        )
    )
    report_id_meta = normalize_named_field("report_id", raw_values.get("numero_de_rapport") or values.get("numero_de_rapport"))
    report_date_meta = normalize_named_field("report_date", raw_values.get("date_du_document") or values.get("date_du_document"))
    encounter_type_meta = normalize_named_field("encounter_type", raw_values.get("type_de_rencontre") or values.get("type_de_rencontre"))
    specialty_meta = normalize_named_field("specialty", raw_values.get("specialite") or values.get("specialite"))
    source_score = _source_score(table, block)
    field_confidence = {
        "report_id": _field_confidence(report_id_meta["canonical"], source_score),
        "report_date": _field_confidence(report_date_meta["canonical"], source_score, parser_applied=True),
        "encounter_type": _field_confidence(encounter_type_meta["canonical"], source_score),
        "prescriber": _field_confidence(values.get("prescripteur"), source_score),
        "specialty": _field_confidence(specialty_meta["canonical"], source_score),
        "status": _field_confidence(values.get("statut"), source_score),
    }
    confidence, confidence_score = _aggregate_field_confidence(field_confidence)
    return {
        "report_id_raw": report_id_meta["raw"],
        "report_id": report_id_meta["canonical"],
        "report_id_normalization_status": report_id_meta["normalization_status"],
        "report_date_raw": report_date_meta["raw"],
        "report_date": report_date_meta["canonical"],
        "report_date_normalization_status": report_date_meta["normalization_status"],
        "encounter_type_raw": encounter_type_meta["raw"],
        "encounter_type": encounter_type_meta["canonical"],
        "encounter_type_normalization_status": encounter_type_meta["normalization_status"],
        "prescriber": values.get("prescripteur"),
        "specialty_raw": specialty_meta["raw"],
        "specialty": specialty_meta["canonical"],
        "specialty_normalization_status": specialty_meta["normalization_status"],
        "status": values.get("statut"),
        "confidence": confidence,
        "confidence_score": confidence_score,
        "field_confidence": field_confidence,
    }


def _parse_result_line(line: str) -> dict | None:
    text = normalize_inline_text(line)
    pattern = re.compile(
        r"^(?P<analyte>.+?)\s+(?P<value>-?\d+(?:[.,]\d+)?)\s+"
        r"(?P<unit>[A-Za-z0-9*\/\[\]()%._-]+)\s+"
        r"(?P<low>-?\d+(?:[.,]\d+)?)\s*-\s*(?P<high>-?\d+(?:[.,]\d+)?)\s+"
        r"(?P<flag>H|L|-)?\s*"
        r"(?P<date>\d{1,2}\s+[A-Za-zéûîô\.]+\s+\d{4})?$",
        flags=re.IGNORECASE,
    )
    match = pattern.match(text)
    if not match:
        return None
    groups = match.groupdict()
    unit_meta = normalize_named_field("unit", groups["unit"])
    alert_meta = normalize_named_field("alert_flag", groups.get("flag"))
    observation_date_meta = normalize_named_field("observation_date", groups.get("date"))
    return {
        "analyte": normalize_inline_text(groups["analyte"]),
        "value_raw": groups["value"],
        "value_numeric": parse_float(groups["value"]),
        "unit_raw": unit_meta["raw"],
        "unit": unit_meta["canonical"],
        "unit_normalization_status": unit_meta["normalization_status"],
        "reference_range": {
            "text": f'{groups["low"]} - {groups["high"]}',
            "low": parse_float(groups["low"]),
            "high": parse_float(groups["high"]),
        },
        "alert_flag_raw": alert_meta["raw"],
        "alert_flag": alert_meta["canonical"],
        "alert_flag_normalization_status": alert_meta["normalization_status"],
        "is_abnormal": alert_meta["canonical"] in {"H", "L"},
        "observation_date_raw": observation_date_meta["raw"],
        "observation_date": observation_date_meta["canonical"],
        "observation_date_normalization_status": observation_date_meta["normalization_status"],
    }


def _normalize_result_unit(unit: str | None) -> str | None:
    return normalize_result_unit_text(unit)


def _normalize_result_analyte(analyte: str | None) -> str:
    return normalize_ocr_analyte_text(analyte)


def _result_fingerprint(analyte: str) -> str:
    tokens = [
        token
        for token in normalize_label(analyte).split("_")
        if token
        and token
        not in {
            "in",
            "of",
            "by",
            "or",
            "the",
            "and",
            "blood",
            "serum",
            "plasma",
            "automated",
            "count",
            "direct",
            "assay",
            "ae",
            "ys",
            "fang",
            "seni",
            "ra",
            "pinema",
            "ininatedonunt",
            "janv",
            "earns",
        }
    ]
    return "_".join(tokens[:6]) if tokens else normalize_label(analyte)


def _result_quality_score(result: dict) -> float:
    score = 0.0
    analyte = normalize_label(result.get("analyte", ""))
    unit = normalize_result_unit_text(result.get("unit")) or ""
    value = result.get("value_numeric")
    reference = result.get("reference_range", {}) or {}
    low = reference.get("low")
    high = reference.get("high")
    flag = result.get("alert_flag")

    suspicious_terms = {
        "ininatedonunt",
        "pinema",
        "sarin",
        "iid",
        "fang",
        "seni",
        "ainantiali",
        "earns",
    }
    if analyte:
        score += 2.0
        if not any(term in analyte for term in suspicious_terms):
            score += 2.0

    normalized_unit = normalize_label(unit)
    if is_known_result_unit(unit):
        score += 2.0
    elif unit:
        score -= 3.0

    if low is not None and high is not None and high >= low:
        score += 2.0
        if value is not None:
            in_range = low <= value <= high
            if in_range and flag is None:
                score += 2.0
            elif (value > high and flag == "H") or (value < low and flag == "L"):
                score += 2.0
            elif (value > high or value < low) and flag is None:
                score -= 2.0
            if flag is None and high > 0 and value > (high * 2.5):
                score -= 4.0
    elif reference.get("text") == "Selon contexte":
        score += 1.0

    return score


def _score_to_confidence(score: float) -> str:
    if score >= 8.0:
        return "high"
    if score >= 6.0:
        return "medium"
    return "low"


def _validate_result_consistency(result: dict) -> list[str]:
    issues: list[str] = []
    reference = result.get("reference_range", {}) or {}
    low = reference.get("low")
    high = reference.get("high")
    value = result.get("value_numeric")
    flag = result.get("alert_flag")

    if value is None:
        issues.append("missing_numeric_value")
        return issues
    if low is not None and high is not None and high < low:
        issues.append("invalid_reference_range")
        return issues
    if low is None or high is None:
        return issues

    if value < low and flag != "L":
        issues.append("value_below_range_without_low_flag")
    if value < low and flag == "H":
        issues.append("value_below_range_with_high_flag")
    if value > high and flag != "H":
        issues.append("value_above_range_without_high_flag")
    if value > high and flag == "L":
        issues.append("value_above_range_with_low_flag")
    if low <= value <= high and flag in {"H", "L"}:
        issues.append("flag_inconsistent_with_in_range_value")
    return issues


def _result_signature(result: dict) -> tuple[str, str, str]:
    return (
        normalize_label(result.get("analyte", "")),
        normalize_inline_text(result.get("value_raw", "") or ""),
        normalize_result_unit_text(result.get("unit")) or "",
    )


def _collect_raw_results_from_tables(tables: list) -> list[dict]:
    results: list[dict] = []
    for table in tables:
        if table.table_role != "results_table":
            continue
        for row_index, record in enumerate(table.records, start=1):
            analyte = _normalize_result_analyte(str(record.get("Analyse / Observation", "")))
            value_raw = normalize_inline_text(str(record.get("Resultat", "")))
            unit_raw = normalize_inline_text(str(record.get("Unites", "")))
            unit_meta = normalize_named_field("unit", unit_raw)
            reference = parse_reference_range(str(record.get("Valeurs de reference", "")))
            alert_raw = normalize_inline_text(str(record.get("Alert\ne", record.get("Alerte", ""))))
            alert_meta = normalize_named_field("alert_flag", alert_raw)
            observation_date_raw = normalize_inline_text(str(record.get("Date", "")))
            observation_date_meta = normalize_named_field("observation_date", observation_date_raw)
            value_numeric_raw = parse_float(value_raw)
            value_numeric, canonical_flag, is_abnormal = _normalize_result_measurements(
                value_raw=value_raw,
                value_numeric=value_numeric_raw,
                reference_range=reference,
                alert_flag=alert_meta["canonical"],
                is_ocr_source=getattr(table, "bbox", None) is None,
            )
            results.append(
                {
                    "page_number": table.page_number,
                    "source_page_number": table.page_number,
                    "source_table_id": table.table_id,
                    "source_kind": "results_table",
                    "row_index": row_index,
                    "analyte": analyte,
                    "value_raw": value_raw,
                    "value_numeric": value_numeric,
                    "unit_raw": unit_meta["raw"],
                    "unit": unit_meta["canonical"],
                    "unit_normalization_status": unit_meta["normalization_status"],
                    "reference_range": reference,
                    "alert_flag_raw": alert_meta["raw"],
                    "alert_flag": canonical_flag,
                    "alert_flag_normalization_status": (
                        "canonicalized"
                        if canonical_flag != alert_meta["canonical"]
                        else alert_meta["normalization_status"]
                    ),
                    "is_abnormal": is_abnormal,
                    "observation_date_raw": observation_date_meta["raw"],
                    "observation_date": observation_date_meta["canonical"],
                    "observation_date_normalization_status": observation_date_meta["normalization_status"],
                }
            )
    return results


def _collect_raw_results_from_blocks(blocks: list) -> list[dict]:
    results: list[dict] = []
    for block in blocks:
        if block.block_type != "results_table_block":
            continue
        for row_index, line in enumerate(block.text.splitlines(), start=1):
            parsed = _parse_result_line(line)
            if not parsed:
                continue
            parsed["analyte"] = _normalize_result_analyte(parsed.get("analyte", ""))
            parsed["unit"] = _normalize_result_unit(parsed.get("unit"))
            value_numeric, canonical_flag, is_abnormal = _normalize_result_measurements(
                value_raw=parsed.get("value_raw", ""),
                value_numeric=parsed.get("value_numeric"),
                reference_range=parsed.get("reference_range", {}) or {},
                alert_flag=parsed.get("alert_flag"),
                is_ocr_source=True,
            )
            parsed["value_numeric"] = value_numeric
            parsed["alert_flag"] = canonical_flag
            parsed["is_abnormal"] = is_abnormal
            parsed["page_number"] = block.page_number
            parsed["source_page_number"] = block.page_number
            parsed["source_table_id"] = None
            parsed["source_kind"] = "derived_text"
            parsed["row_index"] = row_index
            results.append(parsed)
    return results


def _result_dedup_key(result: dict) -> str:
    analyte_key = normalize_label(result.get("analyte", ""))
    date_key = normalize_inline_text(result.get("observation_date", "") or "")
    value_key = normalize_inline_text(result.get("value_raw", "") or "")
    unit_key = normalize_result_unit_text(result.get("unit")) or ""
    return "|".join([analyte_key, date_key, value_key, unit_key])


def _result_completeness_score(result: dict) -> int:
    fields = [
        result.get("analyte"),
        result.get("observation_date"),
        result.get("value_raw"),
        result.get("unit"),
        result.get("reference_range", {}).get("text"),
        result.get("alert_flag"),
    ]
    return sum(1 for value in fields if value not in (None, "", {}))


def _result_source_priority(result: dict) -> int:
    return 3 if result.get("source_kind") == "results_table" else 1


def _canonical_sort_tuple(result: dict) -> tuple[int, int, int, int, float]:
    return (
        _result_source_priority(result),
        _result_completeness_score(result),
        -int(result.get("source_page_number") or result.get("page_number") or 999),
        -int(result.get("row_index") or 999),
        _result_quality_score(result),
    )


def _build_duplicate_source(result: dict) -> dict:
    return {
        "source_table_id": result.get("source_table_id"),
        "source_page_number": result.get("source_page_number", result.get("page_number")),
        "source_kind": result.get("source_kind"),
        "row_index": result.get("row_index"),
    }


def _infer_flag_and_abnormal(value_numeric: float | None, reference_range: dict | None) -> tuple[str | None, bool]:
    reference = reference_range or {}
    low = reference.get("low")
    high = reference.get("high")
    if value_numeric is None or low is None or high is None:
        return (None, False)
    if value_numeric < low:
        return ("L", True)
    if value_numeric > high:
        return ("H", True)
    return (None, False)


def _normalize_result_measurements(*, value_raw: str, value_numeric: float | None, reference_range: dict, alert_flag: str | None, is_ocr_source: bool) -> tuple[float | None, str | None, bool]:
    normalized_value = value_numeric
    if is_ocr_source:
        _, repaired_numeric = repair_numeric_with_reference(value_raw, reference_range, flag=alert_flag)
        if repaired_numeric is not None:
            normalized_value = repaired_numeric

    inferred_flag, inferred_abnormal = _infer_flag_and_abnormal(normalized_value, reference_range)
    canonical_flag = alert_flag if alert_flag in {"H", "L"} else inferred_flag
    is_abnormal = canonical_flag in {"H", "L"} if canonical_flag is not None else inferred_abnormal
    return (normalized_value, canonical_flag, is_abnormal)


def _build_structured_vs_raw_validation(results: list[dict], raw_results: list[dict]) -> dict:
    raw_records: list[tuple[str, str, str]] = []
    for result in raw_results:
        raw_records.append(
            (
                normalize_label(str(result.get("analyte", ""))),
                normalize_inline_text(str(result.get("value_raw", ""))),
                normalize_result_unit_text(str(result.get("unit", ""))) or "",
            )
        )

    raw_set = set(raw_records)
    structured_signatures = [_result_signature(result) for result in results]
    missing_from_raw = [result["analyte"] for result in results if _result_signature(result) not in raw_set]

    return {
        "raw_result_rows": len(raw_records),
        "canonical_result_rows": len(results),
        "structured_result_rows": len(results),
        "duplicate_rows_removed": max(0, len(raw_records) - len(results)),
        "matched_structured_results": sum(1 for signature in structured_signatures if signature in raw_set),
        "missing_from_raw_tables": missing_from_raw,
    }


def build_validation_report(*, results: list[dict], raw_results: list[dict]) -> dict:
    table_summary = _build_structured_vs_raw_validation(results, raw_results)
    consistency_issues = [
        {
            "analyte": result.get("analyte"),
            "page_number": result.get("page_number"),
            "issues": issues,
        }
        for result in results
        if (issues := _validate_result_consistency(result))
    ]
    return {
        "raw_result_rows": table_summary["raw_result_rows"],
        "canonical_result_rows": table_summary["canonical_result_rows"],
        "duplicate_rows_removed": table_summary["duplicate_rows_removed"],
        "result_consistency_issues": consistency_issues,
        "structured_vs_raw_tables": table_summary,
    }


def _is_result_reliable(result: dict, score: float) -> bool:
    analyte_text = normalize_inline_text(result.get("analyte", ""))
    analyte_label = normalize_label(analyte_text)
    suspicious_terms = {
        "ininatedonunt",
        "pinema",
        "sarin",
        "iid",
        "fang",
        "seni",
        "ainantiali",
        "earns",
    }

    if not analyte_label:
        return False
    if analyte_label in {"pinema", "plasma"}:
        return False
    if analyte_label.startswith("automated_count"):
        return False
    if analyte_text.count("[") > 1:
        return False
    if len(analyte_label.split("_")) < 2 and analyte_label not in {"body_height", "body_weight"}:
        return False
    if not is_known_result_unit(result.get("unit")) and result.get("reference_range", {}).get("text") != "Selon contexte":
        return False
    if any(term in analyte_label for term in suspicious_terms) and score < 8.0:
        return False
    if score < 4.5:
        return False
    return True


def _deduplicate_results(results: list[dict]) -> tuple[list[dict], dict]:
    best_by_key: OrderedDict[str, dict] = OrderedDict()
    duplicate_sources_by_key: dict[str, list[dict]] = {}
    for result in results:
        normalized_unit = _normalize_result_unit(result.get("unit"))
        normalized_analyte = _normalize_result_analyte(result.get("analyte", ""))
        result["unit"] = normalized_unit
        result["analyte"] = normalized_analyte
        key = _result_dedup_key(result)
        result["dedup_key"] = key
        current = best_by_key.get(key)
        if current is None:
            best_by_key[key] = dict(result)
            duplicate_sources_by_key[key] = []
            continue
        if _canonical_sort_tuple(result) > _canonical_sort_tuple(current):
            duplicate_sources_by_key[key].append(_build_duplicate_source(current))
            best_by_key[key] = dict(result)
        else:
            duplicate_sources_by_key[key].append(_build_duplicate_source(result))

    deduped: list[dict] = []
    for result in best_by_key.values():
        score = _result_quality_score(result)
        if not _is_result_reliable(result, score):
            continue
        normalized_score = _round_score(score / 10.0)
        dedup_key = result.get("dedup_key") or _result_dedup_key(result)
        result["confidence_score"] = normalized_score
        result["confidence"] = _score_to_confidence(score)
        result["dedup_key"] = dedup_key
        result["source_page_number"] = result.get("source_page_number", result.get("page_number"))
        result["is_canonical"] = True
        result["duplicate_sources"] = duplicate_sources_by_key.get(dedup_key, [])
        result["field_confidence"] = {
            "analyte": _field_confidence(result.get("analyte"), normalized_score),
            "value_raw": _field_confidence(result.get("value_raw"), normalized_score),
            "value_numeric": _field_confidence(result.get("value_numeric"), normalized_score, parser_applied=True),
            "unit": _field_confidence(result.get("unit"), normalized_score),
            "reference_range": _field_confidence(result.get("reference_range", {}).get("text"), normalized_score),
            "alert_flag": _field_confidence(result.get("alert_flag") if result.get("alert_flag") else "-", normalized_score - 0.05),
            "observation_date": _field_confidence(result.get("observation_date"), normalized_score, parser_applied=True),
        }
        deduped.append(result)
    dedup_stats = {
        "raw_result_rows": len(results),
        "canonical_result_rows": len(deduped),
        "duplicate_rows_removed": max(0, len(results) - len(deduped)),
    }
    return deduped, dedup_stats


def extract_results(tables: list, blocks: list) -> tuple[list[dict], list[dict], dict]:
    del blocks
    raw_results = _collect_raw_results_from_tables(tables)
    canonical_results, dedup_stats = _deduplicate_results(raw_results)
    return canonical_results, raw_results, dedup_stats


def extract_results_from_blocks(blocks: list) -> tuple[list[dict], list[dict], dict]:
    raw_results = _collect_raw_results_from_blocks(blocks)
    canonical_results, dedup_stats = _deduplicate_results(raw_results)
    return canonical_results, raw_results, dedup_stats


def extract_interpretation(blocks: list) -> dict:
    interpretation_blocks = [block for block in blocks if block.block_type == "clinical_interpretation_block"]
    summary_blocks = [block for block in blocks if block.block_type == "summary_block"]
    section_scores = [float(getattr(block, "confidence_score", 0.0)) for block in interpretation_blocks if getattr(block, "text", "")]
    interpretation_score = _round_score(sum(section_scores) / len(section_scores)) if section_scores else 0.0
    interpretation_confidence = _score_band(interpretation_score)
    return {
        "text": "\n\n".join(block.text for block in interpretation_blocks if block.text),
        "sections": [
            {
                "page_number": block.page_number,
                "title": block.section_title,
                "text": block.text,
                "confidence": getattr(block, "confidence", "medium"),
                "confidence_score": float(getattr(block, "confidence_score", 0.0)),
            }
            for block in interpretation_blocks
        ],
        "summary_text": "\n\n".join(block.text for block in summary_blocks if block.text),
        "confidence": interpretation_confidence,
        "confidence_score": interpretation_score,
    }


def extract_validation(page_text_data: list[dict], images: list, ocr_visuals: list, report: dict, blocks: list) -> dict:
    signature_ids = [image.image_id for image in images if image.image_type == "signature"]
    stamp_ids = [image.image_id for image in images if image.image_type == "stamp_or_seal"]
    signature_ids.extend(visual.visual_id for visual in ocr_visuals if visual.visual_type == "signature")
    stamp_ids.extend(visual.visual_id for visual in ocr_visuals if visual.visual_type == "stamp_or_seal")

    validation_block = _find_block(blocks, "validation_block")
    structured_fields = getattr(validation_block, "structured_fields", {}) if validation_block else {}

    clinician = structured_fields.get("validated_by")
    specialty = structured_fields.get("specialty")
    if not clinician or not specialty:
        for page in page_text_data:
            text_blocks = page.get("text_blocks", [])
            page_width = page.get("width", 0)
            for index, block in enumerate(text_blocks):
                if "validation_medicale" not in normalize_label(block["text"]):
                    continue
                candidate_blocks = [
                    candidate
                    for candidate in text_blocks[index + 1 :]
                    if candidate["bbox"]["x0"] < page_width * 0.45
                    and candidate["bbox"]["y0"] >= block["bbox"]["y0"]
                    and candidate["bbox"]["y0"] <= block["bbox"]["y0"] + 80
                ]
                if candidate_blocks and not clinician:
                    clinician = normalize_inline_text(candidate_blocks[0]["text"])
                if len(candidate_blocks) >= 2 and not specialty:
                    specialty = normalize_inline_text(candidate_blocks[1]["text"])
                break
            if clinician or specialty:
                break

    report_specialty = report.get("specialty")
    if specialty and report_specialty and normalize_label(report_specialty) in normalize_label(specialty):
        specialty = report_specialty

    field_confidence = {
        "validation_title": _field_confidence(structured_fields.get("validation_title"), float(getattr(validation_block, "confidence_score", 0.65)) if validation_block else 0.0),
        "validated_by": _field_confidence(clinician or report.get("prescriber"), _source_score(block=validation_block) if validation_block else 0.65),
        "specialty": _field_confidence(specialty or report_specialty, _source_score(block=validation_block) if validation_block else 0.65),
        "signature_present": _field_confidence(bool(signature_ids), 0.92 if signature_ids else 0.0),
        "stamp_present": _field_confidence(bool(stamp_ids), 0.92 if stamp_ids else 0.0),
    }
    confidence, confidence_score = _aggregate_field_confidence(field_confidence)
    return {
        "validation_title": structured_fields.get("validation_title", "Validation"),
        "validated_by": clinician or report.get("prescriber"),
        "specialty": specialty or report_specialty,
        "signature_image_ids": signature_ids,
        "stamp_image_ids": stamp_ids,
        "status": report.get("status"),
        "is_signed": bool(signature_ids),
        "is_stamped": bool(stamp_ids),
        "confidence": confidence,
        "confidence_score": confidence_score,
        "field_confidence": field_confidence,
    }


def build_structured_document(
    *,
    page_text_data: list[dict],
    tables: list,
    images: list,
    ocr_visuals: list,
    blocks: list,
) -> dict:
    document_type = detect_document_type(page_text_data)
    facility = extract_facility_info(page_text_data)
    patient = extract_patient_info(tables, blocks, page_text_data)
    report = extract_report_metadata(tables, blocks, page_text_data)
    results, raw_results, dedup_stats = extract_results(tables, blocks)
    if not results:
        results, raw_results, dedup_stats = extract_results_from_blocks(blocks)
    interpretation = extract_interpretation(blocks)
    validation = extract_validation(page_text_data, images, ocr_visuals, report, blocks)
    validation_report = build_validation_report(results=results, raw_results=raw_results)
    validation_report["deduplication"] = dedup_stats
    return {
        "document_type": document_type,
        "facility": facility,
        "patient": patient,
        "report": report,
        "results": results,
        "interpretation": interpretation,
        "validation": validation,
        "validation_report": validation_report,
    }
