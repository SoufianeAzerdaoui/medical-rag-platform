from __future__ import annotations

import re
from collections import OrderedDict
from datetime import date, datetime

from utils import (
    compute_confidence,
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
    if _is_parasitology_stool_report(page_text_data):
        return "parasitology_stool_report"
    text = (page_text_data[0].get("final_text") or page_text_data[0].get("native_text", "")).lower()
    if "laboratoire central" in text or "n° d'échantillon" in text or "n d'échantillon" in text:
        return "biology_report"
    if "analyses biologiques" in text:
        return "biology_report"
    if "imagerie" in text:
        return "imaging_report"
    return "medical_report"


def _page_lines(page: dict) -> list[str]:
    text = page.get("final_text") or page.get("native_text") or ""
    return [normalize_inline_text(line) for line in text.splitlines() if normalize_inline_text(line)]


def _all_lines(page_text_data: list[dict]) -> list[str]:
    lines: list[str] = []
    for page in page_text_data:
        lines.extend(_page_lines(page))
    return lines


def _is_chu_lab_report(page_text_data: list[dict]) -> bool:
    first_text = normalize_label("\n".join(_page_lines(page_text_data[0]))) if page_text_data else ""
    return "laboratoire_central" in first_text and (
        "ip_patient" in first_text or "n_d_echantillon" in first_text or "n_echantillon" in first_text
    )


PARASITOLOGY_MARKERS = (
    "laboratoire_de_parasitologie_mycologie",
    "examen_parasitologique_des_selles",
    "examen_macroscopique",
    "examen_microscopique",
    "resultat_final",
)


def _is_parasitology_stool_report(page_text_data: list[dict]) -> bool:
    label = normalize_label("\n".join(_all_lines(page_text_data)))
    return all(marker in label for marker in PARASITOLOGY_MARKERS)


def _line_value(lines: list[str], label_pattern: str, *, allow_next_line: bool = True) -> str | None:
    pattern = re.compile(label_pattern, flags=re.IGNORECASE)
    for index, line in enumerate(lines):
        match = pattern.search(line)
        if not match:
            continue
        value = normalize_inline_text(line[match.end() :].lstrip(" :;-"))
        if value:
            return value
        if allow_next_line and index + 1 < len(lines):
            candidate = normalize_inline_text(lines[index + 1])
            if candidate and not re.search(r":\s*$", candidate):
                return candidate
    return None


def _line_index(lines: list[str], label_pattern: str) -> int | None:
    pattern = re.compile(label_pattern, flags=re.IGNORECASE)
    for index, line in enumerate(lines):
        if pattern.search(line):
            return index
    return None


def _parse_chu_datetime(value: str | None) -> str | None:
    if not value:
        return None
    text = normalize_inline_text(value)
    match = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})\s+(\d{1,2}):(\d{2})(?::(\d{2}))?", text)
    if not match:
        parsed_date = parse_iso_date(text)
        return parsed_date
    day, month, year, hour, minute, second = match.groups()
    try:
        parsed = datetime(
            int(year),
            int(month),
            int(day),
            int(hour),
            int(minute),
            int(second or 0),
        )
    except ValueError:
        return None
    return parsed.isoformat()


def _compute_age_at_report_date(birth_date: str | None, report_date: str | None) -> int | None:
    if not birth_date or not report_date:
        return None
    try:
        born = date.fromisoformat(birth_date[:10])
        reported = date.fromisoformat(report_date[:10])
    except ValueError:
        return None
    age = reported.year - born.year - ((reported.month, reported.day) < (born.month, born.day))
    return age if age >= 0 else None


def _age_consistency(reported_age: int | None, computed_age: int | None) -> str:
    if reported_age is None or computed_age is None:
        return "unknown"
    return "consistent" if abs(reported_age - computed_age) <= 1 else "inconsistent_with_birth_date"


def _parse_chu_age(birth_context: str | None) -> int | None:
    if not birth_context:
        return None
    match = re.search(r"-\s*(\d+(?:[,.]\d+)?)\s*ans", birth_context, flags=re.IGNORECASE)
    if not match:
        return None
    return parse_int(match.group(1))


def _extract_chu_header(page_text_data: list[dict]) -> dict[str, str | int | None]:
    lines = _page_lines(page_text_data[0]) if page_text_data else []
    all_lines = _all_lines(page_text_data)
    normalized_all = normalize_label("\n".join(all_lines))
    birth_line = _line_value(lines, r"N[ée]\(e\)\s*le\s*:?", allow_next_line=True)
    lab_line = next(
        (
            line
            for line in lines
            if normalize_label(line).startswith("laboratoire_de_")
            or "laboratoire de " in normalize_inline_text(line).lower()
            or normalize_label(line).startswith("hematologie")
        ),
        None,
    )
    validation_line = _line_value(all_lines, r"Valid[ée]\(e\)\s*par\s*:?", allow_next_line=False)
    print_date = None
    edited_date = None
    for index, line in enumerate(all_lines):
        if not re.match(r"^Le\s*:", line, flags=re.IGNORECASE):
            continue
        next_label = normalize_label(all_lines[index + 1]) if index + 1 < len(all_lines) else ""
        if next_label.startswith("imprime_par") and print_date is None:
            print_date = _line_value([line], r"^Le\s*:?", allow_next_line=False)
        elif next_label.startswith(("edite_e_par", "valide_e_par")):
            edited_date = _line_value([line], r"^Le\s*:?", allow_next_line=False)
        elif print_date is None:
            print_date = _line_value([line], r"^Le\s*:?", allow_next_line=False)
    validation_date = None
    for index, line in enumerate(all_lines):
        if not re.search(r"Valid[ée]\(e\)\s*par", line, flags=re.IGNORECASE):
            continue
        if index > 0 and re.match(r"^Le\s*:", all_lines[index - 1], flags=re.IGNORECASE):
            validation_date = _line_value([all_lines[index - 1]], r"^Le\s*:?", allow_next_line=False)
        break

    sample_label = _line_value(all_lines, r"^NATURE\s+DE\s+PR[ÉE]L[ÈE]VEMENT\s*:?", allow_next_line=True)
    if sample_label:
        sample_label = normalize_inline_text(sample_label).lstrip(":").strip()

    return {
        "exam_name": "EXAMEN PARASITOLOGIQUE DES SELLES" if "examen_parasitologique_des_selles" in normalized_all else None,
        "patient": _line_value(lines, r"^Patient\s*:?", allow_next_line=False),
        "patient_id": _line_value(lines, r"^IP\s*Patient\s*:?", allow_next_line=False),
        "birth_date": birth_line,
        "age": _parse_chu_age(birth_line),
        "sex": _line_value(lines, r"^Sexe\s*:?", allow_next_line=False),
        "origin": _line_value(lines, r"^Origine\s*:?", allow_next_line=False),
        "service": _line_value(lines, r"^Service\s*:?", allow_next_line=False),
        "prescriber": _line_value(lines, r"^Prescripteur\s*:?", allow_next_line=True),
        "request_date": _line_value(lines, r"^Date\s+Demande\s*:?", allow_next_line=False),
        "received_date": _line_value(lines, r"^Date\s+R[ée]ception\s*:?", allow_next_line=False),
        "sample_id": _line_value(lines, r"^N[°º]?\s*d['’]?[ée]chantillon\s*:?", allow_next_line=False),
        "specimen": _line_value(lines, r"^Nature\s*:?", allow_next_line=False),
        "sample_label": sample_label,
        "specialty": lab_line,
        "validated_by": validation_line,
        "validation_date": validation_date,
        "print_date": print_date,
        "printed_by": _line_value(all_lines, r"^Imprim[ée]\s+par\s*:?", allow_next_line=False),
        "edited_date": edited_date,
        "edited_by": _line_value(all_lines, r"^Edit[ée]\(e\)\s+par\s*:?", allow_next_line=False),
        "status": "validated" if validation_line else None,
    }


def _confidence_map(source_score: float, values: dict[str, object], field_names: list[str]) -> dict[str, dict[str, object]]:
    return {field: _field_confidence(values.get(field), source_score, parser_applied=field.endswith("_date")) for field in field_names}


def extract_facility_info(page_text_data: list[dict]) -> dict:
    if not page_text_data:
        return {}
    if _is_chu_lab_report(page_text_data):
        lines = _all_lines(page_text_data)
        text = "\n".join(lines)
        phone = None
        fax = None
        phone_match = re.search(r"T[ée]l\s*:\s*([0-9 ]+)", text, flags=re.IGNORECASE)
        fax_match = re.search(r"Fax\s*:\s*([0-9 /]+)", text, flags=re.IGNORECASE)
        if phone_match:
            phone = normalize_inline_text(phone_match.group(1)).replace(" ", "")
        if fax_match:
            fax = normalize_inline_text(fax_match.group(1))
        laboratory = next(
            (
                line
                for line in lines
                if normalize_label(line).startswith("laboratoire_de_")
                or normalize_label(line).startswith("hematologie")
                or "laboratoire de " in line.lower()
            ),
            None,
        )
        return {
            "country": "Maroc",
            "ministry": "Ministère de la santé et de la protection sociale"
            if "Ministère de la santé et de la protection sociale" in text
            else None,
            "organization": "Centre Hospitalo-Universitaire Mohammed VI - Oujda"
            if "Centre Hospitalo-Universitaire Mohammed VI - Oujda" in text
            else None,
            "department": "Laboratoire Central" if "LABORATOIRE CENTRAL" in text else None,
            "laboratory": laboratory,
            "website": "www.chuoujda.ma" if "www.chuoujda.ma" in text else None,
            "phone": phone,
            "fax": fax,
            "confidence": "high",
            "confidence_score": 0.9,
        }
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
    if _is_chu_lab_report(page_text_data):
        header = _extract_chu_header(page_text_data)
        source_score = 0.88

        patient_id_meta = normalize_named_field("patient_id", str(header.get("patient_id") or ""))
        birth_date_meta = normalize_named_field("birth_date", str(header.get("birth_date") or ""))
        sex_meta = normalize_named_field("sex", str(header.get("sex") or ""))

        report_date = parse_iso_date(str(header.get("request_date") or header.get("received_date") or ""))
        computed_age = _compute_age_at_report_date(birth_date_meta["canonical"], report_date)
        reported_age = header.get("age") if isinstance(header.get("age"), int) else None

        age_status = _age_consistency(reported_age, computed_age)

        # ✅ SOURCE OF TRUTH LOGIC
        age_final = None
        age_source = None
        age_warning = None

        if computed_age is not None:
            age_final = computed_age
            age_source = "computed_from_birth_date"
        elif reported_age is not None:
            age_final = reported_age
            age_source = "reported"

        if age_status == "inconsistent_with_birth_date":
            age_warning = {
                "status": "inconsistent",
                "message": f"Reported age ({reported_age}) does not match computed age ({computed_age}) from birth date."
            }

        values = {
            "name": header.get("patient"),
            "patient_id": patient_id_meta["canonical"],
            "birth_date": birth_date_meta["canonical"],
            "age": age_final,
            "sex": sex_meta["canonical"],
            "address": None,
        }

        field_confidence = _confidence_map(
            source_score,
            values,
            ["name", "patient_id", "birth_date", "age", "sex", "address"]
        )
        confidence, confidence_score = _aggregate_field_confidence(field_confidence)

        return {
            "name": header.get("patient"),

            "patient_id_raw": patient_id_meta["raw"],
            "patient_id": patient_id_meta["canonical"],
            "patient_id_normalization_status": patient_id_meta["normalization_status"],

            "birth_date_raw": birth_date_meta["raw"],
            "birth_date": birth_date_meta["canonical"],
            "birth_date_normalization_status": birth_date_meta["normalization_status"],

            "age": age_final,
            "reported_age": reported_age,
            "computed_age_at_request_date": computed_age,
            "age_final": age_final,
            "age_source_of_truth": age_source,
            "age_consistency_status": age_status,
            "age_consistency_warning": age_warning,

            "sex_raw": sex_meta["raw"],
            "sex": sex_meta["canonical"],
            "sex_normalization_status": sex_meta["normalization_status"],

            "address": None,

            "confidence": confidence,
            "confidence_score": confidence_score,
            "field_confidence": field_confidence,
        }

    # =========================
    # GENERIC CASE
    # =========================

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

    patient_id_meta = normalize_named_field(
        "patient_id",
        raw_values.get("identifiant") or values.get("identifiant")
    )

    birth_date_meta = normalize_named_field(
        "birth_date",
        raw_values.get("date_de_naissance") or values.get("date_de_naissance")
    )

    sex_meta = normalize_named_field(
        "sex",
        raw_values.get("sexe") or values.get("sexe")
    )

    reported_age = parse_int(values.get("age"))

    # ⚠️ no report_date → cannot compute safely
    computed_age = None
    age_status = "unknown"

    age_final = reported_age
    age_source = "reported" if reported_age is not None else None
    age_warning = None

    source_score = _source_score(table, block)

    field_confidence = {
        "name": _field_confidence(values.get("patient"), source_score),
        "patient_id": _field_confidence(patient_id_meta["canonical"], source_score),
        "birth_date": _field_confidence(birth_date_meta["canonical"], source_score, parser_applied=True),
        "age": _field_confidence(age_final, source_score, parser_applied=True),
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

        "age": age_final,
        "reported_age": reported_age,
        "computed_age_at_request_date": computed_age,
        "age_final": age_final,
        "age_source_of_truth": age_source,
        "age_consistency_status": age_status,
        "age_consistency_warning": age_warning,

        "sex_raw": sex_meta["raw"],
        "sex": sex_meta["canonical"],
        "sex_normalization_status": sex_meta["normalization_status"],

        "address": values.get("adresse"),

        "confidence": confidence,
        "confidence_score": confidence_score,
        "field_confidence": field_confidence,
    }

def extract_report_metadata(tables: list, blocks: list, page_text_data: list[dict]) -> dict:
    if _is_parasitology_stool_report(page_text_data):
        header = _extract_chu_header(page_text_data)
        source_score = 0.9
        report_id_meta = normalize_named_field("report_id", str(header.get("sample_id") or ""))
        encounter_type_meta = normalize_named_field("encounter_type", str(header.get("origin") or ""))
        specialty_meta = normalize_named_field("specialty", str(header.get("specialty") or ""))
        values = {
            "report_id": report_id_meta["canonical"],
            "encounter_type": encounter_type_meta["canonical"],
            "prescriber": header.get("prescriber"),
            "specialty": specialty_meta["canonical"],
            "status": header.get("status"),
            "exam_name": header.get("exam_name"),
            "sample_label": header.get("sample_label"),
        }
        field_confidence = _confidence_map(
            source_score,
            values,
            ["report_id", "encounter_type", "prescriber", "specialty", "status", "exam_name", "sample_label"],
        )
        confidence, confidence_score = _aggregate_field_confidence(field_confidence)
        return {
            "exam_name": header.get("exam_name"),
            "report_id_raw": report_id_meta["raw"],
            "report_id": report_id_meta["canonical"],
            "report_id_normalization_status": report_id_meta["normalization_status"],
            "request_date_raw": header.get("request_date"),
            "request_date": _parse_chu_datetime(str(header.get("request_date") or "")),
            "received_date_raw": header.get("received_date"),
            "received_date": _parse_chu_datetime(str(header.get("received_date") or "")),
            "encounter_type_raw": encounter_type_meta["raw"],
            "encounter_type": encounter_type_meta["canonical"],
            "encounter_type_normalization_status": encounter_type_meta["normalization_status"],
            "origin": header.get("origin"),
            "service": header.get("service"),
            "prescriber": header.get("prescriber"),
            "specialty_raw": specialty_meta["raw"],
            "specialty": specialty_meta["canonical"],
            "specialty_normalization_status": specialty_meta["normalization_status"],
            "status": header.get("status"),
            "sample_id": report_id_meta["canonical"],
            "sample_type": header.get("specimen"),
            "sample_label": header.get("sample_label"),
            "print_date_raw": header.get("print_date"),
            "print_date": _parse_chu_datetime(str(header.get("print_date") or "")),
            "printed_by": header.get("printed_by"),
            "edited_date_raw": header.get("edited_date"),
            "edited_date": _parse_chu_datetime(str(header.get("edited_date") or "")),
            "edited_by": header.get("edited_by"),
            "validated_by": header.get("validated_by"),
            "validation_date": _parse_chu_datetime(str(header.get("validation_date") or "")),
            "confidence": confidence,
            "confidence_score": confidence_score,
            "field_confidence": field_confidence,
        }

    if _is_chu_lab_report(page_text_data):
        header = _extract_chu_header(page_text_data)
        source_score = 0.88
        report_id_meta = normalize_named_field("report_id", str(header.get("sample_id") or ""))
        report_date_meta = normalize_named_field("report_date", str(header.get("request_date") or header.get("received_date") or ""))
        encounter_type_meta = normalize_named_field("encounter_type", str(header.get("origin") or ""))
        specialty_meta = normalize_named_field("specialty", str(header.get("specialty") or ""))
        values = {
            "report_id": report_id_meta["canonical"],
            "report_date": report_date_meta["canonical"],
            "encounter_type": encounter_type_meta["canonical"],
            "prescriber": header.get("prescriber"),
            "specialty": specialty_meta["canonical"],
            "status": header.get("status"),
        }
        field_confidence = _confidence_map(
            source_score,
            values,
            ["report_id", "report_date", "encounter_type", "prescriber", "specialty", "status"],
        )
        confidence, confidence_score = _aggregate_field_confidence(field_confidence)
        return {
            "report_id_raw": report_id_meta["raw"],
            "report_id": report_id_meta["canonical"],
            "report_id_normalization_status": report_id_meta["normalization_status"],
            "report_date_raw": report_date_meta["raw"],
            "report_date": report_date_meta["canonical"],
            "report_date_normalization_status": report_date_meta["normalization_status"],
            "request_date_raw": header.get("request_date"),
            "request_date": _parse_chu_datetime(str(header.get("request_date") or "")),
            "encounter_type_raw": encounter_type_meta["raw"],
            "encounter_type": encounter_type_meta["canonical"],
            "encounter_type_normalization_status": encounter_type_meta["normalization_status"],
            "origin": header.get("origin"),
            "service": header.get("service"),
            "prescriber": header.get("prescriber"),
            "specialty_raw": specialty_meta["raw"],
            "specialty": specialty_meta["canonical"],
            "specialty_normalization_status": specialty_meta["normalization_status"],
            "status": header.get("status"),
            "sample_id": report_id_meta["canonical"],
            "sample_type": header.get("specimen"),
            "received_date_raw": header.get("received_date"),
            "received_date": _parse_chu_datetime(str(header.get("received_date") or "")),
            "print_date_raw": header.get("print_date"),
            "print_date": _parse_chu_datetime(str(header.get("print_date") or "")),
            "printed_by": header.get("printed_by"),
            "edited_date_raw": header.get("edited_date"),
            "edited_date": _parse_chu_datetime(str(header.get("edited_date") or "")),
            "edited_by": header.get("edited_by"),
            "validated_by": header.get("validated_by"),
            "validation_date": _parse_chu_datetime(str(header.get("validation_date") or "")),
            "confidence": confidence,
            "confidence_score": confidence_score,
            "field_confidence": field_confidence,
        }

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

    """ if low is not None and high is not None and high >= low:
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

    return score """


def _score_to_confidence(score: float) -> str:
    if score >= 8.0:
        return "high"
    if score >= 6.0:
        return "medium"
    return "low"


def _validate_result_consistency(result: dict) -> list[str]:
    issues: list[str] = []
    if result.get("result_kind") in {"qualitative", "pathogen_identification", "microscopy_finding"}:
        return issues
    reference = result.get("reference_range", {}) or {}
    low = reference.get("low")
    high = reference.get("high")
    value = result.get("value_numeric")

    if value is None:
        issues.append("missing_numeric_value")
        return issues
    if low is not None and high is not None and high < low:
        issues.append("invalid_reference_range")
        return issues
    if low is None or high is None:
        return issues

    """ if value < low and flag != "L":
        issues.append("value_below_range_without_low_flag")
    if value < low and flag == "H":
        issues.append("value_below_range_with_high_flag")
    if value > high and flag != "H":
        issues.append("value_above_range_without_high_flag")
    if value > high and flag == "L":
        issues.append("value_above_range_with_low_flag")
    if low <= value <= high and flag in {"H", "L"}:
        issues.append("flag_inconsistent_with_in_range_value") """
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
            table_correction_applied = bool(record.get("Correction OCR appliquee") is True)
            raw_ocr_value = normalize_inline_text(str(record.get("Resultat OCR brut", ""))) if table_correction_applied else None
            table_normalization_reason = (
                normalize_inline_text(str(record.get("Raison normalisation", ""))) if table_correction_applied else None
            )
            unit_raw = normalize_inline_text(str(record.get("Unites", "")))
            unit_meta = normalize_named_field("unit", unit_raw)
            reference = parse_reference_range(str(record.get("Valeurs de reference", "")))
            observation_date_raw = normalize_inline_text(str(record.get("Date", "")))
            observation_date_meta = normalize_named_field("observation_date", observation_date_raw)
            value_numeric_raw = parse_float(value_raw)
            is_ocr_source = getattr(table, "bbox", None) is None
            measurement = _normalize_result_measurements(
                value_raw=value_raw,
                value_numeric=value_numeric_raw,
                reference_range=reference,
                is_ocr_source=is_ocr_source,
            )
            result = {
                    "page_number": table.page_number,
                    "source_page_number": table.page_number,
                    "source_table_id": table.table_id,
                    "source_kind": "ocr_results_table" if is_ocr_source else "results_table",
                    "row_index": row_index,
                    "analyte": analyte,
                    "value_raw": measurement["value_raw"],
                    "value_numeric": measurement["value_numeric"],
                    "unit_raw": unit_meta["raw"],
                    "unit": unit_meta["canonical"],
                    "unit_normalization_status": unit_meta["normalization_status"],
                    "reference_range": reference,
                    "observation_date_raw": observation_date_meta["raw"],
                    "observation_date": observation_date_meta["canonical"],
                    "observation_date_normalization_status": observation_date_meta["normalization_status"],
                }
            if table_correction_applied:
                result.update(
                    {
                        "ocr_correction_applied": True,
                        "raw_ocr_value": raw_ocr_value,
                        "normalized_value": measurement["value_numeric"],
                        "normalization_reason": table_normalization_reason
                        or "decimal inferred from reference range and expected numeric format",
                    }
                )
            elif measurement["ocr_correction_applied"]:
                result.update(
                    {
                        "ocr_correction_applied": True,
                        "raw_ocr_value": measurement["raw_ocr_value"],
                        "normalized_value": measurement["normalized_value"],
                        "normalization_reason": measurement["normalization_reason"],
                    }
                )
            else:
                result["ocr_correction_applied"] = False
            results.append(result)
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
            measurement = _normalize_result_measurements(
                value_raw=parsed.get("value_raw", ""),
                value_numeric=parsed.get("value_numeric"),
                reference_range=parsed.get("reference_range", {}) or {},
                is_ocr_source=True,
            )
            parsed["value_raw"] = measurement["value_raw"]
            parsed["value_numeric"] = measurement["value_numeric"]
            parsed["ocr_correction_applied"] = measurement["ocr_correction_applied"]
            if measurement["ocr_correction_applied"]:
                parsed["raw_ocr_value"] = measurement["raw_ocr_value"]
                parsed["normalized_value"] = measurement["normalized_value"]
                parsed["normalization_reason"] = measurement["normalization_reason"]
            parsed["page_number"] = block.page_number
            parsed["source_page_number"] = block.page_number
            parsed["source_table_id"] = None
            parsed["source_kind"] = "ocr_derived_text"
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
    ]
    return sum(1 for value in fields if value not in (None, "", {}))


def _result_source_priority(result: dict) -> int:
    if result.get("source_kind") == "results_table":
        return 3
    if result.get("source_kind") == "ocr_results_table":
        return 2
    return 1


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


def _normalize_result_measurements(*, value_raw: str, value_numeric: float | None, reference_range: dict, is_ocr_source: bool) -> dict:
    raw_ocr_value = normalize_inline_text(value_raw) if is_ocr_source else None
    normalized_value_raw = normalize_inline_text(value_raw)
    normalized_value = value_numeric
    ocr_correction_applied = False
    normalization_reason = None
    if is_ocr_source:
        repaired_raw, repaired_numeric = repair_numeric_with_reference(value_raw, reference_range)
        if repaired_numeric is not None:
            normalized_value = repaired_numeric
        if repaired_raw and repaired_raw != normalized_value_raw:
            normalized_value_raw = repaired_raw
            ocr_correction_applied = True
            normalization_reason = "decimal inferred from reference range and expected numeric format"

    inferred_flag, inferred_abnormal = _infer_flag_and_abnormal(normalized_value, reference_range)
    return {
        "value_raw": normalized_value_raw,
        "value_numeric": normalized_value,
        "ocr_correction_applied": ocr_correction_applied,
        "raw_ocr_value": raw_ocr_value if ocr_correction_applied else None,
        "normalized_value": normalized_value if ocr_correction_applied else None,
        "normalization_reason": normalization_reason,
    }


RESULT_VALUE_PATTERN = re.compile(
    r"^(?P<comparator>[<>]=?)?\s*(?P<value>-?\d+(?:[.,]\d+)?)\s*(?P<unit>[A-Za-zµ/%]+(?:\[[A-Za-z]+\])?(?:/[A-Za-zµ]+)?|mUI/L|uU/mL)?$",
    flags=re.IGNORECASE,
)

REFERENCE_UNIT_PATTERN = re.compile(
    r"(10\*3/uL|mUI/L|uU/mL|mmol/L|mg/dL|mg/l|mg/L|g/dL|g/l|g/L|µg/dl|UI/ml|UI/L|UI/l|U/L|pg/ml|pg/mL|ng/ml|pmol/l|%)",
    flags=re.IGNORECASE,
)

CHU_SKIP_LABELS = {
    "laboratoire_central",
    "resultats",
    "valeurs_physiologiques",
    "resultats_ant",
    "parametres",
}


def _looks_like_chu_result_value(line: str) -> re.Match[str] | None:
    return RESULT_VALUE_PATTERN.match(normalize_inline_text(line))


def _looks_like_chu_reference(line: str) -> bool:
    text = normalize_inline_text(line)
    label = normalize_label(text)
    if not text:
        return False
    if re.fullmatch(r"[A-Z]{1,5}\s*\d{1,3}(?:-\d{1,3})?", text):
        return False
    if REFERENCE_UNIT_PATTERN.fullmatch(text):
        return True
    reference_terms = (
        "homme",
        "femme",
        "adulte",
        "enfant",
        "nouveau",
        "nourrisson",
        "cordon",
        "sexe_age",
        "valeur_seuil",
        "negatif",
        "positif",
        "risque",
        "normale",
        "taux",
        "indicative",
        "cut_off",
    )
    return (
        any(term in label for term in reference_terms)
        or bool(re.search(r"\d+(?:[,.]\d+)?\s+(?:-|à|a|et)\s+\d+(?:[,.]\d+)?", text, flags=re.IGNORECASE))
        or bool(re.search(r"\(\s*\d+(?:[,.]\d+)?\s*-\s*\d+(?:[,.]\d+)?", text, flags=re.IGNORECASE))
        or text.strip().startswith(("<", ">"))
    )


def _looks_like_chu_admin_line(line: str) -> bool:
    label = normalize_label(line)
    if not label:
        return True
    return (
        label in CHU_SKIP_LABELS
        or label.startswith("ip_patient")
        or label.startswith("patient")
        or label.startswith("ne_e_le")
        or label.startswith("sexe")
        or label.startswith("origine")
        or label.startswith("service")
        or label.startswith("prescripteur")
        or label.startswith("date_demande")
        or label.startswith("date_reception")
        or label.startswith("n_d_echantillon")
        or label.startswith("nature")
        or label.startswith("royaume_du_maroc")
        or label.startswith("ministere")
        or label.startswith("centre_hospitalo")
        or label.startswith("chef")
        or label.startswith("professeur")
        or label.startswith("medecins")
        or label.startswith("infirmier")
        or label.startswith("vice_major")
        or label.startswith("technicien")
        or label.startswith("imprime")
        or label.startswith("page_")
        or label.startswith("adresse_web")
        or label.startswith("tel")
        or label.startswith("le_")
        or label.startswith("valide_e_par")
        or label.startswith("edite_e_par")
        or "المملكة" in line
        or "وزارة" in line
    )


def _looks_like_chu_analyte(line: str) -> bool:
    text = normalize_inline_text(line)
    label = normalize_label(text)
    if not text or _looks_like_chu_admin_line(text) or _looks_like_chu_reference(text) or _looks_like_chu_result_value(text):
        return False
    if label.startswith("automate"):
        return False
    if len(text) > 95:
        return False
    return bool(re.search(r"[A-Za-zÀ-ÿ]", text))


def _simple_reference_range(reference_text: str) -> dict[str, float | str | None]:
    text = normalize_inline_text(reference_text)
    if not text:
        return {"text": None, "low": None, "high": None}
    numeric_values = re.findall(r"-?\d+(?:[,.]\d+)?", text)
    if text.startswith("<") and numeric_values:
        return {"text": text, "low": None, "high": parse_float(numeric_values[0])}
    if text.startswith(">") and numeric_values:
        return {"text": text, "low": parse_float(numeric_values[0]), "high": None}
    if len(numeric_values) == 2 and re.search(r"\d+(?:[,.]\d+)?\s*(?:-|à|a|et)\s*\d+(?:[,.]\d+)?", text, flags=re.IGNORECASE):
        return parse_reference_range(text)
    return {"text": text, "low": None, "high": None}


def _infer_unit_from_reference(reference_text: str) -> str | None:
    matches = REFERENCE_UNIT_PATTERN.findall(reference_text or "")
    if not matches:
        return None
    return normalize_result_unit_text(matches[-1])


def _chu_observation_date(header: dict[str, object]) -> str | None:
    for key in ("request_date", "received_date", "validation_date"):
        parsed = parse_iso_date(str(header.get(key) or ""))
        if parsed:
            return parsed
    return None


def _build_chu_result(
    *,
    analyte: str,
    value_raw: str,
    unit: str | None,
    reference_text: str,
    page_number: int,
    row_index: int,
    observation_date: str | None,
    qualitative: bool = False,
) -> dict:
    reference = {"text": "Qualitatif", "low": None, "high": None} if qualitative else _simple_reference_range(reference_text)
    normalized_unit = normalize_result_unit_text(unit or _infer_unit_from_reference(reference_text) or ("qualitative" if qualitative else "unknown"))
    value_numeric = None if qualitative else parse_float(value_raw)
    confidence_text = f"{analyte} {value_raw} {normalized_unit} {reference_text or ''}"
    confidence_score = compute_confidence(
        confidence_text,
        source="native",
        ocr_correction=False,
        field_length=len(normalize_inline_text(value_raw)),
    )
    return {
        "page_number": page_number,
        "source_page_number": page_number,
        "source_table_id": None,
        "source_kind": "chu_text_fallback",
        "row_index": row_index,
        "analyte": normalize_inline_text(analyte),
        "value_raw": normalize_inline_text(value_raw),
        "value_numeric": value_numeric,
        "unit_raw": unit,
        "unit": normalized_unit,
        "unit_normalization_status": "as_extracted" if normalized_unit and normalized_unit != "unknown" else "needs_review",
        "reference_range": reference,
        "observation_date_raw": observation_date,
        "observation_date": observation_date,
        "observation_date_normalization_status": "as_extracted" if observation_date else "missing",
        "ocr_correction_applied": False,
        "confidence_score": confidence_score,
        "confidence": _score_band(confidence_score),
        "dedup_key": "|".join(
            [
                normalize_label(analyte),
                normalize_inline_text(value_raw),
                normalize_result_unit_text(unit or "") or "",
            ]
        ),
        "is_canonical": True,
        "duplicate_sources": [],
        "result_kind": "qualitative" if qualitative else "numeric",
        "field_confidence": {
            "analyte": _field_confidence(analyte, confidence_score),
            "value_raw": _field_confidence(value_raw, confidence_score),
            "value_numeric": _field_confidence(value_numeric if not qualitative else value_raw, confidence_score, parser_applied=not qualitative),
            "unit": _field_confidence(normalized_unit, confidence_score),
            "reference_range": _field_confidence(reference.get("text"), confidence_score),
            "observation_date": _field_confidence(observation_date, confidence_score, parser_applied=True),
        },
    }


def _extract_chu_qualitative_results(page_text_data: list[dict], observation_date: str | None, start_index: int) -> list[dict]:
    results: list[dict] = []
    row_index = start_index
    for page in page_text_data:
        lines = _page_lines(page)
        for index, line in enumerate(lines[:-1]):
            label = normalize_label(line)
            next_line = normalize_inline_text(lines[index + 1])
            if not next_line.startswith(":"):
                continue
            if _looks_like_chu_admin_line(line) or label in {"resultat_final", "resulat_final"}:
                continue
            value = next_line.lstrip(": ").strip()
            if not value:
                continue
            results.append(
                _build_chu_result(
                    analyte=line,
                    value_raw=value,
                    unit="qualitative",
                    reference_text="Qualitatif",
                    page_number=page["page_number"],
                    row_index=row_index,
                    observation_date=observation_date,
                    qualitative=True,
                )
            )
            row_index += 1

        active = False
        pending_analyte = None
        for index, line in enumerate(lines):
            label = normalize_label(line)
            if label == "parametres" or label.startswith("examen_"):
                active = True
                continue
            if not active:
                continue
            if label.startswith("page_") or label.startswith("adresse_web") or label.startswith("tel"):
                break
            if _looks_like_chu_admin_line(line) or label in CHU_SKIP_LABELS:
                continue
            if label.startswith("commentaire"):
                comment_lines = []
                inline_value = line.split(":", 1)[1].strip() if ":" in line else ""
                if inline_value:
                    comment_lines.append(inline_value)
                for follow in lines[index + 1 :]:
                    follow_label = normalize_label(follow)
                    if follow_label.startswith("page_") or follow_label.startswith("adresse_web") or follow_label.startswith("tel"):
                        break
                    if _looks_like_chu_admin_line(follow):
                        continue
                    comment_lines.append(follow)
                if comment_lines:
                    results.append(
                        _build_chu_result(
                            analyte="Commentaire",
                            value_raw=" ".join(comment_lines),
                            unit="qualitative",
                            reference_text="Qualitatif",
                            page_number=page["page_number"],
                            row_index=row_index,
                            observation_date=observation_date,
                            qualitative=True,
                        )
                    )
                    row_index += 1
                break
            if pending_analyte and label in {"negatif", "positif", "absence", "presence"}:
                results.append(
                    _build_chu_result(
                        analyte=pending_analyte,
                        value_raw=line,
                        unit="qualitative",
                        reference_text="Qualitatif",
                        page_number=page["page_number"],
                        row_index=row_index,
                        observation_date=observation_date,
                        qualitative=True,
                    )
                )
                row_index += 1
                pending_analyte = None
                continue
            if _looks_like_chu_result_value(line) or _looks_like_chu_reference(line):
                pending_analyte = None
                continue
            if _looks_like_chu_analyte(line):
                pending_analyte = line

        for index, line in enumerate(lines):
            if normalize_label(line) not in {"resultat_final", "resulat_final"}:
                continue
            value = ""
            if ":" in line:
                value = line.split(":", 1)[1].strip()
            if not value and index + 1 < len(lines):
                value = lines[index + 1].lstrip(": ").strip()
            if value:
                results.append(
                    _build_chu_result(
                        analyte="Résultat final",
                        value_raw=value,
                        unit="qualitative",
                        reference_text="Qualitatif",
                        page_number=page["page_number"],
                        row_index=row_index,
                        observation_date=observation_date,
                        qualitative=True,
                    )
                )
                row_index += 1
    return results


def extract_chu_lab_results(page_text_data: list[dict]) -> tuple[list[dict], list[dict], dict]:
    header = _extract_chu_header(page_text_data)
    observation_date = _chu_observation_date(header)
    results: list[dict] = []
    current_name_parts: list[str] = []
    reference_parts: list[str] = []
    pending_values: list[tuple[str, str | None]] = []
    row_index = 1

    def flush(page_number: int) -> None:
        nonlocal current_name_parts, reference_parts, pending_values, row_index
        if not current_name_parts or not pending_values:
            current_name_parts = []
            reference_parts = []
            pending_values = []
            return
        value_raw, unit = pending_values[-1]
        results.append(
            _build_chu_result(
                analyte=" ".join(current_name_parts),
                value_raw=value_raw,
                unit=unit,
                reference_text=" ".join(reference_parts),
                page_number=page_number,
                row_index=row_index,
                observation_date=observation_date,
            )
        )
        row_index += 1
        current_name_parts = []
        reference_parts = []
        pending_values = []

    for page in page_text_data:
        page_number = page["page_number"]
        lines = _page_lines(page)
        active = False
        for line in lines:
            label = normalize_label(line)
            if label == "parametres" or label.startswith("examen_"):
                active = True
                continue
            if not active and page_number == 1:
                continue
            if not active and any(marker in label for marker in ("cholesterol", "creatinine", "glucose", "magnesium", "igm_totales")):
                active = True
            if not active:
                continue
            if label.startswith("page_") or label.startswith("adresse_web") or label.startswith("tel"):
                flush(page_number)
                active = False
                continue
            if _looks_like_chu_admin_line(line) or label.startswith("automate"):
                continue
            value_match = _looks_like_chu_result_value(line)
            if value_match and current_name_parts:
                pending_values.append(
                    (
                        normalize_inline_text(
                            f"{value_match.group('comparator') or ''}{value_match.group('value')}"
                        ),
                        normalize_result_unit_text(value_match.group("unit")),
                    )
                )
                continue
            if _looks_like_chu_reference(line) and current_name_parts:
                reference_parts.append(line)
                continue
            if _looks_like_chu_analyte(line):
                if current_name_parts and (reference_parts or pending_values):
                    flush(page_number)
                current_name_parts.append(line)

        flush(page_number)

    qualitative = _extract_chu_qualitative_results(page_text_data, observation_date, len(results) + 1)
    if qualitative and not results:
        results = qualitative

    seen: set[str] = set()
    deduped: list[dict] = []
    for result in results:
        key = result["dedup_key"]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(result)
    stats = {
        "raw_result_rows": len(results),
        "canonical_result_rows": len(deduped),
        "duplicate_rows_removed": max(0, len(results) - len(deduped)),
    }
    return deduped, results, stats


PARASITOLOGY_SECTION_TITLES = {
    "macroscopic_exam": "EXAMEN MACROSCOPIQUE",
    "microscopic_exam": "EXAMEN MICROSCOPIQUE",
    "enrichment_exam": "EXAMEN APRÈS ENRICHISSEMENT",
    "staining_exam": "EXAMEN APRÈS COLORATION",
    "final_result": "RÉSULTAT FINAL",
}


def _clean_qualitative_value(value: str | None) -> str:
    return normalize_inline_text(value or "").lstrip(":").strip()


def _is_false_parasitology_result(section: str, analyte: str, value: str) -> bool:
    analyte_label = normalize_label(analyte)
    value_label = normalize_label(value)
    if not analyte_label or analyte.strip().startswith(":"):
        return True
    if analyte_label in {"examen_macroscopique", "examen_microscopique"}:
        return True
    if section != "final_result" and analyte_label in {"resultat_final", "resulat_final"}:
        return True
    if analyte_label in {"absence", "presence", "negatif", "marron", "molle", "assez_nombreux"}:
        return True
    return analyte_label == "resultat_final" and value_label == "resultat_final"


def _parasitology_result_semantics(section: str, analyte: str, value: str) -> tuple[str, str, bool]:
    analyte_label = normalize_label(analyte)
    value_label = normalize_label(value)
    if section == "final_result":
        return ("pathogen_identification", "pathogen_detected", value_label not in {"negatif", "absence"})
    if section == "staining_exam" and any(term in value_label for term in ("oeufs", "parasite", "ankylostoma", "trichuris")):
        return ("microscopy_finding", "parasite_elements_detected", True)
    if analyte_label in {"leucocytes"} and value_label == "assez_nombreux":
        return ("microscopy_finding", "inflammatory_finding", True)
    if analyte_label in {"hematies"} and value_label == "assez_nombreux":
        return ("microscopy_finding", "blood_cells_detected", True)
    if analyte_label in {"glaire", "pus"} and value_label == "presence":
        return ("qualitative", "positive_finding", True)
    if section == "microscopic_exam":
        return ("microscopy_finding", "normal_or_absent_finding", False)
    return ("qualitative", "not_significant_or_expected", False)


def _parasitology_dedup_key(section: str, analyte: str, value: str | None) -> str:
    return "|".join(
        [
            normalize_label(section),
            normalize_label(analyte),
            normalize_label(value),
        ]
    )


def _make_parasitology_result(
    *,
    section: str,
    section_title: str,
    analyte: str,
    value: str,
    page_number: int,
    row_index: int,
    observation_date: str | None,
    source_line_start: int | None = None,
    source_line_end: int | None = None,
) -> dict | None:
    clean_analyte = normalize_inline_text(analyte).strip(": ")
    clean_value = _clean_qualitative_value(value)
    if _is_false_parasitology_result(section, clean_analyte, clean_value):
        return None
    # _parasitology_result_semantics may also return an abnormality flag; we only
    # need the kind and clinical meaning at this stage.
    semantics = _parasitology_result_semantics(section, clean_analyte, clean_value)
    result_kind = semantics[0]
    confidence_score = compute_confidence(
        f"{section_title} {clean_analyte} {clean_value}",
        source="native",
        ocr_correction=False,
        field_length=len(clean_value),
    )
    return {
        "page_number": page_number,
        "source_page_number": page_number,
        "source_table_id": "logical_results_p001_01",
        "source_kind": "parasitology_section_parser",
        "source_line_start": source_line_start,
        "source_line_end": source_line_end,
        "row_index": row_index,
        "section": section,
        "section_name": section_title,
        "analyte": clean_analyte,
        "value_raw": clean_value,
        "value_numeric": None,
        "unit_raw": "qualitative",
        "unit": "qualitative",
        "unit_normalization_status": "as_extracted",
        "reference_range": {"text": "Qualitatif", "low": None, "high": None},
        "observation_date_raw": observation_date,
        "observation_date": observation_date,
        "observation_date_normalization_status": "as_extracted" if observation_date else "missing",
        "ocr_correction_applied": False,
        "confidence_score": confidence_score,
        "confidence": _score_band(confidence_score),
        "dedup_key": _parasitology_dedup_key(section, clean_analyte, clean_value),
        "is_canonical": True,
        "duplicate_sources": [],
        "result_kind": result_kind,
        "field_confidence": {
            "analyte": _field_confidence(clean_analyte, confidence_score),
            "value_raw": _field_confidence(clean_value, confidence_score),
            "value_numeric": _field_confidence(clean_value, confidence_score),
            "unit": _field_confidence("qualitative", confidence_score),
            "reference_range": _field_confidence("Qualitatif", confidence_score),
            "observation_date": _field_confidence(observation_date, confidence_score, parser_applied=True),
        },
    }


def _find_section_line(lines: list[str], title: str) -> int | None:
    target = normalize_label(title)
    for index, line in enumerate(lines):
        label = normalize_label(line)
        if label == target or target in label:
            return index
    return None


def _parse_label_value_section(
    *,
    lines: list[str],
    start: int,
    end: int,
    section: str,
    section_title: str,
    page_number: int,
    observation_date: str | None,
    row_index: int,
) -> tuple[list[dict], int, int]:
    results: list[dict] = []
    false_positive_count = 0
    index = start
    while index < end - 1:
        analyte = normalize_inline_text(lines[index])
        value = normalize_inline_text(lines[index + 1])
        if value.startswith(":"):
            result = _make_parasitology_result(
                section=section,
                section_title=section_title,
                analyte=analyte,
                value=value,
                page_number=page_number,
                row_index=row_index,
                observation_date=observation_date,
                source_line_start=index + 1,
                source_line_end=index + 2,
            )
            if result is None:
                false_positive_count += 1
            else:
                results.append(result)
                row_index += 1
            index += 2
            continue
        index += 1
    return results, row_index, false_positive_count


def extract_parasitology_stool_results(page_text_data: list[dict]) -> tuple[list[dict], list[dict], dict]:
    header = _extract_chu_header(page_text_data)
    observation_date = parse_iso_date(str(header.get("request_date") or header.get("edited_date") or ""))
    page = page_text_data[0] if page_text_data else {"page_number": 1}
    lines = _page_lines(page)
    page_number = int(page.get("page_number", 1))
    row_index = 1
    raw_results: list[dict] = []
    false_positive_count = 0

    macro = _find_section_line(lines, PARASITOLOGY_SECTION_TITLES["macroscopic_exam"])
    micro = _find_section_line(lines, PARASITOLOGY_SECTION_TITLES["microscopic_exam"])
    enrichment = _find_section_line(lines, PARASITOLOGY_SECTION_TITLES["enrichment_exam"])
    staining = _find_section_line(lines, PARASITOLOGY_SECTION_TITLES["staining_exam"])
    final = _find_section_line(lines, PARASITOLOGY_SECTION_TITLES["final_result"])

    if macro is not None and micro is not None:
        parsed, row_index, false_count = _parse_label_value_section(
            lines=lines,
            start=macro + 1,
            end=micro,
            section="macroscopic_exam",
            section_title=PARASITOLOGY_SECTION_TITLES["macroscopic_exam"],
            page_number=page_number,
            observation_date=observation_date,
            row_index=row_index,
        )
        raw_results.extend(parsed)
        false_positive_count += false_count
    if micro is not None and enrichment is not None:
        parsed, row_index, false_count = _parse_label_value_section(
            lines=lines,
            start=micro + 1,
            end=enrichment,
            section="microscopic_exam",
            section_title=PARASITOLOGY_SECTION_TITLES["microscopic_exam"],
            page_number=page_number,
            observation_date=observation_date,
            row_index=row_index,
        )
        raw_results.extend(parsed)
        false_positive_count += false_count

    for section, title, start_index in [
        ("enrichment_exam", PARASITOLOGY_SECTION_TITLES["enrichment_exam"], enrichment),
        ("staining_exam", PARASITOLOGY_SECTION_TITLES["staining_exam"], staining),
    ]:
        if start_index is None or start_index + 1 >= len(lines):
            continue
        result = _make_parasitology_result(
            section=section,
            section_title=title,
            analyte=title,
            value=lines[start_index + 1],
            page_number=page_number,
            row_index=row_index,
            observation_date=observation_date,
            source_line_start=start_index + 1,
            source_line_end=start_index + 2,
        )
        if result is None:
            false_positive_count += 1
        else:
            raw_results.append(result)
            row_index += 1

    if final is not None:
        final_value = None
        final_line_index = None
        for offset, candidate in enumerate(lines[final + 1 : final + 5]):
            cleaned = _clean_qualitative_value(candidate)
            if normalize_label(cleaned) in {"resultat_final", "resulat_final"}:
                false_positive_count += 1
                continue
            if cleaned:
                final_value = cleaned
                final_line_index = final + 1 + offset
                break
        if final_value:
            result = _make_parasitology_result(
                section="final_result",
                section_title=PARASITOLOGY_SECTION_TITLES["final_result"],
                analyte=PARASITOLOGY_SECTION_TITLES["final_result"],
                value=final_value,
                page_number=page_number,
                row_index=row_index,
                observation_date=observation_date,
                source_line_start=(final + 1),
                source_line_end=(final_line_index + 1) if final_line_index is not None else (final + 2),
            )
            if result is None:
                false_positive_count += 1
            else:
                raw_results.append(result)

    deduped: list[dict] = []
    seen: set[str] = set()
    duplicate_rows_removed = 0
    for result in raw_results:
        key = result["dedup_key"]
        if key in seen:
            duplicate_rows_removed += 1
            continue
        seen.add(key)
        deduped.append(result)

    section_count = len({result["section"] for result in deduped})
    stats = {
        "raw_result_rows": len(raw_results) + duplicate_rows_removed + false_positive_count,
        "canonical_result_rows": len(deduped),
        "duplicate_rows_removed": duplicate_rows_removed,
        "false_positive_result_rows": false_positive_count,
        "section_count": section_count,
        "expected_section_count": 5,
    }
    return deduped, raw_results, stats


def build_parasitology_logical_tables(results: list[dict]) -> list[dict]:
    if not results:
        return []
    records = [
        {
            "section": result.get("section_name"),
            "analyte": result.get("analyte"),
            "value": result.get("value_raw"),
        }
        for result in results
    ]
    return [
        {
            "table_id": "logical_results_p001_01",
            "page_number": 1,
            "row_count": len(records),
            "column_count": 4,
            "csv_path": None,
            "json_path": None,
            "columns": ["section", "analyte", "value"],
            "records": records,
            "preview": records[:5],
            "bbox": None,
            "table_role": "parasitology_results",
            "is_indexable": True,
            "source": "native_text_structured",
        }
    ]


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
        if str(result.get("source_kind", "")).startswith("ocr_"):
            normalized_score = min(normalized_score, 0.92)
        if result.get("ocr_correction_applied"):
            normalized_score = min(normalized_score, 0.88)
        dedup_key = result.get("dedup_key") or _result_dedup_key(result)
        result["confidence_score"] = normalized_score
        result["confidence"] = _score_band(normalized_score)
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


""" def extract_interpretation(blocks: list) -> dict:
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
 """

def extract_validation(page_text_data: list[dict], images: list, ocr_visuals: list, report: dict, blocks: list) -> dict:
    if _is_parasitology_stool_report(page_text_data):
        header = _extract_chu_header(page_text_data)
        edited_by = header.get("edited_by") or report.get("edited_by")
        printed_by = header.get("printed_by") or report.get("printed_by")
        edit_date = _parse_chu_datetime(str(header.get("edited_date") or report.get("edited_date_raw") or ""))
        field_confidence = {
            "validation_title": _field_confidence("Edition laboratoire", 0.9),
            "validated_by": _field_confidence(header.get("validated_by"), 0.0),
            "edited_by": _field_confidence(edited_by, 0.9),
            "edit_date": _field_confidence(edit_date, 0.87, parser_applied=True),
            "printed_by": _field_confidence(printed_by, 0.9),
            "signature_present": _field_confidence(False, 0.0),
            "stamp_present": _field_confidence(False, 0.0),
        }
        confidence, confidence_score = _aggregate_field_confidence(field_confidence)
        return {
            "validation_title": "Edition laboratoire",
            "validated_by": header.get("validated_by"),
            "edited_by": edited_by,
            "edit_date": edit_date,
            "printed_by": printed_by,
            "signature_image_ids": [],
            "stamp_image_ids": [],
            "status": report.get("status"),
            "is_signed": False,
            "is_stamped": False,
            "confidence": confidence,
            "confidence_score": confidence_score,
            "field_confidence": field_confidence,
        }

    if _is_chu_lab_report(page_text_data):
        header = _extract_chu_header(page_text_data)
        validated_by = header.get("validated_by") or report.get("validated_by") or report.get("prescriber")
        specialty = report.get("specialty") or header.get("specialty")
        field_confidence = {
            "validation_title": _field_confidence("Validation laboratoire", 0.82),
            "validated_by": _field_confidence(validated_by, 0.82),
            "specialty": _field_confidence(specialty, 0.78),
            "signature_present": _field_confidence(False, 0.0),
            "stamp_present": _field_confidence(False, 0.0),
        }
        confidence, confidence_score = _aggregate_field_confidence(field_confidence)
        return {
            "validation_title": "Validation laboratoire",
            "validated_by": validated_by,
            "specialty": specialty,
            "signature_image_ids": [],
            "stamp_image_ids": [],
            "status": report.get("status"),
            "is_signed": False,
            "is_stamped": False,
            "confidence": confidence,
            "confidence_score": confidence_score,
            "field_confidence": field_confidence,
        }

    signature_ids = [image.image_id for image in images if image.image_type == "signature"]
    stamp_ids = [image.image_id for image in images if image.image_type == "stamp_or_seal"]
    signature_ids.extend(visual.visual_id for visual in ocr_visuals if visual.visual_type == "signature")
    stamp_ids.extend(visual.visual_id for visual in ocr_visuals if visual.visual_type == "stamp_or_seal")

    validation_block = _find_block(blocks, "validation_block")
    structured_fields = getattr(validation_block, "structured_fields", {}) if validation_block else {}

    clinician = structured_fields.get("validated_by")
    specialty = structured_fields.get("specialty")
    report_prescriber = report.get("prescriber")
    report_specialty = report.get("specialty")

    clinician_label = normalize_label(clinician or "")
    specialty_label = normalize_label(specialty or "")
    if clinician_label and ("validation_medicale" in clinician_label or "cachet_du_service" in clinician_label):
        clinician = None
    if specialty and report_prescriber and normalize_label(specialty) == normalize_label(report_prescriber):
        clinician = report_prescriber
        specialty = report_specialty
    if specialty_label and ("validation_medicale" in specialty_label or "cachet_du_service" in specialty_label):
        specialty = None

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

    clinician_label = normalize_label(clinician or "")
    specialty_label = normalize_label(specialty or "")
    if clinician_label and ("validation_medicale" in clinician_label or "cachet_du_service" in clinician_label):
        clinician = None
    if specialty and report_prescriber and normalize_label(specialty) == normalize_label(report_prescriber):
        clinician = report_prescriber
        specialty = report_specialty
    if specialty_label and ("validation_medicale" in specialty_label or "cachet_du_service" in specialty_label):
        specialty = None

    if specialty and report_specialty and normalize_label(report_specialty) in normalize_label(specialty):
        specialty = report_specialty
    if not clinician and report_prescriber:
        clinician = report_prescriber
    if not specialty and report_specialty:
        specialty = report_specialty

    validation_title = structured_fields.get("validation_title", "Validation")
    validation_title_label = normalize_label(validation_title)
    if "validation_medicale" in validation_title_label:
        validation_title = "Validation medicale"

    field_confidence = {
        "validation_title": _field_confidence(validation_title, float(getattr(validation_block, "confidence_score", 0.65)) if validation_block else 0.0),
        "validated_by": _field_confidence(clinician or report_prescriber, _source_score(block=validation_block) if validation_block else 0.65),
        "specialty": _field_confidence(specialty or report_specialty, _source_score(block=validation_block) if validation_block else 0.65),
        "signature_present": _field_confidence(bool(signature_ids), 0.92 if signature_ids else 0.0),
        "stamp_present": _field_confidence(bool(stamp_ids), 0.92 if stamp_ids else 0.0),
    }
    confidence, confidence_score = _aggregate_field_confidence(field_confidence)
    return {
        "validation_title": validation_title,
        "validated_by": clinician or report_prescriber,
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
    logical_tables: list[dict] = []
    if document_type == "parasitology_stool_report":
        results, raw_results, dedup_stats = extract_parasitology_stool_results(page_text_data)
        logical_tables = build_parasitology_logical_tables(results)
    else:
        results, raw_results, dedup_stats = extract_results(tables, blocks)
    if not results and document_type != "parasitology_stool_report":
        results, raw_results, dedup_stats = extract_results_from_blocks(blocks)
    if _is_chu_lab_report(page_text_data) and not results and document_type != "parasitology_stool_report":
        results, raw_results, dedup_stats = extract_chu_lab_results(page_text_data)
    validation = extract_validation(page_text_data, images, ocr_visuals, report, blocks)
    validation_report = build_validation_report(results=results, raw_results=raw_results)
    validation_report["deduplication"] = dedup_stats
    validation_report.update(
        {
            "false_positive_result_rows": dedup_stats.get("false_positive_result_rows", 0),
            "duplicate_rows_removed": dedup_stats.get("duplicate_rows_removed", validation_report.get("duplicate_rows_removed", 0)),
            "section_count": dedup_stats.get("section_count"),
            "expected_section_count": dedup_stats.get("expected_section_count"),
            "facility_extracted": bool(facility.get("organization") and facility.get("phone")),
            "age_consistency_status": patient.get("age_consistency_status"),
            "computed_age_reference_date": "request_date" if patient.get("computed_age_at_request_date") is not None else None,
            "validation_actor_status": "explicit_validator_present"
            if validation.get("validated_by")
            else ("edition_only" if validation.get("edited_by") else "missing"),
            "result_quality_status": "passed" if results and not validation_report.get("result_consistency_issues") else "needs_review",
        }
    )
    return {
        "document_type": document_type,
        "facility": facility,
        "patient": patient,
        "report": report,
        "results": results,
        "logical_tables": logical_tables,
        "validation": validation,
        "validation_report": validation_report,
    }
