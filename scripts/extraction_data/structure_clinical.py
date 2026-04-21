from __future__ import annotations

import re

from utils import (
    deduplicate_dicts,
    normalize_flag,
    normalize_inline_text,
    normalize_label,
    parse_float,
    parse_int,
    parse_iso_date,
    parse_reference_range,
)


def _find_table(tables: list, role: str):
    return next((table for table in tables if table.table_role == role), None)


def _find_block(blocks: list, block_type: str):
    return next((block for block in blocks if block.block_type == block_type), None)


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


def detect_document_type(page_text_data: list[dict]) -> str:
    if not page_text_data:
        return "medical_report"
    text = page_text_data[0].get("native_text", "").lower()
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


def extract_patient_info(tables: list, blocks: list) -> dict:
    table = _find_table(tables, "patient_info_table")
    values = _table_to_key_values(table)
    if not values:
        block = _find_block(blocks, "patient_info_block")
        values = _extract_labeled_fields(
            block.text if block else "",
            ["Patient", "Identifiant", "Date de naissance", "Age", "Sexe", "Adresse"],
        )
    patient_id = values.get("identifiant")
    return {
        "name": values.get("patient"),
        "patient_id": re.sub(r"\s+", "", patient_id) if patient_id else None,
        "birth_date": parse_iso_date(values.get("date_de_naissance")),
        "age": parse_int(values.get("age")),
        "sex": values.get("sexe"),
        "address": values.get("adresse"),
    }


def extract_report_metadata(tables: list, blocks: list) -> dict:
    table = _find_table(tables, "report_info_table")
    values = _table_to_key_values(table)
    if not values:
        block = _find_block(blocks, "report_info_block")
        values = _extract_labeled_fields(
            block.text if block else "",
            ["Numero de rapport", "Date du document", "Type de rencontre", "Prescripteur", "Specialite", "Statut"],
        )
    return {
        "report_id": values.get("numero_de_rapport"),
        "report_date": parse_iso_date(values.get("date_du_document")),
        "encounter_type": values.get("type_de_rencontre"),
        "prescriber": values.get("prescripteur"),
        "specialty": values.get("specialite"),
        "status": values.get("statut"),
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
    return {
        "analyte": normalize_inline_text(groups["analyte"]),
        "value_raw": groups["value"],
        "value_numeric": parse_float(groups["value"]),
        "unit": groups["unit"],
        "reference_range": {
            "text": f'{groups["low"]} - {groups["high"]}',
            "low": parse_float(groups["low"]),
            "high": parse_float(groups["high"]),
        },
        "alert_flag": normalize_flag(groups.get("flag")),
        "is_abnormal": normalize_flag(groups.get("flag")) in {"H", "L"},
        "observation_date": parse_iso_date(groups.get("date")),
    }


def extract_results(tables: list, blocks: list) -> list[dict]:
    results: list[dict] = []
    for table in tables:
        if table.table_role != "results_table":
            continue
        for record in table.records:
            analyte = normalize_inline_text(str(record.get("Analyse / Observation", "")))
            value_raw = normalize_inline_text(str(record.get("Resultat", "")))
            unit = normalize_inline_text(str(record.get("Unites", "")))
            reference = parse_reference_range(str(record.get("Valeurs de reference", "")))
            alert = normalize_flag(str(record.get("Alert\ne", record.get("Alerte", ""))))
            observation_date = parse_iso_date(str(record.get("Date", "")))
            results.append(
                {
                    "page_number": table.page_number,
                    "source_table_id": table.table_id,
                    "analyte": analyte,
                    "value_raw": value_raw,
                    "value_numeric": parse_float(value_raw),
                    "unit": unit or None,
                    "reference_range": reference,
                    "alert_flag": alert,
                    "is_abnormal": alert in {"H", "L"},
                    "observation_date": observation_date,
                }
            )
    return deduplicate_dicts(
        results,
        keys=["analyte", "value_raw", "unit", "observation_date"],
    )


def extract_results_from_blocks(blocks: list) -> list[dict]:
    results: list[dict] = []
    for block in blocks:
        if block.block_type != "results_table_block":
            continue
        for line in block.text.splitlines():
            parsed = _parse_result_line(line)
            if not parsed:
                continue
            parsed["page_number"] = block.page_number
            parsed["source_table_id"] = None
            results.append(parsed)
    return deduplicate_dicts(results, keys=["analyte", "value_raw", "unit", "observation_date"])


def extract_interpretation(blocks: list) -> dict:
    interpretation_blocks = [block for block in blocks if block.block_type == "clinical_interpretation_block"]
    summary_blocks = [block for block in blocks if block.block_type == "summary_block"]
    return {
        "text": "\n\n".join(block.text for block in interpretation_blocks if block.text),
        "sections": [
            {
                "page_number": block.page_number,
                "title": block.section_title,
                "text": block.text,
            }
            for block in interpretation_blocks
        ],
        "summary_text": "\n\n".join(block.text for block in summary_blocks if block.text),
    }


def extract_validation(page_text_data: list[dict], images: list, report: dict) -> dict:
    signature_ids = [image.image_id for image in images if image.image_type == "signature"]
    stamp_ids = [image.image_id for image in images if image.image_type == "stamp_or_seal"]

    clinician = None
    specialty = None
    for page in page_text_data:
        blocks = page.get("text_blocks", [])
        page_width = page.get("width", 0)
        for index, block in enumerate(blocks):
            if "validation_medicale" not in normalize_label(block["text"]):
                continue
            candidate_blocks = [
                candidate
                for candidate in blocks[index + 1 :]
                if candidate["bbox"]["x0"] < page_width * 0.45
                and candidate["bbox"]["y0"] >= block["bbox"]["y0"]
                and candidate["bbox"]["y0"] <= block["bbox"]["y0"] + 80
            ]
            if candidate_blocks:
                clinician = normalize_inline_text(candidate_blocks[0]["text"])
            if len(candidate_blocks) >= 2:
                specialty = normalize_inline_text(candidate_blocks[1]["text"])
            break
        if clinician or specialty:
            break

    report_specialty = report.get("specialty")
    if specialty and report_specialty and normalize_label(report_specialty) in normalize_label(specialty):
        specialty = report_specialty

    return {
        "validated_by": clinician or report.get("prescriber"),
        "specialty": specialty or report_specialty,
        "signature_image_ids": signature_ids,
        "stamp_image_ids": stamp_ids,
        "status": report.get("status"),
        "is_signed": bool(signature_ids),
        "is_stamped": bool(stamp_ids),
    }


def build_structured_document(
    *,
    page_text_data: list[dict],
    tables: list,
    images: list,
    blocks: list,
) -> dict:
    document_type = detect_document_type(page_text_data)
    facility = extract_facility_info(page_text_data)
    patient = extract_patient_info(tables, blocks)
    report = extract_report_metadata(tables, blocks)
    results = extract_results(tables, blocks)
    if not results:
        results = extract_results_from_blocks(blocks)
    interpretation = extract_interpretation(blocks)
    validation = extract_validation(page_text_data, images, report)
    return {
        "document_type": document_type,
        "facility": facility,
        "patient": patient,
        "report": report,
        "results": results,
        "interpretation": interpretation,
        "validation": validation,
    }
