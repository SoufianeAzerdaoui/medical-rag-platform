from __future__ import annotations

import json
import re
from pathlib import Path

import fitz
import pandas as pd

from schemas import TableAsset
from utils import (
    bbox_from_sequence,
    canonicalize_uuid_like,
    ensure_dir,
    normalize_named_field,
    normalize_inline_text,
    normalize_label,
    normalize_ocr_analyte_text,
    normalize_result_unit_text,
    parse_reference_range,
    repair_numeric_with_reference,
    optional_import,
    sanitize_json_data,
)


pdfplumber = optional_import("pdfplumber")

HEADER_LABEL_MAP = {
    "field": "Field",
    "value": "Value",
    "patient": "Patient",
    "identifiant": "Identifiant",
    "date_de_naissance": "Date de naissance",
    "age": "Age",
    "sexe": "Sexe",
    "adresse": "Adresse",
    "numero_de_rapport": "Numero de rapport",
    "date_du_document": "Date du document",
    "type_de_rencontre": "Type de rencontre",
    "prescripteur": "Prescripteur",
    "specialite": "Specialite",
    "statut": "Statut",
    "analyse_observation": "Analyse / Observation",
    "resultat": "Resultat",
    "unites": "Unites",
    "valeurs_de_reference": "Valeurs de reference",
    "alerte": "Alerte",
    "alert_e": "Alerte",
    "date": "Date",
}

FIELD_VALUE_KEYS = {
    "patient",
    "identifiant",
    "date_de_naissance",
    "age",
    "sexe",
    "adresse",
    "numero_de_rapport",
    "date_du_document",
    "type_de_rencontre",
    "prescripteur",
    "specialite",
    "statut",
}

BOILERPLATE_PATTERNS = (
    "document_synthetique",
    "usage_exclusif_pour_tests_ocr",
    "parsing_retrieval_multimodal",
)


def _clean_cell_text(value: str | None) -> str:
    if value is None:
        return ""
    text = normalize_inline_text(str(value))
    if not text:
        return ""
    if any(pattern in normalize_label(text) for pattern in BOILERPLATE_PATTERNS):
        return ""
    return text


def _normalize_field_label(value: str) -> str:
    return HEADER_LABEL_MAP.get(normalize_label(value), value)


def _normalize_field_value_frame(frame: pd.DataFrame) -> pd.DataFrame:
    records = frame.to_dict(orient="records")
    normalized_records: list[dict[str, str]] = []
    field_name_by_label = {
        "Identifiant": "patient_id",
        "Date de naissance": "birth_date",
        "Date du document": "report_date",
        "Numero de rapport": "report_id",
        "Sexe": "sex",
        "Type de rencontre": "encounter_type",
        "Specialite": "specialty",
    }
    for record in records:
        field = _normalize_field_label(_clean_cell_text(record.get("Field", "")))
        value = _clean_cell_text(record.get("Value", ""))
        field_name = field_name_by_label.get(field)
        if field_name:
            value = normalize_named_field(field_name, value)["canonical"] or value
        normalized_records.append({"Field": field, "Value": value})
    return pd.DataFrame(normalized_records, columns=["Field", "Value"])


def _canonicalize_header(header: list[str]) -> list[str]:
    canonical: list[str] = []
    seen: dict[str, int] = {}
    for index, column in enumerate(header, start=1):
        label = normalize_label(column)
        column_name = HEADER_LABEL_MAP.get(label, column or f"column_{index}")
        count = seen.get(column_name, 0)
        if count:
            column_name = f"{column_name}_{count + 1}"
        seen[column_name] = count + 1
        canonical.append(column_name)
    return canonical


def _looks_like_field_value_table(rows: list[list[str]]) -> bool:
    if not rows:
        return False
    sample = rows[: min(len(rows), 6)]
    hits = 0
    for row in sample:
        if len(row) < 2:
            continue
        if normalize_label(row[0]) in FIELD_VALUE_KEYS:
            hits += 1
    return hits >= max(2, len(sample) // 2)


def _deduplicate_table_rows(rows: list[list[str]]) -> list[list[str]]:
    deduped: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for row in rows:
        key = tuple(row)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _normalize_table(raw_table: list[list[str | None]]) -> pd.DataFrame:
    rows = [[_clean_cell_text(cell) for cell in row] for row in raw_table]
    rows = [row for row in rows if any(cell for cell in row)]
    if not rows:
        return pd.DataFrame()

    rows = _deduplicate_table_rows(rows)
    if len(rows[0]) == 2 and _looks_like_field_value_table(rows):
        return _normalize_field_value_frame(pd.DataFrame(rows, columns=["Field", "Value"]))

    header = _canonicalize_header(rows[0])
    data_rows = rows[1:] if len(rows) > 1 else []
    if len(set(header)) != len(header) or any(not col for col in header):
        header = [f"column_{idx + 1}" for idx in range(len(header))]
        data_rows = rows
    return pd.DataFrame(data_rows, columns=header)


def _classify_table(frame: pd.DataFrame, page_number: int) -> tuple[str, bool]:
    normalized_columns = {normalize_label(column) for column in frame.columns}

    if {"analyse_observation", "resultat", "unites"}.issubset(normalized_columns):
        return "results_table", True
    if {"field", "value"}.issubset(normalized_columns):
        first_column = str(frame.columns[0])
        second_column = str(frame.columns[1])
        first_values = {
            normalize_label(str(record.get(first_column, "")))
            for record in frame.head(6).to_dict(orient="records")
        }
        if first_values & {"patient", "identifiant", "date_de_naissance", "age", "sexe", "adresse"}:
            return "patient_info_table", False
        if first_values & {"numero_de_rapport", "date_du_document", "type_de_rencontre", "prescripteur", "specialite", "statut"}:
            return "report_info_table", False
    if "patient" in normalized_columns:
        return "patient_info_table", False
    if "numero_de_rapport" in normalized_columns:
        return "report_info_table", False
    if page_number == 1 and len(frame.columns) == 2:
        left_key = normalize_label(frame.columns[0])
        if left_key in {"patient", "identifiant"}:
            return "patient_info_table", False
        if left_key in {"numero_de_rapport", "date_du_document"}:
            return "report_info_table", False
    return "unknown", False


def _write_table_asset(
    frame: pd.DataFrame,
    page_index: int,
    table_index: int,
    tables_dir: Path,
    bbox: dict[str, float] | None = None,
    table_role: str | None = None,
    is_indexable: bool | None = None,
) -> TableAsset:
    table_id = f"table_p{page_index:03d}_{table_index:02d}"
    csv_path = tables_dir / f"{table_id}.csv"
    json_path = tables_dir / f"{table_id}.json"
    frame.to_csv(csv_path, index=False)
    records = sanitize_json_data(frame.to_dict(orient="records"))
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    if table_role is None or is_indexable is None:
        detected_role, detected_indexable = _classify_table(frame, page_index)
        table_role = detected_role if table_role is None else table_role
        is_indexable = detected_indexable if is_indexable is None else is_indexable
    return TableAsset(
        table_id=table_id,
        page_number=page_index,
        row_count=int(frame.shape[0]),
        column_count=int(frame.shape[1]),
        csv_path=str(csv_path),
        json_path=str(json_path),
        columns=[str(column) for column in frame.columns],
        records=records,
        preview=sanitize_json_data(frame.head(5).to_dict(orient="records")),
        bbox=bbox,
        table_role=table_role,
        is_indexable=is_indexable,
    )


def _table_fingerprint(table: TableAsset) -> tuple:
    records = tuple(
        tuple((key, normalize_inline_text(str(value))) for key, value in sorted(record.items()))
        for record in table.records
    )
    return (
        table.table_role,
        tuple(table.columns),
        records,
    )


def _deduplicate_tables(assets: list[TableAsset]) -> list[TableAsset]:
    deduped: list[TableAsset] = []
    seen: set[tuple] = set()
    for asset in assets:
        fingerprint = _table_fingerprint(asset)
        if fingerprint in seen:
            Path(asset.csv_path).unlink(missing_ok=True)
            Path(asset.json_path).unlink(missing_ok=True)
            continue
        seen.add(fingerprint)
        deduped.append(asset)
    return deduped


def _result_row_fingerprint(record: dict[str, object]) -> tuple[str, str, str]:
    return (
        normalize_label(str(record.get("Analyse / Observation", ""))),
        normalize_inline_text(str(record.get("Resultat", ""))),
        normalize_result_unit_text(str(record.get("Unites", ""))) or "",
    )


def _deduplicate_result_rows_across_tables(assets: list[TableAsset]) -> list[TableAsset]:
    deduped_assets: list[TableAsset] = []
    seen_rows: set[tuple[str, str, str]] = set()
    for asset in assets:
        if asset.table_role != "results_table":
            deduped_assets.append(asset)
            continue

        kept_records = []
        for record in asset.records:
            fingerprint = _result_row_fingerprint(record)
            if fingerprint in seen_rows:
                continue
            seen_rows.add(fingerprint)
            kept_records.append(record)

        if not kept_records:
            Path(asset.csv_path).unlink(missing_ok=True)
            Path(asset.json_path).unlink(missing_ok=True)
            continue

        asset.records = kept_records
        asset.preview = kept_records[:5]
        asset.row_count = len(kept_records)
        frame = pd.DataFrame(kept_records, columns=asset.columns)
        frame.to_csv(asset.csv_path, index=False)
        Path(asset.json_path).write_text(
            json.dumps(sanitize_json_data(frame.to_dict(orient="records")), ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        deduped_assets.append(asset)
    return deduped_assets


def _ocr_lines_in_page_band(ocr_asset, start_pattern: str, end_patterns: list[str] | None = None) -> list[dict]:
    if not ocr_asset:
        return []
    start_index = None
    end_index = None
    for index, block in enumerate(ocr_asset.blocks):
        label = normalize_label(block["text"])
        if start_index is None and start_pattern in label:
            start_index = index
            continue
        if start_index is not None and end_patterns and any(pattern in label for pattern in end_patterns):
            end_index = index
            break
    if start_index is None:
        return []
    return ocr_asset.blocks[start_index:end_index]


def _header_lines_from_ocr(ocr_asset) -> list[str]:
    if not ocr_asset:
        return []

    lines: list[str] = []
    for block in ocr_asset.blocks:
        text = normalize_inline_text(block["text"])
        label = normalize_label(text)
        if not text or label.startswith("document_synthetique"):
            continue
        if label.startswith("resultats_selectionnes") or "analyse_observation" in label:
            break
        lines.append(text)
    return lines


def _looks_like_labeled_header_line(text: str) -> bool:
    label = normalize_label(text)
    return any(
        label.startswith(pattern)
        for pattern in (
            "patient",
            "numero_de_rapport",
            "identifiant",
            "date_du_document",
            "type_de_rencontre",
            "date_de_naissance",
            "prescripteur",
            "age",
            "specialite",
            "sexe",
            "statut",
            "adresse",
        )
    )


def _clean_identifier_fragment(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9-]", "", normalize_inline_text(text))


def _normalize_report_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9-]", "", normalize_inline_text(text).upper())


def _strip_label(text: str, label: str) -> str:
    return normalize_inline_text(re.sub(rf"^{re.escape(label)}\s*", "", text, flags=re.IGNORECASE))


def _extract_header_fields_from_ocr_asset(ocr_asset) -> tuple[dict[str, str], dict[str, str]]:
    patient: dict[str, str] = {}
    report: dict[str, str] = {}
    lines = _header_lines_from_ocr(ocr_asset)

    address_lines: list[str] = []
    identifier_prefix = ""
    explicit_identifier = ""
    index = 0

    while index < len(lines):
        text = lines[index]
        label = normalize_label(text)

        if "patient" in label and "numero_de_rapport" in label:
            match = re.search(r"Patient\s+(.+?)\s+Numero de rapport\s+(.+)$", text, flags=re.IGNORECASE)
            if match:
                patient["patient"] = normalize_inline_text(match.group(1))
                report_id = _normalize_report_id(match.group(2))
                if report_id:
                    report["report_id"] = report_id
            index += 1
            continue

        if "date_du_document" in label:
            parts = re.split(r"Date du document", text, maxsplit=1, flags=re.IGNORECASE)
            prefix = _clean_identifier_fragment(parts[0]) if parts else ""
            if len(prefix) >= 8:
                identifier_prefix = prefix
            if len(parts) == 2:
                report["report_date"] = normalize_inline_text(parts[1])
            index += 1
            continue

        if label.startswith("identifiant"):
            value = _strip_label(text, "Identifiant")
            if not value and index + 1 < len(lines) and not _looks_like_labeled_header_line(lines[index + 1]):
                index += 1
                value = lines[index]
            explicit_identifier = _clean_identifier_fragment(value)
            index += 1
            continue

        if label.startswith("date_de_naissance"):
            patient["birth_date"] = _strip_label(text, "Date de naissance")
            index += 1
            continue

        if label.startswith("age"):
            patient["age"] = _strip_label(text, "Age")
            index += 1
            continue

        if label.startswith("sexe"):
            patient["sex"] = _strip_label(text, "Sexe")
            index += 1
            continue

        if label.startswith("type_de_rencontre"):
            report["encounter_type"] = _strip_label(text, "Type de rencontre")
            index += 1
            continue

        if label.startswith("prescripteur"):
            report["prescriber"] = _strip_label(text, "Prescripteur")
            index += 1
            continue

        if label.startswith("specialite"):
            report["specialty"] = _strip_label(text, "Specialite")
            index += 1
            continue

        if label.startswith("statut"):
            report["status"] = _strip_label(text, "Statut")
            look_ahead = index + 1
            while look_ahead < len(lines) and not _looks_like_labeled_header_line(lines[look_ahead]):
                address_lines.append(lines[look_ahead])
                look_ahead += 1
            index += 1
            continue

        if label.startswith("adresse"):
            address_value = _strip_label(text, "Adresse")
            if address_value:
                address_lines.append(address_value)
            look_ahead = index + 1
            while look_ahead < len(lines) and not _looks_like_labeled_header_line(lines[look_ahead]):
                address_lines.append(lines[look_ahead])
                look_ahead += 1
            index += 1
            continue

        index += 1

    normalized_identifier = canonicalize_uuid_like(f"{identifier_prefix}{explicit_identifier}")
    if normalized_identifier:
        patient["patient_id"] = normalized_identifier
    elif explicit_identifier:
        patient["patient_id"] = explicit_identifier
    elif identifier_prefix:
        patient["patient_id"] = identifier_prefix

    if address_lines:
        address = normalize_inline_text(" ".join(address_lines))
        digit_match = re.search(r"\d.*", address)
        patient["address"] = digit_match.group(0) if digit_match else address

    return patient, report


def _clean_ocr_continuation_text(text: str) -> str:
    normalized = normalize_inline_text(text)
    normalized = re.sub(r"[^A-Za-z0-9/\[\]()%._#+ -]+", " ", normalized)
    tokens = [token for token in normalized.split() if re.search(r"[A-Za-z]", token)]
    return " ".join(tokens).strip()


def _line_has_date(text: str) -> bool:
    return bool(re.search(r"\d{1,2}\s+\w+\.?\s+\d{4}\s*$", normalize_inline_text(text), flags=re.IGNORECASE))


def _extract_result_reference(prefix: str) -> tuple[str | None, str]:
    cleaned = normalize_inline_text(prefix).strip()
    cleaned = re.sub(r"[\s:;,.]+$", "", cleaned)

    flag = "-"
    flag_match = re.search(r"\b([HL])\b\s*$", cleaned)
    if flag_match:
        flag = flag_match.group(1)
        cleaned = normalize_inline_text(cleaned[: flag_match.start()])
    else:
        cleaned = re.sub(r"[\s:;,.\\/-]+$", "", cleaned)

    reference_match = re.search(
        r"(Selon contexte|-?\d+(?:[.,]\d+)?\s*-\s*-?\d+(?:[.,]\d+)?)\s*$",
        cleaned,
        flags=re.IGNORECASE,
    )
    if not reference_match:
        return None, prefix
    reference_text = normalize_inline_text(reference_match.group(1))
    remainder = normalize_inline_text(cleaned[: reference_match.start()])
    return reference_text, f"{remainder}|||{flag}"


def _row_expects_suffix(prefix_text: str, main_text: str) -> bool:
    text = normalize_inline_text(f"{prefix_text} {main_text}")
    date_match = re.search(r"(\d{1,2}\s+\w+\.?\s+\d{4})\s*$", text, flags=re.IGNORECASE)
    if date_match:
        text = normalize_inline_text(text[: date_match.start()])
    reference_text, remainder = _extract_result_reference(text)
    if not reference_text:
        return False
    seed = remainder.split("|||", 1)[0]
    tokens = seed.split()
    value_index = None
    for index in range(len(tokens) - 1, -1, -1):
        if re.fullmatch(r"-?\d+(?:[.,]\d+)?", tokens[index]):
            value_index = index
            break
    analyte_seed = " ".join(tokens[:value_index]) if value_index is not None else seed
    last_token = normalize_label(analyte_seed).split("_")[-1] if analyte_seed else ""
    return last_token in {"by", "or", "in", "automated", "count", "serum", "plasma"}


def _looks_like_new_row_prefix(text: str) -> bool:
    normalized = normalize_inline_text(text)
    if not normalized:
        return False
    first_word = normalize_label(normalized).split("_")[0]
    if first_word in {"automated", "count", "or", "plasma", "serum", "blood"}:
        return False
    if "[" in normalized or "/" in normalized:
        return True
    return bool(re.search(r"\b(Blood|Serum|Plasma|Pressure|Height)\b", normalized, flags=re.IGNORECASE))


def _parse_ocr_result_rows(ocr_asset) -> list[dict]:
    if not ocr_asset:
        return []
    rows = _ocr_lines_in_page_band(
        ocr_asset,
        start_pattern="analyse_observation",
        end_patterns=["document_synthetique", "validation_medicale", "interpretation_biologique", "resume_du_dossier"],
    )
    row_candidates: list[dict[str, object]] = []
    prefix_buffer: list[str] = []
    current_row: dict[str, object] | None = None

    for block in rows[1:]:
        text = normalize_inline_text(block["text"])
        label = normalize_label(text)
        if not text or label in {"alert", "date"}:
            continue
        if _line_has_date(text):
            current_row = {
                "prefix": list(prefix_buffer),
                "main": text,
                "suffix": [],
            }
            row_candidates.append(current_row)
            prefix_buffer = []
            continue

        continuation = _clean_ocr_continuation_text(text)
        if not continuation:
            continue
        if current_row and _looks_like_new_row_prefix(continuation):
            prefix_buffer = [continuation]
            current_row = None
            continue
        if current_row and _row_expects_suffix(" ".join(current_row["prefix"]), str(current_row["main"])):
            current_row["suffix"].append(continuation)
            continue
        prefix_buffer.append(continuation)

    records: list[dict] = []
    for candidate in row_candidates:
        prefix_text = normalize_inline_text(" ".join(candidate["prefix"]))
        main_text = normalize_inline_text(str(candidate["main"]))
        suffix_text = normalize_inline_text(" ".join(candidate["suffix"]))

        date_match = re.search(r"(\d{1,2}\s+\w+\.?\s+\d{4})$", main_text, flags=re.IGNORECASE)
        if not date_match:
            continue
        date_text = normalize_inline_text(date_match.group(1))
        prefix = normalize_inline_text(main_text[: date_match.start()])
        reference_text, remainder = _extract_result_reference(prefix)
        if not reference_text:
            continue
        prefix, flag = remainder.rsplit("|||", 1)

        tokens = prefix.split()
        if len(tokens) < 2:
            continue

        value_index = None
        for index in range(len(tokens) - 1, -1, -1):
            if re.fullmatch(r"-?\d+(?:[.,]\d+)?", tokens[index]):
                value_index = index
                break
        if value_index is None or value_index == 0 or value_index >= len(tokens) - 1:
            continue

        analyte = normalize_inline_text(" ".join(part for part in [prefix_text, " ".join(tokens[:value_index]), suffix_text] if part))
        analyte = normalize_ocr_analyte_text(analyte)
        if not analyte:
            continue
        value = normalize_inline_text(tokens[value_index])
        unit = normalize_result_unit_text(" ".join(tokens[value_index + 1 :])) or "unknown"
        records.append(
            {
                "Analyse / Observation": analyte,
                "Resultat": value,
                "Unites": unit,
                "Valeurs de reference": reference_text,
                "Alert\ne": flag,
                "Date": date_text,
            }
        )

    return records


def _annotate_ocr_result_corrections(records: list[dict]) -> list[dict]:
    annotated: list[dict] = []
    for record in records:
        enriched = dict(record)
        raw_value = normalize_inline_text(str(enriched.get("Resultat", "")))
        reference_range = parse_reference_range(str(enriched.get("Valeurs de reference", "")))
        flag = normalize_inline_text(str(enriched.get("Alerte", enriched.get("Alert\ne", "")))) or None
        repaired_raw, repaired_numeric = repair_numeric_with_reference(raw_value, reference_range, flag=flag)
        corrected = bool(repaired_raw and repaired_raw != raw_value)
        if corrected:
            enriched["Resultat"] = repaired_raw
            enriched["Correction OCR appliquee"] = True
            enriched["Resultat OCR brut"] = raw_value
            enriched["Valeur normalisee"] = repaired_numeric
            enriched["Raison normalisation"] = "decimal inferred from reference range and expected numeric format"
        else:
            enriched["Correction OCR appliquee"] = False
            enriched["Resultat OCR brut"] = raw_value
            enriched["Valeur normalisee"] = repaired_numeric
            enriched["Raison normalisation"] = ""
        annotated.append(enriched)
    return annotated


def extract_tables_from_ocr(output_dir: str | Path, ocr_results: dict[int, object]) -> list[TableAsset]:
    tables_dir = ensure_dir(Path(output_dir) / "tables")
    assets: list[TableAsset] = []
    table_index = 1

    first_page_ocr = ocr_results.get(1)
    if first_page_ocr and getattr(first_page_ocr, "used", False):
        patient_fields, report_fields = _extract_header_fields_from_ocr_asset(first_page_ocr)
        if patient_fields:
            patient_records = [
                {"Field": label, "Value": value}
                for label, value in [
                    ("Patient", patient_fields.get("patient")),
                    ("Identifiant", patient_fields.get("patient_id")),
                    ("Date de naissance", patient_fields.get("birth_date")),
                    ("Age", patient_fields.get("age")),
                    ("Sexe", patient_fields.get("sex")),
                    ("Adresse", patient_fields.get("address")),
                ]
                if value
            ]
            if patient_records:
                frame = pd.DataFrame(patient_records, columns=["Field", "Value"])
                assets.append(
                    _write_table_asset(
                        frame=frame,
                        page_index=1,
                        table_index=table_index,
                        tables_dir=tables_dir,
                        bbox=None,
                        table_role="patient_info_table",
                        is_indexable=False,
                    )
                )
                table_index += 1

        if report_fields:
            report_records = [
                {"Field": label, "Value": value}
                for label, value in [
                    ("Numero de rapport", report_fields.get("report_id")),
                    ("Date du document", report_fields.get("report_date")),
                    ("Type de rencontre", report_fields.get("encounter_type")),
                    ("Prescripteur", report_fields.get("prescriber")),
                    ("Specialite", report_fields.get("specialty")),
                    ("Statut", report_fields.get("status")),
                ]
                if value
            ]
            if report_records:
                frame = pd.DataFrame(report_records, columns=["Field", "Value"])
                assets.append(
                    _write_table_asset(
                        frame=frame,
                        page_index=1,
                        table_index=table_index,
                        tables_dir=tables_dir,
                        bbox=None,
                        table_role="report_info_table",
                        is_indexable=False,
                    )
                )
                table_index += 1

    for page_number in sorted(ocr_results):
        asset = ocr_results[page_number]
        if not getattr(asset, "used", False):
            continue
        result_records = _parse_ocr_result_rows(asset)
        if not result_records:
            continue
        result_records = _annotate_ocr_result_corrections(result_records)
        frame = pd.DataFrame(
            result_records,
            columns=[
                "Analyse / Observation",
                "Resultat",
                "Unites",
                "Valeurs de reference",
                "Alerte",
                "Date",
                "Correction OCR appliquee",
                "Resultat OCR brut",
                "Valeur normalisee",
                "Raison normalisation",
            ],
        )
        assets.append(
            _write_table_asset(
                frame=frame,
                page_index=page_number,
                table_index=table_index,
                tables_dir=tables_dir,
                bbox=None,
                table_role="results_table",
                is_indexable=True,
            )
        )
        table_index += 1

    return _deduplicate_tables(assets)


def extract_tables(pdf_path: str | Path, output_dir: str | Path) -> list[TableAsset]:
    source = Path(pdf_path).expanduser().resolve()
    tables_dir = ensure_dir(Path(output_dir) / "tables")
    assets: list[TableAsset] = []

    with fitz.open(source) as doc:
        for page_index, page in enumerate(doc, start=1):
            finder = page.find_tables()
            for table_index, table in enumerate(finder.tables, start=1):
                frame = _normalize_table(table.extract())
                if frame.empty:
                    continue
                assets.append(
                    _write_table_asset(
                        frame=frame,
                        page_index=page_index,
                        table_index=table_index,
                        tables_dir=tables_dir,
                        bbox=bbox_from_sequence(table.bbox),
                    )
                )

    if assets or pdfplumber is None:
        return _deduplicate_tables(assets)

    with pdfplumber.open(source) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            raw_tables = page.extract_tables() or []
            for table_index, raw_table in enumerate(raw_tables, start=1):
                frame = _normalize_table(raw_table)
                if frame.empty:
                    continue
                assets.append(
                    _write_table_asset(
                        frame=frame,
                        page_index=page_index,
                        table_index=table_index,
                        tables_dir=tables_dir,
                        bbox=None,
                    )
                )

    return _deduplicate_tables(assets)
