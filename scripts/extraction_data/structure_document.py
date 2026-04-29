from __future__ import annotations

import sys
from pathlib import Path

from schemas import DocumentData, PageData
from utils import page_name, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _promote_ocr_visual_to_image_dict(visual) -> dict:
    visual_dict = visual.to_dict()
    return {
        "image_id": visual.visual_id,
        "page_number": visual.page_number,
        "file_path": visual.file_path,
        "width": visual.width,
        "height": visual.height,
        "ext": visual.ext,
        "bbox": visual.bbox,
        "xref": None,
        "page_coverage": None,
        "image_type": visual.visual_type,
        "role": visual.role,
        "is_indexable": visual.is_indexable,
        "context_text": visual.context_text,
        "source": "ocr_visual",
        "source_visual_id": visual.visual_id,
        "ocr_visual": visual_dict,
    }

def _sync_consistency_checks(document: dict) -> dict:
    """
    Keep one canonical consistency_checks source.

    The clinical layer writes the correct checks inside:
    validation_report.consistency_checks

    The top-level consistency_checks must mirror it, not come from an older checker.
    """
    validation_report = document.get("validation_report", {})
    report_checks = validation_report.get("consistency_checks")

    if isinstance(report_checks, dict) and report_checks:
        document["consistency_checks"] = report_checks

    return document

def _compact_dict(data: dict) -> dict:
    return {key: value for key, value in data.items() if value is not None}


def _as_repo_relative_path(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        path = Path(value).expanduser().resolve()
    except OSError:
        return value
    if not path.is_absolute():
        return value
    try:
        rel = path.relative_to(PROJECT_ROOT)
    except ValueError:
        return value
    return rel.as_posix()


def _parasitology_result_view(result: dict) -> dict:
    view = {
        "page_number": result.get("page_number"),
        "source_page_number": result.get("source_page_number"),
        "source_table_id": result.get("source_table_id"),
        "source_kind": result.get("source_kind"),
        "source_line_start": result.get("source_line_start"),
        "source_line_end": result.get("source_line_end"),
        "row_index": result.get("row_index"),
        "section": result.get("section"),
        "section_name": result.get("section_name"),
        "parameter": result.get("parameter") or result.get("analyte"),
        "result": result.get("result") or result.get("value_raw"),
        "result_kind": result.get("result_kind"),
        "confidence": result.get("confidence"),
        "confidence_score": result.get("confidence_score"),
        "dedup_key": result.get("dedup_key"),
        "is_canonical": result.get("is_canonical"),
    }
    if result.get("duplicate_sources"):
        view["duplicate_sources"] = result.get("duplicate_sources")
    return _compact_dict(view)


def _find_block(document: dict, block_type: str) -> dict | None:
    return next((block for block in document.get("blocks", []) if block.get("block_type") == block_type), None)


def _parasitology_block_fields(document: dict) -> dict[str, dict]:
    patient = document.get("patient", {})
    report = document.get("report", {})
    facility = document.get("facility", {})
    validation = document.get("validation", {})
    results = document.get("results", [])

    result_views = [_parasitology_result_view(result) for result in results]
    results_by_section: dict[str, list[dict]] = {}
    for result in result_views:
        results_by_section.setdefault(result.get("section", ""), []).append(result)

    def first_result(section: str) -> dict | None:
        items = results_by_section.get(section, [])
        return items[0] if items else None

    return {
        "facility_block": _compact_dict(
            {
                "country": facility.get("country"),
                "ministry": facility.get("ministry"),
                "organization": facility.get("organization"),
                "department": facility.get("department"),
                "laboratory": facility.get("laboratory"),
                "website": facility.get("website"),
                "phone": facility.get("phone"),
                "fax": facility.get("fax"),
                "source": "native_text_section_parser",
            }
        ),
        "patient_info_block": _compact_dict(
            {
                "patient_id": patient.get("patient_id"),
                "patient_name": patient.get("name"),
                "birth_date": patient.get("birth_date"),
                "reported_age": patient.get("reported_age"),
                "computed_age_at_request_date": patient.get("computed_age_at_request_date"),
                "age_consistency_status": patient.get("age_consistency_status"),
                "sex": patient.get("sex"),
                "origin": report.get("origin"),
                "service": report.get("service"),
                "prescriber": report.get("prescriber"),
                "request_date": report.get("request_date"),
                "received_date": report.get("received_date"),
                "sample_id": report.get("sample_id"),
                "sample_type": report.get("sample_type"),
                "sample_label": report.get("sample_label"),
                "source": "native_text_section_parser",
            }
        ),
        "report_metadata_block": _compact_dict(
            {
                "exam_name": report.get("exam_name"),
                "sample_id": report.get("sample_id"),
                "request_date": report.get("request_date"),
                "received_date": report.get("received_date"),
                "origin": report.get("origin"),
                "service": report.get("service"),
                "prescriber": report.get("prescriber"),
                "sample_type": report.get("sample_type"),
                "sample_label": report.get("sample_label"),
                "print_date": report.get("print_date"),
                "printed_by": report.get("printed_by"),
                "edited_date": report.get("edited_date"),
                "edited_by": report.get("edited_by"),
                "source": "native_text_section_parser",
            }
        ),
        "sample_block": _compact_dict(
            {
                "exam_name": report.get("exam_name"),
                "sample_id": report.get("sample_id"),
                "sample_type": report.get("sample_type"),
                "sample_label": report.get("sample_label") or "SELLES N°1",
                "source": "native_text_section_parser",
            }
        ),
        "macroscopic_exam_block": {
            "results": results_by_section.get("macroscopic_exam", []),
            "result_count": len(results_by_section.get("macroscopic_exam", [])),
            "source": "native_text_section_parser",
        },
        "microscopic_exam_block": {
            "results": results_by_section.get("microscopic_exam", []),
            "result_count": len(results_by_section.get("microscopic_exam", [])),
            "source": "native_text_section_parser",
        },
        "enrichment_exam_block": _compact_dict(
            {
                "result": first_result("enrichment_exam"),
                "source": "native_text_section_parser",
            }
        ),
        "staining_exam_block": _compact_dict(
            {
                "result": first_result("staining_exam"),
                "source": "native_text_section_parser",
            }
        ),
        "final_result_block": _compact_dict(
            {
                "parameter": "RÉSULTAT FINAL",
                "result": first_result("final_result").get("result") if first_result("final_result") else None,
                "result_kind": first_result("final_result").get("result_kind") if first_result("final_result") else None,
                "clinical_significance": first_result("final_result").get("clinical_significance") if first_result("final_result") else None,
                "source": "native_text_section_parser",
            }
        ),
        "edition_block": _compact_dict(
            {
                "edited_by": validation.get("edited_by"),
                "edited_date": validation.get("edited_date"),
                "printed_by": validation.get("printed_by"),
                "print_date": report.get("print_date"),
                "source": "native_text_section_parser",
            }
        ),
    }


def _project_parasitology_blocks(document: dict) -> None:
    block_fields = _parasitology_block_fields(document)
    for block in document.get("blocks", []):
        block_type = block.get("block_type")
        if block_type not in block_fields:
            continue
        block["structured_fields"] = block_fields[block_type]
        block["normalized_text"] = block.get("text", "")
        block["index_text"] = block.get("text", "")
    for page in document.get("pages", []):
        for block in page.get("blocks", []):
            block_type = block.get("block_type")
            if block_type not in block_fields:
                continue
            block["structured_fields"] = block_fields[block_type]
            block["normalized_text"] = block.get("text", "")
            block["index_text"] = block.get("text", "")


def project_parasitology_stool_report(document: dict) -> dict:
    patient = document.get("patient", {})
    report = document.get("report", {})
    facility = document.get("facility", {})
    validation = document.get("validation", {})

    patient_projected = _compact_dict(
        {
            "ip_patient": patient.get("patient_id") or patient.get("ip_patient"),
            "patient_id": patient.get("patient_id"),
            "patient_id_raw": patient.get("patient_id_raw"),
            "patient_id_source_label": "IP Patient" if (patient.get("patient_id") or patient.get("patient_id_raw")) else None,
            "name": patient.get("name"),
            "birth_date_raw": patient.get("birth_date_raw"),
            "birth_date": patient.get("birth_date"),
            "age": patient.get("age_final") or patient.get("age"),
            "reported_age": patient.get("reported_age", patient.get("age")),
            "computed_age_at_request_date": patient.get("computed_age_at_request_date"),
            "age_final": patient.get("age_final"),
            "age_source_of_truth": patient.get("age_source_of_truth"),
            "age_consistency_status": patient.get("age_consistency_status"),
            "age_consistency_warning": patient.get("age_consistency_warning"),
            "sex_raw": patient.get("sex_raw"),
            "sex": patient.get("sex"),
            "confidence": patient.get("confidence"),
            "confidence_score": patient.get("confidence_score"),
        }
    )
    report_projected = _compact_dict(
        {
            "exam_name": report.get("exam_name"),
            "origin": report.get("origin"),
            "service": report.get("service"),
            "prescriber": report.get("prescriber"),
            "request_date_raw": report.get("request_date_raw"),
            "request_date": report.get("request_date"),
            "received_date_raw": report.get("received_date_raw"),
            "received_date": report.get("received_date"),
            "sample_id_raw": report.get("sample_id") or report.get("report_id"),
            "sample_id": report.get("sample_id") or report.get("report_id"),
            "sample_type": report.get("sample_type"),
            "sample_label": report.get("sample_label") or ("SELLES N°1" if (report.get("sample_type") or "").upper() == "SELLES" else None),
            "print_date_raw": report.get("print_date_raw"),
            "print_date": report.get("print_date"),
            "printed_by": report.get("printed_by"),
            "edited_date_raw": report.get("edited_date_raw"),
            "edited_date": report.get("edited_date"),
            "edited_by": report.get("edited_by"),
            "confidence": report.get("confidence"),
            "confidence_score": report.get("confidence_score"),
        }
    )

    facility_projected = _compact_dict(
        {
            "country": facility.get("country"),
            "ministry": facility.get("ministry"),
            "organization": facility.get("organization"),
            "department": facility.get("department"),
            "laboratory": facility.get("laboratory"),
            "website": facility.get("website"),
            "phone": facility.get("phone"),
            "fax": facility.get("fax"),
            "confidence": facility.get("confidence"),
            "confidence_score": facility.get("confidence_score"),
        }
    )

    projected_results = [_parasitology_result_view(result) for result in document.get("results", [])]
    document["results"] = projected_results
    projected_result_by_key: dict[tuple[object, object, object], dict] = {}
    for result in projected_results:
        key_core = (result.get("section"), result.get("parameter"), result.get("result"))
        projected_result_by_key[key_core] = result
        # Some logical tables store section titles (section_name) instead of section codes.
        key_title = (result.get("section_name"), result.get("parameter"), result.get("result"))
        projected_result_by_key[key_title] = result
    for table in document.get("tables", []):
        if table.get("table_id") != "logical_results_p001_01" or table.get("table_role") != "parasitology_results":
            continue
        # Logical tables are not written as separate files; omit null paths in export.
        if table.get("csv_path") in {None, ""}:
            table.pop("csv_path", None)
        if table.get("json_path") in {None, ""}:
            table.pop("json_path", None)
        table["columns"] = ["section", "parameter", "result", "result_kind"]
        projected_records: list[dict] = []
        for record in table.get("records", []):
            section = record.get("section")
            parameter = record.get("parameter") or record.get("analyte")
            result_value = record.get("result") or record.get("value")
            projected_match = projected_result_by_key.get((section, parameter, result_value))
            projected_records.append(
                _compact_dict(
                    {
                        "section": section,
                        "parameter": parameter,
                        "result": result_value,
                        "result_kind": (
                            projected_match.get("result_kind")
                            if projected_match
                            else record.get("result_kind")
                        ),        
                    }
                )
            )
        table["records"] = projected_records
        table["column_count"] = len(table["columns"])
        table["preview"] = table["records"][:5]
    document["patient"] = patient_projected
    document["report"] = report_projected
    document["facility"] = facility_projected
    document["validation"] = _compact_dict(
        {
            "validation_title": validation.get("validation_title"),
            "edited_by": validation.get("edited_by"),
            "edited_date": validation.get("edit_date") or validation.get("edited_date"),
            "printed_by": validation.get("printed_by"),
            "print_date": report_projected.get("print_date"),
            "is_signed": validation.get("is_signed"),
            "is_stamped": validation.get("is_stamped"),
            "confidence": validation.get("confidence"),
            "confidence_score": validation.get("confidence_score"),
        }
    )
    _project_parasitology_blocks(document)
    return document


def apply_document_type_schema_projection(document: dict) -> dict:
    if document.get("document_type") == "parasitology_stool_report":
        document = project_parasitology_stool_report(document)

    # Export cleanup: if the PDF had no OCR, don't emit an empty ocr_visuals array.
    if not document.get("ocr_available"):
        document.pop("ocr_visuals", None)

    # Export cleanup: avoid absolute, machine-specific paths in production exports.
    source_pdf = _as_repo_relative_path(document.get("source_pdf"))
    if source_pdf:
        document["source_pdf"] = source_pdf
    output_dir = _as_repo_relative_path(document.get("output_dir"))
    if output_dir:
        document["output_dir"] = output_dir

    for image in document.get("images", []) or []:
        if isinstance(image, dict) and image.get("file_path"):
            image["file_path"] = _as_repo_relative_path(image.get("file_path")) or image.get("file_path")
    for visual in document.get("ocr_visuals", []) or []:
        if isinstance(visual, dict) and visual.get("file_path"):
            visual["file_path"] = _as_repo_relative_path(visual.get("file_path")) or visual.get("file_path")
    
    document = _sync_consistency_checks(document)
    return document


def structure_document(
    ingest_result: dict,
    classify_result: dict,
    page_text_data: list[dict],
    tables: list,
    images: list,
    ocr_visuals: list,
    blocks: list,
    structured_data: dict,
    ocr_results: dict[int, object],
    output_dir: str | Path,
) -> DocumentData:
    out_dir = Path(output_dir).expanduser().resolve()
    pages_dir = out_dir / "pages"
    logical_tables = structured_data.get("logical_tables", [])

    table_map: dict[int, list[str]] = {}
    for table in tables:
        table_map.setdefault(table.page_number, []).append(table.table_id)
    for table in logical_tables:
        page_number = int(table.get("page_number", 1))
        table_id = table.get("table_id")
        if table_id:
            table_map.setdefault(page_number, []).append(table_id)

    image_map: dict[int, list[str]] = {}
    for image in images:
        image_map.setdefault(image.page_number, []).append(image.image_id)

    ocr_visual_map: dict[int, list[str]] = {}
    for visual in ocr_visuals:
        ocr_visual_map.setdefault(visual.page_number, []).append(visual.visual_id)
        if visual.is_indexable and visual.visual_type in {"clinical_chart", "medical_illustration"}:
            image_map.setdefault(visual.page_number, []).append(visual.visual_id)

    block_map: dict[int, list[dict]] = {}
    for block in blocks:
        block_map.setdefault(block.page_number, []).append(block.to_dict())

    warnings: list[str] = []
    pages_needing_ocr = [page["page_number"] for page in page_text_data if page["native_text_chars"] < 30]
    ocr_used_on_any_page = any(bool(getattr(asset, "used", False)) for asset in ocr_results.values())
    if not classify_result["native_text_available"]:
        warnings.append("No meaningful native text detected; OCR may be required.")
    if pages_needing_ocr and not ocr_results:
        warnings.append("OCR fallback was not produced for pages requiring OCR.")
    elif pages_needing_ocr and not ocr_used_on_any_page:
        warnings.append("Pages require OCR, but no OCR engine was available.")

    document = DocumentData(
        doc_id=ingest_result["doc_id"],
        source_pdf=ingest_result["source_pdf"],
        output_dir=str(out_dir),
        pdf_type=classify_result["pdf_type"],
        document_type=structured_data["document_type"],
        page_count=ingest_result["page_count"],
        native_text_available=classify_result["native_text_available"],
        ocr_available=ocr_used_on_any_page,
        extraction_warnings=warnings,
        metadata={
            "pdf_metadata": ingest_result["metadata"],
            "page_sizes": ingest_result["page_sizes"],
            "classification": classify_result,
        },
        facility=structured_data.get("facility", {}),
        patient=structured_data.get("patient", {}),
        report=structured_data.get("report", {}),
        results=structured_data.get("results", []),
        validation=structured_data.get("validation", {}),
        validation_report=structured_data.get("validation_report", {}),
    )

    for page in page_text_data:
        page_number = page["page_number"]
        ocr_asset = ocr_results.get(page_number)
        ocr_text = getattr(ocr_asset, "text", "") if ocr_asset else ""
        ocr_used = bool(ocr_asset and getattr(ocr_asset, "used", False) and ocr_text)
        final_text = page.get("final_text", ocr_text if ocr_used else page["native_text"])
        text_source = page.get("text_source", "ocr" if ocr_used and not page["native_text"] else ("hybrid" if ocr_used else "native"))

        page_record = PageData(
            page_number=page_number,
            width=page["width"],
            height=page["height"],
            native_text=page["native_text"],
            ocr_text=page.get("ocr_text", ocr_text),
            final_text=final_text,
            text_source=text_source,
            native_text_chars=page["native_text_chars"],
            ocr_used=ocr_used,
            ocr_text_chars=page.get("ocr_text_chars", len(ocr_text)),
            table_ids=table_map.get(page_number, []),
            image_ids=image_map.get(page_number, []),
            ocr_visual_ids=ocr_visual_map.get(page_number, []),
            blocks=block_map.get(page_number, []),
        )
        write_json(pages_dir / f"{page_name(page_number)}.json", page_record.to_dict())
        document.pages.append(page_record.to_dict())

    document.tables = [table.to_dict() for table in tables] + logical_tables
    promoted_ocr_visual_images = [
        _promote_ocr_visual_to_image_dict(visual)
        for visual in ocr_visuals
        if visual.is_indexable and visual.visual_type in {"clinical_chart", "medical_illustration"}
    ]
    document.images = [image.to_dict() for image in images] + promoted_ocr_visual_images
    document.ocr_visuals = [visual.to_dict() for visual in ocr_visuals]
    document.blocks = [block.to_dict() for block in blocks]
    projected_document = apply_document_type_schema_projection(document.to_dict())
    for page in projected_document.get("pages", []):
        page_number = int(page.get("page_number", 0))
        if page_number:
            write_json(pages_dir / f"{page_name(page_number)}.json", page)
    write_json(out_dir / "document.json", projected_document)
    return document
