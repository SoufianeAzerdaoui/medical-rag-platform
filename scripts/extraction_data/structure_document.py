from __future__ import annotations

from pathlib import Path

from schemas import DocumentData, PageData
from utils import page_name, write_json


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

    table_map: dict[int, list[str]] = {}
    for table in tables:
        table_map.setdefault(table.page_number, []).append(table.table_id)

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
        interpretation=structured_data.get("interpretation", {}),
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

    document.tables = [table.to_dict() for table in tables]
    promoted_ocr_visual_images = [
        _promote_ocr_visual_to_image_dict(visual)
        for visual in ocr_visuals
        if visual.is_indexable and visual.visual_type in {"clinical_chart", "medical_illustration"}
    ]
    document.images = [image.to_dict() for image in images] + promoted_ocr_visual_images
    document.ocr_visuals = [visual.to_dict() for visual in ocr_visuals]
    document.blocks = [block.to_dict() for block in blocks]
    write_json(out_dir / "document.json", document.to_dict())
    return document
