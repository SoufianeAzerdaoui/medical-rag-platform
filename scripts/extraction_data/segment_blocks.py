from __future__ import annotations

from collections import defaultdict

from schemas import DocumentBlock
from utils import bbox_union, normalize_inline_text, normalize_label


def _table_to_text(table) -> str:
    if table.table_role in {"patient_info_table", "report_info_table"} and len(table.columns) >= 2:
        left_key = table.columns[0]
        right_key = table.columns[1]
        lines: list[str] = []
        for record in table.records:
            lines.append(f"{record.get(left_key, '')}: {record.get(right_key, '')}")
        return "\n".join(normalize_inline_text(line) for line in lines if normalize_inline_text(line))

    lines: list[str] = []
    for record in table.records:
        parts = [f"{column}: {normalize_inline_text(str(record.get(column, '')))}" for column in table.columns]
        lines.append(" | ".join(part for part in parts if not part.endswith(": ")))
    return "\n".join(line for line in lines if line)


def _make_block(
    *,
    block_id: str,
    page_number: int,
    block_type: str,
    section_title: str,
    text: str,
    bboxes: list[dict[str, float]],
    source_table_ids: list[str] | None = None,
    source_image_ids: list[str] | None = None,
    source_text_block_ids: list[str] | None = None,
    is_indexable: bool = True,
) -> DocumentBlock:
    return DocumentBlock(
        block_id=block_id,
        page_number=page_number,
        block_type=block_type,
        section_title=section_title,
        text=normalize_inline_text(text),
        bbox=bbox_union([bbox for bbox in bboxes if bbox]),
        source_table_ids=source_table_ids or [],
        source_image_ids=source_image_ids or [],
        source_text_block_ids=source_text_block_ids or [],
        is_indexable=is_indexable,
    )


def _collect_text_blocks_between(
    text_blocks: list[dict],
    start_y: float,
    end_y: float,
    *,
    min_x: float | None = None,
    max_x: float | None = None,
) -> list[dict]:
    selected: list[dict] = []
    for block in text_blocks:
        bbox = block["bbox"]
        if bbox["y0"] < start_y or bbox["y0"] >= end_y:
            continue
        if min_x is not None and bbox["x1"] <= min_x:
            continue
        if max_x is not None and bbox["x0"] >= max_x:
            continue
        selected.append(block)
    return selected


def _find_text_block(text_blocks: list[dict], contains: str) -> dict | None:
    target = normalize_label(contains)
    for block in text_blocks:
        if target in normalize_label(block["text"]):
            return block
    return None


def _blocks_below(text_blocks: list[dict], start_y: float, end_y: float | None = None) -> list[dict]:
    selected = [block for block in text_blocks if block["bbox"]["y0"] >= start_y]
    if end_y is not None:
        selected = [block for block in selected if block["bbox"]["y0"] < end_y]
    return selected


def _create_column_blocks_from_ocr(
    *,
    page_number: int,
    text_blocks: list[dict],
    start_y: float,
    end_y: float,
    page_width: float,
    block_index: int,
) -> tuple[list[DocumentBlock], int]:
    mid_x = page_width * 0.5
    selected = _blocks_below(text_blocks, start_y, end_y)
    left_blocks = [block for block in selected if block["bbox"]["x0"] < mid_x]
    right_blocks = [block for block in selected if block["bbox"]["x0"] >= mid_x]
    created: list[DocumentBlock] = []

    if left_blocks:
        created.append(
            _make_block(
                block_id=f"block_{block_index:03d}",
                page_number=page_number,
                block_type="patient_info_block",
                section_title="Patient information",
                text="\n".join(block["text"] for block in left_blocks),
                bboxes=[block["bbox"] for block in left_blocks],
                source_text_block_ids=[block["text_block_id"] for block in left_blocks],
                is_indexable=False,
            )
        )
        block_index += 1

    if right_blocks:
        created.append(
            _make_block(
                block_id=f"block_{block_index:03d}",
                page_number=page_number,
                block_type="report_info_block",
                section_title="Report metadata",
                text="\n".join(block["text"] for block in right_blocks),
                bboxes=[block["bbox"] for block in right_blocks],
                source_text_block_ids=[block["text_block_id"] for block in right_blocks],
                is_indexable=False,
            )
        )
        block_index += 1

    return created, block_index


def _create_results_block_from_ocr(
    *,
    page_number: int,
    text_blocks: list[dict],
    title_candidates: list[str],
    stop_candidates: list[str],
    block_index: int,
) -> tuple[list[DocumentBlock], int]:
    start_block = None
    for title in title_candidates:
        start_block = _find_text_block(text_blocks, title)
        if start_block:
            break
    if start_block is None:
        return [], block_index

    stop_y = None
    for title in stop_candidates:
        stop_block = _find_text_block(text_blocks, title)
        if stop_block and stop_block["bbox"]["y0"] > start_block["bbox"]["y0"]:
            stop_y = stop_block["bbox"]["y0"]
            break

    candidate_blocks = [
        block
        for block in text_blocks
        if block["bbox"]["y0"] >= start_block["bbox"]["y0"]
        and (stop_y is None or block["bbox"]["y0"] < stop_y)
        and not normalize_label(block["text"]).startswith("document_synthetique")
    ]
    if not candidate_blocks:
        return [], block_index

    block = _make_block(
        block_id=f"block_{block_index:03d}",
        page_number=page_number,
        block_type="results_table_block",
        section_title="Selected results",
        text="\n".join(item["text"] for item in candidate_blocks),
        bboxes=[item["bbox"] for item in candidate_blocks],
        source_text_block_ids=[item["text_block_id"] for item in candidate_blocks],
        is_indexable=True,
    )
    return [block], block_index + 1


def _find_companion_title(text_blocks: list[dict], title_block: dict) -> dict | None:
    candidates = [
        block
        for block in text_blocks
        if block["text_block_id"] != title_block["text_block_id"]
        and block["is_bold"]
        and block["max_font_size"] >= title_block["max_font_size"] - 0.5
        and abs(block["bbox"]["y0"] - title_block["bbox"]["y0"]) <= 20
        and block["bbox"]["x0"] > title_block["bbox"]["x0"] + 40
    ]
    return min(candidates, key=lambda block: block["bbox"]["x0"]) if candidates else None


def _find_title_above_image(text_blocks: list[dict], image_bbox: dict[str, float]) -> dict | None:
    candidates = [
        block
        for block in text_blocks
        if block["is_bold"]
        and block["bbox"]["y1"] <= image_bbox["y0"] + 5
        and image_bbox["y0"] - block["bbox"]["y1"] <= 25
        and block["bbox"]["x1"] >= image_bbox["x0"] - 20
        and block["bbox"]["x0"] <= image_bbox["x1"] + 20
    ]
    return max(candidates, key=lambda block: block["bbox"]["y1"]) if candidates else None


def build_blocks(page_text_data: list[dict], tables: list, images: list) -> list[DocumentBlock]:
    blocks: list[DocumentBlock] = []
    block_index = 1
    tables_by_page: dict[int, list] = defaultdict(list)
    images_by_page: dict[int, list] = defaultdict(list)

    for table in tables:
        tables_by_page[table.page_number].append(table)
    for image in images:
        images_by_page[image.page_number].append(image)

    for page in page_text_data:
        page_number = page["page_number"]
        page_width = page["width"]
        page_height = page["height"]
        text_blocks = page.get("text_blocks", [])
        page_tables = tables_by_page.get(page_number, [])
        page_images = images_by_page.get(page_number, [])

        footer_blocks = [
            block
            for block in text_blocks
            if normalize_label(block["text"]).startswith("document_synthetique")
        ]

        if page_number == 1:
            logo = next((image for image in page_images if image.image_type == "logo_or_branding"), None)
            title_block = next(
                (block for block in text_blocks if "compte rendu" in block["text"].lower()),
                None,
            )
            if title_block:
                bboxes = [title_block["bbox"]]
                image_ids: list[str] = []
                if logo and logo.bbox:
                    bboxes.append(logo.bbox)
                    image_ids.append(logo.image_id)
                blocks.append(
                    _make_block(
                        block_id=f"block_{block_index:03d}",
                        page_number=page_number,
                        block_type="document_header",
                        section_title=title_block["text"],
                        text=title_block["text"],
                        bboxes=bboxes,
                        source_image_ids=image_ids,
                        source_text_block_ids=[title_block["text_block_id"]],
                        is_indexable=False,
                    )
                )
                block_index += 1

            facility_blocks = [
                block
                for block in text_blocks
                if "Tel." in block["text"] or "Tel " in block["text"]
            ]
            if facility_blocks:
                blocks.append(
                    _make_block(
                        block_id=f"block_{block_index:03d}",
                        page_number=page_number,
                        block_type="facility_block",
                        section_title="Facility",
                        text="\n".join(block["text"] for block in facility_blocks),
                        bboxes=[block["bbox"] for block in facility_blocks],
                        source_text_block_ids=[block["text_block_id"] for block in facility_blocks],
                        is_indexable=False,
                    )
                )
                block_index += 1

            for table in page_tables:
                if table.table_role == "patient_info_table":
                    blocks.append(
                        _make_block(
                            block_id=f"block_{block_index:03d}",
                            page_number=page_number,
                            block_type="patient_info_block",
                            section_title="Patient information",
                            text=_table_to_text(table),
                            bboxes=[table.bbox] if table.bbox else [],
                            source_table_ids=[table.table_id],
                            is_indexable=False,
                        )
                    )
                    block_index += 1
                elif table.table_role == "report_info_table":
                    blocks.append(
                        _make_block(
                            block_id=f"block_{block_index:03d}",
                            page_number=page_number,
                            block_type="report_info_block",
                            section_title="Report metadata",
                            text=_table_to_text(table),
                            bboxes=[table.bbox] if table.bbox else [],
                            source_table_ids=[table.table_id],
                            is_indexable=False,
                        )
                    )
                    block_index += 1
                elif table.table_role == "results_table":
                    blocks.append(
                        _make_block(
                            block_id=f"block_{block_index:03d}",
                            page_number=page_number,
                            block_type="results_table_block",
                            section_title="Selected results",
                            text=_table_to_text(table),
                            bboxes=[table.bbox] if table.bbox else [],
                            source_table_ids=[table.table_id],
                            is_indexable=True,
                        )
                    )
                    block_index += 1

            if not any(table.table_role == "patient_info_table" for table in page_tables) and not any(
                table.table_role == "report_info_table" for table in page_tables
            ):
                results_header = _find_text_block(text_blocks, "Resultats selectionnes") or _find_text_block(
                    text_blocks, "Analyse / Observation"
                )
                top_start = 150.0
                top_end = results_header["bbox"]["y0"] if results_header else page_height * 0.45
                fallback_blocks, block_index = _create_column_blocks_from_ocr(
                    page_number=page_number,
                    text_blocks=text_blocks,
                    start_y=top_start,
                    end_y=top_end,
                    page_width=page_width,
                    block_index=block_index,
                )
                blocks.extend(fallback_blocks)

            if not any(table.table_role == "results_table" for table in page_tables):
                fallback_results, block_index = _create_results_block_from_ocr(
                    page_number=page_number,
                    text_blocks=text_blocks,
                    title_candidates=["Resultats selectionnes", "Analyse / Observation"],
                    stop_candidates=["Validation medicale", "Interpretation biologique"],
                    block_index=block_index,
                )
                blocks.extend(fallback_results)

        interpretation_title = (
            _find_text_block(text_blocks, "Interpretation biologique")
            or _find_text_block(text_blocks, "Interpretation et validation")
            or _find_text_block(text_blocks, "Synthese radioclinique")
        )
        if interpretation_title:
            companion_title = _find_companion_title(text_blocks, interpretation_title)
            boundary_y = min(
                [block["bbox"]["y0"] for block in text_blocks if block["bbox"]["y0"] > interpretation_title["bbox"]["y0"] and block["max_font_size"] >= interpretation_title["max_font_size"] and block["is_bold"]]
                + [table.bbox["y0"] for table in page_tables if table.bbox and table.bbox["y0"] > interpretation_title["bbox"]["y0"]]
                + [page_height * 0.95]
            )
            section_blocks = _collect_text_blocks_between(
                text_blocks,
                interpretation_title["bbox"]["y0"],
                boundary_y,
                max_x=(companion_title["bbox"]["x0"] - 10) if companion_title else None,
            )
            blocks.append(
                _make_block(
                    block_id=f"block_{block_index:03d}",
                    page_number=page_number,
                    block_type="clinical_interpretation_block",
                    section_title=interpretation_title["text"],
                    text="\n".join(block["text"] for block in section_blocks),
                    bboxes=[block["bbox"] for block in section_blocks],
                    source_text_block_ids=[block["text_block_id"] for block in section_blocks],
                    is_indexable=True,
                )
            )
            block_index += 1

        summary_title = _find_text_block(text_blocks, "Resume du dossier")
        if summary_title:
            boundary_y = min(
                [block["bbox"]["y0"] for block in text_blocks if block["bbox"]["y0"] > summary_title["bbox"]["y0"] and block["max_font_size"] >= summary_title["max_font_size"] and block["is_bold"]]
                + [table.bbox["y0"] for table in page_tables if table.bbox and table.bbox["y0"] > summary_title["bbox"]["y0"]]
                + [page_height * 0.95]
            )
            section_blocks = _collect_text_blocks_between(text_blocks, summary_title["bbox"]["y0"], boundary_y)
            blocks.append(
                _make_block(
                    block_id=f"block_{block_index:03d}",
                    page_number=page_number,
                    block_type="summary_block",
                    section_title=summary_title["text"],
                    text="\n".join(block["text"] for block in section_blocks),
                    bboxes=[block["bbox"] for block in section_blocks],
                    source_text_block_ids=[block["text_block_id"] for block in section_blocks],
                    is_indexable=True,
                )
            )
            block_index += 1

        analytics_image = next(
            (
                image
                for image in page_images
                if image.image_type in {"clinical_chart", "medical_illustration"}
            ),
            None,
        )
        analytics_title = (
            _find_text_block(text_blocks, "Vue analytique synthetique")
            if analytics_image and analytics_image.image_type == "clinical_chart"
            else (_find_title_above_image(text_blocks, analytics_image.bbox) if analytics_image and analytics_image.bbox else None)
        )
        if analytics_title and analytics_image:
            caption_blocks = [analytics_title]
            caption_blocks.extend(
                _collect_text_blocks_between(
                    text_blocks,
                    analytics_image.bbox["y1"] if analytics_image.bbox else analytics_title["bbox"]["y1"],
                    min(page_height * 0.75, (analytics_image.bbox["y1"] + 60) if analytics_image.bbox else page_height * 0.75),
                    min_x=analytics_image.bbox["x0"] - 10 if analytics_image.bbox else None,
                )
            )
            blocks.append(
                _make_block(
                    block_id=f"block_{block_index:03d}",
                    page_number=page_number,
                    block_type="image_context_block",
                    section_title=analytics_title["text"],
                    text="\n".join([block["text"] for block in caption_blocks] + [analytics_image.context_text]),
                    bboxes=[block["bbox"] for block in caption_blocks] + ([analytics_image.bbox] if analytics_image.bbox else []),
                    source_image_ids=[analytics_image.image_id],
                    source_text_block_ids=[block["text_block_id"] for block in caption_blocks],
                    is_indexable=True,
                )
            )
            block_index += 1

        results_tables = [table for table in page_tables if table.table_role == "results_table"]
        if page_number != 1:
            for table in results_tables:
                blocks.append(
                    _make_block(
                        block_id=f"block_{block_index:03d}",
                        page_number=page_number,
                        block_type="results_table_block",
                        section_title="Selected results",
                        text=_table_to_text(table),
                        bboxes=[table.bbox] if table.bbox else [],
                        source_table_ids=[table.table_id],
                        is_indexable=True,
                    )
                )
                block_index += 1
            if not results_tables:
                fallback_results, block_index = _create_results_block_from_ocr(
                    page_number=page_number,
                    text_blocks=text_blocks,
                    title_candidates=["Analyse / Observation"],
                    stop_candidates=["Validation medicale", "Cachet du service"],
                    block_index=block_index,
                )
                blocks.extend(fallback_results)

        validation_title = _find_text_block(text_blocks, "Validation medicale")
        stamp_title = _find_text_block(text_blocks, "Cachet du service")
        validation_images = [image for image in page_images if image.role == "validation"]
        if validation_title or validation_images or stamp_title:
            validation_blocks = [
                block
                for block in text_blocks
                if block["bbox"]["y0"] >= (validation_title["bbox"]["y0"] if validation_title else page_height * 0.32)
                and block["bbox"]["y0"] < page_height * 0.7
            ]
            validation_text = "\n".join(block["text"] for block in validation_blocks)
            validation_bboxes = [block["bbox"] for block in validation_blocks]
            validation_image_ids = [image.image_id for image in validation_images]
            validation_bboxes.extend([image.bbox for image in validation_images if image.bbox])
            title = validation_title["text"] if validation_title else "Validation"
            blocks.append(
                _make_block(
                    block_id=f"block_{block_index:03d}",
                    page_number=page_number,
                    block_type="validation_block",
                    section_title=title,
                    text=validation_text,
                    bboxes=validation_bboxes,
                    source_image_ids=validation_image_ids,
                    source_text_block_ids=[block["text_block_id"] for block in validation_blocks],
                    is_indexable=False,
                )
            )
            block_index += 1

        for footer in footer_blocks:
            blocks.append(
                _make_block(
                    block_id=f"block_{block_index:03d}",
                    page_number=page_number,
                    block_type="footer_block",
                    section_title="Footer",
                    text=footer["text"],
                    bboxes=[footer["bbox"]],
                    source_text_block_ids=[footer["text_block_id"]],
                    is_indexable=False,
                )
            )
            block_index += 1

    return blocks
