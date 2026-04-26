from __future__ import annotations

import re
from collections import defaultdict

from schemas import DocumentBlock
from utils import bbox_union, normalize_inline_text, normalize_label


OCR_NOISE_PATTERNS = (
    r"=~\s*\w+",
    r"\b\d+\s*:\s*[A-Z]\s+\w+\b",
    r"(?:[~=_-]\s*){2,}",
    r"\b[a-zA-Z]\s*[-_]{2,}\w*\b",
)

OCR_INDEX_NOISE_PATTERNS = (
    r"\ba\s+on\b",
    r"\bNormal\s+i\b",
    r"\bMiscarria\s+je\s+in\s+first\b",
    r"\ba\s*,?\s*trimester\b",
    r"\bols\s+der\b",
    r"\bwr\b",
    r"\bfr\b",
    r"\bI\s+Te\b",
    r"\b\d+\s+so\s+R\b",
    r"\bve\s+Social\b",
)


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
    confidence: str = "medium",
    confidence_score: float = 0.5,
    structured_fields: dict | None = None,
) -> DocumentBlock:
    normalized_text = _normalize_block_text_for_indexing(
        text,
        apply_ocr_cleanup=confidence_score < 0.75 or confidence == "low",
    )
    index_text = _build_index_text(
        normalized_text,
        apply_ocr_cleanup=confidence_score < 0.75 or confidence == "low",
    )
    return DocumentBlock(
        block_id=block_id,
        page_number=page_number,
        block_type=block_type,
        section_title=section_title,
        text=normalize_inline_text(text),
        bbox=bbox_union([bbox for bbox in bboxes if bbox]),
        normalized_text=normalized_text,
        index_text=index_text,
        confidence=confidence,
        confidence_score=round(confidence_score, 3),
        structured_fields=structured_fields or {},
        source_table_ids=source_table_ids or [],
        source_image_ids=source_image_ids or [],
        source_text_block_ids=source_text_block_ids or [],
        is_indexable=is_indexable,
    )


def _normalize_block_text_for_indexing(text: str, *, apply_ocr_cleanup: bool) -> str:
    normalized = normalize_inline_text(text)
    if not apply_ocr_cleanup:
        return normalized

    for pattern in OCR_NOISE_PATTERNS:
        normalized = re.sub(pattern, " ", normalized)
    normalized = re.sub(r"\s{2,}", " ", normalized)
    return normalized.strip()


def _build_index_text(text: str, *, apply_ocr_cleanup: bool) -> str:
    index_text = normalize_inline_text(text)
    if not apply_ocr_cleanup:
        return index_text

    for pattern in (*OCR_NOISE_PATTERNS, *OCR_INDEX_NOISE_PATTERNS):
        index_text = re.sub(pattern, " ", index_text, flags=re.IGNORECASE)

    index_text = re.sub(r"[~=_]+", " ", index_text)
    index_text = re.sub(r"[^\w\s.,;:()/\[\]'%-]", " ", index_text)
    index_text = re.sub(r"\s+[a-zA-Z]\s+[a-zA-Z]\s+", " ", index_text)
    index_text = re.sub(r"\b[a-zA-Z]\s*,\s*", " ", index_text)
    index_text = re.sub(r"\s{2,}", " ", index_text).strip()

    sentences = re.split(r"(?<=[.;:])\s+", index_text)
    kept: list[str] = []
    for sentence in sentences:
        cleaned = sentence.strip(" -;,")
        if not cleaned:
            continue
        symbol_count = len(re.findall(r"[~=_{}|]", cleaned))
        word_count = len(re.findall(r"[A-Za-zÀ-ÿ]{2,}", cleaned))
        if symbol_count:
            continue
        if word_count < 4 and not re.search(r"\b(patient|rapport|validation|image|resultat|antecedents)\b", cleaned, flags=re.IGNORECASE):
            continue
        if re.search(r"\b(?:je|wr|ols|der|fr|trimester)\b", cleaned, flags=re.IGNORECASE):
            continue
        kept.append(cleaned)

    return " ".join(kept) if kept else index_text


def _score_to_confidence(score: float) -> str:
    if score >= 0.85:
        return "high"
    if score >= 0.6:
        return "medium"
    return "low"


def _compute_block_confidence(
    *,
    page_text_source: str,
    block_type: str,
    has_table_source: bool = False,
    has_image_source: bool = False,
    text_length: int = 0,
    source_count: int = 0,
) -> tuple[str, float]:
    if block_type == "footer_block":
        return ("low", 0.2)

    score = 0.45
    if page_text_source == "native":
        score += 0.35
    elif page_text_source == "hybrid":
        score += 0.2
    elif page_text_source == "ocr":
        score += 0.05

    if has_table_source:
        score += 0.2
    if has_image_source:
        score += 0.1
    if source_count >= 2:
        score += 0.05
    if text_length > 80:
        score += 0.05
    if block_type in {"clinical_interpretation_block", "summary_block"} and page_text_source == "ocr":
        score -= 0.15
    if block_type == "validation_block" and not (has_image_source or source_count):
        score -= 0.1

    score = max(0.0, min(1.0, score))
    return (_score_to_confidence(score), score)


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
    page_text_source: str,
) -> tuple[list[DocumentBlock], int]:
    mid_x = page_width * 0.5
    selected = _blocks_below(text_blocks, start_y, end_y)
    left_blocks = [block for block in selected if block["bbox"]["x0"] < mid_x]
    right_blocks = [block for block in selected if block["bbox"]["x0"] >= mid_x]
    created: list[DocumentBlock] = []

    if left_blocks:
        left_text = "\n".join(block["text"] for block in left_blocks)
        confidence, confidence_score = _compute_block_confidence(
            page_text_source=page_text_source,
            block_type="patient_info_block",
            text_length=len(normalize_inline_text(left_text)),
            source_count=len(left_blocks),
        )
        created.append(
            _make_block(
                block_id=f"block_{block_index:03d}",
                page_number=page_number,
                block_type="patient_info_block",
                section_title="Patient information",
                text=left_text,
                bboxes=[block["bbox"] for block in left_blocks],
                source_text_block_ids=[block["text_block_id"] for block in left_blocks],
                is_indexable=False,
                confidence=confidence,
                confidence_score=confidence_score,
            )
        )
        block_index += 1

    if right_blocks:
        right_text = "\n".join(block["text"] for block in right_blocks)
        confidence, confidence_score = _compute_block_confidence(
            page_text_source=page_text_source,
            block_type="report_info_block",
            text_length=len(normalize_inline_text(right_text)),
            source_count=len(right_blocks),
        )
        created.append(
            _make_block(
                block_id=f"block_{block_index:03d}",
                page_number=page_number,
                block_type="report_info_block",
                section_title="Report metadata",
                text=right_text,
                bboxes=[block["bbox"] for block in right_blocks],
                source_text_block_ids=[block["text_block_id"] for block in right_blocks],
                is_indexable=False,
                confidence=confidence,
                confidence_score=confidence_score,
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
    page_text_source: str,
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

    results_text = "\n".join(item["text"] for item in candidate_blocks)
    confidence, confidence_score = _compute_block_confidence(
        page_text_source=page_text_source,
        block_type="results_table_block",
        text_length=len(normalize_inline_text(results_text)),
        source_count=len(candidate_blocks),
    )
    block = _make_block(
        block_id=f"block_{block_index:03d}",
        page_number=page_number,
        block_type="results_table_block",
        section_title="Selected results",
        text=results_text,
        bboxes=[item["bbox"] for item in candidate_blocks],
        source_text_block_ids=[item["text_block_id"] for item in candidate_blocks],
        is_indexable=True,
        confidence=confidence,
        confidence_score=confidence_score,
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


def _next_boundary_y(
    *,
    text_blocks: list[dict],
    page_tables: list,
    start_block: dict,
    page_height: float,
    companion_title: dict | None = None,
    explicit_stops: list[dict] | None = None,
) -> float:
    candidates = [
        block["bbox"]["y0"]
        for block in text_blocks
        if block["bbox"]["y0"] > start_block["bbox"]["y0"]
        and block["max_font_size"] >= start_block["max_font_size"]
        and block["is_bold"]
        and (companion_title is None or block["text_block_id"] != companion_title["text_block_id"])
    ]
    candidates.extend(
        table.bbox["y0"] for table in page_tables if table.bbox and table.bbox["y0"] > start_block["bbox"]["y0"]
    )
    if explicit_stops:
        candidates.extend(stop["bbox"]["y0"] for stop in explicit_stops if stop and stop["bbox"]["y0"] > start_block["bbox"]["y0"])
    candidates.append(page_height * 0.95)
    return min(candidates)


def _filter_content_blocks(blocks: list[dict]) -> list[dict]:
    return [
        block
        for block in blocks
        if block.get("text") and not normalize_label(block["text"]).startswith("document_synthetique")
    ]


def _dedupe_text_parts(parts: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: list[str] = []
    for part in parts:
        fragments = [normalize_inline_text(fragment) for fragment in str(part).replace("|", "\n").splitlines()]
        for text in fragments:
            if not text:
                continue
            key = normalize_label(text)
            if not key:
                continue
            if any(key == existing or key in existing or existing in key for existing in seen):
                continue
            seen.append(key)
            deduped.append(text)
    return deduped


def _select_footer_blocks(text_blocks: list[dict]) -> list[dict]:
    candidates = [
        block
        for block in text_blocks
        if normalize_label(block["text"]).startswith("document_synthetique")
    ]
    if not candidates:
        return []
    return [max(candidates, key=lambda block: block["bbox"]["y0"])]


def _asset_id(asset) -> str:
    return getattr(asset, "image_id", getattr(asset, "visual_id", ""))


def _asset_type(asset) -> str:
    return getattr(asset, "image_type", getattr(asset, "visual_type", "unknown"))


def _build_validation_structured_fields(
    *,
    title: str,
    validation_blocks: list[dict],
    validation_assets: list,
    page_width: float,
) -> dict[str, object]:
    left_side_blocks = [
        block
        for block in validation_blocks
        if block["bbox"]["x0"] < page_width * 0.55 and normalize_label(block["text"]) not in {"validation_medicale"}
    ]
    validated_by = None
    specialty = None
    for block in left_side_blocks:
        text = normalize_inline_text(block["text"])
        label = normalize_label(text)
        if not text or "cachet_du_service" in label or "validation_medicale" in label:
            continue
        if validated_by is None:
            validated_by = text
            continue
        if specialty is None:
            specialty = text
            break

    signature_present = any(_asset_type(asset) == "signature" for asset in validation_assets)
    stamp_present = any(_asset_type(asset) == "stamp_or_seal" for asset in validation_assets)
    return {
        "validation_title": title,
        "validated_by": validated_by,
        "specialty": specialty,
        "stamp_present": stamp_present,
        "signature_present": signature_present,
    }


def build_blocks(page_text_data: list[dict], tables: list, images: list, ocr_visuals: list | None = None) -> list[DocumentBlock]:
    blocks: list[DocumentBlock] = []
    block_index = 1
    tables_by_page: dict[int, list] = defaultdict(list)
    images_by_page: dict[int, list] = defaultdict(list)
    ocr_visuals_by_page: dict[int, list] = defaultdict(list)

    for table in tables:
        tables_by_page[table.page_number].append(table)
    for image in images:
        images_by_page[image.page_number].append(image)
    for visual in ocr_visuals or []:
        ocr_visuals_by_page[visual.page_number].append(visual)

    for page in page_text_data:
        page_number = page["page_number"]
        page_width = page["width"]
        page_height = page["height"]
        page_text_source = page.get("text_source", "native")
        text_blocks = page.get("text_blocks", [])
        page_tables = tables_by_page.get(page_number, [])
        page_images = images_by_page.get(page_number, [])
        page_ocr_visuals = ocr_visuals_by_page.get(page_number, [])

        def scored_kwargs(
            *,
            block_type: str,
            text: str,
            has_table_source: bool = False,
            has_image_source: bool = False,
            source_count: int = 0,
        ) -> dict:
            confidence, confidence_score = _compute_block_confidence(
                page_text_source=page_text_source,
                block_type=block_type,
                has_table_source=has_table_source,
                has_image_source=has_image_source,
                text_length=len(normalize_inline_text(text)),
                source_count=source_count,
            )
            return {"confidence": confidence, "confidence_score": confidence_score}

        footer_blocks = _select_footer_blocks(text_blocks)
        footer_top = footer_blocks[0]["bbox"]["y0"] if footer_blocks else page_height * 0.95

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
                        **scored_kwargs(
                            block_type="document_header",
                            text=title_block["text"],
                            has_image_source=bool(image_ids),
                            source_count=1 + len(image_ids),
                        ),
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
                        **scored_kwargs(
                            block_type="facility_block",
                            text="\n".join(block["text"] for block in facility_blocks),
                            source_count=len(facility_blocks),
                        ),
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
                            **scored_kwargs(
                                block_type="patient_info_block",
                                text=_table_to_text(table),
                                has_table_source=True,
                                source_count=1,
                            ),
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
                            **scored_kwargs(
                                block_type="report_info_block",
                                text=_table_to_text(table),
                                has_table_source=True,
                                source_count=1,
                            ),
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
                            **scored_kwargs(
                                block_type="results_table_block",
                                text=_table_to_text(table),
                                has_table_source=True,
                                source_count=1,
                            ),
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
                    page_text_source=page_text_source,
                )
                blocks.extend(fallback_blocks)

            if not any(table.table_role == "results_table" for table in page_tables):
                fallback_results, block_index = _create_results_block_from_ocr(
                    page_number=page_number,
                    text_blocks=text_blocks,
                    title_candidates=["Resultats selectionnes", "Analyse / Observation"],
                    stop_candidates=["Validation medicale", "Interpretation biologique"],
                    block_index=block_index,
                    page_text_source=page_text_source,
                )
                blocks.extend(fallback_results)

        interpretation_title = (
            _find_text_block(text_blocks, "Interpretation biologique")
            or _find_text_block(text_blocks, "Interpretation et validation")
            or _find_text_block(text_blocks, "Synthese radioclinique")
        )
        if interpretation_title:
            companion_title = _find_companion_title(text_blocks, interpretation_title)
            summary_title = _find_text_block(text_blocks, "Resume du dossier")
            validation_title = _find_text_block(text_blocks, "Validation medicale")
            stamp_title = _find_text_block(text_blocks, "Cachet du service")
            boundary_y = _next_boundary_y(
                text_blocks=text_blocks,
                page_tables=page_tables,
                start_block=interpretation_title,
                companion_title=companion_title,
                page_height=page_height,
                explicit_stops=[summary_title, validation_title, stamp_title, *footer_blocks],
            )
            section_blocks = _collect_text_blocks_between(
                text_blocks,
                interpretation_title["bbox"]["y0"],
                boundary_y,
                max_x=(companion_title["bbox"]["x0"] - 10) if companion_title else None,
            )
            section_blocks = _filter_content_blocks(section_blocks)
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
                    **scored_kwargs(
                        block_type="clinical_interpretation_block",
                        text="\n".join(block["text"] for block in section_blocks),
                        source_count=len(section_blocks),
                    ),
                )
            )
            block_index += 1

        summary_title = _find_text_block(text_blocks, "Resume du dossier")
        if summary_title:
            validation_title = _find_text_block(text_blocks, "Validation medicale")
            stamp_title = _find_text_block(text_blocks, "Cachet du service")
            boundary_y = _next_boundary_y(
                text_blocks=text_blocks,
                page_tables=page_tables,
                start_block=summary_title,
                page_height=page_height,
                explicit_stops=[validation_title, stamp_title, *footer_blocks],
            )
            section_blocks = _collect_text_blocks_between(text_blocks, summary_title["bbox"]["y0"], boundary_y)
            section_blocks = _filter_content_blocks(section_blocks)
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
                    **scored_kwargs(
                        block_type="summary_block",
                        text="\n".join(block["text"] for block in section_blocks),
                        source_count=len(section_blocks),
                    ),
                )
            )
            block_index += 1

        analytics_image = next(
            (
                asset
                for asset in [*page_images, *page_ocr_visuals]
                if _asset_type(asset) in {"clinical_chart", "medical_illustration"}
            ),
            None,
        )
        analytics_title = (
            _find_text_block(text_blocks, "Vue analytique synthetique")
            if analytics_image and _asset_type(analytics_image) == "clinical_chart"
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
            context_parts = _dedupe_text_parts(
                [block["text"] for block in caption_blocks] + [getattr(analytics_image, "context_text", "")]
            )
            blocks.append(
                _make_block(
                    block_id=f"block_{block_index:03d}",
                    page_number=page_number,
                    block_type="image_context_block",
                    section_title=analytics_title["text"],
                    text="\n".join(context_parts),
                    bboxes=[block["bbox"] for block in caption_blocks] + ([analytics_image.bbox] if analytics_image.bbox else []),
                    source_image_ids=[_asset_id(analytics_image)],
                    source_text_block_ids=[block["text_block_id"] for block in caption_blocks],
                    is_indexable=True,
                    **scored_kwargs(
                        block_type="image_context_block",
                        text="\n".join(context_parts),
                        has_image_source=True,
                        source_count=len(caption_blocks) + 1,
                    ),
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
                        **scored_kwargs(
                            block_type="results_table_block",
                            text=_table_to_text(table),
                            has_table_source=True,
                            source_count=1,
                        ),
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
                    page_text_source=page_text_source,
                )
                blocks.extend(fallback_results)

        validation_title = _find_text_block(text_blocks, "Validation medicale")
        stamp_title = _find_text_block(text_blocks, "Cachet du service")
        validation_assets = [
            asset
            for asset in [*page_images, *page_ocr_visuals]
            if getattr(asset, "role", "") == "validation" or _asset_type(asset) in {"signature", "stamp_or_seal"}
        ]
        if validation_title or validation_assets or stamp_title:
            validation_blocks = [
                block
                for block in text_blocks
                if block["bbox"]["y0"] >= (validation_title["bbox"]["y0"] if validation_title else page_height * 0.32)
                and block["bbox"]["y0"] < min(page_height * 0.7, footer_top)
            ]
            validation_blocks = _filter_content_blocks(validation_blocks)
            validation_parts = _dedupe_text_parts([block["text"] for block in validation_blocks])
            validation_text = "\n".join(validation_parts)
            validation_bboxes = [block["bbox"] for block in validation_blocks]
            validation_image_ids = [_asset_id(asset) for asset in validation_assets]
            validation_bboxes.extend([asset.bbox for asset in validation_assets if asset.bbox])
            title = validation_title["text"] if validation_title else "Validation"
            structured_fields = _build_validation_structured_fields(
                title=title,
                validation_blocks=validation_blocks,
                validation_assets=validation_assets,
                page_width=page_width,
            )
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
                    structured_fields=structured_fields,
                    is_indexable=False,
                    **scored_kwargs(
                        block_type="validation_block",
                        text=validation_text,
                        has_image_source=bool(validation_image_ids),
                        source_count=len(validation_blocks) + len(validation_image_ids),
                    ),
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
                    **scored_kwargs(
                        block_type="footer_block",
                        text=footer["text"],
                        source_count=1,
                    ),
                )
            )
            block_index += 1

    return blocks
