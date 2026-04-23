from __future__ import annotations

from pathlib import Path

import fitz

from schemas import ImageAsset, OcrVisualAsset
from utils import (
    bbox_area,
    bbox_from_sequence,
    bbox_horizontal_overlap_ratio,
    bbox_vertical_distance,
    ensure_dir,
    normalize_inline_text,
    normalize_label,
)


def _largest_rect(rects: list[fitz.Rect]) -> fitz.Rect | None:
    if not rects:
        return None
    return max(rects, key=lambda rect: max(rect.width, 0) * max(rect.height, 0))


def _get_context_text(image_bbox: dict[str, float], text_blocks: list[dict]) -> str:
    candidates: list[tuple[float, str]] = []
    for block in text_blocks:
        if normalize_label(block["text"]).startswith("document_synthetique"):
            continue
        block_bbox = block["bbox"]
        overlap = bbox_horizontal_overlap_ratio(image_bbox, block_bbox)
        vertical_distance = bbox_vertical_distance(image_bbox, block_bbox)
        is_clearly_left = block_bbox["x1"] < image_bbox["x0"] - 15
        is_clearly_right = block_bbox["x0"] > image_bbox["x1"] + 15
        if overlap < 0.15 and vertical_distance > 25:
            continue
        if overlap < 0.08 and (is_clearly_left or is_clearly_right):
            continue
        score = vertical_distance - (overlap * 20)
        candidates.append((score, block["text"]))
    candidates.sort(key=lambda item: item[0])
    return " | ".join(text for _, text in candidates[:3])


def _classify_image(
    *,
    page_number: int,
    page_width: float,
    page_height: float,
    image_bbox: dict[str, float],
    context_text: str,
) -> tuple[str, str, bool]:
    context = normalize_label(context_text)
    width_ratio = (image_bbox["x1"] - image_bbox["x0"]) / max(page_width, 1.0)
    height_ratio = (image_bbox["y1"] - image_bbox["y0"]) / max(page_height, 1.0)
    top_ratio = image_bbox["y0"] / max(page_height, 1.0)
    left_ratio = image_bbox["x0"] / max(page_width, 1.0)

    if "cachet_du_service" in context or "valide" in context or "seal" in context:
        return "stamp_or_seal", "validation", False
    if "validation_medicale" in context or "validation" in context:
        return "signature", "validation", False
    if "echographie" in context or "radiographie" in context or "image_medicale" in context or "illustration_d_imagerie" in context:
        return "medical_illustration", "clinical_visual", True
    if "vue_analytique" in context or "visualisation_analytique" in context or "mixed_texte_tableau_image" in context:
        return "clinical_chart", "analytics", True
    if top_ratio < 0.18 and width_ratio > 0.28:
        return "logo_or_branding", "branding", False
    if page_number == 2 and width_ratio > 0.25 and height_ratio > 0.08:
        return "clinical_chart", "analytics", True
    if page_number == 3 and left_ratio < 0.45 and top_ratio > 0.35:
        return "signature", "validation", False
    if page_number == 3 and left_ratio > 0.55 and top_ratio > 0.35:
        return "stamp_or_seal", "validation", False
    return "unknown", "unknown", False


def _find_text_block(text_blocks: list[dict], patterns: list[str]) -> dict | None:
    for block in text_blocks:
        label = normalize_label(block["text"])
        if any(pattern in label for pattern in patterns):
            return block
    return None


def _find_last_text_block(text_blocks: list[dict], patterns: list[str]) -> dict | None:
    matches = []
    for block in text_blocks:
        label = normalize_label(block["text"])
        if any(pattern in label for pattern in patterns):
            matches.append(block)
    if not matches:
        return None
    return max(matches, key=lambda block: block["bbox"]["y0"])


def _clamp_bbox(
    bbox: dict[str, float],
    *,
    page_width: float,
    page_height: float,
    min_size: float = 24.0,
) -> dict[str, float] | None:
    x0 = max(0.0, min(float(bbox["x0"]), page_width))
    y0 = max(0.0, min(float(bbox["y0"]), page_height))
    x1 = max(0.0, min(float(bbox["x1"]), page_width))
    y1 = max(0.0, min(float(bbox["y1"]), page_height))
    if x1 - x0 < min_size or y1 - y0 < min_size:
        return None
    return {"x0": round(x0, 2), "y0": round(y0, 2), "x1": round(x1, 2), "y1": round(y1, 2)}


def _save_native_image_asset(
    *,
    page: fitz.Page,
    page_number: int,
    image_index: int,
    image_type: str,
    role: str,
    is_indexable: bool,
    bbox: dict[str, float],
    context_text: str,
    images_dir: Path,
    extracted: dict,
    xref: int,
) -> ImageAsset:
    ext = extracted.get("ext", "bin")
    file_path = images_dir / f"page_{page_number:03d}_img_{image_index:02d}.{ext}"
    file_path.write_bytes(extracted["image"])
    page_area = max(float(page.rect.width) * float(page.rect.height), 1.0)
    return ImageAsset(
        image_id=f"image_p{page_number:03d}_{image_index:02d}",
        page_number=page_number,
        file_path=str(file_path),
        width=int(extracted.get("width", 0)),
        height=int(extracted.get("height", 0)),
        ext=ext,
        bbox=bbox,
        xref=xref,
        page_coverage=round(bbox_area(bbox) / page_area, 4),
        image_type=image_type,
        role=role,
        is_indexable=is_indexable,
        context_text=normalize_inline_text(context_text),
    )


def _save_scanned_visual_crop(
    *,
    page: fitz.Page,
    page_number: int,
    crop_index: int,
    bbox: dict[str, float],
    visuals_dir: Path,
    label: str,
) -> OcrVisualAsset | None:
    clipped = _clamp_bbox(
        bbox,
        page_width=float(page.rect.width),
        page_height=float(page.rect.height),
    )
    if clipped is None:
        return None
    clip = fitz.Rect(clipped["x0"], clipped["y0"], clipped["x1"], clipped["y1"])
    pix = page.get_pixmap(dpi=220, clip=clip, alpha=False)
    if pix.width <= 8 or pix.height <= 8:
        return None
    file_path = visuals_dir / f"page_{page_number:03d}_{label}_{crop_index:02d}.png"
    pix.save(str(file_path))
    visual_type_map = {
        "branding": ("logo_or_branding", "branding", False),
        "chart": ("clinical_chart", "analytics", True),
        "visual": ("medical_illustration", "clinical_visual", True),
        "signature": ("signature", "validation", False),
        "stamp": ("stamp_or_seal", "validation", False),
    }
    visual_type, role, is_indexable = visual_type_map.get(label, ("unknown", "unknown", False))
    return OcrVisualAsset(
        visual_id=f"ocr_visual_p{page_number:03d}_{crop_index:02d}",
        page_number=page_number,
        file_path=str(file_path),
        width=pix.width,
        height=pix.height,
        ext="png",
        bbox=clipped,
        visual_type=visual_type,
        role=role,
        is_indexable=is_indexable,
        context_text=label,
    )


def _export_scanned_page_visuals(
    *,
    page: fitz.Page,
    page_number: int,
    page_width: float,
    page_height: float,
    text_blocks: list[dict],
    output_dir: Path,
) -> list[OcrVisualAsset]:
    visuals_dir = ensure_dir(output_dir / "ocr" / "visuals")
    footer = _find_last_text_block(text_blocks, ["document_synthetique"])
    footer_top = footer["bbox"]["y0"] if footer else page_height * 0.94
    next_index = 1
    assets: list[OcrVisualAsset] = []

    def export_crop(bbox: dict[str, float], *, label: str) -> None:
        nonlocal next_index
        asset = _save_scanned_visual_crop(
            page=page,
            page_number=page_number,
            crop_index=next_index,
            bbox=bbox,
            visuals_dir=visuals_dir,
            label=label,
        )
        if asset is not None:
            assets.append(asset)
        next_index += 1

    header_block = _find_text_block(text_blocks, ["dossier_patient_synthetique_multimodal", "communicative_health_care_associates"])
    title_block = _find_text_block(text_blocks, ["compte_rendu_d_analyses", "compte_rendu_d_imagerie"])
    if page_number == 1 and title_block:
        top_y = header_block["bbox"]["y0"] - page_height * 0.02 if header_block else page_height * 0.04
        export_crop(
            {
                "x0": page_width * 0.08,
                "y0": top_y,
                "x1": page_width * 0.92,
                "y1": max(page_height * 0.17, title_block["bbox"]["y0"] - page_height * 0.015),
            },
            label="branding",
        )

    chart_caption = _find_text_block(text_blocks, ["bloc_image_rasterise", "visualisation_analytique", "image_medicaie_synthetique", "image_medicale_synthetique"])
    chart_title = _find_text_block(text_blocks, ["vue_analytique_synthetique", "synthese_radioclinique", "radiographie_thoracique_synthetique", "echographie_synthetique"])
    if chart_caption or chart_title:
        has_chart_title = _find_text_block(text_blocks, ["vue_analytique_synthetique", "visualisation_analytique"]) is not None
        if chart_caption:
            crop_y0 = max(
                page_height * 0.10,
                chart_caption["bbox"]["y0"] - page_height * 0.20,
                (chart_title["bbox"]["y1"] + page_height * 0.01) if chart_title else 0.0,
            )
            crop_y1 = chart_caption["bbox"]["y0"] - page_height * 0.008
        else:
            crop_y0 = max(page_height * 0.18, chart_title["bbox"]["y0"] + page_height * 0.04)
            crop_y1 = min(page_height * 0.55, crop_y0 + page_height * 0.24)
        export_crop(
            {
                "x0": page_width * 0.5,
                "y0": crop_y0,
                "x1": page_width * 0.93,
                "y1": crop_y1,
            },
            label="chart" if has_chart_title else "visual",
        )

    validation_block = _find_text_block(text_blocks, ["validation_medicale"])
    stamp_block = _find_text_block(text_blocks, ["cachet_du_service"])
    validation_anchor = validation_block or stamp_block
    if validation_anchor:
        y0 = validation_anchor["bbox"]["y1"] + page_height * 0.02
        y1 = footer_top - page_height * 0.015
        export_crop(
            {
                "x0": page_width * 0.08,
                "y0": y0,
                "x1": page_width * 0.42,
                "y1": y1,
            },
            label="signature",
        )
        export_crop(
            {
                "x0": page_width * 0.58,
                "y0": y0,
                "x1": page_width * 0.86,
                "y1": y1,
            },
            label="stamp",
        )
    return assets


def extract_images(
    pdf_path: str | Path,
    output_dir: str | Path,
    page_text_data: list[dict],
    full_page_threshold: float = 0.90,
) -> tuple[list[ImageAsset], list[OcrVisualAsset]]:
    source = Path(pdf_path).expanduser().resolve()
    output_root = Path(output_dir)
    images_dir = ensure_dir(output_root / "images")
    page_map = {page["page_number"]: page for page in page_text_data}
    assets: list[ImageAsset] = []
    ocr_visual_assets: list[OcrVisualAsset] = []
    seen_xrefs: set[tuple[int, int]] = set()

    with fitz.open(source) as doc:
        for page in doc:
            page_number = page.number + 1
            page_width = float(page.rect.width)
            page_height = float(page.rect.height)
            page_area = max(page_width * page_height, 1.0)
            text_blocks = page_map.get(page_number, {}).get("text_blocks", [])
            page_image_index = 1
            has_full_page_raster = False

            for image_info in page.get_images(full=True):
                xref = image_info[0]
                key = (page_number, xref)
                if key in seen_xrefs:
                    continue
                seen_xrefs.add(key)

                rect = _largest_rect(page.get_image_rects(xref, transform=False))
                if rect is None:
                    continue
                bbox = bbox_from_sequence(rect)
                coverage = bbox_area(bbox) / page_area
                if coverage >= full_page_threshold:
                    has_full_page_raster = True
                    continue

                extracted = doc.extract_image(xref)
                context_text = _get_context_text(bbox, text_blocks)
                image_type, role, is_indexable = _classify_image(
                    page_number=page_number,
                    page_width=page_width,
                    page_height=page_height,
                    image_bbox=bbox,
                    context_text=context_text,
                )

                assets.append(
                    _save_native_image_asset(
                        page=page,
                        page_number=page_number,
                        image_index=page_image_index,
                        image_type=image_type,
                        role=role,
                        is_indexable=is_indexable,
                        bbox=bbox,
                        context_text=context_text,
                        images_dir=images_dir,
                        extracted=extracted,
                        xref=xref,
                    )
                )
                page_image_index += 1

            if page_image_index == 1 and text_blocks and (
                has_full_page_raster or page_map.get(page_number, {}).get("ocr_used")
            ):
                ocr_visual_assets.extend(
                    _export_scanned_page_visuals(
                    page=page,
                    page_number=page_number,
                    page_width=page_width,
                    page_height=page_height,
                    text_blocks=text_blocks,
                    output_dir=output_root,
                )
                )

    return assets, ocr_visual_assets
