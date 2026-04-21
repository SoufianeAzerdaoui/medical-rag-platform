from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import fitz
from PIL import Image, ImageOps

from schemas import OcrAsset
from utils import bbox_from_sequence, clean_text, ensure_dir, optional_import, page_name, write_json


pytesseract = optional_import("pytesseract")


def _pixmap_to_image(page: fitz.Page, dpi: int = 220) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def _prepare_image(image: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(image)
    # Slight contrast boost helps scanned synthetic pages.
    return ImageOps.autocontrast(gray)


def _image_to_ocr_data(image: Image.Image, scale_x: float, scale_y: float) -> tuple[str, list[dict], list[dict]]:
    if pytesseract is None:
        return "", [], []

    output = getattr(pytesseract, "Output", None)
    config = "--oem 3 --psm 6"
    text = clean_text(pytesseract.image_to_string(image, config=config))
    if output is None:
        return text, [], []

    data = pytesseract.image_to_data(image, output_type=output.DICT, config=config)
    words: list[dict] = []
    grouped: dict[tuple[int, int, int, int], list[dict]] = defaultdict(list)
    total = len(data.get("text", []))
    for index in range(total):
        raw_text = str(data["text"][index] or "").strip()
        if not raw_text:
            continue
        try:
            conf = float(data["conf"][index])
        except Exception:
            conf = -1.0
        if conf < 0:
            continue

        left = int(data["left"][index])
        top = int(data["top"][index])
        width = int(data["width"][index])
        height = int(data["height"][index])
        word_bbox = bbox_from_sequence(
            (
                left * scale_x,
                top * scale_y,
                (left + width) * scale_x,
                (top + height) * scale_y,
            )
        )
        word = {
            "text": raw_text,
            "conf": round(conf, 2),
            "bbox": word_bbox,
            "block_num": int(data["block_num"][index]),
            "par_num": int(data["par_num"][index]),
            "line_num": int(data["line_num"][index]),
            "word_num": int(data["word_num"][index]),
        }
        words.append(word)
        grouped[(int(data["page_num"][index]), int(data["block_num"][index]), int(data["par_num"][index]), int(data["line_num"][index]))].append(word)

    blocks: list[dict] = []
    for block_index, (_, line_words) in enumerate(sorted(grouped.items()), start=1):
        line_words.sort(key=lambda item: item["bbox"]["x0"])
        text_line = " ".join(word["text"] for word in line_words).strip()
        if not text_line:
            continue
        bbox = {
            "x0": min(word["bbox"]["x0"] for word in line_words),
            "y0": min(word["bbox"]["y0"] for word in line_words),
            "x1": max(word["bbox"]["x1"] for word in line_words),
            "y1": max(word["bbox"]["y1"] for word in line_words),
        }
        blocks.append(
            {
                "text_block_id": f"ocr_line_{block_index:03d}",
                "text": clean_text(text_line),
                "bbox": bbox,
                "max_font_size": round(max(word["bbox"]["y1"] - word["bbox"]["y0"] for word in line_words), 2),
                "is_bold": False,
                "fonts": ["ocr"],
            }
        )

    blocks.sort(key=lambda item: (item["bbox"]["y0"], item["bbox"]["x0"]))
    return text, blocks, words


def extract_ocr(
    pdf_path: str | Path,
    output_dir: str | Path,
    pages_to_ocr: list[int] | None = None,
    dpi: int = 220,
) -> dict[int, OcrAsset]:
    source = Path(pdf_path).expanduser().resolve()
    ocr_dir = ensure_dir(Path(output_dir) / "ocr")
    wanted_pages = set(pages_to_ocr or [])
    results: dict[int, OcrAsset] = {}

    if pytesseract is None:
        for page_number in wanted_pages:
            asset = OcrAsset(
                page_number=page_number,
                text="",
                text_path=None,
                image_path=None,
                used=False,
                engine=None,
                blocks=[],
                words=[],
            )
            write_json(ocr_dir / f"{page_name(page_number)}.json", asset.to_dict())
            results[page_number] = asset
        return results

    with fitz.open(source) as doc:
        for page in doc:
            page_number = page.number + 1
            if wanted_pages and page_number not in wanted_pages:
                continue

            raw_image = _pixmap_to_image(page, dpi=dpi)
            image = _prepare_image(raw_image)
            image_path = ocr_dir / f"{page_name(page_number)}.png"
            text_path = ocr_dir / f"{page_name(page_number)}.txt"
            image.save(image_path)

            scale_x = float(page.rect.width) / max(image.width, 1)
            scale_y = float(page.rect.height) / max(image.height, 1)
            text, blocks, words = _image_to_ocr_data(image, scale_x, scale_y)
            text_path.write_text(text, encoding="utf-8")

            asset = OcrAsset(
                page_number=page_number,
                text=text,
                text_path=str(text_path),
                image_path=str(image_path),
                used=bool(text),
                engine="pytesseract",
                blocks=blocks,
                words=words,
            )
            write_json(ocr_dir / f"{page_name(page_number)}.json", asset.to_dict())
            results[page_number] = asset

    return results
