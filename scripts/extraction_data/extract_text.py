from __future__ import annotations

from pathlib import Path

import fitz

from utils import bbox_from_sequence, clean_text, strip_page_boilerplate


def _extract_text_blocks(page: fitz.Page) -> list[dict]:
    data = page.get_text("dict")
    blocks: list[dict] = []
    text_block_index = 1

    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue

        line_texts: list[str] = []
        max_font_size = 0.0
        is_bold = False
        fonts: set[str] = set()

        for line in block.get("lines", []):
            spans = line.get("spans", [])
            span_texts = []
            for span in spans:
                text = span.get("text", "")
                if not text.strip():
                    continue
                span_texts.append(text.strip())
                max_font_size = max(max_font_size, float(span.get("size", 0)))
                font_name = str(span.get("font", ""))
                fonts.add(font_name)
                if "bold" in font_name.lower():
                    is_bold = True
            if span_texts:
                line_texts.append(" ".join(span_texts))

        text = clean_text("\n".join(line_texts))
        if not text:
            continue

        blocks.append(
            {
                "text_block_id": f"text_p{page.number + 1:03d}_{text_block_index:02d}",
                "text": text,
                "bbox": bbox_from_sequence(block["bbox"]),
                "max_font_size": round(max_font_size, 2),
                "is_bold": is_bold,
                "fonts": sorted(fonts),
            }
        )
        text_block_index += 1

    blocks.sort(key=lambda item: (item["bbox"]["y0"], item["bbox"]["x0"]))
    return blocks


def extract_text(pdf_path: str | Path) -> list[dict]:
    source = Path(pdf_path).expanduser().resolve()
    pages: list[dict] = []

    with fitz.open(source) as doc:
        for page in doc:
            text_blocks = _extract_text_blocks(page)
            native_text = strip_page_boilerplate(page.get_text("text"))
            pages.append(
                {
                    "page_number": page.number + 1,
                    "width": round(page.rect.width, 2),
                    "height": round(page.rect.height, 2),
                    "native_text": native_text,
                    "native_text_chars": len(native_text),
                    "text_blocks": text_blocks,
                }
            )

    return pages
