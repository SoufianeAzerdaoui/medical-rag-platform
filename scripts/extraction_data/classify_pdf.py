from __future__ import annotations

from pathlib import Path

import fitz

from utils import clean_text


def classify_pdf(pdf_path: str | Path, min_text_chars: int = 30) -> dict:
    source = Path(pdf_path).expanduser().resolve()

    page_signals: list[dict] = []
    with fitz.open(source) as doc:
        for page in doc:
            text = clean_text(page.get_text("text"))
            images = page.get_images(full=True)
            page_signals.append(
                {
                    "page_number": page.number + 1,
                    "native_text_chars": len(text),
                    "has_meaningful_text": len(text) >= min_text_chars,
                    "image_count": len(images),
                }
            )

    text_pages = sum(1 for p in page_signals if p["has_meaningful_text"])
    image_pages = sum(1 for p in page_signals if p["image_count"] > 0)
    page_count = len(page_signals)

    if text_pages == 0:
        pdf_type = "scanned"
    elif text_pages == page_count and image_pages == 0:
        pdf_type = "digital"
    else:
        pdf_type = "mixed"

    return {
        "pdf_type": pdf_type,
        "page_signals": page_signals,
        "native_text_available": text_pages > 0,
        "image_pages": image_pages,
    }
