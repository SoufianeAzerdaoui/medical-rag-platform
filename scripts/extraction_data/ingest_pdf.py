from __future__ import annotations

from pathlib import Path

import fitz

from utils import safe_stem


def ingest_pdf(pdf_path: str | Path) -> dict:
    source = Path(pdf_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"PDF not found: {source}")

    with fitz.open(source) as doc:
        metadata = doc.metadata or {}
        page_sizes = [
            {
                "page_number": page.number + 1,
                "width": round(page.rect.width, 2),
                "height": round(page.rect.height, 2),
                "rotation": page.rotation,
            }
            for page in doc
        ]

    return {
        "doc_id": safe_stem(source.stem),
        "source_pdf": str(source),
        "file_name": source.name,
        "page_count": len(page_sizes),
        "metadata": metadata,
        "page_sizes": page_sizes,
    }
