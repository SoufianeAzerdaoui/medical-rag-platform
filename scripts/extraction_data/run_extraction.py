from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import sys

from classify_pdf import classify_pdf
from extract_images import extract_images
from extract_ocr import extract_ocr
from extract_tables import extract_tables, extract_tables_from_ocr
from extract_text import extract_text
from ingest_pdf import ingest_pdf
from segment_blocks import build_blocks
from structure_clinical import build_structured_document
from structure_document import structure_document
from utils import ensure_dir, strip_page_boilerplate

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "data" / "extraction"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.extraction_pipeline import inject_consistency_checks_in_output_dir


def _merge_ocr_into_pages(page_text_data: list[dict], ocr_results: dict[int, object]) -> list[dict]:
    merged_pages: list[dict] = []
    for page in page_text_data:
        merged = dict(page)
        merged["native_text_blocks"] = list(page.get("text_blocks", []))
        merged["ocr_text"] = ""
        merged["ocr_text_chars"] = 0
        merged["ocr_text_blocks"] = []
        merged["ocr_words"] = []
        merged["final_text"] = page["native_text"]
        merged["text_source"] = "native" if page["native_text"] else "none"
        ocr_asset = ocr_results.get(page["page_number"])
        if ocr_asset and getattr(ocr_asset, "used", False) and getattr(ocr_asset, "text", ""):
            cleaned_ocr_text = strip_page_boilerplate(ocr_asset.text)
            merged["ocr_text"] = cleaned_ocr_text
            merged["ocr_text_chars"] = len(cleaned_ocr_text)
            merged["ocr_text_blocks"] = list(getattr(ocr_asset, "blocks", []))
            merged["ocr_words"] = list(getattr(ocr_asset, "words", []))
            if page["native_text_chars"] < 30:
                merged["final_text"] = cleaned_ocr_text
                merged["text_source"] = "ocr"
                merged["text_blocks"] = list(getattr(ocr_asset, "blocks", []))
            else:
                merged["final_text"] = page["native_text"]
                merged["text_source"] = "hybrid"
        merged_pages.append(merged)
    return merged_pages


def run_pipeline(pdf_path: str | Path, output_root: str | Path | None = None) -> Path:
    ingest_result = ingest_pdf(pdf_path)
    doc_id = ingest_result["doc_id"]
    base_output = Path(output_root) if output_root else DEFAULT_OUTPUT_ROOT
    output_dir = base_output / doc_id
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir = ensure_dir(output_dir)
    ensure_dir(output_dir / "pages")
    ensure_dir(output_dir / "tables")
    ensure_dir(output_dir / "images")
    ensure_dir(output_dir / "ocr")

    classify_result = classify_pdf(pdf_path)
    page_text_data = extract_text(pdf_path)

    pages_to_ocr = [
        page["page_number"]
        for page in page_text_data
        if page["native_text_chars"] < 30
    ]
    ocr_results = extract_ocr(pdf_path, output_dir, pages_to_ocr=pages_to_ocr) if pages_to_ocr else {}
    page_text_data = _merge_ocr_into_pages(page_text_data, ocr_results)
    tables = extract_tables(pdf_path, output_dir)
    if not tables and ocr_results:
        tables = extract_tables_from_ocr(output_dir, ocr_results)
    images, ocr_visuals = extract_images(pdf_path, output_dir, page_text_data)

    blocks = build_blocks(page_text_data, tables, images, ocr_visuals)
    structured_data = build_structured_document(
        page_text_data=page_text_data,
        tables=tables,
        images=images,
        ocr_visuals=ocr_visuals,
        blocks=blocks,
    )

    structure_document(
        ingest_result=ingest_result,
        classify_result=classify_result,
        page_text_data=page_text_data,
        tables=tables,
        images=images,
        ocr_visuals=ocr_visuals,
        blocks=blocks,
        structured_data=structured_data,
        ocr_results=ocr_results,
        output_dir=output_dir,
    )
    inject_consistency_checks_in_output_dir(output_dir)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run extraction on a medical PDF.")
    parser.add_argument("pdf_path", help="Path to the PDF to extract.")
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root output directory.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = run_pipeline(args.pdf_path, args.output_root)
    print(f"Extraction completed: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
