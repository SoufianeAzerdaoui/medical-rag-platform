from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.ingestion_service import reindex_single_doc


def main() -> int:
    parser = argparse.ArgumentParser(description="Reingest and rebuild indexes for a single document.")
    parser.add_argument("doc_id", help="Document identifier, e.g. report_12")
    parser.add_argument("pdf_path", help="Path to the real PDF file to ingest")
    args = parser.parse_args()

    result = reindex_single_doc(
        str(args.doc_id or "").strip().lower(),
        Path(str(args.pdf_path or "")).expanduser().resolve(),
    )
    print(
        json.dumps(
            {
                "filename": result.filename,
                "stored_path": result.stored_path,
                "doc_id": result.doc_id,
                "extraction_dir": result.extraction_dir,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
