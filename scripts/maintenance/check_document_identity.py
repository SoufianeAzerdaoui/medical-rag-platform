from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.document_identity_service import resolve_document_identity


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect document identity consistency for a doc_id.")
    parser.add_argument("doc_id", help="Document identifier, e.g. report_12")
    args = parser.parse_args()
    payload = resolve_document_identity(str(args.doc_id or "").strip().lower())
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
