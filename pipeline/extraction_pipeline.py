from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from validation.consistency_checker import detect_result_consistency


def inject_consistency_checks(document: dict) -> dict:
    """
    Validation-only post-processing.
    Does not modify existing extracted fields; only adds consistency_checks.
    """
    document["consistency_checks"] = detect_result_consistency(document)
    return document


def inject_consistency_checks_in_output_dir(output_dir: str | Path) -> None:
    output_path = Path(output_dir) / "document.json"
    document = json.loads(output_path.read_text(encoding="utf-8"))
    inject_consistency_checks(document)
    output_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")

