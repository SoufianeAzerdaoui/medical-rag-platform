from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATION_ROOT = PROJECT_ROOT / "scripts" / "generation"
if str(GENERATION_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERATION_ROOT))

from followup_scope_utils import resolve_followup_doc_scope


class TestFollowupDocScope(unittest.TestCase):
    def test_reuses_previous_doc_scope_for_deictic_analyte_followup(self) -> None:
        scope = resolve_followup_doc_scope(
            query="et TSHus ?",
            requested_analytes=["tshus"],
            requested_doc_ids=[],
            previous_doc_scope=["report_16"],
        )
        self.assertEqual(scope, ["report_16"])

    def test_does_not_override_explicit_doc_scope(self) -> None:
        scope = resolve_followup_doc_scope(
            query="et TSHus ?",
            requested_analytes=["tshus"],
            requested_doc_ids=["report_31"],
            previous_doc_scope=["report_16"],
        )
        self.assertEqual(scope, ["report_31"])


if __name__ == "__main__":
    unittest.main()

