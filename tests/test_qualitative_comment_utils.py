from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATION_ROOT = PROJECT_ROOT / "scripts" / "generation"
if str(GENERATION_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERATION_ROOT))

from qualitative_comment_utils import (
    build_qualitative_comment_answer,
    build_sourced_comment_block,
    extract_comment_text_for_subject,
)


class TestQualitativeCommentUtils(unittest.TestCase):
    def test_extract_troponine_comment_text_from_mock_text(self) -> None:
        rows = [
            {
                "doc_id": "report_18",
                "source_pdf": "report (18).pdf",
                "page_number": 1,
                "text_for_embedding": (
                    "Commentaire : Valeur seuil au 99ème percentile : 26 ng/l. "
                    "Attention : Elévation de la troponine dans des situations autres que le SCA."
                ),
            }
        ]
        comment, row = extract_comment_text_for_subject("troponine", rows)
        self.assertTrue(comment)
        self.assertIn("Valeur seuil", comment)
        self.assertIsNotNone(row)

    def test_build_comment_answer_not_insufficient_when_comment_exists(self) -> None:
        answer = build_qualitative_comment_answer(
            subject="Troponine",
            comment_text="Valeur seuil au 99ème percentile : 26 ng/l. Attention : élévation possible hors SCA.",
            source_label="report (18).pdf — page 1",
        )
        self.assertIn("Valeur seuil au 99ème percentile", answer)
        self.assertIn("troponine", answer.lower())
        self.assertIn("Source :", answer)
        self.assertNotIn("Information insuffisante", answer)

    def test_build_sourced_comment_block(self) -> None:
        block = build_sourced_comment_block(
            subject="Troponine",
            comment_text="Valeur seuil au 99ème percentile : 26 ng/l.",
            source_label="report (18).pdf — page 1",
        )
        self.assertIn("Bloc commentaire sourcé", block)
        self.assertIn("Sujet : Troponine", block)
        self.assertIn("Source : report (18).pdf — page 1", block)
        self.assertNotIn("Bonjour", block)


if __name__ == "__main__":
    unittest.main()

