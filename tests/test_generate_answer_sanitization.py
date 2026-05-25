from __future__ import annotations

import unittest

from scripts.generation.generate_answer import sanitize_final_answer


class TestGenerateAnswerSanitization(unittest.TestCase):
    def test_sanitize_removes_internal_source_tokens(self) -> None:
        raw = (
            "Les paramètres dépassant sont :\n"
            "- Triglycérides = 8 g/l\n"
            "Sources :\n"
            "- [doc_id=report_10, page=2, row=20, chunk_id=chk_123](#)\n"
        )
        out = sanitize_final_answer(raw)
        self.assertNotIn("doc_id=", out.lower())
        self.assertNotIn("chunk_id=", out.lower())
        self.assertIn("Triglycérides", out)

    def test_sanitize_keeps_clickable_viewer_links(self) -> None:
        raw = (
            "Sources :\n"
            "- [report (10).pdf — page 2, ligne 20](/viewer/pdf?doc_id=report_10&page=2)\n"
        )
        out = sanitize_final_answer(raw)
        self.assertIn("/viewer/pdf?doc_id=report_10&page=2", out)
        self.assertIn("report (10).pdf", out)


if __name__ == "__main__":
    unittest.main()
