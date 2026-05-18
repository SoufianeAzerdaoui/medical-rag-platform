import unittest


class TestCommentVariantMerging(unittest.TestCase):
    def test_merge_comment_variants_caps_total_length(self) -> None:
        from scripts.generation.generate_answer import _merge_comment_variants

        base = "Valeur seuil : 20 mUI/l."
        variants = [base + (" A" * 900), base + (" B" * 900), base + (" C" * 900), base + (" D" * 900), base + (" E" * 900)]
        merged = _merge_comment_variants(variants)
        self.assertLessEqual(len(merged), 1200)

    def test_merge_comment_variants_caps_number_of_blocks(self) -> None:
        from scripts.generation.generate_answer import _merge_comment_variants

        variants = [
            "Commentaire principal long et détaillé.",
            "Variante complémentaire A.",
            "Variante complémentaire B.",
            "Variante complémentaire C.",
            "Variante complémentaire D.",
            "Variante complémentaire E.",
        ]
        merged = _merge_comment_variants(variants)
        # max 4 blocks => max 3 newline separators between blocks
        self.assertLessEqual(merged.count("\n"), 3)


if __name__ == "__main__":
    unittest.main()

