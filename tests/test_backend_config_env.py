from __future__ import annotations

import importlib
import os
import unittest
from unittest.mock import patch


class TestBackendConfigEnv(unittest.TestCase):
    def test_feature_flag_admin_env_parsing(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ENABLE_FEATURE_FLAG_ADMIN_API": "true",
                "ADMIN_EMAILS": "simo@test.ma,admin2@test.ma",
            },
            clear=False,
        ):
            import backend.config as backend_config

            backend_config = importlib.reload(backend_config)
            self.assertTrue(bool(backend_config.ENABLE_FEATURE_FLAG_ADMIN_API))
            self.assertIn("simo@test.ma", backend_config.ADMIN_EMAILS)
            self.assertIn("admin2@test.ma", backend_config.ADMIN_EMAILS)

    def test_generation_model_settings_accept_legacy_env_names(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "gemini",
                "GEMINI_MODEL": "gemini-2.5-flash",
            },
            clear=False,
        ):
            import scripts.generation.model_settings as model_settings

            model_settings = importlib.reload(model_settings)
            self.assertEqual(model_settings.DEFAULT_LLM_PROVIDER, "gemini")
            self.assertEqual(model_settings.DEFAULT_LLM_MODEL, "gemini-2.5-flash")


if __name__ == "__main__":
    unittest.main()
