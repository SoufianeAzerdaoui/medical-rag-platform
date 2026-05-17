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


if __name__ == "__main__":
    unittest.main()
