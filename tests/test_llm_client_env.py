from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATION_DIR = PROJECT_ROOT / "scripts" / "generation"
if str(GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATION_DIR))

from llm_client import LLMClient


class TestLlmClientEnv(unittest.TestCase):
    def test_default_ollama_url_uses_env_host(self) -> None:
        with mock.patch.dict(os.environ, {"MEDICAL_RAG_OLLAMA_URL": "http://ollama:11434"}, clear=False):
            client = LLMClient(provider="ollama")
        self.assertEqual(client.ollama_url, "http://ollama:11434/api/generate")

    def test_default_ollama_url_normalizes_host_only_env(self) -> None:
        with mock.patch.dict(os.environ, {"OLLAMA_HOST": "ollama:11434"}, clear=False):
            client = LLMClient(provider="ollama")
        self.assertEqual(client.ollama_url, "http://ollama:11434/api/generate")


if __name__ == "__main__":
    unittest.main()
