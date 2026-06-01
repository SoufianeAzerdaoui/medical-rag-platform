from __future__ import annotations

import os

def _env_first(*names: str, default: str = "") -> str:
    for name in names:
        value = str(os.getenv(name, "")).strip()
        if value:
            return value
    return default


DEFAULT_LLM_PROVIDER = _env_first("MEDICAL_RAG_LLM_PROVIDER", "LLM_PROVIDER", default="ollama").lower()

if DEFAULT_LLM_PROVIDER == "gemini":
    DEFAULT_LLM_MODEL = _env_first("MEDICAL_RAG_LLM_MODEL", "GEMINI_MODEL", default="gemini-2.5-flash")
else:
    DEFAULT_LLM_MODEL = _env_first("MEDICAL_RAG_LLM_MODEL", default="llama3.2:latest")


DEFAULT_LLM_TEMPERATURE = float(os.getenv("MEDICAL_RAG_LLM_TEMPERATURE", "0.0"))

DEFAULT_LLM_NUM_CTX = int(os.getenv("MEDICAL_RAG_LLM_NUM_CTX", "2048"))

# DEFAULT_LLM_MAX_TOKENS = int(os.getenv("MEDICAL_RAG_LLM_MAX_TOKENS", "400"))

DEFAULT_LLM_MAX_TOKENS = int(os.getenv("MEDICAL_RAG_LLM_MAX_TOKENS", "220"))

DEFAULT_LLM_TIMEOUT = int(os.getenv("MEDICAL_RAG_LLM_TIMEOUT", "180"))
