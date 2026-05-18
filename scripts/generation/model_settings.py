from __future__ import annotations

import os

# Centralized runtime defaults for LLM settings.
# Change values here to affect backend/API + generation/CLI defaults.
DEFAULT_LLM_PROVIDER = os.getenv("MEDICAL_RAG_LLM_PROVIDER", "ollama")
DEFAULT_LLM_MODEL = os.getenv("MEDICAL_RAG_LLM_MODEL", "llama3.2:latest")
DEFAULT_LLM_TEMPERATURE = float(os.getenv("MEDICAL_RAG_LLM_TEMPERATURE", "0.0"))
DEFAULT_LLM_NUM_CTX = int(os.getenv("MEDICAL_RAG_LLM_NUM_CTX", "2048"))
DEFAULT_LLM_MAX_TOKENS = int(os.getenv("MEDICAL_RAG_LLM_MAX_TOKENS", "400"))
DEFAULT_LLM_TIMEOUT = int(os.getenv("MEDICAL_RAG_LLM_TIMEOUT", "180"))

# Hybrid writer runtime flags (all overrideable via .env)
DEFAULT_FORCE_LLM_WRITER = str(os.getenv("MEDICAL_RAG_FORCE_LLM_WRITER", "1")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DEFAULT_VALIDATE_LLM_FACTS = str(os.getenv("MEDICAL_RAG_VALIDATE_LLM_FACTS", "1")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DEFAULT_LLM_REPAIR_RETRY = str(os.getenv("MEDICAL_RAG_LLM_REPAIR_RETRY", "1")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
