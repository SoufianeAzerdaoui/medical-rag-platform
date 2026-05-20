from __future__ import annotations

import os

DEFAULT_LLM_PROVIDER = os.getenv("MEDICAL_RAG_LLM_PROVIDER", "ollama")

DEFAULT_LLM_MODEL = os.getenv("MEDICAL_RAG_LLM_MODEL", "llama3.2:latest")


DEFAULT_LLM_TEMPERATURE = float(os.getenv("MEDICAL_RAG_LLM_TEMPERATURE", "0.0"))

DEFAULT_LLM_NUM_CTX = int(os.getenv("MEDICAL_RAG_LLM_NUM_CTX", "2048"))

# DEFAULT_LLM_MAX_TOKENS = int(os.getenv("MEDICAL_RAG_LLM_MAX_TOKENS", "400"))

DEFAULT_LLM_MAX_TOKENS = int(os.getenv("MEDICAL_RAG_LLM_MAX_TOKENS", "220"))

DEFAULT_LLM_TIMEOUT = int(os.getenv("MEDICAL_RAG_LLM_TIMEOUT", "180"))
