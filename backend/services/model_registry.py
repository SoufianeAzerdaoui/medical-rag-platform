from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scripts.generation.model_settings import (
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_NUM_CTX,
    DEFAULT_LLM_PROVIDER,
)


@dataclass(frozen=True)
class ModelInfo:
    model: str
    provider: str
    context_window: int
    max_output_tokens: int
    tokenizer: str | None = None


MODEL_REGISTRY: dict[str, ModelInfo] = {
    "gpt-4o-mini": ModelInfo(
        model="gpt-4o-mini",
        provider="openai",
        context_window=128_000,
        max_output_tokens=4_096,
        tokenizer="o200k_base",
    ),
    "gpt-4o": ModelInfo(
        model="gpt-4o",
        provider="openai",
        context_window=128_000,
        max_output_tokens=4_096,
        tokenizer="o200k_base",
    ),
    "llama3.2:latest": ModelInfo(
        model="llama3.2:latest",
        provider="ollama",
        context_window=max(2048, int(DEFAULT_LLM_NUM_CTX)),
        max_output_tokens=max(128, int(DEFAULT_LLM_MAX_TOKENS)),
        tokenizer=None,
    ),
}


def active_model_info() -> ModelInfo:
    model = str(DEFAULT_LLM_MODEL or "").strip()
    provider = str(DEFAULT_LLM_PROVIDER or "").strip() or "unknown"
    if model in MODEL_REGISTRY:
        info = MODEL_REGISTRY[model]
        return ModelInfo(
            model=info.model,
            provider=provider or info.provider,
            context_window=info.context_window,
            max_output_tokens=info.max_output_tokens,
            tokenizer=info.tokenizer,
        )
    return ModelInfo(
        model=model or "unknown",
        provider=provider,
        context_window=max(2048, int(DEFAULT_LLM_NUM_CTX)),
        max_output_tokens=max(128, int(DEFAULT_LLM_MAX_TOKENS)),
        tokenizer=None,
    )


def estimate_tokens_from_text(text: str) -> int:
    raw = str(text or "")
    if not raw:
        return 0
    # Lightweight approximation when tokenizer is unavailable: ~4 chars/token.
    return max(1, len(raw) // 4)


def extract_rag_text_from_sources(sources: list[Any]) -> str:
    parts: list[str] = []
    for source in sources:
        if isinstance(source, str):
            parts.append(source)
            continue
        if not isinstance(source, dict):
            continue
        for key in ("excerpt", "label", "section", "documentName", "filename", "doc_id", "source"):
            value = str(source.get(key) or "").strip()
            if value:
                parts.append(value)
    return "\n".join(parts)

