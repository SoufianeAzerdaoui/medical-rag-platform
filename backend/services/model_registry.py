from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any
from urllib import request as urllib_request

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
    recommended_rag_budget: int
    tokenizer: str | None = None


MODEL_REGISTRY: dict[str, ModelInfo] = {
    "gpt-4o-mini": ModelInfo(
        model="gpt-4o-mini",
        provider="openai",
        context_window=128_000,
        max_output_tokens=4_096,
        recommended_rag_budget=12_000,
        tokenizer="o200k_base",
    ),
    "gpt-4o": ModelInfo(
        model="gpt-4o",
        provider="openai",
        context_window=128_000,
        max_output_tokens=4_096,
        recommended_rag_budget=16_000,
        tokenizer="o200k_base",
    ),
    "llama3.2:latest": ModelInfo(
        model="llama3.2:latest",
        provider="ollama",
        context_window=max(8_192, int(DEFAULT_LLM_NUM_CTX)),
        max_output_tokens=max(1_024, int(DEFAULT_LLM_MAX_TOKENS)),
        recommended_rag_budget=5_000,
        tokenizer=None,
    ),
}


def _safe_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except Exception:
        return None
    return parsed if parsed > 0 else None


def _probe_ollama_model_runtime(model: str) -> tuple[int | None, int | None]:
    """
    Best-effort probe of Ollama runtime model settings.
    If unavailable, caller falls back to registry/default values.
    """
    host = str(os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")).strip().rstrip("/")
    url = f"{host}/api/show"
    payload = json.dumps({"name": model}).encode("utf-8")
    req = urllib_request.Request(url, data=payload, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib_request.urlopen(req, timeout=1.2) as resp:  # nosec B310 - local service probe
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None, None
    parameters = str(data.get("parameters") or "")
    context_window = None
    max_output = None
    for line in parameters.splitlines():
        item = line.strip().lower()
        if item.startswith("num_ctx"):
            context_window = _safe_int(item.split()[-1])
        if item.startswith("num_predict"):
            max_output = _safe_int(item.split()[-1])
    return context_window, max_output


def active_model_info() -> ModelInfo:
    model = str(DEFAULT_LLM_MODEL or "").strip()
    provider = str(DEFAULT_LLM_PROVIDER or "").strip() or "unknown"
    if model in MODEL_REGISTRY:
        info = MODEL_REGISTRY[model]
        resolved_context_window = info.context_window
        resolved_max_output = info.max_output_tokens
        if (provider or info.provider).lower() == "ollama":
            runtime_ctx, runtime_out = _probe_ollama_model_runtime(info.model)
            if runtime_ctx:
                resolved_context_window = max(runtime_ctx, info.context_window)
            if runtime_out:
                resolved_max_output = max(128, runtime_out)
        return ModelInfo(
            model=info.model,
            provider=provider or info.provider,
            context_window=resolved_context_window,
            max_output_tokens=resolved_max_output,
            recommended_rag_budget=info.recommended_rag_budget,
            tokenizer=info.tokenizer,
        )
    return ModelInfo(
        model=model or "unknown",
        provider=provider,
        context_window=max(8_192, int(DEFAULT_LLM_NUM_CTX)),
        max_output_tokens=max(1_024, int(DEFAULT_LLM_MAX_TOKENS)),
        recommended_rag_budget=5_000,
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


def trim_text_to_token_budget(text: str, token_budget: int) -> str:
    raw = str(text or "")
    if not raw:
        return ""
    budget = max(1, int(token_budget))
    max_chars = budget * 4
    if len(raw) <= max_chars:
        return raw
    return raw[:max_chars]
