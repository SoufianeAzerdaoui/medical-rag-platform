from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEN_DIR = ROOT / "scripts" / "generation"
if str(GEN_DIR) not in sys.path:
    sys.path.insert(0, str(GEN_DIR))

from llm_client import LLMClient, LLMClientError  # noqa: E402
from model_settings import (  # noqa: E402
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_NUM_CTX,
    DEFAULT_LLM_PROVIDER,
)

PROMPT = (
    "Tu es un rédacteur médical technique. Utilise uniquement les faits fournis. "
    "Ne donne pas de diagnostic. Ne propose pas de traitement. "
    "Rédige en 6 lignes maximum. Sépare Anormaux et Résultats dans la référence uniquement. "
    "Faits: ACIDE URIQUE 23 mg/l sous référence femme 26-60; "
    "Bilirubine Directe 6 mg/l au-dessus référence 0-5; "
    "Créatinine 23 mg/l au-dessus référence femme 5.7-11.1; "
    "LDH 250 UI/L au-dessus référence 125-243; "
    "CKMB 40 UI/L au-dessus référence <25; "
    "APO A1 2.3 g/L au-dessus référence 1.1-1.6."
)


def main() -> int:
    model = os.getenv("MEDICAL_RAG_LLM_MODEL", DEFAULT_LLM_MODEL)
    provider = os.getenv("MEDICAL_RAG_LLM_PROVIDER", DEFAULT_LLM_PROVIDER)
    num_ctx = int(os.getenv("MEDICAL_RAG_LLM_NUM_CTX", str(DEFAULT_LLM_NUM_CTX)))
    num_predict = int(os.getenv("MEDICAL_RAG_DIAG_NUM_PREDICT", "180"))
    timeout = int(os.getenv("MEDICAL_RAG_DIAG_TIMEOUT", "90"))
    keep_alive = os.getenv("MEDICAL_RAG_OLLAMA_KEEP_ALIVE", "10m")

    client = LLMClient(provider=provider)

    print("== Backend LLM writer latency diagnostic ==")
    print(f"provider={provider}")
    print(f"model={model}")
    print(f"num_predict={num_predict}")
    print(f"num_ctx={num_ctx}")
    print(f"keep_alive={keep_alive}")
    print(f"timeout_s={timeout}")
    print(f"prompt_chars={len(PROMPT)}")

    t0 = time.perf_counter()
    response = ""
    raw_error_type = None
    raw_error_message = None
    try:
        response = client.generate(
            prompt=PROMPT,
            model=model,
            temperature=0.0,
            num_ctx=num_ctx,
            max_tokens=num_predict,
            timeout=timeout,
            keep_alive=keep_alive,
        )
    except LLMClientError as exc:
        raw_error_type = type(exc).__name__
        raw_error_message = str(exc)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    dbg = dict(client.last_call_debug or {})
    print(f"elapsed_ms={elapsed_ms}")
    print(f"endpoint={dbg.get('ollama_endpoint')}")
    print(f"api_kind={dbg.get('ollama_api_kind')}")
    print(f"model_effective={dbg.get('ollama_model')}")
    print(f"num_predict_effective={dbg.get('ollama_num_predict')}")
    print(f"num_ctx_effective={dbg.get('ollama_num_ctx')}")
    print(f"keep_alive_effective={dbg.get('ollama_keep_alive')}")
    print(f"stream={dbg.get('stream')}")
    print(f"prompt_chars_effective={dbg.get('prompt_chars')}")
    print(f"llm_elapsed_ms={dbg.get('llm_elapsed_ms')}")
    print(f"raw_error_type={dbg.get('llm_raw_error_type') or raw_error_type}")
    print(f"raw_error_message={dbg.get('llm_raw_error_message') or raw_error_message}")
    print(f"total_duration={dbg.get('total_duration')}")
    print(f"load_duration={dbg.get('load_duration')}")
    print(f"prompt_eval_count={dbg.get('prompt_eval_count')}")
    print(f"prompt_eval_duration={dbg.get('prompt_eval_duration')}")
    print(f"eval_count={dbg.get('eval_count')}")
    print(f"eval_duration={dbg.get('eval_duration')}")
    print(f"tokens_per_second_estimate={dbg.get('tokens_per_second_estimate')}")

    if response:
        preview = response[:500].replace("\n", " ")
        print(f"response_preview={preview}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
