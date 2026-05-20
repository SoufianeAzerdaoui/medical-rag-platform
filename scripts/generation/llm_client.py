from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass
from typing import Any
from urllib import error, request
from model_settings import (
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_NUM_CTX,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_TIMEOUT,
)


class LLMClientError(RuntimeError):
    """Raised when generation fails in a controlled way."""


@dataclass
class LLMClient:
    provider: str = DEFAULT_LLM_PROVIDER
    ollama_url: str = "http://127.0.0.1:11434/api/generate"
    last_call_debug: dict[str, Any] | None = None

    def generate(
        self,
        prompt: str,
        model: str = DEFAULT_LLM_MODEL,
        temperature: float = DEFAULT_LLM_TEMPERATURE,
        num_ctx: int = DEFAULT_LLM_NUM_CTX,
        max_tokens: int = 800,
        timeout: int = DEFAULT_LLM_TIMEOUT,
        keep_alive: str = "10m",
    ) -> str:
        keep_alive = str(os.getenv("MEDICAL_RAG_OLLAMA_KEEP_ALIVE", keep_alive)).strip() or "10m"
        self.last_call_debug = {
            "provider": self.provider,
            "model": model,
            "temperature": float(temperature),
            "num_ctx": int(num_ctx),
            "num_predict": int(max_tokens),
            "keep_alive": keep_alive,
            "stream": False,
            "prompt_chars": len(str(prompt or "")),
            "prompt_preview_first_500": str(prompt or "")[:500],
            "prompt_preview_last_500": str(prompt or "")[-500:] if prompt else "",
            "llm_timeout_ms": int(timeout) * 1000,
        }
        if self.provider == "ollama":
            return self._generate_ollama(
                prompt=prompt,
                model=model,
                temperature=temperature,
                num_ctx=num_ctx,
                max_tokens=max_tokens,
                timeout=timeout,
                keep_alive=keep_alive,
            )
        elif self.provider == "lmstudio":
            return self._generate_lmstudio(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        else:
            raise LLMClientError(f"Unsupported provider: {self.provider}")

    def _generate_lmstudio(
        self,
        *,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: int,
    ) -> str:
        # LM Studio is OpenAI compatible
        import os
        base_url = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
        # Ensure url ends with /chat/completions if it's the v1 base
        url = base_url.rstrip("/") + "/chat/completions"
        
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                status = getattr(resp, "status", 200)
        except Exception as exc:
            raise LLMClientError(f"LM Studio request failed: {str(exc)}") from exc

        if status >= 400:
            raise LLMClientError(f"LM Studio HTTP error: status={status}")

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMClientError("Invalid JSON response from LM Studio.") from exc

        # OpenAI format: data['choices'][0]['message']['content']
        choices = data.get("choices") or []
        if not choices:
            raise LLMClientError("LM Studio returned no choices.")
        
        response_text = choices[0].get("message", {}).get("content", "").strip()
        if not response_text:
            raise LLMClientError("LM Studio response is empty.")

        return response_text

    def _generate_ollama(
        self,
        *,
        prompt: str,
        model: str,
        temperature: float,
        num_ctx: int,
        max_tokens: int,
        timeout: int,
        keep_alive: str,
    ) -> str:
        t0 = time.perf_counter()
        api_kind = "generate"
        endpoint = str(self.ollama_url or "").strip()
        if endpoint.endswith("/api/chat"):
            api_kind = "chat"
        payload: dict[str, Any] = {
            "model": model,
            "stream": False,
            "think": False,
            "keep_alive": keep_alive,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
                "num_predict": max_tokens,
            },
        }
        if api_kind == "chat":
            payload["messages"] = [{"role": "user", "content": prompt}]
        else:
            payload["prompt"] = prompt
        if isinstance(self.last_call_debug, dict):
            self.last_call_debug.update(
                {
                    "ollama_endpoint": endpoint,
                    "ollama_api_kind": api_kind,
                    "ollama_model": model,
                    "ollama_num_predict": int(max_tokens),
                    "ollama_num_ctx": int(num_ctx),
                    "ollama_temperature": float(temperature),
                    "ollama_keep_alive": keep_alive,
                    "stream": False,
                    "messages_count": len(payload.get("messages") or []),
                    "conversation_history_included": bool(len(payload.get("messages") or []) > 1),
                    "system_prompt_chars": 0,
                    "user_prompt_chars": len(str(prompt or "")),
                }
            )

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            self.ollama_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                status = getattr(resp, "status", 200)
        except error.HTTPError as exc:
            if isinstance(self.last_call_debug, dict):
                self.last_call_debug.update(
                    {
                        "llm_elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                        "llm_raw_error_type": "HTTPError",
                        "llm_raw_error_message": str(exc),
                    }
                )
            msg = self._http_error_message(exc)
            raise LLMClientError(msg) from exc
        except error.URLError as exc:
            if isinstance(self.last_call_debug, dict):
                self.last_call_debug.update(
                    {
                        "llm_elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                        "llm_raw_error_type": "URLError",
                        "llm_raw_error_message": str(exc.reason),
                    }
                )
            if isinstance(exc.reason, ConnectionRefusedError) or isinstance(exc.reason, socket.error):
                raise LLMClientError("Ollama service unavailable. Check systemctl status ollama.") from exc
            raise LLMClientError(f"Ollama request failed: {exc.reason}") from exc
        except (TimeoutError, socket.timeout) as exc:
            if isinstance(self.last_call_debug, dict):
                self.last_call_debug.update(
                    {
                        "llm_elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                        "llm_raw_error_type": "TimeoutError",
                        "llm_raw_error_message": str(exc),
                    }
                )
            raise LLMClientError("Ollama timeout. Increase timeout or reduce max_tokens.") from exc

        if status >= 400:
            raise LLMClientError(f"Ollama HTTP error: status={status}")

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            if isinstance(self.last_call_debug, dict):
                self.last_call_debug.update(
                    {
                        "llm_elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                        "llm_raw_error_type": "JSONDecodeError",
                        "llm_raw_error_message": str(exc),
                    }
                )
            raise LLMClientError("Invalid JSON response from Ollama.") from exc

        if isinstance(data, dict) and data.get("error"):
            err = str(data.get("error"))
            if isinstance(self.last_call_debug, dict):
                self.last_call_debug.update(
                    {
                        "llm_elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                        "llm_raw_error_type": "OllamaError",
                        "llm_raw_error_message": err,
                    }
                )
            if "not found" in err.lower() and "model" in err.lower():
                raise LLMClientError(f"Ollama model not installed: {model}. Run: ollama pull {model}")
            raise LLMClientError(f"Ollama error: {err}")

        response_text = str(data.get("response") or "").strip()
        thinking_text = str(data.get("thinking") or "").strip()

        if not response_text and thinking_text:
            raise LLMClientError("LLM produced thinking only. Ensure think=false or increase max_tokens.")
        if not response_text:
            raise LLMClientError("LLM response is empty. Increase max_tokens or check prompt/evidence.")
        if isinstance(self.last_call_debug, dict):
            total_duration = int(data.get("total_duration") or 0)
            eval_count = int(data.get("eval_count") or 0)
            eval_duration = int(data.get("eval_duration") or 0)
            tps = 0.0
            if eval_count > 0 and eval_duration > 0:
                tps = float(eval_count) / (float(eval_duration) / 1_000_000_000.0)
            self.last_call_debug.update(
                {
                    "llm_elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                    "llm_raw_error_type": None,
                    "llm_raw_error_message": None,
                    "total_duration": total_duration,
                    "load_duration": int(data.get("load_duration") or 0),
                    "prompt_eval_count": int(data.get("prompt_eval_count") or 0),
                    "prompt_eval_duration": int(data.get("prompt_eval_duration") or 0),
                    "eval_count": eval_count,
                    "eval_duration": eval_duration,
                    "tokens_per_second_estimate": round(tps, 3) if tps > 0 else 0.0,
                }
            )

        return response_text

    @staticmethod
    def _http_error_message(exc: error.HTTPError) -> str:
        try:
            body = exc.read().decode("utf-8", errors="replace")
            parsed = json.loads(body)
            err = str(parsed.get("error") or "").strip()
            if err:
                if "not found" in err.lower() and "model" in err.lower():
                    return f"Ollama model not installed: {err}"
                return f"Ollama error: {err}"
        except Exception:
            pass
        if exc.code in {502, 503, 504}:
            return "Ollama service unavailable. Check systemctl status ollama."
        return f"Ollama HTTP error: {exc.code}"
