from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Any
from urllib import error, request


class LLMClientError(RuntimeError):
    """Raised when generation fails in a controlled way."""


@dataclass
class LLMClient:
    provider: str = "ollama"
    ollama_url: str = "http://127.0.0.1:11434/api/generate"

    def generate(
        self,
        prompt: str,
        model: str = "qwen3:4b",
        temperature: float = 0.0,
        num_ctx: int = 4096,
        max_tokens: int = 800,
        timeout: int = 180,
        keep_alive: str = "10m",
    ) -> str:
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
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "keep_alive": keep_alive,
            "options": {
                "temperature": temperature,
                "top_p": 0.8,
                "num_ctx": num_ctx,
                "num_predict": max_tokens,
            },
        }

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
            msg = self._http_error_message(exc)
            raise LLMClientError(msg) from exc
        except error.URLError as exc:
            if isinstance(exc.reason, ConnectionRefusedError) or isinstance(exc.reason, socket.error):
                raise LLMClientError("Ollama service unavailable. Check systemctl status ollama.") from exc
            raise LLMClientError(f"Ollama request failed: {exc.reason}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise LLMClientError("Ollama timeout. Increase timeout or reduce max_tokens.") from exc

        if status >= 400:
            raise LLMClientError(f"Ollama HTTP error: status={status}")

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMClientError("Invalid JSON response from Ollama.") from exc

        if isinstance(data, dict) and data.get("error"):
            err = str(data.get("error"))
            if "not found" in err.lower() and "model" in err.lower():
                raise LLMClientError(f"Ollama model not installed: {model}. Run: ollama pull {model}")
            raise LLMClientError(f"Ollama error: {err}")

        response_text = str(data.get("response") or "").strip()
        thinking_text = str(data.get("thinking") or "").strip()

        if not response_text and thinking_text:
            raise LLMClientError("LLM produced thinking only. Ensure think=false or increase max_tokens.")
        if not response_text:
            raise LLMClientError("LLM response is empty. Increase max_tokens or check prompt/evidence.")

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
