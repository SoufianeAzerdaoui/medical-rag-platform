from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass, field
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


def _normalize_ollama_url(raw_url: str | None) -> str:
    raw = str(raw_url or "").strip()
    if not raw:
        return "http://127.0.0.1:11434/api/generate"
    if raw.startswith("http://") or raw.startswith("https://"):
        normalized = raw.rstrip("/")
    else:
        normalized = f"http://{raw}".rstrip("/")
    if normalized.endswith("/api/generate") or normalized.endswith("/api/chat"):
        return normalized
    return normalized + "/api/generate"


def _default_ollama_url() -> str:
    return _normalize_ollama_url(
        os.getenv("MEDICAL_RAG_OLLAMA_URL")
        or os.getenv("OLLAMA_HOST")
        or "http://127.0.0.1:11434/api/generate"
    )


@dataclass
class LLMClient:
    provider: str = DEFAULT_LLM_PROVIDER
    ollama_url: str = field(default_factory=_default_ollama_url)
    last_call_debug: dict[str, Any] | None = None

    def generate(
        self,
        prompt: str | None = None,
        model: str = DEFAULT_LLM_MODEL,
        temperature: float = DEFAULT_LLM_TEMPERATURE,
        num_ctx: int = DEFAULT_LLM_NUM_CTX,
        max_tokens: int = 800,
        timeout: int = DEFAULT_LLM_TIMEOUT,
        keep_alive: str = "10m",
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> str:
        provider_norm = str(self.provider or "").strip().lower()
        keep_alive = str(os.getenv("MEDICAL_RAG_OLLAMA_KEEP_ALIVE", keep_alive)).strip() or "10m"
        message_payload, preview_text = self._build_message_payload(
            prompt=prompt,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            messages=messages,
        )
        system_prompt_chars = sum(len(str(msg.get("content") or "")) for msg in message_payload if str(msg.get("role") or "").strip().lower() == "system")
        user_prompt_chars = sum(len(str(msg.get("content") or "")) for msg in message_payload if str(msg.get("role") or "").strip().lower() != "system")
        self.last_call_debug = {
            "provider": provider_norm,
            "llm_provider": provider_norm,
            "model": model,
            "llm_model": model,
            "temperature": float(temperature),
            "num_ctx": int(num_ctx),
            "num_predict": int(max_tokens),
            "keep_alive": keep_alive,
            "stream": False,
            "prompt_chars": len(preview_text),
            "prompt_preview_first_500": preview_text[:500],
            "prompt_preview_last_500": preview_text[-500:] if preview_text else "",
            "messages_count": len(message_payload),
            "conversation_history_included": len(message_payload) > 1,
            "system_prompt_chars": system_prompt_chars,
            "user_prompt_chars": user_prompt_chars,
            "llm_timeout_ms": int(timeout) * 1000,
        }
        if provider_norm == "ollama":
            return self._generate_ollama(
                prompt=prompt,
                model=model,
                temperature=temperature,
                num_ctx=num_ctx,
                max_tokens=max_tokens,
                timeout=timeout,
                keep_alive=keep_alive,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                messages=message_payload,
            )
        elif provider_norm == "lmstudio":
            return self._generate_lmstudio(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                messages=message_payload,
            )
        elif provider_norm == "gemini":
            return self._generate_gemini(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                messages=message_payload,
            )
        else:
            raise LLMClientError(f"Unsupported provider: {self.provider}")

    @staticmethod
    def _build_message_payload(
        *,
        prompt: str | None,
        system_prompt: str | None,
        user_prompt: str | None,
        messages: list[dict[str, Any]] | None,
    ) -> tuple[list[dict[str, str]], str]:
        if messages:
            normalized: list[dict[str, str]] = []
            for message in messages:
                if not isinstance(message, dict):
                    continue
                role = str(message.get("role") or "user").strip().lower() or "user"
                content = str(message.get("content") or message.get("text") or "").strip()
                if not content:
                    continue
                normalized.append({"role": role, "content": content})
            preview = "\n\n".join(f"[{msg['role']}] {msg['content']}" for msg in normalized)
            return normalized, preview

        normalized = []
        if system_prompt and str(system_prompt).strip():
            normalized.append({"role": "system", "content": str(system_prompt).strip()})
        user_text = str(user_prompt or prompt or "").strip()
        if user_text:
            normalized.append({"role": "user", "content": user_text})
        preview = "\n\n".join(f"[{msg['role']}] {msg['content']}" for msg in normalized)
        return normalized, preview

    def _generate_gemini(
        self,
        *,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: int,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> str:
        t0 = time.perf_counter()
        api_key = str(os.getenv("GEMINI_API_KEY", "")).strip() or str(os.getenv("GOOGLE_API_KEY", "")).strip()
        if not api_key:
            raise LLMClientError("Gemini API key missing. Set GEMINI_API_KEY.")

        base_url = str(os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")).strip().rstrip("/")
        model_name = str(model or "").strip()
        if not model_name:
            raise LLMClientError("Gemini model is missing.")
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"

        contents: list[dict[str, Any]] = []
        system_instruction = str(system_prompt or "").strip()
        normalized_messages = list(messages or [])
        if not normalized_messages:
            user_text = str(user_prompt or prompt or "").strip()
            if user_text:
                normalized_messages = [{"role": "user", "content": user_text}]
        for message in normalized_messages:
            role = str(message.get("role") or "user").strip().lower() or "user"
            content = str(message.get("content") or "").strip()
            if not content:
                continue
            if role == "system":
                system_instruction = system_instruction or content
                continue
            contents.append(
                {
                    "role": "model" if role == "assistant" else "user",
                    "parts": [{"text": content}],
                }
            )

        if not contents:
            contents = [
                {
                    "role": "user",
                    "parts": [{"text": str(user_prompt or prompt or "").strip()}],
                }
            ]

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": float(temperature),
                "maxOutputTokens": int(max_tokens),
            },
        }
        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}],
            }

        url = f"{base_url}/{model_name}:generateContent"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
            method="POST",
        )

        if isinstance(self.last_call_debug, dict):
            self.last_call_debug.update(
                {
                    "gemini_endpoint": url,
                    "gemini_model": model,
                    "gemini_temperature": float(temperature),
                    "gemini_max_output_tokens": int(max_tokens),
                    "messages_count": len(contents) + (1 if system_instruction else 0),
                    "conversation_history_included": len(contents) > 1,
                    "system_prompt_chars": len(system_instruction),
                    "user_prompt_chars": sum(len(str(item.get("parts", [{}])[0].get("text") or "")) for item in contents),
                }
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
            msg = self._gemini_http_error_message(exc)
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
            raise LLMClientError(f"Gemini request failed: {exc.reason}") from exc
        except (TimeoutError, socket.timeout) as exc:
            if isinstance(self.last_call_debug, dict):
                self.last_call_debug.update(
                    {
                        "llm_elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                        "llm_raw_error_type": "TimeoutError",
                        "llm_raw_error_message": str(exc),
                    }
                )
            raise LLMClientError("Gemini timeout. Increase timeout or reduce max_tokens.") from exc

        if status >= 400:
            raise LLMClientError(f"Gemini HTTP error: status={status}")

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMClientError("Invalid JSON response from Gemini.") from exc

        if isinstance(data, dict) and data.get("error"):
            err = data.get("error")
            if isinstance(err, dict):
                msg = str(err.get("message") or "").strip()
            else:
                msg = str(err or "").strip()
            if msg:
                raise LLMClientError(f"Gemini error: {msg}")
            raise LLMClientError("Gemini returned an error response.")

        response_text = self._extract_gemini_text(data)
        if not response_text:
            raise LLMClientError("Gemini response is empty.")

        if isinstance(self.last_call_debug, dict):
            self.last_call_debug.update(
                {
                    "llm_elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                    "llm_raw_error_type": None,
                    "llm_raw_error_message": None,
                }
            )

        return response_text

    @staticmethod
    def _extract_gemini_text(data: dict[str, Any]) -> str:
        candidates = data.get("candidates") or []
        chunks: list[str] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content") or {}
            if not isinstance(content, dict):
                continue
            for part in content.get("parts") or []:
                if not isinstance(part, dict):
                    continue
                text = str(part.get("text") or "").strip()
                if text:
                    chunks.append(text)
            if chunks:
                break
        return "\n".join(chunks).strip()

    def _generate_lmstudio(
        self,
        *,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: int,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> str:
        # LM Studio is OpenAI compatible
        import os
        base_url = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
        # Ensure url ends with /chat/completions if it's the v1 base
        url = base_url.rstrip("/") + "/chat/completions"
        chat_messages: list[dict[str, str]] = []
        normalized_messages = list(messages or [])
        if not normalized_messages:
            if system_prompt and str(system_prompt).strip():
                normalized_messages.append({"role": "system", "content": str(system_prompt).strip()})
            user_text = str(user_prompt or prompt or "").strip()
            if user_text:
                normalized_messages.append({"role": "user", "content": user_text})
        for message in normalized_messages:
            role = str(message.get("role") or "user").strip().lower() or "user"
            content = str(message.get("content") or "").strip()
            if not content:
                continue
            chat_messages.append({"role": role, "content": content})

        payload: dict[str, Any] = {
            "model": model,
            "messages": chat_messages or [{"role": "user", "content": str(user_prompt or prompt or "").strip()}],
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
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> str:
        t0 = time.perf_counter()
        api_kind = "generate"
        endpoint = str(self.ollama_url or "").strip()
        use_chat = bool(messages) or bool(str(system_prompt or "").strip()) or bool(str(user_prompt or "").strip()) or endpoint.endswith("/api/chat")
        if use_chat:
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
            chat_messages: list[dict[str, str]] = []
            normalized_messages = list(messages or [])
            if not normalized_messages:
                if system_prompt and str(system_prompt).strip():
                    normalized_messages.append({"role": "system", "content": str(system_prompt).strip()})
                user_text = str(user_prompt or prompt or "").strip()
                if user_text:
                    normalized_messages.append({"role": "user", "content": user_text})
            for message in normalized_messages:
                role = str(message.get("role") or "user").strip().lower() or "user"
                content = str(message.get("content") or "").strip()
                if not content:
                    continue
                chat_messages.append({"role": role, "content": content})
            payload["messages"] = chat_messages or [{"role": "user", "content": str(user_prompt or prompt or "").strip()}]
        else:
            payload["prompt"] = prompt
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if endpoint.endswith("/api/chat"):
            request_url = endpoint
        elif endpoint.endswith("/api/generate"):
            request_url = endpoint.rsplit("/generate", 1)[0] + "/chat"
        else:
            request_url = endpoint.rstrip("/") + "/api/chat"
        if isinstance(self.last_call_debug, dict):
            self.last_call_debug.update(
                {
                    "llm_provider": "ollama",
                    "llm_model": model,
                    "ollama_endpoint": request_url,
                    "ollama_api_kind": api_kind,
                    "ollama_model": model,
                    "ollama_num_predict": int(max_tokens),
                    "ollama_num_ctx": int(num_ctx),
                    "ollama_temperature": float(temperature),
                    "ollama_keep_alive": keep_alive,
                    "stream": False,
                    "messages_count": len(payload.get("messages") or []),
                    "conversation_history_included": bool(len(payload.get("messages") or []) > 1),
                    "system_prompt_chars": len(str(system_prompt or "").strip()),
                    "user_prompt_chars": len(str(user_prompt or prompt or "").strip()),
                }
            )
        req = request.Request(
            request_url,
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
        if not response_text:
            message = data.get("message") or {}
            if isinstance(message, dict):
                response_text = str(message.get("content") or "").strip()
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
    def _gemini_http_error_message(exc: error.HTTPError) -> str:
        try:
            body = exc.read().decode("utf-8", errors="replace")
            parsed = json.loads(body)
            err = parsed.get("error") or {}
            if isinstance(err, dict):
                message = str(err.get("message") or "").strip()
                if message:
                    return f"Gemini error: {message}"
        except Exception:
            pass
        if exc.code in {429, 500, 503, 504}:
            return "Gemini service unavailable or rate-limited."
        return f"Gemini HTTP error: {exc.code}"

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
