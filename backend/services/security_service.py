from __future__ import annotations

import base64
import json
from typing import Any

from backend import config

try:  # pragma: no cover - optional dependency
    from cryptography.fernet import Fernet, InvalidToken  # type: ignore
except Exception:  # pragma: no cover
    Fernet = None  # type: ignore
    InvalidToken = Exception  # type: ignore


def _fernet() -> Fernet | None:
    if not config.DATA_ENCRYPTION_ENABLED:
        return None
    key = str(config.DATA_ENCRYPTION_KEY or "").strip()
    if not key:
        if config.DATA_ENCRYPTION_REQUIRED:
            raise RuntimeError("DATA_ENCRYPTION_ENABLED=true mais DATA_ENCRYPTION_KEY absent.")
        return None
    if Fernet is None:
        if config.DATA_ENCRYPTION_REQUIRED:
            raise RuntimeError("cryptography absent alors que DATA_ENCRYPTION_REQUIRED=true.")
        return None
    try:
        return Fernet(key.encode("utf-8"))
    except Exception as exc:
        if config.DATA_ENCRYPTION_REQUIRED:
            raise RuntimeError("DATA_ENCRYPTION_KEY invalide (Fernet).") from exc
        return None


def encrypt_json(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload or {}, ensure_ascii=False, sort_keys=True)
    cipher = _fernet()
    if cipher is None:
        return raw
    token = cipher.encrypt(raw.encode("utf-8"))
    return "enc:v1:" + base64.urlsafe_b64encode(token).decode("ascii")


def decrypt_json(raw: str | None) -> dict[str, Any]:
    text = str(raw or "")
    if not text:
        return {}
    if not text.startswith("enc:v1:"):
        try:
            value = json.loads(text)
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}
    cipher = _fernet()
    if cipher is None:
        # Fail closed in strict mode, but tolerate in non-strict to avoid total outage.
        if config.DATA_ENCRYPTION_REQUIRED:
            raise RuntimeError("Impossible de déchiffrer: chiffrement requis mais indisponible.")
        return {}
    encoded = text[len("enc:v1:") :]
    try:
        token = base64.urlsafe_b64decode(encoded.encode("ascii"))
        clear = cipher.decrypt(token).decode("utf-8")
        value = json.loads(clear)
        return value if isinstance(value, dict) else {}
    except (InvalidToken, ValueError, json.JSONDecodeError):
        if config.DATA_ENCRYPTION_REQUIRED:
            raise
        return {}

