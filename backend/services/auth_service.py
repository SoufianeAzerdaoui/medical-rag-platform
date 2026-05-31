from __future__ import annotations

from typing import Any
from uuid import uuid4

import bcrypt
import jwt
from datetime import datetime, timezone
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.config import (
    ADMIN_EMAILS,
    AUTH_LOGIN_BLOCK_SECONDS,
    AUTH_LOGIN_MAX_FAILURES,
    AUTH_LOGIN_WINDOW_SECONDS,
    JWT_ALGORITHM,
    JWT_EXPIRE_MINUTES,
    JWT_SECRET,
    JWT_SECRET_PREVIOUS,
)
from backend.database import db_connect, now_iso, utc_now

AUTH_SCHEME = HTTPBearer(auto_error=False)


def normalize_email(email: str) -> str:
    return str(email or "").strip().lower()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(str(password).encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(str(password).encode("utf-8"), str(password_hash).encode("utf-8"))
    except Exception:
        return False


def create_access_token(user_id: str) -> str:
    now = utc_now()
    payload = {
        "sub": str(user_id),
        "iat": int(now.timestamp()),
        "exp": int((now.timestamp()) + (JWT_EXPIRE_MINUTES * 60)),
    }
    return str(jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM))


def decode_token(token: str) -> dict[str, Any]:
    secrets = [str(JWT_SECRET)] + [str(value) for value in JWT_SECRET_PREVIOUS]
    for secret in secrets:
        try:
            decoded = jwt.decode(token, secret, algorithms=[JWT_ALGORITHM])
        except jwt.PyJWTError:
            continue
        if isinstance(decoded, dict):
            return decoded
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token invalide")


def _record_login_attempt(*, email: str, ip_address: str | None, success: bool) -> None:
    conn = db_connect()
    try:
        conn.execute(
            """
            INSERT INTO auth_login_attempts (email, ip_address, success, attempted_at)
            VALUES (?, ?, ?, ?)
            """,
            (normalize_email(email), str(ip_address or "").strip() or None, 1 if success else 0, now_iso()),
        )
        conn.commit()
    finally:
        conn.close()


def _recent_failures(*, email: str, ip_address: str | None) -> int:
    cutoff_ts = utc_now().timestamp() - max(60, int(AUTH_LOGIN_WINDOW_SECONDS))
    cutoff = datetime.fromtimestamp(cutoff_ts, tz=timezone.utc).isoformat()
    conn = db_connect()
    try:
        row = conn.execute(
            """
            SELECT COUNT(*) AS c
            FROM auth_login_attempts
            WHERE success = 0
              AND attempted_at >= ?
              AND (
                email = ?
                OR (ip_address IS NOT NULL AND ip_address = ?)
              )
            """,
            (cutoff, normalize_email(email), str(ip_address or "").strip() or None),
        ).fetchone()
    finally:
        conn.close()
    return int(row["c"] if row else 0)


def _last_failed_attempt_iso(*, email: str, ip_address: str | None) -> str | None:
    conn = db_connect()
    try:
        row = conn.execute(
            """
            SELECT attempted_at
            FROM auth_login_attempts
            WHERE success = 0
              AND (
                email = ?
                OR (ip_address IS NOT NULL AND ip_address = ?)
              )
            ORDER BY attempted_at DESC
            LIMIT 1
            """,
            (normalize_email(email), str(ip_address or "").strip() or None),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    value = str(row["attempted_at"] or "").strip()
    return value or None


def _assert_login_not_blocked(*, email: str, ip_address: str | None) -> None:
    failures = _recent_failures(email=email, ip_address=ip_address)
    if failures < max(1, int(AUTH_LOGIN_MAX_FAILURES)):
        return
    last = _last_failed_attempt_iso(email=email, ip_address=ip_address)
    if not last:
        return
    try:
        last_dt = datetime.fromisoformat(last)
    except Exception:
        return
    blocked_for = max(30, int(AUTH_LOGIN_BLOCK_SECONDS))
    remaining = int((last_dt.timestamp() + blocked_for) - utc_now().timestamp())
    if remaining > 0:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Trop de tentatives de connexion. Réessaie dans {remaining}s.",
        )


def get_user_by_id(user_id: str) -> dict[str, Any] | None:
    conn = db_connect()
    try:
        row = conn.execute(
            "SELECT id, email, password_hash, role, created_at FROM users WHERE id = ?",
            (str(user_id),),
        ).fetchone()
    finally:
        conn.close()
    return dict(row) if row is not None else None


def get_user_by_email(email: str) -> dict[str, Any] | None:
    conn = db_connect()
    try:
        row = conn.execute(
            "SELECT id, email, password_hash, role, created_at FROM users WHERE email = ?",
            (normalize_email(email),),
        ).fetchone()
    finally:
        conn.close()
    return dict(row) if row is not None else None


def register_user(*, email: str, password: str) -> dict[str, Any]:
    normalized = normalize_email(email)
    if not normalized or "@" not in normalized:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email invalide")
    if get_user_by_email(normalized):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email déjà utilisé")

    user_id = f"user_{uuid4()}"
    created_at = now_iso()
    password_hash = hash_password(password)
    role = "admin" if normalize_email(normalized) in set(ADMIN_EMAILS) else "user"

    conn = db_connect()
    try:
        conn.execute(
            "INSERT INTO users (id, email, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, normalized, password_hash, role, created_at),
        )
        conn.commit()
    finally:
        conn.close()

    return {"id": user_id, "email": normalized, "role": role, "created_at": created_at}


def login_user(*, email: str, password: str, ip_address: str | None = None) -> dict[str, Any]:
    normalized = normalize_email(email)
    _assert_login_not_blocked(email=normalized, ip_address=ip_address)
    user = get_user_by_email(email)
    if not user or not verify_password(password, str(user.get("password_hash") or "")):
        _record_login_attempt(email=normalized, ip_address=ip_address, success=False)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Identifiants invalides")
    _record_login_attempt(email=normalized, ip_address=ip_address, success=True)
    return user


def get_current_user(credentials: HTTPAuthorizationCredentials | None = Depends(AUTH_SCHEME)) -> dict[str, Any]:
    if credentials is None or str(credentials.scheme).lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentification requise")
    payload = decode_token(credentials.credentials)
    user_id = str(payload.get("sub") or "").strip()
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token invalide")

    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Utilisateur introuvable")
    return user
