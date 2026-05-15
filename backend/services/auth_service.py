from __future__ import annotations

from typing import Any
from uuid import uuid4

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.config import JWT_ALGORITHM, JWT_EXPIRE_MINUTES, JWT_SECRET
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
    try:
        decoded = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token invalide") from exc
    if not isinstance(decoded, dict):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token invalide")
    return decoded


def get_user_by_id(user_id: str) -> dict[str, Any] | None:
    conn = db_connect()
    try:
        row = conn.execute(
            "SELECT id, email, password_hash, created_at FROM users WHERE id = ?",
            (str(user_id),),
        ).fetchone()
    finally:
        conn.close()
    return dict(row) if row is not None else None


def get_user_by_email(email: str) -> dict[str, Any] | None:
    conn = db_connect()
    try:
        row = conn.execute(
            "SELECT id, email, password_hash, created_at FROM users WHERE email = ?",
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

    conn = db_connect()
    try:
        conn.execute(
            "INSERT INTO users (id, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (user_id, normalized, password_hash, created_at),
        )
        conn.commit()
    finally:
        conn.close()

    return {"id": user_id, "email": normalized, "created_at": created_at}


def login_user(*, email: str, password: str) -> dict[str, Any]:
    user = get_user_by_email(email)
    if not user or not verify_password(password, str(user.get("password_hash") or "")):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Identifiants invalides")
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
