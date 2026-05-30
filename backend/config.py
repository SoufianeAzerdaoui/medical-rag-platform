from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
GENERATION_DIR = SCRIPTS_DIR / "generation"

for raw_path in (str(ROOT_DIR), str(SCRIPTS_DIR), str(GENERATION_DIR)):
    if raw_path not in sys.path:
        sys.path.insert(0, raw_path)

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


def _load_environment_files() -> None:
    if load_dotenv is None:
        return
    root_env = ROOT_DIR / ".env"
    generation_env = GENERATION_DIR / ".env"
    # Root .env is the recommended source for backend runtime config.
    # Use override=False so explicit system env vars keep precedence.
    if root_env.exists():
        load_dotenv(root_env, override=False)
    # Optional fallback for legacy dev setup; does not override root/system values.
    if generation_env.exists():
        load_dotenv(generation_env, override=False)


_load_environment_files()

APP_DB_PATH = Path(os.getenv("APP_DB_PATH", str(ROOT_DIR / "data" / "app_state.sqlite3"))).resolve()
JWT_SECRET = os.getenv("JWT_SECRET", "pfe-medical-rag-dev-secret-change-me")
JWT_SECRET_PREVIOUS = tuple(
    value.strip()
    for value in str(os.getenv("JWT_SECRET_PREVIOUS", "")).split(",")
    if value.strip()
)
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "720"))
FRONTEND_ORIGIN = str(os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")).strip()
ENABLE_FEATURE_FLAG_ADMIN_API = str(os.getenv("ENABLE_FEATURE_FLAG_ADMIN_API", "false")).strip().lower() in {"1", "true", "yes", "on"}
ADMIN_EMAILS = tuple(
    email.strip().lower()
    for email in str(os.getenv("ADMIN_EMAILS", "")).split(",")
    if email.strip()
)

# Application-level encryption (optional, strongly recommended in production).
DATA_ENCRYPTION_ENABLED = str(os.getenv("DATA_ENCRYPTION_ENABLED", "false")).strip().lower() in {"1", "true", "yes", "on"}
DATA_ENCRYPTION_KEY = str(os.getenv("DATA_ENCRYPTION_KEY", "")).strip()
DATA_ENCRYPTION_REQUIRED = str(os.getenv("DATA_ENCRYPTION_REQUIRED", "false")).strip().lower() in {"1", "true", "yes", "on"}

# Auth brute-force protection.
AUTH_LOGIN_WINDOW_SECONDS = int(os.getenv("AUTH_LOGIN_WINDOW_SECONDS", "900"))
AUTH_LOGIN_MAX_FAILURES = int(os.getenv("AUTH_LOGIN_MAX_FAILURES", "5"))
AUTH_LOGIN_BLOCK_SECONDS = int(os.getenv("AUTH_LOGIN_BLOCK_SECONDS", "900"))

# Generic request rate limiting.
RATE_LIMIT_AUTH_PER_WINDOW = int(os.getenv("RATE_LIMIT_AUTH_PER_WINDOW", "20"))
RATE_LIMIT_CHAT_PER_WINDOW = int(os.getenv("RATE_LIMIT_CHAT_PER_WINDOW", "80"))
RATE_LIMIT_UPLOAD_PER_WINDOW = int(os.getenv("RATE_LIMIT_UPLOAD_PER_WINDOW", "25"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

# Antivirus checks (ClamAV).
ANTIVIRUS_REQUIRED = str(os.getenv("ANTIVIRUS_REQUIRED", "false")).strip().lower() in {"1", "true", "yes", "on"}
ANTIVIRUS_CLAMSCAN_CMD = str(os.getenv("ANTIVIRUS_CLAMSCAN_CMD", "clamscan")).strip() or "clamscan"
ANTIVIRUS_TIMEOUT_SECONDS = int(os.getenv("ANTIVIRUS_TIMEOUT_SECONDS", "20"))

# Retention policy.
RETENTION_JOBS_DAYS = int(os.getenv("RETENTION_JOBS_DAYS", "30"))
RETENTION_AUDIT_DAYS = int(os.getenv("RETENTION_AUDIT_DAYS", "180"))
RETENTION_DOCS_DAYS = int(os.getenv("RETENTION_DOCS_DAYS", "90"))
RETENTION_AUDIO_DAYS = int(os.getenv("RETENTION_AUDIO_DAYS", "7"))
RETENTION_LOGS_DAYS = int(os.getenv("RETENTION_LOGS_DAYS", "30"))
RETENTION_AUTH_ATTEMPTS_DAYS = int(os.getenv("RETENTION_AUTH_ATTEMPTS_DAYS", "14"))

AUDIO_STORAGE_DIR = Path(os.getenv("AUDIO_STORAGE_DIR", str(ROOT_DIR / "data" / "audio"))).resolve()
LOGS_DIR = Path(os.getenv("LOGS_DIR", str(ROOT_DIR / "logs"))).resolve()

# Alerting
SENTRY_DSN = str(os.getenv("SENTRY_DSN", "")).strip()

# Production hard gate
PROD_READINESS_ENFORCE = str(os.getenv("PROD_READINESS_ENFORCE", "false")).strip().lower() in {"1", "true", "yes", "on"}
