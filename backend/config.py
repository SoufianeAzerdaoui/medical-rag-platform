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
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "720"))
FRONTEND_ORIGIN = str(os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")).strip()
ENABLE_FEATURE_FLAG_ADMIN_API = str(os.getenv("ENABLE_FEATURE_FLAG_ADMIN_API", "false")).strip().lower() in {"1", "true", "yes", "on"}
ADMIN_EMAILS = tuple(
    email.strip().lower()
    for email in str(os.getenv("ADMIN_EMAILS", "")).split(",")
    if email.strip()
)
