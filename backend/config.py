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

APP_DB_PATH = Path(os.getenv("APP_DB_PATH", str(ROOT_DIR / "data" / "app_state.sqlite3"))).resolve()
JWT_SECRET = os.getenv("JWT_SECRET", "pfe-medical-rag-dev-secret-change-me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "720"))
FRONTEND_ORIGIN = str(os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")).strip()
