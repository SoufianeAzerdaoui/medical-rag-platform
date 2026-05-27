from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from backend.config import APP_DB_PATH


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def now_iso() -> str:
    return utc_now().isoformat()


def db_connect() -> sqlite3.Connection:
    APP_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(APP_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_schema() -> None:
    conn = db_connect()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                title_source TEXT NOT NULL DEFAULT 'auto',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )
        conversation_columns = {
            str(row[1]).strip().lower()
            for row in cur.execute("PRAGMA table_info(conversations)").fetchall()
        }
        if "title_source" not in conversation_columns:
            cur.execute(
                "ALTER TABLE conversations ADD COLUMN title_source TEXT NOT NULL DEFAULT 'auto'"
            )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources_json TEXT,
                diagnostics_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
            """
        )
        message_columns = {
            str(row[1]).strip().lower()
            for row in cur.execute("PRAGMA table_info(messages)").fetchall()
        }
        if "sources_json" not in message_columns:
            cur.execute("ALTER TABLE messages ADD COLUMN sources_json TEXT")
        if "diagnostics_json" not in message_columns:
            cur.execute("ALTER TABLE messages ADD COLUMN diagnostics_json TEXT")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_states (
                conversation_id TEXT PRIMARY KEY,
                state_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS feature_flags (
                name TEXT PRIMARY KEY,
                enabled INTEGER NOT NULL,
                description TEXT,
                updated_at TEXT NOT NULL,
                updated_by TEXT
            )
            """
        )
        cur.execute(
            """
            INSERT INTO feature_flags (name, enabled, description, updated_at, updated_by)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(name) DO NOTHING
            """,
            (
                "REFERENCE_RANGE_STRICT_MODE",
                1,
                "Enable strict deterministic reference range lookup flow",
                now_iso(),
                "system:init_schema",
            ),
        )
        conn.commit()
    finally:
        conn.close()
