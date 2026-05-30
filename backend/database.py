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
                role TEXT NOT NULL DEFAULT 'user',
                created_at TEXT NOT NULL
            )
            """
        )
        user_columns = {
            str(row[1]).strip().lower()
            for row in cur.execute("PRAGMA table_info(users)").fetchall()
        }
        if "role" not in user_columns:
            cur.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'")
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
            CREATE TABLE IF NOT EXISTS docs_registry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                absolute_path TEXT NOT NULL UNIQUE,
                doc_id TEXT,
                file_hash TEXT,
                text_hash TEXT,
                size_bytes INTEGER NOT NULL DEFAULT 0,
                modified_at TEXT,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                last_ingested_at TEXT,
                is_indexed INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'discovered',
                last_error TEXT,
                duplicate_override INTEGER NOT NULL DEFAULT 0,
                override_reason TEXT,
                override_by TEXT,
                override_at TEXT
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_docs_registry_hash ON docs_registry(file_hash)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_docs_registry_doc_id ON docs_registry(doc_id)")
        docs_registry_columns = {
            str(row[1]).strip().lower()
            for row in cur.execute("PRAGMA table_info(docs_registry)").fetchall()
        }
        if "text_hash" not in docs_registry_columns:
            cur.execute("ALTER TABLE docs_registry ADD COLUMN text_hash TEXT")
        if "duplicate_override" not in docs_registry_columns:
            cur.execute("ALTER TABLE docs_registry ADD COLUMN duplicate_override INTEGER NOT NULL DEFAULT 0")
        if "override_reason" not in docs_registry_columns:
            cur.execute("ALTER TABLE docs_registry ADD COLUMN override_reason TEXT")
        if "override_by" not in docs_registry_columns:
            cur.execute("ALTER TABLE docs_registry ADD COLUMN override_by TEXT")
        if "override_at" not in docs_registry_columns:
            cur.execute("ALTER TABLE docs_registry ADD COLUMN override_at TEXT")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_docs_registry_text_hash ON docs_registry(text_hash)")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ingestion_jobs (
                job_id TEXT PRIMARY KEY,
                owner_user_id TEXT NOT NULL,
                job_type TEXT NOT NULL,
                status TEXT NOT NULL,
                progress_percent INTEGER NOT NULL DEFAULT 0,
                message TEXT,
                error TEXT,
                input_json TEXT,
                result_json TEXT,
                retry_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_owner ON ingestion_jobs(owner_user_id, created_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_status ON ingestion_jobs(status, updated_at)")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                actor_user_id TEXT,
                actor_email TEXT,
                target_type TEXT,
                target_id TEXT,
                status TEXT NOT NULL,
                payload_json TEXT,
                result_json TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_audit_events_created ON audit_events(created_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_audit_events_actor ON audit_events(actor_user_id, created_at)")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS auth_login_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                ip_address TEXT,
                success INTEGER NOT NULL DEFAULT 0,
                attempted_at TEXT NOT NULL
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_auth_attempts_email_time ON auth_login_attempts(email, attempted_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_auth_attempts_ip_time ON auth_login_attempts(ip_address, attempted_at)")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS rate_limit_hits (
                scope TEXT NOT NULL,
                rl_key TEXT NOT NULL,
                window_start_epoch INTEGER NOT NULL,
                count INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (scope, rl_key, window_start_epoch)
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rate_limit_updated ON rate_limit_hits(updated_at)")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS monitoring_metrics (
                metric_name TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                labels_json TEXT NOT NULL DEFAULT '{}',
                value REAL NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (metric_name, labels_json)
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_monitoring_metrics_updated ON monitoring_metrics(updated_at)")
        # Immutable audit trail: block updates/deletes once written.
        cur.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_audit_events_no_update
            BEFORE UPDATE ON audit_events
            BEGIN
                SELECT RAISE(FAIL, 'audit_events is immutable');
            END;
            """
        )
        cur.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_audit_events_no_delete
            BEFORE DELETE ON audit_events
            BEGIN
                SELECT RAISE(FAIL, 'audit_events is immutable');
            END;
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
