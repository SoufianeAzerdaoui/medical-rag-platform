from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import sqlite3

from backend import config
from backend.database import db_connect


def _iso_cutoff(days: int) -> str:
    safe_days = max(0, int(days))
    dt = datetime.now(timezone.utc) - timedelta(days=safe_days)
    return dt.isoformat()


def _safe_unlink(path: Path) -> bool:
    try:
        if path.exists() and path.is_file():
            path.unlink()
            return True
    except Exception:
        return False
    return False


def run_retention(*, hard_delete_docs: bool = False, dry_run: bool = False) -> dict[str, Any]:
    docs_dir = (config.ROOT_DIR / "docs").resolve()
    audio_dir = config.AUDIO_STORAGE_DIR.resolve()
    logs_dir = config.LOGS_DIR.resolve()
    now = datetime.now(timezone.utc)

    cut_jobs = _iso_cutoff(config.RETENTION_JOBS_DAYS)
    cut_audit = _iso_cutoff(config.RETENTION_AUDIT_DAYS)
    cut_docs = _iso_cutoff(config.RETENTION_DOCS_DAYS)
    cut_auth = _iso_cutoff(config.RETENTION_AUTH_ATTEMPTS_DAYS)
    cut_audio = now - timedelta(days=max(0, int(config.RETENTION_AUDIO_DAYS)))
    cut_logs = now - timedelta(days=max(0, int(config.RETENTION_LOGS_DAYS)))

    result: dict[str, Any] = {
        "dry_run": bool(dry_run),
        "hard_delete_docs": bool(hard_delete_docs),
        "jobs_deleted": 0,
        "audit_deleted": 0,
        "auth_attempts_deleted": 0,
        "docs_registry_deleted": 0,
        "docs_files_deleted": 0,
        "audio_files_deleted": 0,
        "log_files_deleted": 0,
        "audit_delete_blocked_immutable": False,
    }

    conn = db_connect()
    try:
        cur = conn.cursor()
        if dry_run:
            row = cur.execute("SELECT COUNT(*) FROM ingestion_jobs WHERE created_at < ?", (cut_jobs,)).fetchone()
            result["jobs_deleted"] = int(row[0] if row else 0)
            row = cur.execute("SELECT COUNT(*) FROM audit_events WHERE created_at < ?", (cut_audit,)).fetchone()
            result["audit_deleted"] = int(row[0] if row else 0)
            row = cur.execute("SELECT COUNT(*) FROM auth_login_attempts WHERE attempted_at < ?", (cut_auth,)).fetchone()
            result["auth_attempts_deleted"] = int(row[0] if row else 0)
            row = cur.execute(
                """
                SELECT COUNT(*) FROM docs_registry
                WHERE last_seen_at < ?
                  AND (is_indexed = 0 OR status IN ('missing', 'error', 'discovered'))
                """,
                (cut_docs,),
            ).fetchone()
            result["docs_registry_deleted"] = int(row[0] if row else 0)
        else:
            cur.execute("DELETE FROM ingestion_jobs WHERE created_at < ?", (cut_jobs,))
            result["jobs_deleted"] = int(cur.rowcount or 0)
            try:
                cur.execute("DELETE FROM audit_events WHERE created_at < ?", (cut_audit,))
                result["audit_deleted"] = int(cur.rowcount or 0)
            except sqlite3.OperationalError:
                result["audit_delete_blocked_immutable"] = True
            cur.execute("DELETE FROM auth_login_attempts WHERE attempted_at < ?", (cut_auth,))
            result["auth_attempts_deleted"] = int(cur.rowcount or 0)

            rows = cur.execute(
                """
                SELECT absolute_path FROM docs_registry
                WHERE last_seen_at < ?
                  AND (is_indexed = 0 OR status IN ('missing', 'error', 'discovered'))
                """,
                (cut_docs,),
            ).fetchall()
            paths = [Path(str(r["absolute_path"] or "")).resolve() for r in rows]
            if hard_delete_docs:
                for p in paths:
                    try:
                        p.relative_to(docs_dir)
                    except Exception:
                        continue
                    if _safe_unlink(p):
                        result["docs_files_deleted"] += 1
            cur.execute(
                """
                DELETE FROM docs_registry
                WHERE last_seen_at < ?
                  AND (is_indexed = 0 OR status IN ('missing', 'error', 'discovered'))
                """,
                (cut_docs,),
            )
            result["docs_registry_deleted"] = int(cur.rowcount or 0)
            conn.commit()
    finally:
        conn.close()

    # File-system retention for audio/logs.
    for root, key, cutoff in (
        (audio_dir, "audio_files_deleted", cut_audio),
        (logs_dir, "log_files_deleted", cut_logs),
    ):
        if not root.exists() or not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            try:
                mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            except Exception:
                continue
            if mtime >= cutoff:
                continue
            if dry_run:
                result[key] += 1
            else:
                if _safe_unlink(path):
                    result[key] += 1

    return result
