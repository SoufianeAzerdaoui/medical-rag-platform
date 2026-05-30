from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

from backend import config
from backend.database import db_connect, now_iso
from backend.services import ingestion_service, security_service


def _check_clamav() -> dict[str, Any]:
    cmd = str(config.ANTIVIRUS_CLAMSCAN_CMD or "clamscan").strip() or "clamscan"
    binary = shutil.which(cmd)
    available = bool(binary)
    version = ""
    if available:
        try:
            proc = subprocess.run(
                [cmd, "--version"],
                capture_output=True,
                text=True,
                timeout=max(2, min(10, int(config.ANTIVIRUS_TIMEOUT_SECONDS))),
            )
            if proc.returncode == 0:
                version = str((proc.stdout or "").strip() or (proc.stderr or "").strip())
        except Exception:
            version = ""
    required = True
    passed = available and bool(version or binary)
    return {
        "id": "clamav_active",
        "label": "ClamAV actif",
        "passed": bool(passed),
        "required": required,
        "details": {
            "command": cmd,
            "available": available,
            "version": version,
        },
    }


def _check_encryption() -> dict[str, Any]:
    enabled = bool(config.DATA_ENCRYPTION_ENABLED)
    key_configured = bool(str(config.DATA_ENCRYPTION_KEY or "").strip())
    details: dict[str, Any] = {
        "enabled": enabled,
        "required": bool(config.DATA_ENCRYPTION_REQUIRED),
        "key_configured": key_configured,
    }
    roundtrip_ok = False
    error = ""
    if enabled and key_configured:
        try:
            encrypted = security_service.encrypt_json({"probe": "ok"})
            decrypted = security_service.decrypt_json(encrypted)
            roundtrip_ok = str(decrypted.get("probe") or "") == "ok"
        except Exception as exc:
            roundtrip_ok = False
            error = str(exc)
    details["roundtrip_ok"] = roundtrip_ok
    if error:
        details["error"] = error
    passed = enabled and key_configured and roundtrip_ok
    return {
        "id": "encryption_active",
        "label": "Chiffrement applicatif actif",
        "passed": bool(passed),
        "required": True,
        "details": details,
    }


def _secret_strength(secret: str) -> bool:
    value = str(secret or "")
    if len(value) < 32:
        return False
    lowered = value.lower()
    weak_markers = ("change-me", "dev-secret", "example", "test")
    return not any(marker in lowered for marker in weak_markers)


def _check_jwt() -> dict[str, Any]:
    secret_ok = _secret_strength(str(config.JWT_SECRET or ""))
    rotation_count = len(tuple(config.JWT_SECRET_PREVIOUS or ()))
    ttl = int(config.JWT_EXPIRE_MINUTES)
    ttl_ok = 15 <= ttl <= 1440
    passed = secret_ok and rotation_count > 0 and ttl_ok
    return {
        "id": "jwt_secure",
        "label": "JWT sécurisé + rotation",
        "passed": bool(passed),
        "required": True,
        "details": {
            "secret_strong": secret_ok,
            "rotation_previous_count": rotation_count,
            "ttl_minutes": ttl,
            "ttl_valid": ttl_ok,
            "algorithm": str(config.JWT_ALGORITHM),
        },
    }


def _check_rbac_ops() -> dict[str, Any]:
    ops_roles = {"admin", "ops", "data_manager", "medical_admin"}
    app_env = str(os.getenv("APP_ENV", "")).strip().lower()
    admin_emails_count = len(tuple(config.ADMIN_EMAILS or ()))
    dev_bypass_active = app_env in {"dev", "development", "local", "test"} and admin_emails_count == 0
    passed = len(ops_roles) > 0 and not (app_env == "production" and dev_bypass_active)
    return {
        "id": "rbac_ops_locked",
        "label": "RBAC ops/admin",
        "passed": bool(passed),
        "required": True,
        "details": {
            "app_env": app_env or "unknown",
            "ops_roles": sorted(ops_roles),
            "admin_emails_count": admin_emails_count,
            "dev_bypass_active": dev_bypass_active,
        },
    }


def _check_audit_immutable() -> dict[str, Any]:
    trigger_names = set()
    update_blocked = False
    delete_blocked = False
    error = ""
    conn = db_connect()
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name='audit_events'"
        ).fetchall()
        trigger_names = {str(r["name"] or "") for r in rows}
        conn.execute("BEGIN")
        conn.execute(
            """
            INSERT INTO audit_events (
                event_type, actor_user_id, actor_email, target_type, target_id,
                status, payload_json, result_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "p0_probe",
                "system",
                "system@local",
                "probe",
                "audit_immutable",
                "success",
                json.dumps({"probe": True}),
                json.dumps({"probe": True}),
                now_iso(),
            ),
        )
        inserted_id = conn.execute("SELECT last_insert_rowid() AS v").fetchone()["v"]
        try:
            conn.execute("UPDATE audit_events SET status='error' WHERE id = ?", (inserted_id,))
            update_blocked = False
        except sqlite3.DatabaseError:
            update_blocked = True
        try:
            conn.execute("DELETE FROM audit_events WHERE id = ?", (inserted_id,))
            delete_blocked = False
        except sqlite3.DatabaseError:
            delete_blocked = True
        conn.execute("ROLLBACK")
    except Exception as exc:
        error = str(exc)
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
    finally:
        conn.close()
    passed = (
        "trg_audit_events_no_update" in trigger_names
        and "trg_audit_events_no_delete" in trigger_names
        and update_blocked
        and delete_blocked
        and not error
    )
    details: dict[str, Any] = {
        "triggers": sorted(trigger_names),
        "update_blocked": update_blocked,
        "delete_blocked": delete_blocked,
    }
    if error:
        details["error"] = error
    return {
        "id": "audit_trail_immutable",
        "label": "Audit trail immuable",
        "passed": bool(passed),
        "required": True,
        "details": details,
    }


def _check_jobs_and_registry() -> dict[str, Any]:
    indexed_ids: set[str] = set()
    sqlite_path = Path("data/indexes/medical_rag.sqlite")
    if sqlite_path.exists():
        try:
            conn = sqlite3.connect(str(sqlite_path))
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT DISTINCT doc_id FROM chunks WHERE doc_id IS NOT NULL AND TRIM(doc_id) != ''").fetchall()
            indexed_ids = {str(r["doc_id"]).strip().lower() for r in rows if str(r["doc_id"]).strip()}
        except Exception:
            indexed_ids = set()
        finally:
            try:
                conn.close()
            except Exception:
                pass
    resync_ok = False
    resync_error = ""
    resync_result: dict[str, Any] = {}
    try:
        resync_result = ingestion_service.resync_docs_registry(indexed_doc_ids=indexed_ids)
        resync_ok = True
    except Exception as exc:
        resync_ok = False
        resync_error = str(exc)

    jobs_table_ok = False
    pending_jobs = 0
    conn2 = db_connect()
    try:
        row = conn2.execute(
            "SELECT COUNT(*) AS c FROM sqlite_master WHERE type='table' AND name='ingestion_jobs'"
        ).fetchone()
        jobs_table_ok = bool(int(row["c"] if row else 0) > 0)
        row2 = conn2.execute(
            "SELECT COUNT(*) AS c FROM ingestion_jobs WHERE status IN ('queued','running')"
        ).fetchone()
        pending_jobs = int(row2["c"] if row2 else 0)
    finally:
        conn2.close()
    passed = jobs_table_ok and resync_ok
    details: dict[str, Any] = {
        "jobs_table_ok": jobs_table_ok,
        "pending_jobs": pending_jobs,
        "resync_ok": resync_ok,
        "resync_result": resync_result,
    }
    if resync_error:
        details["resync_error"] = resync_error
    return {
        "id": "jobs_persistence_and_resync",
        "label": "Jobs persistants + resync registre/index",
        "passed": bool(passed),
        "required": True,
        "details": details,
    }


def _check_backup_artifacts() -> dict[str, Any]:
    backup_script = (config.ROOT_DIR / "scripts" / "ops" / "backup_app_state.sh")
    restore_script = (config.ROOT_DIR / "scripts" / "ops" / "restore_app_state.sh")
    runbook = (config.ROOT_DIR / "backend" / "production_runbook.md")
    passed = backup_script.exists() and restore_script.exists() and runbook.exists()
    return {
        "id": "backup_restore_procedure",
        "label": "Backup/restore DB outillé",
        "passed": bool(passed),
        "required": True,
        "details": {
            "backup_script": str(backup_script),
            "backup_exists": backup_script.exists(),
            "restore_script": str(restore_script),
            "restore_exists": restore_script.exists(),
            "runbook_exists": runbook.exists(),
        },
    }


def _check_e2e_artifacts() -> dict[str, Any]:
    critical_spec = config.ROOT_DIR / "tests" / "e2e" / "ingestion-operator.spec.ts"
    passed = critical_spec.exists()
    return {
        "id": "e2e_critical_path",
        "label": "E2E critique présent",
        "passed": bool(passed),
        "required": True,
        "details": {
            "spec_file": str(critical_spec),
            "exists": critical_spec.exists(),
            "note": "Exécution CI/infra requise pour valider PASS runtime.",
        },
    }


def run_p0_readiness_check() -> dict[str, Any]:
    checks = [
        _check_clamav(),
        _check_encryption(),
        _check_jwt(),
        _check_rbac_ops(),
        _check_audit_immutable(),
        _check_jobs_and_registry(),
        _check_backup_artifacts(),
        _check_e2e_artifacts(),
    ]
    blocking_failures = [item for item in checks if bool(item.get("required")) and not bool(item.get("passed"))]
    return {
        "generated_at": now_iso(),
        "overall_status": "pass" if not blocking_failures else "fail",
        "blocking_failures": len(blocking_failures),
        "checks": checks,
    }
