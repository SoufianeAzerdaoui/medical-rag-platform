from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from backend import config
from backend.database import db_connect, now_iso
from backend.services import audit_service, ingestion_service, monitoring_service, security_service


JobStatus = Literal["queued", "running", "success", "error"]
JOB_TYPE_DOCS_INGESTION = "docs_ingestion"


@dataclass
class IngestionJob:
    job_id: str
    owner_user_id: str
    status: JobStatus
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    message: str | None = None
    error: str | None = None
    progress_percent: int = 0
    result: dict[str, Any] | None = None


class IngestionJobService:
    def __init__(self, *, max_jobs: int = 500) -> None:
        self._max_jobs = max(100, int(max_jobs))
        self._lock = threading.Lock()
        self._running_threads: dict[str, threading.Thread] = {}
        self._resume_incomplete_jobs()

    def start_docs_ingestion_job(self, *, owner_user_id: str, filenames: list[str]) -> IngestionJob:
        job_id = f"job_{uuid.uuid4()}"
        created_at = now_iso()
        payload = {"filenames": list(filenames)}
        conn = db_connect()
        try:
            conn.execute(
                """
                INSERT INTO ingestion_jobs (
                    job_id, owner_user_id, job_type, status, progress_percent, message, error,
                    input_json, result_json, retry_count, created_at, updated_at
                )
                VALUES (?, ?, ?, 'queued', 5, ?, NULL, ?, NULL, 0, ?, ?)
                """,
                (
                    job_id,
                    str(owner_user_id),
                    JOB_TYPE_DOCS_INGESTION,
                    "Job d’ingestion créé.",
                    security_service.encrypt_json(payload),
                    created_at,
                    created_at,
                ),
            )
            self._trim_old_jobs(conn)
            conn.commit()
        finally:
            conn.close()
        self._update_queue_depth_metric()
        self._spawn_worker(job_id)
        job = self.get_job_for_user(job_id=job_id, owner_user_id=owner_user_id)
        if not job:
            raise RuntimeError("Impossible de créer le job d’ingestion.")
        return job

    def get_job_for_user(self, *, job_id: str, owner_user_id: str) -> IngestionJob | None:
        conn = db_connect()
        try:
            row = conn.execute(
                """
                SELECT job_id, owner_user_id, status, created_at, started_at, finished_at,
                       message, error, progress_percent, result_json
                FROM ingestion_jobs
                WHERE job_id = ? AND owner_user_id = ?
                """,
                (str(job_id), str(owner_user_id)),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return self._row_to_job(dict(row))

    def _row_to_job(self, row: dict[str, Any]) -> IngestionJob:
        parsed_result: dict[str, Any] | None = None
        raw_result = row.get("result_json")
        if raw_result:
            try:
                decoded = security_service.decrypt_json(str(raw_result))
                if isinstance(decoded, dict):
                    parsed_result = decoded
            except Exception:
                parsed_result = None
        return IngestionJob(
            job_id=str(row.get("job_id") or ""),
            owner_user_id=str(row.get("owner_user_id") or ""),
            status=str(row.get("status") or "error"),  # type: ignore[arg-type]
            created_at=str(row.get("created_at") or ""),
            started_at=row.get("started_at"),
            finished_at=row.get("finished_at"),
            message=row.get("message"),
            error=row.get("error"),
            progress_percent=int(row.get("progress_percent") or 0),
            result=parsed_result,
        )

    def _spawn_worker(self, job_id: str) -> None:
        with self._lock:
            existing = self._running_threads.get(job_id)
            if existing and existing.is_alive():
                return
            worker = threading.Thread(
                target=self._run_docs_job,
                args=(job_id,),
                daemon=True,
                name=f"ingestion-job-{job_id}",
            )
            self._running_threads[job_id] = worker
            worker.start()

    def _run_docs_job(self, job_id: str) -> None:
        owner_user_id = ""
        try:
            conn = db_connect()
            try:
                row = conn.execute(
                    """
                    SELECT owner_user_id, status, retry_count, input_json
                    FROM ingestion_jobs
                    WHERE job_id = ? AND job_type = ?
                    """,
                    (str(job_id), JOB_TYPE_DOCS_INGESTION),
                ).fetchone()
                if row is None:
                    return
                owner_user_id = str(row["owner_user_id"] or "")
                retry_count = int(row["retry_count"] or 0) + 1
                now = now_iso()
                conn.execute(
                    """
                    UPDATE ingestion_jobs
                    SET status='running',
                        started_at=COALESCE(started_at, ?),
                        progress_percent=20,
                        message=?,
                        error=NULL,
                        retry_count=?,
                        updated_at=?
                    WHERE job_id=?
                    """,
                    (now, "Pipeline en cours d’exécution...", retry_count, now, str(job_id)),
                )
                conn.commit()
                try:
                    payload = security_service.decrypt_json(str(row["input_json"] or "{}"))
                except Exception:
                    payload = {}
            finally:
                conn.close()

            filenames_raw = payload.get("filenames") if isinstance(payload, dict) else None
            filenames = [str(x or "").strip() for x in list(filenames_raw or []) if str(x or "").strip()]
            indexed_doc_ids = self._indexed_doc_ids_from_sqlite(config.ROOT_DIR / "data" / "indexes" / "medical_rag.sqlite")
            results = ingestion_service.ingest_docs_by_filenames(filenames, indexed_doc_ids=indexed_doc_ids)
            result_payload = {
                "success": True,
                "ingested_count": len(results),
                "ingested": [
                    {
                        "filename": item.filename,
                        "doc_id": item.doc_id,
                        "stored_path": item.stored_path,
                        "extraction_dir": item.extraction_dir,
                    }
                    for item in results
                ],
                "skipped": [],
            }
            conn = db_connect()
            try:
                now = now_iso()
                conn.execute(
                    """
                    UPDATE ingestion_jobs
                    SET status='success',
                        finished_at=?,
                        progress_percent=100,
                        message=?,
                        error=NULL,
                        result_json=?,
                        updated_at=?
                    WHERE job_id=?
                    """,
                    (
                        now,
                        f"Ingestion terminée ({len(results)} document(s)).",
                        security_service.encrypt_json(result_payload),
                        now,
                        str(job_id),
                    ),
                )
                conn.commit()
            finally:
                conn.close()
            monitoring_service.inc("ingestion_pipeline_success_total", 1)
            self._observe_job_duration(job_id)
            audit_service.log_event(
                event_type="ingestion_job_completed",
                actor_user_id=owner_user_id,
                actor_email=self._user_email_for_id(owner_user_id),
                target_type="ingestion_job",
                target_id=str(job_id),
                status="success",
                payload={"filenames": filenames},
                result={"ingested_count": len(results)},
            )
        except Exception as exc:
            conn = db_connect()
            try:
                now = now_iso()
                conn.execute(
                    """
                    UPDATE ingestion_jobs
                    SET status='error',
                        finished_at=?,
                        progress_percent=100,
                        message=?,
                        error=?,
                        updated_at=?
                    WHERE job_id=?
                    """,
                    (now, "Le pipeline a échoué.", str(exc), now, str(job_id)),
                )
                conn.commit()
            finally:
                conn.close()
            monitoring_service.inc("ingestion_pipeline_failure_total", 1)
            if "index" in str(exc).lower():
                monitoring_service.inc("ingestion_indexing_errors_total", 1)
            self._observe_job_duration(job_id)
            audit_service.log_event(
                event_type="ingestion_job_failed",
                actor_user_id=owner_user_id,
                actor_email=self._user_email_for_id(owner_user_id),
                target_type="ingestion_job",
                target_id=str(job_id),
                status="error",
                payload={"job_id": str(job_id)},
                result={"error": str(exc)},
            )
        finally:
            with self._lock:
                self._running_threads.pop(str(job_id), None)
            self._update_queue_depth_metric()

    def _resume_incomplete_jobs(self) -> None:
        conn = db_connect()
        try:
            rows = conn.execute(
                """
                SELECT job_id
                FROM ingestion_jobs
                WHERE job_type = ?
                  AND status IN ('queued', 'running')
                ORDER BY created_at ASC
                LIMIT 50
                """,
                (JOB_TYPE_DOCS_INGESTION,),
            ).fetchall()
            if rows:
                now = now_iso()
                conn.execute(
                    """
                    UPDATE ingestion_jobs
                    SET status='queued',
                        message=?,
                        error=NULL,
                        updated_at=?
                    WHERE job_type = ?
                      AND status = 'running'
                    """,
                    ("Reprise du job après redémarrage backend.", now, JOB_TYPE_DOCS_INGESTION),
                )
                conn.commit()
        finally:
            conn.close()
        for row in rows:
            self._spawn_worker(str(row["job_id"]))
        self._update_queue_depth_metric()

    def _trim_old_jobs(self, conn: sqlite3.Connection) -> None:
        cur = conn.cursor()
        total_row = cur.execute("SELECT COUNT(*) FROM ingestion_jobs").fetchone()
        total = int(total_row[0] if total_row else 0)
        overflow = total - self._max_jobs
        if overflow <= 0:
            return
        cur.execute(
            """
            DELETE FROM ingestion_jobs
            WHERE job_id IN (
              SELECT job_id
              FROM ingestion_jobs
              ORDER BY datetime(created_at) ASC
              LIMIT ?
            )
            """,
            (overflow,),
        )

    def _update_queue_depth_metric(self) -> None:
        conn = db_connect()
        try:
            row = conn.execute(
                """
                SELECT COUNT(*) AS c
                FROM ingestion_jobs
                WHERE job_type = ? AND status IN ('queued', 'running')
                """,
                (JOB_TYPE_DOCS_INGESTION,),
            ).fetchone()
        finally:
            conn.close()
        monitoring_service.gauge("ingestion_queue_depth", float(int(row["c"] if row else 0)))

    def _observe_job_duration(self, job_id: str) -> None:
        conn = db_connect()
        try:
            row = conn.execute(
                """
                SELECT started_at, finished_at
                FROM ingestion_jobs
                WHERE job_id = ?
                """,
                (str(job_id),),
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return
        started_at = str(row["started_at"] or "").strip()
        finished_at = str(row["finished_at"] or "").strip()
        if not started_at or not finished_at:
            return
        try:
            delta = (datetime.fromisoformat(finished_at) - datetime.fromisoformat(started_at)).total_seconds()
        except Exception:
            return
        monitoring_service.observe_duration("ingestion_pipeline_duration_seconds", max(0.0, float(delta)))

    @staticmethod
    def _user_email_for_id(user_id: str) -> str | None:
        if not user_id:
            return None
        conn = db_connect()
        try:
            row = conn.execute("SELECT email FROM users WHERE id = ?", (str(user_id),)).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        email = str(row["email"] or "").strip()
        return email or None

    @staticmethod
    def _indexed_doc_ids_from_sqlite(sqlite_path: Path) -> set[str]:
        if not sqlite_path.exists():
            return set()
        out: set[str] = set()
        try:
            conn = sqlite3.connect(str(sqlite_path))
            conn.row_factory = sqlite3.Row
            try:
                cur = conn.cursor()
                for table in ("metadata_chunks", "chunks", "object_references"):
                    try:
                        cur.execute(
                            f"SELECT DISTINCT lower(doc_id) AS doc_id FROM {table} "
                            "WHERE doc_id IS NOT NULL AND trim(doc_id) != ''"
                        )
                    except Exception:
                        continue
                    for row in cur.fetchall():
                        doc_id = str(row["doc_id"] or "").strip().lower()
                        if doc_id:
                            out.add(doc_id)
            finally:
                conn.close()
        except Exception:
            return set()
        return out
