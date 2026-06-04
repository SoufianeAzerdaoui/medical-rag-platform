from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from backend import config
from backend.database import db_connect

try:  # pragma: no cover - runtime dependency
    import fitz  # type: ignore
except Exception:  # pragma: no cover
    fitz = None  # type: ignore


def _repo_relative(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(config.ROOT_DIR.resolve()))
    except Exception:
        try:
            return str(path.resolve())
        except Exception:
            return str(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        return None


def _pdf_page_count(path: Path) -> int | None:
    if fitz is None:
        return None
    try:
        with fitz.open(str(path)) as doc:
            return int(len(doc))
    except Exception:
        return None


def _parse_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except Exception:
        return None


def _registry_rows_for_doc_id(doc_id: str) -> list[dict[str, Any]]:
    conn = db_connect()
    try:
        rows = conn.execute(
            """
            SELECT
                id,
                filename,
                absolute_path,
                doc_id,
                file_hash,
                text_hash,
                size_bytes,
                modified_at,
                first_seen_at,
                last_seen_at,
                last_ingested_at,
                is_indexed,
                status,
                last_error
            FROM docs_registry
            WHERE lower(doc_id) = lower(?)
            ORDER BY
                CASE WHEN last_ingested_at IS NULL OR trim(last_ingested_at) = '' THEN 1 ELSE 0 END,
                last_ingested_at DESC,
                last_seen_at DESC,
                modified_at DESC,
                id DESC
            """,
            (str(doc_id or "").strip().lower(),),
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def _candidate_source_pdf_path(raw_path: str | None) -> Path | None:
    token = str(raw_path or "").strip()
    if not token:
        return None
    candidate = Path(token)
    if not candidate.is_absolute():
        candidate = (config.ROOT_DIR / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def _document_json_for_doc_id(doc_id: str) -> tuple[Path | None, dict[str, Any] | None]:
    path = config.ROOT_DIR / "data" / "extraction" / str(doc_id or "").strip().lower() / "document.json"
    if not path.exists():
        return None, None
    payload = _read_json(path)
    return path, payload


def _resolve_primary_pdf_path(
    *,
    registry_rows: list[dict[str, Any]],
    indexed_source_pdf: Path | None,
) -> Path | None:
    for row in registry_rows:
        raw_path = str(row.get("absolute_path") or "").strip()
        if not raw_path:
            continue
        try:
            candidate = Path(raw_path).resolve()
        except Exception:
            continue
        if candidate.exists():
            return candidate
    if indexed_source_pdf and indexed_source_pdf.exists():
        return indexed_source_pdf
    return None


def resolve_document_identity(doc_id: str) -> dict[str, Any]:
    requested_doc_id = str(doc_id or "").strip().lower()
    extraction_document_path, extraction_payload = _document_json_for_doc_id(requested_doc_id)
    registry_rows = _registry_rows_for_doc_id(requested_doc_id)
    indexed_source_pdf_raw = None
    indexed_page_count = None
    indexed_doc_id = None
    if isinstance(extraction_payload, dict):
        indexed_source_pdf_raw = (
            extraction_payload.get("source_pdf")
            or (extraction_payload.get("metadata") or {}).get("source_pdf")
        )
        indexed_page_count = _parse_int(extraction_payload.get("page_count"))
        indexed_doc_id = str(extraction_payload.get("doc_id") or "").strip().lower() or None
    indexed_source_pdf_path = _candidate_source_pdf_path(str(indexed_source_pdf_raw or "").strip() or None)

    primary_pdf_path = _resolve_primary_pdf_path(
        registry_rows=registry_rows,
        indexed_source_pdf=indexed_source_pdf_path,
    )
    primary_registry_row = registry_rows[0] if registry_rows else {}

    resolved_doc_id = (
        str(primary_registry_row.get("doc_id") or "").strip().lower()
        or indexed_doc_id
        or requested_doc_id
        or None
    )
    resolved_filename = (
        primary_pdf_path.name
        if primary_pdf_path is not None
        else str(primary_registry_row.get("filename") or "").strip()
        or (indexed_source_pdf_path.name if indexed_source_pdf_path is not None else None)
    )
    resolved_file_hash = _sha256_file(primary_pdf_path) if primary_pdf_path is not None else None
    resolved_page_count = _pdf_page_count(primary_pdf_path) if primary_pdf_path is not None else None
    registry_file_hash = str(primary_registry_row.get("file_hash") or "").strip() or None
    ingestion_timestamp = str(primary_registry_row.get("last_ingested_at") or "").strip() or None

    reasons: list[str] = []
    distinct_registry_paths = {
        str(row.get("absolute_path") or "").strip()
        for row in registry_rows
        if str(row.get("absolute_path") or "").strip()
    }
    distinct_registry_hashes = {
        str(row.get("file_hash") or "").strip()
        for row in registry_rows
        if str(row.get("file_hash") or "").strip()
    }
    if len(distinct_registry_paths) > 1 or len(distinct_registry_hashes) > 1:
        reasons.append("multiple_versions_for_doc_id")
    has_any_registered_source = bool(registry_rows or indexed_source_pdf_path is not None)
    if primary_pdf_path is None and has_any_registered_source:
        reasons.append("source_pdf_missing_on_disk")
    if primary_pdf_path is not None and indexed_source_pdf_path is not None:
        try:
            if primary_pdf_path.resolve() != indexed_source_pdf_path.resolve():
                reasons.append("source_pdf_path_mismatch")
        except Exception:
            reasons.append("source_pdf_path_mismatch")
    if (
        resolved_page_count is not None
        and indexed_page_count is not None
        and int(resolved_page_count) != int(indexed_page_count)
    ):
        reasons.append("page_count_mismatch")
    if resolved_file_hash and registry_file_hash and resolved_file_hash != registry_file_hash:
        reasons.append("file_hash_mismatch")
    if extraction_payload is None and has_any_registered_source:
        reasons.append("missing_extraction_artifact")

    reasons = list(dict.fromkeys(reasons))
    mismatch = bool(reasons)
    if mismatch:
        status = "mismatch"
    elif primary_pdf_path is None and extraction_payload is None and not registry_rows:
        status = "missing"
    else:
        status = "matched"

    return {
        "requested_doc_id": requested_doc_id or None,
        "resolved_doc_id": resolved_doc_id,
        "resolved_filename": resolved_filename or None,
        "resolved_file_hash": resolved_file_hash,
        "resolved_page_count": resolved_page_count,
        "indexed_page_count": indexed_page_count,
        "ingestion_timestamp": ingestion_timestamp,
        "source_pdf_path": _repo_relative(primary_pdf_path or indexed_source_pdf_path),
        "document_identity_mismatch": mismatch,
        "document_identity_status": status,
        "document_identity_reasons": reasons,
        "indexed_source_pdf_path": _repo_relative(indexed_source_pdf_path),
        "indexed_registry_file_hash": registry_file_hash,
        "extraction_document_path": _repo_relative(extraction_document_path),
        "registry_row_count": len(registry_rows),
    }


def resolve_document_identities(doc_ids: list[str] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in list(doc_ids or []):
        doc_id = str(raw or "").strip().lower()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        out.append(resolve_document_identity(doc_id))
    return out
