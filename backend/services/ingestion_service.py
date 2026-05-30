from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend import config
from backend.database import db_connect, now_iso
from backend.services import antivirus_service


_INGESTION_LOCK = threading.Lock()


@dataclass
class IngestionResult:
    filename: str
    stored_path: str
    doc_id: str
    extraction_dir: str


@dataclass
class DocsPdfCandidate:
    filename: str
    doc_id: str
    absolute_path: str
    size_bytes: int
    modified_at: str
    file_hash: str
    text_hash: str
    already_indexed: bool
    is_duplicate: bool
    duplicate_with: list[str]
    duplicate_reason: str | None
    blocked: bool
    registry_status: str | None
    first_seen_at: str | None
    last_seen_at: str | None
    last_ingested_at: str | None
    last_error: str | None
    duplicate_entries: list[dict[str, Any]]
    duplicate_override: bool
    override_reason: str | None
    override_by: str | None
    override_at: str | None


_DOC_ID_CLEAN_RE = re.compile(r"[^A-Za-z0-9._-]+")
_TEXT_NORMALIZE_RE = re.compile(r"\s+")


def _run_cmd(args: list[str], *, cwd: Path) -> None:
    proc = subprocess.run(args, cwd=str(cwd), capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        detail = stderr or stdout or "pipeline command failed"
        raise RuntimeError(detail)


def _safe_pdf_filename(name: str) -> str:
    raw = Path(str(name or "document.pdf")).name
    stem = "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in Path(raw).stem)
    stem = stem.strip("._-") or "document"
    return f"{stem}.pdf"


def _doc_id_from_stem(stem: str) -> str:
    return _DOC_ID_CLEAN_RE.sub("_", str(stem or "")).strip("._-").lower() or "document"


def _ensure_unique_path(base_dir: Path, filename: str) -> Path:
    candidate = base_dir / filename
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix or ".pdf"
    i = 1
    while True:
        next_candidate = base_dir / f"{stem}_{i}{suffix}"
        if not next_candidate.exists():
            return next_candidate
        i += 1


def _docs_pdf_dir() -> Path:
    docs_dir = config.ROOT_DIR / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    return docs_dir


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_pdf_text_hash(path: Path) -> str:
    try:  # pragma: no cover - depends on runtime libs
        import fitz  # type: ignore
    except Exception:
        return ""
    try:
        doc = fitz.open(str(path))
    except Exception:
        return ""
    try:
        chunks: list[str] = []
        max_chars = 300_000
        for page in doc:
            chunks.append(str(page.get_text("text") or ""))
            if sum(len(x) for x in chunks) >= max_chars:
                break
        raw = " ".join(chunks).strip().lower()
    except Exception:
        return ""
    finally:
        try:
            doc.close()
        except Exception:
            pass
    if not raw:
        return ""
    normalized = _TEXT_NORMALIZE_RE.sub(" ", raw).strip()
    if len(normalized) < 30:
        return ""
    return hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()


def _registry_upsert_seen(
    *,
    filename: str,
    absolute_path: str,
    doc_id: str,
    file_hash: str,
    text_hash: str,
    size_bytes: int,
    modified_at: str,
) -> None:
    now = now_iso()
    conn = db_connect()
    try:
        conn.execute(
            """
            INSERT INTO docs_registry (
                filename, absolute_path, doc_id, file_hash, text_hash, size_bytes, modified_at,
                first_seen_at, last_seen_at, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'discovered')
            ON CONFLICT(absolute_path) DO UPDATE SET
                filename=excluded.filename,
                doc_id=excluded.doc_id,
                file_hash=excluded.file_hash,
                text_hash=excluded.text_hash,
                size_bytes=excluded.size_bytes,
                modified_at=excluded.modified_at,
                last_seen_at=excluded.last_seen_at
            """,
            (filename, absolute_path, doc_id, file_hash, text_hash, int(size_bytes), modified_at, now, now),
        )
        conn.commit()
    finally:
        conn.close()


def _registry_rows_by_hashes(binary_hashes: set[str], text_hashes: set[str]) -> list[dict[str, Any]]:
    if not binary_hashes and not text_hashes:
        return []
    conn = db_connect()
    try:
        clauses: list[str] = []
        params: list[str] = []
        if binary_hashes:
            placeholders = ",".join("?" for _ in binary_hashes)
            clauses.append(f"file_hash IN ({placeholders})")
            params.extend(sorted(binary_hashes))
        if text_hashes:
            placeholders = ",".join("?" for _ in text_hashes)
            clauses.append(f"text_hash IN ({placeholders})")
            params.extend(sorted(text_hashes))
        rows = conn.execute(
            """
            SELECT
                filename,
                absolute_path,
                doc_id,
                file_hash,
                text_hash,
                is_indexed,
                status,
                first_seen_at,
                last_seen_at,
                last_ingested_at,
                last_error,
                duplicate_override,
                override_reason,
                override_by,
                override_at
            FROM docs_registry
            WHERE
            """
            + " OR ".join(clauses),
            tuple(params),
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def _registry_mark_ingested(results: list[IngestionResult]) -> None:
    if not results:
        return
    now = now_iso()
    conn = db_connect()
    try:
        for item in results:
            abs_path = Path(item.stored_path).resolve()
            try:
                stat = abs_path.stat()
                size_bytes = int(stat.st_size)
                modified_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
                file_hash = _sha256_file(abs_path)
                text_hash = _normalized_pdf_text_hash(abs_path)
            except Exception:
                size_bytes = 0
                modified_at = now
                file_hash = ""
                text_hash = ""
            conn.execute(
                """
                INSERT INTO docs_registry (
                    filename, absolute_path, doc_id, file_hash, text_hash, size_bytes, modified_at,
                    first_seen_at, last_seen_at, last_ingested_at, is_indexed, status, last_error
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 'indexed', NULL)
                ON CONFLICT(absolute_path) DO UPDATE SET
                    filename=excluded.filename,
                    doc_id=excluded.doc_id,
                    file_hash=excluded.file_hash,
                    text_hash=excluded.text_hash,
                    size_bytes=excluded.size_bytes,
                    modified_at=excluded.modified_at,
                    last_seen_at=excluded.last_seen_at,
                    last_ingested_at=excluded.last_ingested_at,
                    is_indexed=1,
                    status='indexed',
                    last_error=NULL
                """,
                (
                    str(abs_path.name),
                    str(abs_path),
                    str(item.doc_id),
                    file_hash,
                    text_hash,
                    size_bytes,
                    modified_at,
                    now,
                    now,
                    now,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def _registry_mark_error(paths: list[Path], error_detail: str) -> None:
    if not paths:
        return
    conn = db_connect()
    try:
        for path in paths:
            conn.execute(
                """
                UPDATE docs_registry
                SET status = 'error', last_error = ?
                WHERE absolute_path = ?
                """,
                (str(error_detail)[:1200], str(path.resolve())),
            )
        conn.commit()
    finally:
        conn.close()


def set_duplicate_override(
    *,
    filename: str,
    enabled: bool,
    updated_by: str,
    reason: str | None = None,
) -> dict[str, Any]:
    docs_dir = _docs_pdf_dir()
    safe_name = Path(str(filename or "")).name
    if not safe_name or not safe_name.lower().endswith(".pdf"):
        raise RuntimeError("Nom de fichier PDF invalide.")
    pdf_path = (docs_dir / safe_name).resolve()
    if not pdf_path.exists() or not pdf_path.is_file():
        raise RuntimeError(f"Fichier introuvable dans docs/: {safe_name}")
    try:
        pdf_path.relative_to(docs_dir.resolve())
    except Exception as exc:
        raise RuntimeError("Chemin de fichier invalide.") from exc

    now = now_iso()
    conn = db_connect()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE docs_registry
            SET duplicate_override = ?,
                override_reason = ?,
                override_by = ?,
                override_at = ?
            WHERE absolute_path = ?
            """,
            (
                1 if enabled else 0,
                str(reason or "").strip()[:500] if enabled else None,
                str(updated_by or "").strip()[:240] if enabled else None,
                now if enabled else None,
                str(pdf_path),
            ),
        )
        if cur.rowcount <= 0:
            raise RuntimeError("Document non présent dans le registre. Lance d’abord Actualiser docs/.")
        conn.commit()
    finally:
        conn.close()
    return {
        "filename": safe_name,
        "enabled": bool(enabled),
        "reason": str(reason or "").strip() if enabled else None,
        "updated_by": str(updated_by or "").strip() if enabled else None,
        "updated_at": now if enabled else None,
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _replace_doc_rows(existing: list[dict[str, Any]], new_rows: list[dict[str, Any]], doc_id: str) -> list[dict[str, Any]]:
    incoming_chunk_ids = {
        str(row.get("chunk_id") or "").strip()
        for row in new_rows
        if str(row.get("chunk_id") or "").strip()
    }
    kept = [
        row
        for row in existing
        if str(row.get("doc_id") or "").strip().lower() != doc_id.lower()
        and str(row.get("chunk_id") or "").strip() not in incoming_chunk_ids
    ]
    kept.extend(new_rows)
    return kept


def _rebuild_indexes(global_anonymized_path: Path) -> None:
    try:
        _run_cmd(
            [
                sys.executable,
                "scripts/indexing/build_indexes.py",
                "--chunks",
                str(global_anonymized_path),
                "--index-dir",
                "data/indexes",
                "--collection",
                "medical_chunks",
                "--embedding-model",
                "BAAI/bge-m3",
                "--batch-size",
                "2",
                "--reset",
            ],
            cwd=config.ROOT_DIR,
        )
    except RuntimeError as exc:
        report_path = config.ROOT_DIR / "data" / "indexes" / "indexing_report.json"
        try:
            report = _read_json(report_path)
            errors = list(report.get("validation_errors") or [])
        except Exception:
            errors = []
        if errors:
            preview = "; ".join(str(err) for err in errors[:3])
            raise RuntimeError(
                f"Indexation bloquée ({len(errors)} erreurs de validation). Exemples: {preview}"
            ) from exc
        raise


def _process_single_pdf(
    pdf_path: Path,
    *,
    mapping_xlsx_path: Path,
    rebuild_indexes: bool = True,
) -> IngestionResult:
    extraction_root = config.ROOT_DIR / "data" / "extraction"
    temp_root = config.ROOT_DIR / "data" / "tmp_ingestion"
    temp_root.mkdir(parents=True, exist_ok=True)

    _run_cmd(
        [
            sys.executable,
            "scripts/extraction_data/run_extraction.py",
            str(pdf_path),
            "--output-root",
            str(extraction_root),
        ],
        cwd=config.ROOT_DIR,
    )

    # doc_id is generated by extraction from file stem and reflected by output dir.
    # read from the freshest document.json under extraction root matching source pdf.
    latest_doc_json = None
    for candidate in sorted(extraction_root.glob("*/document.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        payload = _read_json(candidate)
        source_pdf = str(payload.get("source_pdf") or "")
        if source_pdf and Path(source_pdf).resolve() == pdf_path.resolve():
            latest_doc_json = candidate
            break
    if latest_doc_json is None:
        raise RuntimeError("Extraction réussie mais document.json introuvable.")

    doc_payload = _read_json(latest_doc_json)
    doc_id = str(doc_payload.get("doc_id") or latest_doc_json.parent.name).strip().lower()
    doc_extraction_dir = latest_doc_json.parent
    doc_temp_raw = temp_root / f"{doc_id}.raw.jsonl"
    doc_temp_anon = temp_root / f"{doc_id}.anonymized.jsonl"
    doc_temp_report = temp_root / f"{doc_id}.anonymization_report.json"

    _run_cmd(
        [
            sys.executable,
            "scripts/chunking/build_chunks.py",
            "--input-root",
            str(doc_extraction_dir),
            "--output",
            str(doc_temp_raw),
            "--report",
            str(temp_root / f"{doc_id}.chunking_report.json"),
        ],
        cwd=config.ROOT_DIR,
    )

    _run_cmd(
        [
            sys.executable,
            "scripts/anonymization/anonymize_chunks.py",
            "--input",
            str(doc_temp_raw),
            "--output",
            str(doc_temp_anon),
            "--mapping-xlsx",
            str(mapping_xlsx_path),
            "--report",
            str(doc_temp_report),
        ],
        cwd=config.ROOT_DIR,
    )

    global_raw = config.ROOT_DIR / "data" / "chunks" / "chunks.raw.jsonl"
    global_anon = config.ROOT_DIR / "data" / "chunks" / "chunks.anonymized.jsonl"
    global_raw.parent.mkdir(parents=True, exist_ok=True)
    global_anon.parent.mkdir(parents=True, exist_ok=True)

    doc_raw_rows = _load_jsonl(doc_temp_raw)
    doc_anon_rows = _load_jsonl(doc_temp_anon)
    existing_raw = _load_jsonl(global_raw)
    existing_anon = _load_jsonl(global_anon)
    _write_jsonl(global_raw, _replace_doc_rows(existing_raw, doc_raw_rows, doc_id))
    _write_jsonl(global_anon, _replace_doc_rows(existing_anon, doc_anon_rows, doc_id))
    if rebuild_indexes:
        _rebuild_indexes(global_anon)

    return IngestionResult(
        filename=pdf_path.name,
        stored_path=str(pdf_path),
        doc_id=doc_id,
        extraction_dir=str(doc_extraction_dir),
    )


def ingest_uploaded_pdfs(files: list[tuple[str, bytes]]) -> list[IngestionResult]:
    if not files:
        return []
    docs_dir = _docs_pdf_dir()
    mapping_xlsx_path = config.ROOT_DIR / "data" / "private" / "anonymization_mapping.xlsx"
    mapping_xlsx_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[IngestionResult] = []
    global_anon = config.ROOT_DIR / "data" / "chunks" / "chunks.anonymized.jsonl"
    with _INGESTION_LOCK:
        selected_paths: list[Path] = []
        for original_name, raw in files:
            safe_name = _safe_pdf_filename(original_name)
            antivirus_service.scan_bytes_or_raise(raw, filename=safe_name)
            destination = _ensure_unique_path(docs_dir, safe_name)
            destination.write_bytes(raw)
            selected_paths.append(destination)
            results.append(_process_single_pdf(destination, mapping_xlsx_path=mapping_xlsx_path, rebuild_indexes=False))
        try:
            _rebuild_indexes(global_anon)
            _registry_mark_ingested(results)
        except Exception as exc:
            _registry_mark_error(selected_paths, str(exc))
            raise
    return results


def discover_docs_pdfs(*, indexed_doc_ids: set[str] | None = None) -> list[DocsPdfCandidate]:
    docs_dir = _docs_pdf_dir()
    indexed = {str(value or "").strip().lower() for value in (indexed_doc_ids or set()) if str(value or "").strip()}
    raw_candidates: list[dict[str, Any]] = []
    binary_hashes: set[str] = set()
    text_hashes: set[str] = set()

    for pdf_path in sorted(docs_dir.glob("*.pdf"), key=lambda p: p.name.lower()):
        try:
            stat = pdf_path.stat()
            size_bytes = int(stat.st_size)
            modified_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
            file_hash = _sha256_file(pdf_path)
            text_hash = _normalized_pdf_text_hash(pdf_path)
        except Exception:
            size_bytes = 0
            modified_at = datetime.now(timezone.utc).isoformat()
            file_hash = ""
            text_hash = ""
        doc_id = _doc_id_from_stem(pdf_path.stem)
        absolute_path = str(pdf_path.resolve())
        if file_hash:
            binary_hashes.add(file_hash)
        if text_hash:
            text_hashes.add(text_hash)
        _registry_upsert_seen(
            filename=pdf_path.name,
            absolute_path=absolute_path,
            doc_id=doc_id,
            file_hash=file_hash,
            text_hash=text_hash,
            size_bytes=size_bytes,
            modified_at=modified_at,
        )
        raw_candidates.append(
            {
                "filename": pdf_path.name,
                "doc_id": doc_id,
                "absolute_path": absolute_path,
                "size_bytes": size_bytes,
                "modified_at": modified_at,
                "file_hash": file_hash,
                "text_hash": text_hash,
            }
        )

    registry_rows = _registry_rows_by_hashes(binary_hashes, text_hashes)
    by_hash: dict[str, list[dict[str, Any]]] = {}
    by_text_hash: dict[str, list[dict[str, Any]]] = {}
    for row in registry_rows:
        h = str(row.get("file_hash") or "").strip()
        if not h:
            pass
        else:
            by_hash.setdefault(h, []).append(row)
        th = str(row.get("text_hash") or "").strip()
        if th:
            by_text_hash.setdefault(th, []).append(row)

    candidates: list[DocsPdfCandidate] = []
    for row in raw_candidates:
        group = list(by_hash.get(str(row["file_hash"]), []))
        text_group = by_text_hash.get(str(row.get("text_hash") or ""), [])
        for item in text_group:
            if item not in group:
                group.append(item)
        self_row = next((item for item in group if str(item.get("absolute_path") or "") == str(row["absolute_path"])), None)
        duplicates = [item for item in group if str(item.get("absolute_path") or "") != str(row["absolute_path"])]
        duplicate_with = sorted({str(item.get("filename") or "") for item in duplicates if str(item.get("filename") or "").strip()})
        self_indexed = bool((self_row or {}).get("is_indexed"))
        duplicate_has_indexed = any(
            bool(item.get("is_indexed")) or str(item.get("doc_id") or "").strip().lower() in indexed
            for item in duplicates
        )
        duplicate_override = bool((self_row or {}).get("duplicate_override"))
        is_duplicate = len(duplicates) > 0
        blocked = is_duplicate and not duplicate_override
        reason = None
        same_binary_duplicate = any(str(item.get("file_hash") or "") == str(row["file_hash"]) and str(row["file_hash"]) for item in duplicates)
        same_text_duplicate = any(str(item.get("text_hash") or "") == str(row.get("text_hash") or "") and str(row.get("text_hash") or "") for item in duplicates)
        if is_duplicate and duplicate_override:
            reason = "Doublon détecté mais autorisé par whitelist."
        elif is_duplicate and same_binary_duplicate and duplicate_has_indexed:
            reason = "Doublon binaire détecté avec un document déjà indexé."
        elif is_duplicate and same_binary_duplicate:
            reason = "Doublon binaire détecté."
        elif is_duplicate and same_text_duplicate and duplicate_has_indexed:
            reason = "Quasi-doublon texte détecté avec un document déjà indexé."
        elif is_duplicate and same_text_duplicate:
            reason = "Quasi-doublon texte détecté."
        elif is_duplicate and duplicate_has_indexed:
            reason = "Doublon de contenu détecté avec un document déjà indexé."
        elif is_duplicate:
            reason = "Doublon de contenu détecté."
        duplicate_entries: list[dict[str, Any]] = []
        for item in group:
            duplicate_entries.append(
                {
                    "filename": str(item.get("filename") or ""),
                    "absolute_path": str(item.get("absolute_path") or ""),
                    "doc_id": str(item.get("doc_id") or ""),
                    "is_indexed": bool(item.get("is_indexed")),
                    "status": str(item.get("status") or ""),
                    "first_seen_at": str(item.get("first_seen_at") or ""),
                    "last_seen_at": str(item.get("last_seen_at") or ""),
                    "last_ingested_at": str(item.get("last_ingested_at") or ""),
                    "last_error": str(item.get("last_error") or ""),
                }
            )
        candidates.append(
            DocsPdfCandidate(
                filename=str(row["filename"]),
                doc_id=str(row["doc_id"]),
                absolute_path=str(row["absolute_path"]),
                size_bytes=int(row["size_bytes"]),
                modified_at=str(row["modified_at"]),
                file_hash=str(row["file_hash"]),
                text_hash=str(row.get("text_hash") or ""),
                already_indexed=self_indexed or (str(row["doc_id"]).lower() in indexed) or duplicate_has_indexed,
                is_duplicate=is_duplicate,
                duplicate_with=duplicate_with,
                duplicate_reason=reason,
                blocked=blocked,
                registry_status=str((self_row or {}).get("status") or ""),
                first_seen_at=str((self_row or {}).get("first_seen_at") or ""),
                last_seen_at=str((self_row or {}).get("last_seen_at") or ""),
                last_ingested_at=str((self_row or {}).get("last_ingested_at") or ""),
                last_error=str((self_row or {}).get("last_error") or ""),
                duplicate_entries=duplicate_entries,
                duplicate_override=duplicate_override,
                override_reason=str((self_row or {}).get("override_reason") or ""),
                override_by=str((self_row or {}).get("override_by") or ""),
                override_at=str((self_row or {}).get("override_at") or ""),
            )
        )
    return candidates


def resync_docs_registry(*, indexed_doc_ids: set[str] | None = None) -> dict[str, int]:
    candidates = discover_docs_pdfs(indexed_doc_ids=indexed_doc_ids)
    conn = db_connect()
    try:
        now = now_iso()
        live_paths = {str(Path(item.absolute_path).resolve()) for item in candidates}
        for item in candidates:
            status = "indexed" if item.already_indexed else ("discovered" if not item.last_error else (item.registry_status or "discovered"))
            conn.execute(
                """
                UPDATE docs_registry
                SET is_indexed = ?, status = ?, last_seen_at = ?
                WHERE absolute_path = ?
                """,
                (1 if item.already_indexed else 0, status, now, str(Path(item.absolute_path).resolve())),
            )
        # Mark registry entries that are no longer present on disk.
        stale_rows = conn.execute("SELECT absolute_path FROM docs_registry").fetchall()
        for row in stale_rows:
            abs_path = str(row["absolute_path"] or "").strip()
            if not abs_path:
                continue
            if abs_path not in live_paths:
                conn.execute(
                    """
                    UPDATE docs_registry
                    SET status = 'missing', is_indexed = 0, last_seen_at = ?
                    WHERE absolute_path = ?
                    """,
                    (now, abs_path),
                )
        conn.commit()
    finally:
        conn.close()
    return {
        "discovered_count": len(candidates),
        "indexed_count": sum(1 for item in candidates if item.already_indexed),
        "duplicate_count": sum(1 for item in candidates if item.is_duplicate),
    }


def ingest_docs_by_filenames(filenames: list[str], *, indexed_doc_ids: set[str] | None = None) -> list[IngestionResult]:
    if not filenames:
        return []
    docs_dir = _docs_pdf_dir()
    mapping_xlsx_path = config.ROOT_DIR / "data" / "private" / "anonymization_mapping.xlsx"
    mapping_xlsx_path.parent.mkdir(parents=True, exist_ok=True)

    unique_names = validate_docs_selection(filenames, indexed_doc_ids=indexed_doc_ids)

    results: list[IngestionResult] = []
    global_anon = config.ROOT_DIR / "data" / "chunks" / "chunks.anonymized.jsonl"

    with _INGESTION_LOCK:
        selected_paths: list[Path] = []
        for name in unique_names:
            pdf_path = (docs_dir / name).resolve()
            if not pdf_path.exists() or not pdf_path.is_file():
                raise RuntimeError(f"Fichier introuvable dans docs/: {name}")
            try:
                pdf_path.relative_to(docs_dir.resolve())
            except Exception as exc:
                raise RuntimeError(f"Chemin invalide: {name}") from exc
            antivirus_service.scan_file_or_raise(pdf_path)
            selected_paths.append(pdf_path)
            results.append(_process_single_pdf(pdf_path, mapping_xlsx_path=mapping_xlsx_path, rebuild_indexes=False))
        try:
            _rebuild_indexes(global_anon)
            _registry_mark_ingested(results)
        except Exception as exc:
            _registry_mark_error(selected_paths, str(exc))
            raise

    return results


def validate_docs_selection(filenames: list[str], *, indexed_doc_ids: set[str] | None = None) -> list[str]:
    safe_names: list[str] = []
    for raw_name in filenames:
        name = Path(str(raw_name or "")).name
        if not name or not name.lower().endswith(".pdf"):
            continue
        safe_names.append(name)
    if not safe_names:
        raise RuntimeError("Aucun nom de fichier PDF valide.")

    unique_names = sorted(set(safe_names))
    candidates = discover_docs_pdfs(indexed_doc_ids=indexed_doc_ids)
    by_filename = {item.filename: item for item in candidates}
    blocked_rows = [name for name in unique_names if by_filename.get(name) and by_filename[name].blocked]
    if blocked_rows:
        details = []
        for name in blocked_rows:
            row = by_filename[name]
            if row.duplicate_with:
                details.append(f"{name} -> doublon: {', '.join(row.duplicate_with[:3])}")
            else:
                details.append(f"{name} -> doublon")
        raise RuntimeError("Sélection bloquée. " + " ; ".join(details))
    return unique_names


def reindex_single_doc(doc_id: str, source_pdf_path: Path) -> IngestionResult:
    mapping_xlsx_path = config.ROOT_DIR / "data" / "private" / "anonymization_mapping.xlsx"
    mapping_xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    with _INGESTION_LOCK:
        result = _process_single_pdf(source_pdf_path, mapping_xlsx_path=mapping_xlsx_path)
        _registry_mark_ingested([result])
        return result
