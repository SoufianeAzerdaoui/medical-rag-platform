from __future__ import annotations

import csv
import io
from datetime import datetime
from typing import Any

from backend.services import audit_service, ingestion_service


def build_report_rows(*, indexed_doc_ids: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in ingestion_service.discover_docs_pdfs(indexed_doc_ids=indexed_doc_ids):
        status = "indexed" if item.already_indexed else ("duplicate" if item.is_duplicate else "new")
        rows.append(
            {
                "filename": item.filename,
                "doc_id": item.doc_id,
                "status": status,
                "already_indexed": item.already_indexed,
                "is_duplicate": item.is_duplicate,
                "whitelist": item.duplicate_override,
                "size_bytes": item.size_bytes,
                "modified_at": item.modified_at,
                "first_seen_at": item.first_seen_at or "",
                "last_seen_at": item.last_seen_at or "",
                "last_ingested_at": item.last_ingested_at or "",
                "duplicate_reason": item.duplicate_reason or "",
                "duplicate_with": ", ".join(item.duplicate_with or []),
                "file_hash": item.file_hash,
                "text_hash": item.text_hash,
            }
        )
    return rows


def to_csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    output = io.StringIO()
    fields = [
        "filename",
        "doc_id",
        "status",
        "already_indexed",
        "is_duplicate",
        "whitelist",
        "size_bytes",
        "modified_at",
        "first_seen_at",
        "last_seen_at",
        "last_ingested_at",
        "duplicate_reason",
        "duplicate_with",
        "file_hash",
        "text_hash",
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return output.getvalue().encode("utf-8")


def to_simple_pdf_bytes(rows: list[dict[str, Any]]) -> bytes:
    # Minimal PDF generator for plain-text report (no external dependency).
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "Ingestion Report - Medical RAG Platform",
        f"Generated: {now}",
        f"Rows: {len(rows)}",
        "",
    ]
    for row in rows[:250]:
        lines.append(
            f"- {row['filename']} | {row['status']} | indexed={row['already_indexed']} | duplicate={row['is_duplicate']} | whitelist={row['whitelist']}"
        )
    if len(rows) > 250:
        lines.append(f"... truncated: {len(rows) - 250} additional rows")

    escaped = []
    for line in lines:
        safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        escaped.append(f"({safe}) Tj")
    content_stream = "BT /F1 10 Tf 40 800 Td 0 -14 Td " + " 0 -14 Td ".join(escaped) + " ET"
    content_bytes = content_stream.encode("latin-1", errors="replace")

    objects: list[bytes] = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objects.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>")
    objects.append(f"<< /Length {len(content_bytes)} >>\nstream\n".encode("ascii") + content_bytes + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    xref_offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        xref_offsets.append(out.tell())
        out.write(f"{idx} 0 obj\n".encode("ascii"))
        out.write(obj)
        out.write(b"\nendobj\n")
    xref_start = out.tell()
    out.write(f"xref\n0 {len(objects)+1}\n".encode("ascii"))
    out.write(b"0000000000 65535 f \n")
    for offset in xref_offsets[1:]:
        out.write(f"{offset:010d} 00000 n \n".encode("ascii"))
    out.write(
        f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n".encode("ascii")
    )
    return out.getvalue()


def build_document_timeline(filename: str) -> list[dict[str, Any]]:
    candidates = ingestion_service.discover_docs_pdfs(indexed_doc_ids=set())
    match = next((row for row in candidates if row.filename == filename), None)
    if not match:
        return []
    out: list[dict[str, Any]] = []
    if match.first_seen_at:
        out.append({"at": match.first_seen_at, "type": "discovered", "title": "Document détecté", "detail": match.filename})
    if match.last_ingested_at:
        out.append({"at": match.last_ingested_at, "type": "indexed", "title": "Document indexé", "detail": match.doc_id})
    if match.override_at:
        out.append(
            {
                "at": match.override_at,
                "type": "whitelist",
                "title": "Whitelist doublon modifiée",
                "detail": f"Par {match.override_by or 'unknown'}",
            }
        )
    if match.last_error:
        out.append({"at": match.last_seen_at or "", "type": "error", "title": "Erreur pipeline", "detail": match.last_error})

    # Add audit events tied to this document.
    for event in audit_service.list_events(target_type="document", target_id=filename, limit=100):
        out.append(
            {
                "at": str(event.get("created_at") or ""),
                "type": str(event.get("event_type") or "audit"),
                "title": str(event.get("event_type") or "Audit"),
                "detail": str(event.get("status") or ""),
                "actor": str(event.get("actor_email") or ""),
            }
        )
    out.sort(key=lambda item: str(item.get("at") or ""))
    return out

