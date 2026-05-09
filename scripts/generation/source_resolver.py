from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote


_DOC_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


def is_valid_doc_id(doc_id: str) -> bool:
    return bool(_DOC_ID_RE.fullmatch(str(doc_id or "").strip()))


def normalize_doc_id(doc_id: str) -> str:
    return str(doc_id or "").strip().lower()


def build_source_url(doc_id: str, page: int | None) -> str:
    safe_doc_id = quote(str(doc_id or "").strip(), safe="_-")
    if page is not None:
        return f"/api/documents/{safe_doc_id}/pdf?page={int(page)}"
    return f"/api/documents/{safe_doc_id}/pdf"


def build_viewer_url(doc_id: str, page: int | None) -> str:
    safe_doc_id = quote(str(doc_id or "").strip(), safe="_-")
    if page is not None:
        return f"/viewer/pdf?doc_id={safe_doc_id}&page={int(page)}"
    return f"/viewer/pdf?doc_id={safe_doc_id}"


@dataclass(frozen=True)
class PdfSource:
    doc_id: str
    filename: str | None
    pdf_path: Path | None
    source_pdf: str | None = None


class DocPdfResolver:
    def __init__(
        self,
        *,
        project_root: Path | None = None,
        extraction_root: str | Path = "data/extraction",
        index_dir: str | Path = "data/indexes",
        docs_root: str | Path = "docs",
    ) -> None:
        self.project_root = (project_root or Path(__file__).resolve().parents[2]).resolve()
        self.extraction_root = (self.project_root / extraction_root).resolve()
        self.sqlite_path = (self.project_root / index_dir / "medical_rag.sqlite").resolve()
        self.docs_root = (self.project_root / docs_root).resolve()
        self.allowed_roots = [self.docs_root, self.extraction_root]

    def _is_allowed_pdf_path(self, path: Path) -> bool:
        try:
            resolved = path.resolve()
        except Exception:
            return False
        for root in self.allowed_roots:
            try:
                resolved.relative_to(root.resolve())
                return True
            except Exception:
                continue
        return False

    def _candidate_pdf_path(self, source_pdf: str | None) -> Path | None:
        if not source_pdf:
            return None
        raw = str(source_pdf).strip()
        if not raw:
            return None

        p = Path(raw)
        candidate = p if p.is_absolute() else (self.project_root / p)
        try:
            resolved = candidate.resolve()
        except Exception:
            return None
        if not self._is_allowed_pdf_path(resolved):
            return None
        return resolved

    def _build_pdf_source(self, doc_id: str, source_pdf: str | None) -> PdfSource | None:
        doc = normalize_doc_id(doc_id)
        if not doc or not is_valid_doc_id(doc):
            return None

        path = self._candidate_pdf_path(source_pdf)
        filename: str | None = None
        if source_pdf:
            filename = Path(str(source_pdf)).name or None
        if path and path.exists():
            filename = filename or path.name
            return PdfSource(doc_id=doc, filename=filename, pdf_path=path, source_pdf=source_pdf)

        # Keep a non-existing-but-known source name so UI can still display it.
        if filename:
            return PdfSource(doc_id=doc, filename=filename, pdf_path=None, source_pdf=source_pdf)
        return None

    def _load_from_document_json(self, out: dict[str, PdfSource]) -> None:
        if not self.extraction_root.exists():
            return
        for document_json in self.extraction_root.glob("*/document.json"):
            try:
                payload = json.loads(document_json.read_text(encoding="utf-8"))
            except Exception:
                continue
            doc_id = str(payload.get("doc_id") or document_json.parent.name or "").strip()
            if not is_valid_doc_id(doc_id):
                continue
            key = normalize_doc_id(doc_id)
            if key in out:
                continue
            source_pdf = payload.get("source_pdf") or (payload.get("metadata") or {}).get("source_pdf")
            src = self._build_pdf_source(doc_id, str(source_pdf) if source_pdf else None)
            if src:
                out[key] = src

    def _table_exists(self, conn: sqlite3.Connection, table_name: str) -> bool:
        cur = conn.cursor()
        cur.execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name = ? LIMIT 1",
            (table_name,),
        )
        return cur.fetchone() is not None

    def _load_from_sqlite(self, out: dict[str, PdfSource]) -> None:
        if not self.sqlite_path.exists():
            return
        try:
            conn = sqlite3.connect(str(self.sqlite_path))
            conn.row_factory = sqlite3.Row
        except Exception:
            return
        try:
            queries: list[str] = []
            if self._table_exists(conn, "metadata_chunks"):
                queries.append(
                    """
                    SELECT lower(doc_id) AS doc_id, source_pdf
                    FROM metadata_chunks
                    WHERE doc_id IS NOT NULL AND trim(doc_id) != ''
                      AND source_pdf IS NOT NULL AND trim(source_pdf) != ''
                    GROUP BY lower(doc_id), source_pdf
                    """
                )
            if self._table_exists(conn, "object_references"):
                queries.append(
                    """
                    SELECT lower(doc_id) AS doc_id, source_pdf
                    FROM object_references
                    WHERE doc_id IS NOT NULL AND trim(doc_id) != ''
                      AND source_pdf IS NOT NULL AND trim(source_pdf) != ''
                    GROUP BY lower(doc_id), source_pdf
                    """
                )
            for sql in queries:
                cur = conn.cursor()
                cur.execute(sql)
                for row in cur.fetchall():
                    doc_id = str(row["doc_id"] or "").strip()
                    if not is_valid_doc_id(doc_id):
                        continue
                    if doc_id in out:
                        continue
                    src = self._build_pdf_source(doc_id, str(row["source_pdf"] or ""))
                    if src:
                        out[doc_id] = src
        finally:
            conn.close()

    def _load_convention_fallback(self, out: dict[str, PdfSource]) -> None:
        if not self.docs_root.exists():
            return
        for pdf in self.docs_root.glob("*.pdf"):
            name = pdf.name.strip().lower()
            m = re.fullmatch(r"report\s*\((\d+)\)\.pdf", name)
            if not m:
                continue
            doc_id = f"report_{int(m.group(1))}"
            if doc_id in out:
                continue
            if not self._is_allowed_pdf_path(pdf):
                continue
            out[doc_id] = PdfSource(doc_id=doc_id, filename=pdf.name, pdf_path=pdf.resolve(), source_pdf=f"docs/{pdf.name}")

        default_report = self.docs_root / "report.pdf"
        if default_report.exists() and "report" not in out and self._is_allowed_pdf_path(default_report):
            out["report"] = PdfSource(
                doc_id="report",
                filename=default_report.name,
                pdf_path=default_report.resolve(),
                source_pdf="docs/report.pdf",
            )

    @lru_cache(maxsize=1)
    def _mapping(self) -> dict[str, PdfSource]:
        out: dict[str, PdfSource] = {}
        self._load_from_document_json(out)
        self._load_from_sqlite(out)
        self._load_convention_fallback(out)
        return out

    def resolve_pdf_for_doc_id(self, doc_id: str, source_pdf_hint: str | None = None) -> PdfSource | None:
        key = normalize_doc_id(doc_id)
        if not is_valid_doc_id(key):
            return None
        if source_pdf_hint:
            hinted = self._build_pdf_source(key, source_pdf_hint)
            if hinted:
                return hinted
        mapping = self._mapping()
        return mapping.get(key)

