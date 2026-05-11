from __future__ import annotations

import re
from typing import Any, TypedDict

try:
    from source_resolver import (
        DocPdfResolver,
        build_source_url,
        build_viewer_url,
    )
except Exception:  # pragma: no cover - package import fallback
    from scripts.generation.source_resolver import (
        DocPdfResolver,
        build_source_url,
        build_viewer_url,
    )


class SourceCitation(TypedDict):
    doc_id: str
    filename: str | None
    page: int | None
    row: int | None
    label: str
    url: str | None
    viewer_url: str | None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def _build_label(*, filename: str | None, doc_id: str, page: int | None, row: int | None = None) -> str:
    base = (filename or "").strip() or doc_id
    base = re.sub(r"\[doc_id=.*?\]", "", base, flags=re.IGNORECASE).strip()
    base = re.sub(r"chunk_id\s*=\s*[^\],\s]+", "", base, flags=re.IGNORECASE).strip()
    base = re.sub(r"/home/[^\s\])]+", "", base).strip()
    base = re.sub(r"[A-Za-z]:\\[^\s\])]+", "", base).strip()
    base = re.sub(r"\bpage\s*(\d+)\s*row\s*(\d+)\b", r"page \1, ligne \2", base, flags=re.IGNORECASE)
    base = re.sub(r"\bligne\s*(\d+)\s*ligne\s*\1\b", r"ligne \1", base, flags=re.IGNORECASE)
    base = re.sub(r"(ligne\s*\d+)\s*\1\b", r"\1", base, flags=re.IGNORECASE)
    if page is not None:
        has_page = re.search(r"\bpage\s*\d+\b", base, flags=re.IGNORECASE) is not None
        if not has_page:
            base = f"{base} — page {page}"
    if row is not None:
        has_line = re.search(r"\bligne(?:s)?\s*\d+", base, flags=re.IGNORECASE) is not None
        if not has_line:
            base = f"{base}, ligne {row}"
    return " ".join(base.split())


def build_citations(
    evidence_pack: list[dict[str, Any]],
    *,
    include_chunk_id: bool = False,
    include_source_meta: bool = False,
) -> list[str]:
    citations: list[str] = []
    seen: set[tuple[Any, ...]] = set()

    for ev in evidence_pack:
        key = (
            ev.get("doc_id"),
            ev.get("page_number"),
            ev.get("row_index"),
            ev.get("chunk_id"),
            ev.get("source_kind"),
            ev.get("source_table_id"),
        )
        if key in seen:
            continue
        seen.add(key)

        parts = [f"doc_id={ev.get('doc_id')}", f"page={ev.get('page_number')}", f"row={ev.get('row_index')}"]
        if include_chunk_id and ev.get("chunk_id") not in (None, ""):
            parts.append(f"chunk_id={ev.get('chunk_id')}")
        if include_source_meta and ev.get("source_kind") not in (None, ""):
            parts.append(f"source_kind={ev.get('source_kind')}")
        if include_source_meta and ev.get("source_table_id") not in (None, ""):
            parts.append(f"source_table_id={ev.get('source_table_id')}")

        citations.append("[" + ", ".join(parts) + "]")

    return citations


def build_source_citations(
    evidence_pack: list[dict[str, Any]],
    *,
    resolver: DocPdfResolver | None = None,
) -> list[SourceCitation]:
    out: list[SourceCitation] = []
    seen: set[tuple[str, int | None, int | None]] = set()
    pdf_resolver = resolver or DocPdfResolver()

    for ev in evidence_pack:
        doc_id = str(ev.get("doc_id") or "").strip()
        if not doc_id:
            continue
        page = _safe_int(ev.get("page_number"))
        row = _safe_int(ev.get("row_index"))
        dedupe_key = (doc_id.lower(), page, row)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        source_pdf_hint = ev.get("source_pdf")
        resolved = pdf_resolver.resolve_pdf_for_doc_id(doc_id, str(source_pdf_hint) if source_pdf_hint else None)
        filename = resolved.filename if resolved else None
        page_url = build_source_url(doc_id, page) if resolved and resolved.pdf_path else None
        viewer_url = build_viewer_url(doc_id, page) if resolved and resolved.pdf_path else None

        out.append(
            SourceCitation(
                doc_id=doc_id,
                filename=filename,
                page=page,
                row=row,
                label=_build_label(filename=filename, doc_id=doc_id, page=page, row=row),
                url=page_url,
                viewer_url=viewer_url,
            )
        )

    return out


def append_citations(answer_text: str, citations: list[str]) -> str:
    base = (answer_text or "").strip()
    if not citations:
        return base
    if "sources :" in base.lower():
        return base

    lines = [base, "", "Sources :"]
    lines.extend(f"- {c}" for c in citations)
    return "\n".join(lines).strip()


def append_source_citations(
    answer_text: str,
    sources: list[SourceCitation],
    *,
    fallback_citations: list[str] | None = None,
) -> str:
    base = (answer_text or "").strip()
    if not sources and not fallback_citations:
        return base
    if "sources :" in base.lower():
        return base

    lines = [base, "", "Sources :"]
    if sources:
        for src in sources:
            if src.get("url"):
                lines.append(f"- [{src.get('label')}]({src.get('url')})")
            elif src.get("viewer_url"):
                lines.append(f"- [{src.get('label')}]({src.get('viewer_url')})")
            else:
                lines.append(f"- {src.get('label')}")
    else:
        for citation in fallback_citations or []:
            lines.append(f"- {citation}")
    return "\n".join(lines).strip()
