from __future__ import annotations

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


def _build_label(*, filename: str | None, doc_id: str, page: int | None) -> str:
    base = (filename or "").strip() or doc_id
    if page is not None:
        return f"{base} — page {page}"
    return base


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
                label=_build_label(filename=filename, doc_id=doc_id, page=page),
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
            row = src.get("row")
            row_text = f", row {row}" if row is not None else ""
            doc_meta = f"(doc_id={src.get('doc_id')}{row_text})"
            if src.get("url"):
                lines.append(f"- {src.get('label')} {doc_meta} : {src.get('url')}")
            else:
                lines.append(f"- {src.get('label')} {doc_meta}")
    else:
        for citation in fallback_citations or []:
            lines.append(f"- {citation}")
    return "\n".join(lines).strip()
