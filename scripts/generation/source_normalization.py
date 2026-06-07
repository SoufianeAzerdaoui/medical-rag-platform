from __future__ import annotations

from typing import Any

try:
    from source_resolver import build_source_url, build_viewer_url
except Exception:  # pragma: no cover
    from scripts.generation.source_resolver import build_source_url, build_viewer_url


def normalize_source_for_response(source: dict[str, Any]) -> dict[str, Any]:
    src = dict(source or {})
    doc_id = str(src.get("doc_id") or "").strip() or None
    source_pdf = str(src.get("source_pdf") or src.get("filename") or "").strip() or None
    if source_pdf and source_pdf.lower().startswith("docs/"):
        source_pdf = source_pdf[5:]
    page = src.get("page")
    line = src.get("line")
    row = src.get("row")
    label = str(src.get("label") or "").strip()
    if label.lower().startswith("docs/"):
        label = label[5:]
    viewer_url = str(src.get("viewer_url") or "").strip() or None
    source_url = str(src.get("source_url") or src.get("url") or "").strip() or None
    if not viewer_url and doc_id:
        viewer_url = build_viewer_url(doc_id, page if isinstance(page, int) else None)
    if not source_url and doc_id:
        source_url = build_source_url(doc_id, page if isinstance(page, int) else None)
    url = viewer_url or source_url
    if not label:
        base = source_pdf or doc_id or "source"
        if isinstance(page, int):
            base += f" — page {page}"
        if isinstance(line, int):
            base += f", ligne {line}"
        label = base
    out = {
        "label": label,
        "source_pdf": source_pdf,
        "page": page if isinstance(page, int) else None,
        "line": line if isinstance(line, int) else None,
        "row": row if isinstance(row, int) else None,
        "doc_id": doc_id,
        "viewer_url": viewer_url,
        "source_url": source_url,
        "url": url,
        "is_clickable": bool(url),
    }
    return out


def dedup_normalized_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, Any, Any, str]] = set()
    for src in sources or []:
        n = normalize_source_for_response(src)
        label = str(n.get("label") or "").strip().lower()
        page = n.get("page")
        line = n.get("line")
        row = n.get("row")
        if line is None:
            row = None
        key = (
            str(n.get("source_pdf") or "").strip().lower(),
            page,
            line,
            row,
            str(n.get("url") or "").strip(),
            label,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(n)
    return out
