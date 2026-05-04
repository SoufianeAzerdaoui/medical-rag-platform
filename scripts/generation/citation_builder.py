from __future__ import annotations

from typing import Any


def build_citations(evidence_pack: list[dict[str, Any]]) -> list[str]:
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

        parts = [
            f"doc_id={ev.get('doc_id')}",
            f"page={ev.get('page_number')}",
            f"row={ev.get('row_index')}",
            f"chunk_id={ev.get('chunk_id')}",
        ]
        if ev.get("source_kind") not in (None, ""):
            parts.append(f"source_kind={ev.get('source_kind')}")
        if ev.get("source_table_id") not in (None, ""):
            parts.append(f"source_table_id={ev.get('source_table_id')}")

        citations.append("[" + ", ".join(parts) + "]")

    return citations


def append_citations(answer_text: str, citations: list[str]) -> str:
    base = (answer_text or "").strip()
    if not citations:
        return base
    if "sources :" in base.lower():
        return base

    lines = [base, "", "Sources :"]
    lines.extend(f"- {c}" for c in citations)
    return "\n".join(lines).strip()
