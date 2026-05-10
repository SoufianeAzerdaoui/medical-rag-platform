from __future__ import annotations

import sqlite3
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
GENERATION_DIR = SCRIPTS_DIR / "generation"
for p in (str(SCRIPTS_DIR), str(GENERATION_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.generation.generate_answer import run_generation
from scripts.generation.source_resolver import DocPdfResolver, is_valid_doc_id


class ChatHistoryItem(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    chat_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    history: list[ChatHistoryItem] = Field(default_factory=list)
    document_id: str | None = None
    mode: Literal["general", "document_analysis", "comparison", "summary"] = "general"


class SourceItem(BaseModel):
    id: str
    documentName: str
    documentId: str | None = None
    page: int | None = None
    section: str | None = None
    excerpt: str | None = None
    score: float | None = None
    type: str | None = None
    warning: str | None = None

    # Structured citation fields (new)
    doc_id: str | None = None
    filename: str | None = None
    row: int | None = None
    label: str | None = None
    url: str | None = None
    viewer_url: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem] = Field(default_factory=list)
    confidence: float | None = None
    document_ids: list[str] = Field(default_factory=list)
    response_time: float | None = None
    quality_report: dict[str, Any] | None = None
    validation_status: Literal["pass", "warning", "fail"] | None = None
    generation_mode: str | None = None
    generation_writer: Literal["llm_writer", "professional_fallback"] | None = None


class DocumentItem(BaseModel):
    id: str
    name: str


app = FastAPI(title="Medical RAG Backend API", version="1.1.0")


_CONVERSATION_STATE: dict[str, dict[str, Any]] = defaultdict(dict)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def _resolver() -> DocPdfResolver:
    return DocPdfResolver(project_root=ROOT_DIR)


def _to_source_items(result: dict[str, Any]) -> list[SourceItem]:
    items: list[SourceItem] = []

    # Prefer new structured citations when available.
    structured = list(result.get("sources") or [])
    if structured:
        for idx, src in enumerate(structured, start=1):
            doc_id = str(src.get("doc_id") or "").strip() or None
            filename = str(src.get("filename") or "").strip() or None
            page = src.get("page")
            row = src.get("row")
            label = str(src.get("label") or "").strip() or None
            url = str(src.get("url") or "").strip() or None
            viewer_url = str(src.get("viewer_url") or "").strip() or None
            warning = None if url else "source_pdf_unavailable"
            items.append(
                SourceItem(
                    id=f"source-{idx}",
                    documentName=filename or "Document médical",
                    documentId=doc_id,
                    page=page if isinstance(page, int) else None,
                    section=None,
                    excerpt=label,
                    score=None,
                    type="pdf_source",
                    warning=warning,
                    doc_id=doc_id,
                    filename=filename,
                    row=row if isinstance(row, int) else None,
                    label=label,
                    url=url,
                    viewer_url=viewer_url,
                )
            )
        return items

    # Backward compatibility fallback from evidence pack.
    evidences = list(result.get("displayed_evidences") or result.get("evidence_pack") or [])
    for idx, ev in enumerate(evidences, start=1):
        page_num: int | None = None
        try:
            page_value = ev.get("page_number")
            page_num = int(page_value) if page_value is not None else None
        except Exception:
            page_num = None
        items.append(
            SourceItem(
                id=f"legacy-{idx}",
                documentName=str(ev.get("source_pdf") or ev.get("doc_id") or "Document médical"),
                documentId=str(ev.get("doc_id") or "") or None,
                page=page_num,
                section=str(ev.get("section") or "") or None,
                excerpt=str(ev.get("text_excerpt") or ev.get("source") or "")[:600] or None,
                score=float(ev.get("final_score")) if ev.get("final_score") not in (None, "") else None,
                type=str(ev.get("chunk_type") or "") or None,
                warning=None,
            )
        )
    return items


def _confidence_from_result(result: dict[str, Any]) -> float | None:
    validation = result.get("validation") or {}
    status = str(validation.get("validation_status") or "").lower()
    if status == "pass":
        return 0.9
    if status == "warning":
        return 0.7
    if status == "fail":
        return 0.4
    return None


@app.get("/health")
def health() -> dict[str, Any]:
    index_dir = Path("data/indexes")
    sqlite_exists = (index_dir / "medical_rag.sqlite").exists()
    return {
        "status": "ok",
        "service": "medical-rag-backend",
        "index_ready": sqlite_exists,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    try:
        query = f"{payload.message} doc_id {payload.document_id}" if payload.document_id else payload.message
        state = _CONVERSATION_STATE.get(payload.chat_id) or {}
        generation = run_generation(
            query=query,
            top_k=5,
            mode="hybrid",
            provider="ollama",
            model="qwen3:4b",
            temperature=0.0,
            num_ctx=4096,
            max_tokens=600,
            timeout=120,
            index_dir="data/indexes",
            collection="medical_chunks",
            previous_structured_evidence_pack=state.get("last_evidence_pack"),
            recent_style_history=state.get("recent_style_history") or [],
        )
        _CONVERSATION_STATE[payload.chat_id]["last_evidence_pack"] = generation.get("structured_evidence_pack") or state.get(
            "last_evidence_pack"
        )
        _CONVERSATION_STATE[payload.chat_id]["last_answer"] = generation.get("answer")
        _CONVERSATION_STATE[payload.chat_id]["last_query_understanding"] = generation.get("query_understanding")
        _CONVERSATION_STATE[payload.chat_id]["last_rendered_rows"] = generation.get("displayed_evidences")
        _CONVERSATION_STATE[payload.chat_id]["last_sources"] = generation.get("sources")
        style_hist = list(state.get("recent_style_history") or [])
        style_entry = generation.get("style_memory_entry")
        if isinstance(style_entry, dict):
            style_hist.append(style_entry)
        _CONVERSATION_STATE[payload.chat_id]["recent_style_history"] = style_hist[-20:]
        answer = str(generation.get("answer") or "").strip()
        if not answer:
            answer = "Aucune réponse générée. Cette réponse ne remplace pas l'avis médical."

        sources = _to_source_items(generation)
        document_ids = sorted({s.documentId for s in sources if s.documentId})

        return ChatResponse(
            answer=answer,
            sources=sources,
            confidence=_confidence_from_result(generation),
            document_ids=document_ids,
            response_time=float(generation.get("generation_time_seconds") or 0.0),
            quality_report=(generation.get("quality_report") if isinstance(generation.get("quality_report"), dict) else None),
            validation_status=str((generation.get("validation") or {}).get("validation_status") or "") or None,
            generation_mode=str(generation.get("generation_mode") or "") or None,
            generation_writer=str(((generation.get("debug") or {}).get("generation_writer") or "")) or None,
        )
    except Exception as exc:  # pragma: no cover - defensive API guard
        raise HTTPException(status_code=500, detail=f"Generation error: {exc}") from exc


@app.get("/documents", response_model=list[DocumentItem])
def documents() -> list[DocumentItem]:
    seen: set[str] = set()
    out: list[DocumentItem] = []

    # Preferred source: secure resolver mapping.
    for doc_id, src in sorted(_resolver()._mapping().items()):  # noqa: SLF001 - internal cache read for listing
        if doc_id in seen:
            continue
        seen.add(doc_id)
        out.append(DocumentItem(id=doc_id, name=src.filename or doc_id))

    sqlite_path = Path("data/indexes/medical_rag.sqlite")
    if not sqlite_path.exists():
        return out

    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        for table in ("metadata_chunks", "chunks", "object_references"):
            try:
                cur.execute(
                    f"SELECT DISTINCT lower(doc_id) AS doc_id FROM {table} "
                    "WHERE doc_id IS NOT NULL AND trim(doc_id) != '' ORDER BY doc_id LIMIT 500"
                )
            except Exception:
                continue
            for row in cur.fetchall():
                doc_id = str(row["doc_id"] or "").strip()
                if not doc_id or doc_id in seen:
                    continue
                if not is_valid_doc_id(doc_id):
                    continue
                seen.add(doc_id)
                src = _resolver().resolve_pdf_for_doc_id(doc_id)
                out.append(DocumentItem(id=doc_id, name=(src.filename if src else doc_id) or doc_id))
    finally:
        conn.close()

    return out


@app.get("/api/documents/{doc_id}/pdf")
def get_pdf(doc_id: str, page: int | None = Query(default=None, ge=1)) -> FileResponse:
    _ = page  # kept for frontend viewer deep-linking compatibility
    if not is_valid_doc_id(doc_id):
        raise HTTPException(status_code=404, detail="Document introuvable")

    resolved = _resolver().resolve_pdf_for_doc_id(doc_id)
    if not resolved or not resolved.pdf_path:
        raise HTTPException(status_code=404, detail="PDF source introuvable")

    pdf_path = resolved.pdf_path
    if not pdf_path.exists() or not pdf_path.is_file():
        raise HTTPException(status_code=404, detail="PDF source introuvable")

    filename = resolved.filename or f"{doc_id}.pdf"
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=filename,
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@app.post("/upload")
def upload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "received",
        "payload": payload,
        "message": "Upload endpoint placeholder. Use extraction/indexing pipeline.",
    }


@app.post("/audio/transcribe")
def audio_transcribe(payload: dict[str, Any]) -> dict[str, Any]:
    _ = payload
    return {"transcript": ""}
