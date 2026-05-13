from __future__ import annotations

import sqlite3
import sys
import logging
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from collections import defaultdict
from uuid import uuid4

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

from scripts.generation.source_resolver import DocPdfResolver, is_valid_doc_id
from scripts.generation.conversation_state_utils import (
    evidence_pack_is_transformable,
    get_transformable_context,
    migrate_conversation_state,
    update_conversation_state,
)
from scripts.generation.conversation_state_store import InMemoryConversationStateStore
from scripts.generation.source_normalization import dedup_normalized_sources


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
    generation_writer: Literal["llm_writer", "professional_fallback", "deterministic_metadata_query", "deterministic_response_transform_json"] | None = None
    visualization: dict[str, Any] | None = None
    chart_data: dict[str, Any] | None = None
    patients: list[dict[str, Any]] | None = None
    inventory_view: dict[str, Any] | None = None


class DocumentItem(BaseModel):
    id: str
    name: str


app = FastAPI(title="Medical RAG Backend API", version="1.1.0")
LOGGER = logging.getLogger("medical_rag.backend")


_CONVERSATION_STATE: dict[str, dict[str, Any]] = defaultdict(dict)
_STATE_STORE = InMemoryConversationStateStore(_CONVERSATION_STATE)

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
    structured = dedup_normalized_sources(list(result.get("sources") or []))
    if structured:
        for idx, src in enumerate(structured, start=1):
            doc_id = str(src.get("doc_id") or "").strip() or None
            filename = str(src.get("filename") or src.get("source_pdf") or "").strip() or None
            page = src.get("page")
            row = src.get("line", src.get("row"))
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


def _run_generation(**kwargs: Any) -> dict[str, Any]:
    # Lazy import keeps state tests independent from retrieval-heavy optional deps.
    from scripts.generation.generate_answer import run_generation

    return run_generation(**kwargs)


def _get_transformable_context(state: dict[str, Any]) -> dict[str, Any] | None:
    return get_transformable_context(state)


def _evidence_pack_is_transformable(pack: Any) -> bool:
    return evidence_pack_is_transformable(pack)


def _update_conversation_state(chat_id: str, state: dict[str, Any], generation: dict[str, Any], user_message: str) -> None:
    update_conversation_state(
        state_store=_CONVERSATION_STATE,
        chat_id=chat_id,
        state=state,
        generation=generation,
        user_message=user_message,
    )
    _STATE_STORE.save(chat_id, _CONVERSATION_STATE.get(chat_id) or {})


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
    request_id = str(uuid4())
    try:
        query = f"{payload.message} doc_id {payload.document_id}" if payload.document_id else payload.message
        state = _STATE_STORE.load(payload.chat_id)
        state_version_before = int(state.get("state_version") or 1)
        previous_intent = str(state.get("last_intent") or "").strip().lower() or "none"
        transformable_context = _get_transformable_context(state)
        qualitative_context = state.get("last_qualitative_evidence_pack") if isinstance(state.get("last_qualitative_evidence_pack"), dict) else None
        has_last_evidence_pack = isinstance(state.get("last_evidence_pack"), dict) and bool(state.get("last_evidence_pack"))
        has_last_transformable = isinstance(transformable_context, dict) and bool(transformable_context)
        has_last_qualitative = isinstance(qualitative_context, dict) and bool(qualitative_context)
        last_data_context_type_before = str(state.get("last_data_context_type") or "none")
        qn = payload.message.lower()
        visualization_request_detected = any(
            k in qn for k in ["radar", "chart", "graphique", "graphe", "visualisation", "courbe", "line graph", "bar chart", "diagramme"]
        )
        LOGGER.info(
            "qa_state_pre request_id=%s conversation_id=%s previous_intent=%s has_last_evidence_pack=%s has_last_transformable_evidence_pack=%s transformable_context_used=%s visualization_request_detected=%s",
            request_id,
            payload.chat_id,
            previous_intent,
            bool(has_last_evidence_pack),
            bool(has_last_transformable),
            bool(has_last_transformable),
            bool(visualization_request_detected),
        )
        generation = _run_generation(
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
            previous_structured_evidence_pack=transformable_context,
            previous_displayed_evidence_pack=(
                state.get("last_displayed_evidence_pack")
                if isinstance(state.get("last_displayed_evidence_pack"), dict)
                else None
            ),
            previous_context_intent=str(state.get("last_intent") or ""),
            previous_data_context_intent=str(state.get("last_data_context_intent") or ""),
            previous_data_context_type=str(state.get("last_data_context_type") or ""),
            previous_doc_scope=(
                list((state.get("last_doc_scope") or {}).get("doc_ids") or [])
                if isinstance(state.get("last_doc_scope"), dict)
                else []
            ),
            previous_qualitative_evidence_pack=(
                state.get("last_qualitative_evidence_pack")
                if isinstance(state.get("last_qualitative_evidence_pack"), dict)
                else None
            ),
            previous_has_patient_inventory=bool(state.get("last_patient_inventory")),
            previous_patient_inventory=(state.get("last_patient_inventory") if isinstance(state.get("last_patient_inventory"), list) else None),
            recent_style_history=state.get("recent_style_history") or [],
        )
        _update_conversation_state(payload.chat_id, state, generation, payload.message)
        new_state = _STATE_STORE.load(payload.chat_id)
        state_version_after = int(new_state.get("state_version") or state_version_before)
        current_intent = str(((generation.get("query_understanding") or {}).get("intent") or "")).strip().lower() or "unknown"
        visualization = generation.get("visualization") if isinstance(generation.get("visualization"), dict) else None
        rendered_type = str((visualization or {}).get("rendered_type") or "").strip().lower() or None
        requested_type = str((visualization or {}).get("requested_type") or "").strip().lower() or None
        LOGGER.info(
            "qa_state_post request_id=%s conversation_id=%s current_intent=%s previous_intent=%s has_last_evidence_pack=%s has_last_transformable_evidence_pack=%s transformable_context_used=%s visualization_request_detected=%s retrieval_skipped_due_to_no_transformable_context=%s response_has_visualization=%s response_rendered_type=%s response_requested_type=%s",
            request_id,
            payload.chat_id,
            current_intent,
            previous_intent,
            bool(isinstance(new_state.get("last_evidence_pack"), dict) and new_state.get("last_evidence_pack")),
            bool(isinstance(new_state.get("last_transformable_evidence_pack"), dict) and new_state.get("last_transformable_evidence_pack")),
            bool(has_last_transformable),
            bool(visualization_request_detected),
            bool(((generation.get("debug") or {}).get("retrieval_skipped_due_to_no_transformable_context"))),
            bool(visualization),
            rendered_type,
            requested_type,
        )
        LOGGER.info(
            "qa_state_versions request_id=%s conversation_id=%s state_version_before=%s state_version_after=%s last_data_context_type_before=%s last_data_context_type_after=%s requested_analytes=%s requested_date_iso=%s latest_report=%s requested_clickable_sources=%s previous_doc_scope_used=%s resolved_doc_scope=%s has_transformable_pack=%s has_qualitative_pack=%s sources_count=%s chart_generated=%s smalltalk_blocked=%s validator_status=%s",
            request_id,
            payload.chat_id,
            state_version_before,
            state_version_after,
            last_data_context_type_before,
            str(new_state.get("last_data_context_type") or "none"),
            list((generation.get("query_understanding") or {}).get("requested_analytes") or []),
            (generation.get("query_understanding") or {}).get("requested_date_iso"),
            bool((generation.get("query_understanding") or {}).get("latest_report")),
            bool((generation.get("query_understanding") or {}).get("source_clickable_requested")),
            bool(((generation.get("debug") or {}).get("context_resolution") or {}).get("reuse_doc_scope")),
            ((generation.get("debug") or {}).get("context_resolution") or {}).get("effective_doc_scope"),
            bool(isinstance(new_state.get("last_transformable_evidence_pack"), dict) and new_state.get("last_transformable_evidence_pack")),
            bool(isinstance(new_state.get("last_qualitative_evidence_pack"), dict) and new_state.get("last_qualitative_evidence_pack")),
            len(list(generation.get("sources") or [])),
            bool(visualization),
            (
                "bonjour ! je suis prêt" not in str(generation.get("answer") or "").lower()
                and "comment puis-je vous aider" not in str(generation.get("answer") or "").lower()
            ),
            str((generation.get("validation") or {}).get("validation_status") or ""),
        )
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
            visualization=(generation.get("visualization") if isinstance(generation.get("visualization"), dict) else None),
            chart_data=(generation.get("chart_data") if isinstance(generation.get("chart_data"), dict) else None),
            patients=generation.get("patients"),
            inventory_view=(generation.get("inventory_view") if isinstance(generation.get("inventory_view"), dict) else None),
        )
    except Exception as exc:  # pragma: no cover - defensive API guard
        LOGGER.exception(
            "chat_generation_failed request_id=%s provider=%s intent=unknown query=%r error=%s\n%s",
            request_id,
            "ollama",
            payload.message,
            str(exc),
            traceback.format_exc(),
        )
        safe_answer = (
            "Une erreur interne a empêché la génération complète de la réponse. "
            "Les données indexées restent disponibles ; veuillez relancer la demande ou simplifier la formulation."
        )
        return ChatResponse(
            answer=safe_answer,
            sources=[],
            confidence=0.0,
            document_ids=[],
            response_time=0.0,
            quality_report=None,
            validation_status="warning",
            generation_mode="controlled_error_fallback",
            generation_writer="professional_fallback",
            visualization=None,
            chart_data=None,
            patients=None,
            inventory_view=None,
        )


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
