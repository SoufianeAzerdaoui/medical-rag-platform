from __future__ import annotations

import logging
import os
import sqlite3
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend import config
from backend.database import db_connect as _db_connect, init_schema as _init_auth_db
from backend.models import (
    ActiveModelResponse,
    AuthLoginRequest,
    AuthRegisterRequest,
    AuthResponse,
    ChatRequest,
    ChatResponse,
    ConversationClearRequest,
    ConversationClearResponse,
    ConversationContextUsageResponse,
    ConversationCreateRequest,
    ConversationItem,
    DocsDiscoveryItem,
    DocsIngestRequest,
    DocumentItem,
    FeatureFlagItemResponse,
    FeatureFlagUpdateRequest,
    LogoutResponse,
    MessageItemResponse,
    UserResponse,
)
from backend.services import (
    auth_service,
    conversation_service,
    feature_flag_service,
    ingestion_service,
    message_service,
    transcription_service,
)
from backend.services.chat_service import process_chat
from backend.services.model_registry import (
    active_model_info,
    estimate_tokens_from_text,
    extract_rag_text_from_sources,
    trim_text_to_token_budget,
)
from backend.services.conversation_state_store import (
    ConversationStateService,
    evidence_pack_transformable,
    transformable_context,
)

ROOT_DIR = config.ROOT_DIR
SCRIPTS_DIR = config.SCRIPTS_DIR
GENERATION_DIR = config.GENERATION_DIR
for module_path in (str(SCRIPTS_DIR), str(GENERATION_DIR)):
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

from scripts.generation.conversation_state_utils import update_conversation_state
from scripts.generation.source_resolver import DocPdfResolver, is_valid_doc_id

app = FastAPI(title="Medical RAG Backend API", version="1.1.0")
LOGGER = logging.getLogger("medical_rag.backend")

_STATE_SERVICE = ConversationStateService()
_CONVERSATION_STATE = _STATE_SERVICE.legacy_state
_STATE_STORE = _STATE_SERVICE.memory_store
_DELETED_DOC_IDS: set[str] = set()

JWT_SECRET = config.JWT_SECRET
FRONTEND_ORIGIN = config.FRONTEND_ORIGIN
AUTH_SCHEME = auth_service.AUTH_SCHEME
get_current_user = auth_service.get_current_user

ALLOWED_FRONTEND_ORIGINS = [origin.strip() for origin in FRONTEND_ORIGIN.split(",") if origin.strip()]
# Dev-safe fallback: allow localhost / 127.0.0.1 on any port for preflight CORS.
LOCAL_DEV_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_FRONTEND_ORIGINS,
    allow_origin_regex=LOCAL_DEV_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def _resolver() -> DocPdfResolver:
    return DocPdfResolver(project_root=ROOT_DIR)


def _run_generation(**kwargs: Any) -> dict[str, Any]:
    from scripts.generation.generate_answer import run_generation

    return run_generation(**kwargs)


def _evidence_pack_is_transformable(pack: Any) -> bool:
    return evidence_pack_transformable(pack)


def _get_transformable_context(state: dict[str, Any]) -> dict[str, Any] | None:
    return transformable_context(state)


def _update_conversation_state(chat_id: str, state: dict[str, Any], generation: dict[str, Any], user_message: str) -> None:
    update_conversation_state(
        state_store=_CONVERSATION_STATE,
        chat_id=chat_id,
        state=state,
        generation=generation,
        user_message=user_message,
    )
    _STATE_SERVICE.save(chat_id, _CONVERSATION_STATE.get(chat_id) or {})


def _get_user_by_id(user_id: str) -> dict[str, Any] | None:
    return auth_service.get_user_by_id(user_id)


def _get_user_by_email(email: str) -> dict[str, Any] | None:
    return auth_service.get_user_by_email(email)


def _normalize_email(email: str) -> str:
    return auth_service.normalize_email(email)


def _status_from_usage_percent(usage_percent: float) -> str:
    if usage_percent >= 95.0:
        return "full"
    if usage_percent >= 85.0:
        return "warning"
    if usage_percent >= 70.0:
        return "medium"
    return "safe"


def _stt_debug_enabled() -> bool:
    raw = str(os.getenv("APP_ENV", "")).strip().lower()
    if raw in {"dev", "development", "test", "local"}:
        return True
    return str(os.getenv("STT_DEBUG", "")).strip().lower() in {"1", "true", "yes", "on"}


def _extract_state_context_text(state: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("last_transformable_evidence_pack", "last_qualitative_evidence_pack", "last_evidence_pack"):
        pack = state.get(key)
        if not isinstance(pack, dict):
            continue
        rows = list(pack.get("evidences") or pack.get("results") or [])
        for row in rows:
            if not isinstance(row, dict):
                continue
            for field in ("text_excerpt", "source", "label", "analyte", "value", "reference_range", "status_code", "source_pdf"):
                value = str(row.get(field) or "").strip()
                if value:
                    parts.append(value)
    return "\n".join(parts)[:60_000]


def _chunks_count(sqlite_path: Path) -> int | None:
    if not sqlite_path.exists():
        return None


def _indexed_doc_ids_from_sqlite(sqlite_path: Path) -> set[str]:
    if not sqlite_path.exists():
        return set()
    out: set[str] = set()
    try:
        conn = sqlite3.connect(str(sqlite_path))
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.cursor()
            for table in ("metadata_chunks", "chunks", "object_references"):
                try:
                    cur.execute(
                        f"SELECT DISTINCT lower(doc_id) AS doc_id FROM {table} "
                        "WHERE doc_id IS NOT NULL AND trim(doc_id) != ''"
                    )
                except Exception:
                    continue
                for row in cur.fetchall():
                    doc_id = str(row["doc_id"] or "").strip().lower()
                    if doc_id:
                        out.add(doc_id)
        finally:
            conn.close()
    except Exception:
        return set()
    return out
    try:
        conn = sqlite3.connect(str(sqlite_path))
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM chunks")
            value = cur.fetchone()
            return int(value[0]) if value else 0
        finally:
            conn.close()
    except Exception:
        return None


def _latest_assistant_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in reversed(rows):
        if str(row.get("role") or "") != "assistant":
            continue
        diagnostics = row.get("diagnostics")
        if isinstance(diagnostics, dict):
            return diagnostics
    return {}


def _is_llm_context_expected(diagnostics: dict[str, Any]) -> bool:
    route_class = str(diagnostics.get("llm_route_class") or "").strip().lower()
    final_answer_source = str(diagnostics.get("final_answer_source") or "").strip().lower()
    generation_mode = str(diagnostics.get("generation_mode") or "").strip().lower()
    llm_writer_attempted = bool(diagnostics.get("llm_writer_attempted"))
    llm_writer_accepted = bool(diagnostics.get("llm_writer_accepted"))
    llm_expected = bool(diagnostics.get("llm_expected"))

    if llm_writer_attempted or llm_writer_accepted or llm_expected:
        return True
    if final_answer_source == "llm_writer":
        return True
    if route_class == "llm_allowed":
        return True
    return generation_mode in {"hybrid", "llm_writer", "llm_first"}


def _build_history_text(rows: list[dict[str, Any]], *, max_rows: int, per_message_char_limit: int) -> str:
    selected = rows[-max_rows:]
    parts: list[str] = []
    for row in selected:
        role = str(row.get("role") or "").strip() or "message"
        raw = str(row.get("content") or "").strip()
        if not raw:
            continue
        content = raw[: max(64, per_message_char_limit)]
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _create_conversation_record(*, user_id: str, title: str | None = None, conversation_id: str | None = None) -> ConversationItem:
    return ConversationItem(**conversation_service.create_conversation_record(user_id=user_id, title=title, conversation_id=conversation_id))


def _get_conversation_for_user(conversation_id: str, user_id: str) -> dict[str, Any] | None:
    return conversation_service.get_conversation_for_user(conversation_id, user_id)


def _get_conversation_any_owner(conversation_id: str) -> dict[str, Any] | None:
    return conversation_service.get_conversation_any_owner(conversation_id)


def _require_owned_conversation(conversation_id: str, user_id: str) -> dict[str, Any]:
    return conversation_service.require_owned_conversation(conversation_id, user_id)


def _touch_conversation(conversation_id: str) -> None:
    conversation_service.touch_conversation(conversation_id)


def _save_message(
    conversation_id: str,
    role: str,
    content: str,
    *,
    sources: list[dict[str, Any]] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> MessageItemResponse:
    return MessageItemResponse(
        **message_service.save_message(
            conversation_id,
            role,
            content,
            sources=sources,
            diagnostics=diagnostics,
        )
    )


def _load_state_from_db(conversation_id: str) -> dict[str, Any] | None:
    return _STATE_SERVICE.load_from_db(conversation_id)


def _save_state_to_db(conversation_id: str, state: dict[str, Any]) -> None:
    _STATE_SERVICE.save_to_db(conversation_id, state)


def _delete_conversation_state(conversation_id: str) -> None:
    _STATE_SERVICE.delete_db_state(conversation_id)


def _refresh_resolver_cache() -> None:
    resolver = _resolver()
    try:
        resolver._mapping.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        pass


def _to_user_response(user_row: dict[str, Any]) -> UserResponse:
    return UserResponse(
        id=str(user_row.get("id")),
        email=str(user_row.get("email")),
        created_at=str(user_row.get("created_at")),
    )


_init_auth_db()
feature_flag_service.ensure_feature_flags_seeded()
LOGGER.info(
    "startup_config feature_flag_admin_api_enabled=%s admin_emails_count=%s",
    bool(config.ENABLE_FEATURE_FLAG_ADMIN_API),
    len(tuple(config.ADMIN_EMAILS or ())),
)


def _require_feature_flag_admin(current_user: dict[str, Any]) -> None:
    if not config.ENABLE_FEATURE_FLAG_ADMIN_API:
        raise HTTPException(status_code=403, detail="Feature flag admin API is disabled.")
    admin_emails = set(config.ADMIN_EMAILS)
    if not admin_emails:
        return
    user_email = _normalize_email(str(current_user.get("email") or ""))
    if user_email not in admin_emails:
        raise HTTPException(status_code=403, detail="Forbidden")


@app.get("/health")
def health() -> dict[str, Any]:
    index_dir = Path("data/indexes")
    sqlite_path = index_dir / "medical_rag.sqlite"
    sqlite_exists = sqlite_path.exists()
    chunk_count = _chunks_count(sqlite_path)
    if not sqlite_exists:
        index_status = "missing"
    elif chunk_count is None:
        index_status = "error"
    elif chunk_count <= 0:
        index_status = "empty"
    else:
        index_status = "ready"
    return {
        "status": "ok",
        "service": "medical-rag-backend",
        "index_ready": bool(index_status == "ready"),
        "index_status": index_status,
        "index_chunks_count": chunk_count,
    }


@app.post("/auth/register", response_model=AuthResponse)
def auth_register(payload: AuthRegisterRequest) -> AuthResponse:
    user = auth_service.register_user(email=payload.email, password=payload.password)
    return AuthResponse(
        access_token=auth_service.create_access_token(str(user["id"])),
        user=_to_user_response(user),
    )


@app.post("/auth/login", response_model=AuthResponse)
def auth_login(payload: AuthLoginRequest) -> AuthResponse:
    user = auth_service.login_user(email=payload.email, password=payload.password)
    return AuthResponse(
        access_token=auth_service.create_access_token(str(user["id"])),
        user=_to_user_response(user),
    )


@app.get("/auth/me", response_model=UserResponse)
def auth_me(current_user: dict[str, Any] = Depends(get_current_user)) -> UserResponse:
    return _to_user_response(current_user)


@app.post("/auth/logout", response_model=LogoutResponse)
def auth_logout(_: dict[str, Any] = Depends(get_current_user)) -> LogoutResponse:
    return LogoutResponse(success=True)


@app.get("/conversations", response_model=list[ConversationItem])
def list_conversations(current_user: dict[str, Any] = Depends(get_current_user)) -> list[ConversationItem]:
    rows = conversation_service.list_conversations_for_user(str(current_user["id"]))
    return [ConversationItem(**row) for row in rows]


@app.post("/conversations", response_model=ConversationItem)
def create_conversation(
    payload: ConversationCreateRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> ConversationItem:
    row = conversation_service.create_conversation_record(user_id=str(current_user["id"]), title=payload.title)
    return ConversationItem(**row)


@app.get("/conversations/{conversation_id}/messages", response_model=list[MessageItemResponse])
def get_conversation_messages(conversation_id: str, current_user: dict[str, Any] = Depends(get_current_user)) -> list[MessageItemResponse]:
    conversation_service.require_owned_conversation(conversation_id, str(current_user["id"]))
    rows = message_service.list_messages(conversation_id)
    return [MessageItemResponse(**row) for row in rows]


@app.delete("/conversations/{conversation_id}", response_model=LogoutResponse)
def delete_conversation(conversation_id: str, current_user: dict[str, Any] = Depends(get_current_user)) -> LogoutResponse:
    conversation_service.require_owned_conversation(conversation_id, str(current_user["id"]))
    conversation_service.delete_conversation(conversation_id, str(current_user["id"]))
    _delete_conversation_state(conversation_id)
    return LogoutResponse(success=True)


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, current_user: dict[str, Any] = Depends(get_current_user)) -> ChatResponse:
    return process_chat(
        payload=payload,
        current_user=current_user,
        state_service=_STATE_SERVICE,
        run_generation=_run_generation,
        logger=LOGGER,
    )


@app.get("/api/models/active", response_model=ActiveModelResponse)
def get_active_model(current_user: dict[str, Any] = Depends(get_current_user)) -> ActiveModelResponse:
    _ = current_user
    info = active_model_info()
    return ActiveModelResponse(
        provider=info.provider,
        model=info.model,
        context_window=int(info.context_window),
        max_output_tokens=int(info.max_output_tokens),
        recommended_rag_budget=int(info.recommended_rag_budget),
    )


@app.get("/api/conversations/{conversation_id}/context-usage", response_model=ConversationContextUsageResponse)
def get_conversation_context_usage(
    conversation_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> ConversationContextUsageResponse:
    conversation_service.require_owned_conversation(conversation_id, str(current_user["id"]))
    info = active_model_info()
    rows = message_service.list_messages(conversation_id)
    state = _STATE_SERVICE.load(conversation_id)
    latest_diagnostics = _latest_assistant_diagnostics(rows)
    llm_context_expected = _is_llm_context_expected(latest_diagnostics)

    system_prompt = (
        "Assistant clinique prudent. Réponse basée uniquement sur documents sourcés. "
        "Aucun diagnostic direct. Signaler limites et éléments manquants."
    )
    context_window = max(1, int(info.context_window))
    history_text = _build_history_text(
        rows,
        max_rows=20 if llm_context_expected else 10,
        per_message_char_limit=900 if llm_context_expected else 320,
    )

    latest_rag_sources_text = ""
    if llm_context_expected:
        for row in reversed(rows):
            sources = list(row.get("sources") or [])
            if str(row.get("role") or "") == "assistant" and sources:
                latest_rag_sources_text = extract_rag_text_from_sources(sources)
                if latest_rag_sources_text.strip():
                    break

    state_context_text = _extract_state_context_text(state if (llm_context_expected and isinstance(state, dict)) else {})
    rag_budget = (
        max(512, min(int(info.recommended_rag_budget), int(context_window * 0.6)))
        if llm_context_expected
        else 0
    )
    rag_text = trim_text_to_token_budget("\n".join([latest_rag_sources_text, state_context_text]).strip(), rag_budget)
    history_budget = max(384, int(context_window * (0.35 if llm_context_expected else 0.18)))

    prompt_tokens = estimate_tokens_from_text(system_prompt)
    history_tokens = min(estimate_tokens_from_text(history_text), history_budget)
    rag_tokens = 0 if rag_budget <= 0 else min(estimate_tokens_from_text(rag_text), rag_budget)
    reserved_output_tokens = (
        max(128, min(int(info.max_output_tokens), int(context_window * 0.25)))
        if llm_context_expected
        else max(64, min(256, int(context_window * 0.04)))
    )

    used_tokens = int(prompt_tokens + history_tokens + rag_tokens + reserved_output_tokens)
    remaining_tokens = max(0, context_window - used_tokens)
    usage_percent = round(min(100.0, (used_tokens / context_window) * 100.0), 2)
    status = _status_from_usage_percent(usage_percent)

    return ConversationContextUsageResponse(
        conversation_id=str(conversation_id),
        model=info.model,
        context_window=context_window,
        used_tokens=used_tokens,
        remaining_tokens=remaining_tokens,
        usage_percent=usage_percent,
        status=status,  # type: ignore[arg-type]
    )


@app.post("/chat/clear", response_model=ConversationClearResponse)
def clear_conversation(payload: ConversationClearRequest, current_user: dict[str, Any] = Depends(get_current_user)) -> ConversationClearResponse:
    conversation_service.require_owned_conversation(payload.conversation_id, str(current_user["id"]))
    _delete_conversation_state(payload.conversation_id)
    return ConversationClearResponse(success=True, conversation_id=str(payload.conversation_id))


@app.get("/documents", response_model=list[DocumentItem])
def documents() -> list[DocumentItem]:
    seen: set[str] = set()
    out: list[DocumentItem] = []

    for doc_id, src in sorted(_resolver()._mapping().items()):  # noqa: SLF001
        if doc_id in seen:
            continue
        if doc_id in _DELETED_DOC_IDS:
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
                if doc_id in _DELETED_DOC_IDS:
                    continue
                if not is_valid_doc_id(doc_id):
                    continue
                seen.add(doc_id)
                src = _resolver().resolve_pdf_for_doc_id(doc_id)
                out.append(DocumentItem(id=doc_id, name=(src.filename if src else doc_id) or doc_id))
    finally:
        conn.close()

    return out


@app.get("/documents/discover", response_model=list[DocsDiscoveryItem])
def discover_documents(current_user: dict[str, Any] = Depends(get_current_user)) -> list[DocsDiscoveryItem]:
    _ = current_user
    sqlite_path = Path("data/indexes/medical_rag.sqlite")
    indexed_ids = _indexed_doc_ids_from_sqlite(sqlite_path)
    candidates = ingestion_service.discover_docs_pdfs(indexed_doc_ids=indexed_ids)
    return [DocsDiscoveryItem(**{
        "filename": item.filename,
        "doc_id": item.doc_id,
        "absolute_path": item.absolute_path,
        "size_bytes": item.size_bytes,
        "modified_at": item.modified_at,
        "already_indexed": item.already_indexed,
    }) for item in candidates]


@app.post("/documents/{doc_id}/reindex", response_model=LogoutResponse)
def reindex_document(doc_id: str) -> LogoutResponse:
    if not is_valid_doc_id(doc_id):
        raise HTTPException(status_code=404, detail="Document introuvable")
    if doc_id in _DELETED_DOC_IDS:
        raise HTTPException(status_code=404, detail="Document supprimé")
    resolved = _resolver().resolve_pdf_for_doc_id(doc_id)
    if not resolved or not resolved.pdf_path:
        raise HTTPException(status_code=404, detail="Document introuvable")
    if not resolved.pdf_path.exists():
        raise HTTPException(status_code=404, detail="Fichier source introuvable")
    ingestion_service.reindex_single_doc(doc_id=doc_id, source_pdf_path=resolved.pdf_path)
    _refresh_resolver_cache()
    return LogoutResponse(success=True)


@app.delete("/documents/{doc_id}", response_model=LogoutResponse)
def delete_document(doc_id: str) -> LogoutResponse:
    if not is_valid_doc_id(doc_id):
        raise HTTPException(status_code=404, detail="Document introuvable")
    _DELETED_DOC_IDS.add(doc_id)

    sqlite_path = Path("data/indexes/medical_rag.sqlite")
    if sqlite_path.exists():
        conn = sqlite3.connect(str(sqlite_path))
        try:
            cur = conn.cursor()
            for table in ("metadata_chunks", "chunks", "object_references"):
                try:
                    cur.execute(f"DELETE FROM {table} WHERE lower(doc_id) = ?", (doc_id.lower(),))
                except Exception:
                    continue
            conn.commit()
        finally:
            conn.close()
    _refresh_resolver_cache()

    return LogoutResponse(success=True)


@app.get("/feature-flags", response_model=list[FeatureFlagItemResponse])
def get_feature_flags(current_user: dict[str, Any] = Depends(get_current_user)) -> list[FeatureFlagItemResponse]:
    _require_feature_flag_admin(current_user)
    return [FeatureFlagItemResponse(**flag) for flag in feature_flag_service.list_feature_flags()]


@app.patch("/feature-flags/{flag_name}", response_model=FeatureFlagItemResponse)
def patch_feature_flag(
    flag_name: str,
    payload: FeatureFlagUpdateRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> FeatureFlagItemResponse:
    _require_feature_flag_admin(current_user)
    feature_flag_service.set_feature_flag(
        name=str(flag_name),
        enabled=bool(payload.enabled),
        updated_by=str(current_user.get("email") or ""),
    )
    flags = {f["name"]: f for f in feature_flag_service.list_feature_flags()}
    if str(flag_name) not in flags:
        raise HTTPException(status_code=404, detail="Feature flag not found")
    return FeatureFlagItemResponse(**flags[str(flag_name)])


@app.get("/api/documents/{doc_id}/pdf")
def get_pdf(doc_id: str, page: int | None = Query(default=None, ge=1)) -> FileResponse:
    _ = page
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
async def upload(
    files: list[UploadFile] = File(...),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    _ = current_user
    if not files:
        raise HTTPException(status_code=400, detail="Aucun fichier fourni.")

    accepted: list[tuple[str, bytes]] = []
    skipped: list[dict[str, Any]] = []
    for file in files:
        name = str(file.filename or "").strip()
        if not name.lower().endswith(".pdf"):
            skipped.append({"filename": name or "unknown", "reason": "unsupported_extension"})
            continue
        raw = await file.read()
        if not raw:
            skipped.append({"filename": name or "unknown", "reason": "empty_file"})
            continue
        accepted.append((name, raw))

    if not accepted:
        raise HTTPException(status_code=400, detail="Aucun PDF valide à ingérer.")

    try:
        results = ingestion_service.ingest_uploaded_pdfs(accepted)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Échec ingestion: {exc}") from exc
    _refresh_resolver_cache()
    return {
        "success": True,
        "ingested_count": len(results),
        "ingested": [
            {
                "filename": item.filename,
                "doc_id": item.doc_id,
                "stored_path": item.stored_path,
                "extraction_dir": item.extraction_dir,
            }
            for item in results
        ],
        "skipped": skipped,
    }


@app.post("/upload/from-docs")
def upload_from_docs(
    payload: DocsIngestRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    _ = current_user
    filenames = [str(name or "").strip() for name in payload.filenames if str(name or "").strip()]
    if not filenames:
        raise HTTPException(status_code=400, detail="Aucun fichier sélectionné depuis docs/.")
    try:
        results = ingestion_service.ingest_docs_by_filenames(filenames)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Échec ingestion docs/: {exc}") from exc
    _refresh_resolver_cache()
    return {
        "success": True,
        "ingested_count": len(results),
        "ingested": [
            {
                "filename": item.filename,
                "doc_id": item.doc_id,
                "stored_path": item.stored_path,
                "extraction_dir": item.extraction_dir,
            }
            for item in results
        ],
        "skipped": [],
    }


@app.post("/audio/transcribe")
async def audio_transcribe(
    audio: UploadFile = File(...),
    debug: bool = Query(False),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    _ = current_user
    raw = await audio.read()
    if len(raw) < 2_000:
        raise HTTPException(status_code=422, detail="Audio trop court ou silencieux.")
    suffix = Path(str(audio.filename or "")).suffix or ".webm"
    LOGGER.info("audio_transcribe_request filename=%s bytes=%s suffix=%s", str(audio.filename or ""), len(raw), suffix)
    try:
        detailed = transcription_service.transcribe_audio_bytes_debug(raw, suffix=suffix)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription échouée: {exc}") from exc
    transcript = detailed.transcript
    LOGGER.info(
        "audio_transcribe_response transcript_chars=%s quality=%s rejected_reason=%s accepted_strategy=%s",
        len(transcript or ""),
        detailed.quality_score,
        detailed.rejected_reason,
        detailed.accepted_strategy,
    )
    payload: dict[str, Any] = {"transcript": transcript}
    if debug and _stt_debug_enabled():
        payload["debug"] = detailed.as_dict()
    return payload
