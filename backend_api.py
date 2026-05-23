from __future__ import annotations

import logging
import sqlite3
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend import config
from backend.database import db_connect as _db_connect, init_schema as _init_auth_db
from backend.models import (
    AuthLoginRequest,
    AuthRegisterRequest,
    AuthResponse,
    ChatRequest,
    ChatResponse,
    ConversationClearRequest,
    ConversationClearResponse,
    ConversationCreateRequest,
    ConversationItem,
    DocumentItem,
    FeatureFlagItemResponse,
    FeatureFlagUpdateRequest,
    LogoutResponse,
    MessageItemResponse,
    UserResponse,
)
from backend.services import auth_service, conversation_service, feature_flag_service, message_service
from backend.services.chat_service import process_chat
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

JWT_SECRET = config.JWT_SECRET
FRONTEND_ORIGIN = config.FRONTEND_ORIGIN
AUTH_SCHEME = auth_service.AUTH_SCHEME
get_current_user = auth_service.get_current_user

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in FRONTEND_ORIGIN.split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
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


def _save_message(conversation_id: str, role: str, content: str) -> MessageItemResponse:
    return MessageItemResponse(**message_service.save_message(conversation_id, role, content))


def _load_state_from_db(conversation_id: str) -> dict[str, Any] | None:
    return _STATE_SERVICE.load_from_db(conversation_id)


def _save_state_to_db(conversation_id: str, state: dict[str, Any]) -> None:
    _STATE_SERVICE.save_to_db(conversation_id, state)


def _delete_conversation_state(conversation_id: str) -> None:
    _STATE_SERVICE.delete_db_state(conversation_id)


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
    sqlite_exists = (index_dir / "medical_rag.sqlite").exists()
    return {
        "status": "ok",
        "service": "medical-rag-backend",
        "index_ready": sqlite_exists,
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
