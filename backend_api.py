from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import PlainTextResponse
from fastapi.responses import Response

from backend import config
from backend.database import db_connect as _db_connect, init_schema as _init_auth_db, now_iso
from backend.logging_setup import configure_logging
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
    DuplicateOverrideRequest,
    DuplicateOverrideResponse,
    FeatureFlagItemResponse,
    FeatureFlagUpdateRequest,
    IngestionJobStartResponse,
    IngestionJobStatusResponse,
    LogoutResponse,
    MessageItemResponse,
    ResyncDocsRegistryResponse,
    UserResponse,
)
from backend.services import (
    audit_service,
    auth_service,
    conversation_service,
    feature_flag_service,
    ingestion_service,
    ingestion_jobs_service,
    ingestion_report_service,
    message_service,
    monitoring_service,
    p0_readiness_service,
    rate_limit_service,
    retention_service,
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
configure_logging()
LOGGER = logging.getLogger("medical_rag.backend")

if config.SENTRY_DSN:
    try:  # pragma: no cover - optional runtime integration
        import sentry_sdk  # type: ignore

        sentry_sdk.init(dsn=config.SENTRY_DSN, traces_sample_rate=0.1)
    except Exception:
        LOGGER.warning("Sentry DSN fourni mais sentry_sdk indisponible.")

_STATE_SERVICE = ConversationStateService()
_CONVERSATION_STATE = _STATE_SERVICE.legacy_state
_STATE_STORE = _STATE_SERVICE.memory_store
_DELETED_DOC_IDS: set[str] = set()
_INGESTION_JOBS: ingestion_jobs_service.IngestionJobService | None = None

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
        role=str(user_row.get("role") or "user"),
        created_at=str(user_row.get("created_at")),
    )


_init_auth_db()
_INGESTION_JOBS = ingestion_jobs_service.IngestionJobService()
feature_flag_service.ensure_feature_flags_seeded()
LOGGER.info(
    "startup_config feature_flag_admin_api_enabled=%s admin_emails_count=%s",
    bool(config.ENABLE_FEATURE_FLAG_ADMIN_API),
    len(tuple(config.ADMIN_EMAILS or ())),
)
if str(os.getenv("APP_ENV", "")).strip().lower() == "production" and bool(config.PROD_READINESS_ENFORCE):
    startup_readiness = p0_readiness_service.run_p0_readiness_check()
    if str(startup_readiness.get("overall_status") or "fail").lower() != "pass":
        raise RuntimeError(
            "P0 readiness check failed at startup. "
            f"blocking_failures={int(startup_readiness.get('blocking_failures') or 0)}"
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


def _user_has_ops_permission(current_user: dict[str, Any]) -> bool:
    role = str(current_user.get("role") or "user").strip().lower()
    if role in {"admin", "ops", "data_manager", "medical_admin"}:
        return True
    user_email = _normalize_email(str(current_user.get("email") or ""))
    admin_emails = set(config.ADMIN_EMAILS)
    if user_email in admin_emails:
        return True
    app_env = str(os.getenv("APP_ENV", "")).strip().lower()
    if not admin_emails and app_env in {"dev", "development", "local", "test"}:
        return True
    return False


def _require_ops_permission(current_user: dict[str, Any]) -> None:
    if not _user_has_ops_permission(current_user):
        raise HTTPException(status_code=403, detail="Action réservée aux opérateurs autorisés.")


def _ingestion_jobs() -> ingestion_jobs_service.IngestionJobService:
    if _INGESTION_JOBS is None:
        raise HTTPException(status_code=503, detail="Service ingestion indisponible.")
    return _INGESTION_JOBS


def _client_ip(request: Request) -> str:
    forwarded = str(request.headers.get("x-forwarded-for") or "").strip()
    if forwarded:
        return forwarded.split(",")[0].strip()
    real_ip = str(request.headers.get("x-real-ip") or "").strip()
    if real_ip:
        return real_ip
    if request.client and request.client.host:
        return str(request.client.host)
    return "unknown"


def _enforce_rate_limit(*, request: Request, scope: str, key: str, limit: int) -> None:
    result = rate_limit_service.enforce_limit(
        scope=scope,
        key=key,
        limit=max(1, int(limit)),
        window_seconds=max(1, int(config.RATE_LIMIT_WINDOW_SECONDS)),
    )
    if result.allowed:
        return
    raise HTTPException(
        status_code=429,
        detail=(
            f"Trop de requêtes ({result.count}/{result.limit}) sur {result.window_seconds}s. "
            f"Réessaie dans {result.retry_after_seconds}s."
        ),
    )


def _clamav_status() -> dict[str, Any]:
    cmd = str(config.ANTIVIRUS_CLAMSCAN_CMD or "clamscan").strip() or "clamscan"
    binary = shutil.which(cmd)
    available = bool(binary)
    version = ""
    if available:
        try:
            proc = subprocess.run(
                [cmd, "--version"],
                capture_output=True,
                text=True,
                timeout=max(2, min(10, int(config.ANTIVIRUS_TIMEOUT_SECONDS))),
            )
            if proc.returncode == 0:
                version = str((proc.stdout or "").strip() or (proc.stderr or "").strip())
        except Exception:
            version = ""
    healthy = available and (not config.ANTIVIRUS_REQUIRED or bool(version or binary))
    return {
        "required": bool(config.ANTIVIRUS_REQUIRED),
        "command": cmd,
        "available": available,
        "version": version,
        "healthy": healthy,
    }


def _run_with_retry(fn: Any, *, retries: int = 2, delay_seconds: float = 0.12) -> Any:
    attempts = max(1, int(retries) + 1)
    last_exc: Exception | None = None
    for idx in range(attempts):
        try:
            return fn()
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            last_exc = exc
            if idx >= attempts - 1:
                break
            time.sleep(max(0.0, float(delay_seconds)))
    if last_exc:
        raise last_exc
    return None


def _health_dependencies() -> dict[str, Any]:
    app_db_ok = True
    app_db_error: str | None = None
    try:
        def _probe_db() -> None:
            conn = _db_connect()
            try:
                conn.execute("SELECT 1").fetchone()
            finally:
                conn.close()
        _run_with_retry(_probe_db, retries=1)
    except Exception as exc:
        app_db_ok = False
        app_db_error = str(exc)

    clamav = _clamav_status()
    jobs_service_ready = bool(_INGESTION_JOBS is not None)
    return {
        "app_db": {"ok": app_db_ok, "error": app_db_error},
        "clamav": {
            "ok": bool(clamav.get("healthy")),
            "required": bool(clamav.get("required")),
            "available": bool(clamav.get("available")),
        },
        "ingestion_jobs": {"ok": jobs_service_ready},
    }


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
    deps = _health_dependencies()
    deps_ok = (
        bool(deps.get("app_db", {}).get("ok"))
        and bool(deps.get("ingestion_jobs", {}).get("ok"))
        and (bool(deps.get("clamav", {}).get("ok")) or not bool(deps.get("clamav", {}).get("required")))
    )
    status = "ok" if deps_ok else "degraded"
    return {
        "status": status,
        "service": "medical-rag-backend",
        "index_ready": bool(index_status == "ready"),
        "index_status": index_status,
        "index_chunks_count": chunk_count,
        "dependencies": deps,
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> PlainTextResponse:
    body = monitoring_service.render_prometheus()
    return PlainTextResponse(content=body, media_type="text/plain; version=0.0.4; charset=utf-8")


@app.get("/monitoring/summary")
def monitoring_summary(current_user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    _require_ops_permission(current_user)
    return monitoring_service.compute_summary()


@app.get("/admin/security-status")
def security_status(current_user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    _require_ops_permission(current_user)
    return {
        "server_time": now_iso(),
        "clamav": _clamav_status(),
        "sentry": {
            "configured": bool(config.SENTRY_DSN),
            "dsn_masked": bool(config.SENTRY_DSN),
        },
        "jwt": {
            "algorithm": str(config.JWT_ALGORITHM),
            "expire_minutes": int(config.JWT_EXPIRE_MINUTES),
            "rotation_previous_count": len(tuple(config.JWT_SECRET_PREVIOUS or ())),
        },
        "encryption": {
            "enabled": bool(config.DATA_ENCRYPTION_ENABLED),
            "required": bool(config.DATA_ENCRYPTION_REQUIRED),
            "key_configured": bool(str(config.DATA_ENCRYPTION_KEY or "").strip()),
        },
        "rate_limits": {
            "window_seconds": int(config.RATE_LIMIT_WINDOW_SECONDS),
            "auth_per_window": int(config.RATE_LIMIT_AUTH_PER_WINDOW),
            "chat_per_window": int(config.RATE_LIMIT_CHAT_PER_WINDOW),
            "upload_per_window": int(config.RATE_LIMIT_UPLOAD_PER_WINDOW),
            "login_max_failures": int(config.AUTH_LOGIN_MAX_FAILURES),
            "login_block_seconds": int(config.AUTH_LOGIN_BLOCK_SECONDS),
        },
        "retention": {
            "jobs_days": int(config.RETENTION_JOBS_DAYS),
            "audit_days": int(config.RETENTION_AUDIT_DAYS),
            "docs_days": int(config.RETENTION_DOCS_DAYS),
            "audio_days": int(config.RETENTION_AUDIO_DAYS),
            "logs_days": int(config.RETENTION_LOGS_DAYS),
            "auth_attempts_days": int(config.RETENTION_AUTH_ATTEMPTS_DAYS),
        },
    }


@app.get("/admin/go-live/p0-check")
def go_live_p0_check(current_user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    _require_ops_permission(current_user)
    return p0_readiness_service.run_p0_readiness_check()


@app.post("/admin/retention/run")
def run_retention(
    hard_delete_docs: bool = Query(default=False),
    dry_run: bool = Query(default=False),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    _require_ops_permission(current_user)
    result = retention_service.run_retention(hard_delete_docs=hard_delete_docs, dry_run=dry_run)
    audit_service.log_event(
        event_type="retention_run",
        actor_user_id=str(current_user.get("id") or ""),
        actor_email=str(current_user.get("email") or ""),
        target_type="retention",
        target_id="global",
        status="success",
        payload={"hard_delete_docs": hard_delete_docs, "dry_run": dry_run},
        result=result,
    )
    return {"success": True, **result}


@app.post("/auth/register", response_model=AuthResponse)
def auth_register(payload: AuthRegisterRequest, request: Request) -> AuthResponse:
    _enforce_rate_limit(
        request=request,
        scope="auth_register_ip",
        key=_client_ip(request),
        limit=max(3, int(config.RATE_LIMIT_AUTH_PER_WINDOW)),
    )
    user = auth_service.register_user(email=payload.email, password=payload.password)
    return AuthResponse(
        access_token=auth_service.create_access_token(str(user["id"])),
        user=_to_user_response(user),
    )


@app.post("/auth/login", response_model=AuthResponse)
def auth_login(payload: AuthLoginRequest, request: Request) -> AuthResponse:
    ip = _client_ip(request)
    _enforce_rate_limit(
        request=request,
        scope="auth_login_ip",
        key=ip,
        limit=max(3, int(config.RATE_LIMIT_AUTH_PER_WINDOW)),
    )
    user = auth_service.login_user(email=payload.email, password=payload.password, ip_address=ip)
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
def chat(
    payload: ChatRequest,
    request: Request,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> ChatResponse:
    _enforce_rate_limit(
        request=request,
        scope="chat_user",
        key=str(current_user.get("id") or "anonymous"),
        limit=max(20, int(config.RATE_LIMIT_CHAT_PER_WINDOW)),
    )
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
        "file_hash": item.file_hash,
        "text_hash": item.text_hash,
        "already_indexed": item.already_indexed,
        "is_duplicate": item.is_duplicate,
        "duplicate_with": item.duplicate_with,
        "duplicate_reason": item.duplicate_reason,
        "blocked": item.blocked,
        "registry_status": item.registry_status,
        "first_seen_at": item.first_seen_at,
        "last_seen_at": item.last_seen_at,
        "last_ingested_at": item.last_ingested_at,
        "last_error": item.last_error,
        "duplicate_entries": item.duplicate_entries,
        "duplicate_override": item.duplicate_override,
        "override_reason": item.override_reason,
        "override_by": item.override_by,
        "override_at": item.override_at,
    }) for item in candidates]


@app.get("/documents/timeline")
def document_timeline(
    filename: str = Query(..., min_length=1),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    _ = current_user
    safe_name = Path(str(filename)).name
    if not safe_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Nom de fichier invalide.")
    timeline = ingestion_report_service.build_document_timeline(safe_name)
    return {"filename": safe_name, "events": timeline}


@app.get("/documents/ingestion-report")
def export_ingestion_report(
    format: str = Query(default="csv", pattern="^(csv|pdf)$"),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> Response:
    _ = current_user
    sqlite_path = Path("data/indexes/medical_rag.sqlite")
    indexed_ids = _indexed_doc_ids_from_sqlite(sqlite_path)
    rows = ingestion_report_service.build_report_rows(indexed_doc_ids=indexed_ids)
    if format == "pdf":
        content = ingestion_report_service.to_simple_pdf_bytes(rows)
        filename = "ingestion-report.pdf"
        media_type = "application/pdf"
    else:
        content = ingestion_report_service.to_csv_bytes(rows)
        filename = "ingestion-report.csv"
        media_type = "text/csv; charset=utf-8"
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/documents/resync-registry", response_model=ResyncDocsRegistryResponse)
def resync_docs_registry(current_user: dict[str, Any] = Depends(get_current_user)) -> ResyncDocsRegistryResponse:
    _require_ops_permission(current_user)
    sqlite_path = Path("data/indexes/medical_rag.sqlite")
    indexed_ids = _indexed_doc_ids_from_sqlite(sqlite_path)
    result = ingestion_service.resync_docs_registry(indexed_doc_ids=indexed_ids)
    audit_service.log_event(
        event_type="docs_registry_resync",
        actor_user_id=str(current_user.get("id") or ""),
        actor_email=str(current_user.get("email") or ""),
        target_type="docs_registry",
        target_id="global",
        status="success",
        payload={"indexed_ids_count": len(indexed_ids)},
        result=result,
    )
    return ResyncDocsRegistryResponse(success=True, **result)


@app.post("/documents/duplicates/override", response_model=DuplicateOverrideResponse)
def set_duplicate_override(
    payload: DuplicateOverrideRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> DuplicateOverrideResponse:
    _require_ops_permission(current_user)
    if bool(payload.enabled) and not str(payload.reason or "").strip():
        raise HTTPException(status_code=400, detail="Un motif est obligatoire pour autoriser un doublon.")
    try:
        changed = ingestion_service.set_duplicate_override(
            filename=str(payload.filename),
            enabled=bool(payload.enabled),
            reason=str(payload.reason or "").strip() if payload.reason else None,
            updated_by=str(current_user.get("email") or current_user.get("id") or ""),
        )
    except RuntimeError as exc:
        audit_service.log_event(
            event_type="docs_duplicate_override",
            actor_user_id=str(current_user.get("id") or ""),
            actor_email=str(current_user.get("email") or ""),
            target_type="document",
            target_id=str(payload.filename),
            status="error",
            payload={"enabled": bool(payload.enabled), "reason": payload.reason},
            result={"error": str(exc)},
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    audit_service.log_event(
        event_type="docs_duplicate_override",
        actor_user_id=str(current_user.get("id") or ""),
        actor_email=str(current_user.get("email") or ""),
        target_type="document",
        target_id=str(payload.filename),
        status="success",
        payload={"enabled": bool(payload.enabled), "reason": payload.reason},
        result=changed,
    )
    return DuplicateOverrideResponse(
        success=True,
        filename=str(changed.get("filename") or ""),
        enabled=bool(changed.get("enabled")),
        reason=changed.get("reason"),
        updated_by=changed.get("updated_by"),
        updated_at=changed.get("updated_at"),
    )


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
    request: Request,
    files: list[UploadFile] = File(...),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    _ = current_user
    _enforce_rate_limit(
        request=request,
        scope="upload_user",
        key=str(current_user.get("id") or _client_ip(request)),
        limit=max(5, int(config.RATE_LIMIT_UPLOAD_PER_WINDOW)),
    )
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
    request: Request,
    payload: DocsIngestRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    _enforce_rate_limit(
        request=request,
        scope="upload_from_docs_user",
        key=str(current_user.get("id") or _client_ip(request)),
        limit=max(5, int(config.RATE_LIMIT_UPLOAD_PER_WINDOW)),
    )
    filenames = [str(name or "").strip() for name in payload.filenames if str(name or "").strip()]
    if not filenames:
        raise HTTPException(status_code=400, detail="Aucun fichier sélectionné depuis docs/.")
    sqlite_path = Path("data/indexes/medical_rag.sqlite")
    indexed_ids = _indexed_doc_ids_from_sqlite(sqlite_path)
    try:
        results = ingestion_service.ingest_docs_by_filenames(filenames, indexed_doc_ids=indexed_ids)
    except RuntimeError as exc:
        audit_service.log_event(
            event_type="ingestion_sync_failed",
            actor_user_id=str(current_user.get("id") or ""),
            actor_email=str(current_user.get("email") or ""),
            target_type="ingestion_sync",
            target_id="docs",
            status="error",
            payload={"filenames": filenames},
            result={"error": str(exc)},
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        audit_service.log_event(
            event_type="ingestion_sync_failed",
            actor_user_id=str(current_user.get("id") or ""),
            actor_email=str(current_user.get("email") or ""),
            target_type="ingestion_sync",
            target_id="docs",
            status="error",
            payload={"filenames": filenames},
            result={"error": str(exc)},
        )
        raise HTTPException(status_code=500, detail=f"Échec ingestion docs/: {exc}") from exc
    _refresh_resolver_cache()
    audit_service.log_event(
        event_type="ingestion_sync_completed",
        actor_user_id=str(current_user.get("id") or ""),
        actor_email=str(current_user.get("email") or ""),
        target_type="ingestion_sync",
        target_id="docs",
        status="success",
        payload={"filenames": filenames},
        result={"ingested_count": len(results)},
    )
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


@app.post("/upload/from-docs/jobs", response_model=IngestionJobStartResponse)
def start_upload_from_docs_job(
    request: Request,
    payload: DocsIngestRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> IngestionJobStartResponse:
    _enforce_rate_limit(
        request=request,
        scope="upload_docs_job_user",
        key=str(current_user.get("id") or _client_ip(request)),
        limit=max(5, int(config.RATE_LIMIT_UPLOAD_PER_WINDOW)),
    )
    filenames = [str(name or "").strip() for name in payload.filenames if str(name or "").strip()]
    if not filenames:
        raise HTTPException(status_code=400, detail="Aucun fichier sélectionné depuis docs/.")
    sqlite_path = Path("data/indexes/medical_rag.sqlite")
    indexed_ids = _indexed_doc_ids_from_sqlite(sqlite_path)
    try:
        ingestion_service.validate_docs_selection(filenames, indexed_doc_ids=indexed_ids)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    job = _ingestion_jobs().start_docs_ingestion_job(
        owner_user_id=str(current_user.get("id") or ""),
        filenames=filenames,
    )
    audit_service.log_event(
        event_type="ingestion_job_created",
        actor_user_id=str(current_user.get("id") or ""),
        actor_email=str(current_user.get("email") or ""),
        target_type="ingestion_job",
        target_id=job.job_id,
        status="success",
        payload={"filenames": filenames},
        result={"job_id": job.job_id},
    )
    return IngestionJobStartResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        message=job.message,
    )


@app.get("/upload/jobs/{job_id}", response_model=IngestionJobStatusResponse)
def get_upload_job_status(
    job_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> IngestionJobStatusResponse:
    job = _ingestion_jobs().get_job_for_user(
        job_id=str(job_id),
        owner_user_id=str(current_user.get("id") or ""),
    )
    if not job:
        raise HTTPException(status_code=404, detail="Job introuvable.")
    if job.status == "success":
        _refresh_resolver_cache()
    return IngestionJobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        message=job.message,
        error=job.error,
        progress_percent=max(0, min(100, int(job.progress_percent))),
        result=job.result,
    )


@app.post("/audio/transcribe")
async def audio_transcribe(
    request: Request,
    audio: UploadFile = File(...),
    debug: bool = Query(False),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    _enforce_rate_limit(
        request=request,
        scope="audio_transcribe_user",
        key=str(current_user.get("id") or _client_ip(request)),
        limit=max(10, int(config.RATE_LIMIT_CHAT_PER_WINDOW)),
    )
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
