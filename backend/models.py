from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatHistoryItem(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    conversation_id: str | None = Field(default=None, min_length=1)
    chat_id: str | None = Field(default=None, min_length=1)
    message: str = Field(..., min_length=1)
    history: list[ChatHistoryItem] = Field(default_factory=list)
    document_id: str | None = None
    mode: Literal["general", "document_analysis", "comparison", "summary"] = "general"
    summary_style: Literal["short", "editorial"] | None = None
    llm_provider_override: str | None = Field(default=None, min_length=1)
    llm_model_override: str | None = Field(default=None, min_length=1)

    def resolved_conversation_id(self) -> str:
        return str(self.conversation_id or self.chat_id or "").strip()


class AuthRegisterRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=320)
    password: str = Field(..., min_length=8, max_length=128)


class AuthLoginRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=320)
    password: str = Field(..., min_length=8, max_length=128)


class UserResponse(BaseModel):
    id: str
    email: str
    role: str = "user"
    created_at: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    user: UserResponse


class LogoutResponse(BaseModel):
    success: bool


class ConversationCreateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=240)


class ConversationItem(BaseModel):
    id: str
    user_id: str
    title: str
    created_at: str
    updated_at: str


class MessageItemResponse(BaseModel):
    id: str
    conversation_id: str
    role: Literal["user", "assistant", "system"]
    content: str
    sources: list[SourceItem] = Field(default_factory=list)
    diagnostics: dict[str, Any] | None = None
    created_at: str


class ConversationClearRequest(BaseModel):
    conversation_id: str = Field(..., min_length=1)


class ConversationClearResponse(BaseModel):
    success: bool
    conversation_id: str


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
    doc_id: str | None = None
    filename: str | None = None
    row: int | None = None
    label: str | None = None
    url: str | None = None
    viewer_url: str | None = None


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    sources: list[SourceItem] = Field(default_factory=list)
    confidence: float | None = None
    document_ids: list[str] = Field(default_factory=list)
    response_time: float | None = None
    quality_report: dict[str, Any] | None = None
    validation_status: Literal["pass", "warning", "fail"] | None = None
    generation_mode: str | None = None
    generation_writer: str | None = None
    provider: str | None = None
    model: str | None = None
    llm_provider_effective_runtime: str | None = None
    llm_model_effective_runtime: str | None = None
    llm_quality_escalation_used: bool | None = None
    llm_quality_escalation_reason: str | None = None
    selected_route: str | None = None
    llm_writer_attempted: bool | None = None
    llm_writer_accepted: bool | None = None
    final_answer_source: Literal["llm_writer", "deterministic_renderer"] | None = None
    renderer_used: str | None = None
    fallback_reason: str | None = None
    visualization: dict[str, Any] | None = None
    chart_data: dict[str, Any] | None = None
    patients: list[dict[str, Any]] | None = None
    inventory_view: dict[str, Any] | None = None
    displayed_evidences: list[dict[str, Any]] = Field(default_factory=list)
    debug: dict[str, Any] | None = None


class DocumentItem(BaseModel):
    id: str
    name: str


class DocsDiscoveryItem(BaseModel):
    filename: str
    doc_id: str
    absolute_path: str
    size_bytes: int
    modified_at: str
    file_hash: str
    text_hash: str | None = None
    already_indexed: bool
    is_duplicate: bool = False
    duplicate_with: list[str] = Field(default_factory=list)
    duplicate_reason: str | None = None
    blocked: bool = False
    registry_status: str | None = None
    first_seen_at: str | None = None
    last_seen_at: str | None = None
    last_ingested_at: str | None = None
    last_error: str | None = None
    duplicate_entries: list[dict[str, Any]] = Field(default_factory=list)
    duplicate_override: bool = False
    override_reason: str | None = None
    override_by: str | None = None
    override_at: str | None = None


class DocsIngestRequest(BaseModel):
    filenames: list[str] = Field(default_factory=list)


class DuplicateOverrideRequest(BaseModel):
    filename: str = Field(..., min_length=1)
    enabled: bool = True
    reason: str | None = Field(default=None, max_length=500)


class DuplicateOverrideResponse(BaseModel):
    success: bool
    filename: str
    enabled: bool
    reason: str | None = None
    updated_by: str | None = None
    updated_at: str | None = None


class IngestionJobStartResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "success", "error"]
    created_at: str
    message: str | None = None


class IngestionJobStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "success", "error"]
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    message: str | None = None
    error: str | None = None
    progress_percent: int = 0
    result: dict[str, Any] | None = None


class ResyncDocsRegistryResponse(BaseModel):
    success: bool
    discovered_count: int
    indexed_count: int
    duplicate_count: int


class ActiveModelResponse(BaseModel):
    provider: str
    model: str
    context_window: int
    max_output_tokens: int
    recommended_rag_budget: int | None = None


class ConversationContextUsageResponse(BaseModel):
    conversation_id: str
    model: str
    context_window: int
    used_tokens: int
    remaining_tokens: int
    usage_percent: float
    status: Literal["safe", "medium", "warning", "full"]


class FeatureFlagItemResponse(BaseModel):
    name: str
    enabled: bool
    description: str
    updated_at: str
    updated_by: str


class FeatureFlagUpdateRequest(BaseModel):
    enabled: bool
