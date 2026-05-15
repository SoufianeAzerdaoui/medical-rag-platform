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
    generation_writer: Literal[
        "llm_writer",
        "professional_fallback",
        "deterministic_metadata_query",
        "deterministic_response_transform_json",
        "deterministic_context_summary",
    ] | None = None
    visualization: dict[str, Any] | None = None
    chart_data: dict[str, Any] | None = None
    patients: list[dict[str, Any]] | None = None
    inventory_view: dict[str, Any] | None = None


class DocumentItem(BaseModel):
    id: str
    name: str
