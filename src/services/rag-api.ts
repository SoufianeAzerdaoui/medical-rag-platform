import type { ChatMode, ChatSource, MessageItem, RagResponse } from "@/types/chat";

const API_URL = process.env.NEXT_PUBLIC_RAG_API_URL;

export interface AuthUser {
  id: string;
  email: string;
  role?: string;
  created_at: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: "bearer";
  user: AuthUser;
}

export interface ConversationResponse {
  id: string;
  user_id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface BackendMessageResponse {
  id: string;
  conversation_id: string;
  role: "user" | "assistant" | "system";
  content: string;
  sources?: ChatSource[];
  diagnostics?: Record<string, unknown> | null;
  created_at: string;
}

export interface DocumentRecord {
  id: string;
  name: string;
}

export interface DocsDiscoveryRecord {
  filename: string;
  doc_id: string;
  absolute_path: string;
  size_bytes: number;
  modified_at: string;
  file_hash: string;
  text_hash?: string | null;
  already_indexed: boolean;
  is_duplicate: boolean;
  duplicate_with: string[];
  duplicate_reason?: string | null;
  blocked: boolean;
  registry_status?: string | null;
  first_seen_at?: string | null;
  last_seen_at?: string | null;
  last_ingested_at?: string | null;
  last_error?: string | null;
  duplicate_entries: Array<{
    filename: string;
    absolute_path: string;
    doc_id: string;
    is_indexed: boolean;
    status: string;
    first_seen_at?: string | null;
    last_seen_at?: string | null;
    last_ingested_at?: string | null;
    last_error?: string | null;
  }>;
  duplicate_override: boolean;
  override_reason?: string | null;
  override_by?: string | null;
  override_at?: string | null;
}

export interface DuplicateOverrideResponse {
  success: boolean;
  filename: string;
  enabled: boolean;
  reason?: string | null;
  updated_by?: string | null;
  updated_at?: string | null;
}

export interface ResyncDocsRegistryResponse {
  success: boolean;
  discovered_count: number;
  indexed_count: number;
  duplicate_count: number;
}

export interface MonitoringSummaryResponse {
  avg_pipeline_seconds: number;
  pipeline_success_total: number;
  pipeline_failure_total: number;
  indexing_errors_total: number;
  queue_depth: number;
  generated_at: string;
}

export interface SecurityStatusResponse {
  server_time: string;
  clamav: {
    required: boolean;
    command: string;
    available: boolean;
    version?: string;
    healthy: boolean;
  };
  sentry: {
    configured: boolean;
    dsn_masked: boolean;
  };
  jwt: {
    algorithm: string;
    expire_minutes: number;
    rotation_previous_count: number;
  };
  encryption: {
    enabled: boolean;
    required: boolean;
    key_configured: boolean;
  };
  rate_limits: {
    window_seconds: number;
    auth_per_window: number;
    chat_per_window: number;
    upload_per_window: number;
    login_max_failures: number;
    login_block_seconds: number;
  };
  retention: {
    jobs_days: number;
    audit_days: number;
    docs_days: number;
    audio_days: number;
    logs_days: number;
    auth_attempts_days: number;
  };
}

export interface RetentionRunResponse {
  success: boolean;
  dry_run: boolean;
  hard_delete_docs: boolean;
  jobs_deleted: number;
  audit_deleted: number;
  auth_attempts_deleted: number;
  docs_registry_deleted: number;
  docs_files_deleted: number;
  audio_files_deleted: number;
  log_files_deleted: number;
  audit_delete_blocked_immutable?: boolean;
}

export interface DocumentTimelineEvent {
  at: string;
  type: string;
  title: string;
  detail?: string;
  actor?: string;
}

export interface DocumentTimelineResponse {
  filename: string;
  events: DocumentTimelineEvent[];
}

export interface UploadIngestedItem {
  filename: string;
  doc_id: string;
  stored_path: string;
  extraction_dir: string;
}

export interface UploadResponse {
  success: boolean;
  ingested_count: number;
  ingested: UploadIngestedItem[];
  skipped: Array<{ filename: string; reason: string }>;
}

const INGESTION_TIMEOUT_MS = 20 * 60 * 1000;

export interface IngestionJobStartResponse {
  job_id: string;
  status: "queued" | "running" | "success" | "error";
  created_at: string;
  message?: string | null;
}

export interface IngestionJobStatusResponse {
  job_id: string;
  status: "queued" | "running" | "success" | "error";
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  message?: string | null;
  error?: string | null;
  progress_percent: number;
  result?: UploadResponse | null;
}

export interface ActiveModelInfo {
  provider: string;
  model: string;
  context_window: number;
  max_output_tokens: number;
  recommended_rag_budget?: number | null;
}

export interface ConversationContextUsageInfo {
  conversation_id: string;
  model: string;
  context_window: number;
  used_tokens: number;
  remaining_tokens: number;
  usage_percent: number;
  status: "safe" | "medium" | "warning" | "full";
}

export interface TranscribeDebugAttempt {
  strategy: string;
  language: string;
  vad_filter: boolean;
  transcript_preview: string;
  transcript_chars: number;
  quality_score: number;
  rejected_reason: string | null;
  mean_no_speech: number;
  mean_avg_logprob: number;
  voiced_segments: number;
}

export interface TranscribeDebugInfo {
  quality_score: number;
  rejected_reason: string | null;
  accepted_strategy: string | null;
  attempts: TranscribeDebugAttempt[];
}

export interface TranscribeResponse {
  transcript: string;
  debug?: TranscribeDebugInfo;
}

interface ChatPayload {
  conversation_id?: string;
  chat_id?: string;
  message: string;
  history: Array<{ role: string; content: string }>;
  document_id?: string;
  mode: ChatMode;
}

interface RequestOptions extends RequestInit {
  token?: string | null;
  timeoutMs?: number;
}

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(`API error ${status}${detail ? `: ${detail}` : ""}`);
    this.status = status;
    this.detail = detail;
  }
}

function formatApiDetail(detail: unknown): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => {
        if (typeof item === "string") return item;
        if (item && typeof item === "object" && "msg" in item) {
          const msg = (item as { msg?: unknown }).msg;
          return typeof msg === "string" ? msg : JSON.stringify(item);
        }
        return JSON.stringify(item);
      })
      .filter((msg) => Boolean(msg && msg.trim()));
    return messages.join("; ");
  }
  if (detail && typeof detail === "object") {
    if ("msg" in detail) {
      const msg = (detail as { msg?: unknown }).msg;
      if (typeof msg === "string") return msg;
    }
    return JSON.stringify(detail);
  }
  return "";
}

async function request<T>(path: string, init?: RequestOptions): Promise<T> {
  if (!API_URL) throw new Error("API URL unavailable");
  const headers = new Headers(init?.headers || {});
  if (!headers.has("Content-Type") && !(init?.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }
  if (init?.token) {
    headers.set("Authorization", `Bearer ${init.token}`);
  }

  const timeoutMs = init?.timeoutMs ?? 90_000;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, {
      ...init,
      headers,
      signal: controller.signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new ApiError(408, "Le serveur a mis trop de temps à répondre.");
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }

  if (!res.ok) {
    let detail = "";
    try {
      const payload = (await res.json()) as { detail?: string };
      detail = formatApiDetail(payload.detail);
    } catch {
      detail = "";
    }
    throw new ApiError(res.status, detail);
  }

  if (res.status === 204) {
    return {} as T;
  }
  return res.json() as Promise<T>;
}

export async function healthcheck() {
  try {
    await request("/health");
    return "online";
  } catch {
    return "offline";
  }
}

export async function registerUser(payload: { email: string; password: string }): Promise<AuthResponse> {
  return request<AuthResponse>("/auth/register", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function loginUser(payload: { email: string; password: string }): Promise<AuthResponse> {
  return request<AuthResponse>("/auth/login", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function loadMe(token: string): Promise<AuthUser> {
  return request<AuthUser>("/auth/me", { token });
}

export async function logoutUser(token: string): Promise<{ success: boolean }> {
  return request<{ success: boolean }>("/auth/logout", {
    method: "POST",
    token,
  });
}

export async function listConversations(token: string): Promise<ConversationResponse[]> {
  return request<ConversationResponse[]>("/conversations", { token });
}

export async function createConversation(token: string, payload?: { title?: string }): Promise<ConversationResponse> {
  return request<ConversationResponse>("/conversations", {
    method: "POST",
    token,
    body: JSON.stringify(payload || {}),
  });
}

export async function getConversationMessages(token: string, conversationId: string): Promise<BackendMessageResponse[]> {
  return request<BackendMessageResponse[]>(`/conversations/${conversationId}/messages`, { token });
}

export async function deleteConversation(token: string, conversationId: string): Promise<{ success: boolean }> {
  return request<{ success: boolean }>(`/conversations/${conversationId}`, {
    method: "DELETE",
    token,
  });
}

export async function clearConversation(token: string, conversationId: string): Promise<{ success: boolean; conversation_id: string }> {
  return request<{ success: boolean; conversation_id: string }>("/chat/clear", {
    method: "POST",
    token,
    body: JSON.stringify({ conversation_id: conversationId }),
  });
}

export async function sendChat(payload: ChatPayload, token: string): Promise<RagResponse> {
  const response = await request<RagResponse>("/chat", {
    method: "POST",
    token,
    body: JSON.stringify(payload),
    timeoutMs: 90_000,
  });
  const sources = Array.isArray(response.sources) ? (response.sources as ChatSource[]) : undefined;
  return { ...response, sources };
}

export async function listDocumentsApi(token?: string | null): Promise<DocumentRecord[]> {
  return request<DocumentRecord[]>("/documents", { token: token || null });
}

export async function discoverDocsApi(token?: string | null): Promise<DocsDiscoveryRecord[]> {
  return request<DocsDiscoveryRecord[]>("/documents/discover", { token: token || null });
}

export async function resyncDocsRegistryApi(token?: string | null): Promise<ResyncDocsRegistryResponse> {
  return request<ResyncDocsRegistryResponse>("/documents/resync-registry", {
    method: "POST",
    token: token || null,
  });
}

export async function getDocumentTimelineApi(
  filename: string,
  token?: string | null,
): Promise<DocumentTimelineResponse> {
  return request<DocumentTimelineResponse>(`/documents/timeline?filename=${encodeURIComponent(filename)}`, {
    token: token || null,
  });
}

export async function downloadIngestionReportApi(
  format: "csv" | "pdf",
  token?: string | null,
): Promise<Blob> {
  if (!API_URL) throw new Error("API URL unavailable");
  const headers = new Headers();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const res = await fetch(`${API_URL}/documents/ingestion-report?format=${format}`, {
    method: "GET",
    headers,
  });
  if (!res.ok) {
    let detail = "";
    try {
      const payload = (await res.json()) as { detail?: unknown };
      detail = formatApiDetail(payload.detail);
    } catch {
      detail = "";
    }
    throw new ApiError(res.status, detail || "Export ingestion report échoué.");
  }
  return res.blob();
}

export async function setDuplicateOverrideApi(
  payload: { filename: string; enabled: boolean; reason?: string | null },
  token?: string | null,
): Promise<DuplicateOverrideResponse> {
  return request<DuplicateOverrideResponse>("/documents/duplicates/override", {
    method: "POST",
    token: token || null,
    body: JSON.stringify(payload),
  });
}

export async function reindexDocumentApi(docId: string, token?: string | null): Promise<{ success: boolean }> {
  return request<{ success: boolean }>(`/documents/${encodeURIComponent(docId)}/reindex`, {
    method: "POST",
    token: token || null,
  });
}

export async function deleteDocumentApi(docId: string, token?: string | null): Promise<{ success: boolean }> {
  return request<{ success: boolean }>(`/documents/${encodeURIComponent(docId)}`, {
    method: "DELETE",
    token: token || null,
  });
}

export async function getActiveModelApi(token?: string | null): Promise<ActiveModelInfo> {
  return request<ActiveModelInfo>("/api/models/active", { token: token || null });
}

export async function getConversationContextUsageApi(
  conversationId: string,
  token?: string | null,
): Promise<ConversationContextUsageInfo> {
  return request<ConversationContextUsageInfo>(`/api/conversations/${encodeURIComponent(conversationId)}/context-usage`, {
    token: token || null,
  });
}

export async function getMonitoringSummaryApi(token?: string | null): Promise<MonitoringSummaryResponse> {
  return request<MonitoringSummaryResponse>("/monitoring/summary", { token: token || null });
}

export async function getSecurityStatusApi(token?: string | null): Promise<SecurityStatusResponse> {
  return request<SecurityStatusResponse>("/admin/security-status", { token: token || null });
}

export async function runRetentionApi(
  payload: { dryRun: boolean; hardDeleteDocs: boolean },
  token?: string | null,
): Promise<RetentionRunResponse> {
  const params = new URLSearchParams({
    dry_run: payload.dryRun ? "true" : "false",
    hard_delete_docs: payload.hardDeleteDocs ? "true" : "false",
  });
  return request<RetentionRunResponse>(`/admin/retention/run?${params.toString()}`, {
    method: "POST",
    token: token || null,
  });
}

export async function transcribeAudio(blob: Blob, token?: string): Promise<string> {
  const result = await transcribeAudioDetailed(blob, token);
  return result.transcript || "";
}

export async function transcribeAudioDetailed(blob: Blob, token?: string): Promise<TranscribeResponse> {
  if (!API_URL) throw new Error("API URL unavailable");
  const formData = new FormData();
  const mime = String(blob.type || "").toLowerCase();
  const filename =
    mime.includes("ogg") ? "recording.ogg" :
      mime.includes("webm") ? "recording.webm" :
        mime.includes("wav") ? "recording.wav" : "recording.webm";
  formData.append("audio", blob, filename);
  const headers = new Headers();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const debugEnabled = process.env.NODE_ENV !== "production";
  const url = `${API_URL}/audio/transcribe${debugEnabled ? "?debug=1" : ""}`;
  const res = await fetch(url, {
    method: "POST",
    body: formData,
    headers,
  });
  if (!res.ok) {
    let detail = "";
    try {
      const payload = (await res.json()) as { detail?: unknown };
      detail = formatApiDetail(payload.detail);
    } catch {
      detail = "";
    }
    throw new ApiError(res.status, detail || "Transcription audio échouée.");
  }
  const data = (await res.json()) as TranscribeResponse;
  return { transcript: data.transcript || "", debug: data.debug };
}

export async function uploadDocumentsApi(files: File[], token?: string | null): Promise<UploadResponse> {
  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }
  return request<UploadResponse>("/upload", {
    method: "POST",
    token: token || null,
    body: formData,
    timeoutMs: INGESTION_TIMEOUT_MS,
  });
}

export async function uploadFromDocsApi(filenames: string[], token?: string | null): Promise<UploadResponse> {
  return request<UploadResponse>("/upload/from-docs", {
    method: "POST",
    token: token || null,
    body: JSON.stringify({ filenames }),
    timeoutMs: INGESTION_TIMEOUT_MS,
  });
}

export async function startDocsIngestionJobApi(
  filenames: string[],
  token?: string | null,
): Promise<IngestionJobStartResponse> {
  return request<IngestionJobStartResponse>("/upload/from-docs/jobs", {
    method: "POST",
    token: token || null,
    body: JSON.stringify({ filenames }),
  });
}

export async function getIngestionJobStatusApi(
  jobId: string,
  token?: string | null,
): Promise<IngestionJobStatusResponse> {
  return request<IngestionJobStatusResponse>(`/upload/jobs/${encodeURIComponent(jobId)}`, {
    token: token || null,
  });
}

export function toHistory(messages: MessageItem[]) {
  return messages.map((m) => ({ role: m.role, content: m.content }));
}
