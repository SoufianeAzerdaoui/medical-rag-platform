import type { ChatMode, ChatSource, MessageItem, RagResponse } from "@/types/chat";

const API_URL = process.env.NEXT_PUBLIC_RAG_API_URL;

export interface AuthUser {
  id: string;
  email: string;
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

  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    headers,
  });

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
  });
  const sources = Array.isArray(response.sources) ? (response.sources as ChatSource[]) : undefined;
  return { ...response, sources };
}

export async function transcribeAudio(blob: Blob, token?: string): Promise<string> {
  if (!API_URL) return "";
  const formData = new FormData();
  formData.append("audio", blob);
  const headers = new Headers();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const res = await fetch(`${API_URL}/audio/transcribe`, {
    method: "POST",
    body: formData,
    headers,
  });
  if (!res.ok) return "";
  const data = (await res.json()) as { transcript?: string };
  return data.transcript || "";
}

export function toHistory(messages: MessageItem[]) {
  return messages.map((m) => ({ role: m.role, content: m.content }));
}
