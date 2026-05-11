import type { ChatMode, ChatSource, MessageItem, RagResponse } from "@/types/chat";

const API_URL = process.env.NEXT_PUBLIC_RAG_API_URL;

interface ChatPayload {
  chat_id: string;
  message: string;
  history: Array<{ role: string; content: string }>;
  document_id?: string;
  mode: ChatMode;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  if (!API_URL) throw new Error("API URL unavailable");
  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
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

export async function sendChat(payload: ChatPayload): Promise<RagResponse> {
  const response = await request<RagResponse>("/chat", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  const sources = Array.isArray(response.sources) ? (response.sources as ChatSource[]) : undefined;
  return { ...response, sources };
}

export async function transcribeAudio(blob: Blob): Promise<string> {
  if (!API_URL) return "";
  const formData = new FormData();
  formData.append("audio", blob);
  const res = await fetch(`${API_URL}/audio/transcribe`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) return "";
  const data = (await res.json()) as { transcript?: string };
  return data.transcript || "";
}

export function toHistory(messages: MessageItem[]) {
  return messages.map((m) => ({ role: m.role, content: m.content }));
}
