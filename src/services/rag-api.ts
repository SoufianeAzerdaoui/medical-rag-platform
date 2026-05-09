import { uid } from "@/lib/utils";
import type { ChatMode, MessageItem, RagResponse } from "@/types/chat";

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
  try {
    return await request<RagResponse>("/chat", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  } catch (error) {
    const reason = error instanceof Error ? error.message : "Unknown error";
    return {
      answer:
        `Erreur backend: ${reason}. Mode demo activé. Cette réponse ne remplace pas l'avis médical.`,
      confidence: 0.45,
      response_time: 1.1,
      sources: [
        {
          id: uid("src"),
          documentName: "Demo Clinical Note",
          page: 1,
          section: "Résumé",
          excerpt: `Aucune donnée backend disponible. Détail erreur: ${reason}.`,
          score: 0.45,
          warning: "Vérifier les données sur le backend réel.",
        },
      ],
    };
  }
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
