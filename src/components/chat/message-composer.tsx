"use client";

import { Loader2, Send, Upload } from "lucide-react";
import { useState } from "react";
import { useChatActions } from "@/hooks/use-chat-actions";
import { VoiceRecorder } from "@/components/audio/voice-recorder";
import type { ChatMode } from "@/types/chat";

const modes: Array<{ label: string; value: ChatMode }> = [
  { label: "Général", value: "general" },
  { label: "Analyse document", value: "document_analysis" },
  { label: "Comparaison", value: "comparison" },
  { label: "Résumé", value: "summary" },
];

export function MessageComposer() {
  const [value, setValue] = useState("");
  const [mode, setMode] = useState<ChatMode>("general");
  const { sendMessage, sending } = useChatActions();

  async function onSend() {
    const trimmed = value.trim();
    if (!trimmed || sending) return;
    setValue("");
    try {
      await sendMessage({ content: trimmed, mode });
    } catch {
      // Error state is rendered in chat messages.
    }
  }

  return (
    <div className="border-t border-border p-4">
      <div className="glass mx-auto flex max-w-4xl items-end gap-2 rounded-2xl p-3">
        <select
          aria-label="Mode"
          value={mode}
          onChange={(e) => setMode(e.target.value as ChatMode)}
          disabled={sending}
          className="rounded-lg border border-border bg-transparent px-2 py-1 text-xs"
        >
          {modes.map((m) => (
            <option key={m.value} value={m.value}>
              {m.label}
            </option>
          ))}
        </select>
        <textarea
          aria-label="Message"
          disabled={sending}
          className="max-h-40 min-h-10 flex-1 resize-y bg-transparent p-2 text-sm outline-none"
          placeholder="Écrire une question clinique..."
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void onSend();
            }
          }}
        />
        <button aria-label="Upload document" disabled={sending} className="rounded-lg border border-border p-2 disabled:opacity-50">
          <Upload size={16} />
        </button>
        <div className={sending ? "pointer-events-none opacity-50" : ""}>
          <VoiceRecorder onTranscript={(t) => setValue((prev) => `${prev} ${t}`.trim())} />
        </div>
        <button
          aria-label="Envoyer"
          disabled={sending || value.trim().length === 0}
          onClick={() => void onSend()}
          className="inline-flex items-center justify-center rounded-lg bg-accent/30 p-2 disabled:opacity-50"
        >
          {sending ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
        </button>
      </div>
    </div>
  );
}
