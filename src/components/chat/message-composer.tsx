"use client";

import { ChevronDown, Loader2, Send, Upload } from "lucide-react";
import { useState } from "react";
import { useChatActions } from "@/hooks/use-chat-actions";
import { VoiceRecorder } from "@/components/audio/voice-recorder";
import { useAuthStore } from "@/store/auth-store";
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
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);

  async function onSend() {
    const trimmed = value.trim();
    if (!trimmed || sending || !isAuthenticated) return;
    setValue("");
    try {
      await sendMessage({ content: trimmed, mode });
    } catch {
      // Error state is rendered in chat messages.
    }
  }

  return (
    <div className="border-t border-border/70 bg-bg/70 px-4 py-4 backdrop-blur-xl">
      <div className="glass mx-auto flex max-w-5xl items-end gap-2 rounded-2xl p-2.5">
        <div className="relative hidden shrink-0 sm:block">
          <select
            aria-label="Mode"
            value={mode}
            onChange={(e) => setMode(e.target.value as ChatMode)}
            disabled={sending}
            className="h-10 appearance-none rounded-lg border border-border/80 bg-card/75 py-1 pl-3 pr-8 text-xs font-medium text-fg/[0.82] outline-none transition focus:border-accent/50"
          >
            {modes.map((m) => (
              <option key={m.value} value={m.value}>
                {m.label}
              </option>
            ))}
          </select>
          <ChevronDown className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 text-fg/[0.45]" size={14} />
        </div>
        <textarea
          aria-label="Message"
          disabled={sending || !isAuthenticated}
          className="max-h-40 min-h-10 flex-1 resize-y rounded-xl bg-transparent px-3 py-2 text-sm leading-6 outline-none placeholder:text-fg/[0.42]"
          placeholder={isAuthenticated ? "Écrire une question clinique..." : "Connectez-vous pour discuter"}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void onSend();
            }
          }}
        />
        <button aria-label="Upload document" disabled={sending || !isAuthenticated} className="icon-button">
          <Upload size={16} />
        </button>
        <div className={sending || !isAuthenticated ? "pointer-events-none opacity-50" : ""}>
          <VoiceRecorder onTranscript={(t) => setValue((prev) => `${prev} ${t}`.trim())} />
        </div>
        <button
          aria-label="Envoyer"
          disabled={sending || value.trim().length === 0 || !isAuthenticated}
          onClick={() => void onSend()}
          className="inline-flex h-9 w-9 items-center justify-center rounded-lg bg-accent text-white shadow-sm transition hover:bg-accent/90 disabled:cursor-not-allowed disabled:opacity-45"
        >
          {sending ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
        </button>
      </div>
    </div>
  );
}
