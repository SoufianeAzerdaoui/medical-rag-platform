"use client";

import { ChevronDown, Loader2, Send, Upload } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useChatActions } from "@/hooks/use-chat-actions";
import { VoiceRecorder } from "@/components/audio/voice-recorder";
import { useAuthStore } from "@/store/auth-store";
import { useChatStore } from "@/store/chat-store";
import { getActiveModelApi, getConversationContextUsageApi, type ActiveModelInfo, type ConversationContextUsageInfo } from "@/services/rag-api";
import { ModelBadge } from "@/components/chat/model-badge";
import type { ChatMode } from "@/types/chat";

type PromptMode = "general" | "summary" | "anomalies" | "comparison" | "sources_only" | "simple_explanation";

type PromptModeConfig = {
  label: string;
  value: PromptMode;
  backendMode: ChatMode;
  placeholder: string;
  preface?: string;
};

const modes: PromptModeConfig[] = [
  {
    label: "Général",
    value: "general",
    backendMode: "general",
    placeholder: "Ex : Interprète ce bilan de manière prudente avec sources.",
  },
  {
    label: "Résumé",
    value: "summary",
    backendMode: "summary",
    placeholder: "Ex : Résume ce rapport biologique en points clés.",
  },
  {
    label: "Anomalies",
    value: "anomalies",
    backendMode: "document_analysis",
    placeholder: "Ex : Quels résultats sont hors des valeurs physiologiques ?",
    preface: "Mode anomalies: identifie uniquement les résultats hors référence et leur statut technique.",
  },
  {
    label: "Comparaison",
    value: "comparison",
    backendMode: "comparison",
    placeholder: "Ex : Compare report 16 et report 24",
  },
  {
    label: "Sources uniquement",
    value: "sources_only",
    backendMode: "general",
    placeholder: "Ex : Donne uniquement les passages sources pour cette question.",
    preface: "Mode sources uniquement: réponds uniquement avec les éléments sourcés et sans interprétation additionnelle.",
  },
  {
    label: "Explication simple",
    value: "simple_explanation",
    backendMode: "general",
    placeholder: "Ex : Explique simplement la TSH élevée pour un non spécialiste.",
    preface: "Mode explication simple: explique en langage clair, prudent, sans diagnostic.",
  },
];

function modeConfig(mode: PromptMode): PromptModeConfig {
  return modes.find((m) => m.value === mode) || modes[0];
}

export function MessageComposer() {
  const [value, setValue] = useState("");
  const [mode, setMode] = useState<PromptMode>("general");
  const [activeModel, setActiveModel] = useState<ActiveModelInfo | null>(null);
  const [contextUsage, setContextUsage] = useState<ConversationContextUsageInfo | null>(null);
  const { sendMessage, sending } = useChatActions();
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const accessToken = useAuthStore((s) => s.accessToken);
  const chats = useChatStore((s) => s.chats);
  const activeConversationId = useChatStore((s) => s.activeConversationId);
  const activeChat = useMemo(
    () => chats.find((chat) => chat.conversationId === activeConversationId || chat.id === activeConversationId) || null,
    [activeConversationId, chats],
  );
  const messageCount = activeChat?.messages.length ?? 0;

  useEffect(() => {
    if (!isAuthenticated || !accessToken) {
      setActiveModel(null);
      return;
    }
    let cancelled = false;
    void getActiveModelApi(accessToken)
      .then((payload) => {
        if (!cancelled) setActiveModel(payload);
      })
      .catch(() => {
        if (!cancelled) setActiveModel(null);
      });
    return () => {
      cancelled = true;
    };
  }, [accessToken, isAuthenticated]);

  useEffect(() => {
    if (!isAuthenticated || !accessToken || !activeConversationId) {
      setContextUsage(null);
      return;
    }
    let cancelled = false;
    const timer = setTimeout(() => {
      void getConversationContextUsageApi(activeConversationId, accessToken)
        .then((payload) => {
          if (!cancelled) setContextUsage(payload);
        })
        .catch(() => {
          if (!cancelled) setContextUsage(null);
        });
    }, 120);

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [accessToken, activeConversationId, isAuthenticated, messageCount, sending]);

  async function onSend() {
    const trimmed = value.trim();
    if (!trimmed || sending || !isAuthenticated) return;
    const config = modeConfig(mode);
    const content = config.preface ? `${config.preface}\n\n${trimmed}` : trimmed;
    setValue("");
    try {
      await sendMessage({ content, mode: config.backendMode });
    } catch {
      // Error state is rendered in chat messages.
    }
  }

  return (
    <div className="border-t border-border/70 bg-bg/70 px-4 py-4 backdrop-blur-xl">
      <div className="glass mx-auto flex max-w-5xl flex-nowrap items-center gap-2 rounded-2xl p-2.5">
        <div className="relative hidden shrink-0 sm:block">
          <select
            aria-label="Mode"
            value={mode}
            onChange={(e) => setMode(e.target.value as PromptMode)}
            disabled={sending}
            className="h-10 appearance-none rounded-lg border border-border/80 bg-card/85 py-1 pl-3 pr-8 text-xs font-medium text-fg/[0.82] outline-none transition focus:border-accent/50 focus-visible:ring-2 focus-visible:ring-accent/35"
          >
            {modes.map((m) => (
              <option key={m.value} value={m.value}>
                {m.label}
              </option>
            ))}
          </select>
          <ChevronDown className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 text-fg/[0.45]" size={14} />
        </div>
        <ModelBadge
          modelName={contextUsage?.model || activeModel?.model}
          contextWindow={contextUsage?.context_window || activeModel?.context_window}
          usedTokens={contextUsage?.used_tokens}
          usagePercent={contextUsage?.usage_percent}
          status={contextUsage?.status}
        />
        <span className="hidden h-6 w-px shrink-0 bg-border/70 sm:block" aria-hidden="true" />
        <textarea
          aria-label="Message"
          disabled={sending || !isAuthenticated}
          className="max-h-40 min-h-10 flex-1 resize-y rounded-xl bg-transparent px-3 py-2 text-sm leading-6 outline-none placeholder:text-fg/[0.42] focus-visible:ring-2 focus-visible:ring-accent/30"
          placeholder={isAuthenticated ? modeConfig(mode).placeholder : "Connectez-vous pour discuter"}
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
          className="inline-flex h-9 w-9 items-center justify-center rounded-lg bg-accent text-slate-950 shadow-sm transition hover:bg-accent/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/55 disabled:cursor-not-allowed disabled:opacity-45"
        >
          {sending ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
        </button>
      </div>
    </div>
  );
}
