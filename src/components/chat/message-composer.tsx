"use client";

import { ChevronDown, Loader2, Send, Upload } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useChatActions } from "@/hooks/use-chat-actions";
import { VoiceRecorder } from "@/components/audio/voice-recorder";
import { ContextWindowMeter } from "@/components/chat/context-window-meter";
import { getActiveModelApi, getConversationContextUsageApi, type ContextUsageStatus } from "@/services/rag-api";
import { useAuthStore } from "@/store/auth-store";
import { useChatStore } from "@/store/chat-store";
import type { ChatMode } from "@/types/chat";
import { useRouter } from "next/navigation";

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
  const { sendMessage, sending } = useChatActions();
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const token = useAuthStore((s) => s.accessToken);
  const activeConversationId = useChatStore((s) => s.activeConversationId);
  const chats = useChatStore((s) => s.chats);
  const startNewConversation = useChatStore((s) => s.startNewConversation);
  const router = useRouter();
  const [meter, setMeter] = useState<{
    model: string;
    contextWindow: number;
    usedTokens: number;
    remainingTokens: number;
    usagePercent: number;
    status: ContextUsageStatus;
  } | null>(null);

  const activeMessageCount = useMemo(
    () => chats.find((c) => c.id === activeConversationId)?.messages.length || 0,
    [activeConversationId, chats],
  );

  useEffect(() => {
    if (!isAuthenticated || !token || !activeConversationId) {
      setMeter(null);
      return;
    }
    const conversationId = activeConversationId;
    let active = true;
    async function loadContextUsage() {
      try {
        const [modelInfo, usage] = await Promise.all([
          getActiveModelApi(token),
          getConversationContextUsageApi(conversationId, token),
        ]);
        if (!active) return;
        setMeter({
          model: usage.model || modelInfo.model,
          contextWindow: usage.context_window || modelInfo.context_window,
          usedTokens: usage.used_tokens,
          remainingTokens: usage.remaining_tokens,
          usagePercent: usage.usage_percent,
          status: usage.status,
        });
      } catch {
        if (!active) return;
        setMeter(null);
      }
    }
    void loadContextUsage();
    const timer = window.setInterval(() => void loadContextUsage(), 20000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [activeConversationId, activeMessageCount, isAuthenticated, token]);

  async function onSend() {
    const trimmed = value.trim();
    if (!trimmed || sending || !isAuthenticated) return;
    if (meter?.status === "full") return;
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
      {meter ? (
        <div className="mx-auto mb-2 max-w-5xl">
          <ContextWindowMeter
            model={meter.model}
            contextWindow={meter.contextWindow}
            usedTokens={meter.usedTokens}
            remainingTokens={meter.remainingTokens}
            usagePercent={meter.usagePercent}
            status={meter.status}
          />
          {meter.status === "warning" ? (
            <p className="mt-1 text-xs text-amber-400">Le contexte de cette conversation devient long. Pensez à démarrer une nouvelle conversation.</p>
          ) : null}
          {meter.status === "full" ? (
            <div className="mt-1 flex items-center justify-between gap-2 rounded-lg border border-rose-500/35 bg-rose-500/10 px-3 py-2 text-xs text-rose-300">
              <span>Cette conversation est presque pleine. Créez une nouvelle conversation pour continuer avec de bonnes performances.</span>
              <button
                type="button"
                className="rounded-md border border-rose-400/35 bg-rose-500/10 px-2 py-1 font-medium"
                onClick={async () => {
                  const id = await startNewConversation(token);
                  if (id) router.push(`/chat/${id}`);
                }}
              >
                Nouvelle conversation
              </button>
            </div>
          ) : null}
        </div>
      ) : null}
      <div className="glass mx-auto flex max-w-5xl items-end gap-2 rounded-2xl p-2.5">
        <div className="relative hidden shrink-0 sm:block">
          <select
            aria-label="Mode"
            value={mode}
            onChange={(e) => setMode(e.target.value as PromptMode)}
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
          disabled={sending || value.trim().length === 0 || !isAuthenticated || meter?.status === "full"}
          onClick={() => void onSend()}
          className="inline-flex h-9 w-9 items-center justify-center rounded-lg bg-accent text-white shadow-sm transition hover:bg-accent/90 disabled:cursor-not-allowed disabled:opacity-45"
        >
          {sending ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
        </button>
      </div>
    </div>
  );
}
