"use client";

import { ChevronDown, CircleCheckBig, Loader2, Send, Upload } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useChatActions } from "@/hooks/use-chat-actions";
import { VoiceRecorder } from "@/components/audio/voice-recorder";
import { useAuthStore } from "@/store/auth-store";
import { useChatStore } from "@/store/chat-store";
import { getActiveModelApi, getConversationContextUsageApi, type ActiveModelInfo, type ConversationContextUsageInfo } from "@/services/rag-api";
import { ModelBadge } from "@/components/chat/model-badge";
import type { ChatMode, SummaryStyle } from "@/types/chat";

type PromptMode = "general" | "summary" | "anomalies" | "comparison" | "sources_only" | "simple_explanation";
type LlmProvider = "ollama" | "gemini";
type LlmChoiceId = "local-qwen" | "gemini-cloud";

type PromptModeConfig = {
  label: string;
  value: PromptMode;
  backendMode: ChatMode;
  placeholder: string;
  preface?: string;
};

type LlmChoice = {
  id: LlmChoiceId;
  label: string;
  provider: LlmProvider;
  model: string;
  hint: string;
  contextWindow: number;
  maxOutputTokens: number;
};

const LLM_CHOICE_STORAGE_KEY = "medical-rag-selected-llm-choice";
const SUMMARY_STYLE_STORAGE_KEY = "medical-rag-summary-style";

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

const llmChoices: LlmChoice[] = [
  {
    id: "local-qwen",
    label: "Qwen local",
    provider: "ollama",
    model: "qwen2.5:7b-instruct",
    hint: "Modèle local Ollama",
    contextWindow: 32_768,
    maxOutputTokens: 4_096,
  },
  {
    id: "gemini-cloud",
    label: "Gemini cloud",
    provider: "gemini",
    model: "gemini-2.5-flash",
    hint: "Google AI Studio",
    contextWindow: 1_048_576,
    maxOutputTokens: 65_536,
  },
];

function modeConfig(mode: PromptMode): PromptModeConfig {
  return modes.find((m) => m.value === mode) || modes[0];
}

function inferBackendModeFromDraft(text: string, currentMode: PromptMode): PromptMode {
  if (currentMode !== "general") return currentMode;
  const normalized = text
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/\s+/g, " ")
    .trim();

  const summarySignals = [
    "resume",
    "resume medical",
    "resume medical clair et fidele",
    "resume biologique",
    "synthese",
    "bilan",
    "présente les résultats",
    "presente les resultats",
    "présente le résultat",
    "presente le resultat",
    "fais un resume medical",
    "fais un résumé médical",
    "fais une synthese medicale",
    "mets en avant",
    "en restant prudente",
    "en restant prudent",
    "sans interpretation excessive",
    "sans inventer",
    "points clés",
    "points cles",
  ];

  if (summarySignals.some((signal) => normalized.includes(signal))) {
    return "summary";
  }

  return currentMode;
}

export function MessageComposer() {
  const [selectedModelId, setSelectedModelId] = useState<LlmChoiceId>("local-qwen");
  const [summaryReportStyle, setSummaryReportStyle] = useState<SummaryStyle>("editorial");
  const [activeModel, setActiveModel] = useState<ActiveModelInfo | null>(null);
  const [contextUsage, setContextUsage] = useState<ConversationContextUsageInfo | null>(null);
  const modelSelectionInitialized = useRef(false);
  const userSelectedModel = useRef(false);
  const { sendMessage, sending } = useChatActions();
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const accessToken = useAuthStore((s) => s.accessToken);
  const chats = useChatStore((s) => s.chats);
  const activeConversationId = useChatStore((s) => s.activeConversationId);
  const composerDraft = useChatStore((s) => s.composerDraft);
  const composerPromptMode = useChatStore((s) => s.composerPromptMode);
  const setComposerDraft = useChatStore((s) => s.setComposerDraft);
  const setComposerPromptMode = useChatStore((s) => s.setComposerPromptMode);
  const clearComposerDraft = useChatStore((s) => s.clearComposerDraft);
  const activeChat = useMemo(
    () => chats.find((chat) => chat.conversationId === activeConversationId || chat.id === activeConversationId) || null,
    [activeConversationId, chats],
  );
  const selectedModel = useMemo(
    () => llmChoices.find((choice) => choice.id === selectedModelId) || llmChoices[0],
    [selectedModelId],
  );
  const messageCount = activeChat?.messages.length ?? 0;
  const badgeContextWindow = selectedModel.contextWindow || contextUsage?.context_window || activeModel?.context_window;
  const badgeUsedTokens = contextUsage?.used_tokens;
  const badgeUsagePercent =
    badgeContextWindow && badgeUsedTokens !== undefined && badgeUsedTokens !== null
      ? Math.min(100, Math.round((badgeUsedTokens / badgeContextWindow) * 10000) / 100)
      : contextUsage?.usage_percent;

  useEffect(() => {
    if (typeof window === "undefined") return;
    const storedChoice = window.localStorage.getItem(LLM_CHOICE_STORAGE_KEY) as LlmChoiceId | null;
    if (storedChoice && llmChoices.some((choice) => choice.id === storedChoice)) {
      setSelectedModelId(storedChoice);
      modelSelectionInitialized.current = true;
      return;
    }
    if (userSelectedModel.current) {
      modelSelectionInitialized.current = true;
      return;
    }
    if (activeModel) {
      const matchedChoice = llmChoices.find(
        (choice) => choice.provider === activeModel.provider && choice.model === activeModel.model,
      );
      if (matchedChoice) {
        setSelectedModelId(matchedChoice.id);
      }
    }
    modelSelectionInitialized.current = true;
  }, [activeModel]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const storedStyle = window.localStorage.getItem(SUMMARY_STYLE_STORAGE_KEY) as SummaryStyle | null;
    if (storedStyle === "short" || storedStyle === "editorial") {
      setSummaryReportStyle(storedStyle);
    }
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    if (!modelSelectionInitialized.current) return;
    window.localStorage.setItem(LLM_CHOICE_STORAGE_KEY, selectedModelId);
  }, [selectedModelId]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(SUMMARY_STYLE_STORAGE_KEY, summaryReportStyle);
  }, [summaryReportStyle]);

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
    const trimmed = composerDraft.trim();
    if (!trimmed || sending || !isAuthenticated) return;
    const backendPromptMode = inferBackendModeFromDraft(trimmed, composerPromptMode);
    const config = modeConfig(backendPromptMode);
    const summaryStyleToSend =
      backendPromptMode === "summary"
        ? composerPromptMode === "summary"
          ? summaryReportStyle
          : "short"
        : null;
    if (backendPromptMode !== composerPromptMode) {
      setComposerPromptMode(backendPromptMode);
    }
    const content = config.preface ? `${config.preface}\n\n${trimmed}` : trimmed;
    try {
      await sendMessage({
        content,
        mode: config.backendMode,
        summaryStyle: summaryStyleToSend,
        llmProviderOverride: selectedModel.provider,
        llmModelOverride: selectedModel.model,
      });
      clearComposerDraft();
    } catch {
      // Error state is rendered in chat messages.
    }
  }

  return (
    <div className="message-composer-shell border-t border-border/70 bg-bg/70 px-3 py-3 backdrop-blur-xl sm:px-4 sm:py-4">
      <div className="glass mx-auto flex max-w-5xl flex-col gap-3 rounded-2xl p-2.5 sm:p-3">
        <div className="flex flex-col gap-2 lg:flex-row lg:items-center">
          <div className="grid grid-cols-2 gap-1 rounded-xl border border-border/80 bg-card/80 p-1 shadow-[0_10px_30px_hsl(220_30%_10%_/_0.12)] backdrop-blur-md sm:inline-flex sm:grid-cols-none">
            {llmChoices.map((choice) => {
              const active = selectedModelId === choice.id;
              return (
                <button
                  key={choice.id}
                  type="button"
                  disabled={sending}
                  aria-pressed={active}
                  onClick={() => {
                    userSelectedModel.current = true;
                    setSelectedModelId(choice.id);
                  }}
                  className={[
                    "group relative inline-flex h-10 min-w-0 w-full items-center justify-between gap-2 rounded-lg px-3 text-left text-xs font-medium transition-all duration-150 sm:min-w-[132px] sm:w-auto",
                    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/45",
                    active
                      ? "bg-accent text-slate-950 shadow-[0_8px_20px_hsl(180_100%_50%_/_0.18)]"
                      : "text-fg/72 hover:bg-fg/[0.04] hover:text-fg/90",
                    sending ? "cursor-not-allowed opacity-60" : "cursor-pointer",
                  ].join(" ")}
                >
                  <span className="flex min-w-0 flex-col leading-tight">
                    <span className="flex items-center gap-1.5">
                      {active ? <CircleCheckBig size={12} /> : null}
                      <span className="truncate">{choice.label}</span>
                    </span>
                    <span className={active ? "truncate text-slate-950/70" : "truncate text-fg/45"}>{choice.hint}</span>
                  </span>
                  <span
                    className={[
                      "h-2 w-2 shrink-0 rounded-full transition-colors",
                      active ? "bg-slate-950/70" : choice.provider === "gemini" ? "bg-cyan-400/80" : "bg-emerald-400/80",
                    ].join(" ")}
                    aria-hidden="true"
                  />
                </button>
              );
            })}
          </div>
          <div className="flex flex-1 flex-col gap-2 sm:flex-row sm:items-center">
            <div className="relative w-full sm:w-auto">
              <select
                aria-label="Mode"
                value={composerPromptMode}
                onChange={(e) => setComposerPromptMode(e.target.value as PromptMode)}
                disabled={sending}
                className="h-10 w-full appearance-none rounded-lg border border-border/80 bg-card/85 py-1 pl-3 pr-8 text-xs font-medium text-fg/[0.82] outline-none transition focus:border-accent/50 focus-visible:ring-2 focus-visible:ring-accent/35 sm:w-auto"
              >
                {modes.map((m) => (
                  <option key={m.value} value={m.value}>
                    {m.label}
                  </option>
                ))}
              </select>
              <ChevronDown className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 text-fg/[0.45]" size={14} />
            </div>
            {composerPromptMode === "summary" ? (
              <div className="inline-flex rounded-lg border border-border/70 bg-card/70 p-1 text-[11px] font-medium shadow-sm">
                <button
                  type="button"
                  disabled={sending}
                  aria-pressed={summaryReportStyle === "short"}
                  onClick={() => setSummaryReportStyle("short")}
                  className={[
                    "rounded-md px-3 py-1.5 transition",
                    summaryReportStyle === "short"
                      ? "bg-accent text-slate-950"
                      : "text-fg/70 hover:bg-fg/[0.04] hover:text-fg",
                  ].join(" ")}
                >
                  Rapport court
                </button>
                <button
                  type="button"
                  disabled={sending}
                  aria-pressed={summaryReportStyle === "editorial"}
                  onClick={() => setSummaryReportStyle("editorial")}
                  className={[
                    "rounded-md px-3 py-1.5 transition",
                    summaryReportStyle === "editorial"
                      ? "bg-accent text-slate-950"
                      : "text-fg/70 hover:bg-fg/[0.04] hover:text-fg",
                  ].join(" ")}
                >
                  Rapport éditorial
                </button>
              </div>
            ) : null}
            <div className="hidden md:block">
              <ModelBadge
                modelName={selectedModel.label}
                contextWindow={badgeContextWindow}
                usedTokens={badgeUsedTokens}
                usagePercent={badgeUsagePercent}
              />
            </div>
          </div>
        </div>
        <div className="flex flex-col gap-2 sm:flex-row sm:items-end">
          <textarea
            aria-label="Message"
            disabled={sending || !isAuthenticated}
            className="min-h-24 max-h-40 flex-1 resize-none rounded-xl bg-transparent px-3 py-2 text-sm leading-6 outline-none placeholder:text-fg/[0.42] focus-visible:ring-2 focus-visible:ring-accent/30 sm:min-h-12 sm:resize-y"
            placeholder={isAuthenticated ? modeConfig(composerPromptMode).placeholder : "Connectez-vous pour discuter"}
            value={composerDraft}
            onChange={(e) => setComposerDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void onSend();
              }
            }}
          />
          <div className="flex items-center justify-end gap-2 self-end sm:self-auto">
            <button aria-label="Upload document" disabled={sending || !isAuthenticated} className="icon-button">
              <Upload size={16} />
            </button>
            <div className={sending || !isAuthenticated ? "pointer-events-none opacity-50" : ""}>
              <VoiceRecorder onTranscript={(t) => setComposerDraft(`${composerDraft} ${t}`.trim())} />
            </div>
            <button
              aria-label="Envoyer"
              disabled={sending || composerDraft.trim().length === 0 || !isAuthenticated}
              onClick={() => void onSend()}
              className="inline-flex h-9 w-9 items-center justify-center rounded-lg bg-accent text-slate-950 shadow-sm transition hover:bg-accent/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/55 disabled:cursor-not-allowed disabled:opacity-45"
            >
              {sending ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
