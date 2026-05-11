"use client";

import { motion } from "framer-motion";
import { AlertTriangle, Copy, RotateCcw } from "lucide-react";
import { AssistantMarkdown } from "@/components/chat/assistant-markdown";
import { AssistantLoadingMessage } from "@/components/chat/assistant-loading-message";
import { VisualizationRenderer } from "@/components/chat/visualization-renderer";
import { ConversationQualityPanel } from "@/components/chat/conversation-quality-panel";
import { QualityReportCard } from "@/components/chat/quality-report-card";
import { SourceLinks, stripSourcesSection } from "@/components/sources/source-links";
import { useChatStore } from "@/store/chat-store";
import { useEffect, useRef } from "react";

function isRenderableVisualization(message: {
  visualization?: { type?: string; data?: unknown[] } | undefined;
  chart_data?: { type?: string; data?: unknown[] } | undefined;
}): boolean {
  const chartData = message.chart_data;
  const visualization = message.visualization;
  const data = (Array.isArray(chartData?.data) ? chartData?.data : Array.isArray(visualization?.data) ? visualization?.data : []) || [];
  if (data.length === 0) return false;
  const t = String(chartData?.type || visualization?.type || "bar").toLowerCase().trim();
  return t === "bar" || t === "line";
}

function stripVisualizationUnavailableText(content: string): string {
  return content
    .replace(/Vous avez demandé un [^\n.]+\.?/gi, "")
    .replace(/Recommandation\s*:\s*[^\n.]+\.?/gi, "")
    .replace(/rendu chart\.?/gi, "")
    .replace(/Le rendu graphique n[’']est pas encore disponible dans l[’']interface\s*;?\s*je fournis les données nécessaires pour générer le graphique en barres\.?/gi, "")
    .replace(/Le rendu graphique demandé nécessite un composant côté interface\.?/gi, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function hasMarkdownTable(content: string): boolean {
  return /\|[^\n]+\|\n\|(?:\s*[-:]+[-|\s:]*)+\|/m.test(content);
}

export function ChatMessages() {
  const chats = useChatStore((s) => s.chats);
  const activeChatId = useChatStore((s) => s.activeChatId);
  const qualityDebugEnabled = useChatStore((s) => s.qualityDebugEnabled);
  const chat = chats.find((c) => c.id === activeChatId);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [chat?.messages]);

  if (!chat || chat.messages.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-4 p-6 text-center">
        <h2 className="text-2xl font-semibold">Assistant clinique CHU Oujda</h2>
        <p className="max-w-xl text-fg/75">Cette réponse ne remplace pas l'avis médical.</p>
        <div className="grid w-full max-w-3xl gap-2 md:grid-cols-2">
          {[
            "Résume ce rapport biologique.",
            "Quels résultats semblent anormaux ?",
            "Explique les valeurs importantes.",
            "Compare ces deux documents.",
            "Quels éléments nécessitent vérification ?",
          ].map((s) => (
            <button key={s} className="rounded-xl border border-border px-3 py-2 text-left text-sm hover:bg-card">
              {s}
            </button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4 p-6">
      {qualityDebugEnabled ? <ConversationQualityPanel messages={chat.messages} /> : null}
      {chat.messages.map((message) => {
        const status = message.status || "done";
        const isAssistant = message.role === "assistant";
        const isLoading = isAssistant && status === "loading";
        const isError = isAssistant && status === "error";
        const isDone = isAssistant && status === "done";
        const shouldRenderSourceLinks = isDone && (message.sources?.length || 0) > 0;
        const canRenderVisualization = isAssistant && isDone && isRenderableVisualization(message);
        const withoutSources = shouldRenderSourceLinks ? stripSourcesSection(message.content) : message.content;
        const contentToRender = canRenderVisualization ? stripVisualizationUnavailableText(withoutSources) : withoutSources;
        const contentHasTable = hasMarkdownTable(contentToRender);

        return (
          <motion.article
            key={message.id}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2 }}
            className="rounded-2xl border border-border bg-card/60 p-5 shadow-sm"
          >
            {isLoading ? (
              <AssistantLoadingMessage />
            ) : (
              <>
                <p className="mb-2 text-xs uppercase text-fg/60">{message.role === "user" ? "Vous" : "Assistant"}</p>
                {isError ? (
                  <div
                    role="status"
                    aria-live="polite"
                    className="rounded-xl border border-rose-600/40 bg-rose-950/20 px-4 py-3 text-sm text-rose-100"
                  >
                    {contentToRender}
                  </div>
                ) : isAssistant ? (
                  <>
                    <VisualizationRenderer visualization={message.visualization} chartData={message.chart_data} />
                    {canRenderVisualization && contentHasTable ? (
                      <p className="mt-3 text-xs uppercase tracking-wide text-fg/65">Données utilisées</p>
                    ) : null}
                    <AssistantMarkdown content={contentToRender} />
                  </>
                ) : (
                  <p className="whitespace-pre-wrap text-sm leading-6">{contentToRender}</p>
                )}
                {isDone ? <SourceLinks sources={message.sources} /> : null}
                {isDone && isAssistant && qualityDebugEnabled ? <QualityReportCard diagnostics={message.diagnostics} /> : null}
                {isDone && (
                  <div className="mt-3 flex gap-2">
                    <button aria-label="Copier" className="rounded-lg border border-border p-2">
                      <Copy size={14} />
                    </button>
                    <button aria-label="Régénérer" className="rounded-lg border border-border p-2">
                      <RotateCcw size={14} />
                    </button>
                    <button aria-label="Feedback" className="rounded-lg border border-border p-2">
                      <AlertTriangle size={14} />
                    </button>
                  </div>
                )}
              </>
            )}
          </motion.article>
        );
      })}
      <div ref={bottomRef} />
    </div>
  );
}
