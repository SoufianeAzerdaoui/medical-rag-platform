"use client";

import { motion } from "framer-motion";
import { AlertTriangle, Copy, RotateCcw } from "lucide-react";
import { AssistantMarkdown } from "@/components/chat/assistant-markdown";
import { AssistantLoadingMessage } from "@/components/chat/assistant-loading-message";
import { VisualizationRenderer } from "@/components/chat/visualization-renderer";
import { ConversationQualityPanel } from "@/components/chat/conversation-quality-panel";
import { QualityReportCard } from "@/components/chat/quality-report-card";
import { SourceLinks, stripSourcesSection } from "@/components/sources/source-links";
import { PatientInventoryRenderer } from "@/components/chat/patient-inventory-renderer";
import { SingleAnalyteResultCard } from "@/components/chat/single-analyte-result-card";
import { useChatStore } from "@/store/chat-store";
import { useEffect, useRef } from "react";

function isRenderableVisualization(message: {
  visualization?: { requested?: boolean; rendered_type?: string | null; type?: string; data?: unknown[] } | undefined;
  chart_data?: { rendered_type?: string | null; type?: string; data?: unknown[] } | undefined;
}): boolean {
  const chartData = message.chart_data;
  const visualization = message.visualization;
  if (!visualization?.requested) return false;
  const data = (Array.isArray(chartData?.data) ? chartData?.data : Array.isArray(visualization?.data) ? visualization?.data : []) || [];
  const renderedType = String(chartData?.rendered_type || visualization?.rendered_type || chartData?.type || visualization?.type || "")
    .toLowerCase()
    .trim();
  return renderedType.length > 0 || data.length > 0;
}

function stripVisualizationUnavailableText(content: string): string {
  const cleaned = content
    .replace(/Vous avez demandé un [^\n.]+\.?/gi, "")
    .replace(/Graphique demandé\s*:\s*[^\n.]+\.?/gi, "")
    .replace(/Rendu affiché\s*:\s*[^\n.]+\.?/gi, "")
    .replace(/Recommandation\s*:\s*[^\n.]+\.?/gi, "")
    .replace(/rendu chart\.?/gi, "")
    .replace(/Le rendu graphique n[’']est pas encore disponible dans l[’']interface\s*;?\s*je fournis les données nécessaires pour générer le graphique en barres\.?/gi, "")
    .replace(/Le rendu graphique demandé nécessite un composant côté interface\.?/gi, "")
    .replace(/INSULINET4 LIBRETSHusT3 LIBREANTI-TG-600%0%600%1200%1800%Écart normalisé à la référence/gi, "")
    .replace(/TSHusT3/gi, "")
    .replace(/INSULINE/gi, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();

  // If there's a "Données utilisées" marker followed by a table, we keep the table if visualization is active.
  // But if we want to strip the table when visualization is active (as requested), we do it here.
  return cleaned;
}

function stripMarkdownTable(content: string): string {
  return content.replace(/\|[^\n]+\|\n\|(?:\s*[-:]+[-|\s:]*)+\|[\s\S]*?(?:\n\n|$)/m, "").trim();
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
      {chat.messages.map((message: any, idx: number) => {
        const status = message.status || "done";
        const isAssistant = message.role === "assistant";
        const isLoading = isAssistant && status === "loading";
        const isError = isAssistant && status === "error";
        const isDone = isAssistant && status === "done";
        const shouldRenderSourceLinks = isDone && (message.sources?.length || 0) > 0;
        const canRenderVisualization = isAssistant && isDone && isRenderableVisualization(message);
        const hasPatients = isAssistant && isDone && Array.isArray(message.patients) && message.patients.length > 0;
        const previousUserContent = String(chat.messages[idx - 1]?.role === "user" ? chat.messages[idx - 1]?.content || "" : "").toLowerCase();
        const expandPatientSourcesByDefault =
          hasPatients && (previousUserContent.includes("source") || previousUserContent.includes("cliquable"));
        const isSingleAnalyteDeterministic =
          isAssistant &&
          isDone &&
          String(message?.diagnostics?.generation_mode || "") === "deterministic_single_analyte_lookup";
        
        let contentToRender = shouldRenderSourceLinks ? stripSourcesSection(message.content) : message.content;
        
        if (canRenderVisualization) {
          contentToRender = stripVisualizationUnavailableText(contentToRender);
        }
        
        // If we have interactive components (Visualisation or Patient Inventory), we might want to hide the Markdown table
        if (hasPatients) {
          contentToRender = stripMarkdownTable(contentToRender);
        }

        const contentHasTable = hasMarkdownTable(contentToRender);
        const useSingleAnalyteCard =
          isSingleAnalyteDeterministic &&
          /^###\s+/m.test(contentToRender) &&
          /-\s+\*\*Valeur\*\*/i.test(contentToRender) &&
          /-\s+\*\*Statut technique\*\*/i.test(contentToRender);

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
                    {hasPatients && (
                      <PatientInventoryRenderer
                        patients={message.patients}
                        defaultExpanded={expandPatientSourcesByDefault}
                        inventoryView={message.inventory_view}
                      />
                    )}
                    
                    {canRenderVisualization && contentHasTable ? (
                      <p className="mt-3 text-xs uppercase tracking-wide text-fg/65">Données utilisées</p>
                    ) : null}
                    
                    {useSingleAnalyteCard ? (
                      <SingleAnalyteResultCard content={contentToRender} sources={message.sources} />
                    ) : (
                      <AssistantMarkdown content={contentToRender} />
                    )}
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
