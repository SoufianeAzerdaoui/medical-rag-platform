"use client";

import { motion } from "framer-motion";
import { AlertTriangle, Copy, FileText, RotateCcw, Sparkles } from "lucide-react";
import { AssistantLoadingMessage } from "@/components/chat/assistant-loading-message";
import { AssistantConversationCard } from "@/components/chat/assistant-conversation-card";
import { AssistantMarkdown } from "@/components/chat/assistant-markdown";
import { MedicalAnswerBlocks } from "@/components/chat/medical-answer-blocks";
import { VisualizationRenderer } from "@/components/chat/visualization-renderer";
import { ConversationQualityPanel } from "@/components/chat/conversation-quality-panel";
import { QualityReportCard } from "@/components/chat/quality-report-card";
import { SourceLinks, stripSourcesSection } from "@/components/sources/source-links";
import { PatientInventoryRenderer } from "@/components/chat/patient-inventory-renderer";
import { SingleAnalyteResultCard } from "@/components/chat/single-analyte-result-card";
import { StructuredSummaryCard } from "@/components/chat/structured-summary-card";
import { useChatStore } from "@/store/chat-store";
import { useEffect, useRef } from "react";

function isRenderableVisualization(message: {
  visualization?: { requested?: boolean; rendered_type?: string | null; type?: string; data?: unknown[] } | undefined;
  chart_data?: { rendered_type?: string | null; type?: string; data?: unknown[] } | undefined;
}): boolean {
  const chartData = message.chart_data;
  const viz = message.visualization;
  if (!viz?.requested) return false;
  const data = (Array.isArray(chartData?.data) ? chartData?.data : Array.isArray(viz?.data) ? viz?.data : []) || [];
  const renderedType = String(
    chartData?.rendered_type || viz?.rendered_type || chartData?.type || viz?.type || "",
  )
    .trim()
    .toLowerCase();
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

function isGreetingLike(value: string): boolean {
  const text = String(value || "").trim().toLowerCase();
  if (!text) return false;
  return /^(bonjour|salut|hello|hi|bonsoir|coucou|hey|salam)\b/.test(text);
}

type AssistantRenderType = "medical_structured" | "conversational" | "general_markdown";

function resolveAssistantRenderType(params: {
  explicitAnswerType: string;
  previousUserContent: string;
  content: string;
  hasSources: boolean;
  hasTable: boolean;
  canRenderVisualization: boolean;
  hasPatients: boolean;
  isStructuredSummaryRoute: boolean;
  isSingleAnalyteCard: boolean;
}): AssistantRenderType {
  const explicit = params.explicitAnswerType;
  if ([
    "conversational",
    "small_talk",
    "small-talk",
    "greeting",
    "general_conversation",
    "chitchat",
    "chat",
  ].includes(explicit)) {
    return "conversational";
  }
  if ([
    "medical_structured",
    "medical",
    "medical_report",
    "medical_summary",
    "document_analysis",
    "comparison",
  ].includes(explicit)) {
    return "medical_structured";
  }
  if (explicit === "general_markdown") {
    return "general_markdown";
  }

  const likelyConversational =
    !params.hasSources &&
    !params.hasTable &&
    !params.canRenderVisualization &&
    !params.hasPatients &&
    !params.isStructuredSummaryRoute &&
    !params.isSingleAnalyteCard &&
    (isGreetingLike(params.previousUserContent) || isGreetingLike(params.content));
  if (likelyConversational) {
    return "conversational";
  }

  const likelyMedicalStructured =
    params.hasPatients ||
    params.hasTable ||
    params.canRenderVisualization ||
    params.isStructuredSummaryRoute ||
    params.isSingleAnalyteCard ||
    params.hasSources;
  if (likelyMedicalStructured) {
    return "medical_structured";
  }

  return "general_markdown";
}

export function ChatMessages() {
  const SOURCE_TOP_N = 6;
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
      <div className="mx-auto flex min-h-full w-full max-w-5xl flex-col justify-center gap-6 px-5 py-10">
        <div className="max-w-2xl">
          <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-accent/25 bg-accent/10 px-3 py-1 text-xs font-medium text-accent">
            <Sparkles size={14} />
            Assistant clinique augmenté
          </div>
          <h2 className="text-3xl font-semibold tracking-tight text-fg sm:text-4xl">CHU Oujda Clinical Assistant</h2>
          <p className="mt-3 max-w-xl text-sm leading-6 text-fg/[0.68]">
            Analyse les documents médicaux, compare les résultats et met en avant les points à vérifier avec les sources disponibles.
          </p>
        </div>
        <div className="grid w-full gap-3 md:grid-cols-2">
          {[
            "Résume ce rapport biologique.",
            "Quels résultats semblent anormaux ?",
            "Explique les valeurs importantes.",
            "Compare ces deux documents.",
            "Quels éléments nécessitent vérification ?",
          ].map((s) => (
            <button
              key={s}
              className="premium-surface group flex min-h-16 items-center gap-3 rounded-xl px-4 py-3 text-left text-sm transition hover:-translate-y-0.5 hover:border-accent/35 hover:bg-card"
            >
              <span className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-accent/10 text-accent">
                <FileText size={16} />
              </span>
              <span className="font-medium text-fg/[0.86]">{s}</span>
            </button>
          ))}
        </div>
        <p className="text-xs text-fg/[0.54]">Cette réponse ne remplace pas l'avis médical.</p>
      </div>
    );
  }

  return (
    <div className="mx-auto w-full max-w-5xl space-y-4 px-4 py-6 sm:px-6">
      {qualityDebugEnabled ? <ConversationQualityPanel messages={chat.messages} /> : null}
      {chat.messages.map((message: any, idx: number) => {
        const status = message.status || "done";
        const isAssistant = message.role === "assistant";
        const isLoading = isAssistant && status === "loading";
        const isError = isAssistant && status === "error";
        const isDone = isAssistant && status === "done";
        const shouldRenderSourceLinks = isDone && (message.sources?.length || 0) > 0;
        const canRenderVisualization =
          isAssistant && isDone && isRenderableVisualization(message) && !message.content.includes("Le format demandé est ambigu");
        const hasPatients = isAssistant && isDone && Array.isArray(message.patients) && message.patients.length > 0;
        const previousUserContent = String(chat.messages[idx - 1]?.role === "user" ? chat.messages[idx - 1]?.content || "" : "").toLowerCase();
        const expandPatientSourcesByDefault =
          hasPatients && (previousUserContent.includes("source") || previousUserContent.includes("cliquable"));
        const selectedRoute = String(
          message?.diagnostics?.selected_route || message?.selected_route || "",
        ).toLowerCase();
        const intent = String(
          message?.diagnostics?.intent || message?.intent || "",
        ).toLowerCase();
        const generationMode = String(
          message?.diagnostics?.generation_mode || message?.generation_mode || "",
        ).toLowerCase();
        let contentToRender = shouldRenderSourceLinks ? stripSourcesSection(message.content) : message.content;
        
        if (canRenderVisualization) {
          contentToRender = stripVisualizationUnavailableText(contentToRender);
        }
        
        // If we have interactive components (Visualisation or Patient Inventory), we might want to hide the Markdown table
        if (hasPatients) {
          contentToRender = stripMarkdownTable(contentToRender);
        }

        const looksLikeDoctorNote =
          /^note de synth[èe]se m[ée]dicale\s*[—-]/i.test(contentToRender) ||
          /^note m[ée]dicale\s*[—-]/i.test(contentToRender);
        const looksLikeTechnicalSummary =
          /(?:^|\n)\s*anormaux\s*:/i.test(contentToRender) &&
          /(?:^|\n)\s*conclusion technique\s*:/i.test(contentToRender);
        const isCohortLike =
          selectedRoute === "cohort_search" ||
          intent === "cohort_search" ||
          selectedRoute === "global_analyte_abnormal_search" ||
          intent === "global_analyte_abnormal_search" ||
          selectedRoute === "global_patient_lookup" ||
          intent === "global_patient_lookup" ||
          generationMode === "deterministic_global_analyte_abnormal_search" ||
          generationMode === "deterministic_evidence_template";
        const isSingleAnalyteDeterministic =
          isAssistant &&
          isDone &&
          (generationMode === "deterministic_single_analyte_lookup" ||
            selectedRoute === "doc_scoped_single_analyte_status" ||
            intent === "doc_scoped_single_analyte_status") &&
          (message.diagnostics?.displayed_evidences_count === 1 || message.diagnostics?.included_rows_count === 1) &&
          !hasPatients &&
          !isCohortLike;
        const isStructuredSummaryRoute =
          isAssistant &&
          isDone &&
          (
            selectedRoute === "doc_scoped_biological_summary" ||
            selectedRoute === "reference_ranges_summary" ||
            intent === "doc_scoped_summary" ||
            intent === "reference_ranges_summary" ||
            generationMode === "deterministic_doc_scoped_biological_summary" ||
            generationMode === "deterministic_reference_ranges_summary" ||
            generationMode === "hybrid_structured_llm_writer" ||
            looksLikeDoctorNote ||
            looksLikeTechnicalSummary
          );

        const contentHasTable = hasMarkdownTable(contentToRender);
        const explicitAnswerType = String(message?.diagnostics?.answer_type || message?.answer_type || "").trim().toLowerCase();
        const validationStatus = String(message?.diagnostics?.validation_status || "").toLowerCase();
        const shouldShowFailBanner = isAssistant && isDone && validationStatus === "fail";
        const debugFinalAnswerSource = String(message?.diagnostics?.final_answer_source || "").trim();
        const debugRendererUsed = String(message?.diagnostics?.renderer_used || "").trim();
        const showProvenanceDebug = isAssistant && isDone && qualityDebugEnabled && (debugFinalAnswerSource || debugRendererUsed);
        const useSingleAnalyteCard =
          isSingleAnalyteDeterministic &&
          !contentHasTable &&
          /^###\s+/m.test(contentToRender) &&
          /-\s+\*\*Valeur\*\*/i.test(contentToRender) &&
          /-\s+\*\*Statut technique\*\*/i.test(contentToRender);
        const assistantRenderType = resolveAssistantRenderType({
          explicitAnswerType,
          previousUserContent,
          content: contentToRender,
          hasSources: shouldRenderSourceLinks,
          hasTable: contentHasTable,
          canRenderVisualization,
          hasPatients,
          isStructuredSummaryRoute,
          isSingleAnalyteCard: useSingleAnalyteCard,
        });

        return (
          <motion.article
            key={message.id}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2 }}
            className={
              isAssistant
                ? "premium-surface rounded-2xl p-5"
                : "ml-auto max-w-[86%] rounded-2xl border border-accent/[0.18] bg-accent/10 p-4 shadow-sm"
            }
          >
            {isLoading ? (
              <AssistantLoadingMessage />
            ) : (
              <>
                <div className="mb-3 flex items-center gap-2">
                  <span className={isAssistant ? "h-2 w-2 rounded-full bg-accent" : "h-2 w-2 rounded-full bg-fg/[0.45]"} />
                  <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-fg/[0.52]">
                    {message.role === "user" ? "Vous" : "Assistant"}
                  </p>
                </div>
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
                    {shouldShowFailBanner ? (
                      <div
                        role="status"
                        aria-live="polite"
                        className="mb-3 rounded-xl border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-800 dark:text-amber-100"
                      >
                        <p className="font-medium">Réponse non fiable</p>
                        <p className="mt-1 text-amber-100/90">La validation a détecté une incohérence factuelle. Utilise “Régénérer”.</p>
                        <button
                          type="button"
                          aria-label="Régénérer"
                          className="mt-2 rounded-md border border-amber-400/40 bg-amber-500/10 px-2 py-1 text-xs font-medium text-amber-800 dark:text-amber-100"
                        >
                          Régénérer
                        </button>
                      </div>
                    ) : null}
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
                    ) : isStructuredSummaryRoute ? (
                      <StructuredSummaryCard
                        content={contentToRender}
                        sources={message.sources}
                        diagnostics={message.diagnostics}
                      />
                    ) : assistantRenderType === "conversational" ? (
                      <AssistantConversationCard content={contentToRender} />
                    ) : assistantRenderType === "medical_structured" ? (
                      <MedicalAnswerBlocks content={contentToRender} sources={message.sources} />
                    ) : (
                      <AssistantMarkdown content={contentToRender} />
                    )}
                    {showProvenanceDebug ? (
                      <p className="mt-3 text-xs text-fg/70">
                        Provenance: {debugFinalAnswerSource || "n/a"}
                        {debugRendererUsed ? ` · renderer=${debugRendererUsed}` : ""}
                      </p>
                    ) : null}
                  </>
                ) : (
                  <p className="whitespace-pre-wrap text-sm leading-6">{contentToRender}</p>
                )}
                {isDone && shouldRenderSourceLinks ? (
                  useSingleAnalyteCard ? null : (
                    <div className="mt-4 rounded-xl border border-border/70 bg-fg/[0.025] p-3 shadow-sm">
                      <div className="mb-2 flex items-center justify-between gap-3">
                        <p className="text-xs font-semibold uppercase tracking-wide text-fg/70">Sources cliquables</p>
                        <span className="rounded-full border border-border bg-card px-2 py-0.5 text-[11px] text-fg/60">
                          {message.sources?.length || 0}
                        </span>
                      </div>
                      <SourceLinks
                        sources={message.sources}
                        showTitle={false}
                        compact
                        maxVisible={SOURCE_TOP_N}
                      />
                    </div>
                  )
                ) : null}
                {isDone && isAssistant && qualityDebugEnabled ? <QualityReportCard diagnostics={message.diagnostics} /> : null}
                {isDone && (
                  <div className="mt-4 flex gap-2">
                    <button aria-label="Copier" className="icon-button">
                      <Copy size={14} />
                    </button>
                    <button aria-label="Régénérer" className="icon-button">
                      <RotateCcw size={14} />
                    </button>
                    <button aria-label="Feedback" className="icon-button">
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
