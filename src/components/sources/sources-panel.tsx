"use client";

import { AnimatePresence, motion } from "framer-motion";
import { FileSearch, ShieldCheck, X } from "lucide-react";
import { useMemo, useState } from "react";
import { PdfPreviewPanel } from "@/components/sources/pdf-preview-panel";
import { SourceSnippetCard } from "@/components/sources/source-snippet-card";
import {
  getMessageDocumentaryMetrics,
  getPreferredAssistantMessageForSources,
  type DocumentaryConfidenceLevel,
} from "@/lib/documentary-metrics";
import { buildBackendPdfUrl, buildViewerPreviewUrl } from "@/lib/pdf-preview";
import { useChatStore } from "@/store/chat-store";
import type { ChatSource } from "@/types/chat";
import type { SourceReference } from "@/types/source-reference";

type SourceSummary = {
  usedCount: number;
  ignoredCount: number;
  confidence: DocumentaryConfidenceLevel;
};

const DOC_PAGE_RE = /doc_id=([^,\]\s]+)(?:\s*,\s*page=(\d+))?(?:\s*,\s*row=(\d+))?/i;

function toNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function normalizeSource(source: ChatSource, index: number): SourceReference {
  if (typeof source === "string") {
    const urlMatch = source.match(/(https?:\/\/[^\s)]+|\/(?:viewer\/pdf|api\/documents\/)[^\s)]+)/i);
    const match = source.match(DOC_PAGE_RE);
    const docId = match?.[1] || "";
    const label = docId || `Source ${index + 1}`;
    const page = match?.[2] ? Number(match[2]) : null;
    const row = match?.[3] ? Number(match[3]) : null;
    const parsedScore = source.match(/(?:score|pertinence)\s*[:=]\s*([0-9.]+)/i)?.[1];
    const score = parsedScore ? Number(parsedScore) : undefined;
    const snippet = source.replace(/https?:\/\/[^\s)]+/g, "").slice(0, 180).trim() || "Extrait non disponible.";
    const fallbackDocumentUrl = docId ? buildBackendPdfUrl(docId, page ?? undefined) : undefined;
    return {
      id: `s-${label}-${page ?? "na"}-${row ?? "na"}-${index}`,
      documentName: label,
      documentUrl: urlMatch?.[1] || fallbackDocumentUrl,
      pageNumber: page ?? undefined,
      lineStart: row ?? undefined,
      lineEnd: row ?? undefined,
      score,
      snippet,
    };
  }

  const raw = source as Record<string, unknown>;
  const docId = String(raw.doc_id || raw.documentId || "").trim();
  const filename = String(raw.filename || raw.documentName || raw.label || docId || "Source").trim();
  const page = toNumber(raw.page);
  const row = toNumber(raw.row);
  const rowEnd = toNumber(raw.row_end);
  const score = toNumber(raw.score) ?? undefined;
  const excerptRaw = String(raw.excerpt || raw.section || "").trim();
  const excerpt = excerptRaw ? excerptRaw.slice(0, 180) : `${filename} ${page ? `page ${page}` : ""}`.trim();
  const hrefRaw = String(raw.viewer_url || raw.url || "").trim();
  const fallbackDocumentUrl = docId ? buildBackendPdfUrl(docId, page ?? undefined) : undefined;

  return {
    id: `o-${docId || filename}-${page ?? "na"}-${row ?? "na"}-${index}`,
    documentName: filename,
    documentUrl: hrefRaw || fallbackDocumentUrl,
    pageNumber: page ?? undefined,
    lineStart: row ?? undefined,
    lineEnd: rowEnd && row ? Math.max(rowEnd, row) : row ?? undefined,
    score,
    snippet: excerpt || "Extrait non disponible.",
  };
}

function uniqueSources(sources: SourceReference[]): SourceReference[] {
  const map = new Map<string, SourceReference>();
  for (const source of sources) {
    const pageKey = Number.isFinite(source.pageNumber || NaN) ? String(source.pageNumber) : "na";
    const linesKey = `${source.lineStart ?? "na"}-${source.lineEnd ?? "na"}`;
    const key = `${source.documentName.toLowerCase()}::${pageKey}::${linesKey}::${source.documentUrl || ""}`;
    if (!map.has(key)) {
      map.set(key, source);
    }
  }
  return Array.from(map.values());
}

function confidenceClass(level: DocumentaryConfidenceLevel): string {
  if (level === "elevee") return "doc-confidence-high";
  if (level === "moyenne") return "status-low";
  return "doc-confidence-low";
}

function confidenceLabel(level: DocumentaryConfidenceLevel): string {
  if (level === "elevee") return "élevée";
  if (level === "moyenne") return "moyenne";
  return "faible";
}

function SourcesBody({
  sources,
  summary,
  hasMessages,
  hasResponse,
  isLoading,
  onPreview,
}: {
  sources: SourceReference[];
  summary: SourceSummary;
  hasMessages: boolean;
  hasResponse: boolean;
  isLoading: boolean;
  onPreview: (source: SourceReference) => void;
}) {
  if (!hasMessages) {
    return (
      <div className="flex h-full min-h-52 flex-col items-center justify-center rounded-lg border border-border/60 bg-card/[0.35] px-4 text-center">
        <FileSearch size={40} className="opacity-30" />
        <p className="mt-3 text-[13px] text-fg/55">Les sources apparaîtront ici après votre première question.</p>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="space-y-3 rounded-lg border border-border/65 bg-card/[0.4] p-3">
        <p className="text-[13px] text-fg/58">Recherche des sources…</p>
        <div className="space-y-2">
          <div className="h-3 w-5/6 animate-pulse rounded bg-fg/[0.08]" />
          <div className="h-3 w-2/3 animate-pulse rounded bg-fg/[0.08]" />
          <div className="h-3 w-3/4 animate-pulse rounded bg-fg/[0.08]" />
        </div>
      </div>
    );
  }

  if (!hasResponse) {
    return (
      <div className="flex h-full min-h-52 flex-col items-center justify-center rounded-lg border border-border/60 bg-card/[0.35] px-4 text-center">
        <FileSearch size={40} className="opacity-30" />
        <p className="mt-3 text-[13px] text-fg/55">Les sources apparaîtront après la première réponse de l&apos;assistant.</p>
      </div>
    );
  }

  return (
    <div className="space-y-3 overflow-auto">
      <div className="grid grid-cols-1 gap-2">
        <div className="rounded-lg border border-border/70 bg-card/[0.62] px-3 py-2 text-xs text-fg/78">
          Sources utilisées : <span className="font-semibold text-fg">{summary.usedCount}</span>
        </div>
        <div className="rounded-lg border border-border/70 bg-card/[0.62] px-3 py-2 text-xs text-fg/78">
          Sources ignorées : <span className="font-semibold text-fg">{summary.ignoredCount}</span>
        </div>
        {summary.usedCount > 0 ? (
          <div className={`rounded-lg border px-3 py-2 text-xs font-medium ${confidenceClass(summary.confidence)}`}>
            Confiance documentaire : {confidenceLabel(summary.confidence)}
          </div>
        ) : null}
      </div>

      {sources.length === 0 ? (
        <p className="rounded-lg border border-border/65 bg-card/[0.45] px-3 py-3 text-xs text-fg/60">
          Aucun document pertinent trouvé pour cette question.
        </p>
      ) : (
        <div className="space-y-2">
          {sources.map((source, index) => (
            <SourceSnippetCard
              key={source.id || `${source.documentName}-${source.pageNumber ?? "na"}-${index}`}
              source={source}
              onPreview={onPreview}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export function SourcesPanel({ mobileOpen, onClose }: { mobileOpen: boolean; onClose: () => void }) {
  const chats = useChatStore((s) => s.chats);
  const activeChatId = useChatStore((s) => s.activeChatId);
  const chat = chats.find((c) => c.id === activeChatId);

  const latestAssistantMessage = useMemo(
    () => getPreferredAssistantMessageForSources(chat?.messages || []),
    [chat?.messages],
  );
  const sources = useMemo<ChatSource[]>(
    () => (latestAssistantMessage?.sources || []),
    [latestAssistantMessage?.sources],
  );
  const hasMessages = Boolean(chat?.messages?.some((m) => m.role === "user" && String(m.content || "").trim().length > 0));
  const hasResponse = Boolean(latestAssistantMessage);
  const isLoading = Boolean(chat?.messages?.some((m) => m.role === "assistant" && m.status === "loading"));
  const normalizedSources = useMemo(() => uniqueSources(sources.map(normalizeSource)), [sources]);
  const summary = useMemo<SourceSummary>(() => {
    if (!latestAssistantMessage) {
      return { usedCount: 0, ignoredCount: 0, confidence: "faible" };
    }
    const metrics = getMessageDocumentaryMetrics(latestAssistantMessage);
    return {
      usedCount: metrics.sourceCount,
      ignoredCount: metrics.ignoredCount,
      confidence: metrics.confidence,
    };
  }, [latestAssistantMessage]);
  const [previewSource, setPreviewSource] = useState<SourceReference | null>(null);
  const activePreview = useMemo(() => {
    if (!previewSource) return null;
    if (previewSource.documentUrl) return previewSource;
    const fallbackViewerUrl = buildViewerPreviewUrl(previewSource);
    return fallbackViewerUrl ? { ...previewSource, documentUrl: fallbackViewerUrl } : previewSource;
  }, [previewSource]);

  return (
    <>
      <aside className="glass hidden h-dvh w-80 shrink-0 flex-col border-y-0 border-r-0 p-4 xl:flex">
        <div className="mb-4 flex items-start gap-3">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-accent/10 text-accent">
            <ShieldCheck size={17} />
          </div>
          <div>
            <h3 className="text-sm font-semibold">Sources</h3>
            <p className="mt-1 text-xs leading-5 text-fg/60">Documents et passages utilisés par la réponse.</p>
          </div>
        </div>
        <SourcesBody
          sources={normalizedSources}
          summary={summary}
          hasMessages={hasMessages}
          hasResponse={hasResponse}
          isLoading={isLoading}
          onPreview={setPreviewSource}
        />
      </aside>
      <AnimatePresence>
        {previewSource ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[60] bg-black/45"
            onClick={() => setPreviewSource(null)}
          >
            <motion.aside
              initial={{ x: "100%" }}
              animate={{ x: 0 }}
              exit={{ x: "100%" }}
              transition={{ type: "spring", stiffness: 260, damping: 28 }}
              className="glass absolute right-0 top-0 h-full w-full max-w-3xl border-y-0 border-r-0 p-4"
              onClick={(event) => event.stopPropagation()}
            >
              <div className="h-[calc(100vh-3rem)]">
                <PdfPreviewPanel source={activePreview} onClose={() => setPreviewSource(null)} />
              </div>
            </motion.aside>
          </motion.div>
        ) : null}
      </AnimatePresence>
      <AnimatePresence>
        {mobileOpen ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-black/50 xl:hidden"
            onClick={onClose}
          >
            <motion.aside
              initial={{ x: "100%" }}
              animate={{ x: 0 }}
              exit={{ x: "100%" }}
              transition={{ type: "spring", stiffness: 260, damping: 24 }}
              className="glass absolute right-0 top-0 h-full w-[88%] max-w-sm border-y-0 border-r-0 p-4"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="mb-3 flex items-center justify-between">
                <h3 className="text-sm font-semibold">Sources</h3>
                <button aria-label="Fermer les sources" className="icon-button h-8 w-8" onClick={onClose}>
                  <X size={14} />
                </button>
              </div>
              <SourcesBody
                sources={normalizedSources}
                summary={summary}
                hasMessages={hasMessages}
                hasResponse={hasResponse}
                isLoading={isLoading}
                onPreview={setPreviewSource}
              />
            </motion.aside>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </>
  );
}
