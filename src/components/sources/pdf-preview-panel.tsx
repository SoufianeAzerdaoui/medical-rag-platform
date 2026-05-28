"use client";

import { Copy, ExternalLink, ShieldCheck, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { NativePdfRenderer } from "@/components/sources/pdf-renderers/native-pdf-renderer";
import { PdfJsRenderer } from "@/components/sources/pdf-renderers/pdfjs-renderer";
import { buildViewerPreviewUrl, resolvePdfDocumentUrl, withPdfPageAnchor } from "@/lib/pdf-preview";
import type { SourceReference } from "@/types/source-reference";

export type PdfPreviewPanelProps = {
  source: SourceReference | null;
  onClose?: () => void;
};

type PreviewEngine = "native" | "pdfjs";

function resolvePreviewEngine(): PreviewEngine {
  const raw = (process.env.NEXT_PUBLIC_PDF_PREVIEW_ENGINE || "").toLowerCase();
  return raw === "pdfjs" ? "pdfjs" : "native";
}

function lineLabel(source: SourceReference): string {
  if (Number.isFinite(source.lineStart || NaN) && Number.isFinite(source.lineEnd || NaN) && Number(source.lineEnd) > Number(source.lineStart)) {
    return `lignes ${source.lineStart}-${source.lineEnd}`;
  }
  if (Number.isFinite(source.lineStart || NaN)) {
    return `lignes ${source.lineStart}`;
  }
  return "lignes non précisées";
}

function scoreLabel(score?: number): string {
  if (typeof score !== "number" || !Number.isFinite(score)) return "non renseignée";
  const normalized = score <= 1 ? score * 100 : score;
  return `${Math.max(0, Math.min(100, Math.round(normalized)))}%`;
}

export function PdfPreviewPanel({ source, onClose }: PdfPreviewPanelProps) {
  const [copied, setCopied] = useState(false);
  const [pdfJsFailed, setPdfJsFailed] = useState(false);
  const previewEngine = resolvePreviewEngine();

  useEffect(() => {
    setPdfJsFailed(false);
  }, [source?.id, source?.documentName, source?.pageNumber, source?.lineStart, source?.lineEnd]);

  const pageText = Number.isFinite(source?.pageNumber || NaN) ? `Page ${source?.pageNumber}` : "Page non précisée";
  const nativePreviewUrl = useMemo(() => {
    if (!source) return null;
    if (source.documentUrl) return withPdfPageAnchor(source.documentUrl, source.pageNumber);
    return buildViewerPreviewUrl(source);
  }, [source]);
  const pdfDocumentUrl = useMemo(() => (source ? resolvePdfDocumentUrl(source) : null), [source]);
  const effectiveEngine: PreviewEngine = previewEngine === "pdfjs" && !pdfJsFailed ? "pdfjs" : "native";

  async function copySource() {
    if (!source) return;
    const payload = [
      source.documentName,
      pageText,
      lineLabel(source),
      source.snippet ? `Extrait: ${source.snippet}` : "",
      `Pertinence: ${scoreLabel(source.score)}`,
    ]
      .filter(Boolean)
      .join(" · ");
    try {
      await navigator.clipboard.writeText(payload);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1300);
    } catch {
      setCopied(false);
    }
  }

  if (!source) {
    return (
      <div className="flex h-full items-center justify-center rounded-xl border border-border/70 bg-card/[0.42] p-4 text-sm text-fg/70">
        Sélectionnez une source pour afficher son aperçu.
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="mb-3 flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="inline-flex items-center gap-1 rounded-md border border-emerald-500/30 bg-emerald-500/10 px-2 py-1 text-[11px] font-medium text-emerald-700 dark:text-emerald-200">
            <ShieldCheck size={12} />
            Source utilisée
          </p>
          <h4 className="mt-2 line-clamp-1 text-sm font-semibold text-fg/90">{source.documentName}</h4>
          <p className="mt-1 text-xs text-fg/65">
            {pageText} · {lineLabel(source)}
          </p>
        </div>
        {onClose ? (
          <button aria-label="Fermer aperçu PDF" className="icon-button h-8 w-8" onClick={onClose}>
            <X size={14} />
          </button>
        ) : null}
      </div>

      <div className="mb-3 grid grid-cols-1 gap-2 sm:grid-cols-2">
        <div className="rounded-lg border border-border/70 bg-card/[0.62] px-3 py-2 text-xs text-fg/78">
          Page ciblée : <span className="font-semibold text-fg">{pageText.replace("Page ", "")}</span>
        </div>
        <div className="rounded-lg border border-border/70 bg-card/[0.62] px-3 py-2 text-xs text-fg/78">
          Pertinence : <span className="font-semibold text-fg">{scoreLabel(source.score)}</span>
        </div>
      </div>

      {source.snippet ? (
        <p className="mb-3 line-clamp-3 rounded-lg border border-border/60 bg-fg/[0.025] px-3 py-2 text-xs leading-5 text-fg/80">
          Extrait : {source.snippet}
        </p>
      ) : null}

      <div className="mb-3 flex flex-wrap gap-2">
        {source.documentUrl ? (
          <a
            href={withPdfPageAnchor(source.documentUrl, source.pageNumber)}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 rounded-md border border-border/70 bg-card px-2.5 py-1.5 text-xs font-medium text-fg/75 hover:underline"
          >
            Ouvrir le PDF <ExternalLink size={12} />
          </a>
        ) : null}
        <button
          type="button"
          onClick={() => void copySource()}
          className="inline-flex items-center gap-1 rounded-md border border-border/70 bg-card px-2.5 py-1.5 text-xs font-medium text-fg/75 hover:underline"
        >
          Copier la source <Copy size={12} />
        </button>
      </div>

      {copied ? <p className="mb-2 text-[11px] text-emerald-600 dark:text-emerald-300">Source copiée</p> : null}

      <div className="min-h-0 flex-1 overflow-hidden rounded-xl border border-border/70 bg-card/[0.5]">
        {effectiveEngine === "pdfjs" ? (
          <PdfJsRenderer source={source} src={pdfDocumentUrl} onFatalError={() => setPdfJsFailed(true)} />
        ) : (
          <NativePdfRenderer src={nativePreviewUrl} title={`Aperçu ${source.documentName}`} />
        )}
      </div>

      <p className="mt-2 text-[11px] text-fg/55">
        Le viewer PDF natif ne permet pas un surlignage interne fiable. Le surlignage précis nécessite une intégration pdf.js.
      </p>
    </div>
  );
}
