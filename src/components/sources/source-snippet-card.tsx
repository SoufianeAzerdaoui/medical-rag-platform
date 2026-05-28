"use client";

import { Copy, ExternalLink, FileText } from "lucide-react";
import { useState } from "react";
import type { SourceReference } from "@/types/source-reference";

type SourceSnippetCardProps = {
  source: SourceReference;
  onPreview?: (source: SourceReference) => void;
  compact?: boolean;
};

function toPercent(score?: number): string {
  if (typeof score !== "number" || !Number.isFinite(score)) return "non renseignée";
  const normalized = score <= 1 ? score * 100 : score;
  const clamped = Math.max(0, Math.min(100, Math.round(normalized)));
  return `${clamped}%`;
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

export function SourceSnippetCard({ source, onPreview, compact = false }: SourceSnippetCardProps) {
  const [copied, setCopied] = useState(false);
  const pageText = Number.isFinite(source.pageNumber || NaN) ? `Page ${source.pageNumber}` : "Page non précisée";

  async function copySource() {
    const payload = [
      source.documentName,
      pageText,
      lineLabel(source),
      source.snippet ? `Extrait: ${source.snippet}` : "",
    ]
      .filter(Boolean)
      .join(" · ");
    try {
      await navigator.clipboard.writeText(payload);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    } catch {
      setCopied(false);
    }
  }

  return (
    <article className={`rounded-lg border border-border/70 bg-card/[0.66] p-3 shadow-sm ${compact ? "" : ""}`}>
      <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-fg/50">Source utilisée</p>
      <p className="mt-1 line-clamp-1 text-sm font-semibold text-fg/90">{source.documentName}</p>
      <p className="mt-1 text-xs text-fg/65">{pageText} · {lineLabel(source)}</p>
      <p className="mt-1 text-xs text-fg/65">
        Pertinence : <span className="font-medium text-fg/90">{toPercent(source.score)}</span>
      </p>
      <p className="mt-2 line-clamp-2 rounded-md border border-border/60 bg-fg/[0.025] px-2.5 py-2 text-xs leading-5 text-fg/80">
        Extrait : {source.snippet || "Extrait non disponible."}
      </p>
      <div className="mt-2 flex flex-wrap gap-2">
        {onPreview ? (
          <button
            type="button"
            onClick={() => onPreview(source)}
            className="inline-flex items-center gap-1 rounded-md border border-accent/35 bg-accent/10 px-2.5 py-1.5 text-xs font-medium text-accent hover:underline"
          >
            Aperçu PDF
          </button>
        ) : null}
        {source.documentUrl ? (
          <a
            href={source.documentUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 rounded-md border border-border/70 bg-card px-2.5 py-1.5 text-xs font-medium text-fg/75 hover:underline"
          >
            Ouvrir le PDF <ExternalLink size={12} />
          </a>
        ) : (
          <span className="inline-flex items-center gap-1 rounded-md border border-border/70 bg-card px-2.5 py-1.5 text-xs text-fg/65">
            PDF indisponible <FileText size={12} />
          </span>
        )}
        <button
          type="button"
          onClick={() => void copySource()}
          className="inline-flex items-center gap-1 rounded-md border border-border/70 bg-card px-2.5 py-1.5 text-xs font-medium text-fg/75 hover:underline"
        >
          Copier la source <Copy size={12} />
        </button>
      </div>
      {copied ? <p className="mt-1 text-[11px] text-emerald-600 dark:text-emerald-300">Source copiée</p> : null}
    </article>
  );
}
