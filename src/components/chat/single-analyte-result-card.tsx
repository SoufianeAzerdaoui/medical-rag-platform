"use client";

import type { ChatSource } from "@/types/chat";

type ParsedCard = {
  title: string;
  value: string;
  referenceLabel: string;
  referenceValue: string;
  status: string;
  sourceText: string;
  conclusion: string;
};

function stripMd(value: string): string {
  return String(value || "")
    .replace(/\*\*/g, "")
    .replace(/\[(.*?)\]\((.*?)\)/g, "$1")
    .trim();
}

function parseCard(content: string): ParsedCard | null {
  const lines = String(content || "")
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);
  const titleLine = lines.find((l) => l.startsWith("### "));
  const valueLine = lines.find((l) => l.toLowerCase().startsWith("- **valeur**"));
  const refLine = lines.find((l) => l.toLowerCase().startsWith("- **référence"));
  const statusLine = lines.find((l) => l.toLowerCase().startsWith("- **statut technique**"));
  const sourceLine = lines.find((l) => l.toLowerCase().startsWith("- **source**"));
  const conclusionLine = lines.find((l) => l.toLowerCase().startsWith("conclusion technique"));
  if (!titleLine || !valueLine || !refLine || !statusLine || !sourceLine || !conclusionLine) {
    return null;
  }

  const value = stripMd(valueLine.split(":").slice(1).join(":"));
  const refLabel = stripMd(refLine.split(":")[0].replace(/^-/, ""));
  const refValue = stripMd(refLine.split(":").slice(1).join(":"));
  const status = stripMd(statusLine.split(":").slice(1).join(":"));
  const sourceText = stripMd(sourceLine.split(":").slice(1).join(":"));
  const conclusion = stripMd(conclusionLine);
  if (!value || !status) {
    return null;
  }
  return {
    title: stripMd(titleLine.replace(/^###\s*/, "")),
    value,
    referenceLabel: refLabel || "Référence disponible",
    referenceValue: refValue || "non disponible",
    status,
    sourceText: sourceText || "source non disponible",
    conclusion,
  };
}

function sourceLink(sources: ChatSource[] | undefined): { label: string; href: string } | null {
  const first = Array.isArray(sources) ? sources[0] : null;
  if (!first || typeof first === "string") return null;
  const rec = first as Record<string, unknown>;
  const labelRaw = rec.label ?? rec.documentName ?? rec.documentId;
  const hrefRaw = rec.url ?? rec.viewer_url;
  const label = typeof labelRaw === "string" ? labelRaw : "";
  const href = typeof hrefRaw === "string" ? hrefRaw : "";
  if (!label || !href) return null;
  return { label, href };
}

function statusBadgeClass(status: string): string {
  const s = status.toLowerCase();
  if (s.includes("critique") || s.includes("vérifier")) return "border-rose-500/35 bg-rose-500/10 text-rose-700 dark:text-rose-200";
  if (s.includes("au-dessus")) return "border-amber-500/30 bg-amber-500/12 text-amber-800 dark:text-amber-200";
  if (s.includes("en dessous")) return "border-sky-500/30 bg-sky-500/12 text-sky-800 dark:text-sky-200";
  if (s.includes("dans la référence")) return "border-emerald-500/25 bg-emerald-500/10 text-emerald-700 dark:text-emerald-200";
  return "border-border bg-fg/[0.04] text-fg/65";
}

function statusLabel(status: string): string {
  const s = status.toLowerCase();
  if (s.includes("au-dessus")) return "Haut";
  if (s.includes("en dessous")) return "Bas";
  if (s.includes("dans la référence")) return "Normal";
  if (s.includes("critique") || s.includes("vérifier")) return "À vérifier";
  return status || "Non trouvé";
}

type Props = {
  content: string;
  sources?: ChatSource[];
};

export function SingleAnalyteResultCard({ content, sources }: Props) {
  const parsed = parseCard(content);
  if (!parsed) return null;
  const src = sourceLink(sources);
  return (
    <div className="rounded-xl border border-border/70 bg-card/[0.78] p-4 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <h3 className="min-w-0 text-sm font-semibold">{parsed.title}</h3>
        <span className={`shrink-0 rounded-full border px-2.5 py-1 text-[11px] font-semibold ${statusBadgeClass(parsed.status)}`}>
          {statusLabel(parsed.status)}
        </span>
      </div>
      <p className="mt-3 text-2xl font-semibold tracking-tight">{parsed.value}</p>
      <div className="mt-3 grid gap-2 text-sm">
        <p className="flex gap-2">
          <span className="min-w-24 text-fg/58">{parsed.referenceLabel}</span>
          <span className="text-fg/86">{parsed.referenceValue}</span>
        </p>
        <p className="flex gap-2">
          <span className="min-w-24 text-fg/58">Source</span>
          {src ? (
            <a
              href={src.href}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex min-w-0 items-center gap-1 rounded-lg border border-border/70 bg-card px-2.5 py-1 text-xs font-medium text-accent underline-offset-2 hover:underline"
            >
              <span aria-hidden="true">↗</span>
              <span className="break-words">{src.label}</span>
            </a>
          ) : (
            <span className="text-fg/86">{parsed.sourceText}</span>
          )}
        </p>
      </div>
      <p className="mt-3 rounded-lg border border-border/70 bg-fg/[0.025] px-3 py-2 text-sm leading-6 text-fg/80">{parsed.conclusion}</p>
    </div>
  );
}
