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
  if (s.includes("au-dessus")) return "bg-amber-100 text-amber-800 border-amber-200";
  if (s.includes("en dessous")) return "bg-sky-100 text-sky-800 border-sky-200";
  if (s.includes("dans la référence")) return "bg-emerald-100 text-emerald-800 border-emerald-200";
  return "bg-zinc-100 text-zinc-800 border-zinc-200";
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
    <div className="rounded-xl border border-border bg-card p-4">
      <h3 className="text-sm font-semibold">{parsed.title}</h3>
      <div className="mt-3 grid gap-2 text-sm">
        <p>
          <span className="text-fg/70">Valeur :</span>{" "}
          <span className="font-semibold">{parsed.value}</span>
        </p>
        <p>
          <span className="text-fg/70">{parsed.referenceLabel} :</span>{" "}
          <span>{parsed.referenceValue}</span>
        </p>
        <p>
          <span className="text-fg/70">Statut technique :</span>{" "}
          <span className={`inline-flex rounded-md border px-2 py-0.5 text-xs font-medium ${statusBadgeClass(parsed.status)}`}>
            {parsed.status}
          </span>
        </p>
        <p>
          <span className="text-fg/70">Source :</span>{" "}
          {src ? (
            <a
              href={src.href}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 rounded-lg border border-border/70 bg-card px-2.5 py-1 text-xs font-medium text-accent underline-offset-2 hover:underline"
            >
              <span aria-hidden="true">↗</span>
              <span className="break-words">{src.label}</span>
            </a>
          ) : (
            <span>{parsed.sourceText}</span>
          )}
        </p>
      </div>
      <p className="mt-3 text-sm">{parsed.conclusion}</p>
    </div>
  );
}
