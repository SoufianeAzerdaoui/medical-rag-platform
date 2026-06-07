"use client";

import { FlaskConical, FileText, ShieldCheck } from "lucide-react";
import type { ChatSource } from "@/types/chat";

type ParsedCard = {
  title: string;
  value: string;
  unit: string;
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

function normalize(value: string): string {
  return stripMd(value).replace(/\s+/g, " ").trim();
}

function findLineValue(lines: string[], labels: string[]): string {
  const normalizedLabels = labels.map((label) => normalize(label).toLowerCase());
  for (const line of lines) {
    const plain = normalize(line).replace(/^\s*[-*•]\s*/, "");
    const lower = plain.toLowerCase();
    for (const label of normalizedLabels) {
      if (lower.startsWith(label)) {
        return plain.split(/[:：]/).slice(1).join(":").trim();
      }
    }
  }
  return "";
}

function splitValueAndUnit(value: string): { value: string; unit: string } {
  const normalized = normalize(value);
  const match = normalized.match(/^([+-]?\d+(?:[.,]\d+)?)(?:\s+(.+))?$/);
  if (!match) return { value: normalized, unit: "" };
  return {
    value: match[1].replace(",", "."),
    unit: normalize(match[2] || ""),
  };
}

function parseCard(content: string): ParsedCard | null {
  const lines = String(content || "")
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);
  const titleLine = lines.find((l) => {
    const plain = normalize(l);
    return /^###\s+/.test(plain) || (/^[^:]{4,}$/.test(plain) && !/^(synthèse|synthese|conclusion|source|sources|niveau de support documentaire|qualité de synthèse|qualite de synthese)/i.test(plain));
  });
  const valueLine = lines.find((l) => /(?:^|\s)(?:-\s*)?(?:\*\*)?valeur(?:\*\*)?\s*[:：]/i.test(normalize(l)));
  const refLine = lines.find((l) =>
    /(?:^|\s)(?:-\s*)?(?:\*\*)?(?:référence|reference)(?:\s+(?:disponible|applicable|applicable|féminine|feminine))?(?:\*\*)?\s*[:：]/i.test(
      normalize(l),
    ),
  );
  const statusLine = lines.find((l) => /(?:^|\s)(?:-\s*)?(?:\*\*)?statut (?:technique|interprétatif)(?:\*\*)?\s*[:：]/i.test(normalize(l)));
  const sourceLine = lines.find((l) => /(?:^|\s)(?:-\s*)?(?:\*\*)?source(?:\*\*)?\s*[:：]/i.test(normalize(l)));
  const conclusionLine = lines.find((l) => /conclusion technique/i.test(normalize(l)));
  if (!titleLine || !valueLine || !refLine || !statusLine || !sourceLine || !conclusionLine) {
    return null;
  }

  const valueSplit = splitValueAndUnit(findLineValue(lines, ["Valeur"]));
  const refLabel = stripMd(refLine.split(":")[0].replace(/^-/, ""));
  const refValue = stripMd(
    findLineValue(lines, [
      "Référence applicable",
      "Référence disponible",
      "Référence féminine",
      "Référence femme",
      "Référence",
      "Reference",
      "Intervalle de référence",
      "Plage de référence",
    ]),
  );
  const status = stripMd(findLineValue(lines, ["Statut technique", "Statut interprétatif"]));
  const sourceText = stripMd(findLineValue(lines, ["Source"]));
  const conclusion = stripMd(conclusionLine);
  if (!valueSplit.value || !status) {
    return null;
  }
  return {
    title: stripMd(titleLine.replace(/^###\s*/, "")),
    value: valueSplit.value,
    unit: valueSplit.unit,
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
  if (s.includes("critique") || s.includes("vérifier")) return "status-danger";
  if (s.includes("au-dessus")) return "status-warning";
  if (s.includes("en dessous")) return "status-low";
  if (s.includes("dans la référence")) return "status-success";
  return "status-neutral";
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
  const lineCount = String(content || "")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean).length;
  const isCompact = lineCount <= 8 || String(content || "").length < 260;
  return (
    <div className={`relative overflow-hidden rounded-[28px] border border-border/70 bg-[radial-gradient(circle_at_top_right,rgba(14,165,233,0.10),transparent_30%),linear-gradient(180deg,rgba(255,255,255,0.035),rgba(255,255,255,0.015))] ${isCompact ? "p-4" : "p-5"} shadow-[0_22px_70px_hsl(220_35%_5%_/_0.16)]`}>
      <div className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-accent/40 to-transparent" />
      <div className="pointer-events-none absolute -right-12 -top-16 h-40 w-40 rounded-full bg-accent/10 blur-3xl" />
      <div className="pointer-events-none absolute -left-10 bottom-0 h-36 w-36 rounded-full bg-emerald-500/10 blur-3xl" />

      <div className={`${isCompact ? "mb-3" : "mb-4"} flex flex-wrap items-center gap-2 text-[11px] text-fg/65`}>
        <span className="inline-flex items-center gap-1 rounded-full border border-border/70 bg-card/80 px-2.5 py-1 font-medium text-fg/80">
          <FlaskConical size={12} />
          Résultat documentaire
        </span>
        <span className="inline-flex items-center gap-1 rounded-full border border-emerald-500/25 bg-emerald-500/8 px-2.5 py-1 text-emerald-200">
          <ShieldCheck size={11} />
          Source vérifiée
        </span>
      </div>

      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 space-y-1">
          <h3 className="truncate text-sm font-semibold tracking-[0.01em] text-fg/95">{parsed.title}</h3>
          <p className="text-[11px] uppercase tracking-[0.18em] text-fg/55">Lecture prudente et documentée</p>
        </div>
        <span className={`shrink-0 rounded-full border px-2.5 py-1 text-[11px] font-semibold shadow-sm ${statusBadgeClass(parsed.status)}`}>
          {statusLabel(parsed.status)}
        </span>
      </div>

      <div className={`mt-4 rounded-[24px] border border-border/60 bg-card/55 ${isCompact ? "p-3.5" : "p-4"} shadow-[0_8px_24px_hsl(220_35%_5%_/_0.12)]`}>
        <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-fg/55">Valeur documentée</p>
        <div className={`mt-2 flex flex-wrap items-end ${isCompact ? "gap-1.5" : "gap-2"}`}>
          <p className={`${isCompact ? "text-2xl" : "text-3xl"} font-semibold tracking-tight text-fg/96`}>{parsed.value}</p>
          {parsed.unit ? (
            <span className="mb-0.5 rounded-full border border-border/60 bg-bg/35 px-2.5 py-1 text-xs font-medium text-fg/76">
              {parsed.unit}
            </span>
          ) : null}
        </div>
        <div className={`${isCompact ? "mt-3" : "mt-4"} grid gap-3 sm:grid-cols-2`}>
          <div className="rounded-2xl border border-border/60 bg-bg/35 px-3 py-2.5">
            <p className="text-[10px] uppercase tracking-[0.18em] text-fg/55">{parsed.referenceLabel}</p>
            <p className={`mt-1 ${isCompact ? "text-xs leading-5" : "text-sm"} text-fg/88`}>{parsed.referenceValue}</p>
          </div>
          <div className="rounded-2xl border border-border/60 bg-bg/35 px-3 py-2.5">
            <p className="text-[10px] uppercase tracking-[0.18em] text-fg/55">Statut interprétatif</p>
            <p className={`mt-1 ${isCompact ? "text-xs leading-5" : "text-sm"} text-fg/88`}>{parsed.status}</p>
          </div>
        </div>
      </div>

      <div className={`mt-4 grid gap-3 ${isCompact ? "lg:grid-cols-[1.05fr_0.95fr]" : "lg:grid-cols-2"} lg:items-stretch`}>
        <div className={`rounded-2xl border border-border/60 bg-card/45 ${isCompact ? "p-3.5" : "p-4"}`}>
          <div className="flex items-center gap-2">
            <div className="rounded-full border border-accent/20 bg-accent/10 p-1.5 text-accent">
              <FileText size={12} />
            </div>
            <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-fg/62">Source documentaire</p>
          </div>
          {src ? (
            <a
              href={src.href}
              target="_blank"
              rel="noopener noreferrer"
              className={`mt-2 flex w-full min-w-0 items-start gap-1.5 rounded-xl border border-border/60 bg-bg/45 ${isCompact ? "px-2.5 py-1.5 text-xs leading-5" : "px-3 py-2 text-sm"} font-medium text-accent transition hover:border-accent/30 hover:bg-accent/8 hover:underline`}
            >
              <span aria-hidden="true">↗</span>
              <span className="break-words">{src.label}</span>
            </a>
          ) : (
            <p className={`mt-2 ${isCompact ? "text-xs leading-5" : "text-sm"} text-fg/86`}>{parsed.sourceText}</p>
          )}
        </div>

        <div className={`rounded-2xl border border-emerald-500/20 bg-emerald-500/10 ${isCompact ? "p-3.5" : "p-4"}`}>
          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-emerald-100/80">Conclusion technique</p>
          <p className={`mt-2 ${isCompact ? "text-xs leading-5" : "text-sm leading-6"} text-emerald-50/92`}>{parsed.conclusion}</p>
        </div>
      </div>
    </div>
  );
}
