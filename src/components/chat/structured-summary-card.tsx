"use client";

import { motion } from "framer-motion";
import { AlertTriangle, BadgeCheck, FileText, FlaskConical, ShieldCheck } from "lucide-react";
import { useState } from "react";
import type { AssistantDiagnostics, ChatSource, SourceCitation } from "@/types/chat";

type Props = {
  content: string;
  sources?: ChatSource[];
  diagnostics?: AssistantDiagnostics;
};

type ParsedSummary = {
  kind: "technical_summary" | "doctor_note";
  title: string;
  context: string;
  anomalies: string[];
  normals: string[];
  notableExtra: string[];
  warning: string;
  source: string;
  conclusion: string;
  noteLines: string[];
  rangeItems: string[];
};

function isCitation(source: ChatSource): source is SourceCitation {
  return typeof source === "object" && source !== null && "doc_id" in source && typeof source.doc_id === "string";
}

function firstSourceHint(sources: ChatSource[] = []): string | null {
  for (const source of sources) {
    if (isCitation(source)) {
      const doc = source.filename || source.doc_id;
      if (source.page) return `${doc}, page ${source.page}`;
      return doc;
    }
    if (typeof source === "string") {
      const docMatch = source.match(/doc_id=([^,\]\s]+)/i);
      const pageMatch = source.match(/page=(\d+)/i);
      if (docMatch?.[1]) {
        return pageMatch?.[1] ? `${docMatch[1]}, page ${pageMatch[1]}` : docMatch[1];
      }
    }
  }
  return null;
}

function firstSourceLink(sources: ChatSource[] = []): { label: string; href: string } | null {
  for (const source of sources) {
    if (isCitation(source)) {
      const href = source.viewer_url || source.url || "";
      if (!href) continue;
      const doc = source.filename || source.doc_id;
      const label = source.page ? `${doc} — page ${source.page}` : doc;
      return { label, href };
    }
    if (typeof source === "string") {
      const urlMatch = source.match(/(https?:\/\/[^\s)]+|\/(?:viewer\/pdf|api\/documents\/)[^\s)]+)/i);
      if (!urlMatch?.[1]) continue;
      const label = sanitizeForSentence(source.replace(urlMatch[1], "").trim()) || "Source PDF";
      return { label, href: urlMatch[1] };
    }
    const legacy = source as { viewer_url?: string; url?: string; documentName?: string };
    const href = String(legacy.viewer_url || legacy.url || "").trim();
    if (!href) continue;
    const label = String(legacy.documentName || "Source PDF").trim();
    return { label, href };
  }
  return null;
}

function cleanSegment(value: string): string {
  return value.replace(/^\s*[-*]\s*/, "").replace(/\s+/g, " ").trim();
}

function sanitizeForSentence(value: string): string {
  return normalizeMedicalUnits(cleanSegment(value)).replace(/[.;:,!\s]+$/g, "").trim();
}

function ensureSentence(value: string): string {
  const base = sanitizeForSentence(value);
  return base ? `${base}.` : "";
}

function normalizeMedicalUnits(value: string): string {
  return value
    .replace(/\bmg\/l\b/gi, "mg/L")
    .replace(/\bng\/ml\b/gi, "ng/mL")
    .replace(/\bmmol\/l\b/gi, "mmol/L")
    .replace(/\bpmol\/l\b/gi, "pmol/L")
    .replace(/\bmeq\/l\b/gi, "mEq/L")
    .replace(/\biu\/ml\b/gi, "IU/mL")
    .replace(/\bmui\/l\b/gi, "mUI/L")
    .replace(/\bui\/l\b/gi, "UI/L");
}

function splitLineItems(value: string): string[] {
  const compact = cleanSegment(value);
  if (!compact) return [];
  if (/aucun résultat/i.test(compact) || /aucun resultat/i.test(compact)) return [];
  return compact
    .split(/[;•]/g)
    .map((item) => normalizeMedicalUnits(cleanSegment(item)))
    .filter(Boolean);
}

function extractLine(text: string, keys: string[]): string {
  const lines = text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  for (const line of lines) {
    const lower = line.toLowerCase();
    if (keys.some((key) => lower.startsWith(key))) {
      const idx = line.indexOf(":");
      return idx >= 0 ? line.slice(idx + 1).trim() : line;
    }
  }
  return "";
}

function parseSummary(content: string): ParsedSummary {
  const lines = content
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  const titleLine = lines.find((line) => /^note de synth[eè]se m[eé]dicale/i.test(line));
  if (titleLine) {
    const noteLines: string[] = [];
    const notableRaw = extractLine(content, ["points biologiques notables", "paramètres hors référence notables", "parametres hors reference notables"]);
    const rangesRaw = extractLine(content, ["plages et statuts documentés", "plages et statuts documentes", "plages de référence documentées", "plages de reference documentees"]);
    const extraRaw = extractLine(content, ["autres écarts documentés", "autres ecarts documentes", "autres éléments notables", "autres elements notables"]);
    const normalRaw = extractLine(content, [
      "plusieurs autres paramètres sont dans l’intervalle de référence, notamment",
      "plusieurs autres parametres sont dans l'intervalle de reference, notamment",
      "dans la référence",
      "dans la reference",
    ]);
    const warningRaw = extractLine(content, ["note descriptive uniquement", "avertissement"]);
    const sourceRaw = extractLine(content, ["source"]);
    const conclusionRaw = extractLine(content, ["conclusion technique", "synthèse", "synthese"]);
    const rangeItems = splitLineItems(rangesRaw);

    const contextLine = lines.find(
      (line) =>
        !/^note de synth[eè]se m[eé]dicale/i.test(line) &&
        !/^points biologiques notables\s*:/i.test(line) &&
        !/^param[eè]tres hors r[ée]f[ée]rence notables\s*:/i.test(line) &&
        !/^plages et statuts document[ée]s\s*:/i.test(line) &&
        !/^plages de r[ée]f[ée]rence document[ée]es\s*:/i.test(line) &&
        !/^autres [ée]carts document[ée]s\s*:/i.test(line) &&
        !/^plusieurs autres param[èe]tres sont dans l[’']intervalle de r[ée]f[ée]rence/i.test(line) &&
        !/^note descriptive uniquement/i.test(line) &&
        !/^source\s*:/i.test(line) &&
        !/^conclusion technique\s*:/i.test(line),
    ) || "";

    if (notableRaw) noteLines.push(ensureSentence(`Le bilan montre plusieurs écarts biologiques documentés, notamment : ${notableRaw}`));
    if (extraRaw) noteLines.push(ensureSentence(`Autres écarts documentés : ${extraRaw}`));
    if (normalRaw) noteLines.push(ensureSentence(`Dans les références indiquées : ${normalRaw}`));

    for (const line of lines) {
      if (
        /^note de synth[eè]se m[eé]dicale/i.test(line) ||
        /^points biologiques notables\s*:/i.test(line) ||
        /^param[eè]tres hors r[ée]f[ée]rence notables\s*:/i.test(line) ||
        /^plages et statuts document[ée]s\s*:/i.test(line) ||
        /^plages de r[ée]f[ée]rence document[ée]es\s*:/i.test(line) ||
        /^autres [ée]carts document[ée]s\s*:/i.test(line) ||
        /^plusieurs autres param[èe]tres sont dans l[’']intervalle de r[ée]f[ée]rence/i.test(line) ||
        /^note descriptive uniquement/i.test(line) ||
        /^source\s*:/i.test(line) ||
        /^conclusion technique\s*:/i.test(line)
      ) {
        continue;
      }
      const cleaned = normalizeMedicalUnits(cleanSegment(line));
      if (cleaned && !noteLines.includes(cleaned) && cleaned !== normalizeMedicalUnits(cleanSegment(contextLine))) {
        noteLines.push(ensureSentence(cleaned));
      }
    }

    return {
      kind: "doctor_note",
      title: normalizeMedicalUnits(cleanSegment(titleLine)),
      context: normalizeMedicalUnits(cleanSegment(contextLine)),
      anomalies: splitLineItems(notableRaw),
      normals: splitLineItems(normalRaw),
      notableExtra: splitLineItems(extraRaw),
      warning: normalizeMedicalUnits(cleanSegment(warningRaw)),
      source: normalizeMedicalUnits(cleanSegment(sourceRaw)),
      conclusion: normalizeMedicalUnits(cleanSegment(conclusionRaw)),
      noteLines,
      rangeItems,
    };
  }

  const anomaliesRaw = extractLine(content, ["anormaux", "anomalies"]);
  const normalsRaw = extractLine(content, ["resultats dans la reference", "résultats dans la référence", "principaux resultats dans la reference", "principaux résultats dans la référence"]);
  const conclusionRaw = extractLine(content, ["conclusion technique", "conclusion"]);
  return {
    kind: "technical_summary",
    title: "Résumé technique",
    context: "",
    anomalies: splitLineItems(anomaliesRaw),
    normals: splitLineItems(normalsRaw),
    notableExtra: [],
    warning: "",
    source: "",
    conclusion: normalizeMedicalUnits(cleanSegment(conclusionRaw)),
    noteLines: [],
    rangeItems: [],
  };
}

function hasAmbiguousComparator(item: string): boolean {
  const text = item.toLowerCase();
  const hasStatus = text.includes("au-dessus") || text.includes("au dessus") || text.includes("en dessous") || text.includes("en-dessous");
  const hasCurrentComparator = /=\s*[<>≤≥]/.test(text);
  const hasReferenceComparator = /\b(réf|ref)\b[^;]*[<>≤≥]/.test(text);
  return hasStatus && hasCurrentComparator && hasReferenceComparator;
}

function toneForItem(item: string): "high" | "low" | "normal" | "unknown" | "neutral" {
  if (hasAmbiguousComparator(item)) return "unknown";
  const value = item.toLowerCase();
  if (value.includes("au-dessus") || value.includes("au dessus")) return "high";
  if (value.includes("en dessous") || value.includes("en-dessous")) return "low";
  if (value.includes("dans la référence") || value.includes("dans la reference")) return "normal";
  return "neutral";
}

function badgeClass(tone: "high" | "low" | "normal" | "unknown" | "neutral"): string {
  if (tone === "high") return "border-amber-500/40 bg-amber-500/15 text-amber-100";
  if (tone === "low") return "border-sky-500/40 bg-sky-500/15 text-sky-100";
  if (tone === "normal") return "border-emerald-500/35 bg-emerald-500/10 text-emerald-100";
  if (tone === "unknown") return "border-violet-500/35 bg-violet-500/12 text-violet-100";
  return "border-slate-600/60 bg-slate-800/70 text-slate-100";
}

function statusPrefix(tone: "high" | "low" | "normal" | "unknown" | "neutral"): string {
  if (tone === "high") return "↑";
  if (tone === "low") return "↓";
  if (tone === "normal") return "✓";
  if (tone === "unknown") return "?";
  return "•";
}

function synthesisText(raw: string): string {
  const value = raw.trim();
  if (!value) return "Réponse descriptive limitée aux données du rapport, sans diagnostic médical.";
  const lower = value.toLowerCase();
  if (lower.includes("synthèse descriptive limitée aux données disponibles")) {
    return "Réponse descriptive limitée aux données du rapport, sans diagnostic médical.";
  }
  return value;
}

const DOCTOR_NOTE_DEMO_COMPACT = (() => {
  const raw = String(process.env.NEXT_PUBLIC_DOCTOR_NOTE_DEMO_COMPACT || "").trim().toLowerCase();
  return raw === "1" || raw === "true" || raw === "yes" || raw === "on";
})();

function firstSentence(value: string): string {
  const text = sanitizeForSentence(value);
  if (!text) return "";
  const m = text.match(/^(.+?[.!?])(?:\s|$)/);
  return (m?.[1] || text).trim();
}

export function StructuredSummaryCard({ content, sources = [], diagnostics }: Props) {
  const [showAllRanges, setShowAllRanges] = useState(false);
  const parsed = parseSummary(content);
  const sourceHint = firstSourceHint(sources);
  const sourceLink = firstSourceLink(sources);
  const parsedSource = parsed.source && !/^document fourni\.?$/i.test(parsed.source) ? parsed.source : "";
  const synthesis = synthesisText(parsed.warning || parsed.conclusion);
  const isDoctorNote = parsed.kind === "doctor_note";
  const docScope = Array.isArray(diagnostics?.requested_doc_ids) && diagnostics?.requested_doc_ids?.length
    ? diagnostics.requested_doc_ids.join(", ")
    : null;
  const doctorNoteParagraph = parsed.noteLines
    .map((line) => sanitizeForSentence(line))
    .filter(Boolean)
    .join(" ");
  const rangeItems = parsed.rangeItems || [];
  const rangePreviewCount = 4;
  const visibleRangeItems = showAllRanges ? rangeItems : rangeItems.slice(0, rangePreviewCount);
  const hasMoreRanges = rangeItems.length > rangePreviewCount;
  const hiddenRangesCount = Math.max(0, rangeItems.length - visibleRangeItems.length);
  const rangeSentence = visibleRangeItems.length
    ? `Plages et statuts documentés : ${visibleRangeItems.join("; ")}${hiddenRangesCount > 0 ? `; +${hiddenRangesCount} autre(s)` : ""}.`
    : "";

  return (
    <motion.section
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className="space-y-2 rounded-2xl border border-border/80 bg-gradient-to-b from-fg/[0.04] to-fg/[0.02] p-4 shadow-sm"
    >
      <div className="flex flex-wrap items-center gap-2">
        <span className="inline-flex items-center gap-1 rounded-full border border-border bg-card px-2.5 py-1 text-xs font-medium text-fg/80 transition hover:-translate-y-px">
          <FlaskConical size={12} />
          {isDoctorNote ? "Note médicale" : "Résumé technique"}
        </span>
        <span className="inline-flex items-center gap-1 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-1 text-xs font-medium text-emerald-200 transition hover:-translate-y-px">
          <ShieldCheck size={12} />
          Sans diagnostic
        </span>
        <span className="inline-flex items-center gap-1 rounded-full border border-accent/30 bg-accent/10 px-2.5 py-1 text-xs font-medium text-accent transition hover:-translate-y-px">
          <BadgeCheck size={12} />
          Source vérifiée
        </span>
      </div>

      {isDoctorNote ? (
        <>
          {DOCTOR_NOTE_DEMO_COMPACT ? (
            <section className="space-y-1 rounded-xl border border-border/50 bg-card/35 p-2.5">
              <p className="text-sm text-fg/90">
                <span className="font-medium">Document analysé:</span>{" "}
                {docScope || "document fourni"}
                {parsed.context ? <span> — {sanitizeForSentence(parsed.context)}.</span> : null}
              </p>
              <p className="text-sm text-fg/90">
                <span className="font-medium">Note:</span>{" "}
                {doctorNoteParagraph ? firstSentence(doctorNoteParagraph) : "Synthèse médicale narrative disponible."}
                {rangeSentence ? ` ${firstSentence(rangeSentence)}` : ""}
              </p>
              <p className="text-sm text-fg/90">
                <span className="font-medium">Avertissement:</span> {firstSentence(synthesis)}{" "}
                <span className="font-medium">Source:</span>{" "}
                {sourceLink ? (
                  <a
                    href={sourceLink.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-accent underline-offset-2 hover:underline"
                  >
                    {parsedSource || sourceLink.label}
                  </a>
                ) : (
                  <span>{parsedSource || sourceHint || "voir les sources cliquables ci-dessous"}</span>
                )}
              </p>
            </section>
          ) : (
            <>
          <section className="space-y-1 rounded-xl border border-border/50 bg-card/35 p-2.5">
            <p className="text-xs font-semibold uppercase tracking-wide text-fg/70">Document analysé</p>
            <p className="text-sm text-fg/90">
              {docScope || "document fourni"}
              {parsed.context ? <span> — {parsed.context}</span> : null}
            </p>
          </section>

          <section className="space-y-1 rounded-xl border border-border/50 bg-card/35 p-2.5">
            <p className="text-xs font-semibold uppercase tracking-wide text-fg/70">Note</p>
            {doctorNoteParagraph ? (
              <div className="space-y-2">
                <p className="text-sm leading-6 text-fg/90">
                  {DOCTOR_NOTE_DEMO_COMPACT ? firstSentence(doctorNoteParagraph) : doctorNoteParagraph}
                </p>
                {rangeSentence ? (
                  <div className="space-y-1">
                    <p className="text-sm leading-6 text-fg/90">{rangeSentence}</p>
                    {hasMoreRanges ? (
                      <button
                        type="button"
                        onClick={() => setShowAllRanges((prev) => !prev)}
                        className="text-xs font-medium text-accent underline-offset-2 hover:underline"
                      >
                        {showAllRanges ? "Voir moins" : "Voir plus"}
                      </button>
                    ) : null}
                  </div>
                ) : null}
              </div>
            ) : (
              <p className="text-sm text-fg/90">Synthèse médicale narrative disponible dans la réponse.</p>
            )}
          </section>

          <section className="space-y-1 rounded-xl border border-border/50 bg-card/35 p-2.5">
            <p className="text-xs font-semibold uppercase tracking-wide text-fg/70">Avertissement</p>
            <p className="text-sm leading-6 text-fg/90">
              {DOCTOR_NOTE_DEMO_COMPACT ? firstSentence(synthesis) : synthesis}
            </p>
          </section>

          <section className="space-y-1 rounded-xl border border-border/50 bg-card/35 p-2.5">
            <p className="text-xs font-semibold uppercase tracking-wide text-fg/70">Source</p>
            {sourceLink ? (
              <a
                href={sourceLink.href}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-accent underline-offset-2 hover:underline"
              >
                {parsedSource || sourceLink.label}
              </a>
            ) : (
              <p className="text-sm text-fg/90">{parsedSource || sourceHint || "voir les sources cliquables ci-dessous"}</p>
            )}
          </section>
            </>
          )}
        </>
      ) : (
        <>
          {docScope ? (
            <p className="rounded-lg border border-border/60 bg-card/40 px-3 py-2 text-sm text-fg/90">
              <span className="font-medium">Périmètre d’analyse:</span> {docScope}.
            </p>
          ) : null}

          <motion.section
            initial={{ opacity: 0, y: 3 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.05, duration: 0.18 }}
            className="space-y-2 rounded-xl border border-border/50 bg-card/35 p-3"
          >
            <p className="text-xs font-semibold uppercase tracking-wide text-fg/70">
              Anomalies détectées <span className="text-fg/50">· {parsed.anomalies.length}</span>
            </p>
            {parsed.anomalies.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {parsed.anomalies.map((item) => {
                  const tone = toneForItem(item);
                  const prefix = statusPrefix(tone);
                  return (
                    <span key={`abn-${item}`} className={`inline-flex items-center gap-1 rounded-lg border px-2.5 py-1 text-xs transition hover:-translate-y-px ${badgeClass(tone)}`}>
                      <AlertTriangle size={12} />
                      <span className="font-semibold">{prefix}</span> {item}
                      {tone === "unknown" ? <span className="font-medium">· à vérifier</span> : null}
                    </span>
                  );
                })}
              </div>
            ) : (
              <p className="text-sm text-fg/80">Aucune anomalie signalée dans ce résumé.</p>
            )}
          </motion.section>

          <motion.section
            initial={{ opacity: 0, y: 3 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.08, duration: 0.18 }}
            className="space-y-2 rounded-xl border border-border/50 bg-card/35 p-3"
          >
            <p className="text-xs font-semibold uppercase tracking-wide text-fg/70">
              Résultats dans la référence <span className="text-fg/50">· {parsed.normals.length}</span>
            </p>
            {parsed.normals.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {parsed.normals.map((item) => (
                  <span key={`norm-${item}`} className="inline-flex items-center gap-1 rounded-lg border border-emerald-500/25 bg-emerald-500/10 px-2.5 py-1 text-xs text-emerald-100 transition hover:-translate-y-px">
                    <span className="font-semibold">✓</span> {item}
                  </span>
                ))}
              </div>
            ) : (
              <p className="text-sm text-fg/80">Aucun résultat dans la référence mentionné.</p>
            )}
          </motion.section>

          <p className="rounded-lg border border-border/70 bg-card/60 px-3 py-2 text-sm leading-6 text-fg/90">
            <span className="font-medium">Synthèse:</span> {synthesis}
          </p>

          <p className="text-xs text-fg/70">
            <FileText size={12} className="mr-1 inline-block" />
            Source principale: {parsedSource || sourceHint || "voir les sources cliquables ci-dessous"}.
          </p>
        </>
      )}
    </motion.section>
  );
}
