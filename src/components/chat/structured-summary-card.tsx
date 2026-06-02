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
  kind: "technical_summary" | "doctor_note" | "reference_ranges_note";
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

function cleanFindingChip(value: string): string {
  const cleaned = normalizeMedicalUnits(cleanSegment(value))
    .replace(/\s*\.\s*$/g, "")
    .replace(/\s{2,}/g, " ")
    .trim();
  const compact = cleaned
    .replace(/\s*=\s*/g, " = ")
    .replace(/\s*\(\s*/g, " (")
    .replace(/\s*\)\s*/g, ") ")
    .replace(/\s{2,}/g, " ")
    .trim();
  return compact;
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

function firstSentenceOnly(value: string): string {
  const text = normalizeMedicalUnits(cleanSegment(value));
  if (!text) return "";
  const m = text.match(/^(.+?[.!?])(?:\s|$)/);
  return sanitizeForSentence(m?.[1] || text);
}

function parseSummary(content: string, diagnostics?: AssistantDiagnostics): ParsedSummary {
  const lines = content
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  const titleLine = lines.find(
    (line) =>
      /^note de synth[eè]se m[eé]dicale/i.test(line) ||
      /^note m[eé]dicale/i.test(line) ||
      /^note sur les valeurs physiologiques/i.test(line),
  );
  if (titleLine) {
    const isReferenceRangesNote = /^note sur les valeurs physiologiques/i.test(titleLine);
    const llmNarrativeReferenceNote =
      isReferenceRangesNote && String(diagnostics?.final_answer_source || "").toLowerCase() === "llm_writer";
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

    const documentAnalyzedRaw = extractLine(content, ["document analysé", "document analyse"]);

    const contextLine = lines.find(
      (line) =>
        !/^note de synth[eè]se m[eé]dicale/i.test(line) &&
        !/^note m[eé]dicale/i.test(line) &&
        !/^note sur les valeurs physiologiques/i.test(line) &&
        !/^document analys[ée]\s*:/i.test(line) &&
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

    if (isReferenceRangesNote && !llmNarrativeReferenceNote) {
      for (const line of lines) {
        if (
          /^plages min-max\s*:/i.test(line) ||
          /^seuils\s*\(/i.test(line) ||
          /^seuils et cat[ée]gories interpr[ée]tatives/i.test(line) ||
          /^r[ée]f[ée]rences selon sexe\/[âa]ge\s*:/i.test(line) ||
          /^r[ée]f[ée]rences selon [âa]ge\/sexe\s*:/i.test(line) ||
          /^cat[ée]gories interpr[ée]tatives\s*:/i.test(line)
        ) {
          noteLines.push(ensureSentence(line));
        }
      }
    } else {
      if (notableRaw) noteLines.push(ensureSentence(`Le bilan montre plusieurs écarts biologiques documentés, notamment : ${notableRaw}`));
      if (extraRaw) noteLines.push(ensureSentence(`Autres écarts documentés : ${extraRaw}`));
      if (normalRaw) noteLines.push(ensureSentence(`Dans les références indiquées : ${normalRaw}`));
    }

    const sourcePagesHint = (() => {
      const source = String(sourceRaw || "");
      const rangeMatch = source.match(/pages?\s*\d+\s*[-–]\s*\d+/i);
      const pageMatch = source.match(/page\s*\d+/i);
      return sanitizeForSentence(rangeMatch?.[0] || pageMatch?.[0] || "");
    })();

    for (const line of lines) {
      if (
        /^note de synth[eè]se m[eé]dicale/i.test(line) ||
        /^note m[eé]dicale/i.test(line) ||
        /^note sur les valeurs physiologiques/i.test(line) ||
        /^document analys[ée]\s*:/i.test(line) ||
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
      kind: isReferenceRangesNote ? "reference_ranges_note" : "doctor_note",
      title: normalizeMedicalUnits(cleanSegment(titleLine)),
      context: isReferenceRangesNote
        ? (sourcePagesHint || firstSentenceOnly(documentAnalyzedRaw))
        : normalizeMedicalUnits(cleanSegment(contextLine)),
      anomalies: splitLineItems(notableRaw),
      normals: splitLineItems(normalRaw),
      notableExtra: splitLineItems(extraRaw),
      warning: isReferenceRangesNote
        ? "Note descriptive uniquement, sans diagnostic médical."
        : normalizeMedicalUnits(cleanSegment(warningRaw)),
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
  if (tone === "high") return "status-warning";
  if (tone === "low") return "status-low";
  if (tone === "normal") return "status-success";
  if (tone === "unknown") return "status-neutral";
  return "status-neutral";
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

function compactPreviewItems(items: string[], maxVisible = 4): { visible: string[]; hiddenCount: number } {
  const visible = items.slice(0, maxVisible);
  return { visible, hiddenCount: Math.max(0, items.length - visible.length) };
}

function previewItems(items: string[], maxVisible: number, expanded: boolean): { visible: string[]; hiddenCount: number } {
  const preview = compactPreviewItems(items, maxVisible);
  return {
    visible: expanded ? items : preview.visible,
    hiddenCount: preview.hiddenCount,
  };
}

function summarizeTechnicalFinding(anomalies: string[], normals: string[], sourceHint: string | null): string {
  const abnormalCount = anomalies.length;
  const normalCount = normals.length;
  const abnormalPreview = anomalies.slice(0, 2).map(cleanFindingChip).join("; ");
  const normalPreview = normals.slice(0, 2).map(cleanFindingChip).join("; ");
  const abnormalLabel = abnormalCount === 1 ? "un écart biologique documenté" : `${abnormalCount} écarts biologiques documentés`;
  const normalLabel = normalCount === 1 ? "un résultat dans la référence" : `${normalCount} résultats dans la référence`;
  const contextualEnding = sourceHint ? ` sur ${sourceHint}` : "";
  if (abnormalCount > 0 && normalCount > 0) {
    return `Le bilan met en évidence ${abnormalLabel} et ${normalLabel}. Les écarts principaux concernent ${abnormalPreview}${abnormalCount > 2 ? " et d'autres paramètres." : "."} Plusieurs résultats restent dans la référence, notamment ${normalPreview}${normalCount > 2 ? " et d'autres paramètres." : "."}${contextualEnding}.`;
  }
  if (abnormalCount > 0) {
    return `Le bilan met en évidence ${abnormalLabel}, principalement ${abnormalPreview}${abnormalCount > 2 ? " et d'autres paramètres." : "."} La lecture reste strictement descriptive et ne constitue pas un diagnostic${contextualEnding}.`;
  }
  if (normalCount > 0) {
    return `Le rapport présente ${normalLabel}, sans anomalie mise en avant dans cette synthèse. Les éléments les plus visibles sont ${normalPreview}${normalCount > 2 ? " et d'autres paramètres." : "."}${contextualEnding}.`;
  }
  return sourceHint
    ? `Synthèse descriptive fondée sur ${sourceHint}, sans diagnostic médical.`
    : "Synthèse descriptive limitée aux données du rapport, sans diagnostic médical.";
}

function looksLikeWeakBoilerplate(value: string): boolean {
  const text = value.toLowerCase();
  return [
    "réponse descriptive limitée aux données du rapport",
    "reponse descriptive limitée aux données du rapport",
    "réponse descriptive limitée aux données disponibles",
    "reponse descriptive limitee aux donnees disponibles",
    "nécessitent un contexte clinique",
    "necessitent un contexte clinique",
    "les analytes mentionnés sont anormaux",
    "les analytes mentionnes sont anormaux",
    "synthèse descriptive limitée",
    "synthese descriptive limitee",
    "réponse descriptive limitée",
    "reponse descriptive limitee",
    "lecture prudente",
    "sans diagnostic",
    "interpétés correctement",
    "interpretes correctement",
    "le bilan met en évidence des anomalies",
    "le bilan met en evidence des anomalies",
    "les analytes mentionnés sont anormaux ou nécessitent un contexte clinique",
    "les analytes mentionnes sont anormaux ou necessitent un contexte clinique",
  ].some((pattern) => text.includes(pattern));
}

export function StructuredSummaryCard({ content, sources = [], diagnostics }: Props) {
  const [showAllRanges, setShowAllRanges] = useState(false);
  const [showAllAnomalies, setShowAllAnomalies] = useState(false);
  const [showAllNormals, setShowAllNormals] = useState(false);
  const parsed = parseSummary(content, diagnostics);
  const sourceHint = firstSourceHint(sources);
  const sourceLink = firstSourceLink(sources);
  const parsedSource = parsed.source && !/^document fourni\.?$/i.test(parsed.source) ? parsed.source : "";
  const rawSynthesis = synthesisText(parsed.warning || parsed.conclusion);
  const isDoctorNote = parsed.kind === "doctor_note" || parsed.kind === "reference_ranges_note";
  const isReferenceRangesNote = parsed.kind === "reference_ranges_note";
  const docScope = Array.isArray(diagnostics?.requested_doc_ids) && diagnostics?.requested_doc_ids?.length
    ? diagnostics.requested_doc_ids.join(", ")
    : null;
  const doctorNoteParagraph = parsed.noteLines
    .map((line) => sanitizeForSentence(line))
    .filter(Boolean)
    .join(" ");
  const fallbackNarrative = content
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .join(" ");
  const rangeItems = parsed.rangeItems || [];
  const rangePreviewCount = 4;
  const visibleRangeItems = showAllRanges ? rangeItems : rangeItems.slice(0, rangePreviewCount);
  const hasMoreRanges = rangeItems.length > rangePreviewCount;
  const rangeSentence = visibleRangeItems.length
    ? `Plages et statuts documentés : ${visibleRangeItems.join("; ")}.`
    : "";
  const anomalyPreview = previewItems(parsed.anomalies, 4, showAllAnomalies);
  const normalPreview = previewItems(parsed.normals, 4, showAllNormals);
  const anomalyItems = anomalyPreview.visible;
  const anomalyHiddenCount = anomalyPreview.hiddenCount;
  const normalItems = normalPreview.visible;
  const normalHiddenCount = normalPreview.hiddenCount;
  const editorialSynthesis = summarizeTechnicalFinding(parsed.anomalies, parsed.normals, sourceHint || parsedSource || docScope);
  const synthesis = (
    !isDoctorNote &&
    (looksLikeWeakBoilerplate(rawSynthesis) || rawSynthesis.length < 55)
  )
    ? editorialSynthesis
    : rawSynthesis;

  return (
    <motion.section
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className="space-y-3 rounded-2xl border border-border/70 bg-gradient-to-b from-fg/[0.035] to-fg/[0.015] p-4 shadow-[0_12px_40px_hsl(220_30%_8%_/_0.08)]"
    >
      <div className="flex flex-wrap items-center gap-2">
        <span className="inline-flex items-center gap-1 rounded-full border border-border/70 bg-card/80 px-2.5 py-1 text-xs font-medium text-fg/80">
          <FlaskConical size={12} />
          {isDoctorNote ? (isReferenceRangesNote ? "Note sur les valeurs physiologiques" : "Note médicale") : "Résumé technique"}
        </span>
        {diagnostics?.llm_quality_escalation_used ? (
          <span className="inline-flex items-center gap-1 rounded-full border border-cyan-400/30 bg-cyan-400/10 px-2.5 py-1 text-xs font-medium text-cyan-200">
            <BadgeCheck size={12} />
            Gemini éditorial
          </span>
        ) : null}
        <span className="inline-flex items-center gap-1 rounded-full border border-accent/30 bg-accent/10 px-2.5 py-1 text-xs font-medium text-accent transition hover:-translate-y-px">
          <BadgeCheck size={12} />
          Source vérifiée
        </span>
        <span className="inline-flex items-center rounded-full border border-border/60 bg-bg/40 px-2.5 py-1 text-[11px] text-fg/68">
          Périmètre: {docScope || "document fourni"}
        </span>
      </div>

      <div className="flex flex-wrap items-center gap-2 text-[11px] text-fg/65">
        <span className="inline-flex items-center gap-1 rounded-full border border-emerald-500/25 bg-emerald-500/8 px-2.5 py-0.5 text-emerald-200">
          <ShieldCheck size={11} />
          Lecture prudente, sans diagnostic
        </span>
        <span className="inline-flex items-center rounded-full border border-border/60 bg-card/60 px-2.5 py-0.5">
          Synthèse structurée
        </span>
      </div>

      {isDoctorNote ? (
        <>
          {DOCTOR_NOTE_DEMO_COMPACT ? (
            <section className="space-y-2 rounded-2xl border border-border/50 bg-card/40 p-3">
              <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-fg/62">Résumé narratif</p>
              <p className="text-sm leading-6 text-fg/90">
                {doctorNoteParagraph ? firstSentence(doctorNoteParagraph) : firstSentence(fallbackNarrative)}
                {rangeSentence ? ` ${firstSentence(rangeSentence)}` : ""}
              </p>
              <div className="flex flex-wrap items-center gap-2 text-[11px] text-fg/64">
                <span className="rounded-full border border-border/60 bg-bg/45 px-2 py-0.5">
                  {parsed.context ? sanitizeForSentence(parsed.context) : docScope || "document fourni"}
                </span>
                <span className="rounded-full border border-border/60 bg-bg/45 px-2 py-0.5">
                  {firstSentence(synthesis)}
                </span>
              </div>
            </section>
          ) : (
            <>
              <section className="space-y-2 rounded-2xl border border-border/50 bg-card/40 p-3">
                <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-fg/62">Note clinique</p>
                <p className="text-sm leading-6 text-fg/90">
                  {doctorNoteParagraph ? firstSentence(doctorNoteParagraph) : firstSentence(fallbackNarrative)}
                </p>
                {rangeSentence ? (
                  <div className="flex flex-wrap items-center gap-2 text-[11px] text-fg/62">
                    <span className="rounded-full border border-border/60 bg-bg/45 px-2 py-0.5">
                      {firstSentence(rangeSentence)}
                    </span>
                    {hasMoreRanges ? (
                      <button
                        type="button"
                        onClick={() => setShowAllRanges((prev) => !prev)}
                        className="rounded-full border border-accent/25 bg-accent/8 px-2 py-0.5 font-medium text-accent underline-offset-2 hover:underline"
                      >
                        {showAllRanges ? "Réduire" : "Afficher tout"}
                      </button>
                    ) : null}
                  </div>
                ) : null}
              </section>

              <div className="grid gap-2 sm:grid-cols-2">
                <section className="space-y-1 rounded-2xl border border-border/50 bg-card/35 p-3">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-fg/62">Avertissement</p>
                  <p className="text-sm leading-6 text-fg/90">{DOCTOR_NOTE_DEMO_COMPACT ? firstSentence(synthesis) : synthesis}</p>
                </section>
                <section className="space-y-1 rounded-2xl border border-border/50 bg-card/35 p-3">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-fg/62">Source</p>
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
              </div>
            </>
          )}
        </>
      ) : (
        <>
          <motion.section
            initial={{ opacity: 0, y: 3 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.05, duration: 0.18 }}
            className="space-y-2 rounded-2xl border border-border/50 bg-card/35 p-3 shadow-sm"
          >
            <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-fg/62">
              Anomalies détectées <span className="text-fg/50">· {parsed.anomalies.length}</span>
            </p>
            {parsed.anomalies.length > 0 ? (
              <div className="space-y-2">
                <div className="flex flex-wrap gap-2">
                  {anomalyItems.map((item, index) => {
                    const tone = toneForItem(item);
                    const prefix = statusPrefix(tone);
                    const cleanItem = cleanFindingChip(item);
                    return (
                      <span
                        key={`abn-${index}-${item}`}
                        className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-xs ${badgeClass(tone)}`}
                      >
                        <AlertTriangle size={12} />
                        <span className="font-semibold">{prefix}</span> {cleanItem}
                        {tone === "unknown" ? <span className="font-medium">· à vérifier</span> : null}
                      </span>
                    );
                  })}
                  {anomalyHiddenCount > 0 ? (
                    <button
                      type="button"
                      onClick={() => setShowAllAnomalies((prev) => !prev)}
                      aria-expanded={showAllAnomalies}
                      className="relative z-10 inline-flex cursor-pointer items-center rounded-full border border-border/60 bg-bg/45 px-2.5 py-1 text-xs text-fg/70 transition hover:border-accent/30 hover:bg-accent/8 hover:text-fg"
                    >
                      {showAllAnomalies ? "Réduire" : `+${anomalyHiddenCount} autres`}
                    </button>
                  ) : null}
                </div>
                {showAllAnomalies && anomalyHiddenCount > 0 ? (
                  <div className="flex flex-wrap gap-2 border-t border-border/40 pt-2">
                    {parsed.anomalies.slice(4).map((item, index) => {
                      const tone = toneForItem(item);
                      const prefix = statusPrefix(tone);
                      return (
                        <span
                          key={`abn-extra-${index}-${item}`}
                          className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-xs ${badgeClass(tone)}`}
                        >
                          <AlertTriangle size={12} />
                          <span className="font-semibold">{prefix}</span> {item}
                          {tone === "unknown" ? <span className="font-medium">· à vérifier</span> : null}
                        </span>
                      );
                    })}
                  </div>
                ) : null}
              </div>
            ) : (
              <p className="text-sm text-fg/80">Aucune anomalie signalée dans ce résumé.</p>
            )}
          </motion.section>

          <motion.section
            initial={{ opacity: 0, y: 3 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.08, duration: 0.18 }}
            className="space-y-2 rounded-2xl border border-border/50 bg-card/35 p-3 shadow-sm"
          >
            <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-fg/62">
              Résultats dans la référence <span className="text-fg/50">· {parsed.normals.length}</span>
            </p>
            {parsed.normals.length > 0 ? (
              <div className="space-y-2">
                <div className="flex flex-wrap gap-2">
                  {normalItems.map((item, index) => {
                    const cleanItem = cleanFindingChip(item);
                    return (
                      <span
                        key={`norm-${index}-${item}`}
                        className="inline-flex items-center gap-1 rounded-full border border-emerald-500/25 bg-emerald-500/10 px-2.5 py-1 text-xs text-emerald-100"
                      >
                        <span className="font-semibold">✓</span> {cleanItem}
                      </span>
                    );
                  })}
                  {normalHiddenCount > 0 ? (
                    <button
                      type="button"
                      onClick={() => setShowAllNormals((prev) => !prev)}
                      aria-expanded={showAllNormals}
                      className="relative z-10 inline-flex cursor-pointer items-center rounded-full border border-border/60 bg-bg/45 px-2.5 py-1 text-xs text-fg/70 transition hover:border-accent/30 hover:bg-accent/8 hover:text-fg"
                    >
                      {showAllNormals ? "Réduire" : `+${normalHiddenCount} autres`}
                    </button>
                  ) : null}
                </div>
                {showAllNormals && normalHiddenCount > 0 ? (
                  <div className="flex flex-wrap gap-2 border-t border-border/40 pt-2">
                    {parsed.normals.slice(4).map((item, index) => (
                      <span
                        key={`norm-extra-${index}-${item}`}
                        className="inline-flex items-center gap-1 rounded-full border border-emerald-500/25 bg-emerald-500/10 px-2.5 py-1 text-xs text-emerald-100"
                      >
                        <span className="font-semibold">✓</span> {item}
                      </span>
                    ))}
                  </div>
                ) : null}
              </div>
            ) : (
              <p className="text-sm text-fg/80">Aucun résultat dans la référence mentionné.</p>
            )}
          </motion.section>

          <motion.section
            initial={{ opacity: 0, y: 3 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.11, duration: 0.18 }}
            className="space-y-2 rounded-2xl border border-border/50 bg-card/35 p-3 shadow-sm"
          >
            <p className="text-xs font-semibold uppercase tracking-wide text-fg/70">Synthèse</p>
            <p className="rounded-xl border border-border/50 bg-bg/45 px-3 py-2.5 text-sm leading-6 text-fg/90">
              {synthesis}
            </p>
            <div className="flex flex-wrap items-center gap-2 text-[11px] text-fg/60">
              <span className="inline-flex items-center gap-1.5">
                <FileText size={11} />
                Source principale:
              </span>
              <span className="rounded-full border border-border/60 bg-card/60 px-2 py-0.5 text-fg/72">
                {parsedSource || sourceHint || "voir les sources cliquables"}
              </span>
            </div>
          </motion.section>
        </>
      )}
    </motion.section>
  );
}
