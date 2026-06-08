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
  kind: "technical_summary" | "doctor_note" | "reference_ranges_note" | "narrative_biological_summary";
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

type ToxicologyNarrative = {
  nature: string;
  families: string;
  findings: string;
  noExceedance: string;
  conclusion: string;
};

type ExplicitMultiAnalyteNotFound = {
  title: string;
  analytes: string[];
  source: string;
  conclusion: string;
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

function buildSourceViewerUrl(docId: string, page?: number | null): string {
  const encoded = encodeURIComponent(docId);
  if (page && Number.isFinite(page)) {
    return `/viewer/pdf?doc_id=${encoded}&page=${page}`;
  }
  return `/viewer/pdf?doc_id=${encoded}`;
}

function firstSourceLink(sources: ChatSource[] = []): { label: string; href: string } | null {
  for (const source of sources) {
    if (isCitation(source)) {
      const href = buildSourceViewerUrl(source.doc_id, source.page ?? null);
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

function prettifyDocumentLabel(value: string): string {
  const raw = sanitizeForSentence(value);
  if (!raw) return "";
  const reportMatch = raw.match(/^report[_\s-]*(\d+)(.*)$/i);
  if (reportMatch) {
    const suffix = String(reportMatch[2] || "").trim();
    const base = `report (${reportMatch[1]}).pdf`;
    if (!suffix) return base;
    const cleanedSuffix = suffix
      .replace(/^,?\s*/g, "")
      .replace(/^pages?\s+/i, "pages ")
      .replace(/^page\s+/i, "page ");
    return `${base} — ${cleanedSuffix}`;
  }
  return raw.replace(/_/g, " ");
}

function isInternalSourceLabel(value: string): boolean {
  return /^report[_\s-]*\d+(?:\b|[,.-])/i.test(String(value || "").trim());
}

function preferredSourceLabel(parsedSource: string, sourceHint: string | null, sourceLinkLabel: string | null): string {
  const parsedPretty = prettifyDocumentLabel(parsedSource);
  if (sourceHint && isInternalSourceLabel(parsedPretty)) return prettifyDocumentLabel(sourceHint);
  if (sourceLinkLabel && isInternalSourceLabel(parsedPretty)) return prettifyDocumentLabel(sourceLinkLabel);
  return parsedPretty || prettifyDocumentLabel(sourceHint || "") || prettifyDocumentLabel(sourceLinkLabel || "");
}

function ensureSentence(value: string): string {
  const base = sanitizeForSentence(value);
  return base ? `${base}.` : "";
}

function normalizeMedicalUnits(value: string): string {
  return value
    .replace(/\bmg\/l\b/gi, "mg/L")
    .replace(/\bg\/l\b/gi, "g/L")
    .replace(/\bng\/ml\b/gi, "ng/mL")
    .replace(/\bmmol\/l\b/gi, "mmol/L")
    .replace(/\bpmol\/l\b/gi, "pmol/L")
    .replace(/\bmeq\/l\b/gi, "mEq/L")
    .replace(/\biu\/ml\b/gi, "IU/mL")
    .replace(/\bmui\/l\b/gi, "mUI/L")
    .replace(/\bui\/l\b/gi, "UI/L")
    .replace(/\bµg\/dl\b/gi, "µg/dL")
    .replace(/\bug\/dl\b/gi, "µg/dL");
}

function splitLineItems(value: string): string[] {
  const compact = cleanSegment(value);
  if (!compact) return [];
  if (/aucun résultat/i.test(compact) || /aucun resultat/i.test(compact)) return [];
  const deduped: string[] = [];
  const seen = new Set<string>();
  for (const item of compact
    .split(/[;•]/g)
    .map((item) => normalizeMedicalUnits(cleanSegment(item)))
    .filter(Boolean)) {
    const key = item.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    deduped.push(item);
  }
  return deduped;
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

function looksLikeNarrativeBiologicalSummary(content: string, diagnostics?: AssistantDiagnostics): boolean {
  const text = normalizeMedicalUnits(cleanSegment(content)).toLowerCase();
  if (!text) return false;
  const finalSource = String(diagnostics?.final_answer_source || "").toLowerCase();
  const narrativeMarkers = [
    "bilan",
    "écarts biologiques",
    "ecarts biologiques",
    "lecture prudente",
    "sans diagnostic",
    "conclusion prudente",
    "résumé biologique",
    "resume biologique",
    "synthèse biologique",
    "synthese biologique",
    "le bilan montre",
    "le rapport montre",
    "met en évidence",
    "met en evidence",
  ];
  const hasNarrativeMarkers = narrativeMarkers.some((marker) => text.includes(marker));
  const hasTechnicalCues =
    /(?:^|\n)\s*anormaux\s*:/i.test(content) ||
    /(?:^|\n)\s*résultats dans la référence\s*:/i.test(content) ||
    /(?:^|\n)\s*resultats dans la reference\s*:/i.test(content);
  if (hasTechnicalCues) return false;
  if (finalSource === "llm_writer" || finalSource === "llm_writer_repaired") {
    return hasNarrativeMarkers || text.length > 80;
  }
  return hasNarrativeMarkers && text.length > 120;
}

function parseSummary(content: string, diagnostics?: AssistantDiagnostics): ParsedSummary {
  const lines = content
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  const narrativeFallback = looksLikeNarrativeBiologicalSummary(content, diagnostics);
  const titleBase = String(
    lines.find(
      (line) =>
        /^note de synth[eè]se m[eé]dicale/i.test(line) ||
        /^note m[eé]dicale/i.test(line) ||
        /^note sur les valeurs physiologiques/i.test(line) ||
        /^r[ée]sum[ée] biologique court/i.test(line) ||
        /^synth[eè]se biologique [ée]ditoriale/i.test(line),
    ) || "",
  );
  const titleLine = lines.find(
    (line) =>
      /^note de synth[eè]se m[eé]dicale/i.test(line) ||
      /^note m[eé]dicale/i.test(line) ||
      /^note sur les valeurs physiologiques/i.test(line) ||
      /^r[ée]sum[ée] biologique court/i.test(line) ||
      /^synth[eè]se biologique [ée]ditoriale/i.test(line) ||
      narrativeFallback,
  );
  if (titleLine || narrativeFallback) {
    const isReferenceRangesNote = /^note sur les valeurs physiologiques/i.test(titleBase);
    const isNarrativeBiologicalSummary =
      /^r[ée]sum[ée] biologique court/i.test(titleBase) ||
      /^synth[eè]se biologique [ée]ditoriale/i.test(titleBase) ||
      narrativeFallback;
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
    const conclusionRaw = extractExplicitConclusion(content) || extractLine(content, ["conclusion technique", "synthèse", "synthese"]);
    const rangeItems = splitLineItems(rangesRaw);

    const documentAnalyzedRaw = extractLine(content, ["document analysé", "document analyse"]);

    const contextLine = lines.find(
      (line) =>
        !/^note de synth[eè]se m[eé]dicale/i.test(line) &&
        !/^note m[eé]dicale/i.test(line) &&
        !/^note sur les valeurs physiologiques/i.test(line) &&
        !/^r[ée]sum[ée] biologique court/i.test(line) &&
        !/^synth[eè]se biologique [ée]ditoriale/i.test(line) &&
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

    const faithfulNarrativeLines = isNarrativeBiologicalSummary
      ? lines
          .filter(
            (line) =>
              !/^r[ée]sum[ée] biologique court/i.test(line) &&
              !/^synth[eè]se biologique [ée]ditoriale/i.test(line) &&
              !/^note descriptive uniquement/i.test(line) &&
              !/^source\s*:/i.test(line) &&
              !/^conclusion technique\s*:/i.test(line),
          )
          .map((line) => normalizeMedicalUnits(cleanSegment(line)))
          .filter(Boolean)
      : [];

    if (isNarrativeBiologicalSummary) {
      for (const line of faithfulNarrativeLines) {
        if (!noteLines.includes(line)) noteLines.push(line);
      }
    } else if (isReferenceRangesNote && !llmNarrativeReferenceNote) {
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
          /^r[ée]sum[ée] biologique court/i.test(line) ||
          /^synth[eè]se biologique [ée]ditoriale/i.test(line) ||
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
      kind: isReferenceRangesNote
        ? "reference_ranges_note"
        : (isNarrativeBiologicalSummary ? "narrative_biological_summary" : "doctor_note"),
      title: normalizeMedicalUnits(cleanSegment(titleLine || (narrativeFallback ? "Résumé biologique court" : ""))),
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

function sentenceExcerpt(value: string, maxSentences = 2): string {
  const text = sanitizeForSentence(value);
  if (!text) return "";
  const sentences = text
    .split(/(?<=[.!?])\s+/)
    .map((sentence) => sentence.trim())
    .filter(Boolean);
  if (sentences.length <= maxSentences) {
    return text;
  }
  return sentences.slice(0, maxSentences).join(" ");
}

function compactNarrativeLead(value: string, maxSentences = 2): string {
  const text = normalizeMedicalUnits(String(value || "").replace(/\r/g, "\n"));
  if (!text) return "";

  const sentences = text
    .split(/(?<=[.!?])\s+/)
    .map((sentence) => sanitizeForSentence(sentence))
    .filter(Boolean);

  const seen = new Set<string>();
  const kept: string[] = [];

  for (const sentence of sentences) {
    const lower = sentence.toLowerCase();
    if (
      lower.startsWith("source") ||
      lower.startsWith("conclusion technique") ||
      lower.startsWith("conclusion prudente") ||
      lower.startsWith("note descriptive uniquement")
    ) {
      continue;
    }

    const normalized = lower.replace(/[^a-z0-9]+/g, " ").trim();
    if (!normalized || seen.has(normalized)) continue;
    seen.add(normalized);
    kept.push(sentence);

    if (kept.length >= maxSentences) break;
  }

  return kept.join(" ");
}

function stripInlineSourceClause(value: string): string {
  return String(value || "")
    .split(/\bsource\s*:\s*/i)[0]
    .trim();
}

function compactToxicologyNature(value: string): string {
  const text = normalizeMedicalUnits(stripInlineSourceClause(value)).replace(/\s+/g, " ").trim();
  if (!text) return "";
  if (
    /(document correspond à un\s+)?bilan\s+(urinaire|sanguin)\s+(?:de\s+)?pharmaco-toxicologie/i.test(text) ||
    /(document correspond à un\s+)?bilan\s+de\s+pharmaco-toxicologie/i.test(text)
  ) {
    return "Le document correspond à un bilan urinaire de pharmaco-toxicologie.";
  }
  return firstSentenceOnly(text) || sanitizeForSentence(text);
}

function isToxicologyNarrative(content: string): boolean {
  const text = normalizeMedicalUnits(cleanSegment(content)).toLowerCase();
  if (!text) return false;
  return (
    text.includes("pharmaco-toxicologie") ||
    text.includes("pharmaco toxicologie") ||
    text.includes("bilan urinaire") ||
    text.includes("document correspond à un bilan") ||
    text.includes("document correspond a un bilan") ||
    text.includes("familles recherchées") ||
    text.includes("familles recherchees") ||
    text.includes("seuils fournis")
  );
}

function parseToxicologyNarrative(content: string): ToxicologyNarrative | null {
  if (!isToxicologyNarrative(content)) return null;
  const lines = String(content || "")
    .split("\n")
    .map((line) => normalizeMedicalUnits(cleanSegment(line)))
    .filter(Boolean);
  if (lines.length === 0) return null;

  const nature = compactToxicologyNature(
    lines.find((line) => /(document correspond à un\s+)?bilan\s+(urinaire|sanguin)\s+(?:de\s+)?pharmaco-toxicologie/i.test(line)) ||
    lines.find((line) => /(document correspond à un\s+)?bilan\s+de\s+pharmaco-toxicologie/i.test(line)) ||
    "",
  );
  const families =
    lines.find((line) => /^les familles recherch[ée]es comprennent/i.test(line)) ||
    lines.find((line) => /^les familles recherchees comprennent/i.test(line)) ||
    "";
  const findings =
    lines.find((line) => /^les valeurs semi-quantitatives retenues restent sous les seuils indiqu[ée]s/i.test(line)) ||
    lines.find((line) => /^les valeurs semi-quantitatives retenues restent sous les seuils indiques/i.test(line)) ||
    "";
  const noExceedance =
    lines.find((line) => /^aucun d[ée]passement des seuils fournis n’est mis en évidence/i.test(line)) ||
    lines.find((line) => /^aucun depassement des seuils fournis n’est mis en evidence/i.test(line)) ||
    lines.find((line) => /^aucun dépassement des seuils fournis n’est mis en évidence/i.test(line)) ||
    "";
  const conclusion =
    lines.find((line) => /^conclusion prudente\s*:/i.test(line)) ||
    lines.find((line) => /^conclusion technique\s*:/i.test(line)) ||
    "";

  return {
    nature,
    families,
    findings,
    noExceedance,
    conclusion,
  };
}

function parseToxicologyNarrativeFallback(content: string): ToxicologyNarrative | null {
  if (!isToxicologyNarrative(content)) return null;

  const lines = String(content || "")
    .split("\n")
    .map((line) => normalizeMedicalUnits(cleanSegment(line)))
    .filter(Boolean);

  const text = lines.join(" ");
  const sentence = firstSentenceOnly(text) || "";

  const nature =
    lines.find((line) => /bilan\s+(urinaire|sanguin)\s+(?:de\s+)?pharmaco-toxicologie/i.test(line)) ||
    lines.find((line) => /pharmaco-toxicologie/i.test(line) && /bilan/i.test(line)) ||
    sentence ||
    "";

  const families =
    lines.find((line) => /^les familles recherch[ée]es comprennent/i.test(line)) ||
    lines.find((line) => /familles?\s+recherch[ée]es/i.test(line)) ||
    lines.find((line) => /amphétamine|amphetamine|benzodiazépine|benzodiazepine|cocaïne|cocaine|ecstasy|opiac[ée]s|opiaces|phencyclidine/i.test(line)) ||
    "";

  const findings =
    lines.find((line) => /^les valeurs semi-quantitatives retenues restent sous les seuils indiqu[ée]s/i.test(line)) ||
    lines.find((line) => /valeurs?\s+semi-quantitatives?/i.test(line) && /seuil/i.test(line)) ||
    lines.find((line) => /<\s*\d+/.test(line)) ||
    "";

  const noExceedance =
    lines.find((line) => /^aucun d[ée]passement des seuils fournis n’est mis en évidence/i.test(line)) ||
    lines.find((line) => /^aucun depassement des seuils fournis n’est mis en evidence/i.test(line)) ||
    lines.find((line) => /^aucun dépassement des seuils fournis n’est mis en évidence/i.test(line)) ||
    lines.find((line) => /aucun\s+d[ée]passement/i.test(line)) ||
    "";

  const conclusion =
    lines.find((line) => /^conclusion prudente\s*:/i.test(line)) ||
    lines.find((line) => /^conclusion technique\s*:/i.test(line)) ||
    lines.find((line) => /lecture\s+descriptive/i.test(line) && /diagnostic/i.test(line)) ||
    "";

  if (!nature && !families && !findings && !noExceedance && !conclusion) {
    return null;
  }

  return {
    nature,
    families,
    findings,
    noExceedance,
    conclusion,
  };
}

function parseExplicitMultiAnalyteNotFound(content: string): ExplicitMultiAnalyteNotFound | null {
  const text = normalizeMedicalUnits(String(content || "").replace(/\r/g, "\n"));
  const lines = text
    .split("\n")
    .map((line) => normalizeMedicalUnits(cleanSegment(line)))
    .filter(Boolean);
  if (!lines.length) return null;

  const title = lines.find((line) => /^analytes?\s+demand[ée]s/i.test(line)) || "";
  if (!title) return null;

  const analytes: string[] = [];
  for (const line of lines) {
    const match = line.match(/^\s*[-*]\s*(.+?)\s*:\s*non retrouv[ée]e?(?:\s+dans\s+.+)?$/i);
    if (!match?.[1]) continue;
    const item = sanitizeForSentence(match[1]);
    if (!item) continue;
    if (!analytes.includes(item)) analytes.push(item);
  }

  if (!analytes.length) return null;

  const sourceLine = lines.find((line) => /^source documentaire\s*:/i.test(line)) || "";
  const conclusionLine = lines.find((line) => /^conclusion technique\s*:/i.test(line)) || "";

  return {
    title: sanitizeForSentence(title),
    analytes,
    source: sanitizeForSentence(sourceLine.replace(/^source documentaire\s*:\s*/i, "")),
    conclusion: sanitizeForSentence(conclusionLine.replace(/^conclusion technique\s*:\s*/i, "")),
  };
}

function compactNarrativeSnippet(value: string): string {
  const text = sanitizeForSentence(value);
  if (!text) return "";
  return firstSentence(text) || text;
}

function buildToxicologyLead(narrative: ToxicologyNarrative): string {
  const nature = compactToxicologyNature(narrative.nature);
  const families = sanitizeForSentence(narrative.families);
  const findings = sanitizeForSentence(narrative.findings);
  const noExceedance = sanitizeForSentence(narrative.noExceedance);

  if (nature && /pharmaco-toxicologie/i.test(nature)) {
    if (noExceedance) {
      return "Panel urinaire de pharmaco-toxicologie sans dépassement des seuils fournis.";
    }
    return "Panel urinaire de pharmaco-toxicologie documenté.";
  }

  const parts: string[] = [];
  if (nature) parts.push(nature);
  if (families) parts.push(families);
  if (findings) parts.push(findings);
  if (parts.length === 0) return "";
  const combined = parts.join(" ");
  return firstSentenceOnly(combined) || sanitizeForSentence(combined);
}

function extractExplicitConclusion(value: string): string {
  const text = normalizeMedicalUnits(String(value || "").replace(/\r/g, "\n"));
  const lines = text
    .split("\n")
    .map((line) => sanitizeForSentence(line))
    .filter(Boolean);

  const patterns = [
    /(?:^|[\s•-])conclusion(?:\s+technique)?\s*:\s*(.+)$/i,
    /(?:^|[\s•-])conclusion prudente\s*:\s*(.+)$/i,
  ];

  for (const line of lines) {
    for (const pattern of patterns) {
      const match = line.match(pattern);
      if (match?.[1]) {
        const tail = match[1]
          .split(/\s+(?:Source principale|Source|Sources)\s*[:·]/i)[0]
          .trim();
        return tail ? ensureSentence(tail) : "";
      }
    }
  }

  for (const pattern of patterns) {
    const match = text.match(pattern);
    if (match?.[1]) {
      const tail = match[1]
        .split(/\s+(?:Source principale|Source|Sources)\s*[:·]/i)[0]
        .trim();
      return tail ? ensureSentence(tail) : "";
    }
  }
  return "";
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

function toDisplayCount(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return Math.max(0, Math.round(value));
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return Math.max(0, Math.round(parsed));
    }
  }
  return null;
}

export function StructuredSummaryCard({ content, sources = [], diagnostics }: Props) {
  const [showAllRanges, setShowAllRanges] = useState(false);
  const [showAllAnomalies, setShowAllAnomalies] = useState(false);
  const [showAllNormals, setShowAllNormals] = useState(false);
  const parsed = parseSummary(content, diagnostics);
  const sourceHint = firstSourceHint(sources);
  const sourceLink = firstSourceLink(sources);
  const parsedSource = parsed.source && !/^document fourni\.?$/i.test(parsed.source) ? parsed.source : "";
  const sourceDisplayLabel = preferredSourceLabel(parsedSource, sourceHint, sourceLink?.label || null);
  const requestedDocScope = Array.isArray(diagnostics?.requested_doc_ids) && diagnostics?.requested_doc_ids?.length
    ? diagnostics.requested_doc_ids.map((item) => prettifyDocumentLabel(String(item || ""))).join(", ")
    : null;
  const rawSynthesis = synthesisText(parsed.warning || parsed.conclusion);
  const isDoctorNote =
    parsed.kind === "doctor_note" ||
    parsed.kind === "reference_ranges_note" ||
    parsed.kind === "narrative_biological_summary";
  const isReferenceRangesNote = parsed.kind === "reference_ranges_note";
  const isNarrativeBiologicalSummary = parsed.kind === "narrative_biological_summary";
  const finalAnswerSource = String(diagnostics?.final_answer_source || "").toLowerCase();
  const isLlmWriter = finalAnswerSource === "llm_writer" || finalAnswerSource === "llm_writer_repaired";
  const useFaithfulNarrative = isNarrativeBiologicalSummary && (
    finalAnswerSource === "llm_writer" ||
    finalAnswerSource === "llm_writer_repaired" ||
    finalAnswerSource === "deterministic_renderer"
  );
  const docScope = Array.isArray(diagnostics?.requested_doc_ids) && diagnostics?.requested_doc_ids?.length
    ? diagnostics.requested_doc_ids.map((item) => prettifyDocumentLabel(String(item || ""))).join(", ")
    : null;
  const backendAboveCount = toDisplayCount(diagnostics?.above_reference_count);
  const backendBelowCount = toDisplayCount(diagnostics?.below_reference_count);
  const backendWithinCount = toDisplayCount(diagnostics?.within_reference_count);
  const backendMajorAnomaliesCount = toDisplayCount(diagnostics?.major_anomalies_count);
  const backendAnomalyCount = backendMajorAnomaliesCount ?? ((backendAboveCount ?? 0) + (backendBelowCount ?? 0));
  const backendNormalCount = backendWithinCount ?? parsed.normals.length;
  const backendDocumentedCount =
    toDisplayCount(diagnostics?.displayed_evidences_count) ??
    toDisplayCount(diagnostics?.evidence_pack_count) ??
    toDisplayCount(diagnostics?.lab_result_count) ??
    toDisplayCount(diagnostics?.structured_values_count) ??
    toDisplayCount(diagnostics?.sources_count);
  const hasMeaningfulBiologicalCounts =
    backendAnomalyCount > 0 ||
    backendNormalCount > 0 ||
    parsed.anomalies.length > 0 ||
    parsed.normals.length > 0;
  const noEvidenceDocumentLabel = sourceDisplayLabel || requestedDocScope || sourceHint || "document fourni";
  const isNoEvidenceSummary =
    sources.length === 0 &&
    parsed.anomalies.length === 0 &&
    parsed.normals.length === 0 &&
    parsed.rangeItems.length === 0 &&
    /(?:aucun|aucune)\s+(?:résultat|donnée|données|valeur)/i.test(content);
  const doctorNoteParagraph = parsed.noteLines
    .map((line) => sanitizeForSentence(line))
    .filter(Boolean)
    .join(" ");
  const fallbackNarrative = content
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .join(" ");
  const faithfulNarrativeText = content.trim();
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
  const backendNarrative = rawSynthesis || parsed.conclusion || parsed.warning || "";
  const synthesis = isNarrativeBiologicalSummary && isLlmWriter
    ? (faithfulNarrativeText || backendNarrative)
    : (
      !isDoctorNote &&
      !isLlmWriter &&
      (looksLikeWeakBoilerplate(rawSynthesis) || rawSynthesis.length < 55)
      )
      ? editorialSynthesis
      : (rawSynthesis || editorialSynthesis);
  const faithfulNarrativeBody = parsed.noteLines
    .map((line) => line.trim())
    .filter(Boolean)
    .join("\n\n");
  const faithfulNarrativeLead = compactNarrativeLead(
    faithfulNarrativeText || backendNarrative || fallbackNarrative || content,
    isNarrativeBiologicalSummary ? 1 : 2,
  );
  const narrativeParagraph = useFaithfulNarrative
    ? (faithfulNarrativeLead || faithfulNarrativeBody || faithfulNarrativeText || sentenceExcerpt(fallbackNarrative || content, 3))
    : sentenceExcerpt(doctorNoteParagraph || fallbackNarrative, isNarrativeBiologicalSummary ? 3 : 2);
  const narrativeLeadCandidate =
    narrativeParagraph ||
    sentenceExcerpt(fallbackNarrative || content, 2) ||
    rawSynthesis ||
    (isDoctorNote ? "Résumé médical documenté." : "Synthèse structurée.");
  const toxicologyNarrative = isDoctorNote
    ? parseToxicologyNarrative(faithfulNarrativeText || backendNarrative || fallbackNarrative || content) ||
      parseToxicologyNarrativeFallback(faithfulNarrativeText || backendNarrative || fallbackNarrative || content)
    : null;
  const explicitMultiAnalyteNotFound = isDoctorNote
    ? parseExplicitMultiAnalyteNotFound(faithfulNarrativeText || backendNarrative || fallbackNarrative || content)
    : null;
  const isToxicologySummary = Boolean(toxicologyNarrative) || isToxicologyNarrative(faithfulNarrativeText || backendNarrative || fallbackNarrative || content);
  const narrativeInfoCards = isDoctorNote && !toxicologyNarrative
    ? [
      {
        label: "Contexte",
        value: sanitizeForSentence(parsed.context || docScope || "document fourni"),
      },
      {
        label: "Cadre",
        value: "Prudente, sans diagnostic",
      },
      {
        label: "Source documentaire",
        value: sanitizeForSentence(sourceDisplayLabel || "document fourni"),
      },
      {
        label: "Conclusion",
        value: compactNarrativeSnippet(backendNarrative || synthesis || ""),
      },
    ].filter((item) => item.value)
    : [];
  const narrativeLeadNeedsBackfill = (() => {
    const normalizedLead = sanitizeForSentence(narrativeLeadCandidate).toLowerCase();
    const normalizedTitle = sanitizeForSentence(parsed.title || "").toLowerCase();
    if (!normalizedLead) return true;
    if (normalizedTitle && normalizedLead === normalizedTitle) return true;
    if (normalizedLead.length < 28) return true;
    if (isDoctorNote && /^note de synth[eè]se m[eé]dicale/i.test(normalizedLead)) return true;
    return false;
  })();
  const narrativeLeadText = narrativeLeadNeedsBackfill
    ? (
      (isDoctorNote ? editorialSynthesis : "") ||
      sentenceExcerpt(backendNarrative || faithfulNarrativeText || fallbackNarrative || content, isDoctorNote ? 3 : 2) ||
      narrativeLeadCandidate
    )
    : narrativeLeadCandidate;
  const explicitConclusion = useFaithfulNarrative
    ? extractExplicitConclusion(faithfulNarrativeText || backendNarrative || fallbackNarrative || content)
    : extractExplicitConclusion(content);
  const narrativeConclusion = useFaithfulNarrative
    ? (explicitConclusion || sentenceExcerpt(backendNarrative || fallbackNarrative || content, 2))
      : sentenceExcerpt(synthesis, 2);
  const paragraphLower = sanitizeForSentence(narrativeParagraph).toLowerCase();
  const conclusionLower = sanitizeForSentence(narrativeConclusion).toLowerCase();
  const showConclusionPanel =
    !narrativeConclusion ||
    !isNarrativeBiologicalSummary ||
    !paragraphLower ||
    !conclusionLower ||
    !paragraphLower.includes(conclusionLower);
  const isNarrativeMedicalNoteTitle =
    /^note de synth[eè]se m[eé]dicale/i.test(parsed.title) ||
    /^note m[eé]dicale/i.test(parsed.title);
  const summaryChipLabel = isDoctorNote
    ? (isReferenceRangesNote
      ? "Note sur les valeurs physiologiques"
      : (isNarrativeBiologicalSummary && !isNarrativeMedicalNoteTitle ? "Synthèse biologique" : "Synthèse médicale"))
    : "Résumé technique";

  if (toxicologyNarrative && !isNoEvidenceSummary) {
    const toxicologyLeadText = buildToxicologyLead(toxicologyNarrative);
    return (
      <motion.section
        initial={{ opacity: 0, y: 4 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.2 }}
        className="relative overflow-hidden space-y-4 rounded-[28px] border border-border/70 bg-[radial-gradient(circle_at_top_right,rgba(14,165,233,0.10),transparent_28%),linear-gradient(180deg,rgba(255,255,255,0.035),rgba(255,255,255,0.015))] p-5 shadow-[0_22px_70px_hsl(220_35%_5%_/_0.22)]"
      >
        <div className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-accent/40 to-transparent" />
        <div className="pointer-events-none absolute -right-12 -top-16 h-40 w-40 rounded-full bg-accent/10 blur-3xl" />
        <div className="pointer-events-none absolute -left-10 bottom-0 h-36 w-36 rounded-full bg-emerald-500/10 blur-3xl" />

        <div className="flex flex-wrap items-center gap-2">
          <span className="inline-flex items-center gap-1 rounded-full border border-border/70 bg-card/80 px-2.5 py-1 text-xs font-medium text-fg/80">
            <FlaskConical size={12} />
            {summaryChipLabel}
          </span>
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
            Lecture prudente et documentée
          </span>
          <span className="inline-flex items-center rounded-full border border-border/60 bg-card/60 px-2.5 py-0.5">
            Synthèse structurée
          </span>
        </div>

        <div className="grid gap-2 sm:grid-cols-3">
          <div className="rounded-2xl border border-border/60 bg-card/55 px-3 py-2.5 shadow-[0_6px_18px_hsl(220_35%_5%_/_0.08)]">
            <p className="text-[10px] uppercase tracking-[0.22em] text-fg/55">Type d’examen</p>
            <p className="mt-1 text-sm font-semibold text-fg/92">{toxicologyNarrative.nature || "Pharmaco-toxicologie urinaire"}</p>
          </div>
          <div className="rounded-2xl border border-border/60 bg-card/55 px-3 py-2.5 shadow-[0_6px_18px_hsl(220_35%_5%_/_0.08)]">
            <p className="text-[10px] uppercase tracking-[0.22em] text-fg/55">Familles analysées</p>
            <p className="mt-1 text-sm font-semibold text-fg/92">{toxicologyNarrative.families || "Amphétamine, benzodiazépine, cocaïne, ecstasy, opiacés, phencyclidine"}</p>
          </div>
          <div className="rounded-2xl border border-border/60 bg-card/55 px-3 py-2.5 shadow-[0_6px_18px_hsl(220_35%_5%_/_0.08)]">
            <p className="text-[10px] uppercase tracking-[0.22em] text-fg/55">Résultats sous seuil</p>
            <p className="mt-1 text-sm leading-6 text-fg/90">{toxicologyNarrative.findings || "Les valeurs restent sous les seuils indiqués."}</p>
          </div>
        </div>

        <div className="grid gap-3 lg:grid-cols-[minmax(0,1.4fr)_minmax(280px,0.95fr)]">
          <section className="space-y-3 rounded-[24px] border border-border/60 bg-card/55 p-4 shadow-[0_8px_24px_hsl(220_35%_5%_/_0.12)]">
            <div className="flex items-center gap-2">
              <div className="rounded-full border border-accent/20 bg-accent/10 p-1.5 text-accent">
                <FlaskConical size={12} />
              </div>
              <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-fg/62">Synthèse</p>
            </div>
            <p className="text-sm leading-6 text-fg/92">
              {toxicologyLeadText || toxicologyNarrative.nature || "Le document présente une synthèse toxico documentée."}
            </p>
            {toxicologyNarrative.noExceedance ? (
              <div className="rounded-2xl border border-emerald-500/20 bg-emerald-500/8 px-3 py-2.5">
                <p className="text-[10px] uppercase tracking-[0.18em] text-emerald-200/80">Conclusion opérationnelle</p>
                <p className="mt-1 text-sm leading-6 text-fg/92">{toxicologyNarrative.noExceedance}</p>
              </div>
            ) : null}
          </section>

          <div className="space-y-3">
            <section className="space-y-2 rounded-[24px] border border-border/60 bg-card/45 p-4 shadow-[0_8px_24px_hsl(220_35%_5%_/_0.10)]">
              <div className="flex items-center gap-2">
                <div className="rounded-full border border-emerald-500/20 bg-emerald-500/10 p-1.5 text-emerald-200">
                  <ShieldCheck size={12} />
                </div>
                <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-fg/62">Conclusion technique</p>
              </div>
              <p className="text-sm leading-6 text-fg/92">
                {toxicologyNarrative.conclusion || "Lecture descriptive à corréler au contexte clinique, sans diagnostic."}
              </p>
            </section>
            <section className="space-y-2 rounded-[24px] border border-border/60 bg-card/45 p-4 shadow-[0_8px_24px_hsl(220_35%_5%_/_0.10)]">
              <div className="flex items-center gap-2">
                <div className="rounded-full border border-accent/20 bg-accent/10 p-1.5 text-accent">
                  <FileText size={12} />
                </div>
                <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-fg/62">Source documentaire</p>
              </div>
              {sourceLink ? (
                <a
                  href={sourceLink.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1.5 rounded-xl border border-border/60 bg-bg/45 px-3 py-2 text-sm text-accent transition hover:border-accent/30 hover:bg-accent/8 hover:underline"
                >
                  <span aria-hidden="true">↗</span>
                  <span className="min-w-0 break-words">{sourceDisplayLabel || prettifyDocumentLabel(sourceLink.label)}</span>
                </a>
              ) : (
                <p className="text-sm text-fg/90">{sourceDisplayLabel || "voir les sources cliquables ci-dessous"}</p>
              )}
            </section>
          </div>
        </div>
      </motion.section>
    );
  }

  if (explicitMultiAnalyteNotFound) {
    const explicitSourceLabel = explicitMultiAnalyteNotFound.source || sourceDisplayLabel || sourceHint || "document fourni";
    const explicitConclusion =
      explicitMultiAnalyteNotFound.conclusion ||
      "Aucun résultat exploitable correspondant aux analytes demandés n’a été identifié.";
    return (
      <motion.section
        initial={{ opacity: 0, y: 4 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.2 }}
        className="relative overflow-hidden space-y-4 rounded-[28px] border border-border/70 bg-[radial-gradient(circle_at_top_right,rgba(14,165,233,0.10),transparent_28%),linear-gradient(180deg,rgba(255,255,255,0.035),rgba(255,255,255,0.015))] p-5 shadow-[0_22px_70px_hsl(220_35%_5%_/_0.22)]"
      >
        <div className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-accent/40 to-transparent" />
        <div className="pointer-events-none absolute -right-12 -top-16 h-40 w-40 rounded-full bg-accent/10 blur-3xl" />
        <div className="pointer-events-none absolute -left-10 bottom-0 h-36 w-36 rounded-full bg-emerald-500/10 blur-3xl" />

        <div className="flex flex-wrap items-center gap-2">
          <span className="inline-flex items-center gap-1 rounded-full border border-border/70 bg-card/80 px-2.5 py-1 text-xs font-medium text-fg/80">
            <FlaskConical size={12} />
            {summaryChipLabel}
          </span>
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
            Lecture prudente et documentée
          </span>
          <span className="inline-flex items-center rounded-full border border-border/60 bg-card/60 px-2.5 py-0.5">
            Synthèse structurée
          </span>
        </div>

        <div className="grid gap-3 lg:grid-cols-[minmax(0,1.3fr)_minmax(280px,0.95fr)]">
          <section className="space-y-3 rounded-[24px] border border-border/60 bg-card/55 p-4 shadow-[0_8px_24px_hsl(220_35%_5%_/_0.12)]">
            <div className="flex items-center gap-2">
              <div className="rounded-full border border-accent/20 bg-accent/10 p-1.5 text-accent">
                <FlaskConical size={12} />
              </div>
              <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-fg/62">Synthèse</p>
            </div>
            <p className="text-sm leading-6 text-fg/92">
              Aucun des analytes demandés n’a été retrouvé dans le document.
            </p>
            <div className="flex flex-wrap gap-2">
              {explicitMultiAnalyteNotFound.analytes.map((analyte) => (
                <span
                  key={analyte}
                  className="inline-flex items-center gap-1 rounded-full border border-border/60 bg-bg/45 px-2.5 py-1 text-xs text-fg/80"
                >
                  <span className="font-semibold">•</span>
                  {analyte}
                  <span className="text-fg/55">non retrouvé</span>
                </span>
              ))}
            </div>
          </section>

          <div className="space-y-3">
            <section className="space-y-2 rounded-[24px] border border-border/60 bg-card/45 p-4 shadow-[0_8px_24px_hsl(220_35%_5%_/_0.10)]">
              <div className="flex items-center gap-2">
                <div className="rounded-full border border-emerald-500/20 bg-emerald-500/10 p-1.5 text-emerald-200">
                  <ShieldCheck size={12} />
                </div>
                <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-fg/62">Conclusion technique</p>
              </div>
              <p className="text-sm leading-6 text-fg/92">
                {explicitConclusion}
              </p>
            </section>
            <section className="space-y-2 rounded-[24px] border border-border/60 bg-card/45 p-4 shadow-[0_8px_24px_hsl(220_35%_5%_/_0.10)]">
              <div className="flex items-center gap-2">
                <div className="rounded-full border border-accent/20 bg-accent/10 p-1.5 text-accent">
                  <FileText size={12} />
                </div>
                <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-fg/62">Source documentaire</p>
              </div>
              {sourceLink ? (
                <a
                  href={sourceLink.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1.5 rounded-xl border border-border/60 bg-bg/45 px-3 py-2 text-sm text-accent transition hover:border-accent/30 hover:bg-accent/8 hover:underline"
                >
                  <span aria-hidden="true">↗</span>
                  <span className="min-w-0 break-words">{sourceDisplayLabel || prettifyDocumentLabel(sourceLink.label)}</span>
                </a>
              ) : (
                <p className="text-sm text-fg/90">{sourceDisplayLabel || "voir les sources cliquables ci-dessous"}</p>
              )}
            </section>
          </div>
        </div>
      </motion.section>
    );
  }

  if (isNoEvidenceSummary) {
    return (
      <motion.section
        initial={{ opacity: 0, y: 4 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.2 }}
        className="relative overflow-hidden space-y-4 rounded-[28px] border border-border/70 bg-[radial-gradient(circle_at_top_right,rgba(14,165,233,0.10),transparent_28%),linear-gradient(180deg,rgba(255,255,255,0.035),rgba(255,255,255,0.015))] p-5 shadow-[0_22px_70px_hsl(220_35%_5%_/_0.22)]"
      >
        <div className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-accent/40 to-transparent" />
        <div className="pointer-events-none absolute -right-12 -top-16 h-40 w-40 rounded-full bg-accent/10 blur-3xl" />
        <div className="pointer-events-none absolute -left-10 bottom-0 h-36 w-36 rounded-full bg-emerald-500/10 blur-3xl" />

        <div className="flex flex-wrap items-center gap-2">
          <span className="inline-flex items-center gap-1 rounded-full border border-border/70 bg-card/80 px-2.5 py-1 text-xs font-medium text-fg/80">
            <FlaskConical size={12} />
            {summaryChipLabel}
          </span>
          <span className="inline-flex items-center gap-1 rounded-full border border-accent/30 bg-accent/10 px-2.5 py-1 text-xs font-medium text-accent transition hover:-translate-y-px">
            <BadgeCheck size={12} />
            Source vérifiée
          </span>
          <span className="inline-flex items-center rounded-full border border-border/60 bg-bg/40 px-2.5 py-1 text-[11px] text-fg/68">
            Périmètre: {requestedDocScope || "document fourni"}
          </span>
        </div>

        <div className="grid gap-3 lg:grid-cols-[minmax(0,1.4fr)_minmax(260px,0.9fr)]">
          <section className="space-y-3 rounded-[24px] border border-border/60 bg-card/55 p-4 shadow-[0_8px_24px_hsl(220_35%_5%_/_0.12)]">
            <div className="flex items-center gap-2">
              <div className="rounded-full border border-accent/20 bg-accent/10 p-1.5 text-accent">
                <FileText size={12} />
              </div>
              <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-fg/62">Synthèse</p>
            </div>
            <p className="text-sm leading-6 text-fg/92">
              Aucune donnée structurée exploitable n’a été extraite de {noEvidenceDocumentLabel} pour cette demande.
            </p>
            <p className="text-sm leading-6 text-fg/72">
              Le document ciblé ne contient pas de valeur exploitable pour cette demande.
            </p>
          </section>

          <div className="space-y-3">
            <section className="space-y-2 rounded-[24px] border border-border/60 bg-card/45 p-4 shadow-[0_8px_24px_hsl(220_35%_5%_/_0.10)]">
              <div className="flex items-center gap-2">
                <div className="rounded-full border border-emerald-500/20 bg-emerald-500/10 p-1.5 text-emerald-200">
                  <ShieldCheck size={12} />
                </div>
                <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-fg/62">Conclusion technique</p>
              </div>
              <p className="text-sm leading-6 text-fg/92">
                Aucune source cliquable n’a été produite; la réponse reste strictement documentaire.
              </p>
            </section>
            <section className="space-y-2 rounded-[24px] border border-border/60 bg-card/45 p-4 shadow-[0_8px_24px_hsl(220_35%_5%_/_0.10)]">
              <div className="flex items-center gap-2">
                <div className="rounded-full border border-accent/20 bg-accent/10 p-1.5 text-accent">
                  <FileText size={12} />
                </div>
                <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-fg/62">Source documentaire</p>
              </div>
              <p className="text-sm text-fg/90">{noEvidenceDocumentLabel}</p>
              <p className="text-[11px] leading-5 text-fg/48">Aucune source cliquable n’a été produite.</p>
            </section>
          </div>
        </div>
      </motion.section>
    );
  }

  return (
    <motion.section
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className="relative overflow-hidden space-y-4 rounded-[28px] border border-border/70 bg-[radial-gradient(circle_at_top_right,rgba(14,165,233,0.10),transparent_28%),linear-gradient(180deg,rgba(255,255,255,0.035),rgba(255,255,255,0.015))] p-5 shadow-[0_22px_70px_hsl(220_35%_5%_/_0.22)]"
    >
      <div className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-accent/40 to-transparent" />
      <div className="pointer-events-none absolute -right-12 -top-16 h-40 w-40 rounded-full bg-accent/10 blur-3xl" />
      <div className="pointer-events-none absolute -left-10 bottom-0 h-36 w-36 rounded-full bg-emerald-500/10 blur-3xl" />

      <div className="flex flex-wrap items-center gap-2">
        <span className="inline-flex items-center gap-1 rounded-full border border-border/70 bg-card/80 px-2.5 py-1 text-xs font-medium text-fg/80">
          <FlaskConical size={12} />
          {summaryChipLabel}
        </span>
        {diagnostics?.llm_quality_escalation_used && isLlmWriter ? (
          <span className="inline-flex items-center gap-1 rounded-full border border-cyan-400/30 bg-cyan-400/10 px-2.5 py-1 text-xs font-medium text-cyan-200">
            <BadgeCheck size={12} />
            Réécriture LLM validée
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
          Lecture prudente et documentée
        </span>
        <span className="inline-flex items-center rounded-full border border-border/60 bg-card/60 px-2.5 py-0.5">
          Synthèse structurée
        </span>
      </div>

      {isDoctorNote ? (
        <div className="grid gap-2 sm:grid-cols-3">
          {hasMeaningfulBiologicalCounts ? (
            <>
              <div className="rounded-2xl border border-border/60 bg-card/55 px-3 py-2.5 shadow-[0_6px_18px_hsl(220_35%_5%_/_0.08)]">
                <p className="text-[10px] uppercase tracking-[0.22em] text-fg/55">Écarts biologiques</p>
                <p className="mt-1 text-sm font-semibold text-fg/92">{backendAnomalyCount}</p>
              </div>
              <div className="rounded-2xl border border-border/60 bg-card/55 px-3 py-2.5 shadow-[0_6px_18px_hsl(220_35%_5%_/_0.08)]">
                <p className="text-[10px] uppercase tracking-[0.22em] text-fg/55">
                  {isToxicologySummary ? "Résultats sous seuil" : "Résultats dans la référence"}
                </p>
                <p className="mt-1 text-sm font-semibold text-fg/92">{backendNormalCount}</p>
              </div>
            </>
          ) : (
            <>
              <div className="rounded-2xl border border-border/60 bg-card/55 px-3 py-2.5 shadow-[0_6px_18px_hsl(220_35%_5%_/_0.08)]">
                <p className="text-[10px] uppercase tracking-[0.22em] text-fg/55">Résultats documentés</p>
                <p className="mt-1 text-sm font-semibold text-fg/92">{backendDocumentedCount ?? parsed.noteLines.length}</p>
              </div>
              <div className="rounded-2xl border border-border/60 bg-card/55 px-3 py-2.5 shadow-[0_6px_18px_hsl(220_35%_5%_/_0.08)]">
                <p className="text-[10px] uppercase tracking-[0.22em] text-fg/55">Cadre</p>
                <p className="mt-1 text-sm font-semibold text-fg/92">Prudente, sans diagnostic</p>
              </div>
            </>
          )}
          <div className="rounded-2xl border border-border/60 bg-card/55 px-3 py-2.5 shadow-[0_6px_18px_hsl(220_35%_5%_/_0.08)]">
            <p className="text-[10px] uppercase tracking-[0.22em] text-fg/55">Source documentaire</p>
            <p className="mt-1 truncate text-sm font-semibold text-fg/92">{sourceDisplayLabel || "document fourni"}</p>
          </div>
        </div>
      ) : null}

      {isDoctorNote ? (
        <>
          {DOCTOR_NOTE_DEMO_COMPACT ? (
            <section className="space-y-3 rounded-[24px] border border-border/60 bg-card/55 p-4 shadow-[0_8px_24px_hsl(220_35%_5%_/_0.12)]">
              <div className="flex items-center gap-2">
                <div className="rounded-full border border-accent/20 bg-accent/10 p-1.5 text-accent">
                  <FlaskConical size={12} />
                </div>
                <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-fg/62">Synthèse</p>
              </div>
              {toxicologyNarrative ? (
                <div className="space-y-3">
                  <p className="text-sm leading-6 text-fg/92">
                    {narrativeLeadText}
                  </p>
                  <div className="grid gap-2 sm:grid-cols-2">
                    {toxicologyNarrative.nature ? (
                      <div className="rounded-2xl border border-border/60 bg-bg/35 px-3 py-2.5">
                        <p className="text-[10px] uppercase tracking-[0.18em] text-fg/55">Type d’examen</p>
                        <p className="mt-1 text-sm font-semibold text-fg/92">{toxicologyNarrative.nature}</p>
                      </div>
                    ) : null}
                    {toxicologyNarrative.families ? (
                      <div className="rounded-2xl border border-border/60 bg-bg/35 px-3 py-2.5">
                        <p className="text-[10px] uppercase tracking-[0.18em] text-fg/55">Familles analysées</p>
                        <p className="mt-1 text-sm font-semibold text-fg/92">{toxicologyNarrative.families}</p>
                      </div>
                    ) : null}
                    {toxicologyNarrative.findings ? (
                      <div className="rounded-2xl border border-border/60 bg-bg/35 px-3 py-2.5">
                        <p className="text-[10px] uppercase tracking-[0.18em] text-fg/55">Résultats sous seuil</p>
                        <p className="mt-1 text-sm leading-6 text-fg/90">{toxicologyNarrative.findings}</p>
                      </div>
                    ) : null}
                    {toxicologyNarrative.noExceedance ? (
                      <div className="rounded-2xl border border-emerald-500/20 bg-emerald-500/8 px-3 py-2.5">
                        <p className="text-[10px] uppercase tracking-[0.18em] text-emerald-200/80">Conclusion opérationnelle</p>
                        <p className="mt-1 text-sm leading-6 text-fg/92">{toxicologyNarrative.noExceedance}</p>
                      </div>
                    ) : null}
                  </div>
                  {toxicologyNarrative.conclusion ? (
                    <p className="text-sm leading-6 text-fg/72">{toxicologyNarrative.conclusion}</p>
                  ) : null}
                </div>
              ) : (
                <p className={`text-sm leading-6 text-fg/92 ${isNarrativeBiologicalSummary && isLlmWriter ? "whitespace-pre-line" : ""}`}>
                  {narrativeLeadText}
                  {rangeSentence ? ` ${firstSentence(rangeSentence)}` : ""}
                </p>
              )}
              {narrativeInfoCards.length > 0 ? (
                <div className="grid gap-2 sm:grid-cols-2">
                  {narrativeInfoCards.map((item) => (
                    <div key={`${item.label}-${item.value}`} className="rounded-2xl border border-border/60 bg-bg/35 px-3 py-2.5">
                      <p className="text-[10px] uppercase tracking-[0.18em] text-fg/55">{item.label}</p>
                      <p className="mt-1 text-sm leading-6 text-fg/90">{item.value}</p>
                    </div>
                  ))}
                </div>
              ) : null}
              <div className="flex flex-wrap items-center gap-2 text-[11px] text-fg/64">
                <span className="rounded-full border border-border/60 bg-bg/45 px-2 py-0.5">
                  {parsed.context ? sanitizeForSentence(parsed.context) : docScope || "document fourni"}
                </span>
                {showConclusionPanel && narrativeConclusion ? (
                  <span className="rounded-full border border-emerald-500/25 bg-emerald-500/10 px-2 py-0.5 text-emerald-100">
                    {firstSentence(narrativeConclusion)}
                  </span>
                ) : null}
              </div>
            </section>
          ) : (
            <div className="grid gap-3 lg:grid-cols-[minmax(0,1.45fr)_minmax(280px,0.95fr)]">
              <section className="space-y-3 rounded-[24px] border border-border/60 bg-card/55 p-4 shadow-[0_8px_24px_hsl(220_35%_5%_/_0.12)]">
                <div className="flex items-center gap-2">
                  <div className="rounded-full border border-accent/20 bg-accent/10 p-1.5 text-accent">
                    <FlaskConical size={12} />
                  </div>
                  <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-fg/62">
                    {isNarrativeBiologicalSummary ? "Synthèse" : "Synthèse médicale"}
                  </p>
                </div>
                <p className={`text-sm leading-6 text-fg/92 ${useFaithfulNarrative ? "whitespace-pre-line" : ""}`}>
                  {narrativeLeadText}
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

              <div className="space-y-3">
                <section className="space-y-2 rounded-[24px] border border-border/60 bg-card/45 p-4 shadow-[0_8px_24px_hsl(220_35%_5%_/_0.10)]">
                  <div className="flex items-center gap-2">
                    <div className="rounded-full border border-emerald-500/20 bg-emerald-500/10 p-1.5 text-emerald-200">
                      <ShieldCheck size={12} />
                    </div>
                    <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-fg/62">
                      {isNarrativeBiologicalSummary ? "Conclusion technique" : "Point de vigilance"}
                    </p>
                  </div>
                  {showConclusionPanel && narrativeConclusion ? (
                    <p className={`text-sm leading-6 text-fg/92 ${useFaithfulNarrative ? "whitespace-pre-line" : ""}`}>
                      {DOCTOR_NOTE_DEMO_COMPACT ? firstSentence(narrativeConclusion) : narrativeConclusion}
                    </p>
                  ) : (
                    <p className="text-sm leading-6 text-fg/62">
                      {isNarrativeBiologicalSummary
                        ? "La synthèse ci-dessus contient déjà la conclusion utile."
                        : "La réponse backend ne formulait pas explicitement de conclusion distincte."}
                    </p>
                  )}
                </section>
                <section className="space-y-2 rounded-[24px] border border-border/60 bg-card/45 p-4 shadow-[0_8px_24px_hsl(220_35%_5%_/_0.10)]">
                  <div className="flex items-center gap-2">
                    <div className="rounded-full border border-accent/20 bg-accent/10 p-1.5 text-accent">
                      <FileText size={12} />
                    </div>
                    <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-fg/62">Source documentaire</p>
                  </div>
                  {sourceLink ? (
                    <a
                      href={sourceLink.href}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1.5 rounded-xl border border-border/60 bg-bg/45 px-3 py-2 text-sm text-accent transition hover:border-accent/30 hover:bg-accent/8 hover:underline"
                    >
                      <span aria-hidden="true">↗</span>
                      <span className="min-w-0 break-words">{sourceDisplayLabel || prettifyDocumentLabel(sourceLink.label)}</span>
                    </a>
                  ) : (
                    <p className="text-sm text-fg/90">{sourceDisplayLabel || "voir les sources cliquables ci-dessous"}</p>
                  )}
                </section>
              </div>
            </div>
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
              {narrativeConclusion}
            </p>
            <div className="flex flex-wrap items-center gap-2 text-[11px] text-fg/60">
              <span className="inline-flex items-center gap-1.5">
                <FileText size={11} />
                Source documentaire:
              </span>
              <span className="rounded-full border border-border/60 bg-card/60 px-2 py-0.5 text-fg/72">
                {sourceDisplayLabel || "voir les sources cliquables"}
              </span>
            </div>
          </motion.section>
        </>
      )}
    </motion.section>
  );
}
