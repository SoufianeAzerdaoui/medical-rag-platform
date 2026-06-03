import type { ChatSource, MessageItem } from "@/types/chat";

export type DocumentaryConfidenceLevel = "elevee" | "moyenne" | "faible";

export type DocumentaryMetrics = {
  sourceCount: number;
  ignoredCount: number;
  confidence: DocumentaryConfidenceLevel;
  extractedValues: number;
  extractedValuesLabel: string;
  missingElements: number;
  diagnosisProposed: "Oui" | "Non";
  fromBackendMetrics: boolean;
};

function toNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function pickDiagnosticNumber(diagnostics: Record<string, unknown>, keys: string[]): number | null {
  for (const key of keys) {
    const value = toNumber(diagnostics[key]);
    if (value !== null) return value;
  }
  return null;
}

function sourceKey(source: ChatSource, index: number): string {
  if (typeof source === "string") return `str:${source.trim().toLowerCase()}::${index}`;
  const raw = source as Record<string, unknown>;
  const doc = String(raw.doc_id || raw.documentId || raw.filename || raw.documentName || "").trim().toLowerCase();
  const page = String(raw.page || "").trim();
  const row = String(raw.row || "").trim();
  const rowEnd = String(raw.row_end || "").trim();
  const url = String(raw.viewer_url || raw.url || "").trim().toLowerCase();
  return `obj:${doc}::${page}::${row}::${rowEnd}::${url}`;
}

function uniqueSources(sources: ChatSource[]): ChatSource[] {
  const map = new Map<string, ChatSource>();
  for (let i = 0; i < sources.length; i += 1) {
    const source = sources[i];
    const key = sourceKey(source, i);
    if (!map.has(key)) map.set(key, source);
  }
  return Array.from(map.values());
}

function sourceScores(sources: ChatSource[]): number[] {
  const scores: number[] = [];
  for (const source of sources) {
    if (typeof source === "string") {
      const match = source.match(/(?:score|pertinence)\s*[:=]\s*([0-9.]+)/i)?.[1];
      const value = match ? Number(match) : NaN;
      if (Number.isFinite(value)) scores.push(value <= 1 ? value * 100 : value);
      continue;
    }
    const raw = source as Record<string, unknown>;
    const value = toNumber(raw.score);
    if (value !== null) scores.push(value <= 1 ? value * 100 : value);
  }
  return scores;
}

function estimateExtractedValues(content: string): number {
  const tableRows = content
    .split("\n")
    .filter((line) => /^\s*\|.*\|\s*$/.test(line) && !/^\s*\|?[\s:|-]+\|[\s:|-]*$/.test(line)).length;
  const listedValues = (content.match(/\b(?:TSH|T3|T4|Anti-TG|Hb|CRP|ASAT|ALAT|Leucocytes|Plaquettes)\b/gi) || []).length;
  return Math.max(0, Math.max(tableRows > 1 ? tableRows - 1 : 0, listedValues));
}

function inferConfidenceScore(message: MessageItem, uniqueMessageSources: ChatSource[], sourceCount: number): number {
  const diagnostics = (message.diagnostics || {}) as Record<string, unknown>;
  const scores = sourceScores(uniqueMessageSources);
  if (scores.length > 0) {
    return scores.reduce((sum, value) => sum + value, 0) / scores.length;
  }

  const sourceUx = toNumber((diagnostics.quality_report as Record<string, unknown> | undefined)?.source_ux_score);
  if (sourceUx !== null) return sourceUx <= 1 ? sourceUx * 100 : sourceUx;

  const safetyRaw = toNumber(
    diagnostics.safety_score ??
      (diagnostics.quality_report as Record<string, unknown> | undefined)?.safety_score,
  );
  if (safetyRaw !== null) return safetyRaw <= 1 ? safetyRaw * 100 : safetyRaw;

  if (sourceCount >= 3) return 82;
  if (sourceCount >= 1) return 66;
  return 40;
}

export function getLatestAssistantDoneMessage(messages: MessageItem[]): MessageItem | null {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const message = messages[i];
    if (message.role === "assistant" && message.status === "done") return message;
  }
  return null;
}

export function getMessageDocumentaryMetrics(message: MessageItem): DocumentaryMetrics {
  const content = String(message.content || "");
  const diagnostics = (message.diagnostics || {}) as Record<string, unknown>;
  const uniqueMessageSources = uniqueSources(Array.isArray(message.sources) ? message.sources : []);
  const sourceCountFromDiagnostics = pickDiagnosticNumber(diagnostics, [
    "sources_count",
    "displayed_evidences_count",
    "included_rows_count",
    "used_sources_count",
  ]);
  const sourceCount = Math.max(
    uniqueMessageSources.length,
    sourceCountFromDiagnostics !== null ? Math.max(0, Math.round(sourceCountFromDiagnostics)) : 0,
  );

  const hasDisplayedEvidences =
    diagnostics.displayed_evidences_count !== undefined ||
    diagnostics.evidence_pack_count !== undefined ||
    diagnostics.lab_result_count !== undefined ||
    diagnostics.value_numeric_count !== undefined ||
    diagnostics.included_rows_count !== undefined ||
    diagnostics.used_sources_count !== undefined;
  const hasMissingValues =
    diagnostics.missing_values_count !== undefined ||
    diagnostics.missing_elements_count !== undefined ||
    diagnostics.unresolved_items_count !== undefined;
  const hasSafetyScore =
    diagnostics.safety_score !== undefined ||
    (diagnostics.quality_report as Record<string, unknown> | undefined)?.safety_score !== undefined;

  const candidateCount = pickDiagnosticNumber(diagnostics, [
    "candidate_evidences_count",
    "candidate_rows_count",
    "retrieved_rows_count",
    "retrieved_evidences_count",
    "raw_evidences_count",
    "ranked_rows_count",
    "total_rows_count",
    "total_evidences_count",
    "considered_rows_count",
  ]);
  const ignoredCount = candidateCount !== null ? Math.max(0, Math.round(candidateCount) - sourceCount) : 0;

  const extractedFromDiagnostics = pickDiagnosticNumber(diagnostics, [
    "structured_values_count",
    "extracted_values_count",
    "extracted_items_count",
    "reported_values_count",
  ]);
  const structuredValueCount = pickDiagnosticNumber(diagnostics, [
    "value_numeric_count",
    "structured_values_count",
    "lab_result_count",
    "evidence_pack_count",
    "displayed_evidences_count",
    "included_rows_count",
  ]);
  const extractedValues = extractedFromDiagnostics !== null
    ? Math.max(0, Math.round(extractedFromDiagnostics))
    : structuredValueCount !== null
      ? Math.max(0, Math.round(structuredValueCount))
      : sourceCount > 0
        ? Math.max(1, Math.round(sourceCount))
      : estimateExtractedValues(content);
  const extractedValuesLabel =
    structuredValueCount !== null && structuredValueCount > 0
      ? "Valeurs structurées retrouvées"
      : "Valeurs extraites";

  const missingFromDiagnostics = toNumber(
    diagnostics.missing_values_count ??
      diagnostics.missing_elements_count ??
      diagnostics.unresolved_items_count,
  );
  const uncertainMentions = (content.match(/(non trouv|non disponible|à vérifier|a verifier|indétermin|indetermine)/gi) || []).length;
  const missingElements = Math.max(0, missingFromDiagnostics !== null ? Math.round(missingFromDiagnostics) : uncertainMentions);

  const diagnosisProposed: "Oui" | "Non" = /(diagnostic\s*:|diagnostic proposé|diagnostic propose|diagnostic retenu)/i.test(content)
    ? "Oui"
    : "Non";

  const confidenceScore = inferConfidenceScore(message, uniqueMessageSources, sourceCount);
  const confidence: DocumentaryConfidenceLevel = confidenceScore >= 78 ? "elevee" : confidenceScore >= 58 ? "moyenne" : "faible";

  return {
    sourceCount,
    ignoredCount,
    confidence,
    extractedValues,
    extractedValuesLabel,
    missingElements,
    diagnosisProposed,
    fromBackendMetrics: hasDisplayedEvidences && (hasMissingValues || hasSafetyScore || structuredValueCount !== null),
  };
}

export function getPreferredAssistantMessageForSources(messages: MessageItem[]): MessageItem | null {
  const assistantDone = messages.filter((message) => message.role === "assistant" && message.status === "done");
  if (assistantDone.length === 0) return null;
  for (let i = assistantDone.length - 1; i >= 0; i -= 1) {
    const message = assistantDone[i];
    const metrics = getMessageDocumentaryMetrics(message);
    if (metrics.sourceCount > 0) return message;
  }
  return assistantDone[assistantDone.length - 1];
}
