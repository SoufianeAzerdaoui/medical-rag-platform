"use client";

import { AlertTriangle, BadgeCheck, FileText, HelpCircle, ShieldCheck } from "lucide-react";
import { AssistantMarkdown } from "@/components/chat/assistant-markdown";
import type { AssistantDiagnostics, ChatSource } from "@/types/chat";

type ResultTone = "normal" | "high" | "low" | "critical" | "unknown";

type MedicalResult = {
  analyte: string;
  value: string;
  unit?: string;
  reference: string;
  status: string;
  source: string;
  tone: ResultTone;
};

type MarkdownTable = {
  headers: string[];
  rows: string[][];
  raw: string;
};

type SourceSummary = {
  label: string;
  href?: string;
};

type EvolutionItem = {
  analyte: string;
  previous: number;
  current: number;
  previousRaw: string;
  currentRaw: string;
  variationPct: number | null;
};

type BackendCounts = {
  above: number | null;
  below: number | null;
  within: number | null;
  major: number | null;
};

type NarrativeResultBlock = {
  startIndex: number;
  endIndex: number;
  result: MedicalResult;
};

type CompactResultGroup = {
  startIndex: number;
  endIndex: number;
  heading: string;
  results: MedicalResult[];
};

const SECTION_CHROME =
  "relative overflow-hidden rounded-[24px] border border-border/70 bg-[radial-gradient(circle_at_top_right,rgba(14,165,233,0.08),transparent_28%),linear-gradient(180deg,rgba(255,255,255,0.035),rgba(255,255,255,0.015))] p-4 shadow-[0_18px_50px_hsl(220_35%_5%_/_0.12)]";

function normalize(value: string): string {
  return value
    .replace(/\*\*/g, "")
    .replace(/<br\s*\/?>/gi, " ")
    .replace(/\[(.*?)\]\((.*?)\)/g, "$1")
    .replace(/\s+/g, " ")
    .trim();
}

function normalizeKey(value: string): string {
  return normalize(value)
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "");
}

function splitTableRow(line: string): string[] {
  return line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map(normalize);
}

function parseMarkdownTables(content: string): MarkdownTable[] {
  const lines = content.split("\n");
  const tables: MarkdownTable[] = [];

  for (let i = 0; i < lines.length - 1; i += 1) {
    const headerLine = lines[i];
    const separatorLine = lines[i + 1];
    if (!/^\s*\|.*\|\s*$/.test(headerLine) || !/^\s*\|?[\s:|-]+\|[\s:|-]*$/.test(separatorLine)) {
      continue;
    }

    const rawLines = [headerLine, separatorLine];
    const rows: string[][] = [];
    let cursor = i + 2;
    while (cursor < lines.length && /^\s*\|.*\|\s*$/.test(lines[cursor])) {
      rawLines.push(lines[cursor]);
      rows.push(splitTableRow(lines[cursor]));
      cursor += 1;
    }

    tables.push({
      headers: splitTableRow(headerLine),
      rows,
      raw: rawLines.join("\n"),
    });
    i = cursor - 1;
  }

  return tables;
}

function getCell(headers: string[], row: string[], candidates: string[]): string {
  const normalizedCandidates = candidates.map(normalizeKey);
  const index = headers.findIndex((header) => {
    const key = normalizeKey(header);
    return normalizedCandidates.some((candidate) => key.includes(candidate));
  });
  if (index < 0) return "";
  return normalize(row[index] || "");
}

function toneFromStatus(status: string, value: string): ResultTone {
  const text = normalizeKey(`${status} ${value}`);
  if (text.includes("critique") || text.includes("urgent") || text.includes("alerte")) return "critical";
  if (text.includes("au-dessus") || text.includes("au dessus") || text.includes("haut") || text.includes("eleve") || text.includes("high")) {
    return "high";
  }
  if (text.includes("en dessous") || text.includes("en-dessous") || text.includes("bas") || text.includes("faible") || text.includes("low")) {
    return "low";
  }
  if (text.includes("normal") || text.includes("dans la reference") || text.includes("reference") || text.includes("within")) return "normal";
  if (text.includes("non trouve") || text.includes("non disponible") || text.includes("a verifier") || text.includes("indetermine")) return "unknown";
  return "unknown";
}

function splitValueAndUnit(value: string): { value: string; unit: string } {
  const normalized = normalize(value);
  const match = normalized.match(/^([+-]?\d+(?:[.,]\d+)?)(?:\s+(.+))?$/);
  if (!match) {
    return { value: normalized, unit: "" };
  }
  return {
    value: match[1].replace(",", "."),
    unit: normalize(match[2] || ""),
  };
}

function resultFromTable(table: MarkdownTable): MedicalResult[] {
  const headers = table.headers;
  const hasMedicalShape = headers.some((h) => /analyse|analyte|param|marqueur|resultat|résultat|test|examen/i.test(h)) &&
    headers.some((h) => /valeur|value|resultat|résultat/i.test(h));
  if (!hasMedicalShape) return [];

  return table.rows
    .map((row) => {
      const analyte = getCell(headers, row, ["analyse", "analyte", "parametre", "marqueur", "test", "examen", "resultat"]);
      const value = getCell(headers, row, ["valeur", "value", "resultat"]);
      const reference = getCell(headers, row, ["reference", "intervalle", "norme", "plage"]);
      const status = getCell(headers, row, ["statut", "status", "interpretation", "etat"]);
      const source = getCell(headers, row, ["source", "document", "rapport", "doc"]);
      if (!analyte && !value) return null;
      const split = splitValueAndUnit(value || "Non trouvé");
      return {
        analyte: analyte || "Paramètre",
        value: split.value || "Non trouvé",
        unit: split.unit || undefined,
        reference: reference || "Non disponible",
        status: status || "À vérifier",
        source: source || "Voir sources",
        tone: toneFromStatus(status, value),
      } as MedicalResult;
    })
    .filter((item): item is MedicalResult => Boolean(item));
}

function removeTables(content: string, tables: MarkdownTable[]): string {
  let output = content;
  for (const table of tables) {
    output = output.replace(table.raw, "");
  }
  return output.replace(/\n{3,}/g, "\n\n").trim();
}

function toneBadge(tone: ResultTone): string {
  if (tone === "normal") return "status-success";
  if (tone === "high") return "status-warning";
  if (tone === "low") return "status-low";
  if (tone === "critical") return "status-danger";
  return "status-neutral";
}

function statusLabel(tone: ResultTone, fallback: string): string {
  if (tone === "normal") return "Normal";
  if (tone === "high") return "Haut";
  if (tone === "low") return "Bas";
  if (tone === "critical") return "À vérifier";
  return fallback || "Non trouvé";
}

function sourceSummary(source: ChatSource): SourceSummary | null {
  if (typeof source === "string") {
    const label = source.match(/(?:doc_id=|document=)?([^,\]\n]+)/i)?.[1]?.trim() || source.trim();
    return label ? { label } : null;
  }
  const record = source as Record<string, unknown>;
  const label = String(record.filename || record.label || record.documentName || record.doc_id || record.documentId || "Source").trim();
  const href = String(record.viewer_url || record.url || "").trim();
  const page = record.page ? `, page ${record.page}` : "";
  return { label: `${label}${page}`, href: href || undefined };
}

function parseNumeric(value: string): number | null {
  const cleaned = normalize(value).replace(",", ".").match(/-?\d+(?:\.\d+)?/);
  if (!cleaned) return null;
  const parsed = Number(cleaned[0]);
  return Number.isFinite(parsed) ? parsed : null;
}

function parseEvolutionFromTable(table: MarkdownTable): EvolutionItem[] {
  const headers = table.headers.map((h) => normalizeKey(h));
  const analyteIdx = headers.findIndex((h) => /analyse|analyte|param|marqueur|test|examen/.test(h));
  const oldIdx = headers.findIndex((h) => /ancien|precedent|précédent|old|anterieur|antérieur/.test(h));
  const currentIdx = headers.findIndex((h) => /actuel|current|nouveau|recent|récent/.test(h));
  if (analyteIdx < 0 || oldIdx < 0 || currentIdx < 0) return [];

  return table.rows
    .map((row) => {
      const analyte = normalize(row[analyteIdx] || "");
      const previousRaw = normalize(row[oldIdx] || "");
      const currentRaw = normalize(row[currentIdx] || "");
      const previous = parseNumeric(previousRaw);
      const current = parseNumeric(currentRaw);
      if (!analyte || previous === null || current === null) return null;
      const variationPct = previous !== 0 ? ((current - previous) / Math.abs(previous)) * 100 : null;
      return { analyte, previous, current, previousRaw, currentRaw, variationPct };
    })
    .filter((item): item is EvolutionItem => Boolean(item));
}

function parseEvolutionFromText(content: string): EvolutionItem[] {
  const lines = content.split("\n").map((line) => line.trim()).filter(Boolean);
  const out: EvolutionItem[] = [];
  for (let i = 0; i < lines.length; i += 1) {
    const analyteLine = lines[i].replace(/^[-*]\s*/, "");
    const oldLine = lines[i + 1] || "";
    const currentLine = lines[i + 2] || "";
    if (!/^ancien\s*:/i.test(oldLine) || !/^actuel\s*:/i.test(currentLine)) continue;
    const analyte = normalize(analyteLine.replace(/:$/, ""));
    const previousRaw = normalize(oldLine.split(":").slice(1).join(":"));
    const currentRaw = normalize(currentLine.split(":").slice(1).join(":"));
    const previous = parseNumeric(previousRaw);
    const current = parseNumeric(currentRaw);
    if (!analyte || previous === null || current === null) continue;
    const variationPct = previous !== 0 ? ((current - previous) / Math.abs(previous)) * 100 : null;
    out.push({ analyte, previous, current, previousRaw, currentRaw, variationPct });
  }
  return out;
}

function variationLabel(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return "n/a";
  const rounded = Math.round(value);
  return `${rounded >= 0 ? "+" : ""}${rounded}%`;
}

function dedupeBulletLines(content: string): string {
  const seen = new Set<string>();
  const lines = content.split("\n");
  const filtered = lines.filter((line) => {
    const match = line.match(/^\s*[-*]\s+(.+)/);
    if (!match?.[1]) return true;
    const key = normalizeKey(match[1]);
    if (!key) return true;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
  return filtered.join("\n").replace(/\n{3,}/g, "\n\n").trim();
}

function toDisplayCount(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return Math.max(0, Math.round(value));
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return Math.max(0, Math.round(parsed));
  }
  return null;
}

function getBackendCounts(source: Record<string, unknown>): BackendCounts {
  return {
    above: toDisplayCount(source.above_reference_count),
    below: toDisplayCount(source.below_reference_count),
    within: toDisplayCount(source.within_reference_count),
    major: toDisplayCount(source.major_anomalies_count),
  };
}

function isNarrativeResultTitle(line: string): boolean {
  const normalized = normalize(line);
  if (!normalized) return false;
  if (normalized.includes(":")) return false;
  if (/^[#>*\-•]/.test(normalized)) return false;
  if (
    /^(synthese|synthèse|conclusion|source|sources|niveau de support documentaire|qualite de synthese|qualité de synthèse|ecarts documentes|écarts documentés|elements a lire|éléments à lire|resultats dans la reference|résultats dans la référence|cadre|support documentaire|lecture)\b/i.test(
      normalized,
    )
  ) {
    return false;
  }
  return /[a-zàâçéèêëîïôùûüÿœ]/i.test(normalized);
}

function readNarrativeLabelValue(lines: string[], labels: string[]): string {
  const normalizedLabels = labels.map((label) => normalizeKey(label));
  for (const line of lines) {
    const plain = normalize(line).replace(/^\s*[-*•]\s*/, "");
    const key = normalizeKey(plain);
    for (const label of normalizedLabels) {
      if (key.startsWith(label)) {
        return plain.split(/[:：]/).slice(1).join(":").trim();
      }
    }
  }
  return "";
}

function parseNarrativeResults(content: string): NarrativeResultBlock[] {
  const rawLines = String(content || "").split(/\r?\n/);
  const blocks: NarrativeResultBlock[] = [];

  let i = 0;
  while (i < rawLines.length) {
    const current = normalize(rawLines[i]);
    if (!isNarrativeResultTitle(current)) {
      i += 1;
      continue;
    }

    const blockLines: string[] = [rawLines[i]];
    let j = i + 1;
    let sawLabel = false;
    while (j < rawLines.length) {
      const line = rawLines[j];
      const normalized = normalize(line);
      if (!normalized) {
        if (sawLabel) {
          j += 1;
          break;
        }
        j += 1;
        continue;
      }
      if (
        sawLabel &&
        (isNarrativeResultTitle(normalized) ||
          /^(conclusion technique|source documentaire|niveau de support documentaire|qualite de synthese|qualité de synthèse|sources cliquables)\b/i.test(
            normalized,
          ))
      ) {
        break;
      }
      blockLines.push(line);
      if (/^(?:[-*•]\s*)?(?:\*\*)?(valeur|référence(?: disponible)?|reference(?: disponible)?|statut technique|statut interprétatif|source)(?:\*\*)?\s*[:：]/i.test(normalized)) {
        sawLabel = true;
      }
      j += 1;
    }

    const value = readNarrativeLabelValue(blockLines, ["Valeur"]);
    const splitValue = splitValueAndUnit(value || "");
    const reference = readNarrativeLabelValue(blockLines, ["Référence disponible", "Référence", "Reference", "Intervalle de référence", "Plage de référence"]);
    const status = readNarrativeLabelValue(blockLines, ["Statut technique", "Statut interprétatif"]);
    const source = readNarrativeLabelValue(blockLines, ["Source"]);
    if (!splitValue.value || !source) {
      i += 1;
      continue;
    }

    const title = normalize(blockLines[0]).replace(/^###\s*/, "");
    blocks.push({
      startIndex: i,
      endIndex: j,
      result: {
        analyte: title || "Paramètre",
        value: splitValue.value,
        unit: splitValue.unit || undefined,
        reference: reference || "Non disponible",
        status: status || "À vérifier",
        source: source || "Voir sources",
        tone: toneFromStatus(status, splitValue.value),
      } as MedicalResult,
    });
    i = j;
  }

  return blocks;
}

function parseCompactResultLine(line: string): MedicalResult | null {
  const normalized = normalize(line).replace(/^[-*•]\s*/, "");
  if (!normalized) return null;
  if (/^résultats?\s+\d+\s+à\s+\d+$/i.test(normalized)) return null;
  if (/^conclusion technique\b/i.test(normalized) || /^source(s)?\b/i.test(normalized)) return null;

  const segments = normalized
    .split("|")
    .map((segment) => normalize(segment))
    .filter(Boolean);
  if (segments.length < 2) return null;

  const head = segments[0] || "";
  const headMatch = head.match(/^(.+?)\s*:\s*(.+)$/);
  if (!headMatch) return null;

  const analyte = normalize(headMatch[1]);
  const valueAndUnit = splitValueAndUnit(headMatch[2]);
  const fields = new Map<string, string>();
  for (const segment of segments.slice(1)) {
    const match = segment.match(/^([^:：]+)\s*[:：]\s*(.+)$/);
    if (!match?.[1] || !match[2]) continue;
    fields.set(normalizeKey(match[1]), normalize(match[2]));
  }

  const unit = fields.get(normalizeKey("unité")) || fields.get(normalizeKey("unite")) || valueAndUnit.unit;
  const reference = fields.get(normalizeKey("référence")) || fields.get(normalizeKey("reference")) || "Non disponible";
  const status = fields.get(normalizeKey("statut")) || fields.get(normalizeKey("statut technique")) || fields.get(normalizeKey("statut interprétatif")) || "À vérifier";
  return {
    analyte: analyte || "Paramètre",
    value: valueAndUnit.value || "Non trouvé",
    unit: unit || undefined,
    reference,
    status,
    source: "",
    tone: toneFromStatus(status, valueAndUnit.value),
  };
}

function parseCompactResultGroups(content: string): CompactResultGroup[] {
  const rawLines = String(content || "").split(/\r?\n/);
  const groups: CompactResultGroup[] = [];

  let i = 0;
  while (i < rawLines.length) {
    const heading = normalize(rawLines[i]);
    const headingMatch = heading.match(/^résultats\s+(\d+)\s+à\s+(\d+)$/i);
    if (!headingMatch) {
      i += 1;
      continue;
    }

    const startIndex = i;
    const results: MedicalResult[] = [];
    let j = i + 1;
    while (j < rawLines.length) {
      const current = normalize(rawLines[j]);
      if (!current) {
        j += 1;
        continue;
      }
      if (
        /^résultats\s+\d+\s+à\s+\d+$/i.test(current) ||
        /^conclusion technique\b/i.test(current) ||
        /^source(s)?\b/i.test(current) ||
        /^écarts documentés\b/i.test(current) ||
        /^elements à lire avec prudence\b/i.test(current) ||
        /^éléments à lire avec prudence\b/i.test(current)
      ) {
        break;
      }

      if (/^[-*•]\s+/.test(current)) {
        const parsed = parseCompactResultLine(current);
        if (parsed) results.push(parsed);
      }
      j += 1;
    }

    if (results.length > 0) {
      groups.push({
        startIndex,
        endIndex: j,
        heading,
        results,
      });
    }
    i = j;
  }

  return groups;
}

function removeLineRanges(content: string, ranges: Array<{ startIndex: number; endIndex: number }>): string {
  if (ranges.length === 0) return content;
  const lines = String(content || "").split(/\r?\n/);
  const toRemove = new Set<number>();
  for (const range of ranges) {
    for (let i = range.startIndex; i < range.endIndex; i += 1) {
      toRemove.add(i);
    }
  }
  return lines.filter((_, index) => !toRemove.has(index)).join("\n").replace(/\n{3,}/g, "\n\n").trim();
}

function MedicalResultCard({ result, compact = false, showSource = true }: { result: MedicalResult; compact?: boolean; showSource?: boolean }) {
  return (
    <article className={compact
      ? "group relative overflow-hidden rounded-[22px] border border-border/70 bg-[radial-gradient(circle_at_top_right,rgba(14,165,233,0.08),transparent_30%),linear-gradient(180deg,rgba(255,255,255,0.035),rgba(255,255,255,0.018))] p-3.5 shadow-[0_12px_34px_hsl(220_35%_5%_/_0.10)] transition hover:-translate-y-0.5 hover:border-accent/30"
      : "group relative overflow-hidden rounded-[22px] border border-border/70 bg-[radial-gradient(circle_at_top_right,rgba(14,165,233,0.09),transparent_30%),linear-gradient(180deg,rgba(255,255,255,0.04),rgba(255,255,255,0.02))] p-4 shadow-[0_14px_40px_hsl(220_35%_5%_/_0.12)] transition hover:-translate-y-0.5 hover:border-accent/30"
    }>
      <div className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-accent/35 to-transparent" />
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 space-y-1">
          <h4 className={compact ? "truncate text-[13px] font-semibold tracking-[0.01em] text-fg/96" : "truncate text-sm font-semibold tracking-[0.01em] text-fg/96"}>{result.analyte}</h4>
          <p className="text-[10px] uppercase tracking-[0.18em] text-fg/55">Synthèse médicale structurée</p>
        </div>
        <span className={`shrink-0 rounded-full border px-2.5 py-1 text-[11px] font-semibold shadow-sm ${toneBadge(result.tone)}`}>
          {statusLabel(result.tone, result.status)}
        </span>
      </div>

      <div className={compact ? "mt-3 rounded-[18px] border border-border/60 bg-card/55 p-3" : "mt-4 rounded-[18px] border border-border/60 bg-card/55 p-4"}>
        <p className="text-[10px] uppercase tracking-[0.2em] text-fg/55">Valeur documentée</p>
        <div className="mt-2 flex flex-wrap items-end gap-2">
          <p className={compact ? "text-2xl font-semibold tracking-tight text-fg/96" : "text-3xl font-semibold tracking-tight text-fg/96"}>{result.value}</p>
          {result.unit ? (
            <span className="mb-0.5 rounded-full border border-border/60 bg-bg/35 px-2.5 py-1 text-xs font-medium text-fg/76">
              {result.unit}
            </span>
          ) : null}
        </div>
      </div>

      <dl className={compact ? "mt-3 grid gap-2 text-xs sm:grid-cols-2" : "mt-4 grid gap-3 text-sm sm:grid-cols-2"}>
        <div className="rounded-2xl border border-border/60 bg-card/45 px-3 py-2.5">
          <dt className="text-[10px] uppercase tracking-[0.18em] text-fg/55">Référence</dt>
          <dd className="mt-1 leading-6 text-fg/85">{result.reference}</dd>
        </div>
        {showSource ? (
          <div className="rounded-2xl border border-border/60 bg-card/45 px-3 py-2.5">
            <dt className="text-[10px] uppercase tracking-[0.18em] text-fg/55">Source documentaire</dt>
            <dd className="mt-1 break-words leading-6 text-fg/85">{result.source}</dd>
          </div>
        ) : null}
      </dl>
    </article>
  );
}

export function MedicalAnswerBlocks({ content, sources = [], diagnostics }: { content: string; sources?: ChatSource[]; diagnostics?: AssistantDiagnostics }) {
  const tables = parseMarkdownTables(content);
  const tableResults = tables.flatMap(resultFromTable);
  const compactResultGroups = tableResults.length > 0 ? [] : parseCompactResultGroups(content);
  const compactResultBlocks = compactResultGroups.flatMap((group) => group.results);
  const narrativeResultBlocks = tableResults.length > 0 || compactResultGroups.length > 0 ? [] : parseNarrativeResults(content);
  const results = tableResults.length > 0 ? tableResults : compactResultGroups.length > 0 ? compactResultBlocks : narrativeResultBlocks.map((block) => block.result);
  const evolutionFromTables = tables.flatMap(parseEvolutionFromTable);
  const evolutionFromText = evolutionFromTables.length === 0 ? parseEvolutionFromText(content) : [];
  const evolutionItems = [...evolutionFromTables, ...evolutionFromText];
  const contentWithoutStructuredResults = tableResults.length > 0
    ? removeTables(content, tables)
    : compactResultGroups.length > 0
      ? removeLineRanges(content, compactResultGroups)
      : removeLineRanges(content, narrativeResultBlocks);
  const sanitizedSummaryContent = dedupeBulletLines(contentWithoutStructuredResults);
  const abnormalResults = results.filter((result) => result.tone === "high" || result.tone === "low" || result.tone === "critical");
  const missingResults = results.filter((result) => result.tone === "unknown");
  const sourceItems = sources.map(sourceSummary).filter((item): item is SourceSummary => Boolean(item)).slice(0, 5);
  const backendCounts = getBackendCounts((diagnostics || {}) as Record<string, unknown>);
  const backendAbnormalCount = backendCounts.major ?? ((backendCounts.above ?? 0) + (backendCounts.below ?? 0));
  const backendWithinCount = backendCounts.within ?? results.filter((result) => result.tone === "normal").length;
  const backendResultCount =
    toDisplayCount(diagnostics?.displayed_evidences_count) ??
    toDisplayCount(diagnostics?.lab_result_count) ??
    toDisplayCount(diagnostics?.structured_values_count) ??
    toDisplayCount(diagnostics?.evidence_pack_count);
  const resultCount = results.length > 0 ? results.length : (backendResultCount ?? 0);
  const supportDocumentaryLabel = (() => {
    if (backendAbnormalCount > 0 && backendWithinCount > 0) {
      return `${backendAbnormalCount} hors référence / ${backendWithinCount} dans la référence`;
    }
    if (backendAbnormalCount > 0) {
      return `${backendAbnormalCount} hors référence`;
    }
    if (backendWithinCount > 0) {
      return `${backendWithinCount} dans la référence`;
    }
    if (resultCount > 0) {
      return `${resultCount} résultat${resultCount > 1 ? "s" : ""}`;
    }
    return "Prêt";
  })();
  const resultSummaryLabel = (() => {
    if (backendAbnormalCount > 0 && backendWithinCount > 0) {
      return `${backendAbnormalCount} écart${backendAbnormalCount > 1 ? "s" : ""} et ${backendWithinCount} résultat${backendWithinCount > 1 ? "s" : ""} normal${backendWithinCount > 1 ? "aux" : ""}`;
    }
    if (backendAbnormalCount > 0) {
      return `${backendAbnormalCount} écart${backendAbnormalCount > 1 ? "s" : ""} documenté${backendAbnormalCount > 1 ? "s" : ""}`;
    }
    if (backendWithinCount > 0) {
      return `${backendWithinCount} résultat${backendWithinCount > 1 ? "s" : ""} dans la référence`;
    }
    if (resultCount > 0) {
      return `${resultCount} résultat${resultCount > 1 ? "s" : ""} documenté${resultCount > 1 ? "s" : ""}`;
    }
    return "Lecture descriptive";
  })();

  return (
    <div className="space-y-4">
      <section className={`${SECTION_CHROME} p-5`}>
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <div className="rounded-full border border-accent/20 bg-accent/10 p-1.5 text-accent">
              <ShieldCheck size={14} />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-fg/96">Synthèse structurée</h3>
              <p className="text-xs text-fg/58">Lecture prudente et documentée.</p>
            </div>
          </div>
          <div className="flex flex-wrap gap-2 text-[11px] text-fg/60">
            <span className="rounded-full border border-border/60 bg-card/70 px-2.5 py-1">Résultats documentés: {resultCount}</span>
            <span className="rounded-full border border-border/60 bg-card/70 px-2.5 py-1">Variations: {evolutionItems.length}</span>
            <span className="rounded-full border border-border/60 bg-card/70 px-2.5 py-1">Sources: {sourceItems.length}</span>
          </div>
        </div>
        <div className="grid gap-3 sm:grid-cols-3">
          <div className="rounded-2xl border border-border/60 bg-card/50 px-3 py-2.5">
            <p className="text-[10px] uppercase tracking-[0.18em] text-fg/55">Synthèse</p>
            <p className="mt-1 text-sm font-semibold text-fg/92">{resultSummaryLabel}</p>
          </div>
          <div className="rounded-2xl border border-border/60 bg-card/50 px-3 py-2.5">
            <p className="text-[10px] uppercase tracking-[0.18em] text-fg/55">Cadre</p>
            <p className="mt-1 text-sm font-semibold text-fg/92">Prudente, sans diagnostic</p>
          </div>
          <div className="rounded-2xl border border-border/60 bg-card/50 px-3 py-2.5">
            <p className="text-[10px] uppercase tracking-[0.18em] text-fg/55">Support documentaire</p>
            <p className="mt-1 text-sm font-semibold text-fg/92">{supportDocumentaryLabel}</p>
          </div>
        </div>
      </section>

      <section className={SECTION_CHROME}>
        <div className="mb-4 flex items-center gap-2">
          <div className="rounded-full border border-accent/20 bg-accent/10 p-1.5 text-accent">
            <ShieldCheck size={14} />
          </div>
          <h3 className="text-sm font-semibold">Synthèse</h3>
        </div>
        {sanitizedSummaryContent ? (
          <AssistantMarkdown content={sanitizedSummaryContent} />
        ) : (
          <p className="text-sm leading-6 text-fg/78">Résumé basé uniquement sur les éléments disponibles dans les documents fournis.</p>
        )}
      </section>

      {compactResultGroups.length > 0 ? (
        <section className={SECTION_CHROME}>
          <div className="mb-4 flex items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <div className="rounded-full border border-accent/20 bg-accent/10 p-1.5 text-accent">
                <BadgeCheck size={14} />
              </div>
              <h3 className="text-sm font-semibold">Résultats documentés</h3>
            </div>
            <span className="rounded-full border border-border bg-card px-2.5 py-1 text-[11px] text-fg/58">
              {results.length}
            </span>
          </div>
          <div className="space-y-4">
            {compactResultGroups.map((group, groupIndex) => (
              <div key={`${group.heading}-${groupIndex}`} className="space-y-3">
                <div className="flex items-center justify-between gap-3">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-fg/60">{group.heading}</p>
                  <span className="rounded-full border border-border/60 bg-card/70 px-2.5 py-1 text-[11px] text-fg/60">
                    {group.results.length}
                  </span>
                </div>
                <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                  {group.results.map((result, index) => (
                    <MedicalResultCard key={`${result.analyte}-${result.value}-${groupIndex}-${index}`} result={result} compact showSource={false} />
                  ))}
                </div>
              </div>
            ))}
          </div>
        </section>
      ) : results.length > 0 ? (
        <section className={SECTION_CHROME}>
          <div className="mb-4 flex items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <div className="rounded-full border border-accent/20 bg-accent/10 p-1.5 text-accent">
                <BadgeCheck size={14} />
              </div>
              <h3 className="text-sm font-semibold">Résultats documentés</h3>
            </div>
            <span className="rounded-full border border-border bg-card px-2.5 py-1 text-[11px] text-fg/58">
              {results.length}
            </span>
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            {results.map((result, index) => (
              <MedicalResultCard key={`${result.analyte}-${result.value}-${index}`} result={result} />
            ))}
          </div>
        </section>
      ) : null}

      {evolutionItems.length > 0 ? (
        <section className={SECTION_CHROME}>
          <div className="mb-4 flex items-center gap-2">
            <div className="rounded-full border border-accent/20 bg-accent/10 p-1.5 text-accent">
              <BadgeCheck size={14} />
            </div>
            <h3 className="text-sm font-semibold">Évolution</h3>
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            {evolutionItems.map((item, index) => (
              <article key={`${item.analyte}-${item.previous}-${item.current}-${index}`} className="rounded-[22px] border border-border/70 bg-card/70 p-4 shadow-[0_12px_34px_hsl(220_35%_5%_/_0.10)]">
                <p className="text-sm font-semibold text-fg/96">{item.analyte}</p>
                <dl className="mt-3 space-y-1 text-xs text-fg/78">
                  <div className="flex justify-between gap-3 rounded-lg bg-bg/35 px-2.5 py-2"><dt>Ancien</dt><dd className="font-medium text-fg">{item.previousRaw}</dd></div>
                  <div className="flex justify-between gap-3 rounded-lg bg-bg/35 px-2.5 py-2"><dt>Actuel</dt><dd className="font-medium text-fg">{item.currentRaw}</dd></div>
                  <div className="flex justify-between gap-3 rounded-lg bg-accent/8 px-2.5 py-2"><dt>Variation</dt><dd className="font-semibold text-accent">{variationLabel(item.variationPct)}</dd></div>
                </dl>
                <p className="mt-3 rounded-xl border border-border/60 bg-fg/[0.03] px-3 py-2.5 text-center text-xs text-fg/85">
                  {item.previousRaw} ───────────────▶ {item.currentRaw}
                </p>
              </article>
            ))}
          </div>
        </section>
      ) : null}

      <section className={SECTION_CHROME}>
        <div className="mb-4 flex items-center gap-2">
          <div className="rounded-full border border-amber-500/20 bg-amber-500/10 p-1.5 text-amber-300">
            <AlertTriangle size={14} />
          </div>
          <h3 className="text-sm font-semibold">Écarts documentés</h3>
        </div>
        {abnormalResults.length > 0 ? (
          <div className="grid gap-2">
            {abnormalResults.map((result, index) => (
              <div key={`${result.analyte}-abnormal-${index}`} className={`rounded-2xl border px-3 py-2.5 text-sm shadow-sm ${toneBadge(result.tone)}`}>
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-semibold">{result.analyte}</span>
                  <span className="text-fg/40">·</span>
                  <span>
                    {result.value}
                    {result.unit ? ` ${result.unit}` : ""}
                  </span>
                  <span className="text-fg/40">·</span>
                  <span>{statusLabel(result.tone, result.status)}</span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm leading-6 text-fg/70">
            {resultCount > 0
              ? backendWithinCount > 0
                ? "Les résultats affichés restent dans la référence."
                : "Les résultats affichés ne comportent pas d’écart clairement objectivé."
              : "Aucune valeur hors référence clairement détectée dans la réponse affichée."}
          </p>
        )}
      </section>

      <section className={SECTION_CHROME}>
        <div className="mb-4 flex items-center gap-2">
          <div className="rounded-full border border-border/60 bg-card/70 p-1.5 text-fg/55">
            <HelpCircle size={14} />
          </div>
          <h3 className="text-sm font-semibold">Éléments à lire avec prudence</h3>
        </div>
        {missingResults.length > 0 ? (
          <div className="flex flex-wrap gap-2">
            {missingResults.map((result, index) => (
              <span key={`${result.analyte}-missing-${index}`} className="rounded-full border border-border bg-fg/[0.04] px-3 py-1 text-xs text-fg/68">
                {result.analyte}
              </span>
            ))}
          </div>
        ) : (
          <p className="text-sm leading-6 text-fg/70">Aucun élément manquant explicite détecté. Les résultats restent à confronter au contexte clinique.</p>
        )}
      </section>

      <section className={SECTION_CHROME}>
        <div className="mb-4 flex items-center gap-2">
          <div className="rounded-full border border-accent/20 bg-accent/10 p-1.5 text-accent">
            <FileText size={14} />
          </div>
          <h3 className="text-sm font-semibold">Sources documentaires</h3>
        </div>
        {sourceItems.length > 0 ? (
          <div className="flex flex-wrap gap-2">
            {sourceItems.map((source, index) =>
              source.href ? (
                <a
                  key={`${source.label}-${index}`}
                  href={source.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="rounded-xl border border-border/70 bg-card/70 px-3 py-1.5 text-xs font-medium text-accent underline-offset-2 transition hover:border-accent/30 hover:bg-accent/8 hover:underline"
                >
                  {source.label}
                </a>
              ) : (
                <span key={`${source.label}-${index}`} className="rounded-xl border border-border/70 bg-card/70 px-3 py-1.5 text-xs text-fg/72">
                  {source.label}
                </span>
              ),
            )}
          </div>
        ) : (
          <p className="text-sm leading-6 text-fg/70">Sources détaillées disponibles si elles sont renvoyées par le backend.</p>
        )}
      </section>
    </div>
  );
}
