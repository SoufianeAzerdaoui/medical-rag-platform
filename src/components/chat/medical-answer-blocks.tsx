"use client";

import { AlertTriangle, BadgeCheck, FileText, HelpCircle, ShieldCheck } from "lucide-react";
import { AssistantMarkdown } from "@/components/chat/assistant-markdown";
import type { ChatSource } from "@/types/chat";

type ResultTone = "normal" | "high" | "low" | "critical" | "unknown";

type MedicalResult = {
  analyte: string;
  value: string;
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

const SECTION_CHROME = "rounded-xl border border-border/70 bg-card/[0.58] p-4 shadow-sm";

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
      return {
        analyte: analyte || "Paramètre",
        value: value || "Non trouvé",
        reference: reference || "Non disponible",
        status: status || "À vérifier",
        source: source || "Voir sources",
        tone: toneFromStatus(status, value),
      };
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

function MedicalResultCard({ result }: { result: MedicalResult }) {
  return (
    <article className="rounded-xl border border-border/70 bg-card/80 p-4 shadow-sm transition hover:-translate-y-0.5 hover:border-accent/30">
      <div className="flex items-start justify-between gap-3">
        <h4 className="min-w-0 text-sm font-semibold text-fg">{result.analyte}</h4>
        <span className={`shrink-0 rounded-full border px-2.5 py-1 text-[11px] font-semibold ${toneBadge(result.tone)}`}>
          {statusLabel(result.tone, result.status)}
        </span>
      </div>
      <p className="mt-3 text-2xl font-semibold tracking-tight text-fg">{result.value}</p>
      <dl className="mt-3 grid gap-2 text-xs">
        <div className="flex gap-2">
          <dt className="min-w-20 text-fg/55">Référence</dt>
          <dd className="text-fg/82">{result.reference}</dd>
        </div>
        <div className="flex gap-2">
          <dt className="min-w-20 text-fg/55">Source</dt>
          <dd className="break-words text-fg/82">{result.source}</dd>
        </div>
      </dl>
    </article>
  );
}

export function MedicalAnswerBlocks({ content, sources = [] }: { content: string; sources?: ChatSource[] }) {
  const tables = parseMarkdownTables(content);
  const results = tables.flatMap(resultFromTable);
  const evolutionFromTables = tables.flatMap(parseEvolutionFromTable);
  const evolutionFromText = evolutionFromTables.length === 0 ? parseEvolutionFromText(content) : [];
  const evolutionItems = [...evolutionFromTables, ...evolutionFromText];
  const contentWithoutTables = results.length > 0 ? removeTables(content, tables) : content;
  const abnormalResults = results.filter((result) => result.tone === "high" || result.tone === "low" || result.tone === "critical");
  const missingResults = results.filter((result) => result.tone === "unknown");
  const sourceItems = sources.map(sourceSummary).filter((item): item is SourceSummary => Boolean(item)).slice(0, 5);

  return (
    <div className="space-y-3">
      <section className={SECTION_CHROME}>
        <div className="mb-3 flex items-center gap-2">
          <ShieldCheck size={16} className="text-accent" />
          <h3 className="text-sm font-semibold">Résumé prudent</h3>
        </div>
        {contentWithoutTables ? (
          <AssistantMarkdown content={contentWithoutTables} />
        ) : (
          <p className="text-sm leading-6 text-fg/78">Résumé basé uniquement sur les éléments disponibles dans les documents fournis.</p>
        )}
      </section>

      {results.length > 0 ? (
        <section className={SECTION_CHROME}>
          <div className="mb-3 flex items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <BadgeCheck size={16} className="text-accent" />
              <h3 className="text-sm font-semibold">Résultats importants</h3>
            </div>
            <span className="rounded-full border border-border bg-card px-2.5 py-1 text-[11px] text-fg/58">{results.length}</span>
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
          <div className="mb-3 flex items-center gap-2">
            <BadgeCheck size={16} className="text-accent" />
            <h3 className="text-sm font-semibold">Analyse timeline</h3>
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            {evolutionItems.map((item, index) => (
              <article key={`${item.analyte}-${item.previous}-${item.current}-${index}`} className="rounded-xl border border-border/70 bg-card/80 p-4">
                <p className="text-sm font-semibold text-fg">{item.analyte}</p>
                <dl className="mt-2 space-y-1 text-xs text-fg/78">
                  <div className="flex justify-between gap-3"><dt>Ancien</dt><dd className="font-medium text-fg">{item.previousRaw}</dd></div>
                  <div className="flex justify-between gap-3"><dt>Actuel</dt><dd className="font-medium text-fg">{item.currentRaw}</dd></div>
                  <div className="flex justify-between gap-3"><dt>Variation</dt><dd className="font-semibold text-accent">{variationLabel(item.variationPct)}</dd></div>
                </dl>
                <p className="mt-3 rounded-md border border-border/60 bg-fg/[0.03] px-2 py-2 text-center text-xs text-fg/85">
                  {item.previousRaw} ───────────────▶ {item.currentRaw}
                </p>
              </article>
            ))}
          </div>
        </section>
      ) : null}

      <section className={SECTION_CHROME}>
        <div className="mb-3 flex items-center gap-2">
          <AlertTriangle size={16} className="text-amber-500" />
          <h3 className="text-sm font-semibold">Valeurs hors référence</h3>
        </div>
        {abnormalResults.length > 0 ? (
          <div className="grid gap-2">
            {abnormalResults.map((result, index) => (
              <div key={`${result.analyte}-abnormal-${index}`} className={`rounded-lg border px-3 py-2 text-sm ${toneBadge(result.tone)}`}>
                <span className="font-semibold">{result.analyte}</span>
                <span className="mx-2 text-fg/40">·</span>
                <span>{result.value}</span>
                <span className="mx-2 text-fg/40">·</span>
                <span>{statusLabel(result.tone, result.status)}</span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm leading-6 text-fg/70">Aucune valeur hors référence clairement détectée dans la réponse affichée.</p>
        )}
      </section>

      <section className={SECTION_CHROME}>
        <div className="mb-3 flex items-center gap-2">
          <HelpCircle size={16} className="text-fg/55" />
          <h3 className="text-sm font-semibold">Éléments manquants / à vérifier</h3>
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
        <div className="mb-3 flex items-center gap-2">
          <FileText size={16} className="text-accent" />
          <h3 className="text-sm font-semibold">Sources utilisées</h3>
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
                  className="rounded-lg border border-border/80 bg-card px-3 py-1.5 text-xs font-medium text-accent underline-offset-2 hover:underline"
                >
                  {source.label}
                </a>
              ) : (
                <span key={`${source.label}-${index}`} className="rounded-lg border border-border/80 bg-card px-3 py-1.5 text-xs text-fg/72">
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
