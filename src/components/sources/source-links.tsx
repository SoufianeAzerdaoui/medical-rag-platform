"use client";

import { FileText } from "lucide-react";
import type { ChatSource, LegacySourceItem, SourceCitation } from "@/types/chat";

const LEGACY_DOC_PAGE_RE = /doc_id=([^,\]\s]+)(?:\s*,\s*page=(\d+))?(?:\s*,\s*row=(\d+))?/i;
const RELATIVE_URL_RE = /(\/(?:viewer\/pdf|api\/documents\/)[^\s)]+)/i;
const ABSOLUTE_URL_RE = /(https?:\/\/[^\s)]+)/i;

function isStructuredCitation(source: ChatSource): source is SourceCitation {
  return typeof source === "object" && source !== null && "doc_id" in source && typeof source.doc_id === "string";
}

function sanitizeLabel(raw: string): string {
  const cleaned = (raw || "")
    .replace(/chunk_id=[^,\]\s]+/gi, "")
    .replace(/\(?(doc_id=[^)]+)\)?/gi, "")
    .replace(/\/?home\/[\w./-]+/gi, "")
    .replace(/[A-Za-z]:\\[^\s]+/g, "")
    .replace(/\bpage\s*(\d+)\s*row\s*(\d+)\b/gi, "page $1, ligne $2")
    .replace(/\brow\s*=?\s*(\d+)\b/gi, "ligne $1")
    .replace(/\b(ligne\s*\d+)\s*\1\b/gi, "$1")
    .replace(/\b(ligne\s*\d+)\s*,?\s*ligne\s*\d+\b/gi, "$1")
    .replace(/\s{2,}/g, " ")
    .replace(/\s+,/g, ",")
    .replace(/\s*:\s*$/g, "")
    .replace(/,\s*ligne\s*(\d+)\s*ligne\s*\1/gi, ", ligne $1")
    .trim();
  return cleaned || "Source";
}

function buildViewerUrl(docId: string, page?: number | null): string {
  const encoded = encodeURIComponent(docId);
  if (page && Number.isFinite(page)) {
    return `/viewer/pdf?doc_id=${encoded}&page=${page}`;
  }
  return `/viewer/pdf?doc_id=${encoded}`;
}

function normalizeSource(source: ChatSource, index: number) {
  if (isStructuredCitation(source)) {
    const href = source.viewer_url || source.url || null;
    const fallback = `${source.doc_id}${source.page ? ` — page ${source.page}` : ""}`;
    const label = sanitizeLabel(source.label || fallback);
    return {
      key: `${source.doc_id}-${source.page ?? "na"}-${source.row ?? "na"}-${index}`,
      label,
      href,
      row: source.row ?? null,
      page: source.page ?? null,
      docId: source.doc_id,
      filename: source.filename || null,
      groupable: Boolean(source.doc_id && source.page),
      clickable: Boolean(href),
    };
  }

  if (typeof source === "string") {
    const text = sanitizeLabel(source);
    const urlMatch = source.match(ABSOLUTE_URL_RE) || source.match(RELATIVE_URL_RE);
    const url = urlMatch ? urlMatch[1] : null;
    if (url) {
      const labelCandidate = sanitizeLabel(source.slice(0, urlMatch?.index || 0)).replace(/^[-*]\s*/, "");
      const label = labelCandidate || "Source PDF";
      const rowMatch = source.match(/row\s*=?\s*(\d+)/i);
      return {
        key: `legacy-url-${index}`,
        label,
        href: url,
        row: rowMatch ? Number(rowMatch[1]) : null,
        page: null,
        docId: "",
        filename: null,
        groupable: false,
        clickable: true,
      };
    }
    const match = text.match(LEGACY_DOC_PAGE_RE);
    if (!match) {
      return {
        key: `legacy-text-${index}`,
        label: text,
        href: null,
        row: null,
        page: null,
        docId: "",
        filename: null,
        groupable: false,
        clickable: false,
      };
    }
    const docId = match[1];
    const page = match[2] ? Number(match[2]) : null;
    const row = match[3] ? Number(match[3]) : null;
    return {
      key: `legacy-${docId}-${page ?? "na"}-${row ?? "na"}-${index}`,
      label: `${docId}${page ? ` — page ${page}` : ""}`,
      href: buildViewerUrl(docId, page),
      row,
      page,
      docId,
      filename: null,
      groupable: Boolean(docId && page),
      clickable: true,
    };
  }

  const legacyObject = source as LegacySourceItem;
  const docId = typeof legacyObject.documentId === "string" ? legacyObject.documentId : "";
  const page = typeof legacyObject.page === "number" ? legacyObject.page : null;
  const label = sanitizeLabel(
    String(legacyObject.documentName || legacyObject.excerpt || `${docId}${page ? ` — page ${page}` : ""}` || "Source"),
  );
  const href =
    (typeof legacyObject.viewer_url === "string" && legacyObject.viewer_url) ||
    (typeof legacyObject.url === "string" && legacyObject.url) ||
    (docId ? buildViewerUrl(docId, page) : null);

  return {
    key: `legacy-object-${docId || "unknown"}-${page ?? "na"}-${index}`,
    label,
    href: href || null,
    row: null,
    page,
    docId,
    filename: null,
    groupable: Boolean(docId && page),
    clickable: Boolean(href),
  };
}

function labelAlreadyHasLine(label: string): boolean {
  return /\bligne[s]?\s+\d+/i.test(label);
}

function buildGroupedLabel(filename: string, page: number, rows: number[], fallbackLabel: string): string {
  if (rows.length === 0) {
    return sanitizeLabel(fallbackLabel || `${filename} — page ${page}`);
  }
  const sorted = [...new Set(rows)].sort((a, b) => a - b);
  if (sorted.length === 1) {
    return `${filename} — page ${page}, ligne ${sorted[0]}`;
  }
  const min = sorted[0];
  const max = sorted[sorted.length - 1];
  return `${filename} — page ${page}, lignes ${min}–${max}`;
}

function groupSources(
  items: Array<{
    key: string;
    label: string;
    href: string | null;
    row: number | null;
    page: number | null;
    docId: string;
    filename: string | null;
    groupable: boolean;
    clickable: boolean;
  }>,
) {
  const grouped = new Map<string, (typeof items)[number] & { _rows: number[] }>();
  for (const item of items) {
    if (!item.groupable || !item.page || !item.docId) {
      grouped.set(item.key, { ...item, _rows: typeof item.row === "number" ? [item.row] : [] });
      continue;
    }
    const key = `${item.docId}::${item.page}`;
    const current = grouped.get(key);
    const rows = typeof item.row === "number" ? [item.row] : [];
    if (!current) {
      grouped.set(key, { ...item, key, _rows: rows });
      continue;
    }
    current._rows = [...current._rows, ...rows];
    if (!current.href && item.href) current.href = item.href;
    if (!current.filename && item.filename) current.filename = item.filename;
  }

  return Array.from(grouped.values()).map((entry) => {
    if (!entry.groupable || !entry.page || !entry.docId) {
      const safeLabel = sanitizeLabel(entry.label);
      return {
        ...entry,
        label: safeLabel,
        row: labelAlreadyHasLine(safeLabel) ? null : entry.row,
      };
    }
    const filename = entry.filename || entry.docId;
    const label = buildGroupedLabel(filename, entry.page, entry._rows, entry.label);
    return {
      ...entry,
      label,
      row: null,
    };
  });
}

export function stripSourcesBlock(answer: string): string {
  const text = (answer || "").replace(/\r\n/g, "\n");
  const match = text.match(/(?:^|\n)\s*Sources?\s*:\s*\n/i);
  if (!match || match.index === undefined) {
    return answer;
  }
  const before = text.slice(0, match.index).trimEnd();
  return before || answer;
}

export function stripSourcesSection(answer: string): string {
  return stripSourcesBlock(answer);
}

export function SourceLinks({
  sources,
  showTitle = true,
  compact = false,
}: {
  sources?: ChatSource[];
  showTitle?: boolean;
  compact?: boolean;
}) {
  const normalized = (sources || []).map(normalizeSource).filter((item) => item.label);
  const list = groupSources(normalized);
  if (list.length === 0) return null;

  return (
    <div className={compact ? "rounded-xl border border-border/70 bg-card/40 p-3" : "mt-3 rounded-xl border border-border/70 bg-card/40 p-3"}>
      {showTitle ? <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-fg/70">Sources</div> : null}
      <ul className="space-y-2 text-sm">
        {list.map((source) => (
          <li key={source.key} className="flex items-start gap-2">
            <FileText size={14} className="mt-0.5 shrink-0 text-fg/70" />
            <div className="min-w-0">
              {source.clickable ? (
                <a
                  href={source.href || undefined}
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label={`Ouvrir la source PDF: ${source.label}`}
                  className="break-words text-accent underline-offset-2 hover:underline"
                >
                  {source.label}
                </a>
              ) : (
                <span className="break-words text-fg">{source.label}</span>
              )}
              {typeof source.row === "number" && !labelAlreadyHasLine(source.label) ? (
                <span className="ml-2 text-xs text-fg/60" title={`ligne ${source.row}`}>
                  ligne {source.row}
                </span>
              ) : null}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
