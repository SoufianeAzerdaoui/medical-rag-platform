import type { ChatMode, ChatSource } from "@/types/chat";

export type ConversationTitleInput = {
  userMessage: string;
  assistantMessage?: string;
  answerType?: "smalltalk" | "lab_analysis" | "comparison" | "no_source" | "plain";
  sources?: Array<{
    documentName?: string;
    pageNumber?: number;
  }>;
};

function normalize(input: string): string {
  return String(input || "")
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function extractReports(rawText: string): string[] {
  const reportMatches = rawText.match(/report\s*\(?\d+\)?/gi) || [];
  const normalized = reportMatches.map((r) => r.replace(/\s+/g, " ").trim().toLowerCase());
  return Array.from(new Set(normalized)).slice(0, 2);
}

function trimTitle(value: string, max = 50): string {
  const safe = String(value || "").trim();
  if (safe.length <= max) return safe;
  return `${safe.slice(0, max - 1).trimEnd()}…`;
}

function detectAnswerType(text: string, mode: ChatMode): ConversationTitleInput["answerType"] {
  if (/^(bonjour|salut|hello|hi|bonsoir|coucou)\b/.test(text)) return "smalltalk";
  if (mode === "comparison" || /(compare|comparaison|difference|différence|evolution|évolution)/.test(text)) return "comparison";
  if (mode === "document_analysis" || /(anormal|hors reference|hors référence|thyro|tsh|t3|t4|bilan)/.test(text)) return "lab_analysis";
  if (/(sources uniquement|source uniquement|juste les sources)/.test(text)) return "no_source";
  return "plain";
}

function sourceDocName(source: ChatSource): string | undefined {
  if (typeof source === "string") {
    const found = source.match(/(?:doc_id=)?([^,\]\s]+(?:\.pdf)?)/i)?.[1];
    return found || undefined;
  }
  const raw = source as Record<string, unknown>;
  return String(raw.documentName || raw.filename || raw.doc_id || raw.documentId || "").trim() || undefined;
}

export function inferConversationTitle(input: ConversationTitleInput): string {
  const userMessage = String(input.userMessage || "").trim();
  const text = normalize(userMessage);
  const reports = extractReports(userMessage);

  const hasThyroid =
    text.includes("tsh") ||
    text.includes("t3") ||
    text.includes("t4") ||
    text.includes("thyro");

  const isGreeting = ["bonjour", "salut", "hello", "bonsoir", "hi"].some((k) => text.startsWith(k));
  const isComparison =
    text.includes("compare") ||
    text.includes("comparaison") ||
    text.includes("difference") ||
    text.includes("evolution");
  const isAbnormal =
    text.includes("anormal") ||
    text.includes("anormaux") ||
    text.includes("hors reference") ||
    text.includes("valeurs hors");
  const isSummary =
    text.includes("resume") ||
    text.includes("synthese") ||
    text.includes("bilan");
  const isVerify =
    text.includes("verification") ||
    text.includes("a verifier") ||
    text.includes("à verifier") ||
    text.includes("a vérifier");

  if (isGreeting) return "Accueil assistant";

  if (isComparison && reports.length >= 2) {
    return trimTitle(`Comparaison ${reports[0]} vs ${reports[1]}`);
  }
  if (isComparison) return "Comparaison de rapports";

  if (hasThyroid && reports.length > 0) return trimTitle(`Bilan thyroïdien – ${reports[0]}`);
  if (hasThyroid) return "Analyse thyroïdienne";

  if (isAbnormal && reports.length > 0) return trimTitle(`Anomalies biologiques – ${reports[0]}`);
  if (isAbnormal) return "Valeurs hors référence";

  if (isSummary && reports.length > 0) return trimTitle(`Résumé biologique – ${reports[0]}`);
  if (isSummary) return "Résumé biologique";

  if (isVerify) return "Éléments à vérifier";

  if ((input.sources || []).length > 0) return "Analyse documentaire";
  return "Conversation générale";
}

export function deriveAutoConversationTitle(message: string, mode: ChatMode): string | null {
  const trimmed = String(message || "").trim();
  if (!trimmed) return null;
  const normalized = normalize(trimmed);
  const answerType = detectAnswerType(normalized, mode);
  return inferConversationTitle({ userMessage: trimmed, answerType });
}

export function buildSubtitleMeta(params: {
  updatedAt: string;
  sourceCount: number;
}): string {
  const date = new Date(params.updatedAt);
  const dateLabel = `${date.toLocaleDateString("fr-FR", { day: "numeric", month: "short" })}, ${date.toLocaleTimeString("fr-FR", {
    hour: "2-digit",
    minute: "2-digit",
  })}`;
  if (params.sourceCount > 0) {
    return `${params.sourceCount} source${params.sourceCount > 1 ? "s" : ""} · ${dateLabel}`;
  }
  return `Conversation générale · ${dateLabel}`;
}

export function resolveAutoTitleUpdate(params: {
  currentTitle: string;
  titleSource: "auto" | "manual";
  titleGenerated?: boolean;
  titleEditedByUser?: boolean;
  userMessage?: string;
  message?: string;
  mode: ChatMode;
  assistantMessage?: string;
  sources?: ChatSource[];
  answerType?: ConversationTitleInput["answerType"];
  messageCount?: number;
}): {
  title: string;
  titleSource: "auto" | "manual";
  titleGenerated: boolean;
  titleEditedByUser: boolean;
} {
  const {
    currentTitle,
    titleSource,
    titleGenerated,
    titleEditedByUser,
    userMessage: userMessageRaw,
    mode,
    assistantMessage,
    sources = [],
    answerType,
  } = params;
  const userMessage = String(userMessageRaw || params.message || "").trim();

  if (titleSource === "manual" || titleEditedByUser) {
    return {
      title: currentTitle,
      titleSource: "manual",
      titleGenerated: Boolean(titleGenerated),
      titleEditedByUser: true,
    };
  }

  if (typeof params.messageCount === "number" && params.messageCount > 0 && !titleGenerated) {
    return {
      title: currentTitle,
      titleSource: "auto",
      titleGenerated: false,
      titleEditedByUser: false,
    };
  }

  if (titleGenerated && currentTitle && currentTitle.trim() && !/^nouvelle conversation$/i.test(currentTitle.trim())) {
    return {
      title: currentTitle,
      titleSource: "auto",
      titleGenerated: true,
      titleEditedByUser: false,
    };
  }

  const normalized = normalize(userMessage);
  const safeAnswerType = answerType || detectAnswerType(normalized, mode);
  const title = inferConversationTitle({
    userMessage,
    assistantMessage,
    answerType: safeAnswerType,
    sources: sources.map((s) => ({ documentName: sourceDocName(s) })),
  });

  return {
    title: trimTitle(title || currentTitle || "Conversation générale"),
    titleSource: "auto",
    titleGenerated: true,
    titleEditedByUser: false,
  };
}
