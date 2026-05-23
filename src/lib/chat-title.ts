import type { ChatMode } from "@/types/chat";

const IGNORED_MESSAGES = new Set([
  "bonjour",
  "salut",
  "merci",
  "ok",
  "d'accord",
  "daccord",
  "continue",
  "explique plus",
]);

const ANOMALY_PATTERNS = [
  "quels resultats semblent anormaux",
  "quels résultats semblent anormaux",
  "resultats hors reference",
  "résultats hors référence",
  "resultats necessitant attention",
  "résultats nécessitant attention",
  "resultats anormaux",
  "résultats anormaux",
];

function normalize(input: string): string {
  return input
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function extractReportIds(text: string): string[] {
  const matches = text.match(/\breport[_\s-]?\d+\b/gi) || [];
  const normalized = matches.map((m) => m.toLowerCase().replace(/\s|-/g, "_"));
  return Array.from(new Set(normalized)).slice(0, 2);
}

function containsComparisonIntent(text: string, mode: ChatMode): boolean {
  if (mode === "comparison") return true;
  return (
    /\b(compare|comparaison|comparer)\b/i.test(text) ||
    /\bdeux documents\b/i.test(text) ||
    /\bdocument comparison\b/i.test(text)
  );
}

export function deriveAutoConversationTitle(message: string, mode: ChatMode): string | null {
  const trimmed = message.trim();
  if (!trimmed) return null;

  const normalized = normalize(trimmed);
  if (IGNORED_MESSAGES.has(normalized)) return null;

  if (ANOMALY_PATTERNS.some((pattern) => normalized.includes(normalize(pattern)))) {
    return "Résultats hors référence";
  }

  if (containsComparisonIntent(trimmed, mode)) {
    const reportIds = extractReportIds(trimmed);
    if (reportIds.length >= 2) {
      return `Comparaison ${reportIds[0]} / ${reportIds[1]}`;
    }
    return "Comparaison de rapports";
  }

  if (/\binventaire\b/i.test(trimmed) && /\bpatient/i.test(trimmed)) {
    return "Inventaire patients";
  }

  if (/\bcommentaire\b/i.test(trimmed) && /\btroponine\b/i.test(trimmed)) {
    return "Commentaire troponine";
  }

  if (/\bacth\b/i.test(trimmed)) {
    return "ACTH — dernier rapport";
  }

  // Anti-PHI and unknown-intent guard:
  // avoid using raw user text as title to prevent leaking PII/PHI.
  return null;
}

export function resolveAutoTitleUpdate(params: {
  currentTitle: string;
  titleSource: "auto" | "manual";
  message: string;
  mode: ChatMode;
  messageCount: number;
}): { title: string; titleSource: "auto" | "manual" } {
  const { currentTitle, titleSource, message, mode, messageCount } = params;
  if (titleSource === "manual") {
    return { title: currentTitle, titleSource: "manual" };
  }
  if (messageCount > 0) {
    return { title: currentTitle, titleSource: "auto" };
  }
  return {
    title: deriveAutoConversationTitle(message, mode) ?? currentTitle,
    titleSource: "auto",
  };
}
