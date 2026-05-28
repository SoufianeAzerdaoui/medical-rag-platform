"use client";

import { Compass, FlaskConical, GitCompareArrows, SearchCheck, Sparkles } from "lucide-react";
import { useChatActions } from "@/hooks/use-chat-actions";
import type { ChatMode } from "@/types/chat";

type QuickAction = {
  label: string;
  prompt: string;
  mode: ChatMode;
  icon: typeof FlaskConical;
};

const QUICK_ACTIONS: QuickAction[] = [
  {
    label: "Résumer un rapport",
    prompt: "Résume ce rapport biologique en points clés.",
    mode: "summary",
    icon: Sparkles,
  },
  {
    label: "Détecter hors référence",
    prompt: "Détecte les valeurs hors référence dans ce rapport.",
    mode: "document_analysis",
    icon: SearchCheck,
  },
  {
    label: "Comparer deux rapports",
    prompt: "Compare ces deux rapports et donne les différences importantes.",
    mode: "comparison",
    icon: GitCompareArrows,
  },
  {
    label: "Expliquer une valeur",
    prompt: "Explique cette valeur biologique et son statut technique.",
    mode: "general",
    icon: FlaskConical,
  },
];

function normalizeGreeting(content: string): string {
  const text = String(content || "").trim();
  if (!text) return "Bonjour";
  const lowered = text.toLowerCase();
  if (/^(bonjour|salut|hello|hi|bonsoir|coucou|hey|salam)\b/.test(lowered)) return "Bonjour";
  return text;
}

function normalizeBody(content: string): string {
  const text = String(content || "").trim();
  if (!text) return "Je peux vous aider à analyser les rapports médicaux indexés.";
  if (/^(bonjour|salut|hello|hi|bonsoir|coucou|hey|salam)\s*[.!?]*$/i.test(text)) {
    return "Je peux vous aider à analyser les rapports médicaux indexés.";
  }
  return text;
}

export function AssistantConversationCard({ content }: { content: string }) {
  const { sendMessage, sending } = useChatActions();
  const greeting = normalizeGreeting(content);
  const body = normalizeBody(content);

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <p className="text-lg font-semibold tracking-tight">{greeting} 👋</p>
        <p className="text-sm leading-6 text-fg/80">
          {body}
        </p>
      </div>
      <div className="space-y-2">
        <div className="inline-flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.16em] text-fg/55">
          <Compass size={12} />
          Actions possibles
        </div>
        <div className="grid gap-2 md:grid-cols-2">
          {QUICK_ACTIONS.map((action) => {
            const Icon = action.icon;
            return (
              <button
                key={action.label}
                type="button"
                className="group flex items-center gap-2 rounded-lg border border-border/80 bg-card/[0.72] px-3 py-2 text-left text-sm transition hover:border-accent/35 hover:bg-accent/10 disabled:opacity-45"
                disabled={sending}
                onClick={() => void sendMessage({ content: action.prompt, mode: action.mode })}
              >
                <span className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-accent/10 text-accent">
                  <Icon size={14} />
                </span>
                <span className="font-medium text-fg/90">{action.label}</span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
