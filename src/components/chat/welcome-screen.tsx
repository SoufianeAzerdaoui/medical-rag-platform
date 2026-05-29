"use client";

import { FileSearch, FileText, FlaskConical, GitCompareArrows, Sparkles } from "lucide-react";
import { ActionCards, type QuickAction } from "@/components/chat/action-cards";

const QUICK_ACTIONS: readonly QuickAction[] = [
  {
    id: "analyser-bilan",
    title: "Analyser un bilan",
    description: "Résumé biologique + anomalies principales",
    prompt: "Analyse ce bilan biologique et donne les anomalies principales avec statut technique.",
    mode: "document_analysis",
    icon: FlaskConical,
  },
  {
    id: "comparer-rapports",
    title: "Comparer deux rapports",
    description: "Comparer les valeurs actuelles et antérieures",
    prompt: "Compare ces deux rapports et indique les variations importantes.",
    mode: "comparison",
    icon: GitCompareArrows,
  },
  {
    id: "chercher-valeur",
    title: "Chercher une valeur",
    description: "Ex : Quelle est la TSH dans report 16 ?",
    prompt: "Quelle est la TSH dans report 16 ?",
    mode: "general",
    icon: FileSearch,
  },
  {
    id: "note-clinique",
    title: "Préparer une note clinique",
    description: "Synthèse prudente avec sources",
    prompt: "Prépare une note clinique prudente avec les sources documentaires utilisées.",
    mode: "summary",
    icon: FileText,
  },
];

export function WelcomeScreen({
  sending,
  onActionSelect,
}: {
  sending?: boolean;
  onActionSelect: (action: QuickAction) => void;
}) {
  return (
    <div className="mx-auto flex min-h-full w-full max-w-5xl flex-col justify-center gap-6 px-5 py-10">
      <div className="max-w-2xl">
        <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-accent/25 bg-accent/10 px-3 py-1 text-xs font-medium text-accent">
          <Sparkles size={14} />
          Assistant clinique augmenté
        </div>
        <h2 className="text-3xl font-semibold tracking-tight text-fg sm:text-4xl">CHU Oujda Clinical Assistant</h2>
        <p className="mt-3 max-w-xl text-sm leading-6 text-fg/[0.68]">
          Analyse les documents médicaux, compare les résultats et met en avant les points à vérifier avec les sources disponibles.
        </p>
        <p className="mt-2 text-sm text-fg/[0.62]">Aucun document sélectionné. Importez un rapport ou choisissez une question suggérée.</p>
      </div>
      <ActionCards actions={QUICK_ACTIONS} disabled={sending} onSelect={onActionSelect} />
      <p className="text-xs text-fg/[0.54]">Cette réponse ne remplace pas l&apos;avis médical.</p>
    </div>
  );
}
