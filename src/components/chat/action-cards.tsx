"use client";

import { ArrowRight } from "lucide-react";
import type { LucideIcon } from "lucide-react";

export type QuickAction = {
  id: string;
  title: string;
  description: string;
  prompt: string;
  mode: "general" | "summary" | "comparison" | "document_analysis";
  icon: LucideIcon;
};

export function ActionCards({
  actions,
  disabled,
  onSelect,
}: {
  actions: readonly QuickAction[];
  disabled?: boolean;
  onSelect: (action: QuickAction) => void;
}) {
  return (
    <div className="grid w-full gap-3 md:grid-cols-2">
      {actions.map((action) => (
        <div
          key={action.id}
          className="group flex min-h-24 items-start gap-3 rounded-xl border border-border/85 bg-card/[0.62] px-3 py-3 text-left text-sm transition-all duration-[180ms] ease-in-out hover:-translate-y-0.5 hover:border-accent/40 hover:bg-accent/10 hover:shadow-[0_8px_24px_hsl(var(--accent)/0.18)] data-[disabled=true]:pointer-events-none data-[disabled=true]:opacity-55 sm:px-4"
          data-disabled={disabled ? "true" : "false"}
        >
          <span className="mb-[10px] inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-[8px] bg-accent/15 text-accent sm:h-9 sm:w-9">
            <action.icon size={16} />
          </span>
          <div className="min-w-0 flex-1 space-y-2">
            <div className="space-y-0.5">
              <span className="block text-[13px] font-semibold text-fg/[0.92] sm:text-[14px]">{action.title}</span>
              <span className="block text-[11px] text-fg/[0.58] sm:text-[12px]">{action.description}</span>
            </div>
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
              <span className="text-[10px] text-fg/[0.46] sm:text-[11px]">Préremplit le champ sans envoi automatique.</span>
              <button
                type="button"
                className="group/button inline-flex h-9 w-full items-center justify-between gap-2 rounded-full border border-accent/20 bg-gradient-to-b from-accent/15 to-accent/8 px-3 text-accent shadow-[0_10px_18px_hsl(var(--accent)/0.12)] transition-all duration-200 hover:-translate-y-0.5 hover:border-accent/40 hover:bg-accent/18 hover:shadow-[0_14px_28px_hsl(var(--accent)/0.18)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/55 focus-visible:ring-offset-2 focus-visible:ring-offset-bg/80 active:translate-y-0 active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-50 sm:h-10 sm:w-auto sm:min-w-[118px]"
                disabled={disabled}
                onClick={() => onSelect(action)}
              >
                <span className="flex flex-col items-start leading-tight">
                  <span className="text-[9px] font-semibold uppercase tracking-[0.16em] text-accent/68 sm:text-[10px]">Préremplir</span>
                  <span className="text-[11px] font-medium text-accent sm:text-[12px]">Insérer</span>
                </span>
                <ArrowRight
                  size={12}
                  className="transition-transform duration-200 group-hover/button:translate-x-0.5"
                />
              </button>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
