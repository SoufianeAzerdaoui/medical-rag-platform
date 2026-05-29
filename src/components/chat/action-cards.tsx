"use client";

import type { LucideIcon } from "lucide-react";
import type { KeyboardEvent } from "react";

export type QuickAction = {
  id: string;
  title: string;
  description: string;
  prompt: string;
  mode: "general" | "summary" | "comparison" | "document_analysis";
  icon: LucideIcon;
};

function isKeyboardActivation(event: KeyboardEvent<HTMLElement>): boolean {
  return event.key === "Enter" || event.key === " ";
}

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
          role="button"
          tabIndex={disabled ? -1 : 0}
          aria-disabled={disabled ? "true" : "false"}
          onClick={() => {
            if (!disabled) onSelect(action);
          }}
          onKeyDown={(event) => {
            if (disabled || !isKeyboardActivation(event)) return;
            event.preventDefault();
            onSelect(action);
          }}
          className="group flex min-h-24 cursor-pointer items-start gap-3 rounded-xl border border-border/85 bg-card/[0.62] px-4 py-3 text-left text-sm transition-all duration-[180ms] ease-in-out hover:-translate-y-0.5 hover:border-accent/40 hover:bg-accent/10 hover:shadow-[0_8px_24px_hsl(var(--accent)/0.18)] active:translate-y-0 active:scale-[0.99] active:duration-[80ms] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/55 focus-visible:ring-offset-0 data-[disabled=true]:pointer-events-none data-[disabled=true]:opacity-55"
          data-disabled={disabled ? "true" : "false"}
        >
          <span className="mb-[10px] inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-[8px] bg-accent/15 text-accent">
            <action.icon size={16} />
          </span>
          <span className="space-y-0.5">
            <span className="mb-1 block text-[14px] font-semibold text-fg/[0.92]">{action.title}</span>
            <span className="block text-[12px] text-fg/[0.58]">{action.description}</span>
          </span>
        </div>
      ))}
    </div>
  );
}
