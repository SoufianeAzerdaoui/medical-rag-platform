"use client";

import { useMemo } from "react";

type Status = "safe" | "medium" | "warning" | "full";

function formatCompact(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return "0";
  if (value >= 1000) return `${Math.round(value / 1000)}k`;
  return String(Math.round(value));
}

function clampPercent(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, value));
}

function statusFromPercent(percent: number): Status {
  if (percent >= 90) return "full";
  if (percent >= 70) return "warning";
  return "safe";
}

function toneFromStatus(status: Status): string {
  if (status === "full") return "hsl(var(--danger))";
  if (status === "warning") return "hsl(var(--warning))";
  if (status === "medium") return "hsl(212 86% 49%)";
  return "hsl(var(--success))";
}

export function ModelBadge({
  modelName,
  contextWindow,
  usedTokens,
  usagePercent,
  status,
}: {
  modelName?: string;
  contextWindow?: number;
  usedTokens?: number;
  usagePercent?: number;
  status?: Status;
}) {
  const computedPercent = clampPercent(usagePercent ?? 0);
  const resolvedStatus = status ?? statusFromPercent(computedPercent);
  const tone = toneFromStatus(resolvedStatus);
  const contextLabel = formatCompact(contextWindow ?? 0);
  const usedLabel = Math.max(0, Math.round(usedTokens ?? 0));
  const maxLabel = Math.max(1, Math.round(contextWindow ?? 1));

  const widthStyle = useMemo(
    () => ({ width: `${computedPercent}%`, backgroundColor: tone }),
    [computedPercent, tone],
  );

  return (
    <div className="group relative shrink-0">
      <div className="inline-flex h-6 cursor-default items-center gap-1.5 rounded-[20px] border border-border/80 bg-card/[0.75] px-2.5 text-[11px] text-fg/55">
        <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: tone }} />
        <span>{contextLabel}</span>
      </div>

      <div className="pointer-events-none absolute bottom-full left-0 z-40 mb-2 w-max max-w-[240px] rounded-lg border border-border/80 bg-card/[0.98] p-2.5 text-xs text-fg/80 opacity-0 shadow-[0_10px_30px_hsl(220_30%_10%_/_0.2)] transition-opacity duration-100 ease-out group-hover:opacity-100 group-hover:duration-150 group-focus-within:opacity-100 group-focus-within:duration-150">
        <p className="truncate text-[11px] font-medium text-fg/90">{modelName || "Modèle actif"}</p>
        <p className="mt-1 text-[11px] text-fg/68">
          Contexte utilisé : {usedLabel} / {maxLabel} tokens ({computedPercent.toFixed(1)}%)
        </p>
        <div className="mt-2 h-1 w-40 overflow-hidden rounded-[2px] bg-fg/15">
          <div className="h-full rounded-[2px] transition-all duration-150" style={widthStyle} />
        </div>
      </div>
    </div>
  );
}
