"use client";

import type { ContextUsageStatus } from "@/services/rag-api";

export type ContextWindowMeterProps = {
  model: string;
  contextWindow: number;
  usedTokens: number;
  remainingTokens: number;
  usagePercent: number;
  status: ContextUsageStatus;
};

function statusClasses(status: ContextUsageStatus): string {
  if (status === "safe") return "text-emerald-400 border-emerald-400/35 bg-emerald-500/10";
  if (status === "medium") return "text-sky-400 border-sky-400/35 bg-sky-500/10";
  if (status === "warning") return "text-amber-400 border-amber-400/35 bg-amber-500/10";
  return "text-rose-400 border-rose-400/35 bg-rose-500/10";
}

function barClass(status: ContextUsageStatus): string {
  if (status === "safe") return "bg-emerald-400";
  if (status === "medium") return "bg-sky-400";
  if (status === "warning") return "bg-amber-400";
  return "bg-rose-400";
}

function compactK(value: number): string {
  if (!Number.isFinite(value)) return "0";
  if (value >= 1000) return `${Math.round(value / 1000)}k`;
  return String(Math.max(0, Math.round(value)));
}

export function ContextWindowMeter({
  model,
  contextWindow,
  usedTokens,
  remainingTokens,
  usagePercent,
  status,
}: ContextWindowMeterProps) {
  const pct = Math.max(0, Math.min(100, usagePercent || 0));
  return (
    <div
      className={`rounded-lg border px-2.5 py-2 text-xs ${statusClasses(status)}`}
      title={`Contexte utilisé : ${compactK(usedTokens)} / ${compactK(contextWindow)} tokens`}
    >
      <div className="flex items-center justify-between gap-2">
        <p className="font-medium">{model} · {compactK(contextWindow)}</p>
        <p className="font-semibold">{pct.toFixed(1)}%</p>
      </div>
      <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-white/10">
        <div className={`h-full ${barClass(status)}`} style={{ width: `${pct}%` }} />
      </div>
      <p className="mt-1 text-[11px] text-fg/70">Contexte utilisé : {compactK(usedTokens)} / {compactK(contextWindow)} · restant {compactK(remainingTokens)}</p>
    </div>
  );
}

