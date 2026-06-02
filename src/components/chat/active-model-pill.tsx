"use client";

import type { ActiveModelInfo } from "@/services/rag-api";

function compactContext(value: number | null | undefined): string {
  const n = Number(value || 0);
  if (!Number.isFinite(n) || n <= 0) return "ctx n/a";
  if (n >= 1_000_000) return `ctx ${Math.round(n / 1_000_000)}M`;
  if (n >= 1000) return `ctx ${Math.round(n / 1000)}k`;
  return `ctx ${Math.round(n)}`;
}

function providerLabel(provider: string, model?: string | null): string {
  const normalized = String(provider || "").trim().toLowerCase();
  if (normalized === "gemini") return "Gemini cloud";
  if (normalized === "ollama") {
    const modelNorm = String(model || "").trim().toLowerCase();
    if (modelNorm.startsWith("qwen")) return "Qwen local";
    if (modelNorm.startsWith("llama")) return "Llama local";
    if (modelNorm.startsWith("mistral")) return "Mistral local";
    if (modelNorm.startsWith("gemma") || modelNorm.startsWith("medgemma")) return "Gemma local";
    if (modelNorm.startsWith("deepseek")) return "DeepSeek local";
    return "Local Ollama";
  }
  if (normalized === "lmstudio") return "LM Studio local";
  return provider || "Modèle actif";
}

function tone(provider: string): string {
  const normalized = String(provider || "").trim().toLowerCase();
  if (normalized === "gemini") return "border-cyan-500/30 bg-cyan-500/10 text-cyan-100";
  if (normalized === "ollama") return "border-emerald-500/30 bg-emerald-500/10 text-emerald-100";
  if (normalized === "lmstudio") return "border-violet-500/30 bg-violet-500/10 text-violet-100";
  return "border-border/70 bg-card/70 text-fg/75";
}

export function ActiveModelPill({ model }: { model?: ActiveModelInfo | null }) {
  if (!model) return null;

  return (
    <div className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-[11px] font-medium shadow-sm ${tone(model.provider)}`}>
      <span className="h-1.5 w-1.5 rounded-full bg-current/80" aria-hidden="true" />
      <span className="whitespace-nowrap">{providerLabel(model.provider, model.model)}</span>
      <span className="whitespace-nowrap text-current/70">·</span>
      <span className="max-w-[200px] truncate whitespace-nowrap">{model.model}</span>
      <span className="whitespace-nowrap text-current/70">·</span>
      <span className="whitespace-nowrap text-current/80">{compactContext(model.context_window)}</span>
    </div>
  );
}
