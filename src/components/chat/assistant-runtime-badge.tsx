"use client";

import type { AssistantDiagnostics } from "@/types/chat";

function normalize(value: string | null | undefined): string {
  return String(value || "").trim().toLowerCase();
}

function providerLabel(provider: string | null | undefined, model?: string | null): string | null {
  const p = normalize(provider);
  if (!p) return null;
  if (p === "gemini") return "Gemini cloud";
  if (p === "ollama") {
    const modelNorm = normalize(model);
    if (modelNorm.startsWith("qwen")) return "Qwen local";
    if (modelNorm.startsWith("llama")) return "Llama local";
    if (modelNorm.startsWith("mistral")) return "Mistral local";
    if (modelNorm.startsWith("gemma") || modelNorm.startsWith("medgemma")) return "Gemma local";
    if (modelNorm.startsWith("deepseek")) return "DeepSeek local";
    return "Local Ollama";
  }
  if (p === "lmstudio") return "LM Studio local";
  return provider!;
}

function providerTone(provider: string | null | undefined): string {
  const p = normalize(provider);
  if (p === "gemini") return "border-cyan-500/30 bg-cyan-500/10 text-cyan-200";
  if (p === "ollama") return "border-emerald-500/30 bg-emerald-500/10 text-emerald-200";
  if (p === "lmstudio") return "border-violet-500/30 bg-violet-500/10 text-violet-200";
  return "border-border/80 bg-card/80 text-fg/75";
}

function writerTone(writer: string | null | undefined): string {
  const w = normalize(writer);
  if (w === "llm_writer") return "border-sky-500/30 bg-sky-500/10 text-sky-200";
  return "border-border/80 bg-card/80 text-fg/70";
}

type Props = {
  diagnostics?: AssistantDiagnostics;
  enabled?: boolean;
};

export function AssistantRuntimeBadge({ diagnostics, enabled = false }: Props) {
  if (!enabled) return null;
  const provider = providerLabel(
    diagnostics?.llm_provider_effective_runtime || diagnostics?.provider || null,
    diagnostics?.llm_model_effective_runtime || diagnostics?.model || null,
  );
  const model = String(diagnostics?.llm_model_effective_runtime || diagnostics?.model || "").trim();
  const writer = String(diagnostics?.generation_writer || "").trim();

  if (!provider && !model && !writer) return null;

  return (
    <div className="flex flex-wrap items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.12em]">
      {provider ? (
        <span className={`inline-flex items-center rounded-full border px-2 py-0.5 ${providerTone(diagnostics?.llm_provider_effective_runtime)}`}>
          {provider}
        </span>
      ) : null}
      {model ? (
        <span className="inline-flex items-center rounded-full border border-border/80 bg-card/80 px-2 py-0.5 text-fg/70 normal-case tracking-normal">
          {model}
        </span>
      ) : null}
      {writer ? (
        <span className={`inline-flex items-center rounded-full border px-2 py-0.5 ${writerTone(writer)}`}>
          {writer}
        </span>
      ) : null}
    </div>
  );
}
