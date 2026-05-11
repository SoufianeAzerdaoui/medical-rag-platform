"use client";

import { useEffect, useState } from "react";

const loadingSteps = [
  "Analyse de la question...",
  "Recherche dans les rapports medicaux...",
  "Selection des sources pertinentes...",
  "Verification des valeurs et references...",
  "Preparation de la reponse sourcee...",
];

const STEP_INTERVAL_MS = 1700;

export function AssistantLoadingMessage() {
  const [stepIndex, setStepIndex] = useState(0);

  useEffect(() => {
    const interval = window.setInterval(() => {
      setStepIndex((prev) => (prev + 1) % loadingSteps.length);
    }, STEP_INTERVAL_MS);
    return () => window.clearInterval(interval);
  }, []);

  return (
    <div role="status" aria-live="polite">
      <span className="sr-only">L’assistant prepare une reponse.</span>
      <div className="mb-3 text-xs font-medium uppercase tracking-wide text-fg/60">Assistant</div>
      <div className="flex items-center gap-3">
        <div className="flex gap-1">
          <span className="h-2 w-2 animate-bounce rounded-full bg-slate-300" />
          <span className="h-2 w-2 animate-bounce rounded-full bg-slate-300 [animation-delay:120ms]" />
          <span className="h-2 w-2 animate-bounce rounded-full bg-slate-300 [animation-delay:240ms]" />
        </div>
        <span className="text-sm text-slate-300">{loadingSteps[stepIndex]}</span>
      </div>
      <div className="mt-4 space-y-2">
        <div className="h-3 w-3/4 animate-pulse rounded bg-slate-800" />
        <div className="h-3 w-2/3 animate-pulse rounded bg-slate-800" />
      </div>
    </div>
  );
}
