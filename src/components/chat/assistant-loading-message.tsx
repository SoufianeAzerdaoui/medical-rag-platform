"use client";

import { useEffect, useState } from "react";

const loadingSteps = [
  "Assistant en train d’analyser les documents...",
  "Recherche dans report_16.pdf...",
  "Extraction des valeurs...",
  "Génération de réponse prudente...",
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
        <div className="flex gap-1" aria-hidden>
          <span className="h-2 w-2 animate-bounce rounded-full bg-slate-300" />
          <span className="h-2 w-2 animate-bounce rounded-full bg-slate-300 [animation-delay:120ms]" />
          <span className="h-2 w-2 animate-bounce rounded-full bg-slate-300 [animation-delay:240ms]" />
        </div>
        <span className="text-sm text-slate-300">{loadingSteps[stepIndex]}</span>
      </div>
      <ol className="mt-3 space-y-1">
        {loadingSteps.map((step, index) => (
          <li key={step} className={`text-xs ${index <= stepIndex ? "text-fg/85" : "text-fg/45"}`}>
            {index + 1}. {step}
          </li>
        ))}
      </ol>
      <div className="mt-4 space-y-2">
        <div className="h-3 w-3/4 animate-pulse rounded bg-slate-800" />
        <div className="h-3 w-2/3 animate-pulse rounded bg-slate-800" />
      </div>
    </div>
  );
}
