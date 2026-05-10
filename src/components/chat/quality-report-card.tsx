"use client";

import type { AssistantDiagnostics, QualityReport } from "@/types/chat";

type Props = {
  diagnostics?: AssistantDiagnostics;
};

function scoreTone(score: number): string {
  if (score >= 0.9) return "text-emerald-300";
  if (score >= 0.75) return "text-amber-300";
  return "text-rose-300";
}

function scoreBar(score: number): string {
  if (score >= 0.9) return "bg-emerald-500";
  if (score >= 0.75) return "bg-amber-500";
  return "bg-rose-500";
}

function metricRow(label: string, score: number) {
  const pct = Math.max(0, Math.min(100, Math.round(score * 100)));
  return (
    <div className="space-y-1" key={label}>
      <div className="flex items-center justify-between text-xs">
        <span className="text-fg/70">{label}</span>
        <span className={scoreTone(score)}>{pct}%</span>
      </div>
      <div className="h-1.5 rounded-full bg-card">
        <div className={`h-1.5 rounded-full ${scoreBar(score)}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function badge(report: QualityReport | undefined): string {
  const status = report?.final_status || "warning";
  if (status === "pass") return "border-emerald-500/40 text-emerald-300";
  if (status === "fail") return "border-rose-500/40 text-rose-300";
  return "border-amber-500/40 text-amber-300";
}

export function QualityReportCard({ diagnostics }: Props) {
  const report = diagnostics?.quality_report;
  if (!report) return null;

  return (
    <details className="mt-3 rounded-xl border border-border bg-card/30 p-3">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-2 text-xs">
        <span className="text-fg/80">Quality report (debug)</span>
        <span className={`rounded-full border px-2 py-0.5 ${badge(report)}`}>{report.final_status.toUpperCase()}</span>
      </summary>
      <div className="mt-3 grid gap-2">
        {metricRow("Faithfulness", report.faithfulness_score)}
        {metricRow("Format compliance", report.format_compliance_score)}
        {metricRow("Readability", report.readability_score)}
        {metricRow("Source UX", report.source_ux_score)}
        {metricRow("Style repetition", 1 - report.style_repetition_score)}
        {metricRow("Safety", report.safety_score)}
      </div>
      <div className="mt-3 flex flex-wrap gap-2 text-[11px] text-fg/60">
        {diagnostics?.validation_status ? <span>validation: {diagnostics.validation_status}</span> : null}
        {diagnostics?.generation_writer ? <span>writer: {diagnostics.generation_writer}</span> : null}
        {diagnostics?.generation_mode ? <span>mode: {diagnostics.generation_mode}</span> : null}
        {typeof diagnostics?.response_time === "number" ? <span>latency: {diagnostics.response_time.toFixed(3)}s</span> : null}
      </div>
    </details>
  );
}

