"use client";

import { useMemo } from "react";
import type { MessageItem, QualityReport } from "@/types/chat";

type Props = {
  messages: MessageItem[];
};

type Aggregates = {
  count: number;
  avgFaithfulness: number;
  avgFormat: number;
  avgReadability: number;
  avgSourceUx: number;
  avgStyleRepetition: number;
  avgSafety: number;
  passCount: number;
  warningCount: number;
  failCount: number;
  trend: number[];
};

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

function composite(report: QualityReport): number {
  const styleQuality = 1 - clamp01(report.style_repetition_score);
  return clamp01(
    (clamp01(report.faithfulness_score) +
      clamp01(report.format_compliance_score) +
      clamp01(report.readability_score) +
      clamp01(report.source_ux_score) +
      styleQuality +
      clamp01(report.safety_score)) /
      6,
  );
}

function pct(score: number): string {
  return `${Math.round(clamp01(score) * 100)}%`;
}

function textColor(score: number): string {
  if (score >= 0.9) return "text-emerald-300";
  if (score >= 0.75) return "text-amber-300";
  return "text-rose-300";
}

function tone(status: "pass" | "warning" | "fail"): string {
  if (status === "pass") return "bg-emerald-500";
  if (status === "warning") return "bg-amber-500";
  return "bg-rose-500";
}

function aggregate(messages: MessageItem[]): Aggregates {
  const reports = messages
    .filter((m) => m.role === "assistant" && m.status === "done" && m.diagnostics?.quality_report)
    .map((m) => m.diagnostics?.quality_report as QualityReport);

  if (reports.length === 0) {
    return {
      count: 0,
      avgFaithfulness: 0,
      avgFormat: 0,
      avgReadability: 0,
      avgSourceUx: 0,
      avgStyleRepetition: 0,
      avgSafety: 0,
      passCount: 0,
      warningCount: 0,
      failCount: 0,
      trend: [],
    };
  }

  const sum = reports.reduce(
    (acc, r) => {
      acc.faithfulness += clamp01(r.faithfulness_score);
      acc.format += clamp01(r.format_compliance_score);
      acc.readability += clamp01(r.readability_score);
      acc.sourceUx += clamp01(r.source_ux_score);
      acc.styleRep += clamp01(r.style_repetition_score);
      acc.safety += clamp01(r.safety_score);
      if (r.final_status === "pass") acc.pass += 1;
      else if (r.final_status === "warning") acc.warning += 1;
      else acc.fail += 1;
      return acc;
    },
    { faithfulness: 0, format: 0, readability: 0, sourceUx: 0, styleRep: 0, safety: 0, pass: 0, warning: 0, fail: 0 },
  );

  const trend = reports.slice(-10).map(composite);
  return {
    count: reports.length,
    avgFaithfulness: sum.faithfulness / reports.length,
    avgFormat: sum.format / reports.length,
    avgReadability: sum.readability / reports.length,
    avgSourceUx: sum.sourceUx / reports.length,
    avgStyleRepetition: sum.styleRep / reports.length,
    avgSafety: sum.safety / reports.length,
    passCount: sum.pass,
    warningCount: sum.warning,
    failCount: sum.fail,
    trend,
  };
}

export function ConversationQualityPanel({ messages }: Props) {
  const stats = useMemo(() => aggregate(messages), [messages]);
  if (stats.count === 0) return null;

  return (
    <section className="rounded-2xl border border-border bg-card/50 p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div>
          <p className="text-xs text-fg/60">Conversation quality dashboard (debug)</p>
          <p className="text-sm text-fg/80">{stats.count} réponses assistant analysées</p>
        </div>
        <div className="flex items-center gap-2 text-[11px]">
          <span className="inline-flex items-center gap-1 text-emerald-300">
            <span className={`h-2 w-2 rounded-full ${tone("pass")}`} />
            {stats.passCount}
          </span>
          <span className="inline-flex items-center gap-1 text-amber-300">
            <span className={`h-2 w-2 rounded-full ${tone("warning")}`} />
            {stats.warningCount}
          </span>
          <span className="inline-flex items-center gap-1 text-rose-300">
            <span className={`h-2 w-2 rounded-full ${tone("fail")}`} />
            {stats.failCount}
          </span>
        </div>
      </div>

      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
        <div className="rounded-xl border border-border/70 p-2 text-xs">
          Faithfulness: <span className={textColor(stats.avgFaithfulness)}>{pct(stats.avgFaithfulness)}</span>
        </div>
        <div className="rounded-xl border border-border/70 p-2 text-xs">
          Format: <span className={textColor(stats.avgFormat)}>{pct(stats.avgFormat)}</span>
        </div>
        <div className="rounded-xl border border-border/70 p-2 text-xs">
          Readability: <span className={textColor(stats.avgReadability)}>{pct(stats.avgReadability)}</span>
        </div>
        <div className="rounded-xl border border-border/70 p-2 text-xs">
          Source UX: <span className={textColor(stats.avgSourceUx)}>{pct(stats.avgSourceUx)}</span>
        </div>
        <div className="rounded-xl border border-border/70 p-2 text-xs">
          Style repetition: <span className={textColor(1 - stats.avgStyleRepetition)}>{pct(1 - stats.avgStyleRepetition)}</span>
        </div>
        <div className="rounded-xl border border-border/70 p-2 text-xs">
          Safety: <span className={textColor(stats.avgSafety)}>{pct(stats.avgSafety)}</span>
        </div>
      </div>

      <div className="mt-3">
        <p className="mb-2 text-xs text-fg/60">Trend (10 dernières réponses)</p>
        <div className="flex h-12 items-end gap-1 rounded-lg border border-border/70 bg-card/40 p-2">
          {stats.trend.map((score, idx) => {
            const h = 15 + Math.round(clamp01(score) * 85);
            return <div key={`trend-${idx}`} className={`w-2 rounded-sm ${score >= 0.85 ? "bg-emerald-500" : score >= 0.7 ? "bg-amber-500" : "bg-rose-500"}`} style={{ height: `${h}%` }} />;
          })}
        </div>
      </div>
    </section>
  );
}

