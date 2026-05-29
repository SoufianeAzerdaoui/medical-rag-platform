"use client";

import type { VisualizationDatum, VisualizationPayload } from "@/types/chat";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

function hasMixedUnits(data: VisualizationDatum[]): boolean {
  const units = new Set(
    data
      .map((item) => String(item.unit || "").trim().toLowerCase())
      .filter((item) => item.length > 0),
  );
  return units.size > 1;
}

function safeData(visualization?: VisualizationPayload, chartData?: VisualizationPayload): VisualizationDatum[] {
  const fromChartData = Array.isArray(chartData?.data) ? chartData.data : [];
  const fromVisualization = Array.isArray(visualization?.data) ? visualization.data : [];
  return (fromChartData.length > 0 ? fromChartData : fromVisualization).filter(
    (item) => typeof item === "object" && item !== null,
  );
}

function resolveYField(visualization?: VisualizationPayload, chartData?: VisualizationPayload, data: VisualizationDatum[] = []): string {
  const explicit = String(chartData?.y_field || visualization?.y_field || "").trim();
  if (explicit) return explicit;
  const hasDeviation = data.some((item) => typeof item.reference_deviation === "number");
  if (hasDeviation) return "reference_deviation";
  const hasRatio = data.some((item) => typeof item.reference_ratio === "number");
  return hasRatio ? "reference_ratio" : "value_numeric";
}

function isRenderableType(chartType: string): boolean {
  return chartType === "bar" || chartType === "line";
}

function displayType(chartType: string): string {
  if (chartType === "bar") return "graphique en barres";
  if (chartType === "line") return "courbe";
  if (chartType === "radar") return "graphique radar";
  if (chartType === "scatter") return "nuage de points";
  if (chartType === "heatmap") return "heatmap";
  return "visualisation";
}

function formatTooltipNumber(value: number | string | null | undefined): string {
  if (value === null || value === undefined || value === "") return "n/a";
  if (typeof value === "number") return Number.isFinite(value) ? value.toLocaleString("fr-FR", { maximumFractionDigits: 4 }) : "n/a";
  return String(value);
}

function formatDeviation(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "non calculable";
  const pct = value * 100;
  const sign = pct > 0 ? "+" : "";
  return `${sign}${pct.toLocaleString("fr-FR", { maximumFractionDigits: 1 })}%`;
}

function TooltipContent({ active, payload }: { active?: boolean; payload?: Array<{ payload: VisualizationDatum }> }) {
  if (!active || !payload || payload.length === 0) return null;
  const datum = payload[0]?.payload || {};
  const rawValue = datum.raw_value ?? formatTooltipNumber(datum.value);
  const deviationText = datum.deviation_label || formatDeviation(datum.reference_deviation ?? null);
  return (
    <div className="rounded-lg border border-border/80 bg-card/[0.96] p-3 text-xs text-fg shadow-lg">
      <p className="mb-1 font-semibold">{datum.analyte || "Analyte"}</p>
      <p>Valeur : {rawValue} {datum.unit || ""}</p>
      <p>Référence: {datum.reference || "n/a"}</p>
      <p>Statut: {datum.status || "n/a"}</p>
      <p>Écart : {deviationText}</p>
      {datum.source_label ? <p>Source : {datum.source_label}</p> : null}
    </div>
  );
}

export function VisualizationRenderer({
  visualization,
  chartData,
}: {
  visualization?: VisualizationPayload;
  chartData?: VisualizationPayload;
}) {
  const axisColor = "hsl(var(--text-soft))";
  const gridColor = "hsl(var(--border-strong))";
  const referenceLineColor = "hsl(var(--text-soft))";
  const seriesLineColor = "hsl(var(--primary))";
  const seriesBarColor = "hsl(var(--primary))";

  const data = safeData(visualization, chartData);
  const renderedType = String(
    chartData?.rendered_type || visualization?.rendered_type || chartData?.type || visualization?.type || "",
  )
    .trim()
    .toLowerCase();
  const requestedLabel = String(visualization?.requested_label || displayType(visualization?.requested_type || "unknown"));
  const renderedLabel = String(visualization?.rendered_label || displayType(renderedType || visualization?.type || "unknown"));
  const fallbackUsed = Boolean(visualization?.fallback_used);
  const fallbackReason = String(visualization?.fallback_reason || visualization?.reason || "").trim();

  if (!renderedType && data.length === 0) return null;

  if (!isRenderableType(renderedType)) {
    return (
      <section className="mt-3 rounded-2xl border border-border bg-card/40 p-4">
        <p className="text-sm text-fg/90">
          {fallbackUsed
            ? `Graphique demandé : ${requestedLabel}. Rendu affiché : ${renderedLabel || "format alternatif"}.`
            : "Le type de visualisation demandé n’est pas encore pris en charge dans l’interface."}
        </p>
        {fallbackReason ? <p className="mt-2 text-xs text-fg/70">Raison : {fallbackReason}</p> : null}
        {data.length > 0 ? (
          <div className="mt-3 overflow-x-auto rounded-lg border border-border/70">
            <table className="min-w-full text-xs">
              <thead className="bg-card/70 text-fg/80">
                <tr>
                  <th className="px-3 py-2 text-left">Analyte</th>
                  <th className="px-3 py-2 text-left">Valeur</th>
                  <th className="px-3 py-2 text-left">Unité</th>
                  <th className="px-3 py-2 text-left">Référence</th>
                </tr>
              </thead>
              <tbody>
                {data.map((row, idx) => (
                  <tr key={`${row.analyte || "analyte"}-${idx}`} className="border-t border-border/50">
                    <td className="px-3 py-2">{row.analyte || "n/a"}</td>
                    <td className="px-3 py-2">{formatTooltipNumber(row.value)}</td>
                    <td className="px-3 py-2">{row.unit || "n/a"}</td>
                    <td className="px-3 py-2">{row.reference || "n/a"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </section>
    );
  }

  const yField = resolveYField(visualization, chartData, data);
  const metricLabel = String(chartData?.metric_label || visualization?.metric_label || (yField === "reference_deviation" ? "Écart normalisé à la référence" : "Valeur mesurée"));
  const metricReason = String(chartData?.metric_reason || visualization?.metric_reason || "").trim();
  const seriesData =
    yField === "reference_deviation"
      ? data.filter((row) => row.metric_available !== false && typeof row.reference_deviation === "number")
      : data.filter((row) => typeof row.value_numeric === "number" || typeof row.value === "number");
  const notCalculableCount = yField === "reference_deviation" ? Math.max(0, data.length - seriesData.length) : 0;

  if (seriesData.length === 0) {
    return (
      <section className="mt-3 rounded-2xl border border-border bg-card/40 p-4">
        <p className="text-sm text-fg/90">Aucune barre calculable pour l’écart normalisé à la référence.</p>
        {data.length > 0 ? <p className="mt-2 text-xs text-fg/70">Les valeurs brutes restent disponibles dans le tableau des données.</p> : null}
      </section>
    );
  }

  const mixedUnits = hasMixedUnits(data);
  const title = String(chartData?.title || visualization?.title || `Visualisation: ${displayType(renderedType)}`);
  const fromPrevious = String(visualization?.source || "").trim() === "previous_evidence_pack";
  const intro = fallbackUsed
    ? `Graphique demandé : ${requestedLabel}. Rendu affiché : ${renderedLabel}.`
    : fromPrevious
      ? `J’ai repris les résultats précédents et généré un ${renderedLabel}.`
      : `Voici le ${renderedLabel} généré à partir des résultats retrouvés.`;

  return (
    <section className="mt-3 rounded-2xl border border-border/75 bg-card/[0.62] p-4">
      <p className="text-sm font-medium text-fg">{intro}</p>
      {fallbackUsed && fallbackReason ? <p className="mt-1 text-xs text-fg/72">Raison : {fallbackReason}</p> : null}
      <h4 className="mt-1 text-sm text-fg/75">{title}</h4>
      {metricReason ? (
        <p className="mt-2 text-xs text-fg/68">{metricReason}</p>
      ) : mixedUnits ? (
        <p className="mt-2 text-xs text-fg/68">
          Les unités biologiques étant différentes, l’axe vertical représente l’écart normalisé à la référence.
        </p>
      ) : null}
      {notCalculableCount > 0 ? (
        <p className="mt-1 text-xs text-fg/62">{notCalculableCount} point(s) non calculable(s) ne sont pas tracés.</p>
      ) : null}
      <div className="mt-3 h-72 w-full overflow-hidden rounded-xl border border-border/75 bg-card/[0.82] p-2 sm:h-80">
        <ResponsiveContainer width="100%" height="100%">
          {renderedType === "line" ? (
            <LineChart data={seriesData} margin={{ top: 8, right: 16, left: 8, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
              <XAxis dataKey="analyte" stroke={axisColor} tick={{ fontSize: 12 }} interval={0} angle={-20} textAnchor="end" height={70} />
              <YAxis
                stroke={axisColor}
                tick={{ fontSize: 12 }}
                tickFormatter={yField === "reference_deviation" ? (v) => `${(Number(v) * 100).toFixed(0)}%` : undefined}
                label={{ value: metricLabel, angle: -90, position: "insideLeft", fill: axisColor, fontSize: 12 }}
              />
              {yField === "reference_deviation" ? <ReferenceLine y={0} stroke={referenceLineColor} strokeDasharray="4 4" /> : null}
              <Tooltip content={<TooltipContent />} />
              <Legend />
              <Line type="monotone" dataKey={yField} name={metricLabel} stroke={seriesLineColor} strokeWidth={2.2} dot={{ r: 4 }} />
            </LineChart>
          ) : (
            <BarChart data={seriesData} margin={{ top: 8, right: 16, left: 8, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
              <XAxis dataKey="analyte" stroke={axisColor} tick={{ fontSize: 12 }} interval={0} angle={-20} textAnchor="end" height={70} />
              <YAxis
                stroke={axisColor}
                tick={{ fontSize: 12 }}
                tickFormatter={yField === "reference_deviation" ? (v) => `${(Number(v) * 100).toFixed(0)}%` : undefined}
                label={{ value: metricLabel, angle: -90, position: "insideLeft", fill: axisColor, fontSize: 12 }}
              />
              {yField === "reference_deviation" ? <ReferenceLine y={0} stroke={referenceLineColor} strokeDasharray="4 4" /> : null}
              <Tooltip content={<TooltipContent />} />
              <Legend />
              <Bar dataKey={yField} name={metricLabel} fill={seriesBarColor} radius={[6, 6, 0, 0]} />
            </BarChart>
          )}
        </ResponsiveContainer>
      </div>
    </section>
  );
}
