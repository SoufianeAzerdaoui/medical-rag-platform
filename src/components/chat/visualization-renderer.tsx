"use client";

import type { VisualizationDatum, VisualizationPayload } from "@/types/chat";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
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
  const hasRatio = data.some((item) => typeof item.reference_ratio === "number");
  return hasRatio ? "reference_ratio" : "value_numeric";
}

function isRenderableType(chartType: string): boolean {
  return chartType === "bar" || chartType === "line";
}

function displayType(chartType: string): string {
  if (chartType === "bar") return "graphique en barres";
  if (chartType === "line") return "graphique linéaire";
  return "visualisation";
}

function formatTooltipNumber(value: number | string | null | undefined): string {
  if (value === null || value === undefined || value === "") return "n/a";
  if (typeof value === "number") return Number.isFinite(value) ? value.toLocaleString("fr-FR", { maximumFractionDigits: 4 }) : "n/a";
  return String(value);
}

function TooltipContent({ active, payload }: { active?: boolean; payload?: Array<{ payload: VisualizationDatum }> }) {
  if (!active || !payload || payload.length === 0) return null;
  const datum = payload[0]?.payload || {};
  return (
    <div className="rounded-lg border border-slate-700 bg-slate-900/95 p-3 text-xs text-slate-100 shadow-lg">
      <p className="mb-1 font-semibold">{datum.analyte || "Analyte"}</p>
      <p>Valeur brute: {formatTooltipNumber(datum.value)} {datum.unit || ""}</p>
      <p>Référence: {datum.reference || "n/a"}</p>
      <p>Statut: {datum.status || "n/a"}</p>
      {typeof datum.reference_ratio === "number" ? <p>Ratio référence: {formatTooltipNumber(datum.reference_ratio)}</p> : null}
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
  const data = safeData(visualization, chartData);
  if (data.length === 0) return null;

  const chartType = String(chartData?.type || visualization?.type || "bar").trim().toLowerCase();
  if (!isRenderableType(chartType)) {
    return (
      <section className="mt-3 rounded-2xl border border-border bg-card/40 p-4">
        <p className="text-sm text-fg/85">Le type de visualisation demandé n’est pas encore pris en charge dans l’interface.</p>
      </section>
    );
  }

  const yField = resolveYField(visualization, chartData, data);
  const mixedUnits = hasMixedUnits(data);
  const title = String(chartData?.title || visualization?.title || `Visualisation: ${displayType(chartType)}`);
  const fromPrevious = String(visualization?.source || "").trim() === "previous_evidence_pack";
  const intro = fromPrevious
    ? "J’ai repris les résultats précédents et généré un graphique en barres."
    : "Voici le graphique en barres généré à partir des résultats retrouvés.";

  return (
    <section className="mt-3 rounded-2xl border border-slate-700/70 bg-slate-950/50 p-4">
      <p className="text-sm font-medium text-slate-100">{intro}</p>
      <h4 className="mt-1 text-sm text-slate-300">{title}</h4>
      {mixedUnits ? (
        <p className="mt-2 text-xs text-slate-400">
          Les unités biologiques étant différentes, le graphique utilise un ratio par rapport à la référence lorsque disponible.
        </p>
      ) : null}
      <div className="mt-3 h-72 w-full overflow-hidden rounded-xl border border-slate-800 bg-slate-900/60 p-2 sm:h-80">
        <ResponsiveContainer width="100%" height="100%">
          {chartType === "line" ? (
            <LineChart data={data} margin={{ top: 8, right: 16, left: 8, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="analyte" stroke="#cbd5e1" tick={{ fontSize: 12 }} interval={0} angle={-20} textAnchor="end" height={70} />
              <YAxis stroke="#cbd5e1" tick={{ fontSize: 12 }} />
              <Tooltip content={<TooltipContent />} />
              <Legend />
              <Line type="monotone" dataKey={yField} name={yField} stroke="#60a5fa" strokeWidth={2.2} dot={{ r: 4 }} />
            </LineChart>
          ) : (
            <BarChart data={data} margin={{ top: 8, right: 16, left: 8, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="analyte" stroke="#cbd5e1" tick={{ fontSize: 12 }} interval={0} angle={-20} textAnchor="end" height={70} />
              <YAxis stroke="#cbd5e1" tick={{ fontSize: 12 }} />
              <Tooltip content={<TooltipContent />} />
              <Legend />
              <Bar dataKey={yField} name={yField} fill="#38bdf8" radius={[6, 6, 0, 0]} />
            </BarChart>
          )}
        </ResponsiveContainer>
      </div>
    </section>
  );
}
