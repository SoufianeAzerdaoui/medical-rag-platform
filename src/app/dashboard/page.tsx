"use client";

import { Activity, AlertCircle, Clock3, Database, RefreshCw, Server, Wifi, Workflow } from "lucide-react";
import { useEffect, useMemo, useState, type ComponentType, type ReactNode } from "react";
import { WorkspaceShell } from "@/components/layout/workspace-shell";
import { ApiError, getMonitoringSummaryApi, healthcheck, listDocumentsApi, type MonitoringSummaryResponse } from "@/services/rag-api";
import { useAuthStore } from "@/store/auth-store";

type BackendState = "online" | "offline" | "checking";
type MonitoringAccessState = "ok" | "forbidden" | "unavailable";

export default function DashboardPage() {
  const token = useAuthStore((s) => s.accessToken);
  const [backendStatus, setBackendStatus] = useState<BackendState>("checking");
  const [summary, setSummary] = useState<MonitoringSummaryResponse | null>(null);
  const [documentsCount, setDocumentsCount] = useState<number>(0);
  const [monitoringAccess, setMonitoringAccess] = useState<MonitoringAccessState>("ok");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState<string | null>(null);

  async function refreshDashboard(silent = false) {
    if (!silent) setLoading(true);
    setError(null);
    setNotice(null);
    try {
      const [healthResult, docsResult, monitoringResult] = await Promise.allSettled([
        healthcheck(),
        listDocumentsApi(token),
        getMonitoringSummaryApi(token),
      ]);

      if (healthResult.status === "fulfilled") {
        setBackendStatus(healthResult.value === "online" ? "online" : "offline");
      } else {
        setBackendStatus("offline");
      }

      if (docsResult.status === "fulfilled") {
        setDocumentsCount(docsResult.value.length);
      } else {
        setDocumentsCount(0);
        setNotice("Liste documents indisponible momentanément.");
      }

      if (monitoringResult.status === "fulfilled") {
        setSummary(monitoringResult.value);
        setMonitoringAccess("ok");
      } else {
        const reason = monitoringResult.reason;
        if (reason instanceof ApiError && (reason.status === 401 || reason.status === 403)) {
          setSummary(null);
          setMonitoringAccess("forbidden");
          setNotice("Accès monitoring réservé aux rôles ops/admin. Vue dashboard partielle affichée.");
        } else {
          setSummary(null);
          setMonitoringAccess("unavailable");
          setNotice("Monitoring indisponible. Vue minimale affichée.");
        }
      }

      if (healthResult.status !== "fulfilled" && docsResult.status !== "fulfilled") {
        setError("Impossible de charger les métriques dashboard.");
      }

      setLastRefresh(new Date().toLocaleString("fr-FR"));
    } catch (error) {
      const detail = error instanceof ApiError ? error.detail : "";
      setError(detail || "Impossible de charger les métriques dashboard.");
      setBackendStatus("offline");
    } finally {
      if (!silent) setLoading(false);
    }
  }

  useEffect(() => {
    let mounted = true;
    void (async () => {
      if (!mounted) return;
      await refreshDashboard(false);
    })();
    const timer = window.setInterval(() => void refreshDashboard(true), 30_000);
    return () => {
      mounted = false;
      window.clearInterval(timer);
    };
  }, [token]);

  const cards = useMemo(() => {
    const s = summary;
    return [
      { icon: Database, label: "Documents indexés", value: String(documentsCount), tone: "neutral" as const },
      { icon: Workflow, label: "Queue depth", value: String(s?.queue_depth ?? "—"), tone: "neutral" as const },
      { icon: Clock3, label: "Temps moyen pipeline", value: s ? `${Math.max(0, Number(s.avg_pipeline_seconds || 0)).toFixed(1)}s` : "—", tone: "neutral" as const },
      { icon: Activity, label: "Pipelines succès", value: String(s?.pipeline_success_total ?? "—"), tone: "good" as const },
      { icon: AlertCircle, label: "Pipelines échec", value: String(s?.pipeline_failure_total ?? "—"), tone: "bad" as const },
      { icon: Server, label: "Erreurs indexation", value: String(s?.indexing_errors_total ?? "—"), tone: (s?.indexing_errors_total || 0) > 0 ? ("bad" as const) : ("good" as const) },
      { icon: Wifi, label: "Backend", value: backendStatus === "checking" ? "Checking" : backendStatus === "online" ? "Online" : "Offline", tone: backendStatus === "online" ? ("good" as const) : backendStatus === "offline" ? ("bad" as const) : ("neutral" as const) },
    ];
  }, [backendStatus, documentsCount, summary]);

  return (
    <WorkspaceShell
      title="Dashboard clinique"
      subtitle="Métriques runtime minimales avant Grafana"
      breadcrumbs={["Clinical Assistant", "Dashboard clinique"]}
      actions={[
        { href: "/chat", label: "Retour au chat" },
        { href: "/documents/upload", label: "Importer document" },
        { href: "/chat", label: "Nouvelle conversation" },
      ]}
    >
      <main className="mx-auto max-w-7xl space-y-5 px-5 py-6 sm:px-6">
        <section className="rounded-xl border border-border/70 bg-card/[0.55] p-4">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h2 className="text-sm font-semibold">Vue minimaliste observabilité</h2>
              <p className="mt-1 text-xs text-fg/65">
                Source backend: <code>/monitoring/summary</code>, <code>/documents</code>, <code>/health</code>.
              </p>
              <p className="mt-1 text-xs text-fg/58">Dernière actualisation: {lastRefresh || "—"}</p>
            </div>
            <button
              type="button"
              onClick={() => void refreshDashboard(false)}
              disabled={loading}
              className="inline-flex items-center gap-1 rounded-md border border-border/75 bg-card/[0.7] px-3 py-1.5 text-xs font-medium text-fg/82 transition hover:bg-card disabled:opacity-60"
            >
              <RefreshCw size={13} className={loading ? "animate-spin" : ""} />
              {loading ? "Actualisation…" : "Actualiser"}
            </button>
          </div>
        </section>

        {error ? (
          <section className="rounded-lg border border-rose-500/35 bg-rose-500/10 px-4 py-3 text-sm text-rose-300">
            {error}
          </section>
        ) : null}

        {notice ? (
          <section className="rounded-lg border border-amber-500/35 bg-amber-500/10 px-4 py-3 text-sm text-amber-200">
            {notice}
          </section>
        ) : null}

        <section className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {cards.map((card) => (
            <KpiCard key={card.label} icon={card.icon} label={card.label} value={card.value} tone={card.tone} />
          ))}
        </section>

        <Panel title="État runtime">
          <ul className="space-y-2 text-sm">
            <li className="flex items-center justify-between rounded-md border border-border/65 bg-card/[0.46] px-3 py-2">
              <span className="text-fg/82">API backend</span>
              <span className={backendStatus === "online" ? "text-emerald-400" : backendStatus === "offline" ? "text-rose-400" : "text-fg/65"}>
                {backendStatus === "checking" ? "Checking" : backendStatus === "online" ? "OK" : "Issue"}
              </span>
            </li>
            <li className="flex items-center justify-between rounded-md border border-border/65 bg-card/[0.46] px-3 py-2">
              <span className="text-fg/82">Monitoring summary</span>
              <span
                className={
                  monitoringAccess === "ok"
                    ? "text-emerald-400"
                    : monitoringAccess === "forbidden"
                      ? "text-amber-300"
                      : "text-fg/65"
                }
              >
                {monitoringAccess === "ok" ? "OK" : monitoringAccess === "forbidden" ? "Restreint" : "N/A"}
              </span>
            </li>
            <li className="flex items-center justify-between rounded-md border border-border/65 bg-card/[0.46] px-3 py-2">
              <span className="text-fg/82">Grafana readiness</span>
              <span className="text-sky-300">Ready (metrics endpoint available)</span>
            </li>
          </ul>
        </Panel>
      </main>
    </WorkspaceShell>
  );
}

function KpiCard({
  icon: Icon,
  label,
  value,
  tone = "neutral",
}: {
  icon: ComponentType<{ size?: string | number; className?: string }>;
  label: string;
  value: string;
  tone?: "good" | "bad" | "neutral";
}) {
  return (
    <article className="rounded-lg border border-border/70 bg-card/[0.55] px-4 py-3">
      <div className="mb-2 flex items-center gap-2">
        <Icon size={14} className="text-accent" />
        <p className="text-xs font-medium uppercase tracking-[0.12em] text-fg/58">{label}</p>
      </div>
      <p className={tone === "good" ? "text-2xl font-semibold text-emerald-400" : tone === "bad" ? "text-2xl font-semibold text-rose-400" : "text-2xl font-semibold text-fg"}>
        {value}
      </p>
    </article>
  );
}

function Panel({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="rounded-lg border border-border/70 bg-card/[0.55] p-4">
      <div className="mb-3 flex items-center gap-2">
        <Activity size={15} className="text-accent" />
        <h2 className="text-sm font-semibold text-fg">{title}</h2>
      </div>
      {children}
    </section>
  );
}
