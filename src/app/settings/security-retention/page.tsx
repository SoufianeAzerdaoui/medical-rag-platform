"use client";

import { AlertTriangle, CheckCircle2, Loader2, RefreshCw, Shield, Trash2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { WorkspaceShell } from "@/components/layout/workspace-shell";
import {
  ApiError,
  getMonitoringSummaryApi,
  getSecurityStatusApi,
  runRetentionApi,
  type MonitoringSummaryResponse,
  type RetentionRunResponse,
  type SecurityStatusResponse,
} from "@/services/rag-api";
import { useAuthStore } from "@/store/auth-store";

const POLL_INTERVAL_MS = 8_000;

function Badge({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] ${
        ok
          ? "border-emerald-500/35 bg-emerald-500/10 text-emerald-500"
          : "border-amber-500/40 bg-amber-500/12 text-amber-500"
      }`}
    >
      {ok ? <CheckCircle2 size={11} /> : <AlertTriangle size={11} />}
      {label}
    </span>
  );
}

export default function SecurityRetentionPage() {
  const token = useAuthStore((s) => s.accessToken);
  const user = useAuthStore((s) => s.user);
  const role = String(user?.role || "user").toLowerCase();
  const isOps = ["admin", "ops", "data_manager", "medical_admin"].includes(role);

  const [security, setSecurity] = useState<SecurityStatusResponse | null>(null);
  const [summary, setSummary] = useState<MonitoringSummaryResponse | null>(null);
  const [retentionResult, setRetentionResult] = useState<RetentionRunResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [runningRetention, setRunningRetention] = useState<"dry" | "execute" | null>(null);
  const [hardDeleteDocs, setHardDeleteDocs] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confirmExecute, setConfirmExecute] = useState(false);
  const [lastUpdatedAt, setLastUpdatedAt] = useState<string | null>(null);

  const retentionRows = useMemo(() => {
    if (!security) return [];
    return [
      { label: "Jobs asynchrones", value: `${security.retention.jobs_days} jours` },
      { label: "Audit events", value: `${security.retention.audit_days} jours` },
      { label: "Documents registre", value: `${security.retention.docs_days} jours` },
      { label: "Audio", value: `${security.retention.audio_days} jours` },
      { label: "Logs", value: `${security.retention.logs_days} jours` },
      { label: "Tentatives login", value: `${security.retention.auth_attempts_days} jours` },
    ];
  }, [security]);

  const retentionAffectedCount = useMemo(() => {
    if (!retentionResult) return null;
    return (
      Number(retentionResult.jobs_deleted || 0) +
      Number(retentionResult.audit_deleted || 0) +
      Number(retentionResult.auth_attempts_deleted || 0) +
      Number(retentionResult.docs_registry_deleted || 0) +
      Number(retentionResult.docs_files_deleted || 0) +
      Number(retentionResult.audio_files_deleted || 0) +
      Number(retentionResult.log_files_deleted || 0)
    );
  }, [retentionResult]);

  async function refreshData(opts?: { silent?: boolean }) {
    if (!isOps) {
      setLoading(false);
      return;
    }
    const silent = Boolean(opts?.silent);
    if (silent) {
      setRefreshing(true);
    } else {
      setLoading(true);
    }
    setError(null);
    try {
      const [sec, sum] = await Promise.all([
        getSecurityStatusApi(token),
        getMonitoringSummaryApi(token),
      ]);
      setSecurity(sec);
      setSummary(sum);
      setLastUpdatedAt(new Date().toISOString());
    } catch (err) {
      const detail = err instanceof ApiError ? err.detail : "";
      setError(detail || "Impossible de charger les données Ops.");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }

  async function runDryRun() {
    setRunningRetention("dry");
    setError(null);
    try {
      const out = await runRetentionApi({ dryRun: true, hardDeleteDocs }, token);
      setRetentionResult(out);
      await refreshData({ silent: true });
    } catch (err) {
      const detail = err instanceof ApiError ? err.detail : "";
      setError(detail || "Dry-run retention échoué.");
    } finally {
      setRunningRetention(null);
    }
  }

  async function runExecute() {
    if (!confirmExecute) {
      setError("Confirme d’abord l’exécution de purge.");
      return;
    }
    setRunningRetention("execute");
    setError(null);
    try {
      const out = await runRetentionApi({ dryRun: false, hardDeleteDocs }, token);
      setRetentionResult(out);
      await refreshData({ silent: true });
    } catch (err) {
      const detail = err instanceof ApiError ? err.detail : "";
      setError(detail || "Exécution retention échouée.");
    } finally {
      setRunningRetention(null);
    }
  }

  useEffect(() => {
    void refreshData();
  }, [isOps]);

  useEffect(() => {
    if (!isOps) return;
    const id = window.setInterval(() => {
      void refreshData({ silent: true });
    }, POLL_INTERVAL_MS);
    return () => window.clearInterval(id);
  }, [token, isOps]);

  return (
    <WorkspaceShell
      title="Security & Retention"
      subtitle="Ops console: sécurité runtime, métriques pipeline, rétention contrôlée"
      breadcrumbs={["Clinical Assistant", "Settings", "Security & Retention"]}
      actions={[
        { href: "/chat", label: "Retour au chat" },
        { href: "/settings", label: "Settings" },
        { href: "/documents/upload", label: "Ingestion" },
      ]}
    >
      <main className="mx-auto max-w-6xl space-y-4 px-4 py-6">
        {!isOps ? (
          <section className="rounded-2xl border border-amber-500/35 bg-amber-500/10 p-4 text-sm text-amber-600">
            Accès réservé aux rôles opérateurs (admin/ops/data_manager/medical_admin).
          </section>
        ) : null}

        <section className="rounded-2xl border border-border/70 bg-card/[0.55] p-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="text-[11px] uppercase tracking-[0.14em] text-fg/58">Ops status</p>
              <h2 className="mt-1 text-base font-semibold text-fg">Sécurité & supervision runtime</h2>
              <p className="mt-1 text-xs text-fg/62">
                Dernière actualisation: {lastUpdatedAt ? new Date(lastUpdatedAt).toLocaleString("fr-FR") : "—"}
              </p>
            </div>
            <button
              type="button"
              onClick={() => void refreshData({ silent: true })}
              disabled={refreshing || loading}
              className="inline-flex items-center gap-1.5 rounded-lg border border-border/75 bg-card/[0.66] px-3 py-1.5 text-xs font-medium text-fg/82 transition hover:border-accent/30 hover:bg-card disabled:opacity-55"
            >
              {refreshing || loading ? <Loader2 size={13} className="animate-spin" /> : <RefreshCw size={13} />}
              Actualiser
            </button>
          </div>
          {error ? <p className="mt-3 text-xs text-rose-500">{error}</p> : null}
        </section>

        <section className="grid grid-cols-1 gap-4 xl:grid-cols-[1.2fr,1fr]">
          <article className="rounded-2xl border border-border/70 bg-card/[0.55] p-4">
            <div className="mb-3 flex items-center gap-2">
              <Shield size={16} className="text-accent" />
              <h3 className="text-sm font-semibold text-fg">Security runtime</h3>
            </div>
            {loading || !security ? (
              <p className="text-xs text-fg/62">Chargement…</p>
            ) : (
              <div className="space-y-3 text-xs">
                <div className="flex flex-wrap items-center gap-2">
                  <Badge ok={security.clamav.healthy} label={`ClamAV ${security.clamav.available ? "disponible" : "indisponible"}`} />
                  <Badge ok={security.sentry.configured} label={`Sentry ${security.sentry.configured ? "configuré" : "non configuré"}`} />
                  <Badge
                    ok={security.encryption.enabled && security.encryption.key_configured}
                    label={`Chiffrement ${security.encryption.enabled ? "actif" : "inactif"}`}
                  />
                  <Badge ok={security.jwt.rotation_previous_count > 0} label={`Rotation JWT: ${security.jwt.rotation_previous_count}`} />
                </div>
                <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                  <div className="rounded-lg border border-border/65 bg-card/[0.45] px-3 py-2">
                    <p className="text-fg/58">ClamAV commande</p>
                    <p className="mt-0.5 font-mono text-fg/88">{security.clamav.command}</p>
                    <p className="mt-0.5 text-fg/58">{security.clamav.version || "version non détectée"}</p>
                  </div>
                  <div className="rounded-lg border border-border/65 bg-card/[0.45] px-3 py-2">
                    <p className="text-fg/58">JWT</p>
                    <p className="mt-0.5 text-fg/88">Algo: {security.jwt.algorithm}</p>
                    <p className="text-fg/88">TTL: {security.jwt.expire_minutes} min</p>
                  </div>
                </div>
                <div className="rounded-lg border border-border/65 bg-card/[0.45] px-3 py-2">
                  <p className="mb-1 text-fg/58">Rate limits</p>
                  <p className="text-fg/88">
                    Fenêtre {security.rate_limits.window_seconds}s · Auth {security.rate_limits.auth_per_window} · Chat {security.rate_limits.chat_per_window} · Upload {security.rate_limits.upload_per_window}
                  </p>
                  <p className="text-fg/88">
                    Login lock: {security.rate_limits.login_max_failures} échecs / {security.rate_limits.login_block_seconds}s
                  </p>
                </div>
              </div>
            )}
          </article>

          <article className="rounded-2xl border border-border/70 bg-card/[0.55] p-4">
            <div className="mb-3 flex items-center gap-2">
              <RefreshCw size={16} className="text-accent" />
              <h3 className="text-sm font-semibold text-fg">Monitoring live</h3>
            </div>
            {loading || !summary ? (
              <p className="text-xs text-fg/62">Chargement…</p>
            ) : (
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="rounded-lg border border-border/65 bg-card/[0.45] px-3 py-2">
                  <p className="text-fg/58">Queue depth</p>
                  <p className="mt-0.5 text-lg font-semibold text-fg">{summary.queue_depth}</p>
                </div>
                <div className="rounded-lg border border-border/65 bg-card/[0.45] px-3 py-2">
                  <p className="text-fg/58">Temps moyen pipeline</p>
                  <p className="mt-0.5 text-lg font-semibold text-fg">{summary.avg_pipeline_seconds}s</p>
                </div>
                <div className="rounded-lg border border-border/65 bg-card/[0.45] px-3 py-2">
                  <p className="text-fg/58">Succès pipeline</p>
                  <p className="mt-0.5 text-lg font-semibold text-emerald-500">{summary.pipeline_success_total}</p>
                </div>
                <div className="rounded-lg border border-border/65 bg-card/[0.45] px-3 py-2">
                  <p className="text-fg/58">Échecs pipeline</p>
                  <p className="mt-0.5 text-lg font-semibold text-rose-500">{summary.pipeline_failure_total}</p>
                </div>
                <div className="col-span-2 rounded-lg border border-border/65 bg-card/[0.45] px-3 py-2">
                  <p className="text-fg/58">Erreurs indexation</p>
                  <p className="mt-0.5 text-lg font-semibold text-amber-500">{summary.indexing_errors_total}</p>
                </div>
              </div>
            )}
          </article>
        </section>

        <section className="rounded-2xl border border-border/70 bg-card/[0.55] p-4">
          <div className="mb-3 flex items-center gap-2">
            <Trash2 size={16} className="text-accent" />
            <h3 className="text-sm font-semibold text-fg">Retention console</h3>
          </div>
          {loading || !security ? (
            <p className="text-xs text-fg/62">Chargement…</p>
          ) : (
            <div className="space-y-3">
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
                {retentionRows.map((row) => (
                  <div key={row.label} className="rounded-lg border border-border/65 bg-card/[0.45] px-3 py-2 text-xs">
                    <p className="text-fg/58">{row.label}</p>
                    <p className="mt-0.5 font-medium text-fg">{row.value}</p>
                  </div>
                ))}
              </div>

              <div className="rounded-lg border border-border/65 bg-card/[0.45] p-3 text-xs">
                <label className="inline-flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={hardDeleteDocs}
                    onChange={(e) => setHardDeleteDocs(e.target.checked)}
                    className="h-3.5 w-3.5 accent-cyan-500"
                  />
                  Hard delete fichiers docs (en plus du registre)
                </label>
                <label className="mt-2 inline-flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={confirmExecute}
                    onChange={(e) => setConfirmExecute(e.target.checked)}
                    className="h-3.5 w-3.5 accent-rose-500"
                  />
                  Je confirme l’exécution réelle de la purge
                </label>
                <div className="mt-3 flex flex-wrap gap-2">
                  <button
                    type="button"
                    onClick={() => void runDryRun()}
                    disabled={runningRetention !== null || !isOps}
                    className="inline-flex items-center gap-1.5 rounded-lg border border-accent/40 bg-accent/12 px-3 py-1.5 text-xs font-medium text-accent transition hover:bg-accent/18 disabled:opacity-55"
                  >
                    {runningRetention === "dry" ? <Loader2 size={12} className="animate-spin" /> : null}
                    Preview dry-run
                  </button>
                  <button
                    type="button"
                    onClick={() => void runExecute()}
                    disabled={runningRetention !== null || !isOps}
                    className="inline-flex items-center gap-1.5 rounded-lg border border-rose-500/40 bg-rose-500/12 px-3 py-1.5 text-xs font-medium text-rose-500 transition hover:bg-rose-500/20 disabled:opacity-55"
                  >
                    {runningRetention === "execute" ? <Loader2 size={12} className="animate-spin" /> : null}
                    Exécuter purge
                  </button>
                </div>
                {retentionResult ? (
                  <p className="mt-2 text-xs text-fg/72">
                    {retentionAffectedCount === 0
                      ? "Aucun élément expiré à purger avec les règles actuelles."
                      : `${retentionAffectedCount} élément(s) expiré(s) détecté(s).`}
                  </p>
                ) : null}
              </div>

              {retentionResult ? (
                <div className="rounded-lg border border-border/65 bg-card/[0.45] p-3 text-xs">
                  <p className="mb-2 font-medium text-fg">Résultat retention ({retentionResult.dry_run ? "dry-run" : "exécution"})</p>
                  <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                    <div><p className="text-fg/58">Jobs</p><p className="text-fg">{retentionResult.jobs_deleted}</p></div>
                    <div><p className="text-fg/58">Audit</p><p className="text-fg">{retentionResult.audit_deleted}</p></div>
                    <div><p className="text-fg/58">Auth attempts</p><p className="text-fg">{retentionResult.auth_attempts_deleted}</p></div>
                    <div><p className="text-fg/58">Registry docs</p><p className="text-fg">{retentionResult.docs_registry_deleted}</p></div>
                    <div><p className="text-fg/58">Docs files</p><p className="text-fg">{retentionResult.docs_files_deleted}</p></div>
                    <div><p className="text-fg/58">Audio files</p><p className="text-fg">{retentionResult.audio_files_deleted}</p></div>
                    <div><p className="text-fg/58">Log files</p><p className="text-fg">{retentionResult.log_files_deleted}</p></div>
                    <div><p className="text-fg/58">Audit immutable</p><p className="text-fg">{retentionResult.audit_delete_blocked_immutable ? "bloqué" : "ok"}</p></div>
                  </div>
                </div>
              ) : null}
            </div>
          )}
        </section>
      </main>
    </WorkspaceShell>
  );
}
