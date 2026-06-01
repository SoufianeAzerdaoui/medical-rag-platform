"use client";

import {
  CheckCircle2,
  Circle,
  Clock3,
  Download,
  Eye,
  FileCheck2,
  FileText,
  Files,
  FolderSearch,
  History,
  ListFilter,
  Loader2,
  Play,
  RefreshCw,
  Search,
  ShieldCheck,
  Sparkles,
  UploadCloud,
  X,
  XCircle,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { WorkspaceShell } from "@/components/layout/workspace-shell";
import { useChatActions } from "@/hooks/use-chat-actions";
import {
  ApiError,
  discoverDocsApi,
  downloadIngestionReportApi,
  getDocumentTimelineApi,
  getIngestionJobStatusApi,
  resyncDocsRegistryApi,
  setDuplicateOverrideApi,
  startDocsIngestionJobApi,
  type DocumentTimelineEvent,
  type DocsDiscoveryRecord,
  type UploadResponse,
  uploadDocumentsApi,
} from "@/services/rag-api";
import { useAuthStore } from "@/store/auth-store";

type PipelineState = "idle" | "running" | "success" | "error";
type StepStatus = "pending" | "active" | "done" | "error";
type DocsStatusFilter = "all" | "new" | "indexed" | "duplicate" | "whitelist";

type PipelineStep = {
  id: string;
  title: string;
  description: string;
};

const PIPELINE_STEPS: PipelineStep[] = [
  { id: "extract", title: "Extraction", description: "Extraction structurée des données cliniques depuis PDF." },
  { id: "chunk", title: "Chunking", description: "Découpage médical des résultats, tableaux et sections utiles." },
  { id: "anonymize", title: "Anonymisation", description: "Masquage des données sensibles avant indexation." },
  { id: "index", title: "Indexation", description: "Index vectoriel + keyword + métadonnées pour retrieval." },
  { id: "ready", title: "RAG prêt", description: "Le document est prêt pour résumé et interrogation." },
];

function stepStatus(state: PipelineState, stepIndex: number, activeIndex: number): StepStatus {
  if (state === "idle") return "pending";
  if (state === "success") return "done";
  if (state === "error") {
    if (stepIndex < activeIndex) return "done";
    if (stepIndex === activeIndex) return "error";
    return "pending";
  }
  if (stepIndex < activeIndex) return "done";
  if (stepIndex === activeIndex) return "active";
  return "pending";
}

export default function UploadDocumentsPage() {
  const token = useAuthStore((s) => s.accessToken);
  const { sendMessage, sending } = useChatActions();
  const router = useRouter();

  const [files, setFiles] = useState<File[]>([]);
  const [state, setState] = useState<PipelineState>("idle");
  const [activeStep, setActiveStep] = useState(0);
  const [message, setMessage] = useState("");
  const [result, setResult] = useState<UploadResponse | null>(null);
  const [pipelineError, setPipelineError] = useState<string | null>(null);
  const [docsScan, setDocsScan] = useState<DocsDiscoveryRecord[]>([]);
  const [docsScanLoading, setDocsScanLoading] = useState(false);
  const [selectedDocFilenames, setSelectedDocFilenames] = useState<string[]>([]);
  const [statusFilter, setStatusFilter] = useState<DocsStatusFilter>("all");
  const [selectedDuplicateDoc, setSelectedDuplicateDoc] = useState<DocsDiscoveryRecord | null>(null);
  const [duplicateOverrideReason, setDuplicateOverrideReason] = useState("");
  const [duplicateOverrideLoading, setDuplicateOverrideLoading] = useState(false);
  const [bulkActionLoading, setBulkActionLoading] = useState(false);
  const [reportExportLoading, setReportExportLoading] = useState<"csv" | "pdf" | null>(null);
  const [timelineLoading, setTimelineLoading] = useState(false);
  const [timelineEvents, setTimelineEvents] = useState<DocumentTimelineEvent[]>([]);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [jobStartedAt, setJobStartedAt] = useState<string | null>(null);
  const [jobFinishedAt, setJobFinishedAt] = useState<string | null>(null);
  const [jobLastCheckAt, setJobLastCheckAt] = useState<string | null>(null);
  const [jobElapsedSeconds, setJobElapsedSeconds] = useState<number | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const pollingCancelledRef = useRef(false);

  const hasFiles = files.length > 0;
  const canRun = hasFiles && state !== "running";
  const ingestedRows = result?.ingested || [];
  const skippedRows = result?.skipped || [];
  const selectedDocsCount = selectedDocFilenames.length;
  const discoveredCount = docsScan.length;
  const newDocsCount = docsScan.filter((item) => !item.already_indexed).length;
  const indexedDocsCount = docsScan.filter((item) => item.already_indexed).length;
  const duplicateDocsCount = docsScan.filter((item) => item.is_duplicate && !item.duplicate_override).length;
  const whitelistedDuplicateCount = docsScan.filter((item) => item.is_duplicate && item.duplicate_override).length;
  const selectedDocsEligibleCount = selectedDocFilenames.filter((name) => {
    const row = docsScan.find((item) => item.filename === name);
    return row ? !row.blocked : false;
  }).length;
  const canRunDocsPipeline = selectedDocsEligibleCount > 0 && state !== "running";
  const selectedDuplicateCount = selectedDocFilenames.filter((name) => {
    const row = docsScan.find((item) => item.filename === name);
    return Boolean(row?.is_duplicate);
  }).length;
  const running = state === "running";
  const completedSteps = state === "success" ? PIPELINE_STEPS.length : Math.min(activeStep, PIPELINE_STEPS.length - 1);
  const progressPercent = Math.round((completedSteps / PIPELINE_STEPS.length) * 100);
  const surfaceCardClass =
    "rounded-2xl border border-border/70 bg-card/[0.55] p-4 shadow-[0_14px_36px_rgba(2,8,23,0.14)] transition-all duration-200";
  const primaryActionClass =
    "inline-flex items-center gap-1 rounded-lg border border-accent/40 bg-accent/12 px-3 py-1.5 text-xs font-medium text-accent transition-all duration-200 hover:bg-accent/18 active:scale-[0.985] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/45 disabled:opacity-55";
  const secondaryActionClass =
    "rounded-lg border border-border/75 bg-card/[0.66] px-2.5 py-1.5 text-xs text-fg/78 transition-all duration-200 hover:border-accent/30 hover:bg-card active:scale-[0.985] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/35 disabled:opacity-55";
  const subtleActionClass =
    "inline-flex items-center gap-1 rounded-lg border border-border/75 bg-card/[0.7] px-2.5 py-1.5 text-xs text-fg/82 transition-all duration-200 hover:border-accent/30 hover:bg-card active:scale-[0.985] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/35 disabled:opacity-60";

  const runningStepText = useMemo(() => {
    return PIPELINE_STEPS[Math.min(activeStep, PIPELINE_STEPS.length - 1)]?.description || "";
  }, [activeStep]);

  const filteredDocsScan = useMemo(() => {
    return docsScan.filter((item) => {
      if (statusFilter === "all") return true;
      if (statusFilter === "new") return !item.already_indexed;
      if (statusFilter === "indexed") return item.already_indexed;
      if (statusFilter === "duplicate") return item.is_duplicate && !item.duplicate_override;
      if (statusFilter === "whitelist") return item.is_duplicate && item.duplicate_override;
      return true;
    });
  }, [docsScan, statusFilter]);

  function clearTimer() {
    if (!timerRef.current) return;
    clearInterval(timerRef.current);
    timerRef.current = null;
  }

  useEffect(() => {
    return () => {
      pollingCancelledRef.current = true;
      clearTimer();
    };
  }, []);

  function onFilesPicked(fileList: FileList | null) {
    if (!fileList) return;
    const selected = Array.from(fileList).filter((file) => file.name.toLowerCase().endsWith(".pdf"));
    setFiles(selected);
    setResult(null);
    setPipelineError(null);
    setMessage(selected.length > 0 ? `${selected.length} PDF prêt(s) pour ingestion.` : "Aucun PDF valide sélectionné.");
    setState("idle");
    setActiveStep(0);
  }

  function formatBytes(value: number): string {
    if (!Number.isFinite(value) || value <= 0) return "0 KB";
    const kb = value / 1024;
    if (kb < 1024) return `${Math.max(1, Math.round(kb))} KB`;
    return `${(kb / 1024).toFixed(1)} MB`;
  }

  function formatDate(value: string): string {
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return "Date inconnue";
    return parsed.toLocaleString("fr-FR", { dateStyle: "short", timeStyle: "short" });
  }

  function closeDuplicateDetails() {
    setSelectedDuplicateDoc(null);
    setDuplicateOverrideReason("");
    setTimelineEvents([]);
    setTimelineLoading(false);
  }

  async function openDocumentDetails(item: DocsDiscoveryRecord) {
    setSelectedDuplicateDoc(item);
    setTimelineEvents([]);
    setTimelineLoading(true);
    try {
      const payload = await getDocumentTimelineApi(item.filename, token);
      setTimelineEvents(payload.events || []);
    } catch (error) {
      const detail = error instanceof ApiError ? error.detail : "";
      setPipelineError(detail || "Impossible de charger la timeline document.");
    } finally {
      setTimelineLoading(false);
    }
  }

  function formatDateTimeShort(value: string | null): string {
    if (!value) return "—";
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return "—";
    return parsed.toLocaleString("fr-FR", { dateStyle: "short", timeStyle: "medium" });
  }

  function formatDuration(seconds: number | null): string {
    if (seconds === null || !Number.isFinite(seconds) || seconds < 0) return "—";
    const total = Math.floor(seconds);
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    if (h > 0) return `${h}h ${m.toString().padStart(2, "0")}m`;
    return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  }

  function stateBadgeClass(current: PipelineState): string {
    if (current === "success") return "border-emerald-500/30 bg-emerald-500/12 text-emerald-500";
    if (current === "running") return "border-cyan-500/30 bg-cyan-500/12 text-cyan-500";
    if (current === "error") return "border-rose-500/30 bg-rose-500/10 text-rose-500";
    return "border-border/70 bg-card/[0.55] text-fg/65";
  }

  function stateLabel(current: PipelineState): string {
    if (current === "success") return "Terminé";
    if (current === "running") return "En cours";
    if (current === "error") return "Erreur";
    return "Prêt";
  }

  async function refreshDocsScan(): Promise<DocsDiscoveryRecord[] | null> {
    setDocsScanLoading(true);
    setPipelineError(null);
    try {
      try {
        await resyncDocsRegistryApi(token);
      } catch {
        // Optional resync: continue with discovery even without ops permission.
      }
      const rows = await discoverDocsApi(token);
      setDocsScan(rows);
      const defaults = rows
        .filter((row) => !row.already_indexed && !row.blocked)
        .map((row) => row.filename);
      setSelectedDocFilenames(defaults);
      if (rows.length === 0) {
        setMessage("Aucun PDF trouvé dans le dossier docs/.");
      } else if (rows.some((row) => row.blocked)) {
        const blocked = rows.filter((row) => row.blocked).length;
        setMessage(`${blocked} fichier(s) bloqué(s) (doublon). Corrige les doublons avant ingestion.`);
      } else if (defaults.length > 0) {
        setMessage(`${defaults.length} nouveau(x) PDF détecté(s) dans docs/. Valide puis lance le pipeline.`);
      } else {
        setMessage("Aucun nouveau PDF non indexé détecté dans docs/.");
      }
      return rows;
    } catch (error) {
      const detail = error instanceof ApiError ? error.detail : "";
      setPipelineError(detail || "Impossible de scanner le dossier docs/.");
      return null;
    } finally {
      setDocsScanLoading(false);
    }
  }

  async function toggleDuplicateOverride() {
    if (!selectedDuplicateDoc) return;
    setDuplicateOverrideLoading(true);
    try {
      const nextEnabled = !selectedDuplicateDoc.duplicate_override;
      await setDuplicateOverrideApi(
        {
          filename: selectedDuplicateDoc.filename,
          enabled: nextEnabled,
          reason: nextEnabled ? (duplicateOverrideReason.trim() || "Validation métier contrôlée") : null,
        },
        token,
      );
      const refreshed = await refreshDocsScan();
      if (refreshed) {
        const updated = refreshed.find((row) => row.filename === selectedDuplicateDoc.filename) || null;
        setSelectedDuplicateDoc(updated);
      }
    } catch (error) {
      const detail = error instanceof ApiError ? error.detail : "";
      setPipelineError(detail || "Impossible de mettre à jour la whitelist doublon.");
    } finally {
      setDuplicateOverrideLoading(false);
    }
  }

  async function applyBulkWhitelist(enabled: boolean) {
    const targets = selectedDocFilenames
      .map((name) => docsScan.find((row) => row.filename === name))
      .filter((row): row is DocsDiscoveryRecord => Boolean(row && row.is_duplicate));
    if (targets.length === 0 || bulkActionLoading) return;
    setBulkActionLoading(true);
    setPipelineError(null);
    let processed = 0;
    try {
      for (const row of targets) {
        await setDuplicateOverrideApi(
          {
            filename: row.filename,
            enabled,
            reason: enabled ? "Validation bulk opérateur" : null,
          },
          token,
        );
        processed += 1;
      }
      await refreshDocsScan();
      setMessage(
        enabled
          ? `Whitelist activée pour ${processed} document(s) doublon.`
          : `Whitelist retirée pour ${processed} document(s) doublon.`,
      );
    } catch (error) {
      const detail = error instanceof ApiError ? error.detail : "";
      setPipelineError(detail || "Bulk action échouée.");
    } finally {
      setBulkActionLoading(false);
    }
  }

  async function exportIngestionReport(format: "csv" | "pdf") {
    if (reportExportLoading) return;
    setReportExportLoading(format);
    try {
      const blob = await downloadIngestionReportApi(format, token);
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      const now = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
      anchor.href = url;
      anchor.download = `ingestion-report-${now}.${format}`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
    } catch (error) {
      const detail = error instanceof ApiError ? error.detail : "";
      setPipelineError(detail || "Export du rapport d’ingestion échoué.");
    } finally {
      setReportExportLoading(null);
    }
  }

  function toggleDocSelection(filename: string) {
    const candidate = docsScan.find((row) => row.filename === filename);
    if (candidate?.blocked) return;
    setSelectedDocFilenames((prev) => {
      if (prev.includes(filename)) return prev.filter((item) => item !== filename);
      return [...prev, filename];
    });
  }

  function selectOnlyNewDocs() {
    setSelectedDocFilenames(docsScan.filter((row) => !row.already_indexed && !row.blocked).map((row) => row.filename));
  }

  function selectAllDocs() {
    setSelectedDocFilenames(docsScan.filter((row) => !row.blocked).map((row) => row.filename));
  }

  async function runPipeline() {
    if (!canRun) return;
    setActiveJobId(null);
    setJobStartedAt(null);
    setJobFinishedAt(null);
    setJobLastCheckAt(null);
    setJobElapsedSeconds(null);
    setState("running");
    setPipelineError(null);
    setResult(null);
    setActiveStep(0);
    setMessage("Pipeline démarré...");

    clearTimer();
    timerRef.current = setInterval(() => {
      setActiveStep((prev) => Math.min(prev + 1, PIPELINE_STEPS.length - 2));
    }, 1400);

    try {
      const response = await uploadDocumentsApi(files, token);
      clearTimer();
      setActiveStep(PIPELINE_STEPS.length - 1);
      setState("success");
      setResult(response);
      if (response.ingested_count > 0) {
        setMessage(`${response.ingested_count} document(s) ingéré(s) et indexé(s).`);
      } else {
        setMessage("Aucun document indexé.");
      }
    } catch (error) {
      clearTimer();
      setState("error");
      const detail = error instanceof ApiError ? error.detail : "";
      setPipelineError(detail || "Échec du pipeline d’ingestion.");
      setMessage("Le pipeline a échoué.");
    }
  }

  async function runPipelineFromDocs() {
    if (selectedDocFilenames.length === 0 || state === "running") return;
    pollingCancelledRef.current = false;
    setJobFinishedAt(null);
    setJobElapsedSeconds(0);
    setJobLastCheckAt(new Date().toISOString());
    setState("running");
    setPipelineError(null);
    setResult(null);
    setActiveStep(0);
    setMessage("Création du job d’ingestion...");

    clearTimer();
    timerRef.current = setInterval(() => {
      setActiveStep((prev) => Math.min(prev + 1, PIPELINE_STEPS.length - 2));
    }, 1400);

    try {
      const started = await startDocsIngestionJobApi(selectedDocFilenames, token);
      setActiveJobId(started.job_id);
      setJobStartedAt(started.created_at || new Date().toISOString());
      setJobLastCheckAt(new Date().toISOString());
      setMessage(started.message || "Job lancé. Pipeline en cours...");

      const pollStartedAt = Date.now();
      const maxPollMs = Math.max(
        30 * 60 * 1000,
        selectedDocFilenames.length * 90 * 1000,
      );
      let response: UploadResponse | null = null;
      let pollingTimedOut = false;
      const startedAtMs = Number.isFinite(Date.parse(started.created_at)) ? Date.parse(started.created_at) : Date.now();
      while (!pollingCancelledRef.current) {
        const status = await getIngestionJobStatusApi(started.job_id, token);
        setJobLastCheckAt(new Date().toISOString());
        if (status.started_at) setJobStartedAt(status.started_at);
        if (status.finished_at) setJobFinishedAt(status.finished_at);
        setJobElapsedSeconds(Math.max(0, Math.floor((Date.now() - startedAtMs) / 1000)));
        if (status.message) setMessage(status.message);
        if (Number.isFinite(status.progress_percent)) {
          const ratio = Math.max(0, Math.min(1, status.progress_percent / 100));
          const mappedStep = Math.max(0, Math.min(PIPELINE_STEPS.length - 1, Math.floor(ratio * (PIPELINE_STEPS.length - 1))));
          setActiveStep((prev) => Math.max(prev, mappedStep));
        }

        if (status.status === "success") {
          response = status.result || null;
          if (status.finished_at) {
            const finishedMs = Date.parse(status.finished_at);
            if (Number.isFinite(finishedMs)) {
              setJobElapsedSeconds(Math.max(0, Math.floor((finishedMs - startedAtMs) / 1000)));
            }
          }
          break;
        }
        if (status.status === "error") {
          throw new ApiError(500, status.error || "Le job d’ingestion a échoué.");
        }
        if (Date.now() - pollStartedAt > maxPollMs) {
          pollingTimedOut = true;
          break;
        }
        await new Promise((resolve) => setTimeout(resolve, 2500));
      }

      if (pollingCancelledRef.current) return;
      if (pollingTimedOut && !response) {
        clearTimer();
        setState("idle");
        setPipelineError(null);
        setMessage(
          "Le job continue en arrière-plan. Utilise l'ID job pour vérifier l'état puis relance l'actualisation.",
        );
        return;
      }
      if (!response) {
        throw new ApiError(500, "Job terminé sans résultat exploitable.");
      }
      clearTimer();
      setActiveStep(PIPELINE_STEPS.length - 1);
      setState("success");
      setResult(response);
      setMessage(`${response.ingested_count} document(s) traité(s) depuis docs/.`);
      await refreshDocsScan();
    } catch (error) {
      clearTimer();
      setState("error");
      const detail = error instanceof ApiError ? error.detail : "";
      setPipelineError(detail || "Échec du pipeline docs/.");
      setMessage("Le pipeline a échoué.");
    }
  }

  async function summarizeDocument(docLabel: string) {
    await sendMessage({
      content: `Résume le document ${docLabel} avec les anomalies majeures et les sources.`,
      mode: "summary",
    });
    router.push("/chat");
  }

  async function queryDocument(docLabel: string) {
    await sendMessage({
      content: `Quels résultats hors référence dois-je vérifier dans ${docLabel} ?`,
      mode: "document_analysis",
    });
    router.push("/chat");
  }

  async function summarizeAllNewDocuments() {
    if (ingestedRows.length === 0) return;
    const names = ingestedRows.map((row) => row.doc_id || row.filename).join(", ");
    await sendMessage({
      content: `Fais une synthèse prudente des nouveaux documents ingérés (${names}) avec sources.`,
      mode: "summary",
    });
    router.push("/chat");
  }

  useEffect(() => {
    void refreshDocsScan();
  }, []);

  useEffect(() => {
    if (!selectedDuplicateDoc) return;
    setDuplicateOverrideReason(selectedDuplicateDoc.override_reason || "Validation métier contrôlée");
  }, [selectedDuplicateDoc]);

  return (
    <WorkspaceShell
      title="Ingestion de nouveaux documents"
      subtitle="Pipeline clinique complet: extraction -> chunking -> anonymisation -> indexation"
      breadcrumbs={["Clinical Assistant", "Documents médicaux", "Ingestion"]}
      actions={[
        { href: "/chat", label: "Retour au chat" },
        { href: "/documents", label: "Voir les documents" },
        { href: "/chat", label: "Nouvelle conversation" },
      ]}
    >
      <main className="mx-auto max-w-7xl space-y-4 px-4 py-5 sm:px-6 sm:py-6">
        <section className="rounded-2xl border border-border/70 bg-card/[0.55] p-4 shadow-[0_16px_42px_rgba(2,8,23,0.16)] transition-all duration-200">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="text-[11px] uppercase tracking-[0.14em] text-fg/58">Centre d’ingestion clinique</p>
              <h2 className="mt-1 text-base font-semibold text-fg">Traitement sécurisé des nouveaux rapports PDF</h2>
              <p className="mt-1 text-xs text-fg/62">Extraction, chunking, anonymisation, indexation et disponibilité immédiate côté chat.</p>
            </div>
          <div className="inline-flex w-full items-center justify-between gap-2 rounded-full border border-border/70 bg-card/[0.72] px-3 py-1.5 text-xs sm:w-auto">
              <span className={`inline-flex rounded-full border px-2 py-0.5 text-[11px] font-medium ${stateBadgeClass(state)}`}>{stateLabel(state)}</span>
              <span className="text-fg/70">{progressPercent}%</span>
              {running ? <Loader2 size={12} className="animate-spin text-accent" /> : null}
            </div>
          </div>
          <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-border/45">
            <div className="h-full rounded-full bg-gradient-to-r from-accent/70 to-cyan-400/80 transition-all duration-500" style={{ width: `${progressPercent}%` }} />
          </div>
          <div className="mt-4 grid grid-cols-1 gap-2.5 sm:grid-cols-2 xl:grid-cols-4">
            <article className="rounded-xl border border-border/70 bg-card/[0.62] px-3 py-2.5 transition-all duration-200 hover:border-accent/20 hover:bg-card/[0.7]">
              <p className="text-[11px] uppercase tracking-[0.12em] text-fg/55">PDF détectés</p>
              <p className="mt-1.5 text-xl font-semibold text-fg">{discoveredCount}</p>
            </article>
            <article className="rounded-xl border border-border/70 bg-card/[0.62] px-3 py-2.5 transition-all duration-200 hover:border-accent/20 hover:bg-card/[0.7]">
              <p className="text-[11px] uppercase tracking-[0.12em] text-fg/55">Nouveaux</p>
              <p className="mt-1.5 text-xl font-semibold text-cyan-500">{newDocsCount}</p>
            </article>
            <article className="rounded-xl border border-border/70 bg-card/[0.62] px-3 py-2.5 transition-all duration-200 hover:border-accent/20 hover:bg-card/[0.7]">
              <p className="text-[11px] uppercase tracking-[0.12em] text-fg/55">Déjà indexés</p>
              <p className="mt-1.5 text-xl font-semibold text-fg/88">{indexedDocsCount}</p>
            </article>
            <article className="rounded-xl border border-border/70 bg-card/[0.62] px-3 py-2.5 transition-all duration-200 hover:border-accent/20 hover:bg-card/[0.7]">
              <p className="text-[11px] uppercase tracking-[0.12em] text-fg/55">Sélection active</p>
              <p className="mt-1.5 text-xl font-semibold text-fg">{selectedDocsCount + files.length}</p>
            </article>
            <article className="rounded-xl border border-border/70 bg-card/[0.62] px-3 py-2.5 transition-all duration-200 hover:border-accent/20 hover:bg-card/[0.7] sm:col-span-2 xl:col-span-1">
              <p className="text-[11px] uppercase tracking-[0.12em] text-fg/55">Doublons bloquants</p>
              <p className={`mt-1.5 text-xl font-semibold ${duplicateDocsCount > 0 ? "text-amber-500" : "text-fg/88"}`}>{duplicateDocsCount}</p>
              <p className="mt-0.5 text-[11px] text-fg/55">{whitelistedDuplicateCount} autorisé(s)</p>
            </article>
          </div>
        </section>

        <section className="grid grid-cols-1 gap-4 xl:grid-cols-[1.45fr,1fr]">
          <article className={`${surfaceCardClass} hover:border-accent/20`}>
            <div className="mb-3 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <div className="flex items-center gap-2">
                  <span className="inline-flex h-8 w-8 items-center justify-center rounded-lg border border-accent/30 bg-accent/12">
                    <FolderSearch size={15} className="text-accent" />
                  </span>
                  <h2 className="text-sm font-semibold text-fg">Automatisation depuis le dossier docs/</h2>
                </div>
                <p className="mt-1 text-xs text-fg/65">Détection des nouveaux rapports serveur avec validation avant lancement du pipeline complet.</p>
              </div>
              <div className="flex w-full flex-col gap-2 sm:flex-row sm:flex-wrap sm:items-center sm:justify-end">
                <button
                  type="button"
                  onClick={() => void exportIngestionReport("csv")}
                  disabled={docsScanLoading || running || reportExportLoading !== null}
                  className="inline-flex w-full items-center justify-center gap-1.5 rounded-lg border border-border/75 bg-card/[0.66] px-3 py-1.5 text-xs font-medium text-fg/82 transition-all duration-200 hover:border-accent/30 hover:bg-card active:scale-[0.985] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/35 disabled:opacity-55 sm:w-auto"
                >
                  {reportExportLoading === "csv" ? <Loader2 size={13} className="animate-spin" /> : <Download size={13} />}
                  Export CSV
                </button>
                <button
                  type="button"
                  onClick={() => void exportIngestionReport("pdf")}
                  disabled={docsScanLoading || running || reportExportLoading !== null}
                  className="inline-flex w-full items-center justify-center gap-1.5 rounded-lg border border-border/75 bg-card/[0.66] px-3 py-1.5 text-xs font-medium text-fg/82 transition-all duration-200 hover:border-accent/30 hover:bg-card active:scale-[0.985] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/35 disabled:opacity-55 sm:w-auto"
                >
                  {reportExportLoading === "pdf" ? <Loader2 size={13} className="animate-spin" /> : <FileText size={13} />}
                  Export PDF
                </button>
                <button
                  type="button"
                  onClick={() => void refreshDocsScan()}
                  disabled={docsScanLoading || running}
                  className="inline-flex w-full items-center justify-center gap-1.5 rounded-lg border border-border/75 bg-card/[0.66] px-3 py-1.5 text-xs font-medium text-fg/82 transition-all duration-200 hover:border-accent/30 hover:bg-card active:scale-[0.985] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/35 disabled:opacity-55 sm:w-auto"
                >
                  {docsScanLoading ? <Loader2 size={13} className="animate-spin" /> : <RefreshCw size={13} />}
                  Actualiser docs/
                </button>
              </div>
            </div>

            <div className="mb-3 flex items-center gap-2 overflow-x-auto pb-1">
              <span className="inline-flex items-center gap-1 text-[11px] uppercase tracking-[0.12em] text-fg/55">
                <ListFilter size={12} />
                Filtres
              </span>
              {[
                { id: "all", label: "Tous" },
                { id: "new", label: "Nouveaux" },
                { id: "indexed", label: "Indexés" },
                { id: "duplicate", label: "Doublons" },
                { id: "whitelist", label: "Whitelist" },
              ].map((filter) => {
                const active = statusFilter === (filter.id as DocsStatusFilter);
                return (
                  <button
                    key={filter.id}
                    type="button"
                    onClick={() => setStatusFilter(filter.id as DocsStatusFilter)}
                    className={`rounded-full border px-2.5 py-1 text-[11px] transition ${
                      active
                        ? "border-cyan-500/45 bg-cyan-500/12 text-cyan-500"
                        : "border-border/70 bg-card/[0.62] text-fg/72 hover:border-accent/30"
                    }`}
                  >
                    {filter.label}
                  </button>
                );
              })}
              <span className="ml-auto shrink-0 text-xs text-fg/58">{filteredDocsScan.length} affiché(s)</span>
            </div>

            <div className="rounded-xl border border-border/65 bg-card/[0.42]">
              {filteredDocsScan.length === 0 ? (
                <div className="px-3 py-3 text-xs text-fg/58">{docsScanLoading ? "Scan en cours..." : "Aucun PDF détecté dans docs/."}</div>
              ) : (
                <div className="max-h-[18rem] overflow-auto sm:max-h-[22rem]">
                  {filteredDocsScan.map((item) => {
                    const checked = selectedDocFilenames.includes(item.filename);
                    return (
                      <label
                        key={item.filename}
                        className="group flex cursor-pointer items-center justify-between gap-3 border-b border-border/50 px-3 py-3 text-xs transition-all duration-200 last:border-b-0 hover:bg-card/[0.55]"
                      >
                        <div className="min-w-0 transition-transform duration-200 group-hover:translate-x-0.5">
                          <div className="flex items-center gap-2">
                            <input
                              type="checkbox"
                              checked={checked}
                              disabled={item.blocked}
                              onChange={() => toggleDocSelection(item.filename)}
                              className="h-3.5 w-3.5 accent-cyan-500"
                            />
                            <span className="line-clamp-1 text-fg/90">{item.filename}</span>
                            <span
                              className={`rounded-full border px-1.5 py-0.5 text-[10px] ${
                                item.already_indexed ? "border-border/70 bg-card/[0.62] text-fg/55" : "border-cyan-500/35 bg-cyan-500/10 text-cyan-500"
                              }`}
                            >
                              {item.already_indexed ? "Indexé" : "Nouveau"}
                            </span>
                            {item.blocked ? (
                              <span className="rounded-full border border-amber-500/35 bg-amber-500/12 px-1.5 py-0.5 text-[10px] text-amber-500">
                                Doublon
                              </span>
                            ) : null}
                            {item.is_duplicate && item.duplicate_override ? (
                              <span className="rounded-full border border-emerald-500/35 bg-emerald-500/12 px-1.5 py-0.5 text-[10px] text-emerald-500">
                                Whitelist
                              </span>
                            ) : null}
                          </div>
                          <p className="mt-1 line-clamp-1 text-[11px] text-fg/55">
                            {item.doc_id} · {formatBytes(item.size_bytes)} · {formatDate(item.modified_at)}
                          </p>
                          {item.is_duplicate && item.duplicate_reason ? (
                            <div className="mt-1 flex flex-wrap items-center gap-2">
                              <p className={`line-clamp-2 text-[11px] ${item.duplicate_override ? "text-emerald-500/90" : "text-amber-500/90"}`}>
                                {item.duplicate_override ? "Doublon autorisé (non bloquant)." : item.duplicate_reason}
                                {item.duplicate_with.length > 0 ? ` (${item.duplicate_with.slice(0, 3).join(", ")})` : ""}
                              </p>
                              <button
                                type="button"
                                onClick={(event) => {
                                  event.preventDefault();
                                  event.stopPropagation();
                                  void openDocumentDetails(item);
                                }}
                                className={`inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-[10px] font-medium transition focus-visible:outline-none focus-visible:ring-2 ${
                                  item.duplicate_override
                                    ? "border border-emerald-500/35 bg-emerald-500/10 text-emerald-500 hover:bg-emerald-500/16 focus-visible:ring-emerald-500/45"
                                    : "border border-amber-500/35 bg-amber-500/10 text-amber-500 hover:bg-amber-500/16 focus-visible:ring-amber-500/45"
                                }`}
                              >
                                <Eye size={11} />
                                Voir détails
                              </button>
                            </div>
                          ) : null}
                          {!item.is_duplicate ? (
                            <button
                              type="button"
                              onClick={(event) => {
                                event.preventDefault();
                                event.stopPropagation();
                                void openDocumentDetails(item);
                              }}
                              className="mt-1 inline-flex items-center gap-1 rounded-md border border-border/70 bg-card/[0.65] px-2 py-0.5 text-[10px] text-fg/72 transition hover:border-accent/30"
                            >
                              <History size={11} />
                              Timeline
                            </button>
                          ) : null}
                        </div>
                      </label>
                    );
                  })}
                </div>
              )}
            </div>

            <div className="mt-3 rounded-lg border border-border/65 bg-card/[0.45] px-3 py-2">
              <div className="flex flex-col gap-2 sm:flex-row sm:flex-wrap sm:items-center">
                <span className="text-xs text-fg/65">Actions groupées</span>
                <span className="rounded-full border border-border/70 bg-card/[0.7] px-2 py-0.5 text-[11px] text-fg/72">
                  {selectedDuplicateCount} doublon(s) sélectionné(s)
                </span>
                <button
                  type="button"
                  onClick={() => void applyBulkWhitelist(true)}
                  disabled={selectedDuplicateCount === 0 || bulkActionLoading || running}
                  className="inline-flex w-full items-center justify-center gap-1 rounded-lg border border-emerald-500/35 bg-emerald-500/12 px-2.5 py-1 text-[11px] text-emerald-500 transition hover:bg-emerald-500/18 disabled:opacity-55 sm:w-auto"
                >
                  {bulkActionLoading ? <Loader2 size={11} className="animate-spin" /> : <ShieldCheck size={11} />}
                  Whitelist sélection
                </button>
                <button
                  type="button"
                  onClick={() => void applyBulkWhitelist(false)}
                  disabled={selectedDuplicateCount === 0 || bulkActionLoading || running}
                  className="inline-flex w-full items-center justify-center gap-1 rounded-lg border border-rose-500/35 bg-rose-500/12 px-2.5 py-1 text-[11px] text-rose-500 transition hover:bg-rose-500/18 disabled:opacity-55 sm:w-auto"
                >
                  {bulkActionLoading ? <Loader2 size={11} className="animate-spin" /> : <X size={11} />}
                  Retirer whitelist
                </button>
              </div>
            </div>

            <div className="mt-3 flex flex-col items-stretch gap-2 sm:flex-row sm:items-center">
              <button
                type="button"
                onClick={selectOnlyNewDocs}
                disabled={docsScan.length === 0 || running}
                className={`${secondaryActionClass} w-full justify-center sm:w-auto`}
              >
                Sélectionner nouveaux
              </button>
              <button
                type="button"
                onClick={selectAllDocs}
                disabled={docsScan.length === 0 || running}
                className={`${secondaryActionClass} w-full justify-center sm:w-auto`}
              >
                Tout sélectionner
              </button>
              <div className="flex w-full items-center justify-between gap-2 sm:ml-auto sm:w-auto sm:justify-end">
                <span className="rounded-full border border-border/70 bg-card/[0.66] px-2 py-0.5 text-xs text-fg/62">{selectedDocsEligibleCount} sélectionné(s)</span>
                <button
                  type="button"
                  onClick={() => void runPipelineFromDocs()}
                  disabled={!canRunDocsPipeline}
                  className={`${primaryActionClass} w-full justify-center sm:w-auto`}
                >
                  {running ? <Loader2 size={13} className="animate-spin" /> : <Play size={13} />}
                  Valider et lancer pipeline
                </button>
              </div>
            </div>
          </article>

          <article className={`${surfaceCardClass} hover:border-accent/20`}>
            <div className="mb-3 flex items-center gap-2">
              <span className="inline-flex h-8 w-8 items-center justify-center rounded-lg border border-accent/30 bg-accent/12">
                <UploadCloud size={16} className="text-accent" />
              </span>
              <h2 className="text-sm font-semibold text-fg">Upload manuel</h2>
            </div>
            <p className="text-xs text-fg/68">Import local sécurisé puis traitement complet via pipeline RAG clinique.</p>
            <label className="mt-4 block rounded-xl border border-dashed border-border/70 bg-card/[0.45] p-5 text-center transition-all duration-200 hover:border-accent/35 hover:bg-card/[0.54]">
              <input
                type="file"
                accept=".pdf"
                multiple
                className="hidden"
                onChange={(event) => onFilesPicked(event.target.files)}
              />
              <span className="text-sm font-medium text-fg/90">Sélectionner un ou plusieurs rapports PDF</span>
              <p className="mt-1 text-xs text-fg/58">Formats acceptés: `.pdf`</p>
            </label>

            <div className="mt-3 space-y-2">
              {files.length > 0 ? files.map((file) => (
                <div key={file.name} className="flex items-center justify-between rounded-lg border border-border/65 bg-card/[0.42] px-3 py-2 text-xs transition-all duration-200 hover:border-accent/25">
                  <span className="line-clamp-1 text-fg/85">{file.name}</span>
                  <span className="text-fg/58">{Math.max(1, Math.round(file.size / 1024))} KB</span>
                </div>
              )) : (
                <div className="rounded-lg border border-border/65 bg-card/[0.42] px-3 py-2 text-xs text-fg/58">
                  Aucun PDF sélectionné.
                </div>
              )}
            </div>

            <div className="mt-4 flex flex-col gap-2 sm:flex-row">
              <button
                type="button"
                onClick={() => void runPipeline()}
                disabled={!canRun}
                className={`${primaryActionClass} w-full justify-center sm:w-auto`}
              >
                {running ? <Loader2 size={13} className="animate-spin" /> : <Play size={13} />}
                Lancer le pipeline complet
              </button>
              <button
                type="button"
                onClick={() => {
                  setFiles([]);
                  setState("idle");
                  setResult(null);
                  setPipelineError(null);
                  setMessage("");
                  setActiveStep(0);
                }}
                disabled={running}
                className="inline-flex w-full items-center justify-center gap-1 rounded-lg border border-border/75 bg-card/[0.62] px-3 py-1.5 text-xs font-medium text-fg/78 transition-all duration-200 hover:border-accent/30 hover:bg-card active:scale-[0.985] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/35 disabled:opacity-55 sm:w-auto"
              >
                Réinitialiser
              </button>
            </div>
          </article>
        </section>

        <section className="grid grid-cols-1 gap-4 xl:grid-cols-[1.15fr,1fr]">
          <article className={`${surfaceCardClass} hover:border-accent/20`}>
            <div className="mb-3 flex items-center gap-2">
              <span className="inline-flex h-8 w-8 items-center justify-center rounded-lg border border-accent/30 bg-accent/12">
                <Sparkles size={15} className="text-accent" />
              </span>
              <h2 className="text-sm font-semibold text-fg">Progression pipeline</h2>
            </div>
            <div className="space-y-2">
              {PIPELINE_STEPS.map((step, index) => {
                const status = stepStatus(state, index, activeStep);
                return (
                  <div key={step.id} className="rounded-lg border border-border/65 bg-card/[0.45] px-3 py-2.5 transition-all duration-200 hover:border-accent/20 hover:bg-card/[0.52]">
                    <div className="flex items-center gap-2">
                      {status === "done" ? (
                        <CheckCircle2 size={14} className="text-emerald-500" />
                      ) : status === "active" ? (
                        <Loader2 size={14} className="animate-spin text-accent" />
                      ) : status === "error" ? (
                        <XCircle size={14} className="text-rose-500" />
                      ) : (
                        <Circle size={14} className="text-fg/42" />
                      )}
                      <p className="text-xs font-medium text-fg/88">{step.title}</p>
                    </div>
                    <p className="mt-1 text-xs text-fg/62">{step.description}</p>
                  </div>
                );
              })}
            </div>
            <p className="mt-3 text-xs text-fg/70">
              {state === "running" ? runningStepText : message || "Prêt à lancer le pipeline d’ingestion."}
            </p>
            {pipelineError ? (
              <div className="mt-3 rounded-lg border border-rose-500/35 bg-rose-500/10 px-3 py-2 text-xs text-rose-500">
                {pipelineError}
              </div>
            ) : null}
          </article>

          <article className={`${surfaceCardClass} hover:border-accent/20`}>
            <div className="mb-3 flex items-center gap-2">
              <span className="inline-flex h-8 w-8 items-center justify-center rounded-lg border border-accent/30 bg-accent/12">
                <Clock3 size={15} className="text-accent" />
              </span>
              <h2 className="text-sm font-semibold text-fg">Session d’ingestion</h2>
            </div>
            <ul className="space-y-2 text-xs">
              <li className="flex items-center justify-between rounded-lg border border-border/65 bg-card/[0.45] px-3 py-2 transition-all duration-200 hover:border-accent/20">
                <span className="text-fg/72">État</span>
                <span className={`rounded-full border px-2 py-0.5 font-medium ${stateBadgeClass(state)}`}>{stateLabel(state)}</span>
              </li>
              <li className="flex items-center justify-between rounded-lg border border-border/65 bg-card/[0.45] px-3 py-2 transition-all duration-200 hover:border-accent/20">
                <span className="text-fg/72">Sélection docs/</span>
                <span className="font-medium text-fg">{selectedDocsCount}</span>
              </li>
              <li className="flex items-center justify-between rounded-lg border border-border/65 bg-card/[0.45] px-3 py-2 transition-all duration-200 hover:border-accent/20">
                <span className="text-fg/72">Upload manuel</span>
                <span className="font-medium text-fg">{files.length}</span>
              </li>
              <li className="flex items-center justify-between rounded-lg border border-border/65 bg-card/[0.45] px-3 py-2 transition-all duration-200 hover:border-accent/20">
                <span className="text-fg/72">Dernier résultat</span>
                <span className="font-medium text-fg">{result?.ingested_count ?? 0} ingéré(s)</span>
              </li>
            </ul>
            <div className="mt-3 rounded-lg border border-border/65 bg-card/[0.45] px-3 py-2.5 text-xs">
              <p className="mb-2 text-[11px] uppercase tracking-[0.12em] text-fg/55">Suivi job async</p>
              <div className="grid grid-cols-[auto,1fr] gap-x-3 gap-y-1.5 text-[11px] sm:text-xs">
                <span className="text-fg/62">Job ID</span>
                <span className="truncate font-mono text-[11px] text-fg/85">{activeJobId || "—"}</span>
                <span className="text-fg/62">Démarré</span>
                <span className="text-fg/85">{formatDateTimeShort(jobStartedAt)}</span>
                <span className="text-fg/62">Dernier check</span>
                <span className="text-fg/85">{formatDateTimeShort(jobLastCheckAt)}</span>
                <span className="text-fg/62">Durée</span>
                <span className="text-fg/85">{formatDuration(jobElapsedSeconds)}</span>
                <span className="text-fg/62">Fin</span>
                <span className="text-fg/85">{formatDateTimeShort(jobFinishedAt)}</span>
              </div>
            </div>
            <p className="mt-3 text-xs text-fg/58">
              Le pipeline exécute extraction, anonymisation, chunking et indexation avant disponibilité dans le chat clinique.
            </p>
          </article>
        </section>

        {result ? (
          <section className={`${surfaceCardClass} hover:border-accent/20`}>
            <div className="mb-3 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <h2 className="text-sm font-semibold text-fg">Documents prêts pour interrogation</h2>
                <p className="text-xs text-fg/65">
                  {result.ingested_count} ingéré(s) · {skippedRows.length} ignoré(s)
                </p>
              </div>
              <button
                type="button"
                onClick={() => void summarizeAllNewDocuments()}
                disabled={sending || ingestedRows.length === 0}
                className={`${primaryActionClass} w-full justify-center sm:w-auto`}
              >
                <FileText size={12} />
                Résumer les nouveaux documents
              </button>
            </div>

            <div className="space-y-2">
              {ingestedRows.map((row) => {
                const label = row.doc_id || row.filename;
                return (
                  <article key={`${row.doc_id}-${row.filename}`} className="rounded-lg border border-border/65 bg-card/[0.45] px-3 py-2.5 transition-all duration-200 hover:border-accent/25 hover:bg-card/[0.52]">
                    <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                      <div className="min-w-0">
                        <div className="flex items-center gap-2">
                          <FileCheck2 size={13} className="text-emerald-500" />
                          <p className="line-clamp-1 text-sm font-medium text-fg">{label}</p>
                        </div>
                        <p className="line-clamp-1 text-xs text-fg/62">{row.filename}</p>
                      </div>
                      <div className="flex shrink-0 flex-col gap-2 sm:flex-row sm:flex-wrap">
                        <button
                          type="button"
                          onClick={() => void summarizeDocument(label)}
                          disabled={sending}
                          className={`${subtleActionClass} w-full justify-center sm:w-auto`}
                        >
                          <FileText size={12} /> Résumer
                        </button>
                        <button
                          type="button"
                          onClick={() => void queryDocument(label)}
                          disabled={sending}
                          className={`${subtleActionClass} w-full justify-center sm:w-auto`}
                        >
                          <Search size={12} /> Interroger
                        </button>
                      </div>
                    </div>
                  </article>
                );
              })}
            </div>

            {skippedRows.length > 0 ? (
              <div className="mt-3 rounded-lg border border-amber-500/35 bg-amber-500/10 px-3 py-2 text-xs text-amber-600">
                Ignorés: {skippedRows.map((row) => `${row.filename} (${row.reason})`).join(", ")}
              </div>
            ) : null}
          </section>
        ) : null}

        {!result ? (
          <section className="rounded-xl border border-border/70 bg-card/[0.42] px-4 py-3 text-xs text-fg/62">
            <div className="flex items-center gap-2">
              <Files size={13} />
              Les documents traités ici seront immédiatement disponibles pour résumé, comparaison et interrogation assistée.
            </div>
          </section>
        ) : null}
      </main>
      {selectedDuplicateDoc ? (
        <div className="fixed inset-0 z-[80] flex items-end justify-center bg-slate-950/72 p-3 backdrop-blur-sm sm:items-center sm:p-4">
          <div className="flex max-h-[92dvh] w-full max-w-2xl flex-col overflow-auto rounded-2xl border border-border/80 bg-card/[0.96] p-4 shadow-[0_24px_70px_rgba(2,8,23,0.5)] sm:max-h-[88dvh]">
            <div className="mb-3 flex items-start justify-between gap-3">
              <div>
                <p className="text-[11px] uppercase tracking-[0.13em] text-fg/55">Audit doublon</p>
                <h3 className="mt-1 text-sm font-semibold text-fg">{selectedDuplicateDoc.filename}</h3>
                <p className="mt-1 text-xs text-fg/65">{selectedDuplicateDoc.duplicate_reason || "Doublon détecté."}</p>
              </div>
              <button
                type="button"
                onClick={closeDuplicateDetails}
                className="icon-button h-8 w-8 rounded-lg"
                aria-label="Fermer le détail doublon"
              >
                <X size={14} />
              </button>
            </div>

            <div className="grid max-h-[28dvh] grid-cols-1 gap-2 overflow-auto rounded-lg border border-border/70 bg-card/[0.6] p-3 text-xs sm:max-h-none sm:grid-cols-2 sm:overflow-visible">
              <div>
                <p className="text-fg/55">Hash SHA-256</p>
                <p className="mt-0.5 break-all font-mono text-[11px] text-fg/88">{selectedDuplicateDoc.file_hash || "—"}</p>
              </div>
              <div>
                <p className="text-fg/55">Hash texte normalisé</p>
                <p className="mt-0.5 break-all font-mono text-[11px] text-fg/88">{selectedDuplicateDoc.text_hash || "—"}</p>
              </div>
              <div>
                <p className="text-fg/55">Statut registre</p>
                <p className="mt-0.5 text-fg/88">{selectedDuplicateDoc.registry_status || "—"}</p>
              </div>
              <div>
                <p className="text-fg/55">Première détection</p>
                <p className="mt-0.5 text-fg/88">{formatDateTimeShort(selectedDuplicateDoc.first_seen_at || null)}</p>
              </div>
              <div>
                <p className="text-fg/55">Dernière détection</p>
                <p className="mt-0.5 text-fg/88">{formatDateTimeShort(selectedDuplicateDoc.last_seen_at || null)}</p>
              </div>
              <div>
                <p className="text-fg/55">Dernière ingestion</p>
                <p className="mt-0.5 text-fg/88">{formatDateTimeShort(selectedDuplicateDoc.last_ingested_at || null)}</p>
              </div>
              <div>
                <p className="text-fg/55">Doc ID actuel</p>
                <p className="mt-0.5 text-fg/88">{selectedDuplicateDoc.doc_id || "—"}</p>
              </div>
              <div>
                <p className="text-fg/55">Whitelist</p>
                <p className={`mt-0.5 ${selectedDuplicateDoc.duplicate_override ? "text-emerald-500" : "text-fg/88"}`}>
                  {selectedDuplicateDoc.duplicate_override ? "Activée" : "Désactivée"}
                </p>
              </div>
              <div>
                <p className="text-fg/55">Override par</p>
                <p className="mt-0.5 text-fg/88">
                  {selectedDuplicateDoc.override_by || "—"} · {formatDateTimeShort(selectedDuplicateDoc.override_at || null)}
                </p>
              </div>
            </div>

            <div className="mt-3 rounded-lg border border-border/70 bg-card/[0.6] p-3">
              <p className="mb-1 text-xs font-medium text-fg/88">Contrôle whitelist</p>
              <p className="mb-2 text-[11px] text-fg/62">
                Autorise explicitement ce doublon pour ingestion si validé métier.
              </p>
              <label className="mb-1 block text-[11px] text-fg/60">Motif de validation</label>
              <textarea
                value={duplicateOverrideReason}
                onChange={(event) => setDuplicateOverrideReason(event.target.value)}
                rows={2}
                className="mb-2 w-full resize-none rounded-lg border border-border/70 bg-card/[0.72] px-2.5 py-1.5 text-xs text-fg outline-none transition focus:border-accent/40"
                placeholder="Ex: cas métier validé par biologiste référent"
                disabled={duplicateOverrideLoading}
              />
              <button
                type="button"
                onClick={() => void toggleDuplicateOverride()}
                disabled={duplicateOverrideLoading}
                className={`inline-flex items-center gap-1 rounded-lg border px-3 py-1.5 text-xs font-medium transition ${
                  selectedDuplicateDoc.duplicate_override
                    ? "border-rose-500/35 bg-rose-500/12 text-rose-500 hover:bg-rose-500/18"
                    : "border-emerald-500/35 bg-emerald-500/12 text-emerald-500 hover:bg-emerald-500/18"
                } disabled:opacity-55`}
              >
                {duplicateOverrideLoading ? <Loader2 size={12} className="animate-spin" /> : null}
                {selectedDuplicateDoc.duplicate_override ? "Retirer whitelist" : "Ignorer ce doublon"}
              </button>
            </div>

            <div className="mt-3">
              <p className="mb-2 text-xs font-medium text-fg/88">Timeline d’évènements document</p>
              <div className="mb-3 max-h-36 overflow-auto rounded-lg border border-border/70 sm:max-h-40">
                {timelineLoading ? (
                  <div className="px-3 py-2 text-xs text-fg/60">Chargement timeline...</div>
                ) : timelineEvents.length === 0 ? (
                  <div className="px-3 py-2 text-xs text-fg/60">Aucun évènement trouvé.</div>
                ) : (
                  timelineEvents.map((event, index) => (
                    <div key={`${event.at}-${event.type}-${index}`} className="border-b border-border/55 px-3 py-2 text-xs last:border-b-0">
                      <div className="flex items-center justify-between gap-2">
                        <span className="font-medium text-fg/88">{event.title || event.type}</span>
                        <span className="text-[11px] text-fg/55">{formatDateTimeShort(event.at || null)}</span>
                      </div>
                      <p className="mt-0.5 text-[11px] text-fg/65">{event.detail || "—"}{event.actor ? ` · ${event.actor}` : ""}</p>
                    </div>
                  ))
                )}
              </div>

              <p className="mb-2 text-xs font-medium text-fg/88">Historique des fichiers au même hash</p>
              <div className="max-h-40 overflow-auto rounded-lg border border-border/70 sm:max-h-56">
                {selectedDuplicateDoc.duplicate_entries.length === 0 ? (
                  <div className="px-3 py-2 text-xs text-fg/58">Aucun historique disponible.</div>
                ) : (
                  selectedDuplicateDoc.duplicate_entries.map((entry) => (
                    <div key={`${entry.absolute_path}:${entry.filename}`} className="border-b border-border/55 px-3 py-2 text-xs last:border-b-0">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="font-medium text-fg/90">{entry.filename || "—"}</span>
                        <span className={`rounded-full border px-1.5 py-0.5 text-[10px] ${entry.is_indexed ? "border-emerald-500/35 bg-emerald-500/10 text-emerald-500" : "border-border/70 text-fg/60"}`}>
                          {entry.is_indexed ? "Indexé" : "Non indexé"}
                        </span>
                        <span className="rounded-full border border-border/65 bg-card/[0.66] px-1.5 py-0.5 text-[10px] text-fg/65">{entry.status || "—"}</span>
                      </div>
                      <p className="mt-1 line-clamp-1 text-[11px] text-fg/60">{entry.doc_id || "doc_id —"} · {formatDateTimeShort(entry.last_ingested_at || null)}</p>
                      {entry.last_error ? <p className="mt-1 line-clamp-2 text-[11px] text-rose-500/90">{entry.last_error}</p> : null}
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </WorkspaceShell>
  );
}
