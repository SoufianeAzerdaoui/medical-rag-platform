"use client";

import {
  CheckCircle2,
  Circle,
  Clock3,
  FileCheck2,
  FileText,
  Files,
  FolderSearch,
  Loader2,
  Play,
  RefreshCw,
  Search,
  Sparkles,
  UploadCloud,
  XCircle,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { WorkspaceShell } from "@/components/layout/workspace-shell";
import { useChatActions } from "@/hooks/use-chat-actions";
import { ApiError, discoverDocsApi, type DocsDiscoveryRecord, type UploadResponse, uploadDocumentsApi, uploadFromDocsApi } from "@/services/rag-api";
import { useAuthStore } from "@/store/auth-store";

type PipelineState = "idle" | "running" | "success" | "error";
type StepStatus = "pending" | "active" | "done" | "error";

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
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const hasFiles = files.length > 0;
  const canRun = hasFiles && state !== "running";
  const ingestedRows = result?.ingested || [];
  const skippedRows = result?.skipped || [];
  const selectedDocsCount = selectedDocFilenames.length;
  const discoveredCount = docsScan.length;
  const newDocsCount = docsScan.filter((item) => !item.already_indexed).length;
  const indexedDocsCount = docsScan.filter((item) => item.already_indexed).length;
  const canRunDocsPipeline = selectedDocsCount > 0 && state !== "running";
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

  function clearTimer() {
    if (!timerRef.current) return;
    clearInterval(timerRef.current);
    timerRef.current = null;
  }

  useEffect(() => {
    return () => clearTimer();
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

  async function refreshDocsScan() {
    setDocsScanLoading(true);
    setPipelineError(null);
    try {
      const rows = await discoverDocsApi(token);
      setDocsScan(rows);
      const defaults = rows.filter((row) => !row.already_indexed).map((row) => row.filename);
      setSelectedDocFilenames(defaults);
      if (rows.length === 0) {
        setMessage("Aucun PDF trouvé dans le dossier docs/.");
      } else if (defaults.length > 0) {
        setMessage(`${defaults.length} nouveau(x) PDF détecté(s) dans docs/. Valide puis lance le pipeline.`);
      } else {
        setMessage("Aucun nouveau PDF non indexé détecté dans docs/.");
      }
    } catch (error) {
      const detail = error instanceof ApiError ? error.detail : "";
      setPipelineError(detail || "Impossible de scanner le dossier docs/.");
    } finally {
      setDocsScanLoading(false);
    }
  }

  function toggleDocSelection(filename: string) {
    setSelectedDocFilenames((prev) => {
      if (prev.includes(filename)) return prev.filter((item) => item !== filename);
      return [...prev, filename];
    });
  }

  function selectOnlyNewDocs() {
    setSelectedDocFilenames(docsScan.filter((row) => !row.already_indexed).map((row) => row.filename));
  }

  function selectAllDocs() {
    setSelectedDocFilenames(docsScan.map((row) => row.filename));
  }

  async function runPipeline() {
    if (!canRun) return;
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
    setState("running");
    setPipelineError(null);
    setResult(null);
    setActiveStep(0);
    setMessage("Pipeline démarré depuis docs/...");

    clearTimer();
    timerRef.current = setInterval(() => {
      setActiveStep((prev) => Math.min(prev + 1, PIPELINE_STEPS.length - 2));
    }, 1400);

    try {
      const response = await uploadFromDocsApi(selectedDocFilenames, token);
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
            <div className="inline-flex items-center gap-2 rounded-full border border-border/70 bg-card/[0.72] px-3 py-1.5 text-xs">
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
          </div>
        </section>

        <section className="grid grid-cols-1 gap-4 xl:grid-cols-[1.45fr,1fr]">
          <article className={`${surfaceCardClass} hover:border-accent/20`}>
            <div className="mb-3 flex items-center justify-between gap-3">
              <div>
                <div className="flex items-center gap-2">
                  <span className="inline-flex h-8 w-8 items-center justify-center rounded-lg border border-accent/30 bg-accent/12">
                    <FolderSearch size={15} className="text-accent" />
                  </span>
                  <h2 className="text-sm font-semibold text-fg">Automatisation depuis le dossier docs/</h2>
                </div>
                <p className="mt-1 text-xs text-fg/65">Détection des nouveaux rapports serveur avec validation avant lancement du pipeline complet.</p>
              </div>
              <button
                type="button"
                onClick={() => void refreshDocsScan()}
                disabled={docsScanLoading || running}
                className="inline-flex items-center gap-1.5 rounded-lg border border-border/75 bg-card/[0.66] px-3 py-1.5 text-xs font-medium text-fg/82 transition-all duration-200 hover:border-accent/30 hover:bg-card active:scale-[0.985] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/35 disabled:opacity-55"
              >
                {docsScanLoading ? <Loader2 size={13} className="animate-spin" /> : <RefreshCw size={13} />}
                Actualiser docs/
              </button>
            </div>

            <div className="rounded-xl border border-border/65 bg-card/[0.42]">
              {docsScan.length === 0 ? (
                <div className="px-3 py-3 text-xs text-fg/58">{docsScanLoading ? "Scan en cours..." : "Aucun PDF détecté dans docs/."}</div>
              ) : (
                <div className="max-h-[22rem] overflow-auto">
                  {docsScan.map((item) => {
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
                          </div>
                          <p className="mt-1 line-clamp-1 text-[11px] text-fg/55">
                            {item.doc_id} · {formatBytes(item.size_bytes)} · {formatDate(item.modified_at)}
                          </p>
                        </div>
                      </label>
                    );
                  })}
                </div>
              )}
            </div>

            <div className="mt-3 flex flex-col items-stretch gap-2 sm:flex-row sm:items-center">
              <button
                type="button"
                onClick={selectOnlyNewDocs}
                disabled={docsScan.length === 0 || running}
                className={`${secondaryActionClass} w-full sm:w-auto`}
              >
                Sélectionner nouveaux
              </button>
              <button
                type="button"
                onClick={selectAllDocs}
                disabled={docsScan.length === 0 || running}
                className={`${secondaryActionClass} w-full sm:w-auto`}
              >
                Tout sélectionner
              </button>
              <div className="flex w-full items-center justify-between gap-2 sm:ml-auto sm:w-auto sm:justify-end">
                <span className="rounded-full border border-border/70 bg-card/[0.66] px-2 py-0.5 text-xs text-fg/62">{selectedDocsCount} sélectionné(s)</span>
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
            <p className="mt-3 text-xs text-fg/58">
              Le pipeline exécute extraction, anonymisation, chunking et indexation avant disponibilité dans le chat clinique.
            </p>
          </article>
        </section>

        {result ? (
          <section className={`${surfaceCardClass} hover:border-accent/20`}>
            <div className="mb-3 flex items-center justify-between gap-3">
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
                      <div className="flex shrink-0 flex-wrap gap-2">
                        <button
                          type="button"
                          onClick={() => void summarizeDocument(label)}
                          disabled={sending}
                          className={subtleActionClass}
                        >
                          <FileText size={12} /> Résumer
                        </button>
                        <button
                          type="button"
                          onClick={() => void queryDocument(label)}
                          disabled={sending}
                          className={subtleActionClass}
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
    </WorkspaceShell>
  );
}
