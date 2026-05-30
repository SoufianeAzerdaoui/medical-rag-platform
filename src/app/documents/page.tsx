"use client";

import Link from "next/link";
import {
  AlertCircle,
  CheckCircle2,
  Clock3,
  Eye,
  FileText,
  MessageSquare,
  RefreshCw,
  Scale,
  Search,
  ShieldAlert,
  Trash2,
  UploadCloud,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { WorkspaceShell } from "@/components/layout/workspace-shell";
import { useChatActions } from "@/hooks/use-chat-actions";
import {
  ApiError,
  deleteDocumentApi,
  discoverDocsApi,
  listDocumentsApi,
  reindexDocumentApi,
  type DocsDiscoveryRecord,
  type DocumentRecord,
} from "@/services/rag-api";
import { useAuthStore } from "@/store/auth-store";
import { useChatStore } from "@/store/chat-store";
import type { ChatSource } from "@/types/chat";

type MedicalCategory = "Biochimie" | "Hématologie" | "Toxicologie" | "Parasitologie";
type DocumentStatus =
  | "indexed"
  | "new"
  | "duplicate_blocked"
  | "duplicate_whitelisted"
  | "error"
  | "missing"
  | "processing";
type StatusFilter = "all" | "indexed" | "non_indexed" | "duplicates" | "error" | "processing";
type SortOption = "recent" | "name_asc" | "name_desc" | "status_priority";
type BusyAction = "reindex" | "delete" | "ask";

type ActionNotice = {
  kind: "success" | "error" | "info";
  message: string;
};

type DocumentItem = {
  id: string;
  name: string;
  category: MedicalCategory;
  typeLabel: string;
  patientLabel: string;
  dateLabel: string;
  dateSortValue: number;
  sizeLabel: string;
  analytes: string[];
  indexed: boolean;
  status: DocumentStatus;
  statusLabel: string;
  statusDetail?: string | null;
};

const CATEGORY_OPTIONS: Array<MedicalCategory | "Tous"> = ["Tous", "Biochimie", "Hématologie", "Toxicologie", "Parasitologie"];
const STATUS_FILTERS: Array<{ key: StatusFilter; label: string }> = [
  { key: "all", label: "Tous" },
  { key: "non_indexed", label: "Non indexés" },
  { key: "indexed", label: "Indexés" },
  { key: "duplicates", label: "Doublons" },
  { key: "error", label: "Erreurs" },
  { key: "processing", label: "En cours" },
];

function normalizeDocToken(value: string): string {
  return String(value || "").trim().toLowerCase().replace(/\.pdf$/i, "");
}

function extractSourceDoc(source: ChatSource): string | null {
  if (typeof source === "string") {
    const docId = source.match(/doc_id=([^,\]\s]+)/i)?.[1];
    const filename = source.match(/([A-Za-z0-9_\-().]+\.(?:pdf|PDF))/)?.[1];
    return (docId || filename || "").trim() || null;
  }
  const raw = source as Record<string, unknown>;
  const value = String(raw.filename || raw.documentName || raw.doc_id || raw.documentId || raw.label || "").trim();
  return value || null;
}

function inferCategory(name: string, analytes: string[]): MedicalCategory {
  const text = `${name} ${analytes.join(" ")}`.toLowerCase();
  if (/(tox|drug|benz|opi|coca|alcool|ethanol)/i.test(text)) return "Toxicologie";
  if (/(paras|helminth|protozo|copro|stool)/i.test(text)) return "Parasitologie";
  if (/(h[bc]|plaquette|globule|leuco|numération|hemato|hémato)/i.test(text)) return "Hématologie";
  return "Biochimie";
}

function inferTypeLabel(name: string, analytes: string[]): string {
  const text = `${name} ${analytes.join(" ")}`.toLowerCase();
  if (/(tsh|t3|t4|anti-tg|anti tg|immuno)/i.test(text)) return "Immunoanalyse";
  if (/(h[bc]|plaquette|leucocyte|globule)/i.test(text)) return "Numération";
  if (/(toxic|drug|alcool)/i.test(text)) return "Dépistage";
  return "Bilan biologique";
}

function sourceAnalytesFromText(text: string): string[] {
  const known = ["TSH", "T3", "T4", "Anti-TG", "Hb", "VGM", "Plaquettes", "Leucocytes", "CRP", "ASAT", "ALAT"];
  const upper = text.toUpperCase();
  return known.filter((a) => upper.includes(a.toUpperCase()));
}

function statusFromDiscovery(record: DocsDiscoveryRecord): { status: DocumentStatus; label: string; detail?: string } {
  if (record.last_error && record.last_error.trim()) {
    return { status: "error", label: "Erreur", detail: record.last_error.trim() };
  }
  if ((record.registry_status || "").toLowerCase() === "missing") {
    return { status: "missing", label: "Fichier manquant", detail: "Absent du dossier docs/." };
  }
  if (record.is_duplicate && !record.duplicate_override) {
    return { status: "duplicate_blocked", label: "Doublon bloquant", detail: record.duplicate_reason || "Validation requise." };
  }
  if (record.is_duplicate && record.duplicate_override) {
    return { status: "duplicate_whitelisted", label: "Doublon autorisé", detail: "Whitelist active." };
  }
  if (record.already_indexed) {
    return { status: "indexed", label: "Indexé", detail: "Disponible côté recherche." };
  }
  return { status: "new", label: "Nouveau", detail: "Prêt pour ingestion." };
}

function formatDateLabel(iso: string | null | undefined): { label: string; sortValue: number } {
  if (!iso) return { label: "—", sortValue: 0 };
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return { label: "—", sortValue: 0 };
  return {
    label: date.toLocaleString("fr-FR", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    }),
    sortValue: date.getTime(),
  };
}

function formatBytes(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return "—";
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(0)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

function statusBadgeClass(status: DocumentStatus): string {
  if (status === "indexed") return "border-emerald-500/35 bg-emerald-500/10 text-emerald-500";
  if (status === "new") return "border-sky-500/35 bg-sky-500/10 text-sky-400";
  if (status === "processing") return "border-amber-500/35 bg-amber-500/10 text-amber-500";
  if (status === "duplicate_whitelisted") return "border-orange-500/35 bg-orange-500/10 text-orange-400";
  if (status === "duplicate_blocked") return "border-yellow-500/35 bg-yellow-500/10 text-yellow-500";
  return "border-rose-500/35 bg-rose-500/10 text-rose-400";
}

function statusRank(status: DocumentStatus): number {
  if (status === "processing") return 0;
  if (status === "error") return 1;
  if (status === "duplicate_blocked") return 2;
  if (status === "new") return 3;
  if (status === "missing") return 4;
  if (status === "duplicate_whitelisted") return 5;
  return 6;
}

export default function DocumentsPage() {
  const chats = useChatStore((s) => s.chats);
  const token = useAuthStore((s) => s.accessToken);
  const { sendMessage, sending } = useChatActions();
  const router = useRouter();

  const [query, setQuery] = useState("");
  const [category, setCategory] = useState<MedicalCategory | "Tous">("Tous");
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  const [sortBy, setSortBy] = useState<SortOption>("recent");
  const [serverDocs, setServerDocs] = useState<DocumentRecord[] | null>(null);
  const [discoveredDocs, setDiscoveredDocs] = useState<DocsDiscoveryRecord[] | null>(null);
  const [loadingDocs, setLoadingDocs] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [actionNotice, setActionNotice] = useState<ActionNotice | null>(null);
  const [busyActions, setBusyActions] = useState<Record<string, BusyAction | undefined>>({});
  const [compareIds, setCompareIds] = useState<string[]>([]);

  useEffect(() => {
    let active = true;
    async function load() {
      setLoadingDocs(true);
      setActionError(null);
      try {
        const [docs, discovered] = await Promise.all([listDocumentsApi(token), discoverDocsApi(token)]);
        if (!active) return;
        setServerDocs(docs);
        setDiscoveredDocs(discovered);
      } catch (error) {
        if (!active) return;
        const detail = error instanceof ApiError ? error.detail : "";
        setActionError(detail || "Impossible de charger les documents backend.");
      } finally {
        if (active) setLoadingDocs(false);
      }
    }
    void load();
    return () => {
      active = false;
    };
  }, [token]);

  useEffect(() => {
    if (!actionNotice) return;
    const timer = window.setTimeout(() => setActionNotice(null), 3500);
    return () => window.clearTimeout(timer);
  }, [actionNotice]);

  const analytesByDoc = useMemo(() => {
    const map = new Map<string, Set<string>>();
    for (const chat of chats) {
      for (const message of chat.messages) {
        const textAnalytes = sourceAnalytesFromText(message.content || "");
        for (const source of message.sources || []) {
          const doc = extractSourceDoc(source);
          if (!doc) continue;
          const key = normalizeDocToken(doc);
          const entry = map.get(key) || new Set<string>();
          for (const analyte of textAnalytes) entry.add(analyte);
          map.set(key, entry);
        }
      }
    }
    return map;
  }, [chats]);

  const derivedDocs = useMemo<DocumentItem[]>(() => {
    const byDoc = new Map<string, DocumentItem>();
    const discovered = discoveredDocs || [];
    const listed = serverDocs || [];

    for (const record of discovered) {
      const statusInfo = statusFromDiscovery(record);
      const dateInfo = formatDateLabel(record.modified_at || record.last_ingested_at || record.last_seen_at);
      const key = normalizeDocToken(record.doc_id || record.filename);
      const analytes = Array.from(analytesByDoc.get(key) || []);
      byDoc.set(key, {
        id: record.doc_id,
        name: record.filename,
        category: inferCategory(record.filename, analytes),
        typeLabel: inferTypeLabel(record.filename, analytes),
        patientLabel: "Patient non renseigné",
        dateLabel: dateInfo.label,
        dateSortValue: dateInfo.sortValue,
        sizeLabel: formatBytes(Number(record.size_bytes || 0)),
        analytes,
        indexed: Boolean(record.already_indexed),
        status: statusInfo.status,
        statusLabel: statusInfo.label,
        statusDetail: statusInfo.detail,
      });
    }

    for (const doc of listed) {
      const key = normalizeDocToken(doc.id || doc.name);
      if (byDoc.has(key)) continue;
      const analytes = Array.from(analytesByDoc.get(key) || []);
      byDoc.set(key, {
        id: doc.id,
        name: doc.name,
        category: inferCategory(doc.name, analytes),
        typeLabel: inferTypeLabel(doc.name, analytes),
        patientLabel: "Patient non renseigné",
        dateLabel: "—",
        dateSortValue: 0,
        sizeLabel: "—",
        analytes,
        indexed: true,
        status: "indexed",
        statusLabel: "Indexé",
        statusDetail: "Présent en index (hors scan docs/).",
      });
    }

    return Array.from(byDoc.values());
  }, [analytesByDoc, discoveredDocs, serverDocs]);

  const stats = useMemo(() => {
    const total = derivedDocs.length;
    const indexed = derivedDocs.filter((d) => d.status === "indexed").length;
    const nonIndexed = derivedDocs.filter((d) => d.status !== "indexed").length;
    const duplicates = derivedDocs.filter((d) => d.status === "duplicate_blocked" || d.status === "duplicate_whitelisted").length;
    const errors = derivedDocs.filter((d) => d.status === "error" || d.status === "missing").length;
    return { total, indexed, nonIndexed, duplicates, errors };
  }, [derivedDocs]);

  const visibleDocs = useMemo(() => {
    const withTransientStatus = derivedDocs.map((doc) => {
      const busy = busyActions[doc.id];
      if (!busy) return doc;
      return {
        ...doc,
        status: "processing" as const,
        statusLabel: busy === "delete" ? "Suppression…" : busy === "reindex" ? "Réindexation…" : "Préparation…",
        statusDetail: busy === "ask" ? "Envoi vers le chat en cours." : doc.statusDetail,
      };
    });

    return withTransientStatus
      .filter((doc) => {
        const q = query.trim().toLowerCase();
        const searchMatch = !q
          || `${doc.name} ${doc.typeLabel} ${doc.analytes.join(" ")} ${doc.statusLabel}`.toLowerCase().includes(q);
        const categoryMatch = category === "Tous" || doc.category === category;
        const statusMatch =
          statusFilter === "all"
          || (statusFilter === "indexed" && doc.status === "indexed")
          || (statusFilter === "non_indexed" && doc.status !== "indexed")
          || (statusFilter === "duplicates" && (doc.status === "duplicate_blocked" || doc.status === "duplicate_whitelisted"))
          || (statusFilter === "error" && (doc.status === "error" || doc.status === "missing"))
          || (statusFilter === "processing" && doc.status === "processing");
        return searchMatch && categoryMatch && statusMatch;
      })
      .sort((a, b) => {
        if (sortBy === "name_asc") return a.name.localeCompare(b.name, "fr");
        if (sortBy === "name_desc") return b.name.localeCompare(a.name, "fr");
        if (sortBy === "status_priority") {
          const rank = statusRank(a.status) - statusRank(b.status);
          if (rank !== 0) return rank;
          return b.dateSortValue - a.dateSortValue;
        }
        return b.dateSortValue - a.dateSortValue || a.name.localeCompare(b.name, "fr");
      });
  }, [busyActions, category, derivedDocs, query, sortBy, statusFilter]);

  function setBusy(docId: string, action?: BusyAction) {
    setBusyActions((prev) => {
      const next = { ...prev };
      if (!action) {
        delete next[docId];
      } else {
        next[docId] = action;
      }
      return next;
    });
  }

  async function refreshDocuments(silent = false) {
    if (!silent) setLoadingDocs(true);
    try {
      const [docs, discovered] = await Promise.all([listDocumentsApi(token), discoverDocsApi(token)]);
      setServerDocs(docs);
      setDiscoveredDocs(discovered);
    } catch (error) {
      const detail = error instanceof ApiError ? error.detail : "";
      setActionError(detail || "Impossible de rafraîchir la liste des documents.");
    } finally {
      if (!silent) setLoadingDocs(false);
    }
  }

  async function openDocPreview(doc: DocumentItem) {
    setActionError(null);
    const href = `/viewer/pdf?doc_id=${encodeURIComponent(doc.id)}`;
    const win = window.open(href, "_blank", "noopener,noreferrer");
    if (!win) {
      setActionNotice({ kind: "error", message: "Impossible d’ouvrir l’aperçu PDF (popup bloquée)." });
      return;
    }
    setActionNotice({ kind: "info", message: `Aperçu PDF ouvert: ${doc.name}` });
  }

  async function askAssistant(doc: DocumentItem) {
    setBusy(doc.id, "ask");
    setActionError(null);
    try {
      await sendMessage({
        content: `Analyse le document ${doc.name} et résume les anomalies biologiques avec sources.`,
        mode: "document_analysis",
      });
      setActionNotice({ kind: "success", message: `Document envoyé au chat: ${doc.name}` });
      router.push("/chat");
    } catch (error) {
      const detail = error instanceof Error ? error.message : "";
      setActionNotice({ kind: "error", message: detail || "Échec de l’envoi au chat." });
    } finally {
      setBusy(doc.id);
    }
  }

  async function reindexDoc(docId: string) {
    setBusy(docId, "reindex");
    setActionError(null);
    try {
      await reindexDocumentApi(docId, token);
      await refreshDocuments(true);
      setActionNotice({ kind: "success", message: `Réindexation terminée: ${docId}` });
    } catch (error) {
      const detail = error instanceof ApiError ? error.detail : "";
      setActionError(detail || "Réindexation échouée.");
      setActionNotice({ kind: "error", message: detail || `Réindexation échouée: ${docId}` });
    } finally {
      setBusy(docId);
    }
  }

  async function deleteDoc(docId: string) {
    setBusy(docId, "delete");
    setActionError(null);
    try {
      await deleteDocumentApi(docId, token);
      await refreshDocuments(true);
      setCompareIds((prev) => prev.filter((id) => id !== docId));
      setActionNotice({ kind: "success", message: `Document supprimé: ${docId}` });
    } catch (error) {
      const detail = error instanceof ApiError ? error.detail : "";
      setActionError(detail || "Suppression échouée.");
      setActionNotice({ kind: "error", message: detail || `Suppression échouée: ${docId}` });
    } finally {
      setBusy(docId);
    }
  }

  function toggleCompare(docId: string) {
    setCompareIds((prev) => {
      if (prev.includes(docId)) return prev.filter((id) => id !== docId);
      if (prev.length >= 2) return [prev[1], docId];
      return [...prev, docId];
    });
  }

  async function compareWithAssistant() {
    if (compareIds.length < 2) return;
    setActionError(null);
    try {
      await sendMessage({
        content: `Compare les documents ${compareIds[0]} et ${compareIds[1]} et donne les écarts importants avec sources.`,
        mode: "comparison",
      });
      setActionNotice({ kind: "success", message: "Comparaison envoyée au chat." });
      router.push("/chat");
    } catch (error) {
      const detail = error instanceof Error ? error.message : "";
      setActionNotice({ kind: "error", message: detail || "Échec de la comparaison." });
    }
  }

  return (
    <WorkspaceShell
      title="Documents médicaux"
      subtitle="Gestion documentaire clinique"
      breadcrumbs={["Clinical Assistant", "Documents médicaux"]}
      actions={[
        { href: "/chat", label: "Retour au chat" },
        { href: "/documents/upload", label: "Importer document" },
        { href: "/chat", label: "Nouvelle conversation" },
      ]}
    >
      <main className="mx-auto max-w-7xl space-y-5 px-5 py-6 sm:px-6">
        <section className="rounded-xl border border-border/70 bg-card/[0.55] p-4">
          <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <div>
              <h2 className="text-sm font-semibold text-fg">Ingestion des nouveaux rapports</h2>
              <p className="mt-1 text-xs text-fg/65">
                Utilisez la page d’ingestion dédiée pour lancer le pipeline complet (extraction, anonymisation, chunking, indexation).
              </p>
            </div>
            <Link
              href="/documents/upload"
              className="inline-flex items-center gap-1 self-start rounded-md border border-accent/40 bg-accent/12 px-3 py-1.5 text-xs font-medium text-accent transition hover:bg-accent/18"
            >
              <UploadCloud size={13} />
              Ouvrir l’ingestion pipeline
            </Link>
          </div>
        </section>

        <section className="grid grid-cols-2 gap-2 rounded-xl border border-border/70 bg-card/[0.55] p-3 md:grid-cols-5">
          <div className="rounded-lg border border-border/70 bg-card/[0.7] px-3 py-2">
            <p className="text-[11px] uppercase tracking-wide text-fg/55">Total</p>
            <p className="mt-1 text-base font-semibold">{stats.total}</p>
          </div>
          <div className="rounded-lg border border-emerald-500/25 bg-emerald-500/10 px-3 py-2">
            <p className="text-[11px] uppercase tracking-wide text-emerald-300">Indexés</p>
            <p className="mt-1 text-base font-semibold text-emerald-300">{stats.indexed}</p>
          </div>
          <div className="rounded-lg border border-sky-500/25 bg-sky-500/10 px-3 py-2">
            <p className="text-[11px] uppercase tracking-wide text-sky-300">Non indexés</p>
            <p className="mt-1 text-base font-semibold text-sky-300">{stats.nonIndexed}</p>
          </div>
          <div className="rounded-lg border border-yellow-500/25 bg-yellow-500/10 px-3 py-2">
            <p className="text-[11px] uppercase tracking-wide text-yellow-300">Doublons</p>
            <p className="mt-1 text-base font-semibold text-yellow-300">{stats.duplicates}</p>
          </div>
          <div className="rounded-lg border border-rose-500/25 bg-rose-500/10 px-3 py-2">
            <p className="text-[11px] uppercase tracking-wide text-rose-300">Erreurs</p>
            <p className="mt-1 text-base font-semibold text-rose-300">{stats.errors}</p>
          </div>
        </section>

        <section className="rounded-xl border border-border/70 bg-card/[0.55] p-4">
          <div className="flex flex-col gap-3 xl:flex-row xl:items-center xl:justify-between">
            <div className="relative w-full max-w-lg">
              <Search size={14} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-fg/45" />
              <input
                placeholder="Rechercher un document, une analyse, un type…"
                className="h-10 w-full rounded-lg border border-border/75 bg-card/[0.65] pl-9 pr-3 text-sm outline-none transition focus:border-accent/45"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
              />
            </div>
            <div className="flex w-full flex-wrap items-center justify-end gap-2">
              <select
                value={sortBy}
                onChange={(event) => setSortBy(event.target.value as SortOption)}
                className="h-9 rounded-md border border-border/75 bg-card/[0.65] px-2 text-xs text-fg/85 outline-none"
              >
                <option value="recent">Tri: plus récent</option>
                <option value="status_priority">Tri: statut prioritaire</option>
                <option value="name_asc">Tri: nom A→Z</option>
                <option value="name_desc">Tri: nom Z→A</option>
              </select>
              {CATEGORY_OPTIONS.map((option) => (
                <button
                  key={option}
                  type="button"
                  onClick={() => setCategory(option)}
                  className={`rounded-md border px-3 py-1.5 text-xs font-medium transition ${
                    category === option
                      ? "border-accent/45 bg-accent/12 text-accent"
                      : "border-border/75 bg-card/[0.65] text-fg/75 hover:border-accent/25 hover:bg-card"
                  }`}
                >
                  {option}
                </button>
              ))}
            </div>
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            {STATUS_FILTERS.map((filter) => (
              <button
                key={filter.key}
                type="button"
                onClick={() => setStatusFilter(filter.key)}
                className={`rounded-full border px-3 py-1 text-xs transition ${
                  statusFilter === filter.key
                    ? "border-accent/45 bg-accent/12 text-accent"
                    : "border-border/70 bg-card/[0.55] text-fg/75 hover:border-accent/25"
                }`}
              >
                {filter.label}
              </button>
            ))}
          </div>
        </section>

        {loadingDocs ? (
          <section className="rounded-lg border border-border/70 bg-card/[0.55] px-4 py-3 text-sm text-fg/70">
            Chargement des documents backend…
          </section>
        ) : null}

        {actionNotice ? (
          <section
            className={`rounded-lg border px-4 py-3 text-sm ${
              actionNotice.kind === "success"
                ? "border-emerald-500/35 bg-emerald-500/10 text-emerald-300"
                : actionNotice.kind === "error"
                  ? "border-rose-500/35 bg-rose-500/10 text-rose-300"
                  : "border-sky-500/35 bg-sky-500/10 text-sky-300"
            }`}
          >
            {actionNotice.message}
          </section>
        ) : null}

        {actionError ? (
          <section className="rounded-lg border border-rose-500/35 bg-rose-500/10 px-4 py-3 text-sm text-rose-300">
            {actionError}
          </section>
        ) : null}

        {compareIds.length === 2 ? (
          <section className="flex items-center justify-between rounded-lg border border-accent/35 bg-accent/10 px-4 py-3">
            <p className="text-sm text-fg/85">Comparaison prête : {compareIds[0]} vs {compareIds[1]}</p>
            <button
              type="button"
              onClick={() => void compareWithAssistant()}
              disabled={sending}
              className="rounded-md border border-accent/35 bg-accent/15 px-3 py-1.5 text-xs font-medium text-accent transition hover:bg-accent/20 disabled:opacity-60"
            >
              {sending ? "Envoi…" : "Lancer la comparaison"}
            </button>
          </section>
        ) : null}

        <section className="grid grid-cols-1 gap-3 xl:grid-cols-2">
          {visibleDocs.map((doc) => {
            const busyAction = busyActions[doc.id];
            const docIsBusy = Boolean(busyAction);
            return (
              <article key={doc.id} className="rounded-lg border border-border/70 bg-card/[0.55] p-4">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <h2 className="line-clamp-1 text-sm font-semibold text-fg">{doc.name}</h2>
                    <p className="mt-1 text-xs text-fg/62">
                      Type : {doc.typeLabel} · {doc.patientLabel}
                    </p>
                  </div>
                  <span className={`inline-flex items-center gap-1 rounded-md border px-2 py-1 text-[11px] font-medium ${statusBadgeClass(doc.status)}`}>
                    {doc.status === "indexed" ? <CheckCircle2 size={12} /> : doc.status === "processing" ? <RefreshCw size={12} className="animate-spin" /> : doc.status === "error" || doc.status === "missing" ? <AlertCircle size={12} /> : doc.status === "duplicate_blocked" ? <ShieldAlert size={12} /> : <Clock3 size={12} />}
                    {doc.statusLabel}
                  </span>
                </div>

                <div className="mt-3 grid grid-cols-1 gap-1 text-xs text-fg/72 sm:grid-cols-2">
                  <p>Date : {doc.dateLabel}</p>
                  <p>Taille : {doc.sizeLabel}</p>
                  <p>Discipline : {doc.category}</p>
                  <p className="sm:col-span-2">
                    Analyses : {doc.analytes.length > 0 ? doc.analytes.join(", ") : "—"}
                  </p>
                  {doc.statusDetail ? <p className="sm:col-span-2 text-fg/58">{doc.statusDetail}</p> : null}
                </div>

                <div className="mt-3 flex flex-wrap gap-2">
                  <button
                    type="button"
                    onClick={() => void openDocPreview(doc)}
                    className="inline-flex items-center gap-1 rounded-md border border-border/75 bg-card/[0.7] px-2.5 py-1.5 text-xs font-medium text-fg/80 hover:bg-card"
                  >
                    <Eye size={12} /> Voir
                  </button>
                  <button
                    type="button"
                    onClick={() => void reindexDoc(doc.id)}
                    disabled={docIsBusy}
                    className="inline-flex items-center gap-1 rounded-md border border-border/75 bg-card/[0.7] px-2.5 py-1.5 text-xs font-medium text-fg/80 hover:bg-card disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    <RefreshCw size={12} className={busyAction === "reindex" ? "animate-spin" : ""} />
                    {busyAction === "reindex" ? "Réindexation…" : "Réindexer"}
                  </button>
                  <button
                    type="button"
                    onClick={() => void deleteDoc(doc.id)}
                    disabled={docIsBusy}
                    className="inline-flex items-center gap-1 rounded-md border border-rose-500/30 bg-rose-500/10 px-2.5 py-1.5 text-xs font-medium text-rose-400 hover:bg-rose-500/15 disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    <Trash2 size={12} />
                    {busyAction === "delete" ? "Suppression…" : "Supprimer"}
                  </button>
                  <button
                    type="button"
                    onClick={() => toggleCompare(doc.id)}
                    className={`inline-flex items-center gap-1 rounded-md border px-2.5 py-1.5 text-xs font-medium transition ${
                      compareIds.includes(doc.id)
                        ? "border-accent/45 bg-accent/12 text-accent"
                        : "border-border/75 bg-card/[0.7] text-fg/80 hover:bg-card"
                    }`}
                  >
                    <Scale size={12} /> Comparer
                  </button>
                  <button
                    type="button"
                    onClick={() => void askAssistant(doc)}
                    disabled={sending || docIsBusy}
                    className="inline-flex items-center gap-1 rounded-md border border-accent/35 bg-accent/10 px-2.5 py-1.5 text-xs font-medium text-accent hover:bg-accent/15 disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    <MessageSquare size={12} />
                    {busyAction === "ask" ? "Envoi…" : "Demander à l’assistant"}
                  </button>
                </div>
              </article>
            );
          })}
        </section>

        {visibleDocs.length === 0 ? (
          <section className="rounded-lg border border-border/70 bg-card/[0.55] px-4 py-5 text-sm text-fg/65">
            Aucun document ne correspond aux filtres actifs.
          </section>
        ) : null}

        <footer className="rounded-lg border border-border/70 bg-card/[0.4] px-4 py-3 text-xs text-fg/62">
          <div className="flex items-center gap-2">
            <FileText size={13} />
            <span>
              Les actions Voir/Réindexer/Supprimer/Comparer/Demander affichent désormais un feedback runtime. Les statuts sont basés sur le registre docs + index.
            </span>
          </div>
        </footer>
      </main>
    </WorkspaceShell>
  );
}
