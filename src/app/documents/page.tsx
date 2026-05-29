"use client";

import Link from "next/link";
import { CheckCircle2, Eye, FileText, MessageSquare, RefreshCw, Scale, Search, Trash2, UploadCloud } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { WorkspaceShell } from "@/components/layout/workspace-shell";
import { useChatActions } from "@/hooks/use-chat-actions";
import { ApiError, deleteDocumentApi, listDocumentsApi, reindexDocumentApi, type DocumentRecord } from "@/services/rag-api";
import { useAuthStore } from "@/store/auth-store";
import { useChatStore } from "@/store/chat-store";
import type { ChatSource } from "@/types/chat";

type MedicalCategory = "Biochimie" | "Hématologie" | "Toxicologie" | "Parasitologie";
type DocumentStatus = "indexed" | "processing";

type DocumentItem = {
  id: string;
  name: string;
  category: MedicalCategory;
  typeLabel: string;
  patientLabel: string;
  dateLabel: string;
  analytes: string[];
  indexed: boolean;
  status: DocumentStatus;
};

const CATEGORY_OPTIONS: Array<MedicalCategory | "Tous"> = ["Tous", "Biochimie", "Hématologie", "Toxicologie", "Parasitologie"];

const FALLBACK_DOCS: DocumentItem[] = [
  {
    id: "report_16.pdf",
    name: "report_16.pdf",
    category: "Biochimie",
    typeLabel: "Immunoanalyse",
    patientLabel: "Patient test",
    dateLabel: "20/06/2025",
    analytes: ["TSH", "T3", "T4", "Anti-TG"],
    indexed: true,
    status: "indexed",
  },
  {
    id: "report_29.pdf",
    name: "report_29.pdf",
    category: "Hématologie",
    typeLabel: "Numération",
    patientLabel: "Patient test",
    dateLabel: "18/05/2025",
    analytes: ["Hb", "VGM", "Plaquettes"],
    indexed: true,
    status: "indexed",
  },
];

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

export default function DocumentsPage() {
  const chats = useChatStore((s) => s.chats);
  const token = useAuthStore((s) => s.accessToken);
  const { sendMessage, sending } = useChatActions();
  const router = useRouter();

  const [query, setQuery] = useState("");
  const [category, setCategory] = useState<MedicalCategory | "Tous">("Tous");
  const [serverDocs, setServerDocs] = useState<DocumentRecord[] | null>(null);
  const [loadingDocs, setLoadingDocs] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [reindexingIds, setReindexingIds] = useState<Set<string>>(new Set());
  const [compareIds, setCompareIds] = useState<string[]>([]);

  useEffect(() => {
    let active = true;
    async function load() {
      setLoadingDocs(true);
      setActionError(null);
      try {
        const docs = await listDocumentsApi(token);
        if (active) setServerDocs(docs);
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

  const fallbackFromSources = useMemo(() => {
    const docs = new Map<string, { name: string; analytes: Set<string>; updatedAt?: string }>();
    for (const chat of chats) {
      for (const message of chat.messages) {
        const textAnalytes = sourceAnalytesFromText(message.content || "");
        for (const source of message.sources || []) {
          const doc = extractSourceDoc(source);
          if (!doc) continue;
          const entry = docs.get(doc) || { name: doc, analytes: new Set<string>(), updatedAt: chat.updatedAt };
          for (const analyte of textAnalytes) entry.analytes.add(analyte);
          if (!entry.updatedAt || chat.updatedAt > entry.updatedAt) entry.updatedAt = chat.updatedAt;
          docs.set(doc, entry);
        }
      }
    }

    const mapped: DocumentItem[] = Array.from(docs.values()).map((entry) => {
      const analytes = Array.from(entry.analytes);
      const date = entry.updatedAt ? new Date(entry.updatedAt) : null;
      const dateLabel = date
        ? date.toLocaleDateString("fr-FR")
        : "Date inconnue";
      return {
        id: entry.name,
        name: entry.name,
        category: inferCategory(entry.name, analytes),
        typeLabel: inferTypeLabel(entry.name, analytes),
        patientLabel: "Patient test",
        dateLabel,
        analytes: analytes.length > 0 ? analytes : ["Non précisé"],
        indexed: true,
        status: "indexed",
      };
    });

    return mapped.length > 0 ? mapped : FALLBACK_DOCS;
  }, [chats]);

  const derivedDocs = useMemo(() => {
    if (serverDocs && serverDocs.length > 0) {
      return serverDocs.map((doc) => {
        const analytes = sourceAnalytesFromText(doc.name);
        return {
          id: doc.id,
          name: doc.name,
          category: inferCategory(doc.name, analytes),
          typeLabel: inferTypeLabel(doc.name, analytes),
          patientLabel: "Patient test",
          dateLabel: "Date inconnue",
          analytes: analytes.length > 0 ? analytes : ["Non précisé"],
          indexed: true,
          status: "indexed" as const,
        };
      });
    }
    return fallbackFromSources;
  }, [fallbackFromSources, serverDocs]);

  const visibleDocs = useMemo(() => {
    return derivedDocs
      .map((doc) => ({
        ...doc,
        status: reindexingIds.has(doc.id) ? ("processing" as const) : doc.status,
      }))
      .filter((doc) => {
        const q = query.trim().toLowerCase();
        const searchMatch = !q || `${doc.name} ${doc.typeLabel} ${doc.analytes.join(" ")}`.toLowerCase().includes(q);
        const categoryMatch = category === "Tous" || doc.category === category;
        return searchMatch && categoryMatch;
      });
  }, [category, derivedDocs, query, reindexingIds]);

  async function askAssistant(doc: DocumentItem) {
    await sendMessage({
      content: `Analyse le document ${doc.name} et résume les anomalies biologiques avec sources.`,
      mode: "document_analysis",
    });
    router.push("/chat");
  }

  async function refreshDocuments() {
    try {
      const docs = await listDocumentsApi(token);
      setServerDocs(docs);
    } catch (error) {
      const detail = error instanceof ApiError ? error.detail : "";
      setActionError(detail || "Impossible de rafraîchir la liste des documents.");
    }
  }

  async function reindexDoc(docId: string) {
    setReindexingIds((prev) => new Set(prev).add(docId));
    setActionError(null);
    try {
      await reindexDocumentApi(docId, token);
      await refreshDocuments();
    } catch (error) {
      const detail = error instanceof ApiError ? error.detail : "";
      setActionError(detail || "Réindexation échouée.");
    } finally {
      setReindexingIds((prev) => {
        const next = new Set(prev);
        next.delete(docId);
        return next;
      });
    }
  }

  async function deleteDoc(docId: string) {
    setActionError(null);
    try {
      await deleteDocumentApi(docId, token);
      await refreshDocuments();
      setCompareIds((prev) => prev.filter((id) => id !== docId));
    } catch (error) {
      const detail = error instanceof ApiError ? error.detail : "";
      setActionError(detail || "Suppression échouée.");
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
    await sendMessage({
      content: `Compare les documents ${compareIds[0]} et ${compareIds[1]} et donne les écarts importants avec sources.`,
      mode: "comparison",
    });
    router.push("/chat");
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

      <section className="rounded-xl border border-border/70 bg-card/[0.55] p-4">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
          <div className="relative w-full max-w-lg">
            <Search size={14} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-fg/45" />
            <input
              placeholder="Rechercher un document, une analyse, un type…"
              className="h-10 w-full rounded-lg border border-border/75 bg-card/[0.65] pl-9 pr-3 text-sm outline-none transition focus:border-accent/45"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
          </div>
          <div className="flex flex-wrap gap-2">
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
      </section>

      {loadingDocs ? (
        <section className="rounded-lg border border-border/70 bg-card/[0.55] px-4 py-3 text-sm text-fg/70">
          Chargement des documents backend…
        </section>
      ) : null}

      {actionError ? (
        <section className="rounded-lg border border-rose-500/35 bg-rose-500/10 px-4 py-3 text-sm text-rose-500">
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
            Lancer la comparaison
          </button>
        </section>
      ) : null}

      <section className="grid grid-cols-1 gap-3 xl:grid-cols-2">
        {visibleDocs.map((doc) => (
          <article key={doc.id} className="rounded-lg border border-border/70 bg-card/[0.55] p-4">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <h2 className="line-clamp-1 text-sm font-semibold text-fg">{doc.name}</h2>
                <p className="mt-1 text-xs text-fg/62">
                  Type : {doc.typeLabel} · {doc.patientLabel}
                </p>
              </div>
              <span
                className={`inline-flex items-center gap-1 rounded-md border px-2 py-1 text-[11px] font-medium ${
                  doc.status === "indexed"
                    ? "border-emerald-500/35 bg-emerald-500/10 text-emerald-600"
                    : "border-amber-500/35 bg-amber-500/10 text-amber-600"
                }`}
              >
                {doc.status === "indexed" ? <CheckCircle2 size={12} /> : <RefreshCw size={12} className="animate-spin" />}
                {doc.status === "indexed" ? "Indexé : Oui" : "Réindexation…"}
              </span>
            </div>

            <div className="mt-3 grid grid-cols-1 gap-1 text-xs text-fg/72 sm:grid-cols-2">
              <p>Date : {doc.dateLabel}</p>
              <p>Discipline : {doc.category}</p>
              <p className="sm:col-span-2">Analyses : {doc.analytes.join(", ")}</p>
            </div>

            <div className="mt-3 flex flex-wrap gap-2">
              <a
                href={`/viewer/pdf?doc_id=${encodeURIComponent(doc.name)}`}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 rounded-md border border-border/75 bg-card/[0.7] px-2.5 py-1.5 text-xs font-medium text-fg/80 hover:bg-card"
              >
                <Eye size={12} /> Voir
              </a>
              <button
                type="button"
                onClick={() => void reindexDoc(doc.id)}
                className="inline-flex items-center gap-1 rounded-md border border-border/75 bg-card/[0.7] px-2.5 py-1.5 text-xs font-medium text-fg/80 hover:bg-card"
              >
                <RefreshCw size={12} /> Réindexer
              </button>
              <button
                type="button"
                onClick={() => void deleteDoc(doc.id)}
                className="inline-flex items-center gap-1 rounded-md border border-rose-500/30 bg-rose-500/10 px-2.5 py-1.5 text-xs font-medium text-rose-500 hover:bg-rose-500/15"
              >
                <Trash2 size={12} /> Supprimer
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
                disabled={sending}
                className="inline-flex items-center gap-1 rounded-md border border-accent/35 bg-accent/10 px-2.5 py-1.5 text-xs font-medium text-accent hover:bg-accent/15 disabled:opacity-60"
              >
                <MessageSquare size={12} /> Demander à l’assistant
              </button>
            </div>
          </article>
        ))}
      </section>

      {visibleDocs.length === 0 ? (
        <section className="rounded-lg border border-border/70 bg-card/[0.55] px-4 py-5 text-sm text-fg/65">
          Aucun document ne correspond à votre recherche.
        </section>
      ) : null}

        <footer className="rounded-lg border border-border/70 bg-card/[0.4] px-4 py-3 text-xs text-fg/62">
        <div className="flex items-center gap-2">
          <FileText size={13} />
          <span>Les actions réindexer/supprimer sont reliées au backend. Utilisez “Ingestion pipeline” pour importer de nouveaux rapports.</span>
        </div>
        </footer>
      </main>
    </WorkspaceShell>
  );
}
