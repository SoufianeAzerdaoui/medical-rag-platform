"use client";

import { Activity, Clock3, Database, FileText, MessageSquare, Sparkles, Wifi, Workflow } from "lucide-react";
import { useEffect, useMemo, useState, type ComponentType, type ReactNode } from "react";
import { WorkspaceShell } from "@/components/layout/workspace-shell";
import { healthcheck } from "@/services/rag-api";
import { useChatStore } from "@/store/chat-store";
import type { ChatSource } from "@/types/chat";

function sourceDocName(source: ChatSource): string {
  if (typeof source === "string") {
    const fromDocId = source.match(/doc_id=([^,\]\s]+)/i)?.[1];
    if (fromDocId) return fromDocId;
    const fromPdf = source.match(/([A-Za-z0-9_\-().]+\.(?:pdf|PDF))/)?.[1];
    return fromPdf || "document";
  }
  const raw = source as Record<string, unknown>;
  return String(raw.filename || raw.documentName || raw.doc_id || raw.documentId || raw.label || "document");
}

function normalizeQuestionType(content: string): string {
  const text = content.toLowerCase();
  if (/(résum|synthèse|summary)/i.test(text)) return "Résumé biologique";
  if (/(anormal|hors référence|valeurs? anormales?)/i.test(text)) return "Valeurs anormales";
  if (/(compar|différence|evolution|évolution)/i.test(text)) return "Comparaison de rapports";
  if (/(tsh|valeur|analyte|marqueur|résultat)/i.test(text)) return "Recherche de valeur";
  return "Question clinique";
}

export default function DashboardPage() {
  const chats = useChatStore((s) => s.chats);
  const [backendStatus, setBackendStatus] = useState<"online" | "offline" | "checking">("checking");

  useEffect(() => {
    let active = true;
    async function refreshHealth() {
      const status = await healthcheck();
      if (!active) return;
      setBackendStatus(status === "online" ? "online" : "offline");
    }
    void refreshHealth();
    const timer = window.setInterval(() => void refreshHealth(), 30_000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  const metrics = useMemo(() => {
    const allMessages = chats.flatMap((chat) => chat.messages);
    const assistantDone = allMessages.filter((m) => m.role === "assistant" && m.status === "done");
    const userMessages = allMessages.filter((m) => m.role === "user");
    const messagesWithSources = assistantDone.filter((m) => (m.sources?.length || 0) > 0);
    const sourceRate = assistantDone.length > 0 ? (messagesWithSources.length / assistantDone.length) * 100 : 0;
    const responseTimes = assistantDone
      .map((m) => m.diagnostics?.response_time)
      .filter((v): v is number => typeof v === "number" && Number.isFinite(v) && v > 0);
    const avgResponse = responseTimes.length > 0
      ? responseTimes.reduce((sum, value) => sum + value, 0) / responseTimes.length
      : null;

    const documentSet = new Set<string>();
    for (const message of allMessages) {
      for (const source of message.sources || []) {
        const name = sourceDocName(source).trim();
        if (name) documentSet.add(name);
      }
    }

    const recentDocs = Array.from(documentSet).slice(0, 6);
    const questionCounts = new Map<string, number>();
    for (const message of userMessages) {
      const key = normalizeQuestionType(message.content || "");
      questionCounts.set(key, (questionCounts.get(key) || 0) + 1);
    }
    const topQuestions = Array.from(questionCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([label]) => label);

    return {
      conversations: chats.length,
      questionsAnswered: assistantDone.length,
      indexedDocuments: documentSet.size,
      sourceRate,
      avgResponse,
      recentDocs,
      topQuestions,
    };
  }, [chats]);

  const healthRows = [
    { label: "Vector DB", ok: backendStatus === "online" },
    { label: "LLM", ok: backendStatus === "online" },
    { label: "Embeddings", ok: backendStatus === "online" },
    { label: "API", ok: backendStatus === "online" },
  ];

  return (
    <WorkspaceShell
      title="Dashboard clinique"
      subtitle="Vue globale de la plateforme RAG"
      breadcrumbs={["Clinical Assistant", "Dashboard clinique"]}
      actions={[
        { href: "/chat", label: "Retour au chat" },
        { href: "/documents", label: "Documents" },
        { href: "/chat", label: "Nouvelle conversation" },
      ]}
    >
      <main className="mx-auto max-w-7xl space-y-6 px-5 py-6 sm:px-6">

      <section className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
        <KpiCard icon={FileText} label="Documents indexés" value={String(metrics.indexedDocuments)} />
        <KpiCard icon={MessageSquare} label="Conversations" value={String(metrics.conversations)} />
        <KpiCard icon={Sparkles} label="Questions répondues" value={String(metrics.questionsAnswered)} />
        <KpiCard icon={Workflow} label="Taux avec sources" value={`${Math.round(metrics.sourceRate)}%`} />
        <KpiCard
          icon={Clock3}
          label="Temps moyen réponse"
          value={metrics.avgResponse !== null ? `${metrics.avgResponse.toFixed(1)}s` : "n/a"}
        />
        <KpiCard
          icon={Wifi}
          label="Backend"
          value={backendStatus === "checking" ? "Checking" : backendStatus === "online" ? "Online" : "Offline"}
          tone={backendStatus === "online" ? "good" : backendStatus === "offline" ? "bad" : "neutral"}
        />
      </section>

      <section className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        <Panel title="Documents récents" icon={FileText}>
          {metrics.recentDocs.length === 0 ? (
            <EmptyText text="Aucun document référencé pour le moment." />
          ) : (
            <ul className="space-y-2">
              {metrics.recentDocs.map((doc) => (
                <li key={doc} className="rounded-md border border-border/65 bg-card/[0.46] px-3 py-2 text-sm text-fg/85">
                  {doc}
                </li>
              ))}
            </ul>
          )}
        </Panel>

        <Panel title="Top questions" icon={MessageSquare}>
          {metrics.topQuestions.length === 0 ? (
            <EmptyText text="Aucune question disponible." />
          ) : (
            <ul className="space-y-2">
              {metrics.topQuestions.map((question) => (
                <li key={question} className="rounded-md border border-border/65 bg-card/[0.46] px-3 py-2 text-sm text-fg/85">
                  {question}
                </li>
              ))}
            </ul>
          )}
        </Panel>

        <Panel title="Santé du système" icon={Database}>
          <ul className="space-y-2">
            {healthRows.map((row) => (
              <li key={row.label} className="flex items-center justify-between rounded-md border border-border/65 bg-card/[0.46] px-3 py-2 text-sm">
                <span className="text-fg/85">{row.label}</span>
                <span className={row.ok ? "text-emerald-500" : "text-rose-500"}>{row.ok ? "OK" : "Issue"}</span>
              </li>
            ))}
          </ul>
        </Panel>
      </section>
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
      <p className={tone === "good" ? "text-2xl font-semibold text-emerald-500" : tone === "bad" ? "text-2xl font-semibold text-rose-500" : "text-2xl font-semibold text-fg"}>
        {value}
      </p>
    </article>
  );
}

function Panel({
  title,
  icon: Icon,
  children,
}: {
  title: string;
  icon: ComponentType<{ size?: string | number; className?: string }>;
  children: ReactNode;
}) {
  return (
    <section className="rounded-lg border border-border/70 bg-card/[0.55] p-4">
      <div className="mb-3 flex items-center gap-2">
        <Icon size={15} className="text-accent" />
        <h2 className="text-sm font-semibold text-fg">{title}</h2>
      </div>
      {children}
    </section>
  );
}

function EmptyText({ text }: { text: string }) {
  return <p className="rounded-md border border-border/65 bg-card/[0.46] px-3 py-2 text-sm text-fg/65">{text}</p>;
}
