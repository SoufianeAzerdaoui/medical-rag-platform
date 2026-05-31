"use client";

import Link from "next/link";
import { useMemo } from "react";
import { useTheme } from "next-themes";
import { WorkspaceShell } from "@/components/layout/workspace-shell";
import { useChatStore } from "@/store/chat-store";

export default function SettingsPage() {
  const clearAllData = useChatStore((s) => s.clearAllData);
  const privacyMode = useChatStore((s) => s.privacyMode);
  const togglePrivacyMode = useChatStore((s) => s.togglePrivacyMode);
  const qualityDebugEnabled = useChatStore((s) => s.qualityDebugEnabled);
  const toggleQualityDebug = useChatStore((s) => s.toggleQualityDebug);
  const language = useChatStore((s) => s.language);
  const setLanguage = useChatStore((s) => s.setLanguage);
  const exportAllChats = useChatStore((s) => s.exportAllChats);
  const themePref = useChatStore((s) => s.theme);
  const setThemePref = useChatStore((s) => s.setTheme);
  const { setTheme } = useTheme();
  const apiUrl = useMemo(() => process.env.NEXT_PUBLIC_RAG_API_URL || "Not set", []);

  function onExport(format: "json" | "txt") {
    const content = exportAllChats(format);
    const blob = new Blob([content], { type: format === "json" ? "application/json" : "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `clinical-rag-conversations.${format}`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <WorkspaceShell
      title="Paramètres"
      subtitle="Préférences de la plateforme clinique"
      breadcrumbs={["Clinical Assistant", "Settings"]}
      actions={[
        { href: "/chat", label: "Retour au chat" },
        { href: "/documents", label: "Importer document" },
        { href: "/chat", label: "Nouvelle conversation" },
      ]}
    >
      <main className="mx-auto max-w-4xl space-y-4 p-6">
      <section className="card p-4">
        <p className="text-sm">API Base URL: {apiUrl}</p>
      </section>
      <section className="card p-4">
        <p className="mb-2 text-sm">Administration Ops</p>
        <Link
          href="/settings/security-retention"
          className="inline-flex items-center gap-2 rounded-lg border border-accent/40 bg-accent/12 px-3 py-2 text-sm font-medium text-accent transition hover:bg-accent/18 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/40"
        >
          Ouvrir Security & Retention
        </Link>
      </section>
      <section className="card p-4">
        <p className="mb-2 text-sm">Language</p>
        <select
          aria-label="Choisir la langue"
          className="rounded-lg border border-border bg-card/80 px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/40"
          value={language}
          onChange={(e) => setLanguage(e.target.value as "fr" | "ar" | "en")}
        >
          <option value="fr">Français</option>
          <option value="ar">العربية</option>
          <option value="en">English</option>
        </select>
      </section>
      <section className="card p-4">
        <p className="mb-2 text-sm">Theme</p>
        <div className="flex gap-2">
          {(["dark", "light", "system"] as const).map((theme) => (
            <button
              key={theme}
              onClick={() => {
                setThemePref(theme);
                setTheme(theme);
              }}
              className={`rounded-lg border bg-card/75 px-3 py-2 text-sm transition hover:bg-fg/[0.03] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/40 ${themePref === theme ? "border-accent text-accent" : "border-border text-fg/80"}`}
              aria-label={`Activer thème ${theme}`}
            >
              {theme}
            </button>
          ))}
        </div>
      </section>
      <section className="card p-4">
        <p className="mb-2 text-sm">Microphone permissions are required for voice input.</p>
        <button onClick={togglePrivacyMode} className="rounded-lg border border-border bg-card/75 px-3 py-2 text-sm transition hover:bg-fg/[0.03] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/40">
          Privacy mode: {privacyMode ? "ON" : "OFF"}
        </button>
      </section>
      <section className="card p-4">
        <p className="mb-2 text-sm">Generation quality debug dashboard</p>
        <button onClick={toggleQualityDebug} className="rounded-lg border border-border bg-card/75 px-3 py-2 text-sm transition hover:bg-fg/[0.03] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/40">
          Quality debug: {qualityDebugEnabled ? "ON" : "OFF"}
        </button>
      </section>
      <section className="card p-4">
        <p className="mb-3 text-sm">Local conversations are stored in IndexedDB.</p>
        <div className="flex flex-wrap gap-2">
          <button onClick={() => onExport("json")} className="rounded-lg border border-border bg-card/75 px-3 py-2 text-sm transition hover:bg-fg/[0.03] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/40">
            Export all JSON
          </button>
          <button onClick={() => onExport("txt")} className="rounded-lg border border-border bg-card/75 px-3 py-2 text-sm transition hover:bg-fg/[0.03] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/40">
            Export all TXT
          </button>
          <button onClick={() => void clearAllData()} className="status-danger rounded-lg px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-500/40">
          Clear all conversations
          </button>
        </div>
      </section>
      <p className="text-xs text-fg/70">Cette réponse ne remplace pas l&apos;avis médical.</p>
      </main>
    </WorkspaceShell>
  );
}
