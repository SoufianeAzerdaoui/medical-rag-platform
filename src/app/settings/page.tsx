"use client";

import { useMemo } from "react";
import { useTheme } from "next-themes";
import { useChatStore } from "@/store/chat-store";

export default function SettingsPage() {
  const clearAllData = useChatStore((s) => s.clearAllData);
  const privacyMode = useChatStore((s) => s.privacyMode);
  const togglePrivacyMode = useChatStore((s) => s.togglePrivacyMode);
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
    <main className="mx-auto max-w-4xl space-y-4 p-6">
      <h1 className="text-2xl font-semibold">Settings</h1>
      <section className="rounded-xl border border-border p-4">
        <p className="text-sm">API Base URL: {apiUrl}</p>
      </section>
      <section className="rounded-xl border border-border p-4">
        <p className="mb-2 text-sm">Language</p>
        <select
          aria-label="Choisir la langue"
          className="rounded-lg border border-border bg-transparent px-3 py-2 text-sm"
          value={language}
          onChange={(e) => setLanguage(e.target.value as "fr" | "ar" | "en")}
        >
          <option value="fr">Français</option>
          <option value="ar">العربية</option>
          <option value="en">English</option>
        </select>
      </section>
      <section className="rounded-xl border border-border p-4">
        <p className="mb-2 text-sm">Theme</p>
        <div className="flex gap-2">
          {(["dark", "light", "system"] as const).map((theme) => (
            <button
              key={theme}
              onClick={() => {
                setThemePref(theme);
                setTheme(theme);
              }}
              className={`rounded-lg border px-3 py-2 text-sm ${themePref === theme ? "border-accent" : "border-border"}`}
              aria-label={`Activer thème ${theme}`}
            >
              {theme}
            </button>
          ))}
        </div>
      </section>
      <section className="rounded-xl border border-border p-4">
        <p className="mb-2 text-sm">Microphone permissions are required for voice input.</p>
        <button onClick={togglePrivacyMode} className="rounded-lg border border-border px-3 py-2 text-sm">
          Privacy mode: {privacyMode ? "ON" : "OFF"}
        </button>
      </section>
      <section className="rounded-xl border border-border p-4">
        <p className="mb-3 text-sm">Local conversations are stored in IndexedDB.</p>
        <div className="flex flex-wrap gap-2">
          <button onClick={() => onExport("json")} className="rounded-lg border border-border px-3 py-2 text-sm">
            Export all JSON
          </button>
          <button onClick={() => onExport("txt")} className="rounded-lg border border-border px-3 py-2 text-sm">
            Export all TXT
          </button>
          <button onClick={() => void clearAllData()} className="rounded-lg border border-red-400 px-3 py-2 text-sm text-red-300">
          Clear all conversations
          </button>
        </div>
      </section>
      <p className="text-xs text-fg/70">Cette réponse ne remplace pas l'avis médical.</p>
    </main>
  );
}
