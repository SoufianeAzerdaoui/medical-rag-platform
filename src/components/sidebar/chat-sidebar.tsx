"use client";

import { AnimatePresence, motion } from "framer-motion";
import { Activity, Download, Ellipsis, FileText, Heart, LayoutDashboard, LogOut, MessageSquare, MessageSquarePlus, Search, Settings, SunMoon, Trash2 } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useTheme } from "next-themes";
import { healthcheck } from "@/services/rag-api";
import { buildSubtitleMeta } from "@/lib/chat-title";
import { useAuthStore } from "@/store/auth-store";
import { useChatStore } from "@/store/chat-store";
import { cn } from "@/lib/utils";
import type { ChatItem, ChatSource } from "@/types/chat";

function daysBetween(dateIso: string) {
  const now = new Date();
  const date = new Date(dateIso);
  const startNow = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const startDate = new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();
  return Math.floor((startNow - startDate) / (1000 * 60 * 60 * 24));
}

function sourceDocKey(source: ChatSource): string {
  if (typeof source === "string") {
    const match = source.match(/doc_id=([^,\]\s]+)/i);
    if (match?.[1]) return match[1].trim().toLowerCase();
    return source.slice(0, 48).toLowerCase();
  }
  const raw = source as Record<string, unknown>;
  const docId = String(raw.doc_id || raw.documentId || raw.filename || raw.documentName || raw.label || "").trim().toLowerCase();
  return docId;
}

function summarizeConversation(chat: ChatItem): { preview: string; sourceCount: number; meta: string } {
  const lastMessage = [...chat.messages].reverse().find((m) => m.role === "assistant" || m.role === "user") || chat.messages.at(-1);
  const preview = lastMessage?.content?.replace(/\s+/g, " ").trim() || "Nouvelle conversation";
  const docs = new Set<string>();
  for (const message of chat.messages) {
    for (const source of message.sources || []) {
      const key = sourceDocKey(source);
      if (key) docs.add(key);
    }
  }
  return {
    preview: preview.length > 44 ? `${preview.slice(0, 44)}…` : preview,
    sourceCount: docs.size,
    meta: buildSubtitleMeta({ updatedAt: chat.updatedAt, sourceCount: docs.size }),
  };
}

export function ChatSidebar() {
  const chats = useChatStore((s) => s.chats);
  const activeChatId = useChatStore((s) => s.activeChatId);
  const search = useChatStore((s) => s.search);
  const setSearch = useChatStore((s) => s.setSearch);
  const startNewConversation = useChatStore((s) => s.startNewConversation);
  const removeChat = useChatStore((s) => s.removeChat);
  const toggleFavorite = useChatStore((s) => s.toggleFavorite);
  const renameChat = useChatStore((s) => s.renameChat);
  const exportChat = useChatStore((s) => s.exportChat);
  const themePref = useChatStore((s) => s.theme);
  const setThemePref = useChatStore((s) => s.setTheme);
  const user = useAuthStore((s) => s.user);
  const token = useAuthStore((s) => s.accessToken);
  const logout = useAuthStore((s) => s.logout);
  const { setTheme } = useTheme();
  const router = useRouter();
  const pathname = usePathname();
  const [menuChatId, setMenuChatId] = useState<string | null>(null);
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const [backendStatus, setBackendStatus] = useState<"online" | "offline" | "checking">("checking");
  const searchRef = useRef<HTMLInputElement | null>(null);

  const filtered = useMemo(() => {
    const q = search.toLowerCase();
    return chats.filter((c) => {
      if (!q) return true;
      return (
        c.title.toLowerCase().includes(q) ||
        c.tags.join(" ").toLowerCase().includes(q) ||
        c.messages.some((m) => m.content.toLowerCase().includes(q))
      );
    });
  }, [chats, search]);

  const grouped = useMemo(() => {
    const favorites = filtered.filter((chat) => chat.favorite);
    const nonFavorites = filtered.filter((chat) => !chat.favorite);
    return {
      Favoris: favorites,
      "Aujourd’hui": nonFavorites.filter((chat) => daysBetween(chat.updatedAt) === 0),
      "7 derniers jours": nonFavorites.filter((chat) => {
        const days = daysBetween(chat.updatedAt);
        return days >= 1 && days <= 7;
      }),
      Archives: nonFavorites.filter((chat) => daysBetween(chat.updatedAt) > 7),
    };
  }, [filtered]);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        searchRef.current?.focus();
      }
      if (event.key === "Escape") {
        setMenuChatId(null);
        setUserMenuOpen(false);
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  useEffect(() => {
    let active = true;
    async function checkBackend() {
      const status = await healthcheck();
      if (active) setBackendStatus(status === "online" ? "online" : "offline");
    }
    void checkBackend();
    const timer = window.setInterval(() => void checkBackend(), 30_000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  function downloadChat(chatId: string, format: "json" | "txt") {
    const content = exportChat(chatId, format);
    if (!content) return;
    const blob = new Blob([content], { type: format === "json" ? "application/json" : "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `chat-${chatId}.${format}`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function onRename(chatId: string) {
    const title = window.prompt("Nouveau nom de conversation");
    if (title !== null) renameChat(chatId, title.trim());
  }

  function toggleTheme() {
    const next = themePref === "dark" ? "light" : "dark";
    setThemePref(next);
    setTheme(next);
  }

  async function onStartConversation() {
    const createdId = await startNewConversation(token);
    if (createdId) {
      router.push(`/chat/${createdId}`);
    }
  }

  async function onRemoveConversation(chatId: string) {
    await removeChat(chatId, token);
    if (pathname?.startsWith(`/chat/${chatId}`)) {
      const nextConversationId = useChatStore.getState().activeConversationId;
      if (nextConversationId) {
        router.push(`/chat/${nextConversationId}`);
      } else {
        router.push("/chat");
      }
    }
  }

  const navItems = [
    { href: "/chat", label: "Chat", icon: MessageSquare },
    { href: "/documents", label: "Documents", icon: FileText },
    { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  ] as const;

  return (
    <aside className="glass flex h-screen w-80 shrink-0 flex-col border-y-0 border-l-0 p-4">
      <div className="mb-4 rounded-xl border border-border/70 bg-card/[0.46] p-3">
        <div className="flex items-start gap-3">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-accent text-white shadow-sm">
          <Activity size={18} />
        </div>
        <div className="min-w-0">
          <h1 className="text-sm font-semibold leading-5">CHU Oujda</h1>
          <p className="text-xs text-fg/62">Clinical Assistant</p>
          <p className="mt-0.5 text-xs text-fg/60">
            ●{" "}
            <span
              className={cn(
                "font-medium",
                backendStatus === "online" && "text-[hsl(var(--success))]",
                backendStatus === "offline" && "text-[hsl(var(--danger))]",
                backendStatus === "checking" && "text-[hsl(var(--warning))]",
              )}
            >
              {backendStatus}
            </span>
          </p>
        </div>
        </div>
      </div>
      <button
        aria-label="Nouveau chat"
        onClick={() => void onStartConversation()}
        className="mb-3 flex h-10 items-center justify-center gap-2 rounded-lg bg-accent px-3 py-2 text-sm font-semibold text-slate-950 shadow-sm transition duration-200 hover:-translate-y-0.5 hover:bg-accent/90 active:translate-y-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/55"
      >
        <MessageSquarePlus size={16} /> Nouvelle conversation
      </button>
      <div className="mb-3 flex h-10 items-center gap-2 rounded-lg border border-border/80 bg-card/[0.72] px-3 py-2 shadow-sm">
        <Search size={14} className="text-fg/[0.48]" />
        <input
          ref={searchRef}
          aria-label="Rechercher une conversation"
          placeholder="Rechercher une conversation"
          className="w-full bg-transparent text-sm outline-none placeholder:text-fg/[0.42]"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>
      <section className="mb-3 rounded-lg border border-border/70 bg-card/[0.45] p-2">
        <p className="mb-2 px-1 text-[11px] font-semibold uppercase tracking-[0.14em] text-fg/[0.72]">Navigation</p>
        <div className="space-y-1">
          {navItems.map((item) => {
            const active = pathname === item.href || (item.href === "/chat" && pathname?.startsWith("/chat"));
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex h-[42px] items-center gap-2 rounded-xl border px-3 text-sm transition",
                  "text-fg/65",
                  active
                    ? "nav-active"
                    : "border-transparent hover:border-border/70 hover:bg-card/[0.72]",
                )}
              >
                <item.icon size={15} />
                <span className="font-medium">{item.label}</span>
              </Link>
            );
          })}
        </div>
      </section>
      <div className="flex-1 space-y-3 overflow-auto pr-0.5">
        {Object.entries(grouped).map(([label, items]) =>
          items.length === 0 ? null : (
            <section key={label} aria-label={label}>
              <p className="mb-1 text-[11px] font-semibold uppercase tracking-[0.14em] text-fg/[0.72]">{label}</p>
              <div className="space-y-1.5">
                {items.map((chat) => {
                  const summary = summarizeConversation(chat);
                  return (
                  <motion.div
                    key={chat.id}
                    initial={{ opacity: 0, y: 4 }}
                    animate={{ opacity: 1, y: 0 }}
                    whileHover={{ y: -1 }}
                    transition={{ duration: 0.16, ease: "easeOut" }}
                    className={cn(
                      "rounded-lg border px-3 py-2.5 transition duration-200",
                      chat.id === activeChatId
                        ? "border-accent/[0.55] bg-accent/10 shadow-sm ring-1 ring-accent/25"
                        : "border-border/70 bg-card/[0.42] hover:border-accent/30 hover:bg-card/[0.72] hover:shadow-sm",
                    )}
                  >
                    <button
                      className="w-full text-left focus-visible:outline-none"
                      onClick={() => router.push(`/chat/${chat.id}`)}
                      title={chat.title}
                      aria-current={chat.id === activeChatId ? "page" : undefined}
                    >
                      <p className="line-clamp-1 text-sm font-medium text-fg/90">{chat.title}</p>
                      <p className="mt-1 line-clamp-1 text-xs leading-5 text-fg/[0.78]">
                        {summary.preview.toLowerCase() === "nouvelle conversation" ? "Conversation générale" : summary.preview}
                      </p>
                      <div className="mt-1.5 flex items-center gap-1.5">
                        <span
                          className={cn(
                            "rounded-md border px-1.5 py-0.5 text-[11px] font-medium",
                            summary.sourceCount > 0
                              ? "border-accent/25 bg-accent/10 text-accent"
                              : "status-neutral",
                          )}
                        >
                          {summary.sourceCount > 0 ? `${summary.sourceCount} source${summary.sourceCount > 1 ? "s" : ""}` : "Sans source"}
                        </span>
                        <span className="rounded-md border border-border/70 bg-card/70 px-1.5 py-0.5 text-[11px] text-fg/[0.78]">
                          {summary.meta}
                        </span>
                      </div>
                    </button>
                    <div className="mt-2 flex h-7 items-center justify-between">
                      <div className="flex items-center gap-1">
                      <button aria-label="Favori" title="Favori" onClick={() => toggleFavorite(chat.id)} className="rounded-md p-1 text-fg/[0.58] transition duration-200 hover:-translate-y-0.5 hover:bg-accent/10 hover:text-fg active:translate-y-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/45">
                        <Heart size={14} className={chat.favorite ? "fill-current" : ""} />
                      </button>
                      <button
                        aria-label="Supprimer"
                        title="Supprimer"
                        onClick={() => {
                          if (window.confirm("Supprimer cette conversation ?")) {
                            void onRemoveConversation(chat.id);
                          }
                        }}
                        className="rounded-md p-1 text-fg/[0.58] transition duration-200 hover:-translate-y-0.5 hover:bg-rose-500/10 hover:text-rose-500 active:translate-y-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-500/45"
                      >
                        <Trash2 size={14} />
                      </button>
                      </div>
                      <button
                        aria-label="Menu actions chat"
                        title="Menu actions"
                        onClick={() => setMenuChatId((v) => (v === chat.id ? null : chat.id))}
                        className="rounded-md p-1 text-fg/[0.58] transition duration-200 hover:-translate-y-0.5 hover:bg-accent/10 hover:text-fg active:translate-y-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/45"
                      >
                        <Ellipsis size={14} />
                      </button>
                    </div>
                    <AnimatePresence>
                      {menuChatId === chat.id ? (
                        <motion.div
                          initial={{ opacity: 0, y: -4 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, y: -4 }}
                          className="mt-2 grid grid-cols-2 gap-1 text-xs"
                        >
                          <button className="rounded-lg border border-border/80 bg-card/60 px-2 py-1" onClick={() => onRename(chat.id)}>
                            Rename
                          </button>
                          <button className="rounded-lg border border-border/80 bg-card/60 px-2 py-1" onClick={() => toggleFavorite(chat.id)}>
                            {chat.favorite ? "Unfavorite" : "Favorite"}
                          </button>
                          <button className="rounded-lg border border-border/80 bg-card/60 px-2 py-1" onClick={() => downloadChat(chat.id, "json")}>
                            Export JSON
                          </button>
                          <button className="rounded-lg border border-border/80 bg-card/60 px-2 py-1" onClick={() => downloadChat(chat.id, "txt")}>
                            Export TXT
                          </button>
                        </motion.div>
                      ) : null}
                    </AnimatePresence>
                  </motion.div>
                  );
                })}
              </div>
            </section>
          ),
        )}
      </div>
      <div className="relative mt-3">
        <div className="flex items-center justify-between rounded-lg border border-border/80 bg-card/[0.55] px-3 py-2 text-xs">
          <p className="truncate text-fg/75">{user?.email || "Utilisateur connecté"}</p>
          <button
            aria-label="Menu utilisateur"
            title="Menu utilisateur"
            onClick={() => setUserMenuOpen((v) => !v)}
            className="rounded-md border border-border/70 bg-card/70 px-2 py-1 text-fg/75 transition hover:bg-card"
          >
            <Settings size={12} />
          </button>
        </div>
        <AnimatePresence>
          {userMenuOpen ? (
            <motion.div
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 6 }}
              className="absolute bottom-12 right-0 z-20 w-44 rounded-lg border border-border/75 bg-card p-1.5 shadow-xl"
            >
              <Link href="/settings" className="flex items-center gap-2 rounded-md px-2 py-1.5 text-xs text-fg/80 hover:bg-fg/[0.04]">
                <Settings size={12} />
                Settings
              </Link>
              <button
                className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-xs text-fg/80 hover:bg-fg/[0.04]"
                onClick={toggleTheme}
              >
                <SunMoon size={12} />
                Theme
              </button>
              <button
                className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-xs text-fg/80 hover:bg-fg/[0.04]"
                onClick={() => {
                  const content = exportChat(activeChatId || "", "json");
                  if (!content) return;
                  const blob = new Blob([content], { type: "application/json" });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement("a");
                  a.href = url;
                  a.download = "clinical-rag-conversations.json";
                  a.click();
                  URL.revokeObjectURL(url);
                }}
              >
                <Download size={12} />
                Export
              </button>
              <button
                className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-xs text-rose-400 hover:bg-rose-500/10"
                onClick={() => void logout()}
              >
                <LogOut size={12} />
                Logout
              </button>
            </motion.div>
          ) : null}
        </AnimatePresence>
      </div>
    </aside>
  );
}
