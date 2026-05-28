"use client";

import { AnimatePresence, motion } from "framer-motion";
import { Activity, Download, Ellipsis, Heart, LogOut, MessageSquarePlus, Search, Settings, SunMoon, Trash2 } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useTheme } from "next-themes";
import { useAuthStore } from "@/store/auth-store";
import { useChatStore } from "@/store/chat-store";
import { cn, formatDateLabel } from "@/lib/utils";
import { groupChatsByDate } from "@/lib/chat-groups";

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

  const grouped = groupChatsByDate(filtered);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        searchRef.current?.focus();
      }
      if (event.key === "Escape") setMenuChatId(null);
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
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

  return (
    <aside className="glass flex h-screen w-80 shrink-0 flex-col border-y-0 border-l-0 p-4">
      <div className="mb-4 flex items-start gap-3">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-accent text-white shadow-sm">
          <Activity size={18} />
        </div>
        <div className="min-w-0">
          <h1 className="text-sm font-semibold leading-5">CHU Oujda Clinical Assistant</h1>
          <p className="mt-1 truncate text-xs text-fg/60">{user?.email || "Utilisateur connecté"}</p>
        </div>
      </div>
      <button
        aria-label="Nouveau chat"
        onClick={() => void onStartConversation()}
        className="mb-3 flex h-10 items-center justify-center gap-2 rounded-xl bg-accent px-3 py-2 text-sm font-medium text-white shadow-sm transition hover:bg-accent/90"
      >
        <MessageSquarePlus size={16} /> Nouvelle conversation
      </button>
      <div className="mb-3 flex h-10 items-center gap-2 rounded-xl border border-border/80 bg-card/[0.72] px-3 py-2 shadow-sm">
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
      <div className="mb-2">
        <p className="mb-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-fg/[0.45]">Favoris</p>
        <div className="flex flex-wrap gap-1">
          {filtered
            .filter((c) => c.favorite)
            .slice(0, 4)
            .map((fav) => (
              <button
                key={fav.id}
                className="rounded-full border border-border/80 bg-card/[0.55] px-2 py-1 text-xs text-fg/[0.72] transition hover:border-accent/35 hover:bg-accent/10"
                onClick={() => router.push(`/chat/${fav.id}`)}
              >
                {fav.title.slice(0, 18)}
              </button>
            ))}
        </div>
      </div>
      <div className="flex-1 space-y-4 overflow-auto">
        {Object.entries(grouped).map(([label, items]) =>
          items.length === 0 ? null : (
            <section key={label} aria-label={label}>
              <p className="mb-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-fg/[0.45]">{label}</p>
              <div className="space-y-2">
                {items.map((chat) => (
                  <motion.div
                    key={chat.id}
                    initial={{ opacity: 0, y: 4 }}
                    animate={{ opacity: 1, y: 0 }}
                    className={cn(
                      "rounded-xl border p-3 transition",
                      chat.id === activeChatId
                        ? "border-accent/[0.55] bg-accent/10 shadow-sm"
                        : "border-border/70 bg-card/[0.42] hover:border-accent/25 hover:bg-card/[0.72]",
                    )}
                  >
                    <button
                      className="w-full text-left"
                      onClick={() => router.push(`/chat/${chat.id}`)}
                      title={chat.title}
                      aria-current={chat.id === activeChatId ? "page" : undefined}
                    >
                      <p className="line-clamp-1 text-sm font-medium text-fg/90">{chat.title}</p>
                      <p className="mt-0.5 text-xs text-fg/[0.54]">{formatDateLabel(chat.updatedAt)}</p>
                    </button>
                    <div className="mt-2 flex items-center gap-1">
                      <button aria-label="Favori" title="Favori" onClick={() => toggleFavorite(chat.id)} className="rounded-md p-1 text-fg/[0.58] transition hover:bg-accent/10 hover:text-fg">
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
                        className="rounded-md p-1 text-fg/[0.58] transition hover:bg-rose-500/10 hover:text-rose-500"
                      >
                        <Trash2 size={14} />
                      </button>
                      <button
                        aria-label="Menu actions chat"
                        title="Menu actions"
                        onClick={() => setMenuChatId((v) => (v === chat.id ? null : chat.id))}
                        className="ml-auto rounded-md p-1 text-fg/[0.58] transition hover:bg-accent/10 hover:text-fg"
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
                ))}
              </div>
            </section>
          ),
        )}
      </div>
      <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
        <Link href="/documents" className="rounded-lg border border-border/80 bg-card/[0.55] px-2 py-1.5 text-center transition hover:bg-accent/10">
          Documents
        </Link>
        <Link href="/settings" className="rounded-lg border border-border/80 bg-card/[0.55] px-2 py-1.5 text-center transition hover:bg-accent/10">
          <Settings className="mr-1 inline" size={12} />
          Settings
        </Link>
        <Link href="/dashboard" className="rounded-lg border border-border/80 bg-card/[0.55] px-2 py-1.5 text-center transition hover:bg-accent/10">
          Dashboard
        </Link>
        <button
          className="rounded-lg border border-border/80 bg-card/[0.55] px-2 py-1.5 transition hover:bg-accent/10"
          onClick={toggleTheme}
          aria-label="Basculer thème"
          title="Basculer thème"
        >
          <SunMoon className="mr-1 inline" size={12} />
          Theme
        </button>
        <button className="rounded-lg border border-border/80 bg-card/[0.55] px-2 py-1.5 transition hover:bg-accent/10" aria-label="Exporter" title="Exporter conversations">
          <Download className="mr-1 inline" size={12} />
          Export
        </button>
        <button
          className="rounded-lg border border-border/80 bg-card/[0.55] px-2 py-1.5 transition hover:bg-rose-500/10 hover:text-rose-500"
          onClick={() => void logout()}
          aria-label="Se déconnecter"
          title="Se déconnecter"
        >
          <LogOut className="mr-1 inline" size={12} />
          Logout
        </button>
      </div>
    </aside>
  );
}
