"use client";

import { AnimatePresence, motion } from "framer-motion";
import { Download, Ellipsis, Heart, LogOut, MessageSquarePlus, Search, Settings, SunMoon, Trash2 } from "lucide-react";
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
    <aside className="glass flex h-screen w-80 flex-col p-4">
      <div className="mb-4">
        <h1 className="text-lg font-semibold">CHU Oujda Clinical Assistant</h1>
        <p className="mt-1 text-xs text-fg/70">{user?.email || "Utilisateur connecté"}</p>
      </div>
      <button
        aria-label="Nouveau chat"
        onClick={() => void onStartConversation()}
        className="mb-3 flex items-center gap-2 rounded-xl bg-accent/20 px-3 py-2 text-sm hover:bg-accent/30"
      >
        <MessageSquarePlus size={16} /> Nouvelle conversation
      </button>
      <div className="mb-3 flex items-center gap-2 rounded-xl border border-border px-3 py-2">
        <Search size={14} />
        <input
          ref={searchRef}
          aria-label="Rechercher une conversation"
          placeholder="Rechercher une conversation"
          className="w-full bg-transparent text-sm outline-none"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>
      <div className="mb-2">
        <p className="mb-1 text-xs uppercase tracking-wide text-fg/60">Favoris</p>
        <div className="flex flex-wrap gap-1">
          {filtered
            .filter((c) => c.favorite)
            .slice(0, 4)
            .map((fav) => (
              <button
                key={fav.id}
                className="rounded-full border border-border px-2 py-1 text-xs"
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
              <p className="mb-1 text-xs uppercase tracking-wide text-fg/60">{label}</p>
              <div className="space-y-2">
                {items.map((chat) => (
                  <motion.div
                    key={chat.id}
                    initial={{ opacity: 0, y: 4 }}
                    animate={{ opacity: 1, y: 0 }}
                    className={cn(
                      "rounded-xl border p-3",
                      chat.id === activeChatId ? "border-accent bg-accent/10" : "border-border",
                    )}
                  >
                    <button
                      className="w-full text-left"
                      onClick={() => router.push(`/chat/${chat.id}`)}
                      title={chat.title}
                      aria-current={chat.id === activeChatId ? "page" : undefined}
                    >
                      <p className="line-clamp-1 text-sm font-medium">{chat.title}</p>
                      <p className="text-xs text-fg/70">{formatDateLabel(chat.updatedAt)}</p>
                    </button>
                    <div className="mt-2 flex items-center gap-1">
                      <button aria-label="Favori" title="Favori" onClick={() => toggleFavorite(chat.id)} className="rounded p-1 hover:bg-card">
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
                        className="rounded p-1 hover:bg-card"
                      >
                        <Trash2 size={14} />
                      </button>
                      <button
                        aria-label="Menu actions chat"
                        title="Menu actions"
                        onClick={() => setMenuChatId((v) => (v === chat.id ? null : chat.id))}
                        className="ml-auto rounded p-1 hover:bg-card"
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
                          <button className="rounded border border-border px-2 py-1" onClick={() => onRename(chat.id)}>
                            Rename
                          </button>
                          <button className="rounded border border-border px-2 py-1" onClick={() => toggleFavorite(chat.id)}>
                            {chat.favorite ? "Unfavorite" : "Favorite"}
                          </button>
                          <button className="rounded border border-border px-2 py-1" onClick={() => downloadChat(chat.id, "json")}>
                            Export JSON
                          </button>
                          <button className="rounded border border-border px-2 py-1" onClick={() => downloadChat(chat.id, "txt")}>
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
        <Link href="/documents" className="rounded-lg border border-border px-2 py-1 text-center">
          Documents
        </Link>
        <Link href="/settings" className="rounded-lg border border-border px-2 py-1 text-center">
          <Settings className="mr-1 inline" size={12} />
          Settings
        </Link>
        <Link href="/dashboard" className="rounded-lg border border-border px-2 py-1 text-center">
          Dashboard
        </Link>
        <button
          className="rounded-lg border border-border px-2 py-1"
          onClick={toggleTheme}
          aria-label="Basculer thème"
          title="Basculer thème"
        >
          <SunMoon className="mr-1 inline" size={12} />
          Theme
        </button>
        <button className="rounded-lg border border-border px-2 py-1" aria-label="Exporter" title="Exporter conversations">
          <Download className="mr-1 inline" size={12} />
          Export
        </button>
        <button
          className="rounded-lg border border-border px-2 py-1"
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
