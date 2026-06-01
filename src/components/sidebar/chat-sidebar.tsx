"use client";

import { AnimatePresence, motion } from "framer-motion";
import { Activity, Download, FileText, LayoutDashboard, LogOut, MessageSquare, MessageSquarePlus, Search, Settings, SunMoon, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useTheme } from "next-themes";
import { createPortal } from "react-dom";
import { healthcheck } from "@/services/rag-api";
import { useAuthStore } from "@/store/auth-store";
import { useChatStore } from "@/store/chat-store";
import { cn } from "@/lib/utils";
import type { ChatItem, ChatSource } from "@/types/chat";
import { ConversationCard } from "@/components/sidebar/conversation-card";
import { SidebarFooter } from "@/components/sidebar/sidebar-footer";

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

function summarizeConversation(chat: ChatItem): { sourceCount: number } {
  const docs = new Set<string>();
  for (const message of chat.messages) {
    for (const source of message.sources || []) {
      const key = sourceDocKey(source);
      if (key) docs.add(key);
    }
  }
  return {
    sourceCount: docs.size,
  };
}

type ChatSidebarProps = {
  mobileOpen?: boolean;
  onMobileClose?: () => void;
};

export function ChatSidebar({ mobileOpen = false, onMobileClose }: ChatSidebarProps) {
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
  const [userMenuClosing, setUserMenuClosing] = useState(false);
  const [userMenuPosition, setUserMenuPosition] = useState<{ left: number; right: number; bottom: number } | null>(null);
  const [backendStatus, setBackendStatus] = useState<"online" | "offline" | "checking">("checking");
  const searchRef = useRef<HTMLInputElement | null>(null);
  const footerRef = useRef<HTMLButtonElement | null>(null);
  const userMenuRef = useRef<HTMLDivElement | null>(null);

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

  function computeUserMenuPosition() {
    const footer = footerRef.current;
    if (!footer) return;
    const rect = footer.getBoundingClientRect();
    setUserMenuPosition({
      left: rect.left + 12,
      right: Math.max(window.innerWidth - rect.right + 12, 12),
      bottom: Math.max(window.innerHeight - rect.top + 8, 8),
    });
  }

  function closeUserMenu() {
    setUserMenuClosing(true);
    window.setTimeout(() => {
      setUserMenuOpen(false);
      setUserMenuClosing(false);
    }, 100);
  }

  function openUserMenu() {
    computeUserMenuPosition();
    setUserMenuClosing(false);
    setUserMenuOpen(true);
  }

  function toggleUserMenu() {
    if (userMenuOpen) {
      closeUserMenu();
      return;
    }
    openUserMenu();
  }

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        searchRef.current?.focus();
      }
      if (event.key === "Escape") {
        setMenuChatId(null);
        if (userMenuOpen) closeUserMenu();
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [userMenuOpen]);

  useEffect(() => {
    if (!userMenuOpen) return;

    function handleOutsideClick(event: MouseEvent) {
      const target = event.target as Node | null;
      if (!target) return;
      if (userMenuRef.current?.contains(target)) return;
      if (footerRef.current?.contains(target)) return;
      closeUserMenu();
    }

    function handleReposition() {
      computeUserMenuPosition();
    }

    document.addEventListener("mousedown", handleOutsideClick);
    window.addEventListener("resize", handleReposition);
    window.addEventListener("scroll", handleReposition, true);
    return () => {
      document.removeEventListener("mousedown", handleOutsideClick);
      window.removeEventListener("resize", handleReposition);
      window.removeEventListener("scroll", handleReposition, true);
    };
  }, [userMenuOpen]);

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

  function toggleTheme() {
    const next = themePref === "dark" ? "light" : "dark";
    setThemePref(next);
    setTheme(next);
  }

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

  async function onStartConversation() {
    const createdId = await startNewConversation(token);
    if (createdId) {
      router.push(`/chat/${createdId}`);
      onMobileClose?.();
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
    onMobileClose?.();
  }

  const navItems = [
    { href: "/chat", label: "Chat", icon: MessageSquare },
    { href: "/documents", label: "Documents", icon: FileText },
    { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  ] as const;

  function closeMenuAfterSelection(action: () => void) {
    action();
    window.setTimeout(() => closeUserMenu(), 80);
  }

  return (
    <>
      <AnimatePresence>
        {mobileOpen ? (
          <motion.button
            type="button"
            className="fixed inset-0 z-[55] bg-black/50 backdrop-blur-[2px] xl:hidden"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            aria-label="Fermer le menu latéral"
            onClick={() => onMobileClose?.()}
          />
        ) : null}
      </AnimatePresence>
      <aside
        className={cn(
          "chat-sidebar-shell glass fixed inset-y-0 left-0 z-[60] flex h-dvh w-[92vw] max-w-[18.5rem] shrink-0 flex-col border-y-0 border-l-0 border-border/70 p-3 shadow-[0_24px_80px_rgba(0,0,0,0.42)] transition-[transform,opacity] duration-300 ease-out sm:p-4 xl:static xl:z-auto xl:flex xl:h-dvh xl:w-80 xl:max-w-none xl:translate-x-0 xl:opacity-100 xl:shadow-none",
          mobileOpen ? "translate-x-0 opacity-100 pointer-events-auto" : "-translate-x-full opacity-0 pointer-events-none xl:pointer-events-auto",
        )}
      >
      <div className="sidebar-brand mb-4 rounded-xl border border-border/70 bg-card/[0.46] p-3">
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
          <button
            type="button"
            className="ml-auto inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-border/80 bg-card/70 text-fg/75 transition hover:border-accent/30 hover:bg-accent/10 hover:text-fg xl:hidden"
            aria-label="Fermer le menu latéral"
            onClick={() => onMobileClose?.()}
          >
            <X size={16} />
          </button>
        </div>
      </div>
      <button
        aria-label="Nouveau chat"
        onClick={() => void onStartConversation()}
        className="sidebar-new-chat mb-3 flex h-10 items-center justify-center gap-2 rounded-lg bg-accent px-3 py-2 text-sm font-semibold text-slate-950 shadow-sm transition duration-200 hover:-translate-y-0.5 hover:bg-accent/90 active:translate-y-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/55"
      >
        <MessageSquarePlus size={16} /> Nouvelle conversation
      </button>
      <div className="sidebar-search mb-3 flex h-10 items-center gap-2 rounded-lg border border-border/80 bg-card/[0.72] px-3 py-2 shadow-sm">
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
      <section className="sidebar-nav mb-3 rounded-lg border border-border/70 bg-card/[0.45] p-2">
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
                onClick={() => onMobileClose?.()}
              >
                <item.icon size={15} />
                <span className="font-medium">{item.label}</span>
              </Link>
            );
          })}
        </div>
      </section>
      <div className="sidebar-conversation-list flex-1 space-y-3 overflow-auto pr-0.5">
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
                    className="rounded-lg"
                  >
                    <ConversationCard
                      title={chat.title}
                      updatedAt={chat.updatedAt}
                      sourceCount={summary.sourceCount}
                      isFavorited={chat.favorite}
                      active={chat.id === activeChatId}
                      onClick={() => {
                        router.push(`/chat/${chat.id}`);
                        onMobileClose?.();
                      }}
                      onToggleFavorite={() => toggleFavorite(chat.id)}
                      onDelete={() => {
                        if (window.confirm("Supprimer cette conversation ?")) {
                          void onRemoveConversation(chat.id);
                        }
                      }}
                      onOpenMenu={() => setMenuChatId((current) => (current === chat.id ? null : chat.id))}
                    />
                    <AnimatePresence>
                      {menuChatId === chat.id ? (
                        <motion.div
                          initial={{ opacity: 0, y: -4 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, y: -4 }}
                          className="mt-1.5 grid grid-cols-2 gap-1 text-xs"
                        >
                          <button className="rounded-lg border border-border/80 bg-card/60 px-2 py-1" onClick={() => onRename(chat.id)}>
                            Rename
                          </button>
                          <button className="rounded-lg border border-border/80 bg-card/60 px-2 py-1" onClick={() => toggleFavorite(chat.id)}>
                            {chat.favorite ? "Unfavorite" : "Favorite"}
                          </button>
                          <button
                            className="rounded-lg border border-border/80 bg-card/60 px-2 py-1 text-rose-300 hover:bg-rose-500/10 hover:text-rose-200"
                            onClick={() => {
                              if (window.confirm("Supprimer cette conversation ?")) {
                                void onRemoveConversation(chat.id);
                              }
                            }}
                          >
                            Supprimer
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
      <SidebarFooter ref={footerRef} user={user} onClick={toggleUserMenu} />
      {userMenuOpen && userMenuPosition && typeof document !== "undefined"
        ? createPortal(
            <div
              ref={userMenuRef}
              className={cn(
                "menu-panel z-[70] min-w-[200px] rounded-xl border border-white/10 bg-[#1E2433] p-[6px] shadow-[0_8px_32px_rgba(0,0,0,0.4),0_2px_8px_rgba(0,0,0,0.2)]",
                userMenuClosing && "menu-panel-out",
              )}
              style={{
                position: "fixed",
                left: userMenuPosition.left,
                right: userMenuPosition.right,
                bottom: userMenuPosition.bottom,
              }}
            >
              <button
                className="flex h-9 w-full items-center gap-2.5 rounded-lg px-2.5 text-left text-[13px] text-white/75 transition hover:bg-white/[0.07] hover:text-white"
                onClick={() => closeMenuAfterSelection(() => router.push("/settings"))}
              >
                <Settings size={16} className="text-white/45" />
                Settings
              </button>
              <button
                className="flex h-9 w-full items-center gap-2.5 rounded-lg px-2.5 text-left text-[13px] text-white/75 transition hover:bg-white/[0.07] hover:text-white"
                onClick={() => closeMenuAfterSelection(toggleTheme)}
              >
                <SunMoon size={16} className="text-white/45" />
                Theme
              </button>
              <button
                className="flex h-9 w-full items-center gap-2.5 rounded-lg px-2.5 text-left text-[13px] text-white/75 transition hover:bg-white/[0.07] hover:text-white"
                onClick={() =>
                  closeMenuAfterSelection(() => {
                    const content = exportChat(activeChatId || "", "json");
                    if (!content) return;
                    const blob = new Blob([content], { type: "application/json" });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement("a");
                    a.href = url;
                    a.download = "clinical-rag-conversations.json";
                    a.click();
                    URL.revokeObjectURL(url);
                  })
                }
              >
                <Download size={16} className="text-white/45" />
                Export
              </button>
              <div className="my-1 h-px bg-white/[0.07]" />
              <button
                className="flex h-9 w-full items-center gap-2.5 rounded-lg px-2.5 text-left text-[13px] text-[#E07070] transition hover:bg-[rgba(224,112,112,0.10)]"
                onClick={() => closeMenuAfterSelection(() => void logout())}
              >
                <LogOut size={16} className="text-[#E07070]" />
                Logout
              </button>
            </div>,
            document.body,
          )
        : null}
    </aside>
    </>
  );
}
