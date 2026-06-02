"use client";

import { usePathname, useRouter } from "next/navigation";
import { Menu, PanelRightClose, PanelRightOpen } from "lucide-react";
import { useEffect, useState } from "react";
import { ChatMessages } from "@/components/chat/chat-messages";
import { ActiveModelPill } from "@/components/chat/active-model-pill";
import { WorkspaceTopbar } from "@/components/layout/workspace-shell";
import { MessageComposer } from "@/components/chat/message-composer";
import { ChatSidebar } from "@/components/sidebar/chat-sidebar";
import { SourcesPanel } from "@/components/sources/sources-panel";
import { ApiError, getActiveModelApi, healthcheck, type ActiveModelInfo } from "@/services/rag-api";
import { useAuthStore } from "@/store/auth-store";
import { useChatStore } from "@/store/chat-store";

interface ChatShellProps {
  routeConversationId?: string | null;
}

export function ChatShell({ routeConversationId = null }: ChatShellProps) {
  const router = useRouter();
  const pathname = usePathname();
  const initialize = useChatStore((s) => s.initialize);
  const selectConversation = useChatStore((s) => s.selectConversation);
  const setActiveChat = useChatStore((s) => s.setActiveChat);
  const initializeAuth = useAuthStore((s) => s.initializeAuth);
  const logout = useAuthStore((s) => s.logout);
  const accessToken = useAuthStore((s) => s.accessToken);
  const authStatus = useAuthStore((s) => s.authStatus);
  const chats = useChatStore((s) => s.chats);
  const activeChatId = useChatStore((s) => s.activeChatId);
  const [sidebarOpenMobile, setSidebarOpenMobile] = useState(false);
  const [sourcesOpenMobile, setSourcesOpenMobile] = useState(false);
  const [conversationError, setConversationError] = useState<string | null>(null);
  const [backendOnline, setBackendOnline] = useState(true);
  const [activeModel, setActiveModel] = useState<ActiveModelInfo | null>(null);
  const activeChat = chats.find((chat) => chat.id === activeChatId) || null;
  const topbarTitle = activeChat?.title && activeChat.title.trim() ? activeChat.title : "Workspace Chat";
  const topbarBreadcrumbs = activeChat?.title && !/^nouvelle conversation$/i.test(activeChat.title.trim())
    ? ["Clinical Assistant", "Chat", activeChat.title]
    : ["Clinical Assistant", "Chat"];

  useEffect(() => {
    void initialize();
    void initializeAuth();
  }, [initialize, initializeAuth]);

  useEffect(() => {
    if (authStatus !== "unauthenticated") return;
    const query = typeof window !== "undefined" ? window.location.search.replace(/^\?/, "") : "";
    const nextPath = `${pathname}${query ? `?${query}` : ""}`;
    router.replace(`/auth?next=${encodeURIComponent(nextPath)}`);
  }, [authStatus, pathname, router]);

  useEffect(() => {
    if (authStatus !== "authenticated") {
      setConversationError(null);
      return;
    }
    if (!routeConversationId) {
      setConversationError(null);
      return;
    }
    if (!accessToken) return;

    let cancelled = false;
    setConversationError(null);

    void (async () => {
      try {
        await selectConversation(routeConversationId, accessToken);
        if (!cancelled) setConversationError(null);
      } catch (error) {
        if (cancelled) return;
        setActiveChat(routeConversationId);
        if (error instanceof ApiError && error.status === 401) {
          setConversationError("Session expirée. Veuillez vous reconnecter.");
          await logout();
          return;
        }
        if (error instanceof ApiError && error.status === 403) {
          setConversationError("Accès interdit à cette conversation.");
          return;
        }
        if (error instanceof ApiError && error.status === 404) {
          setConversationError("Conversation introuvable.");
          return;
        }
        setConversationError("Impossible de charger cette conversation.");
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [accessToken, authStatus, logout, routeConversationId, selectConversation, setActiveChat]);

  useEffect(() => {
    let active = true;
    async function checkBackend() {
      const status = await healthcheck();
      if (active) setBackendOnline(status === "online");
    }
    void checkBackend();
    const timer = window.setInterval(() => void checkBackend(), 30_000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    if (authStatus !== "authenticated" || !accessToken) {
      setActiveModel(null);
      return;
    }
    let cancelled = false;
    void getActiveModelApi(accessToken)
      .then((payload) => {
        if (!cancelled) setActiveModel(payload);
      })
      .catch(() => {
        if (!cancelled) setActiveModel(null);
      });
    return () => {
      cancelled = true;
    };
  }, [accessToken, authStatus]);

  useEffect(() => {
    setSidebarOpenMobile(false);
    setSourcesOpenMobile(false);
  }, [pathname]);

  if (authStatus === "loading") {
    return (
      <div className="flex h-dvh items-center justify-center px-6">
        <div className="glass rounded-xl px-5 py-4 text-sm text-fg/70">Chargement de la session...</div>
      </div>
    );
  }

  if (authStatus === "unauthenticated") {
    return (
      <div className="flex h-dvh items-center justify-center px-6">
        <div className="glass rounded-xl px-5 py-4 text-sm text-fg/70">Redirection vers l&apos;authentification...</div>
      </div>
    );
  }

  return (
    <div className="flex h-dvh overflow-hidden bg-transparent">
      <ChatSidebar mobileOpen={sidebarOpenMobile} onMobileClose={() => setSidebarOpenMobile(false)} />
      <main className="flex min-w-0 flex-1 flex-col">
        <div className="workspace-mobile-header border-b border-border/70 bg-card/75 px-3 py-3 backdrop-blur-2xl sm:px-4 xl:hidden">
          <div className="flex items-start gap-2.5">
            <button
              type="button"
              className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border border-border/80 bg-card/[0.72] text-fg/78 shadow-sm transition hover:border-accent/30 hover:bg-accent/10 hover:text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/50 sm:h-10 sm:w-10"
              onClick={() => setSidebarOpenMobile(true)}
              aria-label="Ouvrir le menu latéral"
            >
              <Menu size={16} />
            </button>
            <div className="min-w-0 flex-1">
              <p className="truncate text-[10px] font-medium text-fg/60 sm:text-[11px]">{topbarBreadcrumbs.join(" / ")}</p>
              <h1 className="truncate text-[13px] font-semibold text-fg sm:text-sm">{topbarTitle}</h1>
              <p className="truncate text-[11px] text-fg/72 sm:text-xs">Espace conversationnel clinique</p>
              <div className="mt-2">
                <ActiveModelPill model={activeModel} />
              </div>
            </div>
            <button
              className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border border-border/80 bg-card/[0.72] text-fg/78 shadow-sm transition hover:border-accent/30 hover:bg-accent/10 hover:text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/50 sm:h-10 sm:w-10"
              onClick={() => setSourcesOpenMobile((v) => !v)}
              aria-label="Ouvrir panneau sources"
            >
              {sourcesOpenMobile ? <PanelRightClose size={16} /> : <PanelRightOpen size={16} />}
            </button>
          </div>
        </div>
        <div className="hidden xl:block">
          <WorkspaceTopbar
            title={topbarTitle}
            subtitle="Espace conversationnel clinique"
            breadcrumbs={topbarBreadcrumbs}
            status={<ActiveModelPill model={activeModel} />}
            actions={[
              { href: "/chat", label: "Retour au chat" },
              { href: "/documents/upload", label: "Importer document" },
              { href: "/chat", label: "Nouvelle conversation" },
            ]}
          />
        </div>
        <div className="flex-1 overflow-auto" tabIndex={0} aria-label="Zone de conversation défilable">
          {!backendOnline ? (
            <div className="status-warning mx-6 mt-4 rounded-xl px-4 py-3 text-sm">
              <p className="font-medium">Le service RAG est temporairement indisponible.</p>
              <p className="mt-1 text-xs">Vérifiez l’API backend ou la base vectorielle.</p>
            </div>
          ) : null}
          {conversationError ? (
            <div className="status-danger mx-6 mt-4 rounded-xl px-4 py-3 text-sm">
              {conversationError}
            </div>
          ) : null}
          <ChatMessages />
        </div>
        <MessageComposer />
      </main>
      <SourcesPanel mobileOpen={sourcesOpenMobile} onClose={() => setSourcesOpenMobile(false)} />
    </div>
  );
}
