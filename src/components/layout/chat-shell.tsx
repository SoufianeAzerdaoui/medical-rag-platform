"use client";

import { PanelRightClose, PanelRightOpen } from "lucide-react";
import { useEffect, useState } from "react";
import { AuthPanel } from "@/components/auth/auth-panel";
import { ChatMessages } from "@/components/chat/chat-messages";
import { WorkspaceTopbar } from "@/components/layout/workspace-shell";
import { MessageComposer } from "@/components/chat/message-composer";
import { ChatSidebar } from "@/components/sidebar/chat-sidebar";
import { SourcesPanel } from "@/components/sources/sources-panel";
import { ApiError, healthcheck } from "@/services/rag-api";
import { useAuthStore } from "@/store/auth-store";
import { useChatStore } from "@/store/chat-store";

interface ChatShellProps {
  routeConversationId?: string | null;
}

export function ChatShell({ routeConversationId = null }: ChatShellProps) {
  const initialize = useChatStore((s) => s.initialize);
  const selectConversation = useChatStore((s) => s.selectConversation);
  const setActiveChat = useChatStore((s) => s.setActiveChat);
  const initializeAuth = useAuthStore((s) => s.initializeAuth);
  const logout = useAuthStore((s) => s.logout);
  const accessToken = useAuthStore((s) => s.accessToken);
  const authStatus = useAuthStore((s) => s.authStatus);
  const chats = useChatStore((s) => s.chats);
  const activeChatId = useChatStore((s) => s.activeChatId);
  const [sourcesOpenMobile, setSourcesOpenMobile] = useState(false);
  const [conversationError, setConversationError] = useState<string | null>(null);
  const [backendOnline, setBackendOnline] = useState(true);
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

  if (authStatus === "loading") {
    return (
      <div className="flex h-screen items-center justify-center px-6">
        <div className="glass rounded-xl px-5 py-4 text-sm text-fg/70">Chargement de la session...</div>
      </div>
    );
  }

  if (authStatus === "unauthenticated") {
    return <AuthPanel />;
  }

  return (
    <div className="flex h-screen overflow-hidden bg-transparent">
      <ChatSidebar />
      <main className="flex min-w-0 flex-1 flex-col">
        <div className="hidden xl:block">
          <WorkspaceTopbar
            title={topbarTitle}
            subtitle="Espace conversationnel clinique"
            breadcrumbs={topbarBreadcrumbs}
            actions={[
              { href: "/chat", label: "Retour au chat" },
              { href: "/documents/upload", label: "Importer document" },
              { href: "/chat", label: "Nouvelle conversation" },
            ]}
          />
        </div>
        <div className="border-b border-border/70 bg-card/70 px-4 py-2 backdrop-blur-xl xl:hidden">
          <button
            className="inline-flex items-center gap-2 rounded-lg border border-border/80 bg-card/70 px-3 py-1.5 text-xs font-medium shadow-sm"
            onClick={() => setSourcesOpenMobile((v) => !v)}
            aria-label="Ouvrir panneau sources"
          >
            {sourcesOpenMobile ? <PanelRightClose size={14} /> : <PanelRightOpen size={14} />}
            Sources
          </button>
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
