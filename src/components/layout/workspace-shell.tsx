"use client";

import Link from "next/link";
import { useEffect } from "react";
import { AuthPanel } from "@/components/auth/auth-panel";
import { ChatSidebar } from "@/components/sidebar/chat-sidebar";
import { useAuthStore } from "@/store/auth-store";
import { useChatStore } from "@/store/chat-store";

type WorkspaceAction = {
  href: string;
  label: string;
};

type WorkspaceShellProps = {
  title: string;
  subtitle: string;
  breadcrumbs: string[];
  actions?: WorkspaceAction[];
  children: React.ReactNode;
};

export function WorkspaceShell({ title, subtitle, breadcrumbs, actions = [], children }: WorkspaceShellProps) {
  const initialize = useChatStore((s) => s.initialize);
  const initializeAuth = useAuthStore((s) => s.initializeAuth);
  const authStatus = useAuthStore((s) => s.authStatus);

  useEffect(() => {
    void initialize();
    void initializeAuth();
  }, [initialize, initializeAuth]);

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
        <WorkspaceTopbar title={title} subtitle={subtitle} breadcrumbs={breadcrumbs} actions={actions} />
        <div className="flex-1 overflow-auto">{children}</div>
      </main>
    </div>
  );
}

export function WorkspaceTopbar({
  title,
  subtitle,
  breadcrumbs,
  actions = [],
}: {
  title: string;
  subtitle: string;
  breadcrumbs: string[];
  actions?: WorkspaceAction[];
}) {
  return (
    <header className="topbar h-[72px] border-b border-slate-400/20 bg-slate-900/72 px-5 backdrop-blur-2xl">
      <div className="flex h-full items-center justify-between gap-4">
        <div className="min-w-0">
          <p className="text-xs font-medium text-fg/55">{breadcrumbs.join(" / ")}</p>
          <h1 className="line-clamp-1 text-base font-semibold text-fg">{title}</h1>
          <p className="line-clamp-1 text-xs text-fg/62">{subtitle}</p>
        </div>
        <div className="flex shrink-0 flex-wrap items-center justify-end gap-2">
          {actions.map((action) => (
            <Link
              key={`${action.href}-${action.label}`}
              href={action.href}
              className="rounded-lg border border-border/75 bg-card/[0.62] px-3 py-1.5 text-xs font-medium text-fg/78 transition hover:border-accent/35 hover:bg-card"
            >
              {action.label}
            </Link>
          ))}
        </div>
      </div>
    </header>
  );
}
