"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { Menu } from "lucide-react";
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
  const router = useRouter();
  const pathname = usePathname();
  const initialize = useChatStore((s) => s.initialize);
  const initializeAuth = useAuthStore((s) => s.initializeAuth);
  const authStatus = useAuthStore((s) => s.authStatus);
  const [sidebarOpenMobile, setSidebarOpenMobile] = useState(false);

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
    setSidebarOpenMobile(false);
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
        <div className="workspace-mobile-header border-b border-border/70 bg-card/75 px-4 py-3 backdrop-blur-2xl xl:hidden">
          <div className="flex items-start gap-3">
            <button
              type="button"
              className="inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-border/80 bg-card/[0.72] text-fg/78 shadow-sm transition hover:border-accent/30 hover:bg-accent/10 hover:text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/50"
              onClick={() => setSidebarOpenMobile(true)}
              aria-label="Ouvrir le menu latéral"
            >
              <Menu size={16} />
            </button>
            <div className="min-w-0 flex-1">
              <p className="truncate text-[11px] font-medium text-fg/60">{breadcrumbs.join(" / ")}</p>
              <h1 className="truncate text-sm font-semibold text-fg">{title}</h1>
              <p className="truncate text-xs text-fg/72">{subtitle}</p>
            </div>
          </div>
          {actions.length > 0 ? (
            <div className="mt-3 flex flex-wrap gap-2">
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
          ) : null}
        </div>
        <div className="hidden xl:block">
          <WorkspaceTopbar title={title} subtitle={subtitle} breadcrumbs={breadcrumbs} actions={actions} />
        </div>
        <div className="flex-1 overflow-auto" tabIndex={0} aria-label="Zone de contenu défilable">{children}</div>
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
    <header className="topbar h-[72px] border-b border-border/70 bg-card/75 px-5 backdrop-blur-2xl">
      <div className="flex h-full items-center justify-between gap-4">
        <div className="min-w-0">
          <p className="text-xs font-medium text-fg/72">{breadcrumbs.join(" / ")}</p>
          <h1 className="line-clamp-1 text-base font-semibold text-fg">{title}</h1>
          <p className="line-clamp-1 text-xs text-fg/78">{subtitle}</p>
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
