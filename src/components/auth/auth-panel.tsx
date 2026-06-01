"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { Activity, FileText, ShieldCheck } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useAuthStore } from "@/store/auth-store";

function sanitizeNextPath(nextPath: string | null): string {
  if (!nextPath) return "/chat";
  if (!nextPath.startsWith("/")) return "/chat";
  if (nextPath.startsWith("//")) return "/chat";
  if (nextPath.startsWith("/auth")) return "/chat";
  return nextPath;
}

export function AuthPanel() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const login = useAuthStore((s) => s.login);
  const register = useAuthStore((s) => s.register);
  const loading = useAuthStore((s) => s.loading);
  const error = useAuthStore((s) => s.error);
  const authStatus = useAuthStore((s) => s.authStatus);

  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [localError, setLocalError] = useState("");

  const nextPath = useMemo(
    () => sanitizeNextPath(searchParams.get("next")),
    [searchParams],
  );

  useEffect(() => {
    if (authStatus === "authenticated") {
      router.replace(nextPath);
    }
  }, [authStatus, nextPath, router]);

  async function onSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLocalError("");
    const normalizedEmail = email.trim();
    if (!normalizedEmail || !password) {
      setLocalError("Email et mot de passe sont requis.");
      return;
    }
    try {
      if (mode === "login") {
        await login(normalizedEmail, password);
      } else {
        await register(normalizedEmail, password);
      }
    } catch {
      // Error is reflected by auth-store.
    }
  }

  const displayError = localError || error || "";

  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-bg px-6 py-10">
      <div className="pointer-events-none absolute inset-0 opacity-[0.22]">
        <div className="absolute left-[-8%] top-[14%] h-px w-[42%] rotate-12 bg-accent/35" />
        <div className="absolute left-[10%] top-[42%] h-px w-[35%] -rotate-6 bg-accent/25" />
        <div className="absolute right-[-6%] top-[28%] h-px w-[40%] -rotate-12 bg-accent/28" />
        <div className="absolute right-[8%] bottom-[24%] h-px w-[38%] rotate-6 bg-accent/24" />
        <div className="absolute left-[23%] top-[22%] h-2 w-2 rounded-full bg-accent/40" />
        <div className="absolute left-[34%] top-[45%] h-2 w-2 rounded-full bg-accent/38" />
        <div className="absolute right-[24%] top-[36%] h-2 w-2 rounded-full bg-accent/40" />
        <div className="absolute right-[30%] bottom-[30%] h-2 w-2 rounded-full bg-accent/35" />
      </div>

      <div className="panel-surface grid w-full max-w-5xl grid-cols-1 overflow-hidden backdrop-blur-md lg:grid-cols-[1.1fr_0.9fr]">
        <section className="relative border-b border-border/70 p-7 lg:border-b-0 lg:border-r">
          <div className="inline-flex items-center gap-2 rounded-md border border-accent/35 bg-accent/10 px-2.5 py-1 text-xs font-medium text-accent">
            <Activity size={13} />
            CHU Oujda
          </div>
          <h1 className="mt-4 text-3xl font-semibold text-fg">Clinical RAG Platform</h1>
          <p className="mt-3 max-w-md text-sm leading-6 text-fg/74">
            Assistant sécurisé pour l’analyse documentaire médicale.
          </p>
          <p className="mt-2 text-xs font-medium uppercase tracking-[0.13em] text-accent/85">
            Accès réservé aux professionnels autorisés
          </p>

          <div className="mt-8 grid grid-cols-2 gap-3 max-w-md">
            <div className="rounded-lg border border-accent/25 bg-accent/10 p-3">
              <FileText size={16} className="text-accent" />
              <p className="mt-2 text-xs text-fg/78">Traçabilité documentaire</p>
            </div>
            <div className="rounded-lg border border-accent/25 bg-accent/10 p-3">
              <ShieldCheck size={16} className="text-accent" />
              <p className="mt-2 text-xs text-fg/78">Accès sécurisé</p>
            </div>
          </div>
        </section>

        <section className="p-7">
          <h2 className="text-base font-semibold text-fg">
            {mode === "login" ? "Connexion " : "Création de compte"}
          </h2>
          <p className="mt-1 text-sm text-fg/72">
            {mode === "login" ? "Connectez-vous pour accéder à la plateforme clinique." : "Créez votre accès à la plateforme clinique."}
          </p>

          <form className="mt-6 space-y-4" onSubmit={onSubmit}>
            <div>
              <label className="mb-1 block text-xs font-medium uppercase tracking-wide text-fg/72">Email</label>
              <input
                type="email"
                autoComplete="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                disabled={loading}
                className="input-surface h-10 w-full px-3 text-sm text-fg outline-none transition focus:border-accent/55 focus-visible:ring-2 focus-visible:ring-accent/30"
                placeholder="simo@test.ma"
              />
            </div>

            <div>
              <label className="mb-1 block text-xs font-medium uppercase tracking-wide text-fg/72">Mot de passe</label>
              <input
                type="password"
                autoComplete={mode === "login" ? "current-password" : "new-password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                disabled={loading}
                className="input-surface h-10 w-full px-3 text-sm text-fg outline-none transition focus:border-accent/55 focus-visible:ring-2 focus-visible:ring-accent/30"
                placeholder="••••••••"
              />
            </div>

            {displayError ? (
              <p className="status-danger rounded-md px-3 py-2 text-xs">{displayError}</p>
            ) : null}

            <button
              type="submit"
              disabled={loading}
              className="h-10 w-full rounded-lg bg-accent px-4 text-sm font-semibold text-slate-950 transition hover:bg-accent/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/55 disabled:opacity-60"
            >
              {loading ? "Veuillez patienter..." : mode === "login" ? "Se connecter" : "S’inscrire"}
            </button>
          </form>

          <button
            type="button"
            disabled={loading}
            onClick={() => {
              setMode((current) => (current === "login" ? "register" : "login"));
              setLocalError("");
            }}
            className="mt-4 w-full text-center text-xs text-fg/72 underline underline-offset-4 disabled:opacity-60"
          >
            {mode === "login" ? "Créer un compte" : "J’ai déjà un compte"}
          </button>
        </section>
      </div>
    </div>
  );
}
