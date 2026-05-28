"use client";

import { Activity, FileText, ShieldCheck } from "lucide-react";
import { useState } from "react";
import { useAuthStore } from "@/store/auth-store";

export function AuthPanel() {
  const login = useAuthStore((s) => s.login);
  const register = useAuthStore((s) => s.register);
  const loading = useAuthStore((s) => s.loading);
  const error = useAuthStore((s) => s.error);

  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [localError, setLocalError] = useState("");

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
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-[#061426] px-6 py-10">
      <div className="pointer-events-none absolute inset-0 opacity-[0.2]">
        <div className="absolute left-[-8%] top-[14%] h-px w-[42%] rotate-12 bg-cyan-300/40" />
        <div className="absolute left-[10%] top-[42%] h-px w-[35%] -rotate-6 bg-cyan-200/35" />
        <div className="absolute right-[-6%] top-[28%] h-px w-[40%] -rotate-12 bg-teal-200/35" />
        <div className="absolute right-[8%] bottom-[24%] h-px w-[38%] rotate-6 bg-cyan-200/30" />
        <div className="absolute left-[23%] top-[22%] h-2 w-2 rounded-full bg-cyan-200/50" />
        <div className="absolute left-[34%] top-[45%] h-2 w-2 rounded-full bg-teal-200/50" />
        <div className="absolute right-[24%] top-[36%] h-2 w-2 rounded-full bg-cyan-200/50" />
        <div className="absolute right-[30%] bottom-[30%] h-2 w-2 rounded-full bg-teal-200/50" />
      </div>

      <div className="grid w-full max-w-5xl grid-cols-1 overflow-hidden rounded-xl border border-cyan-200/20 bg-slate-900/65 shadow-2xl backdrop-blur-md lg:grid-cols-[1.1fr_0.9fr]">
        <section className="relative border-b border-cyan-200/10 p-7 lg:border-b-0 lg:border-r">
          <div className="inline-flex items-center gap-2 rounded-md border border-cyan-200/25 bg-cyan-300/10 px-2.5 py-1 text-xs font-medium text-cyan-100">
            <Activity size={13} />
            CHU Oujda
          </div>
          <h1 className="mt-4 text-3xl font-semibold text-slate-50">Clinical RAG Platform</h1>
          <p className="mt-3 max-w-md text-sm leading-6 text-slate-300">
            Assistant sécurisé pour l’analyse documentaire médicale.
          </p>
          <p className="mt-2 text-xs font-medium uppercase tracking-[0.13em] text-cyan-100/80">
            Accès réservé aux professionnels autorisés
          </p>

          <div className="mt-8 grid grid-cols-2 gap-3 max-w-md">
            <div className="rounded-lg border border-cyan-200/20 bg-cyan-300/10 p-3">
              <FileText size={16} className="text-cyan-100" />
              <p className="mt-2 text-xs text-slate-200">Traçabilité documentaire</p>
            </div>
            <div className="rounded-lg border border-teal-200/20 bg-teal-300/10 p-3">
              <ShieldCheck size={16} className="text-teal-100" />
              <p className="mt-2 text-xs text-slate-200">Accès sécurisé</p>
            </div>
          </div>
        </section>

        <section className="p-7">
          <h2 className="text-base font-semibold text-slate-100">
            {mode === "login" ? "Connexion professionnelle" : "Création de compte"}
          </h2>
          <p className="mt-1 text-sm text-slate-300">
            {mode === "login" ? "Connectez-vous pour accéder à la plateforme clinique." : "Créez votre accès à la plateforme clinique."}
          </p>

          <form className="mt-6 space-y-4" onSubmit={onSubmit}>
            <div>
              <label className="mb-1 block text-xs font-medium uppercase tracking-wide text-slate-300">Email</label>
              <input
                type="email"
                autoComplete="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                disabled={loading}
                className="h-10 w-full rounded-lg border border-slate-700 bg-slate-950/85 px-3 text-sm text-white outline-none transition focus:border-teal-400"
                placeholder="simo@test.ma"
              />
            </div>

            <div>
              <label className="mb-1 block text-xs font-medium uppercase tracking-wide text-slate-300">Mot de passe</label>
              <input
                type="password"
                autoComplete={mode === "login" ? "current-password" : "new-password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                disabled={loading}
                className="h-10 w-full rounded-lg border border-slate-700 bg-slate-950/85 px-3 text-sm text-white outline-none transition focus:border-teal-400"
                placeholder="••••••••"
              />
            </div>

            {displayError ? (
              <p className="rounded-md border border-rose-500/35 bg-rose-500/10 px-3 py-2 text-xs text-rose-200">{displayError}</p>
            ) : null}

            <button
              type="submit"
              disabled={loading}
              className="h-10 w-full rounded-lg bg-teal-400 px-4 text-sm font-semibold text-slate-950 transition hover:bg-teal-300 disabled:opacity-60"
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
            className="mt-4 w-full text-center text-xs text-slate-300 underline underline-offset-4 disabled:opacity-60"
          >
            {mode === "login" ? "Créer un compte" : "J’ai déjà un compte"}
          </button>
        </section>
      </div>
    </div>
  );
}
