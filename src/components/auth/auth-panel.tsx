"use client";

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
    <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-6">
      <div className="w-full max-w-md rounded-2xl border border-white/10 bg-slate-900/80 p-6 shadow-xl backdrop-blur">
        <h1 className="text-2xl font-semibold text-white">Medical RAG Platform</h1>
        <p className="mt-1 text-sm text-slate-300">
          {mode === "login" ? "Connectez-vous pour accéder à vos conversations." : "Créez votre compte pour démarrer."}
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
              className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white outline-none focus:border-cyan-400"
              placeholder="you@example.com"
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
              className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white outline-none focus:border-cyan-400"
              placeholder="••••••••"
            />
          </div>

          {displayError ? (
            <p className="rounded-md border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-xs text-rose-200">{displayError}</p>
          ) : null}

          <button
            type="submit"
            disabled={loading}
            className="w-full rounded-lg bg-cyan-500 px-4 py-2 text-sm font-semibold text-slate-950 hover:bg-cyan-400 disabled:opacity-60"
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
      </div>
    </div>
  );
}
