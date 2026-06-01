import { Suspense } from "react";
import { AuthPanel } from "@/components/auth/auth-panel";

export default function AuthPage() {
  return (
    <Suspense
      fallback={
        <div className="flex h-screen items-center justify-center px-6">
          <div className="glass rounded-xl px-5 py-4 text-sm text-fg/70">Chargement de l&apos;authentification...</div>
        </div>
      }
    >
      <AuthPanel />
    </Suspense>
  );
}
