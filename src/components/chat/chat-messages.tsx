"use client";

import { motion } from "framer-motion";
import { AlertTriangle, Copy, RotateCcw } from "lucide-react";
import { useChatStore } from "@/store/chat-store";

export function ChatMessages() {
  const chats = useChatStore((s) => s.chats);
  const activeChatId = useChatStore((s) => s.activeChatId);
  const chat = chats.find((c) => c.id === activeChatId);

  if (!chat || chat.messages.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-4 p-6 text-center">
        <h2 className="text-2xl font-semibold">Assistant clinique CHU Oujda</h2>
        <p className="max-w-xl text-fg/75">Cette réponse ne remplace pas l'avis médical.</p>
        <div className="grid w-full max-w-3xl gap-2 md:grid-cols-2">
          {[
            "Résume ce rapport biologique.",
            "Quels résultats semblent anormaux ?",
            "Explique les valeurs importantes.",
            "Compare ces deux documents.",
            "Quels éléments nécessitent vérification ?",
          ].map((s) => (
            <button key={s} className="rounded-xl border border-border px-3 py-2 text-left text-sm hover:bg-card">
              {s}
            </button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4 p-6">
      {chat.messages.map((message) => (
        <motion.article
          key={message.id}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.2 }}
          className="rounded-2xl border border-border p-4"
        >
          <p className="mb-2 text-xs uppercase text-fg/60">{message.role === "user" ? "Vous" : "Assistant"}</p>
          <p className="whitespace-pre-wrap text-sm leading-6">{message.content}</p>
          {message.role === "assistant" && (
            <div className="mt-3 flex gap-2">
              <button aria-label="Copier" className="rounded-lg border border-border p-2">
                <Copy size={14} />
              </button>
              <button aria-label="Régénérer" className="rounded-lg border border-border p-2">
                <RotateCcw size={14} />
              </button>
              <button aria-label="Feedback" className="rounded-lg border border-border p-2">
                <AlertTriangle size={14} />
              </button>
            </div>
          )}
        </motion.article>
      ))}
    </div>
  );
}
