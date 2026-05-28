"use client";

import { AnimatePresence, motion } from "framer-motion";
import { ShieldCheck, X } from "lucide-react";
import { useMemo } from "react";
import { SourceLinks } from "@/components/sources/source-links";
import { useChatStore } from "@/store/chat-store";

export function SourcesPanel({ mobileOpen, onClose }: { mobileOpen: boolean; onClose: () => void }) {
  const chats = useChatStore((s) => s.chats);
  const activeChatId = useChatStore((s) => s.activeChatId);
  const chat = chats.find((c) => c.id === activeChatId);

  const sources = useMemo(() => {
    if (!chat) return [];
    return chat.messages.flatMap((m) => m.sources || []);
  }, [chat]);

  return (
    <>
      <aside className="glass hidden h-screen w-80 shrink-0 flex-col border-y-0 border-r-0 p-4 xl:flex">
        <div className="mb-4 flex items-start gap-3">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-accent/10 text-accent">
            <ShieldCheck size={17} />
          </div>
          <div>
            <h3 className="text-sm font-semibold">Sources</h3>
            <p className="mt-1 text-xs leading-5 text-fg/60">Documents et passages utilisés par la réponse.</p>
          </div>
        </div>
        <div className="space-y-2 overflow-auto">
          {sources.length === 0 ? <p className="text-xs text-fg/70">Aucune source pour le moment.</p> : <SourceLinks sources={sources} showTitle={false} compact maxVisible={12} />}
        </div>
      </aside>
      <AnimatePresence>
        {mobileOpen ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-black/50 xl:hidden"
            onClick={onClose}
          >
            <motion.aside
              initial={{ x: "100%" }}
              animate={{ x: 0 }}
              exit={{ x: "100%" }}
              transition={{ type: "spring", stiffness: 260, damping: 24 }}
              className="glass absolute right-0 top-0 h-full w-[88%] max-w-sm border-y-0 border-r-0 p-4"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="mb-3 flex items-center justify-between">
                <h3 className="text-sm font-semibold">Sources</h3>
                <button aria-label="Fermer les sources" className="icon-button h-8 w-8" onClick={onClose}>
                  <X size={14} />
                </button>
              </div>
              <div className="space-y-2 overflow-auto">
                {sources.length === 0 ? <p className="text-xs text-fg/70">Aucune source pour le moment.</p> : <SourceLinks sources={sources} showTitle={false} compact maxVisible={12} />}
              </div>
            </motion.aside>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </>
  );
}
