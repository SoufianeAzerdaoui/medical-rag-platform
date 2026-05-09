"use client";

import { PanelRightClose, PanelRightOpen } from "lucide-react";
import { useEffect, useState } from "react";
import { ChatMessages } from "@/components/chat/chat-messages";
import { MessageComposer } from "@/components/chat/message-composer";
import { ChatSidebar } from "@/components/sidebar/chat-sidebar";
import { SourcesPanel } from "@/components/sources/sources-panel";
import { useChatStore } from "@/store/chat-store";

export function ChatShell() {
  const initialize = useChatStore((s) => s.initialize);
  const activeChatId = useChatStore((s) => s.activeChatId);
  const newChat = useChatStore((s) => s.newChat);
  const [sourcesOpenMobile, setSourcesOpenMobile] = useState(false);

  useEffect(() => {
    void initialize();
  }, [initialize]);

  useEffect(() => {
    if (!activeChatId) newChat();
  }, [activeChatId, newChat]);

  return (
    <div className="flex h-screen overflow-hidden">
      <ChatSidebar />
      <main className="flex min-w-0 flex-1 flex-col">
        <div className="border-b border-border px-4 py-2 xl:hidden">
          <button
            className="inline-flex items-center gap-2 rounded-lg border border-border px-3 py-1 text-xs"
            onClick={() => setSourcesOpenMobile((v) => !v)}
            aria-label="Ouvrir panneau sources"
          >
            {sourcesOpenMobile ? <PanelRightClose size={14} /> : <PanelRightOpen size={14} />}
            Sources
          </button>
        </div>
        <div className="flex-1 overflow-auto">
          <ChatMessages />
        </div>
        <MessageComposer />
      </main>
      <SourcesPanel mobileOpen={sourcesOpenMobile} onClose={() => setSourcesOpenMobile(false)} />
    </div>
  );
}
