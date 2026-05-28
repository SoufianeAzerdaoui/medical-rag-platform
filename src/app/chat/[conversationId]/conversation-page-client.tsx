"use client";

import { ChatShell } from "@/components/layout/chat-shell";

interface ConversationPageClientProps {
  conversationId: string;
}

export function ConversationPageClient({ conversationId }: ConversationPageClientProps) {
  return <ChatShell routeConversationId={conversationId} />;
}
