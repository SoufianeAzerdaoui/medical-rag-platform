import { ChatShell } from "@/components/layout/chat-shell";

interface ChatConversationPageProps {
  params: Promise<{
    conversationId: string;
  }>;
}

export default async function ChatConversationPage({ params }: ChatConversationPageProps) {
  const resolvedParams = await params;
  return <ChatShell routeConversationId={resolvedParams.conversationId} />;
}
