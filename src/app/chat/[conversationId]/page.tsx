import { ConversationPageClient } from "@/app/chat/[conversationId]/conversation-page-client";

interface ChatConversationPageProps {
  params: Promise<{
    conversationId: string;
  }>;
}

export default async function ChatConversationPage({ params }: ChatConversationPageProps) {
  const resolvedParams = await params;
  return <ConversationPageClient conversationId={resolvedParams.conversationId} />;
}
