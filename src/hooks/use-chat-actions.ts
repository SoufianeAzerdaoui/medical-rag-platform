import { useMutation } from "@tanstack/react-query";
import { sendChat, toHistory } from "@/services/rag-api";
import { useChatStore } from "@/store/chat-store";
import type { ChatMode } from "@/types/chat";

export function useChatActions() {
  const chats = useChatStore((s) => s.chats);
  const activeChatId = useChatStore((s) => s.activeChatId);
  const addUserMessage = useChatStore((s) => s.addUserMessage);
  const addAssistantMessage = useChatStore((s) => s.addAssistantMessage);

  const mutation = useMutation({
    mutationFn: async ({ content, mode }: { content: string; mode: ChatMode }) => {
      const user = addUserMessage(content, mode);
      if (!user || !activeChatId) return null;
      const chat = chats.find((c) => c.id === activeChatId);
      const response = await sendChat({
        chat_id: activeChatId,
        message: content,
        history: toHistory(chat?.messages || []),
        mode,
      });
      addAssistantMessage(activeChatId, response.answer, response.sources);
      return response;
    },
  });

  return { sendMessage: mutation.mutateAsync, sending: mutation.isPending, error: mutation.error };
}
