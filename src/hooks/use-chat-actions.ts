import { useMutation } from "@tanstack/react-query";
import { sendChat, toHistory } from "@/services/rag-api";
import { useChatStore } from "@/store/chat-store";
import type { ChatMode } from "@/types/chat";

export function useChatActions() {
  const addUserMessage = useChatStore((s) => s.addUserMessage);
  const addAssistantLoadingMessage = useChatStore((s) => s.addAssistantLoadingMessage);
  const resolveAssistantMessage = useChatStore((s) => s.resolveAssistantMessage);
  const failAssistantMessage = useChatStore((s) => s.failAssistantMessage);
  const newChat = useChatStore((s) => s.newChat);

  const mutation = useMutation({
    mutationFn: async ({ content, mode }: { content: string; mode: ChatMode }) => {
      let chatId = useChatStore.getState().activeChatId;
      if (!chatId) {
        chatId = newChat();
      }

      const user = addUserMessage(content, mode);
      if (!user || !chatId) {
        throw new Error("Unable to add user message");
      }

      const chat = useChatStore.getState().chats.find((c) => c.id === chatId);
      const loading = addAssistantLoadingMessage(chatId);
      if (!loading) {
        throw new Error("Unable to create loading message");
      }

      try {
        const response = await sendChat({
          chat_id: chatId,
          message: content,
          history: toHistory(chat?.messages || []),
          mode,
        });
        resolveAssistantMessage(chatId, loading.id, response.answer, response.sources, {
          quality_report: response.quality_report,
          validation_status: response.validation_status,
          generation_mode: response.generation_mode,
          generation_writer: response.generation_writer,
          response_time: response.response_time,
        }, response.visualization, response.chart_data);
        return response;
      } catch (error) {
        failAssistantMessage(
          chatId,
          loading.id,
          "Impossible de generer la reponse pour le moment. Reessayez ou verifiez le backend.",
        );
        throw error;
      }
    },
  });

  return {
    sendMessage: mutation.mutateAsync,
    sending: mutation.isPending,
    error: mutation.error,
  };
}
