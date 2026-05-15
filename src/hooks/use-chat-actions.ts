import { useMutation } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { ApiError, sendChat, toHistory } from "@/services/rag-api";
import { useAuthStore } from "@/store/auth-store";
import { useChatStore } from "@/store/chat-store";
import type { ChatMode } from "@/types/chat";

export function useChatActions() {
  const router = useRouter();
  const addUserMessage = useChatStore((s) => s.addUserMessage);
  const addAssistantLoadingMessage = useChatStore((s) => s.addAssistantLoadingMessage);
  const resolveAssistantMessage = useChatStore((s) => s.resolveAssistantMessage);
  const failAssistantMessage = useChatStore((s) => s.failAssistantMessage);
  const startNewConversation = useChatStore((s) => s.startNewConversation);

  const mutation = useMutation({
    mutationFn: async ({ content, mode }: { content: string; mode: ChatMode }) => {
      const auth = useAuthStore.getState();
      const token = auth.accessToken;
      if (!token) {
        throw new Error("Authentification requise");
      }

      let chatId = useChatStore.getState().activeConversationId;
      if (!chatId) {
        chatId = await startNewConversation(token);
        if (chatId) {
          router.push(`/chat/${chatId}`);
        }
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
          conversation_id: chatId,
          message: content,
          history: toHistory(chat?.messages || []),
          mode,
        }, token);
        resolveAssistantMessage(chatId, loading.id, response.answer, response.sources, {
          quality_report: response.quality_report,
          validation_status: response.validation_status,
          generation_mode: response.generation_mode,
          generation_writer: response.generation_writer,
          response_time: response.response_time,
        }, response.visualization, response.chart_data, response.patients, response.inventory_view);
        return response;
      } catch (error) {
        if (error instanceof ApiError && error.status === 401) {
          await useAuthStore.getState().logout();
          failAssistantMessage(
            chatId,
            loading.id,
            "Session expirée. Veuillez vous reconnecter.",
          );
          throw error;
        }
        if (error instanceof ApiError && error.status === 403) {
          failAssistantMessage(
            chatId,
            loading.id,
            "Accès interdit à cette conversation.",
          );
          throw error;
        }
        if (error instanceof ApiError && error.status === 404) {
          const token = useAuthStore.getState().accessToken;
          if (token) {
            await useChatStore.getState().loadConversations(token);
          }
          failAssistantMessage(
            chatId,
            loading.id,
            "Conversation introuvable. La liste des conversations a été rechargée.",
          );
          throw error;
        }
        failAssistantMessage(
          chatId,
          loading.id,
          "Une erreur interne a empêché la génération complète de la réponse. Les données indexées restent disponibles ; veuillez relancer la demande ou simplifier la formulation.",
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
