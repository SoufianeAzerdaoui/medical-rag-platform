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
  const addAssistantMessage = useChatStore((s) => s.addAssistantMessage);
  const resolveAssistantMessage = useChatStore((s) => s.resolveAssistantMessage);
  const failAssistantMessage = useChatStore((s) => s.failAssistantMessage);
  const startNewConversation = useChatStore((s) => s.startNewConversation);

  const mutation = useMutation({
    mutationFn: async (
      {
        content,
        mode,
        summaryStyle,
        llmProviderOverride,
        llmModelOverride,
      }: {
        content: string;
        mode: ChatMode;
        summaryStyle?: "short" | "editorial" | null;
        llmProviderOverride?: string | null;
        llmModelOverride?: string | null;
      },
    ) => {
      const auth = useAuthStore.getState();
      const token = auth.accessToken;
      if (!token) {
        throw new Error("Authentification requise");
      }

      const hasChatInStore = (id: string) => useChatStore.getState().chats.some((c) => c.id === id);

      let chatId = useChatStore.getState().activeConversationId || useChatStore.getState().activeChatId;
      if (!chatId) {
        chatId = await startNewConversation(token);
        if (chatId) {
          router.push(`/chat/${chatId}`);
        }
      }
      if (!chatId) {
        throw new Error("Conversation introuvable pour l'envoi.");
      }

      if (!hasChatInStore(chatId)) {
        try {
          await useChatStore.getState().loadMessages(chatId, token);
        } catch {
          // Ignore and try a broader resync.
        }
      }
      if (!hasChatInStore(chatId)) {
        try {
          await useChatStore.getState().loadConversations(token);
        } catch {
          // Ignore; we still surface a clear error below.
        }
      }
      if (!hasChatInStore(chatId)) {
        throw new Error(`Conversation ${chatId} absente du store frontend.`);
      }

      const user = addUserMessage(content, mode, chatId);
      if (!user) {
        throw new Error(`Impossible d'ajouter le message utilisateur pour ${chatId}.`);
      }

      const chat = useChatStore.getState().chats.find((c) => c.id === chatId);
      const loading = addAssistantLoadingMessage(chatId);
      if (!loading) {
        throw new Error(`Impossible de créer le message loading pour ${chatId}.`);
      }

      try {
        const response = await sendChat({
          conversation_id: chatId,
          message: content,
          history: toHistory(chat?.messages || []),
          mode,
          summary_style: summaryStyle,
          llm_provider_override: llmProviderOverride,
          llm_model_override: llmModelOverride,
        }, token);
        const enriched = response as typeof response & {
          debug?: Record<string, unknown> | null;
        };
        const debug = (enriched.debug || {}) as Record<string, unknown>;
        const rawDebug = ((debug.raw_debug || {}) as Record<string, unknown>);
        const debugQuery = ((debug.query_understanding || rawDebug.query_understanding || {}) as Record<string, unknown>);
        const llmProviderEffectiveRuntime =
          (response.llm_provider_effective_runtime as string | null) ??
          (debug.llm_provider_effective_runtime as string | null) ??
          (rawDebug.llm_provider_effective_runtime as string | null) ??
          null;
        const llmModelEffectiveRuntime =
          (response.llm_model_effective_runtime as string | null) ??
          (debug.llm_model_effective_runtime as string | null) ??
          (rawDebug.llm_model_effective_runtime as string | null) ??
          null;
        resolveAssistantMessage(chatId, loading.id, response.answer, response.sources, {
          quality_report: response.quality_report,
          validation_status: response.validation_status,
          generation_mode: response.generation_mode,
          generation_writer: response.generation_writer,
          provider: response.provider ?? null,
          model: response.model ?? null,
          llm_provider_effective_runtime: llmProviderEffectiveRuntime,
          llm_model_effective_runtime: llmModelEffectiveRuntime,
          response_time: response.response_time,
          intent: response.intent ?? (debugQuery.intent as string | null) ?? null,
          selected_route: response.selected_route ?? (debug.selected_route as string | null) ?? (rawDebug.selected_route as string | null) ?? null,
          route_reason: response.route_reason ?? (debug.route_reason as string | null) ?? (rawDebug.route_reason as string | null) ?? null,
          technical_condition: response.technical_condition ?? (debugQuery.technical_condition as string | null) ?? null,
          requested_doc_ids:
            response.requested_doc_ids ??
            ((debugQuery.requested_doc_ids as string[] | null) ?? null),
          requested_analytes:
            response.requested_analytes ??
            ((debugQuery.requested_analytes as string[] | null) ?? null),
          answerability_status:
            response.answerability_status ??
            (debug.answerability_status as string | null) ??
            (rawDebug.answerability_status as string | null) ??
            null,
          fallback_kind:
            response.fallback_kind ??
            (debug.fallback_kind as string | null) ??
            (rawDebug.fallback_kind as string | null) ??
            null,
          llm_route_class:
            (debug.llm_route_class as string | null) ??
            (rawDebug.llm_route_class as string | null) ??
            null,
          llm_writer_attempted:
            (response.llm_writer_attempted as boolean | null) ??
            (debug.llm_writer_attempted as boolean | null) ??
            (rawDebug.llm_writer_attempted as boolean | null) ??
            null,
          llm_writer_accepted:
            (response.llm_writer_accepted as boolean | null) ??
            (debug.llm_writer_accepted as boolean | null) ??
            (rawDebug.llm_writer_accepted as boolean | null) ??
            null,
          llm_quality_escalation_used:
            (response.llm_quality_escalation_used as boolean | null) ??
            (debug.llm_quality_escalation_used as boolean | null) ??
            (rawDebug.llm_quality_escalation_used as boolean | null) ??
            null,
          llm_quality_escalation_reason:
            (response.llm_quality_escalation_reason as string | null) ??
            (debug.llm_quality_escalation_reason as string | null) ??
            (rawDebug.llm_quality_escalation_reason as string | null) ??
            null,
          summary_style_requested:
            (response.debug?.summary_style_requested as "short" | "editorial" | null) ??
            (debug.summary_style_requested as "short" | "editorial" | null) ??
            (rawDebug.summary_style_requested as "short" | "editorial" | null) ??
            null,
          final_answer_source:
            (response.final_answer_source as "llm_writer" | "deterministic_renderer" | null) ??
            (debug.final_answer_source as "llm_writer" | "deterministic_renderer" | null) ??
            (rawDebug.final_answer_source as "llm_writer" | "deterministic_renderer" | null) ??
            null,
          renderer_used:
            (response.renderer_used as string | null) ??
            (debug.renderer_used as string | null) ??
            (rawDebug.renderer_used as string | null) ??
            null,
          fallback_reason:
            (response.fallback_reason as string | null) ??
            (debug.fallback_reason as string | null) ??
            (rawDebug.fallback_reason as string | null) ??
            null,
          llm_skipped_reason:
            (debug.llm_skipped_reason as string | null) ??
            (rawDebug.llm_skipped_reason as string | null) ??
            null,
          generation_mode_before_fallback:
            (debug.generation_mode_before_fallback as string | null) ??
            (rawDebug.generation_mode_before_fallback as string | null) ??
            null,
          fallback_decision_path:
            (debug.fallback_decision_path as string | null) ??
            (rawDebug.fallback_decision_path as string | null) ??
            null,
          answer_type:
            (response.answer_type as string | null) ??
            (debug.answer_type as string | null) ??
            (rawDebug.answer_type as string | null) ??
            null,
        }, response.visualization, response.chart_data, response.patients, response.inventory_view);

        const shouldLogRuntime =
          typeof window !== "undefined" &&
          (process.env.NODE_ENV !== "production" ||
            localStorage.getItem("clinical-quality-debug") === "true" ||
            process.env.NEXT_PUBLIC_DEBUG_LLM_RUNTIME === "true");
        if (shouldLogRuntime) {
          console.groupCollapsed("[chat] llm runtime");
          console.table([
            {
              selected_route: response.selected_route ?? (debug.selected_route as string | null) ?? (rawDebug.selected_route as string | null) ?? null,
              generation_mode: response.generation_mode ?? (debug.generation_mode as string | null) ?? (rawDebug.generation_mode as string | null) ?? null,
              final_answer_source: response.final_answer_source ?? (debug.final_answer_source as string | null) ?? (rawDebug.final_answer_source as string | null) ?? null,
              front_provider_selected: llmProviderOverride ?? null,
              front_model_selected: llmModelOverride ?? null,
              provider_requested: response.provider ?? null,
              model_requested: response.model ?? null,
              provider_effective_runtime: llmProviderEffectiveRuntime,
              model_effective_runtime: llmModelEffectiveRuntime,
              llm_writer_attempted: response.llm_writer_attempted ?? (debug.llm_writer_attempted as boolean | null) ?? (rawDebug.llm_writer_attempted as boolean | null) ?? null,
              llm_writer_accepted: response.llm_writer_accepted ?? (debug.llm_writer_accepted as boolean | null) ?? (rawDebug.llm_writer_accepted as boolean | null) ?? null,
              llm_quality_escalation_used:
                response.llm_quality_escalation_used ??
                (debug.llm_quality_escalation_used as boolean | null) ??
                (rawDebug.llm_quality_escalation_used as boolean | null) ??
                null,
              llm_quality_escalation_reason:
                response.llm_quality_escalation_reason ??
                (debug.llm_quality_escalation_reason as string | null) ??
                (rawDebug.llm_quality_escalation_reason as string | null) ??
                null,
              summary_style_requested:
                (response.debug?.summary_style_requested as string | null) ??
                (debug.summary_style_requested as string | null) ??
                (rawDebug.summary_style_requested as string | null) ??
                null,
              fallback_reason: response.fallback_reason ?? (debug.fallback_reason as string | null) ?? (rawDebug.fallback_reason as string | null) ?? null,
            },
          ]);
          console.groupEnd();
        }
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
    onError: (error) => {
      const state = useChatStore.getState();
      const chatId = state.activeConversationId || state.activeChatId;
      if (chatId) {
        addAssistantMessage(
          chatId,
          "Envoi impossible côté interface. Rechargez la page puis relancez la question.",
          [],
        );
      }
      if (typeof window !== "undefined") {
        // Visible only in browser console for rapid production diagnostics.
        console.error("chat_send_failed", {
          error,
          activeChatId: state.activeChatId,
          activeConversationId: state.activeConversationId,
          chatsCount: state.chats.length,
        });
      }
    },
  });

  return {
    sendMessage: mutation.mutateAsync,
    sending: mutation.isPending,
    error: mutation.error,
  };
}
