import { create } from "zustand";
import {
  ApiError,
  type BackendMessageResponse,
  clearConversation,
  createConversation,
  deleteConversation as deleteConversationApi,
  getConversationMessages,
  listConversations,
} from "@/services/rag-api";
import { resolveAutoTitleUpdate } from "@/lib/chat-title";
import { uid } from "@/lib/utils";
import type {
  AssistantDiagnostics,
  ChatItem,
  ChatMode,
  ChatSource,
  MessageItem,
  VisualizationPayload,
} from "@/types/chat";

type Theme = "light" | "dark" | "system";

const ACCESS_TOKEN_KEY = "clinical-access-token";
const inFlightMessageLoads = new Map<string, Promise<void>>();

function getActiveChatTargetId(
  state: Pick<ChatState, "chats" | "activeChatId" | "activeConversationId">,
): string | null {
  const candidates = [state.activeConversationId, state.activeChatId].filter((value): value is string => Boolean(value));
  for (const id of candidates) {
    if (state.chats.some((chat) => chat.id === id)) return id;
  }
  return candidates[0] || null;
}

interface ChatState {
  chats: ChatItem[];
  conversations: ChatItem[];
  activeChatId: string | null;
  activeConversationId: string | null;
  messages: MessageItem[];
  search: string;
  privacyMode: boolean;
  qualityDebugEnabled: boolean;
  theme: Theme;
  language: "fr" | "ar" | "en";

  initialize: () => Promise<void>;
  loadConversations: (token?: string | null) => Promise<void>;
  createConversation: (title?: string, token?: string | null) => Promise<string | null>;
  selectConversation: (id: string, token?: string | null) => Promise<void>;
  loadMessages: (conversationId: string, token?: string | null) => Promise<void>;
  startNewConversation: (token?: string | null) => Promise<string | null>;
  clearMessages: () => void;
  newChat: () => string;
  setActiveChat: (id: string) => void;
  setSearch: (value: string) => void;
  addUserMessage: (content: string, mode: ChatMode, chatId?: string | null) => MessageItem | null;
  addAssistantLoadingMessage: (chatId: string) => MessageItem | null;
  resolveAssistantMessage: (
    chatId: string,
    messageId: string,
    content: string,
    sources?: ChatSource[],
    diagnostics?: AssistantDiagnostics,
    visualization?: VisualizationPayload,
    chartData?: VisualizationPayload,
    patients?: Array<Record<string, unknown>>,
    inventoryView?: { type: "patient_cards" | "report_accordion" | "filterable_table" | "document_timeline" },
  ) => void;
  failAssistantMessage: (chatId: string, messageId: string, content: string) => void;
  addAssistantMessage: (chatId: string, content: string, sources?: ChatSource[]) => void;
  toggleFavorite: (chatId: string) => void;
  renameChat: (chatId: string, title: string) => void;
  removeChat: (chatId: string, token?: string | null) => Promise<void>;
  clearAllData: () => Promise<void>;
  clearForLogout: () => void;
  setTheme: (theme: Theme) => void;
  togglePrivacyMode: () => void;
  toggleQualityDebug: () => void;
  setLanguage: (language: "fr" | "ar" | "en") => void;
  exportChat: (chatId: string, format: "json" | "txt") => string | null;
  exportAllChats: (format: "json" | "txt") => string;
}

const now = () => new Date().toISOString();

function getStoredToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(ACCESS_TOKEN_KEY);
}

function toChatItem(row: { id: string; title: string; created_at: string; updated_at: string }): ChatItem {
  return {
    id: row.id,
    conversationId: row.id,
    title: row.title,
    titleSource: "auto",
    titleGenerated: !/^nouvelle conversation$/i.test(String(row.title || "").trim()),
    titleEditedByUser: false,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    favorite: false,
    tags: [],
    mode: "general",
    documentIds: [],
    messages: [],
    summary: "",
  };
}

function toMessage(chatId: string, row: BackendMessageResponse): MessageItem {
  const sources = Array.isArray(row.sources) ? row.sources : [];
  const diagnostics = row.diagnostics && typeof row.diagnostics === "object" ? (row.diagnostics as AssistantDiagnostics) : undefined;
  return {
    id: row.id,
    chatId,
    role: row.role,
    content: row.content,
    createdAt: row.created_at,
    status: "done",
    sources,
    diagnostics,
  };
}

export const useChatStore = create<ChatState>((set, get) => ({
  chats: [],
  conversations: [],
  activeChatId: null,
  activeConversationId: null,
  messages: [],
  search: "",
  privacyMode: false,
  qualityDebugEnabled: false,
  theme: "dark",
  language: "fr",

  initialize: async () => {
    const theme = (localStorage.getItem("clinical-theme") as Theme | null) ?? "dark";
    const privacyMode = localStorage.getItem("clinical-privacy-mode") === "true";
    const qualityDebugEnabled = localStorage.getItem("clinical-quality-debug") === "true";
    const language = (localStorage.getItem("clinical-lang") as "fr" | "ar" | "en" | null) ?? "fr";
    set({ theme, privacyMode, qualityDebugEnabled, language });
    const token = getStoredToken();
    if (!token) {
      set({ chats: [], conversations: [], activeChatId: null, activeConversationId: null, messages: [] });
    }
  },

  loadConversations: async (token) => {
    const accessToken = token || getStoredToken();
    if (!accessToken) {
      set({ chats: [], conversations: [], activeChatId: null, activeConversationId: null, messages: [] });
      return;
    }
    let rows;
    try {
      rows = await listConversations(accessToken);
    } catch (error) {
      if (error instanceof ApiError && (error.status === 401 || error.status === 403)) {
        localStorage.removeItem(ACCESS_TOKEN_KEY);
        set({ chats: [], conversations: [], activeChatId: null, activeConversationId: null, messages: [] });
      }
      throw error;
    }
    const mapped: ChatItem[] = rows.map(toChatItem);
    const currentActiveId = get().activeConversationId;
    const active = currentActiveId
      ? mapped.find((chat) => chat.id === currentActiveId) ?? null
      : mapped[0] ?? null;
    set({
      chats: mapped,
      conversations: mapped,
      activeChatId: active?.id ?? null,
      activeConversationId: active?.conversationId ?? null,
      messages: active?.messages ?? [],
    });
  },

  createConversation: async (title, token) => {
    const accessToken = token || getStoredToken();
    if (!accessToken) return null;
    const created = await createConversation(accessToken, { title });
    const chat = toChatItem(created);
    set((state) => ({
      chats: [chat, ...state.chats],
      conversations: [chat, ...state.conversations],
      activeChatId: chat.id,
      activeConversationId: chat.conversationId,
      messages: [],
    }));
    return chat.id;
  },

  selectConversation: async (id, token) => {
    set({ activeChatId: id, activeConversationId: id, messages: [] });
    try {
      await get().loadMessages(id, token);
    } catch (error) {
      set((state) => {
        const chats = state.chats.map((chat) =>
          chat.id === id ? { ...chat, messages: [] } : chat,
        );
        return { chats, conversations: chats, messages: [] };
      });
      throw error;
    }
  },

  loadMessages: async (conversationId, token) => {
    const existingLoad = inFlightMessageLoads.get(conversationId);
    if (existingLoad) {
      await existingLoad;
      return;
    }

    const loadPromise = (async () => {
    const accessToken = token || getStoredToken();
    if (!accessToken) return;
    const current = get();
    const existing = current.chats.find((chat) => chat.id === conversationId) || null;
    if (
      current.activeConversationId === conversationId
      && existing
      && existing.messages.length > 0
      && current.messages.length === existing.messages.length
    ) {
      return;
    }
    const rows = await getConversationMessages(accessToken, conversationId);
    const mapped = rows.map((row) => toMessage(conversationId, row));
    set((state) => {
      const hasExistingConversation = state.chats.some((chat) => chat.id === conversationId);
      const chats: ChatItem[] = hasExistingConversation
        ? state.chats.map((chat) =>
            chat.id === conversationId ? { ...chat, messages: mapped, updatedAt: mapped.at(-1)?.createdAt || chat.updatedAt } : chat,
          )
        : [
            {
              id: conversationId,
              conversationId,
              title: "Conversation",
              titleSource: "auto",
              titleGenerated: true,
              titleEditedByUser: false,
              createdAt: now(),
              updatedAt: mapped.at(-1)?.createdAt || now(),
              favorite: false,
              tags: [],
              mode: "general",
              documentIds: [],
              messages: mapped,
              summary: "",
            } as ChatItem,
            ...state.chats,
          ];
      return {
        chats,
        conversations: chats,
        messages: state.activeChatId === conversationId ? mapped : state.messages,
      };
    });
    })();

    inFlightMessageLoads.set(conversationId, loadPromise);
    try {
      await loadPromise;
    } finally {
      inFlightMessageLoads.delete(conversationId);
    }
  },

  startNewConversation: async (token) => {
    const previousConversationId = get().activeConversationId;
    const created = await get().createConversation("Nouvelle conversation", token);
    const accessToken = token || getStoredToken();
    if (previousConversationId && accessToken) {
      try {
        await clearConversation(accessToken, previousConversationId);
      } catch {
        // State clear is best-effort.
      }
    }
    return created;
  },

  clearMessages: () => {
    const { chats } = get();
    const activeTargetId = getActiveChatTargetId(get());
    if (!activeTargetId) return;
    const updated = chats.map((chat) =>
      chat.id === activeTargetId
        ? { ...chat, messages: [], updatedAt: now(), documentIds: [], summary: "" }
        : chat,
    );
    set({ chats: updated, conversations: updated, messages: [] });
  },

  newChat: () => {
    const chat: ChatItem = {
      id: uid("chat-local"),
      conversationId: uid("chat-local-conv"),
      title: "Nouveau chat clinique",
      titleSource: "auto",
      titleGenerated: false,
      titleEditedByUser: false,
      createdAt: now(),
      updatedAt: now(),
      favorite: false,
      tags: [],
      mode: "general",
      documentIds: [],
      messages: [],
      summary: "",
    };
    set((state) => ({
      chats: [chat, ...state.chats],
      conversations: [chat, ...state.conversations],
      activeChatId: chat.id,
      activeConversationId: chat.conversationId,
      messages: [],
    }));
    return chat.id;
  },

  setActiveChat: (id) => {
    const chat = get().chats.find((item) => item.id === id) || null;
    set({
      activeChatId: id,
      activeConversationId: chat?.conversationId || id,
      messages: chat?.messages || [],
    });
  },

  setSearch: (value) => set({ search: value }),

  addUserMessage: (content, mode, chatId) => {
    const { chats } = get();
    const explicitTarget = chatId || null;
    const activeTargetId = explicitTarget && chats.some((chat) => chat.id === explicitTarget)
      ? explicitTarget
      : getActiveChatTargetId(get());
    if (!activeTargetId) return null;
    const hasTargetChat = chats.some((chat) => chat.id === activeTargetId);
    if (!hasTargetChat) return null;
    const msg: MessageItem = {
      id: uid("msg"),
      chatId: activeTargetId,
      role: "user",
      content,
      createdAt: now(),
      status: "done",
    };
    const updated: ChatItem[] = chats.map((chat) =>
      chat.id === activeTargetId
        ? {
            ...chat,
            ...(() => resolveAutoTitleUpdate({
              currentTitle: chat.title,
              titleSource: chat.titleSource ?? "auto",
              titleGenerated: chat.titleGenerated,
              titleEditedByUser: chat.titleEditedByUser,
              userMessage: content,
              mode,
              sources: chat.messages.flatMap((m) => m.sources || []),
            }))(),
            mode,
            messages: [...chat.messages, msg],
            updatedAt: now(),
          }
        : chat,
    );
    const active = updated.find((chat) => chat.id === activeTargetId) || null;
    set({ chats: updated, conversations: updated, messages: active?.messages || [] });
    return msg;
  },

  addAssistantMessage: (chatId, content, sources) => {
    const { chats } = get();
    const msg: MessageItem = {
      id: uid("msg"),
      chatId,
      role: "assistant",
      content,
      createdAt: now(),
      status: "done",
      sources,
    };
    const updated = chats.map((chat) =>
      chat.id === chatId ? { ...chat, messages: [...chat.messages, msg], updatedAt: now() } : chat,
    );
    const active = updated.find((chat) => chat.id === chatId) || null;
    set({ chats: updated, conversations: updated, messages: active?.messages || get().messages });
  },

  addAssistantLoadingMessage: (chatId) => {
    const { chats } = get();
    const preferredTargetId = chats.some((chat) => chat.id === chatId) ? chatId : null;
    const activeTargetId = preferredTargetId || getActiveChatTargetId(get());
    if (!activeTargetId) return null;
    const hasTargetChat = chats.some((chat) => chat.id === activeTargetId);
    if (!hasTargetChat) return null;
    const msg: MessageItem = {
      id: uid("msg"),
      chatId: activeTargetId,
      role: "assistant",
      content: "L’assistant prépare une réponse.",
      createdAt: now(),
      status: "loading",
      sources: [],
    };
    const updated = chats.map((chat) =>
      chat.id === activeTargetId ? { ...chat, messages: [...chat.messages, msg], updatedAt: now() } : chat,
    );
    const active = updated.find((chat) => chat.id === activeTargetId) || null;
    set({ chats: updated, conversations: updated, messages: active?.messages || get().messages });
    return msg;
  },

  resolveAssistantMessage: (chatId, messageId, content, sources, diagnostics, visualization, chartData, patients, inventoryView) => {
    const { chats } = get();
    const updated = chats.map((chat) => {
      if (chat.id !== chatId) return chat;
      const messages: MessageItem[] = chat.messages.map((m): MessageItem =>
        m.id === messageId
          ? {
              ...m,
              content,
              status: "done",
              sources: sources ?? [],
              diagnostics: diagnostics ?? {},
              visualization,
              chart_data: chartData,
              patients,
              inventory_view: inventoryView,
            }
          : m,
      );
      const firstUser = messages.find((m) => m.role === "user")?.content || "";
      const resolvedTitle = resolveAutoTitleUpdate({
        currentTitle: chat.title,
        titleSource: chat.titleSource ?? "auto",
        titleGenerated: chat.titleGenerated,
        titleEditedByUser: chat.titleEditedByUser,
        userMessage: firstUser,
        assistantMessage: content,
        mode: chat.mode,
        sources: sources ?? [],
      });
      return { ...chat, ...resolvedTitle, messages, updatedAt: now() };
    });
    const active = updated.find((chat) => chat.id === chatId) || null;
    set({ chats: updated, conversations: updated, messages: active?.messages || get().messages });
  },

  failAssistantMessage: (chatId, messageId, content) => {
    const { chats } = get();
    const updated = chats.map((chat) => {
      if (chat.id !== chatId) return chat;
      const messages: MessageItem[] = chat.messages.map((m): MessageItem =>
        m.id === messageId
          ? {
              ...m,
              content,
              status: "error",
              sources: [],
            }
          : m,
      );
      return { ...chat, messages, updatedAt: now() };
    });
    const active = updated.find((chat) => chat.id === chatId) || null;
    set({ chats: updated, conversations: updated, messages: active?.messages || get().messages });
  },

  toggleFavorite: (chatId) => {
    const { chats } = get();
    const updated = chats.map((chat) =>
      chat.id === chatId ? { ...chat, favorite: !chat.favorite, updatedAt: now() } : chat,
    );
    set({ chats: updated, conversations: updated });
  },

  renameChat: (chatId, title) => {
    const { chats } = get();
    const updated: ChatItem[] = chats.map((chat) =>
      chat.id === chatId
        ? {
            ...chat,
            title: title || chat.title,
            titleSource: "manual" as const,
            titleGenerated: true,
            titleEditedByUser: true,
            updatedAt: now(),
          }
        : chat,
    );
    set({ chats: updated, conversations: updated });
  },

  removeChat: async (chatId, token) => {
    const accessToken = token || getStoredToken();
    if (accessToken && !chatId.startsWith("chat-local")) {
      await deleteConversationApi(accessToken, chatId);
    }
    const remaining = get().chats.filter((chat) => chat.id !== chatId);
    const next = remaining[0] || null;
    set({
      chats: remaining,
      conversations: remaining,
      activeChatId: next?.id ?? null,
      activeConversationId: next?.conversationId ?? null,
      messages: next?.messages ?? [],
    });
  },

  clearAllData: async () => {
    set({ chats: [], conversations: [], activeChatId: null, activeConversationId: null, messages: [] });
  },

  clearForLogout: () => {
    inFlightMessageLoads.clear();
    set({ chats: [], conversations: [], activeChatId: null, activeConversationId: null, messages: [] });
  },

  setTheme: (theme) => {
    localStorage.setItem("clinical-theme", theme);
    set({ theme });
  },

  togglePrivacyMode: () =>
    set((state) => {
      const nextValue = !state.privacyMode;
      localStorage.setItem("clinical-privacy-mode", String(nextValue));
      return { privacyMode: nextValue };
    }),

  toggleQualityDebug: () =>
    set((state) => {
      const nextValue = !state.qualityDebugEnabled;
      localStorage.setItem("clinical-quality-debug", String(nextValue));
      return { qualityDebugEnabled: nextValue };
    }),

  setLanguage: (language) => {
    localStorage.setItem("clinical-lang", language);
    set({ language });
  },

  exportChat: (chatId, format) => {
    const chat = get().chats.find((item) => item.id === chatId);
    if (!chat) return null;
    if (format === "json") return JSON.stringify(chat, null, 2);
    return [
      `Titre: ${chat.title}`,
      `Créé: ${chat.createdAt}`,
      `Mis à jour: ${chat.updatedAt}`,
      "",
      ...chat.messages.map((m) => `[${m.role}] ${m.content}`),
    ].join("\n");
  },

  exportAllChats: (format) => {
    const chats = get().chats;
    if (format === "json") return JSON.stringify(chats, null, 2);
    return chats
      .map((chat) => [
        `=== ${chat.title} ===`,
        ...chat.messages.map((m) => `[${m.role}] ${m.content}`),
        "",
      ].join("\n"))
      .join("\n");
  },
}));
