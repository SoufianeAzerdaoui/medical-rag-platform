import { create } from "zustand";
import { clearChats, deleteChat, getChats, putChat } from "@/lib/indexeddb";
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

interface ChatState {
  chats: ChatItem[];
  activeChatId: string | null;
  search: string;
  privacyMode: boolean;
  qualityDebugEnabled: boolean;
  theme: Theme;
  initialize: () => Promise<void>;
  newChat: () => string;
  setActiveChat: (id: string) => void;
  setSearch: (value: string) => void;
  addUserMessage: (content: string, mode: ChatMode) => MessageItem | null;
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
  removeChat: (chatId: string) => void;
  clearAllData: () => Promise<void>;
  setTheme: (theme: Theme) => void;
  togglePrivacyMode: () => void;
  toggleQualityDebug: () => void;
  language: "fr" | "ar" | "en";
  setLanguage: (language: "fr" | "ar" | "en") => void;
  exportChat: (chatId: string, format: "json" | "txt") => string | null;
  exportAllChats: (format: "json" | "txt") => string;
}

const now = () => new Date().toISOString();

export const useChatStore = create<ChatState>((set, get) => ({
  chats: [],
  activeChatId: null,
  search: "",
  privacyMode: false,
  qualityDebugEnabled: false,
  theme: "dark",
  language: "fr",
  initialize: async () => {
    const chats = await getChats();
    const theme = (localStorage.getItem("clinical-theme") as Theme | null) ?? "dark";
    const privacyMode = localStorage.getItem("clinical-privacy-mode") === "true";
    const qualityDebugEnabled = localStorage.getItem("clinical-quality-debug") === "true";
    const language = (localStorage.getItem("clinical-lang") as "fr" | "ar" | "en" | null) ?? "fr";
    set({
      chats: chats.sort((a, b) => +new Date(b.updatedAt) - +new Date(a.updatedAt)),
      activeChatId: chats[0]?.id ?? null,
      theme,
      privacyMode,
      qualityDebugEnabled,
      language,
    });
  },
  newChat: () => {
    const chat: ChatItem = {
      id: uid("chat"),
      title: "Nouveau chat clinique",
      createdAt: now(),
      updatedAt: now(),
      favorite: false,
      tags: [],
      mode: "general",
      documentIds: [],
      messages: [],
      summary: "",
    };
    set((state) => ({ chats: [chat, ...state.chats], activeChatId: chat.id }));
    void putChat(chat);
    return chat.id;
  },
  setActiveChat: (id) => set({ activeChatId: id }),
  setSearch: (value) => set({ search: value }),
  addUserMessage: (content, mode) => {
    const { activeChatId, chats } = get();
    if (!activeChatId) return null;
    const msg: MessageItem = {
      id: uid("msg"),
      chatId: activeChatId,
      role: "user",
      content,
      createdAt: now(),
      status: "done",
    };
    const updated = chats.map((chat) =>
      chat.id === activeChatId
        ? {
            ...chat,
            mode,
            title: chat.messages.length === 0 ? content.slice(0, 60) : chat.title,
            messages: [...chat.messages, msg],
            updatedAt: now(),
          }
        : chat,
    );
    const target = updated.find((c) => c.id === activeChatId);
    if (target) void putChat(target);
    set({ chats: updated });
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
    const target = updated.find((c) => c.id === chatId);
    if (target) void putChat(target);
    set({ chats: updated });
  },
  addAssistantLoadingMessage: (chatId) => {
    const { chats } = get();
    const msg: MessageItem = {
      id: uid("msg"),
      chatId,
      role: "assistant",
      content: "L’assistant prépare une réponse.",
      createdAt: now(),
      status: "loading",
      sources: [],
    };
    const updated = chats.map((chat) =>
      chat.id === chatId ? { ...chat, messages: [...chat.messages, msg], updatedAt: now() } : chat,
    );
    const target = updated.find((c) => c.id === chatId);
    if (target) void putChat(target);
    set({ chats: updated });
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
      return { ...chat, messages, updatedAt: now() };
    });
    const target = updated.find((c) => c.id === chatId);
    if (target) void putChat(target);
    set({ chats: updated });
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
    const target = updated.find((c) => c.id === chatId);
    if (target) void putChat(target);
    set({ chats: updated });
  },
  toggleFavorite: (chatId) => {
    const { chats } = get();
    const updated = chats.map((chat) =>
      chat.id === chatId ? { ...chat, favorite: !chat.favorite, updatedAt: now() } : chat,
    );
    const target = updated.find((c) => c.id === chatId);
    if (target) void putChat(target);
    set({ chats: updated });
  },
  renameChat: (chatId, title) => {
    const { chats } = get();
    const updated = chats.map((chat) =>
      chat.id === chatId ? { ...chat, title: title || chat.title, updatedAt: now() } : chat,
    );
    const target = updated.find((c) => c.id === chatId);
    if (target) void putChat(target);
    set({ chats: updated });
  },
  removeChat: (chatId) => {
    const remaining = get().chats.filter((chat) => chat.id !== chatId);
    set({ chats: remaining, activeChatId: remaining[0]?.id ?? null });
    void deleteChat(chatId);
  },
  clearAllData: async () => {
    await clearChats();
    set({ chats: [], activeChatId: null });
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
      .map((chat) =>
        [
          `=== ${chat.title} ===`,
          ...chat.messages.map((m) => `[${m.role}] ${m.content}`),
          "",
        ].join("\n"),
      )
      .join("\n");
  },
}));
