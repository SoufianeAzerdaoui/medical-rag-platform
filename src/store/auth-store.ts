import { create } from "zustand";
import { loadMe, loginUser, logoutUser, registerUser, type AuthUser } from "@/services/rag-api";
import { useChatStore } from "@/store/chat-store";

const ACCESS_TOKEN_KEY = "clinical-access-token";
type AuthStatus = "loading" | "authenticated" | "unauthenticated";
let initAuthPromise: Promise<void> | null = null;

interface AuthState {
  user: AuthUser | null;
  accessToken: string | null;
  isAuthenticated: boolean;
  authStatus: AuthStatus;
  hasInitializedAuth: boolean;
  loading: boolean;
  error: string | null;
  initializeAuth: () => Promise<void>;
  loadMe: () => Promise<void>;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
}

function storeToken(token: string | null) {
  if (typeof window === "undefined") return;
  if (token) {
    localStorage.setItem(ACCESS_TOKEN_KEY, token);
  } else {
    localStorage.removeItem(ACCESS_TOKEN_KEY);
  }
}

function readToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(ACCESS_TOKEN_KEY);
}

export const useAuthStore = create<AuthState>((set, get) => ({
  user: null,
  accessToken: null,
  isAuthenticated: false,
  authStatus: "loading",
  hasInitializedAuth: false,
  loading: false,
  error: null,

  initializeAuth: async () => {
    if (get().hasInitializedAuth && get().authStatus !== "loading") {
      return;
    }
    if (initAuthPromise) {
      await initAuthPromise;
      return;
    }

    const token = get().accessToken || readToken();
    if (!token) {
      set({
        user: null,
        accessToken: null,
        isAuthenticated: false,
        authStatus: "unauthenticated",
        hasInitializedAuth: true,
        error: null,
      });
      return;
    }

    set({
      accessToken: token,
      authStatus: "loading",
      error: null,
    });

    initAuthPromise = (async () => {
      try {
        const user = await loadMe(token);
        set({
          user,
          accessToken: token,
          isAuthenticated: true,
          authStatus: "authenticated",
          hasInitializedAuth: true,
          error: null,
        });
        await useChatStore.getState().loadConversations(token);
      } catch {
        storeToken(null);
        useChatStore.getState().clearForLogout();
        set({
          user: null,
          accessToken: null,
          isAuthenticated: false,
          authStatus: "unauthenticated",
          hasInitializedAuth: true,
          error: "Session invalide",
        });
      } finally {
        initAuthPromise = null;
      }
    })();

    await initAuthPromise;
  },

  loadMe: async () => {
    const token = get().accessToken || readToken();
    if (!token) {
      set({
        user: null,
        accessToken: null,
        isAuthenticated: false,
        authStatus: "unauthenticated",
        hasInitializedAuth: true,
      });
      return;
    }

    set({ loading: true, authStatus: "loading", error: null });
    try {
      const user = await loadMe(token);
      set({
        user,
        accessToken: token,
        isAuthenticated: true,
        loading: false,
        authStatus: "authenticated",
        hasInitializedAuth: true,
        error: null,
      });
      await useChatStore.getState().loadConversations(token);
    } catch {
      storeToken(null);
      useChatStore.getState().clearForLogout();
      set({
        user: null,
        accessToken: null,
        isAuthenticated: false,
        loading: false,
        authStatus: "unauthenticated",
        hasInitializedAuth: true,
        error: "Session invalide",
      });
    }
  },

  login: async (email, password) => {
    set({ loading: true, authStatus: "loading", error: null });
    try {
      const response = await loginUser({ email, password });
      storeToken(response.access_token);
      set({
        user: response.user,
        accessToken: response.access_token,
        isAuthenticated: true,
        authStatus: "authenticated",
        hasInitializedAuth: true,
        loading: false,
        error: null,
      });
      await useChatStore.getState().loadConversations(response.access_token);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Login échoué";
      set({
        loading: false,
        error: message,
        isAuthenticated: false,
        authStatus: "unauthenticated",
        hasInitializedAuth: true,
      });
      throw error;
    }
  },

  register: async (email, password) => {
    set({ loading: true, authStatus: "loading", error: null });
    try {
      const response = await registerUser({ email, password });
      storeToken(response.access_token);
      set({
        user: response.user,
        accessToken: response.access_token,
        isAuthenticated: true,
        authStatus: "authenticated",
        hasInitializedAuth: true,
        loading: false,
        error: null,
      });
      await useChatStore.getState().loadConversations(response.access_token);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Inscription échouée";
      set({
        loading: false,
        error: message,
        isAuthenticated: false,
        authStatus: "unauthenticated",
        hasInitializedAuth: true,
      });
      throw error;
    }
  },

  logout: async () => {
    const token = get().accessToken || readToken();
    if (token) {
      try {
        await logoutUser(token);
      } catch {
        // best-effort
      }
    }
    storeToken(null);
    useChatStore.getState().clearForLogout();
    set({
      user: null,
      accessToken: null,
      isAuthenticated: false,
      authStatus: "unauthenticated",
      hasInitializedAuth: true,
      loading: false,
      error: null,
    });
  },
}));
