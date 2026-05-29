import type { Page, Route } from "@playwright/test";

export type ThemeName = "light" | "dark";
export type ScenarioName = "empty" | "clinical";

const API_BASE = process.env.NEXT_PUBLIC_RAG_API_URL || "http://127.0.0.1:8000";

const authUser = {
  id: "user-e2e",
  email: "simo@test.ma",
  created_at: "2026-05-28T21:00:00.000Z",
};

const conversationsEmpty = [
  {
    id: "conv-empty-1",
    user_id: "user-e2e",
    title: "Accueil assistant",
    created_at: "2026-05-28T20:00:00.000Z",
    updated_at: "2026-05-28T20:00:00.000Z",
  },
];

const conversationsClinical = [
  {
    id: "conv-med-1",
    user_id: "user-e2e",
    title: "Bilan thyroïdien – report 16",
    created_at: "2026-05-28T20:15:00.000Z",
    updated_at: "2026-05-28T20:29:00.000Z",
  },
];

const messagesClinical = [
  {
    id: "m-user-1",
    conversation_id: "conv-med-1",
    role: "user",
    content: "Quels résultats sont anormaux dans report 16 ?",
    created_at: "2026-05-28T20:16:00.000Z",
    sources: [],
    diagnostics: null,
  },
  {
    id: "m-assistant-1",
    conversation_id: "conv-med-1",
    role: "assistant",
    content:
      "Anormaux : TSHus (au-dessus). Résultats dans la référence : T3 Libre, T4 Libre.\n\n| Analyse | Valeur | Référence | Statut |\n|---|---|---|---|\n| TSHus | 55.00 mUI/L | 0.35 - 4.94 | Au-dessus |\n| T3 Libre | 3.18 pg/mL | 2.04 - 4.40 | Dans la référence |\n| T4 Libre | 0.87 ng/dL | 0.93 - 1.70 | En dessous |\n\nConclusion technique : synthèse descriptive sans diagnostic.",
    created_at: "2026-05-28T20:17:00.000Z",
    sources: [
      {
        doc_id: "report_16",
        filename: "report_16.pdf",
        page: 1,
        row: 5,
        label: "report_16.pdf · page 1",
        viewer_url: "/viewer/pdf?doc_id=report_16&page=1",
        score: 0.92,
      },
      {
        doc_id: "report_16",
        filename: "report_16.pdf",
        page: 1,
        row: 7,
        label: "report_16.pdf · page 1",
        viewer_url: "/viewer/pdf?doc_id=report_16&page=1",
        score: 0.88,
      },
    ],
    diagnostics: {
      validation_status: "pass",
      displayed_evidences_count: 2,
      candidate_evidences_count: 5,
      missing_values_count: 1,
      quality_report: {
        safety_score: 0.92,
        source_ux_score: 0.91,
      },
      answer_type: "medical_structured",
      response_time: 1.8,
    },
  },
];

const documentsMock = [
  { id: "report_16", name: "report_16.pdf" },
  { id: "report_29", name: "report_29.pdf" },
];

const corsHeaders = {
  "access-control-allow-origin": "*",
  "access-control-allow-headers": "*",
  "access-control-allow-methods": "GET,POST,PUT,DELETE,OPTIONS",
};

function jsonResponse(route: Route, payload: unknown, status = 200) {
  return route.fulfill({
    status,
    contentType: "application/json",
    headers: corsHeaders,
    body: JSON.stringify(payload),
  });
}

function conversationMessages(pathname: string) {
  const match = pathname.match(/^\/conversations\/([^/]+)\/messages$/);
  if (!match) return null;
  const id = match[1];
  if (id === "conv-med-1") return messagesClinical;
  return [];
}

export async function setClientThemeAndAuth(page: Page, options: { theme: ThemeName; authenticated: boolean }) {
  await page.addInitScript(({ theme, authenticated }) => {
    localStorage.setItem("theme", theme);
    localStorage.setItem("clinical-theme", theme);
    if (authenticated) {
      localStorage.setItem("clinical-access-token", "e2e-token");
    } else {
      localStorage.removeItem("clinical-access-token");
    }
    document.documentElement.classList.toggle("dark", theme === "dark");
    document.documentElement.style.colorScheme = theme;
  }, options);
}

export async function mockRagApi(
  page: Page,
  options: {
    scenario: ScenarioName;
    authenticated: boolean;
    backendOnline?: boolean;
  },
) {
  const { scenario, authenticated, backendOnline = true } = options;

  await page.route(`${API_BASE}/**`, async (route) => {
    const request = route.request();
    const method = request.method();
    const url = new URL(request.url());
    const pathname = url.pathname;

    if (method === "OPTIONS") {
      return route.fulfill({ status: 204, headers: corsHeaders, body: "" });
    }

    if (pathname === "/health" && method === "GET") {
      if (!backendOnline) return jsonResponse(route, { status: "offline" }, 503);
      return jsonResponse(route, { status: "ok", service: "rag-api" });
    }

    if (pathname === "/auth/me" && method === "GET") {
      if (!authenticated) return jsonResponse(route, { detail: "Not authenticated" }, 401);
      return jsonResponse(route, authUser);
    }

    if (pathname === "/conversations" && method === "GET") {
      if (!authenticated) return jsonResponse(route, { detail: "Not authenticated" }, 401);
      return jsonResponse(route, scenario === "clinical" ? conversationsClinical : conversationsEmpty);
    }

    if (pathname.match(/^\/conversations\/[^/]+\/messages$/) && method === "GET") {
      if (!authenticated) return jsonResponse(route, { detail: "Not authenticated" }, 401);
      return jsonResponse(route, conversationMessages(pathname) || []);
    }

    if (pathname === "/api/models/active" && method === "GET") {
      return jsonResponse(route, {
        provider: "ollama",
        model: "llama3.2:latest",
        context_window: 8192,
        max_output_tokens: 2048,
        recommended_rag_budget: 5000,
      });
    }

    if (pathname.match(/^\/api\/conversations\/[^/]+\/context-usage$/) && method === "GET") {
      return jsonResponse(route, {
        conversation_id: "conv-med-1",
        model: "llama3.2:latest",
        context_window: 8192,
        used_tokens: 1128,
        remaining_tokens: 7064,
        usage_percent: 13.8,
        status: "safe",
      });
    }

    if (pathname === "/documents" && method === "GET") {
      return jsonResponse(route, documentsMock);
    }

    if (pathname.match(/^\/documents\/[^/]+\/reindex$/) && method === "POST") {
      return jsonResponse(route, { success: true });
    }

    if (pathname.match(/^\/documents\/[^/]+$/) && method === "DELETE") {
      return jsonResponse(route, { success: true });
    }

    if (pathname === "/chat" && method === "POST") {
      return jsonResponse(route, {
        conversation_id: "conv-med-1",
        answer: "Réponse simulée pour test e2e.",
        sources: [],
      });
    }

    return jsonResponse(route, { detail: `No mock for ${method} ${pathname}` }, 404);
  });
}

export async function waitForStableUI(page: Page, selector = "main") {
  await page.waitForSelector(selector, { state: "visible" });
  await page.waitForLoadState("domcontentloaded");
  await page.waitForTimeout(350);
}

