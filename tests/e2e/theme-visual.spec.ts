import { expect, test } from "@playwright/test";
import { mockRagApi, setClientThemeAndAuth, waitForStableUI, type ScenarioName, type ThemeName } from "./helpers/mock-rag-api";

type VisualCase = {
  name: string;
  path: string;
  scenario: ScenarioName;
  authenticated: boolean;
  backendOnline?: boolean;
  readySelector?: string;
};

const visualCases: VisualCase[] = [
  {
    name: "login-screen",
    path: "/chat",
    scenario: "empty",
    authenticated: false,
    readySelector: "form",
  },
  {
    name: "chat-empty-state",
    path: "/chat",
    scenario: "empty",
    authenticated: true,
    readySelector: "main",
  },
  {
    name: "chat-with-sources",
    path: "/chat/conv-med-1",
    scenario: "clinical",
    authenticated: true,
    readySelector: "main",
  },
  {
    name: "documents-page",
    path: "/documents",
    scenario: "clinical",
    authenticated: true,
    readySelector: "main",
  },
  {
    name: "dashboard-page",
    path: "/dashboard",
    scenario: "clinical",
    authenticated: true,
    readySelector: "main",
  },
  {
    name: "settings-page",
    path: "/settings",
    scenario: "clinical",
    authenticated: true,
    readySelector: "main",
  },
];

for (const theme of ["light", "dark"] as const) {
  test.describe(`visual-${theme}`, () => {
    for (const visualCase of visualCases) {
      test(`${visualCase.name}`, async ({ page }, testInfo) => {
        await setClientThemeAndAuth(page, {
          theme: theme as ThemeName,
          authenticated: visualCase.authenticated,
        });
        await mockRagApi(page, {
          scenario: visualCase.scenario,
          authenticated: visualCase.authenticated,
          backendOnline: visualCase.backendOnline ?? true,
        });

        await page.goto(visualCase.path);
        await waitForStableUI(page, visualCase.readySelector || "main");

        if (visualCase.name === "chat-with-sources") {
          await expect(page.getByText("Sources cliquables")).toBeVisible();
        }

        const screenshotPath = testInfo.outputPath(`${theme}-${visualCase.name}.png`);
        await page.screenshot({
          path: screenshotPath,
          fullPage: true,
          animations: "disabled",
        });
        await testInfo.attach(`${theme}-${visualCase.name}`, {
          path: screenshotPath,
          contentType: "image/png",
        });
      });
    }
  });
}

