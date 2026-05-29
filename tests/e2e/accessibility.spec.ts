import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";
import { mockRagApi, setClientThemeAndAuth, type ThemeName } from "./helpers/mock-rag-api";

type A11yCase = {
  name: string;
  path: string;
  authenticated: boolean;
  scenario: "empty" | "clinical";
  readySelector?: string;
};

const pages: A11yCase[] = [
  { name: "login", path: "/chat", authenticated: false, scenario: "empty", readySelector: "form" },
  { name: "chat-empty", path: "/chat", authenticated: true, scenario: "empty" },
  { name: "chat-clinical", path: "/chat/conv-med-1", authenticated: true, scenario: "clinical" },
  { name: "documents", path: "/documents", authenticated: true, scenario: "clinical" },
  { name: "dashboard", path: "/dashboard", authenticated: true, scenario: "clinical" },
  { name: "settings", path: "/settings", authenticated: true, scenario: "clinical" },
];

for (const theme of ["light", "dark"] as const) {
  test.describe(`a11y-${theme}`, () => {
    for (const item of pages) {
      test(`${item.name}`, async ({ page }) => {
        await setClientThemeAndAuth(page, { theme: theme as ThemeName, authenticated: item.authenticated });
        await mockRagApi(page, {
          scenario: item.scenario,
          authenticated: item.authenticated,
          backendOnline: true,
        });

        await page.goto(item.path, { waitUntil: "domcontentloaded" });
        await expect(page.locator("body")).toBeVisible();
        await page.waitForTimeout(800);

        const axeResults = await new AxeBuilder({ page })
          .exclude("iframe")
          .analyze();

        const criticalViolations = axeResults.violations.filter((violation) => violation.impact === "critical");
        const seriousViolations = axeResults.violations.filter((violation) => violation.impact === "serious");

        await test.info().attach(`${theme}-${item.name}-a11y-report`, {
          body: Buffer.from(
            JSON.stringify(
              {
                path: item.path,
                theme,
                criticalCount: criticalViolations.length,
                seriousCount: seriousViolations.length,
                violations: axeResults.violations,
              },
              null,
              2,
            ),
          ),
          contentType: "application/json",
        });

        expect(criticalViolations, `Critical accessibility violations on ${item.path} (${theme})`).toEqual([]);
        if (process.env.A11Y_STRICT === "1") {
          expect(seriousViolations, `Serious accessibility violations on ${item.path} (${theme})`).toEqual([]);
        }
      });
    }
  });
}
