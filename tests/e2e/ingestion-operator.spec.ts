import { expect, test, type Page } from "@playwright/test";

const API_BASE = process.env.NEXT_PUBLIC_RAG_API_URL || "http://127.0.0.1:8000";

function json(body: unknown, status = 200) {
  return {
    status,
    contentType: "application/json",
    headers: {
      "access-control-allow-origin": "*",
      "access-control-allow-headers": "*",
      "access-control-allow-methods": "GET,POST,PUT,DELETE,OPTIONS",
    },
    body: JSON.stringify(body),
  };
}

async function setupAuth(page: Page) {
  await page.addInitScript(() => {
    localStorage.setItem("clinical-access-token", "e2e-token");
    localStorage.setItem("theme", "dark");
    document.documentElement.classList.add("dark");
  });
}

test.describe("ingestion-operator", () => {
  test("happy-path filters bulk export timeline + async ingestion", async ({ page }) => {
    await setupAuth(page);

    let pollCount = 0;
    await page.route(`${API_BASE}/**`, async (route) => {
      const req = route.request();
      const method = req.method();
      const url = new URL(req.url());
      const path = url.pathname;

      if (method === "OPTIONS") return route.fulfill({ status: 204, body: "" });
      if (path === "/health" && method === "GET") return route.fulfill(json({ status: "ok" }));
      if (path === "/auth/me" && method === "GET") {
        return route.fulfill(json({ id: "u1", email: "simo@test.ma", role: "admin", created_at: "2026-05-30T12:00:00Z" }));
      }
      if (path === "/conversations" && method === "GET") return route.fulfill(json([]));
      if (path === "/documents/resync-registry" && method === "POST") return route.fulfill(json({ success: true, discovered_count: 4, indexed_count: 2, duplicate_count: 2 }));
      if (path === "/documents/discover" && method === "GET") {
        return route.fulfill(
          json([
            {
              filename: "report (10).pdf",
              doc_id: "report_10",
              absolute_path: "/tmp/report (10).pdf",
              size_bytes: 1024,
              modified_at: "2026-05-30T10:00:00Z",
              file_hash: "a1",
              text_hash: "t1",
              already_indexed: true,
              is_duplicate: false,
              duplicate_with: [],
              blocked: false,
              duplicate_override: false,
              duplicate_entries: [],
            },
            {
              filename: "report (100).pdf",
              doc_id: "report_100",
              absolute_path: "/tmp/report (100).pdf",
              size_bytes: 2048,
              modified_at: "2026-05-30T10:02:00Z",
              file_hash: "dup-1",
              text_hash: "dup-t1",
              already_indexed: false,
              is_duplicate: true,
              duplicate_with: ["report (1).pdf"],
              duplicate_reason: "Doublon détecté mais autorisé par whitelist.",
              blocked: false,
              duplicate_override: true,
              override_reason: "validation",
              duplicate_entries: [],
            },
            {
              filename: "report (101).pdf",
              doc_id: "report_101",
              absolute_path: "/tmp/report (101).pdf",
              size_bytes: 1500,
              modified_at: "2026-05-30T10:03:00Z",
              file_hash: "dup-2",
              text_hash: "dup-t2",
              already_indexed: true,
              is_duplicate: true,
              duplicate_with: ["report (31).pdf"],
              duplicate_reason: "Doublon de contenu détecté avec un document déjà indexé.",
              blocked: true,
              duplicate_override: false,
              duplicate_entries: [],
            },
            {
              filename: "report (103).pdf",
              doc_id: "report_103",
              absolute_path: "/tmp/report (103).pdf",
              size_bytes: 1111,
              modified_at: "2026-05-30T10:04:00Z",
              file_hash: "n1",
              text_hash: "nt1",
              already_indexed: false,
              is_duplicate: false,
              duplicate_with: [],
              blocked: false,
              duplicate_override: false,
              duplicate_entries: [],
            },
          ]),
        );
      }
      if (path === "/documents/duplicates/override" && method === "POST") {
        return route.fulfill(json({ success: true, filename: "report (100).pdf", enabled: true, reason: "bulk", updated_by: "simo@test.ma", updated_at: "2026-05-30T11:00:00Z" }));
      }
      if (path === "/documents/ingestion-report" && method === "GET") {
        if (url.searchParams.get("format") === "pdf") {
          return route.fulfill({ status: 200, contentType: "application/pdf", body: "%PDF-1.4\n%%EOF\n" });
        }
        return route.fulfill({ status: 200, contentType: "text/csv", body: "filename,doc_id\nreport (10).pdf,report_10\n" });
      }
      if (path === "/documents/timeline" && method === "GET") {
        return route.fulfill(json({ filename: "report (100).pdf", events: [{ at: "2026-05-30T09:00:00Z", type: "discovered", title: "Document détecté", detail: "report (100).pdf" }] }));
      }
      if (path === "/upload/from-docs/jobs" && method === "POST") {
        return route.fulfill(json({ job_id: "job-e2e-1", status: "queued", created_at: "2026-05-30T11:00:00Z", message: "queued" }));
      }
      if (path === "/upload/jobs/job-e2e-1" && method === "GET") {
        pollCount += 1;
        if (pollCount < 2) {
          return route.fulfill(json({ job_id: "job-e2e-1", status: "running", created_at: "2026-05-30T11:00:00Z", progress_percent: 45, message: "running" }));
        }
        return route.fulfill(
          json({
            job_id: "job-e2e-1",
            status: "success",
            created_at: "2026-05-30T11:00:00Z",
            started_at: "2026-05-30T11:00:01Z",
            finished_at: "2026-05-30T11:00:08Z",
            progress_percent: 100,
            result: {
              success: true,
              ingested_count: 1,
              ingested: [{ filename: "report (103).pdf", doc_id: "report_103", stored_path: "/tmp/report (103).pdf", extraction_dir: "/tmp/report_103" }],
              skipped: [],
            },
          }),
        );
      }
      if (path === "/chat" && method === "POST") {
        return route.fulfill(json({ conversation_id: "conv-x", answer: "ok", sources: [] }));
      }
      return route.fulfill(json({ detail: `No mock for ${method} ${path}` }, 404));
    });

    await page.goto("/documents/upload");
    await expect(page.getByText("Automatisation depuis le dossier docs/")).toBeVisible();

    await page.getByRole("button", { name: "Doublons" }).click();
    await expect(page.getByText("report (100).pdf")).toBeVisible();

    await page.getByRole("checkbox").first().check();
    await page.getByRole("button", { name: "Whitelist sélection" }).click();

    const csvDownload = page.waitForEvent("download");
    await page.getByRole("button", { name: "Export CSV" }).click();
    await (await csvDownload).failure().catch(() => null);

    await page.getByRole("button", { name: "Timeline" }).first().click();
    await expect(page.getByText("Timeline d’évènements document")).toBeVisible();
    await expect(page.getByText("Document détecté")).toBeVisible();
    await page.getByLabel("Fermer le détail doublon").click();

    await page.getByRole("button", { name: "Valider et lancer pipeline" }).click();
    await expect(page.getByText("Documents prêts pour interrogation")).toBeVisible();
  });

  test("error-path shows backend validation error", async ({ page }) => {
    await setupAuth(page);
    await page.route(`${API_BASE}/**`, async (route) => {
      const req = route.request();
      const method = req.method();
      const url = new URL(req.url());
      const path = url.pathname;
      if (method === "OPTIONS") return route.fulfill({ status: 204, body: "" });
      if (path === "/health" && method === "GET") return route.fulfill(json({ status: "ok" }));
      if (path === "/auth/me" && method === "GET") return route.fulfill(json({ id: "u1", email: "simo@test.ma", role: "admin", created_at: "2026-05-30T12:00:00Z" }));
      if (path === "/conversations" && method === "GET") return route.fulfill(json([]));
      if (path === "/documents/resync-registry" && method === "POST") return route.fulfill(json({ success: true, discovered_count: 1, indexed_count: 0, duplicate_count: 0 }));
      if (path === "/documents/discover" && method === "GET") {
        return route.fulfill(
          json([
            {
              filename: "report (200).pdf",
              doc_id: "report_200",
              absolute_path: "/tmp/report (200).pdf",
              size_bytes: 1200,
              modified_at: "2026-05-30T10:05:00Z",
              file_hash: "n2",
              text_hash: "nt2",
              already_indexed: false,
              is_duplicate: false,
              duplicate_with: [],
              blocked: false,
              duplicate_override: false,
              duplicate_entries: [],
            },
          ]),
        );
      }
      if (path === "/upload/from-docs/jobs" && method === "POST") {
        return route.fulfill(json({ detail: "[BLOCKED] Validation errors: 1" }, 400));
      }
      return route.fulfill(json({ detail: `No mock for ${method} ${path}` }, 404));
    });

    await page.goto("/documents/upload");
    await page.getByRole("checkbox").first().check();
    await page.getByRole("button", { name: "Valider et lancer pipeline" }).click();
    await expect(page.getByText("[BLOCKED] Validation errors: 1")).toBeVisible();
  });
});
