function asString(value: string | string[] | undefined): string {
  if (Array.isArray(value)) return value[0] || "";
  return value || "";
}

export const dynamic = "force-dynamic";

function normalizeApiBase(base: string): string {
  const trimmed = (base || "").trim();
  if (!trimmed) return "";
  return trimmed.endsWith("/") ? trimmed.slice(0, -1) : trimmed;
}

function buildPdfHref(docId: string, page: number | null, version: string): string {
  const encoded = encodeURIComponent(docId);
  const params = new URLSearchParams();
  if (page && Number.isFinite(page)) {
    params.set("page", String(page));
  }
  if (version) {
    params.set("v", version);
  }
  const apiBase = normalizeApiBase(process.env.NEXT_PUBLIC_RAG_API_URL || "");
  const suffix = params.toString() ? `?${params.toString()}` : "";
  const relative = `/api/documents/${encoded}/pdf${suffix}`;
  return apiBase ? `${apiBase}${relative}` : relative;
}

export default async function PdfViewerPage({
  searchParams,
}: {
  searchParams: Promise<{
    doc_id?: string | string[];
    page?: string | string[];
    row?: string | string[];
    row_end?: string | string[];
    v?: string | string[];
  }>;
}) {
  const resolvedSearchParams = await searchParams;
  const docId = asString(resolvedSearchParams.doc_id).trim();
  const pageRaw = asString(resolvedSearchParams.page).trim();
  const rowRaw = asString(resolvedSearchParams.row).trim();
  const rowEndRaw = asString(resolvedSearchParams.row_end).trim();
  const versionRaw = asString(resolvedSearchParams.v).trim();
  const page = pageRaw ? Number(pageRaw) : null;
  const row = rowRaw ? Number(rowRaw) : null;
  const rowEnd = rowEndRaw ? Number(rowEndRaw) : null;

  if (!docId) {
    return (
      <main className="mx-auto max-w-4xl p-6">
        <h1 className="text-lg font-semibold">PDF Viewer</h1>
        <p className="mt-3 text-sm text-fg/70">Le paramètre doc_id est requis.</p>
      </main>
    );
  }

  const cacheVersion = versionRaw || String(Date.now());
  const pdfHref = buildPdfHref(docId, Number.isFinite(page || NaN) ? page : null, cacheVersion);
  const title = `Source ${docId}${page ? ` — page ${page}` : ""}`;
  const hasRowFocus = Number.isFinite(row || NaN);
  const lineFocusText = hasRowFocus
    ? (Number.isFinite(rowEnd || NaN) && Number(rowEnd) > Number(row)
      ? `Zone surlignée · lignes ${row}-${rowEnd}`
      : `Zone surlignée · ligne ${row}`)
    : null;

  return (
    <main className="flex min-h-dvh flex-col gap-3 p-3 md:p-4">
      <div className="flex flex-col gap-3 rounded-lg border border-border bg-card/40 px-3 py-3 sm:flex-row sm:items-center sm:justify-between sm:px-3 sm:py-2">
        <div className="min-w-0">
          <div className="text-sm font-medium">{title}</div>
          {lineFocusText ? <div className="mt-0.5 text-xs font-medium text-accent">{lineFocusText}</div> : null}
        </div>
        <div className="flex flex-wrap items-center gap-3">
          {lineFocusText ? (
            <span className="rounded-full border border-accent/30 bg-accent/10 px-2.5 py-1 text-xs font-medium text-accent">
              {lineFocusText}
            </span>
          ) : null}
          <a href={pdfHref} target="_blank" rel="noopener noreferrer" className="text-sm text-accent hover:underline">
            Ouvrir dans un nouvel onglet
          </a>
        </div>
      </div>
      <iframe title={title} src={pdfHref} className="min-h-[70dvh] flex-1 w-full rounded-lg border border-border bg-background" />
    </main>
  );
}
