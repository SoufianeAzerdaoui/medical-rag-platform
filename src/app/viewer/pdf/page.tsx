function asString(value: string | string[] | undefined): string {
  if (Array.isArray(value)) return value[0] || "";
  return value || "";
}

function normalizeApiBase(base: string): string {
  const trimmed = (base || "").trim();
  if (!trimmed) return "";
  return trimmed.endsWith("/") ? trimmed.slice(0, -1) : trimmed;
}

function buildPdfHref(docId: string, page: number | null): string {
  const encoded = encodeURIComponent(docId);
  const suffix = page && Number.isFinite(page) ? `?page=${page}` : "";
  const apiBase = normalizeApiBase(process.env.NEXT_PUBLIC_RAG_API_URL || "");
  const relative = `/api/documents/${encoded}/pdf${suffix}`;
  return apiBase ? `${apiBase}${relative}` : relative;
}

export default async function PdfViewerPage({
  searchParams,
}: {
  searchParams: Promise<{ doc_id?: string | string[]; page?: string | string[] }>;
}) {
  const resolvedSearchParams = await searchParams;
  const docId = asString(resolvedSearchParams.doc_id).trim();
  const pageRaw = asString(resolvedSearchParams.page).trim();
  const page = pageRaw ? Number(pageRaw) : null;

  if (!docId) {
    return (
      <main className="mx-auto max-w-4xl p-6">
        <h1 className="text-lg font-semibold">PDF Viewer</h1>
        <p className="mt-3 text-sm text-fg/70">Le paramètre doc_id est requis.</p>
      </main>
    );
  }

  const pdfHref = buildPdfHref(docId, Number.isFinite(page || NaN) ? page : null);
  const title = `Source ${docId}${page ? ` — page ${page}` : ""}`;

  return (
    <main className="h-screen p-3 md:p-4">
      <div className="mb-3 flex items-center justify-between rounded-lg border border-border bg-card/40 px-3 py-2">
        <div className="text-sm font-medium">{title}</div>
        <a href={pdfHref} target="_blank" rel="noopener noreferrer" className="text-sm text-accent hover:underline">
          Ouvrir dans un nouvel onglet
        </a>
      </div>
      <iframe title={title} src={pdfHref} className="h-[calc(100vh-4.5rem)] w-full rounded-lg border border-border bg-background" />
    </main>
  );
}
