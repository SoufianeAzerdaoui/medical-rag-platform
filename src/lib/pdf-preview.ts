import type { SourceReference } from "@/types/source-reference";

function normalizeApiBase(base: string): string {
  const trimmed = (base || "").trim();
  if (!trimmed) return "";
  return trimmed.endsWith("/") ? trimmed.slice(0, -1) : trimmed;
}

export function buildBackendPdfUrl(docId: string, pageNumber?: number): string {
  const encoded = encodeURIComponent(docId);
  const query = Number.isFinite(pageNumber || NaN) ? `?page=${pageNumber}` : "";
  const apiBase = normalizeApiBase(process.env.NEXT_PUBLIC_RAG_API_URL || "");
  const relative = `/api/documents/${encoded}/pdf${query}`;
  return apiBase ? `${apiBase}${relative}` : relative;
}

export function buildViewerPreviewUrl(source: SourceReference): string | null {
  if (!source.documentName) return null;
  const params = new URLSearchParams();
  params.set("doc_id", source.documentName);
  if (Number.isFinite(source.pageNumber || NaN)) params.set("page", String(source.pageNumber));
  if (Number.isFinite(source.lineStart || NaN)) params.set("row", String(source.lineStart));
  if (Number.isFinite(source.lineEnd || NaN)) params.set("row_end", String(source.lineEnd));
  return `/viewer/pdf?${params.toString()}`;
}

function parseViewerDocParams(urlValue: string): { docId: string; pageNumber?: number } | null {
  try {
    const parsed = new URL(urlValue, "http://localhost");
    if (!parsed.pathname.includes("/viewer/pdf")) return null;
    const docId = parsed.searchParams.get("doc_id") || "";
    if (!docId) return null;
    const pageRaw = parsed.searchParams.get("page");
    const pageNumber = pageRaw ? Number(pageRaw) : undefined;
    return {
      docId,
      pageNumber: Number.isFinite(pageNumber || NaN) ? pageNumber : undefined,
    };
  } catch {
    return null;
  }
}

export function resolvePdfDocumentUrl(source: SourceReference): string | null {
  const rawUrl = (source.documentUrl || "").trim();
  if (rawUrl) {
    if (rawUrl.includes("/viewer/pdf")) {
      const viewer = parseViewerDocParams(rawUrl);
      if (viewer?.docId) {
        return buildBackendPdfUrl(viewer.docId, source.pageNumber ?? viewer.pageNumber);
      }
    }
    if (/\.pdf(\?|#|$)/i.test(rawUrl) || rawUrl.includes("/api/documents/")) {
      return rawUrl;
    }
  }
  if (source.documentName) {
    return buildBackendPdfUrl(source.documentName, source.pageNumber);
  }
  return null;
}

export function withPdfPageAnchor(documentUrl: string, pageNumber?: number): string {
  if (!documentUrl) return "";
  if (!Number.isFinite(pageNumber || NaN)) return documentUrl;
  const safeUrl = documentUrl.split("#")[0];
  return `${safeUrl}#page=${pageNumber}`;
}
