"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { mapLinesToHighlightRects, type PdfLine, type PdfTextItem } from "@/lib/pdfjs/highlight-mapping";
import type { SourceReference } from "@/types/source-reference";

type PdfJsRendererProps = {
  source: SourceReference;
  src: string | null;
  onFatalError?: () => void;
};

type PdfJsModule = {
  GlobalWorkerOptions: { workerSrc: string };
  Util: { transform: (m1: number[], m2: number[]) => number[] };
  getDocument: (source: string) => PdfLoadingTask;
};

type PdfLoadingTask = {
  promise: Promise<any>;
  destroy?: () => void;
};

function toPdfLines(items: PdfTextItem[]): PdfLine[] {
  const sorted = [...items].sort((a, b) => (a.y === b.y ? a.x - b.x : a.y - b.y));
  const lines: PdfLine[] = [];
  const tolerance = 4;

  for (const item of sorted) {
    const last = lines[lines.length - 1];
    if (!last) {
      lines.push({ lineNumber: 1, items: [item] });
      continue;
    }
    const lastY = last.items.reduce((sum, it) => sum + it.y, 0) / last.items.length;
    if (Math.abs(item.y - lastY) <= Math.max(tolerance, item.height * 0.4)) {
      last.items.push(item);
      continue;
    }
    lines.push({ lineNumber: lines.length + 1, items: [item] });
  }

  return lines;
}

export function PdfJsRenderer({ source, src, onFatalError }: PdfJsRendererProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const textLayerRef = useRef<HTMLDivElement | null>(null);
  const overlayRef = useRef<HTMLDivElement | null>(null);
  const frameRef = useRef<HTMLDivElement | null>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [errorText, setErrorText] = useState<string | null>(null);

  const targetPage = useMemo(() => {
    if (Number.isFinite(source.pageNumber || NaN) && Number(source.pageNumber) > 0) return Number(source.pageNumber);
    return 1;
  }, [source.pageNumber]);

  useEffect(() => {
    if (!src) {
      setStatus("error");
      setErrorText("PDF ou page indisponible pour cette source.");
      return;
    }

    let alive = true;
    let loadingTask: PdfLoadingTask | null = null;
    const currentSrc = src;
    if (!currentSrc) return;

    async function renderPdf() {
      try {
        setStatus("loading");
        setErrorText(null);

        const mod = (await import("pdfjs-dist/legacy/build/pdf.mjs")) as unknown as PdfJsModule;
        mod.GlobalWorkerOptions.workerSrc = new URL("pdfjs-dist/legacy/build/pdf.worker.min.mjs", import.meta.url).toString();

        loadingTask = mod.getDocument(currentSrc);
        const pdf = await loadingTask.promise;
        if (!alive) return;

        const safePage = Math.min(Math.max(targetPage, 1), pdf.numPages || targetPage);
        const page = await pdf.getPage(safePage);
        if (!alive) return;

        const host = frameRef.current;
        const canvas = canvasRef.current;
        const textLayer = textLayerRef.current;
        const overlay = overlayRef.current;
        if (!host || !canvas || !textLayer || !overlay) return;

        const baseViewport = page.getViewport({ scale: 1 });
        const containerWidth = Math.max(host.clientWidth - 2, 320);
        const scale = containerWidth / baseViewport.width;
        const viewport = page.getViewport({ scale });
        const outputScale = window.devicePixelRatio || 1;

        const ctx = canvas.getContext("2d", { alpha: false });
        if (!ctx) throw new Error("Canvas context indisponible.");
        canvas.width = Math.floor(viewport.width * outputScale);
        canvas.height = Math.floor(viewport.height * outputScale);
        canvas.style.width = `${Math.floor(viewport.width)}px`;
        canvas.style.height = `${Math.floor(viewport.height)}px`;
        ctx.setTransform(outputScale, 0, 0, outputScale, 0, 0);

        textLayer.style.width = `${Math.floor(viewport.width)}px`;
        textLayer.style.height = `${Math.floor(viewport.height)}px`;
        overlay.style.width = `${Math.floor(viewport.width)}px`;
        overlay.style.height = `${Math.floor(viewport.height)}px`;

        await page.render({ canvasContext: ctx, viewport }).promise;
        if (!alive) return;

        const textContent = await page.getTextContent();
        if (!alive) return;

        textLayer.replaceChildren();
        overlay.replaceChildren();
        const textItems: PdfTextItem[] = [];

        for (const item of textContent.items as Array<any>) {
          if (!item || typeof item.str !== "string" || !item.str.trim()) continue;
          const tx = mod.Util.transform(viewport.transform as number[], item.transform as number[]);
          const fontHeight = Math.hypot(tx[2], tx[3]);
          const left = tx[4];
          const bottom = tx[5];
          const top = viewport.height - bottom - fontHeight;
          const width = (item.width || 0) * scale;

          textItems.push({
            str: item.str,
            x: left,
            y: top,
            width,
            height: fontHeight,
          });

          const span = document.createElement("span");
          span.textContent = item.str;
          span.style.position = "absolute";
          span.style.left = `${left}px`;
          span.style.top = `${top}px`;
          span.style.fontSize = `${Math.max(fontHeight, 1)}px`;
          span.style.lineHeight = "1";
          span.style.whiteSpace = "pre";
          span.style.opacity = "0";
          span.style.color = "transparent";
          span.style.pointerEvents = "auto";
          textLayer.appendChild(span);
        }

        const lines = toPdfLines(textItems);
        const rects = mapLinesToHighlightRects(lines, source);
        for (const rect of rects) {
          const node = document.createElement("div");
          node.style.position = "absolute";
          node.style.left = `${rect.x}px`;
          node.style.top = `${rect.y}px`;
          node.style.width = `${Math.max(rect.width, 2)}px`;
          node.style.height = `${Math.max(rect.height, 10)}px`;
          node.style.background = "rgba(251, 191, 36, 0.32)";
          node.style.border = "1px solid rgba(245, 158, 11, 0.45)";
          node.style.borderRadius = "2px";
          node.style.pointerEvents = "none";
          overlay.appendChild(node);
        }

        setStatus("ready");
      } catch (error) {
        if (!alive) return;
        const message = error instanceof Error ? error.message : "Aperçu pdf.js indisponible.";
        setStatus("error");
        setErrorText(message);
        onFatalError?.();
      }
    }

    void renderPdf();

    return () => {
      alive = false;
      if (loadingTask?.destroy) loadingTask.destroy();
    };
  }, [src, source, targetPage, onFatalError]);

  if (!src) {
    return (
      <div className="flex h-full items-center justify-center p-4 text-sm text-fg/70">
        PDF ou page indisponible pour cette source.
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      <div className="rounded-t-xl border-b border-border/70 bg-card/[0.72] px-3 py-2 text-xs text-fg/65">
        Aperçu pdf.js · page {source.pageNumber ?? 1} · lignes {source.lineStart ?? "?"}
        {source.lineEnd ? `-${source.lineEnd}` : ""}
      </div>
      <div ref={frameRef} className="relative h-full overflow-auto bg-muted/20">
        {status === "loading" ? (
          <div className="flex h-full items-center justify-center p-4 text-sm text-fg/70">Chargement du document…</div>
        ) : null}
        {status === "error" ? (
          <div className="flex h-full items-center justify-center p-4 text-sm text-fg/70">
            {errorText || "Aperçu pdf.js indisponible."}
          </div>
        ) : null}

        <div className={`relative mx-auto my-3 w-fit ${status === "ready" ? "block" : "hidden"}`}>
          <canvas ref={canvasRef} className="block shadow-sm" />
          <div ref={textLayerRef} className="absolute inset-0 overflow-hidden" />
          <div ref={overlayRef} className="absolute inset-0 overflow-hidden" />
        </div>
      </div>
    </div>
  );
}
