import type { SourceReference } from "@/types/source-reference";

export type PdfTextItem = {
  str: string;
  x: number;
  y: number;
  width: number;
  height: number;
};

export type PdfLine = {
  lineNumber: number;
  items: PdfTextItem[];
};

export type HighlightRect = {
  x: number;
  y: number;
  width: number;
  height: number;
};

export type PageHighlight = {
  pageNumber: number;
  rects: HighlightRect[];
};

export function buildLineRange(source: SourceReference): { start?: number; end?: number } {
  return {
    start: source.lineStart,
    end: source.lineEnd ?? source.lineStart,
  };
}

export function mapLinesToHighlightRects(lines: PdfLine[], source: SourceReference): HighlightRect[] {
  const { start, end } = buildLineRange(source);
  if (!Number.isFinite(start || NaN)) return [];
  const safeEnd = Number.isFinite(end || NaN) ? Number(end) : Number(start);

  const target = lines.filter((line) => line.lineNumber >= Number(start) && line.lineNumber <= safeEnd);
  return target.flatMap((line) =>
    line.items.map((item) => ({
      x: item.x,
      y: item.y,
      width: item.width,
      height: item.height,
    })),
  );
}

