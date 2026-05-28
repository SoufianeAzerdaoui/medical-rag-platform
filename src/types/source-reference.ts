export type SourceReference = {
  id?: string;
  documentName: string;
  documentUrl?: string;
  pageNumber?: number;
  lineStart?: number;
  lineEnd?: number;
  snippet?: string;
  score?: number;
};
