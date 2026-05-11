export type MessageRole = "user" | "assistant" | "system";
export type MessageStatus = "idle" | "loading" | "error" | "done";
export type ChatMode = "general" | "document_analysis" | "comparison" | "summary";

export type SourceCitation = {
  doc_id: string;
  filename?: string | null;
  page?: number | null;
  row?: number | null;
  label: string;
  url?: string | null;
  viewer_url?: string | null;
};

export type LegacySourceItem = {
  id?: string;
  documentName?: string;
  documentId?: string;
  page?: number;
  section?: string;
  excerpt?: string;
  score?: number;
  type?: string;
  date?: string;
  warning?: string;
  url?: string;
  viewer_url?: string;
};

export type ChatSource = SourceCitation | LegacySourceItem | string;

export interface VisualizationDatum {
  analyte?: string;
  value?: number | string | null;
  value_numeric?: number | null;
  reference_ratio?: number | null;
  unit?: string;
  reference?: string;
  status?: string;
  source_label?: string;
}

export interface VisualizationPayload {
  requested?: boolean;
  type?: string;
  title?: string;
  source?: string;
  supported?: boolean;
  recommended_type?: string;
  reason?: string;
  x_field?: string;
  y_field?: string;
  data?: VisualizationDatum[];
}

export interface QualityReport {
  faithfulness_score: number;
  format_compliance_score: number;
  readability_score: number;
  source_ux_score: number;
  style_repetition_score: number;
  safety_score: number;
  final_status: "pass" | "warning" | "fail";
}

export interface AssistantDiagnostics {
  quality_report?: QualityReport;
  validation_status?: "pass" | "warning" | "fail";
  generation_mode?: string;
  generation_writer?: "llm_writer" | "professional_fallback";
  response_time?: number;
}

export interface MessageItem {
  id: string;
  chatId: string;
  role: MessageRole;
  content: string;
  createdAt: string;
  status: MessageStatus;
  sources?: ChatSource[];
  visualization?: VisualizationPayload;
  chart_data?: VisualizationPayload;
  diagnostics?: AssistantDiagnostics;
  attachments?: string[];
  audio?: { mimeType: string; blobUrl: string };
}

export interface ChatItem {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: MessageItem[];
  favorite: boolean;
  tags: string[];
  mode: ChatMode;
  documentIds: string[];
  summary?: string;
}

export interface RagResponse {
  answer: string;
  sources?: ChatSource[];
  confidence?: number;
  document_ids?: string[];
  response_time?: number;
  quality_report?: QualityReport;
  validation_status?: "pass" | "warning" | "fail";
  generation_mode?: string;
  generation_writer?: "llm_writer" | "professional_fallback";
  visualization?: VisualizationPayload;
  chart_data?: VisualizationPayload;
}
