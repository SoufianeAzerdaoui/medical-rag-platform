export type MessageRole = "user" | "assistant" | "system";
export type MessageStatus = "idle" | "loading" | "error" | "done";
export type ChatMode = "general" | "document_analysis" | "comparison" | "summary";
export type AnswerType = "medical_structured" | "conversational" | "general_markdown";

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
export type InventoryViewType = "patient_cards" | "report_accordion" | "filterable_table" | "document_timeline";
export type InventoryView = { type: InventoryViewType };

export interface VisualizationDatum {
  analyte?: string;
  value?: number | string | null;
  raw_value?: string | null;
  value_numeric?: number | null;
  reference_deviation?: number | null;
  deviation_label?: string | null;
  metric_available?: boolean;
  lower_bound?: number | null;
  upper_bound?: number | null;
  status_code?: string;
  reference_ratio?: number | null;
  unit?: string;
  reference?: string;
  status?: string;
  source_label?: string;
}

export interface VisualizationPayload {
  requested?: boolean;
  requested_type?: "radar" | "bar" | "line" | "scatter" | "heatmap" | "unknown" | string;
  requested_label?: string;
  rendered_type?: "radar" | "bar" | "line" | "table" | null | string;
  rendered_label?: string;
  suitable?: boolean;
  fallback_used?: boolean;
  fallback_reason?: string | null;
  type?: string;
  title?: string;
  source?: string;
  supported?: boolean;
  recommended_type?: string;
  recommendation_reason?: string | null;
  metric_label?: string;
  metric_reason?: string;
  result_count?: number;
  calculable_count?: number;
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
  provider?: string | null;
  model?: string | null;
  response_time?: number;
  llm_provider_effective_runtime?: string | null;
  llm_model_effective_runtime?: string | null;
  summary_style_requested?: "short" | "editorial" | null;
  intent?: string | null;
  selected_route?: string | null;
  route_reason?: string | null;
  technical_condition?: string | null;
  requested_doc_ids?: string[] | null;
  requested_analytes?: string[] | null;
  answerability_status?: string | null;
  fallback_kind?: string | null;
  llm_route_class?: string | null;
  llm_writer_attempted?: boolean | null;
  llm_writer_accepted?: boolean | null;
  llm_quality_escalation_used?: boolean | null;
  llm_quality_escalation_reason?: string | null;
  final_answer_source?: "llm_writer" | "deterministic_renderer" | null;
  renderer_used?: string | null;
  fallback_reason?: string | null;
  llm_skipped_reason?: string | null;
  generation_mode_before_fallback?: string | null;
  fallback_decision_path?: string | null;
  answer_type?: AnswerType | string | null;
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
  patients?: Array<Record<string, unknown>>;
  inventory_view?: InventoryView;
  diagnostics?: AssistantDiagnostics;
  attachments?: string[];
  audio?: { mimeType: string; blobUrl: string };
}

export type SummaryStyle = "short" | "editorial";

export interface ChatItem {
  id: string;
  conversationId: string;
  title: string;
  subtitle?: string;
  sourceCount?: number;
  lastMessagePreview?: string;
  titleSource?: "auto" | "manual";
  titleGenerated?: boolean;
  titleEditedByUser?: boolean;
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
  conversation_id: string;
  answer: string;
  sources?: ChatSource[];
  confidence?: number;
  document_ids?: string[];
  response_time?: number;
  quality_report?: QualityReport;
  validation_status?: "pass" | "warning" | "fail";
  generation_mode?: string;
  generation_writer?: "llm_writer" | "professional_fallback";
  provider?: string | null;
  model?: string | null;
  llm_provider_effective_runtime?: string | null;
  llm_model_effective_runtime?: string | null;
  llm_writer_attempted?: boolean | null;
  llm_writer_accepted?: boolean | null;
  llm_quality_escalation_used?: boolean | null;
  llm_quality_escalation_reason?: string | null;
  final_answer_source?: "llm_writer" | "deterministic_renderer" | null;
  renderer_used?: string | null;
  fallback_reason?: string | null;
  visualization?: VisualizationPayload;
  chart_data?: VisualizationPayload;
  patients?: Array<Record<string, unknown>>;
  inventory_view?: InventoryView;
  intent?: string | null;
  selected_route?: string | null;
  route_reason?: string | null;
  technical_condition?: string | null;
  requested_doc_ids?: string[] | null;
  requested_analytes?: string[] | null;
  answerability_status?: string | null;
  fallback_kind?: string | null;
  answer_type?: AnswerType | string | null;
  debug?: Record<string, unknown> | null;
}
