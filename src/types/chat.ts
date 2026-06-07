export type MessageRole = "user" | "assistant" | "system";
export type MessageStatus = "idle" | "loading" | "error" | "done";
export type ChatMode = "general" | "document_analysis" | "comparison" | "summary";
export type AnswerType = "medical_structured" | "conversational" | "general_markdown";

export type SourceCitation = {
  doc_id: string;
  filename?: string | null;
  page?: number | null;
  row?: number | null;
  line?: number | null;
  label: string;
  url?: string | null;
  viewer_url?: string | null;
};

export type LegacySourceItem = {
  id?: string;
  documentName?: string;
  documentId?: string;
  page?: number;
  line?: number;
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
  final_answer_source?: "llm_writer" | "llm_writer_repaired" | "deterministic_renderer" | null;
  renderer_used?: string | null;
  fallback_reason?: string | null;
  llm_candidate_answer?: string | null;
  llm_candidate_validation_status?: string | null;
  llm_candidate_validation_errors?: string[] | null;
  llm_candidate_validation_warnings?: string[] | null;
  llm_candidate_rejected_reason?: string | null;
  llm_candidate_contract_errors?: string[] | null;
  llm_repair_attempted?: boolean | null;
  llm_repair_status?: string | null;
  llm_repair_validation_errors?: string[] | null;
  llm_repair_truncation_detected?: boolean | null;
  llm_repaired_answer?: string | null;
  llm_quality_gate?: Record<string, unknown> | null;
  final_answer_quality_gate?: Record<string, unknown> | null;
  quality_final_status?: "pass" | "warning" | "fail" | null;
  synthesis_quality_reason?: string | null;
  displayed_evidences_count?: number | null;
  evidence_pack_count?: number | null;
  lab_result_count?: number | null;
  value_numeric_count?: number | null;
  structured_values_count?: number | null;
  sources_count?: number | null;
  above_reference_count?: number | null;
  below_reference_count?: number | null;
  within_reference_count?: number | null;
  needs_clinical_context_count?: number | null;
  major_anomalies_count?: number | null;
  selected_normal_results_count?: number | null;
  requested_doc_id?: string | null;
  resolved_doc_id?: string | null;
  resolved_filename?: string | null;
  resolved_file_hash?: string | null;
  resolved_page_count?: number | null;
  indexed_page_count?: number | null;
  ingestion_timestamp?: string | null;
  source_pdf_path?: string | null;
  document_identity_mismatch?: boolean | null;
  document_identity_status?: string | null;
  document_identity_reasons?: string[] | null;
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
  final_answer_source?: "llm_writer" | "llm_writer_repaired" | "deterministic_renderer" | null;
  renderer_used?: string | null;
  fallback_reason?: string | null;
  llm_candidate_answer?: string | null;
  llm_candidate_validation_status?: string | null;
  llm_candidate_validation_errors?: string[] | null;
  llm_candidate_validation_warnings?: string[] | null;
  llm_candidate_rejected_reason?: string | null;
  llm_candidate_contract_errors?: string[] | null;
  llm_repair_attempted?: boolean | null;
  llm_repair_status?: string | null;
  llm_repair_validation_errors?: string[] | null;
  llm_repair_truncation_detected?: boolean | null;
  llm_repaired_answer?: string | null;
  llm_quality_gate?: Record<string, unknown> | null;
  final_answer_quality_gate?: Record<string, unknown> | null;
  quality_final_status?: "pass" | "warning" | "fail" | null;
  synthesis_quality_reason?: string | null;
  displayed_evidences_count?: number | null;
  evidence_pack_count?: number | null;
  lab_result_count?: number | null;
  value_numeric_count?: number | null;
  structured_values_count?: number | null;
  sources_count?: number | null;
  above_reference_count?: number | null;
  below_reference_count?: number | null;
  within_reference_count?: number | null;
  needs_clinical_context_count?: number | null;
  major_anomalies_count?: number | null;
  selected_normal_results_count?: number | null;
  requested_doc_id?: string | null;
  resolved_doc_id?: string | null;
  resolved_filename?: string | null;
  resolved_file_hash?: string | null;
  resolved_page_count?: number | null;
  indexed_page_count?: number | null;
  ingestion_timestamp?: string | null;
  source_pdf_path?: string | null;
  document_identity_mismatch?: boolean | null;
  document_identity_status?: string | null;
  document_identity_reasons?: string[] | null;
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
