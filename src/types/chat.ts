export type MessageRole = "user" | "assistant" | "system";
export type MessageStatus = "idle" | "loading" | "error" | "done";
export type ChatMode = "general" | "document_analysis" | "comparison" | "summary";

export interface SourceItem {
  id: string;
  documentName: string;
  documentId?: string;
  page?: number;
  section?: string;
  excerpt?: string;
  score?: number;
  type?: string;
  date?: string;
  warning?: string;
}

export interface MessageItem {
  id: string;
  chatId: string;
  role: MessageRole;
  content: string;
  createdAt: string;
  status: MessageStatus;
  sources?: SourceItem[];
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
  sources?: SourceItem[];
  confidence?: number;
  document_ids?: string[];
  response_time?: number;
}
