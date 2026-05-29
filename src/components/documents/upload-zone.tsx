"use client";

import { UploadCloud } from "lucide-react";
import { useState } from "react";
import { ApiError, uploadDocumentsApi } from "@/services/rag-api";
import { useAuthStore } from "@/store/auth-store";

type UploadStatus = "ready" | "uploading" | "processing" | "indexed" | "error";

export function UploadZone({ onUploaded }: { onUploaded?: () => void }) {
  const [status, setStatus] = useState<UploadStatus>("ready");
  const [filename, setFilename] = useState("");
  const [message, setMessage] = useState("");
  const token = useAuthStore((s) => s.accessToken);

  async function onFile(file?: File) {
    if (!file) return;
    setFilename(file.name);
    setStatus("uploading");
    setMessage("Upload en cours…");
    try {
      setStatus("processing");
      setMessage("Ingestion pipeline: extraction, chunking, anonymisation, indexing…");
      const result = await uploadDocumentsApi([file], token);
      setStatus("indexed");
      setMessage(
        result.ingested_count > 0
          ? `Indexé: ${result.ingested[0]?.doc_id || file.name}`
          : "Aucun document indexé.",
      );
      onUploaded?.();
    } catch (error) {
      const detail = error instanceof ApiError ? error.detail : "";
      setStatus("error");
      setMessage(detail || "Échec de l’ingestion du document.");
    }
  }

  return (
    <div className="rounded-2xl border border-dashed border-border p-6 text-center">
      <UploadCloud className="mx-auto mb-2" size={20} />
      <p className="text-sm">Drag & drop (PDF, images, JSON, TXT)</p>
      <input
        type="file"
        className="mt-3 text-xs"
        accept=".pdf"
        onChange={(e) => void onFile(e.target.files?.[0])}
      />
      <p className="mt-2 text-xs text-fg/70">
        {filename ? `${filename} - ${status}` : "Aucun fichier sélectionné"}
      </p>
      {message ? <p className="mt-1 text-xs text-fg/65">{message}</p> : null}
    </div>
  );
}
