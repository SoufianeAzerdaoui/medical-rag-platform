"use client";

import { UploadCloud } from "lucide-react";
import { useState } from "react";

type UploadStatus = "ready" | "uploading" | "processing" | "indexed" | "error";

export function UploadZone() {
  const [status, setStatus] = useState<UploadStatus>("ready");
  const [filename, setFilename] = useState("");

  function onFile(file?: File) {
    if (!file) return;
    setFilename(file.name);
    setStatus("uploading");
    setTimeout(() => setStatus("processing"), 500);
    setTimeout(() => setStatus("indexed"), 1400);
  }

  return (
    <div className="rounded-2xl border border-dashed border-border p-6 text-center">
      <UploadCloud className="mx-auto mb-2" size={20} />
      <p className="text-sm">Drag & drop (PDF, images, JSON, TXT)</p>
      <input
        type="file"
        className="mt-3 text-xs"
        accept=".pdf,.png,.jpg,.jpeg,.json,.txt"
        onChange={(e) => onFile(e.target.files?.[0])}
      />
      <p className="mt-2 text-xs text-fg/70">
        {filename ? `${filename} - ${status}` : "Aucun fichier sélectionné"}
      </p>
    </div>
  );
}
