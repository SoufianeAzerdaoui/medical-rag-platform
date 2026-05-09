"use client";

import { useMemo, useState } from "react";
import { UploadZone } from "@/components/documents/upload-zone";

const mockDocs = [
  { id: "1", name: "Rapport_Bio_14_02.pdf", type: "PDF", status: "indexed" },
  { id: "2", name: "ECG_patient_A.png", type: "Image", status: "processing" },
];

export default function DocumentsPage() {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState("all");

  const list = useMemo(() => {
    return mockDocs.filter((doc) => {
      const matchQ = doc.name.toLowerCase().includes(query.toLowerCase());
      const matchF = filter === "all" || doc.type === filter;
      return matchQ && matchF;
    });
  }, [query, filter]);

  return (
    <main className="mx-auto max-w-6xl space-y-4 p-6">
      <h1 className="text-2xl font-semibold">Documents</h1>
      <UploadZone />
      <div className="flex flex-wrap gap-2">
        <input
          placeholder="Search documents"
          className="rounded-xl border border-border bg-transparent px-3 py-2 text-sm"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <select value={filter} onChange={(e) => setFilter(e.target.value)} className="rounded-xl border border-border bg-transparent px-3 py-2 text-sm">
          <option value="all">Tous</option>
          <option value="PDF">PDF</option>
          <option value="Image">Image</option>
        </select>
      </div>
      <div className="grid gap-3 md:grid-cols-2">
        {list.map((doc) => (
          <article key={doc.id} className="rounded-xl border border-border p-4">
            <p className="font-medium">{doc.name}</p>
            <p className="text-xs text-fg/70">
              {doc.type} - {doc.status}
            </p>
            <button className="mt-3 rounded-lg border border-border px-2 py-1 text-xs">Ask about this document</button>
          </article>
        ))}
      </div>
      {list.length === 0 && <p className="text-sm text-fg/70">Aucun document disponible.</p>}
    </main>
  );
}
