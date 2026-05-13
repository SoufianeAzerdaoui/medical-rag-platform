"use client";

import React, { useState } from "react";
import { ChevronDown, ChevronRight, FileText, User, ExternalLink } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface PatientReport {
  doc_id: string;
  filename: string;
  source_url: string;
  viewer_url: string;
  date?: string;
}

interface Patient {
  patient: string;
  report_count: number;
  report_range_label: string;
  reports: PatientReport[];
}

interface PatientInventoryRendererProps {
  patients: Patient[];
  defaultExpanded?: boolean;
  inventoryView?: { type?: "patient_cards" | "report_accordion" | "filterable_table" | "document_timeline" };
}

export function PatientInventoryRenderer({ patients, defaultExpanded = false, inventoryView }: PatientInventoryRendererProps) {
  if (!patients || patients.length === 0) return null;
  const viewType = inventoryView?.type || "patient_cards";

  if (viewType === "filterable_table") {
    return (
      <div className="mt-4 rounded-xl border border-border bg-card p-3">
        <p className="mb-2 text-xs uppercase tracking-wide text-fg/65">Table structurée prête à être filtrée</p>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border text-left text-fg/70">
              <th className="py-2">Patient</th>
              <th className="py-2">Rapports</th>
              <th className="py-2">Date</th>
              <th className="py-2">Fichier</th>
              <th className="py-2">Lien</th>
            </tr>
          </thead>
          <tbody>
            {patients.flatMap((p) =>
              p.reports.map((r) => (
                <tr key={`${p.patient}-${r.doc_id}`} className="border-b border-border/60">
                  <td className="py-2">{p.patient}</td>
                  <td className="py-2">{p.report_count}</td>
                  <td className="py-2">{r.date || "Date non disponible"}</td>
                  <td className="py-2">{r.filename}</td>
                  <td className="py-2">
                    <a className="text-primary underline" href={r.viewer_url || r.source_url} target="_blank" rel="noopener noreferrer">
                      Ouvrir
                    </a>
                  </td>
                </tr>
              )),
            )}
          </tbody>
        </table>
      </div>
    );
  }

  if (viewType === "document_timeline") {
    return (
      <div className="mt-4 space-y-4">
        {patients.map((p) => (
          <PatientCard key={p.patient} patient={p} defaultExpanded={defaultExpanded} />
        ))}
      </div>
    );
  }

  return (
    <div className="mt-4 space-y-4">
      {patients.map((p) => (
        <PatientCard key={p.patient} patient={p} defaultExpanded={viewType === "report_accordion" ? true : defaultExpanded} />
      ))}
    </div>
  );
}

function PatientCard({ patient, defaultExpanded }: { patient: Patient; defaultExpanded: boolean }) {
  const [isOpen, setIsOpen] = useState(defaultExpanded);

  return (
    <div className="overflow-hidden rounded-xl border border-border bg-card shadow-sm transition-all hover:border-border/80">
      <div 
        className="flex cursor-pointer items-center justify-between p-4 hover:bg-fg/5"
        onClick={() => setIsOpen(!isOpen)}
      >
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10 text-primary">
            <User size={20} />
          </div>
          <div>
            <h4 className="font-semibold text-fg">{patient.patient}</h4>
            <p className="text-xs text-fg/60">
              {patient.report_count} rapport{patient.report_count > 1 ? "s" : ""} associé{patient.report_count > 1 ? "s" : ""} • {patient.report_range_label}
            </p>
          </div>
        </div>
        <div className="text-fg/40">
          {isOpen ? <ChevronDown size={20} /> : <ChevronRight size={20} />}
        </div>
      </div>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            <div className="border-t border-border bg-fg/5 p-4">
              <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                {patient.reports.map((report) => (
                  <a
                    key={report.doc_id}
                    href={report.viewer_url || report.source_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 rounded-lg border border-border bg-card p-2 text-sm transition-colors hover:bg-primary/5 hover:border-primary/30 group"
                  >
                    <FileText size={16} className="shrink-0 text-primary/70" />
                    <span className="truncate flex-1 font-medium">{report.filename}</span>
                    <ExternalLink size={14} className="shrink-0 opacity-0 group-hover:opacity-100 transition-opacity" />
                  </a>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
