"use client";

import React, { useState } from "react";
import { ChevronDown, ChevronRight, FileText, User, ExternalLink } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface PatientReport {
  doc_id: string;
  filename: string;
  source_url: string;
  viewer_url: string;
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
}

export function PatientInventoryRenderer({ patients, defaultExpanded = false }: PatientInventoryRendererProps) {
  if (!patients || patients.length === 0) return null;

  return (
    <div className="mt-4 space-y-4">
      {patients.map((p) => (
        <PatientCard key={p.patient} patient={p} defaultExpanded={defaultExpanded} />
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
