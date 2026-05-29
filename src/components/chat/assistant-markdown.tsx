"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ReactNode } from "react";

function toPlainText(node: ReactNode): string {
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(toPlainText).join("");
  if (node && typeof node === "object" && "props" in node) {
    const props = node.props as { children?: ReactNode };
    return toPlainText(props.children ?? "");
  }
  return "";
}

function statusTone(text: string):
  | "high"
  | "low"
  | "normal"
  | "unknown"
  | null {
  const value = text.trim().toLowerCase();
  if (!value) return null;
  if (value.includes("au-dessus") || value.includes("au dessus") || value.includes("above")) return "high";
  if (value.includes("en dessous") || value.includes("en-dessous") || value.includes("below")) return "low";
  if (value.includes("dans la référence") || value.includes("dans la reference") || value.includes("within")) return "normal";
  if (value.includes("non interprétable") || value.includes("non interpretable") || value.includes("unknown")) return "unknown";
  return null;
}

function StatusBadge({ text }: { text: string }) {
  const tone = statusTone(text);
  if (!tone) return <>{text}</>;

  const cls =
    tone === "high"
      ? "status-warning"
      : tone === "low"
        ? "status-low"
        : tone === "normal"
          ? "status-success"
          : "status-neutral";

  return (
    <span className={`inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-medium ${cls}`}>
      {text}
    </span>
  );
}

const markdownComponents = {
  h3: ({ children }: { children?: ReactNode }) => (
    <h3 className="mb-2 mt-4 text-base font-semibold text-fg">{children}</h3>
  ),
  p: ({ children }: { children?: ReactNode }) => <p className="my-2 whitespace-pre-wrap leading-6">{children}</p>,
  ul: ({ children }: { children?: ReactNode }) => <ul className="my-3 list-disc space-y-1.5 pl-5">{children}</ul>,
  li: ({ children }: { children?: ReactNode }) => <li className="leading-6 [&>ul]:mt-1.5">{children}</li>,
  strong: ({ children }: { children?: ReactNode }) => <strong className="font-semibold text-fg">{children}</strong>,
  a: ({ href, children }: { href?: string; children?: ReactNode }) => (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="text-accent underline decoration-accent/60 underline-offset-2 hover:decoration-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/70"
    >
      {children}
    </a>
  ),
  code: ({ inline, children }: { inline?: boolean; children?: ReactNode }) => {
    if (inline) {
      return <code className="rounded border border-border/70 bg-fg/[0.06] px-1.5 py-0.5 text-xs text-fg">{children}</code>;
    }
    return (
      <code className="block overflow-x-auto rounded-lg border border-border/70 bg-fg/[0.05] p-3 text-xs text-fg">
        {children}
      </code>
    );
  },
  table: ({ children }: { children?: ReactNode }) => (
    <div className="my-4 overflow-x-auto rounded-xl border border-border/75 bg-card/[0.55]">
      <table className="min-w-full border-collapse text-sm">{children}</table>
    </div>
  ),
  thead: ({ children }: { children?: ReactNode }) => <thead className="bg-fg/[0.07] text-fg">{children}</thead>,
  tbody: ({ children }: { children?: ReactNode }) => <tbody className="divide-y divide-border/80">{children}</tbody>,
  tr: ({ children }: { children?: ReactNode }) => <tr className="transition-colors hover:bg-fg/[0.04]">{children}</tr>,
  th: ({ children }: { children?: ReactNode }) => (
    <th scope="col" className="whitespace-nowrap border-b border-border px-4 py-3 text-left font-semibold">
      {children}
    </th>
  ),
  td: ({ children }: { children?: ReactNode }) => {
    const text = toPlainText(children).replace(/\s+/g, " ").trim();
    const tone = statusTone(text);
    return (
      <td className="border-b border-border/80 px-4 py-3 align-top text-fg/85">
        {tone ? <StatusBadge text={text} /> : children}
      </td>
    );
  },
};

export function AssistantMarkdown({ content }: { content: string }) {
  return (
    <div className="max-w-none text-sm text-fg [&>*:first-child]:mt-0 [&>*:last-child]:mb-0">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
        {content}
      </ReactMarkdown>
    </div>
  );
}
