"use client";

type NativePdfRendererProps = {
  src: string | null;
  title: string;
};

export function NativePdfRenderer({ src, title }: NativePdfRendererProps) {
  if (!src) {
    return (
      <div className="flex h-full items-center justify-center p-4 text-sm text-fg/70">
        PDF ou page indisponible pour cette source.
      </div>
    );
  }

  return <iframe title={title} src={src} className="h-full w-full bg-white" />;
}

