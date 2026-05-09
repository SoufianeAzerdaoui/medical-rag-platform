import type { Metadata } from "next";
import "@/app/globals.css";
import { Providers } from "@/components/layout/providers";

export const metadata: Metadata = {
  title: "CHU Oujda Clinical Assistant",
  description: "Clinical RAG assistant frontend",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="fr" suppressHydrationWarning>
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
