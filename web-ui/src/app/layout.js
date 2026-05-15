import "./globals.css";
import { Inter, Outfit } from "next/font/google";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const outfit = Outfit({ subsets: ["latin"], variable: "--font-outfit" });

export const metadata = {
  title: "Medical RAG Platform | Clinical Assistant",
  description: "Evidence-based medical report analysis with strict provenance.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="fr" className={`${inter.variable} ${outfit.variable}`}>
      <body>{children}</body>
    </html>
  );
}
