"use client";

import { useEffect, useState } from "react";
import { healthcheck } from "@/services/rag-api";
import { useChatStore } from "@/store/chat-store";

export default function DashboardPage() {
  const chats = useChatStore((s) => s.chats);
  const [status, setStatus] = useState("checking");

  useEffect(() => {
    void healthcheck().then(setStatus);
  }, []);

  const totalMessages = chats.reduce((acc, chat) => acc + chat.messages.length, 0);
  const favorites = chats.filter((c) => c.favorite).length;

  return (
    <main className="mx-auto max-w-6xl space-y-6 p-6">
      <h1 className="text-2xl font-semibold">Dashboard clinique</h1>
      <div className="grid gap-3 md:grid-cols-4">
        <Card label="Conversations" value={String(chats.length)} />
        <Card label="Favoris" value={String(favorites)} />
        <Card label="Messages" value={String(totalMessages)} />
        <Card label="Backend" value={status} />
      </div>
      <section className="rounded-2xl border border-border p-4">
        <h2 className="mb-2 font-medium">Conversations récentes</h2>
        <ul className="space-y-2 text-sm text-fg/80">
          {chats.slice(0, 5).map((chat) => (
            <li key={chat.id}>{chat.title}</li>
          ))}
        </ul>
      </section>
    </main>
  );
}

function Card({ label, value }: { label: string; value: string }) {
  return (
    <article className="rounded-xl border border-border p-4">
      <p className="text-xs text-fg/70">{label}</p>
      <p className="text-xl font-semibold">{value}</p>
    </article>
  );
}
