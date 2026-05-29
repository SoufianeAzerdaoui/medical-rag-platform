"use client";

import { Ellipsis, Heart, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import type { KeyboardEvent, MouseEvent } from "react";
import { cn } from "@/lib/utils";
import "./conversation-card.css";

type ConversationCardProps = {
  title: string;
  updatedAt: string;
  sourceCount: number;
  isFavorited?: boolean;
  active?: boolean;
  onClick: () => void;
  onToggleFavorite?: () => void;
  onDelete?: () => void;
  onOpenMenu?: () => void;
};

export function formatRelativeDate(date: string): string {
  const now = new Date();
  const value = new Date(date);
  if (Number.isNaN(value.getTime())) return "";

  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const startOfValue = new Date(value.getFullYear(), value.getMonth(), value.getDate()).getTime();
  const dayDiff = Math.floor((startOfToday - startOfValue) / (1000 * 60 * 60 * 24));
  const hourMinute = new Intl.DateTimeFormat("fr-FR", {
    hour: "2-digit",
    minute: "2-digit",
  }).format(value);

  if (dayDiff === 0) return `Aujourd'hui ${hourMinute}`;
  if (dayDiff === 1) return `Hier ${hourMinute}`;

  if (value.getFullYear() === now.getFullYear()) {
    return new Intl.DateTimeFormat("fr-FR", { day: "numeric", month: "short" }).format(value);
  }

  return new Intl.DateTimeFormat("fr-FR", { day: "numeric", month: "short", year: "numeric" }).format(value);
}

export function ConversationCard({
  title,
  updatedAt,
  sourceCount,
  isFavorited = false,
  active = false,
  onClick,
  onToggleFavorite,
  onDelete,
  onOpenMenu,
}: ConversationCardProps) {
  const [isTouchDevice, setIsTouchDevice] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const media = window.matchMedia("(hover: none), (pointer: coarse)");
    const update = () => setIsTouchDevice(media.matches);
    update();
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);

  function onCardKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onClick();
    }
  }

  function stopCardClick(event: MouseEvent<HTMLButtonElement>) {
    event.stopPropagation();
  }

  const showFavoriteIcon = true;
  const showDeleteIcon = true;

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onClick}
      onKeyDown={onCardKeyDown}
      title={title}
      aria-current={active ? "page" : undefined}
      className={cn(
        "conv-card relative w-full rounded-lg border px-3 py-2.5 text-left transition duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/55",
        sourceCount > 0 && "conv-card-has-source",
        active
          ? "border-accent/[0.55] bg-accent/10 shadow-sm ring-1 ring-accent/25"
          : "border-border/70 bg-card/[0.42] hover:-translate-y-[1px] hover:border-accent/30 hover:bg-card/[0.72] hover:shadow-sm",
      )}
    >
      <p className="truncate text-[13.5px] font-medium leading-[1.25] text-fg">{title}</p>
      <div className="mt-1.5 flex items-center justify-between gap-2">
        <span className="truncate text-[11px] text-fg/70">{formatRelativeDate(updatedAt)}</span>
        {sourceCount > 0 ? (
          <span className="shrink-0 rounded-[20px] border border-[rgba(6,182,212,0.25)] bg-[rgba(6,182,212,0.12)] px-[7px] py-[1px] text-[11px] leading-4 text-[#06B6D4]">
            {sourceCount} source{sourceCount > 1 ? "s" : ""}
          </span>
        ) : null}
      </div>
      <div className={cn("conv-actions", isTouchDevice && "conv-actions-touch", isFavorited && !isTouchDevice && "conv-actions-has-fav")}>
        {showFavoriteIcon ? (
          <button
            type="button"
            aria-label={isFavorited ? "Retirer des favoris" : "Ajouter aux favoris"}
            title={isFavorited ? "Retirer des favoris" : "Ajouter aux favoris"}
            className={cn("conv-action-btn", isFavorited && "conv-action-favorite-persistent")}
            onClick={(event) => {
              stopCardClick(event);
              onToggleFavorite?.();
            }}
          >
            <Heart size={14} className={cn("conv-action-icon", isFavorited && "conv-action-icon-favorited")} />
          </button>
        ) : null}
        {showDeleteIcon ? (
          <button
            type="button"
            aria-label="Supprimer la conversation"
            title="Supprimer la conversation"
            className="conv-action-btn conv-action-extra"
            onClick={(event) => {
              stopCardClick(event);
              onDelete?.();
            }}
          >
            <Trash2 size={14} className="conv-action-icon" />
          </button>
        ) : null}
        <button
          type="button"
          aria-label="Menu actions conversation"
          title="Menu actions conversation"
          className="conv-action-btn conv-action-extra"
          onClick={(event) => {
            stopCardClick(event);
            onOpenMenu?.();
          }}
        >
          <Ellipsis size={14} className="conv-action-icon" />
        </button>
      </div>
    </div>
  );
}
