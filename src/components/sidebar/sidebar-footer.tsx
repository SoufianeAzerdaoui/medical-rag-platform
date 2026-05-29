"use client";

import { ChevronsUpDown } from "lucide-react";
import { forwardRef } from "react";
import type { AuthUser } from "@/services/rag-api";
import "./sidebar-footer.css";

type SidebarFooterProps = {
  user: AuthUser | null;
  onClick: () => void;
};

type UserAvatarProps = {
  email: string;
  name?: string | null;
  photoUrl?: string | null;
};

type AuthUserExtended = AuthUser & {
  name?: string | null;
  photoUrl?: string | null;
  avatarUrl?: string | null;
  image?: string | null;
  picture?: string | null;
};

export function formatName(email: string): string {
  const localPart = String(email || "").split("@")[0]?.trim() || "";
  if (!localPart) return "Utilisateur";
  return localPart.charAt(0).toUpperCase() + localPart.slice(1);
}

export function emailToColor(email: string): string {
  const text = String(email || "").trim().toLowerCase();
  let hash = 0;
  for (let i = 0; i < text.length; i += 1) {
    hash = (hash << 5) - hash + text.charCodeAt(i);
    hash |= 0;
  }
  const hue = Math.abs(hash) % 360;
  return `hsl(${hue} 62% 42%)`;
}

function UserAvatar({ email, name, photoUrl }: UserAvatarProps) {
  const initial = (name?.trim() || email.trim() || "U").charAt(0).toUpperCase();
  if (photoUrl) {
    return <img src={photoUrl} alt={name || email} className="user-avatar user-avatar-image" />;
  }

  return (
    <span className="user-avatar user-avatar-fallback" style={{ backgroundColor: emailToColor(email) }}>
      {initial}
    </span>
  );
}

export const SidebarFooter = forwardRef<HTMLButtonElement, SidebarFooterProps>(function SidebarFooter({ user, onClick }, ref) {
  const safeEmail = user?.email || "utilisateur@clinical.local";
  const userExt = (user as AuthUserExtended | null) || null;
  const displayName = userExt?.name?.trim() || formatName(safeEmail);
  const photoUrl = userExt?.photoUrl || userExt?.avatarUrl || userExt?.image || userExt?.picture || null;

  return (
    <button ref={ref} type="button" className="sidebar-footer" onClick={onClick} aria-label="Ouvrir le menu utilisateur">
      <UserAvatar email={safeEmail} name={displayName} photoUrl={photoUrl} />
      <div className="sidebar-footer-content">
        <p className="user-name">{displayName}</p>
        <p className="user-email">{safeEmail}</p>
      </div>
      <ChevronsUpDown size={16} className="sidebar-footer-chevron" />
    </button>
  );
});
