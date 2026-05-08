import type { ChatItem } from "@/types/chat";

function daysBetween(dateIso: string) {
  const now = new Date();
  const date = new Date(dateIso);
  const startNow = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const startDate = new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();
  return Math.floor((startNow - startDate) / (1000 * 60 * 60 * 24));
}

export function groupChatsByDate(chats: ChatItem[]) {
  const groups: Record<string, ChatItem[]> = {
    "Aujourd'hui": [],
    Hier: [],
    "7 derniers jours": [],
    "30 derniers jours": [],
    "Plus ancien": [],
  };

  chats.forEach((chat) => {
    const days = daysBetween(chat.updatedAt);
    if (days === 0) groups["Aujourd'hui"].push(chat);
    else if (days === 1) groups.Hier.push(chat);
    else if (days <= 7) groups["7 derniers jours"].push(chat);
    else if (days <= 30) groups["30 derniers jours"].push(chat);
    else groups["Plus ancien"].push(chat);
  });

  return groups;
}
