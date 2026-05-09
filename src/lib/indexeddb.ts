import { openDB } from "idb";
import type { ChatItem } from "@/types/chat";

const DB_NAME = "clinical-rag-chat-db";
const STORE = "chats";

async function db() {
  return openDB(DB_NAME, 1, {
    upgrade(database) {
      if (!database.objectStoreNames.contains(STORE)) {
        database.createObjectStore(STORE, { keyPath: "id" });
      }
    },
  });
}

export async function getChats(): Promise<ChatItem[]> {
  const database = await db();
  return database.getAll(STORE);
}

export async function putChat(chat: ChatItem): Promise<void> {
  const database = await db();
  await database.put(STORE, chat);
}

export async function deleteChat(id: string): Promise<void> {
  const database = await db();
  await database.delete(STORE, id);
}

export async function clearChats(): Promise<void> {
  const database = await db();
  await database.clear(STORE);
}
