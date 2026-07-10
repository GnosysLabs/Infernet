export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
};

export type ChatThread = {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: number;
  updatedAt: number;
};

export type ChatHistory = {
  version: 1;
  activeThreadId: string;
  threads: ChatThread[];
};

export type ChatHistoryLoadResult = {
  history: ChatHistory;
  canPersist: boolean;
  error: string | null;
};

const CHAT_HISTORY_STORAGE_KEY = "infernet-chat-history-v1";
const NEW_CHAT_TITLE = "New chat";
const MAX_THREAD_TITLE_LENGTH = 48;

export function createChatThread(now = Date.now()): ChatThread {
  return {
    id: createChatId("thread"),
    title: NEW_CHAT_TITLE,
    messages: [],
    createdAt: now,
    updatedAt: now,
  };
}

export function createEmptyChatHistory(now = Date.now()): ChatHistory {
  const thread = createChatThread(now);
  return {
    version: 1,
    activeThreadId: thread.id,
    threads: [thread],
  };
}

export function loadChatHistory(): ChatHistory {
  return loadChatHistoryResult().history;
}

export function loadChatHistoryResult(): ChatHistoryLoadResult {
  const storage = getLocalStorage();
  if (!storage) {
    return {
      history: createEmptyChatHistory(),
      canPersist: false,
      error: "local storage is unavailable",
    };
  }

  try {
    const stored = storage.getItem(CHAT_HISTORY_STORAGE_KEY);
    if (!stored) {
      return { history: createEmptyChatHistory(), canPersist: true, error: null };
    }
    const parsed: unknown = JSON.parse(stored);
    if (
      isRecord(parsed)
      && typeof parsed.version === "number"
      && parsed.version !== 1
    ) {
      return {
        history: createEmptyChatHistory(),
        canPersist: false,
        error: `unsupported chat history version ${parsed.version}; expected 1`,
      };
    }
    return { history: parseChatHistory(parsed), canPersist: true, error: null };
  } catch {
    return { history: createEmptyChatHistory(), canPersist: true, error: null };
  }
}

export function saveChatHistory(history: ChatHistory): boolean {
  const storage = getLocalStorage();
  if (!storage) return false;

  try {
    storage.setItem(CHAT_HISTORY_STORAGE_KEY, JSON.stringify(history));
    return true;
  } catch {
    return false;
  }
}

export function appendMessageToThread(
  history: ChatHistory,
  threadId: string,
  message: ChatMessage,
  now = Date.now(),
): ChatHistory {
  const thread = history.threads.find((item) => item.id === threadId);
  if (!thread) return history;

  const shouldCreateTitle = message.role === "user"
    && !thread.messages.some((item) => item.role === "user");
  const updatedThread: ChatThread = {
    ...thread,
    title: shouldCreateTitle ? threadTitleFromPrompt(message.text) : thread.title,
    messages: [...thread.messages, message],
    updatedAt: now,
  };

  return {
    ...history,
    threads: [
      updatedThread,
      ...history.threads.filter((item) => item.id !== threadId),
    ],
  };
}

export function removeChatThread(
  history: ChatHistory,
  threadId: string,
  now = Date.now(),
): ChatHistory {
  const threadIndex = history.threads.findIndex((thread) => thread.id === threadId);
  if (threadIndex === -1) return history;

  const remainingThreads = history.threads.filter((thread) => thread.id !== threadId);
  if (remainingThreads.length === 0) {
    return createEmptyChatHistory(now);
  }

  const nextActiveThreadId = history.activeThreadId === threadId
    ? remainingThreads[Math.min(threadIndex, remainingThreads.length - 1)].id
    : history.activeThreadId;

  return {
    ...history,
    activeThreadId: nextActiveThreadId,
    threads: remainingThreads,
  };
}

export function threadTitleFromPrompt(prompt: string): string {
  const normalized = prompt.replace(/\s+/g, " ").trim();
  if (!normalized) return NEW_CHAT_TITLE;

  const characters = Array.from(normalized);
  if (characters.length <= MAX_THREAD_TITLE_LENGTH) return normalized;
  return `${characters.slice(0, MAX_THREAD_TITLE_LENGTH - 1).join("")}…`;
}

export function createChatMessage(
  role: ChatMessage["role"],
  text: string,
): ChatMessage {
  return {
    id: createChatId(role),
    role,
    text,
  };
}

function parseChatHistory(value: unknown): ChatHistory {
  if (!isRecord(value) || value.version !== 1 || !Array.isArray(value.threads)) {
    return createEmptyChatHistory();
  }

  const seenThreadIds = new Set<string>();
  const threads = value.threads.flatMap((item): ChatThread[] => {
    const thread = parseChatThread(item);
    if (!thread || seenThreadIds.has(thread.id)) return [];
    seenThreadIds.add(thread.id);
    return [thread];
  }).sort((left, right) => right.updatedAt - left.updatedAt);

  if (threads.length === 0) return createEmptyChatHistory();

  const activeThreadId = typeof value.activeThreadId === "string"
    && threads.some((thread) => thread.id === value.activeThreadId)
    ? value.activeThreadId
    : threads[0].id;

  return {
    version: 1,
    activeThreadId,
    threads,
  };
}

function parseChatThread(value: unknown): ChatThread | null {
  if (!isRecord(value) || typeof value.id !== "string" || !value.id.trim()) return null;
  if (!Array.isArray(value.messages)) return null;

  const messages = value.messages.flatMap((item): ChatMessage[] => {
    if (!isRecord(item)) return [];
    if (typeof item.id !== "string" || typeof item.text !== "string") return [];
    if (item.role !== "user" && item.role !== "assistant") return [];
    return [{ id: item.id, role: item.role, text: item.text }];
  });
  const createdAt = finiteTimestamp(value.createdAt) ?? Date.now();
  const updatedAt = finiteTimestamp(value.updatedAt) ?? createdAt;

  return {
    id: value.id,
    title: typeof value.title === "string" && value.title.trim()
      ? value.title.trim()
      : NEW_CHAT_TITLE,
    messages,
    createdAt,
    updatedAt,
  };
}

function finiteTimestamp(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : null;
}

function createChatId(prefix: string): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `${prefix}-${crypto.randomUUID()}`;
  }
  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

function getLocalStorage(): Storage | null {
  try {
    return typeof window === "undefined" ? null : window.localStorage;
  } catch {
    return null;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
