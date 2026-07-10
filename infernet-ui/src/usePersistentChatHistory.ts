import { isTauri } from "@tauri-apps/api/core";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  appendPersistentChatMessage,
  createPersistentChatThread,
  deletePersistentChatThread,
  getChatHistory,
  selectPersistentChatThread,
} from "./api";
import {
  appendMessageToThread,
  createChatMessage,
  createChatThread,
  createEmptyChatHistory,
  loadChatHistory as loadBrowserChatHistory,
  removeChatThread,
  saveChatHistory as saveBrowserChatHistory,
} from "./chatHistory";
import type { ChatHistory, ChatMessage } from "./chatHistory";

type HistoryBackend = "loading" | "native" | "browser" | "unavailable";
type BrowserMutation = (history: ChatHistory) => ChatHistory;

export function usePersistentChatHistory() {
  const nativeRuntime = isTauri();
  const [history, setHistory] = useState<ChatHistory>(() =>
    nativeRuntime ? createEmptyChatHistory() : loadBrowserChatHistory()
  );
  const [backend, setBackend] = useState<HistoryBackend>("loading");
  const [pendingMutations, setPendingMutations] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const historyRef = useRef(history);
  const backendRef = useRef<HistoryBackend>("loading");
  const mutationQueueRef = useRef<Promise<void>>(Promise.resolve());

  const commitHistory = useCallback((nextHistory: ChatHistory) => {
    historyRef.current = nextHistory;
    setHistory(nextHistory);
  }, []);

  const commitBackend = useCallback((nextBackend: HistoryBackend) => {
    backendRef.current = nextBackend;
    setBackend(nextBackend);
  }, []);

  useEffect(() => {
    let disposed = false;

    if (!nativeRuntime) {
      const browserHistory = loadBrowserChatHistory();
      if (!disposed) {
        commitHistory(browserHistory);
        commitBackend("browser");
        if (!saveBrowserChatHistory(browserHistory)) {
          setError("Chat history couldn’t be saved in this browser.");
        }
      }
      return () => {
        disposed = true;
      };
    }

    getChatHistory().then((storedHistory) => {
      if (disposed) return;
      commitHistory(storedHistory);
      commitBackend("native");
      setError(null);
    }).catch((loadError) => {
      if (disposed) return;
      commitBackend("unavailable");
      setError(`Chat history couldn’t be opened: ${friendlyPersistenceError(loadError)}`);
    });

    return () => {
      disposed = true;
    };
  }, [commitBackend, commitHistory, nativeRuntime]);

  const mutate = useCallback((
    nativeMutation: () => Promise<ChatHistory>,
    browserMutation: BrowserMutation,
  ): Promise<ChatHistory | null> => {
    if (backendRef.current === "loading" || backendRef.current === "unavailable") {
      return Promise.resolve(null);
    }

    setPendingMutations((count) => count + 1);
    const mutation = mutationQueueRef.current.then(async () => {
      const nextHistory = backendRef.current === "native"
        ? await nativeMutation()
        : browserMutation(historyRef.current);

      if (backendRef.current === "browser" && !saveBrowserChatHistory(nextHistory)) {
        throw new Error("this browser blocked local storage");
      }

      commitHistory(nextHistory);
      setError(null);
    });
    mutationQueueRef.current = mutation.catch(() => undefined);

    return mutation.then(() => historyRef.current).catch((mutationError) => {
      const destination = backendRef.current === "native"
        ? "the Infernet app-data folder"
        : "this browser";
      setError(
        `Chat history couldn’t be saved to ${destination}: ${friendlyPersistenceError(mutationError)}`,
      );
      return null;
    }).finally(() => {
      setPendingMutations((count) => Math.max(0, count - 1));
    });
  }, [commitHistory]);

  const createThread = useCallback(() => mutate(
    createPersistentChatThread,
    (current) => {
      const thread = createChatThread();
      return {
        ...current,
        activeThreadId: thread.id,
        threads: [thread, ...current.threads],
      };
    },
  ), [mutate]);

  const selectThread = useCallback((threadId: string) => mutate(
    () => selectPersistentChatThread(threadId),
    (current) => current.threads.some((thread) => thread.id === threadId)
      ? { ...current, activeThreadId: threadId }
      : current,
  ), [mutate]);

  const appendMessage = useCallback((
    threadId: string,
    role: ChatMessage["role"],
    text: string,
  ) => mutate(
    () => appendPersistentChatMessage(threadId, role, text),
    (current) => appendMessageToThread(current, threadId, createChatMessage(role, text)),
  ), [mutate]);

  const deleteThread = useCallback((threadId: string) => mutate(
    () => deletePersistentChatThread(threadId),
    (current) => removeChatThread(current, threadId),
  ), [mutate]);

  return {
    history,
    ready: backend === "native" || backend === "browser",
    busy: pendingMutations > 0,
    error,
    createThread,
    selectThread,
    appendMessage,
    deleteThread,
  };
}

function friendlyPersistenceError(error: unknown): string {
  const message = String(error).replace(/^Error:\s*/i, "").trim();
  return message || "unknown storage error";
}
