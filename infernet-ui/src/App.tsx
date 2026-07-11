import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import createGlobe from "cobe";
import type { Marker } from "cobe";
import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Activity,
  Box,
  Cpu,
  Download,
  Globe,
  HardDrive,
  CircleHelp,
  Image as ImageIcon,
  Laptop2,
  Layers3,
  LoaderCircle,
  MemoryStick,
  MessageSquare,
  Plus,
  Send,
  Server,
  Settings,
  Trash2,
  Zap,
} from "lucide-react";
import {
  emptySnapshot,
  generateImage,
  getGridSnapshot,
  getImageRuntimeStatus,
  getLocalNodeActivity,
  getVramContributionSettings,
  installOfficialImage,
  listGeneratedImages,
  installOfficialModel,
  listenForProgress,
  listenForModelImportProgress,
  runDistributedInference,
  setVramContribution,
} from "./api";
import type {
  GenerateImageResponse,
  GridSnapshot,
  ImageRuntimeStatus,
  LocalNodeActivitySnapshot,
  MachineView,
  ModelImportProgress,
  ModelView,
  ProgressEvent,
  VramContributionSettings,
} from "./types";
import { createChatMessage } from "./chatHistory";
import type { ChatMessage, ChatThread } from "./chatHistory";
import { buildConversationPrompt } from "./conversationContext";
import { usePersistentChatHistory } from "./usePersistentChatHistory";
import { useAppUpdater, type AppUpdaterState } from "./useAppUpdater";

type PrimaryMode = "chat" | "image";
type Page = PrimaryMode | "activity" | "network" | "downloads" | "about" | "settings";
type TransferStatus = "active" | "complete" | "error";
type TransferActivity = ModelImportProgress & {
  id: string;
  status: TransferStatus;
  startedAt: number;
  updatedAt: number;
};
type NodeJournalEntry = {
  id: string;
  kind: "completion" | "contribution" | "image" | "model" | "sharing" | "error";
  title: string;
  detail?: string;
  occurredAt: number;
};
type RequiredSetupErrors = Partial<Record<"chat" | "image" | "status", string>>;

const DEFAULT_PROMPT = "";
const COMPOSER_MAX_HEIGHT = 160;
const INFERNET_CHAT_MODEL_ID = "infernet-chat-v1";
const INFERNET_IMAGE_MODEL_ID = "infernet-image-v1";
const UNSAVED_RESPONSE_ERROR = "The response is visible, but Infernet couldn’t save it. Keep the app open if you need to copy it.";
const MARKDOWN_COMPONENTS: Components = {
  a: ({ node: _node, ...props }) => (
    <a {...props} target="_blank" rel="noreferrer noopener" />
  ),
  table: ({ node: _node, ...props }) => (
    <div className="markdown-table-wrap" role="region" aria-label="Scrollable table" tabIndex={0}>
      <table {...props} />
    </div>
  ),
};
const EMPTY_LOCAL_NODE_ACTIVITY: LocalNodeActivitySnapshot = {
  computeActive: false,
  computeReady: false,
  computeBackend: "cpu",
  deviceName: "This computer",
  totalMemoryBytes: 0,
  availableMemoryBytes: 0,
  sharingActive: false,
  bytesServed: 0,
  chunksServed: 0,
  lastServedUnixMs: null,
  current: [],
  journal: [],
};

export default function App() {
  const appUpdater = useAppUpdater();
  const [page, setPage] = useState<Page>("chat");
  const [primaryMode, setPrimaryMode] = useState<PrimaryMode>("chat");
  const {
    history: chatHistory,
    ready: chatHistoryReady,
    busy: chatHistoryBusy,
    error: chatHistoryError,
    createThread,
    selectThread,
    appendMessage,
    deleteThread,
  } = usePersistentChatHistory();
  const [localNodeActivity, setLocalNodeActivity] = useState<LocalNodeActivitySnapshot>(
    EMPTY_LOCAL_NODE_ACTIVITY,
  );
  const [localJournal, setLocalJournal] = useState<NodeJournalEntry[]>([]);
  const [snapshot, setSnapshot] = useState<GridSnapshot>(emptySnapshot);
  const [selectedModel, setSelectedModel] = useState("");
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [imagePrompt, setImagePrompt] = useState("");
  const [imageGenerating, setImageGenerating] = useState(false);
  const imageGeneratingRef = useRef(false);
  const [imageGenerationStartedAt, setImageGenerationStartedAt] = useState<number | null>(null);
  const [imageResult, setImageResult] = useState<GenerateImageResponse | null>(null);
  const [imageCreations, setImageCreations] = useState<GenerateImageResponse[]>([]);
  const [imageGenerationError, setImageGenerationError] = useState<string | null>(null);
  const [runningThreadId, setRunningThreadId] = useState<string | null>(null);
  const runningThreadIdRef = useRef<string | null>(null);
  const [threadErrors, setThreadErrors] = useState<Record<string, string>>({});
  const [unsavedAssistantMessages, setUnsavedAssistantMessages] = useState<
    Record<string, ChatMessage[]>
  >({});
  const [composerFocusRequest, setComposerFocusRequest] = useState(0);
  const [imageFocusRequest, setImageFocusRequest] = useState(0);
  const [lastError, setLastError] = useState<string | null>(null);
  const [transferActivities, setTransferActivities] = useState<TransferActivity[]>([]);
  const [connectionGraceExpired, setConnectionGraceExpired] = useState(false);
  const [snapshotChecked, setSnapshotChecked] = useState(false);
  const [imageStatusChecked, setImageStatusChecked] = useState(false);
  const [imageRuntimeStatus, setImageRuntimeStatus] = useState<ImageRuntimeStatus | null>(null);
  const [requiredSetupRunning, setRequiredSetupRunning] = useState(false);
  const [requiredSetupErrors, setRequiredSetupErrors] = useState<RequiredSetupErrors>({});
  const requiredSetupStartedRef = useRef(false);

  useEffect(() => {
    let disposed = false;
    listGeneratedImages()
      .then((creations) => {
        if (!disposed) setImageCreations(creations);
      })
      .catch((error) => console.error("Failed to restore generated images", error));
    return () => {
      disposed = true;
    };
  }, []);

  const activeThread = chatHistory.threads.find(
    (thread) => thread.id === chatHistory.activeThreadId,
  ) ?? chatHistory.threads[0];
  const activeMessages = activeThread
    ? [...activeThread.messages, ...(unsavedAssistantMessages[activeThread.id] ?? [])]
    : [];
  const isRunning = runningThreadId !== null;

  const officialModels = useMemo(
    () => snapshot.availableModels.filter(isOfficialInfernetModel),
    [snapshot.availableModels],
  );
  const selectedModelView = useMemo(
    () => officialModels.find((model) => model.modelId === selectedModel),
    [officialModels, selectedModel],
  );
  const chatModelView = officialModels.find((model) => model.modelId === INFERNET_CHAT_MODEL_ID);
  const chatModelInstalled = Boolean(
    chatModelView?.installed
    || snapshot.distribution.installedModels.includes(INFERNET_CHAT_MODEL_ID),
  );
  const imageModelInstalled = Boolean(imageRuntimeStatus?.verified);
  const activeTransfers = transferActivities.filter((activity) => activity.status === "active").length;
  const connectionPending = !snapshot.localPeerId
    || Boolean(selectedModelView?.installed && !selectedModelView.runnable);
  const isEstablishingConnection = connectionPending && !connectionGraceExpired;

  const appendJournalEntry = useCallback((entry: NodeJournalEntry) => {
    setLocalJournal((current) => {
      if (current.some((item) => item.id === entry.id)) return current;
      return [...current, entry]
        .sort((left, right) => left.occurredAt - right.occurredAt)
        .slice(-50);
    });
  }, []);

  const updateRunningThread = useCallback((threadId: string | null) => {
    runningThreadIdRef.current = threadId;
    setRunningThreadId(threadId);
  }, []);

  const applyProgressEvent = useCallback((event: ProgressEvent) => {
    if (event.type === "routeDiscovered" || event.type === "executionPlan") {
      setLastError(null);
      return;
    }

    if (event.type === "finalOutput") return;

    if (event.type === "error") {
      const threadId = runningThreadIdRef.current;
      if (threadId) {
        setThreadErrors((current) => ({ ...current, [threadId]: event.message }));
      } else {
        setLastError(event.message);
      }
    }
  }, []);

  const refreshSnapshot = useCallback(async (modelId?: string) => {
    try {
      const nextSnapshot = await getGridSnapshot(4000, modelId);
      const nextOfficialModels = nextSnapshot.availableModels.filter(isOfficialInfernetModel);
      const modelStillExists = modelId && nextOfficialModels.some((model) => model.modelId === modelId);
      const nextSelectedModel = modelStillExists
        ? modelId
        : nextOfficialModels.find((model) => model.modelId === nextSnapshot.selectedModel)?.modelId
          || nextOfficialModels[0]?.modelId
          || "";
      setSnapshot(nextSnapshot);
      setLastError(nextSelectedModel ? nextSnapshot.missingRanges ?? null : null);
      setSelectedModel(nextSelectedModel);
    } catch (error) {
      setLastError(String(error));
    } finally {
      setSnapshotChecked(true);
    }
  }, []);

  const beginRequiredSetup = useCallback(async () => {
    if (requiredSetupRunning) return;

    requiredSetupStartedRef.current = true;
    setRequiredSetupRunning(true);
    setRequiredSetupErrors({});

    const tasks: Promise<void>[] = [];
    if (!chatModelInstalled) {
      tasks.push(
        installOfficialModel(INFERNET_CHAT_MODEL_ID)
          .then((next) => {
            setSnapshot(next);
            setSelectedModel(INFERNET_CHAT_MODEL_ID);
          })
          .catch((error) => {
            setRequiredSetupErrors((current) => ({ ...current, chat: String(error) }));
          }),
      );
    }
    if (!imageModelInstalled) {
      tasks.push(
        installOfficialImage()
          .then(setImageRuntimeStatus)
          .catch((error) => {
            setRequiredSetupErrors((current) => ({ ...current, image: imageErrorMessage(error) }));
          }),
      );
    }

    await Promise.all(tasks);

    const [nextSnapshot, nextImageStatus] = await Promise.allSettled([
      getGridSnapshot(4000, INFERNET_CHAT_MODEL_ID),
      getImageRuntimeStatus(),
    ]);
    if (nextSnapshot.status === "fulfilled") {
      setSnapshot(nextSnapshot.value);
      setSelectedModel(INFERNET_CHAT_MODEL_ID);
    } else {
      setRequiredSetupErrors((current) => ({
        ...current,
        status: "Infernet couldn’t confirm the local model status. Try again.",
      }));
    }
    if (nextImageStatus.status === "fulfilled") {
      setImageRuntimeStatus(nextImageStatus.value);
    } else {
      setRequiredSetupErrors((current) => ({
        ...current,
        status: "Infernet couldn’t confirm the local model status. Try again.",
      }));
    }
    setRequiredSetupRunning(false);
  }, [chatModelInstalled, imageModelInstalled, requiredSetupRunning]);

  useEffect(() => {
    let disposed = false;
    let initialized = false;
    let previousComputeActive = false;
    let previousSharingActive = false;
    let previousBytesServed = 0;
    let untrackedComputeStartedAt: number | null = null;
    let sharingStartedAt: number | null = null;
    let sharingStartedBytes = 0;

    const refreshLocalActivity = async () => {
      try {
        const next = await getLocalNodeActivity();
        if (disposed) return;

        const now = Date.now();
        const hasTrackedTask = next.current.length > 0;
        if (hasTrackedTask) untrackedComputeStartedAt = null;
        if (!initialized) {
          initialized = true;
          if (next.computeActive && !hasTrackedTask) untrackedComputeStartedAt = now;
          if (next.sharingActive) {
            sharingStartedAt = next.lastServedUnixMs ?? now;
            sharingStartedBytes = next.bytesServed;
          }
        } else {
          if (!previousComputeActive && next.computeActive && !hasTrackedTask) {
            untrackedComputeStartedAt = now;
          }
          if (previousComputeActive && !next.computeActive && untrackedComputeStartedAt !== null) {
            appendJournalEntry({
              id: `compute-contribution-${untrackedComputeStartedAt}`,
              kind: "contribution",
              title: "You contributed compute",
              detail: "Your node helped the network process a chat request.",
              occurredAt: now,
            });
            untrackedComputeStartedAt = null;
          }

          if (!previousSharingActive && next.sharingActive) {
            sharingStartedAt = next.lastServedUnixMs ?? now;
            sharingStartedBytes = previousBytesServed;
          }
          if (previousSharingActive && !next.sharingActive && sharingStartedAt !== null) {
            const sharedBytes = Math.max(0, next.bytesServed - sharingStartedBytes);
            if (sharedBytes > 0) {
              appendJournalEntry({
                id: `model-sharing-${sharingStartedAt}`,
                kind: "sharing",
                title: "You shared Infernet Chat",
                detail: `${formatBytes(sharedBytes)} sent to another node.`,
                occurredAt: next.lastServedUnixMs ?? now,
              });
            }
            sharingStartedAt = null;
          }
        }

        previousComputeActive = next.computeActive;
        previousSharingActive = next.sharingActive;
        previousBytesServed = next.bytesServed;
        setLocalNodeActivity(next);
      } catch {
        // Browser previews do not have the Tauri command; the network snapshot
        // remains available as a quiet fallback for the HUD.
      }
    };

    void refreshLocalActivity();
    const interval = window.setInterval(refreshLocalActivity, 1000);
    return () => {
      disposed = true;
      window.clearInterval(interval);
    };
  }, [appendJournalEntry]);

  useEffect(() => {
    refreshSnapshot(selectedModel);
  }, [refreshSnapshot, selectedModel]);

  useEffect(() => {
    let disposed = false;
    getImageRuntimeStatus()
      .then((next) => {
        if (!disposed) setImageRuntimeStatus(next);
      })
      .catch((error) => {
        if (!disposed) {
          setRequiredSetupErrors((current) => ({ ...current, status: imageErrorMessage(error) }));
        }
      })
      .finally(() => {
        if (!disposed) setImageStatusChecked(true);
      });
    return () => {
      disposed = true;
    };
  }, []);

  useEffect(() => {
    if (!connectionPending) {
      setConnectionGraceExpired(false);
      return;
    }
    setConnectionGraceExpired(false);
    const timeout = window.setTimeout(() => setConnectionGraceExpired(true), 12_000);
    return () => window.clearTimeout(timeout);
  }, [connectionPending]);

  useEffect(() => {
    if (page !== "chat" || !connectionPending) return;
    let disposed = false;
    let timeout: number | undefined;
    const pollConnection = async () => {
      await refreshSnapshot(selectedModel);
      if (!disposed) {
        timeout = window.setTimeout(pollConnection, 1000);
      }
    };
    timeout = window.setTimeout(pollConnection, 1000);
    return () => {
      disposed = true;
      if (timeout !== undefined) window.clearTimeout(timeout);
    };
  }, [connectionPending, page, refreshSnapshot, selectedModel]);

  useEffect(() => {
    if (page === "chat" && connectionPending) return;
    let disposed = false;
    let inFlight = false;
    const refreshStatus = async () => {
      if (inFlight) return;
      inFlight = true;
      try {
        const nextSnapshot = await getGridSnapshot(2500, selectedModel);
        if (!disposed) {
          setSnapshot(nextSnapshot);
        }
      } catch {
        // The primary refresh path owns user-visible connection errors.
      } finally {
        inFlight = false;
      }
    };
    void refreshStatus();
    const interval = window.setInterval(refreshStatus, 6000);
    return () => {
      disposed = true;
      window.clearInterval(interval);
    };
  }, [connectionPending, page, selectedModel]);

  useEffect(() => {
    let disposed = false;
    let unlisten: (() => void) | undefined;
    listenForProgress(applyProgressEvent).then((dispose) => {
      if (disposed) {
        dispose();
      } else {
        unlisten = dispose;
      }
    });

    return () => {
      disposed = true;
      unlisten?.();
    };
  }, [applyProgressEvent]);

  useEffect(() => {
    let disposed = false;
    let unlisten: (() => void) | undefined;
    listenForModelImportProgress((event) => {
      const isChatModel = isOfficialModelId(event.modelId);
      const isImageModel = isOfficialImageModelId(event.modelId);
      if (!isChatModel && !isImageModel) {
        return;
      }
      setTransferActivities((current) => upsertTransferActivity(current, event));
      const normalizedStage = event.stage.trim().toLowerCase();
      const isReady = isChatModel
        ? normalizedStage === "ready"
        : normalizedStage === "image package ready";
      if (isReady) {
        appendJournalEntry({
          id: `model-ready-${event.modelId}`,
          kind: "model",
          title: isImageModel ? "You prepared Infernet Image" : "You prepared Infernet Chat",
          detail: isImageModel
            ? "The verified image package is ready to use and share."
            : "The verified model is ready to use and share.",
          occurredAt: Date.now(),
        });
      } else if (normalizedStage.includes("failed") || normalizedStage.includes("error")) {
        appendJournalEntry({
          id: `model-error-${event.modelId}-${Date.now()}`,
          kind: "error",
          title: isImageModel
            ? "Infernet Image setup couldn’t finish"
            : "A model task couldn’t finish",
          detail: isImageModel ? event.detail : friendlyActivityError(event.detail),
          occurredAt: Date.now(),
        });
      }
    }).then((dispose) => {
      if (disposed) {
        dispose();
      } else {
        unlisten = dispose;
      }
    });

    return () => {
      disposed = true;
      unlisten?.();
    };
  }, [appendJournalEntry]);

  useEffect(() => {
    if (!snapshotChecked || !imageStatusChecked) {
      return;
    }
    if (chatModelInstalled && imageModelInstalled) {
      requiredSetupStartedRef.current = false;
      return;
    }
    if (requiredSetupStartedRef.current) return;
    void beginRequiredSetup();
  }, [
    beginRequiredSetup,
    chatModelInstalled,
    imageModelInstalled,
    imageStatusChecked,
    snapshotChecked,
  ]);

  async function createNewThread() {
    if (!chatHistoryReady) return;
    setPrimaryMode("chat");
    setPage("chat");
    const nextHistory = await createThread();
    if (nextHistory) {
      setPrompt("");
      setLastError(null);
      setComposerFocusRequest((request) => request + 1);
    }
  }

  async function openThread(threadId: string) {
    setPrimaryMode("chat");
    if (!chatHistoryReady || threadId === chatHistory.activeThreadId) {
      setPage("chat");
      return;
    }
    setPage("chat");
    const nextHistory = await selectThread(threadId);
    if (nextHistory) {
      setPrompt("");
      setLastError(null);
    }
  }

  function selectPrimaryMode(mode: PrimaryMode) {
    setPrimaryMode(mode);
    setPage(mode);
  }

  function startNewImage() {
    setPrimaryMode("image");
    setPage("image");
    if (imageGeneratingRef.current) return;
    setImagePrompt("");
    setImageResult(null);
    setImageGenerationError(null);
    setImageFocusRequest((request) => request + 1);
  }

  async function runImageGeneration() {
    const cleanPrompt = imagePrompt.trim();
    if (
      !cleanPrompt
      || imageGeneratingRef.current
      || imageRuntimeStatus?.busy
      || !imageRuntimeStatus?.runtimeAvailable
      || !imageRuntimeStatus.verified
    ) {
      return;
    }

    imageGeneratingRef.current = true;
    setImageGenerating(true);
    setImageGenerationStartedAt(Date.now());
    setImageGenerationError(null);
    setImageResult(null);
    try {
      const nextImage = await generateImage(cleanPrompt);
      setImageResult(nextImage);
      setImageCreations((current) => [
        nextImage,
        ...current.filter((creation) => creation.imageId !== nextImage.imageId),
      ]);
    } catch (error) {
      setImageGenerationError(imageErrorMessage(error));
    } finally {
      imageGeneratingRef.current = false;
      setImageGenerating(false);
      getImageRuntimeStatus().then(setImageRuntimeStatus).catch(() => undefined);
    }
  }

  function openImageCreation(creation: GenerateImageResponse) {
    setPrimaryMode("image");
    setPage("image");
    setImageResult(creation);
    setImageGenerationError(null);
  }

  async function removeThread(threadId: string) {
    if (!chatHistoryReady || threadId === runningThreadIdRef.current) return;
    const wasActive = threadId === chatHistory.activeThreadId;
    const nextHistory = await deleteThread(threadId);
    if (!nextHistory) return;
    setThreadErrors((current) => {
      const next = { ...current };
      delete next[threadId];
      return next;
    });
    setUnsavedAssistantMessages((current) => {
      const next = { ...current };
      delete next[threadId];
      return next;
    });
    if (wasActive) {
      setPrompt("");
      setLastError(null);
    }
  }

  async function runInference() {
    const userPrompt = prompt.trim();
    const threadId = activeThread?.id;
    if (
      !userPrompt
      || !threadId
      || !chatHistoryReady
      || chatHistoryBusy
      || runningThreadIdRef.current
    ) {
      return;
    }

    if (!selectedModelView) {
      setLastError("Infernet Chat is not available on this computer.");
      return;
    }
    if (!selectedModelView.runnable) {
      setLastError(selectedModelView.status);
      return;
    }

    updateRunningThread(threadId);
    setThreadErrors((current) => {
      const next = { ...current };
      delete next[threadId];
      return next;
    });
    const historyWithPrompt = await appendMessage(threadId, "user", userPrompt);
    if (!historyWithPrompt) {
      updateRunningThread(null);
      return;
    }

    setPrompt("");
    setLastError(null);

    try {
      const conversationPrompt = buildConversationPrompt(activeMessages, userPrompt);
      const output = (await runDistributedInference(conversationPrompt, selectedModel)).output;
      const historyWithResponse = await appendMessage(threadId, "assistant", output);
      if (!historyWithResponse) {
        const unsavedMessage = createChatMessage("assistant", output);
        setUnsavedAssistantMessages((current) => ({
          ...current,
          [threadId]: [...(current[threadId] ?? []), unsavedMessage],
        }));
        setThreadErrors((current) => ({
          ...current,
          [threadId]: UNSAVED_RESPONSE_ERROR,
        }));
      }
    } catch (error) {
      const message = String(error);
      setThreadErrors((current) => ({ ...current, [threadId]: message }));
    } finally {
      if (runningThreadIdRef.current === threadId) {
        updateRunningThread(null);
      }
    }
  }

  const requiredSetupComplete = snapshotChecked
    && imageStatusChecked
    && chatModelInstalled
    && imageModelInstalled;

  if (!requiredSetupComplete) {
    return (
      <RequiredModelsOnboarding
        checking={!snapshotChecked || !imageStatusChecked}
        chatInstalled={chatModelInstalled}
        imageStatus={imageRuntimeStatus}
        transfers={transferActivities}
        running={requiredSetupRunning}
        errors={requiredSetupErrors}
        onRetry={() => void beginRequiredSetup()}
      />
    );
  }

  return (
    <div className="app-shell">
      <UpdateBanner updater={appUpdater} />
      <Sidebar
        threads={chatHistory.threads}
        activeThreadId={chatHistory.activeThreadId}
        chatIsVisible={page === "chat"}
        primaryMode={primaryMode}
        runningThreadId={runningThreadId}
        imageGenerationBusy={imageGenerating || Boolean(imageRuntimeStatus?.busy)}
        imageCreations={imageCreations}
        activeImageId={page === "image" ? imageResult?.imageId ?? null : null}
        disabled={!chatHistoryReady || chatHistoryBusy}
        persistenceError={chatHistoryError}
        onCreateThread={createNewThread}
        onStartImage={startNewImage}
        onOpenImage={openImageCreation}
        onModeChange={selectPrimaryMode}
        onOpenThread={openThread}
        onDeleteThread={removeThread}
      />

      <main className="app-main">
        <AppHeader
          page={page}
          chatTitle={activeThread?.title ?? "New chat"}
          networkNodeCount={snapshot.machines.filter((machine) => machine.connectionStatus !== "unreachable").length}
          networkReadyCount={snapshot.machines.filter((machine) => machine.rpcReady && machine.connectionStatus !== "unreachable").length}
          hasActiveWork={
            imageGenerating
            || imageRuntimeStatus?.busy
            || localNodeActivity.computeActive
            || localNodeActivity.sharingActive
            || activeTransfers > 0
          }
          onNavigate={setPage}
        />

        {page === "chat" ? (
          <ChatPage
            messages={activeMessages}
            prompt={prompt}
            setPrompt={setPrompt}
            runInference={runInference}
            isRunning={runningThreadId === activeThread?.id}
            sendBlocked={isRunning || !chatHistoryReady || chatHistoryBusy}
            model={selectedModelView}
            isEstablishingConnection={isEstablishingConnection}
            lastError={activeThread ? threadErrors[activeThread.id] ?? lastError : lastError}
            focusRequest={composerFocusRequest}
          />
        ) : null}

        {page === "image" ? (
          <ImagePage
            prompt={imagePrompt}
            setPrompt={setImagePrompt}
            focusRequest={imageFocusRequest}
            runtimeStatus={imageRuntimeStatus}
            generating={imageGenerating}
            generationStartedAt={imageGenerationStartedAt}
            result={imageResult}
            imageError={imageGenerationError}
            runImageGeneration={runImageGeneration}
          />
        ) : null}

        {page === "downloads" ? (
          <DownloadsPage
            snapshot={snapshot}
            transferActivities={transferActivities}
          />
        ) : null}

        {page === "network" ? (
          <NetworkPage snapshot={snapshot} />
        ) : null}

        {page === "activity" ? (
          <ActivityPage
            snapshot={snapshot}
            transferActivities={transferActivities}
            localNodeActivity={localNodeActivity}
            localJournal={localJournal}
          />
        ) : null}

        {page === "settings" ? (
          <SettingsPage
            snapshot={snapshot}
            imageRuntimeStatus={imageRuntimeStatus}
            appUpdater={appUpdater}
          />
        ) : null}

        {page === "about" ? <AboutPage /> : null}
      </main>
    </div>
  );
}

function UpdateBanner({ updater }: { updater: AppUpdaterState }) {
  if (updater.phase === "idle") return null;

  return (
    <aside className={`update-banner ${updater.phase}`} role={updater.phase === "error" ? "alert" : "status"}>
      <div>
        <strong>
          {updater.phase === "error"
            ? "Infernet couldn’t update"
            : updater.phase === "installing"
              ? "Installing update…"
              : `Infernet ${updater.version ?? "update"} is ready`}
        </strong>
        <span>
          {updater.phase === "error"
            ? updater.error
            : updater.phase === "installing"
              ? "Workers are stopping safely before Infernet restarts."
              : "The update is signed and will preserve your models, settings, and chats."}
        </span>
      </div>
      {updater.phase === "available" ? (
        <button type="button" onClick={() => void updater.installAndRestart()}>
          Install and restart
        </button>
      ) : null}
      {updater.phase === "error" ? (
        <button type="button" onClick={updater.dismissError}>Dismiss</button>
      ) : null}
    </aside>
  );
}

function RequiredModelsOnboarding({
  checking,
  chatInstalled,
  imageStatus,
  transfers,
  running,
  errors,
  onRetry,
}: {
  checking: boolean;
  chatInstalled: boolean;
  imageStatus: ImageRuntimeStatus | null;
  transfers: TransferActivity[];
  running: boolean;
  errors: RequiredSetupErrors;
  onRetry: () => void;
}) {
  const chatTransfer = transfers.find((item) => item.modelId === INFERNET_CHAT_MODEL_ID);
  const imageTransfer = transfers.find((item) => item.modelId === INFERNET_IMAGE_MODEL_ID);
  const imageInstalled = Boolean(imageStatus?.verified);
  const hasError = Boolean(errors.chat || errors.image || errors.status);

  return (
    <main className="required-setup-screen" aria-busy={checking || running}>
      <section className="required-setup-panel" aria-labelledby="required-setup-title">
        <div className="required-setup-heading">
          <span className="section-eyebrow">First-time setup</span>
          <h1 id="required-setup-title">Preparing Infernet</h1>
          <p>
            Chat and image models are required. Infernet downloads and verifies both packages
            before the rest of the app becomes available.
          </p>
        </div>

        <div className="required-model-list" aria-live="polite">
          <RequiredModelSetupRow
            icon={<MessageSquare size={20} />}
            name="Infernet Chat"
            packageNote="Official chat model"
            installed={chatInstalled}
            checking={checking}
            running={running}
            transfer={chatTransfer}
            error={errors.chat}
          />
          <RequiredModelSetupRow
            icon={<ImageIcon size={20} />}
            name="Infernet Image"
            packageNote="Official image model"
            installed={imageInstalled}
            checking={checking}
            running={running}
            transfer={imageTransfer}
            downloadedBytes={imageStatus?.downloadedBytes}
            totalBytes={imageStatus?.totalBytes}
            error={errors.image}
          />
        </div>

        <div className={hasError ? "required-setup-footer error" : "required-setup-footer"}>
          <div>
            <strong>
              {hasError
                ? "Setup needs another try"
                : checking
                  ? "Checking this computer"
                  : "Keep Infernet open while setup finishes"}
            </strong>
            <span>
              {errors.status
                ?? (hasError
                  ? "Completed downloads are kept, so retrying resumes where possible."
                  : "Each package is checked against its official pinned release.")}
            </span>
          </div>
          {hasError && !running ? (
            <button type="button" className="secondary-button" onClick={onRetry}>
              Try again
            </button>
          ) : null}
        </div>
      </section>
    </main>
  );
}

function RequiredModelSetupRow({
  icon,
  name,
  packageNote,
  installed,
  checking,
  running,
  transfer,
  downloadedBytes: fallbackDownloadedBytes = 0,
  totalBytes: fallbackTotalBytes = 0,
  error,
}: {
  icon: ReactNode;
  name: string;
  packageNote: string;
  installed: boolean;
  checking: boolean;
  running: boolean;
  transfer?: TransferActivity;
  downloadedBytes?: number;
  totalBytes?: number;
  error?: string;
}) {
  const downloadedBytes = transfer?.downloadedBytes ?? fallbackDownloadedBytes;
  const totalBytes = transfer?.totalBytes ?? fallbackTotalBytes;
  const progress = installed
    ? 100
    : totalBytes > 0
      ? Math.min(100, (downloadedBytes / totalBytes) * 100)
      : 0;
  const indeterminate = !installed && !error && (checking || running) && totalBytes === 0;
  const status = installed
    ? "Downloaded and verified"
    : error
      ? "Download paused"
      : checking
        ? "Checking this computer…"
        : transfer?.status === "active"
          ? humanTransferStage(transfer.stage)
          : running
            ? "Starting download…"
            : "Waiting to download";
  const detail = installed
    ? totalBytes > 0
      ? formatBytes(totalBytes)
      : packageNote
    : error
      ? imageErrorMessage(error)
      : totalBytes > 0
        ? `${formatProgressPercent(progress)}% · ${formatBytes(downloadedBytes)} of ${formatBytes(totalBytes)}`
        : packageNote;

  return (
    <article className={error ? "required-model-row error" : "required-model-row"}>
      <div className="required-model-icon" aria-hidden="true">{icon}</div>
      <div className="required-model-copy">
        <div className="required-model-title">
          <strong>{name}</strong>
          <span className={installed ? "model-state ready" : error ? "model-state error" : "model-state"}>
            <i aria-hidden="true" />
            {status}
          </span>
        </div>
        <ProgressBar progress={progress} indeterminate={indeterminate} />
        <small>{detail}</small>
      </div>
    </article>
  );
}

function Sidebar({
  threads,
  activeThreadId,
  chatIsVisible,
  primaryMode,
  runningThreadId,
  imageGenerationBusy,
  imageCreations,
  activeImageId,
  disabled,
  persistenceError,
  onCreateThread,
  onStartImage,
  onOpenImage,
  onModeChange,
  onOpenThread,
  onDeleteThread,
}: {
  threads: ChatThread[];
  activeThreadId: string;
  chatIsVisible: boolean;
  primaryMode: PrimaryMode;
  runningThreadId: string | null;
  imageGenerationBusy: boolean;
  imageCreations: GenerateImageResponse[];
  activeImageId: string | null;
  disabled: boolean;
  persistenceError: string | null;
  onCreateThread: () => Promise<void>;
  onStartImage: () => void;
  onOpenImage: (creation: GenerateImageResponse) => void;
  onModeChange: (mode: PrimaryMode) => void;
  onOpenThread: (threadId: string) => Promise<void>;
  onDeleteThread: (threadId: string) => Promise<void>;
}) {
  const [confirmingDeleteId, setConfirmingDeleteId] = useState<string | null>(null);
  const [deletingThreadId, setDeletingThreadId] = useState<string | null>(null);
  const deleteButtonRefs = useRef(new Map<string, HTMLButtonElement>());
  const threadButtonRefs = useRef(new Map<string, HTMLButtonElement>());

  const restoreDeleteButtonFocus = (threadId: string) => {
    window.requestAnimationFrame(() => deleteButtonRefs.current.get(threadId)?.focus());
  };

  const restoreThreadFocus = () => {
    window.requestAnimationFrame(() => {
      const activeButton = threadButtonRefs.current.get(activeThreadId);
      const fallbackButton = threadButtonRefs.current.values().next().value;
      (activeButton ?? fallbackButton)?.focus();
    });
  };

  useEffect(() => {
    if (confirmingDeleteId && !threads.some((thread) => thread.id === confirmingDeleteId)) {
      setConfirmingDeleteId(null);
    }
  }, [confirmingDeleteId, threads]);

  return (
    <aside className="sidebar" aria-label="Infernet navigation and history">
      <div className="sidebar-brand">
        <div className="brand-block">
          <svg
            className="brand-logo"
            viewBox="230 250 564 524"
            role="img"
            aria-label="Infernet"
          >
            <circle cx="512" cy="512" r="84" fill="currentColor" />
            <circle cx="312" cy="332" r="62" fill="currentColor" />
            <circle cx="712" cy="332" r="62" fill="currentColor" />
            <circle cx="312" cy="692" r="62" fill="currentColor" />
            <circle cx="712" cy="692" r="62" fill="currentColor" />
            <path d="M366 360L458 478" stroke="currentColor" strokeWidth="44" strokeLinecap="round" />
            <path d="M658 360L566 478" stroke="currentColor" strokeWidth="44" strokeLinecap="round" />
            <path d="M366 664L458 546" stroke="currentColor" strokeWidth="44" strokeLinecap="round" />
            <path d="M658 664L566 546" stroke="currentColor" strokeWidth="44" strokeLinecap="round" />
            <path d="M374 332H650" stroke="currentColor" strokeWidth="38" strokeLinecap="round" />
            <path d="M374 692H650" stroke="currentColor" strokeWidth="38" strokeLinecap="round" />
          </svg>
        </div>
        <strong>Infernet</strong>
      </div>

      <div className="mode-switcher" role="group" aria-label="Mode">
        <button
          type="button"
          className={primaryMode === "chat" ? "active" : undefined}
          aria-pressed={primaryMode === "chat"}
          title="Chat"
          onClick={() => onModeChange("chat")}
        >
          <MessageSquare size={16} />
          <span>Chat</span>
        </button>
        <button
          type="button"
          className={`${primaryMode === "image" ? "active " : ""}${imageGenerationBusy ? "mode-image-working" : ""}`.trim() || undefined}
          aria-pressed={primaryMode === "image"}
          aria-label={imageGenerationBusy ? "Image, generating" : "Image"}
          title={imageGenerationBusy ? "Image generation in progress" : "Image"}
          onClick={() => onModeChange("image")}
        >
          {imageGenerationBusy ? <LoaderCircle size={16} /> : <ImageIcon size={16} />}
          <span>Image</span>
        </button>
      </div>

      <button
        className="new-thread-button"
        type="button"
        aria-label={primaryMode === "chat" ? "New chat" : imageGenerationBusy ? "Image generation in progress" : "New image"}
        title={primaryMode === "chat" ? "New chat" : imageGenerationBusy ? "Wait for the current image to finish" : "New image"}
        disabled={primaryMode === "chat" ? disabled : imageGenerationBusy}
        onClick={() => {
          if (primaryMode === "chat") {
            void onCreateThread();
          } else {
            onStartImage();
          }
        }}
      >
        <Plus size={17} />
        <span>{primaryMode === "chat" ? "New chat" : imageGenerationBusy ? "Image in progress" : "New image"}</span>
      </button>

      {primaryMode === "chat" ? (
        <>
          <div className="thread-list-heading">Chats</div>
          <nav className="thread-nav" aria-label="Chat threads" aria-busy={disabled}>
            <ul className="thread-list">
              {threads.map((thread) => {
                const active = thread.id === activeThreadId;
                const isRunning = thread.id === runningThreadId;
                const confirmingDelete = thread.id === confirmingDeleteId;
                const deleting = thread.id === deletingThreadId;

                return (
                  <li
                    className={active ? "thread-list-item active" : "thread-list-item"}
                    key={thread.id}
                  >
                    {confirmingDelete ? (
                      <div
                        className="thread-delete-confirm"
                        role="group"
                        aria-label={`Delete ${thread.title}?`}
                      >
                        <span>Delete this chat?</span>
                        <div>
                          <button
                            type="button"
                            className="thread-confirm-cancel"
                            autoFocus
                            disabled={disabled || deleting}
                            onClick={() => {
                              setConfirmingDeleteId(null);
                              restoreDeleteButtonFocus(thread.id);
                            }}
                          >
                            Cancel
                          </button>
                          <button
                            type="button"
                            className="thread-confirm-delete"
                            disabled={disabled || deleting}
                            onClick={async () => {
                              if (deleting) return;
                              setDeletingThreadId(thread.id);
                              await onDeleteThread(thread.id);
                              setDeletingThreadId(null);
                              setConfirmingDeleteId(null);
                              restoreThreadFocus();
                            }}
                          >
                            Delete
                          </button>
                        </div>
                      </div>
                    ) : (
                      <>
                        <button
                          ref={(element) => {
                            if (element) threadButtonRefs.current.set(thread.id, element);
                            else threadButtonRefs.current.delete(thread.id);
                          }}
                          type="button"
                          className="thread-select-button"
                          disabled={disabled}
                          aria-current={active && chatIsVisible ? "page" : undefined}
                          onClick={() => void onOpenThread(thread.id)}
                        >
                          <span>{thread.title}</span>
                          {isRunning ? (
                            <i
                              className="thread-running-indicator"
                              aria-label="Generating response"
                            />
                          ) : null}
                        </button>
                        <button
                          ref={(element) => {
                            if (element) deleteButtonRefs.current.set(thread.id, element);
                            else deleteButtonRefs.current.delete(thread.id);
                          }}
                          type="button"
                          className="thread-delete-button"
                          disabled={disabled || isRunning}
                          aria-label={isRunning
                            ? `Wait for ${thread.title} to finish before deleting`
                            : `Delete ${thread.title}`}
                          title={isRunning
                            ? "This chat is still responding"
                            : `Delete ${thread.title}`}
                          onClick={() => setConfirmingDeleteId(thread.id)}
                        >
                          <Trash2 size={15} />
                        </button>
                      </>
                    )}
                  </li>
                );
              })}
            </ul>
          </nav>
        </>
      ) : (
        <div className="sidebar-mode-empty">
          <div className="thread-list-heading">Creations</div>
          {imageCreations.length > 0 ? (
            <ul className="creation-list" aria-label="Generated images">
              {imageCreations.map((creation) => (
                <li key={creation.imageId}>
                  <button
                    type="button"
                    className={creation.imageId === activeImageId ? "active" : undefined}
                    aria-current={creation.imageId === activeImageId ? "page" : undefined}
                    aria-label={`Open creation: ${creation.prompt}`}
                    title={creation.prompt}
                    onClick={() => onOpenImage(creation)}
                  >
                    <img src={creation.imageDataUrl} alt="" />
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <div className="creation-empty-state">
              <ImageIcon size={17} aria-hidden="true" />
              <span>Your generated images will appear here.</span>
            </div>
          )}
        </div>
      )}

      {primaryMode === "chat" && persistenceError ? (
        <p className="sidebar-storage-error" role="alert">{persistenceError}</p>
      ) : null}
    </aside>
  );
}

function AppHeader({
  page,
  chatTitle,
  networkNodeCount,
  networkReadyCount,
  hasActiveWork,
  onNavigate,
}: {
  page: Page;
  chatTitle: string;
  networkNodeCount: number;
  networkReadyCount: number;
  hasActiveWork: boolean;
  onNavigate: (page: Page) => void;
}) {
  return (
    <header className="app-header">
      <div>
        <h1>{pageTitle(page, chatTitle)}</h1>
        {page === "network" ? (
          <div className="header-meta">
            <span>
              {networkNodeCount > 0
                ? `${networkNodeCount} node${networkNodeCount === 1 ? "" : "s"} visible · ${networkReadyCount} compute-ready`
                : "Discovering network compute"}
            </span>
          </div>
        ) : null}
      </div>

      <div className="header-actions">
        <HeaderIconButton
          icon={<Globe size={17} />}
          label="Network"
          active={page === "network"}
          onClick={() => onNavigate("network")}
        />
        <button
          className={page === "activity" ? "activity-toggle active" : "activity-toggle"}
          type="button"
          aria-label={hasActiveWork ? "Activity, active work" : "Activity"}
          aria-current={page === "activity" ? "page" : undefined}
          title="Activity"
          onClick={() => onNavigate("activity")}
        >
          <Activity size={16} />
          {hasActiveWork ? <i aria-hidden="true" /> : null}
        </button>
        <HeaderIconButton
          icon={<CircleHelp size={17} />}
          label="Help"
          active={page === "about"}
          onClick={() => onNavigate("about")}
        />
        <HeaderIconButton
          icon={<Settings size={17} />}
          label="Settings"
          active={page === "settings"}
          onClick={() => onNavigate("settings")}
        />
      </div>
    </header>
  );
}

function HeaderIconButton({
  icon,
  label,
  active,
  onClick,
}: {
  icon: React.ReactNode;
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      className={active ? "header-icon-button active" : "header-icon-button"}
      type="button"
      aria-label={label}
      aria-current={active ? "page" : undefined}
      title={label}
      onClick={onClick}
    >
      {icon}
    </button>
  );
}

function ChatPage({
  messages,
  prompt,
  setPrompt,
  runInference,
  isRunning,
  sendBlocked,
  model,
  isEstablishingConnection,
  lastError,
  focusRequest,
}: {
  messages: ChatMessage[];
  prompt: string;
  setPrompt: (prompt: string) => void;
  runInference: () => void;
  isRunning: boolean;
  sendBlocked: boolean;
  model?: ModelView;
  isEstablishingConnection: boolean;
  lastError: string | null;
  focusRequest: number;
}) {
  const conversationRef = useRef<HTMLDivElement>(null);
  const composerInputRef = useRef<HTMLTextAreaElement>(null);
  const canSend = Boolean(model?.runnable);
  const isEmpty = messages.length === 0;
  const responseIsUnsaved = lastError === UNSAVED_RESPONSE_ERROR;

  useEffect(() => {
    const conversation = conversationRef.current;
    if (conversation) {
      conversation.scrollTop = conversation.scrollHeight;
    }
  }, [messages, isRunning, lastError]);

  useLayoutEffect(() => {
    const input = composerInputRef.current;
    if (!input) return;
    resizeComposerInput(input);
  }, [prompt]);

  useLayoutEffect(() => {
    const input = composerInputRef.current;
    if (!input || typeof ResizeObserver === "undefined") return;
    let previousWidth = input.clientWidth;
    const observer = new ResizeObserver(([entry]) => {
      const nextWidth = entry.contentRect.width;
      if (nextWidth === previousWidth) return;
      previousWidth = nextWidth;
      resizeComposerInput(input);
    });
    observer.observe(input);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (focusRequest > 0 && canSend) {
      composerInputRef.current?.focus();
    }
  }, [canSend, focusRequest]);

  return (
    <section className="chat-screen">
      <div className={isEmpty ? "conversation empty" : "conversation"} ref={conversationRef}>
        <div className="conversation-inner">
          {isEmpty ? (
            <div className="empty-chat-hero">
              <span>Infernet</span>
              <h2>{timeGreeting()}</h2>
            </div>
          ) : null}

          {messages.map((message) => (
            <div key={message.id} className={`message-row ${message.role}`}>
              <div className="message-bubble">
                {message.role === "assistant" ? (
                  <MarkdownMessage text={message.text} />
                ) : message.text}
              </div>
            </div>
          ))}

          {isEstablishingConnection ? (
            <div className="connection-establishing" role="status" aria-live="polite">
              <i className="connection-throbber" aria-hidden="true" />
              <strong>Establishing Connection</strong>
            </div>
          ) : !model ? (
            <div className="empty-chat-card">
              <strong>Get Infernet Chat to start</strong>
              <span>The official Infernet model is not available on the network yet.</span>
            </div>
          ) : !model.runnable ? (
            <div className="empty-chat-card warning">
              <strong>{curatedModelName(model)} is not ready yet</strong>
              <span>{model.status}</span>
            </div>
          ) : null}

          {isRunning ? <ThinkingIndicator /> : null}

          {lastError && messages.length > 0 && !isRunning ? (
            <div className="chat-error" role="alert">
              <strong>{responseIsUnsaved
                ? "This response isn’t saved."
                : "Infernet couldn’t finish that response."}</strong>
              <span>{responseIsUnsaved ? lastError : friendlyActivityError(lastError)}</span>
            </div>
          ) : null}
        </div>
      </div>

      <div className="composer-dock">
        <div className="composer">
          <textarea
            ref={composerInputRef}
            rows={1}
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                runInference();
              }
            }}
            placeholder={canSend
              ? "Message Infernet"
              : isEstablishingConnection
                ? "Establishing connection"
                : model
                  ? "Model is not ready"
                  : "Infernet Chat is unavailable"}
            disabled={!canSend}
            aria-label="Message Infernet"
          />
          <button
            className="send-button"
            aria-label="Send message"
            onClick={runInference}
            disabled={sendBlocked || !prompt.trim() || !canSend}
          >
            {isRunning ? <Activity size={18} /> : <Send size={18} />}
            <span>Send</span>
          </button>
        </div>
      </div>
    </section>
  );
}

function ImagePage({
  prompt,
  setPrompt,
  focusRequest,
  runtimeStatus,
  generating,
  generationStartedAt,
  result,
  imageError,
  runImageGeneration,
}: {
  prompt: string;
  setPrompt: (value: string) => void;
  focusRequest: number;
  runtimeStatus: ImageRuntimeStatus | null;
  generating: boolean;
  generationStartedAt: number | null;
  result: GenerateImageResponse | null;
  imageError: string | null;
  runImageGeneration: () => Promise<void>;
}) {
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const elapsedMs = useElapsedTime(generating ? generationStartedAt : null);

  useLayoutEffect(() => {
    const input = inputRef.current;
    if (!input) return;
    resizeComposerInput(input);
  }, [prompt]);

  useLayoutEffect(() => {
    const input = inputRef.current;
    if (!input || typeof ResizeObserver === "undefined") return;
    let previousWidth = input.clientWidth;
    const observer = new ResizeObserver(([entry]) => {
      const nextWidth = entry.contentRect.width;
      if (nextWidth === previousWidth) return;
      previousWidth = nextWidth;
      resizeComposerInput(input);
    });
    observer.observe(input);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (focusRequest <= 0) return;
    inputRef.current?.focus();
  }, [focusRequest]);

  const operationBusy = generating || Boolean(runtimeStatus?.busy);
  const canGenerate = Boolean(
    runtimeStatus?.runtimeAvailable && runtimeStatus.verified && !operationBusy,
  );
  const runtimeNote = generating
    ? `Generating a 1024 × 1024 image · ${formatDuration(elapsedMs)} elapsed`
    : runtimeStatus?.status ?? "Infernet Image is unavailable.";

  return (
    <section className="image-screen" aria-busy={operationBusy}>
      <div className="image-workspace">
        {result ? (
          <figure className="image-result">
            <div className="image-result-frame">
              <img
                src={result.imageDataUrl}
                alt={`Generated image for: ${result.prompt.slice(0, 180)}`}
                decoding="async"
              />
            </div>
            <figcaption>
              <p>{result.prompt}</p>
              <dl className="image-result-meta">
                <div>
                  <dt>Seed</dt>
                  <dd>{result.detailsAvailable ? result.seed : "Unavailable"}</dd>
                </div>
                <div>
                  <dt>Size</dt>
                  <dd>{result.width} × {result.height}</dd>
                </div>
                <div>
                  <dt>Steps</dt>
                  <dd>{result.detailsAvailable ? result.steps : "Unavailable"}</dd>
                </div>
                <div>
                  <dt>Time</dt>
                  <dd>{result.detailsAvailable ? formatDuration(result.durationMs) : "Unavailable"}</dd>
                </div>
              </dl>
            </figcaption>
          </figure>
        ) : (
          <div className="empty-image-hero" aria-live="polite">
            <div className={generating ? "image-mode-mark working" : "image-mode-mark"} aria-hidden="true">
              {generating ? <LoaderCircle size={24} /> : <ImageIcon size={24} />}
            </div>
            <span>Infernet Image</span>
            <h2>{generating ? "Creating your image" : "What do you want to make?"}</h2>
            <p>
              {generating
                ? "You can visit another screen. Come back here to follow progress while Infernet finishes."
                : "Describe a scene, subject, or style. Infernet Image will use the official Z‑Image Turbo edition."}
            </p>
            {generating ? (
              <div className="image-generation-progress">
                <ProgressBar progress={0} indeterminate />
                <span>{formatDuration(elapsedMs)} elapsed</span>
              </div>
            ) : null}
          </div>
        )}

        {imageError ? (
          <div className="chat-error image-error" role="alert">
            <strong>Infernet Image couldn’t finish.</strong>
            <span>{imageError}</span>
          </div>
        ) : null}
      </div>

      {result ? null : <div className="image-composer-dock">
        <div className="composer image-composer">
          <textarea
            ref={inputRef}
            rows={1}
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            disabled={operationBusy}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                event.preventDefault();
                void runImageGeneration();
              }
            }}
            maxLength={4000}
            placeholder="Describe an image"
            aria-label="Describe an image"
            aria-describedby="image-runtime-note"
          />
          <button
            className={generating ? "send-button image-generating" : "send-button"}
            type="button"
            disabled={!canGenerate || !prompt.trim()}
            aria-label="Generate image"
            title={runtimeStatus?.verified ? undefined : runtimeStatus?.status}
            onClick={() => void runImageGeneration()}
          >
            {generating ? <LoaderCircle size={18} /> : <ImageIcon size={18} />}
            <span>{generating ? "Generating…" : "Generate"}</span>
          </button>
        </div>
        <p id="image-runtime-note" className="image-runtime-note" role="status" aria-live="polite">
          {runtimeNote}
        </p>
      </div>}
    </section>
  );
}

function imageErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function resizeComposerInput(input: HTMLTextAreaElement) {
  input.style.height = "0px";
  const nextHeight = Math.min(input.scrollHeight, COMPOSER_MAX_HEIGHT);
  input.style.height = `${nextHeight}px`;
  input.style.overflowY = input.scrollHeight > COMPOSER_MAX_HEIGHT ? "auto" : "hidden";
}

function ThinkingIndicator() {
  return (
    <div className="message-row assistant thinking-row" role="status" aria-live="polite">
      <div className="thinking-indicator">
        <span className="sr-only">Infernet is thinking</span>
        <i aria-hidden="true" />
        <i aria-hidden="true" />
        <i aria-hidden="true" />
      </div>
    </div>
  );
}

function MarkdownMessage({ text }: { text: string }) {
  return (
    <div className="markdown-message">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={MARKDOWN_COMPONENTS}
        skipHtml
        disallowedElements={["img"]}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}

function ActivityPage({
  snapshot,
  transferActivities,
  localNodeActivity,
  localJournal,
}: {
  snapshot: GridSnapshot;
  transferActivities: TransferActivity[];
  localNodeActivity: LocalNodeActivitySnapshot;
  localJournal: NodeJournalEntry[];
}) {
  const localMachine = snapshot.machines.find((machine) => machine.isLocal);
  const activeTransfer = transferActivities.find((activity) => activity.status === "active");
  const currentTask = localNodeActivity.current[0];
  const computeActive = localNodeActivity.computeActive || Boolean(localMachine?.activeSessions);
  const computeReady = localNodeActivity.computeReady || Boolean(localMachine?.rpcReady);
  const sharingActive = localNodeActivity.sharingActive || snapshot.distribution.currentUploads > 0;
  const modelStored = snapshot.distribution.installedModels.includes(INFERNET_CHAT_MODEL_ID);
  const deviceName = localNodeActivity.deviceName !== EMPTY_LOCAL_NODE_ACTIVITY.deviceName
    ? localNodeActivity.deviceName
    : localMachine?.deviceName ?? "This computer";
  const computeBackend = localNodeActivity.computeBackend !== "cpu"
    ? localNodeActivity.computeBackend
    : localMachine?.computeBackend ?? localNodeActivity.computeBackend;
  const availableMemoryBytes = localNodeActivity.totalMemoryBytes > 0
    ? localNodeActivity.availableMemoryBytes
    : localMachine?.availableMemoryBytes ?? 0;
  const totalMemoryBytes = localNodeActivity.totalMemoryBytes > 0
    ? localNodeActivity.totalMemoryBytes
    : localMachine?.totalMemoryBytes ?? 0;
  const isStarting = !localMachine && !snapshot.localPeerId;
  const isWorking = Boolean(activeTransfer || currentTask || computeActive || sharingActive);
  const currentWork = activeTransfer
    ? {
        title: humanTransferStage(activeTransfer.stage),
        detail: transferStageDescription(activeTransfer.stage, activeTransfer.status),
      }
    : currentTask?.kind === "imageGeneration"
      ? {
          title: "Creating an image",
          detail: "Your computer is running the verified Infernet Image package.",
        }
      : currentTask?.kind === "chatCompletion"
      ? {
          title: "Fulfilling a chat completion",
          detail: "Your node is coordinating this response.",
        }
      : currentTask?.kind === "computeContribution" || computeActive
        ? {
            title: "Contributing compute",
            detail: "Your node is processing work for the network.",
          }
        : sharingActive
          ? {
              title: "Sharing Infernet Chat",
              detail: "Sending verified model data to another node.",
            }
          : isStarting
            ? {
                title: "Starting your node",
                detail: "Bringing local services online.",
              }
            : computeReady
              ? {
                  title: "Ready to help",
                  detail: "Standing by for work from the network.",
                }
              : {
                  title: "Online",
                  detail: "Connected to Infernet. Compute is not available right now.",
                };
  const runtimeJournal: NodeJournalEntry[] = localNodeActivity.journal.map((entry) => {
    const duration = formatDuration(entry.completedAtUnixMs - entry.startedAtUnixMs);
    if (entry.outcome === "error") {
      return {
        id: entry.id,
        kind: "error",
        title: entry.kind === "chatCompletion"
          ? "A chat completion couldn’t finish"
          : entry.kind === "imageGeneration"
            ? "An image couldn’t finish"
            : "A compute task couldn’t finish",
        detail: duration === "—" ? undefined : `Your node worked for ${duration}.`,
        occurredAt: entry.completedAtUnixMs,
      };
    }
    if (entry.kind === "imageGeneration") {
      return {
        id: entry.id,
        kind: "image",
        title: "You created an image",
        detail: duration === "—" ? undefined : `Completed in ${duration}.`,
        occurredAt: entry.completedAtUnixMs,
      };
    }
    return {
      id: entry.id,
      kind: entry.kind === "chatCompletion" ? "completion" : "contribution",
      title: entry.kind === "chatCompletion"
        ? "You fulfilled a chat completion"
        : "You contributed compute",
      detail: duration === "—"
        ? undefined
        : entry.kind === "chatCompletion"
          ? `Completed in ${duration}.`
          : `Your part finished in ${duration}.`,
      occurredAt: entry.completedAtUnixMs,
    };
  });
  const journal = [...runtimeJournal, ...localJournal]
    .filter((entry, index, entries) => entries.findIndex((item) => item.id === entry.id) === index)
    .sort((left, right) => left.occurredAt - right.occurredAt)
    .slice(-50);

  return (
    <section className="activity-screen" aria-label="Your node activity">
      <div className="activity-page-content">
        <header className="activity-page-header">
          <div>
            <span>Your node</span>
            <h2>{deviceName}</h2>
          </div>
        </header>

        <div className="activity-page-grid">
          <section className={isWorking ? "node-hud working" : "node-hud"} aria-live="polite">
            <div className="node-current-work">
              <span className={isWorking ? "activity-pulse active" : "activity-pulse"} />
              <div>
                <span className="node-now-label">Now</span>
                <strong>{currentWork.title}</strong>
                <p>{currentWork.detail}</p>
              </div>
            </div>

            {activeTransfer ? <MachineTransferProgress activity={activeTransfer} /> : null}

            <dl className="node-facts">
              <ActivityDataRow
                label="Compute"
                value={computeActive
                  ? "In use"
                  : computeReady
                    ? `${machineBackendLabel(computeBackend)} ready`
                    : "Unavailable"}
              />
              <ActivityDataRow
                label="Memory"
                value={totalMemoryBytes > 0
                  ? `${formatBytes(availableMemoryBytes)} free of ${formatBytes(totalMemoryBytes)}`
                  : "Checking"}
              />
              <ActivityDataRow
                label="Model"
                value={activeTransfer
                  ? "Preparing locally"
                  : modelStored
                    ? "Stored and shareable"
                    : "Compute only"}
              />
              <ActivityDataRow label="Network" value={isStarting ? "Starting" : "Connected"} />
            </dl>
          </section>

          <section className="node-journal" aria-labelledby="node-journal-title">
            <div className="node-journal-heading">
              <strong id="node-journal-title">Journal</strong>
              <span>This session</span>
            </div>
            {journal.length === 0 ? (
              <div className="node-journal-empty">
                <Activity size={17} />
                <span>Your node’s completed work will appear here.</span>
              </div>
            ) : (
              <ol className="node-journal-list" aria-live="polite" aria-relevant="additions">
                {journal.map((entry) => (
                  <li className={`node-journal-entry ${entry.kind}`} key={entry.id}>
                    <span className="node-journal-marker" aria-hidden="true">
                      <NodeJournalIcon kind={entry.kind} />
                    </span>
                    <div>
                      <strong>{entry.title}</strong>
                      {entry.detail ? <p>{entry.detail}</p> : null}
                      <time dateTime={new Date(entry.occurredAt).toISOString()}>
                        {formatJournalTime(entry.occurredAt)}
                      </time>
                    </div>
                  </li>
                ))}
              </ol>
            )}
          </section>
        </div>
      </div>
    </section>
  );
}

function NodeJournalIcon({ kind }: { kind: NodeJournalEntry["kind"] }) {
  if (kind === "completion") return <MessageSquare size={13} />;
  if (kind === "image") return <ImageIcon size={13} />;
  if (kind === "model") return <Download size={13} />;
  if (kind === "sharing") return <Server size={13} />;
  if (kind === "error") return <Activity size={13} />;
  return <Zap size={13} />;
}

function ActivityDataRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="activity-data-row">
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function MachineStatusCard({
  machine,
  localTransfer,
}: {
  machine: MachineView;
  localTransfer?: TransferActivity;
}) {
  const reconnecting = machine.connectionStatus === "reconnecting";
  const unreachable = machine.connectionStatus === "unreachable";
  const connectionState = unreachable
    ? { className: "error", label: "Unreachable" }
    : reconnecting
      ? { className: "waiting", label: "Reconnecting" }
      : { className: "connected", label: "Connected" };
  const serving = machine.activeSessions > 0;
  const busy = machine.maxSessions > 0 && machine.activeSessions >= machine.maxSessions;
  const supportedBackend = machine.computeBackend === "cuda" || machine.computeBackend === "metal";
  const computeState = unreachable
    ? { className: "error", label: "Compute offline" }
    : reconnecting
    ? { className: "waiting", label: "Compute last seen" }
    : serving
      ? { className: "serving", label: "Serving a request" }
      : machine.rpcReady
        ? { className: "ready", label: "Compute ready" }
        : busy
          ? { className: "waiting", label: "Compute busy" }
          : supportedBackend
            ? { className: "error", label: "Compute unavailable" }
            : { className: "muted", label: "No supported GPU" };
  const modelState = localTransfer
    ? { className: "waiting", label: "Downloading model" }
    : unreachable && machine.hostedComponentCount > 0
      ? { className: "error", label: "Model host offline" }
    : machine.hostedComponentCount > 0
      ? { className: "ready", label: "Hosting and sharing" }
      : { className: "muted", label: "Model unavailable" };
  const modelDetail = localTransfer
    ? "The verified package will be shared after the download finishes."
    : unreachable && machine.hostedComponentCount > 0
      ? "This computer last reported the verified package, but it cannot be reached right now."
    : machine.hostedComponentCount > 0
      ? "Infernet Chat is verified and available for other computers to download while Infernet stays open."
      : supportedBackend
        ? "This computer has not completed the required model setup."
        : "This computer can discover the network, but it cannot run a distributed model segment.";
  return (
    <div className="machine-row">
      <div className="machine-row-main">
        <span className={`machine-backend ${machine.computeBackend}`}>
          {machineBackendLabel(machine.computeBackend)}
        </span>
        <div>
          <strong>{machine.isLocal ? "This computer" : machine.deviceName}</strong>
          <span>{machine.isLocal ? machine.deviceName : `${machine.logicalCpuCores} CPU cores`}</span>
        </div>
      </div>

      <div className="machine-state-list" aria-label={`${machine.deviceName} status`}>
        <span className={`machine-state ${connectionState.className}`}>
          <i aria-hidden="true" />
          {connectionState.label}
        </span>
        <span className={`machine-state ${computeState.className}`}>
          <i aria-hidden="true" />
          {computeState.label}
        </span>
        <span className={`machine-state ${modelState.className}`}>
          <i aria-hidden="true" />
          {modelState.label}
        </span>
      </div>

      <p className="machine-model-detail">{modelDetail}</p>

      {localTransfer ? (
        <MachineTransferProgress activity={localTransfer} />
      ) : null}

      <div className="machine-capacity">
        <strong>
          {formatBytes(machine.availableMemoryBytes)} free of {formatBytes(machine.totalMemoryBytes)}
        </strong>
        <span>{machineLoadLabel(machine.activeSessions, machine.maxSessions, machine.queueDepth)}</span>
      </div>
    </div>
  );
}

function MachineTransferProgress({ activity }: { activity: TransferActivity }) {
  const percent = activity.totalBytes && activity.downloadedBytes > 0
    ? Math.min(100, (activity.downloadedBytes / activity.totalBytes) * 100)
    : null;

  return (
    <div className="machine-transfer-progress">
      <ProgressBar progress={percent ?? 0} indeterminate={percent === null} />
      <small>
        {percent === null
          ? humanTransferStage(activity.stage)
          : `${formatProgressPercent(percent)}% · ${formatBytes(activity.downloadedBytes)} of ${formatBytes(activity.totalBytes ?? 0)}`}
      </small>
    </div>
  );
}

function TransferActivityRow({
  activity,
  modelName,
}: {
  activity: TransferActivity;
  modelName: string;
}) {
  const elapsedMs = useElapsedTime(activity.status === "active" ? activity.startedAt : null);
  const percent = activity.totalBytes && (activity.downloadedBytes > 0 || activity.status !== "active")
    ? Math.min(100, (activity.downloadedBytes / activity.totalBytes) * 100)
    : null;
  const stage = humanTransferStage(activity.stage);

  return (
    <div className={`transfer-row ${activity.status}`}>
      <div className="transfer-row-heading">
        <div>
          <strong>{modelName}</strong>
          <span>{stage}</span>
        </div>
        <small>
          {activity.status === "active"
            ? formatDuration(elapsedMs)
            : formatRelativeTime(activity.updatedAt)}
        </small>
      </div>
      <p>{transferStageDescription(activity.stage, activity.status)}</p>
      {activity.status === "active" || percent !== null ? (
        <div className="transfer-progress">
          <ProgressBar progress={percent ?? 0} indeterminate={percent === null && activity.status === "active"} />
          <small>
            {percent !== null
              ? `${formatProgressPercent(percent)}% · ${formatBytes(activity.downloadedBytes)} of ${formatBytes(activity.totalBytes ?? 0)}`
              : activity.status === "active" ? "Working" : stage}
          </small>
        </div>
      ) : null}
    </div>
  );
}

type NodeLocation = {
  latitude: number;
  longitude: number;
  label: string;
  source: "relay" | "illustrative";
};

type GeoPoint = [longitude: number, latitude: number];

function NetworkPage({ snapshot }: { snapshot: GridSnapshot }) {
  const onlineMachines = snapshot.machines.filter(
    (machine) => machine.connectionStatus === "connected",
  );
  const acceleratorMachines = onlineMachines.filter(
    (machine) => machine.computeBackend === "cuda" || machine.computeBackend === "metal",
  );
  const readyWorkers = onlineMachines.filter((machine) => machine.rpcReady);
  const modelHosts = onlineMachines.filter((machine) => machine.hostedComponentCount > 0);
  const allocatedComputeBytes = acceleratorMachines.reduce(
    (total, machine) => total + machine.allocatedMemoryBytes,
    0,
  );
  const activeSessions = onlineMachines.reduce((total, machine) => total + machine.activeSessions, 0);
  const sessionCapacity = onlineMachines.reduce((total, machine) => total + machine.maxSessions, 0);
  const queuedRequests = onlineMachines.reduce((total, machine) => total + machine.queueDepth, 0);
  const logicalCores = onlineMachines.reduce((total, machine) => total + machine.logicalCpuCores, 0);
  const coveredLayers = snapshot.coverage.filter((segment) => segment.covered).length;
  const coveragePercent = snapshot.coverage.length > 0
    ? Math.round((coveredLayers / snapshot.coverage.length) * 100)
    : 0;
  const backendSummaries = ["cuda", "metal"].map((backend) => {
    const machines = onlineMachines.filter((machine) => machine.computeBackend === backend);
    return {
      backend,
      machines,
      allocatedBytes: machines.reduce((total, machine) => total + machine.allocatedMemoryBytes, 0),
    };
  }).filter((summary) => summary.machines.length > 0);
  const locations = useMemo(
    () => Object.fromEntries(onlineMachines.map((machine) => [
      networkMachineKey(machine),
      machine.coarseLocation
        ? { ...machine.coarseLocation, source: "relay" as const }
        : illustrativeNodeLocation(machine),
    ])),
    [onlineMachines],
  );
  const locatedNodeCount = Object.values(locations).filter(
    (location) => location.source === "relay",
  ).length;

  return (
    <section className="network-screen">
      <div className="network-layout">
        <NetworkGlobe
          machines={onlineMachines}
          locations={locations}
          locatedNodeCount={locatedNodeCount}
          isLocating={false}
        />

        <section className="network-pulse" aria-labelledby="network-pulse-title">
          <div className="network-kicker">
            <i aria-hidden="true" />
            <span>Live network</span>
          </div>
          <h2 id="network-pulse-title">
            {onlineMachines.length > 1
              ? "Compute, shared in real time"
              : onlineMachines.length === 1
                ? "This computer is on the network"
                : "Waiting for the network"}
          </h2>
          <p>
            {onlineMachines.length > 0
              ? "A clear view of the accelerator capacity committed to Infernet."
              : "Online computers and their allocated capacity will appear here as Infernet discovers them."}
          </p>

          <dl className="network-capacity-ledger">
            <div className="network-memory-fact">
              <dt>
                <MemoryStick size={17} />
                <span>VRAM + unified memory allocated</span>
              </dt>
              <dd>
                <strong>{formatBytes(allocatedComputeBytes)}</strong>
              </dd>
            </div>
            <NetworkPulseRow
              icon={<Server size={16} />}
              label="Nodes online"
              value={String(onlineMachines.length)}
              detail={onlineMachines.length === 1 ? "This computer" : `${snapshot.networkPeerCount} remote`}
            />
            <NetworkPulseRow
              icon={<Zap size={16} />}
              label="Compute-ready"
              value={String(readyWorkers.length)}
              detail={`${Math.max(0, sessionCapacity - activeSessions)} session slot${Math.max(0, sessionCapacity - activeSessions) === 1 ? "" : "s"} free`}
            />
            <NetworkPulseRow
              icon={<Layers3 size={16} />}
              label="Model coverage"
              value={snapshot.coverage.length > 0 ? `${coveragePercent}%` : "—"}
              detail={snapshot.coverage.length > 0 ? `${coveredLayers} of ${snapshot.coverage.length} layers` : "No route selected"}
            />
          </dl>
        </section>

        <section className="network-capacity-breakdown" aria-labelledby="capacity-breakdown-title">
          <div className="network-section-heading">
            <div>
              <span>Capacity</span>
              <h3 id="capacity-breakdown-title">Compute by runtime</h3>
            </div>
            <small>{acceleratorMachines.length} accelerator node{acceleratorMachines.length === 1 ? "" : "s"}</small>
          </div>

          {backendSummaries.length > 0 ? (
            <div className="network-backend-list">
              {backendSummaries.map((summary) => (
                  <div className="network-backend-row" key={summary.backend}>
                    <div className="network-backend-copy">
                      <span className="machine-backend">{machineBackendLabel(summary.backend)}</span>
                      <div>
                        <strong>{summary.machines.length} node{summary.machines.length === 1 ? "" : "s"}</strong>
                        <span>
                          {formatBytes(summary.allocatedBytes)} allocated
                        </span>
                      </div>
                    </div>
                  </div>
              ))}
            </div>
          ) : (
            <div className="network-empty-inline">
              <Cpu size={18} />
              <span>Capacity reports will appear as nodes connect.</span>
            </div>
          )}
        </section>

        <section className="network-facts" aria-labelledby="network-facts-title">
          <div className="network-section-heading">
            <div>
              <span>Signals</span>
              <h3 id="network-facts-title">What the network is doing</h3>
            </div>
          </div>
          <dl className="network-fact-list">
            <NetworkFactRow label="Active sessions" value={`${activeSessions} / ${sessionCapacity}`} />
            <NetworkFactRow label="Requests waiting" value={queuedRequests.toLocaleString()} />
            <NetworkFactRow label="Model hosts online" value={modelHosts.length.toLocaleString()} />
            <NetworkFactRow label="CPU cores visible" value={logicalCores.toLocaleString()} />
            <NetworkFactRow label="Shared by this computer" value={formatBytes(snapshot.distribution.bytesServed)} />
            <NetworkFactRow label="Verified chunks served" value={snapshot.distribution.chunksServed.toLocaleString()} />
          </dl>
        </section>

        <section className="network-node-directory" aria-labelledby="network-node-directory-title">
          <div className="network-section-heading">
            <div>
              <span>Nodes</span>
              <h3 id="network-node-directory-title">Visible computers</h3>
            </div>
            <small>{locatedNodeCount} relay-verified</small>
          </div>

          {onlineMachines.length > 0 ? (
            <div className="network-node-list">
              {onlineMachines.map((machine) => (
                <div className="network-node-row" key={networkMachineKey(machine)}>
                  <i className={machine.rpcReady ? "ready" : "connected"} aria-hidden="true" />
                  <div className="network-node-copy">
                    <strong>{machine.isLocal ? "This computer" : machine.deviceName}</strong>
                    <span>
                      {locations[networkMachineKey(machine)]?.label ?? "Locating node"}
                      {` · ${machineBackendLabel(machine.computeBackend)}`}
                    </span>
                  </div>
                  <div className="network-node-capacity">
                    <strong>{formatBytes(machine.allocatedMemoryBytes)}</strong>
                    <span>{machine.unifiedMemory ? "unified memory allocated" : "allocated to Infernet"}</span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="network-empty-inline">
              <Globe size={18} />
              <span>No computers are visible yet.</span>
            </div>
          )}
        </section>
      </div>
    </section>
  );
}

function NetworkPulseRow({
  icon,
  label,
  value,
  detail,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  detail: string;
}) {
  return (
    <div className="network-pulse-row">
      <dt>{icon}<span>{label}</span></dt>
      <dd><strong>{value}</strong><span>{detail}</span></dd>
    </div>
  );
}

function NetworkFactRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="network-fact-row">
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function NetworkGlobe({
  machines,
  locations,
  locatedNodeCount,
  isLocating,
}: {
  machines: MachineView[];
  locations: Record<string, NodeLocation>;
  locatedNodeCount: number;
  isLocating: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const markers = useMemo<Marker[]>(
    () => machines.map((machine) => {
      const location = locations[networkMachineKey(machine)] ?? illustrativeNodeLocation(machine);
      return {
        location: [location.latitude, location.longitude],
        size: machine.rpcReady ? 0.055 : 0.038,
        color: location.source === "relay"
          ? [0.96, 0.96, 0.96]
          : [0.52, 0.52, 0.52],
      };
    }),
    [locations, machines],
  );
  const markersRef = useRef(markers);

  useEffect(() => {
    markersRef.current = markers;
  }, [markers]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    const initialSize = canvas.offsetWidth || 520;
    let phi = -0.55;
    let frame = 0;
    let lastTime = performance.now();
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const globe = createGlobe(canvas, {
      devicePixelRatio: pixelRatio,
      width: Math.round(initialSize * pixelRatio),
      height: Math.round(initialSize * pixelRatio),
      phi,
      theta: 0.18,
      dark: 1,
      diffuse: 1.1,
      mapSamples: 24000,
      mapBrightness: 4.2,
      mapBaseBrightness: 0.025,
      baseColor: [0.32, 0.32, 0.32],
      markerColor: [0.96, 0.96, 0.96],
      glowColor: [0.12, 0.12, 0.12],
      markers: markersRef.current,
      markerElevation: 0.035,
      opacity: 0.76,
      scale: 0.98,
      context: { alpha: true, antialias: true },
    });

    const resizeObserver = new ResizeObserver((entries) => {
      const size = entries[0]?.contentRect.width;
      if (!size) return;
      globe.update({
        width: Math.round(size * pixelRatio),
        height: Math.round(size * pixelRatio),
      });
    });
    resizeObserver.observe(canvas);

    const animate = (now: number) => {
      const delta = Math.min(50, now - lastTime);
      lastTime = now;
      phi += (delta * Math.PI * 2) / 64000;
      globe.update({ phi, markers: markersRef.current });
      frame = window.requestAnimationFrame(animate);
    };
    if (!reducedMotion) frame = window.requestAnimationFrame(animate);

    return () => {
      if (frame) window.cancelAnimationFrame(frame);
      resizeObserver.disconnect();
      globe.destroy();
    };
  }, []);

  return (
    <figure className="network-globe-figure" aria-labelledby="network-globe-caption">
      <div className="network-globe-frame">
        <canvas ref={canvasRef} className="network-globe" aria-hidden="true" />
      </div>
      <figcaption id="network-globe-caption">
        <div>
          <i aria-hidden="true" />
          <strong>{machines.length} live node{machines.length === 1 ? "" : "s"}</strong>
        </div>
        <span>
          {isLocating
            ? "Resolving relay-verified regions"
            : locatedNodeCount > 0
              ? `${locatedNodeCount} relay-verified approximate region${locatedNodeCount === 1 ? "" : "s"}`
              : "Waiting for relay-verified approximate regions"}
        </span>
      </figcaption>
    </figure>
  );
}

function illustrativeNodeLocation(machine: MachineView): NodeLocation {
  const identity = networkMachineKey(machine);
  const latitudeUnit = (hashText(`${identity}:latitude`) + 0.5) / 4294967296;
  const longitudeUnit = (hashText(`${identity}:longitude`) + 0.5) / 4294967296;
  return {
    latitude: Math.asin(2 * latitudeUnit - 1) * (180 / Math.PI),
    longitude: longitudeUnit * 360 - 180,
    label: machine.isLocal ? "Private location · this computer" : "Private or relayed location",
    source: "illustrative",
  };
}

function networkMachineKey(machine: MachineView): string {
  return machine.machineId?.trim() || machine.peerId;
}

function hashText(value: string): number {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function DownloadsPage({
  snapshot,
  transferActivities,
}: {
  snapshot: GridSnapshot;
  transferActivities: TransferActivity[];
}) {
  const distribution = snapshot.distribution;
  const activeTransfers = transferActivities.filter((activity) => activity.status === "active");
  const recentTransfers = transferActivities.filter((activity) => activity.status !== "active").slice(0, 8);
  const localModels = groupLocalModels(snapshot);
  const activeModelTransfer = activeTransfers.find((activity) => isOfficialModelId(activity.modelId));
  const modelHosts = snapshot.machines.filter(
    (machine) => machine.hostedComponentCount > 0 && machine.connectionStatus !== "unreachable",
  );
  const storagePercent = distribution.maxStorageBytes > 0
    ? Math.min(100, Math.round((distribution.storageUsedBytes / distribution.maxStorageBytes) * 100))
    : 0;

  return (
    <section className="downloads-screen">
      <div className="section-heading">
        <h2>Storage &amp; sharing</h2>
        <p>See exactly which computer stores the model and which ones contribute compute.</p>
      </div>

      {activeTransfers.length > 0 ? (
        <div className="download-panel active-work-panel">
          <div className="download-panel-heading">
            <div>
              <strong>In progress</strong>
              <span>Current model preparation</span>
            </div>
            <span className="work-count">{activeTransfers.length}</span>
          </div>
          <div className="transfer-list">
            {activeTransfers.map((activity) => (
              <TransferActivityRow
                activity={activity}
                modelName={modelDisplayName(snapshot, activity.modelId)}
                key={activity.id}
              />
            ))}
          </div>
        </div>
      ) : null}

      <div className="download-panel network-model-panel">
        <div className="network-model-heading">
          <div>
            <span className="section-eyebrow">Network model</span>
            <strong>Infernet Chat</strong>
            <p>
              Every Infernet computer downloads the verified package during required setup and can
              share it with the network while the app is open.
            </p>
          </div>
          <span className={`network-model-status ${
            activeModelTransfer ? "downloading" : modelHosts.length > 0 ? "available" : "missing"
          }`}>
            <i aria-hidden="true" />
            {activeModelTransfer
              ? "Downloading here"
              : modelHosts.length > 0
                ? `Available from ${modelHosts.length} computer${modelHosts.length === 1 ? "" : "s"}`
                : "No model host online"}
          </span>
        </div>

        {snapshot.machines.length === 0 ? (
          <div className="empty-state compact">Waiting for computers to report their status.</div>
        ) : (
          <div className="machine-list network-machine-list">
            {snapshot.machines.map((machine) => (
              <MachineStatusCard
                machine={machine}
                localTransfer={machine.isLocal ? activeModelTransfer : undefined}
                key={machine.machineId ?? machine.peerId}
              />
            ))}
          </div>
        )}

        {distribution.currentUploads > 0 || distribution.bytesServed > 0 ? (
          <div className="model-serving-summary">
            <div className={distribution.currentUploads > 0 ? "serving-pulse active" : "serving-pulse"}>
              <i aria-hidden="true" />
              <strong>{distribution.currentUploads > 0 ? "Sharing model now" : "Model sharing ready"}</strong>
            </div>
            <span>
              {formatBytes(distribution.bytesServed)} sent in {distribution.chunksServed.toLocaleString()} verified chunks
            </span>
          </div>
        ) : null}
      </div>

      <div className="storage-overview">
        <div className="storage-overview-heading">
          <div className="storage-icon"><HardDrive size={20} /></div>
          <div>
            <strong>Model storage</strong>
            <span>{localModels.length} model{localModels.length === 1 ? "" : "s"} on this computer</span>
          </div>
          <b>{formatBytes(distribution.storageUsedBytes)}</b>
        </div>
        <ProgressBar progress={storagePercent} />
        <small>
          {distribution.maxStorageBytes > 0
            ? `${formatBytes(distribution.storageUsedBytes)} used of ${formatBytes(distribution.maxStorageBytes)}`
            : `${formatBytes(distribution.storageUsedBytes)} used`}
        </small>
      </div>

      <div className="download-panel local-models-panel">
        <div className="download-panel-heading">
          <div>
            <strong>Models stored on this computer</strong>
            <span>Verified packages this computer can share with the network</span>
          </div>
        </div>
        <div className="local-model-list">
          {localModels.length === 0 ? (
            <div className="empty-state compact">
              Required model setup has not finished on this computer.
            </div>
          ) : (
            localModels.map((item) => (
              <div className="local-model-row" key={item.modelId}>
                <div className="local-model-icon"><Box size={18} /></div>
                <div className="local-model-copy">
                  <strong>{item.displayName}</strong>
                  <span>{item.quantization ? item.quantization.toUpperCase() : "Infernet model"} · Stored locally</span>
                </div>
                <div className="local-model-meta">
                  <strong>{formatBytes(item.sizeBytes)}</strong>
                  <span>{item.replicas <= 1 ? "This computer is the only host" : `Available from ${item.replicas} computers`}</span>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {recentTransfers.length > 0 ? (
        <div className="download-panel recent-activity-panel">
          <div className="download-panel-heading">
            <div>
              <strong>Recent activity</strong>
              <span>Completed during this app session</span>
            </div>
          </div>
          <div className="transfer-list">
            {recentTransfers.map((activity) => (
              <TransferActivityRow
                activity={activity}
                modelName={modelDisplayName(snapshot, activity.modelId)}
                key={activity.id}
              />
            ))}
          </div>
        </div>
      ) : null}
    </section>
  );
}

function AboutPage() {
  return (
    <section className="about-screen">
      <div className="about-document">
        <header className="about-intro">
          <span>About Infernet</span>
          <h2>AI powered by computers that choose to work together.</h2>
          <p>
            Infernet is a local-first AI app connected to a shared compute network. It keeps the
            familiar experience of chatting or creating an image while handling model setup,
            machine discovery, and work distribution in the background.
          </p>
        </header>

        <section className="about-section" aria-labelledby="about-request-title">
          <span className="about-kicker">A request from start to finish</span>
          <h3 id="about-request-title">How it works</h3>
          <ol className="about-flow">
            <li>
              <b>1</b>
              <div><strong>You make a request</strong><p>Enter a message or describe an image, just as you would in any AI app.</p></div>
            </li>
            <li>
              <b>2</b>
              <div><strong>Infernet finds eligible computers</strong><p>It checks which visible machines have the verified model, a compatible runtime, contributed memory, and room for another session.</p></div>
            </li>
            <li>
              <b>3</b>
              <div><strong>The work is planned</strong><p>When multiple eligible physical machines are available, Infernet divides the request across them. Multiple app processes on one computer still count as one machine.</p></div>
            </li>
            <li>
              <b>4</b>
              <div><strong>The result returns here</strong><p>The participating machines process their assigned work and the completed response appears in your conversation or Creations list.</p></div>
            </li>
          </ol>
        </section>

        <section className="about-section" aria-labelledby="about-placement-title">
          <span className="about-kicker">Distribution rules</span>
          <h3 id="about-placement-title">How Infernet chooses machines</h3>
          <div className="about-prose">
            <p>Physical computers, not peer IDs or app processes, are the unit of placement.</p>
            <ul>
              <li>If two or more eligible computers are available, the request is split across them.</li>
              <li>If your computer and a remote computer are both eligible, both participate.</li>
              <li>A request may run on only your computer when it is the sole eligible option.</li>
              <li>Infernet never runs an entire request on one remote computer or silently falls back to one after a distributed plan fails.</li>
            </ul>
          </div>
        </section>

        <section className="about-section" aria-labelledby="about-network-title">
          <span className="about-kicker">Connection and privacy</span>
          <h3 id="about-network-title">What the relay does</h3>
          <div className="about-prose">
            <p>
              The public relay helps computers find and reach one another when private networks or
              routers prevent a direct connection. It forwards encrypted peer-to-peer traffic, but
              it does not advertise itself as compute and does not perform inference.
            </p>
            <p>
              For the globe, each node asks the relay for its own approximate region. The relay
              returns short-lived signed coordinates rounded to a broad area. Other nodes receive
              that coarse result, never the underlying public IP address.
            </p>
          </div>
        </section>

        <section className="about-section" aria-labelledby="about-models-title">
          <span className="about-kicker">Models and contribution</span>
          <h3 id="about-models-title">What lives on your computer</h3>
          <div className="about-prose">
            <p>
              Infernet uses curated, versioned model packages. Packages are downloaded, checked
              against pinned release information, and stored locally. A machine advertises a model
              only after the required package is present and verified.
            </p>
            <p>
              Your contribution setting controls how much GPU VRAM or Apple unified memory this
              computer offers to the network. Turning contribution off removes it from compute
              eligibility. Generated images remain stored on this computer and reappear in
              Creations after restarting the app.
            </p>
          </div>
        </section>

        <section className="about-section about-faq" aria-labelledby="about-faq-title">
          <span className="about-kicker">Useful distinctions</span>
          <h3 id="about-faq-title">A few things to know</h3>
          <dl>
            <div><dt>Does every visible computer run every request?</dt><dd>No. It must be online, compatible, verified, contributing capacity, and available for the requested model.</dd></div>
            <div><dt>Is the relay another compute node?</dt><dd>No. It provides discovery, relayed connections, and signed approximate regions only.</dd></div>
            <div><dt>What does “allocated” memory mean?</dt><dd>It is the contribution ceiling a computer has committed to Infernet, not the amount that happens to be free at that moment.</dd></div>
            <div><dt>Where can I see what is happening?</dt><dd>Activity shows work on this computer. Network shows visible physical machines and shared capacity. Downloads shows model storage and transfers.</dd></div>
          </dl>
        </section>
      </div>
    </section>
  );
}

function SettingsPage({
  snapshot,
  imageRuntimeStatus,
  appUpdater,
}: {
  snapshot: GridSnapshot;
  imageRuntimeStatus: ImageRuntimeStatus | null;
  appUpdater: AppUpdaterState;
}) {
  const [settings, setSettings] = useState<VramContributionSettings | null>(null);
  const [draftBytes, setDraftBytes] = useState(0);
  const [status, setStatus] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let disposed = false;
    getVramContributionSettings()
      .then((next) => {
        if (disposed) return;
        setSettings(next);
        setDraftBytes(next.contributionBytes);
      })
      .catch((error) => {
        if (!disposed) setStatus(String(error));
      });
    return () => {
      disposed = true;
    };
  }, []);

  async function saveContribution() {
    if (!settings || saving || draftBytes === settings.contributionBytes) return;
    setSaving(true);
    setStatus(null);
    try {
      const next = await setVramContribution(draftBytes);
      setSettings(next);
      setDraftBytes(next.contributionBytes);
      setStatus(
        next.contributionBytes === 0
          ? "Network contribution is off."
          : `Contribution limited to ${formatBytes(next.contributionBytes)}.`,
      );
    } catch (error) {
      setStatus(String(error));
    } finally {
      setSaving(false);
    }
  }

  const hasAccelerator = Boolean(settings?.totalBytes);
  const contributionLabel = draftBytes === 0 ? "Off" : formatBytes(draftBytes);
  const chatModel = snapshot.availableModels.find(
    (model) => model.modelId === INFERNET_CHAT_MODEL_ID,
  );
  const chatModelBytes = snapshot.distribution.installedShards
    .filter((shard) => shard.modelId === INFERNET_CHAT_MODEL_ID)
    .reduce((total, shard) => total + shard.sizeBytes, 0);

  return (
    <section className="settings-screen">
      <div className="section-heading">
        <h2>Settings</h2>
        <p>See installed models and control how much of this computer the network can use.</p>
      </div>

      <div className="settings-list">
        <div className="settings-row model-settings">
          <div className="model-settings-heading">
            <strong>Downloaded models</strong>
            <span>Official packages stored and verified on this computer.</span>
          </div>

          <div className="settings-model-list">
            <div className="settings-model-row">
              <div className="settings-model-icon" aria-hidden="true">
                <MessageSquare size={18} />
              </div>
              <div>
                <strong>Infernet Chat</strong>
                <span>
                  {chatModel?.quantization
                    ? `${chatModel.quantization.toUpperCase()} · Official release`
                    : "Official chat model"}
                </span>
              </div>
              <div className="settings-model-status">
                <strong><i aria-hidden="true" />Downloaded</strong>
                <span>{chatModelBytes > 0 ? formatBytes(chatModelBytes) : "Verified"}</span>
              </div>
            </div>

            <div className="settings-model-row">
              <div className="settings-model-icon" aria-hidden="true">
                <ImageIcon size={18} />
              </div>
              <div>
                <strong>Infernet Image</strong>
                <span>
                  {imageRuntimeStatus?.quantization
                    ? `${imageRuntimeStatus.quantization} · Official release`
                    : "Official image model"}
                </span>
              </div>
              <div className="settings-model-status">
                <strong><i aria-hidden="true" />Downloaded</strong>
                <span>
                  {imageRuntimeStatus?.totalBytes
                    ? formatBytes(imageRuntimeStatus.totalBytes)
                    : "Verified"}
                </span>
              </div>
            </div>
          </div>
        </div>

        <div className="settings-row vram-settings">
          <div className="vram-settings-copy">
            <strong>VRAM contribution</strong>
            <span>
              The most memory Infernet will offer to network work. Model data, KV cache,
              and runtime headroom all count toward this limit.
            </span>
            {settings ? (
              <small>
                {settings.deviceName}
                {settings.unifiedMemory ? " · shared GPU and system memory" : ""}
              </small>
            ) : null}
          </div>

          {settings ? (
            <div className="vram-controls">
              <div className="vram-value-row">
                <label htmlFor="vram-contribution">Contribution limit</label>
                <output htmlFor="vram-contribution">{contributionLabel}</output>
              </div>
              <input
                id="vram-contribution"
                type="range"
                min={0}
                max={settings.totalBytes}
                step={1024 * 1024 * 1024}
                value={draftBytes}
                disabled={!hasAccelerator || saving}
                aria-describedby="vram-contribution-help"
                onChange={(event) => {
                  setDraftBytes(Number(event.target.value));
                  setStatus(null);
                }}
              />
              <div className="vram-range-labels" aria-hidden="true">
                <span>Off</span>
                <span>{formatBytes(settings.totalBytes)}</span>
              </div>
              <small id="vram-contribution-help">
                {hasAccelerator
                  ? `${formatBytes(settings.availableBytes)} currently available to Infernet.`
                  : "No supported GPU or Apple unified memory was detected."}
              </small>
              <div className="vram-actions">
                <button
                  type="button"
                  className="secondary-button"
                  disabled={!hasAccelerator || saving || draftBytes === settings.contributionBytes}
                  onClick={saveContribution}
                >
                  {saving ? "Saving…" : "Save limit"}
                </button>
                {status ? <span role="status">{status}</span> : null}
              </div>
            </div>
          ) : (
            <span className="settings-loading" role={status ? "alert" : "status"}>
              {status ?? "Reading graphics memory…"}
            </span>
          )}
        </div>

        <div className="settings-row update-settings">
          <div>
            <strong>Application updates</strong>
            <span>
              Infernet checks GitHub Releases automatically and verifies every update signature
              before installation.
            </span>
          </div>
          <button
            type="button"
            className="secondary-button"
            disabled={appUpdater.phase === "installing"}
            onClick={() => void appUpdater.checkNow()}
          >
            {appUpdater.phase === "installing" ? "Installing…" : "Check for updates"}
          </button>
        </div>
      </div>
    </section>
  );
}

function ProgressBar({ progress, indeterminate = false }: { progress: number; indeterminate?: boolean }) {
  return (
    <div
      className={indeterminate ? "progress-bar indeterminate" : "progress-bar"}
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={indeterminate ? undefined : Math.max(0, Math.min(100, progress))}
    >
      <span style={{ width: `${progress}%` }} />
    </div>
  );
}

function upsertTransferActivity(
  current: TransferActivity[],
  event: ModelImportProgress,
): TransferActivity[] {
  const status = transferStatus(event.stage);
  const updatedAt = Date.now();
  if (status === "error") {
    const modelActivityId = `${event.modelId}:model`;
    const isPeerDownloadFailure = event.stage.toLowerCase() === "download failed";
    let settledActiveActivity = false;
    const settled = current.map((item) => {
      const matchesFailedOperation = isPeerDownloadFailure
        ? item.id !== modelActivityId
        : item.id === modelActivityId;
      if (item.modelId !== event.modelId || item.status !== "active" || !matchesFailedOperation) {
        return item;
      }
      settledActiveActivity = true;
      return {
        ...item,
        stage: event.stage,
        detail: event.detail,
        downloadedBytes: event.downloadedBytes,
        totalBytes: event.totalBytes ?? item.totalBytes,
        status,
        updatedAt,
      };
    });
    if (settledActiveActivity) {
      return settled
        .sort((left, right) => right.updatedAt - left.updatedAt)
        .slice(0, 24);
    }
  }

  const id = transferActivityId(event);
  const existing = current.find((item) => item.id === id);
  const isRestartingImageInstall = isOfficialImageModelId(event.modelId)
    && existing?.status !== "active"
    && status === "active";
  const activity: TransferActivity = {
    ...event,
    id,
    status,
    startedAt: isRestartingImageInstall ? updatedAt : existing?.startedAt ?? updatedAt,
    updatedAt,
  };
  if (
    activity.status === "active"
    && existing
    && existing.status !== "active"
    && activity.id.endsWith(":model")
    && !isOfficialImageModelId(activity.modelId)
  ) {
    return current;
  }
  const next = current.some((item) => item.id === activity.id)
    ? current.map((item) => (item.id === activity.id ? activity : item))
    : [activity, ...current];

  return next
    .sort((left, right) => right.updatedAt - left.updatedAt)
    .slice(0, 24);
}

function formatProgressPercent(percent: number): string {
  if (percent > 0 && percent < 1) {
    return percent.toFixed(2);
  }
  if (percent < 10) {
    return percent.toFixed(1);
  }
  return Math.round(percent).toString();
}

function transferActivityId(event: ModelImportProgress): string {
  const layerMatch = event.detail.match(/layers\s+\d+:\d+/i);
  const isPeerShardTransfer = ["downloading shard", "shard ready"].includes(event.stage.toLowerCase());
  const scope = isPeerShardTransfer && layerMatch ? layerMatch[0].toLowerCase() : "model";
  return `${event.modelId}:${scope}`;
}

function transferStatus(stage: string): TransferStatus {
  const normalized = stage.toLowerCase();
  if (normalized.includes("failed") || normalized.includes("error")) {
    return "error";
  }
  if (normalized.includes("ready")) {
    return "complete";
  }
  return "active";
}

function pageTitle(page: Page, chatTitle = "Chat"): string {
  if (page === "chat") return chatTitle;
  if (page === "image") return "Image";
  if (page === "activity") return "Activity";
  if (page === "network") return "Network";
  if (page === "downloads") return "Downloads";
  if (page === "about") return "Help";
  return "Settings";
}

function timeGreeting(date = new Date()): string {
  const hour = date.getHours();
  if (hour < 12) return "Good morning";
  if (hour < 18) return "Good afternoon";
  return "Good evening";
}

function useElapsedTime(startedAt: number | null): number {
  const [elapsedMs, setElapsedMs] = useState(0);

  useEffect(() => {
    if (startedAt === null) {
      setElapsedMs(0);
      return;
    }

    const updateElapsed = () => setElapsedMs(Date.now() - startedAt);
    updateElapsed();
    const interval = window.setInterval(updateElapsed, 1000);
    return () => window.clearInterval(interval);
  }, [startedAt]);

  return elapsedMs;
}

function formatDuration(durationMs: number | null): string {
  if (durationMs === null || durationMs <= 0) return "—";
  if (durationMs < 1000) return `${Math.round(durationMs)} ms`;
  const seconds = durationMs / 1000;
  if (seconds < 60) return `${seconds < 10 ? seconds.toFixed(1) : Math.round(seconds)} sec`;
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.round(seconds % 60);
  return `${minutes} min ${remainingSeconds} sec`;
}

function formatRelativeTime(timestamp: number): string {
  const seconds = Math.max(0, Math.round((Date.now() - timestamp) / 1000));
  if (seconds < 5) return "Just now";
  if (seconds < 60) return `${seconds} sec ago`;
  const minutes = Math.round(seconds / 60);
  return minutes === 1 ? "1 min ago" : `${minutes} min ago`;
}

function formatJournalTime(timestamp: number): string {
  return new Intl.DateTimeFormat(undefined, {
    hour: "numeric",
    minute: "2-digit",
  }).format(timestamp);
}

function machineBackendLabel(backend: string): string {
  if (backend === "cuda") return "CUDA";
  if (backend === "metal") return "Metal";
  return "CPU";
}

function machineLoadLabel(activeSessions: number, maxSessions: number, queueDepth: number): string {
  if (queueDepth > 0) {
    return `${activeSessions}/${maxSessions} active · ${queueDepth} queued`;
  }
  if (activeSessions > 0) {
    return `${activeSessions}/${maxSessions} sessions active`;
  }
  return "Ready";
}

function friendlyActivityError(error: string): string {
  const normalized = error.toLowerCase();
  if (normalized.includes("at least one other gpu") || normalized.includes("distributed inference needs")) {
    return "Infernet Chat needs one model coordinator and at least one other CUDA or Apple-silicon computer online.";
  }
  if (normalized.includes("compute service") || normalized.includes("rpc worker")) {
    return "A computer is online, but its distributed compute service is not ready. Restart Infernet on that computer.";
  }
  if (normalized.includes("safe model memory") || normalized.includes("enough free memory")) {
    return "The connected computers do not currently have enough free GPU or unified memory for Infernet Chat.";
  }
  if (normalized.includes("incompatible") || normalized.includes("pinned protocol")) {
    return "One computer is running an incompatible Infernet build. Update or rebuild Infernet on every machine.";
  }
  if (normalized.includes("timed out") || normalized.includes("timeout")) {
    return "The model took too long to respond. Try again in a moment.";
  }
  if (normalized.includes("missing") || normalized.includes("no route")) {
    return "The full model is not available on the network yet.";
  }
  if (normalized.includes("offline") || normalized.includes("connect")) {
    return "Infernet could not reach the computer running this part of the model.";
  }
  return "The model stopped unexpectedly. Try again, or restart Infernet if it keeps happening.";
}

function humanTransferStage(stage: string): string {
  const normalized = stage.toLowerCase();
  if (normalized.includes("failed") || normalized.includes("error")) return "Couldn’t finish";
  if (normalized.includes("verifying image package")) return "Checking image package";
  if (normalized.includes("downloading image package")) return "Downloading image package";
  if (normalized.includes("repairing image package")) return "Repairing image package";
  if (normalized.includes("checking file") || normalized.includes("verifying model")) return "Checking model";
  if (normalized.includes("connecting")) return "Connecting to source";
  if (normalized.includes("downloading shard")) return "Downloading part of model";
  if (normalized.includes("downloading") || normalized.includes("starting download")) return "Downloading model";
  if (normalized.includes("preparing")) return "Preparing for Infernet";
  if (normalized.includes("sharing")) return "Making model available";
  if (normalized.includes("ready")) return "Ready";
  return "Preparing model";
}

function transferStageDescription(stage: string, status: TransferStatus): string {
  const normalized = stage.toLowerCase();
  const isImagePackage = normalized.includes("image package");
  if (status === "error") {
    return isImagePackage
      ? "Infernet couldn’t complete this image package task."
      : "Infernet couldn’t complete this model task.";
  }
  if (status === "complete") {
    return isImagePackage ? "Infernet Image is ready to use." : "This model is ready to use.";
  }
  if (normalized.includes("verifying") || normalized.includes("checking")) {
    return "Confirming the model is complete and trusted.";
  }
  if (normalized.includes("sharing")) return "Getting the model ready for other computers.";
  if (normalized.includes("preparing")) return "Optimizing the official package for this computer.";
  return "Receiving the official model package.";
}

function modelDisplayName(snapshot: GridSnapshot, modelId: string): string {
  if (isOfficialImageModelId(modelId)) return "Infernet Image";
  const model = snapshot.availableModels.find(
    (item) => item.modelId === modelId && isOfficialInfernetModel(item),
  );
  return model ? curatedModelName(model) : "Infernet Chat";
}

function isOfficialModelId(modelId: string): boolean {
  return modelId === INFERNET_CHAT_MODEL_ID;
}

function isOfficialImageModelId(modelId: string): boolean {
  return modelId === INFERNET_IMAGE_MODEL_ID;
}

function isOfficialInfernetModel(model: ModelView): boolean {
  return isOfficialModelId(model.modelId);
}

function curatedModelName(_model: ModelView): string {
  return "Infernet Chat";
}

type LocalModelSummary = {
  modelId: string;
  displayName: string;
  quantization?: string | null;
  sizeBytes: number;
  packageCount: number;
  layerStart: number;
  layerEnd: number;
  version: string;
  checksums: string[];
  replicas: number;
};

function groupLocalModels(snapshot: GridSnapshot): LocalModelSummary[] {
  const summaries = new Map<string, LocalModelSummary>();

  for (const shard of snapshot.distribution.installedShards) {
    if (!isOfficialModelId(shard.modelId)) continue;
    const model = snapshot.availableModels.find((item) => item.modelId === shard.modelId);
    const replicaCounts = snapshot.distribution.replicationHealth
      .filter((item) => item.modelId === shard.modelId)
      .map((item) => item.replicas);
    const replicas = replicaCounts.length > 0 ? Math.min(...replicaCounts) : 1;
    const existing = summaries.get(shard.modelId);

    if (existing) {
      existing.sizeBytes += shard.sizeBytes;
      existing.packageCount += 1;
      existing.layerStart = Math.min(existing.layerStart, shard.layerStart);
      existing.layerEnd = Math.max(existing.layerEnd, shard.layerEnd);
      existing.replicas = Math.min(existing.replicas, replicas);
      if (!existing.checksums.includes(shard.checksum)) existing.checksums.push(shard.checksum);
      continue;
    }

    summaries.set(shard.modelId, {
      modelId: shard.modelId,
      displayName: model ? curatedModelName(model) : "Infernet Chat",
      quantization: model?.quantization,
      sizeBytes: shard.sizeBytes,
      packageCount: 1,
      layerStart: shard.layerStart,
      layerEnd: shard.layerEnd,
      version: shard.version,
      checksums: [shard.checksum],
      replicas,
    });
  }

  return [...summaries.values()].sort((left, right) => left.displayName.localeCompare(right.displayName));
}

function formatBytes(bytes: number): string {
  if (!bytes) return "0 B";
  if (bytes < 1024) return `${bytes} B`;

  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1024;
  let unitIndex = 0;

  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }

  return `${value.toFixed(value >= 10 ? 0 : 1)} ${units[unitIndex]}`;
}
