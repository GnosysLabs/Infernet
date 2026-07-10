import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import createGlobe from "cobe";
import type { Marker } from "cobe";
import {
  Activity,
  Box,
  CheckCircle2,
  ChevronDown,
  Cpu,
  Download,
  Globe,
  HardDrive,
  Laptop2,
  Layers3,
  MemoryStick,
  MessageSquare,
  PanelRightClose,
  RefreshCw,
  Send,
  Server,
  Settings,
  Zap,
} from "lucide-react";
import {
  addManualPeer,
  clearManualPeers,
  emptySnapshot,
  getGridSnapshot,
  getLocalIdentity,
  getLocalNodeActivity,
  getManualPeers,
  installOfficialModel,
  listenForProgress,
  listenForModelImportProgress,
  runDistributedInference,
} from "./api";
import type {
  GridSnapshot,
  LocalIdentity,
  LocalNodeActivitySnapshot,
  MachineView,
  ModelImportProgress,
  ModelView,
  ProgressEvent,
} from "./types";

type Page = "chat" | "network" | "models" | "downloads" | "settings";
type Message = { id: string; role: "user" | "assistant"; text: string };
type TransferStatus = "active" | "complete" | "error";
type TransferActivity = ModelImportProgress & {
  id: string;
  status: TransferStatus;
  startedAt: number;
  updatedAt: number;
};
type NodeJournalEntry = {
  id: string;
  kind: "completion" | "contribution" | "model" | "sharing" | "error";
  title: string;
  detail?: string;
  occurredAt: number;
};

const DEFAULT_PROMPT = "";
const INFERNET_CHAT_MODEL_ID = "infernet-chat-v1";
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
  const [page, setPage] = useState<Page>("chat");
  const [activityOpen, setActivityOpen] = useState(false);
  const [identity, setIdentity] = useState<LocalIdentity | null>(null);
  const [localNodeActivity, setLocalNodeActivity] = useState<LocalNodeActivitySnapshot>(
    EMPTY_LOCAL_NODE_ACTIVITY,
  );
  const [localJournal, setLocalJournal] = useState<NodeJournalEntry[]>([]);
  const [snapshot, setSnapshot] = useState<GridSnapshot>(emptySnapshot);
  const [selectedModel, setSelectedModel] = useState("");
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [lastError, setLastError] = useState<string | null>(null);
  const [transferActivities, setTransferActivities] = useState<TransferActivity[]>([]);
  const [installingModelId, setInstallingModelId] = useState<string | null>(null);

  const officialModels = useMemo(
    () => snapshot.availableModels.filter(isOfficialInfernetModel),
    [snapshot.availableModels],
  );
  const selectedModelView = useMemo(
    () => officialModels.find((model) => model.modelId === selectedModel),
    [officialModels, selectedModel],
  );
  const activeTransfers = transferActivities.filter((activity) => activity.status === "active").length;

  const appendJournalEntry = useCallback((entry: NodeJournalEntry) => {
    setLocalJournal((current) => {
      if (current.some((item) => item.id === entry.id)) return current;
      return [...current, entry]
        .sort((left, right) => left.occurredAt - right.occurredAt)
        .slice(-50);
    });
  }, []);

  const applyProgressEvent = useCallback((event: ProgressEvent) => {
    if (event.type === "routeDiscovered" || event.type === "executionPlan") {
      setLastError(null);
      return;
    }

    if (event.type === "finalOutput") {
      setIsRunning(false);
      return;
    }

    if (event.type === "error") {
      setLastError(event.message);
      setIsRunning(false);
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
    }
  }, []);

  useEffect(() => {
    getLocalIdentity().then(setIdentity).catch((error) => setLastError(String(error)));
  }, []);

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
    if (!activityOpen && page !== "downloads" && page !== "network") return;
    let disposed = false;
    let inFlight = false;
    const refreshActivity = async () => {
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
    void refreshActivity();
    const interval = window.setInterval(refreshActivity, 6000);
    return () => {
      disposed = true;
      window.clearInterval(interval);
    };
  }, [activityOpen, page, selectedModel]);

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
      if (!isOfficialModelId(event.modelId)) {
        return;
      }
      setTransferActivities((current) => upsertTransferActivity(current, event));
      const normalizedStage = event.stage.trim().toLowerCase();
      if (normalizedStage === "ready") {
        appendJournalEntry({
          id: `model-ready-${event.modelId}`,
          kind: "model",
          title: "You prepared Infernet Chat",
          detail: "The verified model is ready to use and share.",
          occurredAt: Date.now(),
        });
      } else if (normalizedStage.includes("failed") || normalizedStage.includes("error")) {
        appendJournalEntry({
          id: `model-error-${event.modelId}-${Date.now()}`,
          kind: "error",
          title: "A model task couldn’t finish",
          detail: friendlyActivityError(event.detail),
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

  function handleModelChange(modelId: string) {
    setSelectedModel(modelId);
    setPage("chat");
    setLastError(null);
  }

  async function handleInstallModel(modelId: string) {
    if (installingModelId) return;
    setInstallingModelId(modelId);
    setActivityOpen(true);
    setLastError(null);
    try {
      const nextSnapshot = await installOfficialModel(modelId);
      setSnapshot(nextSnapshot);
    } catch (error) {
      const message = String(error);
      setTransferActivities((current) => upsertTransferActivity(current, {
        modelId,
        stage: "Download failed",
        detail: message,
        downloadedBytes: 0,
        totalBytes: null,
      }));
      setLastError(message);
    } finally {
      setInstallingModelId(null);
    }
  }

  async function runInference() {
    const userPrompt = prompt.trim();
    if (!userPrompt || isRunning) {
      return;
    }

    if (!selectedModelView) {
      setLastError("Install Infernet Chat before sending a message.");
      return;
    }
    if (!selectedModelView.runnable) {
      setLastError(selectedModelView.status);
      return;
    }
    setMessages((current) => [
      ...current,
      { id: `user-${Date.now()}`, role: "user", text: userPrompt },
    ]);
    setPrompt("");
    setIsRunning(true);
    setLastError(null);

    try {
      const output = (await runDistributedInference(userPrompt, selectedModel)).output;

      setMessages((current) => [
        ...current,
        { id: `assistant-${Date.now()}`, role: "assistant", text: output },
      ]);
    } catch (error) {
      const message = String(error);
      setLastError(message);
    } finally {
      setIsRunning(false);
    }
  }

  return (
    <div className={activityOpen ? "app-shell activity-open" : "app-shell"}>
      <Sidebar page={page} setPage={setPage} />

      <main className="app-main">
        <AppHeader
          page={page}
          model={selectedModelView}
          networkNodeCount={snapshot.machines.filter((machine) => machine.connectionStatus !== "unreachable").length}
          networkReadyCount={snapshot.machines.filter((machine) => machine.rpcReady && machine.connectionStatus !== "unreachable").length}
          onRefresh={() => refreshSnapshot(selectedModel)}
          activityOpen={activityOpen}
          hasActiveWork={
            localNodeActivity.computeActive
            || localNodeActivity.sharingActive
            || activeTransfers > 0
          }
          onToggleActivity={() => setActivityOpen((open) => !open)}
        />

        {page === "chat" ? (
          <ChatPage
            messages={messages}
            prompt={prompt}
            setPrompt={setPrompt}
            runInference={runInference}
            isRunning={isRunning}
            model={selectedModelView}
            lastError={lastError}
            onOpenModels={() => setPage("models")}
          />
        ) : null}

        {page === "models" ? (
          <ModelsPage
            snapshot={snapshot}
            selectedModel={selectedModel}
            onModelChange={handleModelChange}
            onInstallModel={handleInstallModel}
            installingModelId={installingModelId}
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

        {page === "settings" ? (
          <SettingsPage
            identity={identity}
            onNetworkChanged={() => refreshSnapshot(selectedModel)}
          />
        ) : null}
      </main>

      {activityOpen ? (
        <ActivitySidebar
          snapshot={snapshot}
          transferActivities={transferActivities}
          localNodeActivity={localNodeActivity}
          localJournal={localJournal}
          onClose={() => setActivityOpen(false)}
        />
      ) : null}
    </div>
  );
}

function Sidebar({
  page,
  setPage,
}: {
  page: Page;
  setPage: (page: Page) => void;
}) {
  return (
    <aside className="sidebar">
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

      <nav className="nav-list" aria-label="Primary">
        <NavButton icon={<MessageSquare size={18} />} label="Chat" active={page === "chat"} onClick={() => setPage("chat")} />
        <NavButton icon={<Globe size={18} />} label="Network" active={page === "network"} onClick={() => setPage("network")} />
        <NavButton icon={<Settings size={18} />} label="Settings" active={page === "settings"} onClick={() => setPage("settings")} />
      </nav>

    </aside>
  );
}

function NavButton({
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
    <button className={active ? "nav-button active" : "nav-button"} onClick={onClick} aria-label={label}>
      {icon}
      <span>{label}</span>
    </button>
  );
}

function AppHeader({
  page,
  model,
  networkNodeCount,
  networkReadyCount,
  onRefresh,
  activityOpen,
  hasActiveWork,
  onToggleActivity,
}: {
  page: Page;
  model?: ModelView;
  networkNodeCount: number;
  networkReadyCount: number;
  onRefresh: () => void;
  activityOpen: boolean;
  hasActiveWork: boolean;
  onToggleActivity: () => void;
}) {
  return (
    <header className="app-header">
      <div>
        <h1>{pageTitle(page)}</h1>
        <div className="header-meta">
          <span>
            {page === "network"
              ? networkNodeCount > 0
                ? `${networkNodeCount} node${networkNodeCount === 1 ? "" : "s"} visible · ${networkReadyCount} compute-ready`
                : "Discovering network compute"
              : model ? curatedModelName(model) : "No model selected"}
          </span>
        </div>
      </div>

      <div className="header-actions">
        <button
          className={activityOpen ? "activity-toggle active" : "activity-toggle"}
          aria-label="Activity"
          aria-expanded={activityOpen}
          aria-controls="activity-sidebar"
          onClick={onToggleActivity}
        >
          <Activity size={16} />
          <span>Activity</span>
          {hasActiveWork ? <i aria-label="Active work" /> : null}
        </button>
        <button className="icon-button" aria-label="Refresh app status" onClick={onRefresh}>
          <RefreshCw size={16} />
        </button>
      </div>
    </header>
  );
}

function ChatPage({
  messages,
  prompt,
  setPrompt,
  runInference,
  isRunning,
  model,
  lastError,
  onOpenModels,
}: {
  messages: Message[];
  prompt: string;
  setPrompt: (prompt: string) => void;
  runInference: () => void;
  isRunning: boolean;
  model?: ModelView;
  lastError: string | null;
  onOpenModels: () => void;
}) {
  const conversationRef = useRef<HTMLDivElement>(null);
  const canSend = Boolean(model?.runnable);
  const isEmpty = messages.length === 0;

  useEffect(() => {
    const conversation = conversationRef.current;
    if (conversation) {
      conversation.scrollTop = conversation.scrollHeight;
    }
  }, [messages, isRunning, lastError]);

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
              <div className="message-bubble">{message.text}</div>
            </div>
          ))}

          {!model ? (
            <div className="empty-chat-card">
              <strong>Get Infernet Chat to start</strong>
              <span>The official Infernet model is not available on the network yet.</span>
              <button className="secondary-button" onClick={onOpenModels}>
                <Download size={16} />
                <span>View Infernet Chat</span>
              </button>
            </div>
          ) : !model.runnable ? (
            <div className="empty-chat-card warning">
              <strong>{curatedModelName(model)} is not ready yet</strong>
              <span>{model.status}</span>
              <button className="secondary-button" onClick={onOpenModels}>
                <Box size={16} />
                <span>Manage Model</span>
              </button>
            </div>
          ) : null}

          {isRunning ? <ThinkingIndicator /> : null}

          {lastError && messages.length > 0 && !isRunning ? (
            <div className="chat-error" role="alert">
              <strong>Infernet couldn’t finish that response.</strong>
              <span>{friendlyActivityError(lastError)}</span>
            </div>
          ) : null}
        </div>
      </div>

      <div className="composer-dock">
        <div className="composer">
          <div className="composer-model">
            <Box size={15} />
            <span>{model ? curatedModelName(model) : "No model selected"}</span>
            <ChevronDown size={15} />
          </div>
          <textarea
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                runInference();
              }
            }}
            placeholder={canSend ? "Message Infernet" : model ? "Model is not ready" : "Install Infernet Chat first"}
            disabled={!canSend}
            aria-label="Message Infernet"
          />
          <button className="send-button" onClick={runInference} disabled={isRunning || !prompt.trim() || !canSend}>
            {isRunning ? <Activity size={18} /> : <Send size={18} />}
            <span>Send</span>
          </button>
        </div>
      </div>
    </section>
  );
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

function ActivitySidebar({
  snapshot,
  transferActivities,
  localNodeActivity,
  localJournal,
  onClose,
}: {
  snapshot: GridSnapshot;
  transferActivities: TransferActivity[];
  localNodeActivity: LocalNodeActivitySnapshot;
  localJournal: NodeJournalEntry[];
  onClose: () => void;
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
          : "A compute task couldn’t finish",
        detail: duration === "—" ? undefined : `Your node worked for ${duration}.`,
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
    <aside className="activity-sidebar" id="activity-sidebar" aria-label="Your node activity">
      <div className="activity-sidebar-header">
        <div>
          <span>Your node</span>
          <h2>{deviceName}</h2>
        </div>
        <button className="icon-button" aria-label="Close activity" onClick={onClose}>
          <PanelRightClose size={18} />
        </button>
      </div>

      <div className="activity-sidebar-scroll">
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
    </aside>
  );
}

function NodeJournalIcon({ kind }: { kind: NodeJournalEntry["kind"] }) {
  if (kind === "completion") return <MessageSquare size={13} />;
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
      : { className: "muted", label: "Compute only" };
  const modelDetail = localTransfer
    ? "The verified package will be shared after the download finishes."
    : unreachable && machine.hostedComponentCount > 0
      ? "This computer last reported the verified package, but it cannot be reached right now."
    : machine.hostedComponentCount > 0
      ? "Infernet Chat is verified and available for other computers to download while Infernet stays open."
      : supportedBackend
        ? "No full model download needed. This computer receives only its assigned model data in memory during a request."
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
  source: "geoip" | "illustrative";
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
  const availableComputeBytes = acceleratorMachines.reduce(
    (total, machine) => total + machine.availableMemoryBytes,
    0,
  );
  const totalComputeBytes = acceleratorMachines.reduce(
    (total, machine) => total + machine.totalMemoryBytes,
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
  const capacityFreePercent = totalComputeBytes > 0
    ? Math.round((availableComputeBytes / totalComputeBytes) * 100)
    : 0;
  const backendSummaries = ["cuda", "metal", "cpu"].map((backend) => {
    const machines = onlineMachines.filter((machine) => machine.computeBackend === backend);
    return {
      backend,
      machines,
      availableBytes: machines.reduce((total, machine) => total + machine.availableMemoryBytes, 0),
      totalBytes: machines.reduce((total, machine) => total + machine.totalMemoryBytes, 0),
    };
  }).filter((summary) => summary.machines.length > 0);
  const { locations, isLocating } = useNodeLocations(onlineMachines);
  const locatedNodeCount = Object.values(locations).filter(
    (location) => location.source === "geoip",
  ).length;

  return (
    <section className="network-screen">
      <div className="network-layout">
        <NetworkGlobe
          machines={onlineMachines}
          locations={locations}
          locatedNodeCount={locatedNodeCount}
          isLocating={isLocating}
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
              ? "A clear view of the capacity Infernet can use right now, without exposing precise device locations."
              : "Online computers and their available capacity will appear here as Infernet discovers them."}
          </p>

          <dl className="network-capacity-ledger">
            <div className="network-memory-fact">
              <dt>
                <MemoryStick size={17} />
                <span>Available VRAM + unified memory</span>
              </dt>
              <dd>
                <strong>{formatBytes(availableComputeBytes)}</strong>
                <span>
                  {totalComputeBytes > 0
                    ? `${capacityFreePercent}% free of ${formatBytes(totalComputeBytes)}`
                    : "Waiting for accelerator capacity"}
                </span>
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
              {backendSummaries.map((summary) => {
                const availablePercent = summary.totalBytes > 0
                  ? Math.round((summary.availableBytes / summary.totalBytes) * 100)
                  : 0;
                return (
                  <div className="network-backend-row" key={summary.backend}>
                    <div className="network-backend-copy">
                      <span className="machine-backend">{machineBackendLabel(summary.backend)}</span>
                      <div>
                        <strong>{summary.machines.length} node{summary.machines.length === 1 ? "" : "s"}</strong>
                        <span>
                          {formatBytes(summary.availableBytes)} {summary.backend === "cpu" ? "RAM" : "available"}
                        </span>
                      </div>
                    </div>
                    <div className="network-capacity-meter" aria-label={`${availablePercent}% available`}>
                      <span style={{ width: `${availablePercent}%` }} />
                    </div>
                  </div>
                );
              })}
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
            <small>{locatedNodeCount} IP-located</small>
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
                    <strong>{formatBytes(machine.availableMemoryBytes)}</strong>
                    <span>{machine.unifiedMemory ? "unified memory free" : "available"}</span>
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
        color: location.source === "geoip"
          ? [0.96, 0.96, 0.96]
          : [0.52, 0.52, 0.52],
      };
    }),
    [locations, machines],
  );

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
      markers,
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
      phi = (phi + (delta * Math.PI * 2) / 64000) % (Math.PI * 2);
      globe.update({ phi });
      frame = window.requestAnimationFrame(animate);
    };
    if (!reducedMotion) frame = window.requestAnimationFrame(animate);

    return () => {
      if (frame) window.cancelAnimationFrame(frame);
      resizeObserver.disconnect();
      globe.destroy();
    };
  }, [markers]);

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
            ? "Resolving public-IP regions"
            : locatedNodeCount > 0
              ? `${locatedNodeCount} approximate IP location${locatedNodeCount === 1 ? "" : "s"}; private and relayed nodes are anonymized`
              : "Private and relayed nodes use anonymous positions"}
        </span>
      </figcaption>
    </figure>
  );
}

function useNodeLocations(machines: MachineView[]): {
  locations: Record<string, NodeLocation>;
  isLocating: boolean;
} {
  const [locations, setLocations] = useState<Record<string, NodeLocation>>(() =>
    Object.fromEntries(machines.map((machine) => [networkMachineKey(machine), illustrativeNodeLocation(machine)])),
  );
  const [isLocating, setIsLocating] = useState(false);
  const locationSignature = machines.map((machine) =>
    `${networkMachineKey(machine)}:${machine.isLocal ? "local" : "remote"}:${machine.addresses.join(",")}`
  ).join("|");

  useEffect(() => {
    const fallbackLocations = Object.fromEntries(
      machines.map((machine) => [networkMachineKey(machine), illustrativeNodeLocation(machine)]),
    );
    setLocations(fallbackLocations);

    const lookupTargets = machines.map((machine) => ({
      machine,
      ip: publicIpFromAddresses(machine.addresses),
    })).filter((target) => target.ip || target.machine.isLocal);

    if (lookupTargets.length === 0) {
      setIsLocating(false);
      return;
    }

    let disposed = false;
    const controller = new AbortController();
    setIsLocating(true);
    Promise.all(lookupTargets.map(async ({ machine, ip }) => {
      const location = await fetchGeoIpLocation(ip, controller.signal);
      return { key: networkMachineKey(machine), location };
    })).then((results) => {
      if (disposed) return;
      const resolved = results.reduce<Record<string, NodeLocation>>((next, result) => {
        if (result.location) next[result.key] = result.location;
        return next;
      }, {});
      setLocations((current) => ({ ...current, ...resolved }));
    }).finally(() => {
      if (!disposed) setIsLocating(false);
    });

    return () => {
      disposed = true;
      controller.abort();
    };
  }, [locationSignature]);

  return { locations, isLocating };
}

async function fetchGeoIpLocation(ip: string | null, signal: AbortSignal): Promise<NodeLocation | null> {
  const cacheKey = `infernet-geo-v1-${hashText(ip ?? "self").toString(16)}`;
  try {
    const cached = window.localStorage.getItem(cacheKey);
    if (cached) {
      const parsed = JSON.parse(cached) as NodeLocation & { cachedAt: number };
      if (Date.now() - parsed.cachedAt < 7 * 24 * 60 * 60 * 1000) {
        return parsed;
      }
    }
  } catch {
    // Location caching is an optimization; lookup still works without storage.
  }

  try {
    const fields = "success,city,region,country,latitude,longitude";
    const endpoint = ip
      ? `https://ipwho.is/${encodeURIComponent(ip)}?fields=${fields}`
      : `https://ipwho.is/?fields=${fields}`;
    const response = await fetch(endpoint, { signal, referrerPolicy: "no-referrer" });
    if (!response.ok) return null;
    const data = await response.json() as {
      success?: boolean;
      city?: string;
      region?: string;
      country?: string;
      latitude?: number;
      longitude?: number;
    };
    if (
      data.success !== true
      || !Number.isFinite(data.latitude)
      || !Number.isFinite(data.longitude)
    ) {
      return null;
    }
    const labelParts = [...new Set([data.city, data.region, data.country].filter(Boolean))];
    const location: NodeLocation = {
      latitude: data.latitude as number,
      longitude: data.longitude as number,
      label: labelParts.join(", ") || "Approximate public-IP region",
      source: "geoip",
    };
    try {
      window.localStorage.setItem(cacheKey, JSON.stringify({ ...location, cachedAt: Date.now() }));
    } catch {
      // Keep the in-memory result if persistent storage is unavailable.
    }
    return location;
  } catch {
    return null;
  }
}

function publicIpFromAddresses(addresses: string[]): string | null {
  for (const address of addresses) {
    if (address.includes("/p2p-circuit")) continue;
    const parts = address.split("/").filter(Boolean);
    for (let index = 0; index < parts.length - 1; index += 1) {
      if (parts[index] !== "ip4" && parts[index] !== "ip6") continue;
      const ip = parts[index + 1];
      if (isPublicIp(ip)) return ip;
    }
  }
  return null;
}

function isPublicIp(ip: string): boolean {
  if (ip.includes(".")) {
    const octets = ip.split(".").map(Number);
    if (octets.length !== 4 || octets.some((octet) => !Number.isInteger(octet) || octet < 0 || octet > 255)) {
      return false;
    }
    const [a, b, c] = octets;
    return !(
      a === 0
      || a === 10
      || a === 127
      || (a === 100 && b >= 64 && b <= 127)
      || (a === 169 && b === 254)
      || (a === 172 && b >= 16 && b <= 31)
      || (a === 192 && b === 168)
      || (a === 192 && b === 0 && (c === 0 || c === 2))
      || (a === 198 && (b === 18 || b === 19 || (b === 51 && c === 100)))
      || (a === 203 && b === 0 && c === 113)
      || a >= 224
    );
  }

  if (!ip.includes(":")) return false;
  const normalized = ip.toLowerCase();
  if (normalized.startsWith("::ffff:")) return isPublicIp(normalized.slice(7));
  return !(
    normalized === "::"
    || normalized === "::1"
    || normalized.startsWith("fc")
    || normalized.startsWith("fd")
    || /^fe[89ab]/.test(normalized)
    || normalized.startsWith("ff")
    || normalized.startsWith("2001:db8")
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

function ModelsPage({
  snapshot,
  selectedModel,
  onModelChange,
  onInstallModel,
  installingModelId,
}: {
  snapshot: GridSnapshot;
  selectedModel: string;
  onModelChange: (modelId: string) => void;
  onInstallModel: (modelId: string) => void;
  installingModelId: string | null;
}) {
  const officialModels = snapshot.availableModels.filter(isOfficialInfernetModel);

  return (
    <section className="library-screen">
      <div className="models-topbar">
        <div className="section-heading">
          <span className="section-eyebrow">Official catalog</span>
          <h2>Infernet models</h2>
          <p>A small collection built, tested, and distributed specifically for Infernet.</p>
        </div>
      </div>

      <div className="official-model-note">
        <CheckCircle2 size={18} />
        <div>
          <strong>Curated from end to end</strong>
          <span>Infernet chooses the model, package layout, and runtime so every release works across the network.</span>
        </div>
      </div>

      <div className="model-library">
        {officialModels.length === 0 ? (
          <div className="library-card flagship-card" aria-label="Infernet Chat flagship model">
            <div>
              <span className="model-edition">Flagship · Infernet edition</span>
              <strong>Infernet Chat</strong>
              <span>Powered by Gemma 4 26B A4B</span>
            </div>
            <div className="library-status">
              <span>Official package</span>
              <ProgressBar progress={0} />
              <small>Preparing the first network release</small>
            </div>
          </div>
        ) : (
          officialModels.map((model) => {
            const installed = model.installed || snapshot.distribution.installedModels.includes(model.modelId);
            const installing = installingModelId === model.modelId;
            return (
              <div
                className={selectedModel === model.modelId ? "library-card active" : "library-card"}
                key={model.modelId}
              >
                <div>
                  <span className="model-edition">Infernet edition</span>
                  <strong>{curatedModelName(model)}</strong>
                  <span>{curatedModelBasis(model)}</span>
                </div>
                <div className="library-status">
                  <span>{installed ? "Stored on this computer" : "Available"}</span>
                  <ProgressBar progress={installed ? 100 : 0} />
                  <small>{model.runnable ? "Ready to chat" : "Preparing for this network"}</small>
                </div>
                <div className="library-actions">
                  <button className="secondary-button" onClick={() => onModelChange(model.modelId)}>
                    Use in chat
                  </button>
                  {!installed ? (
                    <button
                      className="primary-button"
                      disabled={installing}
                      onClick={() => onInstallModel(model.modelId)}
                    >
                      <Download size={15} />
                      <span>{installing ? "Preparing…" : "Store 14.4 GB here"}</span>
                    </button>
                  ) : null}
                </div>
              </div>
            );
          })
        )}
      </div>
    </section>
  );
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
              One computer hosts the verified 14.4 GB package. Compute-only computers do not need
              the full model on disk.
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
              No model is stored here. That is normal for compute-only computers.
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

function SettingsPage({
  identity,
  onNetworkChanged,
}: {
  identity: LocalIdentity | null;
  onNetworkChanged: () => void;
}) {
  const [manualPeer, setManualPeer] = useState("");
  const [manualPeers, setManualPeers] = useState<string[]>([]);
  const [manualPeerStatus, setManualPeerStatus] = useState<string | null>(null);

  useEffect(() => {
    getManualPeers()
      .then(setManualPeers)
      .catch((error) => setManualPeerStatus(String(error)));
  }, []);

  async function connectPeer() {
    try {
      const peers = await addManualPeer(manualPeer);
      setManualPeers(peers);
      setManualPeer("");
      setManualPeerStatus("Peer added. Refreshing network.");
      onNetworkChanged();
    } catch (error) {
      setManualPeerStatus(String(error));
    }
  }

  async function clearPeers() {
    try {
      const peers = await clearManualPeers();
      setManualPeers(peers);
      setManualPeerStatus("Manual peers cleared.");
      onNetworkChanged();
    } catch (error) {
      setManualPeerStatus(String(error));
    }
  }

  return (
    <section className="settings-screen">
      <div className="section-heading">
        <h2>Settings</h2>
        <p>Connection details and controls for this node.</p>
      </div>

      <div className="settings-list">
        <div className="settings-row">
          <div>
            <strong>Local node</strong>
            <span>{identity?.peerId ?? "Starting"}</span>
          </div>
        </div>
        <div className="settings-row">
          <div>
            <strong>LAN address</strong>
            <span>{identity?.connectAddresses[0] ?? "Starting network"}</span>
          </div>
        </div>
        <div className="settings-row manual-peer-settings">
          <div>
            <strong>Connect to another computer</strong>
            <span>Paste the LAN address from the other Infernet app if automatic discovery shows 0 peers.</span>
            {manualPeers.length > 0 ? (
              <small>{manualPeers.length === 1 ? manualPeers[0] : `${manualPeers.length} manual peers`}</small>
            ) : null}
            {manualPeerStatus ? <small>{manualPeerStatus}</small> : null}
          </div>
          <div className="manual-peer-controls">
            <input
              value={manualPeer}
              onChange={(event) => setManualPeer(event.target.value)}
              placeholder="/ip4/192.168.1.10/tcp/9777/p2p/12D3..."
            />
            <button className="secondary-button" onClick={connectPeer} disabled={!manualPeer.trim()}>
              Connect
            </button>
            <button className="text-button" onClick={clearPeers} disabled={manualPeers.length === 0}>
              Clear
            </button>
          </div>
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
  const activity: TransferActivity = {
    ...event,
    id,
    status,
    startedAt: existing?.startedAt ?? updatedAt,
    updatedAt,
  };
  if (activity.status === "active" && existing && existing.status !== "active" && activity.id.endsWith(":model")) {
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

function pageTitle(page: Page): string {
  if (page === "chat") return "Chat";
  if (page === "network") return "Network";
  if (page === "models") return "Models";
  if (page === "downloads") return "Downloads";
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
  if (status === "error") return "Infernet couldn’t complete this model task.";
  if (status === "complete") return "This model is ready to use.";
  const normalized = stage.toLowerCase();
  if (normalized.includes("verifying") || normalized.includes("checking")) {
    return "Confirming the model is complete and trusted.";
  }
  if (normalized.includes("sharing")) return "Getting the model ready for other computers.";
  if (normalized.includes("preparing")) return "Optimizing the official package for this computer.";
  return "Receiving the official model package.";
}

function modelDisplayName(snapshot: GridSnapshot, modelId: string): string {
  const model = snapshot.availableModels.find(
    (item) => item.modelId === modelId && isOfficialInfernetModel(item),
  );
  return model ? curatedModelName(model) : "Infernet Chat";
}

function isOfficialModelId(modelId: string): boolean {
  return modelId === INFERNET_CHAT_MODEL_ID;
}

function isOfficialInfernetModel(model: ModelView): boolean {
  return isOfficialModelId(model.modelId);
}

function curatedModelName(_model: ModelView): string {
  return "Infernet Chat";
}

function curatedModelBasis(model: ModelView): string {
  return isOfficialInfernetModel(model) ? "Powered by Gemma 4 26B A4B" : "Unofficial package";
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
