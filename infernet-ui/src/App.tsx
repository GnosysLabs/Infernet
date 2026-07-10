import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  Box,
  CheckCircle2,
  ChevronDown,
  Download,
  HardDrive,
  Laptop2,
  MessageSquare,
  Network,
  PanelRightClose,
  RefreshCw,
  Send,
  Settings,
} from "lucide-react";
import {
  addManualPeer,
  clearManualPeers,
  emptySnapshot,
  getGridSnapshot,
  getLocalIdentity,
  getManualPeers,
  installOfficialModel,
  listenForProgress,
  listenForModelImportProgress,
  runDistributedInference,
} from "./api";
import type {
  ExecutionParticipantView,
  GridSnapshot,
  HopProgress,
  LocalIdentity,
  MachineView,
  ModelImportProgress,
  ModelView,
  ProgressEvent,
  RouteHopView,
} from "./types";

type Page = "chat" | "models" | "downloads" | "settings";
type Message = { id: string; role: "user" | "assistant"; text: string };
type TransferStatus = "active" | "complete" | "error";
type TransferActivity = ModelImportProgress & {
  id: string;
  status: TransferStatus;
  startedAt: number;
  updatedAt: number;
};

const DEFAULT_PROMPT = "";
const INFERNET_CHAT_MODEL_ID = "infernet-chat-v1";

export default function App() {
  const [page, setPage] = useState<Page>("chat");
  const [developerMode, setDeveloperMode] = useState(false);
  const [activityOpen, setActivityOpen] = useState(false);
  const [identity, setIdentity] = useState<LocalIdentity | null>(null);
  const [snapshot, setSnapshot] = useState<GridSnapshot>(emptySnapshot);
  const [selectedModel, setSelectedModel] = useState("");
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [messages, setMessages] = useState<Message[]>([]);
  const [status, setStatus] = useState("Starting");
  const [isRunning, setIsRunning] = useState(false);
  const [runStartedAt, setRunStartedAt] = useState<number | null>(null);
  const [lastRunDurationMs, setLastRunDurationMs] = useState<number | null>(null);
  const [hops, setHops] = useState<HopProgress[]>([]);
  const [route, setRoute] = useState<RouteHopView[]>([]);
  const [executionPlan, setExecutionPlan] = useState<ExecutionParticipantView[]>([]);
  const [executionConfirmed, setExecutionConfirmed] = useState(false);
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
  const activeRoute = selectedModelView ? (route.length > 0 ? route : snapshot.route) : [];
  const activeTransfers = transferActivities.filter((activity) => activity.status === "active").length;

  const applyProgressEvent = useCallback((event: ProgressEvent) => {
    if (event.type === "routeDiscovered") {
      setRoute(event.route);
      setHops(
        event.route.map((hop) => ({
          key: hopKey(hop.peerId, hop.layerStart, hop.layerEnd),
          peerId: hop.peerId,
          shortPeerId: hop.shortPeerId,
          layerStart: hop.layerStart,
          layerEnd: hop.layerEnd,
          activationSizeBytes: 0,
          status: "pending",
        })),
      );
      setStatus("Finding available compute");
      setLastError(null);
      return;
    }

    if (event.type === "executionPlan") {
      setExecutionPlan(event.participants);
      setExecutionConfirmed(false);
      setStatus("Starting distributed model");
      setLastError(null);
      return;
    }

    if (event.type === "hopStarted") {
      setHops((current) => upsertHop(current, event, "running"));
      setStatus("Running model");
      return;
    }

    if (event.type === "hopCompleted") {
      setHops((current) => upsertHop(current, event, "complete"));
      setStatus("Finishing response");
      return;
    }

    if (event.type === "finalOutput") {
      setExecutionConfirmed(true);
      setStatus("Ready");
      setIsRunning(false);
      return;
    }

    if (event.type === "error") {
      setExecutionConfirmed(false);
      setExecutionPlan([]);
      setLastError(event.message);
      setStatus("Needs attention");
      setIsRunning(false);
    }
  }, []);

  const refreshSnapshot = useCallback(async (modelId?: string) => {
    setStatus("Connecting");
    try {
      const nextSnapshot = await getGridSnapshot(4000, modelId);
      const nextOfficialModels = nextSnapshot.availableModels.filter(isOfficialInfernetModel);
      const modelStillExists = modelId && nextOfficialModels.some((model) => model.modelId === modelId);
      const nextSelectedModel = modelStillExists
        ? modelId
        : nextOfficialModels.find((model) => model.modelId === nextSnapshot.selectedModel)?.modelId
          || nextOfficialModels[0]?.modelId
          || "";
      const nextModel = nextOfficialModels.find((model) => model.modelId === nextSelectedModel);
      setSnapshot(nextSnapshot);
      setRoute(nextSelectedModel ? nextSnapshot.route : []);
      setLastError(nextSelectedModel ? nextSnapshot.missingRanges ?? null : null);
      setSelectedModel(nextSelectedModel);
      setStatus(
        nextOfficialModels.length === 0
          ? "No models"
          : nextModel?.runnable
            ? "Ready"
            : "Connected",
      );
    } catch (error) {
      setLastError(String(error));
      setStatus("Offline");
    }
  }, []);

  useEffect(() => {
    getLocalIdentity().then(setIdentity).catch((error) => setLastError(String(error)));
  }, []);

  useEffect(() => {
    refreshSnapshot(selectedModel);
  }, [refreshSnapshot, selectedModel]);

  useEffect(() => {
    if (!activityOpen && page !== "downloads") return;
    let disposed = false;
    let inFlight = false;
    const refreshActivity = async () => {
      if (inFlight) return;
      inFlight = true;
      try {
        const nextSnapshot = await getGridSnapshot(2500, selectedModel);
        if (!disposed) {
          setSnapshot(nextSnapshot);
          if (!isRunning) setRoute(nextSnapshot.route);
        }
      } catch {
        // The primary refresh path owns user-visible connection errors.
      } finally {
        inFlight = false;
      }
    };
    const interval = window.setInterval(refreshActivity, 6000);
    return () => {
      disposed = true;
      window.clearInterval(interval);
    };
  }, [activityOpen, isRunning, page, selectedModel]);

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
  }, []);

  function handleModelChange(modelId: string) {
    setSelectedModel(modelId);
    setPage("chat");
    setRoute([]);
    setHops([]);
    setExecutionPlan([]);
    setExecutionConfirmed(false);
    setLastError(null);
    setStatus("Connecting");
  }

  async function handleInstallModel(modelId: string) {
    if (installingModelId) return;
    setInstallingModelId(modelId);
    setActivityOpen(true);
    setLastError(null);
    setStatus("Preparing model storage");
    try {
      const nextSnapshot = await installOfficialModel(modelId);
      setSnapshot(nextSnapshot);
      setRoute(nextSnapshot.route);
      setStatus("Ready");
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
      setStatus("Needs attention");
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
      setStatus("No models");
      return;
    }
    if (!selectedModelView.runnable) {
      setLastError(selectedModelView.status);
      setStatus("Model not ready");
      return;
    }
    setMessages((current) => [
      ...current,
      { id: `user-${Date.now()}`, role: "user", text: userPrompt },
    ]);
    setPrompt("");
    setIsRunning(true);
    const startedAt = Date.now();
    setRunStartedAt(startedAt);
    setLastError(null);
    setStatus("Finding available compute");
    setHops([]);
    setExecutionPlan([]);
    setExecutionConfirmed(false);

    try {
      const output = (await runDistributedInference(userPrompt, selectedModel)).output;

      setMessages((current) => [
        ...current,
        { id: `assistant-${Date.now()}`, role: "assistant", text: output },
      ]);
      setStatus("Ready");
    } catch (error) {
      const message = String(error);
      setExecutionPlan([]);
      setExecutionConfirmed(false);
      setLastError(message);
      setStatus("Needs attention");
    } finally {
      setLastRunDurationMs(Date.now() - startedAt);
      setRunStartedAt(null);
      setIsRunning(false);
    }
  }

  return (
    <div className={activityOpen ? "app-shell activity-open" : "app-shell"}>
      <Sidebar
        page={page}
        setPage={setPage}
        developerMode={developerMode}
        setDeveloperMode={setDeveloperMode}
      />

      <main className="app-main">
        <AppHeader
          page={page}
          model={selectedModelView}
          onRefresh={() => refreshSnapshot(selectedModel)}
          activityOpen={activityOpen}
          hasActiveWork={isRunning || activeTransfers > 0}
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
            developerMode={developerMode}
          />
        ) : null}

        {page === "settings" ? (
          <SettingsPage
            identity={identity}
            developerMode={developerMode}
            setDeveloperMode={setDeveloperMode}
            onNetworkChanged={() => refreshSnapshot(selectedModel)}
          />
        ) : null}
      </main>

      {activityOpen ? (
        <ActivitySidebar
          snapshot={snapshot}
          model={selectedModelView}
          route={activeRoute}
          executionPlan={executionPlan}
          executionConfirmed={executionConfirmed}
          hops={hops}
          transferActivities={transferActivities}
          status={status}
          isRunning={isRunning}
          runStartedAt={runStartedAt}
          lastRunDurationMs={lastRunDurationMs}
          lastError={lastError}
          developerMode={developerMode}
          identity={identity}
          onClose={() => setActivityOpen(false)}
        />
      ) : null}
    </div>
  );
}

function Sidebar({
  page,
  setPage,
  developerMode,
  setDeveloperMode,
}: {
  page: Page;
  setPage: (page: Page) => void;
  developerMode: boolean;
  setDeveloperMode: (enabled: boolean) => void;
}) {
  return (
    <aside className="sidebar">
      <div className="brand-block">
        <div className="brand-mark">
          <Network size={24} />
        </div>
        <div>
          <div className="brand-name">Infernet</div>
          <div className="brand-version">v0.2.0</div>
        </div>
      </div>

      <nav className="nav-list" aria-label="Primary">
        <NavButton icon={<MessageSquare size={18} />} label="Chat" active={page === "chat"} onClick={() => setPage("chat")} />
        <NavButton icon={<Box size={18} />} label="Models" active={page === "models"} onClick={() => setPage("models")} />
        <NavButton icon={<Download size={18} />} label="Downloads" active={page === "downloads"} onClick={() => setPage("downloads")} />
        <NavButton icon={<Settings size={18} />} label="Settings" active={page === "settings"} onClick={() => setPage("settings")} />
      </nav>

      <label className="developer-toggle">
        <span>Developer Mode</span>
        <input
          type="checkbox"
          checked={developerMode}
          onChange={(event) => setDeveloperMode(event.target.checked)}
        />
      </label>
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
  onRefresh,
  activityOpen,
  hasActiveWork,
  onToggleActivity,
}: {
  page: Page;
  model?: ModelView;
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
          <span>{model ? curatedModelName(model) : "No model selected"}</span>
        </div>
      </div>

      <div className="header-actions">
        <button
          className={activityOpen ? "activity-toggle active" : "activity-toggle"}
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
  model,
  route,
  executionPlan,
  executionConfirmed,
  hops,
  transferActivities,
  status,
  isRunning,
  runStartedAt,
  lastRunDurationMs,
  lastError,
  developerMode,
  identity,
  onClose,
}: {
  snapshot: GridSnapshot;
  model?: ModelView;
  route: RouteHopView[];
  executionPlan: ExecutionParticipantView[];
  executionConfirmed: boolean;
  hops: HopProgress[];
  transferActivities: TransferActivity[];
  status: string;
  isRunning: boolean;
  runStartedAt: number | null;
  lastRunDurationMs: number | null;
  lastError: string | null;
  developerMode: boolean;
  identity: LocalIdentity | null;
  onClose: () => void;
}) {
  const elapsedMs = useElapsedTime(runStartedAt);
  const peerIds = [...new Set(route.map((hop) => hop.peerId))];
  const ranLocally = peerIds.length > 0 && peerIds.every((peerId) => peerId === snapshot.localPeerId);
  const computeMs = hops.reduce((total, hop) => total + (hop.timingMs ?? 0), 0);
  const activeTransfers = transferActivities.filter((activity) => activity.status === "active");
  const recentTransfer = transferActivities.find((activity) => activity.status !== "active");
  const modelHosts = snapshot.machines.filter(
    (machine) => machine.hostedComponentCount > 0 && machine.connectionStatus !== "unreachable",
  );
  const location = executionPlan.length > 0
    ? executionConfirmed
      ? `${executionPlan.length} computers completed the last response`
      : `Planned across ${executionPlan.length} computers`
    : peerIds.length === 0
    ? isRunning ? "Choosing a computer" : "Not used yet"
    : ranLocally
      ? "This computer"
      : `${peerIds.length} network computer${peerIds.length === 1 ? "" : "s"}`;
  const currentStatus = lastError ? "Needs attention" : status;

  return (
    <aside className="activity-sidebar" id="activity-sidebar" aria-label="Activity">
      <div className="activity-sidebar-header">
        <div>
          <span>Activity</span>
          <h2>What Infernet is doing</h2>
        </div>
        <button className="icon-button" aria-label="Close activity" onClick={onClose}>
          <PanelRightClose size={18} />
        </button>
      </div>

      <div className="activity-sidebar-scroll">
        <section className="activity-primary" aria-live="polite">
          <div className="activity-status-line">
            <span className={isRunning ? "activity-pulse active" : "activity-pulse"} />
            <div>
              <strong>{currentStatus}</strong>
              <span>{isRunning ? "Working on your response" : "Ready when you are"}</span>
            </div>
          </div>

          <dl className="activity-data-list">
            <ActivityDataRow label="Model" value={model ? curatedModelName(model) : "None selected"} />
            <ActivityDataRow label={isRunning ? "Running on" : "Available on"} value={location} />
            <ActivityDataRow
              label={isRunning ? "Elapsed" : "Last response"}
              value={formatDuration(isRunning ? elapsedMs : lastRunDurationMs)}
            />
            <ActivityDataRow label="Compute time" value={computeMs > 0 ? formatDuration(computeMs) : "—"} />
            <ActivityDataRow
              label="Other computers online"
              value={String(snapshot.networkPeerCount)}
            />
            <ActivityDataRow
              label="Distributed workers ready"
              value={String(snapshot.machines.filter((machine) => machine.rpcReady).length)}
            />
            <ActivityDataRow
              label="Model availability"
              value={modelHosts.length > 0
                ? `Hosted by ${modelHosts.length} computer${modelHosts.length === 1 ? "" : "s"}`
                : activeTransfers.length > 0
                  ? "Downloading on this computer"
                  : "Not hosted on the network"}
            />
          </dl>
        </section>

        {lastError ? (
          <section className="activity-alert" role="alert">
            <strong>What happened</strong>
            <span>{friendlyActivityError(lastError)}</span>
          </section>
        ) : null}

        <section className="activity-sidebar-section">
          <div className="sidebar-section-heading">
            <strong>Available compute</strong>
            <span>{snapshot.machines.length} machine{snapshot.machines.length === 1 ? "" : "s"}</span>
          </div>
          {snapshot.machines.length === 0 ? (
            <div className="activity-quiet inline">
              <Laptop2 size={17} />
              <span>Waiting for machine capacity reports.</span>
            </div>
          ) : (
            <div className="machine-list">
              {snapshot.machines.map((machine) => (
                <MachineStatusCard
                  machine={machine}
                  route={route}
                  executionPlan={executionPlan}
                  executionConfirmed={executionConfirmed}
                  isRunning={isRunning}
                  localTransfer={machine.isLocal ? activeTransfers[0] : undefined}
                  developerMode={developerMode}
                  key={machine.machineId ?? machine.peerId}
                />
              ))}
            </div>
          )}
        </section>

        {activeTransfers.length > 0 ? (
          <section className="activity-sidebar-section">
            <div className="sidebar-section-heading">
              <strong>Preparing models</strong>
              <span>{activeTransfers.length} active</span>
            </div>
            <div className="transfer-list">
              {activeTransfers.slice(0, 3).map((activity) => (
                <TransferActivityRow
                  activity={activity}
                  modelName={modelDisplayName(snapshot, activity.modelId)}
                  developerMode={developerMode}
                  key={activity.id}
                />
              ))}
            </div>
          </section>
        ) : recentTransfer ? (
          <section className="activity-sidebar-section">
            <div className="sidebar-section-heading">
              <strong>Latest model activity</strong>
              <span>{formatRelativeTime(recentTransfer.updatedAt)}</span>
            </div>
            <TransferActivityRow
              activity={recentTransfer}
              modelName={modelDisplayName(snapshot, recentTransfer.modelId)}
              developerMode={developerMode}
            />
          </section>
        ) : (
          <div className="activity-quiet">
            <Laptop2 size={18} />
            <span>Model preparation and response activity will appear here.</span>
          </div>
        )}

        {developerMode ? (
          <details className="technical-details">
            <summary>Technical details</summary>
            <div className="technical-detail-list">
              <code>Local peer: {identity?.peerId ?? snapshot.localPeerId ?? "starting"}</code>
              {lastError ? <code>Last error: {lastError}</code> : null}
              {route.length === 0 ? <span>No execution route selected.</span> : route.map((hop) => {
                const progress = hops.find((item) => item.key === hopKey(hop.peerId, hop.layerStart, hop.layerEnd));
                return (
                  <code key={`${hop.peerId}-${hop.layerStart}-${hop.layerEnd}`}>
                    {hop.shortPeerId} · layers {hop.layerStart}:{hop.layerEnd}
                    {progress?.activationSizeBytes ? ` · ${formatBytes(progress.activationSizeBytes)}` : ""}
                    {progress?.timingMs ? ` · ${progress.timingMs} ms` : ""}
                  </code>
                );
              })}
            </div>
          </details>
        ) : null}
      </div>
    </aside>
  );
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
  route,
  executionPlan,
  executionConfirmed,
  isRunning,
  localTransfer,
  developerMode,
}: {
  machine: MachineView;
  route: RouteHopView[];
  executionPlan: ExecutionParticipantView[];
  executionConfirmed: boolean;
  isRunning: boolean;
  localTransfer?: TransferActivity;
  developerMode: boolean;
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
  const executionRole = machineRoleLabel(
    machine.peerId,
    route,
    executionPlan,
    executionConfirmed,
    isRunning,
  );

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
      {executionRole ? <span className="machine-role">{executionRole}</span> : null}
      {developerMode ? (
        <code>
          {machine.peerId} · last seen {machine.lastSeenSeconds}s ago
        </code>
      ) : null}
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
  developerMode,
}: {
  activity: TransferActivity;
  modelName: string;
  developerMode: boolean;
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
      {developerMode ? <code>{activity.detail}</code> : null}
    </div>
  );
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
  developerMode,
}: {
  snapshot: GridSnapshot;
  transferActivities: TransferActivity[];
  developerMode: boolean;
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
                developerMode={developerMode}
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
                route={[]}
                executionPlan={[]}
                executionConfirmed={false}
                isRunning={false}
                localTransfer={machine.isLocal ? activeModelTransfer : undefined}
                developerMode={developerMode}
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
                  {developerMode ? (
                    <details className="model-technical-details">
                      <summary>Technical details</summary>
                      <code>
                        {item.packageCount} package{item.packageCount === 1 ? "" : "s"} · layers {item.layerStart}:{item.layerEnd} · {item.version}
                      </code>
                      {item.checksums.map((checksum) => <code key={checksum}>{checksum}</code>)}
                    </details>
                  ) : null}
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
                developerMode={developerMode}
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
  developerMode,
  setDeveloperMode,
  onNetworkChanged,
}: {
  identity: LocalIdentity | null;
  developerMode: boolean;
  setDeveloperMode: (enabled: boolean) => void;
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
        <p>Keep the app simple by default. Reveal technical details when needed.</p>
      </div>

      <div className="settings-list">
        <label className="settings-row">
          <div>
            <strong>Developer Mode</strong>
            <span>Show peer IDs, layer groups, protocol details, and route timing.</span>
          </div>
          <input
            type="checkbox"
            checked={developerMode}
            onChange={(event) => setDeveloperMode(event.target.checked)}
          />
        </label>
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

function upsertHop(
  current: HopProgress[],
  event: Extract<ProgressEvent, { type: "hopStarted" | "hopCompleted" }>,
  status: HopProgress["status"],
): HopProgress[] {
  const key = hopKey(event.peerId, event.layerStart, event.layerEnd);
  const nextHop: HopProgress = {
    key,
    peerId: event.peerId,
    shortPeerId: event.shortPeerId,
    layerStart: event.layerStart,
    layerEnd: event.layerEnd,
    activationSizeBytes: event.activationSizeBytes,
    status,
    timingMs: event.type === "hopCompleted" ? event.timingMs : undefined,
    activationChecksum: event.type === "hopCompleted" ? event.activationChecksum : undefined,
  };

  return current.some((hop) => hop.key === key)
    ? current.map((hop) => (hop.key === key ? { ...hop, ...nextHop } : hop))
    : [...current, nextHop];
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

function hopKey(peerId: string, layerStart: number, layerEnd: number): string {
  return `${peerId}:${layerStart}:${layerEnd}`;
}

function pageTitle(page: Page): string {
  if (page === "chat") return "Chat";
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

function machineRoleLabel(
  peerId: string,
  route: RouteHopView[],
  executionPlan: ExecutionParticipantView[],
  executionConfirmed: boolean,
  isRunning: boolean,
): string | null {
  const participant = executionPlan.find((item) => item.peerId === peerId);
  if (participant?.role === "coordinator") {
    return executionConfirmed
      ? `Coordinated last response · ≈${participant.estimatedSharePercent}% share`
      : `Planned coordinator · ≈${participant.estimatedSharePercent}% share`;
  }
  if (participant) {
    return executionConfirmed
      ? `Used in last response · ≈${participant.estimatedSharePercent}% share`
      : `Planned worker · ≈${participant.estimatedSharePercent}% share`;
  }
  const assignment = route.find((hop) => hop.peerId === peerId);
  if (assignment && isRunning) {
    return `Working on layers ${assignment.layerStart + 1}–${assignment.layerEnd}`;
  }
  if (assignment) {
    return `Assigned layers ${assignment.layerStart + 1}–${assignment.layerEnd}`;
  }
  return null;
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
  return "The model stopped unexpectedly. The technical error is available in Developer Mode.";
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
