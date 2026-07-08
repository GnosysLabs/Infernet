import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  Box,
  CheckCircle2,
  ChevronDown,
  Cloud,
  Download,
  FilePlus2,
  HardDrive,
  KeyRound,
  MessageSquare,
  Network,
  RefreshCw,
  Send,
  Settings,
  SlidersHorizontal,
  Search,
  UploadCloud,
  Wifi,
} from "lucide-react";
import {
  addHuggingFaceModel,
  addLocalGgufModel,
  chooseLocalModelFile,
  clearHuggingFaceToken,
  getGridSnapshot,
  getHuggingFaceSettings,
  getLocalIdentity,
  inspectHuggingFaceRepo,
  isTauriRuntime,
  listenForProgress,
  listenForModelImportProgress,
  runDistributedInference,
  saveHuggingFaceToken,
} from "./api";
import { runMockDemo, sampleSnapshot } from "./sampleData";
import type {
  AddModelResponse,
  GridSnapshot,
  HuggingFaceFileView,
  HuggingFaceSettings,
  HopProgress,
  LocalIdentity,
  ModelImportProgress,
  ModelView,
  ProgressEvent,
  RouteHopView,
} from "./types";

type Page = "chat" | "models" | "downloads" | "settings";
type Message = { id: string; role: "user" | "assistant"; text: string };

const DEFAULT_PROMPT = "";
const INITIAL_MESSAGES: Message[] = [
  {
    id: "welcome",
    role: "assistant",
    text: "Ask Infernet anything. It will find the right model and start thinking.",
  },
];

export default function App() {
  const [page, setPage] = useState<Page>("chat");
  const [developerMode, setDeveloperMode] = useState(false);
  const [showNetwork, setShowNetwork] = useState(false);
  const [identity, setIdentity] = useState<LocalIdentity | null>(null);
  const [snapshot, setSnapshot] = useState<GridSnapshot>(sampleSnapshot);
  const [selectedModel, setSelectedModel] = useState(sampleSnapshot.selectedModel);
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [messages, setMessages] = useState<Message[]>(INITIAL_MESSAGES);
  const [status, setStatus] = useState("Connected");
  const [isRunning, setIsRunning] = useState(false);
  const [hops, setHops] = useState<HopProgress[]>([]);
  const [route, setRoute] = useState<RouteHopView[]>(sampleSnapshot.route);
  const [lastError, setLastError] = useState<string | null>(null);

  const selectedModelView = useMemo(
    () => snapshot.availableModels.find((model) => model.modelId === snapshot.selectedModel),
    [snapshot.availableModels, snapshot.selectedModel],
  );
  const activeRoute = route.length > 0 ? route : snapshot.route;
  const peerCount = uniquePeerCount(activeRoute);
  const completedHops = hops.filter((hop) => hop.status === "complete").length;

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
      setStatus("Route ready");
      setLastError(null);
      return;
    }

    if (event.type === "hopStarted") {
      setHops((current) => upsertHop(current, event, "running"));
      setStatus("Thinking");
      return;
    }

    if (event.type === "hopCompleted") {
      setHops((current) => upsertHop(current, event, "complete"));
      setStatus("Generating");
      return;
    }

    if (event.type === "finalOutput") {
      setStatus("Connected");
      setIsRunning(false);
      return;
    }

    if (event.type === "error") {
      setLastError(event.message);
      setStatus("Needs attention");
      setIsRunning(false);
    }
  }, []);

  const refreshSnapshot = useCallback(async (modelId: string) => {
    setStatus("Connecting");
    try {
      const nextSnapshot = await getGridSnapshot(4000, modelId);
      setSnapshot(nextSnapshot);
      setRoute(nextSnapshot.route);
      setLastError(nextSnapshot.missingRanges ?? null);
      setStatus(nextSnapshot.route.length > 0 ? "Connected" : "Model unavailable");
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
    let unlisten: (() => void) | undefined;
    listenForProgress(applyProgressEvent).then((dispose) => {
      unlisten = dispose;
    });

    return () => {
      unlisten?.();
    };
  }, [applyProgressEvent]);

  function handleModelChange(modelId: string) {
    setSelectedModel(modelId);
    setPage("chat");
    setRoute([]);
    setHops([]);
    setLastError(null);
    setStatus("Connecting");
  }

  async function handleModelImported(modelId: string) {
    setSelectedModel(modelId);
    await refreshSnapshot(modelId);
  }

  async function runInference() {
    const userPrompt = prompt.trim();
    if (!userPrompt || isRunning) {
      return;
    }

    setMessages((current) => [
      ...current,
      { id: `user-${Date.now()}`, role: "user", text: userPrompt },
    ]);
    setPrompt("");
    setIsRunning(true);
    setShowNetwork(false);
    setLastError(null);
    setStatus("Thinking");
    setHops([]);

    try {
      const output = isTauriRuntime
        ? (await runDistributedInference(userPrompt, selectedModel)).output
        : await runMockDemo(activeRoute, applyProgressEvent);

      setMessages((current) => [
        ...current,
        { id: `assistant-${Date.now()}`, role: "assistant", text: output },
      ]);
      setStatus("Connected");
    } catch (error) {
      const message = String(error);
      const runtimePending = isRuntimePendingMessage(message);
      setLastError(message);
      if (!runtimePending) {
        setMessages((current) => [
          ...current,
          { id: `assistant-error-${Date.now()}`, role: "assistant", text: message },
        ]);
      }
      setStatus(runtimePending ? "Runtime pending" : "Needs attention");
    } finally {
      setIsRunning(false);
    }
  }

  return (
    <div className="app-shell">
      <Sidebar
        page={page}
        setPage={setPage}
        developerMode={developerMode}
        setDeveloperMode={setDeveloperMode}
      />

      <main className="app-main">
        <AppHeader
          model={selectedModelView}
          status={status}
          peerCount={snapshot.peers.length}
          onRefresh={() => refreshSnapshot(selectedModel)}
        />

        {page === "chat" ? (
          <ChatPage
            messages={messages}
            prompt={prompt}
            setPrompt={setPrompt}
            runInference={runInference}
            isRunning={isRunning}
            model={selectedModelView}
            route={activeRoute}
            hops={hops}
            peerCount={peerCount}
            completedHops={completedHops}
            lastError={lastError}
            showNetwork={showNetwork}
            setShowNetwork={setShowNetwork}
            developerMode={developerMode}
          />
        ) : null}

        {page === "models" ? (
          <ModelsPage
            snapshot={snapshot}
            selectedModel={selectedModel}
            onModelChange={handleModelChange}
            onModelImported={handleModelImported}
          />
        ) : null}

        {page === "downloads" ? <DownloadsPage snapshot={snapshot} developerMode={developerMode} /> : null}

        {page === "settings" ? (
          <SettingsPage
            identity={identity}
            developerMode={developerMode}
            setDeveloperMode={setDeveloperMode}
          />
        ) : null}
      </main>
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
  model,
  status,
  peerCount,
  onRefresh,
}: {
  model?: ModelView;
  status: string;
  peerCount: number;
  onRefresh: () => void;
}) {
  return (
    <header className="app-header">
      <div>
        <h1>Infernet</h1>
        <div className="header-meta">
          <span>Model: {model?.displayName ?? "Select a model"}</span>
          <span>Connected to AI Grid</span>
        </div>
      </div>

      <div className="header-actions">
        <span className="connection-pill">
          <Wifi size={15} />
          {peerCount > 0 ? `${peerCount} peers` : status}
        </span>
        <button className="icon-button" aria-label="Refresh network" onClick={onRefresh}>
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
  route,
  hops,
  peerCount,
  completedHops,
  lastError,
  showNetwork,
  setShowNetwork,
  developerMode,
}: {
  messages: Message[];
  prompt: string;
  setPrompt: (prompt: string) => void;
  runInference: () => void;
  isRunning: boolean;
  model?: ModelView;
  route: RouteHopView[];
  hops: HopProgress[];
  peerCount: number;
  completedHops: number;
  lastError: string | null;
  showNetwork: boolean;
  setShowNetwork: (show: boolean) => void;
  developerMode: boolean;
}) {
  const networkVisible = showNetwork || developerMode;

  return (
    <section className="chat-screen">
      <div className="conversation">
        {messages.map((message) => (
          <div key={message.id} className={`message-row ${message.role}`}>
            <div className="message-bubble">{message.text}</div>
          </div>
        ))}

        {(isRunning || hops.length > 0 || lastError) ? (
          <RunStatusCard
            isRunning={isRunning}
            peerCount={peerCount}
            completedHops={completedHops}
            totalHops={route.length}
            lastError={lastError}
            showNetwork={networkVisible}
            setShowNetwork={setShowNetwork}
          />
        ) : null}

        {networkVisible ? (
          <NetworkActivity route={route} hops={hops} model={model} />
        ) : null}
      </div>

      <div className="composer">
        <div className="composer-model">
          <Box size={15} />
          <span>{model?.displayName ?? "No model selected"}</span>
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
          placeholder="Message Infernet"
        />
        <button className="send-button" onClick={runInference} disabled={isRunning || !prompt.trim()}>
          {isRunning ? <Activity size={18} /> : <Send size={18} />}
          <span>Send</span>
        </button>
      </div>
    </section>
  );
}

function RunStatusCard({
  isRunning,
  peerCount,
  completedHops,
  totalHops,
  lastError,
  showNetwork,
  setShowNetwork,
}: {
  isRunning: boolean;
  peerCount: number;
  completedHops: number;
  totalHops: number;
  lastError: string | null;
  showNetwork: boolean;
  setShowNetwork: (show: boolean) => void;
}) {
  const runtimePending = lastError ? isRuntimePendingMessage(lastError) : false;
  const label = lastError
    ? runtimePending ? "Runtime pending" : "Needs attention"
    : isRunning
      ? "Thinking..."
      : "Response ready";
  const detail = runtimePending
    ? `${totalHops} shard groups ready`
    : peerCount > 1
    ? `Connected to ${peerCount} peers`
    : "Running on Community Compute";
  const phase = lastError
    ? runtimePending ? "Token generation runtime not connected yet." : lastError
    : isRunning
      ? phaseLabel(completedHops, totalHops)
      : "Done";

  return (
    <div className={runtimePending ? "run-card notice" : lastError ? "run-card error" : "run-card"}>
      <div className="run-card-main">
        <span className="run-spinner" />
        <div>
          <strong>{label}</strong>
          <span>{detail}</span>
        </div>
      </div>
      <div className="run-card-phase">{phase}</div>
      <button className="text-button" onClick={() => setShowNetwork(!showNetwork)}>
        {showNetwork ? "Hide network activity" : "Show network activity"}
      </button>
    </div>
  );
}

function NetworkActivity({
  route,
  hops,
  model,
}: {
  route: RouteHopView[];
  hops: HopProgress[];
  model?: ModelView;
}) {
  return (
    <div className="network-activity">
      <div className="activity-header">
        <div>
          <span>Network Activity</span>
          <strong>{model?.displayName ?? "Selected model"}</strong>
        </div>
        <SlidersHorizontal size={17} />
      </div>

      <div className="activity-timeline">
        {route.length === 0 ? (
          <div className="empty-state">No route is available for this model yet.</div>
        ) : (
          route.map((hop, index) => {
            const progress = hops.find((item) => item.key === hopKey(hop.peerId, hop.layerStart, hop.layerEnd));
            return (
              <div className="activity-hop" key={`${hop.peerId}-${hop.layerStart}`}>
                <span>{index + 1}</span>
                <div>
                  <strong>{progress?.status === "complete" ? "Completed" : progress?.status === "running" ? "Running" : "Ready"}</strong>
                  <small>
                    Layer group {hop.layerStart}:{hop.layerEnd}
                    {progress?.timingMs ? ` - ${progress.timingMs} ms` : ""}
                  </small>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

function ModelsPage({
  snapshot,
  selectedModel,
  onModelChange,
  onModelImported,
}: {
  snapshot: GridSnapshot;
  selectedModel: string;
  onModelChange: (modelId: string) => void;
  onModelImported: (modelId: string) => Promise<void>;
}) {
  const [showImporter, setShowImporter] = useState(false);
  const [source, setSource] = useState<"local" | "huggingface">("local");
  const [showTokenInput, setShowTokenInput] = useState(false);
  const [localPath, setLocalPath] = useState("");
  const [hfRepo, setHfRepo] = useState("bartowski/Llama-3.2-1B-Instruct-GGUF");
  const [hfToken, setHfToken] = useState("");
  const [hfFiles, setHfFiles] = useState<HuggingFaceFileView[]>([]);
  const [selectedHfFile, setSelectedHfFile] = useState("");
  const [isInspecting, setIsInspecting] = useState(false);
  const [isAdding, setIsAdding] = useState(false);
  const [importProgress, setImportProgress] = useState<ModelImportProgress | null>(null);
  const [result, setResult] = useState<AddModelResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    listenForModelImportProgress((event) => {
      setImportProgress(event);
    }).then((dispose) => {
      unlisten = dispose;
    });

    return () => {
      unlisten?.();
    };
  }, []);

  function openImporter() {
    setResult(null);
    setError(null);
    setImportProgress(null);
    setShowImporter(true);
  }

  async function inspectRepo() {
    setIsInspecting(true);
    setError(null);
    try {
      const files = await inspectHuggingFaceRepo(hfRepo, hfToken);
      setHfFiles(files);
      setSelectedHfFile(files[0]?.filename ?? "");
      if (files.length === 0) {
        setError("No GGUF files were found in that repository.");
      }
    } catch (inspectError) {
      setError(String(inspectError));
    } finally {
      setIsInspecting(false);
    }
  }

  async function chooseFile(): Promise<string | null> {
    setError(null);
    try {
      const selected = await chooseLocalModelFile();
      if (selected) {
        setLocalPath(selected);
      }
      return selected;
    } catch (chooseError) {
      setError(String(chooseError));
      return null;
    }
  }

  async function addModel(pathOverride?: string) {
    setIsAdding(true);
    setError(null);
    setResult(null);
    setImportProgress({
      modelId: "importing",
      stage: source === "huggingface" ? "Starting download" : "Starting import",
      detail: source === "huggingface" ? hfRepo : pathOverride ?? localPath,
      downloadedBytes: 0,
      totalBytes: null,
    });
    try {
      const response = source === "local"
        ? await addLocalGgufModel(pathOverride ?? localPath)
        : await addHuggingFaceModel(hfRepo, selectedHfFile, hfToken);
      setResult(response);
      setLocalPath("");
      setImportProgress({
        modelId: response.modelId,
        stage: "Ready",
        detail: "Infernet is sharing this model",
        downloadedBytes: response.sourceSizeBytes,
        totalBytes: response.sourceSizeBytes,
      });
      await onModelImported(response.modelId);
    } catch (addError) {
      setError(String(addError));
    } finally {
      setIsAdding(false);
    }
  }

  async function runPrimaryImportAction() {
    if (source === "local" && !localPath.trim()) {
      const selected = await chooseFile();
      if (selected) {
        await addModel(selected);
      }
      return;
    }

    if (source === "huggingface" && !selectedHfFile) {
      await inspectRepo();
      return;
    }

    await addModel();
  }

  const canRunPrimary = source === "local"
    ? !isAdding
    : hfRepo.trim().length > 0 && !isAdding && !isInspecting;
  const primaryLabel = source === "local" && !localPath.trim()
    ? "Choose File"
    : source === "huggingface" && !selectedHfFile
      ? isInspecting ? "Finding" : "Find Models"
      : isAdding ? "Adding" : "Add Model";
  const selectedFileParts = localPath.split(/[\\/]/).filter(Boolean);
  const selectedFileName = localPath
    ? selectedFileParts[selectedFileParts.length - 1] ?? localPath
    : null;

  return (
    <section className="library-screen">
      <div className="models-topbar">
        <div className="section-heading">
          <h2>Models</h2>
          <p>Choose what you want to use. Infernet handles the network.</p>
        </div>
        <button className="secondary-button add-model-button" onClick={() => openImporter()}>
          <FilePlus2 size={16} />
          <span>Add Model</span>
        </button>
      </div>

      <div className="model-library">
        {snapshot.availableModels.map((model) => {
          const installed = snapshot.distribution.installedModels.includes(model.modelId);
          const downloading = !installed && isAdding;
          return (
            <button
              className={selectedModel === model.modelId ? "library-card active" : "library-card"}
              key={model.modelId}
              onClick={() => {
                if (installed) {
                  onModelChange(model.modelId);
                } else {
                  openImporter();
                }
              }}
            >
              <div>
                <strong>{model.displayName}</strong>
                <span>{model.activationDtype.toUpperCase()} - {runtimeLabel(model.runtimeKind)}</span>
              </div>
              <div className="library-status">
                <span>{installed ? "Installed" : downloading ? "Preparing" : "Available"}</span>
                <ProgressBar progress={installed ? 100 : downloading ? 56 : 0} />
                <small>{installed ? "Ready to use" : downloading ? "Adding" : "Click to add"}</small>
              </div>
            </button>
          );
        })}
      </div>

      {showImporter ? (
        <div className="modal-backdrop" role="presentation">
          <div className="import-sheet" role="dialog" aria-modal="true" aria-labelledby="add-model-title">
            <div className="import-sheet-header">
              <div>
                <h3 id="add-model-title">Add a model</h3>
                <p>Bring in a model file. Infernet prepares it and starts sharing it.</p>
              </div>
              <button className="text-button" onClick={() => setShowImporter(false)}>Close</button>
            </div>

            <div className="source-choice" aria-label="Choose model source">
              <button className={source === "huggingface" ? "source-card active" : "source-card"} onClick={() => setSource("huggingface")}>
                <Cloud size={20} />
                <strong>Hugging Face</strong>
                <span>Download from a model repo.</span>
              </button>
              <button className={source === "local" ? "source-card active" : "source-card"} onClick={() => setSource("local")}>
                <UploadCloud size={20} />
                <strong>Local file</strong>
                <span>Use a model already on this computer.</span>
              </button>
            </div>

            {source === "huggingface" ? (
              <div className="import-flow">
                <div className="hf-search-row">
                  <label className="field">
                  <span>Repository</span>
                    <input
                      value={hfRepo}
                      onChange={(event) => setHfRepo(event.target.value)}
                      placeholder="bartowski/Llama-3.2-1B-Instruct-GGUF"
                    />
                  </label>
                  <button className="secondary-button" onClick={inspectRepo} disabled={isInspecting || !hfRepo.trim()}>
                    <Search size={16} />
                    <span>{isInspecting ? "Searching" : "Find models"}</span>
                  </button>
                </div>

                <button className="text-button" onClick={() => setShowTokenInput(!showTokenInput)}>
                  {showTokenInput ? "Hide access token" : "Use access token"}
                </button>

                {showTokenInput ? (
                  <label className="field compact-field">
                    <span>Access token</span>
                    <input
                      value={hfToken}
                      onChange={(event) => setHfToken(event.target.value)}
                      placeholder="Only needed for gated or private models"
                      type="password"
                    />
                  </label>
                ) : null}

                {hfFiles.length > 0 ? (
                  <div className="file-list" aria-label="GGUF files">
                    {hfFiles.map((file) => (
                      <button
                        className={selectedHfFile === file.filename ? "file-option active" : "file-option"}
                        onClick={() => setSelectedHfFile(file.filename)}
                        key={file.filename}
                      >
                        <strong>{file.filename}</strong>
                        <span>{file.sizeBytes ? formatBytes(file.sizeBytes) : "GGUF file"}</span>
                      </button>
                    ))}
                  </div>
                ) : (
                  <div className="import-hint">Paste a Hugging Face repo, then find available model files.</div>
                )}
              </div>
            ) : (
              <div className="import-flow">
                <div className={selectedFileName ? "local-file-picker selected" : "local-file-picker"}>
                  <div>
                    <strong>{selectedFileName ?? "Choose a model file"}</strong>
                    <span>{selectedFileName ? localPath : "Select a .gguf file from this computer."}</span>
                  </div>
                  <button className="secondary-button" onClick={chooseFile}>
                    <UploadCloud size={16} />
                    <span>{selectedFileName ? "Change" : "Choose File"}</span>
                  </button>
                </div>
              </div>
            )}

            {(isAdding || importProgress) && !result ? (
              <ImportProgressCard progress={importProgress} />
            ) : null}

            {result ? (
              <div className="import-result">
                <CheckCircle2 size={18} />
                <div>
                  <strong>{result.displayName} added</strong>
                  <span>Infernet is sharing it with the network. Compute support for GGUF layer execution is still in progress.</span>
                </div>
              </div>
            ) : null}

            {error ? (
              <div className="import-result error">
                <Activity size={18} />
                <div>
                  <strong>Could not add model</strong>
                  <span>{friendlyImportError(error)}</span>
                </div>
              </div>
            ) : null}

            <div className="import-actions">
              <button className="send-button" onClick={runPrimaryImportAction} disabled={!canRunPrimary}>
                {source === "huggingface" ? <Cloud size={17} /> : <UploadCloud size={17} />}
                <span>{primaryLabel}</span>
              </button>
              <span>Infernet prepares the model and shares it automatically.</span>
            </div>
          </div>
        </div>
      ) : null}
    </section>
  );
}

function ImportProgressCard({ progress }: { progress: ModelImportProgress | null }) {
  const percent = progress?.totalBytes
    ? Math.min(100, Math.round((progress.downloadedBytes / progress.totalBytes) * 100))
    : null;

  return (
    <div className="import-progress-card">
      <div className="run-spinner" />
      <div>
        <strong>{progress?.stage ?? "Working"}</strong>
        <span>{progress?.detail ?? "Preparing the model"}</span>
      </div>
      <div className="import-progress-meter">
        <ProgressBar progress={percent ?? 18} />
        <small>
          {percent !== null
            ? `${percent}% - ${formatBytes(progress?.downloadedBytes ?? 0)} of ${formatBytes(progress?.totalBytes ?? 0)}`
            : "This can take a while for large models."}
        </small>
      </div>
    </div>
  );
}

function DownloadsPage({ snapshot, developerMode }: { snapshot: GridSnapshot; developerMode: boolean }) {
  const distribution = snapshot.distribution;

  return (
    <section className="downloads-screen">
      <div className="section-heading">
        <h2>Downloads</h2>
        <p>Contribution and storage activity for this node.</p>
      </div>

      <div className="download-metrics">
        <DownloadMetric icon={<HardDrive size={20} />} label="Storage used" value={formatBytes(distribution.storageUsedBytes)} />
        <DownloadMetric icon={<Download size={20} />} label="Downloads" value={String(distribution.currentDownloads)} />
        <DownloadMetric icon={<UploadCloud size={20} />} label="Uploads" value={String(distribution.currentUploads)} />
        <DownloadMetric icon={<CheckCircle2 size={20} />} label="Contribution" value="Enabled" />
      </div>

      <div className="download-list">
        {distribution.installedShards.length === 0 ? (
          <div className="empty-state">No local model shards installed.</div>
        ) : (
          distribution.installedShards.map((shard) => (
            <div className="download-row" key={`${shard.modelId}-${shard.layerStart}-${shard.layerEnd}-${shard.checksum}`}>
              <div>
                <strong>{shard.modelId}</strong>
                <span>{formatBytes(shard.sizeBytes)} - {shard.version}</span>
              </div>
              <span>Hosted</span>
              {developerMode ? <small>{shard.layerStart}:{shard.layerEnd}</small> : null}
            </div>
          ))
        )}
      </div>
    </section>
  );
}

function DownloadMetric({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="download-metric">
      {icon}
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function SettingsPage({
  identity,
  developerMode,
  setDeveloperMode,
}: {
  identity: LocalIdentity | null;
  developerMode: boolean;
  setDeveloperMode: (enabled: boolean) => void;
}) {
  const [hfSettings, setHfSettings] = useState<HuggingFaceSettings>({ hasToken: false });
  const [token, setToken] = useState("");
  const [tokenStatus, setTokenStatus] = useState<string | null>(null);

  useEffect(() => {
    getHuggingFaceSettings()
      .then(setHfSettings)
      .catch((error) => setTokenStatus(String(error)));
  }, []);

  async function saveToken() {
    try {
      const next = await saveHuggingFaceToken(token);
      setHfSettings(next);
      setToken("");
      setTokenStatus("Saved for this app session.");
    } catch (error) {
      setTokenStatus(String(error));
    }
  }

  async function clearToken() {
    try {
      const next = await clearHuggingFaceToken();
      setHfSettings(next);
      setTokenStatus("Token cleared.");
    } catch (error) {
      setTokenStatus(String(error));
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
        <div className="settings-row huggingface-settings">
          <div>
            <strong>Hugging Face</strong>
            <span>
              {hfSettings.hasToken
                ? `Token active: ${hfSettings.tokenPreview ?? "saved"}`
                : "Optional for gated or private model downloads."}
            </span>
          </div>
          <div className="token-controls">
            <KeyRound size={17} />
            <input
              value={token}
              type="password"
              onChange={(event) => setToken(event.target.value)}
              placeholder="hf_..."
            />
            <button className="secondary-button" onClick={saveToken} disabled={!token.trim()}>
              Save
            </button>
            <button className="text-button" onClick={clearToken} disabled={!hfSettings.hasToken}>
              Clear
            </button>
          </div>
        </div>
        {tokenStatus ? <div className="settings-note">{tokenStatus}</div> : null}
      </div>
    </section>
  );
}

function ProgressBar({ progress }: { progress: number }) {
  return (
    <div className="progress-bar">
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

function uniquePeerCount(route: RouteHopView[]): number {
  return new Set(route.map((hop) => hop.peerId)).size;
}

function phaseLabel(completedHops: number, totalHops: number): string {
  if (totalHops === 0) {
    return "Finding compute";
  }
  if (completedHops === 0) {
    return "Peer discovered";
  }
  if (completedHops < totalHops) {
    return "Forwarding";
  }
  return "Receiving response";
}

function hopKey(peerId: string, layerStart: number, layerEnd: number): string {
  return `${peerId}:${layerStart}:${layerEnd}`;
}

function runtimeLabel(runtimeKind: string): string {
  return runtimeKind === "llama_cpp" ? "GGUF" : "Ready";
}

function isRuntimePendingMessage(message: string): boolean {
  return message.toLowerCase().includes("gguf execution runtime is not connected yet");
}

function friendlyImportError(error: string): string {
  if (error.includes("must be a .gguf file")) {
    return "Choose a GGUF model file.";
  }
  if (error.includes("Hugging Face returned 401") || error.includes("Hugging Face returned 403")) {
    return "That model needs access. Add a Hugging Face token and try again.";
  }
  if (error.includes("No GGUF files")) {
    return "No GGUF files were found in that repository.";
  }

  return error;
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
