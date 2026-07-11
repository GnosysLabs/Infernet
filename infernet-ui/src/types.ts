export interface LocalIdentity {
  peerId: string;
  topic: string;
  listen: string;
  connectAddresses: string[];
}

export interface VramContributionSettings {
  contributionBytes: number;
  totalBytes: number;
  availableBytes: number;
  computeBackend: string;
  deviceName: string;
  unifiedMemory: boolean;
}

export interface ModelView {
  modelId: string;
  displayName: string;
  runtimeKind: string;
  layerCount: number;
  activationDtype: string;
  quantization?: string | null;
  installed: boolean;
  runnable: boolean;
  status: string;
}

export interface ShardView {
  modelId: string;
  layerStart: number;
  layerEnd: number;
}

export interface PeerView {
  peerId: string;
  shortPeerId: string;
  addresses: string[];
  protocolVersion: number;
  shards: ShardView[];
}

export interface MachineView {
  peerId: string;
  shortPeerId: string;
  addresses: string[];
  machineId?: string | null;
  isLocal: boolean;
  connectionStatus: "connected" | "reconnecting" | "unreachable";
  lastSeenSeconds: number;
  computeBackend: string;
  deviceName: string;
  logicalCpuCores: number;
  totalMemoryBytes: number;
  availableMemoryBytes: number;
  allocatedMemoryBytes: number;
  unifiedMemory: boolean;
  maxSessions: number;
  activeSessions: number;
  queueDepth: number;
  measuredPrefillTokensPerSecond?: number | null;
  measuredDecodeTokensPerSecond?: number | null;
  hostedComponentCount: number;
  coarseLocation?: {
    latitude: number;
    longitude: number;
    label: string;
  } | null;
  rpcReady: boolean;
}

export interface ExecutionParticipantView {
  peerId: string;
  shortPeerId: string;
  role: "coordinator" | "worker";
  computeBackend: string;
  deviceName: string;
  availableMemoryBytes: number;
  estimatedSharePercent: number;
}

export interface RouteHopView {
  peerId: string;
  shortPeerId: string;
  address: string;
  layerStart: number;
  layerEnd: number;
}

export interface CoverageSegment {
  layer: number;
  covered: boolean;
  peerId?: string;
  layerStart?: number;
  layerEnd?: number;
}

export interface InstalledShardView {
  modelId: string;
  layerStart: number;
  layerEnd: number;
  checksum: string;
  sizeBytes: number;
  version: string;
}

export interface ReplicationHealthView {
  modelId: string;
  layerStart: number;
  layerEnd: number;
  replicas: number;
  targetReplicas: number;
}

export interface DistributionSnapshot {
  installedModels: string[];
  installedShards: InstalledShardView[];
  storageUsedBytes: number;
  maxStorageBytes: number;
  currentUploads: number;
  currentDownloads: number;
  bytesServed: number;
  chunksServed: number;
  lastServedUnixMs?: number | null;
  replicationHealth: ReplicationHealthView[];
}

export interface GridSnapshot {
  localPeerId: string;
  topic: string;
  selectedModel: string;
  availableModels: ModelView[];
  layerCount: number;
  networkPeerCount: number;
  peers: PeerView[];
  machines: MachineView[];
  route: RouteHopView[];
  missingRanges?: string | null;
  coverage: CoverageSegment[];
  distribution: DistributionSnapshot;
}

export interface RunDemoResponse {
  output: string;
  traceId: string;
  snapshot: GridSnapshot;
}

export interface ImageRuntimeStatus {
  modelId: string;
  releaseId: string;
  releaseVersion: string;
  runtimeAbi: string;
  quantization: string;
  runtimeAvailable: boolean;
  busy: boolean;
  installed: boolean;
  verified: boolean;
  downloadedBytes: number;
  totalBytes: number;
  status: string;
}

export interface GenerateImageResponse {
  imageDataUrl: string;
  imageId: string;
  prompt: string;
  seed: number;
  width: number;
  height: number;
  steps: number;
  durationMs: number;
  releaseId: string;
  placement: string;
  detailsAvailable: boolean;
}

export interface ModelImportProgress {
  modelId: string;
  stage: string;
  detail: string;
  downloadedBytes: number;
  totalBytes?: number | null;
}

export type LocalNodeActivityKind = "chatCompletion" | "imageGeneration" | "computeContribution";
export type LocalNodeActivityOutcome = "success" | "error";

export interface LocalNodeActivityTask {
  id: string;
  traceId: string;
  kind: LocalNodeActivityKind;
  startedAtUnixMs: number;
}

export interface LocalNodeActivityEntry extends LocalNodeActivityTask {
  outcome: LocalNodeActivityOutcome;
  completedAtUnixMs: number;
}

export interface LocalNodeActivitySnapshot {
  computeActive: boolean;
  computeReady: boolean;
  computeBackend: string;
  deviceName: string;
  totalMemoryBytes: number;
  availableMemoryBytes: number;
  sharingActive: boolean;
  bytesServed: number;
  chunksServed: number;
  lastServedUnixMs?: number | null;
  current: LocalNodeActivityTask[];
  journal: LocalNodeActivityEntry[];
}

export type ProgressEvent =
  | { type: "routeDiscovered"; route: RouteHopView[] }
  | { type: "executionPlan"; participants: ExecutionParticipantView[] }
  | {
      type: "hopStarted";
      traceId: string;
      peerId: string;
      shortPeerId: string;
      layerStart: number;
      layerEnd: number;
      activationSizeBytes: number;
    }
  | {
      type: "hopCompleted";
      traceId: string;
      peerId: string;
      shortPeerId: string;
      layerStart: number;
      layerEnd: number;
      nextPeerId?: string | null;
      activationSizeBytes: number;
      timingMs: number;
      activationChecksum: string;
    }
  | { type: "finalOutput"; traceId: string; output: string }
  | { type: "error"; message: string };

export interface HopProgress {
  key: string;
  peerId: string;
  shortPeerId: string;
  layerStart: number;
  layerEnd: number;
  activationSizeBytes: number;
  timingMs?: number;
  activationChecksum?: string;
  status: "pending" | "running" | "complete" | "error";
}
