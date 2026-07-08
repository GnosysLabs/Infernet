export interface LocalIdentity {
  peerId: string;
  topic: string;
  listen: string;
}

export interface ModelView {
  modelId: string;
  displayName: string;
  runtimeKind: string;
  layerCount: number;
  activationDtype: string;
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
  replicationHealth: ReplicationHealthView[];
}

export interface GridSnapshot {
  localPeerId: string;
  topic: string;
  selectedModel: string;
  availableModels: ModelView[];
  layerCount: number;
  peers: PeerView[];
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

export interface AddModelResponse {
  modelId: string;
  displayName: string;
  source: string;
  sourceChecksum: string;
  sourceSizeBytes: number;
  plannedShards: number;
  metadataOnly: boolean;
  installedShards: InstalledShardView[];
  message: string;
}

export interface HuggingFaceSettings {
  hasToken: boolean;
  tokenPreview?: string | null;
}

export interface HuggingFaceFileView {
  filename: string;
  sizeBytes?: number | null;
}

export interface ModelImportProgress {
  modelId: string;
  stage: string;
  detail: string;
  downloadedBytes: number;
  totalBytes?: number | null;
}

export type ProgressEvent =
  | { type: "routeDiscovered"; route: RouteHopView[] }
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
