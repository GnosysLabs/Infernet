import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import type {
  GridSnapshot,
  LocalIdentity,
  ModelImportProgress,
  ProgressEvent,
  RunDemoResponse,
} from "./types";

export const emptySnapshot: GridSnapshot = {
  localPeerId: "",
  topic: "infernet/grid-demo/1",
  selectedModel: "",
  availableModels: [],
  layerCount: 0,
  networkPeerCount: 0,
  peers: [],
  machines: [],
  route: [],
  missingRanges: null,
  coverage: [],
  distribution: {
    installedModels: [],
    installedShards: [],
    storageUsedBytes: 0,
    maxStorageBytes: 0,
    currentUploads: 0,
    currentDownloads: 0,
    bytesServed: 0,
    chunksServed: 0,
    lastServedUnixMs: null,
    replicationHealth: [],
  },
};

export async function getLocalIdentity(): Promise<LocalIdentity> {
  return invoke<LocalIdentity>("get_local_identity");
}

export async function getManualPeers(): Promise<string[]> {
  return invoke<string[]>("get_manual_peers");
}

export async function addManualPeer(address: string): Promise<string[]> {
  return invoke<string[]>("add_manual_peer", { address });
}

export async function clearManualPeers(): Promise<string[]> {
  return invoke<string[]>("clear_manual_peers");
}

export async function getGridSnapshot(
  discoveryTimeoutMs = 4000,
  modelId?: string,
): Promise<GridSnapshot> {
  return invoke<GridSnapshot>("get_grid_snapshot", {
    discoveryTimeoutMs,
    modelId: modelId?.trim() ? modelId : null,
  });
}

export async function runDistributedInference(prompt: string, modelId: string): Promise<RunDemoResponse> {
  return invoke<RunDemoResponse>("run_demo_inference", { prompt, modelId });
}

export async function installOfficialModel(modelId: string): Promise<GridSnapshot> {
  return invoke<GridSnapshot>("install_official_model", { modelId });
}

export async function listenForProgress(
  handler: (event: ProgressEvent) => void,
): Promise<() => void> {
  try {
    return await listen<ProgressEvent>("infernet-progress", (event) => {
      handler(event.payload);
    });
  } catch {
    return () => undefined;
  }
}

export async function listenForModelImportProgress(
  handler: (event: ModelImportProgress) => void,
): Promise<() => void> {
  try {
    return await listen<ModelImportProgress>("infernet-model-import-progress", (event) => {
      handler(event.payload);
    });
  } catch {
    return () => undefined;
  }
}
