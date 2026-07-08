import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import type {
  AddModelResponse,
  GridSnapshot,
  HuggingFaceFileView,
  HuggingFaceSettings,
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

export async function addLocalGgufModel(
  path: string,
  version = "v1",
): Promise<AddModelResponse> {
  return invoke<AddModelResponse>("add_local_gguf_model", { path, version });
}

export async function chooseLocalModelFile(): Promise<string | null> {
  const selected = await open({
    multiple: false,
    directory: false,
    title: "Choose a GGUF model",
    filters: [{ name: "GGUF models", extensions: ["gguf"] }],
  });

  return typeof selected === "string" ? selected : null;
}

export async function getHuggingFaceSettings(): Promise<HuggingFaceSettings> {
  return invoke<HuggingFaceSettings>("get_huggingface_settings");
}

export async function saveHuggingFaceToken(token: string): Promise<HuggingFaceSettings> {
  return invoke<HuggingFaceSettings>("save_huggingface_token", { token });
}

export async function clearHuggingFaceToken(): Promise<HuggingFaceSettings> {
  return invoke<HuggingFaceSettings>("clear_huggingface_token");
}

export async function inspectHuggingFaceRepo(
  repoId: string,
  token?: string,
): Promise<HuggingFaceFileView[]> {
  return invoke<HuggingFaceFileView[]>("inspect_huggingface_repo", { repoId, token });
}

export async function addHuggingFaceModel(
  repoId: string,
  filename: string,
  token?: string,
  revision = "main",
  version = "v1",
): Promise<AddModelResponse> {
  return invoke<AddModelResponse>("add_huggingface_model", {
    repoId,
    filename,
    token,
    revision,
    version,
  });
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
