import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { sampleIdentity, sampleSnapshot } from "./sampleData";
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

export const isTauriRuntime = "__TAURI_INTERNALS__" in window;

export async function getLocalIdentity(): Promise<LocalIdentity> {
  if (!isTauriRuntime) {
    return sampleIdentity;
  }

  return invoke<LocalIdentity>("get_local_identity");
}

export async function getGridSnapshot(
  discoveryTimeoutMs = 4000,
  modelId = sampleSnapshot.selectedModel,
): Promise<GridSnapshot> {
  if (!isTauriRuntime) {
    await delay(220);
    return modelSnapshot(modelId);
  }

  return invoke<GridSnapshot>("get_grid_snapshot", { discoveryTimeoutMs, modelId });
}

export async function runDistributedInference(prompt: string, modelId: string): Promise<RunDemoResponse> {
  return invoke<RunDemoResponse>("run_demo_inference", { prompt, modelId });
}

export async function addLocalGgufModel(
  path: string,
  version = "v1",
): Promise<AddModelResponse> {
  if (!isTauriRuntime) {
    await delay(620);
    return mockAddModelResponse(path);
  }

  return invoke<AddModelResponse>("add_local_gguf_model", { path, version });
}

export async function chooseLocalModelFile(): Promise<string | null> {
  if (!isTauriRuntime) {
    await delay(160);
    return "/Users/christopher/Models/gemma-2b-it-Q4_K_M.gguf";
  }

  const selected = await open({
    multiple: false,
    directory: false,
    title: "Choose a GGUF model",
    filters: [{ name: "GGUF models", extensions: ["gguf"] }],
  });

  return typeof selected === "string" ? selected : null;
}

export async function getHuggingFaceSettings(): Promise<HuggingFaceSettings> {
  if (!isTauriRuntime) {
    return { hasToken: false, tokenPreview: null };
  }

  return invoke<HuggingFaceSettings>("get_huggingface_settings");
}

export async function saveHuggingFaceToken(token: string): Promise<HuggingFaceSettings> {
  if (!isTauriRuntime) {
    await delay(180);
    return { hasToken: token.trim().length > 0, tokenPreview: token.trim() ? "hf_...mock" : null };
  }

  return invoke<HuggingFaceSettings>("save_huggingface_token", { token });
}

export async function clearHuggingFaceToken(): Promise<HuggingFaceSettings> {
  if (!isTauriRuntime) {
    await delay(120);
    return { hasToken: false, tokenPreview: null };
  }

  return invoke<HuggingFaceSettings>("clear_huggingface_token");
}

export async function inspectHuggingFaceRepo(
  repoId: string,
  token?: string,
): Promise<HuggingFaceFileView[]> {
  if (!isTauriRuntime) {
    await delay(420);
    return [
      { filename: "llama-3.2-1b-instruct-q4_k_m.gguf", sizeBytes: 812_000_000 },
      { filename: "llama-3.2-1b-instruct-q5_k_m.gguf", sizeBytes: 982_000_000 },
    ];
  }

  return invoke<HuggingFaceFileView[]>("inspect_huggingface_repo", { repoId, token });
}

export async function addHuggingFaceModel(
  repoId: string,
  filename: string,
  token?: string,
  revision = "main",
  version = "v1",
): Promise<AddModelResponse> {
  if (!isTauriRuntime) {
    await delay(900);
    return mockAddModelResponse(`hf://${repoId}/${filename}`);
  }

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
  if (!isTauriRuntime) {
    return () => undefined;
  }

  return listen<ProgressEvent>("infernet-progress", (event) => {
    handler(event.payload);
  });
}

export async function listenForModelImportProgress(
  handler: (event: ModelImportProgress) => void,
): Promise<() => void> {
  if (!isTauriRuntime) {
    return () => undefined;
  }

  return listen<ModelImportProgress>("infernet-model-import-progress", (event) => {
    handler(event.payload);
  });
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function modelSnapshot(modelId: string): GridSnapshot {
  if (modelId === sampleSnapshot.selectedModel) {
    return sampleSnapshot;
  }

  const model = sampleSnapshot.availableModels.find((item) => item.modelId === modelId);
  return {
    ...sampleSnapshot,
    selectedModel: modelId,
    layerCount: model?.layerCount ?? sampleSnapshot.layerCount,
    route: [],
    peers: [],
    coverage: Array.from({ length: model?.layerCount ?? 0 }, (_, layer) => ({
      layer,
      covered: false,
    })),
    distribution: {
      ...sampleSnapshot.distribution,
      installedModels: [],
      installedShards: [],
      storageUsedBytes: 0,
      currentUploads: 0,
      currentDownloads: 0,
      replicationHealth: [],
    },
    missingRanges: model ? `no complete route for model ${model.modelId}` : `unknown model ${modelId}`,
  };
}

function mockAddModelResponse(source: string): AddModelResponse {
  const modelId = modelIdFromSource(source);
  const layerCount = source.toLowerCase().includes("gemma") ? 48 : 16;
  const shardSize = layerCount <= 16 ? 4 : 8;
  const installedShards = Array.from(
    { length: Math.ceil(layerCount / shardSize) },
    (_, index) => {
      const layerStart = index * shardSize;
      const layerEnd = Math.min(layerCount, layerStart + shardSize);
      return {
        modelId,
        layerStart,
        layerEnd,
        checksum: `mock-${index}-${layerStart}-${layerEnd}`,
        sizeBytes: 1_120 + index * 41,
        version: "v1",
      };
    },
  );

  return {
    modelId,
    displayName: displayNameFromSource(source) ?? "Imported GGUF Model",
    source,
    sourceChecksum: "mock-source-checksum",
    sourceSizeBytes: 812_000_000,
    plannedShards: installedShards.length,
    metadataOnly: true,
    installedShards,
    message: "Model seed records are being shared. Physical GGUF tensor shards still require the llama.cpp shard writer.",
  };
}

function modelIdFromSource(source: string): string {
  const displayName = displayNameFromSource(source) ?? "gguf-model";
  const modelId = displayName
    .toLowerCase()
    .replace(/[^a-z0-9.]+/g, "-")
    .replace(/^-+|-+$/g, "");

  return modelId.length > 0 ? modelId : "gguf-model";
}

function displayNameFromSource(source: string): string | null {
  const pathParts = source.split("/").filter(Boolean);
  const fileName = pathParts[pathParts.length - 1];
  if (!fileName) {
    return null;
  }

  const withoutExtension = fileName.replace(/\.gguf$/i, "");
  const displayName = withoutExtension
    .replace(/[_\-.]+/g, " ")
    .trim()
    .split(/\s+/)
    .map(formatModelNamePart)
    .join(" ");

  return displayName.length > 0 ? displayName : null;
}

function formatModelNamePart(part: string): string {
  const lower = part.toLowerCase();
  if (["gguf", "q4", "q5", "q6", "q8", "k", "m", "s", "it"].includes(lower)) {
    return lower.toUpperCase();
  }
  if (lower === "llama") return "Llama";
  if (lower === "gemma") return "Gemma";
  if (lower === "qwen") return "Qwen";
  if (lower === "mistral") return "Mistral";
  if (lower === "instruct") return "Instruct";
  if (/^\d+b$/.test(lower)) return lower.toUpperCase();
  return part;
}
