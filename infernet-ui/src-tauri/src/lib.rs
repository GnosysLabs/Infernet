mod app_settings;
mod chat_history;
mod execution_plan;
mod image_runtime;
mod peer_presence;

#[cfg(test)]
use std::path::Path;
use std::time::Duration;
use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    net::{IpAddr, Ipv4Addr, TcpListener, UdpSocket},
    path::PathBuf,
    sync::{Arc, Mutex},
};

use app_settings::{AppSettingsStore, get_vram_contribution_settings, set_vram_contribution};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64_STANDARD};
use chat_history::{
    ChatHistoryStore, append_chat_message, create_chat_thread, delete_chat_thread,
    get_chat_history, select_chat_thread,
};
use execution_plan::{ExecutionParticipantView, plan_worker_execution, worker_is_usable};
use futures::{StreamExt, channel::oneshot};
use infernet_model::{
    INFERNET_CHAT_KV_CACHE_BYTES_PER_LAYER, LayerRange, ModelManifest, OfficialComponentKind,
    OfficialModelRelease, RuntimeKind, SeedShardManifest, ShardDescriptor,
};
use infernet_node::stable_diffusion_runtime::{
    IMAGE_RPC_DEFAULT_PORT, INFERNET_IMAGE_RPC_RUNTIME_ABI, ImageGenerationRequest,
    StableDiffusionConfig, StableDiffusionPlacement, distributed_diffusion_backend,
    distributed_diffusion_max_vram, find_image_rpc_server_binary, find_sd_cli_binary,
    generate_with_sd_cli,
};
use infernet_node::{
    DiscoveryConfig, INFERNET_LLAMA_RPC_RUNTIME_ABI, LLAMA_RPC_DEFAULT_PORT,
    LLAMA_RPC_PROTOCOL_VERSION, LlamaRpcServer, LlamaRpcServerConfig, LocalNodeActivityEntry,
    LocalNodeActivityKind, LocalNodeActivityOutcome, LocalNodeActivityTask,
    PAYLOAD_KIND_FULL_MODEL, PAYLOAD_KIND_GGUF_SHARD, PAYLOAD_KIND_INFERNET_SHARD, ShardCache,
    ShardCacheConfig, begin_local_node_activity, clear_local_image_rpc_endpoint,
    clear_local_llama_rpc_endpoint, detect_node_capabilities, empty_advertisement,
    enrich_local_advertisement, fetch_coarse_location_assertion,
    fetch_model_shard_over_libp2p_with_progress, find_llama_rpc_server_binary,
    generate_image_over_libp2p, import_seed_model_from_file_consuming_verified, infer_over_libp2p,
    is_executable_shard_record, load_or_generate_keypair, local_capability_advertisement,
    local_node_activity_snapshot, model_serving_telemetry, persistent_infernet_worker_is_resident,
    run_model_distribution_node_with_readiness_and_registry, seed_manifest_for_network,
    set_local_coarse_location, set_local_image_rpc_endpoint, set_local_inference_active,
    set_local_llama_rpc_endpoint, set_local_model_components, set_local_rpc_active, sha256_file,
    spawn_llama_rpc_server, stop_persistent_llama_server, stop_persistent_rpc_tunnels,
    verify_coarse_location_assertion, vram_contribution_limit_bytes,
};
use infernet_protocol::{
    IMAGE_RPC_TUNNEL_PROTOCOL, LLAMA_RPC_TUNNEL_PROTOCOL, LlamaRpcEndpoint, ModelShardInfo,
    NodeAdvertisement, PROTOCOL_VERSION, RouteHop, TraceEvent,
};
use infernet_router::{
    CapacityPlanningConfig, FixedModelComponent, ShardRegistry, plan_fixed_components,
};
use libp2p::{Multiaddr, PeerId, identity, multiaddr::Protocol};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, State};
use tokio::io::AsyncWriteExt;
use tokio::time::{Instant, sleep};

use peer_presence::{ConnectionStatus, PeerPresence, PresenceRecord, PresenceSnapshot};

const DEFAULT_TOPIC: &str = "infernet/grid-demo/1";
const DEFAULT_DISCOVERY_TIMEOUT_MS: u64 = 4_000;
const DEFAULT_INFERENCE_TIMEOUT_MS: u64 = 30_000;
const DEFAULT_MODEL_FETCH_TIMEOUT_MS: u64 = 60 * 60 * 1_000;
const UI_LISTEN_PORT: u16 = 9777;
const OFFICIAL_CHAT_MODEL_ID: &str = "infernet-chat-v1";
const RUNTIME_SCRATCH_BYTES_PER_PEER: u64 = 768 * 1024 * 1024;
const CAPACITY_SAFETY_BYTES: u64 = 1024 * 1024 * 1024;
// The transport delivers 4 MiB chunks. Forward every completed chunk so the
// desktop progress bar reflects the live transfer instead of appearing frozen
// until another 64 MiB has accumulated.
const MODEL_PROGRESS_EMIT_BYTES: u64 = 4 * 1024 * 1024;
const DEFAULT_BOOTSTRAP_PEERS: &[&str] = &[
    "12D3KooWRJrnpHPQTWdThpDGZMwRCHhEBL4JCAxFMwYMfFavxa2h@/ip4/217.77.11.197/tcp/9777/p2p/12D3KooWRJrnpHPQTWdThpDGZMwRCHhEBL4JCAxFMwYMfFavxa2h",
    "12D3KooWRJrnpHPQTWdThpDGZMwRCHhEBL4JCAxFMwYMfFavxa2h@/dns4/infernet.gnosyslabs.xyz/tcp/9777/p2p/12D3KooWRJrnpHPQTWdThpDGZMwRCHhEBL4JCAxFMwYMfFavxa2h",
];
const DEFAULT_RELAY_PEER_ID: &str = "12D3KooWRJrnpHPQTWdThpDGZMwRCHhEBL4JCAxFMwYMfFavxa2h";
const COARSE_LOCATION_ENDPOINT: &str =
    "https://infernet.gnosyslabs.xyz/.well-known/infernet/coarse-location";

enum ModelDistributionServiceState {
    Stopped,
    Starting(Vec<oneshot::Sender<Result<(), String>>>),
    Running,
}

enum LlamaRpcServiceState {
    Stopped,
    Starting(Vec<oneshot::Sender<Result<(), String>>>),
    Running(AdvertisedLlamaRpcServer),
}

enum ImageRpcServiceState {
    Stopped,
    Starting(Vec<oneshot::Sender<Result<(), String>>>),
    Running(AdvertisedImageRpcServer),
}

struct AdvertisedLlamaRpcServer {
    server: LlamaRpcServer,
}

impl Drop for AdvertisedLlamaRpcServer {
    fn drop(&mut self) {
        clear_local_llama_rpc_endpoint();
    }
}

struct AdvertisedImageRpcServer {
    server: LlamaRpcServer,
}

impl Drop for AdvertisedImageRpcServer {
    fn drop(&mut self) {
        clear_local_image_rpc_endpoint();
    }
}

struct UiState {
    keypair: Mutex<identity::Keypair>,
    app_settings: Mutex<Option<AppSettingsStore>>,
    chat_history: Mutex<Option<ChatHistoryStore>>,
    chat_history_error: Mutex<Option<String>>,
    topic: String,
    model_distribution_service: Arc<Mutex<ModelDistributionServiceState>>,
    live_registry: Arc<Mutex<ShardRegistry>>,
    llama_rpc_service: Arc<Mutex<LlamaRpcServiceState>>,
    image_rpc_service: Arc<Mutex<ImageRpcServiceState>>,
    active_model_acquisitions: Arc<Mutex<BTreeSet<String>>>,
    manual_peers: Mutex<Vec<NodeAdvertisement>>,
    peer_presence: Mutex<PeerPresence>,
    execution_plan: Mutex<Option<execution_plan::WorkerExecutionPlan>>,
    image_operation: Arc<Mutex<bool>>,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            keypair: Mutex::new(identity::Keypair::generate_ed25519()),
            app_settings: Mutex::new(None),
            chat_history: Mutex::new(None),
            chat_history_error: Mutex::new(None),
            topic: DEFAULT_TOPIC.to_owned(),
            model_distribution_service: Arc::new(Mutex::new(
                ModelDistributionServiceState::Stopped,
            )),
            live_registry: Arc::new(Mutex::new(ShardRegistry::new())),
            llama_rpc_service: Arc::new(Mutex::new(LlamaRpcServiceState::Stopped)),
            image_rpc_service: Arc::new(Mutex::new(ImageRpcServiceState::Stopped)),
            active_model_acquisitions: Arc::new(Mutex::new(BTreeSet::new())),
            manual_peers: Mutex::new(Vec::new()),
            peer_presence: Mutex::new(PeerPresence::default()),
            execution_plan: Mutex::new(None),
            image_operation: Arc::new(Mutex::new(false)),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct LocalIdentity {
    peer_id: String,
    topic: String,
    listen: String,
    connect_addresses: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct LocalNodeActivityView {
    compute_active: bool,
    compute_ready: bool,
    compute_backend: String,
    device_name: String,
    total_memory_bytes: u64,
    available_memory_bytes: u64,
    sharing_active: bool,
    bytes_served: u64,
    chunks_served: u64,
    last_served_unix_ms: Option<u64>,
    current: Vec<LocalNodeActivityTask>,
    journal: Vec<LocalNodeActivityEntry>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct GridSnapshot {
    local_peer_id: String,
    topic: String,
    selected_model: String,
    available_models: Vec<ModelView>,
    layer_count: u32,
    network_peer_count: usize,
    peers: Vec<PeerView>,
    machines: Vec<MachineView>,
    route: Vec<RouteHopView>,
    missing_ranges: Option<String>,
    coverage: Vec<CoverageSegment>,
    distribution: DistributionSnapshot,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelView {
    model_id: String,
    display_name: String,
    runtime_kind: String,
    layer_count: u32,
    activation_dtype: String,
    quantization: Option<String>,
    installed: bool,
    runnable: bool,
    status: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct PeerView {
    peer_id: String,
    short_peer_id: String,
    addresses: Vec<String>,
    protocol_version: u32,
    shards: Vec<ShardView>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct MachineView {
    peer_id: String,
    short_peer_id: String,
    addresses: Vec<String>,
    machine_id: Option<String>,
    is_local: bool,
    connection_status: ConnectionStatus,
    last_seen_seconds: u64,
    compute_backend: String,
    device_name: String,
    logical_cpu_cores: u32,
    total_memory_bytes: u64,
    available_memory_bytes: u64,
    allocated_memory_bytes: u64,
    unified_memory: bool,
    max_sessions: u32,
    active_sessions: u32,
    queue_depth: u32,
    measured_prefill_tokens_per_second: Option<f32>,
    measured_decode_tokens_per_second: Option<f32>,
    hosted_component_count: usize,
    rpc_ready: bool,
    coarse_location: Option<CoarseLocationView>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct CoarseLocationView {
    latitude: f64,
    longitude: f64,
    label: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ShardView {
    model_id: String,
    layer_start: u32,
    layer_end: u32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct RouteHopView {
    peer_id: String,
    short_peer_id: String,
    address: String,
    layer_start: u32,
    layer_end: u32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct CoverageSegment {
    layer: u32,
    covered: bool,
    peer_id: Option<String>,
    layer_start: Option<u32>,
    layer_end: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct InstalledShardView {
    model_id: String,
    layer_start: u32,
    layer_end: u32,
    checksum: String,
    size_bytes: u64,
    version: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ReplicationHealthView {
    model_id: String,
    layer_start: u32,
    layer_end: u32,
    replicas: usize,
    target_replicas: usize,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct DistributionSnapshot {
    installed_models: Vec<String>,
    installed_shards: Vec<InstalledShardView>,
    storage_used_bytes: u64,
    max_storage_bytes: u64,
    current_uploads: usize,
    current_downloads: usize,
    bytes_served: u64,
    chunks_served: u64,
    last_served_unix_ms: Option<u64>,
    replication_health: Vec<ReplicationHealthView>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct RunDemoResponse {
    output: String,
    trace_id: String,
    snapshot: GridSnapshot,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct GenerateImageResponse {
    image_data_url: String,
    image_id: String,
    prompt: String,
    seed: i64,
    width: u32,
    height: u32,
    steps: u32,
    duration_ms: u64,
    release_id: String,
    placement: String,
    details_available: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeneratedImageMetadata {
    version: u32,
    image_id: String,
    prompt: String,
    seed: i64,
    width: u32,
    height: u32,
    steps: u32,
    duration_ms: u64,
    release_id: String,
    placement: String,
}

struct DiscoveredImagePlacement {
    placement: StableDiffusionPlacement,
    remote_worker_peer_ids: Vec<String>,
    advertisements: Vec<NodeAdvertisement>,
}

struct ImageOperationGuard {
    busy: Arc<Mutex<bool>>,
}

impl Drop for ImageOperationGuard {
    fn drop(&mut self) {
        if let Ok(mut busy) = self.busy.lock() {
            *busy = false;
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelImportProgress {
    model_id: String,
    stage: String,
    detail: String,
    downloaded_bytes: u64,
    total_bytes: Option<u64>,
}

#[derive(Debug, Clone)]
struct AdvertisedModelRecord {
    info: ModelShardInfo,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "camelCase")]
enum ProgressEvent {
    RouteDiscovered {
        route: Vec<RouteHopView>,
    },
    ExecutionPlan {
        participants: Vec<ExecutionParticipantView>,
    },
    HopStarted {
        trace_id: String,
        peer_id: String,
        short_peer_id: String,
        layer_start: u32,
        layer_end: u32,
        activation_size_bytes: usize,
    },
    HopCompleted {
        trace_id: String,
        peer_id: String,
        short_peer_id: String,
        layer_start: u32,
        layer_end: u32,
        next_peer_id: Option<String>,
        activation_size_bytes: usize,
        timing_ms: u64,
        activation_checksum: String,
    },
    FinalOutput {
        trace_id: String,
        output: String,
    },
    Error {
        message: String,
    },
}

#[tauri::command]
async fn get_local_identity(state: State<'_, UiState>) -> Result<LocalIdentity, String> {
    let (peer_id, topic) = identity_from_state(&state)?;
    let connect_addresses = local_connect_addresses(&peer_id);

    Ok(LocalIdentity {
        peer_id: peer_id.clone(),
        topic,
        listen: format!("/ip4/0.0.0.0/tcp/{UI_LISTEN_PORT}/p2p/{peer_id}"),
        connect_addresses,
    })
}

#[tauri::command]
fn get_local_node_activity() -> LocalNodeActivityView {
    let capabilities = detect_node_capabilities();
    let use_accelerator_memory =
        capabilities.compute_backend != "cpu" && capabilities.total_accelerator_memory_bytes > 0;
    let (total_memory_bytes, available_memory_bytes) = if use_accelerator_memory {
        (
            capabilities.total_accelerator_memory_bytes,
            capabilities.available_accelerator_memory_bytes,
        )
    } else {
        (
            capabilities.total_ram_bytes,
            capabilities.available_ram_bytes,
        )
    };
    let compute_ready = capabilities
        .llama_rpc
        .as_ref()
        .is_some_and(|endpoint| endpoint.ready)
        || capabilities
            .image_rpc
            .as_ref()
            .is_some_and(|endpoint| endpoint.ready);
    let serving = model_serving_telemetry();
    let sharing_active = serving
        .last_activity_unix_ms
        .is_some_and(|last| current_unix_ms_u64().saturating_sub(last) <= 5_000);
    let activity = local_node_activity_snapshot();

    LocalNodeActivityView {
        compute_active: capabilities.active_sessions > 0,
        compute_ready,
        compute_backend: capabilities.compute_backend,
        device_name: capabilities.device_name,
        total_memory_bytes,
        available_memory_bytes,
        sharing_active,
        bytes_served: serving.bytes_served,
        chunks_served: serving.chunks_served,
        last_served_unix_ms: serving.last_activity_unix_ms,
        current: activity.current,
        journal: activity.journal,
    }
}

#[tauri::command]
fn get_manual_peers(state: State<'_, UiState>) -> Result<Vec<String>, String> {
    manual_peer_addresses(&state)
}

#[tauri::command]
fn add_manual_peer(state: State<'_, UiState>, address: String) -> Result<Vec<String>, String> {
    let advertisement = parse_manual_peer(&address)?;
    let mut manual_peers = state
        .manual_peers
        .lock()
        .map_err(|_| "failed to lock manual peers".to_owned())?;

    manual_peers.retain(|peer| peer.peer_id != advertisement.peer_id);
    manual_peers.push(advertisement);
    Ok(manual_peers.iter().flat_map(peer_address_labels).collect())
}

#[tauri::command]
fn clear_manual_peers(state: State<'_, UiState>) -> Result<Vec<String>, String> {
    state
        .manual_peers
        .lock()
        .map_err(|_| "failed to lock manual peers".to_owned())?
        .clear();
    Ok(Vec::new())
}

#[tauri::command]
async fn get_grid_snapshot(
    app: AppHandle,
    state: State<'_, UiState>,
    discovery_timeout_ms: Option<u64>,
    model_id: Option<String>,
) -> Result<GridSnapshot, String> {
    let cache_config = cache_config_for_app(&app);
    ensure_model_distribution_service(&state, cache_config.clone()).await?;

    collect_snapshot(
        &app,
        &state,
        discovery_timeout_ms.unwrap_or(DEFAULT_DISCOVERY_TIMEOUT_MS),
        model_id.as_deref(),
    )
    .await
}

#[tauri::command]
fn get_image_runtime_status(
    app: AppHandle,
    state: State<'_, UiState>,
) -> Result<image_runtime::ImageRuntimeStatus, String> {
    let cache_config = cache_config_for_app(&app);
    sync_local_image_components(&cache_config);
    let mut status = image_runtime::image_runtime_status(&cache_config);
    status.busy = image_operation_is_busy(&state)?;
    Ok(status)
}

#[tauri::command]
async fn install_official_image(
    app: AppHandle,
    state: State<'_, UiState>,
) -> Result<image_runtime::ImageRuntimeStatus, String> {
    let _operation = begin_image_operation(&state)?;
    let cache_config = cache_config_for_app(&app);
    let progress_app = app.clone();
    let result = image_runtime::install_official_package(&cache_config, move |progress| {
        emit_model_import_progress(
            &progress_app,
            image_runtime::IMAGE_MODEL_ID,
            progress.stage,
            progress.detail,
            progress.downloaded_bytes,
            Some(progress.total_bytes),
        );
    })
    .await;

    match result {
        Ok(mut status) => {
            sync_local_image_components(&cache_config);
            status.busy = false;
            Ok(status)
        }
        Err(error) => {
            sync_local_image_components(&cache_config);
            emit_model_import_progress(
                &app,
                image_runtime::IMAGE_MODEL_ID,
                "Image package download failed",
                error.to_string(),
                image_runtime::image_runtime_status(&cache_config).downloaded_bytes,
                Some(image_runtime::IMAGE_TOTAL_BYTES),
            );
            Err(error.to_string())
        }
    }
}

#[tauri::command]
async fn list_generated_images(app: AppHandle) -> Result<Vec<GenerateImageResponse>, String> {
    let image_dir = app_data_dir(&app).join("generated-images");
    let mut directory = match tokio::fs::read_dir(&image_dir).await {
        Ok(directory) => directory,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => {
            return Err(format!(
                "failed to read generated images at {}: {error}",
                image_dir.display()
            ));
        }
    };
    let mut creations = Vec::new();

    while let Some(entry) = directory
        .next_entry()
        .await
        .map_err(|error| format!("failed to inspect generated images: {error}"))?
    {
        let path = entry.path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("png") {
            continue;
        }
        let Some(image_id) = path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        if uuid::Uuid::parse_str(image_id).is_err() {
            continue;
        }

        let image_bytes = match tokio::fs::read(&path).await {
            Ok(bytes) => bytes,
            Err(error) => {
                eprintln!(
                    "failed to reload generated image {}: {error}",
                    path.display()
                );
                continue;
            }
        };
        let Some((png_width, png_height)) = png_dimensions(&image_bytes) else {
            eprintln!("ignored invalid generated PNG at {}", path.display());
            continue;
        };
        let metadata_path = image_dir.join(format!("{image_id}.json"));
        let saved_metadata = match tokio::fs::read(&metadata_path).await {
            Ok(bytes) => serde_json::from_slice::<GeneratedImageMetadata>(&bytes)
                .ok()
                .filter(|metadata| metadata.version == 1 && metadata.image_id == image_id),
            Err(_) => None,
        };
        let modified_at = entry
            .metadata()
            .await
            .ok()
            .and_then(|metadata| metadata.modified().ok())
            .and_then(|modified| modified.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|duration| duration.as_millis())
            .unwrap_or(0);

        let response = if let Some(metadata) = saved_metadata {
            GenerateImageResponse {
                image_data_url: format!(
                    "data:image/png;base64,{}",
                    BASE64_STANDARD.encode(&image_bytes)
                ),
                image_id: metadata.image_id,
                prompt: metadata.prompt,
                seed: metadata.seed,
                width: metadata.width,
                height: metadata.height,
                steps: metadata.steps,
                duration_ms: metadata.duration_ms,
                release_id: metadata.release_id,
                placement: metadata.placement,
                details_available: true,
            }
        } else {
            GenerateImageResponse {
                image_data_url: format!(
                    "data:image/png;base64,{}",
                    BASE64_STANDARD.encode(&image_bytes)
                ),
                image_id: image_id.to_owned(),
                prompt: "Saved creation".to_owned(),
                seed: 0,
                width: png_width,
                height: png_height,
                steps: 0,
                duration_ms: 0,
                release_id: image_runtime::IMAGE_RELEASE_ID.to_owned(),
                placement: "Saved on this computer".to_owned(),
                details_available: false,
            }
        };
        creations.push((modified_at, response));
    }

    creations.sort_by(|left, right| right.0.cmp(&left.0));
    Ok(creations
        .into_iter()
        .map(|(_, creation)| creation)
        .collect())
}

fn png_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    const PNG_SIGNATURE: &[u8; 8] = b"\x89PNG\r\n\x1a\n";
    if bytes.len() < 24 || &bytes[..8] != PNG_SIGNATURE || &bytes[12..16] != b"IHDR" {
        return None;
    }
    let width = u32::from_be_bytes(bytes[16..20].try_into().ok()?);
    let height = u32::from_be_bytes(bytes[20..24].try_into().ok()?);
    (width > 0 && height > 0).then_some((width, height))
}

async fn persist_generated_image_metadata(
    image_dir: &std::path::Path,
    metadata: &GeneratedImageMetadata,
) -> Result<(), String> {
    let metadata_path = image_dir.join(format!("{}.json", metadata.image_id));
    let temporary_path = image_dir.join(format!(".{}.json.tmp", metadata.image_id));
    let bytes = serde_json::to_vec_pretty(metadata)
        .map_err(|error| format!("failed to serialize generated image metadata: {error}"))?;
    tokio::fs::write(&temporary_path, bytes)
        .await
        .map_err(|error| format!("failed to save generated image metadata: {error}"))?;
    tokio::fs::rename(&temporary_path, &metadata_path)
        .await
        .map_err(|error| format!("failed to commit generated image metadata: {error}"))
}

#[tauri::command]
async fn generate_image(
    app: AppHandle,
    state: State<'_, UiState>,
    prompt: String,
    seed: Option<i64>,
) -> Result<GenerateImageResponse, String> {
    let prompt = prompt.trim().to_owned();
    if prompt.is_empty() {
        return Err("Describe an image before generating.".to_owned());
    }
    if prompt.chars().count() > 4_000 {
        return Err("Image prompts must be 4,000 characters or fewer.".to_owned());
    }
    let _operation = begin_image_operation(&state)?;
    let cache_config = cache_config_for_app(&app);
    ensure_image_rpc_service(&state, &cache_config).await?;
    ensure_model_distribution_service(&state, cache_config.clone()).await?;
    let package_paths = image_runtime::ensure_verified_package(&cache_config)
        .await
        .map_err(|error| error.to_string())?;
    sync_local_image_components(&cache_config);
    let binary = find_sd_cli_binary().ok_or_else(|| {
        "The Infernet Image runtime is not prepared for this platform.".to_owned()
    })?;
    let local_capabilities = detect_node_capabilities();
    let origin_machine_id = local_capabilities
        .machine_id
        .as_deref()
        .map(str::trim)
        .filter(|machine_id| !machine_id.is_empty())
        .ok_or_else(|| "Infernet could not verify this computer's identity.".to_owned())?;
    let discovered_placement = discover_image_placement(
        &app,
        &state,
        &cache_config,
        origin_machine_id,
        Duration::from_millis(DEFAULT_DISCOVERY_TIMEOUT_MS),
    )
    .await?;
    let placement = discovered_placement.placement.clone();

    let local_backend = match local_capabilities.compute_backend.as_str() {
        "metal" => "metal",
        "cuda" => "cuda0",
        _ => {
            return Err(
                "Infernet Image currently requires a Metal or CUDA accelerator.".to_owned(),
            );
        }
    };
    let (runtime_backend, params_backend, split_mode, max_vram, placement_label) = match placement {
        StableDiffusionPlacement::RequesterLocal => (
            local_backend.to_owned(),
            Some("*=cpu".to_owned()),
            None,
            None,
            "requester-local sole eligible machine".to_owned(),
        ),
        StableDiffusionPlacement::Distributed { machine_count } => (
            distributed_diffusion_backend(local_backend, machine_count)
                .map_err(|error| error.to_string())?,
            Some("te=cpu,vae=cpu".to_owned()),
            Some("layer".to_owned()),
            Some(
                distributed_diffusion_max_vram(local_backend, machine_count)
                    .map_err(|error| error.to_string())?,
            ),
            format!(
                "DiT blocks split across {machine_count} physical machines, including requester"
            ),
        ),
    };
    let image_trace_id = uuid::Uuid::new_v4();
    let image_id = image_trace_id.to_string();
    let seed = seed.unwrap_or_else(|| (current_unix_ms_u64() & i64::MAX as u64) as i64);
    let image_dir = app_data_dir(&app).join("generated-images");
    let config = StableDiffusionConfig {
        binary,
        diffusion_model_path: package_paths.diffusion_model,
        text_encoder_path: package_paths.text_encoder,
        vae_path: package_paths.vae,
        output_dir: image_dir.clone(),
        log_dir: app_data_dir(&app).join("runtime-logs").join("image"),
        backend: runtime_backend,
        params_backend,
        split_mode,
        max_vram,
        rpc_servers: Vec::new(),
        placement: placement.clone(),
        timeout: Duration::from_secs(10 * 60),
    };
    let request = ImageGenerationRequest {
        job_id: image_id.clone(),
        prompt: prompt.clone(),
        seed,
        width: 1024,
        height: 1024,
        steps: 8,
    };

    let activity =
        begin_local_node_activity(image_trace_id, LocalNodeActivityKind::ImageGeneration);
    set_local_inference_active(true);
    let generation: Result<_, String> = async {
        match placement {
            StableDiffusionPlacement::RequesterLocal => {
                tokio::task::spawn_blocking(move || generate_with_sd_cli(&config, &request))
                    .await
                    .map_err(|error| format!("Infernet Image runtime task failed: {error}"))?
                    .map_err(|error| error.to_string())
            }
            StableDiffusionPlacement::Distributed { .. } => {
                let (mut discovery, _) = discovery_config_from_state(&state)?;
                discovery.keypair = identity::Keypair::generate_ed25519();
                discovery.advertisement = None;
                discovery.advertise_listen_addresses = false;
                merge_static_peer_advertisements(
                    &mut discovery.static_peers,
                    discovered_placement.advertisements,
                );
                discovery
                    .set_rpc_worker_peer_ids(discovered_placement.remote_worker_peer_ids)
                    .map_err(|error| error.to_string())?;
                generate_image_over_libp2p(discovery, config, request, Duration::from_secs(30))
                    .await
                    .map_err(|error| error.to_string())
            }
        }
    }
    .await;
    set_local_inference_active(false);
    let output = generation?;
    let image_bytes = tokio::fs::read(&output.png_path)
        .await
        .map_err(|error| format!("failed to read generated image: {error}"))?;
    let metadata = GeneratedImageMetadata {
        version: 1,
        image_id: image_id.clone(),
        prompt: prompt.clone(),
        seed: output.seed,
        width: output.width,
        height: output.height,
        steps: output.steps,
        duration_ms: output.duration_ms,
        release_id: image_runtime::IMAGE_RELEASE_ID.to_owned(),
        placement: placement_label.clone(),
    };
    if let Err(error) = persist_generated_image_metadata(&image_dir, &metadata).await {
        eprintln!("{error}");
    }
    activity.complete(LocalNodeActivityOutcome::Success);

    Ok(GenerateImageResponse {
        image_data_url: format!(
            "data:image/png;base64,{}",
            BASE64_STANDARD.encode(image_bytes)
        ),
        image_id,
        prompt,
        seed: output.seed,
        width: output.width,
        height: output.height,
        steps: output.steps,
        duration_ms: output.duration_ms,
        release_id: image_runtime::IMAGE_RELEASE_ID.to_owned(),
        placement: placement_label,
        details_available: true,
    })
}

fn begin_image_operation(state: &State<'_, UiState>) -> Result<ImageOperationGuard, String> {
    let busy = Arc::clone(&state.image_operation);
    {
        let mut active = busy
            .lock()
            .map_err(|_| "failed to lock Infernet Image operation state".to_owned())?;
        if *active {
            return Err("Infernet Image is already installing or generating.".to_owned());
        }
        *active = true;
    }
    Ok(ImageOperationGuard { busy })
}

fn image_operation_is_busy(state: &State<'_, UiState>) -> Result<bool, String> {
    state
        .image_operation
        .lock()
        .map(|busy| *busy)
        .map_err(|_| "failed to lock Infernet Image operation state".to_owned())
}

fn eligible_image_machine_ids(
    advertisements: &[NodeAdvertisement],
    local_peer_id: &str,
) -> BTreeSet<String> {
    advertisements
        .iter()
        .filter_map(|advertisement| {
            let capabilities = image_advertisement_is_eligible(advertisement, local_peer_id)?;
            capabilities
                .machine_id
                .as_deref()
                .map(str::trim)
                .filter(|machine_id| !machine_id.is_empty())
                .map(str::to_owned)
        })
        .collect()
}

fn image_advertisement_is_eligible<'a>(
    advertisement: &'a NodeAdvertisement,
    local_peer_id: &str,
) -> Option<&'a infernet_protocol::NodeCapabilities> {
    let capabilities = advertisement.capabilities.as_ref()?;
    image_advertisement_ineligibility_reason(advertisement, local_peer_id)
        .is_none()
        .then_some(capabilities)
}

fn image_advertisement_ineligibility_reason(
    advertisement: &NodeAdvertisement,
    local_peer_id: &str,
) -> Option<String> {
    let Some(capabilities) = advertisement.capabilities.as_ref() else {
        return Some("has no current compute capability report".to_owned());
    };
    if capabilities
        .machine_id
        .as_deref()
        .map(str::trim)
        .is_none_or(str::is_empty)
    {
        return Some("has no verified physical-machine identity".to_owned());
    }
    let expected_components = image_runtime::component_infos();
    if !expected_components.iter().all(|expected| {
        advertisement
            .model_components
            .iter()
            .any(|actual| actual == expected)
    }) {
        return Some("is not advertising the complete verified Infernet Image package".to_owned());
    }
    if !matches!(capabilities.compute_backend.as_str(), "cuda" | "metal") {
        return Some(format!(
            "uses the unsupported {} compute backend",
            capabilities.compute_backend
        ));
    }
    if capabilities.active_sessions >= capabilities.max_sessions {
        return Some(format!(
            "has no free inference session ({}/{})",
            capabilities.active_sessions, capabilities.max_sessions
        ));
    }
    if capabilities.total_accelerator_memory_bytes == 0 {
        return Some("reports no accelerator memory capacity".to_owned());
    }
    if advertisement.peer_id != local_peer_id {
        if advertisement.addresses.is_empty() {
            return Some("has no authenticated network route".to_owned());
        }
        if capabilities.vram_contribution_limit_bytes == Some(0) {
            return Some("has accelerator sharing disabled".to_owned());
        }
        let Some(endpoint) = capabilities.image_rpc.as_ref() else {
            return Some("is not advertising an Infernet Image RPC worker".to_owned());
        };
        if !endpoint.ready {
            return Some("has an Infernet Image RPC worker that is not ready".to_owned());
        }
        if endpoint.rpc_protocol_version != LLAMA_RPC_PROTOCOL_VERSION
            || endpoint.runtime_abi != INFERNET_IMAGE_RPC_RUNTIME_ABI
            || endpoint.tunnel_protocol.as_deref() != Some(IMAGE_RPC_TUNNEL_PROTOCOL)
            || !matches!(endpoint.backend.as_str(), "cuda" | "metal")
        {
            return Some("is advertising an incompatible Infernet Image RPC worker".to_owned());
        }
    }
    None
}

fn local_image_ineligibility_message(
    advertisements: &[NodeAdvertisement],
    local_peer_id: &str,
) -> Option<String> {
    let advertisement = advertisements
        .iter()
        .find(|advertisement| advertisement.peer_id == local_peer_id)?;
    image_advertisement_ineligibility_reason(advertisement, local_peer_id).map(|reason| {
        format!("This computer cannot participate in the image split because it {reason}.")
    })
}

fn select_image_rpc_workers(
    advertisements: &[NodeAdvertisement],
    local_peer_id: &str,
    origin_machine_id: &str,
) -> Vec<String> {
    let mut best_by_machine = BTreeMap::<String, (u64, String)>::new();
    for advertisement in advertisements {
        let Some(capabilities) = image_advertisement_is_eligible(advertisement, local_peer_id)
        else {
            continue;
        };
        let Some(machine_id) = capabilities
            .machine_id
            .as_deref()
            .map(str::trim)
            .filter(|machine_id| !machine_id.is_empty() && *machine_id != origin_machine_id)
        else {
            continue;
        };
        let candidate = (
            capabilities.available_accelerator_memory_bytes,
            advertisement.peer_id.clone(),
        );
        best_by_machine
            .entry(machine_id.to_owned())
            .and_modify(|current| {
                if candidate.0 > current.0 || (candidate.0 == current.0 && candidate.1 < current.1)
                {
                    *current = candidate.clone();
                }
            })
            .or_insert(candidate);
    }
    let mut workers = best_by_machine.into_values().collect::<Vec<_>>();
    // The final device receives the uncapped remainder of the transformer, so
    // put the largest remote last while every earlier machine gets a forced
    // non-empty contiguous range.
    workers.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
    workers.into_iter().map(|(_, peer_id)| peer_id).collect()
}

fn plan_image_placement(
    eligible_machine_ids: &BTreeSet<String>,
    origin_machine_id: &str,
) -> Result<StableDiffusionPlacement, String> {
    match eligible_machine_ids.len() {
        0 => Err("No eligible computer has the verified Infernet Image package.".to_owned()),
        1 if eligible_machine_ids.contains(origin_machine_id) => {
            Ok(StableDiffusionPlacement::RequesterLocal)
        }
        1 => Err(
            "Infernet will not assign an entire image request to one remote computer; waiting for another eligible computer."
                .to_owned(),
        ),
        _ if !eligible_machine_ids.contains(origin_machine_id) => Err(
            "This computer must participate in the image generation split, but it is not currently eligible."
                .to_owned(),
        ),
        machine_count => Ok(StableDiffusionPlacement::Distributed { machine_count }),
    }
}

async fn discover_image_placement(
    app: &AppHandle,
    state: &State<'_, UiState>,
    cache_config: &ShardCacheConfig,
    origin_machine_id: &str,
    timeout: Duration,
) -> Result<DiscoveredImagePlacement, String> {
    let deadline = Instant::now() + timeout;
    loop {
        let (registry, local_peer_id, _, _) =
            discover_registry(app, state, cache_config, 0).await?;
        let advertisements = registry.advertisements();
        let eligible_machines = eligible_image_machine_ids(&advertisements, &local_peer_id);
        let planned = plan_image_placement(&eligible_machines, origin_machine_id);

        match &planned {
            Ok(StableDiffusionPlacement::Distributed { machine_count }) => {
                let remote_worker_peer_ids =
                    select_image_rpc_workers(&advertisements, &local_peer_id, origin_machine_id);
                if remote_worker_peer_ids.len() + 1 != *machine_count {
                    return Err(
                        "Infernet could not bind every eligible physical machine to one image worker."
                            .to_owned(),
                    );
                }
                return Ok(DiscoveredImagePlacement {
                    placement: planned?,
                    remote_worker_peer_ids,
                    advertisements,
                });
            }
            Ok(StableDiffusionPlacement::RequesterLocal) if Instant::now() >= deadline => {
                return Ok(DiscoveredImagePlacement {
                    placement: planned?,
                    remote_worker_peer_ids: Vec::new(),
                    advertisements,
                });
            }
            Err(error) if Instant::now() >= deadline => {
                return Err(
                    local_image_ineligibility_message(&advertisements, &local_peer_id)
                        .unwrap_or_else(|| error.clone()),
                );
            }
            _ => sleep(Duration::from_millis(100)).await,
        }
    }
}

fn sync_local_image_components(cache_config: &ShardCacheConfig) {
    set_local_model_components(image_runtime::advertised_components(cache_config));
}

#[tauri::command]
async fn install_official_model(
    app: AppHandle,
    state: State<'_, UiState>,
    model_id: String,
) -> Result<GridSnapshot, String> {
    let cache_config = cache_config_for_app(&app);
    ensure_model_distribution_service(&state, cache_config.clone()).await?;
    let (registry, _, _, _) =
        discover_registry(&app, &state, &cache_config, DEFAULT_DISCOVERY_TIMEOUT_MS).await?;
    let manifest = manifest_for_model(Some(&model_id), &cache_config, Some(&registry))
        .map_err(|error| error.to_string())?;
    if let Err(https_error) =
        acquire_official_model_over_https(&app, &cache_config, &manifest).await
    {
        eprintln!("official HTTPS model acquisition failed: {https_error}");
        let p2p_plan = advertised_model_record_plan(&registry, &manifest.model_id);
        if p2p_plan.is_empty() {
            emit_model_import_progress(
                &app,
                &manifest.model_id,
                "Download failed",
                format!("HTTPS failed and no P2P seed is online: {https_error}"),
                0,
                None,
            );
            return Err(format!(
                "The official model download failed, and no P2P seed is online: {https_error}"
            ));
        }
        let release = OfficialModelRelease::infernet_chat_v1_compatibility();
        let full_layers =
            LayerRange::new(0, manifest.layer_count).map_err(|error| error.to_string())?;
        let (_, total_bytes) = release
            .components
            .iter()
            .find(|component| component.layers == Some(full_layers))
            .map(|component| (component.sha256.as_str(), component.size_bytes))
            .ok_or_else(|| "official release has no downloadable component".to_owned())?;
        let (partial_path, _) =
            official_release_download_paths(&cache_config, &release, full_layers);
        let downloaded_bytes = std::fs::metadata(partial_path)
            .map(|metadata| metadata.len())
            .unwrap_or(0);
        emit_model_import_progress(
            &app,
            &manifest.model_id,
            "Downloading shard",
            format!(
                "layers {}:{} · P2P fallback",
                full_layers.start, full_layers.end
            ),
            downloaded_bytes,
            Some(total_bytes),
        );
        acquire_advertised_model_records(&app, &state, &cache_config, &manifest, &registry, true)
            .await?;
    }
    collect_snapshot(
        &app,
        &state,
        DEFAULT_DISCOVERY_TIMEOUT_MS,
        Some(&manifest.model_id),
    )
    .await
}

#[tauri::command]
async fn run_demo_inference(
    app: AppHandle,
    state: State<'_, UiState>,
    prompt: String,
    model_id: Option<String>,
) -> Result<RunDemoResponse, String> {
    let cache_config = cache_config_for_app(&app);
    ensure_llama_rpc_service(&state, &cache_config).await?;
    ensure_model_distribution_service(&state, cache_config.clone()).await?;
    let (registry, _, _, _) =
        discover_registry(&app, &state, &cache_config, DEFAULT_DISCOVERY_TIMEOUT_MS).await?;
    let manifest = manifest_for_model(model_id.as_deref(), &cache_config, Some(&registry))
        .map_err(|error| error.to_string())?;

    let snapshot = collect_snapshot(
        &app,
        &state,
        DEFAULT_DISCOVERY_TIMEOUT_MS,
        Some(&manifest.model_id),
    )
    .await?;

    if snapshot.route.is_empty() {
        let message = snapshot
            .missing_ranges
            .clone()
            .unwrap_or_else(|| "no complete route discovered".to_owned());
        emit_progress(
            &app,
            ProgressEvent::Error {
                message: message.clone(),
            },
        );
        return Err(message);
    }

    let worker_plan = match leased_worker_execution_plan(&state, &registry, &manifest) {
        Ok(plan) => plan,
        Err(message) => {
            emit_progress(
                &app,
                ProgressEvent::Error {
                    message: message.clone(),
                },
            );
            return Err(message);
        }
    };

    emit_progress(
        &app,
        ProgressEvent::RouteDiscovered {
            route: worker_plan.route.iter().map(route_hop_view).collect(),
        },
    );
    emit_progress(
        &app,
        ProgressEvent::ExecutionPlan {
            participants: worker_plan.participants.clone(),
        },
    );

    let (mut config, local_peer_id) = discovery_config_from_state(&state)?;
    config.keypair = identity::Keypair::generate_ed25519();
    let mut local_advertisement = registry
        .advertisements()
        .into_iter()
        .find(|advertisement| advertisement.peer_id == local_peer_id)
        .unwrap_or_else(|| local_capability_advertisement(local_peer_id.clone(), String::new()));
    local_advertisement.addresses = local_connect_addresses(&local_peer_id);
    config.static_peers.push(local_advertisement);
    merge_static_peer_advertisements(&mut config.static_peers, registry.advertisements());
    config.set_planned_route(worker_plan.route.clone());
    let hidden_size = manifest.hidden_size;
    let inference_result = infer_over_libp2p(
        config,
        manifest,
        prompt,
        hidden_size,
        Duration::from_millis(DEFAULT_INFERENCE_TIMEOUT_MS),
    )
    .await;
    set_local_inference_active(false);
    let result = match inference_result {
        Ok(result) => result,
        Err(error) => {
            let message = error.to_string();
            if let Ok(mut lease) = state.execution_plan.lock() {
                *lease = None;
            }
            emit_progress(
                &app,
                ProgressEvent::Error {
                    message: message.clone(),
                },
            );
            return Err(message);
        }
    };

    let trace_id = result.response.trace_id.to_string();
    replay_trace_progress(&app, &trace_id, &result.response.trace).await;
    let output = result
        .response
        .output_text
        .clone()
        .filter(|output| !output.trim().is_empty());
    let Some(output) = output else {
        let message = "Infernet Chat completed without generating text.".to_owned();
        if let Ok(mut lease) = state.execution_plan.lock() {
            *lease = None;
        }
        emit_progress(
            &app,
            ProgressEvent::Error {
                message: message.clone(),
            },
        );
        return Err(message);
    };
    emit_progress(
        &app,
        ProgressEvent::FinalOutput {
            trace_id: trace_id.clone(),
            output: output.clone(),
        },
    );

    Ok(RunDemoResponse {
        output,
        trace_id,
        snapshot,
    })
}

fn leased_worker_execution_plan(
    state: &State<'_, UiState>,
    registry: &ShardRegistry,
    manifest: &ModelManifest,
) -> Result<execution_plan::WorkerExecutionPlan, String> {
    let origin_peer_id = identity_from_state(state)?.0;
    let mut lease = state
        .execution_plan
        .lock()
        .map_err(|_| "failed to lock distributed execution plan".to_owned())?;
    if let Some(plan) = lease.as_ref() {
        if plan.remains_usable(registry, manifest) {
            return Ok(plan.clone());
        }
    }
    let plan = plan_worker_execution(registry, manifest, &origin_peer_id)?;
    *lease = Some(plan.clone());
    Ok(plan)
}

async fn collect_snapshot(
    app: &AppHandle,
    state: &State<'_, UiState>,
    discovery_timeout_ms: u64,
    model_id: Option<&str>,
) -> Result<GridSnapshot, String> {
    let cache_config = cache_config_for_app(app);
    let (registry, local_peer_id, topic, presence) =
        discover_registry(app, state, &cache_config, discovery_timeout_ms).await?;
    let manifest = match manifest_for_model(model_id, &cache_config, Some(&registry)) {
        Ok(manifest) => manifest,
        Err(error) => {
            if model_id.is_none_or(|value| value.trim().is_empty()) {
                return Ok(empty_snapshot(
                    local_peer_id,
                    topic,
                    &cache_config,
                    &registry,
                    &presence,
                ));
            }
            return Err(error.to_string());
        }
    };
    spawn_background_model_record_acquisition(app, state, &cache_config, &manifest, &registry)?;

    Ok(snapshot_from_registry(
        local_peer_id,
        topic,
        &manifest,
        &registry,
        &cache_config,
        &presence,
    ))
}

async fn discover_registry(
    _app: &AppHandle,
    state: &State<'_, UiState>,
    cache_config: &ShardCacheConfig,
    _discovery_timeout_ms: u64,
) -> Result<(ShardRegistry, String, String, PresenceSnapshot), String> {
    let (local_peer_id, topic) = identity_from_state(state)?;
    let mut registry = state
        .live_registry
        .lock()
        .map_err(|_| "failed to lock live peer registry".to_owned())?
        .clone();
    let mut local_advertisement = local_node_advertisement(cache_config, local_peer_id.clone());
    local_advertisement.addresses = local_connect_addresses(&local_peer_id);
    registry.upsert(local_advertisement);
    let fresh_registry = trusted_launch_registry(registry, &local_peer_id);
    // Static bootstrap/manual descriptors are inserted before discovery, so
    // their mere presence is not proof that a machine is online. Only a
    // current capability or executable model report refreshes last-seen.
    let observed_advertisements = fresh_registry
        .advertisements()
        .into_iter()
        .filter(advertisement_has_capacity)
        .collect::<Vec<_>>();
    let presence = state
        .peer_presence
        .lock()
        .map_err(|_| "failed to lock peer presence state".to_owned())?
        .observe(observed_advertisements);
    let mut routable_registry = ShardRegistry::new();
    routable_registry.extend(presence.routable_advertisements());

    Ok((routable_registry, local_peer_id, topic, presence))
}

fn trusted_launch_registry(registry: ShardRegistry, local_peer_id: &str) -> ShardRegistry {
    let mut trusted = ShardRegistry::new();
    for mut advertisement in registry.advertisements() {
        if advertisement.peer_id != local_peer_id {
            advertisement
                .addresses
                .retain(|address| remote_route_address_is_usable(address));
            advertisement.addresses.sort_by_key(|address| {
                let is_tcp_circuit = address.contains("/tcp/") && address.contains("/p2p-circuit/");
                let is_circuit = address.contains("/p2p-circuit/");
                (!is_tcp_circuit, !is_circuit, address.clone())
            });
            advertisement.addresses.dedup();
        }
        let trusted_records = advertisement
            .model_shards
            .iter()
            .filter(|info| official_info_matches_release(info))
            .cloned()
            .collect::<Vec<_>>();
        let runnable_components = advertisement
            .hosted_shards
            .iter()
            .filter_map(|descriptor| {
                let info = trusted_records.iter().find(|info| {
                    info.model_id == descriptor.model_id && info.layers == descriptor.layers
                })?;
                let manifest = descriptor.seed_manifest.as_deref()?;
                official_record_matches_release(info, manifest).then(|| {
                    (
                        descriptor.model_id.clone(),
                        descriptor.layers.start,
                        descriptor.layers.end,
                    )
                })
            })
            .collect::<BTreeSet<_>>();
        advertisement.hosted_shards.retain(|descriptor| {
            runnable_components.contains(&(
                descriptor.model_id.clone(),
                descriptor.layers.start,
                descriptor.layers.end,
            ))
        });
        advertisement.model_shards = trusted_records;
        trusted.upsert(advertisement);
    }
    trusted
}

fn remote_route_address_is_usable(address: &str) -> bool {
    let Ok(address) = address.parse::<Multiaddr>() else {
        return false;
    };
    !address.iter().any(|protocol| match protocol {
        Protocol::Ip4(host) => host.is_loopback() || host.is_unspecified(),
        Protocol::Ip6(host) => host.is_loopback() || host.is_unspecified(),
        _ => false,
    })
}

fn official_release_download_url(release: &OfficialModelRelease) -> String {
    format!(
        "https://huggingface.co/{}/resolve/{}/{}",
        release.upstream.repository, release.upstream.revision, release.upstream.artifact
    )
}

fn official_release_download_paths(
    cache_config: &ShardCacheConfig,
    release: &OfficialModelRelease,
    layers: LayerRange,
) -> (PathBuf, PathBuf) {
    let checksum_prefix =
        &release.upstream.source_sha256[..release.upstream.source_sha256.len().min(16)];
    let base = format!(
        "{}-{}-{}-{}",
        sanitize_download_path_segment(&release.model_id),
        layers.start,
        layers.end,
        checksum_prefix
    );
    let temp_dir = cache_config.root.join("tmp");
    (
        temp_dir.join(format!("{base}.gguf.partial")),
        temp_dir.join(format!("{base}.gguf")),
    )
}

fn sanitize_download_path_segment(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric()
                || character == '-'
                || character == '_'
                || character == '.'
            {
                character
            } else {
                '_'
            }
        })
        .collect()
}

fn content_range_start(value: &str) -> Option<u64> {
    value
        .strip_prefix("bytes ")?
        .split_once('-')?
        .0
        .parse()
        .ok()
}

async fn acquire_official_model_over_https(
    app: &AppHandle,
    cache_config: &ShardCacheConfig,
    manifest: &ModelManifest,
) -> anyhow::Result<()> {
    let release = OfficialModelRelease::infernet_chat_v1_compatibility();
    release.validate_for_model(manifest)?;
    let full_layers = LayerRange::new(0, manifest.layer_count)?;
    let component = release
        .components
        .iter()
        .find(|component| {
            component.kind == OfficialComponentKind::Transformer
                && component.layers == Some(full_layers)
        })
        .ok_or_else(|| anyhow::anyhow!("official release has no full-model component"))?;
    let cache = ShardCache::new(cache_config.clone())?;
    if cache
        .find(
            &manifest.model_id,
            full_layers,
            Some(&component.sha256),
            Some(&release.version),
        )?
        .is_some()
    {
        return Ok(());
    }

    let (partial_path, completed_path) =
        official_release_download_paths(cache_config, &release, full_layers);
    let temp_dir = partial_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("official download has no temporary directory"))?;
    tokio::fs::create_dir_all(temp_dir).await?;

    if tokio::fs::metadata(&completed_path)
        .await
        .is_ok_and(|metadata| metadata.len() != component.size_bytes)
    {
        tokio::fs::remove_file(&completed_path).await?;
    }

    let mut downloaded_bytes = tokio::fs::metadata(&completed_path)
        .await
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    if downloaded_bytes == 0 {
        downloaded_bytes = tokio::fs::metadata(&partial_path)
            .await
            .map(|metadata| metadata.len())
            .unwrap_or(0);
    }
    if downloaded_bytes > component.size_bytes {
        tokio::fs::remove_file(&partial_path).await?;
        downloaded_bytes = 0;
    }

    let progress_detail = format!(
        "layers {}:{} · Hugging Face",
        full_layers.start, full_layers.end
    );
    emit_model_import_progress(
        app,
        &manifest.model_id,
        "Downloading shard",
        progress_detail.clone(),
        downloaded_bytes,
        Some(component.size_bytes),
    );

    if downloaded_bytes < component.size_bytes {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(20))
            .user_agent(concat!("infernet/", env!("CARGO_PKG_VERSION")))
            .build()?;
        let mut request = client.get(official_release_download_url(&release));
        if downloaded_bytes > 0 {
            request = request.header(reqwest::header::RANGE, format!("bytes={downloaded_bytes}-"));
        }
        let response = request.send().await?;
        let status = response.status();
        if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE
            && downloaded_bytes == component.size_bytes
        {
            // The partial file completed before the previous process exited.
        } else if !status.is_success() {
            return Err(anyhow::anyhow!(
                "Hugging Face returned HTTP {}",
                status.as_u16()
            ));
        } else {
            let append = downloaded_bytes > 0 && status == reqwest::StatusCode::PARTIAL_CONTENT;
            if append {
                let range_start = response
                    .headers()
                    .get(reqwest::header::CONTENT_RANGE)
                    .and_then(|value| value.to_str().ok())
                    .and_then(content_range_start);
                if range_start != Some(downloaded_bytes) {
                    return Err(anyhow::anyhow!(
                        "Hugging Face returned an invalid resume range"
                    ));
                }
            } else if downloaded_bytes > 0 {
                // The origin ignored Range. Restart safely instead of appending
                // a complete response to the existing partial file.
                downloaded_bytes = 0;
            }

            let mut options = tokio::fs::OpenOptions::new();
            options.create(true).write(true);
            if append {
                options.append(true);
            } else {
                options.truncate(true);
            }
            let mut file = options.open(&partial_path).await?;
            let mut stream = response.bytes_stream();
            let mut last_progress_emit = downloaded_bytes;
            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                let next_downloaded = downloaded_bytes
                    .checked_add(chunk.len() as u64)
                    .ok_or_else(|| anyhow::anyhow!("official model download size overflow"))?;
                if next_downloaded > component.size_bytes {
                    return Err(anyhow::anyhow!(
                        "Hugging Face download exceeded the pinned model size"
                    ));
                }
                file.write_all(&chunk).await?;
                downloaded_bytes = next_downloaded;
                if downloaded_bytes == component.size_bytes
                    || downloaded_bytes.saturating_sub(last_progress_emit)
                        >= MODEL_PROGRESS_EMIT_BYTES
                {
                    emit_model_import_progress(
                        app,
                        &manifest.model_id,
                        "Downloading shard",
                        progress_detail.clone(),
                        downloaded_bytes,
                        Some(component.size_bytes),
                    );
                    last_progress_emit = downloaded_bytes;
                }
            }
            file.flush().await?;
        }
    }

    if downloaded_bytes != component.size_bytes {
        return Err(anyhow::anyhow!(
            "Hugging Face download ended at {} of {} bytes",
            downloaded_bytes,
            component.size_bytes
        ));
    }

    if !tokio::fs::try_exists(&completed_path).await? {
        tokio::fs::rename(&partial_path, &completed_path).await?;
    }
    emit_model_import_progress(
        app,
        &manifest.model_id,
        "Verifying download",
        "Checking the official model checksum",
        component.size_bytes,
        Some(component.size_bytes),
    );

    let checksum_path = completed_path.clone();
    let actual_checksum =
        tokio::task::spawn_blocking(move || sha256_file(&checksum_path)).await??;
    if actual_checksum != component.sha256 {
        let _ = tokio::fs::remove_file(&completed_path).await;
        return Err(anyhow::anyhow!(
            "official model checksum mismatch; expected {}, got {}",
            component.sha256,
            actual_checksum
        ));
    }

    let import_cache_config = cache_config.clone();
    let import_manifest = manifest.clone();
    let import_version = release.version.clone();
    let import_checksum = actual_checksum.clone();
    tokio::task::spawn_blocking(move || {
        let cache = ShardCache::new(import_cache_config)?;
        import_seed_model_from_file_consuming_verified(
            &cache,
            &completed_path,
            &import_manifest,
            import_version,
            import_checksum,
        )
    })
    .await??;

    emit_model_import_progress(
        app,
        &manifest.model_id,
        "Shard ready",
        format!(
            "layers {}:{} · verified HTTPS",
            full_layers.start, full_layers.end
        ),
        component.size_bytes,
        Some(component.size_bytes),
    );
    Ok(())
}

async fn acquire_advertised_model_records(
    app: &AppHandle,
    state: &State<'_, UiState>,
    cache_config: &ShardCacheConfig,
    manifest: &ModelManifest,
    registry: &ShardRegistry,
    allow_direct_fetch: bool,
) -> Result<(), String> {
    if manifest.runtime_kind == RuntimeKind::Demo {
        return Ok(());
    }

    let cache = ShardCache::new(cache_config.clone()).map_err(|error| error.to_string())?;
    let plan = advertised_model_record_plan(registry, &manifest.model_id);
    if plan.is_empty() {
        return Ok(());
    }
    let missing_ranges = missing_ranges_from_layer_ranges(
        manifest.layer_count,
        plan.iter().map(|record| record.info.layers),
    );
    if !missing_ranges.is_empty() {
        if !allow_direct_fetch {
            return Ok(());
        }
        return Err(format!(
            "{} is visible on the network, but its advertised model records are incomplete; missing layer ranges: {}",
            manifest.display_name,
            format_ranges(&missing_ranges),
        ));
    }

    let local_peer_id = identity_from_state(state)?.0;
    let plan = model_records_to_download_for_local_contribution(
        &cache,
        &manifest.model_id,
        &local_peer_id,
        &registry.advertisements(),
        plan,
    )
    .map_err(|error| error.to_string())?;

    let mut static_peers = configured_static_peers(state)?;
    merge_static_peer_advertisements(&mut static_peers, registry.advertisements());
    for record in plan {
        if cache
            .find(
                &record.info.model_id,
                record.info.layers,
                Some(&record.info.checksum),
                Some(&record.info.version),
            )
            .map_err(|error| error.to_string())?
            .is_some()
        {
            continue;
        }

        if !allow_direct_fetch {
            continue;
        }

        emit_model_import_progress(
            app,
            &manifest.model_id,
            "Downloading shard",
            format!(
                "layers {}:{}",
                record.info.layers.start, record.info.layers.end
            ),
            0,
            Some(record.info.size_bytes),
        );
        let (mut config, _) = discovery_config_from_state(state)?;
        config.static_peers = static_peers.clone();
        remove_relay_servers_from_download_targets(&mut config);
        prefer_tcp_circuit_addresses_for_downloads(&mut config);
        // The persistent desktop node already owns the machine's stable
        // relay reservation. A one-shot download using the same PeerId creates
        // a competing reservation and can wait forever for readiness. Give
        // each transfer swarm its own short-lived authenticated identity.
        config.keypair = identity::Keypair::generate_ed25519();
        config.advertisement = None;
        let progress_app = app.clone();
        let progress_model_id = manifest.model_id.clone();
        let progress_detail = format!(
            "layers {}:{}",
            record.info.layers.start, record.info.layers.end
        );
        let mut last_progress_emit = 0_u64;
        fetch_model_shard_over_libp2p_with_progress(
            config,
            cache_config.clone(),
            manifest.model_id.clone(),
            record.info.layers,
            Some(record.info.checksum.clone()),
            Some(record.info.version.clone()),
            Duration::from_millis(model_shard_fetch_timeout_ms(record.info.size_bytes)),
            move |downloaded, total| {
                if downloaded == total
                    || downloaded == 0
                    || downloaded.saturating_sub(last_progress_emit) >= MODEL_PROGRESS_EMIT_BYTES
                {
                    emit_model_import_progress(
                        &progress_app,
                        &progress_model_id,
                        "Downloading shard",
                        progress_detail.clone(),
                        downloaded,
                        Some(total),
                    );
                    last_progress_emit = downloaded;
                }
            },
        )
        .await
        .map_err(|error| error.to_string())?;
        emit_model_import_progress(
            app,
            &manifest.model_id,
            "Shard ready",
            format!(
                "layers {}:{}",
                record.info.layers.start, record.info.layers.end
            ),
            record.info.size_bytes,
            Some(record.info.size_bytes),
        );
    }

    Ok(())
}

fn spawn_background_model_record_acquisition(
    app: &AppHandle,
    state: &State<'_, UiState>,
    cache_config: &ShardCacheConfig,
    manifest: &ModelManifest,
    registry: &ShardRegistry,
) -> Result<(), String> {
    if manifest.runtime_kind == RuntimeKind::Demo {
        return Ok(());
    }

    let cache = ShardCache::new(cache_config.clone()).map_err(|error| error.to_string())?;
    let plan = advertised_model_record_plan(registry, &manifest.model_id);
    if plan.is_empty() {
        return Ok(());
    }
    let missing_ranges = missing_ranges_from_layer_ranges(
        manifest.layer_count,
        plan.iter().map(|record| record.info.layers),
    );
    if !missing_ranges.is_empty() {
        return Ok(());
    }

    let local_peer_id = identity_from_state(state)?.0;
    let plan = match model_records_to_download_for_local_contribution(
        &cache,
        &manifest.model_id,
        &local_peer_id,
        &registry.advertisements(),
        plan,
    ) {
        Ok(plan) => plan,
        Err(error) => {
            // Capacity can legitimately be incomplete while machines are
            // still joining. A snapshot must remain usable; the next refresh
            // retries placement after new capability reports arrive.
            eprintln!(
                "model host placement for {} is not ready: {error}",
                manifest.model_id
            );
            return Ok(());
        }
    };
    if plan.is_empty() {
        return Ok(());
    }

    let acquisition_key = model_acquisition_key(&manifest.model_id, &plan);
    {
        let mut active = state
            .active_model_acquisitions
            .lock()
            .map_err(|_| "failed to lock model acquisition state".to_owned())?;
        if !active.insert(acquisition_key.clone()) {
            return Ok(());
        }
    }

    let mut static_peers = configured_static_peers(state)?;
    merge_static_peer_advertisements(&mut static_peers, registry.advertisements());
    let (mut config, _) = discovery_config_from_state(state)?;
    config.static_peers = static_peers;
    remove_relay_servers_from_download_targets(&mut config);
    prefer_tcp_circuit_addresses_for_downloads(&mut config);
    config.keypair = identity::Keypair::generate_ed25519();
    config.advertisement = None;

    let app = app.clone();
    let cache_config = cache_config.clone();
    let manifest = manifest.clone();
    let active_model_acquisitions = Arc::clone(&state.active_model_acquisitions);

    tauri::async_runtime::spawn(async move {
        let result = async {
            // Snapshot refreshes can automatically prepare this machine for a
            // compute assignment. Keep that path on the same source policy as
            // explicit installs: pinned HTTPS first, peers only after failure.
            let https_error = if manifest.model_id == OFFICIAL_CHAT_MODEL_ID {
                match acquire_official_model_over_https(&app, &cache_config, &manifest).await {
                    Ok(()) => {
                        emit_model_import_progress(
                            &app,
                            &manifest.model_id,
                            "Ready",
                            "Official model verified and seeding",
                            1,
                            Some(1),
                        );
                        return Ok::<(), anyhow::Error>(());
                    }
                    Err(error) => {
                        let error = error.to_string();
                        eprintln!(
                            "official HTTPS model acquisition for {} failed; trying P2P fallback: {error}",
                            manifest.model_id
                        );
                        Some(error)
                    }
                }
            } else {
                None
            };

            for record in plan {
                let cache = ShardCache::new(cache_config.clone())?;
                if cache
                    .find(
                        &record.info.model_id,
                        record.info.layers,
                        Some(&record.info.checksum),
                        Some(&record.info.version),
                    )?
                    .is_some()
                {
                    continue;
                }

                emit_model_import_progress(
                    &app,
                    &manifest.model_id,
                    "Downloading shard",
                    model_download_progress_detail(&record, https_error.is_some()),
                    0,
                    Some(record.info.size_bytes),
                );
                let progress_app = app.clone();
                let progress_model_id = manifest.model_id.clone();
                let progress_detail = model_download_progress_detail(&record, https_error.is_some());
                let mut last_progress_emit = 0_u64;
                let p2p_result = fetch_model_shard_over_libp2p_with_progress(
                    config.clone(),
                    cache_config.clone(),
                    manifest.model_id.clone(),
                    record.info.layers,
                    Some(record.info.checksum.clone()),
                    Some(record.info.version.clone()),
                    Duration::from_millis(model_shard_fetch_timeout_ms(record.info.size_bytes)),
                    move |downloaded, total| {
                        if downloaded == total
                            || downloaded == 0
                            || downloaded.saturating_sub(last_progress_emit)
                                >= MODEL_PROGRESS_EMIT_BYTES
                        {
                            emit_model_import_progress(
                                &progress_app,
                                &progress_model_id,
                                "Downloading shard",
                                progress_detail.clone(),
                                downloaded,
                                Some(total),
                            );
                            last_progress_emit = downloaded;
                        }
                    },
                )
                .await;
                if let Err(p2p_error) = p2p_result {
                    return match https_error.as_deref() {
                        Some(https_error) => Err(anyhow::anyhow!(
                            "Hugging Face download failed ({https_error}); P2P fallback also failed: {p2p_error}"
                        )),
                        None => Err(p2p_error),
                    };
                }
                emit_model_import_progress(
                    &app,
                    &manifest.model_id,
                    "Shard ready",
                    model_download_progress_detail(&record, https_error.is_some()),
                    record.info.size_bytes,
                    Some(record.info.size_bytes),
                );
            }

            emit_model_import_progress(
                &app,
                &manifest.model_id,
                "Ready",
                "Local shards are verified and seeding",
                1,
                Some(1),
            );
            Ok::<(), anyhow::Error>(())
        }
        .await;

        if let Err(error) = result {
            eprintln!(
                "background model shard acquisition for {} failed: {error}",
                manifest.model_id
            );
            emit_model_import_progress(
                &app,
                &manifest.model_id,
                "Download failed",
                error.to_string(),
                0,
                None,
            );
        }

        if let Ok(mut active) = active_model_acquisitions.lock() {
            active.remove(&acquisition_key);
        }
    });

    Ok(())
}

fn model_download_progress_detail(record: &AdvertisedModelRecord, p2p_fallback: bool) -> String {
    let fallback = if p2p_fallback { " · P2P fallback" } else { "" };
    format!(
        "layers {}:{}{}",
        record.info.layers.start, record.info.layers.end, fallback
    )
}

fn model_records_to_download_for_local_contribution(
    cache: &ShardCache,
    model_id: &str,
    local_peer_id: &str,
    advertisements: &[NodeAdvertisement],
    mut plan: Vec<AdvertisedModelRecord>,
) -> anyhow::Result<Vec<AdvertisedModelRecord>> {
    plan.sort_by_key(|record| (record.info.layers.start, record.info.layers.end));
    let local_executable = cache
        .list()?
        .into_iter()
        .filter(|record| record.info.model_id == model_id && is_executable_shard_record(record))
        .map(|record| {
            (
                record.info.layers,
                record.info.checksum,
                record.info.version,
            )
        })
        .collect::<Vec<_>>();
    let missing_records = plan
        .iter()
        .filter(|record| {
            !local_executable.iter().any(|(layers, checksum, version)| {
                *layers == record.info.layers
                    && checksum == &record.info.checksum
                    && version == &record.info.version
            })
        })
        .cloned()
        .collect::<Vec<_>>();

    let components = plan
        .iter()
        .map(|record| FixedModelComponent {
            content_hash: record.info.checksum.clone(),
            layers: record.info.layers,
            weight_bytes: record.info.size_bytes,
        })
        .collect::<Vec<_>>();
    let mut compute_nodes = advertisements.to_vec();
    if !compute_nodes
        .iter()
        .any(|advertisement| advertisement.peer_id == local_peer_id)
    {
        compute_nodes.push(local_capability_advertisement(
            local_peer_id.to_owned(),
            String::new(),
        ));
    }

    // Official Infernet workers execute different layer ranges from the same
    // signed package. Every CUDA/Metal node keeps that verified package on its
    // own disk; inference never redistributes weights between peers.
    if model_id == OFFICIAL_CHAT_MODEL_ID
        && plan.len() == 1
        && plan[0].info.layers.start == 0
        && compute_nodes.iter().any(|advertisement| {
            advertisement.peer_id == local_peer_id
                && advertisement
                    .capabilities
                    .as_ref()
                    .is_some_and(|capabilities| {
                        matches!(capabilities.compute_backend.as_str(), "cuda" | "metal")
                    })
        })
    {
        return Ok(missing_records);
    }

    let config = |minimum_peer_count| CapacityPlanningConfig {
        kv_cache_bytes_per_layer: INFERNET_CHAT_KV_CACHE_BYTES_PER_LAYER,
        scratch_bytes_per_peer: RUNTIME_SCRATCH_BYTES_PER_PEER,
        safety_margin_bytes: CAPACITY_SAFETY_BYTES,
        safety_margin_basis_points: 1_000,
        minimum_peer_count,
    };
    let reported_compute_nodes = compute_nodes
        .iter()
        .filter(|advertisement| {
            if let Some(capabilities) = advertisement.capabilities.as_ref() {
                capabilities.max_sessions > capabilities.active_sessions
                    && (capabilities.available_accelerator_memory_bytes > 0
                        || capabilities.available_ram_bytes > 0)
            } else {
                advertisement
                    .available_vram_bytes
                    .is_some_and(|bytes| bytes > 0)
                    || advertisement
                        .available_ram_bytes
                        .is_some_and(|bytes| bytes > 0)
            }
        })
        .map(|advertisement| advertisement.peer_id.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    let desired_peer_count = components.len().min(reported_compute_nodes).min(8).max(1);
    let mut capacity_plan = None;
    for minimum_peer_count in (2..=desired_peer_count).rev() {
        if let Ok(candidate) =
            plan_fixed_components(&components, &compute_nodes, config(minimum_peer_count))
        {
            capacity_plan = Some(candidate);
            break;
        }
    }
    let capacity_plan = match capacity_plan {
        Some(plan) => plan,
        None => plan_fixed_components(&components, &compute_nodes, config(1))?,
    };
    let local_component_hashes = capacity_plan
        .assignments
        .iter()
        .filter(|assignment| assignment.peer_id == local_peer_id)
        .flat_map(|assignment| assignment.component_hashes.iter().cloned())
        .collect::<BTreeSet<_>>();

    let selected = plan
        .iter()
        .filter(|record| local_component_hashes.contains(&record.info.checksum))
        .filter(|record| {
            missing_records.iter().any(|missing| {
                missing.info.layers == record.info.layers
                    && missing.info.checksum == record.info.checksum
                    && missing.info.version == record.info.version
            })
        })
        .cloned()
        .collect::<Vec<_>>();

    Ok(selected)
}

fn model_acquisition_key(model_id: &str, plan: &[AdvertisedModelRecord]) -> String {
    let mut ranges = plan
        .iter()
        .map(|record| {
            format!(
                "{}:{}:{}:{}",
                record.info.layers.start,
                record.info.layers.end,
                record.info.checksum,
                record.info.version
            )
        })
        .collect::<Vec<_>>();
    ranges.sort();
    format!("{model_id}:{}", ranges.join(","))
}

fn model_shard_fetch_timeout_ms(size_bytes: u64) -> u64 {
    let one_megabyte_per_second = size_bytes
        .saturating_div(1024 * 1024)
        .saturating_mul(1_000)
        .saturating_mul(2);
    DEFAULT_MODEL_FETCH_TIMEOUT_MS.max(one_megabyte_per_second)
}

fn snapshot_from_registry(
    local_peer_id: String,
    topic: String,
    manifest: &ModelManifest,
    registry: &ShardRegistry,
    cache_config: &ShardCacheConfig,
    presence: &PresenceSnapshot,
) -> GridSnapshot {
    let all_advertisements =
        advertisements_with_local_node(registry.advertisements(), cache_config, &local_peer_id);
    let advertisements =
        ui_visible_advertisements(all_advertisements.clone(), Some(&manifest.model_id));
    let route_result = registry.route_for_model(manifest);
    let (route, missing_ranges) = match route_result {
        Ok(route) => (route, None),
        Err(error) => (Vec::new(), Some(error.to_string())),
    };
    let machines = machine_views_from_presence(presence.records(), &local_peer_id);
    let network_peer_count = connected_remote_machine_count(&machines);
    let available_models =
        available_model_views(cache_config, Some(registry), Some(&local_peer_id));

    GridSnapshot {
        local_peer_id,
        topic,
        selected_model: manifest.model_id.clone(),
        available_models,
        layer_count: manifest.layer_count,
        network_peer_count,
        peers: advertisements
            .iter()
            .map(peer_view_from_advertisement)
            .collect(),
        machines,
        route: route.iter().map(route_hop_view).collect(),
        missing_ranges,
        coverage: build_coverage(manifest, &route, &advertisements),
        distribution: build_distribution_snapshot(cache_config, &advertisements),
    }
}

fn empty_snapshot(
    local_peer_id: String,
    topic: String,
    cache_config: &ShardCacheConfig,
    registry: &ShardRegistry,
    presence: &PresenceSnapshot,
) -> GridSnapshot {
    let all_advertisements =
        advertisements_with_local_node(registry.advertisements(), cache_config, &local_peer_id);
    let advertisements = ui_visible_advertisements(all_advertisements.clone(), None);
    let machines = machine_views_from_presence(presence.records(), &local_peer_id);
    let network_peer_count = connected_remote_machine_count(&machines);
    let available_models =
        available_model_views(cache_config, Some(registry), Some(&local_peer_id));
    GridSnapshot {
        local_peer_id,
        topic,
        selected_model: String::new(),
        available_models,
        layer_count: 0,
        network_peer_count,
        peers: advertisements
            .iter()
            .map(peer_view_from_advertisement)
            .collect(),
        machines,
        route: Vec::new(),
        missing_ranges: None,
        coverage: Vec::new(),
        distribution: build_distribution_snapshot(cache_config, &advertisements),
    }
}

fn model_view_from_manifest(
    manifest: &ModelManifest,
    installed: bool,
    network_runnable: bool,
    cache_config: &ShardCacheConfig,
) -> ModelView {
    let runnable = manifest.runtime_kind == RuntimeKind::Demo || network_runnable;
    ModelView {
        model_id: manifest.model_id.clone(),
        display_name: manifest.display_name.clone(),
        runtime_kind: manifest.runtime_kind.as_str().to_owned(),
        layer_count: manifest.layer_count,
        activation_dtype: manifest.activation_dtype.clone(),
        quantization: manifest
            .quantization
            .as_deref()
            .map(normalize_quantization_label),
        installed,
        runnable,
        status: model_status(manifest, installed, runnable, cache_config),
    }
}

fn available_model_views(
    cache_config: &ShardCacheConfig,
    registry: Option<&ShardRegistry>,
    origin_peer_id: Option<&str>,
) -> Vec<ModelView> {
    let installed_ids = installed_model_ids(cache_config);
    let manifest = ModelManifest::infernet_chat_v1();
    let network_runnable =
        registry
            .zip(origin_peer_id)
            .is_some_and(|(registry, origin_peer_id)| {
                plan_worker_execution(registry, &manifest, origin_peer_id).is_ok()
            });
    vec![model_view_from_manifest(
        &manifest,
        installed_ids.contains(&manifest.model_id),
        network_runnable,
        cache_config,
    )]
}

fn installed_model_manifests(cache_config: &ShardCacheConfig) -> Vec<ModelManifest> {
    let Ok(cache) = ShardCache::new(cache_config.clone()) else {
        return Vec::new();
    };
    let Ok(records) = cache.list() else {
        return Vec::new();
    };

    let mut manifests = records
        .into_iter()
        .filter_map(|record| {
            if !is_executable_shard_record(&record) {
                return None;
            }
            let manifest = record.manifest?;
            if !official_record_matches_release(&record.info, &manifest) {
                return None;
            }
            Some(ModelManifest {
                model_id: manifest.model_id,
                display_name: manifest.display_name,
                architecture: manifest.architecture,
                layer_count: manifest.layer_count,
                hidden_size: manifest.hidden_size,
                activation_dtype: manifest.activation_dtype,
                quantization: manifest
                    .metadata
                    .quantization
                    .as_deref()
                    .map(normalize_quantization_label),
                runtime_kind: manifest.runtime_kind,
            })
        })
        .collect::<Vec<_>>();
    manifests.sort_by(|left, right| left.model_id.cmp(&right.model_id));
    manifests.dedup_by(|left, right| left.model_id == right.model_id);
    manifests
}

fn installed_model_ids(cache_config: &ShardCacheConfig) -> Vec<String> {
    let mut ids = installed_model_manifests(cache_config)
        .into_iter()
        .map(|manifest| manifest.model_id)
        .collect::<Vec<_>>();
    ids.sort();
    ids.dedup();
    ids
}

fn ui_visible_advertisements(
    advertisements: Vec<NodeAdvertisement>,
    model_id: Option<&str>,
) -> Vec<NodeAdvertisement> {
    advertisements
        .into_iter()
        .filter_map(|mut advertisement| {
            advertisement.hosted_shards.retain(|shard| {
                shard.model_id == OFFICIAL_CHAT_MODEL_ID
                    && shard.runtime_kind != RuntimeKind::Demo
                    && executable_seed_manifest_for_descriptor(shard).is_some()
                    && model_id.is_none_or(|model_id| shard.model_id == model_id)
            });
            let executable_keys = advertisement
                .hosted_shards
                .iter()
                .map(|shard| (shard.model_id.clone(), shard.layers))
                .collect::<Vec<_>>();
            advertisement.model_shards.retain(|shard| {
                shard.model_id == OFFICIAL_CHAT_MODEL_ID
                    && executable_keys.iter().any(|(model_id, layers)| {
                        model_id == &shard.model_id && *layers == shard.layers
                    })
                    && model_id.is_none_or(|model_id| shard.model_id == model_id)
            });

            if advertisement.hosted_shards.is_empty() && advertisement.model_shards.is_empty() {
                None
            } else {
                Some(advertisement)
            }
        })
        .collect()
}

#[cfg(test)]
fn remote_network_peer_count(local_peer_id: &str, advertisements: &[NodeAdvertisement]) -> usize {
    let bootstrap_peer_ids = default_bootstrap_peer_ids();
    advertisements
        .iter()
        .filter(|advertisement| advertisement.peer_id != local_peer_id)
        .filter(|advertisement| !bootstrap_peer_ids.contains(&advertisement.peer_id))
        .filter(|advertisement| advertisement_has_capacity(advertisement))
        .map(machine_identity_key)
        .collect::<BTreeSet<_>>()
        .len()
}

fn machine_identity_key(advertisement: &NodeAdvertisement) -> String {
    advertisement
        .capabilities
        .as_ref()
        .and_then(|capabilities| capabilities.machine_id.as_deref())
        .filter(|machine_id| !machine_id.trim().is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| format!("peer:{}", advertisement.peer_id))
}

fn advertisement_has_capacity(advertisement: &NodeAdvertisement) -> bool {
    advertisement.capabilities.is_some()
        || advertisement
            .hosted_shards
            .iter()
            .any(|shard| executable_seed_manifest_for_descriptor(shard).is_some())
        || advertisement
            .model_shards
            .iter()
            .any(|shard| executable_seed_manifest_for_model_shard(advertisement, shard).is_some())
}

fn executable_seed_manifest_for_descriptor(
    descriptor: &ShardDescriptor,
) -> Option<&SeedShardManifest> {
    descriptor
        .seed_manifest
        .as_deref()
        .filter(|manifest| seed_manifest_has_executable_payload(manifest))
}

fn executable_seed_manifest_for_model_shard<'a>(
    advertisement: &'a NodeAdvertisement,
    shard: &ModelShardInfo,
) -> Option<&'a SeedShardManifest> {
    advertisement
        .hosted_shards
        .iter()
        .find(|descriptor| {
            descriptor.model_id == shard.model_id && descriptor.layers == shard.layers
        })
        .and_then(executable_seed_manifest_for_descriptor)
}

fn official_record_matches_release(info: &ModelShardInfo, manifest: &SeedShardManifest) -> bool {
    let model = ModelManifest::infernet_chat_v1();
    let release = OfficialModelRelease::infernet_chat_v1_compatibility();
    if release.validate_for_model(&model).is_err()
        || !official_info_matches_release(info)
        || manifest.model_id != model.model_id
        || manifest.display_name != model.display_name
        || manifest.architecture != model.architecture
        || manifest.layer_count != model.layer_count
        || manifest.hidden_size != model.hidden_size
        || manifest.activation_dtype != model.activation_dtype
        || manifest.metadata.quantization.as_deref() != model.quantization.as_deref()
        || manifest.runtime_kind != model.runtime_kind
        || manifest.layers != info.layers
        || manifest.metadata.source_checksum.as_deref()
            != Some(release.upstream.source_sha256.as_str())
        || manifest.source.checksum_sha256 != release.upstream.source_sha256
        || manifest.source.file_size_bytes != release.expected_total_bytes
    {
        return false;
    }

    release.components.iter().any(|component| {
        component.kind == OfficialComponentKind::Transformer
            && component.layers == Some(info.layers)
            && component.sha256 == info.checksum
            && component.size_bytes == info.size_bytes
    })
}

fn official_info_matches_release(info: &ModelShardInfo) -> bool {
    let release = OfficialModelRelease::infernet_chat_v1_compatibility();
    info.model_id == release.model_id
        && info.version == release.version
        && info.protocol_version == PROTOCOL_VERSION
        && release.components.iter().any(|component| {
            component.kind == OfficialComponentKind::Transformer
                && component.layers == Some(info.layers)
                && component.sha256 == info.checksum
                && component.size_bytes == info.size_bytes
        })
}

#[cfg(test)]
fn default_bootstrap_peer_ids() -> Vec<String> {
    DEFAULT_BOOTSTRAP_PEERS
        .iter()
        .filter_map(|address| parse_manual_peer(address).ok())
        .map(|advertisement| advertisement.peer_id)
        .collect()
}

fn model_status(
    manifest: &ModelManifest,
    installed: bool,
    runnable: bool,
    cache_config: &ShardCacheConfig,
) -> String {
    if runnable {
        return "Ready to chat".to_owned();
    }

    if manifest.runtime_kind == RuntimeKind::Demo {
        return "Ready to chat".to_owned();
    }

    if !installed {
        return "Waiting for an online machine to seed the verified release.".to_owned();
    }

    if !local_model_has_complete_coverage(cache_config, manifest) {
        return "This computer does not have a complete verified model package yet.".to_owned();
    }

    "Stored and verified, but no online machine currently reports enough free compute memory."
        .to_owned()
}

fn normalize_quantization_label(value: &str) -> String {
    match value.strip_prefix("gguf_file_type_") {
        Some("0") => "F32",
        Some("1") => "F16",
        Some("2") => "Q4_0",
        Some("3") => "Q4_1",
        Some("7") => "Q8_0",
        Some("8") => "Q5_0",
        Some("9") => "Q5_1",
        Some("10") => "Q2_K",
        Some("11") => "Q3_K_S",
        Some("12") => "Q3_K_M",
        Some("13") => "Q3_K_L",
        Some("14") => "Q4_K_S",
        Some("15") => "Q4_K_M",
        Some("16") => "Q5_K_S",
        Some("17") => "Q5_K_M",
        Some("18") => "Q6_K",
        Some("19") => "IQ2_XXS",
        Some("20") => "IQ2_XS",
        Some("21") => "Q2_K_S",
        Some("22") => "IQ3_XS",
        Some("23") => "IQ3_XXS",
        Some("24") => "IQ1_S",
        Some("25") => "IQ4_NL",
        Some("26") => "IQ3_S",
        Some("27") => "IQ3_M",
        Some("28") => "IQ2_S",
        Some("29") => "IQ2_M",
        Some("30") => "IQ4_XS",
        Some("31") => "IQ1_M",
        Some("32") => "BF16",
        Some("36") => "TQ1_0",
        Some("37") => "TQ2_0",
        Some("38") => "MXFP4_MOE",
        Some("39") => "NVFP4",
        Some("40") => "Q1_0",
        Some("41") => "Q2_0",
        _ => value,
    }
    .to_owned()
}

fn peer_view_from_advertisement(advertisement: &NodeAdvertisement) -> PeerView {
    let mut shards = advertisement
        .hosted_shards
        .iter()
        .map(|shard| ShardView {
            model_id: shard.model_id.clone(),
            layer_start: shard.layers.start,
            layer_end: shard.layers.end,
        })
        .collect::<Vec<_>>();
    for shard in &advertisement.model_shards {
        if shards.iter().any(|existing| {
            existing.model_id == shard.model_id
                && existing.layer_start == shard.layers.start
                && existing.layer_end == shard.layers.end
        }) {
            continue;
        }
        shards.push(ShardView {
            model_id: shard.model_id.clone(),
            layer_start: shard.layers.start,
            layer_end: shard.layers.end,
        });
    }
    shards.sort_by_key(|shard| (shard.model_id.clone(), shard.layer_start, shard.layer_end));

    PeerView {
        peer_id: advertisement.peer_id.clone(),
        short_peer_id: short_peer_id(&advertisement.peer_id),
        addresses: advertisement.addresses.clone(),
        protocol_version: advertisement.protocol_version,
        shards,
    }
}

fn advertisements_with_local_node(
    mut advertisements: Vec<NodeAdvertisement>,
    cache_config: &ShardCacheConfig,
    local_peer_id: &str,
) -> Vec<NodeAdvertisement> {
    let mut local = local_node_advertisement(cache_config, local_peer_id.to_owned());
    if let Some(existing) = advertisements
        .iter()
        .find(|advertisement| advertisement.peer_id == local_peer_id)
    {
        if local.addresses.is_empty() {
            local.addresses = existing.addresses.clone();
        }
    }
    advertisements.retain(|advertisement| advertisement.peer_id != local_peer_id);
    advertisements.push(local);
    advertisements
}

#[cfg(test)]
fn machine_views(advertisements: &[NodeAdvertisement], local_peer_id: &str) -> Vec<MachineView> {
    let records = advertisements
        .iter()
        .cloned()
        .map(|advertisement| PresenceRecord {
            advertisement,
            status: ConnectionStatus::Connected,
            last_seen_age: Duration::ZERO,
        })
        .collect::<Vec<_>>();
    machine_views_from_presence(&records, local_peer_id)
}

fn machine_views_from_presence(
    records: &[PresenceRecord],
    local_peer_id: &str,
) -> Vec<MachineView> {
    let mut by_machine = BTreeMap::<String, Vec<&PresenceRecord>>::new();
    for record in records
        .iter()
        .filter(|record| record.advertisement.capabilities.is_some())
    {
        by_machine
            .entry(machine_identity_key(&record.advertisement))
            .or_default()
            .push(record);
    }

    let mut views = by_machine
        .into_values()
        .filter_map(|mut aliases| {
            aliases.sort_by(|left, right| {
                let left_capabilities = left.advertisement.capabilities.as_ref().unwrap();
                let right_capabilities = right.advertisement.capabilities.as_ref().unwrap();
                let left_local = left.advertisement.peer_id == local_peer_id;
                let right_local = right.advertisement.peer_id == local_peer_id;
                let left_rpc = left.status.is_connected()
                    && worker_is_usable(left_capabilities)
                    && left_capabilities.active_sessions < left_capabilities.max_sessions;
                let right_rpc = right.status.is_connected()
                    && worker_is_usable(right_capabilities)
                    && right_capabilities.active_sessions < right_capabilities.max_sessions;

                right_local
                    .cmp(&left_local)
                    .then_with(|| right.status.priority().cmp(&left.status.priority()))
                    .then_with(|| right_rpc.cmp(&left_rpc))
                    .then_with(|| left.last_seen_age.cmp(&right.last_seen_age))
                    .then_with(|| left.advertisement.peer_id.cmp(&right.advertisement.peer_id))
            });

            let primary = aliases.first()?;
            let advertisement = &primary.advertisement;
            let capabilities = advertisement.capabilities.as_ref()?;
            let is_local = aliases
                .iter()
                .any(|alias| alias.advertisement.peer_id == local_peer_id);
            let connection_status = aliases
                .iter()
                .map(|alias| alias.status)
                .max_by_key(|status| status.priority())
                .unwrap_or(ConnectionStatus::Unreachable);
            let last_seen_age = aliases
                .iter()
                .map(|alias| alias.last_seen_age)
                .min()
                .unwrap_or_default();
            let use_accelerator_memory = capabilities.compute_backend != "cpu"
                && capabilities.total_accelerator_memory_bytes > 0;
            let (total_memory_bytes, available_memory_bytes) = if use_accelerator_memory {
                (
                    capabilities.total_accelerator_memory_bytes,
                    capabilities.available_accelerator_memory_bytes,
                )
            } else {
                (
                    capabilities.total_ram_bytes,
                    capabilities.available_ram_bytes,
                )
            };
            let allocated_memory_bytes = if use_accelerator_memory {
                capabilities
                    .vram_contribution_limit_bytes
                    .unwrap_or(capabilities.total_accelerator_memory_bytes)
                    .min(capabilities.total_accelerator_memory_bytes)
            } else {
                0
            };
            let hosted_components = aliases
                .iter()
                .flat_map(|alias| {
                    alias
                        .advertisement
                        .hosted_shards
                        .iter()
                        .map(|shard| (shard.model_id.clone(), shard.layers.start, shard.layers.end))
                        .chain(alias.advertisement.model_shards.iter().map(|shard| {
                            (shard.model_id.clone(), shard.layers.start, shard.layers.end)
                        }))
                })
                .collect::<BTreeSet<_>>();
            let addresses = aliases
                .iter()
                .flat_map(|alias| alias.advertisement.addresses.iter().cloned())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            let rpc_ready = aliases.iter().any(|alias| {
                let capabilities = alias.advertisement.capabilities.as_ref().unwrap();
                alias.status.is_connected()
                    && worker_is_usable(capabilities)
                    && capabilities.active_sessions < capabilities.max_sessions
            });
            let coarse_location = aliases.iter().find_map(|alias| {
                let assertion = alias.advertisement.coarse_location.as_ref()?;
                let trusted_relays = vec![DEFAULT_RELAY_PEER_ID.to_owned()];
                verify_coarse_location_assertion(
                    assertion,
                    &alias.advertisement.peer_id,
                    &trusted_relays,
                    current_unix_ms_u64(),
                )
                .ok()?;
                let label = [assertion.region.trim(), assertion.country.trim()]
                    .into_iter()
                    .filter(|part| !part.is_empty())
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect::<Vec<_>>()
                    .join(", ");
                Some(CoarseLocationView {
                    latitude: assertion.latitude_e4 as f64 / 10_000.0,
                    longitude: assertion.longitude_e4 as f64 / 10_000.0,
                    label: if label.is_empty() {
                        "Approximate region".to_owned()
                    } else {
                        label
                    },
                })
            });

            Some(MachineView {
                peer_id: advertisement.peer_id.clone(),
                short_peer_id: short_peer_id(&advertisement.peer_id),
                addresses,
                machine_id: capabilities.machine_id.clone(),
                is_local,
                connection_status: if is_local {
                    ConnectionStatus::Connected
                } else {
                    connection_status
                },
                last_seen_seconds: if is_local { 0 } else { last_seen_age.as_secs() },
                compute_backend: capabilities.compute_backend.clone(),
                device_name: capabilities.device_name.clone(),
                logical_cpu_cores: capabilities.logical_cpu_cores,
                total_memory_bytes,
                available_memory_bytes,
                allocated_memory_bytes,
                unified_memory: capabilities.unified_memory,
                max_sessions: capabilities.max_sessions,
                active_sessions: capabilities.active_sessions,
                queue_depth: capabilities.queue_depth,
                measured_prefill_tokens_per_second: capabilities.measured_prefill_tokens_per_second,
                measured_decode_tokens_per_second: capabilities.measured_decode_tokens_per_second,
                hosted_component_count: hosted_components.len(),
                rpc_ready,
                coarse_location,
            })
        })
        .collect::<Vec<_>>();

    views.sort_by(|left, right| {
        right
            .is_local
            .cmp(&left.is_local)
            .then_with(|| left.device_name.cmp(&right.device_name))
            .then_with(|| left.peer_id.cmp(&right.peer_id))
    });
    views
}

fn connected_remote_machine_count(machines: &[MachineView]) -> usize {
    machines
        .iter()
        .filter(|machine| !machine.is_local && machine.connection_status.is_connected())
        .count()
}

fn advertised_model_record_plan(
    registry: &ShardRegistry,
    model_id: &str,
) -> Vec<AdvertisedModelRecord> {
    let mut by_range = BTreeMap::<(u32, u32), AdvertisedModelRecord>::new();
    for advertisement in registry.advertisements() {
        for info in advertisement
            .model_shards
            .iter()
            .filter(|shard| shard.model_id == model_id)
        {
            if !official_info_matches_release(info) {
                continue;
            }
            let record = AdvertisedModelRecord { info: info.clone() };

            by_range
                .entry((info.layers.start, info.layers.end))
                .and_modify(|existing| {
                    if (
                        record.info.version.clone(),
                        record.info.size_bytes,
                        record.info.checksum.clone(),
                    ) < (
                        existing.info.version.clone(),
                        existing.info.size_bytes,
                        existing.info.checksum.clone(),
                    ) {
                        *existing = record.clone();
                    }
                })
                .or_insert(record);
        }
    }

    by_range.into_values().collect()
}

fn missing_ranges_from_layer_ranges(
    layer_count: u32,
    ranges: impl IntoIterator<Item = LayerRange>,
) -> Vec<LayerRange> {
    let mut ranges = ranges
        .into_iter()
        .filter_map(|range| {
            let start = range.start.min(layer_count);
            let end = range.end.min(layer_count);
            (start < end).then_some(LayerRange { start, end })
        })
        .collect::<Vec<_>>();
    ranges.sort_by_key(|range| (range.start, range.end));

    let mut cursor = 0;
    let mut missing = Vec::new();
    for range in ranges {
        if range.end <= cursor {
            continue;
        }
        if range.start > cursor {
            missing.push(LayerRange {
                start: cursor,
                end: range.start,
            });
        }
        cursor = cursor.max(range.end);
    }
    if cursor < layer_count {
        missing.push(LayerRange {
            start: cursor,
            end: layer_count,
        });
    }

    missing
}

fn format_ranges(ranges: &[LayerRange]) -> String {
    ranges
        .iter()
        .map(|range| format!("{}:{}", range.start, range.end))
        .collect::<Vec<_>>()
        .join(", ")
}

fn route_hop_view(hop: &RouteHop) -> RouteHopView {
    RouteHopView {
        peer_id: hop.peer_id.clone(),
        short_peer_id: short_peer_id(&hop.peer_id),
        address: hop.address.clone(),
        layer_start: hop.layers.start,
        layer_end: hop.layers.end,
    }
}

fn build_coverage(
    manifest: &ModelManifest,
    route: &[RouteHop],
    advertisements: &[NodeAdvertisement],
) -> Vec<CoverageSegment> {
    (0..manifest.layer_count)
        .map(|layer| {
            if let Some(hop) = route
                .iter()
                .find(|hop| hop.layers.start <= layer && layer < hop.layers.end)
            {
                return coverage_segment(layer, Some(hop.peer_id.clone()), Some(hop.layers));
            }

            let shard = advertisements.iter().find_map(|advertisement| {
                advertisement
                    .hosted_shards
                    .iter()
                    .find(|shard| {
                        shard.layers.start <= layer
                            && layer < shard.layers.end
                            && shard.model_id == manifest.model_id
                    })
                    .map(|shard| (advertisement.peer_id.clone(), shard.layers))
            });

            match shard {
                Some((peer_id, layers)) => coverage_segment(layer, Some(peer_id), Some(layers)),
                None => coverage_segment(layer, None, None),
            }
        })
        .collect()
}

fn build_distribution_snapshot(
    cache_config: &ShardCacheConfig,
    advertisements: &[NodeAdvertisement],
) -> DistributionSnapshot {
    let serving = model_serving_telemetry();
    let serving_recently = serving
        .last_activity_unix_ms
        .is_some_and(|last| current_unix_ms_u64().saturating_sub(last) <= 5_000);
    let cache = ShardCache::new(cache_config.clone());
    let (installed_shards, storage_used_bytes, max_storage_bytes) = match cache {
        Ok(cache) => {
            let records = cache.list().unwrap_or_default();
            let stats = cache.stats().ok();
            (
                records
                    .iter()
                    .filter_map(|record| {
                        let manifest = record.manifest.as_ref()?;
                        official_record_matches_release(&record.info, manifest).then(|| {
                            InstalledShardView {
                                model_id: record.info.model_id.clone(),
                                layer_start: record.info.layers.start,
                                layer_end: record.info.layers.end,
                                checksum: record.info.checksum.clone(),
                                size_bytes: record.info.size_bytes,
                                version: record.info.version.clone(),
                            }
                        })
                    })
                    .collect::<Vec<_>>(),
                stats
                    .as_ref()
                    .map(|stats| stats.storage_used_bytes)
                    .unwrap_or(0),
                stats
                    .as_ref()
                    .map(|stats| stats.max_storage_bytes)
                    .unwrap_or(0),
            )
        }
        Err(_) => (Vec::new(), 0, 0),
    };
    let mut installed_models = installed_shards
        .iter()
        .map(|shard| shard.model_id.clone())
        .collect::<Vec<_>>();
    installed_models.sort();
    installed_models.dedup();

    let mut replicas = BTreeMap::<(String, u32, u32), usize>::new();
    for shard in advertisements
        .iter()
        .flat_map(|advertisement| advertisement.model_shards.iter())
    {
        *replicas
            .entry((shard.model_id.clone(), shard.layers.start, shard.layers.end))
            .or_default() += 1;
    }
    for shard in &installed_shards {
        replicas
            .entry((shard.model_id.clone(), shard.layer_start, shard.layer_end))
            .or_insert(1);
    }

    DistributionSnapshot {
        installed_models,
        installed_shards,
        storage_used_bytes,
        max_storage_bytes,
        current_uploads: usize::from(serving_recently),
        current_downloads: 0,
        bytes_served: serving.bytes_served,
        chunks_served: serving.chunks_served,
        last_served_unix_ms: serving.last_activity_unix_ms,
        replication_health: replicas
            .into_iter()
            .map(
                |((model_id, layer_start, layer_end), replicas)| ReplicationHealthView {
                    model_id,
                    layer_start,
                    layer_end,
                    replicas,
                    target_replicas: 10,
                },
            )
            .collect(),
    }
}

fn current_unix_ms_u64() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}

fn local_cache_advertisement(
    cache_config: &ShardCacheConfig,
    peer_id: String,
) -> Option<NodeAdvertisement> {
    let cache = ShardCache::new(cache_config.clone()).ok()?;
    let records = cache.list().ok()?;
    let mut hosted_shards = Vec::new();
    let mut model_shards = Vec::new();

    for record in records {
        let Some(manifest) = record.manifest.clone() else {
            continue;
        };
        if manifest.model_id != OFFICIAL_CHAT_MODEL_ID {
            continue;
        }
        if !official_record_matches_release(&record.info, &manifest) {
            continue;
        }
        if is_executable_shard_record(&record) && seed_record_is_executable(cache_config, &manifest)
        {
            let seed_manifest = Box::new(seed_manifest_for_network(&manifest));
            hosted_shards.push(ShardDescriptor {
                model_id: manifest.model_id.clone(),
                layers: manifest.layers,
                runtime_kind: manifest.runtime_kind.clone(),
                resident: persistent_infernet_worker_is_resident(
                    &manifest.model_id,
                    manifest.layers,
                ),
                tokenizer: Some(manifest.tokenizer.clone()),
                metadata: Some(manifest.metadata.clone()),
                shard_hash: Some(manifest.shard_hash.clone()),
                seed_manifest: Some(seed_manifest),
            });
        }
        if is_executable_shard_record(&record) {
            model_shards.push(record.info);
        }
    }

    if hosted_shards.is_empty() && model_shards.is_empty() {
        return None;
    }

    Some(NodeAdvertisement {
        protocol_version: PROTOCOL_VERSION,
        peer_id,
        addresses: Vec::new(),
        available_ram_bytes: None,
        available_vram_bytes: None,
        latency_hint_ms: Some(0),
        capabilities: None,
        hosted_shards,
        model_shards,
        model_components: Vec::new(),
        coarse_location: None,
    })
}

fn local_node_advertisement(cache_config: &ShardCacheConfig, peer_id: String) -> NodeAdvertisement {
    sync_local_image_components(cache_config);
    let advertisement = local_cache_advertisement(cache_config, peer_id.clone())
        .unwrap_or_else(|| empty_advertisement(peer_id, String::new()));
    enrich_local_advertisement(advertisement)
}

fn seed_record_is_executable(config: &ShardCacheConfig, manifest: &SeedShardManifest) -> bool {
    let _ = config;
    manifest.runtime_kind == RuntimeKind::Demo || seed_manifest_has_executable_payload(manifest)
}

fn seed_manifest_has_executable_payload(manifest: &SeedShardManifest) -> bool {
    match manifest.runtime_kind {
        RuntimeKind::Demo => matches!(
            manifest.payload_kind.as_str(),
            PAYLOAD_KIND_GGUF_SHARD | PAYLOAD_KIND_INFERNET_SHARD
        ),
        RuntimeKind::LlamaCpp => {
            manifest.payload_kind == PAYLOAD_KIND_FULL_MODEL
                && manifest.layers.start == 0
                && manifest.layers.end == manifest.layer_count
        }
    }
}

fn local_model_has_complete_coverage(
    cache_config: &ShardCacheConfig,
    model: &ModelManifest,
) -> bool {
    let Ok(cache) = ShardCache::new(cache_config.clone()) else {
        return false;
    };
    let Ok(records) = cache.list() else {
        return false;
    };
    records.into_iter().any(|record| {
        let Some(manifest) = record.manifest.as_ref() else {
            return false;
        };
        is_executable_shard_record(&record)
            && official_record_matches_release(&record.info, manifest)
            && record.info.model_id == model.model_id
            && record.info.layers.start == 0
            && record.info.layers.end == model.layer_count
            && manifest.model_id == model.model_id
            && manifest.layers == record.info.layers
            && manifest.layer_count == model.layer_count
            && manifest.hidden_size == model.hidden_size
            && manifest.architecture == model.architecture
            && manifest.runtime_kind == model.runtime_kind
    })
}

#[cfg(test)]
fn unix_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

fn app_data_dir(app: &AppHandle) -> PathBuf {
    app.path()
        .app_data_dir()
        .unwrap_or_else(|_| PathBuf::from(".infernet"))
}

fn cache_config_for_app(app: &AppHandle) -> ShardCacheConfig {
    // The launch app reads only official release packages. The former `shards`
    // cache is intentionally left untouched so changing product direction
    // never deletes a user's files without consent.
    let mut config = ShardCacheConfig::new(app_data_dir(app).join("official-models").join("v1"));
    config.preferred_models = vec![OFFICIAL_CHAT_MODEL_ID.to_owned()];
    config.pinned_models = vec![OFFICIAL_CHAT_MODEL_ID.to_owned()];
    config
}

#[cfg(test)]
fn migrate_legacy_cache_roots(
    target_config: &ShardCacheConfig,
    legacy_roots: impl IntoIterator<Item = PathBuf>,
) -> anyhow::Result<usize> {
    let target_cache = ShardCache::new(target_config.clone())?;
    let target_root = target_config.root.clone();
    let mut migrated = 0_usize;

    for legacy_root in legacy_roots {
        if same_path(&legacy_root, &target_root) || !legacy_root.join("meta").is_dir() {
            continue;
        }

        let legacy_cache = ShardCache::new(ShardCacheConfig::new(legacy_root))?;
        for record in legacy_cache.list()? {
            if target_cache
                .find(
                    &record.info.model_id,
                    record.info.layers,
                    Some(&record.info.checksum),
                    Some(&record.info.version),
                )?
                .is_some()
            {
                continue;
            }

            let payload = legacy_cache.read_payload(&record.info)?;
            target_cache.store_downloaded(&record.info, payload)?;
            migrated += 1;
        }
    }

    Ok(migrated)
}

#[cfg(test)]
fn same_path(left: &Path, right: &Path) -> bool {
    match (std::fs::canonicalize(left), std::fs::canonicalize(right)) {
        (Ok(left), Ok(right)) => left == right,
        _ => left == right,
    }
}

async fn ensure_model_distribution_service(
    state: &State<'_, UiState>,
    cache_config: ShardCacheConfig,
) -> Result<(), String> {
    let keypair = state
        .keypair
        .lock()
        .map_err(|_| "failed to lock local node identity".to_owned())?
        .clone();
    let local_peer_id = keypair.public().to_peer_id().to_string();
    let mut discovery = DiscoveryConfig::new(state.topic.clone());
    discovery.keypair = keypair;
    discovery.p2p_listen = format!("/ip4/0.0.0.0/tcp/{UI_LISTEN_PORT}");
    discovery.static_peers = configured_static_peers(state)?;
    discovery.relay_peers = default_relay_peer_addresses()?;
    // Desktop discovery is public-bootstrap based. Broadcasting the node over
    // mDNS makes legacy LAN builds repeatedly connect and fail modern
    // Identify/Ping negotiation, producing noisy errors and needless sockets.
    discovery.enable_mdns = false;
    discovery.advertisement = Some(local_node_advertisement(&cache_config, local_peer_id));

    let (waiter_sender, waiter_receiver) = oneshot::channel();
    let should_start = {
        let mut service = state
            .model_distribution_service
            .lock()
            .map_err(|_| "failed to lock model distribution service state".to_owned())?;
        match &mut *service {
            ModelDistributionServiceState::Running => return Ok(()),
            ModelDistributionServiceState::Starting(waiters) => {
                waiters.push(waiter_sender);
                false
            }
            ModelDistributionServiceState::Stopped => {
                *service = ModelDistributionServiceState::Starting(vec![waiter_sender]);
                true
            }
        }
    };

    if should_start {
        spawn_model_distribution_service(
            Arc::clone(&state.model_distribution_service),
            Arc::clone(&state.live_registry),
            discovery,
            cache_config,
        );
    }

    waiter_receiver.await.map_err(|_| {
        "model distribution startup coordinator stopped before reporting readiness".to_owned()
    })?
}

async fn ensure_llama_rpc_service(
    state: &State<'_, UiState>,
    cache_config: &ShardCacheConfig,
) -> Result<(), String> {
    if environment_flag("INFERNET_DISABLE_LLAMA_RPC") || vram_contribution_limit_bytes() == Some(0)
    {
        clear_local_llama_rpc_endpoint();
        return Ok(());
    }

    let (waiter_sender, waiter_receiver) = oneshot::channel();
    let should_start = {
        let mut service = state
            .llama_rpc_service
            .lock()
            .map_err(|_| "failed to lock llama.cpp RPC service state".to_owned())?;

        if let LlamaRpcServiceState::Running(running) = &mut *service {
            if running.server.is_running() {
                return Ok(());
            }
            *service = LlamaRpcServiceState::Stopped;
            clear_local_llama_rpc_endpoint();
        }

        match &mut *service {
            LlamaRpcServiceState::Running(_) => return Ok(()),
            LlamaRpcServiceState::Starting(waiters) => {
                waiters.push(waiter_sender);
                false
            }
            LlamaRpcServiceState::Stopped => {
                *service = LlamaRpcServiceState::Starting(vec![waiter_sender]);
                true
            }
        }
    };

    if should_start {
        clear_local_llama_rpc_endpoint();
        let configuration = llama_rpc_configuration(cache_config);
        let startup = match configuration {
            Ok((server_config, endpoint)) => tauri::async_runtime::spawn_blocking(move || {
                let server =
                    spawn_llama_rpc_server(server_config).map_err(|error| format!("{error:#}"))?;
                set_local_llama_rpc_endpoint(Some(endpoint))?;
                Ok::<_, String>(AdvertisedLlamaRpcServer { server })
            })
            .await
            .map_err(|error| format!("llama.cpp RPC startup task failed: {error}"))
            .and_then(|result| result),
            Err(error) => Err(error),
        };

        let startup_result = startup.as_ref().map(|_| ()).map_err(Clone::clone);
        let waiters = {
            let mut service = state
                .llama_rpc_service
                .lock()
                .map_err(|_| "failed to lock llama.cpp RPC service state".to_owned())?;
            let waiters = match std::mem::replace(&mut *service, LlamaRpcServiceState::Stopped) {
                LlamaRpcServiceState::Starting(waiters) => waiters,
                current => {
                    *service = current;
                    Vec::new()
                }
            };
            if let Ok(server) = startup {
                *service = LlamaRpcServiceState::Running(server);
            } else {
                clear_local_llama_rpc_endpoint();
            }
            waiters
        };

        for waiter in waiters {
            let _ = waiter.send(startup_result.clone());
        }
    }

    waiter_receiver
        .await
        .map_err(|_| "llama.cpp RPC startup coordinator stopped before readiness".to_owned())?
}

async fn ensure_image_rpc_service(
    state: &State<'_, UiState>,
    cache_config: &ShardCacheConfig,
) -> Result<(), String> {
    if environment_flag("INFERNET_DISABLE_IMAGE_RPC") || vram_contribution_limit_bytes() == Some(0)
    {
        clear_local_image_rpc_endpoint();
        return Ok(());
    }

    let (waiter_sender, waiter_receiver) = oneshot::channel();
    let should_start = {
        let mut service = state
            .image_rpc_service
            .lock()
            .map_err(|_| "failed to lock Infernet Image RPC service state".to_owned())?;
        if let ImageRpcServiceState::Running(running) = &mut *service {
            if running.server.is_running() {
                return Ok(());
            }
            *service = ImageRpcServiceState::Stopped;
            clear_local_image_rpc_endpoint();
        }
        match &mut *service {
            ImageRpcServiceState::Running(_) => return Ok(()),
            ImageRpcServiceState::Starting(waiters) => {
                waiters.push(waiter_sender);
                false
            }
            ImageRpcServiceState::Stopped => {
                *service = ImageRpcServiceState::Starting(vec![waiter_sender]);
                true
            }
        }
    };

    if should_start {
        clear_local_image_rpc_endpoint();
        let startup = match image_rpc_configuration(cache_config) {
            Ok((server_config, endpoint)) => tauri::async_runtime::spawn_blocking(move || {
                let server =
                    spawn_llama_rpc_server(server_config).map_err(|error| format!("{error:#}"))?;
                set_local_image_rpc_endpoint(Some(endpoint))?;
                Ok::<_, String>(AdvertisedImageRpcServer { server })
            })
            .await
            .map_err(|error| format!("Infernet Image RPC startup task failed: {error}"))
            .and_then(|result| result),
            Err(error) => Err(error),
        };
        let startup_result = startup.as_ref().map(|_| ()).map_err(Clone::clone);
        let waiters = {
            let mut service = state
                .image_rpc_service
                .lock()
                .map_err(|_| "failed to lock Infernet Image RPC service state".to_owned())?;
            let waiters = match std::mem::replace(&mut *service, ImageRpcServiceState::Stopped) {
                ImageRpcServiceState::Starting(waiters) => waiters,
                current => {
                    *service = current;
                    Vec::new()
                }
            };
            if let Ok(server) = startup {
                *service = ImageRpcServiceState::Running(server);
            } else {
                clear_local_image_rpc_endpoint();
            }
            waiters
        };
        for waiter in waiters {
            let _ = waiter.send(startup_result.clone());
        }
    }

    waiter_receiver
        .await
        .map_err(|_| "Infernet Image RPC startup coordinator stopped before readiness".to_owned())?
}

fn image_rpc_configuration(
    cache_config: &ShardCacheConfig,
) -> Result<(LlamaRpcServerConfig, LlamaRpcEndpoint), String> {
    let binary = find_image_rpc_server_binary().ok_or_else(|| {
        "Infernet Image RPC worker was not found; rebuild the pinned image runtime".to_owned()
    })?;
    let host = Ipv4Addr::LOCALHOST;
    let requested_port = configured_image_rpc_port()?;
    let port = available_rpc_port(
        host,
        requested_port,
        IMAGE_RPC_DEFAULT_PORT,
        "Infernet Image RPC",
    )?;
    let cache_dir = infernet_runtime_dir(cache_config).join("image-rpc");
    let threads = configured_rpc_threads()?;
    let device = env::var("INFERNET_IMAGE_RPC_DEVICE")
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty());
    let backend = rpc_backend_for_device(
        &detect_node_capabilities().compute_backend,
        device.as_deref(),
    )?;
    let endpoint = LlamaRpcEndpoint {
        host: host.to_string(),
        port,
        rpc_protocol_version: LLAMA_RPC_PROTOCOL_VERSION.to_owned(),
        runtime_abi: INFERNET_IMAGE_RPC_RUNTIME_ABI.to_owned(),
        backend: backend.clone(),
        ready: true,
        tunnel_protocol: Some(IMAGE_RPC_TUNNEL_PROTOCOL.to_owned()),
    };
    Ok((
        LlamaRpcServerConfig {
            binary,
            bind_host: host.to_string(),
            advertised_host: host.to_string(),
            port,
            cache_dir,
            cache_tensors: false,
            threads,
            device,
            expected_backend: backend,
            startup_timeout: Duration::from_secs(30),
        },
        endpoint,
    ))
}

fn llama_rpc_configuration(
    cache_config: &ShardCacheConfig,
) -> Result<(LlamaRpcServerConfig, LlamaRpcEndpoint), String> {
    let binary = find_llama_rpc_server_binary().ok_or_else(|| {
        "ggml-rpc-server was not found; rebuild the bundled llama.cpp runtime with GGML_RPC=ON"
            .to_owned()
    })?;
    // ggml-rpc-server is unauthenticated. It is process-local only; remote
    // workers reach it through the authenticated libp2p tunnel.
    let host = Ipv4Addr::LOCALHOST;
    let requested_port = configured_rpc_port()?;
    let port = available_rpc_port(
        host,
        requested_port,
        LLAMA_RPC_DEFAULT_PORT,
        "llama.cpp RPC",
    )?;
    let cache_dir = infernet_runtime_dir(cache_config).join("llama-rpc");
    let threads = configured_rpc_threads()?;
    let device = env::var("INFERNET_LLAMA_RPC_DEVICE")
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty());
    let backend = rpc_backend_for_device(
        &detect_node_capabilities().compute_backend,
        device.as_deref(),
    )?;
    let endpoint = LlamaRpcEndpoint {
        host: host.to_string(),
        port,
        rpc_protocol_version: LLAMA_RPC_PROTOCOL_VERSION.to_owned(),
        runtime_abi: INFERNET_LLAMA_RPC_RUNTIME_ABI.to_owned(),
        backend: backend.clone(),
        ready: true,
        tunnel_protocol: Some(LLAMA_RPC_TUNNEL_PROTOCOL.to_owned()),
    };
    let config = LlamaRpcServerConfig {
        binary,
        bind_host: host.to_string(),
        advertised_host: host.to_string(),
        port,
        cache_dir,
        cache_tensors: false,
        threads,
        device,
        expected_backend: backend,
        startup_timeout: Duration::from_secs(30),
    };
    Ok((config, endpoint))
}

fn infernet_runtime_dir(cache_config: &ShardCacheConfig) -> PathBuf {
    cache_config
        .root
        .parent()
        .and_then(std::path::Path::parent)
        .unwrap_or(&cache_config.root)
        .join("runtime")
}

fn configured_private_rpc_host() -> Result<Ipv4Addr, String> {
    if let Ok(value) = env::var("INFERNET_LLAMA_RPC_HOST") {
        let host = value
            .trim()
            .parse::<Ipv4Addr>()
            .map_err(|_| "INFERNET_LLAMA_RPC_HOST must be a private IPv4 address".to_owned())?;
        return validate_private_rpc_host(host);
    }

    preferred_lan_ipv4().ok_or_else(|| {
        "no private IPv4 address is available for the llama.cpp RPC sidecar".to_owned()
    })
}

fn validate_private_rpc_host(host: Ipv4Addr) -> Result<Ipv4Addr, String> {
    if is_private_or_cgnat_ipv4(host) {
        Ok(host)
    } else {
        Err(format!(
            "refusing to expose unauthenticated llama.cpp RPC on non-private address {host}"
        ))
    }
}

fn is_private_or_cgnat_ipv4(host: Ipv4Addr) -> bool {
    let [first, second, _, _] = host.octets();
    first == 10
        || (first == 172 && (16..=31).contains(&second))
        || (first == 192 && second == 168)
        || (first == 100 && (64..=127).contains(&second))
}

fn preferred_lan_ipv4() -> Option<Ipv4Addr> {
    match preferred_lan_ip()? {
        IpAddr::V4(host) if is_private_or_cgnat_ipv4(host) => Some(host),
        _ => None,
    }
}

fn configured_rpc_port() -> Result<Option<u16>, String> {
    let Ok(value) = env::var("INFERNET_LLAMA_RPC_PORT") else {
        return Ok(None);
    };
    let port = value
        .trim()
        .parse::<u16>()
        .map_err(|_| "INFERNET_LLAMA_RPC_PORT must be between 1 and 65535".to_owned())?;
    if port == 0 {
        return Err("INFERNET_LLAMA_RPC_PORT must be between 1 and 65535".to_owned());
    }
    Ok(Some(port))
}

fn configured_image_rpc_port() -> Result<Option<u16>, String> {
    let Ok(value) = env::var("INFERNET_IMAGE_RPC_PORT") else {
        return Ok(None);
    };
    let port = value
        .trim()
        .parse::<u16>()
        .map_err(|_| "INFERNET_IMAGE_RPC_PORT must be between 1 and 65535".to_owned())?;
    if port == 0 {
        return Err("INFERNET_IMAGE_RPC_PORT must be between 1 and 65535".to_owned());
    }
    Ok(Some(port))
}

fn available_rpc_port(
    host: Ipv4Addr,
    requested: Option<u16>,
    default_port: u16,
    label: &str,
) -> Result<u16, String> {
    let preferred = requested.unwrap_or(default_port);
    match TcpListener::bind((host, preferred)) {
        Ok(listener) => {
            drop(listener);
            Ok(preferred)
        }
        Err(error) if requested.is_some() => Err(format!(
            "configured {label} address {host}:{preferred} is unavailable: {error}"
        )),
        Err(_) => {
            let listener = TcpListener::bind((host, 0)).map_err(|error| {
                format!("could not allocate a private {label} port on {host}: {error}")
            })?;
            let port = listener
                .local_addr()
                .map_err(|error| format!("could not inspect allocated RPC port: {error}"))?
                .port();
            drop(listener);
            Ok(port)
        }
    }
}

fn configured_rpc_threads() -> Result<usize, String> {
    if let Ok(value) = env::var("INFERNET_LLAMA_RPC_THREADS") {
        return value
            .trim()
            .parse::<usize>()
            .ok()
            .filter(|threads| *threads > 0)
            .ok_or_else(|| "INFERNET_LLAMA_RPC_THREADS must be a positive integer".to_owned());
    }

    Ok(std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(2)
        .div_ceil(2)
        .clamp(1, 8))
}

fn rpc_backend_for_device(detected: &str, device: Option<&str>) -> Result<String, String> {
    let backend = match device {
        Some(device) => {
            let device = device.to_ascii_lowercase();
            if device.contains("cuda") {
                "cuda"
            } else if device.contains("metal") {
                "metal"
            } else if device.contains("cpu") {
                "cpu"
            } else {
                return Err(format!(
                    "cannot advertise unknown llama.cpp RPC device {device}"
                ));
            }
        }
        None => detected,
    }
    .to_owned();
    if matches!(backend.as_str(), "cuda" | "metal") {
        Ok(backend)
    } else {
        Err(format!(
            "distributed inference requires a CUDA or Metal backend, got {backend}"
        ))
    }
}

fn environment_flag(name: &str) -> bool {
    env::var(name).is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn monitor_llama_rpc_service(service_state: Arc<Mutex<LlamaRpcServiceState>>) {
    tauri::async_runtime::spawn(async move {
        let mut health_check = tokio::time::interval(Duration::from_secs(1));
        loop {
            health_check.tick().await;
            let (stopped, active) = {
                let mut service = service_state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if let LlamaRpcServiceState::Running(running) = &mut *service {
                    if !running.server.is_running() {
                        *service = LlamaRpcServiceState::Stopped;
                        (true, false)
                    } else {
                        (false, running.server.has_active_client())
                    }
                } else {
                    (false, false)
                }
            };
            set_local_rpc_active(active);
            if stopped {
                clear_local_llama_rpc_endpoint();
                eprintln!("llama.cpp RPC sidecar stopped; compute advertisement was withdrawn");
            }
        }
    });
}

fn spawn_model_distribution_service(
    service_state: Arc<Mutex<ModelDistributionServiceState>>,
    live_registry: Arc<Mutex<ShardRegistry>>,
    discovery: DiscoveryConfig,
    cache_config: ShardCacheConfig,
) {
    tauri::async_runtime::spawn(async move {
        let (readiness_sender, readiness_receiver) = oneshot::channel();
        let registry_observer = Arc::new(move |registry: ShardRegistry| {
            *live_registry
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = registry;
        });
        let mut service_task = Some(tauri::async_runtime::spawn(
            run_model_distribution_node_with_readiness_and_registry(
                discovery,
                cache_config,
                readiness_sender,
                registry_observer,
            ),
        ));

        let startup_result = match tokio::time::timeout(Duration::from_secs(10), readiness_receiver)
            .await
        {
            Ok(Ok(Ok(_address))) => Ok(()),
            Ok(Ok(Err(error))) => Err(error),
            Ok(Err(_)) => {
                let task = service_task
                    .take()
                    .expect("model distribution service task should exist");
                match task.await {
                    Ok(Ok(())) => Err(
                        "model distribution service stopped before its listener became ready"
                            .to_owned(),
                    ),
                    Ok(Err(error)) => Err(format!("{error:#}")),
                    Err(error) => Err(format!(
                        "model distribution service failed before its listener became ready: {error}"
                    )),
                }
            }
            Err(_) => {
                if let Some(task) = service_task.as_ref() {
                    task.abort();
                }
                Err("timed out waiting for the model distribution listener to start".to_owned())
            }
        };

        if startup_result.is_err() {
            if let Some(task) = service_task.take() {
                let _ = task.await;
            }
        }

        let waiters = complete_model_distribution_startup(&service_state, &startup_result);
        for waiter in waiters {
            let _ = waiter.send(startup_result.clone());
        }

        if startup_result.is_ok() {
            if let Some(task) = service_task.take() {
                match task.await {
                    Ok(Ok(())) => {
                        eprintln!("model distribution node stopped unexpectedly");
                    }
                    Ok(Err(error)) => {
                        eprintln!("model distribution node stopped: {error:#}");
                    }
                    Err(error) => {
                        eprintln!("model distribution node task failed: {error}");
                    }
                }
            }
            reset_model_distribution_service(&service_state);
        }
    });
}

fn complete_model_distribution_startup(
    service_state: &Arc<Mutex<ModelDistributionServiceState>>,
    startup_result: &Result<(), String>,
) -> Vec<oneshot::Sender<Result<(), String>>> {
    let mut service = service_state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let waiters = match std::mem::replace(&mut *service, ModelDistributionServiceState::Stopped) {
        ModelDistributionServiceState::Starting(waiters) => waiters,
        current => {
            *service = current;
            return Vec::new();
        }
    };

    if startup_result.is_ok() {
        *service = ModelDistributionServiceState::Running;
    }

    waiters
}

fn reset_model_distribution_service(service_state: &Arc<Mutex<ModelDistributionServiceState>>) {
    let mut service = service_state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if matches!(*service, ModelDistributionServiceState::Running) {
        *service = ModelDistributionServiceState::Stopped;
    }
}

fn parse_manual_peer(input: &str) -> Result<NodeAdvertisement, String> {
    let input = input.trim();
    if input.is_empty() {
        return Err("Paste a peer address from the other computer.".to_owned());
    }

    let (peer_id, address) = if let Some((peer_id, address)) = input.split_once('@') {
        (peer_id.trim().to_owned(), address.trim().to_owned())
    } else if let Some((_, peer_id)) = input.rsplit_once("/p2p/") {
        (peer_id.trim().to_owned(), input.to_owned())
    } else {
        return Err(
            "Peer address must look like /ip4/192.168.1.10/tcp/9777/p2p/12D3...".to_owned(),
        );
    };

    peer_id
        .parse::<PeerId>()
        .map_err(|_| format!("invalid peer id {peer_id}"))?;
    address
        .parse::<Multiaddr>()
        .map_err(|_| format!("invalid peer address {address}"))?;

    Ok(empty_advertisement(peer_id, address))
}

fn manual_peer_addresses(state: &State<'_, UiState>) -> Result<Vec<String>, String> {
    Ok(state
        .manual_peers
        .lock()
        .map_err(|_| "failed to lock manual peers".to_owned())?
        .iter()
        .flat_map(peer_address_labels)
        .collect())
}

fn configured_static_peers(state: &State<'_, UiState>) -> Result<Vec<NodeAdvertisement>, String> {
    let mut by_peer = BTreeMap::<String, NodeAdvertisement>::new();
    for peer in default_bootstrap_peers()?.into_iter().chain(
        state
            .manual_peers
            .lock()
            .map_err(|_| "failed to lock manual peers".to_owned())?
            .clone(),
    ) {
        by_peer
            .entry(peer.peer_id.clone())
            .and_modify(|existing| {
                for address in &peer.addresses {
                    if !existing.addresses.contains(address) {
                        existing.addresses.push(address.clone());
                    }
                }
            })
            .or_insert(peer);
    }

    Ok(by_peer.into_values().collect())
}

fn merge_static_peer_advertisements(
    peers: &mut Vec<NodeAdvertisement>,
    discovered: Vec<NodeAdvertisement>,
) {
    let mut by_peer = BTreeMap::<String, NodeAdvertisement>::new();
    for peer in peers.drain(..).chain(discovered) {
        by_peer
            .entry(peer.peer_id.clone())
            .and_modify(|existing| merge_peer_advertisement(existing, &peer))
            .or_insert(peer);
    }
    *peers = by_peer.into_values().collect();
}

fn merge_peer_advertisement(existing: &mut NodeAdvertisement, peer: &NodeAdvertisement) {
    for address in &peer.addresses {
        if !existing.addresses.contains(address) {
            existing.addresses.push(address.clone());
        }
    }
    for shard in &peer.hosted_shards {
        if let Some(existing_shard) = existing.hosted_shards.iter_mut().find(|existing| {
            existing.model_id == shard.model_id
                && existing.layers == shard.layers
                && existing.runtime_kind == shard.runtime_kind
        }) {
            if existing_shard.seed_manifest.is_none() && shard.seed_manifest.is_some() {
                existing_shard.seed_manifest = shard.seed_manifest.clone();
            }
        } else {
            existing.hosted_shards.push(shard.clone());
        }
    }
    for shard in &peer.model_shards {
        if !existing
            .model_shards
            .iter()
            .any(|existing| existing == shard)
        {
            existing.model_shards.push(shard.clone());
        }
    }
    for component in &peer.model_components {
        if !existing
            .model_components
            .iter()
            .any(|existing| existing == component)
        {
            existing.model_components.push(component.clone());
        }
    }
    if peer.available_ram_bytes.is_some() {
        existing.available_ram_bytes = peer.available_ram_bytes;
    }
    if peer.available_vram_bytes.is_some() {
        existing.available_vram_bytes = peer.available_vram_bytes;
    }
    if peer.latency_hint_ms.is_some() {
        existing.latency_hint_ms = peer.latency_hint_ms;
    }
    if peer.capabilities.is_some() {
        existing.capabilities = peer.capabilities.clone();
    }
    if peer.coarse_location.is_some() {
        existing.coarse_location = peer.coarse_location.clone();
    }
}

fn default_bootstrap_peers() -> Result<Vec<NodeAdvertisement>, String> {
    let mut peers = Vec::new();
    for address in DEFAULT_BOOTSTRAP_PEERS {
        peers.push(parse_manual_peer(address)?);
    }
    if let Ok(addresses) = env::var("INFERNET_BOOTSTRAP_PEERS") {
        for address in addresses
            .split(',')
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            peers.push(parse_manual_peer(address)?);
        }
    }
    Ok(peers)
}

fn default_relay_peer_addresses() -> Result<Vec<String>, String> {
    Ok(default_bootstrap_peers()?
        .into_iter()
        .flat_map(|advertisement| advertisement.addresses)
        .collect())
}

fn remove_relay_servers_from_download_targets(config: &mut DiscoveryConfig) {
    let relay_peer_ids = config
        .relay_peers
        .iter()
        .filter_map(|address| address.parse::<Multiaddr>().ok())
        .filter_map(|address| match address.iter().last() {
            Some(Protocol::P2p(peer_id)) => Some(peer_id.to_string()),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    config
        .static_peers
        .retain(|advertisement| !relay_peer_ids.contains(&advertisement.peer_id));
}

fn prefer_tcp_circuit_addresses_for_downloads(config: &mut DiscoveryConfig) {
    for advertisement in &mut config.static_peers {
        let mut circuit_addresses = advertisement
            .addresses
            .iter()
            .filter(|address| address.contains("/tcp/") && address.contains("/p2p-circuit/"))
            .cloned()
            .collect::<Vec<_>>();
        if circuit_addresses.is_empty() {
            continue;
        }
        circuit_addresses.sort_by_key(|address| !address.starts_with("/ip4/"));
        circuit_addresses.dedup();
        advertisement.addresses = circuit_addresses;
    }
}

fn peer_address_labels(advertisement: &NodeAdvertisement) -> Vec<String> {
    advertisement
        .addresses
        .iter()
        .map(|address| format!("{}@{}", advertisement.peer_id, address))
        .collect()
}

fn local_connect_addresses(peer_id: &str) -> Vec<String> {
    let mut private_ips = private_interface_ipv4s();
    if private_ips.is_empty() {
        private_ips.extend(preferred_lan_ipv4());
    }
    let mut addresses = private_ips
        .into_iter()
        .map(|ip| format!("/ip4/{ip}/tcp/{UI_LISTEN_PORT}/p2p/{peer_id}"))
        .collect::<Vec<_>>();
    // Loopback is useful only when the machine has no private interface. It
    // must not be advertised alongside real addresses to remote peers.
    if addresses.is_empty() {
        addresses.push(format!("/ip4/127.0.0.1/tcp/{UI_LISTEN_PORT}/p2p/{peer_id}"));
    }
    addresses.sort();
    addresses.dedup();
    addresses
}

fn private_interface_ipv4s() -> Vec<Ipv4Addr> {
    let candidates = if_addrs::get_if_addrs()
        .unwrap_or_default()
        .into_iter()
        .filter(|interface| interface.is_oper_up())
        .filter_map(|interface| match interface.ip() {
            IpAddr::V4(ip) => Some(ip),
            IpAddr::V6(_) => None,
        });
    filter_private_interface_ipv4s(candidates)
}

fn filter_private_interface_ipv4s(candidates: impl IntoIterator<Item = Ipv4Addr>) -> Vec<Ipv4Addr> {
    let mut addresses = candidates
        .into_iter()
        .filter(|address| is_private_or_cgnat_ipv4(*address))
        .collect::<Vec<_>>();
    addresses.sort();
    addresses.dedup();
    addresses
}

fn preferred_lan_ip() -> Option<IpAddr> {
    let socket = UdpSocket::bind("0.0.0.0:0").ok()?;
    socket.connect("8.8.8.8:80").ok()?;
    let ip = socket.local_addr().ok()?.ip();
    (!ip.is_loopback()).then_some(ip)
}

fn manifest_for_model(
    model_id: Option<&str>,
    _cache_config: &ShardCacheConfig,
    _registry: Option<&ShardRegistry>,
) -> anyhow::Result<ModelManifest> {
    let requested = model_id.map(str::trim).filter(|value| !value.is_empty());
    if requested.is_none_or(|model_id| model_id == OFFICIAL_CHAT_MODEL_ID) {
        return Ok(ModelManifest::infernet_chat_v1());
    }

    let model_id = requested.unwrap_or_default();
    Err(anyhow::anyhow!(
        "unknown model {model_id}; the launch catalog contains only {OFFICIAL_CHAT_MODEL_ID}"
    ))
}

fn coverage_segment(
    layer: u32,
    peer_id: Option<String>,
    range: Option<LayerRange>,
) -> CoverageSegment {
    CoverageSegment {
        layer,
        covered: peer_id.is_some(),
        peer_id,
        layer_start: range.map(|range| range.start),
        layer_end: range.map(|range| range.end),
    }
}

async fn replay_trace_progress(app: &AppHandle, trace_id: &str, trace: &[TraceEvent]) {
    for event in trace {
        emit_progress(
            app,
            ProgressEvent::HopStarted {
                trace_id: trace_id.to_owned(),
                peer_id: event.peer_id.clone(),
                short_peer_id: short_peer_id(&event.peer_id),
                layer_start: event.layers.start,
                layer_end: event.layers.end,
                activation_size_bytes: event.activation_size_bytes,
            },
        );

        emit_progress(
            app,
            ProgressEvent::HopCompleted {
                trace_id: trace_id.to_owned(),
                peer_id: event.peer_id.clone(),
                short_peer_id: short_peer_id(&event.peer_id),
                layer_start: event.layers.start,
                layer_end: event.layers.end,
                next_peer_id: event.next_peer_id.clone(),
                activation_size_bytes: event.activation_size_bytes,
                timing_ms: event.timing_ms,
                activation_checksum: format!("{:016x}", event.activation_checksum),
            },
        );
    }
}

fn emit_progress(app: &AppHandle, event: ProgressEvent) {
    let _ = app.emit("infernet-progress", event);
}

fn emit_model_import_progress(
    app: &AppHandle,
    model_id: impl Into<String>,
    stage: impl Into<String>,
    detail: impl Into<String>,
    downloaded_bytes: u64,
    total_bytes: Option<u64>,
) {
    let model_id = model_id.into();
    let stage = stage.into();
    let detail = detail.into();
    if let Some(total_bytes) = total_bytes {
        let percent = if total_bytes == 0 {
            0
        } else {
            downloaded_bytes.saturating_mul(100) / total_bytes
        };
        println!(
            "model_download_progress model_id={} stage={} downloaded_bytes={} total_bytes={} percent={}",
            model_id, stage, downloaded_bytes, total_bytes, percent
        );
    }
    let _ = app.emit(
        "infernet-model-import-progress",
        ModelImportProgress {
            model_id,
            stage,
            detail,
            downloaded_bytes,
            total_bytes,
        },
    );
}

fn discovery_config_from_state(
    state: &State<'_, UiState>,
) -> Result<(DiscoveryConfig, String), String> {
    let keypair = state
        .keypair
        .lock()
        .map_err(|_| "failed to lock local node identity".to_owned())?
        .clone();
    let local_peer_id = keypair.public().to_peer_id().to_string();
    let mut config = DiscoveryConfig::new(state.topic.clone());
    config.keypair = keypair;
    config.static_peers = configured_static_peers(state)?;
    config.relay_peers = default_relay_peer_addresses()?;
    // One-shot fetch/inference swarms already have explicit peer and relay
    // addresses. Disabling mDNS prevents Windows interface/socket tasks from
    // accumulating when those short-lived swarms are dropped.
    config.enable_mdns = false;
    config.advertisement = Some(local_capability_advertisement(
        local_peer_id.clone(),
        String::new(),
    ));

    Ok((config, local_peer_id))
}

fn identity_from_state(state: &State<'_, UiState>) -> Result<(String, String), String> {
    let peer_id = state
        .keypair
        .lock()
        .map_err(|_| "failed to lock local node identity".to_owned())?
        .public()
        .to_peer_id()
        .to_string();

    Ok((peer_id, state.topic.clone()))
}

fn short_peer_id(peer_id: &str) -> String {
    if peer_id.len() <= 16 {
        return peer_id.to_owned();
    }

    format!("{}...{}", &peer_id[..8], &peer_id[peer_id.len() - 6..])
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(UiState::default())
        .setup(|app| {
            let app_handle = app.handle().clone();
            let app_data_dir = app.path().app_data_dir()?;
            let app_settings = AppSettingsStore::open(app_data_dir.join("app-settings-v1.json"))?;
            app_settings.apply_saved_limit();
            *app_handle
                .state::<UiState>()
                .app_settings
                .lock()
                .expect("UI app settings lock poisoned during startup") = Some(app_settings);
            match ChatHistoryStore::open(app_data_dir.join("chat-history-v1.json")) {
                Ok(chat_history) => {
                    *app_handle
                        .state::<UiState>()
                        .chat_history
                        .lock()
                        .expect("UI chat history lock poisoned during startup") =
                        Some(chat_history);
                }
                Err(error) => {
                    eprintln!("failed to open chat history: {error}");
                    *app_handle
                        .state::<UiState>()
                        .chat_history_error
                        .lock()
                        .expect("UI chat history error lock poisoned during startup") = Some(error);
                }
            }
            let identity_path = app_data_dir.join("identity.key");
            let keypair = load_or_generate_keypair(&identity_path)?;
            let location_keypair = keypair.clone();
            *app_handle
                .state::<UiState>()
                .keypair
                .lock()
                .expect("UI identity lock poisoned during startup") = keypair;
            tauri::async_runtime::spawn(async move {
                let trusted_relays = vec![DEFAULT_RELAY_PEER_ID.to_owned()];
                loop {
                    match fetch_coarse_location_assertion(
                        COARSE_LOCATION_ENDPOINT,
                        &location_keypair,
                        &trusted_relays,
                    )
                    .await
                    {
                        Ok(assertion) => {
                            set_local_coarse_location(Some(assertion));
                            sleep(Duration::from_secs(3 * 60 * 60)).await;
                        }
                        Err(error) => {
                            eprintln!("failed to refresh relay-signed coarse location: {error:#}");
                            sleep(Duration::from_secs(5 * 60)).await;
                        }
                    }
                }
            });
            let cache_config = cache_config_for_app(app.handle());
            tauri::async_runtime::spawn(async move {
                let state = app_handle.state::<UiState>();
                if let Err(error) = ensure_image_rpc_service(&state, &cache_config).await {
                    eprintln!("failed to start Infernet Image RPC service: {error}");
                }
                if let Err(error) = ensure_llama_rpc_service(&state, &cache_config).await {
                    eprintln!("failed to start GGML RPC service: {error}");
                }
                if let Err(error) =
                    ensure_model_distribution_service(&state, cache_config.clone()).await
                {
                    eprintln!("failed to start model distribution service: {error}");
                }
            });
            Ok(())
        })
        .on_window_event(|window, event| {
            if matches!(event, tauri::WindowEvent::CloseRequested { .. }) {
                stop_persistent_llama_server();
                stop_persistent_rpc_tunnels();
                let state = window.state::<UiState>();
                if let Ok(mut service) = state.llama_rpc_service.lock() {
                    *service = LlamaRpcServiceState::Stopped;
                }
                if let Ok(mut service) = state.image_rpc_service.lock() {
                    *service = ImageRpcServiceState::Stopped;
                }
                clear_local_llama_rpc_endpoint();
                clear_local_image_rpc_endpoint();
                set_local_inference_active(false);
                set_local_rpc_active(false);
            }
        })
        .invoke_handler(tauri::generate_handler![
            get_vram_contribution_settings,
            set_vram_contribution,
            get_local_identity,
            get_local_node_activity,
            get_manual_peers,
            add_manual_peer,
            clear_manual_peers,
            get_chat_history,
            create_chat_thread,
            select_chat_thread,
            append_chat_message,
            delete_chat_thread,
            get_grid_snapshot,
            get_image_runtime_status,
            install_official_image,
            list_generated_images,
            generate_image,
            install_official_model,
            run_demo_inference
        ])
        .run(tauri::generate_context!())
        .expect("error while running Infernet UI");
}

#[cfg(test)]
mod tests {
    use super::*;
    use infernet_protocol::{ImageComponentRole, ModelComponentInfo};

    #[test]
    fn reads_dimensions_from_a_png_header() {
        let mut header = Vec::from(b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR".as_slice());
        header.extend_from_slice(&1024_u32.to_be_bytes());
        header.extend_from_slice(&768_u32.to_be_bytes());

        assert_eq!(png_dimensions(&header), Some((1024, 768)));
        assert_eq!(png_dimensions(b"not a png"), None);
    }

    fn eligible_image_advertisement(peer_id: &str, machine_id: &str) -> NodeAdvertisement {
        const GIB: u64 = 1024 * 1024 * 1024;
        let mut advertisement = local_capability_advertisement(
            peer_id.to_owned(),
            format!("/ip4/192.168.1.20/tcp/9777/p2p/{peer_id}"),
        );
        let capabilities = advertisement.capabilities.as_mut().unwrap();
        capabilities.compute_backend = "metal".to_owned();
        capabilities.machine_id = Some(machine_id.to_owned());
        capabilities.total_accelerator_memory_bytes = 16 * GIB;
        capabilities.available_accelerator_memory_bytes = 12 * GIB;
        capabilities.max_sessions = 1;
        capabilities.active_sessions = 0;
        capabilities.vram_contribution_limit_bytes = None;
        capabilities.image_rpc = Some(LlamaRpcEndpoint {
            host: String::new(),
            port: 0,
            rpc_protocol_version: LLAMA_RPC_PROTOCOL_VERSION.to_owned(),
            runtime_abi: INFERNET_IMAGE_RPC_RUNTIME_ABI.to_owned(),
            backend: "metal".to_owned(),
            ready: true,
            tunnel_protocol: Some(IMAGE_RPC_TUNNEL_PROTOCOL.to_owned()),
        });
        advertisement.available_vram_bytes = Some(12 * GIB);
        advertisement.model_components = image_runtime::component_infos();
        advertisement
    }

    #[test]
    fn image_placement_allows_the_sole_requester_machine() {
        let advertisements = vec![eligible_image_advertisement(
            "requester-peer",
            "requester-machine",
        )];
        let machines = eligible_image_machine_ids(&advertisements, "requester-peer");

        assert_eq!(machines, BTreeSet::from(["requester-machine".to_owned()]));
        assert_eq!(
            plan_image_placement(&machines, "requester-machine").unwrap(),
            StableDiffusionPlacement::RequesterLocal
        );
    }

    #[test]
    fn image_eligibility_does_not_reject_transient_memory_pressure() {
        let mut advertisement = eligible_image_advertisement("requester-peer", "requester-machine");
        advertisement
            .capabilities
            .as_mut()
            .unwrap()
            .available_accelerator_memory_bytes = 0;
        advertisement.available_vram_bytes = Some(0);

        assert!(image_advertisement_is_eligible(&advertisement, "requester-peer").is_some());
    }

    #[test]
    fn image_eligibility_explains_the_actual_failed_gate() {
        let mut advertisement = eligible_image_advertisement("requester-peer", "requester-machine");
        advertisement.model_components.clear();

        let message = local_image_ineligibility_message(
            std::slice::from_ref(&advertisement),
            "requester-peer",
        )
        .unwrap();

        assert!(message.contains("complete verified Infernet Image package"));
    }

    #[test]
    fn image_placement_rejects_a_sole_remote_machine() {
        let advertisements = vec![eligible_image_advertisement(
            "remote-peer",
            "remote-machine",
        )];
        let machines = eligible_image_machine_ids(&advertisements, "requester-peer");

        let error = plan_image_placement(&machines, "requester-machine").unwrap_err();
        assert!(error.contains("one remote computer"));
    }

    #[test]
    fn image_placement_requires_distribution_for_requester_and_remote() {
        let advertisements = vec![
            eligible_image_advertisement("requester-peer", "requester-machine"),
            eligible_image_advertisement("remote-peer", "remote-machine"),
        ];
        let machines = eligible_image_machine_ids(&advertisements, "requester-peer");

        assert_eq!(
            plan_image_placement(&machines, "requester-machine").unwrap(),
            StableDiffusionPlacement::Distributed { machine_count: 2 }
        );
    }

    #[test]
    fn image_placement_rejects_two_remotes_without_the_requester() {
        let advertisements = vec![
            eligible_image_advertisement("remote-a", "remote-machine-a"),
            eligible_image_advertisement("remote-b", "remote-machine-b"),
        ];
        let machines = eligible_image_machine_ids(&advertisements, "requester-peer");

        let error = plan_image_placement(&machines, "requester-machine").unwrap_err();
        assert!(error.contains("must participate"));
    }

    #[test]
    fn image_worker_selection_uses_every_remote_machine_once() {
        const GIB: u64 = 1024 * 1024 * 1024;
        let mut smaller = eligible_image_advertisement("remote-small", "remote-machine-a");
        smaller
            .capabilities
            .as_mut()
            .unwrap()
            .available_accelerator_memory_bytes = 6 * GIB;
        let mut duplicate = eligible_image_advertisement("remote-duplicate", "remote-machine-a");
        duplicate
            .capabilities
            .as_mut()
            .unwrap()
            .available_accelerator_memory_bytes = 5 * GIB;
        let mut larger = eligible_image_advertisement("remote-large", "remote-machine-b");
        larger
            .capabilities
            .as_mut()
            .unwrap()
            .available_accelerator_memory_bytes = 10 * GIB;
        let advertisements = vec![
            eligible_image_advertisement("requester-peer", "requester-machine"),
            larger,
            duplicate,
            smaller,
        ];

        assert_eq!(
            select_image_rpc_workers(&advertisements, "requester-peer", "requester-machine"),
            vec!["remote-small".to_owned(), "remote-large".to_owned()]
        );
    }

    #[test]
    fn image_placement_does_not_count_two_peers_on_one_remote_machine() {
        let advertisements = vec![
            eligible_image_advertisement("remote-peer-a", "shared-remote-machine"),
            eligible_image_advertisement("remote-peer-b", "shared-remote-machine"),
        ];
        let machines = eligible_image_machine_ids(&advertisements, "requester-peer");

        assert_eq!(
            machines,
            BTreeSet::from(["shared-remote-machine".to_owned()])
        );
        assert!(plan_image_placement(&machines, "requester-machine").is_err());
    }

    #[test]
    fn discovered_advertisement_enriches_a_manual_peer_placeholder() {
        let mut placeholder = empty_advertisement(
            "manual-peer".to_owned(),
            "/ip4/10.0.0.2/tcp/9777".to_owned(),
        );
        let mut discovered = local_capability_advertisement(
            "manual-peer".to_owned(),
            "/ip4/10.0.0.3/tcp/9777".to_owned(),
        );
        discovered.capabilities.as_mut().unwrap().machine_id = Some("machine-a".to_owned());
        discovered.available_ram_bytes = Some(8);
        discovered.available_vram_bytes = Some(4);
        discovered.latency_hint_ms = Some(3);
        let component = ModelComponentInfo {
            release_id: "image-release".to_owned(),
            model_id: "image-model".to_owned(),
            component_id: "vae".to_owned(),
            role: ImageComponentRole::Vae,
            checksum: "checksum".to_owned(),
            size_bytes: 42,
            version: "1".to_owned(),
            runtime_abi: "stable-diffusion.cpp-v1".to_owned(),
        };
        discovered.model_components.push(component.clone());

        merge_peer_advertisement(&mut placeholder, &discovered);

        assert_eq!(
            placeholder
                .capabilities
                .as_ref()
                .and_then(|capabilities| capabilities.machine_id.as_deref()),
            Some("machine-a")
        );
        assert_eq!(placeholder.available_ram_bytes, Some(8));
        assert_eq!(placeholder.available_vram_bytes, Some(4));
        assert_eq!(placeholder.latency_hint_ms, Some(3));
        assert_eq!(placeholder.model_components, vec![component]);
        assert_eq!(placeholder.addresses.len(), 2);
    }

    #[test]
    fn rpc_binding_rejects_public_loopback_and_wildcard_addresses() {
        assert!(is_private_or_cgnat_ipv4(Ipv4Addr::new(10, 1, 2, 3)));
        assert!(is_private_or_cgnat_ipv4(Ipv4Addr::new(100, 100, 2, 3)));
        assert!(!is_private_or_cgnat_ipv4(Ipv4Addr::UNSPECIFIED));
        assert!(!is_private_or_cgnat_ipv4(Ipv4Addr::LOCALHOST));
        assert!(!is_private_or_cgnat_ipv4(Ipv4Addr::new(8, 8, 8, 8)));
        assert!(validate_private_rpc_host(Ipv4Addr::new(192, 168, 1, 2)).is_ok());
        assert!(validate_private_rpc_host(Ipv4Addr::new(217, 77, 11, 197)).is_err());
        assert!(rpc_backend_for_device("metal", Some("CPU")).is_err());
        assert!(rpc_backend_for_device("cpu", Some("Vulkan0")).is_err());
    }

    #[test]
    fn connect_addresses_include_only_unique_private_ipv4_interfaces() {
        let addresses = filter_private_interface_ipv4s([
            Ipv4Addr::new(192, 168, 1, 20),
            Ipv4Addr::new(10, 0, 0, 7),
            Ipv4Addr::new(100, 76, 1, 2),
            Ipv4Addr::new(192, 168, 1, 20),
            Ipv4Addr::UNSPECIFIED,
            Ipv4Addr::LOCALHOST,
            Ipv4Addr::new(169, 254, 10, 20),
            Ipv4Addr::new(8, 8, 8, 8),
        ]);

        assert_eq!(
            addresses,
            vec![
                Ipv4Addr::new(10, 0, 0, 7),
                Ipv4Addr::new(100, 76, 1, 2),
                Ipv4Addr::new(192, 168, 1, 20),
            ]
        );
    }

    #[test]
    fn empty_cache_publishes_only_the_official_launch_model() {
        let root = std::env::temp_dir().join(format!("infernet-ui-empty-{}", unix_ms()));
        let cache_config = ShardCacheConfig::new(root.clone());

        let models = available_model_views(&cache_config, None, None);
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].model_id, OFFICIAL_CHAT_MODEL_ID);
        assert!(!models[0].installed);
        assert_eq!(
            manifest_for_model(None, &cache_config, None)
                .unwrap()
                .model_id,
            OFFICIAL_CHAT_MODEL_ID
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn remote_peer_count_ignores_connection_only_advertisements() {
        let connection_only = empty_advertisement("remote-connection".to_owned(), String::new());
        let mut capacity = empty_advertisement("remote-capacity".to_owned(), String::new());
        capacity.model_shards.push(ModelShardInfo {
            model_id: "gemma".to_owned(),
            layers: LayerRange::new(0, 8).unwrap(),
            checksum: "checksum".to_owned(),
            size_bytes: 8,
            version: "v1".to_owned(),
            protocol_version: PROTOCOL_VERSION,
        });

        assert_eq!(
            remote_network_peer_count("local-peer", &[connection_only, capacity]),
            0
        );

        let connection_only = empty_advertisement("remote-connection".to_owned(), String::new());
        let mut capacity = empty_advertisement("remote-capacity".to_owned(), String::new());
        let layers = LayerRange::new(0, 8).unwrap();
        capacity.model_shards.push(ModelShardInfo {
            model_id: "gemma".to_owned(),
            layers,
            checksum: "checksum".to_owned(),
            size_bytes: 8,
            version: "v1".to_owned(),
            protocol_version: PROTOCOL_VERSION,
        });
        capacity.hosted_shards.push(ShardDescriptor {
            model_id: "gemma".to_owned(),
            layers,
            runtime_kind: RuntimeKind::LlamaCpp,
            resident: false,
            tokenizer: None,
            metadata: None,
            shard_hash: None,
            seed_manifest: Some(Box::new(SeedShardManifest {
                model_id: "gemma".to_owned(),
                display_name: "Gemma".to_owned(),
                architecture: "gemma3".to_owned(),
                layer_count: 8,
                hidden_size: 16,
                activation_dtype: "f16".to_owned(),
                runtime_kind: RuntimeKind::LlamaCpp,
                layers,
                tokenizer: infernet_model::TokenizerCompatibility {
                    family: "gemma3".to_owned(),
                    checksum: None,
                },
                metadata: infernet_model::ShardMetadata {
                    architecture: "gemma3".to_owned(),
                    quantization: Some("IQ4_XS".to_owned()),
                    source_checksum: Some("source".to_owned()),
                    protocol_version: PROTOCOL_VERSION,
                },
                source: infernet_model::SeedSourceMetadata {
                    path: "/tmp/gemma.gguf".to_owned(),
                    checksum_sha256: "source".to_owned(),
                    file_size_bytes: 8,
                },
                shard_hash: "hash".to_owned(),
                payload_kind: PAYLOAD_KIND_FULL_MODEL.to_owned(),
            })),
        });

        assert_eq!(
            remote_network_peer_count("local-peer", &[connection_only, capacity]),
            1
        );

        let capability_only =
            local_capability_advertisement("remote-machine".to_owned(), String::new());
        assert_eq!(
            remote_network_peer_count("local-peer", &[capability_only]),
            1,
            "a compute node does not need a model component yet to count as online"
        );
    }

    #[test]
    fn empty_local_cache_still_reports_this_machine() {
        let root = std::env::temp_dir().join(format!("infernet-ui-capacity-{}", unix_ms()));
        let cache_config = ShardCacheConfig::new(root.clone());
        let advertisement = local_node_advertisement(&cache_config, "local-peer".to_owned());
        let machines = machine_views(&[advertisement], "local-peer");

        assert_eq!(machines.len(), 1);
        assert!(machines[0].is_local);
        assert!(!machines[0].compute_backend.is_empty());
        assert!(!machines[0].device_name.is_empty());
        assert!(machines[0].max_sessions > 0);

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn machine_view_reports_configured_accelerator_allocation_not_free_memory() {
        const GIB: u64 = 1024 * 1024 * 1024;
        let mut advertisement = eligible_image_advertisement("peer-a", "machine-a");
        advertisement
            .capabilities
            .as_mut()
            .unwrap()
            .vram_contribution_limit_bytes = Some(8 * GIB);

        let machines = machine_views(&[advertisement], "local-peer");

        assert_eq!(machines[0].total_memory_bytes, 16 * GIB);
        assert_eq!(machines[0].available_memory_bytes, 12 * GIB);
        assert_eq!(machines[0].allocated_memory_bytes, 8 * GIB);
    }

    #[test]
    fn two_peer_identities_for_one_physical_machine_render_once() {
        let mut lan = local_capability_advertisement("peer-lan".to_owned(), String::new());
        let mut secondary =
            local_capability_advertisement("peer-secondary".to_owned(), String::new());
        lan.capabilities.as_mut().unwrap().machine_id = Some("machine-a".to_owned());
        secondary.capabilities.as_mut().unwrap().machine_id = Some("machine-a".to_owned());

        let machines = machine_views(&[lan.clone(), secondary.clone()], "local-peer");

        assert_eq!(machines.len(), 1);
        assert_eq!(machines[0].machine_id.as_deref(), Some("machine-a"));
        assert_eq!(
            remote_network_peer_count("local-peer", &[lan, secondary]),
            1
        );
    }

    #[test]
    fn available_models_ignore_orphan_model_shards() {
        let root = std::env::temp_dir().join(format!("infernet-ui-orphan-{}", unix_ms()));
        let cache_config = ShardCacheConfig::new(root.clone());
        let mut registry = ShardRegistry::new();
        let mut orphan = empty_advertisement("remote-orphan".to_owned(), String::new());
        orphan.model_shards.push(ModelShardInfo {
            model_id: "gemma".to_owned(),
            layers: LayerRange::new(0, 8).unwrap(),
            checksum: "checksum".to_owned(),
            size_bytes: 8,
            version: "v1".to_owned(),
            protocol_version: PROTOCOL_VERSION,
        });
        registry.upsert(orphan);

        assert_eq!(
            available_model_views(&cache_config, Some(&registry), Some("local-peer")).len(),
            1
        );
        assert!(manifest_for_model(Some("gemma"), &cache_config, Some(&registry)).is_err());

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn launch_registry_accepts_only_the_pinned_release_bytes() {
        let release = OfficialModelRelease::infernet_chat_v1_compatibility();
        let model = ModelManifest::infernet_chat_v1();
        let component = &release.components[0];
        let layers = component.layers.unwrap();
        let seed_manifest = SeedShardManifest {
            model_id: model.model_id.clone(),
            display_name: model.display_name.clone(),
            architecture: model.architecture.clone(),
            layer_count: model.layer_count,
            hidden_size: model.hidden_size,
            activation_dtype: model.activation_dtype.clone(),
            runtime_kind: model.runtime_kind.clone(),
            layers,
            tokenizer: infernet_model::TokenizerCompatibility {
                family: model.architecture.clone(),
                checksum: None,
            },
            metadata: infernet_model::ShardMetadata {
                architecture: model.architecture.clone(),
                quantization: model.quantization.clone(),
                source_checksum: Some(release.upstream.source_sha256.clone()),
                protocol_version: PROTOCOL_VERSION,
            },
            source: infernet_model::SeedSourceMetadata {
                path: String::new(),
                checksum_sha256: release.upstream.source_sha256.clone(),
                file_size_bytes: release.expected_total_bytes,
            },
            shard_hash: "release-manifest-hash".to_owned(),
            payload_kind: PAYLOAD_KIND_FULL_MODEL.to_owned(),
        };
        let info = ModelShardInfo {
            model_id: model.model_id.clone(),
            layers,
            checksum: component.sha256.clone(),
            size_bytes: component.size_bytes,
            version: release.version.clone(),
            protocol_version: PROTOCOL_VERSION,
        };
        let mut valid = empty_advertisement("valid-peer".to_owned(), String::new());
        valid.available_vram_bytes = Some(24 * 1024 * 1024 * 1024);
        valid.model_shards.push(info.clone());
        valid.hosted_shards.push(ShardDescriptor {
            model_id: model.model_id.clone(),
            layers,
            runtime_kind: model.runtime_kind.clone(),
            resident: false,
            tokenizer: Some(seed_manifest.tokenizer.clone()),
            metadata: Some(seed_manifest.metadata.clone()),
            shard_hash: Some(seed_manifest.shard_hash.clone()),
            seed_manifest: Some(Box::new(seed_manifest.clone())),
        });
        let mut forged = valid.clone();
        forged.peer_id = "forged-peer".to_owned();
        forged.model_shards[0].checksum = "0".repeat(64);
        let mut too_small = valid.clone();
        too_small.peer_id = "small-peer".to_owned();
        too_small.available_vram_bytes = Some(8 * 1024 * 1024 * 1024);

        let mut registry = ShardRegistry::new();
        registry.upsert(valid);
        registry.upsert(forged);
        registry.upsert(too_small);
        let trusted = trusted_launch_registry(registry, "local-peer");
        let advertisements = trusted.advertisements();
        let route = trusted.route_for_model(&model).unwrap();

        assert_eq!(route.len(), 1);
        assert_eq!(route[0].peer_id, "small-peer");
        assert_eq!(
            advertisements
                .iter()
                .find(|advertisement| advertisement.peer_id == "valid-peer")
                .unwrap()
                .model_shards
                .len(),
            1
        );
        let forged = advertisements
            .iter()
            .find(|advertisement| advertisement.peer_id == "forged-peer")
            .unwrap();
        assert!(forged.model_shards.is_empty());
        assert!(forged.hosted_shards.is_empty());
        let too_small = advertisements
            .iter()
            .find(|advertisement| advertisement.peer_id == "small-peer")
            .unwrap();
        assert_eq!(too_small.model_shards.len(), 1);
        assert_eq!(
            too_small.hosted_shards.len(),
            1,
            "verified storage availability must not depend on one host fitting the full distributed model"
        );
    }

    #[test]
    fn legacy_cache_is_migrated_to_app_cache() {
        let root = std::env::temp_dir().join(format!("infernet-ui-migrate-{}", unix_ms()));
        let legacy_root = root.join("legacy").join(".infernet").join("shards");
        let target_root = root.join("app-data").join("shards");
        let legacy_config = ShardCacheConfig::new(legacy_root.clone());
        let target_config = ShardCacheConfig::new(target_root.clone());
        let range = LayerRange::new(0, 8).unwrap();

        let legacy_cache = ShardCache::new(legacy_config).unwrap();
        let legacy_record = legacy_cache
            .import_payload(
                b"legacy gemma metadata".to_vec(),
                "gemma-4-12b-it-iq4-xs",
                range,
                "v1",
            )
            .unwrap();

        let migrated =
            migrate_legacy_cache_roots(&target_config, vec![legacy_root.clone()]).unwrap();
        assert_eq!(migrated, 1);

        let target_cache = ShardCache::new(target_config).unwrap();
        let migrated_record = target_cache
            .find(
                "gemma-4-12b-it-iq4-xs",
                range,
                Some(&legacy_record.info.checksum),
                Some("v1"),
            )
            .unwrap()
            .expect("migrated shard should exist in app cache");

        assert!(migrated_record.path.starts_with(&target_root));
        assert_eq!(
            target_cache.read_payload(&legacy_record.info).unwrap(),
            b"legacy gemma metadata"
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn launch_app_does_not_advertise_or_select_legacy_gguf_records() {
        let root = std::env::temp_dir().join(format!("infernet-ui-local-gguf-{}", unix_ms()));
        let cache_config = ShardCacheConfig::new(root.join("shards"));
        let cache = ShardCache::new(cache_config.clone()).unwrap();
        let source = root.join("gemma-4-12b-it-IQ4_XS.gguf");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(&source, b"local physical gguf shard placeholder").unwrap();

        let layers = LayerRange::new(0, 48).unwrap();
        let seed_manifest = SeedShardManifest {
            model_id: "gemma-4-12b-it-iq4-xs".to_owned(),
            display_name: "Gemma 4 12B IT IQ4 XS".to_owned(),
            architecture: "gemma4".to_owned(),
            layer_count: 48,
            hidden_size: 3840,
            activation_dtype: "f16".to_owned(),
            runtime_kind: RuntimeKind::LlamaCpp,
            layers,
            tokenizer: infernet_model::TokenizerCompatibility {
                family: "gemma4".to_owned(),
                checksum: None,
            },
            metadata: infernet_model::ShardMetadata {
                architecture: "gemma4".to_owned(),
                quantization: Some("gguf_file_type_30".to_owned()),
                source_checksum: Some("checksum".to_owned()),
                protocol_version: PROTOCOL_VERSION,
            },
            source: infernet_model::SeedSourceMetadata {
                path: source.display().to_string(),
                checksum_sha256: "checksum".to_owned(),
                file_size_bytes: 123,
            },
            shard_hash: "hash".to_owned(),
            payload_kind: PAYLOAD_KIND_FULL_MODEL.to_owned(),
        };
        cache
            .import_physical_shard_file(
                &source,
                seed_manifest.model_id.clone(),
                layers,
                "v1",
                seed_manifest,
            )
            .unwrap();

        assert!(local_cache_advertisement(&cache_config, "local-peer".to_owned()).is_none());
        assert!(manifest_for_model(Some("gemma-4-12b-it-iq4-xs"), &cache_config, None).is_err());
        let views = available_model_views(&cache_config, None, None);
        assert_eq!(views.len(), 1);
        assert_eq!(views[0].model_id, OFFICIAL_CHAT_MODEL_ID);
        assert!(!views[0].installed);

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn physical_shard_record_keeps_manifest_and_payload() {
        let root = std::env::temp_dir().join(format!("infernet-ui-advertised-{}", unix_ms()));
        let cache_config = ShardCacheConfig::new(root.join("shards"));
        let cache = ShardCache::new(cache_config.clone()).unwrap();
        std::fs::create_dir_all(&root).unwrap();
        let shard_file = root.join("remote-shard.gguf");
        let shard_bytes = b"infernet shard payload";
        std::fs::write(&shard_file, shard_bytes).unwrap();
        let layers = LayerRange::new(0, 8).unwrap();
        let seed_manifest = SeedShardManifest {
            model_id: "gemma-4-12b-it-iq4-xs".to_owned(),
            display_name: "Gemma 4 12B IT IQ4 XS".to_owned(),
            architecture: "gemma4".to_owned(),
            layer_count: 48,
            hidden_size: 3840,
            activation_dtype: "f16".to_owned(),
            runtime_kind: RuntimeKind::LlamaCpp,
            layers,
            tokenizer: infernet_model::TokenizerCompatibility {
                family: "gemma4".to_owned(),
                checksum: None,
            },
            metadata: infernet_model::ShardMetadata {
                architecture: "gemma4".to_owned(),
                quantization: Some("IQ4_XS".to_owned()),
                source_checksum: Some("source-checksum".to_owned()),
                protocol_version: PROTOCOL_VERSION,
            },
            source: infernet_model::SeedSourceMetadata {
                path: "/remote/gemma.gguf".to_owned(),
                checksum_sha256: "source-checksum".to_owned(),
                file_size_bytes: 12_345,
            },
            shard_hash: "seed-hash".to_owned(),
            payload_kind: PAYLOAD_KIND_INFERNET_SHARD.to_owned(),
        };

        let record = cache
            .import_physical_shard_file(
                &shard_file,
                seed_manifest.model_id.clone(),
                layers,
                "v1",
                seed_manifest.clone(),
            )
            .unwrap();

        let installed = cache
            .find(
                &record.info.model_id,
                layers,
                Some(&record.info.checksum),
                Some(&record.info.version),
            )
            .unwrap()
            .expect("advertised record should be installed");
        assert_eq!(installed.manifest, Some(seed_manifest));
        assert_eq!(cache.read_payload(&installed.info).unwrap(), shard_bytes);

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn local_contribution_planner_replaces_legacy_partial_with_complete_package() {
        let root = std::env::temp_dir().join(format!("infernet-ui-partial-{}", unix_ms()));
        let cache_config = ShardCacheConfig::new(root.join("shards"));
        let cache = ShardCache::new(cache_config.clone()).unwrap();
        std::fs::create_dir_all(&root).unwrap();
        let shard_file = root.join("local-shard.gguf");
        std::fs::write(&shard_file, b"local physical gguf shard placeholder").unwrap();
        let installed_layers = LayerRange::new(0, 8).unwrap();
        let complete_layers = LayerRange::new(0, 16).unwrap();
        let seed_manifest = SeedShardManifest {
            model_id: "gemma".to_owned(),
            display_name: "Gemma".to_owned(),
            architecture: "gemma".to_owned(),
            layer_count: 16,
            hidden_size: 1024,
            activation_dtype: "f16".to_owned(),
            runtime_kind: RuntimeKind::LlamaCpp,
            layers: installed_layers,
            tokenizer: infernet_model::TokenizerCompatibility {
                family: "gemma".to_owned(),
                checksum: None,
            },
            metadata: infernet_model::ShardMetadata {
                architecture: "gemma".to_owned(),
                quantization: Some("IQ4_XS".to_owned()),
                source_checksum: Some("source".to_owned()),
                protocol_version: PROTOCOL_VERSION,
            },
            source: infernet_model::SeedSourceMetadata {
                path: "/tmp/gemma.gguf".to_owned(),
                checksum_sha256: "source".to_owned(),
                file_size_bytes: 16,
            },
            shard_hash: "hash".to_owned(),
            payload_kind: PAYLOAD_KIND_INFERNET_SHARD.to_owned(),
        };
        let installed = cache
            .import_physical_shard_file(
                &shard_file,
                "gemma".to_owned(),
                installed_layers,
                "v1",
                seed_manifest,
            )
            .unwrap();
        let complete = ModelShardInfo {
            model_id: "gemma".to_owned(),
            layers: complete_layers,
            checksum: "complete-checksum".to_owned(),
            size_bytes: 16,
            version: "v1".to_owned(),
            protocol_version: PROTOCOL_VERSION,
        };
        let mut local_advertisement = empty_advertisement("local-peer".to_owned(), String::new());
        local_advertisement.available_ram_bytes = Some(8 * 1024 * 1024 * 1024);
        let advertisements = vec![local_advertisement];
        let selected = model_records_to_download_for_local_contribution(
            &cache,
            "gemma",
            "local-peer",
            &advertisements,
            vec![AdvertisedModelRecord { info: complete }],
        )
        .unwrap();

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].info.layers, complete_layers);
        assert!(!is_executable_shard_record(&installed));
        assert!(
            available_model_views(&cache_config, None, None)
                .into_iter()
                .all(|model| model.model_id != "gemma"),
            "unsupported legacy partials must not appear runnable or advertised"
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn every_accelerated_worker_downloads_the_verified_full_package() {
        const GIB: u64 = 1024 * 1024 * 1024;
        let root = std::env::temp_dir().join(format!("infernet-ui-host-plan-{}", unix_ms()));
        let cache = ShardCache::new(ShardCacheConfig::new(root.clone())).unwrap();
        let layers = LayerRange::new(0, 46).unwrap();
        let package = AdvertisedModelRecord {
            info: ModelShardInfo {
                model_id: OFFICIAL_CHAT_MODEL_ID.to_owned(),
                layers,
                checksum: "full-package".to_owned(),
                size_bytes: 14_439_361_440,
                version: "1.0.0-compat.1".to_owned(),
                protocol_version: PROTOCOL_VERSION,
            },
        };

        let machine = |peer_id: &str,
                       machine_id: &str,
                       backend: &str,
                       device: &str,
                       total_accelerator_memory_bytes: u64,
                       available_accelerator_memory_bytes: u64,
                       unified_memory: bool| {
            let mut advertisement =
                local_capability_advertisement(peer_id.to_owned(), String::new());
            let capabilities = advertisement.capabilities.as_mut().unwrap();
            capabilities.machine_id = Some(machine_id.to_owned());
            capabilities.compute_backend = backend.to_owned();
            capabilities.device_name = device.to_owned();
            capabilities.total_ram_bytes = if unified_memory { 16 * GIB } else { 32 * GIB };
            capabilities.available_ram_bytes = if unified_memory { 12 * GIB } else { 24 * GIB };
            capabilities.total_accelerator_memory_bytes = total_accelerator_memory_bytes;
            capabilities.available_accelerator_memory_bytes = available_accelerator_memory_bytes;
            capabilities.unified_memory = unified_memory;
            capabilities.max_sessions = 1;
            capabilities.active_sessions = 0;
            advertisement.available_ram_bytes = Some(capabilities.available_ram_bytes);
            advertisement.available_vram_bytes =
                Some(capabilities.available_accelerator_memory_bytes);
            advertisement
        };
        let advertisements = vec![
            machine(
                "peer-3090",
                "machine-3090",
                "cuda",
                "NVIDIA GeForce RTX 3090",
                24 * GIB,
                23 * GIB,
                false,
            ),
            machine(
                "peer-4060",
                "machine-4060",
                "cuda",
                "NVIDIA GeForce RTX 4060",
                8 * GIB,
                7 * GIB,
                false,
            ),
            machine(
                "peer-mac",
                "machine-mac",
                "metal",
                "Apple M5",
                16 * GIB,
                12 * GIB,
                true,
            ),
        ];

        let selected_3090 = model_records_to_download_for_local_contribution(
            &cache,
            OFFICIAL_CHAT_MODEL_ID,
            "peer-3090",
            &advertisements,
            vec![package.clone()],
        )
        .unwrap();
        let selected_4060 = model_records_to_download_for_local_contribution(
            &cache,
            OFFICIAL_CHAT_MODEL_ID,
            "peer-4060",
            &advertisements,
            vec![package.clone()],
        )
        .unwrap();
        let selected_mac = model_records_to_download_for_local_contribution(
            &cache,
            OFFICIAL_CHAT_MODEL_ID,
            "peer-mac",
            &advertisements,
            vec![package],
        )
        .unwrap();

        assert_eq!(selected_3090.len(), 1);
        assert_eq!(selected_4060.len(), 1);
        assert_eq!(selected_mac.len(), 1);

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn download_targets_exclude_the_relay_server() {
        let relay_key = identity::Keypair::generate_ed25519();
        let relay_peer_id = relay_key.public().to_peer_id();
        let mut config = DiscoveryConfig::new("infernet/test");
        config.relay_peers = vec![format!("/ip4/217.77.11.197/tcp/9777/p2p/{relay_peer_id}")];
        config.static_peers = vec![
            empty_advertisement(
                relay_peer_id.to_string(),
                format!("/ip4/217.77.11.197/udp/9777/quic-v1/p2p/{relay_peer_id}"),
            ),
            {
                let mut seed = empty_advertisement(
                    "model-seed".to_owned(),
                    "/ip4/10.0.0.2/tcp/9777".to_owned(),
                );
                seed.addresses.push(
                    "/ip4/217.77.11.197/udp/9777/quic-v1/p2p/relay/p2p-circuit/p2p/model-seed"
                        .to_owned(),
                );
                seed.addresses.push(
                    "/ip4/217.77.11.197/tcp/9777/p2p/relay/p2p-circuit/p2p/model-seed".to_owned(),
                );
                seed
            },
        ];

        remove_relay_servers_from_download_targets(&mut config);
        prefer_tcp_circuit_addresses_for_downloads(&mut config);

        assert_eq!(config.static_peers.len(), 1);
        assert_eq!(config.static_peers[0].peer_id, "model-seed");
        assert_eq!(config.static_peers[0].addresses.len(), 1);
        assert!(config.static_peers[0].addresses[0].contains("/tcp/"));
        assert!(config.static_peers[0].addresses[0].contains("/p2p-circuit/"));
    }

    #[test]
    fn official_https_download_is_revision_pinned_and_resumable_by_p2p() {
        let release = OfficialModelRelease::infernet_chat_v1_compatibility();
        let layers = LayerRange::new(0, 30).unwrap();
        let cache = ShardCacheConfig::new("/tmp/infernet-official-download-test");
        let (partial, completed) = official_release_download_paths(&cache, &release, layers);

        assert_eq!(
            official_release_download_url(&release),
            "https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf/resolve/dfc00409adc70be497fee9c90bfe76b3ee130f2e/gemma-4-26B_q4_0-it.gguf"
        );
        assert!(partial.ends_with("tmp/infernet-chat-v1-0-30-4c856523d61d7792.gguf.partial"));
        assert!(completed.ends_with("tmp/infernet-chat-v1-0-30-4c856523d61d7792.gguf"));
        assert_eq!(
            content_range_start("bytes 4194304-8388607/14439361440"),
            Some(4_194_304)
        );
        assert_eq!(content_range_start("invalid"), None);
    }

    #[test]
    fn peer_download_progress_is_identified_as_a_fallback() {
        let record = AdvertisedModelRecord {
            info: ModelShardInfo {
                model_id: OFFICIAL_CHAT_MODEL_ID.to_owned(),
                layers: LayerRange::new(0, 30).unwrap(),
                checksum: "checksum".to_owned(),
                size_bytes: 16,
                version: "v1".to_owned(),
                protocol_version: PROTOCOL_VERSION,
            },
        };

        assert_eq!(
            model_download_progress_detail(&record, true),
            "layers 0:30 · P2P fallback"
        );
        assert_eq!(
            model_download_progress_detail(&record, false),
            "layers 0:30"
        );
    }

    #[test]
    fn remote_routes_reject_loopback_but_keep_public_circuits() {
        let relay = identity::Keypair::generate_ed25519().public().to_peer_id();
        let target = identity::Keypair::generate_ed25519().public().to_peer_id();
        assert!(!remote_route_address_is_usable(&format!(
            "/ip4/127.0.0.1/tcp/9777/p2p/{target}"
        )));
        assert!(!remote_route_address_is_usable(&format!(
            "/ip4/0.0.0.0/tcp/9777/p2p/{target}"
        )));
        assert!(remote_route_address_is_usable(&format!(
            "/ip4/217.77.11.197/tcp/9777/p2p/{relay}/p2p-circuit/p2p/{target}"
        )));
    }
}
