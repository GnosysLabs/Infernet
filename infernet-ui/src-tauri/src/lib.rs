mod execution_plan;
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

use execution_plan::{ExecutionParticipantView, plan_rpc_execution, rpc_endpoint_is_usable};
use futures::channel::oneshot;
use infernet_model::{
    LayerRange, ModelManifest, OfficialComponentKind, OfficialModelRelease, RuntimeKind,
    SeedShardManifest, ShardDescriptor,
};
use infernet_node::{
    DiscoveryConfig, INFERNET_LLAMA_RPC_RUNTIME_ABI, LLAMA_RPC_DEFAULT_PORT,
    LLAMA_RPC_PROTOCOL_VERSION, LlamaRpcServer, LlamaRpcServerConfig, PAYLOAD_KIND_FULL_MODEL,
    PAYLOAD_KIND_GGUF_SHARD, PAYLOAD_KIND_INFERNET_SHARD, ShardCache, ShardCacheConfig,
    clear_local_llama_rpc_endpoint, detect_node_capabilities, empty_advertisement,
    enrich_local_advertisement, fetch_model_shard_over_libp2p_with_progress,
    find_llama_rpc_server_binary, infer_over_libp2p, is_executable_shard_record,
    load_or_generate_keypair, local_capability_advertisement, model_serving_telemetry,
    run_model_distribution_node_with_readiness_and_registry, seed_manifest_for_network,
    set_local_inference_active, set_local_llama_rpc_endpoint, set_local_rpc_active,
    spawn_llama_rpc_server, stop_persistent_llama_server, stop_persistent_rpc_tunnels,
};
use infernet_protocol::{
    LLAMA_RPC_TUNNEL_PROTOCOL, LlamaRpcEndpoint, ModelShardInfo, NodeAdvertisement,
    PROTOCOL_VERSION, RouteHop, TraceEvent,
};
use infernet_router::{
    CapacityPlanningConfig, FixedModelComponent, ShardRegistry, plan_fixed_components,
};
use libp2p::{Multiaddr, PeerId, identity, multiaddr::Protocol};
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, State};

use peer_presence::{ConnectionStatus, PeerPresence, PresenceRecord, PresenceSnapshot};

const DEFAULT_TOPIC: &str = "infernet/grid-demo/1";
const DEFAULT_DISCOVERY_TIMEOUT_MS: u64 = 4_000;
const DEFAULT_INFERENCE_TIMEOUT_MS: u64 = 30_000;
const DEFAULT_MODEL_FETCH_TIMEOUT_MS: u64 = 60 * 60 * 1_000;
const UI_LISTEN_PORT: u16 = 9777;
const OFFICIAL_CHAT_MODEL_ID: &str = "infernet-chat-v1";
const LAUNCH_KV_CACHE_BYTES_PER_LAYER: u64 = 32 * 1024 * 1024;
const RUNTIME_SCRATCH_BYTES_PER_PEER: u64 = 768 * 1024 * 1024;
const CAPACITY_SAFETY_BYTES: u64 = 1024 * 1024 * 1024;
const MODEL_PROGRESS_EMIT_BYTES: u64 = 64 * 1024 * 1024;
const DEFAULT_BOOTSTRAP_PEERS: &[&str] = &[
    "12D3KooWRJrnpHPQTWdThpDGZMwRCHhEBL4JCAxFMwYMfFavxa2h@/ip4/217.77.11.197/tcp/9777/p2p/12D3KooWRJrnpHPQTWdThpDGZMwRCHhEBL4JCAxFMwYMfFavxa2h",
    "12D3KooWRJrnpHPQTWdThpDGZMwRCHhEBL4JCAxFMwYMfFavxa2h@/dns4/infernet.gnosyslabs.xyz/tcp/9777/p2p/12D3KooWRJrnpHPQTWdThpDGZMwRCHhEBL4JCAxFMwYMfFavxa2h",
];

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

struct AdvertisedLlamaRpcServer {
    server: LlamaRpcServer,
}

impl Drop for AdvertisedLlamaRpcServer {
    fn drop(&mut self) {
        clear_local_llama_rpc_endpoint();
    }
}

struct UiState {
    keypair: Mutex<identity::Keypair>,
    topic: String,
    model_distribution_service: Arc<Mutex<ModelDistributionServiceState>>,
    live_registry: Arc<Mutex<ShardRegistry>>,
    llama_rpc_service: Arc<Mutex<LlamaRpcServiceState>>,
    active_model_acquisitions: Arc<Mutex<BTreeSet<String>>>,
    manual_peers: Mutex<Vec<NodeAdvertisement>>,
    peer_presence: Mutex<PeerPresence>,
    execution_plan: Mutex<Option<execution_plan::RpcExecutionPlan>>,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            keypair: Mutex::new(identity::Keypair::generate_ed25519()),
            topic: DEFAULT_TOPIC.to_owned(),
            model_distribution_service: Arc::new(Mutex::new(
                ModelDistributionServiceState::Stopped,
            )),
            live_registry: Arc::new(Mutex::new(ShardRegistry::new())),
            llama_rpc_service: Arc::new(Mutex::new(LlamaRpcServiceState::Stopped)),
            active_model_acquisitions: Arc::new(Mutex::new(BTreeSet::new())),
            manual_peers: Mutex::new(Vec::new()),
            peer_presence: Mutex::new(PeerPresence::default()),
            execution_plan: Mutex::new(None),
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
    machine_id: Option<String>,
    is_local: bool,
    connection_status: ConnectionStatus,
    last_seen_seconds: u64,
    compute_backend: String,
    device_name: String,
    logical_cpu_cores: u32,
    total_memory_bytes: u64,
    available_memory_bytes: u64,
    unified_memory: bool,
    max_sessions: u32,
    active_sessions: u32,
    queue_depth: u32,
    measured_prefill_tokens_per_second: Option<f32>,
    measured_decode_tokens_per_second: Option<f32>,
    hosted_component_count: usize,
    rpc_ready: bool,
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
    if advertised_model_record_plan(&registry, &manifest.model_id).is_empty() {
        return Err(
            "The verified Infernet Chat release is not being seeded by an online machine yet."
                .to_owned(),
        );
    }
    acquire_advertised_model_records(&app, &state, &cache_config, &manifest, &registry, true)
        .await?;
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

    let execution_route = registry
        .route_for_model(&manifest)
        .map_err(|error| error.to_string())?;
    let rpc_plan = match leased_rpc_execution_plan(&state, &registry, &execution_route, &manifest) {
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
            route: snapshot.route.clone(),
        },
    );
    emit_progress(
        &app,
        ProgressEvent::ExecutionPlan {
            participants: rpc_plan.participants.clone(),
        },
    );

    let (mut config, local_peer_id) = discovery_config_from_state(&state)?;
    config
        .set_rpc_worker_peer_ids(rpc_plan.worker_peer_ids)
        .map_err(|error| error.to_string())?;
    config.keypair = identity::Keypair::generate_ed25519();
    let mut local_advertisement = registry
        .advertisements()
        .into_iter()
        .find(|advertisement| advertisement.peer_id == local_peer_id)
        .unwrap_or_else(|| local_capability_advertisement(local_peer_id.clone(), String::new()));
    local_advertisement.addresses = local_connect_addresses(&local_peer_id);
    config.static_peers.push(local_advertisement);
    merge_static_peer_advertisements(&mut config.static_peers, registry.advertisements());
    config.set_planned_route(execution_route);
    let hidden_size = manifest.hidden_size;
    let result = match infer_over_libp2p(
        config,
        manifest,
        prompt,
        hidden_size,
        Duration::from_millis(DEFAULT_INFERENCE_TIMEOUT_MS),
    )
    .await
    {
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

fn leased_rpc_execution_plan(
    state: &State<'_, UiState>,
    registry: &ShardRegistry,
    route: &[RouteHop],
    manifest: &ModelManifest,
) -> Result<execution_plan::RpcExecutionPlan, String> {
    let mut lease = state
        .execution_plan
        .lock()
        .map_err(|_| "failed to lock distributed execution plan".to_owned())?;
    if let Some(plan) = lease.as_ref() {
        if plan.remains_usable(registry, route) {
            return Ok(plan.clone());
        }
    }
    let plan = plan_rpc_execution(registry, route, manifest)?;
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
    registry.upsert(local_node_advertisement(
        cache_config,
        local_peer_id.clone(),
    ));
    let fresh_registry = trusted_launch_registry(registry);
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

fn trusted_launch_registry(registry: ShardRegistry) -> ShardRegistry {
    let mut trusted = ShardRegistry::new();
    for mut advertisement in registry.advertisements() {
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
                    format!(
                        "layers {}:{}",
                        record.info.layers.start, record.info.layers.end
                    ),
                    0,
                    Some(record.info.size_bytes),
                );
                let progress_app = app.clone();
                let progress_model_id = manifest.model_id.clone();
                let progress_detail = format!(
                    "layers {}:{}",
                    record.info.layers.start, record.info.layers.end
                );
                let mut last_progress_emit = 0_u64;
                fetch_model_shard_over_libp2p_with_progress(
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
                .await?;
                emit_model_import_progress(
                    &app,
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

    let config = |minimum_peer_count| CapacityPlanningConfig {
        kv_cache_bytes_per_layer: LAUNCH_KV_CACHE_BYTES_PER_LAYER,
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

    GridSnapshot {
        local_peer_id,
        topic,
        selected_model: manifest.model_id.clone(),
        available_models: available_model_views(cache_config, Some(registry)),
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
    GridSnapshot {
        local_peer_id,
        topic,
        selected_model: String::new(),
        available_models: available_model_views(cache_config, Some(registry)),
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
) -> Vec<ModelView> {
    let installed_ids = installed_model_ids(cache_config);
    let manifest = ModelManifest::infernet_chat_v1();
    let network_runnable = registry.is_some_and(|registry| {
        registry.route_for_model(&manifest).is_ok_and(|route| {
            let Some(coordinator) = route.first().map(|hop| hop.peer_id.as_str()) else {
                return false;
            };
            registry.advertisements().iter().any(|advertisement| {
                advertisement.peer_id != coordinator
                    && advertisement
                        .capabilities
                        .as_ref()
                        .is_some_and(rpc_endpoint_is_usable)
            })
        })
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
                    && rpc_endpoint_is_usable(left_capabilities)
                    && left_capabilities.active_sessions < left_capabilities.max_sessions;
                let right_rpc = right.status.is_connected()
                    && rpc_endpoint_is_usable(right_capabilities)
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
            let rpc_ready = aliases.iter().any(|alias| {
                let capabilities = alias.advertisement.capabilities.as_ref().unwrap();
                alias.status.is_connected()
                    && rpc_endpoint_is_usable(capabilities)
                    && capabilities.active_sessions < capabilities.max_sessions
            });

            Some(MachineView {
                peer_id: advertisement.peer_id.clone(),
                short_peer_id: short_peer_id(&advertisement.peer_id),
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
                unified_memory: capabilities.unified_memory,
                max_sessions: capabilities.max_sessions,
                active_sessions: capabilities.active_sessions,
                queue_depth: capabilities.queue_depth,
                measured_prefill_tokens_per_second: capabilities.measured_prefill_tokens_per_second,
                measured_decode_tokens_per_second: capabilities.measured_decode_tokens_per_second,
                hosted_component_count: hosted_components.len(),
                rpc_ready,
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
    })
}

fn local_node_advertisement(cache_config: &ShardCacheConfig, peer_id: String) -> NodeAdvertisement {
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
    if let Err(error) = ensure_llama_rpc_service(state, &cache_config).await {
        // Model discovery and downloads remain useful when a development build
        // does not contain the optional sidecar. Crucially, clearing the
        // endpoint prevents this node from claiming compute it cannot serve.
        clear_local_llama_rpc_endpoint();
        eprintln!("llama.cpp RPC sidecar is unavailable: {error}");
    }

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
    if environment_flag("INFERNET_DISABLE_LLAMA_RPC") {
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
    let port = available_rpc_port(host, requested_port)?;
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

fn available_rpc_port(host: Ipv4Addr, requested: Option<u16>) -> Result<u16, String> {
    let preferred = requested.unwrap_or(LLAMA_RPC_DEFAULT_PORT);
    match TcpListener::bind((host, preferred)) {
        Ok(listener) => {
            drop(listener);
            Ok(preferred)
        }
        Err(error) if requested.is_some() => Err(format!(
            "configured llama.cpp RPC address {host}:{preferred} is unavailable: {error}"
        )),
        Err(_) => {
            let listener = TcpListener::bind((host, 0)).map_err(|error| {
                format!("could not allocate a private llama.cpp RPC port on {host}: {error}")
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
    let _ = app.emit(
        "infernet-model-import-progress",
        ModelImportProgress {
            model_id: model_id.into(),
            stage: stage.into(),
            detail: detail.into(),
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
            let identity_path = app.path().app_data_dir()?.join("identity.key");
            let keypair = load_or_generate_keypair(&identity_path)?;
            *app_handle
                .state::<UiState>()
                .keypair
                .lock()
                .expect("UI identity lock poisoned during startup") = keypair;
            let cache_config = cache_config_for_app(app.handle());
            monitor_llama_rpc_service(Arc::clone(&app_handle.state::<UiState>().llama_rpc_service));
            tauri::async_runtime::spawn(async move {
                let state = app_handle.state::<UiState>();
                if let Err(error) = ensure_model_distribution_service(&state, cache_config).await {
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
                clear_local_llama_rpc_endpoint();
                set_local_inference_active(false);
                set_local_rpc_active(false);
            }
        })
        .invoke_handler(tauri::generate_handler![
            get_local_identity,
            get_manual_peers,
            add_manual_peer,
            clear_manual_peers,
            get_grid_snapshot,
            install_official_model,
            run_demo_inference
        ])
        .run(tauri::generate_context!())
        .expect("error while running Infernet UI");
}

#[cfg(test)]
mod tests {
    use super::*;

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

        let models = available_model_views(&cache_config, None);
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
            available_model_views(&cache_config, Some(&registry)).len(),
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
        let trusted = trusted_launch_registry(registry);
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
        let views = available_model_views(&cache_config, None);
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
            available_model_views(&cache_config, None)
                .into_iter()
                .all(|model| model.model_id != "gemma"),
            "unsupported legacy partials must not appear runnable or advertised"
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn full_package_auto_host_is_the_3090_not_the_4060_or_small_mac() {
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
        assert!(selected_4060.is_empty());
        assert!(selected_mac.is_empty());

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
}
