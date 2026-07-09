use std::time::Duration;
use std::{
    collections::BTreeMap,
    env,
    fs::File,
    io::Write,
    net::{IpAddr, UdpSocket},
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
    time::{SystemTime, UNIX_EPOCH},
};

use futures::StreamExt;
use infernet_model::{
    LayerRange, ModelManifest, RuntimeKind, SeedShardManifest, ShardDescriptor,
    gguf::parse_gguf_info,
};
use infernet_node::{
    DiscoveryConfig, SeededModelSummary, ShardCache, ShardCacheConfig, discover_for,
    empty_advertisement, fetch_model_shard_over_libp2p, import_seed_model_from_file_with_progress,
    infer_over_libp2p, run_model_distribution_node, sha256_bytes,
};
use infernet_protocol::{
    ModelShardInfo, NodeAdvertisement, PROTOCOL_VERSION, RouteHop, TraceEvent,
};
use infernet_router::ShardRegistry;
use libp2p::{Multiaddr, PeerId, identity};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, State};
use tokio::process::Command;

const DEFAULT_TOPIC: &str = "infernet/grid-demo/1";
const DEFAULT_DISCOVERY_TIMEOUT_MS: u64 = 4_000;
const DEFAULT_INFERENCE_TIMEOUT_MS: u64 = 6_000;
const DEFAULT_MODEL_FETCH_TIMEOUT_MS: u64 = 6_000;
const MIN_LOCAL_GGUF_LIMIT_BYTES: u64 = 3 * 1024 * 1024 * 1024;
const UI_LISTEN_PORT: u16 = 9777;
const DEFAULT_BOOTSTRAP_PEERS: &[&str] = &[
    "12D3KooWRJrnpHPQTWdThpDGZMwRCHhEBL4JCAxFMwYMfFavxa2h@/ip4/217.77.11.197/tcp/9777/p2p/12D3KooWRJrnpHPQTWdThpDGZMwRCHhEBL4JCAxFMwYMfFavxa2h",
    "12D3KooWRJrnpHPQTWdThpDGZMwRCHhEBL4JCAxFMwYMfFavxa2h@/dns4/infernet.gnosyslabs.xyz/tcp/9777/p2p/12D3KooWRJrnpHPQTWdThpDGZMwRCHhEBL4JCAxFMwYMfFavxa2h",
];

struct UiState {
    keypair: Mutex<identity::Keypair>,
    topic: String,
    huggingface_token: Mutex<Option<String>>,
    model_distribution_started: Mutex<bool>,
    manual_peers: Mutex<Vec<NodeAdvertisement>>,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            keypair: Mutex::new(identity::Keypair::generate_ed25519()),
            topic: DEFAULT_TOPIC.to_owned(),
            huggingface_token: Mutex::new(None),
            model_distribution_started: Mutex::new(false),
            manual_peers: Mutex::new(Vec::new()),
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
struct AddModelResponse {
    model_id: String,
    display_name: String,
    source: String,
    source_checksum: String,
    source_size_bytes: u64,
    planned_shards: usize,
    metadata_only: bool,
    installed_shards: Vec<InstalledShardView>,
    message: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct HuggingFaceSettings {
    has_token: bool,
    token_preview: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct HuggingFaceFileView {
    filename: String,
    size_bytes: Option<u64>,
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
    seed_manifest: Option<SeedShardManifest>,
}

#[derive(Debug, Deserialize)]
struct HuggingFaceModelInfo {
    siblings: Option<Vec<HuggingFaceSibling>>,
}

#[derive(Debug, Deserialize)]
struct HuggingFaceSibling {
    rfilename: String,
    size: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "camelCase")]
enum ProgressEvent {
    RouteDiscovered {
        route: Vec<RouteHopView>,
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
    ensure_model_distribution_service(&state, cache_config.clone())?;

    collect_snapshot(
        &app,
        &state,
        discovery_timeout_ms.unwrap_or(DEFAULT_DISCOVERY_TIMEOUT_MS),
        model_id.as_deref(),
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
    ensure_model_distribution_service(&state, cache_config.clone())?;
    let (registry, _, _) =
        discover_registry(&app, &state, &cache_config, DEFAULT_DISCOVERY_TIMEOUT_MS).await?;
    let manifest = manifest_for_model(model_id.as_deref(), &cache_config, Some(&registry))
        .map_err(|error| error.to_string())?;

    if manifest.runtime_kind != RuntimeKind::Demo {
        acquire_advertised_model_records(&app, &state, &cache_config, &manifest, &registry, true)
            .await?;
        let (refreshed_registry, local_peer_id, topic) =
            discover_registry(&app, &state, &cache_config, DEFAULT_DISCOVERY_TIMEOUT_MS).await?;
        let manifest = manifest_for_model(
            Some(&manifest.model_id),
            &cache_config,
            Some(&refreshed_registry),
        )
        .map_err(|error| error.to_string())?;
        let trace_id = format!("llama-{}", unix_ms());
        let local_route = vec![RouteHopView {
            peer_id: local_peer_id.clone(),
            short_peer_id: short_peer_id(&local_peer_id),
            address: "local".to_owned(),
            layer_start: 0,
            layer_end: manifest.layer_count,
        }];
        emit_progress(
            &app,
            ProgressEvent::RouteDiscovered {
                route: local_route.clone(),
            },
        );
        replay_route_progress(&app, &trace_id, &local_route, manifest.hidden_size).await;
        let output = generate_with_llama_cli(&app, &cache_config, &manifest, &prompt).await?;
        emit_progress(
            &app,
            ProgressEvent::FinalOutput {
                trace_id: trace_id.clone(),
                output: output.clone(),
            },
        );
        let snapshot = snapshot_from_registry(
            local_peer_id,
            topic,
            &manifest,
            &refreshed_registry,
            &cache_config,
        );
        return Ok(RunDemoResponse {
            output,
            trace_id,
            snapshot,
        });
    }

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

    emit_progress(
        &app,
        ProgressEvent::RouteDiscovered {
            route: snapshot.route.clone(),
        },
    );

    let (config, _) = discovery_config_from_state(&state)?;
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
        .unwrap_or_else(|| "<missing output>".to_owned());
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

#[tauri::command]
async fn add_local_gguf_model(
    app: AppHandle,
    state: State<'_, UiState>,
    path: String,
    version: Option<String>,
) -> Result<AddModelResponse, String> {
    let cache_config = cache_config_for_app(&app);
    let cache = ShardCache::new(cache_config.clone()).map_err(|error| error.to_string())?;
    let source = PathBuf::from(path);
    let progress_model_id = model_id_from_source_path(&source);
    emit_model_import_progress(
        &app,
        &progress_model_id,
        "Checking file",
        source.display().to_string(),
        0,
        None,
    );
    let manifest = manifest_from_gguf_source(&source).map_err(|error| error.to_string())?;
    let summary = import_seed_model_from_file_with_progress(
        &cache,
        &source,
        &manifest,
        version.unwrap_or_else(|| "v1".to_owned()),
        |downloaded_bytes, total_bytes| {
            emit_model_import_progress(
                &app,
                &manifest.model_id,
                "Verifying model",
                "Reading and verifying the selected file",
                downloaded_bytes,
                Some(total_bytes),
            );
        },
    )
    .map_err(|error| error.to_string())?;
    emit_model_import_progress(
        &app,
        &manifest.model_id,
        "Starting sharing",
        "Publishing the model to the network",
        summary.source_size_bytes,
        Some(summary.source_size_bytes),
    );
    ensure_model_distribution_service(&state, cache_config)?;
    emit_model_import_progress(
        &app,
        &manifest.model_id,
        "Ready",
        "Infernet is sharing this model",
        summary.source_size_bytes,
        Some(summary.source_size_bytes),
    );

    Ok(add_model_response_from_summary(summary))
}

#[tauri::command]
async fn get_huggingface_settings(
    state: State<'_, UiState>,
) -> Result<HuggingFaceSettings, String> {
    let token = state
        .huggingface_token
        .lock()
        .map_err(|_| "failed to lock Hugging Face settings".to_owned())?
        .clone();

    Ok(huggingface_settings_from_token(token.as_deref()))
}

#[tauri::command]
async fn save_huggingface_token(
    state: State<'_, UiState>,
    token: String,
) -> Result<HuggingFaceSettings, String> {
    let token = token.trim().to_owned();
    let mut stored = state
        .huggingface_token
        .lock()
        .map_err(|_| "failed to lock Hugging Face settings".to_owned())?;
    *stored = (!token.is_empty()).then_some(token);

    Ok(huggingface_settings_from_token(stored.as_deref()))
}

#[tauri::command]
async fn clear_huggingface_token(state: State<'_, UiState>) -> Result<HuggingFaceSettings, String> {
    let mut stored = state
        .huggingface_token
        .lock()
        .map_err(|_| "failed to lock Hugging Face settings".to_owned())?;
    *stored = None;

    Ok(huggingface_settings_from_token(None))
}

#[tauri::command]
async fn inspect_huggingface_repo(
    state: State<'_, UiState>,
    repo_id: String,
    token: Option<String>,
) -> Result<Vec<HuggingFaceFileView>, String> {
    let repo_id = repo_id.trim();
    if repo_id.is_empty() {
        return Err("enter a Hugging Face repository id".to_owned());
    }

    let client = reqwest::Client::new();
    let url = format!("https://huggingface.co/api/models/{repo_id}");
    let response = apply_huggingface_auth(client.get(url), &state, token.as_deref())?
        .send()
        .await
        .map_err(|error| error.to_string())?;

    if !response.status().is_success() {
        return Err(format!(
            "Hugging Face returned {} while reading {repo_id}",
            response.status()
        ));
    }

    let info = response
        .json::<HuggingFaceModelInfo>()
        .await
        .map_err(|error| error.to_string())?;
    let mut files = info
        .siblings
        .unwrap_or_default()
        .into_iter()
        .filter(|file| file.rfilename.to_ascii_lowercase().ends_with(".gguf"))
        .map(|file| HuggingFaceFileView {
            filename: file.rfilename,
            size_bytes: file.size,
        })
        .collect::<Vec<_>>();
    files.sort_by(|left, right| left.filename.cmp(&right.filename));

    Ok(files)
}

#[tauri::command]
async fn add_huggingface_model(
    app: AppHandle,
    state: State<'_, UiState>,
    repo_id: String,
    filename: String,
    token: Option<String>,
    revision: Option<String>,
    version: Option<String>,
) -> Result<AddModelResponse, String> {
    let repo_id = repo_id.trim();
    let filename = filename.trim();
    if repo_id.is_empty() || filename.is_empty() {
        return Err("choose a Hugging Face repo and GGUF file".to_owned());
    }

    let revision = revision.unwrap_or_else(|| "main".to_owned());
    let target = huggingface_download_path(&app, repo_id, filename)?;
    let progress_model_id = model_id_from_source_path(Path::new(filename));
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    emit_model_import_progress(
        &app,
        &progress_model_id,
        "Connecting",
        format!("Opening {repo_id}"),
        0,
        None,
    );

    let client = reqwest::Client::new();
    let url = format!("https://huggingface.co/{repo_id}/resolve/{revision}/{filename}");
    let response = apply_huggingface_auth(client.get(url), &state, token.as_deref())?
        .send()
        .await
        .map_err(|error| error.to_string())?;
    if !response.status().is_success() {
        return Err(format!(
            "Hugging Face returned {} while downloading {repo_id}/{filename}",
            response.status()
        ));
    }

    let total_bytes = response.content_length();
    let mut file = File::create(&target).map_err(|error| error.to_string())?;
    let mut stream = response.bytes_stream();
    let mut downloaded_bytes = 0_u64;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|error| error.to_string())?;
        file.write_all(&chunk).map_err(|error| error.to_string())?;
        downloaded_bytes = downloaded_bytes.saturating_add(chunk.len() as u64);
        emit_model_import_progress(
            &app,
            &progress_model_id,
            "Downloading",
            filename.to_owned(),
            downloaded_bytes,
            total_bytes,
        );
    }

    let manifest = manifest_from_gguf_source(&target).map_err(|error| error.to_string())?;
    let cache_config = cache_config_for_app(&app);
    let cache = ShardCache::new(cache_config.clone()).map_err(|error| error.to_string())?;
    let summary = import_seed_model_from_file_with_progress(
        &cache,
        &target,
        &manifest,
        version.unwrap_or_else(|| "v1".to_owned()),
        |verified_bytes, total_bytes| {
            emit_model_import_progress(
                &app,
                &manifest.model_id,
                "Verifying model",
                "Reading and verifying the downloaded file",
                verified_bytes,
                Some(total_bytes),
            );
        },
    )
    .map_err(|error| error.to_string())?;
    emit_model_import_progress(
        &app,
        &manifest.model_id,
        "Starting sharing",
        "Publishing the model to the network",
        summary.source_size_bytes,
        Some(summary.source_size_bytes),
    );
    ensure_model_distribution_service(&state, cache_config)?;
    emit_model_import_progress(
        &app,
        &manifest.model_id,
        "Ready",
        "Infernet is sharing this model",
        summary.source_size_bytes,
        Some(summary.source_size_bytes),
    );

    Ok(add_model_response_from_summary(summary))
}

async fn collect_snapshot(
    app: &AppHandle,
    state: &State<'_, UiState>,
    discovery_timeout_ms: u64,
    model_id: Option<&str>,
) -> Result<GridSnapshot, String> {
    let cache_config = cache_config_for_app(app);
    let (registry, local_peer_id, topic) =
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
                ));
            }
            return Err(error.to_string());
        }
    };
    acquire_advertised_model_records(app, state, &cache_config, &manifest, &registry, false)
        .await?;

    Ok(snapshot_from_registry(
        local_peer_id,
        topic,
        &manifest,
        &registry,
        &cache_config,
    ))
}

async fn discover_registry(
    _app: &AppHandle,
    state: &State<'_, UiState>,
    cache_config: &ShardCacheConfig,
    discovery_timeout_ms: u64,
) -> Result<(ShardRegistry, String, String), String> {
    let (mut config, local_peer_id) = discovery_config_from_state(state)?;
    let topic = config.topic.clone();
    if let Some(local_advertisement) =
        local_cache_advertisement(cache_config, local_peer_id.clone())
    {
        config.advertisement = Some(local_advertisement);
    }
    let registry = discover_for(config, Duration::from_millis(discovery_timeout_ms))
        .await
        .map_err(|error| error.to_string())?;

    Ok((registry, local_peer_id, topic))
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

        if let Some(seed_manifest) = record.seed_manifest.as_ref() {
            emit_model_import_progress(
                app,
                &manifest.model_id,
                "Installing model record",
                format!(
                    "layers {}:{}",
                    record.info.layers.start, record.info.layers.end
                ),
                0,
                Some(record.info.size_bytes),
            );
            install_advertised_seed_record(&cache, &record.info, seed_manifest)?;
            emit_model_import_progress(
                app,
                &manifest.model_id,
                "Model record ready",
                format!(
                    "layers {}:{}",
                    record.info.layers.start, record.info.layers.end
                ),
                record.info.size_bytes,
                Some(record.info.size_bytes),
            );
            continue;
        }

        if !allow_direct_fetch {
            continue;
        }

        emit_model_import_progress(
            app,
            &manifest.model_id,
            "Downloading model record",
            format!(
                "layers {}:{}",
                record.info.layers.start, record.info.layers.end
            ),
            0,
            Some(record.info.size_bytes),
        );
        let (mut config, _) = discovery_config_from_state(state)?;
        config.static_peers = static_peers.clone();
        fetch_model_shard_over_libp2p(
            config,
            cache_config.clone(),
            manifest.model_id.clone(),
            record.info.layers,
            Some(record.info.checksum.clone()),
            Some(record.info.version.clone()),
            Duration::from_millis(DEFAULT_MODEL_FETCH_TIMEOUT_MS),
        )
        .await
        .map_err(|error| error.to_string())?;
        emit_model_import_progress(
            app,
            &manifest.model_id,
            "Model record ready",
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

fn snapshot_from_registry(
    local_peer_id: String,
    topic: String,
    manifest: &ModelManifest,
    registry: &ShardRegistry,
    cache_config: &ShardCacheConfig,
) -> GridSnapshot {
    let all_advertisements = registry.advertisements();
    let network_peer_count = remote_network_peer_count(&local_peer_id, &all_advertisements);
    let advertisements =
        ui_visible_advertisements(all_advertisements.clone(), Some(&manifest.model_id));
    let route_result = registry.route_for_model(manifest);
    let (route, missing_ranges) = match route_result {
        Ok(route) => (route, None),
        Err(error) => (Vec::new(), Some(error.to_string())),
    };

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
) -> GridSnapshot {
    let all_advertisements = registry.advertisements();
    let network_peer_count = remote_network_peer_count(&local_peer_id, &all_advertisements);
    let advertisements = ui_visible_advertisements(all_advertisements, None);
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
        route: Vec::new(),
        missing_ranges: None,
        coverage: Vec::new(),
        distribution: build_distribution_snapshot(cache_config, &advertisements),
    }
}

fn model_view_from_manifest(
    manifest: &ModelManifest,
    installed: bool,
    cache_config: &ShardCacheConfig,
) -> ModelView {
    let runnable = manifest.runtime_kind == RuntimeKind::Demo
        || (manifest.runtime_kind == RuntimeKind::LlamaCpp
            && installed
            && local_source_path_for_model(cache_config, &manifest.model_id)
                .and_then(|path| std::fs::metadata(path).ok())
                .is_some_and(|metadata| metadata.len() <= local_gguf_size_limit_bytes())
            && find_llama_cli(None).is_some());
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
    let mut manifests = installed_model_manifests(cache_config);
    if let Some(registry) = registry {
        manifests.extend(discovered_model_manifests(registry));
    }
    manifests.retain(|manifest| manifest.runtime_kind != RuntimeKind::Demo);
    manifests.sort_by(|left, right| left.model_id.cmp(&right.model_id));
    manifests.dedup_by(|left, right| left.model_id == right.model_id);
    manifests.sort_by(|left, right| left.display_name.cmp(&right.display_name));

    manifests
        .iter()
        .map(|manifest| {
            model_view_from_manifest(
                manifest,
                installed_ids.contains(&manifest.model_id),
                cache_config,
            )
        })
        .collect()
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
            let payload = cache.read_payload(&record.info).ok()?;
            let manifest = serde_json::from_slice::<SeedShardManifest>(&payload).ok()?;
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

fn discovered_model_manifests(registry: &ShardRegistry) -> Vec<ModelManifest> {
    let mut by_model = BTreeMap::<String, ModelManifest>::new();
    for shard in registry
        .advertisements()
        .iter()
        .flat_map(|advertisement| advertisement.hosted_shards.iter())
        .filter(|shard| shard.runtime_kind != RuntimeKind::Demo)
    {
        if let Some(seed_manifest) = shard.seed_manifest.as_deref() {
            by_model
                .entry(shard.model_id.clone())
                .and_modify(|manifest| {
                    manifest.layer_count = manifest.layer_count.max(seed_manifest.layer_count);
                    manifest.hidden_size = manifest.hidden_size.max(seed_manifest.hidden_size);
                    if manifest.quantization.is_none() {
                        manifest.quantization = seed_manifest.metadata.quantization.clone();
                    }
                })
                .or_insert_with(|| ModelManifest {
                    model_id: seed_manifest.model_id.clone(),
                    display_name: seed_manifest.display_name.clone(),
                    architecture: seed_manifest.architecture.clone(),
                    layer_count: seed_manifest.layer_count,
                    hidden_size: seed_manifest.hidden_size,
                    activation_dtype: seed_manifest.activation_dtype.clone(),
                    quantization: seed_manifest.metadata.quantization.clone(),
                    runtime_kind: seed_manifest.runtime_kind.clone(),
                });
            continue;
        }
        by_model
            .entry(shard.model_id.clone())
            .and_modify(|manifest| {
                manifest.layer_count = manifest.layer_count.max(shard.layers.end);
            })
            .or_insert_with(|| ModelManifest {
                model_id: shard.model_id.clone(),
                display_name: display_name_from_model_id(&shard.model_id),
                architecture: shard
                    .metadata
                    .as_ref()
                    .map(|metadata| metadata.architecture.clone())
                    .unwrap_or_else(|| "unknown".to_owned()),
                layer_count: shard.layers.end,
                hidden_size: 0,
                activation_dtype: "f16".to_owned(),
                quantization: shard
                    .metadata
                    .as_ref()
                    .and_then(|metadata| metadata.quantization.as_deref())
                    .map(normalize_quantization_label),
                runtime_kind: shard.runtime_kind.clone(),
            });
    }
    for shard in registry
        .advertisements()
        .iter()
        .flat_map(|advertisement| advertisement.model_shards.iter())
        .filter(|shard| shard.model_id != ModelManifest::demo().model_id)
    {
        by_model
            .entry(shard.model_id.clone())
            .and_modify(|manifest| {
                manifest.layer_count = manifest.layer_count.max(shard.layers.end);
            })
            .or_insert_with(|| ModelManifest {
                model_id: shard.model_id.clone(),
                display_name: display_name_from_model_id(&shard.model_id),
                architecture: "unknown".to_owned(),
                layer_count: shard.layers.end,
                hidden_size: 0,
                activation_dtype: "f16".to_owned(),
                quantization: None,
                runtime_kind: RuntimeKind::LlamaCpp,
            });
    }

    by_model.into_values().collect()
}

fn ui_visible_advertisements(
    advertisements: Vec<NodeAdvertisement>,
    model_id: Option<&str>,
) -> Vec<NodeAdvertisement> {
    advertisements
        .into_iter()
        .filter_map(|mut advertisement| {
            advertisement.hosted_shards.retain(|shard| {
                shard.runtime_kind != RuntimeKind::Demo
                    && model_id.is_none_or(|model_id| shard.model_id == model_id)
            });
            advertisement.model_shards.retain(|shard| {
                shard.model_id != ModelManifest::demo().model_id
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

fn remote_network_peer_count(local_peer_id: &str, advertisements: &[NodeAdvertisement]) -> usize {
    let bootstrap_peer_ids = default_bootstrap_peer_ids();
    advertisements
        .iter()
        .filter(|advertisement| advertisement.peer_id != local_peer_id)
        .filter(|advertisement| !bootstrap_peer_ids.contains(&advertisement.peer_id))
        .count()
}

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
        return "Available on the network".to_owned();
    }

    let Some(source_path) = local_source_path_for_model(cache_config, &manifest.model_id) else {
        return "Available for sharing. This machine has shard records, but not executable GGUF tensors yet.".to_owned();
    };
    if std::fs::metadata(source_path)
        .map(|metadata| metadata.len() > local_gguf_size_limit_bytes())
        .unwrap_or(false)
    {
        return "Installed locally. This machine does not have enough memory for safe local fallback execution.".to_owned();
    }

    "Installed locally. Token runtime is being bundled with the app.".to_owned()
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
            let seed_manifest = advertisement
                .hosted_shards
                .iter()
                .find(|descriptor| {
                    descriptor.model_id == info.model_id && descriptor.layers == info.layers
                })
                .and_then(|descriptor| descriptor.seed_manifest.as_deref())
                .cloned();
            let record = AdvertisedModelRecord {
                info: info.clone(),
                seed_manifest,
            };

            by_range
                .entry((info.layers.start, info.layers.end))
                .and_modify(|existing| {
                    if existing.seed_manifest.is_none() && record.seed_manifest.is_some() {
                        *existing = record.clone();
                    } else if existing.seed_manifest.is_none()
                        && record.seed_manifest.is_none()
                        && (
                            record.info.version.clone(),
                            record.info.size_bytes,
                            record.info.checksum.clone(),
                        ) < (
                            existing.info.version.clone(),
                            existing.info.size_bytes,
                            existing.info.checksum.clone(),
                        )
                    {
                        *existing = record.clone();
                    }
                })
                .or_insert(record);
        }
    }

    by_range.into_values().collect()
}

fn verify_advertised_seed_payload(info: &ModelShardInfo, payload: &[u8]) -> Result<(), String> {
    let actual_checksum = sha256_bytes(payload);
    if actual_checksum != info.checksum {
        return Err(format!(
            "advertised model record checksum mismatch for {} {}:{}; expected {}, got {}",
            info.model_id, info.layers.start, info.layers.end, info.checksum, actual_checksum
        ));
    }
    if payload.len() as u64 != info.size_bytes {
        return Err(format!(
            "advertised model record size mismatch for {} {}:{}; expected {}, got {}",
            info.model_id,
            info.layers.start,
            info.layers.end,
            info.size_bytes,
            payload.len()
        ));
    }

    Ok(())
}

fn install_advertised_seed_record(
    cache: &ShardCache,
    info: &ModelShardInfo,
    seed_manifest: &SeedShardManifest,
) -> Result<(), String> {
    let payload = serde_json::to_vec_pretty(seed_manifest).map_err(|error| error.to_string())?;
    verify_advertised_seed_payload(info, &payload)?;
    cache
        .store_downloaded(info, payload)
        .map(|_| ())
        .map_err(|error| error.to_string())
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
    let cache = ShardCache::new(cache_config.clone());
    let (installed_shards, storage_used_bytes, max_storage_bytes) = match cache {
        Ok(cache) => {
            let records = cache.list().unwrap_or_default();
            let stats = cache.stats().ok();
            (
                records
                    .iter()
                    .map(|record| InstalledShardView {
                        model_id: record.info.model_id.clone(),
                        layer_start: record.info.layers.start,
                        layer_end: record.info.layers.end,
                        checksum: record.info.checksum.clone(),
                        size_bytes: record.info.size_bytes,
                        version: record.info.version.clone(),
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
        current_uploads: advertisements
            .iter()
            .map(|advertisement| advertisement.model_shards.len())
            .sum(),
        current_downloads: 0,
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

fn local_cache_advertisement(
    cache_config: &ShardCacheConfig,
    peer_id: String,
) -> Option<NodeAdvertisement> {
    let cache = ShardCache::new(cache_config.clone()).ok()?;
    let records = cache.list().ok()?;
    let mut hosted_shards = Vec::new();
    let mut model_shards = Vec::new();

    for record in records {
        let payload = cache.read_payload(&record.info).ok()?;
        let manifest = serde_json::from_slice::<SeedShardManifest>(&payload).ok()?;
        if seed_record_is_executable(&manifest) {
            let seed_manifest = Box::new(manifest.clone());
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
        model_shards.push(record.info);
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
        hosted_shards,
        model_shards,
    })
}

fn seed_record_is_executable(manifest: &SeedShardManifest) -> bool {
    manifest.runtime_kind == RuntimeKind::Demo
        || manifest.payload_kind != "metadata-only"
        || Path::new(&manifest.source.path).is_file()
}

async fn generate_with_llama_cli(
    app: &AppHandle,
    cache_config: &ShardCacheConfig,
    manifest: &ModelManifest,
    prompt: &str,
) -> Result<String, String> {
    let model_path = local_source_path_for_model(cache_config, &manifest.model_id).ok_or_else(|| {
        format!(
            "{} is available, but this machine does not have executable GGUF tensors yet. Infernet can share metadata records today; real split-GGUF token execution still needs the tensor shard runtime.",
            manifest.display_name
        )
    })?;
    let llama_cli = find_llama_cli(Some(app)).ok_or_else(|| {
        "The bundled llama.cpp runtime is missing from this app build. Rebuild Infernet so it can package the token runtime.".to_owned()
    })?;
    let model_size = std::fs::metadata(&model_path)
        .map_err(|error| format!("failed to read {}: {error}", model_path.display()))?
        .len();
    let allow_large = env::var("INFERNET_ALLOW_LARGE_GGUF").as_deref() == Ok("1");
    let size_limit = local_gguf_size_limit_bytes();
    if model_size > size_limit && !allow_large {
        return Err(format!(
            "{} is {}. This machine's local GGUF safety limit is {}. This model needs a smaller quantization, more memory, or the split GGUF token runtime.",
            model_path.display(),
            format_bytes(model_size),
            format_bytes(size_limit),
        ));
    }

    let mut command = Command::new(&llama_cli);
    if let Some(runtime_dir) = llama_cli.parent() {
        command.current_dir(runtime_dir);
    }
    #[cfg(target_os = "windows")]
    {
        let mut library_dirs = llama_runtime_library_dirs(Some(app), &llama_cli);
        if let Some(path) = env::var_os("PATH") {
            library_dirs.extend(env::split_paths(&path));
        }
        if let Ok(path) = env::join_paths(library_dirs) {
            command.env("PATH", path);
        }
    }

    let output = command
        .arg("-m")
        .arg(&model_path)
        .arg("-p")
        .arg(prompt)
        .arg("-n")
        .arg("64")
        .arg("--no-display-prompt")
        .arg("--simple-io")
        .env("LLAMA_LOG_LEVEL", "error")
        .output()
        .await
        .map_err(|error| format!("failed to launch {}: {error}", llama_cli.display()))?;

    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    if !output.status.success() {
        return Err(if stderr.is_empty() {
            format!("llama.cpp exited with {}", output.status)
        } else {
            stderr
        });
    }

    if stdout.is_empty() {
        return Err(if stderr.is_empty() {
            "llama.cpp produced no output".to_owned()
        } else {
            stderr
        });
    }

    Ok(stdout)
}

#[cfg(target_os = "windows")]
fn llama_runtime_library_dirs(app: Option<&AppHandle>, llama_cli: &Path) -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    if let Some(parent) = llama_cli.parent() {
        dirs.push(parent.to_path_buf());
    }
    if let Some(app) = app {
        if let Ok(resource_dir) = app.path().resource_dir() {
            dirs.push(resource_dir.clone());
            dirs.push(resource_dir.join("binaries"));
        }
    }
    if let Ok(current_exe) = env::current_exe() {
        if let Some(parent) = current_exe.parent() {
            dirs.push(parent.to_path_buf());
            dirs.push(parent.join("binaries"));
        }
    }

    dirs.sort();
    dirs.dedup();
    dirs
}

fn local_source_path_for_model(cache_config: &ShardCacheConfig, model_id: &str) -> Option<PathBuf> {
    let cache = ShardCache::new(cache_config.clone()).ok()?;
    let records = cache.list().ok()?;

    for record in records {
        let payload = cache.read_payload(&record.info).ok()?;
        let manifest = serde_json::from_slice::<SeedShardManifest>(&payload).ok()?;
        if manifest.model_id == model_id {
            let path = PathBuf::from(manifest.source.path);
            if path.is_file() {
                return Some(path);
            }
        }
    }

    None
}

fn find_llama_cli(app: Option<&AppHandle>) -> Option<PathBuf> {
    if let Ok(path) = env::var("INFERNET_LLAMA_CLI") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
    }

    for candidate in bundled_llama_cli_candidates(app) {
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    let executable_names = if cfg!(windows) {
        vec!["llama-cli.exe", "llama.exe", "main.exe"]
    } else {
        vec!["llama-cli", "llama", "main"]
    };
    let mut search_dirs = Vec::new();
    if let Ok(current_exe) = env::current_exe() {
        if let Some(parent) = current_exe.parent() {
            search_dirs.push(parent.to_path_buf());
        }
    }
    if let Some(path) = env::var_os("PATH") {
        search_dirs.extend(env::split_paths(&path));
    }

    for directory in search_dirs {
        for name in &executable_names {
            let candidate = directory.join(*name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }

    None
}

fn bundled_llama_cli_candidates(app: Option<&AppHandle>) -> Vec<PathBuf> {
    let executable_name = if cfg!(windows) {
        "llama-cli.exe"
    } else {
        "llama-cli"
    };
    let sidecar_name = bundled_llama_cli_sidecar_name();
    let mut candidates = Vec::new();

    if let Some(app) = app {
        if let Ok(resource_dir) = app.path().resource_dir() {
            candidates.push(resource_dir.join(executable_name));
            candidates.push(resource_dir.join("binaries").join(executable_name));
            if let Some(sidecar_name) = sidecar_name {
                candidates.push(resource_dir.join(sidecar_name));
                candidates.push(resource_dir.join("binaries").join(sidecar_name));
            }
        }
    }

    if let Ok(current_exe) = env::current_exe() {
        if let Some(parent) = current_exe.parent() {
            candidates.push(parent.join(executable_name));
            candidates.push(parent.join("binaries").join(executable_name));
            if let Some(resources) = parent.parent().map(|path| path.join("Resources")) {
                candidates.push(resources.join(executable_name));
                candidates.push(resources.join("binaries").join(executable_name));
            }
        }
    }

    if let Some(sidecar_name) = sidecar_name {
        candidates.push(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("binaries")
                .join(sidecar_name),
        );
    }

    candidates
}

fn bundled_llama_cli_sidecar_name() -> Option<&'static str> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        return Some("llama-cli-aarch64-apple-darwin");
    }
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    {
        return Some("llama-cli-x86_64-apple-darwin");
    }
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    {
        return Some("llama-cli-x86_64-pc-windows-msvc.exe");
    }
    #[cfg(all(target_os = "windows", target_arch = "aarch64"))]
    {
        return Some("llama-cli-aarch64-pc-windows-msvc.exe");
    }
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        return Some("llama-cli-x86_64-unknown-linux-gnu");
    }
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        return Some("llama-cli-aarch64-unknown-linux-gnu");
    }
    #[allow(unreachable_code)]
    None
}

fn local_gguf_size_limit_bytes() -> u64 {
    static LIMIT: OnceLock<u64> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        env::var("INFERNET_MAX_LOCAL_GGUF_BYTES")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or_else(|| {
                total_system_memory_bytes()
                    .map(|bytes| (bytes.saturating_mul(45) / 100).max(MIN_LOCAL_GGUF_LIMIT_BYTES))
                    .unwrap_or(MIN_LOCAL_GGUF_LIMIT_BYTES)
            })
    })
}

#[cfg(target_os = "macos")]
fn total_system_memory_bytes() -> Option<u64> {
    let output = std::process::Command::new("sysctl")
        .arg("-n")
        .arg("hw.memsize")
        .output()
        .ok()?;
    String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse::<u64>()
        .ok()
}

#[cfg(target_os = "linux")]
fn total_system_memory_bytes() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = meminfo.lines().find(|line| line.starts_with("MemTotal:"))?;
    let kib = line
        .split_whitespace()
        .nth(1)
        .and_then(|value| value.parse::<u64>().ok())?;
    Some(kib.saturating_mul(1024))
}

#[cfg(target_os = "windows")]
fn total_system_memory_bytes() -> Option<u64> {
    let output = std::process::Command::new("powershell")
        .args([
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
        ])
        .output()
        .ok()?;
    String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse::<u64>()
        .ok()
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
fn total_system_memory_bytes() -> Option<u64> {
    None
}

fn format_bytes(bytes: u64) -> String {
    let gib = bytes as f64 / 1024.0 / 1024.0 / 1024.0;
    if gib >= 1.0 {
        format!("{gib:.1} GB")
    } else {
        let mib = bytes as f64 / 1024.0 / 1024.0;
        format!("{mib:.0} MB")
    }
}

async fn replay_route_progress(
    app: &AppHandle,
    trace_id: &str,
    route: &[RouteHopView],
    hidden_size: usize,
) {
    let activation_size_bytes = hidden_size.saturating_mul(2);

    for (index, hop) in route.iter().enumerate() {
        emit_progress(
            app,
            ProgressEvent::HopStarted {
                trace_id: trace_id.to_owned(),
                peer_id: hop.peer_id.clone(),
                short_peer_id: hop.short_peer_id.clone(),
                layer_start: hop.layer_start,
                layer_end: hop.layer_end,
                activation_size_bytes,
            },
        );
        tokio::time::sleep(Duration::from_millis(120)).await;
        emit_progress(
            app,
            ProgressEvent::HopCompleted {
                trace_id: trace_id.to_owned(),
                peer_id: hop.peer_id.clone(),
                short_peer_id: hop.short_peer_id.clone(),
                layer_start: hop.layer_start,
                layer_end: hop.layer_end,
                next_peer_id: route.get(index + 1).map(|next| next.peer_id.clone()),
                activation_size_bytes,
                timing_ms: 120,
                activation_checksum: format!("{:016x}", route_progress_checksum(hop, index)),
            },
        );
    }
}

fn route_progress_checksum(hop: &RouteHopView, index: usize) -> u64 {
    hop.peer_id
        .bytes()
        .fold(0xcbf29ce484222325 ^ index as u64, |hash, byte| {
            hash.wrapping_mul(0x100000001b3) ^ u64::from(byte)
        })
        ^ u64::from(hop.layer_start)
        ^ (u64::from(hop.layer_end) << 32)
}

fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

fn add_model_response_from_summary(summary: SeededModelSummary) -> AddModelResponse {
    let display_name = display_name_from_source_path(&summary.source_path)
        .filter(|name| !name.is_empty())
        .unwrap_or(summary.display_name);

    AddModelResponse {
        model_id: summary.model_id,
        display_name,
        source: summary.source_path.display().to_string(),
        source_checksum: summary.source_checksum,
        source_size_bytes: summary.source_size_bytes,
        planned_shards: summary.shard_count,
        metadata_only: summary.metadata_only,
        installed_shards: summary
            .records
            .into_iter()
            .map(|record| InstalledShardView {
                model_id: record.info.model_id,
                layer_start: record.info.layers.start,
                layer_end: record.info.layers.end,
                checksum: record.info.checksum,
                size_bytes: record.info.size_bytes,
                version: record.info.version,
            })
            .collect(),
        message: if summary.metadata_only {
            "Model seed records are being shared. Physical GGUF tensor shards still require the llama.cpp shard writer.".to_owned()
        } else {
            "Model shards are installed and seeding to the network.".to_owned()
        },
    }
}

fn display_name_from_source_path(path: &Path) -> Option<String> {
    let file_name = path.file_name()?.to_str()?;
    let without_ext = file_name
        .strip_suffix(".gguf")
        .or_else(|| file_name.strip_suffix(".GGUF"))
        .unwrap_or(file_name);
    let name = without_ext
        .replace(['_', '-', '.'], " ")
        .split_whitespace()
        .map(format_model_name_part)
        .collect::<Vec<_>>()
        .join(" ");

    (!name.is_empty()).then_some(name)
}

fn display_name_from_model_id(model_id: &str) -> String {
    let name = model_id
        .replace(['_', '-', '.'], " ")
        .split_whitespace()
        .map(format_model_name_part)
        .collect::<Vec<_>>()
        .join(" ");

    if name.is_empty() {
        "Imported GGUF Model".to_owned()
    } else {
        name
    }
}

fn model_id_from_source_path(path: &Path) -> String {
    let name = path
        .file_stem()
        .and_then(|value| value.to_str())
        .or_else(|| path.file_name().and_then(|value| value.to_str()))
        .unwrap_or("gguf-model");
    model_id_from_name(name)
}

fn model_id_from_name(name: &str) -> String {
    let mut output = String::new();
    let mut last_was_separator = false;

    for ch in name.chars().flat_map(|ch| ch.to_lowercase()) {
        if ch.is_ascii_alphanumeric() || ch == '.' {
            output.push(ch);
            last_was_separator = false;
        } else if !last_was_separator && !output.is_empty() {
            output.push('-');
            last_was_separator = true;
        }
    }

    while output.ends_with('-') {
        output.pop();
    }

    if output.is_empty() {
        "gguf-model".to_owned()
    } else {
        output
    }
}

fn manifest_from_gguf_source(source: &Path) -> anyhow::Result<ModelManifest> {
    let info = parse_gguf_info(source)?;
    let display_name =
        display_name_from_source_path(source).unwrap_or_else(|| "Imported GGUF Model".to_owned());
    let model_id = model_id_from_source_path(source);
    Ok(ModelManifest::from_gguf_info(
        model_id,
        display_name,
        &info,
    )?)
}

fn format_model_name_part(part: &str) -> String {
    let lower = part.to_ascii_lowercase();
    match lower.as_str() {
        "gguf" => "GGUF".to_owned(),
        "llama" => "Llama".to_owned(),
        "gemma" => "Gemma".to_owned(),
        "qwen" => "Qwen".to_owned(),
        "mistral" => "Mistral".to_owned(),
        "instruct" => "Instruct".to_owned(),
        "it" => "IT".to_owned(),
        "q4" | "q5" | "q6" | "q8" | "k" | "m" | "s" => lower.to_ascii_uppercase(),
        value
            if value.ends_with('b')
                && value[..value.len() - 1]
                    .chars()
                    .all(|ch| ch.is_ascii_digit()) =>
        {
            value.to_ascii_uppercase()
        }
        _ => part.to_owned(),
    }
}

fn app_data_dir(app: &AppHandle) -> PathBuf {
    app.path()
        .app_data_dir()
        .unwrap_or_else(|_| PathBuf::from(".infernet"))
}

fn cache_config_for_app(app: &AppHandle) -> ShardCacheConfig {
    let config = ShardCacheConfig::new(app_data_dir(app).join("shards"));
    if let Err(error) = migrate_legacy_caches(&config) {
        eprintln!("failed to migrate legacy Infernet cache: {error}");
    }
    config
}

fn migrate_legacy_caches(target_config: &ShardCacheConfig) -> anyhow::Result<usize> {
    migrate_legacy_cache_roots(target_config, legacy_cache_roots())
}

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

fn legacy_cache_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    roots.push(PathBuf::from(".infernet/shards"));
    if let Ok(current_dir) = env::current_dir() {
        roots.push(current_dir.join(".infernet/shards"));
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    roots.push(manifest_dir.join(".infernet/shards"));
    roots.push(manifest_dir.join("../../.infernet/shards"));

    roots.sort();
    roots.dedup();
    roots
}

fn same_path(left: &Path, right: &Path) -> bool {
    match (std::fs::canonicalize(left), std::fs::canonicalize(right)) {
        (Ok(left), Ok(right)) => left == right,
        _ => left == right,
    }
}

fn ensure_model_distribution_service(
    state: &State<'_, UiState>,
    cache_config: ShardCacheConfig,
) -> Result<(), String> {
    let mut started = state
        .model_distribution_started
        .lock()
        .map_err(|_| "failed to lock model distribution service state".to_owned())?;
    if *started {
        return Ok(());
    }

    let keypair = state
        .keypair
        .lock()
        .map_err(|_| "failed to lock local node identity".to_owned())?
        .clone();
    let mut discovery = DiscoveryConfig::new(state.topic.clone());
    discovery.keypair = keypair;
    discovery.p2p_listen = format!("/ip4/0.0.0.0/tcp/{UI_LISTEN_PORT}");
    discovery.static_peers = configured_static_peers(state)?;
    *started = true;

    tauri::async_runtime::spawn(async move {
        if let Err(error) = run_model_distribution_node(discovery, cache_config).await {
            eprintln!("model distribution node stopped: {error}");
        }
    });

    Ok(())
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

fn peer_address_labels(advertisement: &NodeAdvertisement) -> Vec<String> {
    advertisement
        .addresses
        .iter()
        .map(|address| format!("{}@{}", advertisement.peer_id, address))
        .collect()
}

fn local_connect_addresses(peer_id: &str) -> Vec<String> {
    let mut addresses = Vec::new();
    if let Some(ip) = preferred_lan_ip() {
        addresses.push(format!(
            "/{}/{}/tcp/{}/p2p/{}",
            ip_protocol(ip),
            ip,
            UI_LISTEN_PORT,
            peer_id
        ));
    }
    addresses.push(format!("/ip4/127.0.0.1/tcp/{UI_LISTEN_PORT}/p2p/{peer_id}"));
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

fn ip_protocol(ip: IpAddr) -> &'static str {
    match ip {
        IpAddr::V4(_) => "ip4",
        IpAddr::V6(_) => "ip6",
    }
}

fn huggingface_settings_from_token(token: Option<&str>) -> HuggingFaceSettings {
    HuggingFaceSettings {
        has_token: token.is_some_and(|token| !token.is_empty()),
        token_preview: token.map(mask_token),
    }
}

fn mask_token(token: &str) -> String {
    if token.len() <= 8 {
        return "saved".to_owned();
    }

    format!("{}...{}", &token[..4], &token[token.len() - 4..])
}

fn apply_huggingface_auth(
    request: reqwest::RequestBuilder,
    state: &State<'_, UiState>,
    token_override: Option<&str>,
) -> Result<reqwest::RequestBuilder, String> {
    let token = match token_override
        .map(str::trim)
        .filter(|token| !token.is_empty())
    {
        Some(token) => Some(token.to_owned()),
        None => state
            .huggingface_token
            .lock()
            .map_err(|_| "failed to lock Hugging Face settings".to_owned())?
            .clone(),
    };

    Ok(match token {
        Some(token) => request.bearer_auth(token),
        None => request,
    })
}

fn huggingface_download_path(
    app: &AppHandle,
    repo_id: &str,
    filename: &str,
) -> Result<PathBuf, String> {
    let file_name = Path::new(filename)
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| "Hugging Face filename is invalid".to_owned())?;

    Ok(app_data_dir(app)
        .join("imports")
        .join(sanitize_path_segment(repo_id))
        .join(sanitize_path_segment(file_name)))
}

fn sanitize_path_segment(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' || ch == '.' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn manifest_for_model(
    model_id: Option<&str>,
    cache_config: &ShardCacheConfig,
    registry: Option<&ShardRegistry>,
) -> anyhow::Result<ModelManifest> {
    let mut available = installed_model_manifests(cache_config);
    if let Some(registry) = registry {
        available.extend(discovered_model_manifests(registry));
    }
    available.sort_by(|left, right| left.model_id.cmp(&right.model_id));
    available.dedup_by(|left, right| left.model_id == right.model_id);

    let requested = model_id.map(str::trim).filter(|value| !value.is_empty());
    let Some(model_id) = requested else {
        return available
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("no models are installed or discovered yet"));
    };

    available
        .into_iter()
        .find(|manifest| manifest.model_id == model_id)
        .ok_or_else(|| {
            let supported = available_model_views(cache_config, registry)
                .into_iter()
                .map(|model| model.model_id)
                .collect::<Vec<_>>()
                .join(", ");
            if supported.is_empty() {
                anyhow::anyhow!(
                    "unknown model {model_id}; no models are installed or discovered yet"
                )
            } else {
                anyhow::anyhow!("unknown model {model_id}; available models are {supported}")
            }
        })
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

        tokio::time::sleep(Duration::from_millis(140)).await;

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
        .plugin(tauri_plugin_dialog::init())
        .manage(UiState::default())
        .setup(|app| {
            let state = app.state::<UiState>();
            let cache_config = cache_config_for_app(app.handle());
            ensure_model_distribution_service(&state, cache_config)
                .map_err(|error| Box::<dyn std::error::Error>::from(error))?;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_local_identity,
            get_manual_peers,
            add_manual_peer,
            clear_manual_peers,
            get_grid_snapshot,
            run_demo_inference,
            add_local_gguf_model,
            get_huggingface_settings,
            save_huggingface_token,
            clear_huggingface_token,
            inspect_huggingface_repo,
            add_huggingface_model
        ])
        .run(tauri::generate_context!())
        .expect("error while running Infernet UI");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_cache_does_not_publish_builtin_models() {
        let root = std::env::temp_dir().join(format!("infernet-ui-empty-{}", unix_ms()));
        let cache_config = ShardCacheConfig::new(root.clone());

        assert!(available_model_views(&cache_config, None).is_empty());
        assert!(manifest_for_model(None, &cache_config, None).is_err());

        let _ = std::fs::remove_dir_all(root);
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
    fn local_seed_record_with_source_file_advertises_route_and_quantization() {
        let root = std::env::temp_dir().join(format!("infernet-ui-local-gguf-{}", unix_ms()));
        let cache_config = ShardCacheConfig::new(root.join("shards"));
        let cache = ShardCache::new(cache_config.clone()).unwrap();
        let source = root.join("gemma-4-12b-it-IQ4_XS.gguf");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(&source, b"local gguf placeholder").unwrap();

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
            payload_kind: "metadata-only".to_owned(),
        };
        cache
            .import_payload(
                serde_json::to_vec(&seed_manifest).unwrap(),
                seed_manifest.model_id.clone(),
                layers,
                "v1",
            )
            .unwrap();

        let advertisement =
            local_cache_advertisement(&cache_config, "local-peer".to_owned()).unwrap();
        assert_eq!(advertisement.hosted_shards.len(), 1);
        assert_eq!(advertisement.hosted_shards[0].layers, layers);

        let mut registry = ShardRegistry::new();
        registry.upsert(advertisement);
        let manifest = manifest_for_model(
            Some("gemma-4-12b-it-iq4-xs"),
            &cache_config,
            Some(&registry),
        )
        .unwrap();
        let route = registry.route_for_model(&manifest).unwrap();
        assert_eq!(route.len(), 1);
        assert_eq!(route[0].layers, layers);

        let view = available_model_views(&cache_config, Some(&registry))
            .into_iter()
            .find(|model| model.model_id == "gemma-4-12b-it-iq4-xs")
            .unwrap();
        assert_eq!(view.quantization.as_deref(), Some("IQ4_XS"));

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn advertised_seed_manifest_installs_verified_model_record() {
        let root = std::env::temp_dir().join(format!("infernet-ui-advertised-{}", unix_ms()));
        let cache_config = ShardCacheConfig::new(root.join("shards"));
        let cache = ShardCache::new(cache_config.clone()).unwrap();
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
            payload_kind: "metadata-only".to_owned(),
        };
        let payload = serde_json::to_vec_pretty(&seed_manifest).unwrap();
        let info = ModelShardInfo {
            model_id: seed_manifest.model_id.clone(),
            layers,
            checksum: sha256_bytes(&payload),
            size_bytes: payload.len() as u64,
            version: "v1".to_owned(),
            protocol_version: PROTOCOL_VERSION,
        };

        install_advertised_seed_record(&cache, &info, &seed_manifest).unwrap();

        let installed = cache
            .find(
                &info.model_id,
                layers,
                Some(&info.checksum),
                Some(&info.version),
            )
            .unwrap()
            .expect("advertised record should be installed");
        assert_eq!(installed.info, info);
        assert_eq!(cache.read_payload(&installed.info).unwrap(), payload);

        let _ = std::fs::remove_dir_all(root);
    }
}
