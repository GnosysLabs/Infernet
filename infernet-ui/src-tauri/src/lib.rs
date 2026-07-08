use std::time::Duration;
use std::{
    collections::BTreeMap,
    env,
    fs::File,
    io::Write,
    net::{IpAddr, UdpSocket},
    path::{Path, PathBuf},
    sync::Mutex,
    time::{SystemTime, UNIX_EPOCH},
};

use futures::StreamExt;
use infernet_model::{
    LayerRange, ModelManifest, RuntimeKind, SeedShardManifest, ShardDescriptor,
    gguf::parse_gguf_info,
};
use infernet_node::{
    DiscoveryConfig, SeededModelSummary, ShardCache, ShardCacheConfig, discover_for,
    empty_advertisement, import_seed_model_from_file_with_progress, infer_over_libp2p,
    run_model_distribution_node,
};
use infernet_protocol::{NodeAdvertisement, PROTOCOL_VERSION, RouteHop, TraceEvent};
use infernet_router::ShardRegistry;
use libp2p::{Multiaddr, PeerId, identity};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, State};
use tokio::process::Command;

const DEFAULT_TOPIC: &str = "infernet/grid-demo/1";
const DEFAULT_DISCOVERY_TIMEOUT_MS: u64 = 4_000;
const DEFAULT_INFERENCE_TIMEOUT_MS: u64 = 6_000;
const MAX_SAFE_LOCAL_GGUF_BYTES: u64 = 3 * 1024 * 1024 * 1024;
const UI_LISTEN_PORT: u16 = 9777;

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
    if local_cache_has_shards(&cache_config) {
        ensure_model_distribution_service(&state, cache_config.clone())?;
    }

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
    if local_cache_has_shards(&cache_config) {
        ensure_model_distribution_service(&state, cache_config.clone())?;
    }
    let manifest = manifest_for_model(model_id.as_deref(), &cache_config, None)
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

    emit_progress(
        &app,
        ProgressEvent::RouteDiscovered {
            route: snapshot.route.clone(),
        },
    );

    if manifest.runtime_kind != RuntimeKind::Demo {
        let trace_id = format!("llama-{}", unix_ms());
        replay_route_progress(&app, &trace_id, &snapshot.route, manifest.hidden_size).await;
        let output = generate_with_llama_cli(&cache_config, &manifest, &prompt).await?;
        emit_progress(
            &app,
            ProgressEvent::FinalOutput {
                trace_id: trace_id.clone(),
                output: output.clone(),
            },
        );
        return Ok(RunDemoResponse {
            output,
            trace_id,
            snapshot,
        });
    }

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
    let (mut config, local_peer_id) = discovery_config_from_state(state)?;
    let cache_config = cache_config_for_app(app);
    let topic = config.topic.clone();
    if let Some(local_advertisement) =
        local_cache_advertisement(&cache_config, local_peer_id.clone())
    {
        config.advertisement = Some(local_advertisement);
    }
    let registry = discover_for(config, Duration::from_millis(discovery_timeout_ms))
        .await
        .map_err(|error| error.to_string())?;
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

    Ok(snapshot_from_registry(
        local_peer_id,
        topic,
        &manifest,
        &registry,
        &cache_config,
    ))
}

fn snapshot_from_registry(
    local_peer_id: String,
    topic: String,
    manifest: &ModelManifest,
    registry: &ShardRegistry,
    cache_config: &ShardCacheConfig,
) -> GridSnapshot {
    let advertisements =
        ui_visible_advertisements(registry.advertisements(), Some(&manifest.model_id));
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
    let advertisements = ui_visible_advertisements(registry.advertisements(), None);
    GridSnapshot {
        local_peer_id,
        topic,
        selected_model: String::new(),
        available_models: available_model_views(cache_config, Some(registry)),
        layer_count: 0,
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
            && source_path_for_model(cache_config, &manifest.model_id)
                .and_then(|path| std::fs::metadata(path).ok())
                .is_some_and(|metadata| metadata.len() <= MAX_SAFE_LOCAL_GGUF_BYTES)
            && find_llama_cli().is_some());
    ModelView {
        model_id: manifest.model_id.clone(),
        display_name: manifest.display_name.clone(),
        runtime_kind: manifest.runtime_kind.as_str().to_owned(),
        layer_count: manifest.layer_count,
        activation_dtype: manifest.activation_dtype.clone(),
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

    let Some(source_path) = source_path_for_model(cache_config, &manifest.model_id) else {
        return "Installed for sharing. Token execution is not connected yet.".to_owned();
    };
    if std::fs::metadata(source_path)
        .map(|metadata| metadata.len() > MAX_SAFE_LOCAL_GGUF_BYTES)
        .unwrap_or(false)
    {
        return "Installed for sharing. This model is too large for local fallback execution."
            .to_owned();
    }

    "Installed for sharing. Configure the split GGUF runtime to chat.".to_owned()
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
        hosted_shards.push(ShardDescriptor {
            model_id: manifest.model_id,
            layers: manifest.layers,
            runtime_kind: manifest.runtime_kind,
            tokenizer: Some(manifest.tokenizer),
            metadata: Some(manifest.metadata),
            shard_hash: Some(manifest.shard_hash),
        });
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

async fn generate_with_llama_cli(
    cache_config: &ShardCacheConfig,
    manifest: &ModelManifest,
    prompt: &str,
) -> Result<String, String> {
    let model_path = source_path_for_model(cache_config, &manifest.model_id).ok_or_else(|| {
        format!(
            "{} is installed, but the source GGUF path is missing",
            manifest.display_name
        )
    })?;
    let llama_cli = find_llama_cli().ok_or_else(|| {
        "Token generation is not connected yet. Set INFERNET_LLAMA_CLI to a llama.cpp binary for small local GGUF tests; split GGUF token execution still needs the Infernet runtime bridge.".to_owned()
    })?;
    let model_size = std::fs::metadata(&model_path)
        .map_err(|error| format!("failed to read {}: {error}", model_path.display()))?
        .len();
    let allow_large = env::var("INFERNET_ALLOW_LARGE_GGUF").as_deref() == Ok("1");
    if model_size > MAX_SAFE_LOCAL_GGUF_BYTES && !allow_large {
        return Err(format!(
            "{} is {}. Infernet will not full-load models over {} in the UI. This model needs the split GGUF token runtime, not a local llama.cpp fallback.",
            model_path.display(),
            format_bytes(model_size),
            format_bytes(MAX_SAFE_LOCAL_GGUF_BYTES),
        ));
    }

    let output = Command::new(&llama_cli)
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

fn source_path_for_model(cache_config: &ShardCacheConfig, model_id: &str) -> Option<PathBuf> {
    let cache = ShardCache::new(cache_config.clone()).ok()?;
    let records = cache.list().ok()?;

    for record in records {
        let payload = cache.read_payload(&record.info).ok()?;
        let manifest = serde_json::from_slice::<SeedShardManifest>(&payload).ok()?;
        if manifest.model_id == model_id {
            return Some(PathBuf::from(manifest.source.path));
        }
    }

    None
}

fn find_llama_cli() -> Option<PathBuf> {
    if let Ok(path) = env::var("INFERNET_LLAMA_CLI") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
    }

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

fn local_cache_has_shards(cache_config: &ShardCacheConfig) -> bool {
    ShardCache::new(cache_config.clone())
        .and_then(|cache| cache.list())
        .map(|records| !records.is_empty())
        .unwrap_or(false)
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
    discovery.static_peers = state
        .manual_peers
        .lock()
        .map_err(|_| "failed to lock manual peers".to_owned())?
        .clone();
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
    config.static_peers = state
        .manual_peers
        .lock()
        .map_err(|_| "failed to lock manual peers".to_owned())?
        .clone();

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
            if local_cache_has_shards(&cache_config) {
                ensure_model_distribution_service(&state, cache_config)
                    .map_err(|error| Box::<dyn std::error::Error>::from(error))?;
            }
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
}
