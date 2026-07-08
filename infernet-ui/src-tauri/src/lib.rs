use std::time::Duration;
use std::{
    collections::BTreeMap,
    fs::File,
    io::Write,
    path::{Path, PathBuf},
    sync::Mutex,
};

use futures::StreamExt;
use infernet_model::{LayerRange, ModelManifest};
use infernet_node::{
    DiscoveryConfig, SeededModelSummary, ShardCache, ShardCacheConfig, discover_for,
    import_seed_model_from_file, infer_over_libp2p, run_model_distribution_node,
};
use infernet_protocol::{NodeAdvertisement, RouteHop, TraceEvent};
use infernet_router::ShardRegistry;
use libp2p::identity;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, State};

const DEFAULT_TOPIC: &str = "infernet/grid-demo/1";
const DEFAULT_DISCOVERY_TIMEOUT_MS: u64 = 4_000;
const DEFAULT_INFERENCE_TIMEOUT_MS: u64 = 6_000;

struct UiState {
    keypair: Mutex<identity::Keypair>,
    topic: String,
    huggingface_token: Mutex<Option<String>>,
    model_distribution_started: Mutex<bool>,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            keypair: Mutex::new(identity::Keypair::generate_ed25519()),
            topic: DEFAULT_TOPIC.to_owned(),
            huggingface_token: Mutex::new(None),
            model_distribution_started: Mutex::new(false),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct LocalIdentity {
    peer_id: String,
    topic: String,
    listen: String,
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

    Ok(LocalIdentity {
        peer_id,
        topic,
        listen: "/ip4/0.0.0.0/tcp/0".to_owned(),
    })
}

#[tauri::command]
async fn get_grid_snapshot(
    state: State<'_, UiState>,
    discovery_timeout_ms: Option<u64>,
    model_id: Option<String>,
) -> Result<GridSnapshot, String> {
    if local_cache_has_shards() {
        ensure_model_distribution_service(&state)?;
    }

    collect_snapshot(
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
    let manifest = manifest_for_model(model_id.as_deref()).map_err(|error| error.to_string())?;
    let snapshot = collect_snapshot(
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
    state: State<'_, UiState>,
    model_id: String,
    path: String,
    version: Option<String>,
) -> Result<AddModelResponse, String> {
    let manifest = manifest_for_model(Some(&model_id)).map_err(|error| error.to_string())?;
    let cache = ShardCache::new(default_cache_config()).map_err(|error| error.to_string())?;
    let source = PathBuf::from(path);
    let summary = import_seed_model_from_file(
        &cache,
        &source,
        &manifest,
        version.unwrap_or_else(|| "v1".to_owned()),
    )
    .map_err(|error| error.to_string())?;
    ensure_model_distribution_service(&state)?;

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
    state: State<'_, UiState>,
    repo_id: String,
    filename: String,
    model_id: String,
    token: Option<String>,
    revision: Option<String>,
    version: Option<String>,
) -> Result<AddModelResponse, String> {
    let repo_id = repo_id.trim();
    let filename = filename.trim();
    if repo_id.is_empty() || filename.is_empty() {
        return Err("choose a Hugging Face repo and GGUF file".to_owned());
    }

    let manifest = manifest_for_model(Some(&model_id)).map_err(|error| error.to_string())?;
    let revision = revision.unwrap_or_else(|| "main".to_owned());
    let target = huggingface_download_path(repo_id, filename)?;
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }

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

    let mut file = File::create(&target).map_err(|error| error.to_string())?;
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|error| error.to_string())?;
        file.write_all(&chunk).map_err(|error| error.to_string())?;
    }

    let cache = ShardCache::new(default_cache_config()).map_err(|error| error.to_string())?;
    let summary = import_seed_model_from_file(
        &cache,
        &target,
        &manifest,
        version.unwrap_or_else(|| "v1".to_owned()),
    )
    .map_err(|error| error.to_string())?;
    ensure_model_distribution_service(&state)?;

    Ok(add_model_response_from_summary(summary))
}

async fn collect_snapshot(
    state: &State<'_, UiState>,
    discovery_timeout_ms: u64,
    model_id: Option<&str>,
) -> Result<GridSnapshot, String> {
    let (config, local_peer_id) = discovery_config_from_state(state)?;
    let topic = config.topic.clone();
    let manifest = manifest_for_model(model_id).map_err(|error| error.to_string())?;
    let registry = discover_for(config, Duration::from_millis(discovery_timeout_ms))
        .await
        .map_err(|error| error.to_string())?;

    Ok(snapshot_from_registry(
        local_peer_id,
        topic,
        &manifest,
        &registry,
    ))
}

fn snapshot_from_registry(
    local_peer_id: String,
    topic: String,
    manifest: &ModelManifest,
    registry: &ShardRegistry,
) -> GridSnapshot {
    let advertisements = registry.advertisements();
    let route_result = registry.route_for_model(manifest);
    let (route, missing_ranges) = match route_result {
        Ok(route) => (route, None),
        Err(error) => (Vec::new(), Some(error.to_string())),
    };

    GridSnapshot {
        local_peer_id,
        topic,
        selected_model: manifest.model_id.clone(),
        available_models: ModelManifest::catalog()
            .iter()
            .map(model_view_from_manifest)
            .collect(),
        layer_count: manifest.layer_count,
        peers: advertisements
            .iter()
            .map(peer_view_from_advertisement)
            .collect(),
        route: route.iter().map(route_hop_view).collect(),
        missing_ranges,
        coverage: build_coverage(manifest, &route, &advertisements),
        distribution: build_distribution_snapshot(&advertisements),
    }
}

fn model_view_from_manifest(manifest: &ModelManifest) -> ModelView {
    ModelView {
        model_id: manifest.model_id.clone(),
        display_name: manifest.display_name.clone(),
        runtime_kind: manifest.runtime_kind.as_str().to_owned(),
        layer_count: manifest.layer_count,
        activation_dtype: manifest.activation_dtype.clone(),
    }
}

fn peer_view_from_advertisement(advertisement: &NodeAdvertisement) -> PeerView {
    PeerView {
        peer_id: advertisement.peer_id.clone(),
        short_peer_id: short_peer_id(&advertisement.peer_id),
        addresses: advertisement.addresses.clone(),
        protocol_version: advertisement.protocol_version,
        shards: advertisement
            .hosted_shards
            .iter()
            .map(|shard| ShardView {
                model_id: shard.model_id.clone(),
                layer_start: shard.layers.start,
                layer_end: shard.layers.end,
            })
            .collect(),
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

fn build_distribution_snapshot(advertisements: &[NodeAdvertisement]) -> DistributionSnapshot {
    let cache = ShardCache::new(default_cache_config());
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

fn add_model_response_from_summary(summary: SeededModelSummary) -> AddModelResponse {
    AddModelResponse {
        model_id: summary.model_id,
        display_name: summary.display_name,
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

fn default_cache_config() -> ShardCacheConfig {
    ShardCacheConfig::new(".infernet/shards")
}

fn local_cache_has_shards() -> bool {
    ShardCache::new(default_cache_config())
        .and_then(|cache| cache.list())
        .map(|records| !records.is_empty())
        .unwrap_or(false)
}

fn ensure_model_distribution_service(state: &State<'_, UiState>) -> Result<(), String> {
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
    *started = true;

    tokio::spawn(async move {
        if let Err(error) = run_model_distribution_node(discovery, default_cache_config()).await {
            eprintln!("model distribution node stopped: {error}");
        }
    });

    Ok(())
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

fn huggingface_download_path(repo_id: &str, filename: &str) -> Result<PathBuf, String> {
    let file_name = Path::new(filename)
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| "Hugging Face filename is invalid".to_owned())?;

    Ok(PathBuf::from(".infernet")
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

fn manifest_for_model(model_id: Option<&str>) -> anyhow::Result<ModelManifest> {
    let model_id = model_id.unwrap_or("grid-demo-12");
    ModelManifest::by_id(model_id).ok_or_else(|| {
        let supported = ModelManifest::catalog()
            .into_iter()
            .map(|manifest| manifest.model_id)
            .collect::<Vec<_>>()
            .join(", ");
        anyhow::anyhow!("unknown model {model_id}; supported models are {supported}")
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
        .invoke_handler(tauri::generate_handler![
            get_local_identity,
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
