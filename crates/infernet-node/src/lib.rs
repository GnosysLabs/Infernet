pub mod model_distribution;

use std::collections::HashMap;
use std::env;
use std::io::{Read, Seek, SeekFrom, Write};
use std::mem;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;
use std::{fs, io};

use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use futures::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, StreamExt};
use infernet_model::{LayerRange, ModelManifest, RuntimeKind, SeedShardManifest, ShardDescriptor};
use infernet_protocol::{
    ACTIVATION_PROTOCOL, ActivationRequest, ActivationResponse, MODEL_BLOB_PROTOCOL,
    MODEL_PROTOCOL, ModelBlobRequest, ModelBlobResponse, ModelShardInfo, ModelShardRequest,
    ModelShardResponse, NodeAdvertisement, PROTOCOL_VERSION, PromptMetadata, RouteHop, TraceEvent,
};
use infernet_router::ShardRegistry;
use infernet_runtime::{DemoRuntime, LayerRuntime, activation_checksum};
use libp2p::{
    Multiaddr, PeerId, StreamProtocol, Swarm, SwarmBuilder,
    core::connection::ConnectedPoint,
    gossipsub, identity, mdns,
    multiaddr::Protocol,
    noise, request_response,
    swarm::{NetworkBehaviour, SwarmEvent},
    tcp, yamux,
};
pub use model_distribution::{
    CachedShardRecord, PAYLOAD_KIND_GGUF_SHARD, PAYLOAD_KIND_METADATA_ONLY, SeedShardBuildProgress,
    SeededModelSummary, ShardCache, ShardCacheConfig, ShardCacheStats,
    executable_source_path_for_manifest, import_seed_model_from_file,
    import_seed_model_from_file_with_build_progress, import_seed_model_from_file_with_progress,
    is_executable_shard_record, sha256_bytes, sha256_file, source_cache_path, source_cache_root,
};
use serde::Deserialize;
use tokio::time::{Instant, interval, sleep};

#[derive(Debug, Clone)]
pub struct WorkerConfig {
    pub peer_id: String,
    pub model_id: String,
    pub runtime_kind: RuntimeKind,
    pub owned_layers: LayerRange,
    pub hidden_size: usize,
    pub shard_cache: Option<ShardCacheConfig>,
}

#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    pub keypair: identity::Keypair,
    pub topic: String,
    pub p2p_listen: String,
    pub advertisement: Option<NodeAdvertisement>,
    pub static_peers: Vec<NodeAdvertisement>,
    pub publish_interval: Duration,
    pub advertise_listen_addresses: bool,
    pub dial_discovered_peers: bool,
    pub relay_advertisements: bool,
}

impl DiscoveryConfig {
    pub fn new(topic: impl Into<String>) -> Self {
        Self {
            keypair: identity::Keypair::generate_ed25519(),
            topic: topic.into(),
            p2p_listen: "/ip4/0.0.0.0/tcp/0".to_owned(),
            advertisement: None,
            static_peers: Vec::new(),
            publish_interval: Duration::from_millis(750),
            advertise_listen_addresses: true,
            dial_discovered_peers: true,
            relay_advertisements: false,
        }
    }

    pub fn peer_id(&self) -> PeerId {
        self.keypair.public().to_peer_id()
    }
}

pub fn load_or_generate_keypair(path: impl AsRef<Path>) -> Result<identity::Keypair> {
    let path = path.as_ref();
    match fs::read(path) {
        Ok(bytes) => identity::Keypair::from_protobuf_encoding(&bytes)
            .with_context(|| format!("failed to decode libp2p identity {}", path.display())),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).with_context(|| {
                    format!("failed to create identity directory {}", parent.display())
                })?;
            }

            let keypair = identity::Keypair::generate_ed25519();
            let bytes = keypair
                .to_protobuf_encoding()
                .context("failed to encode libp2p identity")?;
            fs::write(path, bytes)
                .with_context(|| format!("failed to write libp2p identity {}", path.display()))?;
            #[cfg(unix)]
            fs::set_permissions(path, fs::Permissions::from_mode(0o600)).with_context(|| {
                format!(
                    "failed to restrict libp2p identity permissions {}",
                    path.display()
                )
            })?;
            Ok(keypair)
        }
        Err(error) => {
            Err(error).with_context(|| format!("failed to read libp2p identity {}", path.display()))
        }
    }
}

#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub route: Vec<RouteHop>,
    pub response: ActivationResponse,
}

#[derive(Debug, Clone)]
pub struct ModelFetchResult {
    pub shard: ModelShardInfo,
    pub source_peer_id: String,
    pub cache_record: CachedShardRecord,
}

#[derive(Debug, Clone)]
pub struct ModelSourceFetchResult {
    pub model_id: String,
    pub source_checksum: String,
    pub source_peer_id: String,
    pub path: PathBuf,
    pub size_bytes: u64,
}

const MODEL_BLOB_CHUNK_BYTES: u32 = 4 * 1024 * 1024;
const MODEL_BLOB_HEADER_MAX_BYTES: usize = 64 * 1024;

#[derive(NetworkBehaviour)]
struct GridBehaviour {
    gossipsub: gossipsub::Behaviour,
    mdns: mdns::tokio::Behaviour,
    activation: request_response::json::Behaviour<ActivationRequest, ActivationResponse>,
    model: request_response::json::Behaviour<ModelShardRequest, ModelShardResponse>,
    blob: request_response::Behaviour<ModelBlobCodec>,
}

#[derive(Debug, Clone, Default)]
struct ModelBlobCodec;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ModelBlobResponseHeader {
    protocol_version: u32,
    request_id: uuid::Uuid,
    peer_id: String,
    model_id: String,
    layers: Option<LayerRange>,
    source_checksum: String,
    offset: u64,
    total_size_bytes: u64,
    payload_len: u32,
    error: Option<String>,
}

#[async_trait]
impl request_response::Codec for ModelBlobCodec {
    type Protocol = StreamProtocol;
    type Request = ModelBlobRequest;
    type Response = ModelBlobResponse;

    async fn read_request<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Request>
    where
        T: AsyncRead + Unpin + Send,
    {
        let bytes = read_blob_frame(io, MODEL_BLOB_HEADER_MAX_BYTES).await?;
        serde_json::from_slice(&bytes).map_err(invalid_data)
    }

    async fn read_response<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Response>
    where
        T: AsyncRead + Unpin + Send,
    {
        let header_bytes = read_blob_frame(io, MODEL_BLOB_HEADER_MAX_BYTES).await?;
        let header: ModelBlobResponseHeader =
            serde_json::from_slice(&header_bytes).map_err(invalid_data)?;
        let payload_len = header.payload_len as usize;
        if payload_len > MODEL_BLOB_CHUNK_BYTES as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "model blob payload is too large: {} > {}",
                    payload_len, MODEL_BLOB_CHUNK_BYTES
                ),
            ));
        }
        let mut payload = vec![0_u8; payload_len];
        if payload_len > 0 {
            io.read_exact(&mut payload).await?;
        }

        Ok(ModelBlobResponse {
            protocol_version: header.protocol_version,
            request_id: header.request_id,
            peer_id: header.peer_id,
            model_id: header.model_id,
            layers: header.layers,
            source_checksum: header.source_checksum,
            offset: header.offset,
            total_size_bytes: header.total_size_bytes,
            payload,
            error: header.error,
        })
    }

    async fn write_request<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
        req: Self::Request,
    ) -> io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        let bytes = serde_json::to_vec(&req).map_err(invalid_data)?;
        write_blob_frame(io, &bytes).await?;
        io.close().await
    }

    async fn write_response<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
        res: Self::Response,
    ) -> io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        if res.payload.len() > MODEL_BLOB_CHUNK_BYTES as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "model blob payload is too large: {} > {}",
                    res.payload.len(),
                    MODEL_BLOB_CHUNK_BYTES
                ),
            ));
        }
        let header = ModelBlobResponseHeader {
            protocol_version: res.protocol_version,
            request_id: res.request_id,
            peer_id: res.peer_id,
            model_id: res.model_id,
            layers: res.layers,
            source_checksum: res.source_checksum,
            offset: res.offset,
            total_size_bytes: res.total_size_bytes,
            payload_len: res.payload.len() as u32,
            error: res.error,
        };
        let header_bytes = serde_json::to_vec(&header).map_err(invalid_data)?;
        write_blob_frame(io, &header_bytes).await?;
        if !res.payload.is_empty() {
            io.write_all(&res.payload).await?;
        }
        io.close().await
    }
}

async fn read_blob_frame<T>(io: &mut T, max_len: usize) -> io::Result<Vec<u8>>
where
    T: AsyncRead + Unpin + Send,
{
    let mut len_bytes = [0_u8; 4];
    io.read_exact(&mut len_bytes).await?;
    let len = u32::from_be_bytes(len_bytes) as usize;
    if len > max_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("model blob frame too large: {len} > {max_len}"),
        ));
    }
    let mut bytes = vec![0_u8; len];
    if len > 0 {
        io.read_exact(&mut bytes).await?;
    }
    Ok(bytes)
}

async fn write_blob_frame<T>(io: &mut T, bytes: &[u8]) -> io::Result<()>
where
    T: AsyncWrite + Unpin + Send,
{
    if bytes.len() > u32::MAX as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "model blob frame exceeds u32 length limit",
        ));
    }
    io.write_all(&(bytes.len() as u32).to_be_bytes()).await?;
    io.write_all(bytes).await
}

fn invalid_data(error: impl std::error::Error + Send + Sync + 'static) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error)
}

enum ActivationNetworkEvent {
    Request {
        request: ActivationRequest,
        channel: request_response::ResponseChannel<ActivationResponse>,
    },
    Response {
        request_id: request_response::OutboundRequestId,
        response: ActivationResponse,
    },
    OutboundFailure {
        peer: PeerId,
        request_id: request_response::OutboundRequestId,
        error: request_response::OutboundFailure,
    },
}

#[allow(dead_code)]
enum ModelNetworkEvent {
    Request {
        request: ModelShardRequest,
        channel: request_response::ResponseChannel<ModelShardResponse>,
    },
    Response {
        request_id: request_response::OutboundRequestId,
        response: ModelShardResponse,
    },
    OutboundFailure {
        peer: PeerId,
        request_id: request_response::OutboundRequestId,
        error: request_response::OutboundFailure,
    },
}

enum ModelBlobNetworkEvent {
    Request {
        request: ModelBlobRequest,
        channel: request_response::ResponseChannel<ModelBlobResponse>,
    },
    Response {
        request_id: request_response::OutboundRequestId,
        response: ModelBlobResponse,
    },
    OutboundFailure {
        peer: PeerId,
        request_id: request_response::OutboundRequestId,
        error: request_response::OutboundFailure,
    },
}

enum GridNetworkEvent {
    Activation(ActivationNetworkEvent),
    Model(ModelNetworkEvent),
    ModelBlob(ModelBlobNetworkEvent),
}

enum PendingOutbound {
    Forward {
        channel: request_response::ResponseChannel<ActivationResponse>,
        trace_id: uuid::Uuid,
        peer_id: String,
        trace: Vec<TraceEvent>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum ActivationStep {
    Forward(ActivationRequest),
    Final(ActivationResponse),
}

pub async fn run_worker_node(mut discovery: DiscoveryConfig, worker: WorkerConfig) -> Result<()> {
    if worker.peer_id != discovery.peer_id().to_string() {
        bail!(
            "worker peer_id {} does not match libp2p identity {}",
            worker.peer_id,
            discovery.peer_id()
        );
    }

    let topic = gossipsub::IdentTopic::new(discovery.topic.clone());
    let mut registry = ShardRegistry::new();
    registry.extend(discovery.static_peers.clone());

    let mut swarm = build_grid_swarm(discovery.keypair.clone(), &topic)?;
    add_static_peer_addresses(&mut swarm, &discovery.static_peers);
    listen_on(&mut swarm, &discovery.p2p_listen)?;
    let shard_cache = worker
        .shard_cache
        .clone()
        .map(ShardCache::new)
        .transpose()?;

    let mut publish_interval = interval(discovery.publish_interval);
    let mut static_peer_dial_interval = interval(Duration::from_secs(10));
    let mut pending_forwards = HashMap::new();

    loop {
        tokio::select! {
            _ = static_peer_dial_interval.tick(), if !discovery.static_peers.is_empty() => {
                add_static_peer_addresses(&mut swarm, &discovery.static_peers);
            }
            _ = publish_interval.tick(), if discovery.advertisement.is_some() => {
                refresh_advertisement_model_shards(
                    &mut discovery.advertisement,
                    shard_cache.as_ref(),
                )?;
                if let Some(advertisement) = &discovery.advertisement {
                    publish_advertisement(&mut swarm, &topic, advertisement)?;
                }
            }
            event = swarm.select_next_some() => {
                if let Some(network_event) = handle_grid_event(
                    &mut swarm,
                    event,
                    &mut registry,
                    &mut discovery.advertisement,
                    &topic,
                    discovery.advertise_listen_addresses,
                    discovery.dial_discovered_peers,
                )? {
                    match network_event {
                        GridNetworkEvent::Activation(event) => {
                            handle_worker_activation_event(
                                &mut swarm,
                                &worker,
                                &discovery.static_peers,
                                event,
                                &mut pending_forwards,
                            )?;
                        }
                        GridNetworkEvent::Model(event) => {
                            handle_model_network_event(
                                &mut swarm,
                                shard_cache.as_ref(),
                                &worker.peer_id,
                                event,
                            )?;
                        }
                        GridNetworkEvent::ModelBlob(event) => {
                            handle_model_blob_network_event(
                                &mut swarm,
                                shard_cache.as_ref(),
                                &worker.peer_id,
                                event,
                            )?;
                        }
                    }
                }
            }
        }
    }
}

pub async fn run_model_distribution_node(
    mut discovery: DiscoveryConfig,
    cache_config: ShardCacheConfig,
) -> Result<()> {
    let topic = gossipsub::IdentTopic::new(discovery.topic.clone());
    let mut registry = ShardRegistry::new();
    registry.extend(discovery.static_peers.clone());
    let shard_cache = ShardCache::new(cache_config)?;
    let peer_id = discovery.peer_id().to_string();

    if discovery.advertisement.is_none() {
        discovery.advertisement = Some(empty_advertisement(peer_id.clone(), String::new()));
    }
    refresh_advertisement_model_shards(&mut discovery.advertisement, Some(&shard_cache))?;

    let mut swarm = build_grid_swarm(discovery.keypair.clone(), &topic)?;
    add_static_peer_addresses(&mut swarm, &discovery.static_peers);
    listen_on(&mut swarm, &discovery.p2p_listen)?;

    let mut publish_interval = interval(discovery.publish_interval);
    let mut static_peer_dial_interval = interval(Duration::from_secs(10));
    let mut pending_forwards = HashMap::new();

    loop {
        tokio::select! {
            _ = static_peer_dial_interval.tick(), if !discovery.static_peers.is_empty() => {
                add_static_peer_addresses(&mut swarm, &discovery.static_peers);
            }
            _ = publish_interval.tick() => {
                refresh_advertisement_model_shards(&mut discovery.advertisement, Some(&shard_cache))?;
                if let Some(advertisement) = &discovery.advertisement {
                    publish_advertisement(&mut swarm, &topic, advertisement)?;
                }
                if discovery.relay_advertisements {
                    publish_known_advertisements(
                        &mut swarm,
                        &topic,
                        &registry,
                        discovery.peer_id(),
                    )?;
                }
            }
            event = swarm.select_next_some() => {
                if let Some(network_event) = handle_grid_event(
                    &mut swarm,
                    event,
                    &mut registry,
                    &mut discovery.advertisement,
                    &topic,
                    discovery.advertise_listen_addresses,
                    discovery.dial_discovered_peers,
                )? {
                    match network_event {
                        GridNetworkEvent::Model(event) => {
                            handle_model_network_event(&mut swarm, Some(&shard_cache), &peer_id, event)?;
                        }
                        GridNetworkEvent::ModelBlob(event) => {
                            handle_model_blob_network_event(&mut swarm, Some(&shard_cache), &peer_id, event)?;
                        }
                        GridNetworkEvent::Activation(event) => {
                            handle_cached_activation_event(
                                &mut swarm,
                                &shard_cache,
                                &peer_id,
                                &discovery.static_peers,
                                event,
                                &mut pending_forwards,
                            )?;
                        }
                    }
                }
            }
        }
    }
}

pub async fn discover_for(mut config: DiscoveryConfig, timeout: Duration) -> Result<ShardRegistry> {
    let topic = gossipsub::IdentTopic::new(config.topic.clone());
    let mut registry = ShardRegistry::new();
    registry.extend(config.static_peers.clone());
    if let Some(advertisement) = config.advertisement.clone() {
        registry.upsert(advertisement);
    }

    let mut swarm = build_grid_swarm(config.keypair.clone(), &topic)?;
    add_static_peer_addresses(&mut swarm, &config.static_peers);
    listen_on(&mut swarm, &config.p2p_listen)?;

    let deadline = Instant::now() + timeout;

    loop {
        tokio::select! {
            event = swarm.select_next_some() => {
                let _ = handle_grid_event(
                    &mut swarm,
                    event,
                    &mut registry,
                    &mut config.advertisement,
                    &topic,
                    config.advertise_listen_addresses,
                    config.dial_discovered_peers,
                )?;
            }
            _ = sleep_until(deadline) => {
                return Ok(registry);
            }
        }
    }
}

pub async fn fetch_model_shard_over_libp2p(
    mut config: DiscoveryConfig,
    cache_config: ShardCacheConfig,
    model_id: String,
    layers: LayerRange,
    checksum: Option<String>,
    version: Option<String>,
    discovery_timeout: Duration,
) -> Result<ModelFetchResult> {
    let cache = ShardCache::new(cache_config)?;
    if let Some(record) = cache.find(&model_id, layers, checksum.as_deref(), version.as_deref())? {
        return Ok(ModelFetchResult {
            shard: record.info.clone(),
            source_peer_id: "local-cache".to_owned(),
            cache_record: record,
        });
    }

    let topic = gossipsub::IdentTopic::new(config.topic.clone());
    let mut registry = ShardRegistry::new();
    registry.extend(config.static_peers.clone());
    let local_peer_id = config.peer_id().to_string();

    if config.advertisement.is_none() {
        config.advertisement = Some(empty_advertisement(local_peer_id.clone(), String::new()));
    }
    refresh_advertisement_model_shards(&mut config.advertisement, Some(&cache))?;

    let mut swarm = build_grid_swarm(config.keypair.clone(), &topic)?;
    add_static_peer_addresses(&mut swarm, &config.static_peers);
    listen_on(&mut swarm, &config.p2p_listen)?;

    let deadline = Instant::now() + discovery_timeout;
    let mut publish_interval = interval(config.publish_interval);
    let partial_dir = cache.config().root.join("tmp");
    fs::create_dir_all(&partial_dir)
        .with_context(|| format!("failed to create {}", partial_dir.display()))?;
    let partial_path = partial_dir.join(format!(
        "{}-{}-{}-{}.gguf.partial",
        sanitize_path_segment(&model_id),
        layers.start,
        layers.end,
        uuid::Uuid::new_v4()
    ));
    let _ = fs::remove_file(&partial_path);
    let mut downloaded_bytes = 0_u64;
    let mut pending_request: Option<(
        request_response::OutboundRequestId,
        ModelShardCandidate,
        u64,
    )> = None;
    let mut failed_peers = Vec::<String>::new();

    loop {
        if pending_request.is_none() {
            if let Some(candidate) = select_model_shard_candidate(
                &registry,
                &local_peer_id,
                &model_id,
                layers,
                checksum.as_deref(),
                version.as_deref(),
                &failed_peers,
            ) {
                let request = ModelBlobRequest::new_shard(
                    model_id.clone(),
                    layers,
                    candidate.shard.checksum.clone(),
                    downloaded_bytes,
                    MODEL_BLOB_CHUNK_BYTES,
                );
                let request_id =
                    send_model_blob_request(&mut swarm, &candidate.advertisement, request)?;
                pending_request = Some((request_id, candidate, downloaded_bytes));
            }
        }

        tokio::select! {
            _ = publish_interval.tick() => {
                refresh_advertisement_model_shards(&mut config.advertisement, Some(&cache))?;
                if let Some(advertisement) = &config.advertisement {
                    publish_advertisement(&mut swarm, &topic, advertisement)?;
                }
            }
            event = swarm.select_next_some() => {
                if let Some(network_event) = handle_grid_event(
                    &mut swarm,
                    event,
                    &mut registry,
                    &mut config.advertisement,
                    &topic,
                    config.advertise_listen_addresses,
                    config.dial_discovered_peers,
                )? {
                    match network_event {
                        GridNetworkEvent::ModelBlob(ModelBlobNetworkEvent::Response { request_id, response }) => {
                            if let Some((pending_id, candidate, expected_offset)) = pending_request.take() {
                                if request_id != pending_id {
                                    pending_request = Some((pending_id, candidate, expected_offset));
                                    continue;
                                }

                                if let Some(error) = response.error {
                                    failed_peers.push(candidate.advertisement.peer_id);
                                    eprintln!("model shard chunk request failed: {error}");
                                    continue;
                                }
                                if response.model_id != model_id
                                    || response.source_checksum != candidate.shard.checksum
                                    || response.layers != Some(layers)
                                {
                                    failed_peers.push(candidate.advertisement.peer_id);
                                    eprintln!("model shard chunk response identity mismatch");
                                    continue;
                                }
                                if response.offset != expected_offset || response.offset != downloaded_bytes {
                                    failed_peers.push(candidate.advertisement.peer_id);
                                    eprintln!("model shard chunk response offset mismatch: got {}, expected {}", response.offset, downloaded_bytes);
                                    continue;
                                }
                                if response.total_size_bytes != candidate.shard.size_bytes {
                                    failed_peers.push(candidate.advertisement.peer_id);
                                    eprintln!(
                                        "model shard chunk size mismatch: got {}, expected {}",
                                        response.total_size_bytes, candidate.shard.size_bytes
                                    );
                                    continue;
                                }
                                if response.payload.is_empty() && downloaded_bytes < candidate.shard.size_bytes {
                                    failed_peers.push(candidate.advertisement.peer_id);
                                    eprintln!("model shard chunk response returned empty payload before EOF");
                                    continue;
                                }

                                append_source_chunk(&partial_path, &response.payload)?;
                                downloaded_bytes = downloaded_bytes.saturating_add(response.payload.len() as u64);

                                if downloaded_bytes >= candidate.shard.size_bytes {
                                    let cache_record = cache.store_downloaded_file(
                                        &candidate.shard,
                                        &partial_path,
                                        candidate.seed_manifest.clone(),
                                    )?;
                                    refresh_advertisement_model_shards(&mut config.advertisement, Some(&cache))?;
                                    if let Some(advertisement) = &config.advertisement {
                                        publish_advertisement(&mut swarm, &topic, advertisement)?;
                                    }

                                    return Ok(ModelFetchResult {
                                        shard: candidate.shard,
                                        source_peer_id: candidate.advertisement.peer_id,
                                        cache_record,
                                    });
                                }
                            }
                        }
                        GridNetworkEvent::ModelBlob(ModelBlobNetworkEvent::OutboundFailure { peer, request_id, error }) => {
                            if let Some((pending_id, candidate, expected_offset)) = pending_request.take() {
                                if request_id == pending_id {
                                    failed_peers.push(peer.to_string());
                                    eprintln!("model shard chunk request to {peer} failed: {error}");
                                    continue;
                                }
                                pending_request = Some((pending_id, candidate, expected_offset));
                            }
                        }
                        GridNetworkEvent::Model(event) => {
                            handle_model_network_event(&mut swarm, Some(&cache), &local_peer_id, event)?;
                        }
                        GridNetworkEvent::ModelBlob(event) => {
                            handle_model_blob_network_event(&mut swarm, Some(&cache), &local_peer_id, event)?;
                        }
                        GridNetworkEvent::Activation(_) => {}
                    }
                }
            }
            _ = sleep_until(deadline) => {
                bail!(
                    "timed out downloading executable model shard {} {}:{} checksum {}; downloaded {} bytes",
                    model_id,
                    layers.start,
                    layers.end,
                    checksum.as_deref().unwrap_or("<any>"),
                    downloaded_bytes
                );
            }
        }
    }
}

pub async fn fetch_model_source_over_libp2p(
    mut config: DiscoveryConfig,
    cache_config: ShardCacheConfig,
    model_id: String,
    source_checksum: String,
    expected_size_bytes: u64,
    discovery_timeout: Duration,
    mut on_progress: impl FnMut(u64, u64) + Send,
) -> Result<ModelSourceFetchResult> {
    let cache = ShardCache::new(cache_config.clone())?;
    let final_path = source_cache_path(&cache_config, &model_id, &source_checksum);
    if final_path.is_file() {
        let actual = sha256_file(&final_path)
            .with_context(|| format!("failed to verify cached source {}", final_path.display()))?;
        if actual == source_checksum {
            let size_bytes = fs::metadata(&final_path)
                .map(|metadata| metadata.len())
                .unwrap_or(expected_size_bytes);
            on_progress(size_bytes, size_bytes);
            return Ok(ModelSourceFetchResult {
                model_id,
                source_checksum,
                source_peer_id: "local-cache".to_owned(),
                path: final_path,
                size_bytes,
            });
        }
        let _ = fs::remove_file(&final_path);
    }

    let topic = gossipsub::IdentTopic::new(config.topic.clone());
    let mut registry = ShardRegistry::new();
    registry.extend(config.static_peers.clone());
    let local_peer_id = config.peer_id().to_string();

    if config.advertisement.is_none() {
        config.advertisement = Some(empty_advertisement(local_peer_id.clone(), String::new()));
    }
    refresh_advertisement_model_shards(&mut config.advertisement, Some(&cache))?;

    let mut swarm = build_grid_swarm(config.keypair.clone(), &topic)?;
    add_static_peer_addresses(&mut swarm, &config.static_peers);
    listen_on(&mut swarm, &config.p2p_listen)?;

    let deadline = Instant::now() + discovery_timeout;
    let mut publish_interval = interval(config.publish_interval);
    let partial_path = final_path.with_extension("gguf.partial");
    if let Some(parent) = final_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let _ = fs::remove_file(&partial_path);
    let mut downloaded_bytes = 0_u64;
    let mut total_size_bytes = expected_size_bytes;
    let mut pending_request: Option<(request_response::OutboundRequestId, NodeAdvertisement, u64)> =
        None;
    let mut failed_peers = Vec::<String>::new();
    on_progress(downloaded_bytes, total_size_bytes);

    loop {
        if pending_request.is_none() {
            if let Some(advertisement) = select_model_blob_candidate(
                &registry,
                &local_peer_id,
                &model_id,
                &source_checksum,
                &failed_peers,
            ) {
                let request = ModelBlobRequest::new(
                    model_id.clone(),
                    source_checksum.clone(),
                    downloaded_bytes,
                    MODEL_BLOB_CHUNK_BYTES,
                );
                let request_id = send_model_blob_request(&mut swarm, &advertisement, request)?;
                pending_request = Some((request_id, advertisement, downloaded_bytes));
            }
        }

        tokio::select! {
            _ = publish_interval.tick() => {
                refresh_advertisement_model_shards(&mut config.advertisement, Some(&cache))?;
                if let Some(advertisement) = &config.advertisement {
                    publish_advertisement(&mut swarm, &topic, advertisement)?;
                }
            }
            event = swarm.select_next_some() => {
                if let Some(network_event) = handle_grid_event(
                    &mut swarm,
                    event,
                    &mut registry,
                    &mut config.advertisement,
                    &topic,
                    config.advertise_listen_addresses,
                    config.dial_discovered_peers,
                )? {
                    match network_event {
                        GridNetworkEvent::ModelBlob(ModelBlobNetworkEvent::Response { request_id, response }) => {
                            let Some((pending_id, advertisement, expected_offset)) = pending_request.take() else {
                                continue;
                            };
                            if request_id != pending_id {
                                pending_request = Some((pending_id, advertisement, expected_offset));
                                continue;
                            }

                            if let Some(error) = response.error {
                                failed_peers.push(advertisement.peer_id);
                                eprintln!("model blob request failed: {error}");
                                continue;
                            }
                            if response.model_id != model_id || response.source_checksum != source_checksum {
                                failed_peers.push(advertisement.peer_id);
                                eprintln!("model blob response identity mismatch");
                                continue;
                            }
                            if response.offset != expected_offset || response.offset != downloaded_bytes {
                                failed_peers.push(advertisement.peer_id);
                                eprintln!("model blob response offset mismatch: got {}, expected {}", response.offset, downloaded_bytes);
                                continue;
                            }
                            if expected_size_bytes > 0 && response.total_size_bytes != expected_size_bytes {
                                failed_peers.push(advertisement.peer_id);
                                eprintln!(
                                    "model blob size mismatch: got {}, expected {}",
                                    response.total_size_bytes, expected_size_bytes
                                );
                                continue;
                            }
                            total_size_bytes = response.total_size_bytes;
                            if response.payload.is_empty() && downloaded_bytes < total_size_bytes {
                                failed_peers.push(advertisement.peer_id);
                                eprintln!("model blob response returned an empty chunk before EOF");
                                continue;
                            }

                            append_source_chunk(&partial_path, &response.payload)?;
                            downloaded_bytes = downloaded_bytes.saturating_add(response.payload.len() as u64);
                            on_progress(downloaded_bytes, total_size_bytes);

                            if downloaded_bytes >= total_size_bytes {
                                let downloaded_size = fs::metadata(&partial_path)
                                    .with_context(|| format!("failed to inspect {}", partial_path.display()))?
                                    .len();
                                if downloaded_size != total_size_bytes {
                                    bail!(
                                        "downloaded source size mismatch for {}; expected {}, got {}",
                                        model_id,
                                        total_size_bytes,
                                        downloaded_size
                                    );
                                }
                                let actual_checksum = sha256_file(&partial_path)
                                    .with_context(|| format!("failed to verify {}", partial_path.display()))?;
                                if actual_checksum != source_checksum {
                                    bail!(
                                        "downloaded source checksum mismatch for {}; expected {}, got {}",
                                        model_id,
                                        source_checksum,
                                        actual_checksum
                                    );
                                }
                                if final_path.exists() {
                                    let _ = fs::remove_file(&final_path);
                                }
                                fs::rename(&partial_path, &final_path).with_context(|| {
                                    format!(
                                        "failed to move {} to {}",
                                        partial_path.display(),
                                        final_path.display()
                                    )
                                })?;
                                refresh_advertisement_model_shards(&mut config.advertisement, Some(&cache))?;
                                if let Some(advertisement) = &config.advertisement {
                                    publish_advertisement(&mut swarm, &topic, advertisement)?;
                                }
                                return Ok(ModelSourceFetchResult {
                                    model_id,
                                    source_checksum,
                                    source_peer_id: advertisement.peer_id,
                                    path: final_path,
                                    size_bytes: total_size_bytes,
                                });
                            }
                        }
                        GridNetworkEvent::ModelBlob(ModelBlobNetworkEvent::OutboundFailure { peer, request_id, error }) => {
                            if let Some((pending_id, advertisement, expected_offset)) = pending_request.take() {
                                if request_id == pending_id {
                                    failed_peers.push(peer.to_string());
                                    eprintln!("model blob request to {peer} failed: {error}");
                                    continue;
                                }
                                pending_request = Some((pending_id, advertisement, expected_offset));
                            }
                        }
                        GridNetworkEvent::ModelBlob(event) => {
                            handle_model_blob_network_event(&mut swarm, Some(&cache), &local_peer_id, event)?;
                        }
                        GridNetworkEvent::Model(event) => {
                            handle_model_network_event(&mut swarm, Some(&cache), &local_peer_id, event)?;
                        }
                        GridNetworkEvent::Activation(_) => {}
                    }
                }
            }
            _ = sleep_until(deadline) => {
                bail!(
                    "timed out downloading GGUF source for {} checksum {}; downloaded {}/{} bytes",
                    model_id,
                    source_checksum,
                    downloaded_bytes,
                    total_size_bytes
                );
            }
        }
    }
}

fn append_source_chunk(path: &Path, payload: &[u8]) -> Result<()> {
    if payload.is_empty() {
        return Ok(());
    }
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    file.write_all(payload)
        .with_context(|| format!("failed to write {}", path.display()))
}

pub async fn infer_over_libp2p(
    mut config: DiscoveryConfig,
    manifest: ModelManifest,
    prompt: String,
    hidden_size: usize,
    discovery_timeout: Duration,
) -> Result<InferenceResult> {
    let topic = gossipsub::IdentTopic::new(config.topic.clone());
    let mut registry = ShardRegistry::new();
    registry.extend(config.static_peers.clone());

    let mut swarm = build_grid_swarm(config.keypair.clone(), &topic)?;
    add_static_peer_addresses(&mut swarm, &config.static_peers);
    listen_on(&mut swarm, &config.p2p_listen)?;

    let route = discover_route_on_swarm(
        &mut swarm,
        &mut registry,
        &mut config,
        &topic,
        &manifest,
        discovery_timeout,
    )
    .await?;

    let demo_mode = manifest.runtime_kind == RuntimeKind::Demo;
    let activation = if demo_mode {
        DemoRuntime::prompt_to_activation(&prompt, hidden_size)
    } else {
        Vec::new()
    };
    let request = ActivationRequest::new(
        manifest.model_id.clone(),
        route.clone(),
        hidden_size,
        activation,
        Some(PromptMetadata { prompt, demo_mode }),
    );
    let first_hop = request
        .current_hop()
        .cloned()
        .ok_or_else(|| anyhow!("route must contain at least one hop"))?;
    let outbound_id =
        send_activation_request_with_relays(&mut swarm, &config.static_peers, &first_hop, request)?;
    let response = wait_for_client_response(
        &mut swarm,
        &mut registry,
        &mut config,
        &topic,
        outbound_id,
        match manifest.runtime_kind {
            RuntimeKind::Demo => Duration::from_secs(15),
            RuntimeKind::LlamaCpp => Duration::from_secs(600),
        },
    )
    .await?;

    if let Some(error) = &response.error {
        bail!("remote activation error: {error}");
    }

    Ok(InferenceResult { route, response })
}

pub fn demo_advertisement(
    peer_id: String,
    address: String,
    model_id: String,
    layers: LayerRange,
) -> NodeAdvertisement {
    let addresses = if address.is_empty() {
        Vec::new()
    } else {
        vec![address]
    };

    NodeAdvertisement {
        protocol_version: PROTOCOL_VERSION,
        peer_id,
        addresses,
        available_ram_bytes: None,
        available_vram_bytes: None,
        latency_hint_ms: None,
        hosted_shards: vec![ShardDescriptor::demo(model_id, layers)],
        model_shards: Vec::new(),
    }
}

pub fn shard_advertisement(
    peer_id: String,
    address: String,
    manifest: &ModelManifest,
    layers: LayerRange,
) -> NodeAdvertisement {
    let addresses = if address.is_empty() {
        Vec::new()
    } else {
        vec![address]
    };

    NodeAdvertisement {
        protocol_version: PROTOCOL_VERSION,
        peer_id,
        addresses,
        available_ram_bytes: None,
        available_vram_bytes: None,
        latency_hint_ms: None,
        hosted_shards: vec![ShardDescriptor::for_manifest(manifest, layers)],
        model_shards: Vec::new(),
    }
}

pub fn empty_advertisement(peer_id: String, address: String) -> NodeAdvertisement {
    let addresses = if address.is_empty() {
        Vec::new()
    } else {
        vec![address]
    };

    NodeAdvertisement {
        protocol_version: PROTOCOL_VERSION,
        peer_id,
        addresses,
        available_ram_bytes: None,
        available_vram_bytes: None,
        latency_hint_ms: None,
        hosted_shards: Vec::new(),
        model_shards: Vec::new(),
    }
}

fn observed_peer_advertisement(
    peer_id: PeerId,
    endpoint: &ConnectedPoint,
) -> Option<NodeAdvertisement> {
    let remote_address = endpoint.get_remote_address();
    if remote_address.is_empty() {
        return None;
    }

    let mut address = remote_address.clone();
    if !address
        .iter()
        .any(|protocol| matches!(protocol, Protocol::P2p(_)))
    {
        address.push(Protocol::P2p(peer_id));
    }

    Some(empty_advertisement(
        peer_id.to_string(),
        address.to_string(),
    ))
}

fn refresh_advertisement_model_shards(
    advertisement: &mut Option<NodeAdvertisement>,
    cache: Option<&ShardCache>,
) -> Result<()> {
    let Some(advertisement) = advertisement else {
        return Ok(());
    };

    match cache {
        Some(cache) => {
            let records = cache.list()?;
            advertisement.model_shards = records
                .iter()
                .filter(|record| is_executable_shard_record(record))
                .map(|record| record.info.clone())
                .collect();
            advertisement.hosted_shards = records
                .iter()
                .filter_map(|record| {
                    if !is_executable_shard_record(record) {
                        return None;
                    }
                    let manifest = record.manifest.clone()?;
                    if !seed_record_is_executable(cache.config(), &manifest) {
                        return None;
                    }
                    let seed_manifest = Box::new(manifest.clone());
                    Some(ShardDescriptor {
                        model_id: manifest.model_id.clone(),
                        layers: manifest.layers,
                        runtime_kind: manifest.runtime_kind.clone(),
                        tokenizer: Some(manifest.tokenizer.clone()),
                        metadata: Some(manifest.metadata.clone()),
                        shard_hash: Some(manifest.shard_hash.clone()),
                        seed_manifest: Some(seed_manifest),
                    })
                })
                .collect();
        }
        None => {
            advertisement.model_shards = Vec::new();
            advertisement.hosted_shards = Vec::new();
        }
    }

    Ok(())
}

fn seed_record_is_executable(config: &ShardCacheConfig, manifest: &SeedShardManifest) -> bool {
    let _ = config;
    manifest.runtime_kind == RuntimeKind::Demo
        || manifest.payload_kind == model_distribution::PAYLOAD_KIND_GGUF_SHARD
}

pub fn process_activation_step(
    config: &WorkerConfig,
    mut request: ActivationRequest,
) -> Result<ActivationStep, ActivationResponse> {
    let trace_id = request.trace_id;

    if let Err(error) = validate_activation_request(config, &request) {
        return Err(ActivationResponse::failure(
            trace_id,
            config.peer_id.clone(),
            error.to_string(),
            request.trace,
        ));
    }

    let hop = request
        .current_hop()
        .cloned()
        .expect("validation ensures a current hop exists");
    let started = Instant::now();
    let mut output_text = None;
    let timing_ms;

    match config.runtime_kind {
        RuntimeKind::Demo => {
            let runtime = DemoRuntime::new(config.owned_layers, config.hidden_size);
            request.activation = match runtime.execute(hop.layers, &request.activation) {
                Ok(activation) => activation,
                Err(error) => {
                    return Err(ActivationResponse::failure(
                        trace_id,
                        config.peer_id.clone(),
                        error.to_string(),
                        request.trace,
                    ));
                }
            };
            timing_ms = elapsed_ms(started);
        }
        RuntimeKind::LlamaCpp => match execute_llama_cpp_shard(config, hop.layers, &request) {
            Ok(output) => {
                request.activation = output.activation;
                output_text = output.output_text;
                timing_ms = output.timing_ms;
            }
            Err(error) => {
                return Err(ActivationResponse::failure(
                    trace_id,
                    config.peer_id.clone(),
                    error.to_string(),
                    request.trace,
                ));
            }
        },
    }

    let next_peer_id = request.next_hop().map(|next| next.peer_id.clone());
    let trace_event = TraceEvent {
        peer_id: config.peer_id.clone(),
        layers: hop.layers,
        next_peer_id,
        activation_size_bytes: request.activation.len() * mem::size_of::<f32>(),
        activation_checksum: activation_checksum(&request.activation),
        timing_ms,
    };
    log_hop(trace_id, &trace_event);
    request.trace.push(trace_event);

    if request.next_hop().is_some() {
        request.current_hop_index += 1;
        Ok(ActivationStep::Forward(request))
    } else {
        let output =
            output_text.unwrap_or_else(|| DemoRuntime::decode_activation(&request.activation));
        Ok(ActivationStep::Final(ActivationResponse::success(
            request,
            config.peer_id.clone(),
            Some(output),
            timing_ms,
        )))
    }
}

#[derive(Debug)]
struct LlamaShardOutput {
    activation: Vec<f32>,
    output_text: Option<String>,
    timing_ms: u64,
}

#[derive(Debug, Deserialize)]
struct LlamaBridgeJson {
    ok: bool,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    n_tokens: Option<usize>,
    #[serde(default)]
    hidden_size: Option<usize>,
    #[serde(default)]
    output_f32_count: Option<usize>,
    #[serde(default)]
    output_text: Option<String>,
    #[serde(default)]
    timing_ms: Option<f64>,
}

fn execute_llama_cpp_shard(
    config: &WorkerConfig,
    layers: LayerRange,
    request: &ActivationRequest,
) -> Result<LlamaShardOutput> {
    let prompt = request
        .prompt
        .as_ref()
        .ok_or_else(|| anyhow!("llama.cpp activation request is missing prompt metadata"))?;
    if prompt.demo_mode {
        bail!("llama.cpp activation request was marked as demo_mode");
    }

    let cache_config = config
        .shard_cache
        .as_ref()
        .ok_or_else(|| anyhow!("llama.cpp worker has no shard cache configured"))?;
    let cache = ShardCache::new(cache_config.clone())?;
    let (manifest, model_path) =
        executable_seed_manifest_for_layers(&cache, &config.model_id, layers)?.ok_or_else(
            || {
                anyhow!(
                    "missing verified executable GGUF source for {} {}:{}",
                    config.model_id,
                    layers.start,
                    layers.end
                )
            },
        )?;
    if manifest.runtime_kind != RuntimeKind::LlamaCpp {
        bail!(
            "expected llama.cpp shard for {} {}:{}, got {}",
            config.model_id,
            layers.start,
            layers.end,
            manifest.runtime_kind.as_str()
        );
    }

    let bridge = find_llama_bridge_binary().ok_or_else(|| {
        anyhow!(
            "infernet-llama-bridge binary is missing; run npm run prepare-runtime or set INFERNET_LLAMA_BRIDGE"
        )
    })?;

    let temp_root = env::temp_dir().join("infernet-activation-frames");
    fs::create_dir_all(&temp_root)
        .with_context(|| format!("failed to create {}", temp_root.display()))?;
    let frame_id = format!("{}-{}", request.trace_id, request.current_hop_index);
    let input_path = temp_root.join(format!("{frame_id}-in.f32"));
    let output_path = temp_root.join(format!("{frame_id}-out.f32"));

    if !request.activation.is_empty() {
        write_f32_activation(&input_path, &request.activation)?;
    }

    let mut command = Command::new(&bridge);
    if let Some(runtime_dir) = bridge.parent() {
        command.current_dir(runtime_dir);
    }
    #[cfg(target_os = "windows")]
    {
        let mut library_dirs = Vec::new();
        if let Some(parent) = bridge.parent() {
            library_dirs.push(parent.to_path_buf());
        }
        if let Some(path) = env::var_os("PATH") {
            library_dirs.extend(env::split_paths(&path));
        }
        if let Ok(path) = env::join_paths(library_dirs) {
            command.env("PATH", path);
        }
    }

    command
        .arg("--model")
        .arg(&model_path)
        .arg("--layer-start")
        .arg(layers.start.to_string())
        .arg("--layer-end")
        .arg(layers.end.to_string())
        .arg("--hidden-size")
        .arg(config.hidden_size.to_string())
        .arg("--prompt")
        .arg(&prompt.prompt);
    if !request.activation.is_empty() {
        command.arg("--input").arg(&input_path);
    }
    if request.next_hop().is_some() {
        command.arg("--output").arg(&output_path);
    }

    let output = command.output().with_context(|| {
        format!(
            "failed to run {} for {} {}:{}",
            bridge.display(),
            config.model_id,
            layers.start,
            layers.end
        )
    })?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let json_line = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .ok_or_else(|| {
            anyhow!(
                "infernet-llama-bridge produced no JSON output; status={:?}; stderr={}",
                output.status.code(),
                stderr.trim()
            )
        })?;
    let bridge_output: LlamaBridgeJson = serde_json::from_str(json_line)
        .with_context(|| format!("failed to parse infernet-llama-bridge JSON: {json_line}"))?;

    if !output.status.success() || !bridge_output.ok {
        bail!(
            "infernet-llama-bridge failed for {} {}:{}: {}{}",
            config.model_id,
            layers.start,
            layers.end,
            bridge_output
                .error
                .unwrap_or_else(|| format!("exit status {:?}", output.status.code())),
            if stderr.trim().is_empty() {
                String::new()
            } else {
                format!("; stderr={}", stderr.trim())
            }
        );
    }

    if let Some(hidden_size) = bridge_output.hidden_size {
        if hidden_size != config.hidden_size {
            bail!(
                "infernet-llama-bridge hidden size mismatch: {} vs {}",
                hidden_size,
                config.hidden_size
            );
        }
    }

    let activation = if request.next_hop().is_some() {
        let activation = read_f32_activation(&output_path)?;
        if let Some(expected) = bridge_output.output_f32_count {
            if activation.len() != expected {
                bail!(
                    "infernet-llama-bridge wrote {} f32 values, JSON reported {}",
                    activation.len(),
                    expected
                );
            }
        }
        if let Some(n_tokens) = bridge_output.n_tokens {
            let expected = n_tokens
                .checked_mul(config.hidden_size)
                .ok_or_else(|| anyhow!("activation shape overflow"))?;
            if activation.len() != expected {
                bail!(
                    "activation shape mismatch from bridge: got {} f32 values, expected {} tokens x {} hidden = {}",
                    activation.len(),
                    n_tokens,
                    config.hidden_size,
                    expected
                );
            }
        }
        activation
    } else {
        Vec::new()
    };

    let _ = fs::remove_file(&input_path);
    let _ = fs::remove_file(&output_path);

    Ok(LlamaShardOutput {
        activation,
        output_text: bridge_output.output_text,
        timing_ms: bridge_output
            .timing_ms
            .map(|value| value.max(0.0).round() as u64)
            .unwrap_or(0),
    })
}

fn write_f32_activation(path: &Path, values: &[f32]) -> Result<()> {
    let mut file =
        fs::File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    for value in values {
        file.write_all(&value.to_le_bytes())
            .with_context(|| format!("failed to write {}", path.display()))?;
    }
    Ok(())
}

fn read_f32_activation(path: &Path) -> Result<Vec<f32>> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    if bytes.len() % mem::size_of::<f32>() != 0 {
        bail!(
            "{} is not aligned to f32 values: {} bytes",
            path.display(),
            bytes.len()
        );
    }
    let mut values = Vec::with_capacity(bytes.len() / mem::size_of::<f32>());
    for chunk in bytes.chunks_exact(mem::size_of::<f32>()) {
        values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(values)
}

fn find_llama_bridge_binary() -> Option<PathBuf> {
    if let Ok(path) = env::var("INFERNET_LLAMA_BRIDGE") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
    }

    for candidate in bundled_llama_bridge_candidates() {
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    let mut executable_names = vec![platform_executable_name("infernet-llama-bridge")];
    if let Some(sidecar_name) = bundled_llama_bridge_sidecar_name() {
        executable_names.push(sidecar_name.to_owned());
    }
    if let Some(path) = env::var_os("PATH") {
        for directory in env::split_paths(&path) {
            for name in &executable_names {
                let candidate = directory.join(name);
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
        }
    }

    None
}

fn bundled_llama_bridge_candidates() -> Vec<PathBuf> {
    let executable_name = platform_executable_name("infernet-llama-bridge");
    let sidecar_name = bundled_llama_bridge_sidecar_name();
    let mut candidates = Vec::new();

    if let Ok(current_exe) = env::current_exe() {
        if let Some(parent) = current_exe.parent() {
            candidates.push(parent.join(&executable_name));
            candidates.push(parent.join("binaries").join(&executable_name));
            if let Some(sidecar_name) = sidecar_name {
                candidates.push(parent.join(sidecar_name));
                candidates.push(parent.join("binaries").join(sidecar_name));
            }
            if let Some(resources) = parent.parent().map(|path| path.join("Resources")) {
                candidates.push(resources.join(&executable_name));
                candidates.push(resources.join("binaries").join(&executable_name));
                if let Some(sidecar_name) = sidecar_name {
                    candidates.push(resources.join(sidecar_name));
                    candidates.push(resources.join("binaries").join(sidecar_name));
                }
            }
        }
    }

    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if let Some(repo_root) = crate_dir.parent().and_then(Path::parent) {
        let binaries = repo_root
            .join("infernet-ui")
            .join("src-tauri")
            .join("binaries");
        candidates.push(binaries.join(&executable_name));
        if let Some(sidecar_name) = sidecar_name {
            candidates.push(binaries.join(sidecar_name));
        }
    }

    candidates
}

fn platform_executable_name(base: &str) -> String {
    if cfg!(windows) {
        format!("{base}.exe")
    } else {
        base.to_owned()
    }
}

fn bundled_llama_bridge_sidecar_name() -> Option<&'static str> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        return Some("infernet-llama-bridge-aarch64-apple-darwin");
    }
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    {
        return Some("infernet-llama-bridge-x86_64-apple-darwin");
    }
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    {
        return Some("infernet-llama-bridge-x86_64-pc-windows-msvc.exe");
    }
    #[cfg(all(target_os = "windows", target_arch = "aarch64"))]
    {
        return Some("infernet-llama-bridge-aarch64-pc-windows-msvc.exe");
    }
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        return Some("infernet-llama-bridge-x86_64-unknown-linux-gnu");
    }
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        return Some("infernet-llama-bridge-aarch64-unknown-linux-gnu");
    }
    #[allow(unreachable_code)]
    None
}

fn validate_activation_request(config: &WorkerConfig, request: &ActivationRequest) -> Result<()> {
    if request.protocol_version != PROTOCOL_VERSION {
        bail!(
            "unsupported protocol version {}; expected {}",
            request.protocol_version,
            PROTOCOL_VERSION
        );
    }

    if request.model_id != config.model_id {
        bail!(
            "worker {} hosts model {}, got {}",
            config.peer_id,
            config.model_id,
            request.model_id
        );
    }

    if request.hidden_size != config.hidden_size {
        bail!(
            "worker {} hidden size {}, got {}",
            config.peer_id,
            config.hidden_size,
            request.hidden_size
        );
    }

    let hop = request
        .current_hop()
        .ok_or_else(|| anyhow!("missing route hop {}", request.current_hop_index))?;

    if hop.peer_id != config.peer_id {
        bail!(
            "route expected peer {}, but request reached {}",
            hop.peer_id,
            config.peer_id
        );
    }

    if !config.owned_layers.contains(&hop.layers) {
        bail!(
            "peer {} owns {:?}, route requested {:?}",
            config.peer_id,
            config.owned_layers,
            hop.layers
        );
    }

    Ok(())
}

async fn discover_route_on_swarm(
    swarm: &mut Swarm<GridBehaviour>,
    registry: &mut ShardRegistry,
    config: &mut DiscoveryConfig,
    topic: &gossipsub::IdentTopic,
    manifest: &ModelManifest,
    timeout: Duration,
) -> Result<Vec<RouteHop>> {
    let deadline = Instant::now() + timeout;

    loop {
        let route_error = match registry.route_for_model(manifest) {
            Ok(route) => return Ok(route),
            Err(error) => error,
        };

        tokio::select! {
            event = swarm.select_next_some() => {
                let _ = handle_grid_event(
                    swarm,
                    event,
                    registry,
                    &mut config.advertisement,
                    topic,
                    config.advertise_listen_addresses,
                    config.dial_discovered_peers,
                )?;
            }
            _ = sleep_until(deadline) => {
                return Err(route_error.into());
            }
        }
    }
}

async fn wait_for_client_response(
    swarm: &mut Swarm<GridBehaviour>,
    registry: &mut ShardRegistry,
    config: &mut DiscoveryConfig,
    topic: &gossipsub::IdentTopic,
    outbound_id: request_response::OutboundRequestId,
    timeout: Duration,
) -> Result<ActivationResponse> {
    let deadline = Instant::now() + timeout;

    loop {
        tokio::select! {
            event = swarm.select_next_some() => {
                match handle_grid_event(
                    swarm,
                    event,
                    registry,
                    &mut config.advertisement,
                    topic,
                    config.advertise_listen_addresses,
                    config.dial_discovered_peers,
                )? {
                    Some(GridNetworkEvent::Activation(ActivationNetworkEvent::Response { request_id, response }))
                        if request_id == outbound_id => return Ok(response),
                    Some(GridNetworkEvent::Activation(ActivationNetworkEvent::OutboundFailure { peer, request_id, error }))
                        if request_id == outbound_id => {
                            bail!("activation request to {peer} failed: {error}");
                        }
                    _ => {}
                }
            }
            _ = sleep_until(deadline) => {
                bail!("timed out waiting for activation response");
            }
        }
    }
}

fn handle_worker_activation_event(
    swarm: &mut Swarm<GridBehaviour>,
    worker: &WorkerConfig,
    activation_relays: &[NodeAdvertisement],
    network_event: ActivationNetworkEvent,
    pending_forwards: &mut HashMap<request_response::OutboundRequestId, PendingOutbound>,
) -> Result<()> {
    match network_event {
        ActivationNetworkEvent::Request { request, channel } => handle_worker_activation_request(
            swarm,
            worker,
            activation_relays,
            request,
            channel,
            pending_forwards,
        )?,
        ActivationNetworkEvent::Response {
            request_id,
            response,
        } => {
            if let Some(PendingOutbound::Forward { channel, .. }) =
                pending_forwards.remove(&request_id)
            {
                send_response(swarm, channel, response);
            }
        }
        ActivationNetworkEvent::OutboundFailure {
            peer,
            request_id,
            error,
        } => {
            if let Some(PendingOutbound::Forward {
                channel,
                trace_id,
                peer_id,
                trace,
            }) = pending_forwards.remove(&request_id)
            {
                send_response(
                    swarm,
                    channel,
                    ActivationResponse::failure(
                        trace_id,
                        peer_id,
                        format!("forward to {peer} failed: {error}"),
                        trace,
                    ),
                );
            }
        }
    }

    Ok(())
}

fn handle_worker_activation_request(
    swarm: &mut Swarm<GridBehaviour>,
    worker: &WorkerConfig,
    activation_relays: &[NodeAdvertisement],
    mut request: ActivationRequest,
    channel: request_response::ResponseChannel<ActivationResponse>,
    pending_forwards: &mut HashMap<request_response::OutboundRequestId, PendingOutbound>,
) -> Result<()> {
    if request
        .current_hop()
        .is_some_and(|hop| hop.peer_id != worker.peer_id)
    {
        return forward_activation_request(
            swarm,
            &worker.peer_id,
            activation_relays,
            request,
            channel,
            pending_forwards,
        );
    }

    loop {
        match process_activation_step(worker, request) {
            Ok(ActivationStep::Final(response)) => {
                send_response(swarm, channel, response);
                return Ok(());
            }
            Ok(ActivationStep::Forward(next_request)) => {
                if next_request
                    .current_hop()
                    .is_some_and(|hop| hop.peer_id == worker.peer_id)
                {
                    request = next_request;
                    continue;
                }
                return forward_activation_request(
                    swarm,
                    &worker.peer_id,
                    activation_relays,
                    next_request,
                    channel,
                    pending_forwards,
                );
            }
            Err(response) => {
                send_response(swarm, channel, response);
                return Ok(());
            }
        }
    }
}

fn handle_cached_activation_event(
    swarm: &mut Swarm<GridBehaviour>,
    cache: &ShardCache,
    peer_id: &str,
    activation_relays: &[NodeAdvertisement],
    network_event: ActivationNetworkEvent,
    pending_forwards: &mut HashMap<request_response::OutboundRequestId, PendingOutbound>,
) -> Result<()> {
    match network_event {
        ActivationNetworkEvent::Request { request, channel } => {
            handle_cached_activation_request(
                swarm,
                cache,
                peer_id,
                activation_relays,
                request,
                channel,
                pending_forwards,
            )?;
        }
        ActivationNetworkEvent::Response {
            request_id,
            response,
        } => {
            if let Some(PendingOutbound::Forward { channel, .. }) =
                pending_forwards.remove(&request_id)
            {
                send_response(swarm, channel, response);
            }
        }
        ActivationNetworkEvent::OutboundFailure {
            peer,
            request_id,
            error,
        } => {
            if let Some(PendingOutbound::Forward {
                channel,
                trace_id,
                peer_id,
                trace,
            }) = pending_forwards.remove(&request_id)
            {
                send_response(
                    swarm,
                    channel,
                    ActivationResponse::failure(
                        trace_id,
                        peer_id,
                        format!("forward to {peer} failed: {error}"),
                        trace,
                    ),
                );
            }
        }
    }

    Ok(())
}

fn handle_cached_activation_request(
    swarm: &mut Swarm<GridBehaviour>,
    cache: &ShardCache,
    peer_id: &str,
    activation_relays: &[NodeAdvertisement],
    mut request: ActivationRequest,
    channel: request_response::ResponseChannel<ActivationResponse>,
    pending_forwards: &mut HashMap<request_response::OutboundRequestId, PendingOutbound>,
) -> Result<()> {
    loop {
        let trace_id = request.trace_id;
        if request
            .current_hop()
            .is_some_and(|hop| hop.peer_id != peer_id)
        {
            return forward_activation_request(
                swarm,
                peer_id,
                activation_relays,
                request,
                channel,
                pending_forwards,
            );
        }

        let worker = match worker_config_for_activation(cache, peer_id, &request) {
            Ok(worker) => worker,
            Err(error) => {
                send_response(
                    swarm,
                    channel,
                    ActivationResponse::failure(
                        trace_id,
                        peer_id.to_owned(),
                        error.to_string(),
                        request.trace,
                    ),
                );
                return Ok(());
            }
        };

        match process_activation_step(&worker, request) {
            Ok(ActivationStep::Final(response)) => {
                send_response(swarm, channel, response);
                return Ok(());
            }
            Ok(ActivationStep::Forward(next_request)) => {
                if next_request
                    .current_hop()
                    .is_some_and(|hop| hop.peer_id == peer_id)
                {
                    request = next_request;
                    continue;
                }
                return forward_activation_request(
                    swarm,
                    peer_id,
                    activation_relays,
                    next_request,
                    channel,
                    pending_forwards,
                );
            }
            Err(response) => {
                send_response(swarm, channel, response);
                return Ok(());
            }
        }
    }
}

fn forward_activation_request(
    swarm: &mut Swarm<GridBehaviour>,
    local_peer_id: &str,
    activation_relays: &[NodeAdvertisement],
    request: ActivationRequest,
    channel: request_response::ResponseChannel<ActivationResponse>,
    pending_forwards: &mut HashMap<request_response::OutboundRequestId, PendingOutbound>,
) -> Result<()> {
    let trace_id = request.trace_id;
    let next_hop = request
        .current_hop()
        .cloned()
        .ok_or_else(|| anyhow!("forwarded request has no current hop"))?;
    match send_activation_request_with_relays(swarm, activation_relays, &next_hop, request.clone())
    {
        Ok(request_id) => {
            pending_forwards.insert(
                request_id,
                PendingOutbound::Forward {
                    channel,
                    trace_id,
                    peer_id: local_peer_id.to_owned(),
                    trace: request.trace,
                },
            );
        }
        Err(error) => {
            send_response(
                swarm,
                channel,
                ActivationResponse::failure(
                    trace_id,
                    local_peer_id.to_owned(),
                    format!("failed to forward activation: {error:#}"),
                    request.trace.clone(),
                ),
            );
        }
    }
    Ok(())
}

fn worker_config_for_activation(
    cache: &ShardCache,
    peer_id: &str,
    request: &ActivationRequest,
) -> Result<WorkerConfig> {
    let hop = request
        .current_hop()
        .ok_or_else(|| anyhow!("missing route hop {}", request.current_hop_index))?;
    let (manifest, _) = executable_seed_manifest_for_layers(cache, &request.model_id, hop.layers)?
        .ok_or_else(|| {
            anyhow!(
                "peer {} does not have executable shard {} {}:{}",
                peer_id,
                request.model_id,
                hop.layers.start,
                hop.layers.end
            )
        })?;

    Ok(WorkerConfig {
        peer_id: peer_id.to_owned(),
        model_id: manifest.model_id,
        runtime_kind: manifest.runtime_kind,
        owned_layers: manifest.layers,
        hidden_size: manifest.hidden_size,
        shard_cache: Some(cache.config().clone()),
    })
}

fn handle_model_network_event(
    swarm: &mut Swarm<GridBehaviour>,
    shard_cache: Option<&ShardCache>,
    peer_id: &str,
    network_event: ModelNetworkEvent,
) -> Result<()> {
    if let ModelNetworkEvent::Request { request, channel } = network_event {
        let response = match shard_cache {
            Some(cache) => model_shard_response_from_cache(cache, peer_id, &request),
            None => ModelShardResponse::failure(
                &request,
                peer_id.to_owned(),
                "node has no model shard cache configured",
            ),
        };
        send_model_response(swarm, channel, response);
    }

    Ok(())
}

fn handle_model_blob_network_event(
    swarm: &mut Swarm<GridBehaviour>,
    shard_cache: Option<&ShardCache>,
    peer_id: &str,
    network_event: ModelBlobNetworkEvent,
) -> Result<()> {
    if let ModelBlobNetworkEvent::Request { request, channel } = network_event {
        let response = match shard_cache {
            Some(cache) => model_blob_response_from_cache(cache, peer_id, &request),
            None => ModelBlobResponse::failure(
                &request,
                peer_id.to_owned(),
                "node has no model source cache configured",
            ),
        };
        send_model_blob_response(swarm, channel, response);
    }

    Ok(())
}

fn model_shard_response_from_cache(
    cache: &ShardCache,
    peer_id: &str,
    request: &ModelShardRequest,
) -> ModelShardResponse {
    if request.protocol_version != PROTOCOL_VERSION {
        return ModelShardResponse::failure(
            request,
            peer_id.to_owned(),
            format!(
                "unsupported model protocol version {}; expected {}",
                request.protocol_version, PROTOCOL_VERSION
            ),
        );
    }

    let record = match cache.find(
        &request.model_id,
        request.layers,
        request.checksum.as_deref(),
        request.version.as_deref(),
    ) {
        Ok(Some(record)) => record,
        Ok(None) => {
            return ModelShardResponse::failure(
                request,
                peer_id.to_owned(),
                format!(
                    "shard not found: {} {}:{} checksum {} version {}",
                    request.model_id,
                    request.layers.start,
                    request.layers.end,
                    request.checksum.as_deref().unwrap_or("<any>"),
                    request.version.as_deref().unwrap_or("<any>")
                ),
            );
        }
        Err(error) => {
            return ModelShardResponse::failure(request, peer_id.to_owned(), error.to_string());
        }
    };

    if is_executable_shard_record(&record) {
        return ModelShardResponse::failure(
            request,
            peer_id.to_owned(),
            "physical GGUF shards must be fetched with the chunked model blob protocol",
        );
    }

    match cache.read_payload(&record.info) {
        Ok(payload) => {
            ModelShardResponse::success(request, peer_id.to_owned(), record.info, payload)
        }
        Err(error) => ModelShardResponse::failure(request, peer_id.to_owned(), error.to_string()),
    }
}

fn model_blob_response_from_cache(
    cache: &ShardCache,
    peer_id: &str,
    request: &ModelBlobRequest,
) -> ModelBlobResponse {
    if request.protocol_version != PROTOCOL_VERSION {
        return ModelBlobResponse::failure(
            request,
            peer_id.to_owned(),
            format!(
                "unsupported model blob protocol version {}; expected {}",
                request.protocol_version, PROTOCOL_VERSION
            ),
        );
    }

    if request.max_bytes == 0 {
        return ModelBlobResponse::failure(
            request,
            peer_id.to_owned(),
            "model blob request max_bytes must be greater than zero",
        );
    }

    if let Some(layers) = request.layers {
        let record = match cache.find(
            &request.model_id,
            layers,
            Some(&request.source_checksum),
            None,
        ) {
            Ok(Some(record)) if is_executable_shard_record(&record) => record,
            Ok(Some(_)) => {
                return ModelBlobResponse::failure(
                    request,
                    peer_id.to_owned(),
                    format!(
                        "cached shard {} {}:{} is not executable",
                        request.model_id, layers.start, layers.end
                    ),
                );
            }
            Ok(None) => {
                return ModelBlobResponse::failure(
                    request,
                    peer_id.to_owned(),
                    format!(
                        "executable shard not found: {} {}:{} checksum {}",
                        request.model_id, layers.start, layers.end, request.source_checksum
                    ),
                );
            }
            Err(error) => {
                return ModelBlobResponse::failure(request, peer_id.to_owned(), error.to_string());
            }
        };

        if request.offset >= record.info.size_bytes {
            return ModelBlobResponse::success(
                request,
                peer_id.to_owned(),
                record.info.size_bytes,
                Vec::new(),
            );
        }
        let bytes_to_read = request
            .max_bytes
            .min(MODEL_BLOB_CHUNK_BYTES)
            .min((record.info.size_bytes - request.offset).min(u64::from(u32::MAX)) as u32);
        return match read_source_chunk(&record.path, request.offset, bytes_to_read as usize) {
            Ok(payload) => ModelBlobResponse::success(
                request,
                peer_id.to_owned(),
                record.info.size_bytes,
                payload,
            ),
            Err(error) => {
                ModelBlobResponse::failure(request, peer_id.to_owned(), error.to_string())
            }
        };
    }

    let (manifest, source_path) = match executable_seed_manifest_for_source(
        cache,
        &request.model_id,
        &request.source_checksum,
    ) {
        Ok(Some(value)) => value,
        Ok(None) => {
            return ModelBlobResponse::failure(
                request,
                peer_id.to_owned(),
                format!(
                    "source GGUF not found for {} checksum {}",
                    request.model_id, request.source_checksum
                ),
            );
        }
        Err(error) => {
            return ModelBlobResponse::failure(request, peer_id.to_owned(), error.to_string());
        }
    };

    let total_size_bytes = manifest.source.file_size_bytes;
    if request.offset >= total_size_bytes {
        return ModelBlobResponse::success(
            request,
            peer_id.to_owned(),
            total_size_bytes,
            Vec::new(),
        );
    }

    let bytes_to_read = request
        .max_bytes
        .min(MODEL_BLOB_CHUNK_BYTES)
        .min((total_size_bytes - request.offset).min(u64::from(u32::MAX)) as u32);

    match read_source_chunk(&source_path, request.offset, bytes_to_read as usize) {
        Ok(payload) => {
            ModelBlobResponse::success(request, peer_id.to_owned(), total_size_bytes, payload)
        }
        Err(error) => ModelBlobResponse::failure(request, peer_id.to_owned(), error.to_string()),
    }
}

fn executable_seed_manifest_for_source(
    cache: &ShardCache,
    model_id: &str,
    source_checksum: &str,
) -> Result<Option<(SeedShardManifest, PathBuf)>> {
    for record in cache.list()? {
        let Some(manifest) = record.manifest.clone() else {
            continue;
        };
        if manifest.model_id != model_id || manifest.source.checksum_sha256 != source_checksum {
            continue;
        }
        if is_executable_shard_record(&record) {
            return Ok(Some((manifest, record.path)));
        }
    }

    Ok(None)
}

fn executable_seed_manifest_for_layers(
    cache: &ShardCache,
    model_id: &str,
    layers: LayerRange,
) -> Result<Option<(SeedShardManifest, PathBuf)>> {
    for record in cache.list()? {
        if record.info.model_id != model_id || record.info.layers != layers {
            continue;
        }
        let Some(manifest) = record.manifest.clone() else {
            continue;
        };
        if manifest.model_id != model_id || manifest.layers != layers {
            continue;
        }
        if manifest.runtime_kind == RuntimeKind::Demo {
            return Ok(Some((manifest, PathBuf::new())));
        }
        if is_executable_shard_record(&record) {
            return Ok(Some((manifest, record.path)));
        }
    }

    Ok(None)
}

fn read_source_chunk(path: &Path, offset: u64, len: usize) -> Result<Vec<u8>> {
    let mut file =
        fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    file.seek(SeekFrom::Start(offset))
        .with_context(|| format!("failed to seek {}", path.display()))?;
    let mut payload = vec![0_u8; len];
    let read = file
        .read(&mut payload)
        .with_context(|| format!("failed to read {}", path.display()))?;
    payload.truncate(read);
    Ok(payload)
}

fn send_activation_request(
    swarm: &mut Swarm<GridBehaviour>,
    hop: &RouteHop,
    request: ActivationRequest,
) -> Result<request_response::OutboundRequestId> {
    let peer_id = hop
        .peer_id
        .parse::<PeerId>()
        .with_context(|| format!("invalid libp2p peer id {}", hop.peer_id))?;
    let addresses = hop_addresses(hop)?;
    let request_id = if addresses.is_empty() {
        swarm
            .behaviour_mut()
            .activation
            .send_request(&peer_id, request)
    } else {
        swarm
            .behaviour_mut()
            .activation
            .send_request_with_addresses(&peer_id, request, addresses)
    };

    Ok(request_id)
}

fn send_activation_request_with_relays(
    swarm: &mut Swarm<GridBehaviour>,
    activation_relays: &[NodeAdvertisement],
    hop: &RouteHop,
    request: ActivationRequest,
) -> Result<request_response::OutboundRequestId> {
    let local_peer_id = swarm.local_peer_id().to_string();
    let relay_hop = activation_relay_hop(&local_peer_id, activation_relays, hop);
    let send_hop = relay_hop.as_ref().unwrap_or(hop);
    if send_hop.peer_id != hop.peer_id {
        println!(
            "activation_relay target_peer={} relay_peer={} trace_id={}",
            hop.peer_id, send_hop.peer_id, request.trace_id
        );
    }
    send_activation_request(swarm, send_hop, request)
}

fn activation_relay_hop(
    local_peer_id: &str,
    activation_relays: &[NodeAdvertisement],
    target_hop: &RouteHop,
) -> Option<RouteHop> {
    if route_hop_is_loopback(target_hop) {
        return None;
    }

    activation_relays
        .iter()
        .filter(|relay| relay.peer_id != local_peer_id && relay.peer_id != target_hop.peer_id)
        .find_map(|relay| {
            relay.addresses.first().map(|address| RouteHop {
                peer_id: relay.peer_id.clone(),
                address: address.clone(),
                layers: target_hop.layers,
            })
        })
}

fn route_hop_is_loopback(hop: &RouteHop) -> bool {
    hop.address.contains("/ip4/127.")
        || hop.address.contains("/ip6/::1")
        || hop.address.contains("/ip6/0:0:0:0:0:0:0:1")
}

fn send_model_blob_request(
    swarm: &mut Swarm<GridBehaviour>,
    advertisement: &NodeAdvertisement,
    request: ModelBlobRequest,
) -> Result<request_response::OutboundRequestId> {
    let peer_id = advertisement
        .peer_id
        .parse::<PeerId>()
        .with_context(|| format!("invalid libp2p peer id {}", advertisement.peer_id))?;
    let addresses = advertisement
        .addresses
        .iter()
        .filter_map(|address| address.parse::<Multiaddr>().ok())
        .collect::<Vec<_>>();
    let request_id = if addresses.is_empty() {
        swarm.behaviour_mut().blob.send_request(&peer_id, request)
    } else {
        swarm
            .behaviour_mut()
            .blob
            .send_request_with_addresses(&peer_id, request, addresses)
    };

    Ok(request_id)
}

fn send_response(
    swarm: &mut Swarm<GridBehaviour>,
    channel: request_response::ResponseChannel<ActivationResponse>,
    response: ActivationResponse,
) {
    if let Err(response) = swarm
        .behaviour_mut()
        .activation
        .send_response(channel, response)
    {
        eprintln!(
            "failed to send activation response trace_id={} error={:?}",
            response.trace_id, response.error
        );
    }
}

fn send_model_response(
    swarm: &mut Swarm<GridBehaviour>,
    channel: request_response::ResponseChannel<ModelShardResponse>,
    response: ModelShardResponse,
) {
    if let Err(response) = swarm.behaviour_mut().model.send_response(channel, response) {
        eprintln!(
            "failed to send model shard response request_id={} error={:?}",
            response.request_id, response.error
        );
    }
}

fn send_model_blob_response(
    swarm: &mut Swarm<GridBehaviour>,
    channel: request_response::ResponseChannel<ModelBlobResponse>,
    response: ModelBlobResponse,
) {
    if let Err(response) = swarm.behaviour_mut().blob.send_response(channel, response) {
        eprintln!(
            "failed to send model blob response request_id={} offset={} error={:?}",
            response.request_id, response.offset, response.error
        );
    }
}

fn handle_grid_event(
    swarm: &mut Swarm<GridBehaviour>,
    event: SwarmEvent<GridBehaviourEvent>,
    registry: &mut ShardRegistry,
    advertisement: &mut Option<NodeAdvertisement>,
    topic: &gossipsub::IdentTopic,
    advertise_listen_addresses: bool,
    dial_discovered_peers: bool,
) -> Result<Option<GridNetworkEvent>> {
    match event {
        SwarmEvent::NewListenAddr { address, .. } => {
            let peer_id = *swarm.local_peer_id();
            if advertise_listen_addresses && update_listen_address(advertisement, peer_id, address)
            {
                if let Some(advertisement) = advertisement {
                    println!(
                        "libp2p_listen={}",
                        advertisement
                            .addresses
                            .last()
                            .map(String::as_str)
                            .unwrap_or("<no-address>")
                    );
                    publish_advertisement(swarm, topic, advertisement)?;
                }
            }
        }
        SwarmEvent::ConnectionEstablished {
            peer_id, endpoint, ..
        } => {
            swarm.behaviour_mut().gossipsub.add_explicit_peer(&peer_id);
            if let Some(advertisement) = observed_peer_advertisement(peer_id, &endpoint) {
                registry.merge(advertisement);
            }
            println!(
                "libp2p_connected peer_id={} endpoint={:?}",
                peer_id, endpoint
            );
        }
        SwarmEvent::ConnectionClosed {
            peer_id,
            endpoint,
            cause,
            ..
        } => {
            println!(
                "libp2p_disconnected peer_id={} endpoint={:?} cause={:?}",
                peer_id, endpoint, cause
            );
        }
        SwarmEvent::Behaviour(GridBehaviourEvent::Mdns(mdns::Event::Discovered(peers))) => {
            if dial_discovered_peers {
                for (peer_id, address) in peers {
                    swarm.behaviour_mut().gossipsub.add_explicit_peer(&peer_id);
                    swarm.add_peer_address(peer_id, address);
                }
            }
        }
        SwarmEvent::Behaviour(GridBehaviourEvent::Mdns(mdns::Event::Expired(peers))) => {
            for (peer_id, _) in peers {
                swarm
                    .behaviour_mut()
                    .gossipsub
                    .remove_explicit_peer(&peer_id);
            }
        }
        SwarmEvent::Behaviour(GridBehaviourEvent::Gossipsub(gossipsub::Event::Message {
            message,
            ..
        })) => {
            let advertisement = serde_json::from_slice::<NodeAdvertisement>(&message.data)?;
            if advertisement.protocol_version == PROTOCOL_VERSION {
                if dial_discovered_peers {
                    add_advertisement_addresses(swarm, &advertisement);
                }
                registry.upsert(advertisement);
            }
        }
        SwarmEvent::Behaviour(GridBehaviourEvent::Activation(
            request_response::Event::Message { message, .. },
        )) => match message {
            request_response::Message::Request {
                request, channel, ..
            } => {
                return Ok(Some(GridNetworkEvent::Activation(
                    ActivationNetworkEvent::Request { request, channel },
                )));
            }
            request_response::Message::Response {
                request_id,
                response,
            } => {
                return Ok(Some(GridNetworkEvent::Activation(
                    ActivationNetworkEvent::Response {
                        request_id,
                        response,
                    },
                )));
            }
        },
        SwarmEvent::Behaviour(GridBehaviourEvent::Activation(
            request_response::Event::OutboundFailure {
                peer,
                request_id,
                error,
                ..
            },
        )) => {
            return Ok(Some(GridNetworkEvent::Activation(
                ActivationNetworkEvent::OutboundFailure {
                    peer,
                    request_id,
                    error,
                },
            )));
        }
        SwarmEvent::Behaviour(GridBehaviourEvent::Model(request_response::Event::Message {
            message,
            ..
        })) => match message {
            request_response::Message::Request {
                request, channel, ..
            } => {
                return Ok(Some(GridNetworkEvent::Model(ModelNetworkEvent::Request {
                    request,
                    channel,
                })));
            }
            request_response::Message::Response {
                request_id,
                response,
            } => {
                return Ok(Some(GridNetworkEvent::Model(ModelNetworkEvent::Response {
                    request_id,
                    response,
                })));
            }
        },
        SwarmEvent::Behaviour(GridBehaviourEvent::Model(
            request_response::Event::OutboundFailure {
                peer,
                request_id,
                error,
                ..
            },
        )) => {
            return Ok(Some(GridNetworkEvent::Model(
                ModelNetworkEvent::OutboundFailure {
                    peer,
                    request_id,
                    error,
                },
            )));
        }
        SwarmEvent::Behaviour(GridBehaviourEvent::Blob(request_response::Event::Message {
            message,
            ..
        })) => match message {
            request_response::Message::Request {
                request, channel, ..
            } => {
                return Ok(Some(GridNetworkEvent::ModelBlob(
                    ModelBlobNetworkEvent::Request { request, channel },
                )));
            }
            request_response::Message::Response {
                request_id,
                response,
            } => {
                return Ok(Some(GridNetworkEvent::ModelBlob(
                    ModelBlobNetworkEvent::Response {
                        request_id,
                        response,
                    },
                )));
            }
        },
        SwarmEvent::Behaviour(GridBehaviourEvent::Blob(
            request_response::Event::OutboundFailure {
                peer,
                request_id,
                error,
                ..
            },
        )) => {
            return Ok(Some(GridNetworkEvent::ModelBlob(
                ModelBlobNetworkEvent::OutboundFailure {
                    peer,
                    request_id,
                    error,
                },
            )));
        }
        _ => {}
    }

    Ok(None)
}

fn build_grid_swarm(
    keypair: identity::Keypair,
    topic: &gossipsub::IdentTopic,
) -> Result<Swarm<GridBehaviour>> {
    let peer_id = keypair.public().to_peer_id();
    let mut swarm = SwarmBuilder::with_existing_identity(keypair)
        .with_tokio()
        .with_tcp(
            tcp::Config::default(),
            noise::Config::new,
            yamux::Config::default,
        )?
        .with_dns()?
        .with_behaviour(|key| {
            let gossipsub = gossipsub::Behaviour::new(
                gossipsub::MessageAuthenticity::Signed(key.clone()),
                gossipsub::Config::default(),
            )?;
            let mdns = mdns::tokio::Behaviour::new(mdns::Config::default(), peer_id)?;
            let activation = request_response::json::Behaviour::new(
                [(
                    StreamProtocol::new(ACTIVATION_PROTOCOL),
                    request_response::ProtocolSupport::Full,
                )],
                request_response::Config::default().with_request_timeout(Duration::from_secs(5)),
            );
            let model = request_response::json::Behaviour::new(
                [(
                    StreamProtocol::new(MODEL_PROTOCOL),
                    request_response::ProtocolSupport::Full,
                )],
                request_response::Config::default().with_request_timeout(Duration::from_secs(20)),
            );
            let blob = request_response::Behaviour::new(
                [(
                    StreamProtocol::new(MODEL_BLOB_PROTOCOL),
                    request_response::ProtocolSupport::Full,
                )],
                request_response::Config::default().with_request_timeout(Duration::from_secs(60)),
            );

            Ok(GridBehaviour {
                gossipsub,
                mdns,
                activation,
                model,
                blob,
            })
        })?
        .build();

    swarm.behaviour_mut().gossipsub.subscribe(topic)?;

    Ok(swarm)
}

fn listen_on(swarm: &mut Swarm<GridBehaviour>, listen: &str) -> Result<()> {
    let p2p_listen = listen
        .parse::<Multiaddr>()
        .with_context(|| format!("invalid libp2p listen address {listen}"))?;
    swarm.listen_on(p2p_listen)?;
    Ok(())
}

fn publish_advertisement(
    swarm: &mut Swarm<GridBehaviour>,
    topic: &gossipsub::IdentTopic,
    advertisement: &NodeAdvertisement,
) -> Result<()> {
    let bytes = serde_json::to_vec(advertisement)?;
    swarm
        .behaviour_mut()
        .gossipsub
        .publish(topic.clone(), bytes)
        .map(|_| ())
        .or_else(|error| match error {
            gossipsub::PublishError::NoPeersSubscribedToTopic => Ok(()),
            error => Err(anyhow!("failed to publish shard advertisement: {error}")),
        })
}

fn publish_known_advertisements(
    swarm: &mut Swarm<GridBehaviour>,
    topic: &gossipsub::IdentTopic,
    registry: &ShardRegistry,
    local_peer_id: PeerId,
) -> Result<()> {
    let local_peer_id = local_peer_id.to_string();
    for advertisement in registry.advertisements() {
        if advertisement.peer_id == local_peer_id {
            continue;
        }
        publish_advertisement(swarm, topic, &advertisement)?;
    }

    Ok(())
}

fn update_listen_address(
    advertisement: &mut Option<NodeAdvertisement>,
    peer_id: PeerId,
    address: Multiaddr,
) -> bool {
    let Some(advertisement) = advertisement else {
        return false;
    };

    let address = match address.with_p2p(peer_id) {
        Ok(address) | Err(address) => address.to_string(),
    };
    if advertisement.addresses.contains(&address) {
        return false;
    }

    advertisement.addresses.push(address);
    true
}

fn add_static_peer_addresses(
    swarm: &mut Swarm<GridBehaviour>,
    advertisements: &[NodeAdvertisement],
) {
    for advertisement in advertisements {
        add_advertisement_addresses(swarm, advertisement);
    }
}

fn add_advertisement_addresses(
    swarm: &mut Swarm<GridBehaviour>,
    advertisement: &NodeAdvertisement,
) {
    let Ok(peer_id) = advertisement.peer_id.parse::<PeerId>() else {
        return;
    };

    swarm.behaviour_mut().gossipsub.add_explicit_peer(&peer_id);
    for address in &advertisement.addresses {
        if let Ok(address) = address.parse::<Multiaddr>() {
            swarm.add_peer_address(peer_id, address.clone());
            let _ = swarm.dial(address);
        }
    }
}

#[derive(Debug, Clone)]
struct ModelShardCandidate {
    advertisement: NodeAdvertisement,
    shard: ModelShardInfo,
    seed_manifest: Option<SeedShardManifest>,
}

fn select_model_shard_candidate(
    registry: &ShardRegistry,
    local_peer_id: &str,
    model_id: &str,
    layers: LayerRange,
    checksum: Option<&str>,
    version: Option<&str>,
    failed_peers: &[String],
) -> Option<ModelShardCandidate> {
    registry
        .advertisements()
        .into_iter()
        .filter(|advertisement| advertisement.peer_id != local_peer_id)
        .filter(|advertisement| !failed_peers.contains(&advertisement.peer_id))
        .flat_map(|advertisement| {
            let hosted_shards = advertisement.hosted_shards.clone();
            advertisement
                .model_shards
                .clone()
                .into_iter()
                .filter_map(move |shard| {
                    let seed_manifest = hosted_shards
                        .iter()
                        .find(|descriptor| {
                            descriptor.model_id == shard.model_id
                                && descriptor.layers == shard.layers
                        })
                        .and_then(|descriptor| descriptor.seed_manifest.as_deref())
                        .filter(|manifest| {
                            manifest.payload_kind == model_distribution::PAYLOAD_KIND_GGUF_SHARD
                        })
                        .cloned();

                    seed_manifest.map(|seed_manifest| ModelShardCandidate {
                        advertisement: advertisement.clone(),
                        shard,
                        seed_manifest: Some(seed_manifest),
                    })
                })
        })
        .filter(|candidate| {
            candidate.shard.model_id == model_id
                && candidate.shard.layers == layers
                && checksum.is_none_or(|checksum| candidate.shard.checksum == checksum)
                && version.is_none_or(|version| candidate.shard.version == version)
        })
        .min_by_key(|candidate| {
            (
                candidate.advertisement.latency_hint_ms.unwrap_or(u32::MAX),
                candidate.shard.size_bytes,
                candidate.advertisement.peer_id.clone(),
            )
        })
}

fn select_model_blob_candidate(
    registry: &ShardRegistry,
    local_peer_id: &str,
    model_id: &str,
    source_checksum: &str,
    failed_peers: &[String],
) -> Option<NodeAdvertisement> {
    registry
        .advertisements()
        .into_iter()
        .filter(|advertisement| advertisement.peer_id != local_peer_id)
        .filter(|advertisement| !failed_peers.contains(&advertisement.peer_id))
        .filter(|advertisement| {
            advertisement.hosted_shards.iter().any(|descriptor| {
                descriptor.model_id == model_id
                    && descriptor
                        .seed_manifest
                        .as_deref()
                        .is_some_and(|manifest| manifest.source.checksum_sha256 == source_checksum)
            })
        })
        .min_by_key(|advertisement| {
            (
                advertisement.latency_hint_ms.unwrap_or(u32::MAX),
                advertisement.peer_id.clone(),
            )
        })
}

fn hop_addresses(hop: &RouteHop) -> Result<Vec<Multiaddr>> {
    if hop.address.trim().is_empty() {
        return Ok(Vec::new());
    }

    Ok(vec![hop.address.parse::<Multiaddr>().with_context(
        || format!("invalid route multiaddr {}", hop.address),
    )?])
}

fn sleep_until(deadline: Instant) -> tokio::time::Sleep {
    sleep(deadline.saturating_duration_since(Instant::now()))
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

fn elapsed_ms(started: Instant) -> u64 {
    started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64
}

fn log_hop(trace_id: uuid::Uuid, event: &TraceEvent) {
    println!(
        "trace_id={} peer={} layers={}:{} next={} activation_bytes={} timing_ms={} checksum={:016x}",
        trace_id,
        event.peer_id,
        event.layers.start,
        event.layers.end,
        event.next_peer_id.as_deref().unwrap_or("<final>"),
        event.activation_size_bytes,
        event.timing_ms,
        event.activation_checksum
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn worker(peer_id: &str, layers: LayerRange) -> WorkerConfig {
        WorkerConfig {
            peer_id: peer_id.to_owned(),
            model_id: "grid-demo-12".to_owned(),
            runtime_kind: RuntimeKind::Demo,
            owned_layers: layers,
            hidden_size: 4,
            shard_cache: None,
        }
    }

    fn hop(peer_id: &str, start: u32, end: u32) -> RouteHop {
        RouteHop {
            peer_id: peer_id.to_owned(),
            address: String::new(),
            layers: LayerRange::new(start, end).unwrap(),
        }
    }

    fn relay(peer_id: &str) -> NodeAdvertisement {
        empty_advertisement(
            peer_id.to_owned(),
            format!("/ip4/203.0.113.10/tcp/9777/p2p/{peer_id}"),
        )
    }

    #[test]
    fn activation_step_forwards_to_next_hop() {
        let route = vec![hop("peer-a", 0, 3), hop("peer-b", 3, 6)];
        let request = ActivationRequest::new("grid-demo-12", route, 4, vec![0.1; 4], None);

        let step =
            process_activation_step(&worker("peer-a", LayerRange::new(0, 3).unwrap()), request)
                .unwrap();

        match step {
            ActivationStep::Forward(request) => {
                assert_eq!(request.current_hop_index, 1);
                assert_eq!(request.current_hop().unwrap().peer_id, "peer-b");
                assert_eq!(request.trace.len(), 1);
                assert_eq!(request.trace[0].next_peer_id.as_deref(), Some("peer-b"));
            }
            ActivationStep::Final(_) => panic!("expected forwarded activation"),
        }
    }

    #[test]
    fn activation_step_rejects_wrong_layer_range() {
        let route = vec![hop("peer-a", 3, 6)];
        let request = ActivationRequest::new("grid-demo-12", route, 4, vec![0.1; 4], None);

        let response =
            process_activation_step(&worker("peer-a", LayerRange::new(0, 3).unwrap()), request)
                .unwrap_err();

        assert!(
            response
                .error
                .unwrap()
                .contains("route requested LayerRange")
        );
    }

    #[test]
    fn activation_relay_selected_for_non_loopback_remote_hop() {
        let target = RouteHop {
            peer_id: "peer-b".to_owned(),
            address: "/ip4/10.0.0.2/tcp/9777/p2p/peer-b".to_owned(),
            layers: LayerRange::new(0, 3).unwrap(),
        };

        let relay = activation_relay_hop("peer-a", &[relay("relay-peer")], &target).unwrap();

        assert_eq!(relay.peer_id, "relay-peer");
        assert_eq!(relay.layers, target.layers);
    }

    #[test]
    fn activation_relay_skips_loopback_target() {
        let target = RouteHop {
            peer_id: "peer-b".to_owned(),
            address: "/ip4/127.0.0.1/tcp/9777/p2p/peer-b".to_owned(),
            layers: LayerRange::new(0, 3).unwrap(),
        };

        assert!(activation_relay_hop("peer-a", &[relay("relay-peer")], &target).is_none());
    }

    #[tokio::test]
    async fn model_blob_codec_roundtrips_raw_payload() {
        let protocol = StreamProtocol::new(MODEL_BLOB_PROTOCOL);
        let request = ModelBlobRequest::new("gemma", "source-checksum", 8, 16);
        let mut writer = futures::io::Cursor::new(Vec::new());
        let mut codec = ModelBlobCodec;
        request_response::Codec::write_request(&mut codec, &protocol, &mut writer, request.clone())
            .await
            .unwrap();

        let mut reader = futures::io::Cursor::new(writer.into_inner());
        let mut codec = ModelBlobCodec;
        let decoded = request_response::Codec::read_request(&mut codec, &protocol, &mut reader)
            .await
            .unwrap();
        assert_eq!(decoded, request);

        let response = ModelBlobResponse::success(&request, "peer-a", 32, vec![1, 2, 3, 4]);
        let mut writer = futures::io::Cursor::new(Vec::new());
        let mut codec = ModelBlobCodec;
        request_response::Codec::write_response(
            &mut codec,
            &protocol,
            &mut writer,
            response.clone(),
        )
        .await
        .unwrap();

        let mut reader = futures::io::Cursor::new(writer.into_inner());
        let mut codec = ModelBlobCodec;
        let decoded = request_response::Codec::read_response(&mut codec, &protocol, &mut reader)
            .await
            .unwrap();
        assert_eq!(decoded, response);
    }

    #[test]
    fn model_blob_response_serves_chunk_from_physical_shard() {
        let root = std::env::temp_dir().join(format!("infernet-blob-{}", uuid::Uuid::new_v4()));
        let cache_config = ShardCacheConfig::new(root.join("shards"));
        let cache = ShardCache::new(cache_config).unwrap();
        fs::create_dir_all(&root).unwrap();
        let shard_file = root.join("gemma-0-8.gguf");
        fs::write(&shard_file, b"0123456789abcdef").unwrap();
        let checksum = sha256_file(&shard_file).unwrap();
        let layers = LayerRange::new(0, 8).unwrap();
        let manifest = SeedShardManifest {
            model_id: "gemma".to_owned(),
            display_name: "Gemma".to_owned(),
            architecture: "gemma".to_owned(),
            layer_count: 8,
            hidden_size: 16,
            activation_dtype: "f16".to_owned(),
            runtime_kind: RuntimeKind::LlamaCpp,
            layers,
            tokenizer: infernet_model::TokenizerCompatibility {
                family: "gemma".to_owned(),
                checksum: None,
            },
            metadata: infernet_model::ShardMetadata {
                architecture: "gemma".to_owned(),
                quantization: Some("IQ4_XS".to_owned()),
                source_checksum: Some(checksum.clone()),
                protocol_version: PROTOCOL_VERSION,
            },
            source: infernet_model::SeedSourceMetadata {
                path: "/seed/gemma.gguf".to_owned(),
                checksum_sha256: "source-checksum".to_owned(),
                file_size_bytes: 16,
            },
            shard_hash: "seed-hash".to_owned(),
            payload_kind: model_distribution::PAYLOAD_KIND_GGUF_SHARD.to_owned(),
        };
        cache
            .import_physical_shard_file(&shard_file, "gemma", layers, "v1", manifest)
            .unwrap();

        let request = ModelBlobRequest::new_shard("gemma", layers, checksum, 4, 6);
        let response = model_blob_response_from_cache(&cache, "peer-a", &request);

        assert!(response.error.is_none(), "{:?}", response.error);
        assert_eq!(response.layers, Some(layers));
        assert_eq!(response.offset, 4);
        assert_eq!(response.total_size_bytes, 16);
        assert_eq!(response.payload, b"456789");

        let _ = fs::remove_dir_all(root);
    }
}
