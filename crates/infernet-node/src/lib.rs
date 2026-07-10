pub mod capabilities;
pub mod llama_server_runtime;
pub mod model_distribution;
pub mod rpc_runtime;
pub mod rpc_tunnel;

use std::collections::{HashMap, HashSet};
use std::env;
use std::io::{Read, Seek, SeekFrom, Write};
use std::mem;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use std::{fs, io};

use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use futures::{
    AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, StreamExt,
    channel::{mpsc, oneshot},
};
use infernet_model::{
    LayerRange, ModelManifest, OfficialModelRelease, RuntimeKind, SeedShardManifest,
    ShardDescriptor,
};
use infernet_protocol::{
    ACTIVATION_PROTOCOL, ActivationRequest, ActivationResponse, MODEL_BLOB_PROTOCOL,
    MODEL_PROTOCOL, ModelBlobRequest, ModelBlobResponse, ModelShardInfo, ModelShardRequest,
    ModelShardResponse, NodeAdvertisement, PROTOCOL_VERSION, PromptMetadata, RouteHop, TraceEvent,
};
use infernet_router::ShardRegistry;
use infernet_runtime::{DemoRuntime, LayerRuntime, activation_checksum};
use libp2p::{
    Multiaddr, PeerId, StreamProtocol, Swarm, SwarmBuilder,
    core::{connection::ConnectedPoint, transport::ListenerId},
    dcutr, gossipsub, identify, identity, mdns,
    multiaddr::Protocol,
    noise, ping, relay, request_response,
    swarm::{
        NetworkBehaviour, SwarmEvent,
        behaviour::toggle::Toggle,
        dial_opts::{DialOpts, PeerCondition},
    },
    tcp, yamux,
};
pub use model_distribution::{
    CachedShardRecord, INFERNET_FULL_MODEL_RUNTIME_ABI, INFERNET_SHARD_FORMAT_VERSION,
    INFERNET_SHARD_MANIFEST_FILE, INFERNET_SHARD_RUNTIME_ABI, INFERNET_SHARD_TENSOR_FILE,
    InfernetShardPackageManifest, InfernetShardPayloadManifest, PAYLOAD_KIND_FULL_MODEL,
    PAYLOAD_KIND_GGUF_SHARD, PAYLOAD_KIND_INFERNET_SHARD, PAYLOAD_KIND_METADATA_ONLY,
    SeedShardBuildProgress, SeededModelSummary, ShardCache, ShardCacheConfig, ShardCacheStats,
    executable_source_path_for_manifest, import_seed_model_from_file,
    import_seed_model_from_file_consuming, import_seed_model_from_file_consuming_verified,
    import_seed_model_from_file_with_build_progress, import_seed_model_from_file_with_progress,
    is_executable_shard_record, seed_manifest_for_network, sha256_bytes, sha256_file,
    source_cache_path, source_cache_root,
};
use serde::Deserialize;
use tokio::time::{Instant, interval, sleep};

pub use capabilities::{
    PINNED_GGML_RPC_PROTOCOL_VERSION, clear_local_llama_rpc_endpoint,
    configured_llama_rpc_endpoint, detect_node_capabilities, local_llama_rpc_target,
    set_local_inference_active, set_local_llama_rpc_endpoint, set_local_rpc_active,
    validate_llama_rpc_endpoint,
};
pub use llama_server_runtime::{
    LlamaServerCompletion, LlamaServerConfig, complete_with_persistent_llama_server,
    find_llama_server_binary, stop_persistent_llama_server,
};
pub use rpc_runtime::{
    INFERNET_LLAMA_RPC_RUNTIME_ABI, LLAMA_RPC_DEFAULT_PORT, LLAMA_RPC_PROTOCOL_VERSION,
    LlamaRpcServer, LlamaRpcServerConfig, find_llama_rpc_server_binary, spawn_llama_rpc_server,
};
pub use rpc_tunnel::{
    RPC_TUNNEL_PROTOCOL, RpcTunnelAdmission, RpcTunnelAdmissionLimits, RpcTunnelProxy,
    RpcTunnelProxyConfig, RpcTunnelTicket, RpcTunnelWorker, RpcTunnelWorkerConfig,
};

pub type ModelDistributionReadiness = oneshot::Sender<std::result::Result<String, String>>;
pub type ModelDistributionRegistryObserver = Arc<dyn Fn(ShardRegistry) + Send + Sync>;

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
    /// Enable local mDNS discovery. Long-lived desktop services generally
    /// enable it; one-shot fetch/infer swarms may disable it to avoid spawning
    /// short-lived interface watchers.
    pub enable_mdns: bool,
    pub relay_advertisements: bool,
    /// Run a Circuit Relay v2 hop service. This is intended for explicitly
    /// deployed public bootstrap nodes, never ordinary desktop nodes.
    pub relay_server: bool,
    /// Public relay server multiaddresses. Each address must end in the
    /// relay's `/p2p/<peer-id>` component. A reservation is maintained on
    /// every configured relay while direct TCP/QUIC paths remain enabled.
    pub relay_peers: Vec<String>,
    /// Explicitly trusted llama.cpp RPC backends selected by the caller for
    /// this request. These are validated again before they cross the network.
    pub rpc_endpoints: Vec<String>,
    /// Authenticated worker identities selected for RPC-over-libp2p. The
    /// coordinator creates loopback proxies for these peers.
    pub rpc_worker_peer_ids: Vec<String>,
    /// Exact route selected alongside `rpc_endpoints`. This prevents a second
    /// discovery pass from switching coordinators after the RPC plan is built.
    pub planned_route: Option<Vec<RouteHop>>,
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
            enable_mdns: true,
            relay_advertisements: false,
            relay_server: false,
            relay_peers: Vec::new(),
            rpc_endpoints: Vec::new(),
            rpc_worker_peer_ids: Vec::new(),
            planned_route: None,
        }
    }

    pub fn peer_id(&self) -> PeerId {
        self.keypair.public().to_peer_id()
    }

    pub fn set_trusted_rpc_endpoints(
        &mut self,
        endpoints: impl IntoIterator<Item = String>,
    ) -> Result<()> {
        self.rpc_endpoints =
            normalize_trusted_rpc_endpoints(&endpoints.into_iter().collect::<Vec<_>>())?;
        Ok(())
    }

    pub fn set_rpc_worker_peer_ids(
        &mut self,
        peers: impl IntoIterator<Item = String>,
    ) -> Result<()> {
        let mut normalized = Vec::new();
        let mut seen = HashSet::new();
        for peer in peers {
            let peer = peer.trim();
            peer.parse::<PeerId>()
                .with_context(|| format!("invalid RPC worker peer id {peer}"))?;
            if seen.insert(peer.to_owned()) {
                normalized.push(peer.to_owned());
            }
        }
        if normalized.len() > MAX_TRUSTED_RPC_ENDPOINTS {
            bail!(
                "distributed RPC requested {} workers; maximum is {}",
                normalized.len(),
                MAX_TRUSTED_RPC_ENDPOINTS
            );
        }
        self.rpc_worker_peer_ids = normalized;
        Ok(())
    }

    pub fn set_planned_route(&mut self, route: Vec<RouteHop>) {
        self.planned_route = Some(route);
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

/// Process-wide counters for model bytes accepted by the libp2p response
/// behaviour for serving to peers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct ModelServingTelemetry {
    pub bytes_served: u64,
    pub chunks_served: u64,
    pub last_activity_unix_ms: Option<u64>,
}

struct ModelServingTelemetryCounters {
    bytes_served: AtomicU64,
    chunks_served: AtomicU64,
    last_activity_unix_ms: AtomicU64,
}

impl ModelServingTelemetryCounters {
    const fn new() -> Self {
        Self {
            bytes_served: AtomicU64::new(0),
            chunks_served: AtomicU64::new(0),
            last_activity_unix_ms: AtomicU64::new(0),
        }
    }

    fn record_chunk(&self, bytes: u64, activity_unix_ms: u64) {
        if bytes == 0 {
            return;
        }
        self.bytes_served.fetch_add(bytes, Ordering::Relaxed);
        self.chunks_served.fetch_add(1, Ordering::Relaxed);
        self.last_activity_unix_ms
            .store(activity_unix_ms, Ordering::Relaxed);
    }

    fn snapshot(&self) -> ModelServingTelemetry {
        let last_activity_unix_ms = self.last_activity_unix_ms.load(Ordering::Relaxed);
        ModelServingTelemetry {
            bytes_served: self.bytes_served.load(Ordering::Relaxed),
            chunks_served: self.chunks_served.load(Ordering::Relaxed),
            last_activity_unix_ms: (last_activity_unix_ms != 0).then_some(last_activity_unix_ms),
        }
    }
}

static MODEL_SERVING_TELEMETRY: ModelServingTelemetryCounters =
    ModelServingTelemetryCounters::new();

struct PersistentRpcTunnelSet {
    worker_peer_ids: Vec<String>,
    endpoints: Vec<String>,
    _proxies: Vec<RpcTunnelProxy>,
}

static PERSISTENT_RPC_TUNNELS: OnceLock<Mutex<Option<PersistentRpcTunnelSet>>> = OnceLock::new();

/// Returns process-wide model serving totals. The counters are monotonic for
/// the lifetime of the process and include both full-source and shard chunks.
pub fn model_serving_telemetry() -> ModelServingTelemetry {
    MODEL_SERVING_TELEMETRY.snapshot()
}

async fn ensure_persistent_rpc_tunnels(
    control: libp2p_stream::Control,
    worker_peer_ids: &[String],
) -> Result<Vec<String>> {
    let state = PERSISTENT_RPC_TUNNELS.get_or_init(|| Mutex::new(None));
    if let Some(existing) = state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .as_ref()
        .filter(|existing| existing.worker_peer_ids == worker_peer_ids)
    {
        return Ok(existing.endpoints.clone());
    }

    let mut proxies = Vec::with_capacity(worker_peer_ids.len());
    let mut endpoints = Vec::with_capacity(worker_peer_ids.len());
    for peer_id in worker_peer_ids {
        let peer = peer_id
            .parse::<PeerId>()
            .with_context(|| format!("invalid selected RPC worker peer id {peer_id}"))?;
        let proxy = RpcTunnelProxy::bind(
            control.clone(),
            RpcTunnelProxyConfig::new(peer, RpcTunnelTicket::default()),
        )
        .await
        .with_context(|| format!("failed to start RPC tunnel proxy for worker {peer_id}"))?;
        endpoints.push(proxy.llama_cpp_endpoint());
        proxies.push(proxy);
    }

    *state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(PersistentRpcTunnelSet {
        worker_peer_ids: worker_peer_ids.to_vec(),
        endpoints: endpoints.clone(),
        _proxies: proxies,
    });
    Ok(endpoints)
}

pub fn stop_persistent_rpc_tunnels() {
    if let Some(state) = PERSISTENT_RPC_TUNNELS.get() {
        state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
    }
}

fn model_serving_now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}

const MODEL_BLOB_CHUNK_BYTES: u32 = 4 * 1024 * 1024;
const MODEL_BLOB_HEADER_MAX_BYTES: usize = 64 * 1024;
const MODEL_FETCH_PEER_RETRY_COOLDOWN: Duration = Duration::from_secs(5);
const IDENTIFY_PROTOCOL: &str = "/infernet/identify/1";
const IDENTIFY_AGENT: &str = concat!("infernet/", env!("CARGO_PKG_VERSION"));
const PUBLIC_RELAY_RESERVATION_DURATION: Duration = Duration::from_secs(2 * 60 * 60);
const PUBLIC_RELAY_CIRCUIT_DURATION: Duration = Duration::from_secs(2 * 60 * 60);
const PUBLIC_RELAY_MAX_CIRCUIT_BYTES: u64 = 64 * 1024 * 1024 * 1024;
const LLAMA_BRIDGE_EXECUTION_TIMEOUT: Duration = Duration::from_secs(5 * 60);
const LLAMA_BRIDGE_POLL_INTERVAL: Duration = Duration::from_millis(25);
const LLAMA_BRIDGE_MAX_LIBRARY_THREADS: usize = 4;
// llama.cpp supports at most 16 devices total, including the coordinator's
// local backend. Launch nodes expose one accelerator each; keep ample headroom
// for local and future multi-device hosts.
const MAX_TRUSTED_RPC_ENDPOINTS: usize = 8;

struct RemoveFileOnDrop(PathBuf);

impl Drop for RemoveFileOnDrop {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

#[derive(NetworkBehaviour)]
struct GridBehaviour {
    relay_client: relay::client::Behaviour,
    relay_server: Toggle<relay::Behaviour>,
    dcutr: dcutr::Behaviour,
    identify: identify::Behaviour,
    ping: ping::Behaviour,
    gossipsub: gossipsub::Behaviour,
    mdns: Toggle<mdns::tokio::Behaviour>,
    activation: request_response::json::Behaviour<ActivationRequest, ActivationResponse>,
    model: request_response::json::Behaviour<ModelShardRequest, ModelShardResponse>,
    blob: request_response::Behaviour<ModelBlobCodec>,
    rpc_tunnel: libp2p_stream::Behaviour,
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

enum LocalActivationOutcome {
    Response(ActivationResponse),
    Forward(ActivationRequest),
}

struct CompletedLocalActivation {
    job_id: uuid::Uuid,
    outcome: LocalActivationOutcome,
}

struct PendingLocalActivation {
    channel: request_response::ResponseChannel<ActivationResponse>,
    peer_id: String,
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

    let mut swarm = build_grid_swarm(
        discovery.keypair.clone(),
        &topic,
        discovery.relay_server,
        discovery.enable_mdns,
    )?;
    // Reserve relay circuits before dialing the same bootstrap as a static
    // peer. Starting the static dial first races libp2p's relay listener and
    // can permanently leave this node without a circuit address.
    start_grid_listeners(&mut swarm, &discovery)?;
    add_static_peer_addresses(&mut swarm, &discovery.static_peers, &discovery.relay_peers);
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
                add_static_peer_addresses(&mut swarm, &discovery.static_peers, &discovery.relay_peers);
            }
            _ = publish_interval.tick(), if discovery.advertisement.is_some() => {
                refresh_advertisement_model_shards(
                    &mut discovery.advertisement,
                    shard_cache.as_ref(),
                )?;
                publish_local_advertisement(
                    &mut swarm,
                    &topic,
                    &mut discovery.advertisement,
                    &worker.peer_id,
                )?;
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
    discovery: DiscoveryConfig,
    cache_config: ShardCacheConfig,
) -> Result<()> {
    let mut readiness = None;
    run_model_distribution_node_inner(discovery, cache_config, &mut readiness, None).await
}

pub async fn run_model_distribution_node_with_readiness(
    discovery: DiscoveryConfig,
    cache_config: ShardCacheConfig,
    readiness: ModelDistributionReadiness,
) -> Result<()> {
    let mut readiness = Some(readiness);
    let result =
        run_model_distribution_node_inner(discovery, cache_config, &mut readiness, None).await;

    if let Err(error) = &result {
        signal_model_distribution_readiness(&mut readiness, Err(format!("{error:#}")));
    }

    result
}

/// Runs the long-lived model service and publishes its current peer registry
/// to the owner. Desktop clients use this instead of creating a second swarm
/// for every UI refresh, which otherwise causes visible connection churn and
/// can race active model transfers.
pub async fn run_model_distribution_node_with_readiness_and_registry(
    discovery: DiscoveryConfig,
    cache_config: ShardCacheConfig,
    readiness: ModelDistributionReadiness,
    registry_observer: ModelDistributionRegistryObserver,
) -> Result<()> {
    let mut readiness = Some(readiness);
    let result = run_model_distribution_node_inner(
        discovery,
        cache_config,
        &mut readiness,
        Some(&registry_observer),
    )
    .await;

    if let Err(error) = &result {
        signal_model_distribution_readiness(&mut readiness, Err(format!("{error:#}")));
    }

    result
}

async fn run_model_distribution_node_inner(
    mut discovery: DiscoveryConfig,
    cache_config: ShardCacheConfig,
    readiness: &mut Option<ModelDistributionReadiness>,
    registry_observer: Option<&ModelDistributionRegistryObserver>,
) -> Result<()> {
    let topic = gossipsub::IdentTopic::new(discovery.topic.clone());
    let mut registry = ShardRegistry::new();
    registry.extend(discovery.static_peers.clone());
    notify_registry_observer(registry_observer, &registry);
    let shard_cache = ShardCache::new(cache_config)?;
    let peer_id = discovery.peer_id().to_string();

    if discovery.advertisement.is_none() {
        discovery.advertisement = Some(empty_advertisement(peer_id.clone(), String::new()));
    }
    refresh_advertisement_model_shards(&mut discovery.advertisement, Some(&shard_cache))?;

    let mut swarm = build_grid_swarm(
        discovery.keypair.clone(),
        &topic,
        discovery.relay_server,
        discovery.enable_mdns,
    )?;
    // The relay listener must own the first dial to a shared
    // bootstrap/relay peer so its reservation cannot lose a dial race.
    let listener_id = start_grid_listeners(&mut swarm, &discovery)?;
    add_static_peer_addresses(&mut swarm, &discovery.static_peers, &discovery.relay_peers);
    let rpc_tunnel_control = swarm.behaviour().rpc_tunnel.new_control();
    let _rpc_tunnel_worker = start_local_rpc_tunnel_worker(&swarm)?;

    let mut publish_interval = interval(discovery.publish_interval);
    let mut static_peer_dial_interval = interval(Duration::from_secs(10));
    let mut pending_forwards = HashMap::new();
    let (local_completion_sender, mut local_completion_receiver) = mpsc::unbounded();
    let mut pending_local_activations = HashMap::new();

    loop {
        tokio::select! {
            completed = local_completion_receiver.next() => {
                let completed = completed.ok_or_else(|| anyhow!("local inference completion channel closed"))?;
                handle_completed_local_activation(
                    &mut swarm,
                    &discovery.static_peers,
                    completed,
                    &mut pending_local_activations,
                    &mut pending_forwards,
                )?;
            }
            _ = static_peer_dial_interval.tick(), if !discovery.static_peers.is_empty() => {
                add_static_peer_addresses(&mut swarm, &discovery.static_peers, &discovery.relay_peers);
            }
            _ = publish_interval.tick() => {
                refresh_advertisement_model_shards(&mut discovery.advertisement, Some(&shard_cache))?;
                publish_local_advertisement(
                    &mut swarm,
                    &topic,
                    &mut discovery.advertisement,
                    &peer_id,
                )?;
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
                let ready_address = match &event {
                    SwarmEvent::NewListenAddr {
                        listener_id: event_listener_id,
                        address,
                    } if *event_listener_id == listener_id => Some(address.to_string()),
                    SwarmEvent::ListenerClosed {
                        listener_id: event_listener_id,
                        reason,
                        ..
                    } if *event_listener_id == listener_id => {
                        let reason = match reason {
                            Ok(()) => "listener closed unexpectedly".to_owned(),
                            Err(error) => error.to_string(),
                        };
                        bail!("libp2p listener {listener_id} closed: {reason}");
                    }
                    SwarmEvent::ListenerError {
                        listener_id: event_listener_id,
                        error,
                    } if *event_listener_id == listener_id => {
                        let phase = if readiness.is_some() {
                            "before becoming ready"
                        } else {
                            "after startup"
                        };
                        bail!("libp2p listener {listener_id} failed {phase}: {error}");
                    }
                    _ => None,
                };

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
                                &local_completion_sender,
                                &mut pending_local_activations,
                                rpc_tunnel_control.clone(),
                            )?;
                        }
                    }
                }
                notify_registry_observer(registry_observer, &registry);

                if let Some(address) = ready_address {
                    signal_model_distribution_readiness(readiness, Ok(address));
                }
            }
        }
    }
}

fn start_local_rpc_tunnel_worker(swarm: &Swarm<GridBehaviour>) -> Result<Option<RpcTunnelWorker>> {
    let Some(endpoint) = local_llama_rpc_target().filter(|endpoint| endpoint.ready) else {
        return Ok(None);
    };
    let host = endpoint
        .host
        .parse::<IpAddr>()
        .with_context(|| format!("invalid process-local RPC host {}", endpoint.host))?;
    let target = SocketAddr::new(host, endpoint.port);
    if !target.ip().is_loopback() {
        bail!("refusing non-loopback RPC tunnel target {target}");
    }
    let admission = RpcTunnelAdmission::allow_authenticated_peers(RpcTunnelAdmissionLimits {
        max_sessions: 1,
        max_sessions_per_peer: 1,
    })?;
    let mut control = swarm.behaviour().rpc_tunnel.new_control();
    let incoming = control
        .accept(RPC_TUNNEL_PROTOCOL)
        .map_err(|_| anyhow!("RPC tunnel protocol was already registered"))?;
    Ok(Some(RpcTunnelWorker::spawn(
        incoming,
        RpcTunnelWorkerConfig::new(target, admission),
    )?))
}

fn notify_registry_observer(
    observer: Option<&ModelDistributionRegistryObserver>,
    registry: &ShardRegistry,
) {
    if let Some(observer) = observer {
        observer(registry.clone());
    }
}

fn signal_model_distribution_readiness(
    readiness: &mut Option<ModelDistributionReadiness>,
    result: std::result::Result<String, String>,
) {
    if let Some(readiness) = readiness.take() {
        let _ = readiness.send(result);
    }
}

pub async fn discover_for(mut config: DiscoveryConfig, timeout: Duration) -> Result<ShardRegistry> {
    let topic = gossipsub::IdentTopic::new(config.topic.clone());
    let mut registry = ShardRegistry::new();
    registry.extend(config.static_peers.clone());
    if let Some(advertisement) = config.advertisement.clone() {
        registry.upsert(advertisement);
    }

    let mut swarm = build_grid_swarm(
        config.keypair.clone(),
        &topic,
        config.relay_server,
        config.enable_mdns,
    )?;
    start_grid_listeners(&mut swarm, &config)?;
    add_static_peer_addresses(&mut swarm, &config.static_peers, &config.relay_peers);

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
    config: DiscoveryConfig,
    cache_config: ShardCacheConfig,
    model_id: String,
    layers: LayerRange,
    checksum: Option<String>,
    version: Option<String>,
    discovery_timeout: Duration,
) -> Result<ModelFetchResult> {
    fetch_model_shard_over_libp2p_with_progress(
        config,
        cache_config,
        model_id,
        layers,
        checksum,
        version,
        discovery_timeout,
        |_, _| {},
    )
    .await
}

pub async fn fetch_model_shard_over_libp2p_with_progress(
    mut config: DiscoveryConfig,
    cache_config: ShardCacheConfig,
    model_id: String,
    layers: LayerRange,
    checksum: Option<String>,
    version: Option<String>,
    discovery_timeout: Duration,
    mut on_progress: impl FnMut(u64, u64) + Send,
) -> Result<ModelFetchResult> {
    let cache = ShardCache::new(cache_config)?;
    if let Some(record) = cache.find(&model_id, layers, checksum.as_deref(), version.as_deref())? {
        on_progress(record.info.size_bytes, record.info.size_bytes);
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

    let mut swarm = build_grid_swarm(
        config.keypair.clone(),
        &topic,
        config.relay_server,
        config.enable_mdns,
    )?;
    start_grid_listeners(&mut swarm, &config)?;
    // A download request must own the target dial. Proactively dialing every
    // discovered seed here races request-response and produces an immediate
    // `DialFailure` while the correct connection is still in progress.
    for advertisement in &config.static_peers {
        add_advertisement_addresses(&mut swarm, advertisement);
    }

    let deadline = Instant::now() + discovery_timeout;
    let mut publish_interval = interval(config.publish_interval);
    let mut relay_ready = config.relay_peers.is_empty();
    let partial_dir = cache.config().root.join("tmp");
    fs::create_dir_all(&partial_dir)
        .with_context(|| format!("failed to create {}", partial_dir.display()))?;
    let partial_identity = sanitize_path_segment(checksum.as_deref().unwrap_or("unresolved"));
    let partial_path = partial_dir.join(format!(
        "{}-{}-{}-{}.gguf.partial",
        sanitize_path_segment(&model_id),
        layers.start,
        layers.end,
        &partial_identity[..partial_identity.len().min(16)]
    ));
    // Keep a content-addressed partial file across restarts. The completed
    // package is still SHA-256 verified before entering the cache, so resume
    // never weakens package integrity.
    let mut downloaded_bytes = fs::metadata(&partial_path)
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    let mut pending_request: Option<(
        request_response::OutboundRequestId,
        ModelShardCandidate,
        u64,
    )> = None;
    let mut failed_peers = HashMap::<String, Instant>::new();
    let mut progress_started = false;

    loop {
        if pending_request.is_none() && relay_ready {
            if let Some(candidate) = select_model_shard_candidate(
                &registry,
                &local_peer_id,
                &model_id,
                layers,
                checksum.as_deref(),
                version.as_deref(),
                &failed_peers,
            ) {
                if candidate.shard.size_bytes == 0 {
                    bail!(
                        "refusing zero-byte model package {} {}:{}",
                        model_id,
                        layers.start,
                        layers.end
                    );
                }
                if candidate.shard.size_bytes > cache.config().max_storage_bytes {
                    bail!(
                        "refusing model package {} {}:{}: advertised size {} exceeds cache limit {}",
                        model_id,
                        layers.start,
                        layers.end,
                        candidate.shard.size_bytes,
                        cache.config().max_storage_bytes
                    );
                }
                if downloaded_bytes > candidate.shard.size_bytes {
                    fs::remove_file(&partial_path).with_context(|| {
                        format!(
                            "failed to reset oversized partial {}",
                            partial_path.display()
                        )
                    })?;
                    downloaded_bytes = 0;
                }
                if !progress_started {
                    on_progress(downloaded_bytes, candidate.shard.size_bytes);
                    progress_started = true;
                }
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
                publish_local_advertisement(
                    &mut swarm,
                    &topic,
                    &mut config.advertisement,
                    &local_peer_id,
                )?;
            }
            event = swarm.select_next_some() => {
                if matches!(
                    &event,
                    SwarmEvent::Behaviour(GridBehaviourEvent::RelayClient(
                        relay::client::Event::ReservationReqAccepted { .. }
                    ))
                ) {
                    relay_ready = true;
                }
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
                                    record_model_fetch_peer_failure(
                                        &mut failed_peers,
                                        candidate.advertisement.peer_id.clone(),
                                    );
                                    eprintln!(
                                        "model shard chunk request failed: {error}; retrying peer after cooldown"
                                    );
                                    continue;
                                }
                                if response.model_id != model_id
                                    || response.source_checksum != candidate.shard.checksum
                                    || response.layers != Some(layers)
                                {
                                    record_model_fetch_peer_failure(
                                        &mut failed_peers,
                                        candidate.advertisement.peer_id.clone(),
                                    );
                                    eprintln!("model shard chunk response identity mismatch");
                                    continue;
                                }
                                if response.offset != expected_offset || response.offset != downloaded_bytes {
                                    record_model_fetch_peer_failure(
                                        &mut failed_peers,
                                        candidate.advertisement.peer_id.clone(),
                                    );
                                    eprintln!("model shard chunk response offset mismatch: got {}, expected {}", response.offset, downloaded_bytes);
                                    continue;
                                }
                                if response.total_size_bytes != candidate.shard.size_bytes {
                                    record_model_fetch_peer_failure(
                                        &mut failed_peers,
                                        candidate.advertisement.peer_id.clone(),
                                    );
                                    eprintln!(
                                        "model shard chunk size mismatch: got {}, expected {}",
                                        response.total_size_bytes, candidate.shard.size_bytes
                                    );
                                    continue;
                                }
                                if response.payload.is_empty() && downloaded_bytes < candidate.shard.size_bytes {
                                    record_model_fetch_peer_failure(
                                        &mut failed_peers,
                                        candidate.advertisement.peer_id.clone(),
                                    );
                                    eprintln!("model shard chunk response returned empty payload before EOF");
                                    continue;
                                }

                                let next_downloaded = downloaded_bytes
                                    .checked_add(response.payload.len() as u64)
                                    .ok_or_else(|| anyhow!("model package download size overflow"))?;
                                if next_downloaded > candidate.shard.size_bytes {
                                    bail!(
                                        "model package download exceeded advertised size for {} {}:{}; got {}, expected {}",
                                        model_id,
                                        layers.start,
                                        layers.end,
                                        next_downloaded,
                                        candidate.shard.size_bytes
                                    );
                                }
                                append_source_chunk(&partial_path, &response.payload)?;
                                downloaded_bytes = next_downloaded;
                                on_progress(downloaded_bytes, candidate.shard.size_bytes);

                                if downloaded_bytes >= candidate.shard.size_bytes {
                                    let cache_record = match cache.store_downloaded_file(
                                        &candidate.shard,
                                        &partial_path,
                                        candidate.seed_manifest.clone(),
                                    ) {
                                        Ok(record) => record,
                                        Err(error) => {
                                            // A complete file that fails its final digest must not
                                            // poison every future resume attempt.
                                            let _ = fs::remove_file(&partial_path);
                                            return Err(error);
                                        }
                                    };
                                    refresh_advertisement_model_shards(&mut config.advertisement, Some(&cache))?;
                                    publish_local_advertisement(
                                        &mut swarm,
                                        &topic,
                                        &mut config.advertisement,
                                        &local_peer_id,
                                    )?;

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
                                    record_model_fetch_peer_failure(&mut failed_peers, peer.to_string());
                                    eprintln!(
                                        "model shard chunk request to {peer} failed: {error}; retrying peer after cooldown"
                                    );
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
    if expected_size_bytes > cache.config().max_storage_bytes {
        bail!(
            "refusing GGUF source for {}: expected size {} exceeds cache limit {}",
            model_id,
            expected_size_bytes,
            cache.config().max_storage_bytes
        );
    }
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

    let mut swarm = build_grid_swarm(
        config.keypair.clone(),
        &topic,
        config.relay_server,
        config.enable_mdns,
    )?;
    start_grid_listeners(&mut swarm, &config)?;
    for advertisement in &config.static_peers {
        add_advertisement_addresses(&mut swarm, advertisement);
    }

    let deadline = Instant::now() + discovery_timeout;
    let mut publish_interval = interval(config.publish_interval);
    let mut relay_ready = config.relay_peers.is_empty();
    let partial_path = final_path.with_extension("gguf.partial");
    if let Some(parent) = final_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let _ = fs::remove_file(&partial_path);
    let _partial_cleanup = RemoveFileOnDrop(partial_path.clone());
    let mut downloaded_bytes = 0_u64;
    let mut total_size_bytes = expected_size_bytes;
    let mut pending_request: Option<(request_response::OutboundRequestId, NodeAdvertisement, u64)> =
        None;
    let mut failed_peers = HashMap::<String, Instant>::new();
    on_progress(downloaded_bytes, total_size_bytes);

    loop {
        if pending_request.is_none() && relay_ready {
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
                publish_local_advertisement(
                    &mut swarm,
                    &topic,
                    &mut config.advertisement,
                    &local_peer_id,
                )?;
            }
            event = swarm.select_next_some() => {
                if matches!(
                    &event,
                    SwarmEvent::Behaviour(GridBehaviourEvent::RelayClient(
                        relay::client::Event::ReservationReqAccepted { .. }
                    ))
                ) {
                    relay_ready = true;
                }
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
                                record_model_fetch_peer_failure(
                                    &mut failed_peers,
                                    advertisement.peer_id.clone(),
                                );
                                eprintln!(
                                    "model blob request failed: {error}; retrying peer after cooldown"
                                );
                                continue;
                            }
                            if response.model_id != model_id || response.source_checksum != source_checksum {
                                record_model_fetch_peer_failure(
                                    &mut failed_peers,
                                    advertisement.peer_id.clone(),
                                );
                                eprintln!("model blob response identity mismatch");
                                continue;
                            }
                            if response.offset != expected_offset || response.offset != downloaded_bytes {
                                record_model_fetch_peer_failure(
                                    &mut failed_peers,
                                    advertisement.peer_id.clone(),
                                );
                                eprintln!("model blob response offset mismatch: got {}, expected {}", response.offset, downloaded_bytes);
                                continue;
                            }
                            if expected_size_bytes > 0 && response.total_size_bytes != expected_size_bytes {
                                record_model_fetch_peer_failure(
                                    &mut failed_peers,
                                    advertisement.peer_id.clone(),
                                );
                                eprintln!(
                                    "model blob size mismatch: got {}, expected {}",
                                    response.total_size_bytes, expected_size_bytes
                                );
                                continue;
                            }
                            total_size_bytes = response.total_size_bytes;
                            if total_size_bytes > cache.config().max_storage_bytes {
                                bail!(
                                    "refusing GGUF source for {}: advertised size {} exceeds cache limit {}",
                                    model_id,
                                    total_size_bytes,
                                    cache.config().max_storage_bytes
                                );
                            }
                            if response.payload.is_empty() && downloaded_bytes < total_size_bytes {
                                record_model_fetch_peer_failure(
                                    &mut failed_peers,
                                    advertisement.peer_id.clone(),
                                );
                                eprintln!("model blob response returned an empty chunk before EOF");
                                continue;
                            }

                            let next_downloaded = downloaded_bytes
                                .checked_add(response.payload.len() as u64)
                                .ok_or_else(|| anyhow!("GGUF source download size overflow"))?;
                            if next_downloaded > total_size_bytes {
                                bail!(
                                    "GGUF source download exceeded advertised size for {}; got {}, expected {}",
                                    model_id,
                                    next_downloaded,
                                    total_size_bytes
                                );
                            }
                            append_source_chunk(&partial_path, &response.payload)?;
                            downloaded_bytes = next_downloaded;
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
                                publish_local_advertisement(
                                    &mut swarm,
                                    &topic,
                                    &mut config.advertisement,
                                    &local_peer_id,
                                )?;
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
                                    record_model_fetch_peer_failure(&mut failed_peers, peer.to_string());
                                    eprintln!(
                                        "model blob request to {peer} failed: {error}; retrying peer after cooldown"
                                    );
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
    if !config.rpc_endpoints.is_empty() {
        bail!("raw RPC endpoints cannot cross the network; select authenticated worker peers");
    }
    if !config.rpc_worker_peer_ids.is_empty() {
        validate_rpc_model_manifest(&manifest)?;
    }

    let topic = gossipsub::IdentTopic::new(config.topic.clone());
    let mut registry = ShardRegistry::new();
    registry.extend(config.static_peers.clone());

    let mut swarm = build_grid_swarm(
        config.keypair.clone(),
        &topic,
        config.relay_server,
        config.enable_mdns,
    )?;
    start_grid_listeners(&mut swarm, &config)?;
    add_static_peer_addresses(&mut swarm, &config.static_peers, &config.relay_peers);

    let route = if let Some(route) = config.planned_route.clone() {
        validate_planned_route(&route, &manifest, &config.static_peers)?;
        route
    } else {
        discover_route_on_swarm(
            &mut swarm,
            &mut registry,
            &mut config,
            &topic,
            &manifest,
            discovery_timeout,
        )
        .await?
    };

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
        Some(PromptMetadata {
            prompt,
            demo_mode,
            rpc_endpoints: Vec::new(),
            rpc_worker_peer_ids: config.rpc_worker_peer_ids.clone(),
        }),
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
            RuntimeKind::LlamaCpp => Duration::from_secs(18 * 60),
        },
    )
    .await?;

    if let Some(error) = &response.error {
        bail!("remote activation error: {error}");
    }

    Ok(InferenceResult { route, response })
}

fn normalize_trusted_rpc_endpoints(endpoints: &[String]) -> Result<Vec<String>> {
    let mut normalized = Vec::new();
    let mut seen = HashSet::new();

    for endpoint in endpoints {
        let endpoint = endpoint.trim();
        let (host, port) = endpoint.split_once(':').ok_or_else(|| {
            anyhow!("llama RPC endpoint must use the loopback IPv4 host:port form")
        })?;
        if host.is_empty() || port.is_empty() || port.contains(':') {
            bail!("llama RPC endpoint must use the loopback IPv4 host:port form");
        }
        let host = host.parse::<Ipv4Addr>().with_context(|| {
            format!("llama RPC endpoint {endpoint:?} must use a loopback IPv4 address")
        })?;
        if !is_trusted_rpc_ipv4(host) {
            bail!("llama RPC endpoint {endpoint:?} is not process-local loopback");
        }
        if !port.bytes().all(|byte| byte.is_ascii_digit()) {
            bail!("llama RPC endpoint {endpoint:?} has an invalid port");
        }
        let port = port
            .parse::<u16>()
            .ok()
            .filter(|port| *port > 0)
            .ok_or_else(|| anyhow!("llama RPC endpoint {endpoint:?} has an invalid port"))?;
        let endpoint = format!("{host}:{port}");
        if seen.insert(endpoint.clone()) {
            normalized.push(endpoint);
            if normalized.len() > MAX_TRUSTED_RPC_ENDPOINTS {
                bail!(
                    "llama RPC request exceeds the {MAX_TRUSTED_RPC_ENDPOINTS}-backend safety limit"
                );
            }
        }
    }

    Ok(normalized)
}

fn validate_planned_route(
    route: &[RouteHop],
    manifest: &ModelManifest,
    advertisements: &[NodeAdvertisement],
) -> Result<()> {
    if route.is_empty() {
        bail!("planned inference route must not be empty");
    }
    let mut expected_start = 0_u32;
    for hop in route {
        if hop.layers.start != expected_start || hop.layers.end > manifest.layer_count {
            bail!("planned inference route is not contiguous for the selected model");
        }
        let advertised = advertisements.iter().any(|advertisement| {
            advertisement.peer_id == hop.peer_id
                && advertisement
                    .hosted_shards
                    .iter()
                    .any(|shard| shard.model_id == manifest.model_id && shard.layers == hop.layers)
        });
        if !advertised {
            bail!(
                "planned coordinator {} no longer advertises the selected verified model",
                hop.peer_id
            );
        }
        expected_start = hop.layers.end;
    }
    if expected_start != manifest.layer_count {
        bail!("planned inference route does not cover the full selected model");
    }
    Ok(())
}

fn is_trusted_rpc_ipv4(address: Ipv4Addr) -> bool {
    address.is_loopback()
}

fn validate_rpc_model_manifest(manifest: &ModelManifest) -> Result<()> {
    let official_model = ModelManifest::infernet_chat_v1();
    let release = OfficialModelRelease::infernet_chat_v1_compatibility();
    release
        .validate_for_model(&official_model)
        .context("pinned Infernet Chat release metadata is invalid")?;
    if manifest != &official_model || manifest.model_id != release.model_id {
        bail!(
            "distributed llama RPC execution is restricted to the pinned {} release",
            release.release_id
        );
    }
    Ok(())
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
    enrich_local_advertisement(NodeAdvertisement {
        protocol_version: PROTOCOL_VERSION,
        peer_id,
        addresses,
        available_ram_bytes: None,
        available_vram_bytes: None,
        latency_hint_ms: None,
        capabilities: None,
        hosted_shards: vec![ShardDescriptor::demo(model_id, layers)],
        model_shards: Vec::new(),
    })
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
    enrich_local_advertisement(NodeAdvertisement {
        protocol_version: PROTOCOL_VERSION,
        peer_id,
        addresses,
        available_ram_bytes: None,
        available_vram_bytes: None,
        latency_hint_ms: None,
        capabilities: None,
        hosted_shards: vec![ShardDescriptor::for_manifest(manifest, layers)],
        model_shards: Vec::new(),
    })
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
        capabilities: None,
        hosted_shards: Vec::new(),
        model_shards: Vec::new(),
    }
}

pub fn local_capability_advertisement(peer_id: String, address: String) -> NodeAdvertisement {
    enrich_local_advertisement(empty_advertisement(peer_id, address))
}

pub fn enrich_local_advertisement(mut advertisement: NodeAdvertisement) -> NodeAdvertisement {
    let local_peer_id = advertisement.peer_id.clone();
    refresh_local_advertisement_capabilities(&mut advertisement, &local_peer_id);
    advertisement
}

pub fn refresh_local_advertisement_capabilities(
    advertisement: &mut NodeAdvertisement,
    local_peer_id: &str,
) -> bool {
    if advertisement.peer_id != local_peer_id {
        return false;
    }

    let capabilities = detect_node_capabilities();
    advertisement.available_ram_bytes =
        (capabilities.available_ram_bytes > 0).then_some(capabilities.available_ram_bytes);
    advertisement.available_vram_bytes = (capabilities.available_accelerator_memory_bytes > 0)
        .then_some(capabilities.available_accelerator_memory_bytes);
    advertisement.capabilities = Some(capabilities);
    true
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
                    let seed_manifest = Box::new(seed_manifest_for_network(&manifest));
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
    match manifest.runtime_kind {
        RuntimeKind::Demo => matches!(
            manifest.payload_kind.as_str(),
            model_distribution::PAYLOAD_KIND_GGUF_SHARD
                | model_distribution::PAYLOAD_KIND_INFERNET_SHARD
        ),
        RuntimeKind::LlamaCpp => {
            manifest.payload_kind == model_distribution::PAYLOAD_KIND_FULL_MODEL
                && manifest.layers.start == 0
                && manifest.layers.end == manifest.layer_count
        }
    }
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
    let rpc_endpoints =
        validated_rpc_endpoints_for_execution(config, layers, request, &manifest, &model_path)?;
    if !rpc_endpoints.is_empty() {
        let binary = find_llama_server_binary().ok_or_else(|| {
            anyhow!(
                "llama-server binary is missing; run npm run prepare-runtime or set INFERNET_LLAMA_SERVER"
            )
        })?;
        let release = OfficialModelRelease::infernet_chat_v1_compatibility();
        let completion = complete_with_persistent_llama_server(
            LlamaServerConfig {
                binary,
                model_path,
                rpc_endpoints,
                context_size: release.launch_context_cap_tokens,
                threads: llama_bridge_thread_cap(),
                cache_ram_mib: 0,
                startup_timeout: Duration::from_secs(8 * 60),
                request_timeout: Duration::from_secs(7 * 60),
                log_dir: cache_config.root.join("runtime"),
            },
            &prompt.prompt,
            64,
        )?;
        return Ok(LlamaShardOutput {
            activation: Vec::new(),
            output_text: Some(completion.text),
            timing_ms: completion.timing_ms,
        });
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
    constrain_llama_bridge_library_threads(&mut command);
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
        .arg("--threads")
        .arg(llama_bridge_thread_cap().to_string())
        .arg("--prompt")
        .arg(&prompt.prompt);
    if layers.start == 0 && layers.end == manifest.layer_count {
        // Compatibility packages must use llama.cpp's unmodified full graph.
        // Enabling Infernet's partial-graph patch here breaks architectures for
        // which no split executor exists yet (including Qwen 3.5).
        command.arg("--full-model");
    }
    append_llama_rpc_arguments(&mut command, &rpc_endpoints);
    if !request.activation.is_empty() {
        command.arg("--input").arg(&input_path);
    }
    if request.next_hop().is_some() {
        command.arg("--output").arg(&output_path);
    }

    let output = run_llama_bridge_with_timeout(&mut command, LLAMA_BRIDGE_EXECUTION_TIMEOUT)
        .with_context(|| {
            format!(
                "failed to run {} for {} {}:{}",
                bridge.display(),
                config.model_id,
                layers.start,
                layers.end
            )
        })?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    let bridge_output = parse_llama_bridge_json(&output)?;

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

fn validated_rpc_endpoints_for_execution(
    config: &WorkerConfig,
    layers: LayerRange,
    request: &ActivationRequest,
    manifest: &SeedShardManifest,
    model_path: &Path,
) -> Result<Vec<String>> {
    let endpoints = normalize_trusted_rpc_endpoints(
        &request
            .prompt
            .as_ref()
            .map(|prompt| prompt.rpc_endpoints.as_slice())
            .unwrap_or_default(),
    )?;
    if endpoints.is_empty() {
        return Ok(endpoints);
    }

    let package_path = model_path
        .parent()
        .ok_or_else(|| anyhow!("full model RPC payload has no package directory"))?
        .join(INFERNET_SHARD_MANIFEST_FILE);
    let package_bytes = fs::read(&package_path).with_context(|| {
        format!(
            "distributed RPC requires a complete Infernet package manifest at {}",
            package_path.display()
        )
    })?;
    let package = serde_json::from_slice::<InfernetShardPackageManifest>(&package_bytes)
        .with_context(|| format!("failed to parse {}", package_path.display()))?;
    let model_size_bytes = fs::metadata(model_path)
        .with_context(|| format!("failed to inspect {}", model_path.display()))?
        .len();
    validate_official_rpc_package(
        config,
        layers,
        request,
        manifest,
        &package,
        model_size_bytes,
    )?;

    Ok(endpoints)
}

fn validate_official_rpc_package(
    config: &WorkerConfig,
    layers: LayerRange,
    request: &ActivationRequest,
    manifest: &SeedShardManifest,
    package: &InfernetShardPackageManifest,
    model_size_bytes: u64,
) -> Result<()> {
    let official_model = ModelManifest::infernet_chat_v1();
    let release = OfficialModelRelease::infernet_chat_v1_compatibility();
    release
        .validate_for_model(&official_model)
        .context("pinned Infernet Chat release metadata is invalid")?;
    let component = release
        .components
        .iter()
        .find(|component| component.layers == Some(layers))
        .ok_or_else(|| anyhow!("RPC layer range is not part of the pinned official release"))?;

    let single_full_model_hop = request.route.len() == 1
        && request.current_hop_index == 0
        && request.next_hop().is_none()
        && request.activation.is_empty();
    let exact_model = config.runtime_kind == RuntimeKind::LlamaCpp
        && config.model_id == official_model.model_id
        && config.hidden_size == official_model.hidden_size
        && layers.start == 0
        && layers.end == official_model.layer_count
        && manifest.model_id == official_model.model_id
        && manifest.display_name == official_model.display_name
        && manifest.architecture == official_model.architecture
        && manifest.layer_count == official_model.layer_count
        && manifest.hidden_size == official_model.hidden_size
        && manifest.activation_dtype == official_model.activation_dtype
        && manifest.runtime_kind == official_model.runtime_kind
        && manifest.layers == layers
        && manifest.metadata.architecture == official_model.architecture
        && manifest.metadata.quantization.as_deref() == official_model.quantization.as_deref()
        && manifest.metadata.protocol_version == PROTOCOL_VERSION
        && manifest.metadata.source_checksum.as_deref()
            == Some(release.upstream.source_sha256.as_str())
        && manifest.source.checksum_sha256 == release.upstream.source_sha256
        && manifest.source.file_size_bytes == release.expected_total_bytes
        && manifest.payload_kind == PAYLOAD_KIND_FULL_MODEL;
    let exact_package = package.format_version == INFERNET_SHARD_FORMAT_VERSION
        && package.runtime_abi == INFERNET_FULL_MODEL_RUNTIME_ABI
        && package.component == "full_model"
        && package.seed_manifest == *manifest
        && package.payload.kind == "gguf_tensor_payload"
        && package.payload.file == INFERNET_SHARD_TENSOR_FILE
        && package.payload.checksum_sha256 == component.sha256
        && package.payload.size_bytes == component.size_bytes
        && model_size_bytes == component.size_bytes;

    if !single_full_model_hop || !exact_model || !exact_package {
        bail!(
            "distributed llama RPC requires the exact pinned {} full-model package on a single coordinator hop",
            release.release_id
        );
    }

    Ok(())
}

fn append_llama_rpc_arguments(command: &mut Command, endpoints: &[String]) {
    if !endpoints.is_empty() {
        command.arg("--rpc").arg(endpoints.join(","));
    }
}

fn constrain_llama_bridge_library_threads(command: &mut Command) {
    let thread_cap = llama_bridge_thread_cap();

    // Keep BLAS/OpenMP helpers under the same limit as llama.cpp's native
    // thread pool, without increasing a lower limit explicitly set by a user.
    for variable in [
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ] {
        let configured = env::var(variable)
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .map(|value| value.min(thread_cap))
            .unwrap_or(thread_cap);
        command.env(variable, configured.to_string());
    }
}

fn llama_bridge_thread_cap() -> usize {
    let available_threads = thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    available_threads
        .min(LLAMA_BRIDGE_MAX_LIBRARY_THREADS)
        .max(1)
}

fn parse_llama_bridge_json(output: &Output) -> Result<LlamaBridgeJson> {
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
    serde_json::from_str(json_line)
        .with_context(|| format!("failed to parse infernet-llama-bridge JSON: {json_line}"))
}

fn run_llama_bridge_with_timeout(command: &mut Command, timeout: Duration) -> Result<Output> {
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = command
        .spawn()
        .context("failed to spawn infernet-llama-bridge")?;
    let child_id = child.id();
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("failed to capture infernet-llama-bridge stdout"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow!("failed to capture infernet-llama-bridge stderr"))?;
    let stdout_reader = thread::spawn(move || read_child_output(stdout));
    let stderr_reader = thread::spawn(move || read_child_output(stderr));
    let started = std::time::Instant::now();

    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let stdout = join_child_output(stdout_reader, "stdout")?;
                let stderr = join_child_output(stderr_reader, "stderr")?;
                return Ok(Output {
                    status,
                    stdout,
                    stderr,
                });
            }
            Ok(None) if started.elapsed() >= timeout => {
                let kill_error = child.kill().err();
                let wait_error = child.wait().err();
                let _stdout = join_child_output(stdout_reader, "stdout")?;
                let stderr = join_child_output(stderr_reader, "stderr")?;
                let stderr = String::from_utf8_lossy(&stderr);
                let mut details = Vec::new();
                if let Some(error) = kill_error {
                    details.push(format!("kill failed: {error}"));
                }
                if let Some(error) = wait_error {
                    details.push(format!("reap failed: {error}"));
                }
                if !stderr.trim().is_empty() {
                    details.push(format!("stderr={}", stderr.trim()));
                }
                let details = if details.is_empty() {
                    String::new()
                } else {
                    format!("; {}", details.join("; "))
                };
                bail!(
                    "infernet-llama-bridge process {child_id} exceeded the {:.1}s execution deadline and was terminated{details}",
                    timeout.as_secs_f64()
                );
            }
            Ok(None) => {
                thread::sleep(
                    LLAMA_BRIDGE_POLL_INTERVAL.min(timeout.saturating_sub(started.elapsed())),
                );
            }
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = join_child_output(stdout_reader, "stdout");
                let _ = join_child_output(stderr_reader, "stderr");
                return Err(error).context("failed to poll infernet-llama-bridge");
            }
        }
    }
}

fn read_child_output(mut stream: impl Read) -> io::Result<Vec<u8>> {
    let mut output = Vec::new();
    stream.read_to_end(&mut output)?;
    Ok(output)
}

fn join_child_output(
    reader: thread::JoinHandle<io::Result<Vec<u8>>>,
    stream_name: &str,
) -> Result<Vec<u8>> {
    reader
        .join()
        .map_err(|_| anyhow!("infernet-llama-bridge {stream_name} reader panicked"))?
        .with_context(|| format!("failed to read infernet-llama-bridge {stream_name}"))
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
    completion_sender: &mpsc::UnboundedSender<CompletedLocalActivation>,
    pending_local_activations: &mut HashMap<uuid::Uuid, PendingLocalActivation>,
    rpc_tunnel_control: libp2p_stream::Control,
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
                completion_sender,
                pending_local_activations,
                rpc_tunnel_control,
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
    request: ActivationRequest,
    channel: request_response::ResponseChannel<ActivationResponse>,
    pending_forwards: &mut HashMap<request_response::OutboundRequestId, PendingOutbound>,
    completion_sender: &mpsc::UnboundedSender<CompletedLocalActivation>,
    pending_local_activations: &mut HashMap<uuid::Uuid, PendingLocalActivation>,
    rpc_tunnel_control: libp2p_stream::Control,
) -> Result<()> {
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

    if !pending_local_activations.is_empty() {
        send_response(
            swarm,
            channel,
            ActivationResponse::failure(
                trace_id,
                peer_id.to_owned(),
                "the model coordinator is already processing a request".to_owned(),
                request.trace,
            ),
        );
        return Ok(());
    }

    if request
        .prompt
        .as_ref()
        .is_some_and(|prompt| !prompt.rpc_endpoints.is_empty())
    {
        send_response(
            swarm,
            channel,
            ActivationResponse::failure(
                trace_id,
                peer_id.to_owned(),
                "raw RPC endpoints are forbidden; use authenticated Infernet worker identities"
                    .to_owned(),
                request.trace,
            ),
        );
        return Ok(());
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

    let job_id = uuid::Uuid::new_v4();
    pending_local_activations.insert(
        job_id,
        PendingLocalActivation {
            channel,
            peer_id: peer_id.to_owned(),
        },
    );
    let completion_sender = completion_sender.clone();
    let peer_id = peer_id.to_owned();
    let worker_peer_ids = request
        .prompt
        .as_ref()
        .map(|prompt| prompt.rpc_worker_peer_ids.clone())
        .unwrap_or_default();
    set_local_inference_active(true);
    tokio::spawn(async move {
        let trace_id = request.trace_id;
        let failure_trace = request.trace.clone();
        let outcome = match ensure_persistent_rpc_tunnels(rpc_tunnel_control, &worker_peer_ids)
            .await
        {
            Ok(endpoints) => {
                let mut request = request;
                if let Some(prompt) = request.prompt.as_mut() {
                    prompt.rpc_endpoints = endpoints;
                    prompt.rpc_worker_peer_ids.clear();
                }
                let processing_peer_id = peer_id.clone();
                match tokio::task::spawn_blocking(move || {
                    process_local_activation_steps(&worker, &processing_peer_id, request)
                })
                .await
                {
                    Ok(outcome) => outcome,
                    Err(error) => LocalActivationOutcome::Response(ActivationResponse::failure(
                        trace_id,
                        peer_id,
                        format!("local inference task failed: {error}"),
                        failure_trace,
                    )),
                }
            }
            Err(error) => LocalActivationOutcome::Response(ActivationResponse::failure(
                trace_id,
                peer_id,
                format!("failed to prepare distributed RPC tunnels: {error:#}"),
                failure_trace,
            )),
        };
        let _ = completion_sender.unbounded_send(CompletedLocalActivation { job_id, outcome });
    });

    Ok(())
}

fn process_local_activation_steps(
    worker: &WorkerConfig,
    peer_id: &str,
    mut request: ActivationRequest,
) -> LocalActivationOutcome {
    loop {
        match process_activation_step(worker, request) {
            Ok(ActivationStep::Final(response)) | Err(response) => {
                return LocalActivationOutcome::Response(response);
            }
            Ok(ActivationStep::Forward(next_request)) => {
                if next_request
                    .current_hop()
                    .is_some_and(|hop| hop.peer_id == peer_id)
                {
                    request = next_request;
                } else {
                    return LocalActivationOutcome::Forward(next_request);
                }
            }
        }
    }
}

fn handle_completed_local_activation(
    swarm: &mut Swarm<GridBehaviour>,
    activation_relays: &[NodeAdvertisement],
    completed: CompletedLocalActivation,
    pending_local_activations: &mut HashMap<uuid::Uuid, PendingLocalActivation>,
    pending_forwards: &mut HashMap<request_response::OutboundRequestId, PendingOutbound>,
) -> Result<()> {
    let Some(pending) = pending_local_activations.remove(&completed.job_id) else {
        return Ok(());
    };
    set_local_inference_active(false);
    match completed.outcome {
        LocalActivationOutcome::Response(response) => {
            send_response(swarm, pending.channel, response);
            Ok(())
        }
        LocalActivationOutcome::Forward(request) => forward_activation_request(
            swarm,
            &pending.peer_id,
            activation_relays,
            request,
            pending.channel,
            pending_forwards,
        ),
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
            "executable Infernet shards must be fetched with the chunked model blob protocol",
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
    _activation_relays: &[NodeAdvertisement],
    hop: &RouteHop,
    request: ActivationRequest,
) -> Result<request_response::OutboundRequestId> {
    // Circuit Relay v2 and DCUtR are transport-level connections to the exact
    // target PeerId. Never substitute an arbitrary bootstrap peer as an
    // application-level activation hop.
    send_activation_request(swarm, hop, request)
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
    let served_bytes = response
        .error
        .is_none()
        .then_some(response.payload.len() as u64)
        .unwrap_or(0);
    match swarm.behaviour_mut().blob.send_response(channel, response) {
        Ok(()) => MODEL_SERVING_TELEMETRY.record_chunk(served_bytes, model_serving_now_unix_ms()),
        Err(response) => {
            eprintln!(
                "failed to send model blob response request_id={} offset={} error={:?}",
                response.request_id, response.offset, response.error
            );
        }
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
                if let Some(advertisement) = advertisement.as_ref() {
                    println!(
                        "libp2p_listen={}",
                        advertisement
                            .addresses
                            .last()
                            .map(String::as_str)
                            .unwrap_or("<no-address>")
                    );
                }
                publish_local_advertisement(swarm, topic, advertisement, &peer_id.to_string())?;
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
        SwarmEvent::Behaviour(GridBehaviourEvent::RelayClient(
            relay::client::Event::ReservationReqAccepted {
                relay_peer_id,
                renewal,
                ..
            },
        )) => {
            println!(
                "libp2p_relay_reserved peer_id={} renewal={}",
                relay_peer_id, renewal
            );
        }
        SwarmEvent::Behaviour(GridBehaviourEvent::RelayServer(
            relay::Event::ReservationReqAccepted {
                src_peer_id,
                renewed,
            },
        )) => {
            println!(
                "libp2p_relay_client_reserved peer_id={} renewal={}",
                src_peer_id, renewed
            );
        }
        SwarmEvent::Behaviour(GridBehaviourEvent::Dcutr(event)) => match event.result {
            Ok(_) => println!("libp2p_direct_upgrade peer_id={}", event.remote_peer_id),
            Err(error) => eprintln!(
                "libp2p_direct_upgrade_failed peer_id={} error={error}",
                event.remote_peer_id
            ),
        },
        SwarmEvent::Behaviour(GridBehaviourEvent::Identify(identify::Event::Received {
            peer_id,
            info,
            ..
        })) => {
            let mut observed = empty_advertisement(peer_id.to_string(), String::new());
            for address in info.listen_addrs.into_iter().take(32) {
                let address = match address.iter().last() {
                    Some(Protocol::P2p(address_peer_id)) if address_peer_id != peer_id => continue,
                    Some(Protocol::P2p(_)) => address,
                    _ => address.with(Protocol::P2p(peer_id)),
                };
                let address_text = address.to_string();
                if !observed.addresses.contains(&address_text) {
                    observed.addresses.push(address_text);
                }
                swarm.add_peer_address(peer_id, address);
            }
            registry.merge(observed);
        }
        SwarmEvent::Behaviour(GridBehaviourEvent::Identify(identify::Event::Error {
            peer_id,
            error,
            ..
        })) => {
            eprintln!("libp2p_identify_failed peer_id={peer_id} error={error}");
        }
        SwarmEvent::Behaviour(GridBehaviourEvent::Ping(ping::Event {
            peer,
            result: Err(error),
            ..
        })) => {
            eprintln!("libp2p_ping_failed peer_id={peer} error={error}");
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
    relay_server_enabled: bool,
    enable_mdns: bool,
) -> Result<Swarm<GridBehaviour>> {
    let peer_id = keypair.public().to_peer_id();
    let mut swarm = SwarmBuilder::with_existing_identity(keypair)
        .with_tokio()
        .with_tcp(
            tcp::Config::default(),
            noise::Config::new,
            yamux::Config::default,
        )?
        .with_quic()
        .with_dns()?
        .with_relay_client(noise::Config::new, yamux::Config::default)?
        .with_behaviour(|key, relay_client| {
            let relay_server = relay_server_enabled
                .then(|| relay::Behaviour::new(peer_id, public_relay_server_config()))
                .into();
            let dcutr = dcutr::Behaviour::new(peer_id);
            let identify = identify::Behaviour::new(
                identify::Config::new_with_signed_peer_record(IDENTIFY_PROTOCOL.to_owned(), key)
                    .with_agent_version(IDENTIFY_AGENT.to_owned())
                    .with_push_listen_addr_updates(true),
            );
            let ping = ping::Behaviour::new(
                ping::Config::new()
                    .with_interval(Duration::from_secs(15))
                    .with_timeout(Duration::from_secs(20)),
            );
            let gossipsub = gossipsub::Behaviour::new(
                gossipsub::MessageAuthenticity::Signed(key.clone()),
                gossipsub::Config::default(),
            )?;
            let mdns = enable_mdns
                .then(|| mdns::tokio::Behaviour::new(mdns::Config::default(), peer_id))
                .transpose()?
                .into();
            let activation = request_response::json::Behaviour::new(
                [(
                    StreamProtocol::new(ACTIVATION_PROTOCOL),
                    request_response::ProtocolSupport::Full,
                )],
                // A real GGUF request includes model load and prompt evaluation. The
                // previous five-second transport deadline guaranteed failure while
                // the worker kept computing after the client had already given up.
                request_response::Config::default()
                    .with_request_timeout(Duration::from_secs(20 * 60)),
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
                relay_client,
                relay_server,
                dcutr,
                identify,
                ping,
                gossipsub,
                mdns,
                activation,
                model,
                blob,
                rpc_tunnel: libp2p_stream::Behaviour::new(),
            })
        })?
        .build();

    swarm.behaviour_mut().gossipsub.subscribe(topic)?;

    Ok(swarm)
}

fn public_relay_server_config() -> relay::Config {
    relay::Config {
        max_reservations: 4_096,
        max_reservations_per_peer: 4,
        reservation_duration: PUBLIC_RELAY_RESERVATION_DURATION,
        max_circuits: 2_048,
        max_circuits_per_peer: 16,
        max_circuit_duration: PUBLIC_RELAY_CIRCUIT_DURATION,
        // A model is transferred as authenticated libp2p streams. The relay
        // default (128 KiB) is intentionally tiny and would sever the first
        // model chunk, so public Infernet relays allow a full verified package.
        max_circuit_bytes: PUBLIC_RELAY_MAX_CIRCUIT_BYTES,
        ..relay::Config::default()
    }
}

fn listen_on(swarm: &mut Swarm<GridBehaviour>, listen: &str) -> Result<ListenerId> {
    let p2p_listen = listen
        .parse::<Multiaddr>()
        .with_context(|| format!("invalid libp2p listen address {listen}"))?;
    swarm
        .listen_on(p2p_listen)
        .with_context(|| format!("failed to start libp2p listener on {listen}"))
}

fn start_grid_listeners(
    swarm: &mut Swarm<GridBehaviour>,
    config: &DiscoveryConfig,
) -> Result<ListenerId> {
    let direct_address = config
        .p2p_listen
        .parse::<Multiaddr>()
        .with_context(|| format!("invalid libp2p listen address {}", config.p2p_listen))?;
    let direct_listener = listen_on(swarm, &config.p2p_listen)?;

    if let Some(quic_address) = quic_listen_address(&direct_address) {
        swarm
            .listen_on(quic_address.clone())
            .with_context(|| format!("failed to start libp2p QUIC listener on {quic_address}"))?;
    }

    if config.relay_server {
        add_relay_server_external_addresses(swarm, config)?;
    }

    // The relay client associates one reservation with one direct relay
    // connection. Starting two listeners for the same PeerId (for example an
    // IP and DNS spelling of one bootstrap) can replace the first reservation
    // inside libp2p, so reserve once per relay identity.
    let mut seen_relays = HashSet::new();
    for relay_peer in &config.relay_peers {
        let (relay_peer_id, circuit_address) = relay_circuit_listen_address(relay_peer)?;
        if !seen_relays.insert(relay_peer_id) {
            continue;
        }
        swarm
            .behaviour_mut()
            .gossipsub
            .add_explicit_peer(&relay_peer_id);
        swarm
            .listen_on(circuit_address.clone())
            .with_context(|| format!("failed to reserve Infernet relay {circuit_address}"))?;
        println!("libp2p_relay_reserving={circuit_address}");
    }

    Ok(direct_listener)
}

fn quic_listen_address(address: &Multiaddr) -> Option<Multiaddr> {
    let mut quic = Multiaddr::empty();
    let mut converted = false;
    for protocol in address.iter() {
        if converted {
            // TCP encapsulations such as WebSocket cannot be mechanically
            // translated into QUIC listen addresses.
            return None;
        }
        match protocol {
            Protocol::Tcp(port) => {
                quic.push(Protocol::Udp(port));
                quic.push(Protocol::QuicV1);
                converted = true;
            }
            protocol => quic.push(protocol),
        }
    }
    converted.then_some(quic)
}

fn relay_circuit_listen_address(relay_peer: &str) -> Result<(PeerId, Multiaddr)> {
    let relay_address = relay_peer
        .parse::<Multiaddr>()
        .with_context(|| format!("invalid Infernet relay multiaddress {relay_peer}"))?;
    if relay_address
        .iter()
        .any(|protocol| matches!(protocol, Protocol::P2pCircuit))
    {
        bail!(
            "Infernet relay address must identify the relay itself, without /p2p-circuit: {relay_peer}"
        );
    }
    let Some(Protocol::P2p(relay_peer_id)) = relay_address.iter().last() else {
        bail!("Infernet relay address must end in /p2p/<relay-peer-id>: {relay_peer}");
    };
    if relay_address.iter().count() < 2 {
        bail!("Infernet relay address is missing a transport: {relay_peer}");
    }

    Ok((relay_peer_id, relay_address.with(Protocol::P2pCircuit)))
}

fn add_relay_server_external_addresses(
    swarm: &mut Swarm<GridBehaviour>,
    config: &DiscoveryConfig,
) -> Result<()> {
    let local_peer_id = *swarm.local_peer_id();
    let Some(advertisement) = config.advertisement.as_ref() else {
        return Ok(());
    };
    for address in &advertisement.addresses {
        let mut address = address
            .parse::<Multiaddr>()
            .with_context(|| format!("invalid relay server public address {address}"))?;
        if address
            .iter()
            .any(|protocol| matches!(protocol, Protocol::P2pCircuit))
        {
            continue;
        }
        match address.iter().last() {
            Some(Protocol::P2p(peer_id)) if peer_id != local_peer_id => {
                bail!("relay server public address identifies {peer_id}, expected {local_peer_id}");
            }
            Some(Protocol::P2p(_)) => {}
            _ => address.push(Protocol::P2p(local_peer_id)),
        }
        swarm.add_external_address(address);
    }
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

fn publish_local_advertisement(
    swarm: &mut Swarm<GridBehaviour>,
    topic: &gossipsub::IdentTopic,
    advertisement: &mut Option<NodeAdvertisement>,
    local_peer_id: &str,
) -> Result<()> {
    let Some(advertisement) = advertisement else {
        return Ok(());
    };

    refresh_local_advertisement_capabilities(advertisement, local_peer_id);
    publish_advertisement(swarm, topic, advertisement)
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
    relay_peers: &[String],
) {
    let relay_peer_ids = relay_peers
        .iter()
        .filter_map(|address| address.parse::<Multiaddr>().ok())
        .filter_map(|address| match address.iter().last() {
            Some(Protocol::P2p(peer_id)) => Some(peer_id),
            _ => None,
        })
        .collect::<HashSet<_>>();

    for advertisement in advertisements {
        let Ok(peer_id) = advertisement.peer_id.parse::<PeerId>() else {
            continue;
        };
        // The relay behaviour owns this peer's addresses and connection.
        // Feeding the same peer into gossipsub/request-response discovery can
        // schedule a competing normal dial before the reservation is sent.
        if relay_peer_ids.contains(&peer_id) {
            continue;
        }

        add_advertisement_addresses(swarm, advertisement);
        let dial_addresses = advertisement
            .addresses
            .iter()
            .filter_map(|address| address.parse::<Multiaddr>().ok())
            .collect::<Vec<_>>();
        if dial_addresses.is_empty() {
            continue;
        }

        // Configured bootstrap/static peers form the initial network mesh, so
        // they are the only advertisements that should proactively connect.
        // Periodic calls are harmless because peer-aware dialing is rejected
        // while already connected or dialing.
        swarm.behaviour_mut().gossipsub.add_explicit_peer(&peer_id);
        // A relay listener must establish and own its reservation connection.
        // Issuing a normal static dial to that same PeerId races the relay
        // behaviour and can leave the client connected but without a circuit.
        let dial = DialOpts::peer_id(peer_id)
            .condition(PeerCondition::DisconnectedAndNotDialing)
            .addresses(dial_addresses)
            .extend_addresses_through_behaviour()
            .build();
        let _ = swarm.dial(dial);
    }
}

fn add_advertisement_addresses(
    swarm: &mut Swarm<GridBehaviour>,
    advertisement: &NodeAdvertisement,
) {
    let Ok(peer_id) = advertisement.peer_id.parse::<PeerId>() else {
        return;
    };

    for address in &advertisement.addresses {
        if let Ok(address) = address.parse::<Multiaddr>() {
            swarm.add_peer_address(peer_id, address);
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
    failed_peers: &HashMap<String, Instant>,
) -> Option<ModelShardCandidate> {
    registry
        .advertisements()
        .into_iter()
        .filter(|advertisement| advertisement.peer_id != local_peer_id)
        .filter(|advertisement| !model_fetch_peer_is_blocked(failed_peers, &advertisement.peer_id))
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
                            matches!(
                                manifest.payload_kind.as_str(),
                                model_distribution::PAYLOAD_KIND_GGUF_SHARD
                                    | model_distribution::PAYLOAD_KIND_INFERNET_SHARD
                                    | model_distribution::PAYLOAD_KIND_FULL_MODEL
                            )
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
    failed_peers: &HashMap<String, Instant>,
) -> Option<NodeAdvertisement> {
    registry
        .advertisements()
        .into_iter()
        .filter(|advertisement| advertisement.peer_id != local_peer_id)
        .filter(|advertisement| !model_fetch_peer_is_blocked(failed_peers, &advertisement.peer_id))
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

fn record_model_fetch_peer_failure(
    failed_peers: &mut HashMap<String, Instant>,
    peer_id: impl Into<String>,
) {
    failed_peers.insert(
        peer_id.into(),
        Instant::now() + MODEL_FETCH_PEER_RETRY_COOLDOWN,
    );
}

fn model_fetch_peer_is_blocked(failed_peers: &HashMap<String, Instant>, peer_id: &str) -> bool {
    failed_peers
        .get(peer_id)
        .is_some_and(|retry_at| Instant::now() < *retry_at)
}

fn hop_addresses(hop: &RouteHop) -> Result<Vec<Multiaddr>> {
    if hop.address.trim().is_empty() {
        return Ok(Vec::new());
    }

    let mut address = hop
        .address
        .parse::<Multiaddr>()
        .with_context(|| format!("invalid route multiaddr {}", hop.address))?;

    // `request_response::send_request_with_addresses` already receives the
    // destination PeerId separately. Its explicit dial addresses therefore
    // must not repeat that PeerId as a trailing `/p2p/...` component. Keeping
    // it causes libp2p to discard the otherwise-valid direct address and fall
    // back to unrelated observed addresses (for example a stale private-IP
    // address), which breaks worker-to-worker forwarding.
    if let Some(Protocol::P2p(address_peer_id)) = address.iter().last() {
        let route_peer_id = hop
            .peer_id
            .parse::<PeerId>()
            .with_context(|| format!("invalid libp2p peer id {}", hop.peer_id))?;
        if address_peer_id != route_peer_id {
            bail!(
                "route multiaddr identifies peer {}, expected {}",
                address_peer_id,
                route_peer_id
            );
        }
        address.pop();
    }

    if address.is_empty() {
        bail!("route multiaddr {} has no transport", hop.address);
    }

    Ok(vec![address])
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

    fn official_rpc_fixture() -> (
        WorkerConfig,
        LayerRange,
        ActivationRequest,
        SeedShardManifest,
        InfernetShardPackageManifest,
    ) {
        let model = ModelManifest::infernet_chat_v1();
        let release = OfficialModelRelease::infernet_chat_v1_compatibility();
        let layers = LayerRange::new(0, model.layer_count).unwrap();
        let component = release
            .components
            .iter()
            .find(|component| component.layers == Some(layers))
            .unwrap();
        let manifest = SeedShardManifest {
            model_id: model.model_id.clone(),
            display_name: model.display_name.clone(),
            architecture: model.architecture.clone(),
            layer_count: model.layer_count,
            hidden_size: model.hidden_size,
            activation_dtype: model.activation_dtype.clone(),
            runtime_kind: model.runtime_kind.clone(),
            layers,
            tokenizer: infernet_model::TokenizerCompatibility {
                family: "gemma".to_owned(),
                checksum: None,
            },
            metadata: infernet_model::ShardMetadata {
                architecture: model.architecture.clone(),
                quantization: model.quantization.clone(),
                source_checksum: Some(release.upstream.source_sha256.clone()),
                protocol_version: PROTOCOL_VERSION,
            },
            source: infernet_model::SeedSourceMetadata {
                path: "/official/infernet-chat-v1.gguf".to_owned(),
                checksum_sha256: release.upstream.source_sha256.clone(),
                file_size_bytes: release.expected_total_bytes,
            },
            shard_hash: "official-full-model-shard".to_owned(),
            payload_kind: PAYLOAD_KIND_FULL_MODEL.to_owned(),
        };
        let package = InfernetShardPackageManifest {
            format_version: INFERNET_SHARD_FORMAT_VERSION.to_owned(),
            runtime_abi: INFERNET_FULL_MODEL_RUNTIME_ABI.to_owned(),
            component: "full_model".to_owned(),
            seed_manifest: manifest.clone(),
            payload: InfernetShardPayloadManifest {
                kind: "gguf_tensor_payload".to_owned(),
                file: INFERNET_SHARD_TENSOR_FILE.to_owned(),
                checksum_sha256: component.sha256.clone(),
                size_bytes: component.size_bytes,
            },
        };
        let request = ActivationRequest::new(
            model.model_id.clone(),
            vec![hop("coordinator", layers.start, layers.end)],
            model.hidden_size,
            Vec::new(),
            Some(PromptMetadata {
                prompt: "hello".to_owned(),
                demo_mode: false,
                rpc_endpoints: vec!["192.168.1.20:50052".to_owned()],
                rpc_worker_peer_ids: Vec::new(),
            }),
        );
        let worker = WorkerConfig {
            peer_id: "coordinator".to_owned(),
            model_id: model.model_id,
            runtime_kind: model.runtime_kind,
            owned_layers: layers,
            hidden_size: model.hidden_size,
            shard_cache: None,
        };

        (worker, layers, request, manifest, package)
    }

    #[test]
    fn discovery_config_has_no_rpc_backends_by_default() {
        assert!(
            DiscoveryConfig::new("infernet/test")
                .rpc_endpoints
                .is_empty()
        );
    }

    #[test]
    fn discovery_config_does_not_accidentally_run_a_public_relay() {
        let discovery = DiscoveryConfig::new("infernet/test");
        assert!(!discovery.relay_server);
        assert!(discovery.relay_peers.is_empty());
    }

    #[test]
    fn tcp_listeners_get_a_matching_quic_fallback() {
        let tcp = "/ip4/0.0.0.0/tcp/9777".parse::<Multiaddr>().unwrap();
        assert_eq!(
            quic_listen_address(&tcp).unwrap().to_string(),
            "/ip4/0.0.0.0/udp/9777/quic-v1"
        );

        let websocket = "/ip4/0.0.0.0/tcp/9777/ws".parse::<Multiaddr>().unwrap();
        assert!(quic_listen_address(&websocket).is_none());
    }

    #[test]
    fn relay_peer_address_becomes_a_circuit_listener() {
        let relay_peer_id = identity::Keypair::generate_ed25519().public().to_peer_id();
        let address = format!("/dns4/relay.example/tcp/9777/p2p/{relay_peer_id}");
        let (parsed_peer_id, circuit) = relay_circuit_listen_address(&address).unwrap();

        assert_eq!(parsed_peer_id, relay_peer_id);
        assert_eq!(circuit.to_string(), format!("{address}/p2p-circuit"));
        assert!(relay_circuit_listen_address("/dns4/relay.example/tcp/9777").is_err());
        assert!(relay_circuit_listen_address(&format!("{address}/p2p-circuit")).is_err());
    }

    #[test]
    fn public_relay_can_carry_a_full_model_transfer() {
        let config = public_relay_server_config();
        assert!(config.max_circuit_bytes >= 14_400_000_000);
        assert!(config.max_circuit_duration >= Duration::from_secs(60 * 60));
        assert!(config.reservation_duration >= config.max_circuit_duration);
    }

    #[tokio::test]
    async fn relay_server_behaviour_is_explicitly_toggled() {
        let topic = gossipsub::IdentTopic::new("infernet/test/relay-toggle");
        let disabled =
            build_grid_swarm(identity::Keypair::generate_ed25519(), &topic, false, true).unwrap();
        assert!(!disabled.behaviour().relay_server.is_enabled());

        let enabled =
            build_grid_swarm(identity::Keypair::generate_ed25519(), &topic, true, true).unwrap();
        assert!(enabled.behaviour().relay_server.is_enabled());
    }

    #[tokio::test]
    async fn arbitrary_advertisements_do_not_queue_unsolicited_dials() {
        let topic = gossipsub::IdentTopic::new("infernet/test/single-dial");
        let mut swarm =
            build_grid_swarm(identity::Keypair::generate_ed25519(), &topic, false, false).unwrap();
        let remote_peer_id = identity::Keypair::generate_ed25519().public().to_peer_id();
        let advertisement = empty_advertisement(
            remote_peer_id.to_string(),
            format!("/ip4/127.0.0.1/tcp/9/p2p/{remote_peer_id}"),
        );

        add_advertisement_addresses(&mut swarm, &advertisement);
        add_advertisement_addresses(&mut swarm, &advertisement);

        assert_eq!(
            swarm
                .network_info()
                .connection_counters()
                .num_pending_outgoing(),
            0
        );
    }

    #[tokio::test]
    async fn repeated_static_peer_updates_queue_only_one_dial() {
        let topic = gossipsub::IdentTopic::new("infernet/test/static-single-dial");
        let mut swarm =
            build_grid_swarm(identity::Keypair::generate_ed25519(), &topic, false, false).unwrap();
        let remote_peer_id = identity::Keypair::generate_ed25519().public().to_peer_id();
        let advertisement = empty_advertisement(
            remote_peer_id.to_string(),
            format!("/ip4/127.0.0.1/tcp/9/p2p/{remote_peer_id}"),
        );

        add_static_peer_addresses(&mut swarm, std::slice::from_ref(&advertisement), &[]);
        add_static_peer_addresses(&mut swarm, std::slice::from_ref(&advertisement), &[]);

        assert_eq!(
            swarm
                .network_info()
                .connection_counters()
                .num_pending_outgoing(),
            1
        );
    }

    #[tokio::test]
    async fn static_bootstrap_that_is_also_a_relay_does_not_race_reservation() {
        let topic = gossipsub::IdentTopic::new("infernet/test/static-relay-no-race");
        let mut swarm =
            build_grid_swarm(identity::Keypair::generate_ed25519(), &topic, false, false).unwrap();
        let relay_peer_id = identity::Keypair::generate_ed25519().public().to_peer_id();
        let relay_address = format!("/ip4/127.0.0.1/tcp/9/p2p/{relay_peer_id}");
        let advertisement = empty_advertisement(relay_peer_id.to_string(), relay_address.clone());

        add_static_peer_addresses(
            &mut swarm,
            std::slice::from_ref(&advertisement),
            &[relay_address],
        );

        assert_eq!(
            swarm
                .network_info()
                .connection_counters()
                .num_pending_outgoing(),
            0
        );
    }

    #[tokio::test]
    async fn relay_client_obtains_a_real_circuit_listen_address() {
        let topic = gossipsub::IdentTopic::new(format!(
            "infernet/test/relay-reservation/{}",
            uuid::Uuid::new_v4()
        ));
        let mut relay =
            build_grid_swarm(identity::Keypair::generate_ed25519(), &topic, true, false).unwrap();
        let relay_peer_id = *relay.local_peer_id();
        let relay_listener = listen_on(&mut relay, "/ip4/127.0.0.1/tcp/0").unwrap();
        let relay_transport_address = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if let SwarmEvent::NewListenAddr {
                    listener_id,
                    address,
                } = relay.select_next_some().await
                    && listener_id == relay_listener
                {
                    break address;
                }
            }
        })
        .await
        .expect("relay did not start listening");
        let relay_public_address = relay_transport_address.with(Protocol::P2p(relay_peer_id));
        relay.add_external_address(relay_public_address.clone());

        let mut client =
            build_grid_swarm(identity::Keypair::generate_ed25519(), &topic, false, false).unwrap();
        let client_peer_id = *client.local_peer_id();
        let circuit_listener = client
            .listen_on(relay_public_address.with(Protocol::P2pCircuit))
            .unwrap();

        let circuit_address = tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                tokio::select! {
                    _ = relay.select_next_some() => {}
                    event = client.select_next_some() => {
                        if let SwarmEvent::NewListenAddr { listener_id, address } = event
                            && listener_id == circuit_listener
                            && address.iter().any(|protocol| matches!(protocol, Protocol::P2pCircuit))
                        {
                            break address;
                        }
                    }
                }
            }
        })
        .await
        .expect("client did not obtain a relay reservation");

        assert!(matches!(
            circuit_address.iter().last(),
            Some(Protocol::P2p(peer_id)) if peer_id == client_peer_id
        ));
    }

    #[test]
    fn trusted_rpc_endpoints_are_canonicalized_and_deduplicated() {
        let endpoints = normalize_trusted_rpc_endpoints(&[
            " 127.0.0.1:50052 ".to_owned(),
            "127.0.0.1:50052".to_owned(),
            "127.0.0.2:6000".to_owned(),
        ])
        .unwrap();

        assert_eq!(endpoints, ["127.0.0.1:50052", "127.0.0.2:6000"]);
    }

    #[test]
    fn rpc_endpoints_reject_public_dns_and_malformed_targets() {
        for endpoint in [
            "8.8.8.8:50052",
            "worker.example.com:50052",
            "192.168.1.20:50052",
            "100.64.2.3:6000",
            "169.254.10.8:50052",
            "[::1]:50052",
            "127.0.0.1:0",
            "127.0.0.1:not-a-port",
        ] {
            assert!(
                normalize_trusted_rpc_endpoints(&[endpoint.to_owned()]).is_err(),
                "accepted unsafe RPC endpoint {endpoint}"
            );
        }
    }

    #[test]
    fn rpc_execution_accepts_exact_official_single_hop_package() {
        let (worker, layers, request, manifest, package) = official_rpc_fixture();
        let release = OfficialModelRelease::infernet_chat_v1_compatibility();

        validate_official_rpc_package(
            &worker,
            layers,
            &request,
            &manifest,
            &package,
            release.expected_total_bytes,
        )
        .unwrap();
    }

    #[test]
    fn rpc_execution_rejects_partial_multihop_and_incompatible_packages() {
        let (worker, layers, request, manifest, package) = official_rpc_fixture();
        let release = OfficialModelRelease::infernet_chat_v1_compatibility();

        let partial_layers = LayerRange::new(0, layers.end - 1).unwrap();
        assert!(
            validate_official_rpc_package(
                &worker,
                partial_layers,
                &request,
                &manifest,
                &package,
                release.expected_total_bytes,
            )
            .is_err()
        );

        let mut multihop = request.clone();
        multihop.route.push(hop("unexpected-peer", 0, 1));
        assert!(
            validate_official_rpc_package(
                &worker,
                layers,
                &multihop,
                &manifest,
                &package,
                release.expected_total_bytes,
            )
            .is_err()
        );

        let mut incompatible = package.clone();
        incompatible.runtime_abi = "wrong-runtime-abi".to_owned();
        assert!(
            validate_official_rpc_package(
                &worker,
                layers,
                &request,
                &manifest,
                &incompatible,
                release.expected_total_bytes,
            )
            .is_err()
        );
    }

    #[test]
    fn llama_bridge_receives_one_rpc_comma_list() {
        let mut command = Command::new("infernet-llama-bridge");
        append_llama_rpc_arguments(
            &mut command,
            &[
                "192.168.1.20:50052".to_owned(),
                "100.64.2.3:50052".to_owned(),
            ],
        );
        let args = command
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect::<Vec<_>>();

        assert_eq!(args, ["--rpc", "192.168.1.20:50052,100.64.2.3:50052"]);
    }

    #[cfg(unix)]
    #[test]
    fn llama_bridge_deadline_preserves_normal_json_output() {
        let mut command = Command::new("sh");
        command.arg("-c").arg(
            r#"printf '%s\n' 'bridge startup' '{"ok":true,"output_text":"token","timing_ms":12.5}'"#,
        );

        let output = run_llama_bridge_with_timeout(&mut command, Duration::from_secs(2)).unwrap();
        assert!(output.status.success());

        let parsed = parse_llama_bridge_json(&output).unwrap();
        assert!(parsed.ok);
        assert_eq!(parsed.output_text.as_deref(), Some("token"));
        assert_eq!(parsed.timing_ms, Some(12.5));
    }

    #[cfg(unix)]
    #[test]
    fn llama_bridge_deadline_kills_and_reaps_process() {
        let pid_path = env::temp_dir().join(format!(
            "infernet-bridge-timeout-{}.pid",
            uuid::Uuid::new_v4()
        ));
        let mut command = Command::new("sh");
        command
            .arg("-c")
            .arg(r#"printf '%s' "$$" > "$1"; exec sleep 30"#)
            .arg("infernet-bridge-timeout-test")
            .arg(&pid_path);

        let started = std::time::Instant::now();
        let error = run_llama_bridge_with_timeout(&mut command, Duration::from_millis(150))
            .expect_err("long-running bridge should be terminated");
        assert!(started.elapsed() < Duration::from_secs(2));
        let error = format!("{error:#}");
        assert!(
            error.contains("exceeded the 0.1s execution deadline"),
            "{error}"
        );
        assert!(error.contains("was terminated"), "{error}");

        let pid = fs::read_to_string(&pid_path).unwrap();
        let status = Command::new("kill")
            .arg("-0")
            .arg(pid.trim())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .unwrap();
        assert!(
            !status.success(),
            "timed-out bridge process {pid} is still alive"
        );

        let _ = fs::remove_file(pid_path);
    }

    #[tokio::test]
    async fn model_distribution_reports_actual_listen_address_when_ready() {
        let root = std::env::temp_dir().join(format!(
            "infernet-distribution-ready-{}",
            uuid::Uuid::new_v4()
        ));
        let mut discovery = DiscoveryConfig::new("infernet/test/readiness");
        discovery.p2p_listen = "/ip4/127.0.0.1/tcp/0".to_owned();
        let (readiness_sender, readiness_receiver) = oneshot::channel();
        let service = tokio::spawn(run_model_distribution_node_with_readiness(
            discovery,
            ShardCacheConfig::new(root.clone()),
            readiness_sender,
        ));

        let address = tokio::time::timeout(Duration::from_secs(5), readiness_receiver)
            .await
            .expect("model distribution service did not report readiness")
            .expect("model distribution readiness channel closed")
            .expect("model distribution listener failed");

        assert!(address.starts_with("/ip4/127.0.0.1/tcp/"), "{address}");
        assert!(!address.ends_with("/tcp/0"), "{address}");
        assert!(!service.is_finished());

        service.abort();
        let _ = service.await;
        let _ = fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn model_distribution_reports_startup_failure_to_readiness_waiter() {
        let root = std::env::temp_dir().join(format!(
            "infernet-distribution-failed-{}",
            uuid::Uuid::new_v4()
        ));
        let mut discovery = DiscoveryConfig::new("infernet/test/readiness-failure");
        discovery.p2p_listen = "not-a-multiaddr".to_owned();
        let (readiness_sender, readiness_receiver) = oneshot::channel();
        let service = tokio::spawn(run_model_distribution_node_with_readiness(
            discovery,
            ShardCacheConfig::new(root.clone()),
            readiness_sender,
        ));

        let error = tokio::time::timeout(Duration::from_secs(5), readiness_receiver)
            .await
            .expect("model distribution service did not report startup failure")
            .expect("model distribution readiness channel closed")
            .expect_err("invalid listen address should fail startup");
        assert!(error.contains("invalid libp2p listen address"), "{error}");
        assert!(service.await.unwrap().is_err());

        let _ = fs::remove_dir_all(root);
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
                source_checksum: Some("source-checksum".to_owned()),
                protocol_version: PROTOCOL_VERSION,
            },
            source: infernet_model::SeedSourceMetadata {
                path: "/seed/gemma.gguf".to_owned(),
                checksum_sha256: "source-checksum".to_owned(),
                file_size_bytes: 16,
            },
            shard_hash: "seed-hash".to_owned(),
            payload_kind: model_distribution::PAYLOAD_KIND_FULL_MODEL.to_owned(),
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

    #[test]
    fn full_model_package_is_a_download_candidate() {
        let (_, layers, _, manifest, package) = official_rpc_fixture();
        let mut advertisement =
            empty_advertisement("seed-peer".to_owned(), "/ip4/127.0.0.1/tcp/9777".to_owned());
        advertisement.hosted_shards.push(ShardDescriptor {
            model_id: manifest.model_id.clone(),
            layers,
            runtime_kind: manifest.runtime_kind.clone(),
            tokenizer: Some(manifest.tokenizer.clone()),
            metadata: Some(manifest.metadata.clone()),
            shard_hash: Some(manifest.shard_hash.clone()),
            seed_manifest: Some(Box::new(manifest.clone())),
        });
        advertisement.model_shards.push(ModelShardInfo {
            model_id: manifest.model_id.clone(),
            layers,
            checksum: package.payload.checksum_sha256.clone(),
            size_bytes: package.payload.size_bytes,
            version: "1".to_owned(),
            protocol_version: PROTOCOL_VERSION,
        });
        let mut registry = ShardRegistry::new();
        registry.upsert(advertisement);

        let candidate = select_model_shard_candidate(
            &registry,
            "downloader",
            &manifest.model_id,
            layers,
            Some(&package.payload.checksum_sha256),
            Some("1"),
            &HashMap::new(),
        );

        assert!(candidate.is_some());
        assert_eq!(
            candidate.unwrap().seed_manifest.unwrap().payload_kind,
            PAYLOAD_KIND_FULL_MODEL
        );
    }
}
