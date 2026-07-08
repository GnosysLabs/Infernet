pub mod model_distribution;

use std::collections::HashMap;
use std::mem;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::time::Duration;
use std::{fs, io};

use anyhow::{Context, Result, anyhow, bail};
use futures::StreamExt;
use infernet_model::{LayerRange, ModelManifest, RuntimeKind, SeedShardManifest, ShardDescriptor};
use infernet_protocol::{
    ACTIVATION_PROTOCOL, ActivationRequest, ActivationResponse, MODEL_PROTOCOL, ModelShardInfo,
    ModelShardRequest, ModelShardResponse, NodeAdvertisement, PROTOCOL_VERSION, PromptMetadata,
    RouteHop, TraceEvent,
};
use infernet_router::ShardRegistry;
use infernet_runtime::{DemoRuntime, LayerRuntime, activation_checksum};
use libp2p::{
    Multiaddr, PeerId, StreamProtocol, Swarm, SwarmBuilder, gossipsub, identity, mdns, noise,
    request_response,
    swarm::{NetworkBehaviour, SwarmEvent},
    tcp, yamux,
};
pub use model_distribution::{
    CachedShardRecord, SeededModelSummary, ShardCache, ShardCacheConfig, ShardCacheStats,
    import_seed_model_from_file, import_seed_model_from_file_with_progress, sha256_bytes,
};
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

#[derive(NetworkBehaviour)]
struct GridBehaviour {
    gossipsub: gossipsub::Behaviour,
    mdns: mdns::tokio::Behaviour,
    activation: request_response::json::Behaviour<ActivationRequest, ActivationResponse>,
    model: request_response::json::Behaviour<ModelShardRequest, ModelShardResponse>,
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

enum GridNetworkEvent {
    Activation(ActivationNetworkEvent),
    Model(ModelNetworkEvent),
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
    let mut pending_forwards = HashMap::new();

    loop {
        tokio::select! {
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
                )? {
                    match network_event {
                        GridNetworkEvent::Activation(event) => {
                            handle_worker_activation_event(
                                &mut swarm,
                                &worker,
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

    loop {
        tokio::select! {
            _ = publish_interval.tick() => {
                refresh_advertisement_model_shards(&mut discovery.advertisement, Some(&shard_cache))?;
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
                )? {
                    if let GridNetworkEvent::Model(event) = network_event {
                        handle_model_network_event(&mut swarm, Some(&shard_cache), &peer_id, event)?;
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
    let mut pending_request: Option<(request_response::OutboundRequestId, ModelShardInfo, String)> =
        None;

    loop {
        if pending_request.is_none() {
            if let Some((advertisement, shard)) = select_model_shard_candidate(
                &registry,
                &local_peer_id,
                &model_id,
                layers,
                checksum.as_deref(),
                version.as_deref(),
            ) {
                let request = ModelShardRequest::new(
                    model_id.clone(),
                    layers,
                    Some(shard.checksum.clone()),
                    Some(shard.version.clone()),
                );
                let request_id = send_model_shard_request(&mut swarm, &advertisement, request)?;
                pending_request = Some((request_id, shard, advertisement.peer_id));
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
                )? {
                    match network_event {
                        GridNetworkEvent::Model(ModelNetworkEvent::Response { request_id, response }) => {
                            if let Some((pending_id, expected, peer_id)) = pending_request.take() {
                                if request_id != pending_id {
                                    pending_request = Some((pending_id, expected, peer_id));
                                    continue;
                                }

                                if let Some(error) = response.error {
                                    bail!("model shard request to {peer_id} failed: {error}");
                                }

                                let response_shard = response
                                    .shard
                                    .ok_or_else(|| anyhow!("model shard response from {peer_id} omitted shard metadata"))?;
                                if response_shard != expected {
                                    bail!(
                                        "model shard metadata mismatch from {peer_id}; expected {:?}, got {:?}",
                                        expected,
                                        response_shard
                                    );
                                }

                                let cache_record = cache.store_downloaded(&expected, response.payload)?;
                                refresh_advertisement_model_shards(&mut config.advertisement, Some(&cache))?;
                                if let Some(advertisement) = &config.advertisement {
                                    publish_advertisement(&mut swarm, &topic, advertisement)?;
                                }

                                return Ok(ModelFetchResult {
                                    shard: expected,
                                    source_peer_id: peer_id,
                                    cache_record,
                                });
                            }
                        }
                        GridNetworkEvent::Model(ModelNetworkEvent::OutboundFailure { peer, request_id, error }) => {
                            if let Some((pending_id, expected, peer_id)) = pending_request.take() {
                                if request_id == pending_id {
                                    bail!("model shard request to {peer} failed: {error}");
                                }
                                pending_request = Some((pending_id, expected, peer_id));
                            }
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
                    "timed out discovering model shard {} {}:{} checksum {}",
                    model_id,
                    layers.start,
                    layers.end,
                    checksum.as_deref().unwrap_or("<any>")
                );
            }
        }
    }
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

    if manifest.runtime_kind != RuntimeKind::Demo {
        bail!(
            "model {} discovered a complete route, but the {} shard runtime is not linked yet; see docs/gguf-split-inference-design.md",
            manifest.model_id,
            manifest.runtime_kind.as_str()
        );
    }

    let activation = DemoRuntime::prompt_to_activation(&prompt, hidden_size);
    let request = ActivationRequest::new(
        manifest.model_id.clone(),
        route.clone(),
        hidden_size,
        activation,
        Some(PromptMetadata {
            prompt,
            demo_mode: true,
        }),
    );
    let first_hop = request
        .current_hop()
        .cloned()
        .ok_or_else(|| anyhow!("route must contain at least one hop"))?;
    let outbound_id = send_activation_request(&mut swarm, &first_hop, request)?;
    let response = wait_for_client_response(
        &mut swarm,
        &mut registry,
        &mut config,
        &topic,
        outbound_id,
        Duration::from_secs(15),
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
            advertisement.model_shards = records.iter().map(|record| record.info.clone()).collect();
            advertisement.hosted_shards = records
                .iter()
                .filter_map(|record| {
                    let payload = cache.read_payload(&record.info).ok()?;
                    let manifest = serde_json::from_slice::<SeedShardManifest>(&payload).ok()?;
                    Some(ShardDescriptor {
                        model_id: manifest.model_id,
                        layers: manifest.layers,
                        runtime_kind: manifest.runtime_kind,
                        tokenizer: Some(manifest.tokenizer),
                        metadata: Some(manifest.metadata),
                        shard_hash: Some(manifest.shard_hash),
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
    let runtime = match config.runtime_kind {
        RuntimeKind::Demo => DemoRuntime::new(config.owned_layers, config.hidden_size),
        RuntimeKind::LlamaCpp => {
            return Err(ActivationResponse::failure(
                trace_id,
                config.peer_id.clone(),
                "llama.cpp shard runtime is not linked yet; route discovery and metadata are available, but real GGUF layer execution requires the Infernet llama.cpp bridge described in docs/gguf-split-inference-design.md",
                request.trace,
            ));
        }
    };
    let started = Instant::now();

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

    let timing_ms = elapsed_ms(started);
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
        let output = DemoRuntime::decode_activation(&request.activation);
        Ok(ActivationStep::Final(ActivationResponse::success(
            request,
            config.peer_id.clone(),
            Some(output),
            timing_ms,
        )))
    }
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
    network_event: ActivationNetworkEvent,
    pending_forwards: &mut HashMap<request_response::OutboundRequestId, PendingOutbound>,
) -> Result<()> {
    match network_event {
        ActivationNetworkEvent::Request { request, channel } => {
            let trace_id = request.trace_id;

            match process_activation_step(worker, request) {
                Ok(ActivationStep::Final(response)) => {
                    send_response(swarm, channel, response);
                }
                Ok(ActivationStep::Forward(request)) => {
                    let next_hop = request
                        .current_hop()
                        .cloned()
                        .ok_or_else(|| anyhow!("forwarded request has no current hop"))?;
                    match send_activation_request(swarm, &next_hop, request.clone()) {
                        Ok(request_id) => {
                            pending_forwards.insert(
                                request_id,
                                PendingOutbound::Forward {
                                    channel,
                                    trace_id,
                                    peer_id: worker.peer_id.clone(),
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
                                    worker.peer_id.clone(),
                                    format!("failed to forward activation: {error:#}"),
                                    request.trace.clone(),
                                ),
                            );
                        }
                    }
                }
                Err(response) => {
                    send_response(swarm, channel, response);
                }
            }
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

    match cache.read_payload(&record.info) {
        Ok(payload) => {
            ModelShardResponse::success(request, peer_id.to_owned(), record.info, payload)
        }
        Err(error) => ModelShardResponse::failure(request, peer_id.to_owned(), error.to_string()),
    }
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

fn send_model_shard_request(
    swarm: &mut Swarm<GridBehaviour>,
    advertisement: &NodeAdvertisement,
    request: ModelShardRequest,
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
        swarm.behaviour_mut().model.send_request(&peer_id, request)
    } else {
        swarm
            .behaviour_mut()
            .model
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

fn handle_grid_event(
    swarm: &mut Swarm<GridBehaviour>,
    event: SwarmEvent<GridBehaviourEvent>,
    registry: &mut ShardRegistry,
    advertisement: &mut Option<NodeAdvertisement>,
    topic: &gossipsub::IdentTopic,
) -> Result<Option<GridNetworkEvent>> {
    match event {
        SwarmEvent::NewListenAddr { address, .. } => {
            let peer_id = *swarm.local_peer_id();
            if update_listen_address(advertisement, peer_id, address) {
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
            println!("libp2p_connected peer_id={} endpoint={:?}", peer_id, endpoint);
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
            for (peer_id, address) in peers {
                swarm.behaviour_mut().gossipsub.add_explicit_peer(&peer_id);
                swarm.add_peer_address(peer_id, address);
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
                add_advertisement_addresses(swarm, &advertisement);
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

            Ok(GridBehaviour {
                gossipsub,
                mdns,
                activation,
                model,
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

fn select_model_shard_candidate(
    registry: &ShardRegistry,
    local_peer_id: &str,
    model_id: &str,
    layers: LayerRange,
    checksum: Option<&str>,
    version: Option<&str>,
) -> Option<(NodeAdvertisement, ModelShardInfo)> {
    registry
        .advertisements()
        .into_iter()
        .filter(|advertisement| advertisement.peer_id != local_peer_id)
        .flat_map(|advertisement| {
            advertisement
                .model_shards
                .clone()
                .into_iter()
                .map(move |shard| (advertisement.clone(), shard))
        })
        .filter(|(_, shard)| {
            shard.model_id == model_id
                && shard.layers == layers
                && checksum.is_none_or(|checksum| shard.checksum == checksum)
                && version.is_none_or(|version| shard.version == version)
        })
        .min_by_key(|(advertisement, shard)| {
            (
                advertisement.latency_hint_ms.unwrap_or(u32::MAX),
                shard.size_bytes,
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
}
