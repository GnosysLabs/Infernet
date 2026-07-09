use std::{
    env, fs,
    net::{IpAddr, Ipv4Addr, TcpListener, UdpSocket},
    path::PathBuf,
    time::Duration,
};

use anyhow::{Context, Result, anyhow};
use clap::{Args, Parser, Subcommand};
use infernet_model::{
    GgufShardManifest, GgufSourceMetadata, LayerRange, ModelManifest, OfficialModelRelease,
    RuntimeKind, ShardDescriptor, ShardMetadata, TokenizerCompatibility, gguf,
};
use infernet_node::{
    DiscoveryConfig, INFERNET_LLAMA_RPC_RUNTIME_ABI, LLAMA_RPC_DEFAULT_PORT,
    LLAMA_RPC_PROTOCOL_VERSION, LlamaRpcServer, LlamaRpcServerConfig, ShardCache, ShardCacheConfig,
    WorkerConfig, clear_local_llama_rpc_endpoint, detect_node_capabilities, discover_for,
    empty_advertisement, enrich_local_advertisement, fetch_model_shard_over_libp2p,
    find_llama_rpc_server_binary, import_seed_model_from_file,
    import_seed_model_from_file_consuming, import_seed_model_from_file_consuming_verified,
    infer_over_libp2p, load_or_generate_keypair, local_capability_advertisement,
    run_model_distribution_node, run_worker_node, set_local_inference_active,
    set_local_llama_rpc_endpoint, set_local_rpc_active, spawn_llama_rpc_server,
    stop_persistent_llama_server,
};
use infernet_protocol::{LlamaRpcEndpoint, ModelShardInfo, NodeAdvertisement, RouteHop};
use infernet_router::ShardRegistry;
use sha2::{Digest, Sha256};

const DEFAULT_TOPIC: &str = "infernet/grid-demo/1";
const DEFAULT_DISCOVERY_TIMEOUT_MS: u64 = 4_000;

#[derive(Debug, Parser)]
#[command(name = "infernet-worker")]
#[command(about = "Headless Infernet worker and Phase 1 grid client")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Bootstrap(BootstrapArgs),
    Serve(ServeArgs),
    Peers(DiscoveryArgs),
    Route(RouteArgs),
    Infer(InferArgs),
    Shard(ShardArgs),
    Model(ModelArgs),
}

#[derive(Debug, Args)]
struct BootstrapArgs {
    #[arg(long, default_value = DEFAULT_TOPIC)]
    topic: String,
    #[arg(long, default_value = "/ip4/0.0.0.0/tcp/9777")]
    p2p_listen: String,
    #[arg(long, default_value = "/var/lib/infernet-bootstrap/identity.key")]
    identity_file: PathBuf,
    #[arg(long, default_value = "/var/lib/infernet-bootstrap/shards")]
    cache_dir: PathBuf,
    #[arg(long, default_value_t = 10 * 1024 * 1024 * 1024_u64)]
    max_storage_bytes: u64,
    #[arg(long)]
    public_domain: Option<String>,
    #[arg(long)]
    public_ip: Option<String>,
}

#[derive(Debug, Args)]
struct ServeArgs {
    #[arg(long, default_value = "grid-demo-12")]
    model: String,
    #[arg(long)]
    layers: String,
    #[arg(long, default_value = "/ip4/0.0.0.0/tcp/0")]
    p2p_listen: String,
    #[arg(long, default_value = DEFAULT_TOPIC)]
    topic: String,
    #[arg(long)]
    hidden_size: Option<usize>,
}

#[derive(Debug, Args)]
struct DiscoveryArgs {
    #[arg(long, default_value = DEFAULT_TOPIC)]
    topic: String,
    #[arg(long, default_value = "/ip4/0.0.0.0/tcp/0")]
    p2p_listen: String,
    #[arg(long, default_value_t = DEFAULT_DISCOVERY_TIMEOUT_MS)]
    discovery_timeout_ms: u64,
    #[arg(long = "static-peer")]
    static_peers: Vec<String>,
}

#[derive(Debug, Args)]
struct RouteArgs {
    #[command(flatten)]
    discovery: DiscoveryArgs,
    #[arg(long, default_value = "grid-demo-12")]
    model: String,
}

#[derive(Debug, Args)]
struct InferArgs {
    #[command(flatten)]
    discovery: DiscoveryArgs,
    #[arg(long, default_value = "grid-demo-12")]
    model: String,
    #[arg(long)]
    prompt: String,
    #[arg(long)]
    hidden_size: Option<usize>,
}

#[derive(Debug, Args)]
struct ShardArgs {
    #[command(subcommand)]
    command: ShardCommand,
}

#[derive(Debug, Subcommand)]
enum ShardCommand {
    Build(ShardBuildArgs),
}

#[derive(Debug, Args)]
struct ShardBuildArgs {
    #[arg(long, default_value = "llama-3.2-1b")]
    model: String,
    #[arg(long)]
    gguf: PathBuf,
    #[arg(long)]
    layers: String,
    #[arg(long)]
    out: PathBuf,
}

#[derive(Debug, Args)]
struct ModelArgs {
    #[command(subcommand)]
    command: ModelCommand,
}

#[derive(Debug, Subcommand)]
enum ModelCommand {
    AddLocal(ModelAddLocalArgs),
    Import(ModelImportArgs),
    List(ModelCacheArgs),
    Serve(ModelServeArgs),
    Fetch(ModelFetchArgs),
    Mirror(ModelFetchArgs),
}

#[derive(Debug, Args)]
struct ModelAddLocalArgs {
    #[command(flatten)]
    cache: ModelCacheArgs,
    #[arg(long, default_value = "infernet-chat-v1")]
    model: String,
    #[arg(long)]
    gguf: PathBuf,
    #[arg(long, default_value = "1.0.0-compat.1")]
    version: String,
    /// Move the verified source into Infernet's package instead of retaining a duplicate.
    #[arg(long, default_value_t = false)]
    consume_source: bool,
}

#[derive(Debug, Args)]
struct ModelCacheArgs {
    #[arg(long, default_value = ".infernet/shards")]
    cache_dir: PathBuf,
    #[arg(long, default_value_t = 50 * 1024 * 1024 * 1024_u64)]
    max_storage_bytes: u64,
    #[arg(long = "preferred-model")]
    preferred_models: Vec<String>,
    #[arg(long = "pinned-model")]
    pinned_models: Vec<String>,
    #[arg(long, default_value_t = false)]
    no_auto_cleanup: bool,
}

#[derive(Debug, Args)]
struct ModelImportArgs {
    #[command(flatten)]
    cache: ModelCacheArgs,
    #[arg(long)]
    model: String,
    #[arg(long)]
    layers: String,
    #[arg(long)]
    file: PathBuf,
    #[arg(long, default_value = "v1")]
    version: String,
}

#[derive(Debug, Args)]
struct ModelServeArgs {
    #[command(flatten)]
    cache: ModelCacheArgs,
    #[arg(long, default_value = DEFAULT_TOPIC)]
    topic: String,
    #[arg(long, default_value = "/ip4/0.0.0.0/tcp/0")]
    p2p_listen: String,
    /// Disable the private llama.cpp RPC compute sidecar.
    #[arg(long, default_value_t = false)]
    no_rpc: bool,
    /// Private RFC1918 or Tailscale IPv4 address to bind and advertise.
    #[arg(long)]
    rpc_host: Option<Ipv4Addr>,
    /// RPC port. The default is 50052, with an automatic private-port fallback.
    #[arg(long)]
    rpc_port: Option<u16>,
    /// CPU helper threads used by the RPC sidecar (defaults to half, capped at 8).
    #[arg(long)]
    rpc_threads: Option<usize>,
    /// Optional llama.cpp backend device, such as CUDA0 or Metal.
    #[arg(long)]
    rpc_device: Option<String>,
    /// Explicit ggml-rpc-server binary (or use INFERNET_LLAMA_RPC_SERVER).
    #[arg(long)]
    rpc_server_binary: Option<PathBuf>,
    /// RPC tensor cache and log directory.
    #[arg(long)]
    rpc_cache_dir: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct ModelFetchArgs {
    #[command(flatten)]
    cache: ModelCacheArgs,
    #[command(flatten)]
    discovery: DiscoveryArgs,
    #[arg(long)]
    model: String,
    #[arg(long)]
    layers: String,
    #[arg(long)]
    checksum: Option<String>,
    #[arg(long)]
    version: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Bootstrap(args) => bootstrap(args).await,
        Command::Serve(args) => serve(args).await,
        Command::Peers(args) => {
            let manifest = ModelManifest::demo();
            let registry = discover_registry(args, &manifest).await?;
            print_peers(&registry);
            Ok(())
        }
        Command::Route(args) => {
            let manifest = manifest_for_model(&args.model)?;
            let registry = discover_registry(args.discovery, &manifest).await?;
            let route = registry.route_for_model(&manifest)?;
            print_route(&route);
            Ok(())
        }
        Command::Infer(args) => infer(args).await,
        Command::Shard(args) => match args.command {
            ShardCommand::Build(args) => build_shard(args),
        },
        Command::Model(args) => match args.command {
            ModelCommand::AddLocal(args) => add_local_model(args),
            ModelCommand::Import(args) => import_model_shard(args),
            ModelCommand::List(args) => list_model_shards(args),
            ModelCommand::Serve(args) => serve_model_distribution(args).await,
            ModelCommand::Fetch(args) => fetch_model_shard(args, false).await,
            ModelCommand::Mirror(args) => fetch_model_shard(args, true).await,
        },
    }
}

async fn bootstrap(args: BootstrapArgs) -> Result<()> {
    let keypair = load_or_generate_keypair(&args.identity_file)?;
    let peer_id = keypair.public().to_peer_id().to_string();
    let tcp_port = tcp_port_from_multiaddr(&args.p2p_listen).unwrap_or("9777");
    let mut discovery = DiscoveryConfig::new(args.topic);
    discovery.keypair = keypair;
    discovery.p2p_listen = args.p2p_listen.clone();
    discovery.advertise_listen_addresses = false;
    discovery.dial_discovered_peers = false;
    discovery.relay_advertisements = true;

    let mut bootstrap_advertisement =
        local_capability_advertisement(peer_id.clone(), String::new());
    if let Some(ip) = args.public_ip.as_deref() {
        bootstrap_advertisement
            .addresses
            .push(format!("/ip4/{ip}/tcp/{tcp_port}/p2p/{peer_id}"));
    }
    if let Some(domain) = args.public_domain.as_deref() {
        bootstrap_advertisement
            .addresses
            .push(format!("/dns4/{domain}/tcp/{tcp_port}/p2p/{peer_id}"));
    }
    discovery.advertisement = Some(bootstrap_advertisement);

    println!("peer_id={peer_id}");
    println!("listen={}", args.p2p_listen);
    println!("model_protocol=/infernet/model/1");
    println!("activation_protocol=/infernet/activation/1");
    println!("identity_file={}", args.identity_file.display());
    println!("cache={}", args.cache_dir.display());
    if let Some(domain) = args.public_domain.as_deref() {
        println!("public_multiaddr=/dns4/{domain}/tcp/{tcp_port}/p2p/{peer_id}");
    }
    if let Some(ip) = args.public_ip.as_deref() {
        println!("public_multiaddr=/ip4/{ip}/tcp/{tcp_port}/p2p/{peer_id}");
    }

    run_model_distribution_node(
        discovery,
        ShardCacheConfig {
            root: args.cache_dir,
            max_storage_bytes: args.max_storage_bytes,
            preferred_models: Vec::new(),
            pinned_models: Vec::new(),
            automatic_cleanup: true,
        },
    )
    .await
}

fn add_local_model(args: ModelAddLocalArgs) -> Result<()> {
    let manifest = manifest_for_model(&args.model)?;
    let release = OfficialModelRelease::infernet_chat_v1_compatibility();
    let test_model_allowed = cfg!(debug_assertions)
        && std::env::var("INFERNET_ALLOW_TEST_MODELS").as_deref() == Ok("1")
        && manifest.runtime_kind == RuntimeKind::Demo;
    if !test_model_allowed
        && (manifest.model_id != release.model_id || args.version != release.version)
    {
        return Err(anyhow!(
            "model add-local is reserved for official release {} version {}",
            release.model_id,
            release.version
        ));
    }
    let verified_source_checksum = if !test_model_allowed {
        let source_size_bytes = fs::metadata(&args.gguf)
            .with_context(|| format!("failed to inspect {}", args.gguf.display()))?
            .len();
        if source_size_bytes != release.expected_total_bytes {
            return Err(anyhow!(
                "official source size mismatch; expected {}, got {}",
                release.expected_total_bytes,
                source_size_bytes
            ));
        }
        let source_checksum = gguf::sha256_file(&args.gguf)?;
        if source_checksum != release.upstream.source_sha256 {
            return Err(anyhow!(
                "official source checksum mismatch; expected {}, got {}",
                release.upstream.source_sha256,
                source_checksum
            ));
        }
        Some(source_checksum)
    } else {
        None
    };
    let cache = ShardCache::new(cache_config(&args.cache))?;
    let summary = if args.consume_source {
        match verified_source_checksum {
            Some(checksum) => import_seed_model_from_file_consuming_verified(
                &cache,
                &args.gguf,
                &manifest,
                args.version,
                checksum,
            )?,
            None => {
                import_seed_model_from_file_consuming(&cache, &args.gguf, &manifest, args.version)?
            }
        }
    } else {
        import_seed_model_from_file(&cache, &args.gguf, &manifest, args.version)?
    };

    println!("model={}", summary.model_id);
    println!("display_name={}", summary.display_name);
    println!("source={}", summary.source_path.display());
    println!("source_checksum={}", summary.source_checksum);
    println!("source_size_bytes={}", summary.source_size_bytes);
    println!("planned_shards={}", summary.shard_count);
    println!("metadata_only={}", summary.metadata_only);

    for record in &summary.records {
        print_model_shard_info(&record.info);
    }

    Ok(())
}

async fn serve(args: ServeArgs) -> Result<()> {
    let manifest = manifest_for_model(&args.model)?;
    let owned_layers = parse_layer_range(&args.layers)?;
    owned_layers.validate_for_model(manifest.layer_count)?;
    let hidden_size = args.hidden_size.unwrap_or(manifest.hidden_size);

    let mut discovery = DiscoveryConfig::new(args.topic);
    discovery.p2p_listen = args.p2p_listen;
    let peer_id = discovery.peer_id().to_string();

    discovery.advertisement = Some(enrich_local_advertisement(shard_descriptor_advertisement(
        peer_id.clone(),
        String::new(),
        &manifest,
        owned_layers,
    )));

    println!("peer_id={peer_id}");
    println!("model={}", manifest.model_id);
    println!("runtime={}", manifest.runtime_kind.as_str());
    println!("layers={}:{}", owned_layers.start, owned_layers.end);
    println!("activation_protocol=/infernet/activation/1");

    run_worker_node(
        discovery,
        WorkerConfig {
            peer_id,
            model_id: manifest.model_id,
            runtime_kind: manifest.runtime_kind,
            owned_layers,
            hidden_size,
            shard_cache: None,
        },
    )
    .await
}

async fn infer(args: InferArgs) -> Result<()> {
    let manifest = manifest_for_model(&args.model)?;
    let mut discovery = DiscoveryConfig::new(args.discovery.topic);
    discovery.p2p_listen = args.discovery.p2p_listen;
    discovery.static_peers = args
        .discovery
        .static_peers
        .iter()
        .map(|peer| parse_static_peer(peer, &manifest))
        .collect::<Result<Vec<_>>>()?;
    let discovery_timeout = Duration::from_millis(args.discovery.discovery_timeout_ms);
    let hidden_size = args.hidden_size.unwrap_or(manifest.hidden_size);
    let result = infer_over_libp2p(
        discovery,
        manifest.clone(),
        args.prompt,
        hidden_size,
        discovery_timeout,
    )
    .await?;
    let response = result.response;

    println!(
        "{}",
        response
            .output_text
            .unwrap_or_else(|| "<missing output>".to_owned())
    );
    println!("model={}", manifest.display_name);
    println!("trace_id={}", response.trace_id);
    println!("route:");

    for (index, hop) in result.route.iter().enumerate() {
        println!(
            "  {} {} layers {}:{} {}",
            index,
            hop.peer_id,
            hop.layers.start,
            hop.layers.end,
            if hop.address.is_empty() {
                "<mdns>"
            } else {
                hop.address.as_str()
            }
        );
    }

    println!("activation_path:");

    for event in response.trace {
        println!(
            "  {} layers {}:{} next {} bytes {} timing_ms {} checksum {:016x}",
            event.peer_id,
            event.layers.start,
            event.layers.end,
            event.next_peer_id.as_deref().unwrap_or("<final>"),
            event.activation_size_bytes,
            event.timing_ms,
            event.activation_checksum
        );
    }

    Ok(())
}

async fn discover_registry(args: DiscoveryArgs, manifest: &ModelManifest) -> Result<ShardRegistry> {
    let mut discovery = DiscoveryConfig::new(args.topic);
    discovery.p2p_listen = args.p2p_listen;
    discovery.static_peers = args
        .static_peers
        .iter()
        .map(|peer| parse_static_peer(peer, manifest))
        .collect::<Result<Vec<_>>>()?;

    discover_for(discovery, Duration::from_millis(args.discovery_timeout_ms)).await
}

fn manifest_for_model(model: &str) -> Result<ModelManifest> {
    ModelManifest::by_id(model).ok_or_else(|| {
        let supported = ModelManifest::catalog()
            .into_iter()
            .map(|manifest| manifest.model_id)
            .collect::<Vec<_>>()
            .join(", ");
        anyhow!("unknown model {model}; supported models are {supported}")
    })
}

fn parse_layer_range(input: &str) -> Result<LayerRange> {
    let separator = if input.contains(':') { ':' } else { '-' };
    let (start, end) = input
        .split_once(separator)
        .ok_or_else(|| anyhow!("layer range must look like start:end"))?;
    let start = start
        .parse::<u32>()
        .with_context(|| format!("invalid range start {start}"))?;
    let end = end
        .parse::<u32>()
        .with_context(|| format!("invalid range end {end}"))?;

    LayerRange::new(start, end).map_err(Into::into)
}

fn tcp_port_from_multiaddr(address: &str) -> Option<&str> {
    let mut parts = address.split('/');
    while let Some(part) = parts.next() {
        if part == "tcp" {
            return parts.next();
        }
    }
    None
}

fn parse_static_peer(input: &str, manifest: &ModelManifest) -> Result<NodeAdvertisement> {
    let (peer, rest) = input
        .split_once('@')
        .ok_or_else(|| anyhow!("static peer must look like peer@multiaddr#start:end"))?;
    let (address, layers) = rest
        .rsplit_once('#')
        .or_else(|| rest.rsplit_once('/'))
        .ok_or_else(|| anyhow!("static peer must include #start:end"))?;
    let layers = parse_layer_range(layers)?;

    Ok(shard_descriptor_advertisement(
        peer.to_owned(),
        address.to_owned(),
        manifest,
        layers,
    ))
}

fn shard_descriptor_advertisement(
    peer_id: String,
    address: String,
    manifest: &ModelManifest,
    layers: LayerRange,
) -> NodeAdvertisement {
    let mut advertisement = empty_advertisement(peer_id, address);
    advertisement
        .hosted_shards
        .push(ShardDescriptor::for_manifest(manifest, layers));
    advertisement
}

fn build_shard(args: ShardBuildArgs) -> Result<()> {
    let manifest = manifest_for_model(&args.model)?;
    if manifest.runtime_kind != RuntimeKind::LlamaCpp {
        return Err(anyhow!(
            "shard build currently targets GGUF/llama.cpp models, got {}",
            manifest.runtime_kind.as_str()
        ));
    }

    let layers = parse_layer_range(&args.layers)?;
    layers.validate_for_model(manifest.layer_count)?;

    let info = gguf::parse_gguf_info(&args.gguf)?;
    validate_gguf_compatibility(&manifest, &info)?;
    let source_checksum = gguf::sha256_file(&args.gguf)?;
    let file_size_bytes = fs::metadata(&args.gguf)
        .with_context(|| format!("failed to inspect {}", args.gguf.display()))?
        .len();
    let (required_tensors, boundary_tensors) =
        select_shard_tensors(&info.tensor_names, layers, manifest.layer_count);

    if required_tensors.is_empty() {
        return Err(anyhow!(
            "no layer tensors found for {}:{} in {}",
            layers.start,
            layers.end,
            args.gguf.display()
        ));
    }

    let tokenizer = TokenizerCompatibility {
        family: info
            .tokenizer_family
            .clone()
            .unwrap_or_else(|| manifest.architecture.clone()),
        checksum: Some(info.tokenizer_checksum.clone()),
    };
    let metadata = ShardMetadata {
        architecture: manifest.architecture.clone(),
        quantization: info.quantization.clone(),
        source_checksum: Some(source_checksum.clone()),
        protocol_version: 1,
    };
    let shard_hash = shard_manifest_hash(
        &manifest.model_id,
        layers,
        &source_checksum,
        &required_tensors,
        &boundary_tensors,
    );
    let shard_manifest = GgufShardManifest {
        model_id: manifest.model_id.clone(),
        display_name: manifest.display_name.clone(),
        architecture: manifest.architecture.clone(),
        layer_count: manifest.layer_count,
        hidden_size: manifest.hidden_size,
        activation_dtype: manifest.activation_dtype.clone(),
        runtime_kind: manifest.runtime_kind,
        layers,
        tokenizer,
        metadata,
        source: GgufSourceMetadata {
            path: args.gguf.display().to_string(),
            checksum_sha256: source_checksum,
            gguf_version: info.version,
            metadata_kv_count: info.metadata_kv_count,
            tensor_count: info.tensor_count,
            file_size_bytes,
        },
        required_tensors,
        boundary_tensors,
        shard_hash,
    };

    let json = serde_json::to_string_pretty(&shard_manifest)?;
    fs::write(&args.out, format!("{json}\n"))
        .with_context(|| format!("failed to write {}", args.out.display()))?;

    println!("wrote={}", args.out.display());
    println!(
        "model={} layers={}:{} required_tensors={} boundary_tensors={} shard_hash={}",
        shard_manifest.model_id,
        shard_manifest.layers.start,
        shard_manifest.layers.end,
        shard_manifest.required_tensors.len(),
        shard_manifest.boundary_tensors.len(),
        shard_manifest.shard_hash
    );

    Ok(())
}

fn import_model_shard(args: ModelImportArgs) -> Result<()> {
    let layers = parse_layer_range(&args.layers)?;
    let cache = ShardCache::new(cache_config(&args.cache))?;
    let record = cache.import_file(&args.file, args.model, layers, args.version)?;

    println!("imported={}", record.path.display());
    print_model_shard_info(&record.info);
    Ok(())
}

fn list_model_shards(args: ModelCacheArgs) -> Result<()> {
    let cache = ShardCache::new(cache_config(&args))?;
    let stats = cache.stats()?;
    println!("cache={}", stats.root.display());
    println!(
        "storage_used_bytes={} max_storage_bytes={} shard_count={}",
        stats.storage_used_bytes, stats.max_storage_bytes, stats.shard_count
    );

    for record in cache.list()? {
        print_model_shard_info(&record.info);
    }

    Ok(())
}

async fn serve_model_distribution(args: ModelServeArgs) -> Result<()> {
    // Install the signal listener before spawning the child. Otherwise a Ctrl-C
    // arriving during sidecar readiness could terminate this process before
    // Rust gets a chance to drop and reap the child.
    let mut shutdown_receiver = tokio::spawn(wait_for_shutdown_signal());
    tokio::task::yield_now().await;

    let mut rpc_server = start_worker_rpc_server(&args)?;
    let mut discovery = DiscoveryConfig::new(args.topic);
    discovery.p2p_listen = args.p2p_listen;
    let peer_id = discovery.peer_id().to_string();
    discovery.advertisement = Some(local_capability_advertisement(
        peer_id.clone(),
        String::new(),
    ));
    println!("peer_id={peer_id}");
    println!("model_protocol=/infernet/model/1");
    println!("cache={}", args.cache.cache_dir.display());
    if let Some(server) = rpc_server.as_ref() {
        println!("llama_rpc={}", server.server.endpoint());
        println!("llama_rpc_security=private-network-only");
    } else {
        println!("llama_rpc=disabled");
    }

    let mut service = Box::pin(run_model_distribution_node(
        discovery,
        cache_config(&args.cache),
    ));
    let mut rpc_health_check = tokio::time::interval(Duration::from_secs(1));
    let result = loop {
        tokio::select! {
            result = &mut service => break result,
            signal = &mut shutdown_receiver => {
                signal
                    .context("shutdown signal listener stopped")?
                    .context("failed to install Ctrl-C handler")?;
                break Ok(());
            }
            _ = rpc_health_check.tick(), if rpc_server.is_some() => {
                let is_running = rpc_server
                    .as_mut()
                    .is_some_and(|server| server.server.is_running());
                let active = rpc_server
                    .as_ref()
                    .is_some_and(|server| server.server.has_active_client());
                set_local_rpc_active(active);
                if !is_running {
                    clear_local_llama_rpc_endpoint();
                    break Err(anyhow!(
                        "llama.cpp RPC sidecar stopped unexpectedly; compute advertisement withdrawn"
                    ));
                }
            }
        }
    };
    stop_persistent_llama_server();
    set_local_inference_active(false);
    set_local_rpc_active(false);
    drop(rpc_server);
    result
}

async fn wait_for_shutdown_signal() -> std::io::Result<()> {
    #[cfg(unix)]
    {
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
        tokio::select! {
            result = tokio::signal::ctrl_c() => result,
            _ = terminate.recv() => Ok(()),
        }
    }
    #[cfg(not(unix))]
    {
        tokio::signal::ctrl_c().await
    }
}

struct AdvertisedWorkerRpcServer {
    server: LlamaRpcServer,
}

impl Drop for AdvertisedWorkerRpcServer {
    fn drop(&mut self) {
        clear_local_llama_rpc_endpoint();
    }
}

fn start_worker_rpc_server(args: &ModelServeArgs) -> Result<Option<AdvertisedWorkerRpcServer>> {
    if args.no_rpc || env_flag("INFERNET_DISABLE_LLAMA_RPC") {
        clear_local_llama_rpc_endpoint();
        return Ok(None);
    }

    clear_local_llama_rpc_endpoint();
    let binary = match args.rpc_server_binary.clone() {
        Some(binary) if binary.is_file() => binary,
        Some(binary) => {
            return Err(anyhow!(
                "llama.cpp RPC server binary is missing: {}",
                binary.display()
            ));
        }
        None => find_llama_rpc_server_binary().ok_or_else(|| {
            anyhow!(
                "ggml-rpc-server was not found; set --rpc-server-binary or INFERNET_LLAMA_RPC_SERVER"
            )
        })?,
    };
    let host = configured_worker_rpc_host(args.rpc_host)?;
    let requested_port = args.rpc_port.or(configured_env_port()?);
    let port = available_worker_rpc_port(host, requested_port)?;
    let threads = configured_worker_rpc_threads(args.rpc_threads)?;
    let device = args
        .rpc_device
        .clone()
        .or_else(|| env::var("INFERNET_LLAMA_RPC_DEVICE").ok())
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty());
    let cache_dir = args
        .rpc_cache_dir
        .clone()
        .or_else(|| env::var_os("INFERNET_LLAMA_RPC_CACHE_DIR").map(PathBuf::from))
        .unwrap_or_else(|| args.cache.cache_dir.join("rpc-runtime"));
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
    };
    let server = spawn_llama_rpc_server(LlamaRpcServerConfig {
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
    })?;
    if let Err(error) = set_local_llama_rpc_endpoint(Some(endpoint)) {
        drop(server);
        clear_local_llama_rpc_endpoint();
        return Err(anyhow!(error));
    }

    Ok(Some(AdvertisedWorkerRpcServer { server }))
}

fn configured_worker_rpc_host(explicit: Option<Ipv4Addr>) -> Result<Ipv4Addr> {
    let host = if let Some(host) = explicit {
        host
    } else if let Ok(value) = env::var("INFERNET_LLAMA_RPC_HOST") {
        value
            .trim()
            .parse::<Ipv4Addr>()
            .context("INFERNET_LLAMA_RPC_HOST must be a private IPv4 address")?
    } else {
        preferred_private_ipv4().ok_or_else(|| {
            anyhow!(
                "no private LAN/Tailscale IPv4 address was detected; pass --rpc-host explicitly"
            )
        })?
    };

    if is_private_or_tailscale_ipv4(host) {
        Ok(host)
    } else {
        Err(anyhow!(
            "refusing to expose unauthenticated llama.cpp RPC on non-private address {host}"
        ))
    }
}

fn is_private_or_tailscale_ipv4(host: Ipv4Addr) -> bool {
    let [first, second, _, _] = host.octets();
    first == 10
        || (first == 172 && (16..=31).contains(&second))
        || (first == 192 && second == 168)
        || (first == 100 && (64..=127).contains(&second))
}

fn preferred_private_ipv4() -> Option<Ipv4Addr> {
    let socket = UdpSocket::bind("0.0.0.0:0").ok()?;
    socket.connect("8.8.8.8:80").ok()?;
    match socket.local_addr().ok()?.ip() {
        IpAddr::V4(host) if is_private_or_tailscale_ipv4(host) => Some(host),
        _ => None,
    }
}

fn configured_env_port() -> Result<Option<u16>> {
    let Ok(value) = env::var("INFERNET_LLAMA_RPC_PORT") else {
        return Ok(None);
    };
    let port = value
        .trim()
        .parse::<u16>()
        .context("INFERNET_LLAMA_RPC_PORT must be between 1 and 65535")?;
    if port == 0 {
        return Err(anyhow!(
            "INFERNET_LLAMA_RPC_PORT must be between 1 and 65535"
        ));
    }
    Ok(Some(port))
}

fn available_worker_rpc_port(host: Ipv4Addr, requested: Option<u16>) -> Result<u16> {
    let preferred = requested.unwrap_or(LLAMA_RPC_DEFAULT_PORT);
    match TcpListener::bind((host, preferred)) {
        Ok(listener) => {
            drop(listener);
            Ok(preferred)
        }
        Err(error) if requested.is_some() => Err(anyhow!(
            "configured llama.cpp RPC address {host}:{preferred} is unavailable: {error}"
        )),
        Err(_) => {
            let listener = TcpListener::bind((host, 0)).with_context(|| {
                format!("could not allocate a private llama.cpp RPC port on {host}")
            })?;
            let port = listener
                .local_addr()
                .context("could not inspect allocated llama.cpp RPC port")?
                .port();
            drop(listener);
            Ok(port)
        }
    }
}

fn configured_worker_rpc_threads(explicit: Option<usize>) -> Result<usize> {
    if let Some(threads) = explicit {
        if threads == 0 {
            return Err(anyhow!("--rpc-threads must be a positive integer"));
        }
        return Ok(threads);
    }
    if let Ok(value) = env::var("INFERNET_LLAMA_RPC_THREADS") {
        return value
            .trim()
            .parse::<usize>()
            .ok()
            .filter(|threads| *threads > 0)
            .ok_or_else(|| anyhow!("INFERNET_LLAMA_RPC_THREADS must be a positive integer"));
    }

    Ok(std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(2)
        .div_ceil(2)
        .clamp(1, 8))
}

fn rpc_backend_for_device(detected: &str, device: Option<&str>) -> Result<String> {
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
                return Err(anyhow!(
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
        Err(anyhow!(
            "distributed inference requires a CUDA or Metal backend, got {backend}"
        ))
    }
}

fn env_flag(name: &str) -> bool {
    env::var(name).is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

async fn fetch_model_shard(args: ModelFetchArgs, mirror_after_download: bool) -> Result<()> {
    let layers = parse_layer_range(&args.layers)?;
    let mut discovery = DiscoveryConfig::new(args.discovery.topic.clone());
    discovery.p2p_listen = args.discovery.p2p_listen.clone();
    discovery.static_peers = args
        .discovery
        .static_peers
        .iter()
        .map(|peer| parse_static_model_peer(peer))
        .collect::<Result<Vec<_>>>()?;
    discovery.advertisement = Some(local_capability_advertisement(
        discovery.peer_id().to_string(),
        String::new(),
    ));
    let serving_discovery = discovery.clone();
    let cache_config = cache_config(&args.cache);
    let result = fetch_model_shard_over_libp2p(
        discovery,
        cache_config.clone(),
        args.model,
        layers,
        args.checksum,
        args.version,
        Duration::from_millis(args.discovery.discovery_timeout_ms),
    )
    .await?;

    println!("downloaded_from={}", result.source_peer_id);
    println!("stored={}", result.cache_record.path.display());
    print_model_shard_info(&result.shard);

    if mirror_after_download {
        println!("peer_id={}", serving_discovery.peer_id());
        println!("mirroring=true");
        run_model_distribution_node(serving_discovery, cache_config).await?;
    }

    Ok(())
}

fn parse_static_model_peer(input: &str) -> Result<NodeAdvertisement> {
    let (peer, rest) = input.split_once('@').ok_or_else(|| {
        anyhow!(
            "static model peer must look like peer@multiaddr#model:start:end:checksum:size:version"
        )
    })?;
    let (address, descriptor) = rest.rsplit_once('#').ok_or_else(|| {
        anyhow!("static model peer must include #model:start:end:checksum:size:version")
    })?;
    let parts = descriptor.split(':').collect::<Vec<_>>();
    if parts.len() != 6 {
        return Err(anyhow!(
            "static model peer descriptor must be model:start:end:checksum:size:version"
        ));
    }

    let start = parts[1]
        .parse::<u32>()
        .with_context(|| format!("invalid static model peer layer start {}", parts[1]))?;
    let end = parts[2]
        .parse::<u32>()
        .with_context(|| format!("invalid static model peer layer end {}", parts[2]))?;
    let size_bytes = parts[4]
        .parse::<u64>()
        .with_context(|| format!("invalid static model peer size {}", parts[4]))?;
    let mut advertisement = empty_advertisement(peer.to_owned(), address.to_owned());
    advertisement.model_shards.push(ModelShardInfo {
        model_id: parts[0].to_owned(),
        layers: LayerRange::new(start, end)?,
        checksum: parts[3].to_owned(),
        size_bytes,
        version: parts[5].to_owned(),
        protocol_version: 1,
    });

    Ok(advertisement)
}

fn cache_config(args: &ModelCacheArgs) -> ShardCacheConfig {
    ShardCacheConfig {
        root: args.cache_dir.clone(),
        max_storage_bytes: args.max_storage_bytes,
        preferred_models: args.preferred_models.clone(),
        pinned_models: args.pinned_models.clone(),
        automatic_cleanup: !args.no_auto_cleanup,
    }
}

fn print_model_shard_info(info: &ModelShardInfo) {
    println!(
        "model_shard model={} layers={}:{} checksum={} size={} version={} protocol={}",
        info.model_id,
        info.layers.start,
        info.layers.end,
        info.checksum,
        info.size_bytes,
        info.version,
        info.protocol_version
    );
}

fn validate_gguf_compatibility(manifest: &ModelManifest, info: &gguf::GgufInfo) -> Result<()> {
    if let Some(architecture) = &info.architecture {
        if architecture != &manifest.architecture {
            return Err(anyhow!(
                "GGUF architecture {architecture} does not match model {} architecture {}",
                manifest.model_id,
                manifest.architecture
            ));
        }
    }

    if let Some(layer_count) = info.layer_count {
        if layer_count != manifest.layer_count {
            return Err(anyhow!(
                "GGUF layer count {layer_count} does not match model {} layer count {}",
                manifest.model_id,
                manifest.layer_count
            ));
        }
    }

    if let Some(hidden_size) = info.hidden_size {
        if hidden_size != manifest.hidden_size {
            return Err(anyhow!(
                "GGUF hidden size {hidden_size} does not match model {} hidden size {}",
                manifest.model_id,
                manifest.hidden_size
            ));
        }
    }

    Ok(())
}

fn select_shard_tensors(
    tensor_names: &[String],
    layers: LayerRange,
    layer_count: u32,
) -> (Vec<String>, Vec<String>) {
    let mut required_tensors = Vec::new();
    let mut boundary_tensors = Vec::new();

    for name in tensor_names {
        if let Some(layer) = layer_index_from_tensor_name(name) {
            if layers.start <= layer && layer < layers.end {
                required_tensors.push(name.clone());
            }
            continue;
        }

        let is_token_embedding = name.contains("token_embd");
        let is_output = name.starts_with("output") || name.contains("output_norm");
        let is_shared_global = !is_token_embedding && !is_output;

        if is_shared_global
            || (layers.start == 0 && is_token_embedding)
            || (layers.end == layer_count && is_output)
        {
            boundary_tensors.push(name.clone());
        }
    }

    (required_tensors, boundary_tensors)
}

fn layer_index_from_tensor_name(name: &str) -> Option<u32> {
    let rest = name.strip_prefix("blk.")?;
    let (index, _) = rest.split_once('.')?;
    index.parse().ok()
}

fn shard_manifest_hash(
    model_id: &str,
    layers: LayerRange,
    source_checksum: &str,
    required_tensors: &[String],
    boundary_tensors: &[String],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(model_id.as_bytes());
    hasher.update(layers.start.to_le_bytes());
    hasher.update(layers.end.to_le_bytes());
    hasher.update(source_checksum.as_bytes());

    for tensor in required_tensors.iter().chain(boundary_tensors) {
        hasher.update([0]);
        hasher.update(tensor.as_bytes());
    }

    hex_lower(&hasher.finalize())
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);

    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }

    output
}

fn print_peers(registry: &ShardRegistry) {
    for advertisement in registry.advertisements() {
        if let Some(capabilities) = advertisement.capabilities.as_ref() {
            println!(
                "machine peer={} machine_id={} backend={} device={} free_accelerator_bytes={} active_sessions={}/{} rpc={}",
                advertisement.peer_id,
                capabilities.machine_id.as_deref().unwrap_or("unknown"),
                capabilities.compute_backend,
                capabilities.device_name,
                capabilities.available_accelerator_memory_bytes,
                capabilities.active_sessions,
                capabilities.max_sessions,
                capabilities
                    .llama_rpc
                    .as_ref()
                    .filter(|endpoint| endpoint.ready)
                    .map(LlamaRpcEndpoint::llama_cpp_endpoint)
                    .unwrap_or_else(|| "not-ready".to_owned()),
            );
        }
        for shard in advertisement.hosted_shards {
            println!(
                "hosted_shard peer={} model={} layers={}:{} runtime={} address={}",
                advertisement.peer_id,
                shard.model_id,
                shard.layers.start,
                shard.layers.end,
                shard.runtime_kind.as_str(),
                advertisement
                    .addresses
                    .first()
                    .map(String::as_str)
                    .unwrap_or("<no-address>")
            );
        }
        for shard in advertisement.model_shards {
            println!(
                "model_shard peer={} model={} layers={}:{} checksum={} size={} version={} address={}",
                advertisement.peer_id,
                shard.model_id,
                shard.layers.start,
                shard.layers.end,
                shard.checksum,
                shard.size_bytes,
                shard.version,
                advertisement
                    .addresses
                    .first()
                    .map(String::as_str)
                    .unwrap_or("<no-address>")
            );
        }
    }
}

fn print_route(route: &[RouteHop]) {
    for (index, hop) in route.iter().enumerate() {
        println!(
            "{} {} {}:{} {}",
            index, hop.peer_id, hop.layers.start, hop.layers.end, hop.address
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rpc_binding_accepts_only_private_and_tailscale_ipv4() {
        for host in [
            Ipv4Addr::new(10, 0, 0, 1),
            Ipv4Addr::new(172, 16, 0, 1),
            Ipv4Addr::new(192, 168, 1, 10),
            Ipv4Addr::new(100, 64, 0, 1),
            Ipv4Addr::new(100, 127, 255, 254),
        ] {
            assert!(is_private_or_tailscale_ipv4(host), "{host}");
        }

        for host in [
            Ipv4Addr::UNSPECIFIED,
            Ipv4Addr::LOCALHOST,
            Ipv4Addr::new(8, 8, 8, 8),
            Ipv4Addr::new(100, 128, 0, 1),
            Ipv4Addr::new(217, 77, 11, 197),
        ] {
            assert!(!is_private_or_tailscale_ipv4(host), "{host}");
        }
    }

    #[test]
    fn explicit_rpc_device_controls_advertised_backend() {
        assert!(rpc_backend_for_device("metal", Some("CPU")).is_err());
        assert_eq!(
            rpc_backend_for_device("cpu", Some("CUDA0,CPU")).unwrap(),
            "cuda"
        );
        assert!(rpc_backend_for_device("cpu", Some("Vulkan0")).is_err());
    }

    #[test]
    fn static_peer_descriptor_does_not_claim_local_capabilities() {
        let advertisement = parse_static_peer(
            "remote-peer@/ip4/127.0.0.1/tcp/7001#0:3",
            &ModelManifest::demo(),
        )
        .unwrap();

        assert!(advertisement.capabilities.is_none());
        assert!(advertisement.available_ram_bytes.is_none());
        assert!(advertisement.available_vram_bytes.is_none());
        assert_eq!(advertisement.hosted_shards.len(), 1);
        assert_eq!(
            advertisement.hosted_shards[0].layers,
            LayerRange::new(0, 3).unwrap()
        );
    }
}
