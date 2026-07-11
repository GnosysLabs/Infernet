use infernet_model::{LayerRange, SeedShardManifest, ShardDescriptor};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub const PROTOCOL_VERSION: u32 = 1;
/// A distributed Infernet job uses at least two physical machines. The only
/// one-machine exception is execution entirely on the requester's own machine.
pub const MIN_DISTRIBUTED_MACHINE_COUNT: usize = 2;
pub const ACTIVATION_PROTOCOL: &str = "/infernet/activation/2";
pub const MODEL_PROTOCOL: &str = "/infernet/model/1";
pub const MODEL_BLOB_PROTOCOL: &str = "/infernet/model-blob/1";
pub const LLAMA_RPC_TUNNEL_PROTOCOL: &str = "/infernet/llama-rpc-tunnel/1";
pub const IMAGE_RPC_TUNNEL_PROTOCOL: &str = "/infernet/image-rpc-tunnel/1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelShardInfo {
    pub model_id: String,
    pub layers: LayerRange,
    pub checksum: String,
    pub size_bytes: u64,
    pub version: String,
    pub protocol_version: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageComponentRole {
    DiffusionTransformer,
    TextEncoder,
    Vae,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelComponentInfo {
    pub release_id: String,
    pub model_id: String,
    pub component_id: String,
    pub role: ImageComponentRole,
    pub checksum: String,
    pub size_bytes: u64,
    pub version: String,
    pub runtime_abi: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelBlobRequest {
    pub protocol_version: u32,
    pub request_id: Uuid,
    pub model_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layers: Option<LayerRange>,
    pub source_checksum: String,
    pub offset: u64,
    pub max_bytes: u32,
}

impl ModelBlobRequest {
    pub fn new(
        model_id: impl Into<String>,
        source_checksum: impl Into<String>,
        offset: u64,
        max_bytes: u32,
    ) -> Self {
        Self {
            protocol_version: PROTOCOL_VERSION,
            request_id: Uuid::new_v4(),
            model_id: model_id.into(),
            layers: None,
            source_checksum: source_checksum.into(),
            offset,
            max_bytes,
        }
    }

    pub fn new_shard(
        model_id: impl Into<String>,
        layers: LayerRange,
        shard_checksum: impl Into<String>,
        offset: u64,
        max_bytes: u32,
    ) -> Self {
        Self {
            protocol_version: PROTOCOL_VERSION,
            request_id: Uuid::new_v4(),
            model_id: model_id.into(),
            layers: Some(layers),
            source_checksum: shard_checksum.into(),
            offset,
            max_bytes,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelBlobResponse {
    pub protocol_version: u32,
    pub request_id: Uuid,
    pub peer_id: String,
    pub model_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layers: Option<LayerRange>,
    pub source_checksum: String,
    pub offset: u64,
    pub total_size_bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed_manifest: Option<SeedShardManifest>,
    pub payload: Vec<u8>,
    pub error: Option<String>,
}

impl ModelBlobResponse {
    pub fn success(
        request: &ModelBlobRequest,
        peer_id: impl Into<String>,
        total_size_bytes: u64,
        payload: Vec<u8>,
    ) -> Self {
        Self {
            protocol_version: PROTOCOL_VERSION,
            request_id: request.request_id,
            peer_id: peer_id.into(),
            model_id: request.model_id.clone(),
            layers: request.layers,
            source_checksum: request.source_checksum.clone(),
            offset: request.offset,
            total_size_bytes,
            seed_manifest: None,
            payload,
            error: None,
        }
    }

    pub fn failure(
        request: &ModelBlobRequest,
        peer_id: impl Into<String>,
        error: impl Into<String>,
    ) -> Self {
        Self {
            protocol_version: PROTOCOL_VERSION,
            request_id: request.request_id,
            peer_id: peer_id.into(),
            model_id: request.model_id.clone(),
            layers: request.layers,
            source_checksum: request.source_checksum.clone(),
            offset: request.offset,
            total_size_bytes: 0,
            seed_manifest: None,
            payload: Vec::new(),
            error: Some(error.into()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlamaRpcEndpoint {
    /// Process-local host accepted by llama.cpp's `--rpc host:port`
    /// convention. It is deliberately never advertised over the network.
    #[serde(default, skip_serializing)]
    pub host: String,
    /// Process-local port. It is deliberately never advertised over the
    /// network; strangers connect through the authenticated libp2p tunnel.
    #[serde(default, skip_serializing)]
    pub port: u16,
    /// ggml RPC wire protocol implemented by the endpoint.
    pub rpc_protocol_version: String,
    /// Infernet runtime/package ABI accepted by this worker.
    pub runtime_abi: String,
    /// Backend actually exposed by the running RPC server (cuda/metal/cpu).
    pub backend: String,
    /// True only after the configured RPC backend has become reachable.
    pub ready: bool,
    /// Authenticated stream protocol exposed by this worker. New launch nodes
    /// must advertise this instead of a raw TCP endpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tunnel_protocol: Option<String>,
}

impl LlamaRpcEndpoint {
    pub fn llama_cpp_endpoint(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeCapabilities {
    pub os: String,
    pub arch: String,
    pub compute_backend: String,
    pub device_name: String,
    /// Stable, privacy-preserving hash of the physical host identity. Used to
    /// avoid counting two app identities/interfaces as two computers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub machine_id: Option<String>,
    pub logical_cpu_cores: u32,
    pub total_ram_bytes: u64,
    pub available_ram_bytes: u64,
    pub total_accelerator_memory_bytes: u64,
    pub available_accelerator_memory_bytes: u64,
    /// User-selected ceiling for accelerator or unified memory offered to
    /// network work. `None` keeps the automatic all-available behavior.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vram_contribution_limit_bytes: Option<u64>,
    pub unified_memory: bool,
    pub max_sessions: u32,
    pub active_sessions: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measured_prefill_tokens_per_second: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measured_decode_tokens_per_second: Option<f32>,
    pub queue_depth: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llama_rpc: Option<LlamaRpcEndpoint>,
    /// Exact stable-diffusion.cpp GGML worker used only for image DiT blocks.
    /// It is separate from llama.cpp because equal wire versions do not imply
    /// an equal in-memory tensor ABI.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_rpc: Option<LlamaRpcEndpoint>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeAdvertisement {
    pub protocol_version: u32,
    pub peer_id: String,
    pub addresses: Vec<String>,
    pub available_ram_bytes: Option<u64>,
    pub available_vram_bytes: Option<u64>,
    pub latency_hint_ms: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<NodeCapabilities>,
    pub hosted_shards: Vec<ShardDescriptor>,
    #[serde(default)]
    pub model_shards: Vec<ModelShardInfo>,
    #[serde(default)]
    pub model_components: Vec<ModelComponentInfo>,
    /// Short-lived, relay-signed display metadata. It is never consulted for
    /// routing, placement, eligibility, or machine identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coarse_location: Option<CoarseLocationAssertion>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoarseLocationAssertion {
    pub version: u32,
    pub subject_peer_id: String,
    pub latitude_e4: i32,
    pub longitude_e4: i32,
    pub region: String,
    pub country: String,
    pub issued_at_ms: u64,
    pub expires_at_ms: u64,
    pub relay_peer_id: String,
    pub relay_public_key: Vec<u8>,
    pub signature: Vec<u8>,
}

impl CoarseLocationAssertion {
    pub fn signing_bytes(&self) -> Vec<u8> {
        format!(
            "infernet-coarse-location-v1\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}",
            self.subject_peer_id,
            self.latitude_e4,
            self.longitude_e4,
            self.region,
            self.country,
            self.issued_at_ms,
            self.expires_at_ms,
            self.relay_peer_id,
        )
        .into_bytes()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteHop {
    pub peer_id: String,
    /// Stable physical-machine identity for placement validation. Multiple
    /// peer/process identities on one computer share this value.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub machine_id: String,
    pub address: String,
    pub layers: LayerRange,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraceEvent {
    pub peer_id: String,
    pub layers: LayerRange,
    pub next_peer_id: Option<String>,
    pub activation_size_bytes: usize,
    pub activation_checksum: u64,
    pub timing_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromptMetadata {
    pub prompt: String,
    pub demo_mode: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rpc_endpoints: Vec<String>,
    /// Exact authenticated worker identities selected for llama.cpp RPC.
    /// The coordinator converts these to process-local loopback proxies; raw
    /// worker host/port pairs never cross the network.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rpc_worker_peer_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActivationRequest {
    pub protocol_version: u32,
    pub trace_id: Uuid,
    /// Physical machine that originated the request. This distinguishes the
    /// sole-requester local exception from an invalid sole-remote route.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin_machine_id: Option<String>,
    pub model_id: String,
    pub route: Vec<RouteHop>,
    pub current_hop_index: usize,
    pub hidden_size: usize,
    pub sequence_position: u32,
    /// The token sampled by the final worker on the previous pipeline pass.
    /// `None` marks the initial prompt-prefill pass.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_token_id: Option<i32>,
    pub activation: Vec<f32>,
    pub prompt: Option<PromptMetadata>,
    pub trace: Vec<TraceEvent>,
}

impl ActivationRequest {
    pub fn new(
        model_id: impl Into<String>,
        route: Vec<RouteHop>,
        hidden_size: usize,
        activation: Vec<f32>,
        prompt: Option<PromptMetadata>,
    ) -> Self {
        Self {
            protocol_version: PROTOCOL_VERSION,
            trace_id: Uuid::new_v4(),
            origin_machine_id: None,
            model_id: model_id.into(),
            route,
            current_hop_index: 0,
            hidden_size,
            sequence_position: 0,
            input_token_id: None,
            activation,
            prompt,
            trace: Vec::new(),
        }
    }

    pub fn current_hop(&self) -> Option<&RouteHop> {
        self.route.get(self.current_hop_index)
    }

    pub fn next_hop(&self) -> Option<&RouteHop> {
        self.route.get(self.current_hop_index + 1)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActivationResponse {
    pub protocol_version: u32,
    pub trace_id: Uuid,
    pub peer_id: String,
    pub processed_layer_start: u32,
    pub processed_layer_end: u32,
    pub output_activation: Vec<f32>,
    pub timing_ms: u64,
    pub trace: Vec<TraceEvent>,
    pub output_text: Option<String>,
    /// Token selected by the final layer worker. The client sends this token
    /// back through the same route while every worker keeps its KV cache hot.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampled_token_id: Option<i32>,
    #[serde(default)]
    pub generation_complete: bool,
    #[serde(default)]
    pub next_sequence_position: u32,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelShardRequest {
    pub protocol_version: u32,
    pub request_id: Uuid,
    pub model_id: String,
    pub layers: LayerRange,
    pub checksum: Option<String>,
    pub version: Option<String>,
}

impl ModelShardRequest {
    pub fn new(
        model_id: impl Into<String>,
        layers: LayerRange,
        checksum: Option<String>,
        version: Option<String>,
    ) -> Self {
        Self {
            protocol_version: PROTOCOL_VERSION,
            request_id: Uuid::new_v4(),
            model_id: model_id.into(),
            layers,
            checksum,
            version,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelShardResponse {
    pub protocol_version: u32,
    pub request_id: Uuid,
    pub peer_id: String,
    pub shard: Option<ModelShardInfo>,
    pub payload: Vec<u8>,
    pub error: Option<String>,
}

impl ModelShardResponse {
    pub fn success(
        request: &ModelShardRequest,
        peer_id: impl Into<String>,
        shard: ModelShardInfo,
        payload: Vec<u8>,
    ) -> Self {
        Self {
            protocol_version: PROTOCOL_VERSION,
            request_id: request.request_id,
            peer_id: peer_id.into(),
            shard: Some(shard),
            payload,
            error: None,
        }
    }

    pub fn failure(
        request: &ModelShardRequest,
        peer_id: impl Into<String>,
        error: impl Into<String>,
    ) -> Self {
        Self {
            protocol_version: PROTOCOL_VERSION,
            request_id: request.request_id,
            peer_id: peer_id.into(),
            shard: None,
            payload: Vec::new(),
            error: Some(error.into()),
        }
    }
}

impl ActivationResponse {
    pub fn success(
        request: ActivationRequest,
        peer_id: impl Into<String>,
        output_text: Option<String>,
        timing_ms: u64,
    ) -> Self {
        let layers = request
            .current_hop()
            .map(|hop| hop.layers)
            .unwrap_or(LayerRange { start: 0, end: 0 });

        Self {
            protocol_version: PROTOCOL_VERSION,
            trace_id: request.trace_id,
            peer_id: peer_id.into(),
            processed_layer_start: layers.start,
            processed_layer_end: layers.end,
            output_activation: request.activation,
            timing_ms,
            trace: request.trace,
            output_text,
            sampled_token_id: None,
            generation_complete: false,
            next_sequence_position: 0,
            error: None,
        }
    }

    pub fn failure(
        trace_id: Uuid,
        peer_id: impl Into<String>,
        error: impl Into<String>,
        trace: Vec<TraceEvent>,
    ) -> Self {
        Self {
            protocol_version: PROTOCOL_VERSION,
            trace_id,
            peer_id: peer_id.into(),
            processed_layer_start: 0,
            processed_layer_end: 0,
            output_activation: Vec::new(),
            timing_ms: 0,
            trace,
            output_text: None,
            sampled_token_id: None,
            generation_complete: false,
            next_sequence_position: 0,
            error: Some(error.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activation_request_response_roundtrip_json() {
        let route = vec![RouteHop {
            peer_id: "peer-a".to_owned(),
            machine_id: "machine-a".to_owned(),
            address: "/ip4/127.0.0.1/tcp/10000".to_owned(),
            layers: LayerRange::new(0, 3).unwrap(),
        }];
        let mut request = ActivationRequest::new(
            "grid-demo-12",
            route,
            4,
            vec![0.1, 0.2, 0.3, 0.4],
            Some(PromptMetadata {
                prompt: "hello infernet".to_owned(),
                demo_mode: true,
                rpc_endpoints: vec!["192.0.2.10:50052".to_owned()],
                rpc_worker_peer_ids: Vec::new(),
            }),
        );
        request.origin_machine_id = Some("machine-a".to_owned());
        request.trace.push(TraceEvent {
            peer_id: "peer-a".to_owned(),
            layers: LayerRange::new(0, 3).unwrap(),
            next_peer_id: None,
            activation_size_bytes: 16,
            activation_checksum: 0x1234,
            timing_ms: 7,
        });

        let request_bytes = serde_json::to_vec(&request).unwrap();
        let decoded_request: ActivationRequest = serde_json::from_slice(&request_bytes).unwrap();
        assert_eq!(decoded_request, request);

        let response =
            ActivationResponse::success(decoded_request, "peer-a", Some("demo".to_owned()), 7);
        let response_bytes = serde_json::to_vec(&response).unwrap();
        let decoded_response: ActivationResponse = serde_json::from_slice(&response_bytes).unwrap();

        assert_eq!(decoded_response, response);
        assert_eq!(decoded_response.protocol_version, PROTOCOL_VERSION);
        assert_eq!(ACTIVATION_PROTOCOL, "/infernet/activation/2");
    }

    #[test]
    fn model_request_response_roundtrip_json() {
        let shard = ModelShardInfo {
            model_id: "grid-demo-12".to_owned(),
            layers: LayerRange::new(0, 3).unwrap(),
            checksum: "abc123".to_owned(),
            size_bytes: 4,
            version: "v1".to_owned(),
            protocol_version: PROTOCOL_VERSION,
        };
        let request = ModelShardRequest::new(
            shard.model_id.clone(),
            shard.layers,
            Some(shard.checksum.clone()),
            Some(shard.version.clone()),
        );
        let response = ModelShardResponse::success(&request, "peer-a", shard, vec![1, 2, 3, 4]);

        let request_bytes = serde_json::to_vec(&request).unwrap();
        let decoded_request: ModelShardRequest = serde_json::from_slice(&request_bytes).unwrap();
        assert_eq!(decoded_request, request);

        let response_bytes = serde_json::to_vec(&response).unwrap();
        let decoded_response: ModelShardResponse = serde_json::from_slice(&response_bytes).unwrap();
        assert_eq!(decoded_response, response);
        assert_eq!(MODEL_PROTOCOL, "/infernet/model/1");
    }

    #[test]
    fn legacy_advertisement_without_capabilities_still_deserializes() {
        let json = r#"{
            "protocol_version": 1,
            "peer_id": "legacy-peer",
            "addresses": [],
            "available_ram_bytes": 4096,
            "available_vram_bytes": null,
            "latency_hint_ms": 12,
            "hosted_shards": []
        }"#;

        let advertisement: NodeAdvertisement = serde_json::from_str(json).unwrap();
        assert_eq!(advertisement.peer_id, "legacy-peer");
        assert_eq!(advertisement.available_ram_bytes, Some(4096));
        assert_eq!(advertisement.capabilities, None);
        assert!(advertisement.model_shards.is_empty());
    }

    #[test]
    fn capabilities_roundtrip_and_none_is_omitted() {
        let capabilities = NodeCapabilities {
            os: "linux".to_owned(),
            arch: "x86_64".to_owned(),
            compute_backend: "cuda".to_owned(),
            device_name: "NVIDIA GeForce RTX 3090".to_owned(),
            machine_id: Some("machine-a".to_owned()),
            logical_cpu_cores: 16,
            total_ram_bytes: 64 * 1024 * 1024 * 1024,
            available_ram_bytes: 48 * 1024 * 1024 * 1024,
            total_accelerator_memory_bytes: 24 * 1024 * 1024 * 1024,
            available_accelerator_memory_bytes: 20 * 1024 * 1024 * 1024,
            vram_contribution_limit_bytes: Some(16 * 1024 * 1024 * 1024),
            unified_memory: false,
            max_sessions: 1,
            active_sessions: 0,
            measured_prefill_tokens_per_second: Some(92.5),
            measured_decode_tokens_per_second: Some(31.25),
            queue_depth: 0,
            llama_rpc: Some(LlamaRpcEndpoint {
                host: "192.0.2.10".to_owned(),
                port: 50052,
                rpc_protocol_version: "4.0.1".to_owned(),
                runtime_abi: "infernet-llama-rpc-v1".to_owned(),
                backend: "cuda".to_owned(),
                ready: true,
                tunnel_protocol: Some(LLAMA_RPC_TUNNEL_PROTOCOL.to_owned()),
            }),
            image_rpc: None,
        };
        let advertisement = NodeAdvertisement {
            protocol_version: PROTOCOL_VERSION,
            peer_id: "gpu-peer".to_owned(),
            addresses: Vec::new(),
            available_ram_bytes: Some(capabilities.available_ram_bytes),
            available_vram_bytes: Some(capabilities.available_accelerator_memory_bytes),
            latency_hint_ms: None,
            capabilities: Some(capabilities),
            hosted_shards: Vec::new(),
            model_shards: Vec::new(),
            model_components: Vec::new(),
            coarse_location: None,
        };

        let bytes = serde_json::to_vec(&advertisement).unwrap();
        let encoded: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let advertised_rpc = &encoded["capabilities"]["llama_rpc"];
        assert!(advertised_rpc.get("host").is_none());
        assert!(advertised_rpc.get("port").is_none());
        assert_eq!(
            encoded["capabilities"]["vram_contribution_limit_bytes"],
            16_u64 * 1024 * 1024 * 1024
        );
        let decoded: NodeAdvertisement = serde_json::from_slice(&bytes).unwrap();
        let mut expected = advertisement.clone();
        let expected_rpc = expected
            .capabilities
            .as_mut()
            .and_then(|capabilities| capabilities.llama_rpc.as_mut())
            .unwrap();
        expected_rpc.host.clear();
        expected_rpc.port = 0;
        assert_eq!(decoded, expected);
        assert_eq!(
            decoded
                .capabilities
                .as_ref()
                .and_then(|capabilities| capabilities.llama_rpc.as_ref())
                .and_then(|endpoint| endpoint.tunnel_protocol.as_deref()),
            Some(LLAMA_RPC_TUNNEL_PROTOCOL)
        );

        let mut without_capabilities = advertisement;
        without_capabilities.capabilities = None;
        let value = serde_json::to_value(without_capabilities).unwrap();
        assert!(value.get("capabilities").is_none());
    }

    #[test]
    fn older_capabilities_and_prompt_metadata_default_rpc_fields() {
        let capabilities_json = r#"{
            "os":"linux",
            "arch":"x86_64",
            "compute_backend":"cuda",
            "device_name":"RTX 4060",
            "logical_cpu_cores":8,
            "total_ram_bytes":32000,
            "available_ram_bytes":16000,
            "total_accelerator_memory_bytes":8000,
            "available_accelerator_memory_bytes":6000,
            "unified_memory":false,
            "max_sessions":1,
            "active_sessions":0,
            "queue_depth":0
        }"#;
        let capabilities: NodeCapabilities = serde_json::from_str(capabilities_json).unwrap();
        assert!(capabilities.llama_rpc.is_none());
        assert!(capabilities.vram_contribution_limit_bytes.is_none());
        let capabilities_value = serde_json::to_value(capabilities).unwrap();
        assert!(capabilities_value.get("llama_rpc").is_none());
        assert!(
            capabilities_value
                .get("vram_contribution_limit_bytes")
                .is_none()
        );

        let prompt_json = r#"{"prompt":"hello","demo_mode":false}"#;
        let prompt: PromptMetadata = serde_json::from_str(prompt_json).unwrap();
        assert!(prompt.rpc_endpoints.is_empty());
        assert!(prompt.rpc_worker_peer_ids.is_empty());

        let value = serde_json::to_value(prompt).unwrap();
        assert!(value.get("rpc_endpoints").is_none());
        assert!(value.get("rpc_worker_peer_ids").is_none());
    }
}
