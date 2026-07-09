use infernet_model::{LayerRange, ShardDescriptor};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub const PROTOCOL_VERSION: u32 = 1;
pub const ACTIVATION_PROTOCOL: &str = "/infernet/activation/1";
pub const MODEL_PROTOCOL: &str = "/infernet/model/1";
pub const MODEL_BLOB_PROTOCOL: &str = "/infernet/model-blob/1";

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
            payload: Vec::new(),
            error: Some(error.into()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeAdvertisement {
    pub protocol_version: u32,
    pub peer_id: String,
    pub addresses: Vec<String>,
    pub available_ram_bytes: Option<u64>,
    pub available_vram_bytes: Option<u64>,
    pub latency_hint_ms: Option<u32>,
    pub hosted_shards: Vec<ShardDescriptor>,
    #[serde(default)]
    pub model_shards: Vec<ModelShardInfo>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteHop {
    pub peer_id: String,
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
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActivationRequest {
    pub protocol_version: u32,
    pub trace_id: Uuid,
    pub model_id: String,
    pub route: Vec<RouteHop>,
    pub current_hop_index: usize,
    pub hidden_size: usize,
    pub sequence_position: u32,
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
            model_id: model_id.into(),
            route,
            current_hop_index: 0,
            hidden_size,
            sequence_position: 0,
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
            }),
        );
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
        assert_eq!(ACTIVATION_PROTOCOL, "/infernet/activation/1");
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
}
