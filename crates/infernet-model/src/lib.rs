use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod gguf;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ModelError {
    #[error("layer range {start}..{end} is empty")]
    EmptyRange { start: u32, end: u32 },
    #[error("layer range {start}..{end} exceeds model layer count {layer_count}")]
    RangeOutOfBounds {
        start: u32,
        end: u32,
        layer_count: u32,
    },
    #[error("expected layer {expected}, found layer {actual}")]
    NonContiguous { expected: u32, actual: u32 },
    #[error("route ended at layer {actual}, expected {expected}")]
    IncompleteCoverage { actual: u32, expected: u32 },
    #[error("cannot plan shards for a model with zero layers")]
    EmptyModel,
    #[error("GGUF metadata is missing required field {field}")]
    MissingGgufMetadata { field: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerRange {
    pub start: u32,
    pub end: u32,
}

impl LayerRange {
    pub fn new(start: u32, end: u32) -> Result<Self, ModelError> {
        if start >= end {
            return Err(ModelError::EmptyRange { start, end });
        }

        Ok(Self { start, end })
    }

    pub fn len(&self) -> u32 {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }

    pub fn contains(&self, other: &LayerRange) -> bool {
        self.start <= other.start && other.end <= self.end
    }

    pub fn validate_for_model(&self, layer_count: u32) -> Result<(), ModelError> {
        if self.start >= self.end {
            return Err(ModelError::EmptyRange {
                start: self.start,
                end: self.end,
            });
        }

        if self.end > layer_count {
            return Err(ModelError::RangeOutOfBounds {
                start: self.start,
                end: self.end,
                layer_count,
            });
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimeKind {
    Demo,
    LlamaCpp,
}

impl RuntimeKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Demo => "demo",
            Self::LlamaCpp => "llama_cpp",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizerCompatibility {
    pub family: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checksum: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardMetadata {
    pub architecture: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_checksum: Option<String>,
    pub protocol_version: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelManifest {
    pub model_id: String,
    pub display_name: String,
    pub architecture: String,
    pub layer_count: u32,
    pub hidden_size: usize,
    pub activation_dtype: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
    pub runtime_kind: RuntimeKind,
}

impl ModelManifest {
    pub fn demo() -> Self {
        Self {
            model_id: "grid-demo-12".to_owned(),
            display_name: "Infernet Demo 12 Layer Model".to_owned(),
            architecture: "demo-transformer".to_owned(),
            layer_count: 12,
            hidden_size: 16,
            activation_dtype: "f32".to_owned(),
            quantization: None,
            runtime_kind: RuntimeKind::Demo,
        }
    }

    pub fn llama32_1b() -> Self {
        Self {
            model_id: "llama-3.2-1b".to_owned(),
            display_name: "Llama 3.2 1B".to_owned(),
            architecture: "llama".to_owned(),
            layer_count: 16,
            hidden_size: 2048,
            activation_dtype: "f16".to_owned(),
            quantization: None,
            runtime_kind: RuntimeKind::LlamaCpp,
        }
    }

    pub fn catalog() -> Vec<Self> {
        vec![Self::demo(), Self::llama32_1b()]
    }

    pub fn by_id(model_id: &str) -> Option<Self> {
        Self::catalog()
            .into_iter()
            .find(|manifest| manifest.model_id == model_id)
    }

    pub fn from_gguf_info(
        model_id: String,
        display_name: String,
        info: &gguf::GgufInfo,
    ) -> Result<Self, ModelError> {
        Ok(Self {
            model_id,
            display_name,
            architecture: info.architecture.clone().ok_or_else(|| {
                ModelError::MissingGgufMetadata {
                    field: "general.architecture".to_owned(),
                }
            })?,
            layer_count: info
                .layer_count
                .ok_or_else(|| ModelError::MissingGgufMetadata {
                    field: "block_count".to_owned(),
                })?,
            hidden_size: info
                .hidden_size
                .ok_or_else(|| ModelError::MissingGgufMetadata {
                    field: "embedding_length".to_owned(),
                })?,
            activation_dtype: "f16".to_owned(),
            quantization: info.quantization.clone(),
            runtime_kind: RuntimeKind::LlamaCpp,
        })
    }

    pub fn automatic_layer_plan(&self) -> Result<Vec<LayerRange>, ModelError> {
        let target_layers = match self.runtime_kind {
            RuntimeKind::Demo => 3,
            RuntimeKind::LlamaCpp if self.layer_count <= 32 => 4,
            RuntimeKind::LlamaCpp => 8,
        };

        plan_layer_ranges(self.layer_count, target_layers)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardDescriptor {
    pub model_id: String,
    pub layers: LayerRange,
    pub runtime_kind: RuntimeKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer: Option<TokenizerCompatibility>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<ShardMetadata>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shard_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed_manifest: Option<Box<SeedShardManifest>>,
}

impl ShardDescriptor {
    pub fn demo(model_id: impl Into<String>, layers: LayerRange) -> Self {
        Self {
            model_id: model_id.into(),
            layers,
            runtime_kind: RuntimeKind::Demo,
            tokenizer: None,
            metadata: None,
            shard_hash: None,
            seed_manifest: None,
        }
    }

    pub fn for_manifest(manifest: &ModelManifest, layers: LayerRange) -> Self {
        Self {
            model_id: manifest.model_id.clone(),
            layers,
            runtime_kind: manifest.runtime_kind.clone(),
            tokenizer: None,
            metadata: Some(ShardMetadata {
                architecture: manifest.architecture.clone(),
                quantization: manifest.quantization.clone(),
                source_checksum: None,
                protocol_version: 1,
            }),
            shard_hash: None,
            seed_manifest: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GgufSourceMetadata {
    pub path: String,
    pub checksum_sha256: String,
    pub gguf_version: u32,
    pub metadata_kv_count: u64,
    pub tensor_count: u64,
    pub file_size_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GgufShardManifest {
    pub model_id: String,
    pub display_name: String,
    pub architecture: String,
    pub layer_count: u32,
    pub hidden_size: usize,
    pub activation_dtype: String,
    pub runtime_kind: RuntimeKind,
    pub layers: LayerRange,
    pub tokenizer: TokenizerCompatibility,
    pub metadata: ShardMetadata,
    pub source: GgufSourceMetadata,
    pub required_tensors: Vec<String>,
    pub boundary_tensors: Vec<String>,
    pub shard_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeedSourceMetadata {
    pub path: String,
    pub checksum_sha256: String,
    pub file_size_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeedShardManifest {
    pub model_id: String,
    pub display_name: String,
    pub architecture: String,
    pub layer_count: u32,
    pub hidden_size: usize,
    pub activation_dtype: String,
    pub runtime_kind: RuntimeKind,
    pub layers: LayerRange,
    pub tokenizer: TokenizerCompatibility,
    pub metadata: ShardMetadata,
    pub source: SeedSourceMetadata,
    pub shard_hash: String,
    pub payload_kind: String,
}

pub fn plan_layer_ranges(
    layer_count: u32,
    target_layers_per_shard: u32,
) -> Result<Vec<LayerRange>, ModelError> {
    if layer_count == 0 {
        return Err(ModelError::EmptyModel);
    }

    let target_layers_per_shard = target_layers_per_shard.max(1);
    let mut ranges = Vec::new();
    let mut start = 0;

    while start < layer_count {
        let end = (start + target_layers_per_shard).min(layer_count);
        ranges.push(LayerRange::new(start, end)?);
        start = end;
    }

    Ok(ranges)
}

pub fn validate_contiguous_coverage(
    layer_count: u32,
    ranges: impl IntoIterator<Item = LayerRange>,
) -> Result<(), ModelError> {
    let mut expected = 0;

    for range in ranges {
        range.validate_for_model(layer_count)?;

        if range.start != expected {
            return Err(ModelError::NonContiguous {
                expected,
                actual: range.start,
            });
        }

        expected = range.end;
    }

    if expected != layer_count {
        return Err(ModelError::IncompleteCoverage {
            actual: expected,
            expected: layer_count,
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_contiguous_coverage() {
        let ranges = [
            LayerRange::new(0, 3).unwrap(),
            LayerRange::new(3, 6).unwrap(),
            LayerRange::new(6, 9).unwrap(),
            LayerRange::new(9, 12).unwrap(),
        ];

        assert_eq!(validate_contiguous_coverage(12, ranges), Ok(()));
    }

    #[test]
    fn plans_llama32_1b_automatically() {
        let manifest = ModelManifest::llama32_1b();
        let ranges = manifest.automatic_layer_plan().unwrap();

        assert_eq!(
            ranges,
            vec![
                LayerRange::new(0, 4).unwrap(),
                LayerRange::new(4, 8).unwrap(),
                LayerRange::new(8, 12).unwrap(),
                LayerRange::new(12, 16).unwrap(),
            ]
        );
        assert_eq!(
            validate_contiguous_coverage(manifest.layer_count, ranges),
            Ok(())
        );
    }

    #[test]
    fn automatic_planner_handles_remainder_layers() {
        let ranges = plan_layer_ranges(10, 4).unwrap();

        assert_eq!(
            ranges,
            vec![
                LayerRange::new(0, 4).unwrap(),
                LayerRange::new(4, 8).unwrap(),
                LayerRange::new(8, 10).unwrap(),
            ]
        );
    }

    #[test]
    fn rejects_gap() {
        let ranges = [
            LayerRange::new(0, 3).unwrap(),
            LayerRange::new(4, 6).unwrap(),
        ];

        assert_eq!(
            validate_contiguous_coverage(6, ranges),
            Err(ModelError::NonContiguous {
                expected: 3,
                actual: 4
            })
        );
    }
}
