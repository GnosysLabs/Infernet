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

#[derive(Debug, Error, PartialEq, Eq)]
pub enum OfficialReleaseError {
    #[error("official release field {field} must not be empty")]
    EmptyField { field: String },
    #[error(
        "official release model {release_model_id} does not match manifest model {manifest_model_id}"
    )]
    ModelMismatch {
        release_model_id: String,
        manifest_model_id: String,
    },
    #[error(
        "component {component_id} targets model {component_model_id}, expected {release_model_id}"
    )]
    ComponentModelMismatch {
        component_id: String,
        component_model_id: String,
        release_model_id: String,
    },
    #[error("official release has no components")]
    EmptyComponents,
    #[error("component id {component_id} appears more than once")]
    DuplicateComponentId { component_id: String },
    #[error("component SHA-256 {sha256} appears more than once")]
    DuplicateComponentHash { sha256: String },
    #[error("official release field {field} is not a lowercase 64-character SHA-256")]
    InvalidSha256 { field: String },
    #[error("transformer component {component_id} has no layer range")]
    MissingTransformerLayers { component_id: String },
    #[error("shared component {component_id} must not declare a transformer layer range")]
    UnexpectedSharedLayers { component_id: String },
    #[error("component {component_id} has zero payload bytes")]
    EmptyComponent { component_id: String },
    #[error("launch context cap must be greater than zero")]
    EmptyLaunchContextCap,
    #[error("official release component byte total overflowed u64")]
    TotalBytesOverflow,
    #[error("official release expected {expected} total bytes, components contain {actual}")]
    TotalBytesMismatch { expected: u64, actual: u64 },
    #[error(transparent)]
    InvalidTransformerCoverage(#[from] ModelError),
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

    pub fn infernet_chat_v1() -> Self {
        Self {
            model_id: "infernet-chat-v1".to_owned(),
            display_name: "Infernet Chat".to_owned(),
            architecture: "gemma4".to_owned(),
            layer_count: 30,
            hidden_size: 2816,
            activation_dtype: "f16".to_owned(),
            quantization: Some("Q4_0".to_owned()),
            runtime_kind: RuntimeKind::LlamaCpp,
        }
    }

    pub fn catalog() -> Vec<Self> {
        vec![Self::infernet_chat_v1(), Self::demo(), Self::llama32_1b()]
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
            // Until partial-graph execution is proven for each llama.cpp architecture,
            // keep imported GGUF models in one executable package. This avoids
            // duplicating global tensors into every transformer-layer shard and also
            // gives unsupported split architectures a correct full-model fallback.
            RuntimeKind::LlamaCpp => self.layer_count,
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
pub struct OfficialUpstreamProvenance {
    pub publisher: String,
    pub repository: String,
    pub revision: String,
    pub artifact: String,
    pub source_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OfficialComponentKind {
    Transformer,
    Shared,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OfficialModelComponent {
    pub component_id: String,
    pub model_id: String,
    pub kind: OfficialComponentKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layers: Option<LayerRange>,
    pub size_bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OfficialModelRelease {
    pub release_id: String,
    pub version: String,
    pub model_id: String,
    pub upstream: OfficialUpstreamProvenance,
    pub expected_total_bytes: u64,
    pub launch_context_cap_tokens: u32,
    pub components: Vec<OfficialModelComponent>,
}

impl OfficialModelRelease {
    /// Pinned, byte-for-byte trust anchor for the temporary full-model package.
    /// This is kept separate from the future multi-component network release.
    pub fn infernet_chat_v1_compatibility() -> Self {
        const SOURCE_SHA256: &str =
            "4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5";
        const SOURCE_BYTES: u64 = 14_439_361_440;

        Self {
            release_id: "infernet-chat-v1-compatibility".to_owned(),
            version: "1.0.0-compat.1".to_owned(),
            model_id: "infernet-chat-v1".to_owned(),
            upstream: OfficialUpstreamProvenance {
                publisher: "Google".to_owned(),
                repository: "google/gemma-4-26B-A4B-it-qat-q4_0-gguf".to_owned(),
                revision: "dfc00409adc70be497fee9c90bfe76b3ee130f2e".to_owned(),
                artifact: "gemma-4-26B_q4_0-it.gguf".to_owned(),
                source_sha256: SOURCE_SHA256.to_owned(),
            },
            expected_total_bytes: SOURCE_BYTES,
            launch_context_cap_tokens: 8_192,
            components: vec![OfficialModelComponent {
                component_id: "compatibility-full-model".to_owned(),
                model_id: "infernet-chat-v1".to_owned(),
                kind: OfficialComponentKind::Transformer,
                layers: Some(LayerRange { start: 0, end: 30 }),
                size_bytes: SOURCE_BYTES,
                sha256: SOURCE_SHA256.to_owned(),
            }],
        }
    }

    pub fn validate_for_model(&self, model: &ModelManifest) -> Result<(), OfficialReleaseError> {
        validate_nonempty("release_id", &self.release_id)?;
        validate_nonempty("version", &self.version)?;
        validate_nonempty("upstream.publisher", &self.upstream.publisher)?;
        validate_nonempty("upstream.repository", &self.upstream.repository)?;
        validate_nonempty("upstream.revision", &self.upstream.revision)?;
        validate_nonempty("upstream.artifact", &self.upstream.artifact)?;
        validate_sha256("upstream.source_sha256", &self.upstream.source_sha256)?;

        if self.model_id != model.model_id {
            return Err(OfficialReleaseError::ModelMismatch {
                release_model_id: self.model_id.clone(),
                manifest_model_id: model.model_id.clone(),
            });
        }
        if self.launch_context_cap_tokens == 0 {
            return Err(OfficialReleaseError::EmptyLaunchContextCap);
        }
        if self.components.is_empty() {
            return Err(OfficialReleaseError::EmptyComponents);
        }

        let mut component_ids = std::collections::BTreeSet::new();
        let mut component_hashes = std::collections::BTreeSet::new();
        let mut transformer_ranges = Vec::new();
        let mut total_bytes = 0_u64;

        for component in &self.components {
            validate_nonempty("component.component_id", &component.component_id)?;
            if component.model_id != self.model_id {
                return Err(OfficialReleaseError::ComponentModelMismatch {
                    component_id: component.component_id.clone(),
                    component_model_id: component.model_id.clone(),
                    release_model_id: self.model_id.clone(),
                });
            }
            if !component_ids.insert(component.component_id.clone()) {
                return Err(OfficialReleaseError::DuplicateComponentId {
                    component_id: component.component_id.clone(),
                });
            }
            validate_sha256(
                &format!("component.{}.sha256", component.component_id),
                &component.sha256,
            )?;
            if !component_hashes.insert(component.sha256.clone()) {
                return Err(OfficialReleaseError::DuplicateComponentHash {
                    sha256: component.sha256.clone(),
                });
            }
            if component.size_bytes == 0 {
                return Err(OfficialReleaseError::EmptyComponent {
                    component_id: component.component_id.clone(),
                });
            }
            total_bytes = total_bytes
                .checked_add(component.size_bytes)
                .ok_or(OfficialReleaseError::TotalBytesOverflow)?;

            match (component.kind, component.layers) {
                (OfficialComponentKind::Transformer, Some(layers)) => {
                    transformer_ranges.push(layers)
                }
                (OfficialComponentKind::Transformer, None) => {
                    return Err(OfficialReleaseError::MissingTransformerLayers {
                        component_id: component.component_id.clone(),
                    });
                }
                (OfficialComponentKind::Shared, Some(_)) => {
                    return Err(OfficialReleaseError::UnexpectedSharedLayers {
                        component_id: component.component_id.clone(),
                    });
                }
                (OfficialComponentKind::Shared, None) => {}
            }
        }

        transformer_ranges.sort_by_key(|range| (range.start, range.end));
        validate_contiguous_coverage(model.layer_count, transformer_ranges)?;

        if total_bytes != self.expected_total_bytes {
            return Err(OfficialReleaseError::TotalBytesMismatch {
                expected: self.expected_total_bytes,
                actual: total_bytes,
            });
        }

        Ok(())
    }
}

fn validate_nonempty(field: &str, value: &str) -> Result<(), OfficialReleaseError> {
    if value.trim().is_empty() {
        return Err(OfficialReleaseError::EmptyField {
            field: field.to_owned(),
        });
    }
    Ok(())
}

fn validate_sha256(field: &str, value: &str) -> Result<(), OfficialReleaseError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(OfficialReleaseError::InvalidSha256 {
            field: field.to_owned(),
        });
    }
    Ok(())
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

    fn synthetic_sha256(character: char) -> String {
        character.to_string().repeat(64)
    }

    fn valid_official_release() -> OfficialModelRelease {
        OfficialModelRelease {
            release_id: "infernet-chat-v1-release".to_owned(),
            version: "1.0.0-test".to_owned(),
            model_id: "infernet-chat-v1".to_owned(),
            upstream: OfficialUpstreamProvenance {
                publisher: "Google".to_owned(),
                repository: "google/gemma-4-test".to_owned(),
                revision: "synthetic-test-revision".to_owned(),
                artifact: "synthetic-test.gguf".to_owned(),
                source_sha256: synthetic_sha256('a'),
            },
            expected_total_bytes: 600,
            launch_context_cap_tokens: 16_384,
            components: vec![
                OfficialModelComponent {
                    component_id: "transformer-000".to_owned(),
                    model_id: "infernet-chat-v1".to_owned(),
                    kind: OfficialComponentKind::Transformer,
                    layers: Some(LayerRange::new(0, 10).unwrap()),
                    size_bytes: 100,
                    sha256: synthetic_sha256('b'),
                },
                OfficialModelComponent {
                    component_id: "transformer-001".to_owned(),
                    model_id: "infernet-chat-v1".to_owned(),
                    kind: OfficialComponentKind::Transformer,
                    layers: Some(LayerRange::new(10, 30).unwrap()),
                    size_bytes: 200,
                    sha256: synthetic_sha256('c'),
                },
                OfficialModelComponent {
                    component_id: "shared".to_owned(),
                    model_id: "infernet-chat-v1".to_owned(),
                    kind: OfficialComponentKind::Shared,
                    layers: None,
                    size_bytes: 300,
                    sha256: synthetic_sha256('d'),
                },
            ],
        }
    }

    #[test]
    fn infernet_chat_v1_is_the_canonical_gemma4_manifest() {
        let manifest = ModelManifest::infernet_chat_v1();

        assert_eq!(manifest.model_id, "infernet-chat-v1");
        assert_eq!(manifest.display_name, "Infernet Chat");
        assert_eq!(manifest.architecture, "gemma4");
        assert_eq!(manifest.layer_count, 30);
        assert_eq!(manifest.hidden_size, 2816);
        assert_eq!(manifest.activation_dtype, "f16");
        assert_eq!(manifest.quantization.as_deref(), Some("Q4_0"));
        assert_eq!(manifest.runtime_kind, RuntimeKind::LlamaCpp);
        assert_eq!(ModelManifest::by_id("infernet-chat-v1"), Some(manifest));
    }

    #[test]
    fn pinned_compatibility_release_is_valid_and_byte_exact() {
        let release = OfficialModelRelease::infernet_chat_v1_compatibility();

        assert_eq!(
            release.validate_for_model(&ModelManifest::infernet_chat_v1()),
            Ok(())
        );
        assert_eq!(release.expected_total_bytes, 14_439_361_440);
        assert_eq!(
            release.components[0].sha256,
            "4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5"
        );
    }

    #[test]
    fn official_release_types_are_serializable() {
        fn assert_serde<T: serde::Serialize + serde::de::DeserializeOwned>() {}

        assert_serde::<OfficialUpstreamProvenance>();
        assert_serde::<OfficialModelComponent>();
        assert_serde::<OfficialModelRelease>();
    }

    #[test]
    fn validates_complete_official_release() {
        assert_eq!(
            valid_official_release().validate_for_model(&ModelManifest::infernet_chat_v1()),
            Ok(())
        );
    }

    #[test]
    fn official_release_requires_supplied_sha256_values() {
        let mut release = valid_official_release();
        release.upstream.source_sha256.clear();

        assert_eq!(
            release.validate_for_model(&ModelManifest::infernet_chat_v1()),
            Err(OfficialReleaseError::InvalidSha256 {
                field: "upstream.source_sha256".to_owned()
            })
        );
    }

    #[test]
    fn official_release_rejects_model_mismatch() {
        let release = valid_official_release();

        assert_eq!(
            release.validate_for_model(&ModelManifest::llama32_1b()),
            Err(OfficialReleaseError::ModelMismatch {
                release_model_id: "infernet-chat-v1".to_owned(),
                manifest_model_id: "llama-3.2-1b".to_owned(),
            })
        );
    }

    #[test]
    fn official_release_rejects_noncontiguous_transformer_coverage() {
        let mut release = valid_official_release();
        release.components[0].layers = Some(LayerRange::new(0, 9).unwrap());

        assert_eq!(
            release.validate_for_model(&ModelManifest::infernet_chat_v1()),
            Err(OfficialReleaseError::InvalidTransformerCoverage(
                ModelError::NonContiguous {
                    expected: 9,
                    actual: 10,
                }
            ))
        );
    }

    #[test]
    fn official_release_rejects_duplicate_component_ids_and_hashes() {
        let mut duplicate_id = valid_official_release();
        duplicate_id.components[1].component_id = duplicate_id.components[0].component_id.clone();
        assert_eq!(
            duplicate_id.validate_for_model(&ModelManifest::infernet_chat_v1()),
            Err(OfficialReleaseError::DuplicateComponentId {
                component_id: "transformer-000".to_owned()
            })
        );

        let mut duplicate_hash = valid_official_release();
        duplicate_hash.components[1].sha256 = duplicate_hash.components[0].sha256.clone();
        assert_eq!(
            duplicate_hash.validate_for_model(&ModelManifest::infernet_chat_v1()),
            Err(OfficialReleaseError::DuplicateComponentHash {
                sha256: synthetic_sha256('b')
            })
        );
    }

    #[test]
    fn official_release_rejects_incorrect_total_bytes() {
        let mut release = valid_official_release();
        release.expected_total_bytes += 1;

        assert_eq!(
            release.validate_for_model(&ModelManifest::infernet_chat_v1()),
            Err(OfficialReleaseError::TotalBytesMismatch {
                expected: 601,
                actual: 600,
            })
        );
    }

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
    fn plans_llama32_1b_as_one_complete_runtime_package() {
        let manifest = ModelManifest::llama32_1b();
        let ranges = manifest.automatic_layer_plan().unwrap();

        assert_eq!(ranges, vec![LayerRange::new(0, 16).unwrap()]);
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
