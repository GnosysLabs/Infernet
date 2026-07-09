use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow, bail};
use infernet_model::{
    LayerRange, ModelManifest, SeedShardManifest, SeedSourceMetadata, ShardMetadata,
    TokenizerCompatibility,
};
use infernet_protocol::{ModelShardInfo, PROTOCOL_VERSION};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardCacheConfig {
    pub root: PathBuf,
    pub max_storage_bytes: u64,
    pub preferred_models: Vec<String>,
    pub pinned_models: Vec<String>,
    pub automatic_cleanup: bool,
}

impl ShardCacheConfig {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            max_storage_bytes: 50 * 1024 * 1024 * 1024,
            preferred_models: Vec::new(),
            pinned_models: Vec::new(),
            automatic_cleanup: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedShardRecord {
    pub info: ModelShardInfo,
    pub path: PathBuf,
    pub last_access_unix_ms: u64,
    pub pinned: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardCacheStats {
    pub root: PathBuf,
    pub shard_count: usize,
    pub storage_used_bytes: u64,
    pub max_storage_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeededModelSummary {
    pub model_id: String,
    pub display_name: String,
    pub source_path: PathBuf,
    pub source_checksum: String,
    pub source_size_bytes: u64,
    pub shard_count: usize,
    pub metadata_only: bool,
    pub records: Vec<CachedShardRecord>,
}

#[derive(Debug, Clone)]
pub struct ShardCache {
    config: ShardCacheConfig,
}

impl ShardCache {
    pub fn new(config: ShardCacheConfig) -> Result<Self> {
        fs::create_dir_all(config.root.join("data"))
            .with_context(|| format!("failed to create {}", config.root.join("data").display()))?;
        fs::create_dir_all(config.root.join("meta"))
            .with_context(|| format!("failed to create {}", config.root.join("meta").display()))?;
        Ok(Self { config })
    }

    pub fn config(&self) -> &ShardCacheConfig {
        &self.config
    }

    pub fn import_file(
        &self,
        source: &Path,
        model_id: impl Into<String>,
        layers: LayerRange,
        version: impl Into<String>,
    ) -> Result<CachedShardRecord> {
        let bytes = fs::read(source)
            .with_context(|| format!("failed to read shard file {}", source.display()))?;
        let checksum = sha256_bytes(&bytes);
        let info = ModelShardInfo {
            model_id: model_id.into(),
            layers,
            checksum,
            size_bytes: bytes.len() as u64,
            version: version.into(),
            protocol_version: PROTOCOL_VERSION,
        };

        self.store_verified(info, bytes)
    }

    pub fn import_payload(
        &self,
        payload: Vec<u8>,
        model_id: impl Into<String>,
        layers: LayerRange,
        version: impl Into<String>,
    ) -> Result<CachedShardRecord> {
        let checksum = sha256_bytes(&payload);
        let info = ModelShardInfo {
            model_id: model_id.into(),
            layers,
            checksum,
            size_bytes: payload.len() as u64,
            version: version.into(),
            protocol_version: PROTOCOL_VERSION,
        };

        self.store_verified(info, payload)
    }

    pub fn store_downloaded(
        &self,
        expected: &ModelShardInfo,
        payload: Vec<u8>,
    ) -> Result<CachedShardRecord> {
        let actual_checksum = sha256_bytes(&payload);
        if actual_checksum != expected.checksum {
            bail!(
                "checksum verification failed for {} {}:{}; expected {}, got {}",
                expected.model_id,
                expected.layers.start,
                expected.layers.end,
                expected.checksum,
                actual_checksum
            );
        }

        if payload.len() as u64 != expected.size_bytes {
            bail!(
                "size verification failed for {} {}:{}; expected {}, got {}",
                expected.model_id,
                expected.layers.start,
                expected.layers.end,
                expected.size_bytes,
                payload.len()
            );
        }

        self.store_verified(expected.clone(), payload)
    }

    pub fn list(&self) -> Result<Vec<CachedShardRecord>> {
        let meta_dir = self.config.root.join("meta");
        let mut records = Vec::new();

        if !meta_dir.exists() {
            return Ok(records);
        }

        for entry in fs::read_dir(&meta_dir)
            .with_context(|| format!("failed to read {}", meta_dir.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|value| value.to_str()) != Some("json") {
                continue;
            }

            let bytes =
                fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
            let record = serde_json::from_slice::<CachedShardRecord>(&bytes)
                .with_context(|| format!("failed to parse {}", path.display()))?;
            records.push(record);
        }

        records.sort_by_key(|record| {
            (
                record.info.model_id.clone(),
                record.info.layers.start,
                record.info.layers.end,
            )
        });
        Ok(records)
    }

    pub fn stats(&self) -> Result<ShardCacheStats> {
        let shards = self.list()?;
        let storage_used_bytes = shards.iter().map(|record| record.info.size_bytes).sum();

        Ok(ShardCacheStats {
            root: self.config.root.clone(),
            shard_count: shards.len(),
            storage_used_bytes,
            max_storage_bytes: self.config.max_storage_bytes,
        })
    }

    pub fn shard_infos(&self) -> Result<Vec<ModelShardInfo>> {
        Ok(self.list()?.into_iter().map(|record| record.info).collect())
    }

    pub fn find(
        &self,
        model_id: &str,
        layers: LayerRange,
        checksum: Option<&str>,
        version: Option<&str>,
    ) -> Result<Option<CachedShardRecord>> {
        Ok(self.list()?.into_iter().find(|record| {
            record.info.model_id == model_id
                && record.info.layers == layers
                && checksum.is_none_or(|checksum| record.info.checksum == checksum)
                && version.is_none_or(|version| record.info.version == version)
        }))
    }

    pub fn read_payload(&self, info: &ModelShardInfo) -> Result<Vec<u8>> {
        let record = self
            .find(
                &info.model_id,
                info.layers,
                Some(&info.checksum),
                Some(&info.version),
            )?
            .ok_or_else(|| anyhow!("shard {} not found in local cache", shard_label(info)))?;

        let payload = fs::read(&record.path)
            .with_context(|| format!("failed to read shard {}", record.path.display()))?;
        let actual_checksum = sha256_bytes(&payload);
        if actual_checksum != info.checksum {
            bail!(
                "local cache checksum mismatch for {}; expected {}, got {}",
                shard_label(info),
                info.checksum,
                actual_checksum
            );
        }

        self.touch(&record)?;
        Ok(payload)
    }

    pub fn evict_lru_if_needed(&self) -> Result<Vec<CachedShardRecord>> {
        if !self.config.automatic_cleanup {
            return Ok(Vec::new());
        }

        let mut records = self.list()?;
        let mut used = records
            .iter()
            .map(|record| record.info.size_bytes)
            .sum::<u64>();
        if used <= self.config.max_storage_bytes {
            return Ok(Vec::new());
        }

        records.sort_by_key(|record| record.last_access_unix_ms);
        let mut evicted = Vec::new();

        for record in records {
            if used <= self.config.max_storage_bytes {
                break;
            }
            if record.pinned || self.config.pinned_models.contains(&record.info.model_id) {
                continue;
            }

            let _ = fs::remove_file(&record.path);
            let _ = fs::remove_file(self.meta_path(&record.info));
            used = used.saturating_sub(record.info.size_bytes);
            evicted.push(record);
        }

        Ok(evicted)
    }

    fn store_verified(&self, info: ModelShardInfo, payload: Vec<u8>) -> Result<CachedShardRecord> {
        fs::create_dir_all(self.config.root.join("data"))?;
        fs::create_dir_all(self.config.root.join("meta"))?;

        let data_path = self.data_path(&info);
        fs::write(&data_path, payload)
            .with_context(|| format!("failed to write {}", data_path.display()))?;

        let record = CachedShardRecord {
            pinned: self.config.pinned_models.contains(&info.model_id),
            info,
            path: data_path,
            last_access_unix_ms: now_unix_ms(),
        };
        let meta_path = self.meta_path(&record.info);
        let json = serde_json::to_vec_pretty(&record)?;
        fs::write(&meta_path, json)
            .with_context(|| format!("failed to write {}", meta_path.display()))?;

        let _ = self.evict_lru_if_needed()?;
        Ok(record)
    }

    fn touch(&self, record: &CachedShardRecord) -> Result<()> {
        let mut updated = record.clone();
        updated.last_access_unix_ms = now_unix_ms();
        fs::write(
            self.meta_path(&updated.info),
            serde_json::to_vec_pretty(&updated)?,
        )?;
        Ok(())
    }

    fn data_path(&self, info: &ModelShardInfo) -> PathBuf {
        self.config.root.join("data").join(format!(
            "{}-{}-{}-{}.shard",
            sanitize(&info.model_id),
            info.layers.start,
            info.layers.end,
            &info.checksum[..16.min(info.checksum.len())]
        ))
    }

    fn meta_path(&self, info: &ModelShardInfo) -> PathBuf {
        self.config.root.join("meta").join(format!(
            "{}-{}-{}-{}.json",
            sanitize(&info.model_id),
            info.layers.start,
            info.layers.end,
            &info.checksum[..16.min(info.checksum.len())]
        ))
    }
}

pub fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex_lower(&hasher.finalize())
}

pub fn import_seed_model_from_file(
    cache: &ShardCache,
    source: &Path,
    manifest: &ModelManifest,
    version: impl Into<String>,
) -> Result<SeededModelSummary> {
    import_seed_model_from_file_with_progress(cache, source, manifest, version, |_, _| {})
}

pub fn import_seed_model_from_file_with_progress(
    cache: &ShardCache,
    source: &Path,
    manifest: &ModelManifest,
    version: impl Into<String>,
    mut on_hash_progress: impl FnMut(u64, u64),
) -> Result<SeededModelSummary> {
    if source.extension().and_then(|value| value.to_str()) != Some("gguf") {
        bail!(
            "{} must be a .gguf file for model bootstrap",
            source.display()
        );
    }

    let metadata =
        fs::metadata(source).with_context(|| format!("failed to inspect {}", source.display()))?;
    if !metadata.is_file() {
        bail!("{} is not a file", source.display());
    }

    let source_size_bytes = metadata.len();
    let source_checksum =
        sha256_file_with_progress(source, source_size_bytes, &mut on_hash_progress)?;
    let version = version.into();
    let ranges = manifest.automatic_layer_plan()?;
    let source_metadata = SeedSourceMetadata {
        path: source.display().to_string(),
        checksum_sha256: source_checksum.clone(),
        file_size_bytes: source_size_bytes,
    };
    let mut records = Vec::with_capacity(ranges.len());

    for layers in ranges {
        let shard_manifest = SeedShardManifest {
            model_id: manifest.model_id.clone(),
            display_name: manifest.display_name.clone(),
            architecture: manifest.architecture.clone(),
            layer_count: manifest.layer_count,
            hidden_size: manifest.hidden_size,
            activation_dtype: manifest.activation_dtype.clone(),
            runtime_kind: manifest.runtime_kind.clone(),
            layers,
            tokenizer: TokenizerCompatibility {
                family: manifest.architecture.clone(),
                checksum: None,
            },
            metadata: ShardMetadata {
                architecture: manifest.architecture.clone(),
                quantization: manifest.quantization.clone(),
                source_checksum: Some(source_checksum.clone()),
                protocol_version: PROTOCOL_VERSION,
            },
            source: source_metadata.clone(),
            shard_hash: seed_shard_hash(&manifest.model_id, layers, &source_checksum),
            payload_kind: "metadata-only".to_owned(),
        };
        let payload = serde_json::to_vec_pretty(&shard_manifest)?;
        let record = cache.import_payload(payload, manifest.model_id.clone(), layers, &version)?;
        records.push(record);
    }

    Ok(SeededModelSummary {
        model_id: manifest.model_id.clone(),
        display_name: manifest.display_name.clone(),
        source_path: source.to_path_buf(),
        source_checksum,
        source_size_bytes,
        shard_count: records.len(),
        metadata_only: true,
        records,
    })
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let total_bytes = fs::metadata(path)
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    sha256_file_with_progress(path, total_bytes, &mut |_, _| {})
}

fn sha256_file_with_progress(
    path: &Path,
    total_bytes: u64,
    on_progress: &mut impl FnMut(u64, u64),
) -> Result<String> {
    use std::io::Read;

    let mut file =
        fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0; 1024 * 64];
    let mut read_total = 0_u64;

    on_progress(read_total, total_bytes);

    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        read_total = read_total.saturating_add(read as u64);
        on_progress(read_total, total_bytes);
    }

    Ok(hex_lower(&hasher.finalize()))
}

fn seed_shard_hash(model_id: &str, layers: LayerRange, source_checksum: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(model_id.as_bytes());
    hasher.update(layers.start.to_le_bytes());
    hasher.update(layers.end.to_le_bytes());
    hasher.update(source_checksum.as_bytes());

    hex_lower(&hasher.finalize())
}

pub fn shard_label(info: &ModelShardInfo) -> String {
    format!(
        "{} {}:{} {} {}",
        info.model_id, info.layers.start, info.layers.end, info.version, info.checksum
    )
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}

fn sanitize(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stores_and_verifies_shard() {
        let temp =
            std::env::temp_dir().join(format!("infernet-cache-test-{}", uuid::Uuid::new_v4()));
        let source = temp.join("source.shard");
        fs::create_dir_all(&temp).unwrap();
        fs::write(&source, b"shard payload").unwrap();
        let cache = ShardCache::new(ShardCacheConfig::new(temp.clone())).unwrap();
        let record = cache
            .import_file(
                &source,
                "grid-demo-12",
                LayerRange::new(0, 3).unwrap(),
                "v1",
            )
            .unwrap();

        let payload = cache.read_payload(&record.info).unwrap();

        assert_eq!(payload, b"shard payload");
        assert_eq!(cache.list().unwrap().len(), 1);

        let _ = fs::remove_dir_all(temp);
    }

    #[test]
    fn rejects_bad_checksum() {
        let temp =
            std::env::temp_dir().join(format!("infernet-cache-test-{}", uuid::Uuid::new_v4()));
        let cache = ShardCache::new(ShardCacheConfig::new(temp.clone())).unwrap();
        let expected = ModelShardInfo {
            model_id: "grid-demo-12".to_owned(),
            layers: LayerRange::new(0, 3).unwrap(),
            checksum: "not-the-checksum".to_owned(),
            size_bytes: 4,
            version: "v1".to_owned(),
            protocol_version: PROTOCOL_VERSION,
        };

        let err = cache
            .store_downloaded(&expected, b"test".to_vec())
            .unwrap_err();
        assert!(err.to_string().contains("checksum verification failed"));

        let _ = fs::remove_dir_all(temp);
    }
}
