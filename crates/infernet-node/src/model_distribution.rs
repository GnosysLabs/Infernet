use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow, bail};
use infernet_model::{
    LayerRange, ModelManifest, SeedShardManifest, SeedSourceMetadata, ShardMetadata,
    TokenizerCompatibility, gguf::write_layer_shard_with_progress,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manifest: Option<SeedShardManifest>,
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

#[derive(Debug, Clone, Copy)]
pub struct SeedShardBuildProgress {
    pub shard_index: usize,
    pub shard_count: usize,
    pub layers: LayerRange,
    pub written_bytes: u64,
    pub total_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct ShardCache {
    config: ShardCacheConfig,
}

pub const PAYLOAD_KIND_METADATA_ONLY: &str = "metadata-only";
pub const PAYLOAD_KIND_GGUF_SHARD: &str = "gguf-shard";
pub const PAYLOAD_KIND_INFERNET_SHARD: &str = "infernet-shard";
pub const INFERNET_SHARD_FORMAT_VERSION: &str = "infernet-shard-v1";
pub const INFERNET_SHARD_RUNTIME_ABI: &str = "infernet-llama-layer-v1";
pub const INFERNET_SHARD_MANIFEST_FILE: &str = "manifest.json";
pub const INFERNET_SHARD_TENSOR_FILE: &str = "tensors.gguf";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfernetShardPackageManifest {
    pub format_version: String,
    pub runtime_abi: String,
    pub component: String,
    pub seed_manifest: SeedShardManifest,
    pub payload: InfernetShardPayloadManifest,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfernetShardPayloadManifest {
    pub kind: String,
    pub file: String,
    pub checksum_sha256: String,
    pub size_bytes: u64,
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

        self.store_verified_payload(info, bytes, None)
    }

    pub fn import_physical_shard_file(
        &self,
        source: &Path,
        model_id: impl Into<String>,
        layers: LayerRange,
        version: impl Into<String>,
        manifest: SeedShardManifest,
    ) -> Result<CachedShardRecord> {
        let size_bytes = fs::metadata(source)
            .with_context(|| format!("failed to inspect shard file {}", source.display()))?
            .len();
        let checksum = sha256_file(source)?;
        let info = ModelShardInfo {
            model_id: model_id.into(),
            layers,
            checksum,
            size_bytes,
            version: version.into(),
            protocol_version: PROTOCOL_VERSION,
        };

        self.store_verified_file(info, source, Some(manifest), true)
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

        self.store_verified_payload(info, payload, None)
    }

    pub fn store_downloaded(
        &self,
        expected: &ModelShardInfo,
        payload: Vec<u8>,
    ) -> Result<CachedShardRecord> {
        self.store_downloaded_with_manifest(expected, payload, None)
    }

    pub fn store_downloaded_with_manifest(
        &self,
        expected: &ModelShardInfo,
        payload: Vec<u8>,
        manifest: Option<SeedShardManifest>,
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

        self.store_verified_payload(expected.clone(), payload, manifest)
    }

    pub fn store_downloaded_file(
        &self,
        expected: &ModelShardInfo,
        path: &Path,
        manifest: Option<SeedShardManifest>,
    ) -> Result<CachedShardRecord> {
        let actual_checksum = sha256_file(path)?;
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

        let actual_size = fs::metadata(path)
            .with_context(|| format!("failed to inspect {}", path.display()))?
            .len();
        if actual_size != expected.size_bytes {
            bail!(
                "size verification failed for {} {}:{}; expected {}, got {}",
                expected.model_id,
                expected.layers.start,
                expected.layers.end,
                expected.size_bytes,
                actual_size
            );
        }

        self.store_verified_file(expected.clone(), path, manifest, true)
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

            let _ = remove_cached_payload_path(&record.path);
            let _ = fs::remove_file(self.meta_path(&record.info));
            used = used.saturating_sub(record.info.size_bytes);
            evicted.push(record);
        }

        Ok(evicted)
    }

    fn store_verified_payload(
        &self,
        info: ModelShardInfo,
        payload: Vec<u8>,
        manifest: Option<SeedShardManifest>,
    ) -> Result<CachedShardRecord> {
        fs::create_dir_all(self.config.root.join("data"))?;
        fs::create_dir_all(self.config.root.join("meta"))?;

        let data_path = self.data_path(&info, manifest.as_ref());
        prepare_data_path(&data_path)?;
        fs::write(&data_path, payload)
            .with_context(|| format!("failed to write {}", data_path.display()))?;
        write_infernet_shard_package_manifest(&info, &data_path, manifest.as_ref())?;

        let record = CachedShardRecord {
            pinned: self.config.pinned_models.contains(&info.model_id),
            info,
            path: data_path,
            last_access_unix_ms: now_unix_ms(),
            manifest,
        };
        self.write_record(record)
    }

    fn store_verified_file(
        &self,
        info: ModelShardInfo,
        source: &Path,
        manifest: Option<SeedShardManifest>,
        remove_source_after_copy: bool,
    ) -> Result<CachedShardRecord> {
        fs::create_dir_all(self.config.root.join("data"))?;
        fs::create_dir_all(self.config.root.join("meta"))?;

        let data_path = self.data_path(&info, manifest.as_ref());
        prepare_data_path(&data_path)?;
        if remove_source_after_copy {
            match fs::rename(source, &data_path) {
                Ok(()) => {}
                Err(_) => {
                    fs::copy(source, &data_path).with_context(|| {
                        format!(
                            "failed to copy shard file {} to {}",
                            source.display(),
                            data_path.display()
                        )
                    })?;
                    let _ = fs::remove_file(source);
                }
            }
        } else {
            fs::copy(source, &data_path).with_context(|| {
                format!(
                    "failed to copy shard file {} to {}",
                    source.display(),
                    data_path.display()
                )
            })?;
        }
        write_infernet_shard_package_manifest(&info, &data_path, manifest.as_ref())?;

        let record = CachedShardRecord {
            pinned: self.config.pinned_models.contains(&info.model_id),
            info,
            path: data_path,
            last_access_unix_ms: now_unix_ms(),
            manifest,
        };
        self.write_record(record)
    }

    fn write_record(&self, record: CachedShardRecord) -> Result<CachedShardRecord> {
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

    fn data_path(&self, info: &ModelShardInfo, manifest: Option<&SeedShardManifest>) -> PathBuf {
        let basename = format!(
            "{}-{}-{}-{}.{}",
            sanitize(&info.model_id),
            info.layers.start,
            info.layers.end,
            &info.checksum[..16.min(info.checksum.len())],
            data_extension(manifest)
        );
        if manifest.is_some_and(|manifest| manifest.payload_kind == PAYLOAD_KIND_INFERNET_SHARD) {
            self.config
                .root
                .join("data")
                .join(basename)
                .join(INFERNET_SHARD_TENSOR_FILE)
        } else {
            self.config.root.join("data").join(basename)
        }
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

fn data_extension(manifest: Option<&SeedShardManifest>) -> &'static str {
    match manifest.map(|manifest| manifest.payload_kind.as_str()) {
        Some(PAYLOAD_KIND_INFERNET_SHARD) => "infershard",
        Some(PAYLOAD_KIND_GGUF_SHARD) => "gguf",
        _ => "shard",
    }
}

fn prepare_data_path(path: &Path) -> Result<()> {
    if let Some(package_dir) = infernet_shard_package_dir(path) {
        if package_dir.exists() {
            fs::remove_dir_all(package_dir)
                .with_context(|| format!("failed to replace {}", package_dir.display()))?;
        }
        fs::create_dir_all(package_dir)
            .with_context(|| format!("failed to create {}", package_dir.display()))?;
        return Ok(());
    }

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    if path.exists() {
        fs::remove_file(path).with_context(|| format!("failed to replace {}", path.display()))?;
    }
    Ok(())
}

fn remove_cached_payload_path(path: &Path) -> Result<()> {
    if let Some(package_dir) = infernet_shard_package_dir(path) {
        if package_dir.exists() {
            fs::remove_dir_all(package_dir)
                .with_context(|| format!("failed to remove {}", package_dir.display()))?;
        }
        return Ok(());
    }

    if path.exists() {
        fs::remove_file(path).with_context(|| format!("failed to remove {}", path.display()))?;
    }
    Ok(())
}

fn infernet_shard_package_dir(path: &Path) -> Option<&Path> {
    let parent = path.parent()?;
    (parent.extension().and_then(|value| value.to_str()) == Some("infershard")).then_some(parent)
}

fn write_infernet_shard_package_manifest(
    info: &ModelShardInfo,
    payload_path: &Path,
    manifest: Option<&SeedShardManifest>,
) -> Result<()> {
    let Some(manifest) =
        manifest.filter(|manifest| manifest.payload_kind == PAYLOAD_KIND_INFERNET_SHARD)
    else {
        return Ok(());
    };
    let Some(package_dir) = infernet_shard_package_dir(payload_path) else {
        return Ok(());
    };

    let package_manifest = InfernetShardPackageManifest {
        format_version: INFERNET_SHARD_FORMAT_VERSION.to_owned(),
        runtime_abi: INFERNET_SHARD_RUNTIME_ABI.to_owned(),
        component: "transformer_layer".to_owned(),
        seed_manifest: manifest.clone(),
        payload: InfernetShardPayloadManifest {
            kind: "gguf_tensor_payload".to_owned(),
            file: INFERNET_SHARD_TENSOR_FILE.to_owned(),
            checksum_sha256: info.checksum.clone(),
            size_bytes: info.size_bytes,
        },
    };
    let path = package_dir.join(INFERNET_SHARD_MANIFEST_FILE);
    fs::write(&path, serde_json::to_vec_pretty(&package_manifest)?)
        .with_context(|| format!("failed to write {}", path.display()))
}

pub fn source_cache_root(config: &ShardCacheConfig) -> PathBuf {
    config
        .root
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| config.root.clone())
        .join("sources")
}

pub fn source_cache_path(
    config: &ShardCacheConfig,
    model_id: &str,
    source_checksum: &str,
) -> PathBuf {
    source_cache_root(config).join(format!(
        "{}-{}.gguf",
        sanitize(model_id),
        &source_checksum[..16.min(source_checksum.len())]
    ))
}

pub fn executable_source_path_for_manifest(
    config: &ShardCacheConfig,
    manifest: &SeedShardManifest,
) -> Option<PathBuf> {
    let original = PathBuf::from(&manifest.source.path);
    if original.is_file() {
        return Some(original);
    }

    let cached = source_cache_path(config, &manifest.model_id, &manifest.source.checksum_sha256);
    cached.is_file().then_some(cached)
}

pub fn is_executable_shard_record(record: &CachedShardRecord) -> bool {
    record.manifest.as_ref().is_some_and(|manifest| {
        matches!(
            manifest.payload_kind.as_str(),
            PAYLOAD_KIND_GGUF_SHARD | PAYLOAD_KIND_INFERNET_SHARD
        ) && record.path.is_file()
    })
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
    import_seed_model_from_file_with_build_progress(
        cache,
        source,
        manifest,
        version,
        |_, _| {},
        |_| {},
    )
}

pub fn import_seed_model_from_file_with_progress(
    cache: &ShardCache,
    source: &Path,
    manifest: &ModelManifest,
    version: impl Into<String>,
    mut on_hash_progress: impl FnMut(u64, u64),
) -> Result<SeededModelSummary> {
    import_seed_model_from_file_with_build_progress(
        cache,
        source,
        manifest,
        version,
        &mut on_hash_progress,
        |_| {},
    )
}

pub fn import_seed_model_from_file_with_build_progress(
    cache: &ShardCache,
    source: &Path,
    manifest: &ModelManifest,
    version: impl Into<String>,
    mut on_hash_progress: impl FnMut(u64, u64),
    mut on_shard_progress: impl FnMut(SeedShardBuildProgress),
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
    let temp_root = cache
        .config
        .root
        .join("tmp")
        .join(format!("build-{}", now_unix_ms()));
    fs::create_dir_all(&temp_root)
        .with_context(|| format!("failed to create {}", temp_root.display()))?;
    let mut records = Vec::with_capacity(ranges.len());
    let shard_count = ranges.len();

    for (index, layers) in ranges.into_iter().enumerate() {
        let shard_index = index + 1;
        let shard_path = temp_root.join(format!(
            "{}-{}-{}.gguf",
            sanitize(&manifest.model_id),
            layers.start,
            layers.end
        ));
        let shard_summary = write_layer_shard_with_progress(
            source,
            &shard_path,
            layers,
            manifest.layer_count,
            |written_bytes, total_bytes| {
                on_shard_progress(SeedShardBuildProgress {
                    shard_index,
                    shard_count,
                    layers,
                    written_bytes,
                    total_bytes,
                });
            },
        )?;
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
            payload_kind: PAYLOAD_KIND_INFERNET_SHARD.to_owned(),
        };
        let record = cache.import_physical_shard_file(
            &shard_summary.path,
            manifest.model_id.clone(),
            layers,
            version.clone(),
            shard_manifest,
        )?;
        records.push(record);
    }
    let _ = fs::remove_dir_all(&temp_root);

    Ok(SeededModelSummary {
        model_id: manifest.model_id.clone(),
        display_name: manifest.display_name.clone(),
        source_path: source.to_path_buf(),
        source_checksum,
        source_size_bytes,
        shard_count: records.len(),
        metadata_only: false,
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

    #[test]
    fn executable_source_path_finds_downloaded_source_cache() {
        let temp = std::env::temp_dir().join(format!(
            "infernet-source-cache-test-{}",
            uuid::Uuid::new_v4()
        ));
        let config = ShardCacheConfig::new(temp.join("shards"));
        let source_checksum = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let cached_source = source_cache_path(&config, "gemma", source_checksum);
        fs::create_dir_all(cached_source.parent().unwrap()).unwrap();
        fs::write(&cached_source, b"gguf bytes").unwrap();
        let manifest = SeedShardManifest {
            model_id: "gemma".to_owned(),
            display_name: "Gemma".to_owned(),
            architecture: "gemma".to_owned(),
            layer_count: 8,
            hidden_size: 16,
            activation_dtype: "f16".to_owned(),
            runtime_kind: infernet_model::RuntimeKind::LlamaCpp,
            layers: LayerRange::new(0, 8).unwrap(),
            tokenizer: TokenizerCompatibility {
                family: "gemma".to_owned(),
                checksum: None,
            },
            metadata: ShardMetadata {
                architecture: "gemma".to_owned(),
                quantization: Some("IQ4_XS".to_owned()),
                source_checksum: Some(source_checksum.to_owned()),
                protocol_version: PROTOCOL_VERSION,
            },
            source: SeedSourceMetadata {
                path: temp.join("missing-source.gguf").display().to_string(),
                checksum_sha256: source_checksum.to_owned(),
                file_size_bytes: 9,
            },
            shard_hash: "hash".to_owned(),
            payload_kind: "metadata-only".to_owned(),
        };

        assert_eq!(
            executable_source_path_for_manifest(&config, &manifest),
            Some(cached_source)
        );

        let _ = fs::remove_dir_all(temp);
    }

    #[test]
    fn infernet_shards_are_stored_as_packages() {
        let temp =
            std::env::temp_dir().join(format!("infernet-package-test-{}", uuid::Uuid::new_v4()));
        let source = temp.join("layer.gguf");
        fs::create_dir_all(&temp).unwrap();
        fs::write(&source, b"layer tensor payload").unwrap();
        let cache = ShardCache::new(ShardCacheConfig::new(temp.join("cache"))).unwrap();
        let layers = LayerRange::new(3, 4).unwrap();
        let manifest = SeedShardManifest {
            model_id: "gemma".to_owned(),
            display_name: "Gemma".to_owned(),
            architecture: "gemma".to_owned(),
            layer_count: 48,
            hidden_size: 3840,
            activation_dtype: "f16".to_owned(),
            runtime_kind: infernet_model::RuntimeKind::LlamaCpp,
            layers,
            tokenizer: TokenizerCompatibility {
                family: "gemma".to_owned(),
                checksum: None,
            },
            metadata: ShardMetadata {
                architecture: "gemma".to_owned(),
                quantization: Some("IQ4_XS".to_owned()),
                source_checksum: Some("source".to_owned()),
                protocol_version: PROTOCOL_VERSION,
            },
            source: SeedSourceMetadata {
                path: "/models/gemma.gguf".to_owned(),
                checksum_sha256: "source".to_owned(),
                file_size_bytes: 128,
            },
            shard_hash: "hash".to_owned(),
            payload_kind: PAYLOAD_KIND_INFERNET_SHARD.to_owned(),
        };

        let record = cache
            .import_physical_shard_file(&source, "gemma", layers, "v1", manifest.clone())
            .unwrap();

        assert_eq!(
            record.path.file_name().and_then(|value| value.to_str()),
            Some(INFERNET_SHARD_TENSOR_FILE)
        );
        let package_dir = record.path.parent().unwrap();
        assert_eq!(
            package_dir.extension().and_then(|value| value.to_str()),
            Some("infershard")
        );
        let package_manifest_path = package_dir.join(INFERNET_SHARD_MANIFEST_FILE);
        let package_manifest = serde_json::from_slice::<InfernetShardPackageManifest>(
            &fs::read(package_manifest_path).unwrap(),
        )
        .unwrap();

        assert_eq!(
            package_manifest.format_version,
            INFERNET_SHARD_FORMAT_VERSION
        );
        assert_eq!(package_manifest.runtime_abi, INFERNET_SHARD_RUNTIME_ABI);
        assert_eq!(package_manifest.component, "transformer_layer");
        assert_eq!(package_manifest.seed_manifest, manifest);
        assert_eq!(package_manifest.payload.file, INFERNET_SHARD_TENSOR_FILE);
        assert_eq!(
            cache.read_payload(&record.info).unwrap(),
            b"layer tensor payload"
        );
        assert!(is_executable_shard_record(&record));

        let _ = fs::remove_dir_all(temp);
    }
}
