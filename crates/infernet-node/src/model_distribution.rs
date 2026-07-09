use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow, bail};
use infernet_model::{
    LayerRange, ModelManifest, SeedShardManifest, SeedSourceMetadata, ShardMetadata,
    TokenizerCompatibility,
    gguf::{parse_gguf_info, validate_gguf_file, write_layer_shard_with_progress},
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
pub const PAYLOAD_KIND_FULL_MODEL: &str = "infernet-full-model";
pub const INFERNET_SHARD_FORMAT_VERSION: &str = "infernet-shard-v1";
pub const INFERNET_SHARD_RUNTIME_ABI: &str = "infernet-llama-layer-v1";
pub const INFERNET_FULL_MODEL_RUNTIME_ABI: &str = "infernet-llama-full-v1";
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

    pub fn import_verified_external_shard_file(
        &self,
        source: &Path,
        model_id: impl Into<String>,
        layers: LayerRange,
        version: impl Into<String>,
        source_checksum: impl Into<String>,
        manifest: SeedShardManifest,
    ) -> Result<CachedShardRecord> {
        let expected_checksum = source_checksum.into();
        let actual_checksum = sha256_file(source)?;
        if actual_checksum != expected_checksum {
            bail!(
                "checksum verification failed for external shard {}; expected {}, got {}",
                source.display(),
                expected_checksum,
                actual_checksum
            );
        }
        let size_bytes = fs::metadata(source)
            .with_context(|| format!("failed to inspect shard file {}", source.display()))?
            .len();
        let info = ModelShardInfo {
            model_id: model_id.into(),
            layers,
            checksum: actual_checksum,
            size_bytes,
            version: version.into(),
            protocol_version: PROTOCOL_VERSION,
        };

        self.store_verified_file(info, source, Some(manifest), false)
    }

    fn commit_verified_staged_shard_file(
        &self,
        staged: &Path,
        model_id: impl Into<String>,
        layers: LayerRange,
        version: impl Into<String>,
        checksum: impl Into<String>,
        manifest: SeedShardManifest,
    ) -> Result<CachedShardRecord> {
        let size_bytes = fs::metadata(staged)
            .with_context(|| format!("failed to inspect staged shard {}", staged.display()))?
            .len();
        let info = ModelShardInfo {
            model_id: model_id.into(),
            layers,
            checksum: checksum.into(),
            size_bytes,
            version: version.into(),
            protocol_version: PROTOCOL_VERSION,
        };
        self.store_verified_file(info, staged, Some(manifest), true)
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

            let record = match fs::read(&path)
                .with_context(|| format!("failed to read {}", path.display()))
                .and_then(|bytes| {
                    serde_json::from_slice::<CachedShardRecord>(&bytes)
                        .with_context(|| format!("failed to parse {}", path.display()))
                }) {
                Ok(record) => record,
                Err(error) => {
                    eprintln!("ignoring corrupt Infernet cache metadata: {error:#}");
                    quarantine_cache_metadata(&path);
                    continue;
                }
            };
            if let Err(error) = validate_cached_record(&record) {
                eprintln!(
                    "ignoring inconsistent Infernet cache metadata {}: {error:#}",
                    path.display()
                );
                quarantine_cache_metadata(&path);
                continue;
            }
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

    fn retire_model_records_except(
        &self,
        model_id: &str,
        keep: &ModelShardInfo,
    ) -> Result<Vec<CachedShardRecord>> {
        let mut retired = Vec::new();
        for record in self.list()? {
            if record.info.model_id != model_id
                || (record.info.layers == keep.layers
                    && record.info.checksum == keep.checksum
                    && record.info.version == keep.version)
            {
                continue;
            }
            self.remove_record(&record)?;
            retired.push(record);
        }
        Ok(retired)
    }

    fn remove_record(&self, record: &CachedShardRecord) -> Result<()> {
        remove_cached_payload_path(&record.path)?;
        let meta_path = self.meta_path(&record.info);
        if meta_path.exists() {
            fs::remove_file(&meta_path)
                .with_context(|| format!("failed to remove {}", meta_path.display()))?;
        }
        Ok(())
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
        let record = self.write_record(record)?;
        if !record.path.is_file() {
            bail!(
                "cache capacity is too small to retain {}",
                shard_label(&record.info)
            );
        }
        Ok(record)
    }

    fn store_verified_file(
        &self,
        info: ModelShardInfo,
        source: &Path,
        manifest: Option<SeedShardManifest>,
        remove_source_after_copy: bool,
    ) -> Result<CachedShardRecord> {
        if let Some(existing) = self.find(
            &info.model_id,
            info.layers,
            Some(&info.checksum),
            Some(&info.version),
        )? {
            let shares_source_inode = manifest.as_ref().is_some_and(|manifest| {
                paths_share_inode(&existing.path, Path::new(&manifest.source.path))
            });
            if !shares_source_inode
                && sha256_file(&existing.path).is_ok_and(|checksum| checksum == info.checksum)
            {
                if remove_source_after_copy {
                    let _ = fs::remove_file(source);
                }
                return Ok(existing);
            }
        }
        let data_path = self.data_path(&info, manifest.as_ref());
        let cleanup_info = info.clone();
        let data_root = cached_payload_root(&data_path);
        let backup_root = data_root.parent().map(|parent| {
            parent.join(format!(
                ".{}.{}.backup",
                data_root
                    .file_name()
                    .and_then(|value| value.to_str())
                    .unwrap_or("payload"),
                uuid::Uuid::new_v4()
            ))
        });
        let backup_root = match backup_root {
            Some(backup) if data_root.exists() => {
                fs::rename(&data_root, &backup).with_context(|| {
                    format!(
                        "failed to stage existing cache payload {} for replacement",
                        data_root.display()
                    )
                })?;
                Some(backup)
            }
            _ => None,
        };
        let meta_path = self.meta_path(&cleanup_info);
        let previous_meta = match fs::read(&meta_path) {
            Ok(bytes) => Some(bytes),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => {
                if let Some(backup) = &backup_root {
                    let _ = fs::rename(backup, &data_root);
                }
                return Err(error)
                    .with_context(|| format!("failed to read {}", meta_path.display()));
            }
        };
        let result: Result<CachedShardRecord> = (|| {
            fs::create_dir_all(self.config.root.join("data"))?;
            fs::create_dir_all(self.config.root.join("meta"))?;
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
                clone_or_copy_file(source, &data_path, |_, _| {}).with_context(|| {
                    format!(
                        "failed to clone or copy shard file {} to {}",
                        source.display(),
                        data_path.display()
                    )
                })?;
            }
            write_infernet_shard_package_manifest(&info, &data_path, manifest.as_ref())?;

            let record = CachedShardRecord {
                pinned: self.config.pinned_models.contains(&info.model_id),
                info,
                path: data_path.clone(),
                last_access_unix_ms: now_unix_ms(),
                manifest,
            };
            let record = self.write_record(record)?;
            if !record.path.is_file() {
                bail!(
                    "cache capacity is too small to retain {}",
                    shard_label(&record.info)
                );
            }
            Ok(record)
        })();

        if result.is_err() {
            let _ = remove_cached_payload_path(&data_path);
            if let Some(backup) = &backup_root {
                let _ = fs::rename(backup, &data_root);
            }
            match previous_meta {
                Some(bytes) => {
                    let _ = atomic_write(&meta_path, &bytes);
                }
                None => {
                    let _ = fs::remove_file(&meta_path);
                }
            }
        } else if let Some(backup) = &backup_root {
            let _ = remove_any_path(backup);
        }

        result
    }

    fn write_record(&self, record: CachedShardRecord) -> Result<CachedShardRecord> {
        let meta_path = self.meta_path(&record.info);
        let json = serde_json::to_vec_pretty(&record)?;
        atomic_write(&meta_path, &json)?;

        let _ = self.evict_lru_if_needed()?;
        Ok(record)
    }

    fn touch(&self, record: &CachedShardRecord) -> Result<()> {
        let mut updated = record.clone();
        updated.last_access_unix_ms = now_unix_ms();
        atomic_write(
            &self.meta_path(&updated.info),
            &serde_json::to_vec_pretty(&updated)?,
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
        Some(PAYLOAD_KIND_INFERNET_SHARD | PAYLOAD_KIND_FULL_MODEL) => "infershard",
        Some(PAYLOAD_KIND_GGUF_SHARD) => "gguf",
        _ => "shard",
    }
}

fn validate_cached_record(record: &CachedShardRecord) -> Result<()> {
    if !record.path.is_file() {
        bail!("cached payload {} is missing", record.path.display());
    }
    let actual_size = fs::metadata(&record.path)?.len();
    if actual_size != record.info.size_bytes {
        bail!(
            "cached payload size mismatch for {}; expected {}, got {}",
            record.path.display(),
            record.info.size_bytes,
            actual_size
        );
    }
    if let Some(manifest) = &record.manifest {
        if manifest.model_id != record.info.model_id || manifest.layers != record.info.layers {
            bail!("cache record identity does not match its shard manifest");
        }
        manifest.layers.validate_for_model(manifest.layer_count)?;
        if manifest.metadata.architecture != manifest.architecture {
            bail!("cache record architecture metadata is inconsistent");
        }
        if manifest.metadata.source_checksum.as_deref()
            != Some(manifest.source.checksum_sha256.as_str())
        {
            bail!("cache record source checksums are inconsistent");
        }
    }
    Ok(())
}

fn quarantine_cache_metadata(path: &Path) {
    let Some(parent) = path.parent() else {
        return;
    };
    let corrupt_dir = parent.join("corrupt");
    if fs::create_dir_all(&corrupt_dir).is_err() {
        return;
    }
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("record.json");
    let destination = corrupt_dir.join(format!("{name}.{}.bad", uuid::Uuid::new_v4()));
    let _ = fs::rename(path, destination);
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("{} has no parent directory", path.display()))?;
    fs::create_dir_all(parent).with_context(|| format!("failed to create {}", parent.display()))?;
    let temp_path = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("record"),
        uuid::Uuid::new_v4()
    ));
    let result = (|| -> Result<()> {
        let mut file = fs::File::create(&temp_path)
            .with_context(|| format!("failed to create {}", temp_path.display()))?;
        file.write_all(bytes)
            .with_context(|| format!("failed to write {}", temp_path.display()))?;
        file.flush()
            .with_context(|| format!("failed to flush {}", temp_path.display()))?;
        file.sync_all()
            .with_context(|| format!("failed to sync {}", temp_path.display()))?;
        fs::rename(&temp_path, path).with_context(|| {
            format!(
                "failed to atomically replace {} with {}",
                path.display(),
                temp_path.display()
            )
        })?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp_path);
    }
    result
}

fn cached_payload_root(path: &Path) -> PathBuf {
    infernet_shard_package_dir(path)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| path.to_path_buf())
}

#[cfg(unix)]
fn paths_share_inode(left: &Path, right: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;

    match (fs::metadata(left), fs::metadata(right)) {
        (Ok(left), Ok(right)) => left.dev() == right.dev() && left.ino() == right.ino(),
        _ => false,
    }
}

#[cfg(not(unix))]
fn paths_share_inode(_left: &Path, _right: &Path) -> bool {
    // New imports never hard-link. On platforms where std does not expose a
    // stable file identity, replace an existing exact package on re-import.
    true
}

fn clone_or_copy_file(
    source: &Path,
    destination: &Path,
    mut on_progress: impl FnMut(u64, u64),
) -> Result<u64> {
    let total_bytes = fs::metadata(source)
        .with_context(|| format!("failed to inspect {}", source.display()))?
        .len();
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    if destination.exists() {
        fs::remove_file(destination)
            .with_context(|| format!("failed to replace {}", destination.display()))?;
    }
    on_progress(0, total_bytes);

    #[cfg(target_os = "macos")]
    if clonefile_macos(source, destination) {
        on_progress(total_bytes, total_bytes);
        return Ok(total_bytes);
    }

    let mut input =
        fs::File::open(source).with_context(|| format!("failed to open {}", source.display()))?;
    let mut output = fs::File::create(destination)
        .with_context(|| format!("failed to create {}", destination.display()))?;
    let mut buffer = vec![0_u8; 8 * 1024 * 1024];
    let mut written = 0_u64;
    loop {
        let read = input
            .read(&mut buffer)
            .with_context(|| format!("failed to read {}", source.display()))?;
        if read == 0 {
            break;
        }
        output
            .write_all(&buffer[..read])
            .with_context(|| format!("failed to write {}", destination.display()))?;
        written = written.saturating_add(read as u64);
        on_progress(written.min(total_bytes), total_bytes);
    }
    output
        .flush()
        .with_context(|| format!("failed to flush {}", destination.display()))?;
    if written != total_bytes {
        bail!(
            "copy size mismatch for {}; expected {}, wrote {}",
            source.display(),
            total_bytes,
            written
        );
    }
    Ok(written)
}

#[cfg(target_os = "macos")]
fn clonefile_macos(source: &Path, destination: &Path) -> bool {
    use std::{ffi::CString, os::unix::ffi::OsStrExt};

    unsafe extern "C" {
        fn clonefile(src: *const std::ffi::c_char, dst: *const std::ffi::c_char, flags: u32)
        -> i32;
    }

    let Ok(source) = CString::new(source.as_os_str().as_bytes()) else {
        return false;
    };
    let Ok(destination) = CString::new(destination.as_os_str().as_bytes()) else {
        return false;
    };
    // clonefile creates a copy-on-write APFS clone: separate inode/contents,
    // near-zero initial physical cost, and no mutation link to the user file.
    unsafe { clonefile(source.as_ptr(), destination.as_ptr(), 0) == 0 }
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

fn remove_any_path(path: &Path) -> Result<()> {
    if path.is_dir() {
        fs::remove_dir_all(path).with_context(|| format!("failed to remove {}", path.display()))?;
    } else if path.exists() {
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
    let Some(manifest) = manifest.filter(|manifest| {
        matches!(
            manifest.payload_kind.as_str(),
            PAYLOAD_KIND_INFERNET_SHARD | PAYLOAD_KIND_FULL_MODEL
        )
    }) else {
        return Ok(());
    };
    let Some(package_dir) = infernet_shard_package_dir(payload_path) else {
        return Ok(());
    };

    let full_model = manifest.layers.start == 0 && manifest.layers.end == manifest.layer_count;
    let package_manifest = InfernetShardPackageManifest {
        format_version: INFERNET_SHARD_FORMAT_VERSION.to_owned(),
        runtime_abi: if full_model {
            INFERNET_FULL_MODEL_RUNTIME_ABI
        } else {
            INFERNET_SHARD_RUNTIME_ABI
        }
        .to_owned(),
        component: if full_model {
            "full_model"
        } else {
            "transformer_layer"
        }
        .to_owned(),
        seed_manifest: manifest.clone(),
        payload: InfernetShardPayloadManifest {
            kind: "gguf_tensor_payload".to_owned(),
            file: INFERNET_SHARD_TENSOR_FILE.to_owned(),
            checksum_sha256: info.checksum.clone(),
            size_bytes: info.size_bytes,
        },
    };
    let path = package_dir.join(INFERNET_SHARD_MANIFEST_FILE);
    atomic_write(&path, &serde_json::to_vec_pretty(&package_manifest)?)
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
        let stable_payload = match manifest.runtime_kind {
            infernet_model::RuntimeKind::Demo => matches!(
                manifest.payload_kind.as_str(),
                PAYLOAD_KIND_GGUF_SHARD | PAYLOAD_KIND_INFERNET_SHARD
            ),
            infernet_model::RuntimeKind::LlamaCpp => {
                manifest.payload_kind == PAYLOAD_KIND_FULL_MODEL
                    && manifest.layers.start == 0
                    && manifest.layers.end == manifest.layer_count
            }
        };
        record.path.is_file()
            && stable_payload
            && record.info.model_id == manifest.model_id
            && record.info.layers == manifest.layers
    })
}

pub fn seed_manifest_for_network(manifest: &SeedShardManifest) -> SeedShardManifest {
    let mut network_manifest = manifest.clone();
    // Absolute source paths can contain usernames and private directory names.
    // They are useful only on the importing machine and must never be gossiped.
    network_manifest.source.path.clear();
    network_manifest
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
    if source_size_bytes > cache.config().max_storage_bytes {
        bail!(
            "model is {} bytes, larger than the configured Infernet cache limit of {} bytes",
            source_size_bytes,
            cache.config().max_storage_bytes
        );
    }
    let version = version.into();
    let ranges = manifest.automatic_layer_plan()?;
    validate_gguf_matches_manifest(source, manifest)?;
    let full_range = LayerRange::new(0, manifest.layer_count)?;
    if ranges.len() == 1 && ranges[0] == full_range {
        let temp_root = cache
            .config
            .root
            .join("tmp")
            .join(format!("import-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&temp_root)
            .with_context(|| format!("failed to create {}", temp_root.display()))?;
        let staged = temp_root.join("model.gguf");
        let result = (|| -> Result<(CachedShardRecord, String)> {
            clone_or_copy_file(source, &staged, |written_bytes, total_bytes| {
                on_shard_progress(SeedShardBuildProgress {
                    shard_index: 1,
                    shard_count: 1,
                    layers: full_range,
                    written_bytes,
                    total_bytes,
                });
            })?;
            let staged_size = fs::metadata(&staged)
                .with_context(|| format!("failed to inspect {}", staged.display()))?
                .len();
            if staged_size != source_size_bytes {
                bail!(
                    "model changed while it was being imported; expected {} bytes, staged {} bytes",
                    source_size_bytes,
                    staged_size
                );
            }
            validate_gguf_matches_manifest(&staged, manifest)?;
            let source_checksum =
                sha256_file_with_progress(&staged, staged_size, &mut on_hash_progress)?;
            let source_metadata = SeedSourceMetadata {
                path: source.display().to_string(),
                checksum_sha256: source_checksum.clone(),
                file_size_bytes: staged_size,
            };
            let record = cache.commit_verified_staged_shard_file(
                &staged,
                manifest.model_id.clone(),
                full_range,
                version.clone(),
                source_checksum.clone(),
                seed_shard_manifest(manifest, full_range, &source_metadata, &source_checksum),
            )?;
            Ok((record, source_checksum))
        })();
        let _ = fs::remove_dir_all(&temp_root);
        let (record, source_checksum) = result?;

        // Commit first, then retire unsupported legacy layer splits. A failed
        // replacement therefore never destroys the last usable package.
        cache.retire_model_records_except(&manifest.model_id, &record.info)?;

        return Ok(SeededModelSummary {
            model_id: manifest.model_id.clone(),
            display_name: manifest.display_name.clone(),
            source_path: source.to_path_buf(),
            source_checksum,
            source_size_bytes,
            shard_count: 1,
            metadata_only: false,
            records: vec![record],
        });
    }
    let source_checksum =
        sha256_file_with_progress(source, source_size_bytes, &mut on_hash_progress)?;
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
    let preexisting_records = cache
        .list()?
        .into_iter()
        .map(|record| {
            (
                record.info.model_id,
                record.info.layers,
                record.info.checksum,
                record.info.version,
            )
        })
        .collect::<Vec<_>>();
    let mut records = Vec::with_capacity(ranges.len());
    let shard_count = ranges.len();

    let build_result = (|| -> Result<()> {
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
            let shard_manifest =
                seed_shard_manifest(manifest, layers, &source_metadata, &source_checksum);
            let record = cache.import_physical_shard_file(
                &shard_summary.path,
                manifest.model_id.clone(),
                layers,
                version.clone(),
                shard_manifest,
            )?;
            records.push(record);
        }
        Ok(())
    })();
    let _ = fs::remove_dir_all(&temp_root);
    if let Err(error) = build_result {
        for record in &records {
            let existed_before =
                preexisting_records
                    .iter()
                    .any(|(model_id, layers, checksum, version)| {
                        model_id == &record.info.model_id
                            && layers == &record.info.layers
                            && checksum == &record.info.checksum
                            && version == &record.info.version
                    });
            if !existed_before {
                let _ = cache.remove_record(record);
            }
        }
        return Err(error);
    }

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

fn seed_shard_manifest(
    manifest: &ModelManifest,
    layers: LayerRange,
    source: &SeedSourceMetadata,
    source_checksum: &str,
) -> SeedShardManifest {
    SeedShardManifest {
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
            source_checksum: Some(source_checksum.to_owned()),
            protocol_version: PROTOCOL_VERSION,
        },
        source: source.clone(),
        shard_hash: seed_shard_hash(&manifest.model_id, layers, source_checksum),
        payload_kind: if manifest.runtime_kind == infernet_model::RuntimeKind::LlamaCpp
            && layers.start == 0
            && layers.end == manifest.layer_count
        {
            PAYLOAD_KIND_FULL_MODEL
        } else {
            PAYLOAD_KIND_INFERNET_SHARD
        }
        .to_owned(),
    }
}

fn validate_gguf_matches_manifest(source: &Path, manifest: &ModelManifest) -> Result<()> {
    validate_gguf_file(source)
        .with_context(|| format!("invalid GGUF structure in {}", source.display()))?;
    let info = parse_gguf_info(source)
        .with_context(|| format!("failed to inspect GGUF metadata in {}", source.display()))?;
    let architecture = info
        .architecture
        .as_deref()
        .ok_or_else(|| anyhow!("GGUF is missing general.architecture"))?;
    if architecture != manifest.architecture {
        bail!(
            "GGUF architecture mismatch: file reports {}, manifest expects {}",
            architecture,
            manifest.architecture
        );
    }
    let layer_count = info
        .layer_count
        .ok_or_else(|| anyhow!("GGUF is missing {architecture}.block_count"))?;
    if layer_count != manifest.layer_count {
        bail!(
            "GGUF layer-count mismatch: file reports {}, manifest expects {}",
            layer_count,
            manifest.layer_count
        );
    }
    let hidden_size = info
        .hidden_size
        .ok_or_else(|| anyhow!("GGUF is missing {architecture}.embedding_length"))?;
    if hidden_size != manifest.hidden_size {
        bail!(
            "GGUF hidden-size mismatch: file reports {}, manifest expects {}",
            hidden_size,
            manifest.hidden_size
        );
    }
    Ok(())
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
    let mut file =
        fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0; 1024 * 64];
    let mut read_total = 0_u64;
    let mut last_reported = 0_u64;
    const PROGRESS_INTERVAL_BYTES: u64 = 64 * 1024 * 1024;

    on_progress(read_total, total_bytes);

    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        read_total = read_total.saturating_add(read as u64);
        if read_total >= total_bytes
            || read_total.saturating_sub(last_reported) >= PROGRESS_INTERVAL_BYTES
        {
            on_progress(read_total, total_bytes);
            last_reported = read_total;
        }
    }

    if read_total != last_reported {
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
        let layers = LayerRange::new(0, 48).unwrap();
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
            payload_kind: PAYLOAD_KIND_FULL_MODEL.to_owned(),
        };
        let checksum = sha256_file(&source).unwrap();

        let record = cache
            .import_verified_external_shard_file(
                &source,
                "gemma",
                layers,
                "v1",
                checksum,
                manifest.clone(),
            )
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
        assert_eq!(
            package_manifest.runtime_abi,
            INFERNET_FULL_MODEL_RUNTIME_ABI
        );
        assert_eq!(package_manifest.component, "full_model");
        assert_eq!(package_manifest.seed_manifest, manifest);
        assert_eq!(package_manifest.payload.file, INFERNET_SHARD_TENSOR_FILE);
        assert_eq!(
            cache.read_payload(&record.info).unwrap(),
            b"layer tensor payload"
        );
        assert!(is_executable_shard_record(&record));
        assert!(
            source.is_file(),
            "external source must not be moved or deleted"
        );

        let _ = fs::remove_dir_all(temp);
    }

    #[test]
    fn full_model_import_creates_one_isolated_complete_package() {
        let temp = std::env::temp_dir().join(format!(
            "infernet-full-model-import-test-{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&temp).unwrap();
        let source = temp.join("qwen35-test.gguf");
        write_test_gguf(&source, "qwen35", 32, 1024).unwrap();
        let cache = ShardCache::new(ShardCacheConfig::new(temp.join("cache"))).unwrap();
        let manifest = ModelManifest {
            model_id: "qwen35-test".to_owned(),
            display_name: "Qwen 3.5 Test".to_owned(),
            architecture: "qwen35".to_owned(),
            layer_count: 32,
            hidden_size: 1024,
            activation_dtype: "f16".to_owned(),
            quantization: Some("Q4_K_M".to_owned()),
            runtime_kind: infernet_model::RuntimeKind::LlamaCpp,
        };

        let summary = import_seed_model_from_file(&cache, &source, &manifest, "v1").unwrap();

        assert_eq!(summary.shard_count, 1);
        assert_eq!(summary.records.len(), 1);
        assert_eq!(
            summary.records[0].info.layers,
            LayerRange::new(0, 32).unwrap()
        );
        assert_eq!(
            summary.records[0].info.size_bytes,
            summary.source_size_bytes
        );
        assert!(source.is_file());
        assert_eq!(
            fs::read(&summary.records[0].path).unwrap(),
            fs::read(&source).unwrap()
        );

        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            assert_ne!(
                fs::metadata(&summary.records[0].path).unwrap().ino(),
                fs::metadata(&source).unwrap().ino(),
                "the verified cache must not share a mutable inode with the source"
            );
        }

        let cached_before = fs::read(&summary.records[0].path).unwrap();
        fs::write(&source, vec![0_u8; cached_before.len()]).unwrap();
        assert_eq!(fs::read(&summary.records[0].path).unwrap(), cached_before);

        let _ = fs::remove_dir_all(temp);
    }

    #[test]
    fn full_model_import_rejects_corrupt_and_mismatched_gguf() {
        let temp = std::env::temp_dir().join(format!(
            "infernet-full-model-validation-test-{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&temp).unwrap();
        let cache = ShardCache::new(ShardCacheConfig::new(temp.join("cache"))).unwrap();
        let manifest = test_model_manifest("qwen35", 32, 1024);

        let corrupt = temp.join("corrupt.gguf");
        fs::write(&corrupt, b"not a model").unwrap();
        assert!(
            import_seed_model_from_file(&cache, &corrupt, &manifest, "v1")
                .unwrap_err()
                .to_string()
                .contains("GGUF")
        );

        let mismatch = temp.join("mismatch.gguf");
        write_test_gguf(&mismatch, "llama", 32, 1024).unwrap();
        assert!(
            import_seed_model_from_file(&cache, &mismatch, &manifest, "v1")
                .unwrap_err()
                .to_string()
                .contains("architecture mismatch")
        );
        assert!(cache.list().unwrap().is_empty());

        let _ = fs::remove_dir_all(temp);
    }

    #[cfg(unix)]
    #[test]
    fn reimport_replaces_a_legacy_mutable_hard_link() {
        use std::os::unix::fs::MetadataExt;

        let temp = std::env::temp_dir().join(format!(
            "infernet-hard-link-repair-test-{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&temp).unwrap();
        let source = temp.join("qwen35.gguf");
        write_test_gguf(&source, "qwen35", 4, 8).unwrap();
        let cache = ShardCache::new(ShardCacheConfig::new(temp.join("cache"))).unwrap();
        let manifest = test_model_manifest("qwen35", 4, 8);
        let checksum = sha256_file(&source).unwrap();
        let layers = LayerRange::new(0, 4).unwrap();
        let info = ModelShardInfo {
            model_id: manifest.model_id.clone(),
            layers,
            checksum: checksum.clone(),
            size_bytes: fs::metadata(&source).unwrap().len(),
            version: "v1".to_owned(),
            protocol_version: PROTOCOL_VERSION,
        };
        let source_metadata = SeedSourceMetadata {
            path: source.display().to_string(),
            checksum_sha256: checksum.clone(),
            file_size_bytes: info.size_bytes,
        };
        let seed_manifest = seed_shard_manifest(&manifest, layers, &source_metadata, &checksum);
        let data_path = cache.data_path(&info, Some(&seed_manifest));
        prepare_data_path(&data_path).unwrap();
        fs::hard_link(&source, &data_path).unwrap();
        write_infernet_shard_package_manifest(&info, &data_path, Some(&seed_manifest)).unwrap();
        cache
            .write_record(CachedShardRecord {
                info: info.clone(),
                path: data_path.clone(),
                last_access_unix_ms: now_unix_ms(),
                pinned: false,
                manifest: Some(seed_manifest),
            })
            .unwrap();
        assert_eq!(
            fs::metadata(&source).unwrap().ino(),
            fs::metadata(&data_path).unwrap().ino()
        );

        let summary = import_seed_model_from_file(&cache, &source, &manifest, "v1").unwrap();

        assert_ne!(
            fs::metadata(&source).unwrap().ino(),
            fs::metadata(&summary.records[0].path).unwrap().ino()
        );
        assert!(source.is_file());

        let _ = fs::remove_dir_all(temp);
    }

    #[test]
    fn successful_full_import_retires_legacy_splits_without_global_temp_deletion() {
        let temp = std::env::temp_dir().join(format!(
            "infernet-full-model-recovery-test-{}",
            uuid::Uuid::new_v4()
        ));
        let config = ShardCacheConfig::new(temp.join("cache"));
        let cache = ShardCache::new(config.clone()).unwrap();
        let manifest = test_model_manifest("qwen35", 4, 8);
        let legacy_source = SeedSourceMetadata {
            path: "/old/model.gguf".to_owned(),
            checksum_sha256: "legacy-source".to_owned(),
            file_size_bytes: 64,
        };
        let mut legacy_paths = Vec::new();
        for start in 0..4 {
            let layers = LayerRange::new(start, start + 1).unwrap();
            let payload = vec![start as u8; 16];
            let info = ModelShardInfo {
                model_id: manifest.model_id.clone(),
                layers,
                checksum: sha256_bytes(&payload),
                size_bytes: payload.len() as u64,
                version: "legacy".to_owned(),
                protocol_version: PROTOCOL_VERSION,
            };
            let record = cache
                .store_verified_payload(
                    info,
                    payload,
                    Some(seed_shard_manifest(
                        &manifest,
                        layers,
                        &legacy_source,
                        "legacy-source",
                    )),
                )
                .unwrap();
            legacy_paths.push(record.path);
        }
        let stale_build = config.root.join("tmp/build-interrupted");
        fs::create_dir_all(&stale_build).unwrap();
        fs::write(stale_build.join("partial.gguf"), vec![1_u8; 128]).unwrap();
        let source = temp.join("qwen35.gguf");
        write_test_gguf(&source, "qwen35", 4, 8).unwrap();

        let summary = import_seed_model_from_file(&cache, &source, &manifest, "v2").unwrap();

        assert_eq!(summary.records.len(), 1);
        assert_eq!(cache.list().unwrap().len(), 1);
        assert!(
            stale_build.exists(),
            "an import must not delete another process's unowned temp directory"
        );
        assert!(legacy_paths.into_iter().all(|path| !path.exists()));

        let _ = fs::remove_dir_all(temp);
    }

    #[test]
    fn corrupt_metadata_does_not_brick_a_later_import() {
        let temp = std::env::temp_dir().join(format!(
            "infernet-corrupt-metadata-recovery-test-{}",
            uuid::Uuid::new_v4()
        ));
        let config = ShardCacheConfig::new(temp.join("cache"));
        let cache = ShardCache::new(config.clone()).unwrap();
        fs::write(config.root.join("meta/torn.json"), b"{\"truncated\"").unwrap();
        let source = temp.join("qwen35.gguf");
        write_test_gguf(&source, "qwen35", 4, 8).unwrap();

        let summary = import_seed_model_from_file(
            &cache,
            &source,
            &test_model_manifest("qwen35", 4, 8),
            "v1",
        )
        .unwrap();

        assert_eq!(summary.records.len(), 1);
        assert_eq!(cache.list().unwrap().len(), 1);
        assert!(config.root.join("meta/corrupt").is_dir());

        let _ = fs::remove_dir_all(temp);
    }

    fn test_model_manifest(
        architecture: &str,
        layer_count: u32,
        hidden_size: usize,
    ) -> ModelManifest {
        ModelManifest {
            model_id: format!("{architecture}-test"),
            display_name: format!("{architecture} test"),
            architecture: architecture.to_owned(),
            layer_count,
            hidden_size,
            activation_dtype: "f16".to_owned(),
            quantization: Some("Q4_K_M".to_owned()),
            runtime_kind: infernet_model::RuntimeKind::LlamaCpp,
        }
    }

    fn write_test_gguf(
        path: &Path,
        architecture: &str,
        layer_count: u32,
        hidden_size: usize,
    ) -> Result<()> {
        use std::io::Seek;

        fn write_u32(output: &mut fs::File, value: u32) -> Result<()> {
            output.write_all(&value.to_le_bytes())?;
            Ok(())
        }
        fn write_u64(output: &mut fs::File, value: u64) -> Result<()> {
            output.write_all(&value.to_le_bytes())?;
            Ok(())
        }
        fn write_string(output: &mut fs::File, value: &str) -> Result<()> {
            write_u64(output, value.len() as u64)?;
            output.write_all(value.as_bytes())?;
            Ok(())
        }

        let mut output = fs::File::create(path)?;
        output.write_all(b"GGUF")?;
        write_u32(&mut output, 3)?;
        write_u64(&mut output, 1)?;
        write_u64(&mut output, 4)?;
        write_string(&mut output, "general.architecture")?;
        write_u32(&mut output, 8)?;
        write_string(&mut output, architecture)?;
        write_string(&mut output, &format!("{architecture}.block_count"))?;
        write_u32(&mut output, 4)?;
        write_u32(&mut output, layer_count)?;
        write_string(&mut output, &format!("{architecture}.embedding_length"))?;
        write_u32(&mut output, 4)?;
        write_u32(&mut output, hidden_size as u32)?;
        write_string(&mut output, "general.alignment")?;
        write_u32(&mut output, 4)?;
        write_u32(&mut output, 32)?;
        write_string(&mut output, "token_embd.weight")?;
        write_u32(&mut output, 1)?;
        write_u64(&mut output, 1)?;
        write_u32(&mut output, 0)?;
        write_u64(&mut output, 0)?;
        let position = output.stream_position()?;
        let padding = (32 - (position % 32)) % 32;
        output.write_all(&vec![0_u8; padding as usize])?;
        output.write_all(&1_f32.to_le_bytes())?;
        Ok(())
    }
}
