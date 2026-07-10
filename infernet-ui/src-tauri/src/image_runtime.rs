use std::{
    env, fs,
    path::{Path, PathBuf},
    time::UNIX_EPOCH,
};

use futures::StreamExt;
use infernet_node::{ShardCacheConfig, find_sd_cli_binary, sha256_file};
use infernet_protocol::{ImageComponentRole, ModelComponentInfo};
use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;

pub const IMAGE_MODEL_ID: &str = "infernet-image-v1";
pub const IMAGE_RELEASE_ID: &str = "infernet-image-v1-z-image-turbo-q4-k-m";
pub const IMAGE_RELEASE_VERSION: &str = "2026-07-10.1";
pub const IMAGE_RUNTIME_ABI: &str = "infernet-sdcpp-image-v1";
pub const IMAGE_TOTAL_BYTES: u64 = 7_850_198_884;

const VERIFICATION_MARKER_FILE: &str = "verified-package-v1.json";
const PACKAGE_FINGERPRINT: &str = concat!(
    "infernet-image-v1|infernet-sdcpp-image-v1|",
    "e6494f87de6abaf6a561924f50317a5f271fc34bb4222aabbd801197df8f7daa|",
    "3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597|",
    "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38",
);

#[derive(Debug, Clone)]
pub struct ImageComponentSpec {
    pub component_id: &'static str,
    pub role: ImageComponentRole,
    pub file_name: &'static str,
    pub repository: &'static str,
    pub revision: &'static str,
    pub artifact: &'static str,
    pub sha256: &'static str,
    pub size_bytes: u64,
    pub environment_override: &'static str,
}

pub const IMAGE_COMPONENTS: &[ImageComponentSpec] = &[
    ImageComponentSpec {
        component_id: "z-image-turbo-q4-k-m",
        role: ImageComponentRole::DiffusionTransformer,
        file_name: "z-image-turbo-Q4_K_M.gguf",
        repository: "unsloth/Z-Image-Turbo-GGUF",
        revision: "6c80814333b7b6a70a2e5b469a7c6437ce65de0f",
        artifact: "z-image-turbo-Q4_K_M.gguf",
        sha256: "e6494f87de6abaf6a561924f50317a5f271fc34bb4222aabbd801197df8f7daa",
        size_bytes: 5_017_613_376,
        environment_override: "INFERNET_IMAGE_DIFFUSION_MODEL",
    },
    ImageComponentSpec {
        component_id: "qwen3-4b-instruct-2507-q4-k-m",
        role: ImageComponentRole::TextEncoder,
        file_name: "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        repository: "unsloth/Qwen3-4B-Instruct-2507-GGUF",
        revision: "a06e946bb6b655725eafa393f4a9745d460374c9",
        artifact: "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        sha256: "3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597",
        size_bytes: 2_497_281_120,
        environment_override: "INFERNET_IMAGE_TEXT_ENCODER",
    },
    ImageComponentSpec {
        component_id: "z-image-ae",
        role: ImageComponentRole::Vae,
        file_name: "ae.safetensors",
        repository: "Comfy-Org/z_image_turbo",
        revision: "d24c4cf2a0cd98a42f23467e27e3d76ee9438b8e",
        artifact: "split_files/vae/ae.safetensors",
        sha256: "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38",
        size_bytes: 335_304_388,
        environment_override: "INFERNET_IMAGE_VAE",
    },
];

#[derive(Debug, Clone)]
pub struct ImagePackagePaths {
    pub diffusion_model: PathBuf,
    pub text_encoder: PathBuf,
    pub vae: PathBuf,
}

impl ImagePackagePaths {
    fn as_array(&self) -> [&Path; 3] {
        [&self.diffusion_model, &self.text_encoder, &self.vae]
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageRuntimeStatus {
    pub model_id: String,
    pub release_id: String,
    pub release_version: String,
    pub runtime_abi: String,
    pub quantization: String,
    pub runtime_available: bool,
    pub busy: bool,
    pub installed: bool,
    pub verified: bool,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub status: String,
}

#[derive(Debug, Clone)]
pub struct ImageInstallProgress {
    pub stage: &'static str,
    pub detail: String,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct VerificationMarker {
    fingerprint: String,
    files: Vec<VerifiedFile>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct VerifiedFile {
    path: PathBuf,
    size_bytes: u64,
    modified_unix_ms: u64,
}

pub fn image_runtime_status(cache_config: &ShardCacheConfig) -> ImageRuntimeStatus {
    let paths = resolve_package_paths(cache_config);
    let installed = package_has_exact_sizes(&paths);
    let verified = installed && package_is_verified(cache_config, &paths);
    let runtime_available = find_sd_cli_binary().is_some();
    let downloaded_bytes = if installed {
        IMAGE_TOTAL_BYTES
    } else {
        official_downloaded_bytes(cache_config)
    };
    let status = match (runtime_available, installed, verified) {
        (false, _, _) => "The Infernet Image runtime is not prepared for this platform.".to_owned(),
        (true, false, _) => format!(
            "Install the verified Infernet Image package ({:.2} GiB).",
            IMAGE_TOTAL_BYTES as f64 / 1024_f64.powi(3)
        ),
        (true, true, false) => "The image package must be verified before generation.".to_owned(),
        (true, true, true) => "Infernet Image is ready on this computer.".to_owned(),
    };

    ImageRuntimeStatus {
        model_id: IMAGE_MODEL_ID.to_owned(),
        release_id: IMAGE_RELEASE_ID.to_owned(),
        release_version: IMAGE_RELEASE_VERSION.to_owned(),
        runtime_abi: IMAGE_RUNTIME_ABI.to_owned(),
        quantization: "Q4_K_M".to_owned(),
        runtime_available,
        busy: false,
        installed,
        verified,
        downloaded_bytes,
        total_bytes: IMAGE_TOTAL_BYTES,
        status,
    }
}

pub fn advertised_components(cache_config: &ShardCacheConfig) -> Vec<ModelComponentInfo> {
    let paths = resolve_package_paths(cache_config);
    if find_sd_cli_binary().is_none()
        || !package_has_exact_sizes(&paths)
        || !package_is_verified(cache_config, &paths)
    {
        return Vec::new();
    }
    component_infos()
}

pub fn component_infos() -> Vec<ModelComponentInfo> {
    IMAGE_COMPONENTS
        .iter()
        .map(|component| ModelComponentInfo {
            release_id: IMAGE_RELEASE_ID.to_owned(),
            model_id: IMAGE_MODEL_ID.to_owned(),
            component_id: component.component_id.to_owned(),
            role: component.role.clone(),
            checksum: component.sha256.to_owned(),
            size_bytes: component.size_bytes,
            version: IMAGE_RELEASE_VERSION.to_owned(),
            runtime_abi: IMAGE_RUNTIME_ABI.to_owned(),
        })
        .collect()
}

pub fn resolve_package_paths(cache_config: &ShardCacheConfig) -> ImagePackagePaths {
    let official = official_package_paths(cache_config);
    let overridden = ImagePackagePaths {
        diffusion_model: env::var_os(IMAGE_COMPONENTS[0].environment_override)
            .map(PathBuf::from)
            .unwrap_or_else(|| official.diffusion_model.clone()),
        text_encoder: env::var_os(IMAGE_COMPONENTS[1].environment_override)
            .map(PathBuf::from)
            .unwrap_or_else(|| official.text_encoder.clone()),
        vae: env::var_os(IMAGE_COMPONENTS[2].environment_override)
            .map(PathBuf::from)
            .unwrap_or_else(|| official.vae.clone()),
    };
    if overridden.as_array().iter().all(|path| path.is_file()) {
        return overridden;
    }
    if official.as_array().iter().all(|path| path.is_file()) {
        return official;
    }

    let development_root = repository_root().join("target").join(IMAGE_MODEL_ID);
    let development = ImagePackagePaths {
        diffusion_model: development_root.join(IMAGE_COMPONENTS[0].file_name),
        text_encoder: development_root.join(IMAGE_COMPONENTS[1].file_name),
        vae: development_root.join(IMAGE_COMPONENTS[2].file_name),
    };
    if development.as_array().iter().all(|path| path.is_file()) {
        development
    } else {
        overridden
    }
}

pub async fn install_official_package<F>(
    cache_config: &ShardCacheConfig,
    mut progress: F,
) -> anyhow::Result<ImageRuntimeStatus>
where
    F: FnMut(ImageInstallProgress) + Send,
{
    let current_paths = resolve_package_paths(cache_config);
    if package_has_exact_sizes(&current_paths) && package_is_verified(cache_config, &current_paths)
    {
        return Ok(image_runtime_status(cache_config));
    }

    let official_paths = official_package_paths(cache_config);
    if package_has_exact_sizes(&official_paths) {
        progress(ImageInstallProgress {
            stage: "Verifying image package",
            detail: "Checking all three pinned SHA-256 checksums".to_owned(),
            downloaded_bytes: IMAGE_TOTAL_BYTES,
            total_bytes: IMAGE_TOTAL_BYTES,
        });
        match verify_and_mark_package(cache_config, &official_paths, true).await {
            Ok(()) => return Ok(image_runtime_status(cache_config)),
            Err(error) if package_has_exact_sizes(&official_paths) => return Err(error),
            Err(_) => {
                progress(ImageInstallProgress {
                    stage: "Repairing image package",
                    detail: "Replacing a component that failed verification".to_owned(),
                    downloaded_bytes: official_downloaded_bytes(cache_config),
                    total_bytes: IMAGE_TOTAL_BYTES,
                });
            }
        }
    }

    let package_root = official_package_root(cache_config);
    tokio::fs::create_dir_all(&package_root).await?;
    let _ = tokio::fs::remove_file(verification_marker_path(cache_config)).await;
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(20))
        .user_agent(concat!("infernet/", env!("CARGO_PKG_VERSION")))
        .build()?;
    let mut completed_before = 0_u64;

    for component in IMAGE_COMPONENTS {
        let final_path = package_root.join(component.file_name);
        let partial_path = package_root.join(format!("{}.partial", component.file_name));
        if tokio::fs::metadata(&final_path)
            .await
            .is_ok_and(|metadata| metadata.len() != component.size_bytes)
        {
            tokio::fs::remove_file(&final_path).await?;
        }
        if tokio::fs::metadata(&final_path)
            .await
            .is_ok_and(|metadata| metadata.len() == component.size_bytes)
        {
            completed_before += component.size_bytes;
            progress(ImageInstallProgress {
                stage: "Downloading image package",
                detail: format!("{} ready", component_label(component.role.clone())),
                downloaded_bytes: completed_before,
                total_bytes: IMAGE_TOTAL_BYTES,
            });
            continue;
        }

        let mut downloaded = tokio::fs::metadata(&partial_path)
            .await
            .map(|metadata| metadata.len())
            .unwrap_or(0);
        if downloaded > component.size_bytes {
            tokio::fs::remove_file(&partial_path).await?;
            downloaded = 0;
        }
        progress(ImageInstallProgress {
            stage: "Downloading image package",
            detail: component_label(component.role.clone()).to_owned(),
            downloaded_bytes: completed_before + downloaded,
            total_bytes: IMAGE_TOTAL_BYTES,
        });

        if downloaded < component.size_bytes {
            let mut request = client.get(component_download_url(component));
            if downloaded > 0 {
                request = request.header(reqwest::header::RANGE, format!("bytes={downloaded}-"));
            }
            let response = request.send().await?;
            let status = response.status();
            if !status.is_success() {
                anyhow::bail!(
                    "Hugging Face returned HTTP {} for {}",
                    status.as_u16(),
                    component.file_name
                );
            }
            let append = downloaded > 0 && status == reqwest::StatusCode::PARTIAL_CONTENT;
            if append {
                let range_start = response
                    .headers()
                    .get(reqwest::header::CONTENT_RANGE)
                    .and_then(|value| value.to_str().ok())
                    .and_then(content_range_start);
                if range_start != Some(downloaded) {
                    anyhow::bail!("Hugging Face returned an invalid resume range");
                }
            } else if downloaded > 0 {
                downloaded = 0;
            }

            let mut options = tokio::fs::OpenOptions::new();
            options.create(true).write(true);
            if append {
                options.append(true);
            } else {
                options.truncate(true);
            }
            let mut file = options.open(&partial_path).await?;
            let mut stream = response.bytes_stream();
            let mut last_progress_emit = downloaded;
            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                downloaded = downloaded
                    .checked_add(chunk.len() as u64)
                    .ok_or_else(|| anyhow::anyhow!("image package download size overflow"))?;
                if downloaded > component.size_bytes {
                    anyhow::bail!("image package download exceeded its pinned size");
                }
                file.write_all(&chunk).await?;
                if downloaded == component.size_bytes
                    || downloaded.saturating_sub(last_progress_emit) >= 4 * 1024 * 1024
                {
                    progress(ImageInstallProgress {
                        stage: "Downloading image package",
                        detail: component_label(component.role.clone()).to_owned(),
                        downloaded_bytes: completed_before + downloaded,
                        total_bytes: IMAGE_TOTAL_BYTES,
                    });
                    last_progress_emit = downloaded;
                }
            }
            file.flush().await?;
        }
        if downloaded != component.size_bytes {
            anyhow::bail!(
                "{} ended at {} of {} bytes",
                component.file_name,
                downloaded,
                component.size_bytes
            );
        }
        tokio::fs::rename(&partial_path, &final_path).await?;
        completed_before += component.size_bytes;
    }

    progress(ImageInstallProgress {
        stage: "Verifying image package",
        detail: "Checking all three pinned SHA-256 checksums".to_owned(),
        downloaded_bytes: IMAGE_TOTAL_BYTES,
        total_bytes: IMAGE_TOTAL_BYTES,
    });
    let paths = official_package_paths(cache_config);
    verify_and_mark_package(cache_config, &paths, true).await?;
    progress(ImageInstallProgress {
        stage: "Image package ready",
        detail: "Z-Image Turbo Q4_K_M verified".to_owned(),
        downloaded_bytes: IMAGE_TOTAL_BYTES,
        total_bytes: IMAGE_TOTAL_BYTES,
    });
    Ok(image_runtime_status(cache_config))
}

pub async fn ensure_verified_package(
    cache_config: &ShardCacheConfig,
) -> anyhow::Result<ImagePackagePaths> {
    let paths = resolve_package_paths(cache_config);
    if !package_has_exact_sizes(&paths) {
        anyhow::bail!("Install Infernet Image before generating an image");
    }
    if !package_is_verified(cache_config, &paths) {
        verify_and_mark_package(cache_config, &paths, false).await?;
    }
    Ok(paths)
}

fn official_package_root(cache_config: &ShardCacheConfig) -> PathBuf {
    cache_config
        .root
        .join("image-components")
        .join(IMAGE_RELEASE_ID)
}

fn official_package_paths(cache_config: &ShardCacheConfig) -> ImagePackagePaths {
    let root = official_package_root(cache_config);
    ImagePackagePaths {
        diffusion_model: root.join(IMAGE_COMPONENTS[0].file_name),
        text_encoder: root.join(IMAGE_COMPONENTS[1].file_name),
        vae: root.join(IMAGE_COMPONENTS[2].file_name),
    }
}

fn verification_marker_path(cache_config: &ShardCacheConfig) -> PathBuf {
    official_package_root(cache_config).join(VERIFICATION_MARKER_FILE)
}

fn package_has_exact_sizes(paths: &ImagePackagePaths) -> bool {
    paths
        .as_array()
        .into_iter()
        .zip(IMAGE_COMPONENTS)
        .all(|(path, spec)| {
            fs::metadata(path).is_ok_and(|metadata| metadata.len() == spec.size_bytes)
        })
}

fn package_is_verified(cache_config: &ShardCacheConfig, paths: &ImagePackagePaths) -> bool {
    let Ok(bytes) = fs::read(verification_marker_path(cache_config)) else {
        return false;
    };
    let Ok(marker) = serde_json::from_slice::<VerificationMarker>(&bytes) else {
        return false;
    };
    if marker.fingerprint != PACKAGE_FINGERPRINT {
        return false;
    }
    verified_file_records(paths).is_ok_and(|files| files == marker.files)
}

async fn verify_and_mark_package(
    cache_config: &ShardCacheConfig,
    paths: &ImagePackagePaths,
    remove_corrupt_official_files: bool,
) -> anyhow::Result<()> {
    if !package_has_exact_sizes(paths) {
        anyhow::bail!("Infernet Image package has an unexpected component size");
    }
    let mut mismatches = Vec::new();
    for (path, component) in paths.as_array().into_iter().zip(IMAGE_COMPONENTS) {
        let path = path.to_owned();
        let checksum_path = path.clone();
        let actual = tokio::task::spawn_blocking(move || sha256_file(&checksum_path)).await??;
        if actual != component.sha256 {
            if remove_corrupt_official_files {
                let _ = tokio::fs::remove_file(&path).await;
            }
            mismatches.push(format!(
                "{} (expected {}, got {})",
                component.file_name, component.sha256, actual
            ));
        }
    }
    if !mismatches.is_empty() {
        anyhow::bail!("image package checksum mismatch: {}", mismatches.join(", "));
    }

    let marker = VerificationMarker {
        fingerprint: PACKAGE_FINGERPRINT.to_owned(),
        files: verified_file_records(paths)?,
    };
    let marker_path = verification_marker_path(cache_config);
    let temporary_path = marker_path.with_extension("json.tmp");
    tokio::fs::create_dir_all(
        marker_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("image verification marker has no parent"))?,
    )
    .await?;
    let _ = tokio::fs::remove_file(&temporary_path).await;
    tokio::fs::write(&temporary_path, serde_json::to_vec_pretty(&marker)?).await?;
    let _ = tokio::fs::remove_file(&marker_path).await;
    tokio::fs::rename(temporary_path, marker_path).await?;
    Ok(())
}

fn verified_file_records(paths: &ImagePackagePaths) -> anyhow::Result<Vec<VerifiedFile>> {
    paths
        .as_array()
        .into_iter()
        .map(|path| {
            let path = fs::canonicalize(path)?;
            let metadata = fs::metadata(&path)?;
            let modified_unix_ms = metadata
                .modified()?
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
                .try_into()
                .unwrap_or(u64::MAX);
            Ok(VerifiedFile {
                path,
                size_bytes: metadata.len(),
                modified_unix_ms,
            })
        })
        .collect()
}

fn official_downloaded_bytes(cache_config: &ShardCacheConfig) -> u64 {
    let root = official_package_root(cache_config);
    IMAGE_COMPONENTS
        .iter()
        .map(|component| {
            let final_path = root.join(component.file_name);
            let partial_path = root.join(format!("{}.partial", component.file_name));
            fs::metadata(final_path)
                .or_else(|_| fs::metadata(partial_path))
                .map(|metadata| metadata.len().min(component.size_bytes))
                .unwrap_or(0)
        })
        .sum()
}

fn component_download_url(component: &ImageComponentSpec) -> String {
    format!(
        "https://huggingface.co/{}/resolve/{}/{}",
        component.repository, component.revision, component.artifact
    )
}

fn component_label(role: ImageComponentRole) -> &'static str {
    match role {
        ImageComponentRole::DiffusionTransformer => "Diffusion transformer",
        ImageComponentRole::TextEncoder => "Text encoder",
        ImageComponentRole::Vae => "VAE",
    }
}

fn content_range_start(value: &str) -> Option<u64> {
    value
        .strip_prefix("bytes ")?
        .split_once('-')?
        .0
        .parse()
        .ok()
}

fn repository_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_owned)
        .unwrap_or_else(|| PathBuf::from("."))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_contract_has_all_three_exact_components() {
        assert_eq!(IMAGE_COMPONENTS.len(), 3);
        assert_eq!(
            IMAGE_COMPONENTS
                .iter()
                .map(|component| component.size_bytes)
                .sum::<u64>(),
            IMAGE_TOTAL_BYTES
        );
        assert!(component_infos().iter().all(|component| {
            component.release_id == IMAGE_RELEASE_ID
                && component.runtime_abi == IMAGE_RUNTIME_ABI
                && component.checksum.len() == 64
        }));
    }

    #[test]
    fn downloads_are_bound_to_immutable_revisions() {
        for component in IMAGE_COMPONENTS {
            let url = component_download_url(component);
            assert!(url.contains(component.revision));
            assert!(!url.contains("/main/"));
        }
    }
}
