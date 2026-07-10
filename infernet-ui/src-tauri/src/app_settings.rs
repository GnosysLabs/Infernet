use std::{
    fs::{self, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use infernet_node::{
    clear_local_image_rpc_endpoint, clear_local_llama_rpc_endpoint, detect_node_capabilities,
    set_local_rpc_active, set_vram_contribution_limit_bytes, stop_persistent_infernet_workers,
};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, State};
use uuid::Uuid;

use super::{
    ImageRpcServiceState, LlamaRpcServiceState, UiState, cache_config_for_app,
    ensure_image_rpc_service, ensure_llama_rpc_service,
};

const APP_SETTINGS_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AppSettingsDocument {
    version: u32,
    #[serde(default)]
    vram_contribution_limit_bytes: Option<u64>,
}

impl Default for AppSettingsDocument {
    fn default() -> Self {
        Self {
            version: APP_SETTINGS_VERSION,
            vram_contribution_limit_bytes: None,
        }
    }
}

pub(crate) struct AppSettingsStore {
    path: PathBuf,
    document: AppSettingsDocument,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct VramContributionSettings {
    contribution_bytes: u64,
    total_bytes: u64,
    available_bytes: u64,
    compute_backend: String,
    device_name: String,
    unified_memory: bool,
}

impl AppSettingsStore {
    pub(crate) fn open(path: PathBuf) -> Result<Self, String> {
        let document = match fs::read(&path) {
            Ok(bytes) => match parse_document(&bytes) {
                Ok(document) => document,
                Err(reason) => {
                    let quarantined = quarantine_corrupt_file(&path).map_err(|error| {
                        format!(
                            "failed to quarantine malformed app settings at {}: {error}",
                            path.display()
                        )
                    })?;
                    eprintln!(
                        "recovered malformed app settings ({reason}) from {}; contribution was disabled and the original was moved to {}",
                        path.display(),
                        quarantined.display()
                    );
                    AppSettingsDocument {
                        version: APP_SETTINGS_VERSION,
                        vram_contribution_limit_bytes: Some(0),
                    }
                }
            },
            Err(error) if error.kind() == io::ErrorKind::NotFound => AppSettingsDocument::default(),
            Err(error) => {
                return Err(format!(
                    "failed to read app settings at {}: {error}",
                    path.display()
                ));
            }
        };

        let store = Self { path, document };
        store.persist(&store.document)?;
        Ok(store)
    }

    pub(crate) fn apply_saved_limit(&self) {
        let total_bytes = detect_node_capabilities().total_accelerator_memory_bytes;
        let limit = self
            .document
            .vram_contribution_limit_bytes
            .map(|bytes| bytes.min(total_bytes));
        set_vram_contribution_limit_bytes(limit);
    }

    fn snapshot(&self) -> VramContributionSettings {
        let capabilities = detect_node_capabilities();
        let total_bytes = capabilities.total_accelerator_memory_bytes;
        VramContributionSettings {
            contribution_bytes: self
                .document
                .vram_contribution_limit_bytes
                .unwrap_or(total_bytes)
                .min(total_bytes),
            total_bytes,
            available_bytes: capabilities.available_accelerator_memory_bytes,
            compute_backend: capabilities.compute_backend,
            device_name: capabilities.device_name,
            unified_memory: capabilities.unified_memory,
        }
    }

    fn set_vram_contribution(
        &mut self,
        contribution_bytes: u64,
    ) -> Result<VramContributionSettings, String> {
        let total_bytes = detect_node_capabilities().total_accelerator_memory_bytes;
        if contribution_bytes > total_bytes {
            return Err(format!(
                "VRAM contribution cannot exceed the detected capacity of {total_bytes} bytes"
            ));
        }

        let next = AppSettingsDocument {
            version: APP_SETTINGS_VERSION,
            vram_contribution_limit_bytes: Some(contribution_bytes),
        };
        self.persist(&next)?;
        self.document = next;
        set_vram_contribution_limit_bytes(Some(contribution_bytes));
        Ok(self.snapshot())
    }

    fn persist(&self, document: &AppSettingsDocument) -> Result<(), String> {
        persist_document(&self.path, document).map_err(|error| {
            format!(
                "failed to persist app settings at {}: {error}",
                self.path.display()
            )
        })
    }
}

#[tauri::command]
pub(crate) fn get_vram_contribution_settings(
    state: State<'_, UiState>,
) -> Result<VramContributionSettings, String> {
    with_store(&state, |store| Ok(store.snapshot()))
}

#[tauri::command]
pub(crate) async fn set_vram_contribution(
    app: AppHandle,
    state: State<'_, UiState>,
    contribution_bytes: u64,
) -> Result<VramContributionSettings, String> {
    let settings = with_store(&state, |store| {
        store.set_vram_contribution(contribution_bytes)
    })?;

    stop_persistent_infernet_workers();
    {
        let mut service = state
            .llama_rpc_service
            .lock()
            .map_err(|_| "failed to lock llama.cpp RPC service state".to_owned())?;
        *service = LlamaRpcServiceState::Stopped;
    }
    {
        let mut service = state
            .image_rpc_service
            .lock()
            .map_err(|_| "failed to lock Infernet Image RPC service state".to_owned())?;
        *service = ImageRpcServiceState::Stopped;
    }
    clear_local_llama_rpc_endpoint();
    clear_local_image_rpc_endpoint();
    set_local_rpc_active(false);

    if contribution_bytes > 0 {
        let cache_config = cache_config_for_app(&app);
        ensure_image_rpc_service(&state, &cache_config).await?;
        ensure_llama_rpc_service(&state, &cache_config).await?;
    }

    Ok(settings)
}

fn with_store<T>(
    state: &State<'_, UiState>,
    operation: impl FnOnce(&mut AppSettingsStore) -> Result<T, String>,
) -> Result<T, String> {
    let mut settings = state
        .app_settings
        .lock()
        .map_err(|_| "failed to lock app settings".to_owned())?;
    let store = settings
        .as_mut()
        .ok_or_else(|| "app settings are not initialized".to_owned())?;
    operation(store)
}

fn parse_document(bytes: &[u8]) -> Result<AppSettingsDocument, String> {
    let document: AppSettingsDocument =
        serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
    if document.version != APP_SETTINGS_VERSION {
        return Err(format!(
            "unsupported app settings version {}; expected {APP_SETTINGS_VERSION}",
            document.version
        ));
    }
    Ok(document)
}

fn persist_document(path: &Path, document: &AppSettingsDocument) -> io::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("app-settings-v1.json");
    let temporary_path = parent.join(format!(".{file_name}.tmp-{}", Uuid::new_v4()));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }

    let result = (|| {
        let mut file = options.open(&temporary_path)?;
        serde_json::to_writer_pretty(&mut file, document)?;
        file.write_all(b"\n")?;
        file.flush()?;
        file.sync_all()?;
        drop(file);
        replace_file(&temporary_path, path)
    })();

    if result.is_err() {
        let _ = fs::remove_file(&temporary_path);
    }
    result
}

#[cfg(not(windows))]
fn replace_file(source: &Path, destination: &Path) -> io::Result<()> {
    fs::rename(source, destination)
}

#[cfg(windows)]
fn replace_file(source: &Path, destination: &Path) -> io::Result<()> {
    if destination.exists() {
        fs::remove_file(destination)?;
    }
    fs::rename(source, destination)
}

fn quarantine_corrupt_file(path: &Path) -> io::Result<PathBuf> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("app-settings-v1.json");
    let quarantined = path.with_file_name(format!("{file_name}.corrupt-{timestamp}"));
    fs::rename(path, &quarantined)?;
    Ok(quarantined)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new(label: &str) -> Self {
            let path = std::env::temp_dir()
                .join(format!("infernet-app-settings-{label}-{}", Uuid::new_v4()));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn settings_path(&self) -> PathBuf {
            self.0.join("app-settings-v1.json")
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn defaults_to_automatic_contribution() {
        let directory = TestDirectory::new("default");
        let store = AppSettingsStore::open(directory.settings_path()).unwrap();

        assert_eq!(store.document.vram_contribution_limit_bytes, None);
    }

    #[test]
    fn contribution_limit_survives_reopen() {
        let directory = TestDirectory::new("roundtrip");
        let path = directory.settings_path();
        let store = AppSettingsStore::open(path.clone()).unwrap();
        let expected = AppSettingsDocument {
            version: APP_SETTINGS_VERSION,
            vram_contribution_limit_bytes: Some(4 * 1024 * 1024 * 1024),
        };
        store.persist(&expected).unwrap();
        drop(store);

        let reopened = AppSettingsStore::open(path).unwrap();
        assert_eq!(reopened.document, expected);
    }

    #[test]
    fn corrupt_settings_fail_closed() {
        let directory = TestDirectory::new("corrupt");
        let path = directory.settings_path();
        fs::write(&path, b"not-json").unwrap();

        let store = AppSettingsStore::open(path).unwrap();

        assert_eq!(store.document.vram_contribution_limit_bytes, Some(0));
        assert!(
            fs::read_dir(&directory.0)
                .unwrap()
                .filter_map(Result::ok)
                .any(|entry| entry.file_name().to_string_lossy().contains(".corrupt-"))
        );
    }
}
