use std::{
    env, fs,
    net::{TcpStream, ToSocketAddrs},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow, bail};

pub const LLAMA_RPC_DEFAULT_PORT: u16 = 50_052;
pub const LLAMA_RPC_PROTOCOL_VERSION: &str = "4.0.1";
pub const INFERNET_LLAMA_RPC_RUNTIME_ABI: &str = "infernet-llama-rpc-v1";

#[derive(Debug, Clone)]
pub struct LlamaRpcServerConfig {
    pub binary: PathBuf,
    pub bind_host: String,
    pub advertised_host: String,
    pub port: u16,
    pub cache_dir: PathBuf,
    /// Enables llama.cpp's persistent RPC tensor cache. Launch nodes keep this
    /// false so topology changes cannot silently duplicate a model on disk.
    pub cache_tensors: bool,
    pub threads: usize,
    pub device: Option<String>,
    pub expected_backend: String,
    pub startup_timeout: Duration,
}

pub struct LlamaRpcServer {
    child: Child,
    endpoint: String,
    log_path: PathBuf,
}

impl LlamaRpcServer {
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub fn process_id(&self) -> u32 {
        self.child.id()
    }

    pub fn is_running(&mut self) -> bool {
        self.child.try_wait().ok().flatten().is_none()
    }

    pub fn has_active_client(&self) -> bool {
        let Ok(log) = fs::read_to_string(&self.log_path) else {
            return false;
        };
        log.matches("Accepted client connection").count()
            > log.matches("Client connection closed").count()
    }
}

impl Drop for LlamaRpcServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub fn spawn_llama_rpc_server(config: LlamaRpcServerConfig) -> Result<LlamaRpcServer> {
    if !config.binary.is_file() {
        bail!(
            "llama.cpp RPC server binary is missing: {}",
            config.binary.display()
        );
    }
    if config.port == 0
        || config.bind_host.trim().is_empty()
        || config.advertised_host.trim().is_empty()
    {
        bail!("llama.cpp RPC server requires a host and non-zero port");
    }

    fs::create_dir_all(&config.cache_dir)
        .with_context(|| format!("failed to create RPC cache {}", config.cache_dir.display()))?;
    let log_path = config.cache_dir.join("rpc-server.log");
    let stdout = fs::File::create(&log_path)
        .with_context(|| format!("failed to open {}", log_path.display()))?;
    let stderr = stdout
        .try_clone()
        .with_context(|| format!("failed to clone {}", log_path.display()))?;

    let mut command = Command::new(&config.binary);
    command
        .arg("--host")
        .arg(&config.bind_host)
        .arg("--port")
        .arg(config.port.to_string())
        .arg("--threads")
        .arg(config.threads.max(1).to_string())
        .env("LLAMA_CACHE", &config.cache_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr));
    if config.cache_tensors {
        command.arg("--cache");
    }
    if let Some(device) = config
        .device
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        command.arg("--device").arg(device);
    }

    let probe_host = if matches!(config.bind_host.as_str(), "0.0.0.0" | "::") {
        "127.0.0.1"
    } else {
        config.bind_host.as_str()
    };
    let probe = format!("{probe_host}:{}", config.port);
    let probe_address = resolve_one(&probe)?;
    let mut child = command.spawn().with_context(|| {
        format!(
            "failed to start llama.cpp RPC server {}",
            config.binary.display()
        )
    })?;
    let endpoint = format!("{}:{}", config.advertised_host, config.port);
    let deadline = Instant::now() + config.startup_timeout;
    let startup = (|| -> Result<()> {
        loop {
            let reachable =
                TcpStream::connect_timeout(&probe_address, Duration::from_millis(200)).is_ok();
            if reachable && rpc_log_confirms_backend(&log_path, &config.expected_backend) {
                return Ok(());
            }
            if let Some(status) = child.try_wait()? {
                bail!(
                    "llama.cpp RPC server exited during startup with {status}; see {}",
                    log_path.display()
                );
            }
            if Instant::now() >= deadline {
                bail!(
                    "timed out validating llama.cpp RPC {} backend at {probe}; see {}",
                    config.expected_backend,
                    log_path.display()
                );
            }
            thread::sleep(Duration::from_millis(50));
        }
    })();
    if let Err(error) = startup {
        let _ = child.kill();
        let _ = child.wait();
        return Err(error);
    }

    Ok(LlamaRpcServer {
        child,
        endpoint,
        log_path,
    })
}

fn rpc_log_confirms_backend(path: &Path, expected_backend: &str) -> bool {
    let Ok(log) = fs::read_to_string(path) else {
        return false;
    };
    if !log.contains(&format!(
        "Starting RPC server v{LLAMA_RPC_PROTOCOL_VERSION}"
    )) {
        return false;
    }
    match expected_backend {
        "cuda" => log.contains("CUDA") || log.contains("NVIDIA"),
        "metal" => log.contains("MTL") || log.contains("Metal") || log.contains("Apple"),
        _ => false,
    }
}

pub fn find_llama_rpc_server_binary() -> Option<PathBuf> {
    if let Ok(path) = env::var("INFERNET_LLAMA_RPC_SERVER") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
    }

    let executable_name = platform_executable_name("ggml-rpc-server");
    let sidecar_name = bundled_sidecar_name();
    for candidate in bundled_candidates(&executable_name, sidecar_name) {
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    if let Some(path) = env::var_os("PATH") {
        for directory in env::split_paths(&path) {
            for name in std::iter::once(executable_name.as_str()).chain(sidecar_name) {
                let candidate = directory.join(name);
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
        }
    }
    None
}

fn bundled_candidates(executable_name: &str, sidecar_name: Option<&str>) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(current_exe) = env::current_exe()
        && let Some(parent) = current_exe.parent()
    {
        push_binary_candidates(&mut candidates, parent, executable_name, sidecar_name);
        if let Some(resources) = parent.parent().map(|path| path.join("Resources")) {
            push_binary_candidates(&mut candidates, &resources, executable_name, sidecar_name);
        }
    }
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if let Some(repo_root) = crate_dir.parent().and_then(Path::parent) {
        push_binary_candidates(
            &mut candidates,
            &repo_root
                .join("infernet-ui")
                .join("src-tauri")
                .join("binaries"),
            executable_name,
            sidecar_name,
        );
    }
    candidates
}

fn push_binary_candidates(
    candidates: &mut Vec<PathBuf>,
    root: &Path,
    executable_name: &str,
    sidecar_name: Option<&str>,
) {
    candidates.push(root.join(executable_name));
    candidates.push(root.join("binaries").join(executable_name));
    if let Some(sidecar_name) = sidecar_name {
        candidates.push(root.join(sidecar_name));
        candidates.push(root.join("binaries").join(sidecar_name));
    }
}

fn platform_executable_name(base: &str) -> String {
    if cfg!(windows) {
        format!("{base}.exe")
    } else {
        base.to_owned()
    }
}

fn bundled_sidecar_name() -> Option<&'static str> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    return Some("ggml-rpc-server-aarch64-apple-darwin");
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    return Some("ggml-rpc-server-x86_64-apple-darwin");
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    return Some("ggml-rpc-server-x86_64-pc-windows-msvc.exe");
    #[cfg(all(target_os = "windows", target_arch = "aarch64"))]
    return Some("ggml-rpc-server-aarch64-pc-windows-msvc.exe");
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    return Some("ggml-rpc-server-x86_64-unknown-linux-gnu");
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    return Some("ggml-rpc-server-aarch64-unknown-linux-gnu");
    #[allow(unreachable_code)]
    None
}

fn resolve_one(endpoint: &str) -> Result<std::net::SocketAddr> {
    endpoint
        .to_socket_addrs()
        .with_context(|| format!("invalid RPC endpoint {endpoint}"))?
        .next()
        .ok_or_else(|| anyhow!("RPC endpoint {endpoint} did not resolve"))
}
