use std::env;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};

const MAX_HTTP_RESPONSE_BYTES: usize = 16 * 1024 * 1024;
const HEALTH_POLL_INTERVAL: Duration = Duration::from_millis(250);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlamaServerConfig {
    pub binary: PathBuf,
    pub model_path: PathBuf,
    pub rpc_endpoints: Vec<String>,
    pub context_size: u32,
    pub threads: usize,
    pub cache_ram_mib: u32,
    pub startup_timeout: Duration,
    pub request_timeout: Duration,
    pub log_dir: PathBuf,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LlamaServerCompletion {
    pub text: String,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub timing_ms: u64,
}

#[derive(Debug)]
struct PersistentLlamaServer {
    child: Child,
    endpoint: String,
    config: LlamaServerConfig,
    log_path: PathBuf,
}

impl Drop for PersistentLlamaServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

static PERSISTENT_SERVER: OnceLock<Mutex<Option<PersistentLlamaServer>>> = OnceLock::new();
static SERVER_REQUEST: OnceLock<Mutex<()>> = OnceLock::new();

pub fn complete_with_persistent_llama_server(
    config: LlamaServerConfig,
    prompt: &str,
    max_tokens: u32,
) -> Result<LlamaServerCompletion> {
    validate_config(&config)?;
    if prompt.trim().is_empty() {
        bail!("llama.cpp prompt must not be empty");
    }
    if max_tokens == 0 {
        bail!("llama.cpp max_tokens must be greater than zero");
    }

    let _request = SERVER_REQUEST
        .get_or_init(|| Mutex::new(()))
        .lock()
        .map_err(|_| anyhow!("persistent llama.cpp request lock is poisoned"))?;
    let (endpoint, request_timeout, log_path) = {
        let server_state = PERSISTENT_SERVER.get_or_init(|| Mutex::new(None));
        let mut server_state = server_state
            .lock()
            .map_err(|_| anyhow!("persistent llama.cpp server lock is poisoned"))?;
        let must_restart = match server_state.as_mut() {
            Some(server) if server.config == config => !server_is_running(server)?,
            Some(_) => true,
            None => true,
        };
        if must_restart {
            *server_state = None;
            *server_state = Some(spawn_persistent_server(config)?);
        }
        let server = server_state
            .as_ref()
            .expect("persistent server is initialized above");
        (
            server.endpoint.clone(),
            server.config.request_timeout,
            server.log_path.clone(),
        )
    };
    let started = Instant::now();
    let completion = request_chat_completion(&endpoint, prompt, max_tokens, request_timeout)
        .with_context(|| {
            format!(
                "llama.cpp server request failed; coordinator log: {}",
                log_path.display()
            )
        })?;

    Ok(LlamaServerCompletion {
        text: completion.text,
        prompt_tokens: completion.prompt_tokens,
        completion_tokens: completion.completion_tokens,
        timing_ms: started.elapsed().as_millis().try_into().unwrap_or(u64::MAX),
    })
}

pub fn stop_persistent_llama_server() {
    let Some(server_state) = PERSISTENT_SERVER.get() else {
        return;
    };
    if let Ok(mut server) = server_state.lock() {
        *server = None;
    }
}

pub fn find_llama_server_binary() -> Option<PathBuf> {
    if let Some(path) = env::var_os("INFERNET_LLAMA_SERVER").map(PathBuf::from) {
        if path.is_file() {
            return Some(path);
        }
    }

    let executable_name = platform_executable_name("llama-server");
    let sidecar_name = bundled_sidecar_name();
    let mut candidates = Vec::new();
    if let Ok(current_exe) = env::current_exe() {
        if let Some(directory) = current_exe.parent() {
            push_candidates(
                &mut candidates,
                directory,
                &executable_name,
                sidecar_name.as_deref(),
            );
            push_candidates(
                &mut candidates,
                &directory.join("resources"),
                &executable_name,
                sidecar_name.as_deref(),
            );
        }
    }
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if let Some(repo_root) = crate_dir.parent().and_then(Path::parent) {
        push_candidates(
            &mut candidates,
            &repo_root
                .join("infernet-ui")
                .join("src-tauri")
                .join("binaries"),
            &executable_name,
            sidecar_name.as_deref(),
        );
        candidates.push(
            repo_root
                .join("target")
                .join("llama.cpp-runtime")
                .join(format!("build-{}", target_triple()))
                .join("bin")
                .join(&executable_name),
        );
    }
    if let Some(path) = env::var_os("PATH") {
        for directory in env::split_paths(&path) {
            candidates.push(directory.join(&executable_name));
        }
    }
    candidates.into_iter().find(|candidate| candidate.is_file())
}

fn validate_config(config: &LlamaServerConfig) -> Result<()> {
    if !config.binary.is_file() {
        bail!(
            "llama-server binary is missing: {}",
            config.binary.display()
        );
    }
    if !config.model_path.is_file() {
        bail!("model payload is missing: {}", config.model_path.display());
    }
    if config.rpc_endpoints.is_empty() {
        bail!("distributed llama.cpp execution requires at least one RPC worker");
    }
    if config.context_size == 0 || config.threads == 0 {
        bail!("llama.cpp server limits must be non-zero");
    }
    Ok(())
}

fn spawn_persistent_server(config: LlamaServerConfig) -> Result<PersistentLlamaServer> {
    fs::create_dir_all(&config.log_dir)
        .with_context(|| format!("failed to create {}", config.log_dir.display()))?;
    let log_path = config.log_dir.join("llama-coordinator.log");
    rotate_log_if_needed(&log_path)?;
    let stdout = File::options()
        .create(true)
        .append(true)
        .open(&log_path)
        .with_context(|| format!("failed to open {}", log_path.display()))?;
    let stderr = stdout.try_clone()?;
    let listener = TcpListener::bind(("127.0.0.1", 0))
        .context("failed to reserve a loopback llama.cpp server port")?;
    let port = listener.local_addr()?.port();
    drop(listener);

    let mut command = Command::new(&config.binary);
    if let Some(runtime_dir) = config.binary.parent() {
        command.current_dir(runtime_dir);
    }
    constrain_library_threads(&mut command, config.threads);
    command
        .args(llama_server_arguments(&config, port))
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr));
    #[cfg(target_os = "windows")]
    augment_windows_path(&mut command, &config.binary);

    let mut child = command.spawn().with_context(|| {
        format!(
            "failed to start persistent llama.cpp coordinator {}",
            config.binary.display()
        )
    })?;
    let endpoint = format!("127.0.0.1:{port}");
    let deadline = Instant::now() + config.startup_timeout;
    let startup = (|| -> Result<()> {
        loop {
            if let Some(status) = child.try_wait()? {
                bail!(
                    "llama.cpp coordinator exited during model load with {status}; see {}",
                    log_path.display()
                );
            }
            match health_status(&endpoint, Duration::from_secs(2)) {
                Ok(HttpStatus { code: 200, .. }) => return Ok(()),
                Ok(HttpStatus { code: 503, .. }) | Err(_) => {}
                Ok(status) => {
                    let body = String::from_utf8_lossy(&status.body);
                    bail!(
                        "llama.cpp coordinator health check returned HTTP {}: {}",
                        status.code,
                        body.trim()
                    );
                }
            }
            if Instant::now() >= deadline {
                bail!(
                    "timed out loading the distributed model; see {}",
                    log_path.display()
                );
            }
            thread::sleep(HEALTH_POLL_INTERVAL);
        }
    })();
    if let Err(error) = startup {
        let _ = child.kill();
        let _ = child.wait();
        return Err(error);
    }

    Ok(PersistentLlamaServer {
        child,
        endpoint,
        config,
        log_path,
    })
}

fn llama_server_arguments(config: &LlamaServerConfig, port: u16) -> Vec<OsString> {
    [
        OsString::from("--rpc"),
        OsString::from(config.rpc_endpoints.join(",")),
        OsString::from("--model"),
        config.model_path.as_os_str().to_owned(),
        OsString::from("--host"),
        OsString::from("127.0.0.1"),
        OsString::from("--port"),
        OsString::from(port.to_string()),
        OsString::from("--gpu-layers"),
        OsString::from("all"),
        OsString::from("--split-mode"),
        OsString::from("layer"),
        OsString::from("--ctx-size"),
        OsString::from(config.context_size.to_string()),
        OsString::from("--parallel"),
        OsString::from("1"),
        OsString::from("--cache-ram"),
        OsString::from(config.cache_ram_mib.to_string()),
        OsString::from("--threads"),
        OsString::from(config.threads.to_string()),
        OsString::from("--jinja"),
    ]
    .into_iter()
    .collect()
}

fn server_is_running(server: &mut PersistentLlamaServer) -> Result<bool> {
    if server.child.try_wait()?.is_some() {
        return Ok(false);
    }
    Ok(matches!(
        health_status(&server.endpoint, Duration::from_secs(2)),
        Ok(HttpStatus { code: 200, .. })
    ))
}

#[derive(Debug)]
struct ParsedCompletion {
    text: String,
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
}

#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'static str,
    messages: [ChatMessage<'a>; 1],
    max_tokens: u32,
    stream: bool,
}

#[derive(Debug, Serialize)]
struct ChatMessage<'a> {
    role: &'static str,
    content: &'a str,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    #[serde(default)]
    choices: Vec<ChatChoice>,
    #[serde(default)]
    usage: Option<ChatUsage>,
    #[serde(default)]
    error: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatResponseMessage,
}

#[derive(Debug, Deserialize)]
struct ChatResponseMessage {
    #[serde(default)]
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatUsage {
    #[serde(default)]
    prompt_tokens: Option<u64>,
    #[serde(default)]
    completion_tokens: Option<u64>,
}

fn request_chat_completion(
    endpoint: &str,
    prompt: &str,
    max_tokens: u32,
    timeout: Duration,
) -> Result<ParsedCompletion> {
    let body = serde_json::to_vec(&ChatRequest {
        model: "infernet-chat-v1",
        messages: [ChatMessage {
            role: "user",
            content: prompt,
        }],
        max_tokens,
        stream: false,
    })?;
    let response = http_request(
        endpoint,
        "POST",
        "/v1/chat/completions",
        Some(("application/json", &body)),
        timeout,
    )?;
    if response.code != 200 {
        bail!(
            "llama.cpp completion returned HTTP {}: {}",
            response.code,
            String::from_utf8_lossy(&response.body).trim()
        );
    }
    let response: ChatResponse = serde_json::from_slice(&response.body)
        .context("llama.cpp returned invalid completion JSON")?;
    if let Some(error) = response.error {
        bail!("llama.cpp completion failed: {error}");
    }
    let text = response
        .choices
        .first()
        .and_then(|choice| choice.message.content.clone())
        .filter(|content| !content.is_empty())
        .ok_or_else(|| anyhow!("llama.cpp completion did not contain generated text"))?;
    Ok(ParsedCompletion {
        text,
        prompt_tokens: response
            .usage
            .as_ref()
            .and_then(|usage| usage.prompt_tokens),
        completion_tokens: response
            .usage
            .as_ref()
            .and_then(|usage| usage.completion_tokens),
    })
}

fn health_status(endpoint: &str, timeout: Duration) -> Result<HttpStatus> {
    http_request(endpoint, "GET", "/health", None, timeout)
}

#[derive(Debug)]
struct HttpStatus {
    code: u16,
    body: Vec<u8>,
}

fn http_request(
    endpoint: &str,
    method: &str,
    path: &str,
    body: Option<(&str, &[u8])>,
    timeout: Duration,
) -> Result<HttpStatus> {
    let mut stream = TcpStream::connect(endpoint)
        .with_context(|| format!("failed to connect to llama.cpp server at {endpoint}"))?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(Duration::from_secs(10)))?;
    let (content_type, payload) = body.unwrap_or(("application/octet-stream", &[]));
    write!(
        stream,
        "{method} {path} HTTP/1.1\r\nHost: {endpoint}\r\nConnection: close\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\r\n",
        payload.len()
    )?;
    stream.write_all(payload)?;
    stream.flush()?;

    let mut raw = Vec::new();
    stream
        .take((MAX_HTTP_RESPONSE_BYTES + 1) as u64)
        .read_to_end(&mut raw)?;
    if raw.len() > MAX_HTTP_RESPONSE_BYTES {
        bail!("llama.cpp HTTP response exceeded the safety limit");
    }
    parse_http_response(&raw)
}

fn parse_http_response(raw: &[u8]) -> Result<HttpStatus> {
    let header_end = raw
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .ok_or_else(|| anyhow!("llama.cpp returned an invalid HTTP response"))?;
    let header = std::str::from_utf8(&raw[..header_end])?;
    let status_line = header
        .lines()
        .next()
        .ok_or_else(|| anyhow!("llama.cpp HTTP response has no status line"))?;
    let code = status_line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| anyhow!("llama.cpp HTTP response has no status code"))?
        .parse::<u16>()?;
    let mut body = raw[header_end + 4..].to_vec();
    let chunked = header.lines().any(|line| {
        line.split_once(':').is_some_and(|(name, value)| {
            name.eq_ignore_ascii_case("transfer-encoding")
                && value.to_ascii_lowercase().contains("chunked")
        })
    });
    if chunked {
        body = decode_chunked_body(&body)?;
    }
    Ok(HttpStatus { code, body })
}

fn decode_chunked_body(raw: &[u8]) -> Result<Vec<u8>> {
    let mut cursor = 0_usize;
    let mut decoded = Vec::new();
    loop {
        let line_end = raw[cursor..]
            .windows(2)
            .position(|window| window == b"\r\n")
            .map(|offset| cursor + offset)
            .ok_or_else(|| anyhow!("invalid chunked HTTP response"))?;
        let size_text = std::str::from_utf8(&raw[cursor..line_end])?
            .split(';')
            .next()
            .unwrap_or_default();
        let size = usize::from_str_radix(size_text.trim(), 16)?;
        cursor = line_end + 2;
        if size == 0 {
            return Ok(decoded);
        }
        let chunk_end = cursor
            .checked_add(size)
            .ok_or_else(|| anyhow!("chunked HTTP response size overflow"))?;
        if chunk_end + 2 > raw.len() || &raw[chunk_end..chunk_end + 2] != b"\r\n" {
            bail!("truncated chunked HTTP response");
        }
        decoded.extend_from_slice(&raw[cursor..chunk_end]);
        if decoded.len() > MAX_HTTP_RESPONSE_BYTES {
            bail!("decoded HTTP response exceeded the safety limit");
        }
        cursor = chunk_end + 2;
    }
}

fn rotate_log_if_needed(path: &Path) -> Result<()> {
    const MAX_LOG_BYTES: u64 = 16 * 1024 * 1024;
    if fs::metadata(path).is_ok_and(|metadata| metadata.len() >= MAX_LOG_BYTES) {
        let rotated = path.with_extension("log.previous");
        let _ = fs::remove_file(&rotated);
        fs::rename(path, rotated)?;
    }
    Ok(())
}

fn constrain_library_threads(command: &mut Command, threads: usize) {
    let threads = threads.max(1).to_string();
    command
        .env("OMP_NUM_THREADS", &threads)
        .env("OPENBLAS_NUM_THREADS", &threads)
        .env("VECLIB_MAXIMUM_THREADS", &threads)
        .env("BLIS_NUM_THREADS", &threads);
}

#[cfg(target_os = "windows")]
fn augment_windows_path(command: &mut Command, binary: &Path) {
    let mut directories = Vec::new();
    if let Some(parent) = binary.parent() {
        directories.push(parent.to_path_buf());
    }
    if let Some(path) = env::var_os("PATH") {
        directories.extend(env::split_paths(&path));
    }
    if let Ok(path) = env::join_paths(directories) {
        command.env("PATH", path);
    }
}

fn push_candidates(
    candidates: &mut Vec<PathBuf>,
    directory: &Path,
    executable_name: &str,
    sidecar_name: Option<&str>,
) {
    candidates.push(directory.join(executable_name));
    if let Some(sidecar_name) = sidecar_name {
        candidates.push(directory.join(sidecar_name));
    }
}

fn platform_executable_name(base: &str) -> String {
    if cfg!(target_os = "windows") {
        format!("{base}.exe")
    } else {
        base.to_owned()
    }
}

fn bundled_sidecar_name() -> Option<String> {
    Some(format!(
        "llama-server-{}{}",
        target_triple(),
        if cfg!(target_os = "windows") {
            ".exe"
        } else {
            ""
        }
    ))
}

fn target_triple() -> String {
    let arch = match env::consts::ARCH {
        "x86_64" => "x86_64",
        "aarch64" => "aarch64",
        other => other,
    };
    let platform = if cfg!(target_os = "macos") {
        "apple-darwin"
    } else if cfg!(target_os = "windows") {
        "pc-windows-msvc"
    } else {
        "unknown-linux-gnu"
    };
    format!("{arch}-{platform}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_content_length_response() {
        let response = parse_http_response(
            b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nConnection: close\r\n\r\nhello",
        )
        .unwrap();
        assert_eq!(response.code, 200);
        assert_eq!(response.body, b"hello");
    }

    #[test]
    fn decodes_chunked_response() {
        let response = parse_http_response(
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n",
        )
        .unwrap();
        assert_eq!(response.body, b"hello world");
    }

    #[test]
    fn distributed_server_arguments_keep_rpc_before_model_and_persist_one_slot() {
        let config = LlamaServerConfig {
            binary: PathBuf::from("llama-server"),
            model_path: PathBuf::from("infernet-chat.gguf"),
            rpc_endpoints: vec![
                "192.168.1.11:50052".to_owned(),
                "192.168.1.12:50052".to_owned(),
            ],
            context_size: 8192,
            threads: 4,
            cache_ram_mib: 1024,
            startup_timeout: Duration::from_secs(1),
            request_timeout: Duration::from_secs(1),
            log_dir: PathBuf::from("logs"),
        };
        let args = llama_server_arguments(&config, 18080)
            .into_iter()
            .map(|value| value.to_string_lossy().into_owned())
            .collect::<Vec<_>>();

        assert_eq!(
            args[..4],
            [
                "--rpc",
                "192.168.1.11:50052,192.168.1.12:50052",
                "--model",
                "infernet-chat.gguf"
            ]
        );
        assert!(args.windows(2).any(|pair| pair == ["--parallel", "1"]));
        assert!(args.windows(2).any(|pair| pair == ["--cache-ram", "1024"]));
        assert!(args.windows(2).any(|pair| pair == ["--gpu-layers", "all"]));
    }
}
