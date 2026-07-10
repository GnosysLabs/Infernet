use std::{
    env,
    ffi::OsString,
    fs,
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{Mutex, OnceLock},
    thread,
    time::{Duration, Instant, SystemTime},
};

use anyhow::{Context, Result, anyhow, bail};

const MAX_IMAGE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_DECODED_IMAGE_BYTES: usize = 32 * 1024 * 1024;
const PINNED_STABLE_DIFFUSION_CPP_REVISION: &str = "cc73429";

#[derive(Debug, Clone, PartialEq, Eq)]
struct ValidatedRuntime {
    path: PathBuf,
    size_bytes: u64,
    modified: Option<SystemTime>,
}

static VALIDATED_RUNTIMES: OnceLock<Mutex<Vec<ValidatedRuntime>>> = OnceLock::new();

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StableDiffusionPlacement {
    RequesterLocal,
    Distributed { machine_count: usize },
}

#[derive(Debug, Clone)]
pub struct StableDiffusionConfig {
    pub binary: PathBuf,
    pub diffusion_model_path: PathBuf,
    pub text_encoder_path: PathBuf,
    pub vae_path: PathBuf,
    pub output_dir: PathBuf,
    pub log_dir: PathBuf,
    pub backend: String,
    pub params_backend: Option<String>,
    pub rpc_servers: Vec<String>,
    pub placement: StableDiffusionPlacement,
    pub timeout: Duration,
}

#[derive(Debug, Clone)]
pub struct ImageGenerationRequest {
    pub job_id: String,
    pub prompt: String,
    pub seed: i64,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageGenerationOutput {
    pub png_path: PathBuf,
    pub seed: i64,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub duration_ms: u64,
}

pub fn generate_with_sd_cli(
    config: &StableDiffusionConfig,
    request: &ImageGenerationRequest,
) -> Result<ImageGenerationOutput> {
    validate_config(config, request)?;
    fs::create_dir_all(&config.output_dir)
        .with_context(|| format!("failed to create {}", config.output_dir.display()))?;
    fs::create_dir_all(&config.log_dir)
        .with_context(|| format!("failed to create {}", config.log_dir.display()))?;
    let temporary_dir = config.output_dir.join(".tmp");
    fs::create_dir_all(&temporary_dir)
        .with_context(|| format!("failed to create {}", temporary_dir.display()))?;

    let temporary_path = temporary_dir.join(format!("{}.png", request.job_id));
    let output_path = config.output_dir.join(format!("{}.png", request.job_id));
    let log_path = config.log_dir.join(format!("{}.log", request.job_id));
    let _ = fs::remove_file(&temporary_path);

    let result = (|| -> Result<ImageGenerationOutput> {
        let stdout = File::options()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&log_path)
            .with_context(|| format!("failed to open {}", log_path.display()))?;
        let stderr = stdout.try_clone()?;
        let mut command = Command::new(&config.binary);
        if let Some(runtime_dir) = config.binary.parent() {
            command.current_dir(runtime_dir);
        }
        command
            .args(sd_cli_arguments(config, request, &temporary_path))
            .stdin(Stdio::null())
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(stderr));

        let started = Instant::now();
        let mut child = command.spawn().with_context(|| {
            format!(
                "failed to start Infernet Image runtime {}",
                config.binary.display()
            )
        })?;
        let status = loop {
            if let Some(status) = child.try_wait()? {
                break status;
            }
            if started.elapsed() >= config.timeout {
                let _ = child.kill();
                let _ = child.wait();
                bail!(
                    "Infernet Image exceeded its {} second generation limit; see {}",
                    config.timeout.as_secs(),
                    log_path.display()
                );
            }
            thread::sleep(Duration::from_millis(50));
        };
        if !status.success() {
            bail!(
                "Infernet Image runtime exited with {status}; see {}",
                log_path.display()
            );
        }

        validate_png(&temporary_path, request.width, request.height)?;
        if output_path.exists() {
            fs::remove_file(&output_path)
                .with_context(|| format!("failed to replace {}", output_path.display()))?;
        }
        fs::rename(&temporary_path, &output_path).with_context(|| {
            format!("failed to commit generated image {}", output_path.display())
        })?;

        Ok(ImageGenerationOutput {
            png_path: output_path,
            seed: request.seed,
            width: request.width,
            height: request.height,
            steps: request.steps,
            duration_ms: started.elapsed().as_millis().try_into().unwrap_or(u64::MAX),
        })
    })();

    if result.is_err() {
        let _ = fs::remove_file(temporary_path);
    }
    result
}

pub fn find_sd_cli_binary() -> Option<PathBuf> {
    if environment_flag("INFERNET_ALLOW_EXTERNAL_SD_RUNTIME")
        && let Some(path) = env::var_os("INFERNET_SD_CLI").map(PathBuf::from)
        && validate_sd_cli_binary(&path).is_ok()
    {
        return Some(path);
    }

    let executable_name = platform_executable_name("sd-cli");
    let sidecar_name = bundled_sidecar_name();
    let mut candidates = Vec::new();
    if let Ok(current_exe) = env::current_exe()
        && let Some(directory) = current_exe.parent()
    {
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
                .join("stable-diffusion.cpp-runtime")
                .join("build")
                .join("bin")
                .join(&executable_name),
        );
    }
    candidates
        .into_iter()
        .find(|candidate| validate_sd_cli_binary(candidate).is_ok())
}

fn validate_config(config: &StableDiffusionConfig, request: &ImageGenerationRequest) -> Result<()> {
    for (label, path) in [
        ("sd-cli", &config.binary),
        ("diffusion transformer", &config.diffusion_model_path),
        ("text encoder", &config.text_encoder_path),
        ("VAE", &config.vae_path),
    ] {
        if !path.is_file() {
            bail!("Infernet Image {label} is missing: {}", path.display());
        }
    }
    validate_sd_cli_binary(&config.binary)?;
    if request.prompt.trim().is_empty() {
        bail!("image prompt must not be empty");
    }
    if request.job_id.is_empty()
        || request.job_id.len() > 80
        || !request
            .job_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
    {
        bail!("image job id is invalid");
    }
    if !(512..=1536).contains(&request.width)
        || !(512..=1536).contains(&request.height)
        || request.width % 64 != 0
        || request.height % 64 != 0
    {
        bail!("image dimensions must be 512..1536 pixels and divisible by 64");
    }
    if request.steps == 0 || request.steps > 50 {
        bail!("image step count must be between 1 and 50");
    }
    if config.backend.trim().is_empty() || config.backend.eq_ignore_ascii_case("cpu") {
        bail!("Infernet Image requires an accelerated Metal, CUDA, Vulkan, or RPC backend");
    }
    validate_placement(config)
}

fn validate_placement(config: &StableDiffusionConfig) -> Result<()> {
    match config.placement {
        StableDiffusionPlacement::RequesterLocal if !config.rpc_servers.is_empty() => {
            bail!("requester-local image execution cannot use remote RPC workers");
        }
        StableDiffusionPlacement::Distributed { .. } => bail!(
            "distributed image execution requires Infernet's role-scoped stage runtime; sd-cli RPC is not a valid placement plan"
        ),
        StableDiffusionPlacement::RequesterLocal => {}
    }
    Ok(())
}

fn validate_sd_cli_binary(path: &Path) -> Result<()> {
    let metadata = fs::metadata(path)
        .with_context(|| format!("Infernet Image runtime is missing: {}", path.display()))?;
    if !metadata.is_file() {
        bail!("Infernet Image runtime is not a file: {}", path.display());
    }
    let validated = ValidatedRuntime {
        path: fs::canonicalize(path).unwrap_or_else(|_| path.to_owned()),
        size_bytes: metadata.len(),
        modified: metadata.modified().ok(),
    };
    let runtimes = VALIDATED_RUNTIMES.get_or_init(|| Mutex::new(Vec::new()));
    if runtimes
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .contains(&validated)
    {
        return Ok(());
    }

    let output = Command::new(path)
        .arg("--help")
        .stdin(Stdio::null())
        .output()
        .with_context(|| {
            format!(
                "failed to inspect Infernet Image runtime {}",
                path.display()
            )
        })?;
    if !output.status.success() {
        bail!(
            "Infernet Image runtime validation failed with {}",
            output.status
        );
    }
    let version = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    if !version.contains("stable-diffusion.cpp version")
        || !version.contains(PINNED_STABLE_DIFFUSION_CPP_REVISION)
    {
        bail!(
            "Infernet Image requires pinned stable-diffusion.cpp revision {PINNED_STABLE_DIFFUSION_CPP_REVISION}"
        );
    }
    let mut runtimes = runtimes
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    runtimes.retain(|runtime| runtime.path != validated.path);
    runtimes.push(validated);
    Ok(())
}

fn environment_flag(name: &str) -> bool {
    env::var(name).is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn sd_cli_arguments(
    config: &StableDiffusionConfig,
    request: &ImageGenerationRequest,
    output_path: &Path,
) -> Vec<OsString> {
    let mut arguments = vec![
        OsString::from("--diffusion-model"),
        config.diffusion_model_path.as_os_str().to_owned(),
        OsString::from("--llm"),
        config.text_encoder_path.as_os_str().to_owned(),
        OsString::from("--vae"),
        config.vae_path.as_os_str().to_owned(),
        OsString::from("--prompt"),
        OsString::from(&request.prompt),
        OsString::from("--output"),
        output_path.as_os_str().to_owned(),
        OsString::from("--width"),
        OsString::from(request.width.to_string()),
        OsString::from("--height"),
        OsString::from(request.height.to_string()),
        OsString::from("--steps"),
        OsString::from(request.steps.to_string()),
        OsString::from("--cfg-scale"),
        OsString::from("1.0"),
        OsString::from("--seed"),
        OsString::from(request.seed.to_string()),
        OsString::from("--rng"),
        OsString::from("cpu"),
        OsString::from("--backend"),
        OsString::from(&config.backend),
        OsString::from("--diffusion-fa"),
        OsString::from("--vae-conv-direct"),
        OsString::from("--vae-tiling"),
    ];
    if let Some(params_backend) = config.params_backend.as_deref() {
        arguments.push(OsString::from("--params-backend"));
        arguments.push(OsString::from(params_backend));
    }
    if !config.rpc_servers.is_empty() {
        arguments.push(OsString::from("--rpc-servers"));
        arguments.push(OsString::from(config.rpc_servers.join(",")));
    }
    arguments
}

fn validate_png(path: &Path, expected_width: u32, expected_height: u32) -> Result<()> {
    let metadata = fs::metadata(path)
        .with_context(|| format!("image runtime did not create {}", path.display()))?;
    if metadata.len() < 33 || metadata.len() > MAX_IMAGE_BYTES {
        bail!("generated PNG has invalid size {} bytes", metadata.len());
    }
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let decoder = png::Decoder::new(BufReader::new(file));
    let mut reader = decoder.read_info().with_context(|| {
        format!(
            "image runtime output is not a valid PNG: {}",
            path.display()
        )
    })?;
    let width = reader.info().width;
    let height = reader.info().height;
    if width != expected_width || height != expected_height {
        bail!(
            "image runtime returned {width}x{height}, expected {expected_width}x{expected_height}"
        );
    }
    let decoded_size = reader
        .output_buffer_size()
        .ok_or_else(|| anyhow!("generated PNG dimensions overflow the decoder"))?;
    if decoded_size > MAX_DECODED_IMAGE_BYTES {
        bail!("generated PNG expands beyond the image safety limit");
    }
    let mut decoded = vec![0_u8; decoded_size];
    reader
        .next_frame(&mut decoded)
        .with_context(|| format!("generated PNG data is corrupt: {}", path.display()))?;
    Ok(())
}

fn push_candidates(
    candidates: &mut Vec<PathBuf>,
    directory: &Path,
    executable_name: &str,
    sidecar_name: Option<&str>,
) {
    candidates.push(directory.join(executable_name));
    candidates.push(directory.join("binaries").join(executable_name));
    if let Some(sidecar_name) = sidecar_name {
        candidates.push(directory.join(sidecar_name));
        candidates.push(directory.join("binaries").join(sidecar_name));
    }
}

fn platform_executable_name(base: &str) -> String {
    if cfg!(windows) {
        format!("{base}.exe")
    } else {
        base.to_owned()
    }
}

fn bundled_sidecar_name() -> Option<String> {
    let triple = target_triple()?;
    Some(if cfg!(windows) {
        format!("sd-cli-{triple}.exe")
    } else {
        format!("sd-cli-{triple}")
    })
}

fn target_triple() -> Option<&'static str> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    return Some("aarch64-apple-darwin");
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    return Some("x86_64-apple-darwin");
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    return Some("x86_64-pc-windows-msvc");
    #[cfg(all(target_os = "windows", target_arch = "aarch64"))]
    return Some("aarch64-pc-windows-msvc");
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    return Some("x86_64-unknown-linux-gnu");
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    return Some("aarch64-unknown-linux-gnu");
    #[allow(unreachable_code)]
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arguments_keep_the_prompt_as_one_process_argument() {
        let config = test_config();
        let request = ImageGenerationRequest {
            job_id: "image-1".to_owned(),
            prompt: "a fox; rm -rf /".to_owned(),
            seed: 42,
            width: 1024,
            height: 1024,
            steps: 8,
        };
        let arguments = sd_cli_arguments(&config, &request, Path::new("output.png"));

        assert!(arguments.contains(&OsString::from("a fox; rm -rf /")));
        assert_eq!(
            arguments
                .iter()
                .filter(|argument| *argument == "a fox; rm -rf /")
                .count(),
            1
        );
        assert!(arguments.contains(&OsString::from("--diffusion-fa")));
        assert!(arguments.contains(&OsString::from("--vae-tiling")));
    }

    #[test]
    fn sd_cli_rpc_never_claims_to_be_a_valid_distributed_plan() {
        let mut config = test_config();
        config.placement = StableDiffusionPlacement::Distributed { machine_count: 2 };
        config.rpc_servers.push("127.0.0.1:50052".to_owned());
        assert!(validate_placement(&config).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn pinned_runtime_wrapper_commits_a_validated_png() {
        use std::os::unix::fs::PermissionsExt;

        let root = env::temp_dir().join(format!("infernet-image-runtime-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let binary = root.join("sd-cli");
        fs::write(
            &binary,
            r#"#!/bin/sh
if [ "$1" = "--help" ]; then
  echo "stable-diffusion.cpp version master-769-cc73429, commit cc73429"
  exit 0
fi
output=""
while [ "$#" -gt 0 ]; do
  if [ "$1" = "--output" ]; then
    shift
    output="$1"
  fi
  shift
done
cp "$(dirname "$0")/fixture.png" "$output"
"#,
        )
        .unwrap();
        fs::set_permissions(&binary, fs::Permissions::from_mode(0o755)).unwrap();
        for file in ["dit.gguf", "te.gguf", "vae.safetensors"] {
            fs::write(root.join(file), b"model").unwrap();
        }
        {
            let file = File::create(root.join("fixture.png")).unwrap();
            let mut encoder = png::Encoder::new(file, 1024, 1024);
            encoder.set_color(png::ColorType::Grayscale);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(&vec![0_u8; 1024 * 1024]).unwrap();
        }

        let config = StableDiffusionConfig {
            binary,
            diffusion_model_path: root.join("dit.gguf"),
            text_encoder_path: root.join("te.gguf"),
            vae_path: root.join("vae.safetensors"),
            output_dir: root.join("output"),
            log_dir: root.join("logs"),
            backend: "metal".to_owned(),
            params_backend: Some("*=cpu".to_owned()),
            rpc_servers: Vec::new(),
            placement: StableDiffusionPlacement::RequesterLocal,
            timeout: Duration::from_secs(5),
        };
        let request = ImageGenerationRequest {
            job_id: "generated-image".to_owned(),
            prompt: "a small red kite".to_owned(),
            seed: 42,
            width: 1024,
            height: 1024,
            steps: 8,
        };

        let output = generate_with_sd_cli(&config, &request).unwrap();

        assert_eq!(output.png_path, root.join("output/generated-image.png"));
        assert!(output.png_path.is_file());
        assert!(!root.join("output/.tmp/generated-image.png").exists());
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn runtime_validation_rejects_an_unpinned_sd_cli() {
        use std::os::unix::fs::PermissionsExt;

        let root = env::temp_dir().join(format!("infernet-image-runtime-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let binary = root.join("sd-cli");
        fs::write(
            &binary,
            "#!/bin/sh\necho 'stable-diffusion.cpp version unknown'\n",
        )
        .unwrap();
        fs::set_permissions(&binary, fs::Permissions::from_mode(0o755)).unwrap();

        assert!(validate_sd_cli_binary(&binary).is_err());
        let _ = fs::remove_dir_all(root);
    }

    fn test_config() -> StableDiffusionConfig {
        StableDiffusionConfig {
            binary: PathBuf::from("sd-cli"),
            diffusion_model_path: PathBuf::from("dit.gguf"),
            text_encoder_path: PathBuf::from("te.gguf"),
            vae_path: PathBuf::from("vae.safetensors"),
            output_dir: PathBuf::from("output"),
            log_dir: PathBuf::from("logs"),
            backend: "metal".to_owned(),
            params_backend: None,
            rpc_servers: Vec::new(),
            placement: StableDiffusionPlacement::RequesterLocal,
            timeout: Duration::from_secs(60),
        }
    }
}
