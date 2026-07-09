use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::time::Instant;

#[derive(Debug, Default)]
struct Args {
    model: Option<PathBuf>,
    prompt: Option<String>,
    input: Option<PathBuf>,
    output: Option<PathBuf>,
}

fn main() -> ExitCode {
    let started = Instant::now();
    match run(started) {
        Ok(output_text) => {
            println!(
                "{}",
                serde_json::json!({
                    "ok": true,
                    "output_text": output_text,
                    "timing_ms": started.elapsed().as_secs_f64() * 1000.0
                })
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            println!(
                "{}",
                serde_json::json!({
                    "ok": false,
                    "error": error,
                    "timing_ms": started.elapsed().as_secs_f64() * 1000.0
                })
            );
            ExitCode::from(2)
        }
    }
}

fn run(started: Instant) -> Result<String, String> {
    let args = parse_args(env::args().skip(1).collect())?;
    if args.input.is_some() || args.output.is_some() {
        return Err("Infernet started with the fallback llama.cpp bridge because the real split-layer bridge was not built on this machine. Install CMake and C++ build tools, rerun npm run prepare-runtime, or set INFERNET_LLAMA_BRIDGE to a real infernet-llama-bridge binary.".to_owned());
    }

    let model = args
        .model
        .ok_or_else(|| "missing --model for fallback llama.cpp bridge".to_owned())?;
    let prompt = args.prompt.unwrap_or_default();
    let llama_cli = find_llama_cli()
        .ok_or_else(|| "fallback llama.cpp bridge could not find bundled llama-cli".to_owned())?;

    let output = Command::new(&llama_cli)
        .arg("-m")
        .arg(&model)
        .arg("-p")
        .arg(&prompt)
        .arg("-n")
        .arg("128")
        .arg("--no-display-prompt")
        .output()
        .map_err(|error| format!("failed to run {}: {error}", llama_cli.display()))?;

    if !output.status.success() {
        return Err(format!(
            "fallback llama-cli failed with status {:?}: {}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if stdout.is_empty() {
        return Err(format!(
            "fallback llama-cli produced no text after {:.0} ms",
            started.elapsed().as_secs_f64() * 1000.0
        ));
    }
    Ok(stdout)
}

fn parse_args(raw: Vec<String>) -> Result<Args, String> {
    let mut parsed = Args::default();
    let mut i = 0;
    while i < raw.len() {
        let key = &raw[i];
        let Some(value) = raw.get(i + 1) else {
            return Err(format!("missing value for {key}"));
        };
        match key.as_str() {
            "--model" => parsed.model = Some(PathBuf::from(value)),
            "--prompt" => parsed.prompt = Some(value.clone()),
            "--input" => parsed.input = Some(PathBuf::from(value)),
            "--output" => parsed.output = Some(PathBuf::from(value)),
            "--layer-start" | "--layer-end" | "--hidden-size" => {}
            _ => return Err(format!("unsupported argument {key}")),
        }
        i += 2;
    }
    Ok(parsed)
}

fn find_llama_cli() -> Option<PathBuf> {
    if let Ok(path) = env::var("INFERNET_LLAMA_CLI") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
    }

    let mut names = vec![
        platform_name("llama-cli"),
        platform_name("llama"),
        platform_name("main"),
    ];
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        names.push("llama-cli-aarch64-apple-darwin".to_owned());
    }
    if cfg!(all(target_os = "macos", target_arch = "x86_64")) {
        names.push("llama-cli-x86_64-apple-darwin".to_owned());
    }
    if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        names.push("llama-cli-x86_64-pc-windows-msvc.exe".to_owned());
    }
    if cfg!(all(target_os = "windows", target_arch = "aarch64")) {
        names.push("llama-cli-aarch64-pc-windows-msvc.exe".to_owned());
    }

    if let Ok(current_exe) = env::current_exe() {
        if let Some(parent) = current_exe.parent() {
            for dir in candidate_dirs(parent) {
                if let Some(path) = find_named_file(&dir, &names) {
                    return Some(path);
                }
            }
        }
    }

    if let Some(path) = env::var_os("PATH") {
        for dir in env::split_paths(&path) {
            if let Some(path) = find_named_file(&dir, &names) {
                return Some(path);
            }
        }
    }

    None
}

fn candidate_dirs(parent: &Path) -> Vec<PathBuf> {
    let mut dirs = vec![parent.to_path_buf(), parent.join("binaries")];
    if let Some(resources) = parent.parent().map(|path| path.join("Resources")) {
        dirs.push(resources.clone());
        dirs.push(resources.join("binaries"));
    }
    dirs
}

fn find_named_file(dir: &Path, names: &[String]) -> Option<PathBuf> {
    names
        .iter()
        .map(|name| dir.join(name))
        .find(|path| path.is_file())
}

fn platform_name(base: &str) -> String {
    if cfg!(windows) {
        format!("{base}.exe")
    } else {
        base.to_owned()
    }
}
