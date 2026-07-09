fn main() {
    prepare_llama_runtime();
    tauri_build::build()
}

fn prepare_llama_runtime() {
    println!("cargo:rerun-if-changed=../../scripts/prepare-llama-runtime.mjs");
    println!("cargo:rerun-if-changed=../../llama-runtime/infernet-bridge.cpp");
    println!("cargo:rerun-if-env-changed=INFERNET_SKIP_RUNTIME_PREPARE");
    println!("cargo:rerun-if-env-changed=INFERNET_LLAMA_CLI");
    println!("cargo:rerun-if-env-changed=INFERNET_LLAMA_BRIDGE");
    println!("cargo:rerun-if-env-changed=INFERNET_LLAMA_RPC_SERVER");
    println!("cargo:rerun-if-env-changed=INFERNET_LLAMA_SERVER");
    println!("cargo:rerun-if-env-changed=INFERNET_LLAMA_BUILD_JOBS");
    println!("cargo:rerun-if-env-changed=CMAKE_BUILD_PARALLEL_LEVEL");
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_REF");
    println!("cargo:rerun-if-env-changed=INFERNET_CUDA");
    println!("cargo:rerun-if-env-changed=INFERNET_ALLOW_EXTERNAL_LLAMA_RUNTIME");

    if std::env::var("INFERNET_SKIP_RUNTIME_PREPARE").as_deref() == Ok("1") {
        return;
    }

    let script = std::path::Path::new("../../scripts/prepare-llama-runtime.mjs");
    let status = std::process::Command::new("node")
        .arg(script)
        .arg("--quiet")
        .status();

    match status {
        Ok(status) if status.success() => {}
        Ok(status) => {
            eprintln!("failed to prepare bundled llama.cpp runtime: {status}");
            std::process::exit(1);
        }
        Err(error) => {
            eprintln!("failed to launch bundled runtime preparation script: {error}");
            std::process::exit(1);
        }
    }
}
