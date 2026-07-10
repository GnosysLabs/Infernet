#!/usr/bin/env node
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  rmSync,
  statSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(scriptDir, "..");
const quiet = process.argv.includes("--quiet");
const targetTriple = process.env.TARGET || rustTargetTriple();
const isWindows = targetTriple.includes("windows");
const isMacos = targetTriple.includes("darwin");
const executableSuffix = isWindows ? ".exe" : "";
const sidecarDir = resolve(repoRoot, "infernet-ui", "src-tauri", "binaries");
const cliSidecarBase = resolve(sidecarDir, "llama-cli");
const bridgeSidecarBase = resolve(sidecarDir, "infernet-llama-bridge");
const rpcServerSidecarBase = resolve(sidecarDir, "ggml-rpc-server");
const serverSidecarBase = resolve(sidecarDir, "llama-server");
const sidecarPath = `${cliSidecarBase}-${targetTriple}${executableSuffix}`;
const bridgeSidecarPath = `${bridgeSidecarBase}-${targetTriple}${executableSuffix}`;
const rpcServerSidecarPath = `${rpcServerSidecarBase}-${targetTriple}${executableSuffix}`;
const serverSidecarPath = `${serverSidecarBase}-${targetTriple}${executableSuffix}`;
const runtimeStampPath = resolve(sidecarDir, `.infernet-llama-runtime-${targetTriple}.stamp`);
// This file is created and removed during every runtime check. Keeping it in
// src-tauri/binaries makes `tauri dev` rebuild the entire app, which starts
// another runtime check and creates an endless hot-reload/peer-identity loop.
const runtimeLockPath = resolve(
  repoRoot,
  "target",
  "llama.cpp-runtime",
  `.infernet-llama-runtime-${targetTriple}.lock`,
);
const runtimePatchVersion = "infernet-persistent-local-layer-workers-v12";
const buildRoot = resolve(repoRoot, "target", "llama.cpp-runtime");
const sourceDir = join(buildRoot, "llama.cpp");
const downloadDir = join(buildRoot, "downloads");
const prebuiltDir = join(buildRoot, `prebuilt-${targetTriple}`);
const llamaRef = process.env.LLAMA_CPP_REF || "049326a00025d00b08cc188ed716b681e984a3f8";
const rpcServerTarget = "ggml-rpc-server";
const serverTarget = "llama-server";
const buildJobs = readBuildJobs();
const cudaEnabled = shouldEnableCuda();
const cudaArchitectures = process.env.INFERNET_CUDA_ARCHITECTURES?.trim() || "native";
const windowsCudaGenerator = isWindows && cudaEnabled
  ? process.env.CMAKE_GENERATOR?.trim() || "Visual Studio 17 2022"
  : null;
const cudaToolkitRoot = process.env.CUDA_PATH?.trim() || null;
const bridgeSourceHash = createHash("sha256")
  .update(readFileSync(resolve(repoRoot, "llama-runtime", "infernet-bridge.cpp")))
  .digest("hex")
  .slice(0, 16);
const runtimeBackend = cudaEnabled ? `cuda-${cudaArchitectures}` : isMacos ? "metal" : "cpu";
const buildDir = join(buildRoot, `build-${targetTriple}-${runtimeBackend.replaceAll(";", "-")}`);
const runtimeStamp = `${runtimePatchVersion}:${llamaRef}:${runtimeBackend}:${bridgeSourceHash}`;

if (process.argv.includes("--dry-run")) {
  printBuildPlan();
} else {
  withRuntimeLock(main);
}

function withRuntimeLock(task) {
  acquireRuntimeLock();
  try {
    task();
  } finally {
    rmSync(runtimeLockPath, { recursive: true, force: true });
  }
}

function acquireRuntimeLock() {
  const waitBuffer = new Int32Array(new SharedArrayBuffer(4));
  while (true) {
    try {
      mkdirSync(runtimeLockPath);
      writeFileSync(join(runtimeLockPath, "pid"), `${process.pid}\n`);
      return;
    } catch (error) {
      if (error?.code !== "EEXIST") {
        throw error;
      }
    }

    let ownerAlive = false;
    try {
      const ownerPid = Number.parseInt(readFileSync(join(runtimeLockPath, "pid"), "utf8"), 10);
      if (Number.isInteger(ownerPid) && ownerPid > 0) {
        process.kill(ownerPid, 0);
        ownerAlive = true;
      }
    } catch {
      ownerAlive = false;
    }
    if (!ownerAlive) {
      rmSync(runtimeLockPath, { recursive: true, force: true });
      continue;
    }
    Atomics.wait(waitBuffer, 0, 0, 250);
  }
}

function main() {
  mkdirSync(sidecarDir, { recursive: true });
  if (
    fileExists(sidecarPath)
    && fileExists(bridgeSidecarPath)
    && fileExists(rpcServerSidecarPath)
    && fileExists(serverSidecarPath)
  ) {
    const validation = validateRuntime(sidecarPath);
    const bridgeValidation = validateRuntime(bridgeSidecarPath);
    const rpcServerValidation = validateRuntime(rpcServerSidecarPath);
    const serverValidation = validateRuntime(serverSidecarPath);
    const stampMatches = readStamp() === runtimeStamp;
    if (validation.ok && bridgeValidation.ok && rpcServerValidation.ok && serverValidation.ok && stampMatches) {
      log(`bundled llama.cpp runtime already exists: ${relative(sidecarPath)}`);
      log(`bundled Infernet llama.cpp bridge already exists: ${relative(bridgeSidecarPath)}`);
      log(`bundled llama.cpp RPC server already exists: ${relative(rpcServerSidecarPath)}`);
      log(`bundled llama.cpp HTTP server already exists: ${relative(serverSidecarPath)}`);
      return;
    }
    const invalidRuntime = [validation, bridgeValidation, rpcServerValidation, serverValidation]
      .find((candidate) => !candidate.ok);
    log(`rebuilding bundled llama.cpp runtime: ${stampMatches ? invalidRuntime.reason : "runtime patch stamp is stale"}`);
    // Keep the last known-good sidecars in place until replacements have been
    // built and validated. This avoids breaking a running dev app if a rebuild
    // is interrupted.
  }

  const allowExternalRuntime = environmentFlag("INFERNET_ALLOW_EXTERNAL_LLAMA_RUNTIME");
  const configuredCli = allowExternalRuntime && process.env.INFERNET_LLAMA_CLI?.trim();
  const configuredBridge = allowExternalRuntime && process.env.INFERNET_LLAMA_BRIDGE?.trim();
  const configuredRpcServer = allowExternalRuntime && process.env.INFERNET_LLAMA_RPC_SERVER?.trim();
  const configuredServer = allowExternalRuntime && process.env.INFERNET_LLAMA_SERVER?.trim();
  if (configuredCli && configuredBridge && configuredRpcServer && configuredServer) {
    copyRuntime(configuredCli, sidecarPath, "INFERNET_LLAMA_CLI");
    copyRuntime(configuredBridge, bridgeSidecarPath, "INFERNET_LLAMA_BRIDGE");
    copyRuntime(configuredRpcServer, rpcServerSidecarPath, "INFERNET_LLAMA_RPC_SERVER");
    copyRuntime(configuredServer, serverSidecarPath, "INFERNET_LLAMA_SERVER");
    writeStamp();
    return;
  }
  if (allowExternalRuntime && [configuredCli, configuredBridge, configuredRpcServer, configuredServer].some(Boolean)) {
    fail("external llama.cpp runtime override requires all four INFERNET_LLAMA_* binary paths");
  }

  // RPC client/server wire compatibility must be exact. Never combine a
  // current upstream prebuilt with Infernet's pinned bridge and then stamp the
  // result as pinned; build the complete runtime from the exact commit.
  buildFromSource();
  writeStamp();
}

function prepareWindowsPrebuilt() {
  const arch = targetTriple.includes("aarch64") ? "arm64" : "x64";
  const assetPattern = new RegExp(`^llama-.+-bin-win-cpu-${arch}\\.zip$`);
  const api = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest";

  try {
    const release = JSON.parse(downloadText(api));
    const asset = release.assets?.find((asset) => assetPattern.test(asset.name));
    if (!asset?.browser_download_url) {
      log(`no official llama.cpp Windows CPU ${arch} prebuilt asset found`);
      return false;
    }

    mkdirSync(downloadDir, { recursive: true });
    rmSync(prebuiltDir, { recursive: true, force: true });
    mkdirSync(prebuiltDir, { recursive: true });

    const zipPath = join(downloadDir, asset.name);
    if (!fileExists(zipPath)) {
      downloadFile(asset.browser_download_url, zipPath);
    }

    run("powershell", [
      "-NoProfile",
      "-ExecutionPolicy",
      "Bypass",
      "-Command",
      `Expand-Archive -LiteralPath ${powershellQuote(zipPath)} -DestinationPath ${powershellQuote(prebuiltDir)} -Force`,
    ]);

    const cli = findFileRecursive(prebuiltDir, "llama-cli.exe");
    const rpcServer = findFileRecursive(prebuiltDir, "ggml-rpc-server.exe");
    const server = findFileRecursive(prebuiltDir, "llama-server.exe");
    if (!cli || !rpcServer || !server) {
      log(`official llama.cpp prebuilt ${asset.name} did not contain llama-cli.exe, llama-server.exe, and ggml-rpc-server.exe`);
      return false;
    }

    copyRuntime(cli, sidecarPath, `official llama.cpp ${asset.name}`);
    copyRuntime(rpcServer, rpcServerSidecarPath, `official llama.cpp ${asset.name}`);
    copyRuntime(server, serverSidecarPath, `official llama.cpp ${asset.name}`);
    copyRuntimeDlls(dirname(cli));
    return true;
  } catch (error) {
    log(`failed to prepare official Windows llama.cpp runtime: ${error.message}`);
    return false;
  }
}

function prepareMacosPrebuilt() {
  const arch = targetTriple.includes("aarch64") ? "arm64" : "x64";
  const assetPattern = new RegExp(`^llama-.+-bin-macos-${arch}\\.tar\\.gz$`);
  const api = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest";

  try {
    const release = JSON.parse(downloadText(api));
    const asset = release.assets?.find((asset) => assetPattern.test(asset.name));
    if (!asset?.browser_download_url) {
      log(`no official llama.cpp macOS ${arch} prebuilt asset found`);
      return false;
    }

    mkdirSync(downloadDir, { recursive: true });
    rmSync(prebuiltDir, { recursive: true, force: true });
    mkdirSync(prebuiltDir, { recursive: true });

    const tarPath = join(downloadDir, asset.name);
    if (!fileExists(tarPath)) {
      downloadFile(asset.browser_download_url, tarPath);
    }

    run("tar", ["-xzf", tarPath, "-C", prebuiltDir]);

    const cli = findFileRecursive(prebuiltDir, "llama-cli");
    const rpcServer = findFileRecursive(prebuiltDir, "ggml-rpc-server");
    const server = findFileRecursive(prebuiltDir, "llama-server");
    if (!cli || !rpcServer || !server) {
      log(`official llama.cpp prebuilt ${asset.name} did not contain llama-cli, llama-server, and ggml-rpc-server`);
      return false;
    }

    const label = `official llama.cpp ${asset.name}`;
    copyRuntime(cli, sidecarPath, label, { validate: false });
    copyRuntime(rpcServer, rpcServerSidecarPath, label, { validate: false });
    copyRuntime(server, serverSidecarPath, label, { validate: false });
    copyRuntimeDylibs(dirname(cli));
    validatePreparedRuntime(sidecarPath, label);
    validatePreparedRuntime(rpcServerSidecarPath, label);
    validatePreparedRuntime(serverSidecarPath, label);
    return true;
  } catch (error) {
    log(`failed to prepare official macOS llama.cpp runtime: ${error.message}`);
    return false;
  }
}

function buildFromSource() {
  ensureSourceBuildRequirements();

  mkdirSync(buildRoot, { recursive: true });
  if (!existsSync(join(sourceDir, ".git"))) {
    run("git", ["clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp.git", sourceDir]);
  }

  run("git", ["fetch", "--depth", "1", "origin", llamaRef], { cwd: sourceDir, optional: true });
  run("git", ["checkout", llamaRef], { cwd: sourceDir });
  prepareInfernetBridgeSource();

  const cmakeArgs = [
    ...(windowsCudaGenerator ? ["-G", windowsCudaGenerator] : []),
    ...(windowsCudaGenerator?.startsWith("Visual Studio ") && cudaToolkitRoot
      ? ["-T", `cuda=${cudaToolkitRoot}`, `-DCUDAToolkit_ROOT=${cudaToolkitRoot}`]
      : []),
    "-S", sourceDir,
    "-B", buildDir,
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_SHARED_LIBS=OFF",
    "-DGGML_BACKEND_DL=OFF",
    "-DGGML_RPC=ON",
    "-DGGML_NATIVE=OFF",
    `-DGGML_CUDA=${cudaEnabled ? "ON" : "OFF"}`,
    `-DGGML_METAL=${isMacos ? "ON" : "OFF"}`,
    "-DLLAMA_CURL=OFF",
    "-DLLAMA_OPENSSL=OFF",
    "-DLLAMA_BUILD_COMMON=ON",
    "-DLLAMA_BUILD_TOOLS=ON",
    "-DLLAMA_BUILD_SERVER=ON",
    "-DLLAMA_BUILD_UI=OFF",
    "-DLLAMA_USE_PREBUILT_UI=OFF",
    "-DLLAMA_BUILD_TESTS=OFF",
  ];
  if (cudaEnabled) {
    cmakeArgs.push(`-DCMAKE_CUDA_ARCHITECTURES=${cudaArchitectures}`);
  }

  run("cmake", cmakeArgs);
  const targets = cmakeTargets();
  const cliTarget = ["llama-cli", "main"].find((target) => targets.includes(target));
  if (!cliTarget) {
    fail(`no supported llama.cpp CLI target found in ${buildDir}`);
  }
  if (!targets.includes(rpcServerTarget)) {
    fail(`required llama.cpp RPC target ${rpcServerTarget} was not generated in ${buildDir}`);
  }
  if (!targets.includes(serverTarget)) {
    fail(`required llama.cpp server target ${serverTarget} was not generated in ${buildDir}`);
  }
  run("cmake", [
    "--build", buildDir,
    "--config", "Release",
    "--parallel", String(buildJobs),
    "--target", cliTarget, "infernet-llama-bridge", rpcServerTarget, serverTarget,
  ]);

  const built = [
    join(buildDir, "bin", `llama-cli${executableSuffix}`),
    join(buildDir, "bin", "Release", `llama-cli${executableSuffix}`),
    join(buildDir, "bin", `main${executableSuffix}`),
    join(buildDir, "bin", "Release", `main${executableSuffix}`),
    join(buildDir, "examples", "main", `llama-cli${executableSuffix}`),
  ].find(fileExists);

  if (!built) {
    fail(`llama.cpp built, but llama-cli was not found under ${buildDir}`);
  }

  const bridgeBuilt = [
    join(buildDir, "bin", `infernet-llama-bridge${executableSuffix}`),
    join(buildDir, "bin", "Release", `infernet-llama-bridge${executableSuffix}`),
    join(buildDir, "examples", "infernet-bridge", `infernet-llama-bridge${executableSuffix}`),
    join(buildDir, "examples", "infernet-bridge", "Release", `infernet-llama-bridge${executableSuffix}`),
  ].find(fileExists);

  if (!bridgeBuilt) {
    fail(`llama.cpp built, but infernet-llama-bridge was not found under ${buildDir}`);
  }

  const rpcServerBuilt = [
    join(buildDir, "bin", `${rpcServerTarget}${executableSuffix}`),
    join(buildDir, "bin", "Release", `${rpcServerTarget}${executableSuffix}`),
    join(buildDir, "tools", "rpc", `${rpcServerTarget}${executableSuffix}`),
    join(buildDir, "tools", "rpc", "Release", `${rpcServerTarget}${executableSuffix}`),
  ].find(fileExists);

  if (!rpcServerBuilt) {
    fail(`llama.cpp built, but ${rpcServerTarget} was not found under ${buildDir}`);
  }

  const serverBuilt = [
    join(buildDir, "bin", `${serverTarget}${executableSuffix}`),
    join(buildDir, "bin", "Release", `${serverTarget}${executableSuffix}`),
    join(buildDir, "tools", "server", `${serverTarget}${executableSuffix}`),
    join(buildDir, "tools", "server", "Release", `${serverTarget}${executableSuffix}`),
  ].find(fileExists);

  if (!serverBuilt) {
    fail(`llama.cpp built, but ${serverTarget} was not found under ${buildDir}`);
  }

  copyRuntime(built, sidecarPath, "llama.cpp source build");
  copyRuntime(bridgeBuilt, bridgeSidecarPath, "Infernet llama.cpp bridge source build");
  copyRuntime(rpcServerBuilt, rpcServerSidecarPath, "llama.cpp RPC server source build");
  copyRuntime(serverBuilt, serverSidecarPath, "llama.cpp HTTP server source build");
  if (isWindows && cudaEnabled) {
    copyCudaRuntimeDlls();
  }
}

function copyCudaRuntimeDlls() {
  const directories = [];
  if (process.env.CUDA_PATH) {
    directories.push(join(process.env.CUDA_PATH, "bin"));
  }
  directories.push(...(process.env.PATH || "").split(";").filter(Boolean));
  const copied = new Set();
  for (const directory of directories) {
    if (!existsSync(directory)) continue;
    for (const name of readdirSync(directory)) {
      if (!/^(?:cublas|cublasLt|cudart|nvrtc|nvJitLink).+\.dll$/i.test(name) || copied.has(name.toLowerCase())) {
        continue;
      }
      copyFileSync(join(directory, name), join(sidecarDir, name));
      copied.add(name.toLowerCase());
    }
  }
  if (![...copied].some((name) => name.startsWith("cublas")) || ![...copied].some((name) => name.startsWith("cudart"))) {
    fail("CUDA runtime DLLs were not found; ensure CUDA_PATH/bin is available while packaging the NVIDIA worker");
  }
}

function prepareInfernetBridgeSource() {
  copyInfernetBridgeExample();
  patchCMakeLists();
  patchLlamaHeader();
  patchLlamaModelParams();
  patchLlamaCparams();
  patchLlamaContext();
  patchLlamaGraph();
  patchModelLoader();
  patchDecoderGraphs();
}

function copyInfernetBridgeExample() {
  const exampleDir = join(sourceDir, "examples", "infernet-bridge");
  mkdirSync(exampleDir, { recursive: true });
  copyFileSync(resolve(repoRoot, "llama-runtime", "infernet-bridge.cpp"), join(exampleDir, "infernet-bridge.cpp"));
  copyFileSync(resolve(repoRoot, "llama-runtime", "CMakeLists.txt"), join(exampleDir, "CMakeLists.txt"));
}

function patchCMakeLists() {
  const path = join(sourceDir, "examples", "CMakeLists.txt");
  let text = readText(path);
  if (!text.includes("add_subdirectory(infernet-bridge)")) {
    text += "\nadd_subdirectory(infernet-bridge)\n";
    writeFileSync(path, text);
  }
}

function patchLlamaHeader() {
  const path = join(sourceDir, "include", "llama.h");
  let text = readText(path);
  text = replaceOnce(text,
    "        const float * tensor_split;\n\n        // Called with a progress value",
    "        const float * tensor_split;\n\n        // Infernet layer-range loading. Experimental: load only tensors needed by one contiguous shard.\n        uint32_t infernet_layer_start;\n        uint32_t infernet_layer_end;\n\n        // Called with a progress value");
  text = replaceOnce(text,
    "        bool no_alloc;        // only load metadata and simulate memory allocations\n",
    "        bool no_alloc;        // only load metadata and simulate memory allocations\n        bool infernet_partial; // load/evaluate only infernet_layer_start..infernet_layer_end\n");
  text = replaceOnce(text,
    "        uint32_t n_outputs_max;     // max outputs in a ubatch (0 = n_batch)\n",
    "        uint32_t n_outputs_max;     // max outputs in a ubatch (0 = n_batch)\n        uint32_t infernet_layer_start; // first resident layer for an Infernet worker\n        uint32_t infernet_layer_end;   // exclusive resident layer end\n        bool infernet_partial;         // allocate KV/graphs only for this range\n");
  text = replaceOnce(text,
    "    // Frees all allocated memory\n    LLAMA_API void llama_free(struct llama_context * ctx);\n",
    "    // Infernet experimental split-layer execution. Must be called before llama_decode().\n    LLAMA_API void llama_infernet_set_layer_range(struct llama_context * ctx, uint32_t layer_start, uint32_t layer_end);\n\n    // Frees all allocated memory\n    LLAMA_API void llama_free(struct llama_context * ctx);\n");
  writeFileSync(path, text);
}

function patchLlamaModelParams() {
  const path = join(sourceDir, "src", "llama-model.cpp");
  let text = readText(path);
  text = replaceOnce(text,
    "        /*.tensor_split                =*/ nullptr,\n        /*.progress_callback           =*/ nullptr,",
    "        /*.tensor_split                =*/ nullptr,\n        /*.infernet_layer_start        =*/ 0,\n        /*.infernet_layer_end          =*/ UINT32_MAX,\n        /*.progress_callback           =*/ nullptr,");
  text = replaceOnce(text,
    "        /*.no_alloc                    =*/ false,\n",
    "        /*.no_alloc                    =*/ false,\n        /*.infernet_partial             =*/ false,\n");
  writeFileSync(path, text);
}

function patchLlamaCparams() {
  const path = join(sourceDir, "src", "llama-cparams.h");
  let text = readText(path);
  text = replaceOnce(text,
    "    int32_t  nextn_layer_offset = 0;\n\n    float rope_freq_base;",
    "    int32_t  nextn_layer_offset = 0;\n\n    uint32_t infernet_layer_start = 0;\n    uint32_t infernet_layer_end   = UINT32_MAX;\n\n    float rope_freq_base;");
  text = replaceOnce(text,
    "    bool pipeline_parallel;\n\n    std::vector<bool> embeddings_layer_inp;",
    "    bool pipeline_parallel;\n    bool infernet_partial;\n\n    std::vector<bool> embeddings_layer_inp;");
  writeFileSync(path, text);
}

function patchLlamaContext() {
  const headerPath = join(sourceDir, "src", "llama-context.h");
  let header = readText(headerPath);
  header = replaceOnce(header,
    "    void set_embeddings_layer_inp(uint32_t lid, bool enable);\n    void set_nextn_layer_offset(int32_t offset);",
    "    void set_embeddings_layer_inp(uint32_t lid, bool enable);\n    void infernet_set_layer_range(uint32_t layer_start, uint32_t layer_end);\n    void set_nextn_layer_offset(int32_t offset);");
  writeFileSync(headerPath, header);

  const cppPath = join(sourceDir, "src", "llama-context.cpp");
  let text = readText(cppPath);
  text = replaceOnce(text,
    "    cparams.embeddings_nextn_masked = false;\n",
    "    cparams.embeddings_nextn_masked = false;\n    cparams.infernet_partial = params.infernet_partial;\n    cparams.infernet_layer_start = params.infernet_layer_start;\n    cparams.infernet_layer_end = params.infernet_layer_end;\n");
  text = replaceOnce(text,
    "        /*.n_outputs_max               =*/ 0,\n        /*.n_threads",
    "        /*.n_outputs_max               =*/ 0,\n        /*.infernet_layer_start        =*/ 0,\n        /*.infernet_layer_end          =*/ UINT32_MAX,\n        /*.infernet_partial            =*/ false,\n        /*.n_threads");
  text = replaceOnce(text,
    "void llama_context::set_nextn_layer_offset(int32_t offset) {\n    cparams.nextn_layer_offset = offset;\n}\n",
    "void llama_context::infernet_set_layer_range(uint32_t layer_start, uint32_t layer_end) {\n    GGML_ASSERT(layer_start < layer_end);\n    GGML_ASSERT(layer_end <= model.hparams.n_layer());\n    cparams.infernet_partial = true;\n    cparams.infernet_layer_start = layer_start;\n    cparams.infernet_layer_end = layer_end;\n    sched_need_reserve = true;\n}\n\nvoid llama_context::set_nextn_layer_offset(int32_t offset) {\n    cparams.nextn_layer_offset = offset;\n}\n");
  text = replaceOnce(text,
    "void llama_set_embeddings_layer_inp(llama_context * ctx, uint32_t lid, bool value) {\n    ctx->set_embeddings_layer_inp(lid, value);\n}\n",
    "void llama_set_embeddings_layer_inp(llama_context * ctx, uint32_t lid, bool value) {\n    ctx->set_embeddings_layer_inp(lid, value);\n}\n\nvoid llama_infernet_set_layer_range(llama_context * ctx, uint32_t layer_start, uint32_t layer_end) {\n    ctx->infernet_set_layer_range(layer_start, layer_end);\n}\n");
  writeFileSync(cppPath, text);
}

function patchLlamaGraph() {
  const headerPath = join(sourceDir, "src", "llama-graph.h");
  let header = readText(headerPath);
  header = replaceOnce(header,
    "        if (cparams.nextn_layer_offset != other.cparams.nextn_layer_offset) {\n            return false;\n        }\n\n        return\n            cparams.embeddings",
    "        if (cparams.nextn_layer_offset != other.cparams.nextn_layer_offset) {\n            return false;\n        }\n\n        if (cparams.infernet_partial != other.cparams.infernet_partial ||\n            cparams.infernet_layer_start != other.cparams.infernet_layer_start ||\n            cparams.infernet_layer_end != other.cparams.infernet_layer_end) {\n            return false;\n        }\n\n        return\n            cparams.embeddings");
  header = replaceOnce(header,
    "    ggml_tensor * build_inp_embd(ggml_tensor * tok_embd) const;\n    ggml_tensor * build_inp_pos() const;",
    "    ggml_tensor * build_inp_embd(ggml_tensor * tok_embd) const;\n    ggml_tensor * build_inp_hidden() const;\n    int infernet_layer_start() const;\n    int infernet_layer_end() const;\n    bool infernet_is_partial() const;\n    bool infernet_is_final_shard() const;\n    ggml_tensor * infernet_finish_or_forward(ggml_tensor * cur) const;\n    ggml_tensor * build_inp_pos() const;");
  writeFileSync(headerPath, header);

  const cppPath = join(sourceDir, "src", "llama-graph.cpp");
  let text = readText(cppPath);
  text = replaceAll(
    text,
    "std::min((int) cparams.infernet_layer_end, n_layer) : n_layer",
    "std::min((int) cparams.infernet_layer_end, (int) n_layer) : (int) n_layer"
  );
  text = replaceOnce(text,
    "ggml_tensor * llm_graph_context::build_inp_embd(ggml_tensor * tok_embd) const {\n",
    "int llm_graph_context::infernet_layer_start() const {\n    return cparams.infernet_partial ? (int) cparams.infernet_layer_start : 0;\n}\n\nint llm_graph_context::infernet_layer_end() const {\n    return cparams.infernet_partial ? std::min((int) cparams.infernet_layer_end, (int) n_layer) : (int) n_layer;\n}\n\nbool llm_graph_context::infernet_is_partial() const {\n    return cparams.infernet_partial;\n}\n\nbool llm_graph_context::infernet_is_final_shard() const {\n    return infernet_layer_end() >= n_layer;\n}\n\nggml_tensor * llm_graph_context::infernet_finish_or_forward(ggml_tensor * cur) const {\n    if (infernet_is_partial() && !infernet_is_final_shard()) {\n        cb(cur, \"infernet_boundary\", -1);\n        res->t_embd = cur;\n        ggml_build_forward_expand(gf, cur);\n        return cur;\n    }\n    return nullptr;\n}\n\nggml_tensor * llm_graph_context::build_inp_hidden() const {\n    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd);\n\n    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);\n    cb(inp->tokens, \"inp_tokens\", -1);\n    ggml_set_input(inp->tokens);\n    res->t_inp_tokens = inp->tokens;\n\n    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, ubatch.n_tokens);\n    cb(inp->embd, \"infernet_inp_hidden\", -1);\n    ggml_set_input(inp->embd);\n    inp->h = inp->embd;\n    res->t_inp_embd = inp->embd;\n\n    res->add_input(std::move(inp));\n    return res->t_inp_embd;\n}\n\nggml_tensor * llm_graph_context::build_inp_embd(ggml_tensor * tok_embd) const {\n");
  writeFileSync(cppPath, text);
}

function patchModelLoader() {
  const path = join(sourceDir, "src", "llama-model.cpp");
  let text = readText(path);
  text = replaceOnce(text,
    "    ml.done_getting_tensors();\n",
    "    ml.done_getting_tensors(params.infernet_partial);\n");
  writeFileSync(path, text);

  const loaderPath = join(sourceDir, "src", "llama-model-loader.cpp");
  let loader = readText(loaderPath);
  loader = replaceOnce(loader,
    "        if (!t_meta) {\n            if (flags & TENSOR_NOT_REQUIRED) {\n                return nullptr;\n            }\n            throw std::runtime_error(format(\"missing tensor '%s'\", tn.str().c_str()));\n        }\n",
    "        if (!t_meta) {\n            if ((flags & TENSOR_SKIP) || (flags & TENSOR_NOT_REQUIRED)) {\n                return nullptr;\n            }\n            throw std::runtime_error(format(\"missing tensor '%s'\", tn.str().c_str()));\n        }\n");
  loader = replaceOnce(loader,
    "            n_created++;\n\n            return nullptr;\n        }\n",
    "            if (!(flags & TENSOR_SKIP)) {\n                n_created++;\n            }\n\n            return nullptr;\n        }\n");
  writeFileSync(loaderPath, loader);

  const modelPath = join(sourceDir, "src", "llama-model.cpp");
  let model = readText(modelPath);
  model = replaceOnce(model,
    "                    if (arch == LLM_ARCH_DEEPSEEK4) {\n",
    "                    if (cparams.infernet_partial) {\n                        const uint32_t infernet_start = cparams.infernet_layer_start;\n                        const uint32_t infernet_end = cparams.infernet_layer_end;\n                        filter = [&, infernet_start, infernet_end](uint32_t il) {\n                            const bool assigned = il >= infernet_start && il < infernet_end;\n                            const bool gemma_shared_kv = (arch == LLM_ARCH_GEMMA3N || arch == LLM_ARCH_GEMMA4)\n                                && il < (uint32_t) hparams.n_layer_kv_from_start;\n                            return assigned || gemma_shared_kv;\n                        };\n                    }\n\n                    if (arch == LLM_ARCH_DEEPSEEK4) {\n");
  writeFileSync(modelPath, model);
}

function patchDecoderGraphs() {
  patchLlamaModel();
  patchGemmaModel("gemma.cpp", "llama_model_gemma::graph::graph", "LLM_FFN_GELU");
  patchGemma3Model();
  patchGemma4Model();
}

function patchLlamaModel() {
  const path = join(sourceDir, "src", "models", "llama.cpp");
  let text = readText(path);
  text = replaceOnce(text,
    "    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, \"weight\"), {n_embd, n_vocab}, 0);\n\n    // output\n    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, \"weight\"), {n_embd}, 0);\n    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      \"weight\"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);\n",
    "    const bool infernet_partial = params.infernet_partial;\n    const int infernet_start = infernet_partial ? (int) params.infernet_layer_start : 0;\n    const int infernet_end = infernet_partial ? std::min((int) params.infernet_layer_end, n_layer) : n_layer;\n    const bool infernet_needs_input = !infernet_partial || infernet_start == 0 || infernet_end >= n_layer;\n    const bool infernet_needs_output = !infernet_partial || infernet_end >= n_layer;\n    auto infernet_layer_flags = [&](int il) { return infernet_partial && (il < infernet_start || il >= infernet_end) ? TENSOR_SKIP : 0; };\n\n    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, \"weight\"), {n_embd, n_vocab}, infernet_needs_input ? 0 : TENSOR_SKIP);\n\n    // output\n    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, \"weight\"), {n_embd}, infernet_needs_output ? 0 : TENSOR_SKIP);\n    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      \"weight\"), {n_embd, n_vocab}, infernet_needs_output ? TENSOR_NOT_REQUIRED : TENSOR_SKIP);\n");
  text = replaceOnce(text,
    "        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, \"weight\", i), {n_embd}, 0);\n\n        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, 0);\n        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, \"weight\", i), {n_embd_head_k * n_head, n_embd}, 0);\n",
    "        const int flags = infernet_layer_flags(i);\n        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, \"weight\", i), {n_embd}, flags);\n\n        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, flags);\n        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, \"weight\", i), {n_embd_head_k * n_head, n_embd}, flags);\n");
  text = replaceOnce(text,
    "        layer.wo_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT, \"bias\", i), {n_embd}, TENSOR_NOT_REQUIRED);\n\n        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, \"weight\", i), {n_embd}, 0);\n",
    "        layer.wo_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT, \"bias\", i), {n_embd}, TENSOR_NOT_REQUIRED | flags);\n\n        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, \"weight\", i), {n_embd}, flags);\n");
  text = replaceOnce(text,
    "            layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, \"weight\", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));\n",
    "            layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, \"weight\", i), {n_rot/2}, TENSOR_NOT_REQUIRED | flags | (i != 0 ? TENSOR_DUPLICATED : 0));\n");
  text = replaceOnce(text,
    "            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, \"weight\", i), {n_embd,   n_ff}, 0);\n            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, \"weight\", i), {  n_ff, n_embd}, 0);\n            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   \"weight\", i), {n_embd,   n_ff}, 0);\n",
    "            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, \"weight\", i), {n_embd,   n_ff}, flags);\n            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, \"weight\", i), {  n_ff, n_embd}, flags);\n            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   \"weight\", i), {n_embd,   n_ff}, flags);\n");
  text = replaceOnce(text,
    "            layer.ffn_gate_b = create_tensor(tn(LLM_TENSOR_FFN_GATE, \"bias\", i), {n_ff}, TENSOR_NOT_REQUIRED);\n            layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, \"bias\", i), {n_embd}, TENSOR_NOT_REQUIRED);\n            layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   \"bias\", i), {n_ff}, TENSOR_NOT_REQUIRED);\n",
    "            layer.ffn_gate_b = create_tensor(tn(LLM_TENSOR_FFN_GATE, \"bias\", i), {n_ff}, TENSOR_NOT_REQUIRED | flags);\n            layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, \"bias\", i), {n_embd}, TENSOR_NOT_REQUIRED | flags);\n            layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   \"bias\", i), {n_ff}, TENSOR_NOT_REQUIRED | flags);\n");
  text = replaceOnce(text,
    "    inpL = build_inp_embd(model.tok_embd);\n",
    "    const int il_start = infernet_layer_start();\n    const int il_end = infernet_layer_end();\n\n    inpL = il_start > 0 ? build_inp_hidden() : build_inp_embd(model.tok_embd);\n");
  text = replaceOnce(text,
    "    for (int il = 0; il < n_layer; ++il) {\n",
    "    for (int il = il_start; il < il_end; ++il) {\n");
  text = replaceOnce(text,
    "        if (il == n_layer - 1 && inp_out_ids) {\n",
    "        if (il == il_end - 1 && il_end == n_layer && inp_out_ids) {\n");
  text = replaceOnce(text,
    "    cur = inpL;\n\n    cur = build_norm(cur,",
    "    cur = inpL;\n\n    if (infernet_finish_or_forward(cur)) {\n        return;\n    }\n\n    cur = build_norm(cur,");
  writeFileSync(path, text);
}

function patchGemmaModel(filename) {
  const path = join(sourceDir, "src", "models", filename);
  let text = readText(path);
  text = replaceOnce(text,
    "    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, \"weight\"), {n_embd, n_vocab}, 0);\n\n    // output\n    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, \"weight\"), {n_embd}, 0);\n    output      = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  \"weight\"), {n_embd, n_vocab}, TENSOR_DUPLICATED); // same as tok_embd, duplicated to allow offloading\n",
    "    const bool infernet_partial = params.infernet_partial;\n    const int infernet_start = infernet_partial ? (int) params.infernet_layer_start : 0;\n    const int infernet_end = infernet_partial ? std::min((int) params.infernet_layer_end, n_layer) : n_layer;\n    const bool infernet_needs_input = !infernet_partial || infernet_start == 0 || infernet_end >= n_layer;\n    const bool infernet_needs_output = !infernet_partial || infernet_end >= n_layer;\n    auto infernet_layer_flags = [&](int il) { return infernet_partial && (il < infernet_start || il >= infernet_end) ? TENSOR_SKIP : 0; };\n\n    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, \"weight\"), {n_embd, n_vocab}, infernet_needs_input ? 0 : TENSOR_SKIP);\n\n    // output\n    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, \"weight\"), {n_embd}, infernet_needs_output ? 0 : TENSOR_SKIP);\n    output      = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  \"weight\"), {n_embd, n_vocab}, infernet_needs_output ? TENSOR_DUPLICATED : TENSOR_SKIP); // same as tok_embd, duplicated to allow offloading\n");
  text = replaceOnce(text,
    "        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, \"weight\", i), {n_embd}, 0);\n\n        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, 0);\n        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, \"weight\", i), {n_embd_head_k * n_head, n_embd}, 0);\n\n        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, \"weight\", i), {n_embd}, 0);\n        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, \"weight\", i), {n_embd,   n_ff}, 0);\n        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   \"weight\", i), {n_embd,   n_ff}, 0);\n        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, \"weight\", i), {  n_ff, n_embd}, 0);\n",
    "        const int flags = infernet_layer_flags(i);\n        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, \"weight\", i), {n_embd}, flags);\n\n        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, flags);\n        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, \"weight\", i), {n_embd_head_k * n_head, n_embd}, flags);\n\n        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, \"weight\", i), {n_embd}, flags);\n        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, \"weight\", i), {n_embd,   n_ff}, flags);\n        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   \"weight\", i), {n_embd,   n_ff}, flags);\n        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, \"weight\", i), {  n_ff, n_embd}, flags);\n");
  text = replaceOnce(text,
    "    inpL = build_inp_embd(model.tok_embd);\n\n    inpL = ggml_scale(ctx0, inpL, sqrtf(n_embd));",
    "    const int il_start = infernet_layer_start();\n    const int il_end = infernet_layer_end();\n\n    inpL = il_start > 0 ? build_inp_hidden() : build_inp_embd(model.tok_embd);\n\n    if (il_start == 0) {\n        inpL = ggml_scale(ctx0, inpL, sqrtf(n_embd));\n    }");
  text = replaceOnce(text,
    "    for (int il = 0; il < n_layer; ++il) {\n",
    "    for (int il = il_start; il < il_end; ++il) {\n");
  text = replaceOnce(text,
    "        if (il == n_layer - 1 && inp_out_ids) {\n",
    "        if (il == il_end - 1 && il_end == n_layer && inp_out_ids) {\n");
  text = replaceOnce(text,
    "    cur = inpL;\n\n    cur = build_norm(cur,",
    "    cur = inpL;\n\n    if (infernet_finish_or_forward(cur)) {\n        return;\n    }\n\n    cur = build_norm(cur,");
  writeFileSync(path, text);
}

function patchGemma3Model() {
  const path = join(sourceDir, "src", "models", "gemma3.cpp");
  let text = readText(path);
  text = replaceOnce(text,
    "    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, \"weight\"), {n_embd, n_vocab}, 0);\n\n    // output\n    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, \"weight\"), {n_embd}, 0);\n    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      \"weight\"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);\n",
    "    const bool infernet_partial = params.infernet_partial;\n    const int infernet_start = infernet_partial ? (int) params.infernet_layer_start : 0;\n    const int infernet_end = infernet_partial ? std::min((int) params.infernet_layer_end, n_layer) : n_layer;\n    const bool infernet_needs_input = !infernet_partial || infernet_start == 0 || infernet_end >= n_layer;\n    const bool infernet_needs_output = !infernet_partial || infernet_end >= n_layer;\n    auto infernet_layer_flags = [&](int il) { return infernet_partial && (il < infernet_start || il >= infernet_end) ? TENSOR_SKIP : 0; };\n\n    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, \"weight\"), {n_embd, n_vocab}, infernet_needs_input ? 0 : TENSOR_SKIP);\n\n    // output\n    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, \"weight\"), {n_embd}, infernet_needs_output ? 0 : TENSOR_SKIP);\n    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      \"weight\"), {n_embd, n_vocab}, infernet_needs_output ? TENSOR_NOT_REQUIRED : TENSOR_SKIP);\n");
  text = replaceAll(text, "TENSOR_NOT_REQUIRED);", "TENSOR_NOT_REQUIRED);");
  text = replaceOnce(text,
    "        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, \"weight\", i), {n_embd}, 0);\n\n        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, 0);\n        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, \"weight\", i), {n_embd_head_k * n_head, n_embd}, 0);\n",
    "        const int flags = infernet_layer_flags(i);\n        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, \"weight\", i), {n_embd}, flags);\n\n        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, flags);\n        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, \"weight\", i), {n_embd_head_k * n_head, n_embd}, flags);\n");
  text = replaceOnce(text,
    "        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, \"weight\", i), {n_embd}, 0);\n        layer.attn_k_norm    = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM,    \"weight\", i), {n_embd_head_k}, 0);\n        layer.attn_q_norm    = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM,    \"weight\", i), {n_embd_head_k}, 0);\n\n        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, \"weight\", i), {n_embd}, 0);\n        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, \"weight\", i), {n_embd,   n_ff}, 0);\n        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   \"weight\", i), {n_embd,   n_ff}, 0);\n        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, \"weight\", i), {  n_ff, n_embd}, 0);\n        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, \"weight\", i), {n_embd}, 0);\n",
    "        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, \"weight\", i), {n_embd}, flags);\n        layer.attn_k_norm    = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM,    \"weight\", i), {n_embd_head_k}, flags);\n        layer.attn_q_norm    = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM,    \"weight\", i), {n_embd_head_k}, flags);\n\n        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, \"weight\", i), {n_embd}, flags);\n        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, \"weight\", i), {n_embd,   n_ff}, flags);\n        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   \"weight\", i), {n_embd,   n_ff}, flags);\n        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, \"weight\", i), {  n_ff, n_embd}, flags);\n        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, \"weight\", i), {n_embd}, flags);\n");
  text = replaceOnce(text,
    "    inpL = build_inp_embd(model.tok_embd);\n\n    // important: do not normalize weights for raw embeddings input (i.e. encoded image embeddings)\n    inpL = ggml_scale(ctx0, inpL, ubatch.token ? sqrtf(n_embd) : 1.0f);",
    "    const int il_start = infernet_layer_start();\n    const int il_end = infernet_layer_end();\n\n    inpL = il_start > 0 ? build_inp_hidden() : build_inp_embd(model.tok_embd);\n\n    // important: do not normalize weights for raw embeddings input (i.e. encoded image embeddings)\n    if (il_start == 0) {\n        inpL = ggml_scale(ctx0, inpL, ubatch.token ? sqrtf(n_embd) : 1.0f);\n    }");
  text = replaceOnce(text,
    "    for (int il = 0; il < n_layer; ++il) {\n",
    "    for (int il = il_start; il < il_end; ++il) {\n");
  text = replaceOnce(text,
    "        if (il == n_layer - 1 && inp_out_ids) {\n",
    "        if (il == il_end - 1 && il_end == n_layer && inp_out_ids) {\n");
  text = replaceOnce(text,
    "    cur = inpL;\n\n    cur = build_norm(cur,",
    "    cur = inpL;\n\n    if (infernet_finish_or_forward(cur)) {\n        return;\n    }\n\n    cur = build_norm(cur,");
  writeFileSync(path, text);
}

function patchGemma4Model() {
  const path = join(sourceDir, "src", "models", "gemma4.cpp");
  let text = readText(path);
  text = replaceAll(
    text,
    "        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, \"weight\", i), {n_embd}, flags);\n\n        // note: use_alternative_attention (v_proj is optional, if it's not present, use k_proj)\n        const int flags = infernet_layer_flags(i);\n",
    "        const int flags = infernet_layer_flags(i);\n        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, \"weight\", i), {n_embd}, flags);\n\n        // note: use_alternative_attention (v_proj is optional, if it's not present, use k_proj)\n"
  );
  text = replaceOnce(text,
    "    output = create_tensor(tn(LLM_TENSOR_OUTPUT, \"weight\"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);\n",
    "    const bool infernet_partial = params.infernet_partial;\n    const int infernet_start = infernet_partial ? (int) params.infernet_layer_start : 0;\n    const int infernet_end = infernet_partial ? std::min((int) params.infernet_layer_end, n_layer) : n_layer;\n    const bool infernet_needs_output = !infernet_partial || infernet_end >= n_layer;\n    auto infernet_layer_flags = [&](int il) { return infernet_partial && (il < infernet_start || il >= infernet_end) ? TENSOR_SKIP : 0; };\n\n    output = create_tensor(tn(LLM_TENSOR_OUTPUT, \"weight\"), {n_embd, n_vocab}, infernet_needs_output ? TENSOR_NOT_REQUIRED : TENSOR_SKIP);\n");
  text = replaceOnce(text,
    "    if (output == NULL) {\n        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, \"weight\"), {n_embd, n_vocab}, TENSOR_DUPLICATED);\n    }\n",
    "    if (output == NULL && infernet_needs_output) {\n        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, \"weight\"), {n_embd, n_vocab}, TENSOR_DUPLICATED);\n    }\n");
  text = replaceOnce(text,
    "    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, \"weight\"), {n_embd, n_vocab}, 0);\n",
    "    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, \"weight\"), {n_embd, n_vocab}, 0);\n");
  text = replaceOnce(text,
    "        per_layer_tok_embd   = create_tensor(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, \"weight\"),    {n_embd_per_layer * n_layer, n_vocab}, 0);\n",
    "        per_layer_tok_embd   = create_tensor(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, \"weight\"),    {n_embd_per_layer * n_layer, n_vocab}, 0);\n");
  text = replaceOnce(text,
    "    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, \"weight\"), {n_embd}, 0);\n",
    "    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, \"weight\"), {n_embd}, infernet_needs_output ? 0 : TENSOR_SKIP);\n");
  const unpatchedWq =
    "        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   \"weight\", i), {n_embd, n_embd_head * n_head}, 0);\n";
  const patchedWq =
    "        const int flags = infernet_layer_flags(i);\n        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   \"weight\", i), {n_embd, n_embd_head * n_head}, flags);\n";
  const patchedWqWithoutFlag =
    "        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   \"weight\", i), {n_embd, n_embd_head * n_head}, flags);\n";
  if (text.includes(patchedWqWithoutFlag)) {
    if (!text.includes("const int flags = infernet_layer_flags(i);")) {
      text = text.replace(patchedWqWithoutFlag, patchedWq);
    }
  } else {
    text = replaceOnce(text, unpatchedWq, patchedWq);
  }
  text = replaceAll(
    text,
    "        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, \"weight\", i), {n_embd}, flags);\n\n        // note: use_alternative_attention (v_proj is optional, if it's not present, use k_proj)\n        const int flags = infernet_layer_flags(i);\n",
    "        const int flags = infernet_layer_flags(i);\n        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, \"weight\", i), {n_embd}, flags);\n\n        // note: use_alternative_attention (v_proj is optional, if it's not present, use k_proj)\n"
  );
  text = text
    .replaceAll("{n_embd, n_embd_head * n_head_kv}, kv_flags)", "{n_embd, n_embd_head * n_head_kv}, kv_flags | flags)")
    .replaceAll("{n_embd_head * n_head, n_embd}, 0)", "{n_embd_head * n_head, n_embd}, flags)")
    .replaceAll("{n_embd_head}, 0)", "{n_embd_head}, flags)")
    .replaceAll("{n_embd_head}, kv_flags)", "{n_embd_head}, kv_flags | flags)")
    .replaceAll("{n_embd_head/2}, rope_freqs_flag)", "{n_embd_head/2}, rope_freqs_flag | flags)")
    .replaceAll("{n_embd,   n_ff}, 0)", "{n_embd,   n_ff}, flags)")
    .replaceAll("{  n_ff, n_embd}, 0)", "{  n_ff, n_embd}, flags)")
    .replaceAll("{n_embd}, 0)", "{n_embd}, flags)");
  text = text
    .replaceAll("{n_embd,   n_ff_cur}, 0)", "{n_embd,   n_ff_cur}, flags)")
    .replaceAll("{n_ff_cur, n_embd}, 0)", "{n_ff_cur, n_embd}, flags)")
    .replaceAll(
      "tn(LLM_TENSOR_FFN_GATE_INP, \"weight\", i), {n_embd, n_expert}, TENSOR_NOT_REQUIRED)",
      "tn(LLM_TENSOR_FFN_GATE_INP, \"weight\", i), {n_embd, n_expert}, TENSOR_NOT_REQUIRED | flags)"
    )
    .replaceAll("{n_embd, n_ff_exp, n_expert}, 0)", "{n_embd, n_ff_exp, n_expert}, flags)")
    .replaceAll("{n_ff_exp, n_embd, n_expert}, 0)", "{n_ff_exp, n_embd, n_expert}, flags)");
  text = replaceAll(
    text,
    "        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, \"weight\", i), {n_embd}, flags);\n\n        // note: use_alternative_attention (v_proj is optional, if it's not present, use k_proj)\n        const int flags = infernet_layer_flags(i);\n",
    "        const int flags = infernet_layer_flags(i);\n        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, \"weight\", i), {n_embd}, flags);\n\n        // note: use_alternative_attention (v_proj is optional, if it's not present, use k_proj)\n"
  );
  text = replaceOnce(text,
    "    inpL = build_inp_embd(model.tok_embd);\n\n    // important: do not normalize weights for raw embeddings input (i.e. encoded image emdeddings)\n    inpL = ggml_scale(ctx0, inpL, ubatch.token ? sqrtf(n_embd) : 1.0f);",
    "    const int il_start = infernet_layer_start();\n    const int il_end = infernet_layer_end();\n\n    ggml_tensor * tok_inp = nullptr;\n    if (il_start > 0) {\n        inpL = build_inp_hidden();\n        tok_inp = build_inp_embd(model.tok_embd);\n        tok_inp = ggml_scale(ctx0, tok_inp, ubatch.token ? sqrtf(n_embd) : 1.0f);\n    } else {\n        inpL = build_inp_embd(model.tok_embd);\n        // important: do not normalize weights for raw embeddings input (i.e. encoded image emdeddings)\n        inpL = ggml_scale(ctx0, inpL, ubatch.token ? sqrtf(n_embd) : 1.0f);\n        tok_inp = inpL;\n    }");
  text = replaceOnce(text,
    "        inp_per_layer = project_per_layer_inputs(inpL, inp_per_layer);\n",
    "        inp_per_layer = project_per_layer_inputs(tok_inp, inp_per_layer);\n");
  text = replaceOnce(text,
    "    for (int il = 0; il < n_layer; ++il) {\n",
    "    for (int il = il_start; il < il_end; ++il) {\n");
  text = replaceAll(text, "if (il == n_layer - 1 && inp_out_ids", "if (il == il_end - 1 && il_end == n_layer && inp_out_ids");
  text = replaceOnce(text,
    "    cur = inpL;\n\n    cur = build_norm(cur,",
    "    cur = inpL;\n\n    if (infernet_finish_or_forward(cur)) {\n        return;\n    }\n\n    cur = build_norm(cur,");
  writeFileSync(path, text);
}

function cmakeTargets() {
  const result = spawnSync("cmake", ["--build", buildDir, "--target", "help"], {
    cwd: repoRoot,
    encoding: "utf8",
  });
  if (result.status !== 0) {
    if (isWindows) {
      return findFilesRecursive(buildDir, (path) => path.toLowerCase().endsWith(".vcxproj"))
        .map((path) => basename(path, ".vcxproj"));
    }
    return [];
  }
  return result.stdout
    .split(/\r?\n/)
    .map((line) => line.trim().replace(/^\.\.\.\s*/, ""))
    .map((line) => line.split(/\s+/)[0])
    .filter(Boolean);
}

function copyRuntime(source, destination, label, options = {}) {
  const absolute = resolve(source);
  if (!fileExists(absolute)) {
    fail(`${label} did not point to an executable file: ${source}`);
  }
  copyFileSync(absolute, destination);
  if (!isWindows) {
    run("chmod", ["755", destination]);
  }
  if (options.validate === false) {
    return;
  }
  validatePreparedRuntime(destination, label);
}

function validatePreparedRuntime(path, label) {
  const validation = validateRuntime(path);
  if (!validation.ok) {
    safeUnlink(path);
    fail(`prepared runtime is not portable: ${validation.reason}`);
  }
  log(`prepared bundled runtime from ${label}: ${relative(path)}`);
}

function copyRuntimeDlls(directory) {
  const outputDir = dirname(sidecarPath);
  for (const dll of findFilesRecursive(directory, (path) => path.toLowerCase().endsWith(".dll"))) {
    copyFileSync(dll, join(outputDir, basename(dll)));
  }
}

function copyRuntimeDylibs(directory) {
  const outputDir = dirname(sidecarPath);
  for (const dylib of findFilesRecursive(directory, (path) => path.toLowerCase().endsWith(".dylib"))) {
    copyFileSync(dylib, join(outputDir, basename(dylib)));
  }
}

function validateRuntime(path) {
  if (process.platform === "darwin") {
    const result = spawnSync("otool", ["-L", path], { encoding: "utf8" });
    if (result.status !== 0) {
      return { ok: false, reason: "otool could not inspect Mach-O dependencies" };
    }
    const badDependency = result.stdout
      .split(/\r?\n/)
      .map((line) => line.trim())
      .find((line) => invalidMacosDependency(line, dirname(path)));
    if (badDependency) {
      return { ok: false, reason: `depends on non-bundled library ${badDependency.split(/\s+/)[0]}` };
    }
  }

  if (process.platform === "linux") {
    const result = spawnSync("ldd", [path], { encoding: "utf8" });
    if (result.status === 0) {
      const badDependency = result.stdout
        .split(/\r?\n/)
        .map((line) => line.trim())
        .find((line) => line.includes("libggml") || line.includes("libllama"));
      if (badDependency) {
        return { ok: false, reason: `depends on non-bundled library ${badDependency}` };
      }
    }
  }

  return { ok: true };
}

function readStamp() {
  try {
    return readFileSync(runtimeStampPath, "utf8").trim();
  } catch {
    return "";
  }
}

function writeStamp() {
  writeFileSync(runtimeStampPath, runtimeStamp + "\n");
}

function invalidMacosDependency(line, runtimeDir) {
  const dependency = line.split(/\s+/)[0];
  if (!dependency || dependency.endsWith(":")) {
    return false;
  }
  if (dependency.startsWith("/opt/homebrew/") || dependency.startsWith("/usr/local/")) {
    return true;
  }

  const name = basename(dependency);
  const isLlamaDependency =
    name.startsWith("libggml")
    || name.startsWith("libllama")
    || name.startsWith("libmtmd");
  if (!isLlamaDependency) {
    return false;
  }

  if (
    dependency.startsWith("@rpath/")
    || dependency.startsWith("@loader_path/")
    || dependency.startsWith("@executable_path/")
  ) {
    return !fileExists(join(runtimeDir, name));
  }

  return true;
}

function readBuildJobs() {
  const configured =
    process.env.INFERNET_LLAMA_BUILD_JOBS?.trim()
    || process.env.CMAKE_BUILD_PARALLEL_LEVEL?.trim()
    || "2";
  const jobs = Number.parseInt(configured, 10);
  if (!/^\d+$/.test(configured) || !Number.isInteger(jobs) || jobs < 1) {
    fail("INFERNET_LLAMA_BUILD_JOBS must be a positive integer");
  }
  // Native llama.cpp compilation is intentionally capped. Unbounded parallel
  // builds previously made development machines unresponsive.
  return Math.min(jobs, 2);
}

function environmentFlag(name) {
  return ["1", "true", "yes", "on"].includes((process.env[name] || "").trim().toLowerCase());
}

function shouldEnableCuda() {
  const configured = process.env.INFERNET_CUDA?.trim().toLowerCase();
  if (["0", "false", "no", "off"].includes(configured)) {
    return false;
  }
  if (["1", "true", "yes", "on"].includes(configured)) {
    return true;
  }
  return !isMacos && commandAvailable(isWindows ? "nvidia-smi.exe" : "nvidia-smi");
}

function printBuildPlan() {
  console.log(JSON.stringify({
    llamaRef,
    targetTriple,
    buildJobs,
    cmakeOptions: [
      "-DGGML_RPC=ON",
      "-DBUILD_SHARED_LIBS=OFF",
      "-DGGML_BACKEND_DL=OFF",
      "-DGGML_NATIVE=OFF",
      `-DGGML_CUDA=${cudaEnabled ? "ON" : "OFF"}`,
      `-DGGML_METAL=${isMacos ? "ON" : "OFF"}`,
      ...(cudaEnabled ? [`-DCMAKE_CUDA_ARCHITECTURES=${cudaArchitectures}`] : []),
    ],
    targets: ["llama-cli", "infernet-llama-bridge", rpcServerTarget, serverTarget],
    sidecars: [
      relative(sidecarPath),
      relative(bridgeSidecarPath),
      relative(rpcServerSidecarPath),
      relative(serverSidecarPath),
    ],
  }, null, 2));
}

function rustTargetTriple() {
  const output = spawnSync("rustc", ["-vV"], { encoding: "utf8" });
  if (output.status !== 0) {
    fail("could not determine Rust target triple with rustc -vV");
  }
  const host = output.stdout
    .split(/\r?\n/)
    .find((line) => line.startsWith("host:"))
    ?.replace("host:", "")
    .trim();
  if (!host) {
    fail("rustc -vV did not report a host triple");
  }
  return host;
}

function findOnPath(names) {
  const path = process.env.PATH || "";
  for (const directory of path.split(process.platform === "win32" ? ";" : ":")) {
    if (!directory) {
      continue;
    }
    for (const name of names) {
      const candidate = join(directory, name);
      if (fileExists(candidate)) {
        return candidate;
      }
    }
  }
  return null;
}

function ensureSourceBuildRequirements() {
  const missing = missingSourceBuildRequirements();
  if (missing.length === 0) {
    return;
  }

  fail(
    `cannot build required Infernet split-layer bridge; missing requirement(s): ${missing.join("; ")}.\n` +
    sourceBuildInstallHint()
  );
}

function missingSourceBuildRequirements() {
  const missing = [];
  if (!commandAvailable("git")) {
    missing.push("git");
  }
  if (!commandAvailable("cmake")) {
    missing.push("cmake");
  }
  if (!hasCppToolchain()) {
    missing.push(cppToolchainRequirement());
  }
  return missing;
}

function hasCppToolchain() {
  if (isWindows) {
    return commandExists("cl", ["/?"])
      || commandSuccessful("clang++", ["--version"])
      || commandSuccessful("g++", ["--version"])
      || visualStudioCppToolsInstalled();
  }
  if (isMacos) {
    return commandSuccessful("clang++", ["--version"]) || commandSuccessful("xcrun", ["-find", "clang++"]);
  }
  return commandSuccessful("c++", ["--version"])
    || commandSuccessful("g++", ["--version"])
    || commandSuccessful("clang++", ["--version"]);
}

function cppToolchainRequirement() {
  if (isWindows) {
    return "C++ compiler toolchain (Visual Studio Build Tools 2022 with Desktop development with C++, or clang++)";
  }
  if (isMacos) {
    return "C++ compiler toolchain (Xcode Command Line Tools: xcode-select --install)";
  }
  return "C++ compiler toolchain (build-essential/g++ or clang++)";
}

function sourceBuildInstallHint() {
  if (isWindows) {
    return [
      "Install on Windows:",
      "  winget install --id Kitware.CMake -e",
      "  winget install --id Microsoft.VisualStudio.2022.BuildTools -e --override \"--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive --wait\"",
      "Then open a new PowerShell and rerun npm run prepare-runtime."
    ].join("\n");
  }
  if (isMacos) {
    return [
      "Install on macOS:",
      "  xcode-select --install",
      "  brew install cmake",
      "Then rerun npm run prepare-runtime."
    ].join("\n");
  }
  return [
    "Install on Debian/Ubuntu:",
    "  sudo apt-get update && sudo apt-get install -y git cmake build-essential",
    "Then rerun npm run prepare-runtime."
  ].join("\n");
}

function visualStudioCppToolsInstalled() {
  const candidates = ["vswhere"];
  const programFilesX86 = process.env["ProgramFiles(x86)"];
  if (programFilesX86) {
    candidates.push(join(programFilesX86, "Microsoft Visual Studio", "Installer", "vswhere.exe"));
  }
  for (const candidate of candidates) {
    const result = spawnSync(candidate, [
      "-latest",
      "-products",
      "*",
      "-requires",
      "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
      "-property",
      "installationPath",
    ], { encoding: "utf8" });
    if (!result.error && result.status === 0 && result.stdout.trim()) {
      return true;
    }
  }
  return false;
}

function commandAvailable(command) {
  const result = spawnSync(command, ["--version"], { encoding: "utf8" });
  if (result.error || result.status !== 0) {
    return false;
  }
  return true;
}

function commandExists(command, args = ["--version"]) {
  const result = spawnSync(command, args, { encoding: "utf8" });
  return !result.error;
}

function commandSuccessful(command, args = ["--version"]) {
  const result = spawnSync(command, args, { encoding: "utf8" });
  return !result.error && result.status === 0;
}

function downloadText(url) {
  return download(url).toString("utf8");
}

function downloadFile(url, destination) {
  const data = download(url);
  writeFileSync(destination, data);
}

function download(url, redirects = 0) {
  if (redirects > 5) {
    fail(`too many redirects while downloading ${url}`);
  }

  const script = `
const chunks = [];
const { get: httpGet } = require(${JSON.stringify(url.startsWith("https:") ? "https" : "http")});
httpGet(${JSON.stringify(url)}, { headers: { "User-Agent": "infernet-runtime-prep" } }, (res) => {
  if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
    console.error("INFERNET_REDIRECT " + new URL(res.headers.location, ${JSON.stringify(url)}).toString());
    process.exit(23);
  }
  if (res.statusCode !== 200) {
    console.error("download failed with HTTP " + res.statusCode);
    process.exit(1);
  }
  res.on("data", (chunk) => chunks.push(chunk));
  res.on("end", () => process.stdout.write(Buffer.concat(chunks)));
}).on("error", (error) => {
  console.error(error.message);
  process.exit(1);
});
`;
  const result = spawnSync(process.execPath, ["-e", script], {
    encoding: "buffer",
    maxBuffer: 1024 * 1024 * 512,
  });
  if (result.status === 23) {
    const match = result.stderr.toString("utf8").match(/INFERNET_REDIRECT (.+)/);
    if (!match) {
      fail(`download redirect missing location for ${url}`);
    }
    return download(match[1].trim(), redirects + 1);
  }
  if (result.status !== 0) {
    fail(result.stderr.toString("utf8").trim() || `download failed for ${url}`);
  }
  return result.stdout;
}

function run(command, args, options = {}) {
  log(`${command} ${args.join(" ")}`);
  const result = spawnSync(command, args, {
    cwd: options.cwd || repoRoot,
    stdio: quiet ? "ignore" : "inherit",
  });
  if (result.status !== 0 && !options.optional) {
    fail(`${command} failed with exit code ${result.status}`);
  }
}

function findFileRecursive(root, name) {
  return findFilesRecursive(root, (path) => basename(path).toLowerCase() === name.toLowerCase())[0] || null;
}

function findFilesRecursive(root, predicate) {
  const matches = [];
  const stack = [root];
  while (stack.length > 0) {
    const current = stack.pop();
    for (const entry of readdirSync(current, { withFileTypes: true })) {
      const path = join(current, entry.name);
      if (entry.isDirectory()) {
        stack.push(path);
      } else if ((entry.isFile() || entry.isSymbolicLink()) && predicate(path)) {
        matches.push(path);
      }
    }
  }
  return matches;
}

function fileExists(path) {
  try {
    return statSync(path).isFile();
  } catch {
    return false;
  }
}

function safeUnlink(path) {
  try {
    unlinkSync(path);
  } catch (error) {
    if (error.code !== "ENOENT") {
      throw error;
    }
  }
}

function readText(path) {
  return readFileSync(path, "utf8").replace(/\r\n?/g, "\n");
}

function replaceOnce(text, search, replacement) {
  if (text.includes(replacement)) {
    return text;
  }
  if (!text.includes(search)) {
    fail(`llama.cpp patch target not found:\n${search.slice(0, 240)}`);
  }
  return text.replace(search, replacement);
}

function replaceAll(text, search, replacement) {
  if (!text.includes(search)) {
    return text;
  }
  return text.split(search).join(replacement);
}

function relative(path) {
  return path.replace(`${repoRoot}/`, "");
}

function log(message) {
  if (!quiet) {
    console.log(message);
  }
}

function powershellQuote(value) {
  return `'${value.replace(/'/g, "''")}'`;
}

function fail(message) {
  console.error(message);
  process.exit(1);
}
