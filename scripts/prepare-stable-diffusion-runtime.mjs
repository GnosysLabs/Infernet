#!/usr/bin/env node

import {
  chmodSync,
  copyFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  renameSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const STABLE_DIFFUSION_CPP_REF =
  process.env.STABLE_DIFFUSION_CPP_REF?.trim()
  || "cc734292286f85f9c48305d94d7fd22f42838522";
const EXPECTED_STABLE_DIFFUSION_CPP_REF =
  "cc734292286f85f9c48305d94d7fd22f42838522";
const RUNTIME_FORMAT_VERSION = "infernet-sd-cli-v2";
const SOURCE_REPOSITORY = "https://github.com/leejet/stable-diffusion.cpp.git";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(scriptDir, "..");
const quiet = process.argv.includes("--quiet");
const dryRun = process.argv.includes("--dry-run");
const verifyOnly = process.argv.includes("--verify-only");
const hostTriple = rustHostTriple();
const targetTriple = process.env.TARGET?.trim() || hostTriple;
const isWindowsTarget = targetTriple.includes("windows");
const isMacosTarget = targetTriple.includes("apple-darwin");
const executableSuffix = isWindowsTarget ? ".exe" : "";
const backend = isMacosTarget ? "metal" : "cpu";
const buildJobs = readBuildJobs();
const imageRpcSourceDir = resolve(repoRoot, "stable-diffusion-runtime");
const imageRpcOverlay = join(imageRpcSourceDir, "infernet-overlay.cmake");

const runtimeRoot = resolve(repoRoot, "target", "stable-diffusion.cpp-runtime");
const sourceDir = join(runtimeRoot, "source");
// Keep this path stable so an existing pinned build can be reused incrementally.
const buildDir = join(runtimeRoot, "build");
const sidecarDir = resolve(repoRoot, "infernet-ui", "src-tauri", "binaries");
const sidecarPath = join(
  sidecarDir,
  `sd-cli-${targetTriple}${executableSuffix}`,
);
const imageRpcSidecarPath = join(
  sidecarDir,
  `infernet-image-rpc-server-${targetTriple}${executableSuffix}`,
);
const stampPath = join(
  sidecarDir,
  `.infernet-stable-diffusion-runtime-${targetTriple}.stamp`,
);
const lockPath = join(runtimeRoot, `.prepare-${targetTriple}.lock`);
const runtimeStamp = [
  RUNTIME_FORMAT_VERSION,
  STABLE_DIFFUSION_CPP_REF,
  targetTriple,
  backend,
  "rpc-on",
  "image-rpc-worker",
  "webp-off",
  "webm-off",
  "static",
].join(":");

const cmakeConfigureArgs = [
  "-S", sourceDir,
  "-B", buildDir,
  "-DCMAKE_BUILD_TYPE=Release",
  "-DSD_BUILD_EXAMPLES=ON",
  "-DSD_BUILD_SHARED_LIBS=OFF",
  "-DSD_BUILD_SHARED_GGML_LIB=OFF",
  "-DSD_USE_SYSTEM_GGML=OFF",
  "-DSD_WEBP=OFF",
  "-DSD_WEBM=OFF",
  "-DSD_RPC=ON",
  `-DINFERNET_IMAGE_RPC_SOURCE_DIR=${imageRpcSourceDir}`,
  "-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=",
  `-DCMAKE_PROJECT_INCLUDE=${imageRpcOverlay}`,
  "-DSD_CUDA=OFF",
  `-DSD_METAL=${isMacosTarget ? "ON" : "OFF"}`,
  "-DSD_VULKAN=OFF",
];
if (isWindowsTarget) {
  cmakeConfigureArgs.push("-DCMAKE_CXX_FLAGS=/bigobj");
}
if (targetTriple === "aarch64-apple-darwin") {
  cmakeConfigureArgs.push("-DCMAKE_OSX_ARCHITECTURES=arm64");
} else if (targetTriple === "x86_64-apple-darwin") {
  cmakeConfigureArgs.push("-DCMAKE_OSX_ARCHITECTURES=x86_64");
}

if (STABLE_DIFFUSION_CPP_REF !== EXPECTED_STABLE_DIFFUSION_CPP_REF) {
  fail(
    "STABLE_DIFFUSION_CPP_REF must remain pinned to "
      + `${EXPECTED_STABLE_DIFFUSION_CPP_REF} for the Infernet Image v1 runtime`,
  );
}

if (dryRun) {
  printBuildPlan();
} else if (verifyOnly) {
  validatePreparedRuntime(sidecarPath);
  validatePreparedImageRpcRuntime(imageRpcSidecarPath);
  log(`verified bundled stable-diffusion.cpp runtime: ${relative(sidecarPath)}`);
  log(`verified bundled Infernet Image RPC worker: ${relative(imageRpcSidecarPath)}`);
} else {
  withRuntimeLock(main);
}

function main() {
  mkdirSync(sidecarDir, { recursive: true });

  if (fileExists(sidecarPath) && fileExists(imageRpcSidecarPath) && readStamp() === runtimeStamp) {
    const validation = validateRuntime(sidecarPath);
    const rpcValidation = validateImageRpcRuntime(imageRpcSidecarPath);
    if (validation.ok && rpcValidation.ok) {
      log(`bundled stable-diffusion.cpp runtime already exists: ${relative(sidecarPath)}`);
      log(`bundled Infernet Image RPC worker already exists: ${relative(imageRpcSidecarPath)}`);
      return;
    }
    log(`rebuilding bundled stable-diffusion.cpp runtime: ${(validation.ok ? rpcValidation : validation).reason}`);
  }

  const allowExternal = environmentFlag("INFERNET_ALLOW_EXTERNAL_SD_RUNTIME");
  const configuredCli = process.env.INFERNET_SD_CLI?.trim();
  const configuredImageRpcServer = process.env.INFERNET_IMAGE_RPC_SERVER?.trim();
  if (configuredCli) {
    if (!allowExternal) {
      fail(
        "INFERNET_SD_CLI requires INFERNET_ALLOW_EXTERNAL_SD_RUNTIME=1 so an "
          + "unpinned runtime cannot be selected accidentally",
      );
    }
    if (targetTriple !== hostTriple) {
      fail("cross-target stable-diffusion.cpp sidecars cannot be executed and verified on this host");
    }
    if (!configuredImageRpcServer) {
      fail("INFERNET_SD_CLI also requires an exact INFERNET_IMAGE_RPC_SERVER build");
    }
    installSidecar(configuredCli, sidecarPath, "INFERNET_SD_CLI", validatePreparedRuntime);
    installSidecar(
      configuredImageRpcServer,
      imageRpcSidecarPath,
      "INFERNET_IMAGE_RPC_SERVER",
      validatePreparedImageRpcRuntime,
    );
    writeStamp();
    return;
  }

  if (targetTriple !== hostTriple) {
    fail(
      `cross-target stable-diffusion.cpp preparation (${hostTriple} -> ${targetTriple}) `
        + "requires a separately verified native build on the target platform",
    );
  }

  requireCommand("git");
  requireCommand("cmake");
  prepareSource();
  configureAndBuild();
  installSidecar(
    findBuiltCli(),
    sidecarPath,
    "pinned stable-diffusion.cpp source build",
    validatePreparedRuntime,
  );
  installSidecar(
    findBuiltImageRpcServer(),
    imageRpcSidecarPath,
    "pinned stable-diffusion.cpp GGML source build",
    validatePreparedImageRpcRuntime,
  );
  writeStamp();
}

function prepareSource() {
  let freshClone = false;
  if (!existsSync(sourceDir)) {
    mkdirSync(runtimeRoot, { recursive: true });
    run("git", ["clone", "--filter=blob:none", "--no-checkout", SOURCE_REPOSITORY, sourceDir]);
    freshClone = true;
  } else if (!existsSync(join(sourceDir, ".git"))) {
    fail(`${relative(sourceDir)} exists but is not a git checkout`);
  }

  if (!freshClone) {
    const dirty = capture("git", ["status", "--porcelain", "--untracked-files=no"], {
      cwd: sourceDir,
    }).trim();
    if (dirty) {
      fail(
        `${relative(sourceDir)} has local changes; preserve or remove them before preparing the runtime`,
      );
    }
  }

  const currentRef = capture("git", ["rev-parse", "HEAD"], { cwd: sourceDir }).trim();
  if (freshClone || currentRef !== STABLE_DIFFUSION_CPP_REF) {
    run("git", ["fetch", "--depth", "1", "origin", STABLE_DIFFUSION_CPP_REF], {
      cwd: sourceDir,
    });
    run("git", ["checkout", "--detach", STABLE_DIFFUSION_CPP_REF], { cwd: sourceDir });
  }
  run("git", ["submodule", "sync", "--recursive"], { cwd: sourceDir });
  run("git", ["submodule", "update", "--init", "--recursive", "--depth", "1"], {
    cwd: sourceDir,
  });

  const preparedRef = capture("git", ["rev-parse", "HEAD"], { cwd: sourceDir }).trim();
  if (preparedRef !== STABLE_DIFFUSION_CPP_REF) {
    fail(`stable-diffusion.cpp checkout is ${preparedRef}; expected ${STABLE_DIFFUSION_CPP_REF}`);
  }
}

function configureAndBuild() {
  run("cmake", cmakeConfigureArgs);
  run("cmake", [
    "--build", buildDir,
    "--config", "Release",
    "--target", "sd-cli", "infernet-image-rpc-server",
    "--parallel", String(buildJobs),
  ]);
}

function findBuiltCli() {
  const executableName = `sd-cli${executableSuffix}`;
  const candidates = [
    join(buildDir, "bin", executableName),
    join(buildDir, "bin", "Release", executableName),
    ...findFilesRecursive(buildDir, (path) => basename(path) === executableName),
  ];
  const built = candidates.find(fileExists);
  if (!built) {
    fail(`stable-diffusion.cpp build did not produce ${executableName}`);
  }
  return built;
}

function findBuiltImageRpcServer() {
  const executableName = `infernet-image-rpc-server${executableSuffix}`;
  const candidates = [
    join(buildDir, "bin", executableName),
    join(buildDir, "bin", "Release", executableName),
    ...findFilesRecursive(buildDir, (path) => basename(path) === executableName),
  ];
  const built = candidates.find(fileExists);
  if (!built) {
    fail(`stable-diffusion.cpp build did not produce ${executableName}`);
  }
  return built;
}

function installSidecar(source, destination, label, validate) {
  const absoluteSource = resolve(source);
  if (!fileExists(absoluteSource)) {
    fail(`${label} did not point to an executable file: ${source}`);
  }
  const temporary = `${destination}.tmp-${process.pid}`;
  rmSync(temporary, { force: true });
  copyFileSync(absoluteSource, temporary);
  if (!isWindowsTarget) {
    chmodSync(temporary, 0o755);
  }
  validate(temporary);
  rmSync(destination, { force: true });
  renameSync(temporary, destination);
  log(`prepared ${relative(destination)} from ${label}`);
}

function validatePreparedRuntime(path) {
  const validation = validateRuntime(path);
  if (!validation.ok) {
    fail(`stable-diffusion.cpp runtime validation failed: ${validation.reason}`);
  }
}

function validatePreparedImageRpcRuntime(path) {
  const validation = validateImageRpcRuntime(path);
  if (!validation.ok) {
    fail(`Infernet Image RPC runtime validation failed: ${validation.reason}`);
  }
}

function validateRuntime(path) {
  if (!fileExists(path)) {
    return { ok: false, reason: `missing executable ${relative(path)}` };
  }

  const help = spawnSync(path, ["--help"], {
    encoding: "utf8",
    timeout: 15_000,
    maxBuffer: 8 * 1024 * 1024,
  });
  if (help.error || help.status !== 0) {
    return { ok: false, reason: "sd-cli --help did not exit successfully" };
  }
  const versionText = `${help.stdout || ""}\n${help.stderr || ""}`;
  if (!versionText.includes("stable-diffusion.cpp version") || !versionText.includes("cc73429")) {
    return { ok: false, reason: "sd-cli does not report the pinned cc73429 revision" };
  }

  return validatePortableBinary(path, "sd-cli");
}

function validateImageRpcRuntime(path) {
  if (!fileExists(path)) {
    return { ok: false, reason: `missing executable ${relative(path)}` };
  }
  const version = spawnSync(path, ["--version"], {
    encoding: "utf8",
    timeout: 15_000,
    maxBuffer: 8 * 1024 * 1024,
  });
  const versionText = `${version.stdout || ""}\n${version.stderr || ""}`;
  if (version.error || version.status !== 0
      || !versionText.includes("Infernet Image RPC stable-diffusion.cpp")
      || !versionText.includes(STABLE_DIFFUSION_CPP_REF)) {
    return { ok: false, reason: "image RPC worker does not report the pinned runtime revision" };
  }
  return validatePortableBinary(path, "Infernet Image RPC worker");
}

function validatePortableBinary(path, label) {

  if (process.platform === "darwin") {
    const inspection = spawnSync("otool", ["-L", path], { encoding: "utf8" });
    if (inspection.status !== 0) {
      return { ok: false, reason: `otool could not inspect ${label} dependencies` };
    }
    const badDependency = inspection.stdout
      .split(/\r?\n/)
      .map((line) => line.trim().split(/\s+/)[0])
      .find((dependency) => dependency?.startsWith("/opt/homebrew/")
        || dependency?.startsWith("/usr/local/"));
    if (badDependency) {
      return { ok: false, reason: `${label} depends on non-system library ${badDependency}` };
    }
    if (targetTriple === "aarch64-apple-darwin") {
      const architecture = spawnSync("file", [path], { encoding: "utf8" });
      if (architecture.status !== 0 || !architecture.stdout.includes("arm64")) {
        return { ok: false, reason: `${label} is not an arm64 Mach-O executable` };
      }
    }
  }

  if (process.platform === "linux") {
    const inspection = spawnSync("ldd", [path], { encoding: "utf8" });
    const badDependency = inspection.status === 0
      ? inspection.stdout
        .split(/\r?\n/)
        .find((line) => line.includes("libggml") || line.includes("libstable-diffusion"))
      : null;
    if (badDependency) {
      return { ok: false, reason: `${label} depends on non-bundled library ${badDependency.trim()}` };
    }
  }

  return { ok: true };
}

function withRuntimeLock(task) {
  mkdirSync(runtimeRoot, { recursive: true });
  const waitBuffer = new Int32Array(new SharedArrayBuffer(4));
  while (true) {
    try {
      mkdirSync(lockPath);
      writeFileSync(join(lockPath, "pid"), `${process.pid}\n`);
      break;
    } catch (error) {
      if (error?.code !== "EEXIST") throw error;
    }

    let ownerAlive = false;
    try {
      const ownerPid = Number.parseInt(readFileSync(join(lockPath, "pid"), "utf8"), 10);
      if (Number.isInteger(ownerPid) && ownerPid > 0) {
        process.kill(ownerPid, 0);
        ownerAlive = true;
      }
    } catch {
      ownerAlive = false;
    }
    if (!ownerAlive) {
      rmSync(lockPath, { recursive: true, force: true });
      continue;
    }
    Atomics.wait(waitBuffer, 0, 0, 250);
  }

  try {
    task();
  } finally {
    rmSync(lockPath, { recursive: true, force: true });
  }
}

function printBuildPlan() {
  process.stdout.write(`${JSON.stringify({
    sourceRepository: SOURCE_REPOSITORY,
    sourceRef: STABLE_DIFFUSION_CPP_REF,
    hostTriple,
    targetTriple,
    backend,
    sourceDir,
    buildDir,
    sidecarPath,
    imageRpcSidecarPath,
    stampPath,
    cmakeConfigureArgs,
    buildJobs,
  }, null, 2)}\n`);
}

function readStamp() {
  try {
    return readFileSync(stampPath, "utf8").trim();
  } catch {
    return "";
  }
}

function writeStamp() {
  writeFileSync(stampPath, `${runtimeStamp}\n`);
}

function rustHostTriple() {
  const result = spawnSync("rustc", ["-vV"], { encoding: "utf8" });
  if (!result.error && result.status === 0) {
    const host = result.stdout.match(/^host:\s*(.+)$/m)?.[1]?.trim();
    if (host) return host;
  }
  if (process.platform === "darwin") {
    return process.arch === "arm64" ? "aarch64-apple-darwin" : "x86_64-apple-darwin";
  }
  if (process.platform === "win32") {
    return process.arch === "arm64" ? "aarch64-pc-windows-msvc" : "x86_64-pc-windows-msvc";
  }
  return process.arch === "arm64" ? "aarch64-unknown-linux-gnu" : "x86_64-unknown-linux-gnu";
}

function readBuildJobs() {
  const configured = process.env.INFERNET_SD_BUILD_JOBS?.trim()
    || process.env.CMAKE_BUILD_PARALLEL_LEVEL?.trim()
    || "2";
  if (!/^\d+$/.test(configured)) {
    fail("INFERNET_SD_BUILD_JOBS must be a positive integer");
  }
  const jobs = Number.parseInt(configured, 10);
  if (!Number.isInteger(jobs) || jobs < 1) {
    fail("INFERNET_SD_BUILD_JOBS must be a positive integer");
  }
  return Math.min(jobs, 4);
}

function environmentFlag(name) {
  return ["1", "true", "yes", "on"].includes(
    (process.env[name] || "").trim().toLowerCase(),
  );
}

function requireCommand(command) {
  const result = spawnSync(command, ["--version"], { encoding: "utf8" });
  if (result.error || result.status !== 0) {
    fail(`${command} is required to prepare stable-diffusion.cpp`);
  }
}

function capture(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: options.cwd || repoRoot,
    encoding: "utf8",
    maxBuffer: 16 * 1024 * 1024,
  });
  if (result.error || result.status !== 0) {
    fail(`${command} ${args.join(" ")} failed`);
  }
  return result.stdout;
}

function run(command, args, options = {}) {
  log(`${command} ${args.join(" ")}`);
  const result = spawnSync(command, args, {
    cwd: options.cwd || repoRoot,
    stdio: quiet ? "ignore" : "inherit",
  });
  if (result.error || result.status !== 0) {
    fail(`${command} failed with exit code ${result.status ?? "unknown"}`);
  }
}

function findFilesRecursive(root, predicate) {
  if (!existsSync(root)) return [];
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

function relative(path) {
  return path.startsWith(`${repoRoot}/`) ? path.slice(repoRoot.length + 1) : path;
}

function log(message) {
  if (!quiet) process.stdout.write(`${message}\n`);
}

function fail(message) {
  process.stderr.write(`stable-diffusion.cpp runtime preparation failed: ${message}\n`);
  process.exit(1);
}
