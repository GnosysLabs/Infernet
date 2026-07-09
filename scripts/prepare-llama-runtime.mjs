#!/usr/bin/env node
import { copyFileSync, existsSync, mkdirSync, statSync, unlinkSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(scriptDir, "..");
const quiet = process.argv.includes("--quiet");
const targetTriple = process.env.TARGET || rustTargetTriple();
const isWindows = targetTriple.includes("windows");
const executableSuffix = isWindows ? ".exe" : "";
const sidecarBase = resolve(repoRoot, "infernet-ui", "src-tauri", "binaries", "llama-cli");
const sidecarPath = `${sidecarBase}-${targetTriple}${executableSuffix}`;
const buildRoot = resolve(repoRoot, "target", "llama.cpp-runtime");
const sourceDir = join(buildRoot, "llama.cpp");
const buildDir = join(buildRoot, `build-${targetTriple}`);
const llamaRef = process.env.LLAMA_CPP_REF || "master";

main();

function main() {
  mkdirSync(dirname(sidecarPath), { recursive: true });
  if (fileExists(sidecarPath)) {
    const validation = validateRuntime(sidecarPath);
    if (validation.ok) {
      log(`bundled llama.cpp runtime already exists: ${relative(sidecarPath)}`);
      return;
    }
    log(`rebuilding bundled llama.cpp runtime: ${validation.reason}`);
    unlinkSync(sidecarPath);
  }

  const configured = process.env.INFERNET_LLAMA_CLI?.trim();
  if (configured) {
    copyRuntime(configured, "INFERNET_LLAMA_CLI");
    return;
  }

  const fromPath = findOnPath(isWindows ? ["llama-cli.exe", "llama.exe", "main.exe"] : ["llama-cli", "llama", "main"]);
  if (fromPath) {
    copyRuntime(fromPath, "PATH");
    return;
  }

  buildFromSource();
}

function buildFromSource() {
  ensureCommand("git");
  ensureCommand("cmake");

  mkdirSync(buildRoot, { recursive: true });
  if (!existsSync(join(sourceDir, ".git"))) {
    run("git", ["clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp.git", sourceDir]);
  }

  run("git", ["fetch", "--depth", "1", "origin", llamaRef], { cwd: sourceDir, optional: true });
  run("git", ["checkout", llamaRef], { cwd: sourceDir });

  const cmakeArgs = [
    "-S", sourceDir,
    "-B", buildDir,
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_SHARED_LIBS=OFF",
    "-DGGML_BACKEND_DL=OFF",
    "-DLLAMA_CURL=OFF",
    "-DLLAMA_OPENSSL=OFF",
    "-DLLAMA_BUILD_COMMON=ON",
    "-DLLAMA_BUILD_TOOLS=ON",
    "-DLLAMA_BUILD_SERVER=ON",
    "-DLLAMA_BUILD_UI=OFF",
    "-DLLAMA_USE_PREBUILT_UI=OFF",
    "-DLLAMA_BUILD_TESTS=OFF",
  ];
  if (process.platform === "darwin") {
    cmakeArgs.push("-DGGML_METAL=ON");
  }

  run("cmake", cmakeArgs);
  const targets = cmakeTargets();
  const cliTarget = ["llama-cli", "main"].find((target) => targets.includes(target));
  if (!cliTarget) {
    fail(`no supported llama.cpp CLI target found in ${buildDir}`);
  }
  run("cmake", ["--build", buildDir, "--config", "Release", "--target", cliTarget]);

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

  copyRuntime(built, "llama.cpp source build");
}

function cmakeTargets() {
  const result = spawnSync("cmake", ["--build", buildDir, "--target", "help"], {
    cwd: repoRoot,
    encoding: "utf8",
  });
  if (result.status !== 0) {
    return [];
  }
  return result.stdout
    .split(/\r?\n/)
    .map((line) => line.trim().replace(/^\.\.\.\s*/, ""))
    .map((line) => line.split(/\s+/)[0])
    .filter(Boolean);
}

function copyRuntime(source, label) {
  const absolute = resolve(source);
  if (!fileExists(absolute)) {
    fail(`${label} did not point to an executable file: ${source}`);
  }
  copyFileSync(absolute, sidecarPath);
  if (!isWindows) {
    run("chmod", ["755", sidecarPath]);
  }
  const validation = validateRuntime(sidecarPath);
  if (!validation.ok) {
    unlinkSync(sidecarPath);
    fail(`prepared runtime is not portable: ${validation.reason}`);
  }
  log(`prepared bundled llama.cpp runtime from ${label}: ${relative(sidecarPath)}`);
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
      .find((line) =>
        line.startsWith("/opt/homebrew/")
        || line.startsWith("/usr/local/")
        || line.includes("libggml")
        || line.includes("libllama")
      );
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

function ensureCommand(command) {
  const result = spawnSync(command, ["--version"], { encoding: "utf8" });
  if (result.error || result.status !== 0) {
    fail(`required build tool is missing: ${command}`);
  }
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

function fileExists(path) {
  try {
    return statSync(path).isFile();
  } catch {
    return false;
  }
}

function relative(path) {
  return path.replace(`${repoRoot}/`, "");
}

function log(message) {
  if (!quiet) {
    console.log(message);
  }
}

function fail(message) {
  console.error(message);
  process.exit(1);
}
