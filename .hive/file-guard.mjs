#!/usr/bin/env node
import { readFile } from "node:fs/promises";

const [, , command, ...rawPaths] = process.argv;
const usage = "Usage: node .hive/file-guard.mjs <claim|release> <path> [path...]";

if ((command !== "claim" && command !== "release") || rawPaths.length === 0) {
  console.error(usage);
  process.exit(64);
}

const session = JSON.parse(await readFile(new URL("./session.json", import.meta.url), "utf8"));
const paths = rawPaths.map(normalizePath).filter(Boolean);

if (paths.length === 0) {
  console.error(usage);
  process.exit(64);
}

if (command === "claim") {
  let denied = false;

  for (const path of paths) {
    const decision = await post("/ownership/request", {
      taskId: session.taskId,
      agentId: session.agentId,
      ownerUserId: session.userId,
      resources: [{ kind: "file", pattern: path }]
    });

    if (decision.granted) {
      console.log(`Hive granted ${path}`);
      continue;
    }

    denied = true;
    console.error(`Hive denied ${path}: ${decision.reason}`);
  }

  process.exit(denied ? 2 : 0);
}

const snapshot = await get("/snapshot");
const activeClaims = snapshot.ownership.filter((claim) => claim.status === "active" && claim.taskId === session.taskId);

for (const path of paths) {
  const claims = activeClaims.filter((claim) => claim.resources.some((resource) => resource.kind === "file" && selectorMatchesPath(resource.pattern, path)));

  if (claims.length === 0) {
    console.log(`Hive had no active claim for ${path}`);
    continue;
  }

  await Promise.all(claims.map((claim) => post("/ownership/release", {
    taskId: session.taskId,
    claimId: claim.id
  })));
  console.log(`Hive released ${path}`);
}

async function get(path) {
  const response = await fetch(`${session.serverUrl}${path}`);

  if (!response.ok) {
    throw new Error(`Hive request failed: ${response.status} ${response.statusText}`);
  }

  return response.json();
}

async function post(path, body) {
  const response = await fetch(`${session.serverUrl}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body)
  });

  if (!response.ok) {
    throw new Error(`Hive request failed: ${response.status} ${response.statusText}`);
  }

  return response.json();
}

function normalizePath(path) {
  return path.replaceAll("\\", "/").replace(/^\/+/, "").replace(/^\.\//, "");
}

function selectorMatchesPath(pattern, path) {
  const normalizedPattern = normalizePath(pattern);
  const normalizedPath = normalizePath(path);

  if (normalizedPattern === normalizedPath) {
    return true;
  }

  if (normalizedPattern.endsWith("/**")) {
    const prefix = normalizedPattern.slice(0, -3);
    return normalizedPath.startsWith(prefix);
  }

  return false;
}
