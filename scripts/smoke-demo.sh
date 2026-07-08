#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROMPT="${1:-hello infernet}"
TOPIC="infernet/smoke/$$"

cd "$ROOT_DIR"

cargo build -p infernet-worker >/dev/null 2>&1

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait "${pids[@]:-}" 2>/dev/null || true
}
trap cleanup EXIT

target/debug/infernet-worker serve --model grid-demo-12 --layers 0:3 --topic "$TOPIC" \
  >target/infernet-peer-a.log 2>&1 &
pids+=("$!")

target/debug/infernet-worker serve --model grid-demo-12 --layers 3:6 --topic "$TOPIC" \
  >target/infernet-peer-b.log 2>&1 &
pids+=("$!")

target/debug/infernet-worker serve --model grid-demo-12 --layers 6:9 --topic "$TOPIC" \
  >target/infernet-peer-c.log 2>&1 &
pids+=("$!")

target/debug/infernet-worker serve --model grid-demo-12 --layers 9:12 --topic "$TOPIC" \
  >target/infernet-peer-d.log 2>&1 &
pids+=("$!")

sleep 2

target/debug/infernet-worker infer \
  --model grid-demo-12 \
  --prompt "$PROMPT" \
  --topic "$TOPIC" \
  --discovery-timeout-ms 6000
