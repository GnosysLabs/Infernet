#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOPIC="infernet/grid-demo/1"
WITH_DEMO_PEERS=0

for arg in "$@"; do
  case "$arg" in
    --with-demo-peers)
      WITH_DEMO_PEERS=1
      ;;
    *)
      echo "usage: $0 [--with-demo-peers]" >&2
      exit 2
      ;;
  esac
done

cd "$ROOT_DIR"

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait "${pids[@]:-}" 2>/dev/null || true
}
trap cleanup EXIT

if [ "$WITH_DEMO_PEERS" -eq 1 ]; then
  cargo build -p infernet-worker >/dev/null 2>&1

  target/debug/infernet-worker serve --model grid-demo-12 --layers 0:3 --topic "$TOPIC" \
    >target/infernet-ui-peer-a.log 2>&1 &
  pids+=("$!")

  target/debug/infernet-worker serve --model grid-demo-12 --layers 3:6 --topic "$TOPIC" \
    >target/infernet-ui-peer-b.log 2>&1 &
  pids+=("$!")

  target/debug/infernet-worker serve --model grid-demo-12 --layers 6:9 --topic "$TOPIC" \
    >target/infernet-ui-peer-c.log 2>&1 &
  pids+=("$!")

  target/debug/infernet-worker serve --model grid-demo-12 --layers 9:12 --topic "$TOPIC" \
    >target/infernet-ui-peer-d.log 2>&1 &
  pids+=("$!")

  sleep 2
fi

if [ ! -d infernet-ui/node_modules ]; then
  npm --prefix infernet-ui install
fi

npm --prefix infernet-ui run tauri dev
