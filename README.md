# Infernet

Infernet is an experimental peer-to-peer split-inference system. Infernet is
always a distributed network: your computer is one node in the same peer graph
as every other participant. Users select a model; the scheduler decides whether
the route is local-only, mixed local/remote, or remote-only.

The current stable milestone proves a real peer-to-peer topology for a tiny demo
model:

- workers advertise shard metadata over libp2p mDNS + gossipsub;
- each worker hosts only its assigned layer range;
- clients discover peers, build a complete ordered route, and send activation
  requests over libp2p request-response;
- each worker processes its hop and forwards to the next peer over
  `/infernet/activation/1`;
- no client command needs to hardcode peer order.

Read the designs first:

- [docs/infernet-technical-design.md](docs/infernet-technical-design.md)
- [docs/gguf-split-inference-design.md](docs/gguf-split-inference-design.md)
- [docs/model-distribution-design.md](docs/model-distribution-design.md)

The next milestone targets `llama-3.2-1b`, a Llama 3.2 1B GGUF model. The
repository can now build Infernet GGUF shard sidecar metadata, advertise
LlamaCpp shard ranges, and route them by runtime kind. Real llama.cpp
layer-range execution is intentionally guarded until the native bridge can load
only the assigned layer range.

## Peer Discovery

Each `infernet-worker serve` process creates a libp2p peer identity and joins a
gossipsub topic. mDNS discovers other libp2p peers on the LAN. Once peers are in
the mesh, workers repeatedly publish a `NodeAdvertisement` containing:

- `peer_id`
- libp2p listen address
- `model_id`
- hosted `layer_start:layer_end`
- protocol version

Discovery traffic is metadata only. It tells a client which peer hosts which
model layers, but it does not carry activation tensors.

Advertisements now carry two independent shard lists:

- `hosted_shards` for inference-capable layer execution;
- `model_shards` for locally cached model files available over
  `/infernet/model/1`.

## Model Distribution

Model shard files are distributed peer-to-peer over:

```text
/infernet/model/1
```

This protocol is separate from activation forwarding. A node can seed model
files without serving inference, and a downloader becomes a seeder as soon as it
stores and verifies a shard.

Import an initial shard:

```sh
cargo run -p infernet-worker -- model import \
  --cache-dir /tmp/infernet-seed \
  --model grid-demo-12 \
  --layers 0:3 \
  --file /path/to/shard.bin \
  --version v1
```

Serve cached shards:

```sh
cargo run -p infernet-worker -- model serve --cache-dir /tmp/infernet-seed
```

Download and immediately mirror:

```sh
cargo run -p infernet-worker -- model mirror \
  --cache-dir /tmp/infernet-peer \
  --model grid-demo-12 \
  --layers 0:3 \
  --checksum <sha256>
```

List local cache state:

```sh
cargo run -p infernet-worker -- model list --cache-dir /tmp/infernet-peer
```

Storage options include `--max-storage-bytes`, repeated `--preferred-model`,
repeated `--pinned-model`, and `--no-auto-cleanup`. Received shards are verified
with SHA-256 and size checks before they are accepted into the local cache.

## Activation Forwarding

Activation tensors use a libp2p request-response protocol:

```text
/infernet/activation/1
```

The client sends the first `ActivationRequest` to the first peer in the route.
That peer executes its local layers, appends a trace event, and forwards the
updated request to the next peer. The final peer returns an
`ActivationResponse`; intermediate peers relay that response back upstream.

Each trace event records:

- trace id
- current peer id
- processed layer range
- next peer id
- activation byte size
- timing in milliseconds
- activation checksum

There is no direct TCP activation path in the default inference flow.

## Route Construction

Clients run discovery for a short window, store advertisements in a shard
registry, and ask the router for a complete route for the selected model. The
router filters by model id and runtime kind, sorts layer ranges from
`0..layer_count`, and rejects incomplete coverage with a clear missing-range
error, for example:

```text
no complete route for model grid-demo-12; missing layer ranges: 3:6, 9:12
```

Manual peers can be added as fallback with repeated `--static-peer` flags:

```sh
--static-peer 12D3...peer@/ip4/192.168.1.20/tcp/7001/p2p/12D3...peer#0:3
```

## Local Smoke Demo

Run the repeatable dynamic discovery smoke test:

```sh
scripts/smoke-demo.sh "hello infernet"
```

The script starts four independent workers, lets them discover one another over
libp2p, and runs inference without a hardcoded route.

## GGUF Shard Metadata

Build a sidecar manifest for a Llama 3.2 1B GGUF layer range:

```sh
cargo run -p infernet-worker -- shard build \
  --model llama-3.2-1b \
  --gguf /path/to/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --layers 0:4 \
  --out /tmp/llama-3.2-1b-0-4.infernet-shard.json
```

The sidecar records model id, architecture, layer range, tokenizer checksum,
source GGUF checksum, GGUF tensor directory selection, and shard hash. It does
not yet materialize a physical tensor-only shard. Physical shards and the
llama.cpp layer-range bridge are required before claiming that a real worker
does not possess the full model file.

## Desktop UI Demo

The minimal desktop UI lives in `infernet-ui` and uses Tauri v2, React, and
TypeScript. It visualizes the same P2P route used by the CLI. The UI no longer
presents Local and AI Grid execution modes or static test models; it presents
installed/imported models and the distributed route chosen for the selected
model:

- model list and selected model;
- local node identity as one peer in the network;
- discovered peers and advertised shard ranges;
- route coverage;
- chat prompt, hop-by-hop progress, and final output.

Start four workers and launch the desktop UI:

```sh
scripts/ui-demo.sh
```

For frontend-only visual development without Tauri commands:

```sh
cd infernet-ui
npm install
npm run dev
```

The browser-only Vite view uses mock data. Use `npm run tauri dev` or
`scripts/ui-demo.sh` when you need real libp2p discovery and activation
forwarding.

## Manual Local Run

Start four workers in separate terminals:

```sh
cargo run -p infernet-worker -- serve --model grid-demo-12 --layers 0:3
cargo run -p infernet-worker -- serve --model grid-demo-12 --layers 3:6
cargo run -p infernet-worker -- serve --model grid-demo-12 --layers 6:9
cargo run -p infernet-worker -- serve --model grid-demo-12 --layers 9:12
```

Inspect discovered peers:

```sh
cargo run -p infernet-worker -- peers
```

Build the dynamic route:

```sh
cargo run -p infernet-worker -- route --model grid-demo-12
```

Run inference:

```sh
cargo run -p infernet-worker -- infer --model grid-demo-12 --prompt "hello infernet"
```

The output includes a deterministic demo token and a trace showing each peer and
layer range that participated.

## LAN Run

mDNS works across machines on the same LAN when local firewalls allow traffic.
For cross-machine activation forwarding, bind libp2p to a LAN interface or use
the default `0.0.0.0` listener when the OS advertises a dialable LAN address:

```sh
cargo run -p infernet-worker -- serve \
  --model grid-demo-12 \
  --layers 0:3 \
  --p2p-listen /ip4/0.0.0.0/tcp/7001
```

Run the other layer ranges on other machines with their own LAN IPs. Then run:

```sh
cargo run -p infernet-worker -- route --model grid-demo-12
cargo run -p infernet-worker -- infer --model grid-demo-12 --prompt "hello infernet"
```

Current limitations:

- mDNS is LAN/local only; WAN discovery still needs Kademlia/bootstrap work.
- Model distribution uses `/infernet/model/1`, but currently transfers one JSON
  encoded payload from one source. Binary streaming, resume, and multi-source
  downloads are future work.
- The activation payload is JSON-encoded `f32` demo data, not a real model
  tensor codec.
- Real GGUF execution is not linked yet; LlamaCpp routes fail explicitly instead
  of falling back to the toy runtime.
- Peers are trusted for this phase; there is no correctness proof or privacy
  protection from the worker executing a layer.
