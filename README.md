# Infernet

Infernet is an experimental peer-to-peer split-inference system for local and
community-owned AI compute.

The core idea is simple: the app is always connected to an AI grid. Your
computer is one node in that grid. A model can be present as many physical layer
shards spread across many machines, and the scheduler builds an ordered route
through the peers that can execute those layers.

The user experience should feel like a normal AI chat app. The network exists
under the surface. Users choose a model and send a message; Infernet discovers
peers, verifies available shards, downloads missing executable shards when
needed, builds a route, forwards activations hop by hop, and returns a response.

## Current State

This repository is not a production LLM runtime yet. It is a working prototype
of three separate peer-to-peer protocols:

1. Peer discovery
2. Model shard distribution
3. Distributed activation forwarding

The current implementation includes:

- libp2p peer discovery over mDNS, gossipsub, static peers, and a public
  bootstrap node.
- A Tauri desktop app with a chat-first UI, model library, downloads view, and
  optional network activity details.
- Physical GGUF layer shard creation from a local or Hugging Face GGUF file.
- A local shard cache with verified checksums and executable-shard metadata.
- P2P model shard transfer over `/infernet/model-blob/1`.
- P2P activation forwarding over `/infernet/activation/1`.
- Route construction from live peer advertisements.
- A patched llama.cpp bridge that can load and evaluate a contiguous layer range
  from an Infernet physical shard.

The current GGUF runtime is correctness-first. It is not fast and it is not a
finished product. The first bridge is intended to prove that a real GGUF model
can be partitioned into physical executable layer shards and routed across peers.
Multi-token distributed KV-cache streaming, advanced scheduling, proofs of
correct execution, privacy protections, and production NAT traversal are still
future work.

Infernet intentionally does not ship a fake bridge or fake peers by default. If
the real runtime cannot build or load, that is treated as a real error.

## What Infernet Enables

Infernet is trying to prove that a model does not have to live on one computer.

In the target architecture:

- One node can introduce a model to the network.
- Infernet automatically splits the model into executable layer shards.
- Other nodes can discover those shards.
- Nodes can download only the shards they should host.
- Downloaders immediately become seeders.
- The network can route inference through the machines that collectively cover
  the model.
- No single participating node needs to possess the whole model forever.

Today, the prototype has the first version of that loop:

- A user can add a GGUF model.
- Infernet builds real physical GGUF layer shard files.
- Those shards are cached locally and advertised to peers.
- Other peers can fetch shard files peer-to-peer and start serving them.
- The app only treats a peer as inference-capable when it advertises executable
  physical shards, not loose metadata.

## Architecture

Infernet is split into three protocols. They are intentionally separate.

### 1. Peer Discovery

Discovery is metadata only. It tells the app which peers exist and what they
claim to host.

Discovery uses libp2p:

- mDNS for LAN discovery.
- gossipsub for repeated node advertisements.
- static peers for manual testing and fallback.
- public bootstrap peers for internet rendezvous.

Each node advertises:

- `peer_id`
- listen addresses
- protocol version
- hosted executable shard descriptors
- cached model shard records
- model id
- layer range
- checksum
- shard version

Only executable shard descriptors are used to build inference routes.

### 2. Model Distribution

Model distribution is independent from inference. It moves model shard files
between peers.

The model protocols are:

```text
/infernet/model/1
/infernet/model-blob/1
```

`/infernet/model/1` exists for model metadata and legacy record exchange.

`/infernet/model-blob/1` transfers physical shard bytes in verified chunks. A
peer asks for a specific model id, layer range, checksum, version, offset, and
chunk size. The receiver responds with bytes from the matching local shard file.

Every downloaded shard is verified before it is accepted:

- checksum must match
- byte size must match
- protocol version must match
- layer range must match

After verification, the downloader stores the shard in its local cache and
immediately advertises it. That makes every downloader a seeder.

### 3. Distributed Inference

Inference uses a separate libp2p request-response protocol:

```text
/infernet/activation/1
```

The client builds a route from discovered executable shard descriptors. It sends
the first activation request to the first peer in that route. Each peer:

1. validates that the request layer range matches its hosted shard
2. loads/evaluates its local contiguous layer range
3. records trace information
4. forwards the activation to the next peer

The final peer returns the response back through the request chain.

Trace events include:

- trace id
- current peer id
- processed layer range
- next peer id
- activation size
- timing in milliseconds
- activation checksum

There is no direct TCP activation path in the default inference flow.

## Physical GGUF Shards

This is the most important current subsystem.

When a user adds a `.gguf` model, Infernet does not merely store a JSON record
or a pointer to the full file. It creates physical layer shard `.gguf` files in
the local shard cache.

### What A Shard Contains

A physical Infernet GGUF shard is still a GGUF file. It contains:

- the original GGUF metadata section
- tokenizer-compatible metadata
- all non-layer/global tensors currently required by llama.cpp loading
- only the `blk.N.*` tensors for the shard's assigned layer range
- rewritten tensor directory entries
- rewritten tensor offsets
- shard checksum, size, version, and protocol metadata

For a layer range `16:24`, the shard includes block tensors for layers
`16 <= N < 24`. It excludes block tensors outside that range.

### What Is Duplicated

The current shard writer deliberately keeps global tensors in every shard. That
includes tensors such as embeddings, output norms, and other non-`blk.N.*`
tensors.

This is not storage-optimal. It is a correctness-first design. Duplicating
global tensors lets the patched llama.cpp loader initialize a partial model
without inventing a new GGUF container format.

Future shard formats should reduce this duplication.

### What Is Not A Shard

Infernet no longer treats these as executable model capacity:

- metadata-only seed records
- arbitrary files imported with the legacy `model import` command
- bare `model_shards` rows without an executable seed manifest
- stale cache entries whose physical file is missing

Those records may exist in older caches, but they do not count toward route
construction and they should not produce model cards in the app.

### Automatic Shard Planning

Users do not choose a shard size.

Infernet plans contiguous layer ranges automatically from GGUF metadata. The
current planner is simple:

- demo models use small fixed ranges
- llama.cpp/GGUF models use contiguous layer groups
- larger models use larger groups

This is not yet hardware-aware. The future scheduler should account for RAM,
VRAM, disk budget, bandwidth, latency, and network replication health.

## What Happens When You Add A Model

From the desktop app:

1. The user chooses a local `.gguf` file or downloads one from Hugging Face.
2. Infernet parses GGUF metadata:
   - architecture
   - layer count
   - hidden size
   - quantization
   - tokenizer metadata
3. Infernet hashes the source file.
4. Infernet automatically plans layer ranges.
5. Infernet writes physical shard `.gguf` files into the shard cache.
6. Each shard is checksummed and recorded with metadata.
7. The node starts or refreshes the model distribution service.
8. The node advertises only executable physical shards.
9. Other nodes can discover and download those shards.

The UI shows progress for:

- checking/parsing the file
- verifying/hash reading
- building shards
- writing shard `X of Y` for layer range `A:B`
- starting sharing
- ready

For multi-GB models, the shard-build step can take real time and significant
disk I/O. The current format can also require substantial extra disk space
because global tensors are duplicated into each shard.

## What Happens On Another Machine

When another node sees a model on the network, it should not assume the model is
already installed locally.

The current behavior is:

- discovery finds peers advertising executable shards
- the app can show that the model is available on the network
- before chat, the node downloads executable shard files from peers when needed
- downloaded shards are verified
- downloaded shards are cached locally
- the node immediately begins seeding those shards
- route construction uses the union of local and remote executable shards

This is the beginning of Infernet becoming a self-hosting model repository. Once
one node introduces a model, other nodes can obtain shards from peers instead of
from a central HTTP file server.

## Route Construction

The router builds an ordered sequence of peers for a requested model.

It uses only `hosted_shards` that are executable:

- model id must match
- runtime kind must match
- layer ranges must cover `0..layer_count`
- the descriptor must include a valid executable GGUF seed manifest

If no full route exists, Infernet returns a clear missing-range error:

```text
no complete route for model gemma-4-12b-it-iq4-xs; missing layer ranges: 0:48
```

That error means the network does not currently have executable shard coverage
for the full model. It does not mean the model name exists somewhere in a cache.
It means route construction cannot find every required layer.

## Desktop App

The app lives in `infernet-ui` and uses Tauri, React, and TypeScript.

The current UI is chat-first:

- Chat
- Models
- Downloads
- Settings

The app no longer exposes "Local" and "AI Grid" modes. Infernet is always a
network. Your computer is one node in that network.

### Chat

The chat screen is the primary interface. Network details are hidden until they
matter.

During inference, the app can show:

- thinking state
- peer count
- route discovered
- hop started
- hop completed
- activation size
- timing
- final output

Developer/network details are available through the network activity view.

### Models

The Models page is where users add models.

Supported sources:

- local `.gguf` file
- Hugging Face GGUF file download

The app chooses shard layout automatically. Users do not choose "4 layers per
shard" or similar parameters.

### Downloads

The Downloads page is for technical storage and contribution state:

- installed models
- installed shards
- storage used
- uploads
- downloads
- replication health placeholders

The current replication health display is informational. Automatic replication
and rebalancing are future work.

## Setup Requirements

Required for development:

- Rust toolchain
- Node.js and npm
- Git
- CMake
- C++ compiler toolchain

Infernet downloads official llama.cpp prebuilt binaries when available, but the
Infernet split-layer bridge is an Infernet patch and may need to be built
locally. Setup does not silently install OS-level build tools.

Windows:

```powershell
winget install --id Kitware.CMake -e
winget install --id Microsoft.VisualStudio.2022.BuildTools -e --override "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive --wait"
```

After installing, open a new PowerShell so `cmake`, `cl`, and Visual Studio
environment discovery are visible.

macOS:

```sh
xcode-select --install
brew install cmake
```

Debian/Ubuntu:

```sh
sudo apt-get update
sudo apt-get install -y git cmake build-essential
```

If a bridge dependency is missing, `npm run prepare-runtime` fails instead of
launching a degraded app.

## Running The Desktop App

Install dependencies:

```sh
npm --prefix infernet-ui install
```

Launch the real Tauri app node:

```sh
scripts/ui-demo.sh
```

On Windows PowerShell:

```powershell
.\scripts\ui-demo.ps1
```

This starts the desktop app with the real Rust backend. It does not start fake
demo peers.

The old four-peer demo is still available, but it is explicit:

```sh
scripts/ui-demo.sh --with-demo-peers
```

```powershell
.\scripts\ui-demo.ps1 -WithDemoPeers
```

For frontend-only visual work:

```sh
cd infernet-ui
npm install
npm run dev
```

The browser-only Vite view uses mock data. Use Tauri for real networking,
sharding, and inference commands.

## CLI Commands

Add a local GGUF model and build physical shards:

```sh
cargo run -p infernet-worker -- model add-local \
  --cache-dir /tmp/infernet-node \
  --gguf /path/to/model.gguf \
  --version v1
```

Serve cached model shards:

```sh
cargo run -p infernet-worker -- model serve \
  --cache-dir /tmp/infernet-node
```

Fetch a specific shard from peers:

```sh
cargo run -p infernet-worker -- model fetch \
  --cache-dir /tmp/infernet-peer \
  --model gemma-4-12b-it-iq4-xs \
  --layers 0:8 \
  --checksum <sha256> \
  --version v1
```

Download and immediately mirror:

```sh
cargo run -p infernet-worker -- model mirror \
  --cache-dir /tmp/infernet-peer \
  --model gemma-4-12b-it-iq4-xs \
  --layers 0:8 \
  --checksum <sha256> \
  --version v1
```

List local cache state:

```sh
cargo run -p infernet-worker -- model list \
  --cache-dir /tmp/infernet-peer
```

`model import` is a legacy/debug path for arbitrary cache payloads. It does not
create executable GGUF shards and should not be used for real inference.

## Toy Split-Inference Demo

The toy demo is still useful for protocol testing.

Run the repeatable dynamic discovery smoke test:

```sh
scripts/smoke-demo.sh "hello infernet"
```

On Windows PowerShell:

```powershell
.\scripts\smoke-demo.ps1 "hello infernet"
```

The script starts four independent workers, lets them discover one another over
libp2p, builds a route dynamically, and runs inference without a hardcoded peer
order.

Manual local run:

```sh
cargo run -p infernet-worker -- serve --model grid-demo-12 --layers 0:3
cargo run -p infernet-worker -- serve --model grid-demo-12 --layers 3:6
cargo run -p infernet-worker -- serve --model grid-demo-12 --layers 6:9
cargo run -p infernet-worker -- serve --model grid-demo-12 --layers 9:12
```

Inspect peers:

```sh
cargo run -p infernet-worker -- peers
```

Build a route:

```sh
cargo run -p infernet-worker -- route --model grid-demo-12
```

Run inference:

```sh
cargo run -p infernet-worker -- infer \
  --model grid-demo-12 \
  --prompt "hello infernet"
```

## LAN And Internet Discovery

mDNS works across machines on the same LAN when local firewalls allow it.

For LAN testing, bind libp2p to a dialable interface:

```sh
cargo run -p infernet-worker -- serve \
  --model grid-demo-12 \
  --layers 0:3 \
  --p2p-listen /ip4/0.0.0.0/tcp/7001
```

For internet discovery, the desktop app dials the public bootstrap node:

```text
/ip4/217.77.11.197/tcp/9777/p2p/12D3KooWRJrnpHPQTWdThpDGZMwRCHhEBL4JCAxFMwYMfFavxa2h
```

`infernet.gnosyslabs.xyz` is also configured as a DNS bootstrap address. The DNS
record must be DNS-only for raw libp2p TCP. If it is proxied through Cloudflare,
raw libp2p TCP will not work through that hostname.

Current WAN limitations:

- public bootstrap helps peers discover each other
- private NAT-to-NAT connectivity still needs circuit relay or hole punching
- firewalls can still block inbound model and activation requests
- multi-source downloads and resume are not implemented yet

## Troubleshooting

### A model appears and then says "unknown model"

That means the UI saw a stale or incomplete advertisement. Current builds only
trust executable `gguf-shard` manifests. Pull latest on every machine and re-add
the model on the seed node so it builds physical shards.

### A route is missing `0:N`

No peer is advertising executable shard coverage for those layers. Adding a
model name is not enough. At least one peer must have physical verified shard
files for the required layer ranges.

### The model import reaches 100% and keeps working

Hash verification is done, but shard building is still writing physical GGUF
shard files. Current builds show `Building shards` and `Writing shard X of Y`.
Large models can still take time because this is real disk I/O.

### The app says there are peers but no runnable model

Peer connectivity and model availability are different. A peer only helps
inference if it advertises executable shard descriptors for the selected model.

### Runtime preparation fails

Install the missing CMake/C++ build tools or provide real bridge binaries with:

```sh
INFERNET_LLAMA_CLI=/path/to/llama-cli \
INFERNET_LLAMA_BRIDGE=/path/to/infernet-llama-bridge \
npm --prefix infernet-ui run prepare-runtime
```

Infernet does not fall back to a non-functional bridge.

## Current Limitations

Infernet is still a prototype.

Important limitations:

- Physical shard v1 duplicates global tensors into every shard.
- Shard planning is layer-count based, not hardware-aware.
- The scheduler does not yet optimize for RAM, VRAM, latency, or bandwidth.
- WAN connectivity still needs relay/hole-punching for difficult NAT cases.
- Downloads are chunked and verified, but not resumable.
- Multi-source downloads are not implemented.
- Automatic replication and self-healing are not implemented.
- GGUF split execution is experimental and currently focused on contiguous
  layer ranges.
- The bridge is intended to prove a prompt pass plus a sampled token; persistent
  distributed KV-cache streaming for long chat responses is future work.
- There is no cryptographic proof that a peer executed a layer correctly.
- There is no privacy layer; peers can inspect the activations they process.
- The trust model currently assumes participants are cooperative.

## Design Documents

Read these for deeper architecture notes:

- [docs/infernet-technical-design.md](docs/infernet-technical-design.md)
- [docs/gguf-split-inference-design.md](docs/gguf-split-inference-design.md)
- [docs/model-distribution-design.md](docs/model-distribution-design.md)
